"""The review page: accept, reject or mark-for-correction, with the slot quotas in front of you.

Spec decisions D4/D5 and 4 M4: the automatic score only sorts and shows; a person decides. So this page is a plain
grid of what the pipeline produced, four buttons per image, and the quota board that says which slots are still short.
It is a local page (spec 7 Q3) on stdlib http.server, and the decisions go into the project's `review.json`.

Corrections are not applied here: marking an image `fix` records the decision, and the fix stage (which drives tool2)
picks it up, so a page refresh can never start GPU work by accident.

Two things the page and the API agree on:

* the image URL is `/image/<id><the file's own suffix>` and the route answers exactly that (with the file's media type),
  so a picture that is shown in the page is a picture the server can actually deliver;
* a decision may carry a structured `reason`. `different_character` is the one reason that also limits what else may
  be done with the image: it may only be rejected, because correcting another person's picture only produces another
  person. The page, the API and the correction stage all read that one field;
* the keyboard decides only what the reviewer can see. The cursor, the movement and the A/R/F/D keys all read the
  visible cards in the same order, a filter control that owns the keyboard silences them, zero visible cards mean no
  decision, and a decision that no longer matches the active filter removes its card from the view at once (R1).
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
import time
import webbrowser
from pathlib import Path
import urllib.parse
from urllib.parse import unquote

AUTO = Path(__file__).resolve().parents[1]
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))

from pipeline import project as pj  # noqa: E402

LABELS = {"accept": "채택", "reject": "기각", "fix": "보정"}
KEYS = {"accept": "A", "reject": "R", "fix": "F"}
# The reasons shown as their own button, so "another character" is a value and not a sentence someone typed.
REASON_BUTTONS = {"different_character": ("다른 캐릭터", "D")}
# A candidate marked fix cannot be a different character: correcting the wrong person would only produce another one.
FIX_BLOCKED = "different_character"
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def fixable(state):
    """May this candidate still be sent to the correction stage? Only a named different-character image may not."""
    return (state or {}).get("reason") not in pj.FIX_BLOCKED_REASONS


def image_url(row, prefix="/image"):
    """The URL of one candidate's image: its id plus the suffix of the file behind it."""
    return f"{prefix}/{row['id']}{row.get('suffix') or '.png'}"


def candidate_rows(project, *, decision=None, generator=None, slot=None, images_prefix="/image", scores=None):
    """The candidates as page rows, newest first, each with its decision and whether it can still be corrected.

    `scores` is `{id: row}` from `score.table`. It is passed in rather than read here so the page still renders on a
    project that has never been scored, and so a scoring pass that failed cannot stop a reviewer from working. Every
    score field is `None` when there is no score, never 0: a picture ArcFace found no face in has no identity score,
    and showing it as 0.0 would read as "the worst picture in the set".
    """
    review = project.review()
    index = project.index()
    scores = scores or {}
    rows = []
    for record in reversed(project.candidates()):
        state = review.get(record["id"]) or {}
        mark = state.get("decision")
        if decision and mark != decision:
            continue
        if generator and record.get("generator") != generator:
            continue
        slot_info = record.get("slot") or {}
        slot_id = slot_info.get("slot_id") or slot_info.get("id") or slot_info.get("category")
        if slot and slot_id != slot:
            continue
        path = project.resolve_file(record.get("file"))
        before = None
        fix_of = record.get("fix_of") or next((p for p in (record.get("parent_ids") or [])
                                               if p.startswith("c_")), None) if record.get("edit_depth") else None
        if fix_of and fix_of in index:
            before_path = project.resolve_file(index[fix_of].get("file"))
            before = {"id": fix_of, "url": f"{images_prefix}/{fix_of}"
                                           f"{(before_path.suffix.lower() if before_path else '.png')}",
                      "exists": bool(before_path and before_path.is_file()),
                      "decision": (review.get(fix_of) or {}).get("decision")}
        scored = scores.get(record["id"]) or {}
        rows.append({
            "id": record["id"],
            "file": record.get("file"),
            "rel": Path(record.get("file") or "").name,
            "suffix": (path.suffix.lower() if path else Path(record.get("file") or "").suffix.lower()) or ".png",
            "exists": bool(path and path.is_file()),
            "generator": record.get("generator"),
            "slot": slot_id,
            "category": slot_info.get("category"),
            "generation": record.get("train_generation"),
            "edit_depth": record.get("edit_depth"),
            "parents": record.get("parent_ids") or [],
            "caption": record.get("caption", ""),
            "decision": mark,
            "reason": state.get("reason"),
            "by": state.get("by"),
            "note": state.get("note", ""),
            "fixable": fixable(state),
            "fix_of": fix_of,
            "before": before,
            "clip_max": scored.get("clip_max_cosine"),
            "clip_mean": scored.get("clip_mean_cosine"),
            "arcface_max": scored.get("arcface_max_cosine"),
            "arcface_note": scored.get("arcface_no_face") or "",
            "white_ratio": scored.get("white_ratio"),
        })
    return rows


def attach_crops(project, rows, *, store=None):
    """Put each candidate's stored crop into its row, so the page renders it after a reload (spec N7).

    The score a crop produced is not stored with the box: it is a measurement of a model and a reference set, and both
    can change. What is stored is the box and the image's SHA - the part a person decided - and the score shown on a
    reload is only the one this page measured in the current session. Keeping the two apart is what stops an old
    number from looking current.
    """
    store = read_crop_store(project) if store is None else store
    for row in rows:
        entry = store.get(row["id"])
        if isinstance(entry, dict) and entry.get("box"):
            row["crops"] = {str(entry.get("kind") or "face"):
                            {"box": list(entry["box"]),
                             "max": (row.get("crop_scores") or {}).get(str(entry.get("kind") or "face"))}}
    return rows


def _duplicate_groups(project, rows, *, max_distance=6):
    """Annotate rows with their near-duplicate group, using the scorer's own pHash clustering (spec R7 P2).

    Group 1 is the largest. A row in no group stays without `dup_group` - "not a near-duplicate" is the normal case
    and must not look like group 0. Files that are missing are simply not grouped; nothing is removed anywhere.
    """
    scorable = [row for row in rows if row.get("exists")]
    if len(scorable) < 2:
        return []
    try:
        from .scorers import common
        groups = common.phash_groups([{"id": row["id"], "path": str(project.resolve_file(row["file"]))}
                                      for row in scorable], max_distance=max_distance)["rows"]
    except Exception:  # noqa: BLE001 - a hashing failure must not take the review page down
        return []
    sizes = {}
    for number, group in enumerate(groups, start=1):
        sizes[number] = group["count"]
        for row_id in group["ids"]:
            for row in rows:
                if row["id"] == row_id:
                    row["dup_group"] = number
                    row["dup_count"] = group["count"]
    return sizes


def _is_on(row, decision):
    """Is this plain button the current state? A reasoned rejection lights its own button, not "기각"."""
    if row["decision"] != decision:
        return False
    if decision == "reject" and (row["reason"] or "") in REASON_BUTTONS:
        return False
    return True


def _cell(row, images_prefix):
    """One image card: the picture, what it is, and the decision buttons."""
    from html import escape
    rel = image_url(row, images_prefix)
    if not row["exists"]:
        body = f'<div class="ph">파일 없음<br>{escape(Path(row["file"] or "?").name)}</div>'
    else:
        body = (f'<a href="{escape(rel)}" target="_blank"><img loading="lazy" src="{escape(rel)}" '
                f'alt="{escape(row["id"])}"></a>')
    buttons = "".join(
        f'<button type="button" class="dec {"on" if _is_on(row, name) else ""}" data-id="{row["id"]}" '
        f'data-decision="{name}"{"" if name != "fix" or row["fixable"] else " disabled"}>{LABELS[name]} '
        f'<span class="k">{KEYS[name]}</span></button>'
        for name in pj.DECISIONS)
    for reason, (label, key) in REASON_BUTTONS.items():
        on = row["decision"] == "reject" and row["reason"] == reason
        buttons += (f'<button type="button" class="dec reason {"on" if on else ""}" data-id="{row["id"]}" '
                    f'data-decision="reject" data-reason="{escape(reason)}">{escape(label)} '
                    f'<span class="k">{escape(key)}</span></button>')
    meta = (f'<div class="meta">{escape(str(row["generator"]))}'
            f' · gen {row["generation"]} · depth {row["edit_depth"]}'
            f' · {escape(str(row["slot"] or "-"))}</div>')
    # The scores are shown as measurements, kept apart: CLIP looks at the whole frame and ArcFace at the face, and the
    # two disagree enough on this data (+0.34 between them) that one blended number would hide which said what.
    score_bits = []
    if row.get("clip_max") is not None:
        score_bits.append(f'CLIP {row["clip_max"]:.3f}')
    if row.get("arcface_max") is not None:
        score_bits.append(f'face {row["arcface_max"]:.3f}')
    elif row.get("arcface_note"):
        score_bits.append(f'face — ({escape(row["arcface_note"])})')
    if row.get("white_ratio") is not None:
        score_bits.append(f'흰배경 {row["white_ratio"]:.2f}')
    scores_line = (f'<div class="scores">{" · ".join(score_bits)}</div>' if score_bits else "")
    # A stored crop is part of what the page shows (spec N7): the box and its score are rendered inline, so a reload
    # restores the measurement instead of losing it. `crops` is `{kind: {box, max}}` for this candidate's own image.
    crops = row.get("crops") or {}
    crop_lines = ""
    for kind, entry in sorted(crops.items()):
        if not isinstance(entry, dict) or not entry.get("box"):
            continue
        value = entry.get("max")
        shown = "점수 없음" if value is None else f"max {float(value):.3f}"
        crop_lines += (f'<div class="crop-score">{escape(kind)} {escape(shown)} · box '
                       f'{escape(",".join(str(v) for v in entry["box"]))}</div>')
    # Sortable values, kept as data so the page can order without re-rendering.
    sort_data = (f'data-clip="{"" if row.get("clip_max") is None else row["clip_max"]}" '
                 f'data-face="{"" if row.get("arcface_max") is None else row["arcface_max"]}" '
                 f'data-white="{"" if row.get("white_ratio") is None else row["white_ratio"]}" '
                 f'data-slot="{escape(str(row["slot"] or ""))}" '
                 f'data-decision-key="{escape(str(row["decision"] or ""))}" '
                 f'data-dup="{row.get("dup_group") or ""}"')
    dup = row.get("dup_count") or 0
    flags = "".join(f'<div class="flag">{escape(text)}</div>' for text in (
        ("다른 캐릭터" if row["reason"] == FIX_BLOCKED else ""),
        ("" if row["fixable"] else "보정 불가"),
        (f"근접 중복 G{row.get('dup_group')} ({dup}장)" if dup > 1 else ""),
        (f"결정: {escape(str(row['by']))}" if row["by"] else ""),
    ) if text)
    before = row.get("before")
    pair = ""
    if before is not None:
        # A correction is only judgeable next to what it corrected: same card, the original on the left.
        thumb = (f'<img loading="lazy" src="{escape(before["url"])}" alt="{escape(before["id"])}">'
                 if before["exists"] else '<div class="ph small">원본 파일 없음</div>')
        pair = (f'<div class="pair"><div class="side"><a href="{escape(before["url"])}" target="_blank">{thumb}</a>'
                f'<div class="id">원본 {escape(before["id"])}</div></div>'
                f'<div class="arrow">→</div></div>')
    return (f'<div class="card" data-id="{row["id"]}" data-decision="{row["decision"] or ""}" '
            f'data-reason="{escape(row["reason"] or "")}" {sort_data} '
            f'data-keys="{KEYS["accept"]}{KEYS["reject"]}{KEYS["fix"]}D">'
            f'{pair}{body}{meta}{scores_line}{crop_lines}<div class="id">{escape(row["id"])}</div>'
            f'<div class="caption">{escape((row["caption"] or "")[:120])}</div>'
            f'{flags}<div class="buttons">{buttons}</div></div>')


def write_page(project, destination, *, images_prefix="/image", quota_plan=None, log=print, scores=None):
    """Render the review page. Decisions are posted back, so it is meant to be served, not opened from disk."""
    if scores is None:
        # Reading the stored scores here keeps the page useful without every caller having to pass them, while the
        # `scores` parameter still lets a caller show a pass that has not been written yet.
        try:
            from . import score as score_mod
            scores = score_mod.table(project)
        except Exception:  # noqa: BLE001 - a missing or broken score file must not hide the review page
            scores = {}
    rows = candidate_rows(project, images_prefix=images_prefix, scores=scores)
    # A stored crop box renders with its card, so a reload shows what was measured (spec N7). Any score beside it is
    # this session's, never a stored number: a model or reference change would make an old one wrong.
    attach_crops(project, rows)
    # Near-duplicate groups (spec R7 P2): computed with the same pHash the scorer uses, shown as a number on each
    # grouped card, and filterable - grouped for review, never removed.
    dup_sizes = _duplicate_groups(project, rows)
    quotas = project.quotas(quota_plan or _plan_from_card(project.card))
    short = {slot: info for slot, info in quotas.items() if info["short"]}
    counts = {name: sum(1 for r in rows if r["decision"] == name) for name in pj.DECISIONS}
    blocked = sum(1 for r in rows if r["reason"] == FIX_BLOCKED)
    synthetic_share, source_counts = project.source_share()
    scored = sum(1 for row in rows if row["clip_max"] is not None or row["arcface_max"] is not None)
    quota_rows = "".join(
        f'<tr class="{"short" if info["short"] else ""}"><td>{html.escape(slot)}</td><td>{info["accepted"]}</td>'
        f'<td>{info["target"]}</td><td>{info["short"] or ""}</td></tr>'
        for slot, info in sorted(quotas.items()))
    page = _TEMPLATE.format(
        title=html.escape(f"{project.card.get('id')} 검수"),
        summary=html.escape(
            f"후보 {len(rows)} · 미검수 {len(project.undecided())} · 채택 {counts['accept']} · "
            f"기각 {counts['reject']} · 보정 {counts['fix']} · 슬롯 부족 {len(short)}"
            + (f" · 점수 {scored}" if scored else " · 점수 없음")
            + (f" · 다른 캐릭터 {blocked}" if blocked else "")
            + (f" · 근접 중복 묶음 {len(dup_sizes)}" if dup_sizes else "")),
        sources=html.escape(f"원본 {source_counts['user']}장 · 합성 {source_counts['synthetic']}장 "
                            f"({synthetic_share:.0%})"),
        cards="".join(_cell(row, images_prefix) for row in rows),
        quotas=quota_rows or '<tr><td colspan="4">계획된 슬롯이 없습니다</td></tr>',
        slot_options="".join(
            f'<option value="{html.escape(slot)}">{html.escape(slot)}</option>'
            for slot in sorted({str(row["slot"]) for row in rows if row["slot"]})),
        keys=json.dumps(KEYS),
        reason_keys=json.dumps({key: reason for reason, (_label, key) in REASON_BUTTONS.items()}),
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(destination, page)
    log(f"review page: {destination}")
    return destination


def _plan_from_card(card):
    """Slot targets from the card, when it carries a plan; otherwise no quota board."""
    plan = card.get("slot_quotas") or {}
    return {slot: int(target) for slot, target in plan.items()}


_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font: 14px/1.45 system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }}
 h1 {{ font-size: 18px; margin: 0 0 4px; }}
 .summary, .sources {{ color: #9ab; }}
 .sources {{ margin-bottom: 10px; }}
 table.quota {{ border-collapse: collapse; margin: 10px 0 16px; }}
 table.quota th, table.quota td {{ border: 1px solid #333; padding: 3px 8px; font-size: 12px; }}
 table.quota tr.short {{ background: #3a2418; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
 .card {{ width: 246px; border: 1px solid #333; border-radius: 4px; padding: 6px; background: #161616; }}
 .card img {{ width: 232px; height: auto; display: block; }}
 .card .ph {{ width: 232px; height: 160px; display: flex; align-items: center; justify-content: center;
              color: #a55; background: #1c1414; font-size: 12px; text-align: center; }}
 .card .meta {{ color: #789; font-size: 11px; margin-top: 3px; }}
 .card .id {{ color: #567; font-size: 10px; font-family: monospace; }}
 .card .caption {{ color: #8ab; font-size: 11px; min-height: 30px; }}
 .card .flag {{ color: #d08; font-size: 11px; }}
 .buttons {{ display: flex; gap: 4px; margin-top: 4px; flex-wrap: wrap; }}
 .buttons button {{ flex: 1 1 40%; padding: 4px 0; font-size: 12px; background: #1d2228; color: #9ab;
                    border: 1px solid #333b44; border-radius: 3px; cursor: pointer; }}
 .buttons button .k {{ color: #567; font-size: 10px; }}
 .buttons button:disabled {{ opacity: .35; cursor: not-allowed; }}
 .buttons button.on {{ color: #eafff4; font-weight: 600; }}
 .buttons button.on[data-decision=accept] {{ background: #2f6f4f; border-color: #47a878; }}
 .buttons button.on[data-decision=reject] {{ background: #7a2f2f; border-color: #b85555; }}
 .buttons button.on[data-decision=fix] {{ background: #6b5a1f; border-color: #b09a3a; }}
 .buttons button.reason.on {{ background: #5c2244; border-color: #b0508c; }}
 .card[data-decision=reject] img {{ opacity: .45; }}
 .card.cursor {{ outline: 2px solid #4a9ed9; outline-offset: 1px; }}
 .card .pair {{ display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }}
 .card .pair img {{ width: 74px; height: auto; display: block; border: 1px solid #333; }}
 .card .pair .ph.small {{ width: 74px; height: 60px; font-size: 10px; }}
 .card .pair .arrow {{ color: #789; font-size: 14px; }}
 .card .scores {{ color: #8a9; font-size: 11px; font-family: monospace; margin-top: 2px; }}
 .card .crop-score {{ color: #9bd; font-size: 11px; font-family: monospace; }}
 #controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0 14px;
              padding: 6px 10px; background: #191d21; border: 1px solid #2a3138; border-radius: 4px; }}
 #controls label {{ color: #9ab; font-size: 12px; }}
 #controls select, #controls input {{ background: #111; color: #cde; border: 1px solid #333b44;
                                      border-radius: 3px; padding: 3px 6px; font-size: 12px; }}
 #controls button {{ background: #1d2228; color: #9ab; border: 1px solid #333b44;
                     border-radius: 3px; padding: 3px 9px; font-size: 12px; cursor: pointer; }}
 #controls button:hover {{ color: #eafff4; }}
 #state {{ position: sticky; bottom: 0; background: #1b1b1b; border-top: 1px solid #333; padding: 6px 10px; }}
</style></head><body>
<h1>{title}</h1>
<div class="summary">{summary}</div>
<div class="sources">{sources}</div>
<table class="quota"><thead><tr><th>슬롯</th><th>채택</th><th>목표</th><th>부족</th></tr></thead>
<tbody>{quotas}</tbody></table>
<div id="controls">
 <label>정렬 <select id="sort">
  <option value="">기본(최신순)</option>
  <option value="clip-desc">CLIP 높은 순</option>
  <option value="clip-asc">CLIP 낮은 순</option>
  <option value="face-desc">얼굴 점수 높은 순</option>
  <option value="face-asc">얼굴 점수 낮은 순</option>
  <option value="white-asc">흰 배경 적은 순</option>
 </select></label>
 <label>슬롯 <select id="slot"><option value="">전체</option>{slot_options}</select></label>
 <label>상태 <select id="status">
  <option value="">전체</option>
  <option value="__none">미검수</option>
  <option value="accept">채택</option>
  <option value="reject">기각</option>
  <option value="fix">보정</option>
 </select></label>
 <label>점수 있는 것만 <input type="checkbox" id="scored"></label>
 <label>근접 중복만 <input type="checkbox" id="dup"></label>
 <button type="button" id="reset">초기화</button>
 <span id="shown" style="color:#789;font-size:12px"></span>
 <span style="color:#789;font-size:12px">크롭 재채점(대상 카드):
  <select id="crop-preset">
   <option value="">직접 입력</option>
   <option value="face">얼굴(중앙 35%)</option>
   <option value="upper">상반신(중앙 70%)</option>
  </select>
  <input id="crop-box" placeholder="x,y,w,h (0~1)" size="12">
  <button type="button" id="crop-run">재채점</button>
  <label title="원본마다 다른 위치의 크롭을 지정합니다 (spec N7)">원본별
   <input type="checkbox" id="crop-per-image"></label>
  <button type="button" id="crop-clear" title="저장된 크롭을 모두 지웁니다">크롭 지우기</button>
  <span id="crop-state" style="color:#8a9"></span>
 </span>
</div>
<div class="grid" id="grid">{cards}</div>
<div id="state"><span id="target"></span><span id="last"></span>이미지를 클릭하면 확대, A=채택 · R=기각 · F=보정 · D=다른 캐릭터.
결정은 review.json에 바로 기록됩니다.</div>
<script>
const KEYS = {keys};
const REASON_KEYS = {reason_keys};
let cursor = 0;
let cursorCard = null;
// Only the cards the reviewer can see are reviewable: a filter hides a card from the eye, so it must hide it from
// the keyboard too. The cursor, the movement and the decision keys all read this one list, in the same order.
function cards() {{
  return Array.from(document.querySelectorAll('#grid .card')).filter(c => c.style.display !== 'none');
}}
function paint(card, decision, reason) {{
  card.dataset.decision = decision || '';
  card.dataset.reason = reason || '';
  card.querySelectorAll('.dec').forEach(button => {{
    const on = (button.dataset.decision || '') === (decision || '')
      && (button.dataset.reason || '') === (reason || '');
    button.classList.toggle('on', on);
  }});
}}
async function send(card, decision, reason) {{
  // one request per click: the server writes the final state once, never "decided" then "undecided"
  const toggleOff = card.dataset.decision === decision && (card.dataset.reason || '') === (reason || '');
  const next = toggleOff ? null : decision;
  const payload = {{id: card.dataset.id, decision: next}};
  if (next && reason) {{ payload.reason = reason; }}
  const response = await fetch('/decide', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
                                          body: JSON.stringify(payload)}});
  const data = await response.json();
  if (!response.ok) {{ document.getElementById('last').textContent = data.error || 'failed'; return; }}
  paint(card, data.decision, data.reason);
  // Written into its own span, not over the state bar: the state bar carries the cursor's target element, and
  // overwriting it would destroy that element and break the next view re-sync.
  document.getElementById('last').textContent = card.dataset.id + ' → ' + (data.decision || '미검수')
    + (data.reason ? ' (' + data.reason + ')' : '');
  // A card whose decision no longer matches the filter must leave the view at once - under the undecided filter a
  // decided card is gone the moment its decision lands - so the view is re-applied after every stored decision.
  applyView();
}}
function setCursor(index) {{
  const list = cards();
  if (cursorCard) cursorCard.classList.remove('cursor');
  const target = document.getElementById('target');
  if (!list.length) {{
    cursorCard = null;
    if (target) target.textContent = '대상 없음 (표시된 카드 0장)';
    return;
  }}
  cursor = Math.max(0, Math.min(index, list.length - 1));
  cursorCard = list[cursor];
  cursorCard.classList.add('cursor');
  if (target) target.textContent = '대상 ' + cursorCard.dataset.id + ' · ';
}}
function syncCursor() {{
  const list = cards();
  if (cursorCard && list.includes(cursorCard)) {{ cursor = list.indexOf(cursorCard); return; }}
  setCursor(cursor);
}}
document.addEventListener('click', e => {{
  const card = e.target.closest('.card');
  if (card) {{
    const at = cards().indexOf(card);
    if (at >= 0) setCursor(at);
  }}
  const button = e.target.closest('.dec');
  if (button && !button.disabled) {{
    send(button.closest('.card'), button.dataset.decision, button.dataset.reason);
  }}
}});
document.addEventListener('keydown', e => {{
  const list = cards();
  // A filter control that owns the keyboard must never let a letter decide a picture: arrows stay with the control
  // and A/R/F/D do nothing until the reviewer clicks back into the grid.
  if (document.activeElement && document.getElementById('controls').contains(document.activeElement)) return;
  if (e.key === 'j' || e.key === 'ArrowDown') {{
    e.preventDefault();
    setCursor(cursor + 1);
    if (cursorCard) cursorCard.scrollIntoView({{block: 'center'}});
  }}
  else if (e.key === 'k' || e.key === 'ArrowUp') {{
    e.preventDefault();
    setCursor(cursor - 1);
    if (cursorCard) cursorCard.scrollIntoView({{block: 'center'}});
  }}
  else if (list.length) {{
    // With nothing visible there is nothing to decide, and cursor always names a visible card.
    const card = list[Math.min(cursor, list.length - 1)];
    const key = e.key.toUpperCase();
    if (!card) return;
    for (const [decision, wanted] of Object.entries(KEYS)) {{
      if (key === wanted) {{ send(card, decision, ''); return; }}
    }}
    if (REASON_KEYS[key]) {{ send(card, 'reject', REASON_KEYS[key]); }}
  }}
}});
cards()[0]?.setAttribute('tabindex', '0');

// Sorting and filtering works on the DOM the server already rendered, so the review keeps working with no round trip
// and no second source of truth for what a card contains.
const allCards = Array.from(document.querySelectorAll('#grid .card'));
const originalOrder = allCards.slice();
function num(card, key) {{
  const raw = card.dataset[key];
  return raw === undefined || raw === '' ? null : Number(raw);
}}
function applyView() {{
  const sort = document.getElementById('sort').value;
  const slot = document.getElementById('slot').value;
  const status = document.getElementById('status').value;
  const scoredOnly = document.getElementById('scored').checked;
  const dupOnly = document.getElementById('dup').checked;

  let visible = allCards.filter(card => {{
    if (slot && card.dataset.slot !== slot) return false;
    if (status === '__none') {{ if (card.dataset.decision) return false; }}
    else if (status && card.dataset.decision !== status) return false;
    if (scoredOnly && num(card, 'clip') === null && num(card, 'face') === null) return false;
    if (dupOnly && !card.dataset.dup) return false;
    return true;
  }});

  if (sort) {{
    const [field, direction] = sort.split('-');
    const key = field === 'white' ? 'white' : field;
    visible.sort((a, b) => {{
      const av = num(a, key), bv = num(b, key);
      // A picture with no score is not a zero: it stays at the end in both directions rather than sorting as the worst.
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return direction === 'desc' ? bv - av : av - bv;
    }});
  }} else {{
    visible = visible.slice().sort((a, b) => originalOrder.indexOf(a) - originalOrder.indexOf(b));
  }}

  const grid = document.getElementById('grid');
  allCards.forEach(card => {{ card.style.display = 'none'; }});
  visible.forEach(card => {{ card.style.display = ''; grid.appendChild(card); }});
  document.getElementById('shown').textContent = visible.length + ' / ' + allCards.length + '장 표시';
  // Every view change re-syncs the cursor, so the highlighted card is always one the reviewer can actually see.
  syncCursor();
}}
['sort', 'slot', 'status', 'scored', 'dup'].forEach(id =>
  document.getElementById(id).addEventListener('change', applyView));
const CROP_PRESETS = {{face: [0.325, 0.05, 0.35, 0.35], upper: [0.15, 0.05, 0.7, 0.6]}};
function cropState(text) {{ document.getElementById('crop-state').textContent = text; }}
document.getElementById('crop-preset').addEventListener('change', e => {{
  const box = CROP_PRESETS[e.target.value];
  if (box) document.getElementById('crop-box').value = box.join(',');
}});
document.getElementById('crop-run').addEventListener('click', async () => {{
  const card = cursorCard || cards()[0];
  if (!card) {{ cropState('대상 카드가 없습니다'); return; }}
  const parts = document.getElementById('crop-box').value.split(',').map(s => parseFloat(s.trim()));
  if (parts.length !== 4 || parts.some(v => !isFinite(v) || v < 0 || v > 1) || parts[2] <= 0 || parts[3] <= 0) {{
    cropState('박스는 x,y,w,h (0~1, 너비·높이 > 0)'); return;
  }}
  const kind = document.getElementById('crop-preset').value || 'custom';
  cropState('채점 중…');
  try {{
    // With 원본별 on, the box typed here belongs to every *reference* as well as the candidate, which is the
    // assumption the audit found false for real datasets. So the request names no per-image boxes yet: it reports
    // which references it could not measure, and the reviewer fills those in from /crops/<id> (spec N7).
    const perImage = document.getElementById('crop-per-image').checked;
    let url = '/crop-score/' + card.dataset.id + '?kind=' + kind + '&box=' + parts.join(',');
    if (perImage) {{
      const stored = await loadCrops(card.dataset.id);
      const boxes = Object.entries(stored)
        .filter(([key, row]) => row && row.box && (row.kind || 'face') === kind)
        .map(([key, row]) => encodeURIComponent(key) + ':' + row.box.join(','))
        .join(';');
      if (boxes) {{ url += '&boxes=' + boxes; }}
    }}
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) {{ cropState(data.error || 'failed'); return; }}
    const value = (data.max === null || data.max === undefined)
      ? kind + ' 점수 없음' : kind + ' max ' + Number(data.max).toFixed(3);
    let line = card.querySelector('.crop-score');
    if (!line) {{
      line = document.createElement('div');
      line.className = 'crop-score';
      const anchor = card.querySelector('.scores');
      if (anchor) {{ anchor.after(line); }} else {{ card.appendChild(line); }}
    }}
    const detail = (data.unmeasured && data.unmeasured.length)
      ? ' · 원본 ' + data.unmeasured.length + '장 박스 미지정' : '';
    line.textContent = '크롭 ' + value + ' · box ' + parts.join(',') + detail;
    card.dataset.cropBox = parts.join(',');
    card.dataset.cropKind = kind;
    card.dataset.cropMax = data.max === null || data.max === undefined ? '' : String(data.max);
    cropState(card.dataset.id + ' → ' + value + detail);
  }} catch (error) {{ cropState(String(error)); }}
}});
async function loadCrops(id) {{
  try {{
    const response = await fetch('/crops/' + encodeURIComponent(id));
    if (!response.ok) return {{}};
    const data = await response.json();
    return data.crops || {{}};
  }} catch (error) {{ return {{}}; }}
}}
document.getElementById('crop-clear').addEventListener('click', async () => {{
  const card = cursorCard || cards()[0];
  if (!card) {{ cropState('대상 카드가 없습니다'); return; }}
  // Clearing only forgets the measurement; no image file is touched (spec N7).
  try {{
    const response = await fetch('/crop-clear/' + encodeURIComponent(card.dataset.id), {{method: 'POST'}});
    const data = await response.json();
    if (!response.ok) {{ cropState(data.error || 'failed'); return; }}
    const line = card.querySelector('.crop-score');
    if (line) line.remove();
    cropState('크롭 기록을 지웠습니다 (' + (data.cleared || 0) + '장)');
  }} catch (error) {{ cropState(String(error)); }}
}});
document.getElementById('reset').addEventListener('click', () => {{
  document.getElementById('sort').value = '';
  document.getElementById('slot').value = '';
  document.getElementById('status').value = '';
  document.getElementById('scored').checked = false;
  applyView();
}});
applyView();
</script></body></html>
"""


def check_decision(project, candidate_id, decision):
    """(record, state) for a decision that may be recorded, or a ProjectError saying why it may not.

    A decision about an id that is not in the project is a client error, not a new row in review.json - otherwise a
    typo leaves a decision behind that nothing can ever show. An image judged to be another character may only be
    rejected.
    """
    record = project.candidate_by_id(candidate_id)
    if record is None:
        raise pj.ProjectError(f"존재하지 않는 후보입니다: {candidate_id}")
    state = project.state_of(candidate_id)
    if decision is not None and decision not in pj.DECISIONS:
        raise pj.ProjectError(f"decision must be one of {', '.join(pj.DECISIONS)}, not {decision!r}")
    if decision in ("accept", "fix") and not fixable(state):
        raise pj.ProjectError("다른 캐릭터로 판단된 이미지는 보정할 수 없습니다 (기각만 가능)")
    return record, state


def parse_crop_box(raw):
    """A user-typed crop box "x,y,w,h" in 0~1 fractions, or a ProjectError saying why it is not one (spec R7 P2)."""
    parts = [part.strip() for part in str(raw or "").split(",")]
    if len(parts) != 4:
        raise pj.ProjectError("크롭 박스는 x,y,w,h 네 값입니다")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise pj.ProjectError("크롭 박스 값은 숫자입니다") from exc
    if any(v < 0 or v > 1 for v in values):
        raise pj.ProjectError("크롭 박스 값은 0~1 비율입니다")
    if values[2] <= 0 or values[3] <= 0:
        raise pj.ProjectError("크롭의 너비와 높이는 0보다 커야 합니다")
    return values


def _pixel_box(path, box):
    """A 0~1 box turned into this image's pixels; raises when the crop would have no area."""
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
    x, y, w, h = box
    pixel = [max(0, round(x * width)), max(0, round(y * height)),
             round(w * width), round(h * height)]
    if pixel[2] < 1 or pixel[3] < 1:
        raise pj.ProjectError(f"크롭이 {Path(path).name}에서 면적 0이 됩니다")
    return pixel


def parse_crop_boxes(raw, *, decode=True):
    """`id:x,y,w,h;id:x,y,w,h` into `{id: box}` (spec N7). Empty string means "no per-image boxes".

    `decode` is on when the text has already been through `urllib.parse.parse_qs`, which is the HTTP path: a Windows
    path arrives as `C%3A/x/y.png` and has to be turned back into `C:/x/y.png` to match a real file. A caller that
    passes a string it built itself sets `decode=False` and gets its keys back unchanged.

    A malformed entry is refused rather than skipped: silently dropping one image's box would score it with the
    candidate's box and make the comparison look complete when it was not.
    """
    text = str(raw or "").strip()
    if not text:
        return {}
    out = {}
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, sep, body = chunk.partition(":")
        if not sep or not key.strip():
            raise pj.ProjectError(f"이미지별 박스는 id:x,y,w,h 형식입니다: {chunk[:40]}")
        # The last colon is the one before the numbers; a Windows drive letter puts an earlier one in the key.
        key, sep, body = chunk.rpartition(":")
        if not sep or not key.strip():
            raise pj.ProjectError(f"이미지별 박스는 id:x,y,w,h 형식입니다: {chunk[:40]}")
        name = unquote(key.strip()) if decode else key.strip()
        out[name] = parse_crop_box(body)
    return out


def _now_stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def crop_store_path(project):
    """Where the per-image crop boxes live: `experiments/crops.json` (spec N7).

    A sidecar rather than a field on the candidate record, because a crop is a *reference* the reviewer chose, not
    something the generator produced - and because a candidate's record is rewritten by other steps. Never inside the
    image files themselves: the original bytes are never touched.
    """
    return Path(project.dir) / "experiments" / "crops.json"


def read_crop_store(project):
    """`{image_id_or_path: {"box": [...], "kind": ..., "image_sha256": ..., "at": ..., "by": ...}}`."""
    path = crop_store_path(project)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_crop_store(project, data):
    path = crop_store_path(project)
    pj.write_json(path, data)
    return path


def _crop_key(project, path):
    """How one image is named in the crop store: its candidate id when it has one, else its path."""
    resolved = Path(path).resolve()
    for record in project.candidates():
        candidate = project.resolve_file(record.get("file"))
        if candidate and Path(candidate).resolve() == resolved:
            return record["id"]
    for record in project.source_records():
        source = project.resolve_file(record.get("file") or record.get("src_path"))
        if source and Path(source).resolve() == resolved:
            return record["id"]
    return str(resolved)


def crop_for(project, path, *, name=None, store=None):
    """The stored box for this image under this kind, or None (spec N7).

    `name` selects a box measured as a `face` or an `upper` crop; the store keeps them apart, because a face crop and
    an upper-body crop are not comparable and mixing them would silently compare different regions.
    """
    store = read_crop_store(project) if store is None else store
    entry = store.get(_crop_key(project, path))
    if not isinstance(entry, dict):
        return None
    if name and str(entry.get("kind") or "") != str(name):
        return None
    box = entry.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    return [float(v) for v in box]


def set_crop(project, path, box, *, kind="face", by="user"):
    """Record one image's crop box, with the image's SHA so a changed picture invalidates it (spec N7)."""
    from . import lineage
    data = read_crop_store(project)
    key = _crop_key(project, path)
    data[key] = {"box": [float(v) for v in box], "kind": str(kind),
                 "image_sha256": lineage.sha256_file(path) if Path(path).is_file() else None,
                 "at": _now_stamp(), "by": str(by), "path": str(path)}
    write_crop_store(project, data)
    return data[key]


def crop_scores(project, candidate_id, box, *, name="face", embedder=None, boxes=None, store=True):
    """The identity score of the user's crop, against the same crop of every original (spec R7 P2, N7).

    The box arrives in 0~1 fractions. By default it is applied to the candidate **and to every reference original** -
    a face box is only comparable with other face boxes, which is the rule the scorer already enforces by name. That
    default assumes the face sits at the same fractions in every picture, which is not true of real datasets: the
    audit's finding (`review.crop_scores` applying one box to all originals) was that a group shot and a head-and-
    shoulders portrait put the face in very different places.

    So `boxes` may carry a **per-image** box: `{path_or_id: box}`. Any image it does not name is reported as
    unmeasured rather than silently scored with the candidate's box, and the result says which references were used.
    `store=True` writes every box that was used into the project's crop sidecar, so a reloaded page restores what was
    measured instead of losing it.

    Model residency policy (spec N1): the embedder **stays resident on the GPU between requests** - reload per
    request would make every re-score pay a model load - so the server process holds roughly the CLIP model's VRAM
    (about 1.3 GB, measured) even when idle. What changes per request is the *compute*: the /crop-score/ handler
    takes the common GPU claim for the request span, so concurrent training or generation refuses the re-score with
    a 503 and the re-score never races a training run for the card.
    """
    from . import score as score_mod
    from .scorers import clip_vision
    record = project.candidate_by_id(candidate_id)
    if record is None:
        raise pj.ProjectError(f"존재하지 않는 후보입니다: {candidate_id}")
    target = project.resolve_file(record.get("file"))
    if not target or not Path(target).is_file():
        raise pj.ProjectError("후보 이미지가 디스크에 없습니다")
    references = score_mod.reference_paths(project)
    if not references:
        raise pj.ProjectError("비교할 사용자 원본이 없습니다")
    per_image = {str(k): v for k, v in (boxes or {}).items()}
    paths = [Path(p) for p in references]
    paths.append(Path(target))
    crops = {}
    used, unmeasured = {}, []
    for path in paths:
        is_target = Path(path).resolve() == Path(target).resolve()
        own = per_image.get(str(path)) or per_image.get(path.name) or per_image.get(_crop_key(project, path))
        chosen = own if own is not None else box
        if own is None and per_image and not is_target:
            # The typed box is the *candidate's* box; a reference the request does not name has no box of its own, so
            # it cannot be compared with the candidate's crop. Reported, never guessed (spec N7).
            unmeasured.append(path.name)
        crops[path.name] = {name: _pixel_box(path, chosen)}
        used[path.name] = [float(v) for v in chosen]
    if embedder is None:
        embedder = crop_scores.embedder
        if embedder is None:
            embedder = crop_scores.embedder = clip_vision.Embedder()
    result = clip_vision.identity_scores([Path(target)], [Path(p) for p in references], ids=[candidate_id],
                                         crops=crops, embedder=embedder)
    row = result["rows"][0]
    if store:
        try:
            for path in paths:
                set_crop(project, path, used[path.name], kind=name)
        except OSError:
            pass                     # a read-only project must not fail the measurement over its sidecar
    return {"id": candidate_id, "kind": name, "box": list(box),
            "boxes": used, "per_image": bool(per_image), "unmeasured": unmeasured,
            "max": row.get(f"crop_{name}_max"), "mean": row.get(f"crop_{name}_mean"),
            "reference_count": row.get("reference_count")}



crop_scores.embedder = None


from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402  (imported for the server class)


class ExclusiveServer(ThreadingHTTPServer):
    """A server that refuses to share its port, and says which build is answering (spec N5).

    Windows lets a second `bind()` on a port the first server holds succeed, and requests then reach whichever server
    started first - so a new review page could print "serving ... at 8195" while the browser talked to an old build
    left running for days (`duplicate_review_bind`: the second bind succeeded and all eight requests went to the
    first). `allow_reuse_address = False` plus `SO_EXCLUSIVEADDRUSE` on Windows is what makes the bind fail instead.
    The started-at stamp and the source digest travel in every response, so a page can be asked which code it is.
    """

    allow_reuse_address = False

    def server_bind(self):
        if sys.platform == "win32":
            import socket
            # SO_EXCLUSIVEADDRUSE must be set before bind; SO_REUSEADDR (the F/Windows default in some builds) would
            # otherwise let another process take the port.
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except (AttributeError, OSError):
                pass
        super().server_bind()


def code_identity():
    """Which build of this module is running, so a page can be judged as current or not (spec N5).

    A digest of the server's own source plus the process start time. The audit found a review server started the day
    before still answering on the port, serving a page that had since lost its crop-score control; the response now
    carries the identity a person or a test can compare against.
    """
    source = Path(__file__).read_bytes()
    return {"code_sha256": hashlib.sha256(source).hexdigest()[:16],
            "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def make_server(project, port=8195, log=print, quota_plan=None, embedder_factory=None, claim_factory=None,
                exclusive=True):
    """Serve the review page for one project: the images, the decisions, and nothing else.

    `embedder_factory` builds the CLIP embedder the crop re-score uses, lazily on first use; tests inject a fake so
    the page's GPU path is exercised without a model. `claim_factory` builds the GPU claim each /crop-score/
    request takes (spec N1); tests inject one bound to a scratch folder so the page exercises the claim path
    without touching the real automation/gpu.lock. `exclusive=False` lets a test bind port 0 freely; a real server
    keeps the default.
    """
    from http.server import BaseHTTPRequestHandler

    root = Path(project.dir).resolve()
    embedder_state = {"embedder": None}
    identity = {**code_identity(), "project": project.card.get("id")}


    def embedder():
        if embedder_state["embedder"] is None:
            from .scorers import clip_vision
            embedder_state["embedder"] = (embedder_factory or clip_vision.Embedder)()
        return embedder_state["embedder"]

    def build_claim():
        if claim_factory is not None:
            return claim_factory()
        from . import gpu
        return gpu.claim_gpu(what="crop scoring")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            # Which build and which project answered (spec N5): a page that shows this can be told apart from an old
            # server that kept the port. The header is on every response so even a plain image request carries it.
            self.send_header("X-Review-Code", identity["code_sha256"])
            self.send_header("X-Review-Project", str(identity["project"] or ""))
            self.send_header("X-Review-Started", identity["started_at"])
            self.end_headers()
            self.wfile.write(data)

        def _image(self, wanted):
            """`<id>` or `<id><suffix>`: the page asks for the file's own suffix, and a request without one works too."""
            name = unquote(wanted)
            candidate_id, suffix = name, ""
            stem, dot, tail = name.rpartition(".")
            if dot and "." + tail.lower() in MEDIA_TYPES:
                candidate_id, suffix = stem, "." + tail.lower()
            record = project.candidate_by_id(candidate_id)
            if not record:
                return self._send(404, "not found")
            target = project.resolve_file(record.get("file"))
            if not target or not target.is_file():
                return self._send(404, "not found")
            if suffix and suffix != target.suffix.lower():
                return self._send(404, "not found")
            return self._send(200, target.read_bytes(), MEDIA_TYPES.get(target.suffix.lower(), "image/png"))

        def do_GET(self):
            split = urllib.parse.urlsplit(self.path)
            path = split.path
            if path == "/health":
                # Which code is answering on this port (spec N5). The audit's stale server answered every review
                # route from yesterday's build; this is the question a person or a GUI can actually ask it.
                return self._send(200, json.dumps(identity, ensure_ascii=False),
                                  "application/json; charset=utf-8")
            if path in ("/", "/index.html"):
                page = root / "review.html"
                write_page(project, page, quota_plan=quota_plan, log=lambda *_: None)
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/image/"):
                return self._image(path[len("/image/"):])
            if path.startswith("/crop-score/"):
                candidate_id = unquote(path[len("/crop-score/"):])
                query = urllib.parse.parse_qs(split.query)
                kind = (query.get("kind") or ["face"])[0]
                try:
                    if kind not in ("face", "upper", "custom"):
                        raise pj.ProjectError("크롭 이름은 face, upper 또는 custom입니다")
                    box = parse_crop_box((query.get("box") or [""])[0])
                    # A per-image box may arrive as `boxes=<id>:x,y,w,h;<id>:x,y,w,h` (spec N7). Anything it does not
                    # name is reported as unmeasured rather than scored with the candidate's own box.
                    boxes = parse_crop_boxes((query.get("boxes") or [""])[0])
                    # The CLIP model runs on the one GPU, so a re-score takes the common GPU claim for the request
                    # (spec N1). The model itself stays resident between requests - that policy is documented at
                    # crop_scores(), and the VRAM it holds shows up in the read-only free-VRAM gate.
                    claim = build_claim()
                    from . import gpu as gpu_mod
                    try:
                        claim.acquire()
                    except gpu_mod.GpuBusy as busy:
                        return self._send(503, json.dumps({"error": str(busy)}, ensure_ascii=False),
                                          "application/json; charset=utf-8")
                    try:
                        row = crop_scores(project, candidate_id, box, name=kind, embedder=embedder(),
                                          boxes=boxes or None)
                    finally:
                        claim.release()
                except pj.ProjectError as exc:
                    return self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False),
                                      "application/json; charset=utf-8")
                except Exception as exc:  # noqa: BLE001 - a scorer failure is a message, not a crash
                    return self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"},
                                                      ensure_ascii=False), "application/json; charset=utf-8")
                log(f"{candidate_id} crop {kind} max {row.get('max')}")
                return self._send(200, json.dumps(row, ensure_ascii=False), "application/json; charset=utf-8")
            if path.startswith("/crops/"):
                # What the page restores after a reload: the boxes measured for this image, per kind (spec N7).
                wanted = unquote(path[len("/crops/"):])
                store = read_crop_store(project)
                rows = {}
                for key, value in store.items():
                    if key != wanted and not (isinstance(value, dict) and str(value.get("path") or "").endswith(wanted)):
                        continue
                    if isinstance(value, dict):
                        rows[str(value.get("kind") or "face")] = value
                return self._send(200, json.dumps({"id": wanted, "crops": rows}, ensure_ascii=False),
                                  "application/json; charset=utf-8")
            return self._send(404, "not found")

        def do_POST(self):
            split = urllib.parse.urlsplit(self.path)
            if split.path.startswith("/crop-clear/"):
                # Forget the crop boxes recorded for one image (spec N7). No picture is touched: the store is a small
                # JSON sidecar, and the images are the user's originals.
                candidate_id = unquote(split.path[len("/crop-clear/"):])
                try:
                    store = read_crop_store(project)
                    cleared = 0
                    for key in list(store):
                        value = store.get(key)
                        if key == candidate_id or (isinstance(value, dict)
                                                   and str(value.get("path") or "").endswith(candidate_id)):
                            store.pop(key, None)
                            cleared += 1
                    if cleared:
                        write_crop_store(project, store)
                except OSError as exc:
                    return self._send(500, json.dumps({"error": str(exc)}, ensure_ascii=False),
                                      "application/json; charset=utf-8")
                return self._send(200, json.dumps({"id": candidate_id, "cleared": cleared}, ensure_ascii=False),
                                  "application/json; charset=utf-8")
            if self.path != "/decide":
                return self._send(404, "not found")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, json.dumps({"error": "bad json"}))
            candidate_id = payload.get("id")
            decision = payload.get("decision")
            reason = payload.get("reason") or None
            try:
                check_decision(project, candidate_id, decision)
                if reason is not None and reason not in pj.REVIEW_REASONS:
                    raise pj.ProjectError(f"reason must be one of {', '.join(pj.REVIEW_REASONS)}, not {reason!r}")
                if decision is None:
                    project.undecide(candidate_id)
                    log(f"{candidate_id} → undecided")
                    return self._send(200, json.dumps({"id": candidate_id, "decision": None, "reason": None}))
                state = project.decide(candidate_id, decision, note=payload.get("note", ""),
                                       reason=None if decision == "accept" else reason)
            except pj.ProjectError as exc:
                return self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False))
            log(f"{candidate_id} → {decision}" + (f" ({reason})" if reason else ""))
            self._send(200, json.dumps({"id": candidate_id, "decision": state["decision"],
                                        "reason": state.get("reason"),
                                        "fixable": fixable(state)}))

    server = ExclusiveServer(("127.0.0.1", port), Handler)
    server.code_identity = identity
    log(f"serving {root} at http://127.0.0.1:{server.server_address[1]}/ "
        f"(code {identity['code_sha256']}, pid {identity['pid']})")
    return server


def serve(project, port=8195, log=print, quota_plan=None, open_browser=True):
    server = make_server(project, port=port, log=log, quota_plan=quota_plan)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Serve a project's review page (accept / reject / fix)")
    parser.add_argument("--project", required=True, help="project folder (holds project.json)")
    parser.add_argument("--port", type=int, default=8195)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)
    project = pj.Project.open(args.project)
    return 0 if serve(project, port=args.port, quota_plan=pj.slot_plan_of(project),
                      open_browser=not args.no_browser) else 0


if __name__ == "__main__":
    raise SystemExit(main())
