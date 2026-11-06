"""M3-1: the seed LoRA's training set, and the run that trains it.

Spec 7 Q5: the seed set is the originals (generation 0) plus the Qwen edits a person accepted (generation 1). Nothing
newer than `lineage.seed_max_generation` may enter it, and the holdout originals are left out entirely (Q6) together
with every image derived from them, so the seed LoRA never sees - directly or through an edit - the pictures used to
judge it.

Eligibility is decided by `audit()`, which answers three questions per image and records the answer:

* is this a user original, or an explicitly accepted `qwen_edit` candidate? (nothing else is a seed input)
* does its whole recorded ancestry resolve, without a cycle, and without touching a holdout id?
* do the bytes on disk still match the recorded sha256?

An image that cannot be vouched for is excluded with a reason code, never quietly dropped. A synthetic anchor whose own
parents were never recorded is quarantined for the same reason: an unverifiable ancestor could be a holdout.

Captions come from the tool2 dataset where the same image is already there under its content hash: tool2 copies the
originals into its own dataset and writes captions for them, and its caption instructions already omit the character's
permanent features, which is exactly the `omit` policy in Q4. A caption built from the card is the fallback, and which
one was used is recorded per image.

The snapshot itself is written to a temporary folder, checked there, and only then published under its id. Rerunning
with the same inputs verifies the published snapshot and reuses it untouched, so a run that is interrupted, restarted
or repeated never rewrites the files a training run may already be reading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))

from pipeline import lineage, models, project as pj  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
INSTRUCTION_FILES = ("caption_instructions.txt", "caption_instructions_anime.txt")

# The rules below are part of a snapshot's identity: change one, and old snapshots stay valid under the old policy
# while new ones are written under this tag.
SEED_POLICY = "seed-v1"
SEED_GENERATORS = ("qwen_edit",)

# Exclusion reasons: code -> (category for the report, why it happened, what to do about it)
EXCLUSIONS = {
    "holdout_source": ("holdout", "this original is the holdout itself",
                       "nothing: it exists so the LoRA can be judged on something it never saw"),
    "holdout_direct": ("holdout", "a parent of this image is a holdout original",
                       "nothing: the holdout exists so this cannot be trained on; make a new edit from a non-holdout"),
    "holdout_indirect": ("holdout", "an ancestor of this image is a holdout original",
                         "nothing: same reason, one edit further away"),
    "holdout_via_lora": ("holdout", "the LoRA that made this image was trained on a snapshot holding a holdout",
                         "retrain the generator LoRA on a snapshot without holdouts"),
    "missing_parent": ("lineage", "a recorded parent id is not in sources.jsonl or candidates.jsonl",
                       "restore the missing record, or re-generate with the parent recorded"),
    "cyclic_lineage": ("lineage", "the parent chain leads back to the image itself",
                       "fix the parent ids; a cycle cannot be traced to an original"),
    "unverified_ancestor": ("lineage", "an ancestor is a synthetic anchor with no recorded parent",
                            "record the anchor's parent, or generate from a labelled original instead"),
    "no_ancestry": ("lineage", "a generated image with neither parents nor a LoRA snapshot",
                    "record where the image came from"),
    "lora_snapshot_unverified": ("lineage", "the LoRA names a training snapshot this project has no record of",
                                 "keep the snapshot folder, or retrain and record it"),
    "synthetic_source": ("eligibility", "a synthetic image in sources/ is not a user original",
                         "keep it out of the seed set; it can still be a reference"),
    "not_reviewed": ("review", "no decision was recorded for this candidate", "review it on the web page"),
    "rejected": ("review", "the reviewer rejected it", "nothing"),
    "marked_for_fix": ("review", "the reviewer asked for a correction; the corrected image is the candidate",
                       "run the fix stage"),
    "different_character": ("review", "the reviewer judged it another person", "nothing: it cannot be corrected"),
    "generator_not_eligible": ("eligibility", "only tool2 Qwen edits are seed inputs (spec 7 Q5)",
                               "use the Qwen edit generator, or accept it as a final set instead"),
    "generation_cap": ("lineage", "train_generation is past seed_max_generation", "use a lower generation"),
    "edit_depth_cap": ("lineage", "edit_depth is past max_edit_depth", "use a shallower edit"),
    "missing_file": ("integrity", "the image file is not on disk", "re-import or re-generate it"),
    "tampered_file": ("integrity", "the bytes on disk no longer hash to the recorded sha256",
                      "restore the original bytes, or re-record the candidate"),
    "duplicate_bytes": ("integrity", "another kept image has the same bytes; this one is not the representative",
                        "nothing: one picture is one training image"),
}


def card_caption(card):
    """The caption the card alone can produce: the trigger, the style, and the traits when the policy includes them."""
    parts = [card.get("trigger", ""), card.get("caption_base", "")]
    if str(card.get("caption_traits") or "omit").lower() == "include":
        parts += list(card.get("fixed_traits") or [])
        parts.append((card.get("outfits") or {}).get("default", ""))
    return ", ".join(part.strip() for part in parts if part and part.strip())


def add_traits(caption, card):
    """Append the fixed traits to an existing caption, for the `include` policy."""
    if str(card.get("caption_traits") or "omit").lower() != "include":
        return caption
    extra = [t for t in (card.get("fixed_traits") or []) if t and t.lower() not in caption.lower()]
    outfit = (card.get("outfits") or {}).get("default", "")
    if outfit and outfit.lower() not in caption.lower():
        extra.append(outfit)
    return ", ".join([caption] + extra) if extra else caption


def tool2_captions(folder):
    """{sha256: caption} for the images tool2 already captioned in its own dataset folder.

    Matched by content, not by name: the same picture under two names is one picture, and that is the only link
    between a project source and the caption tool2 wrote for the copy it made.
    """
    out = {}
    folder = Path(folder) if folder else None
    if not folder or not folder.is_dir():
        return out
    for image in sorted(folder.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        sidecar = image.with_suffix(".txt")
        if not sidecar.is_file():
            continue
        try:
            digest = lineage.sha256_file(image)
        except OSError:
            continue
        caption = sidecar.read_text(encoding="utf-8-sig").strip()
        if caption:
            out.setdefault(digest, caption)
    return out


# ---------------------------------------------------------------------- ancestry
def walk(record, index):
    """(ancestor ids, unknown parent ids, cyclic) for one record, following every recorded parent."""
    ancestors, unknown = set(), set()
    stack = [(record, frozenset([record["id"]]))]
    cyclic = False
    while stack:
        node, path = stack.pop()
        for parent_id in node.get("parent_ids") or []:
            if parent_id in path:
                cyclic = True
                continue
            if parent_id in ancestors:
                continue
            ancestors.add(parent_id)
            parent = index.get(parent_id)
            if parent is None:
                unknown.add(parent_id)
                continue
            stack.append((parent, path | {parent_id}))
    return ancestors, unknown, cyclic


def _snapshot_manifests(project):
    """{snapshot id: manifest} for the seed snapshots this project has a record of."""
    out = {}
    folder = project.dir / "seed_lora"
    if not folder.is_dir():
        return out
    for child in sorted(folder.iterdir()):
        manifest = child / "snapshot.json"
        if manifest.is_file():
            data = pj.read_json(manifest)
            if data.get("id"):
                out[data["id"]] = data
    return out


def _reason(code, detail=""):
    category, why, fix = EXCLUSIONS[code]
    return {"code": code, "category": category, "reason": why + (f" ({detail})" if detail else ""), "fix": fix}


def audit(project, *, caps=None, log=None):
    """Every source and candidate with its verdict, the reason for it, and the counts that follow.

    The result is a report as well as a plan: `kept` (deduplicated by content, with every lineage id that shares the
    bytes), `excluded` (one row per refusal, with a reason code, a sentence and what to do), `counts`, `duplicates`
    and `shortfall`. Nothing here writes to the project.
    """
    caps = caps or lineage.caps(project.card.get("lineage"))
    holdout = project.holdout_entry()
    holdout_ids = set(holdout["ids"])
    holdout_shas = set(holdout["shas"])
    index = project.index()
    review = project.review()
    manifests = _snapshot_manifests(project)

    rows, kept, excluded = [], [], []

    def refuse(record, code, detail="", kind=None, sha=None):
        row = {"id": record["id"], "kind": kind or ("source" if record["id"].startswith("s_") else "candidate"),
               "sha256": sha or record.get("sha256"), "file": record.get("file") or record.get("src_path"),
               **_reason(code, detail)}
        rows.append({**row, "eligible": False})
        excluded.append(row)
        return row

    # ---- synthetic anchors whose own provenance is unknown: quarantined, and anything descending from them
    quarantined = set()
    for record in project.source_records():
        if record.get("origin") != "synthetic":
            continue
        parents = [p for p in (record.get("parent_ids") or []) if p in index]
        if not parents:
            quarantined.add(record["id"])

    for record in project.source_records():
        if record["id"] in holdout_ids:
            refuse(record, "holdout_source")
            continue
        if record.get("origin") != "user":
            detail = "; no recorded parent" if record["id"] in quarantined else ""
            refuse(record, "synthetic_source", f"origin {record.get('origin')}{detail}")
            continue
        path = project.resolve_file(record.get("file") or record.get("src_path"))
        if not path or not path.is_file():
            refuse(record, "missing_file", str(record.get("file") or record.get("src_path")))
            continue
        actual = lineage.sha256_file(path)
        if actual != record.get("sha256"):
            refuse(record, "tampered_file", f"recorded {str(record.get('sha256'))[:12]}, disk {actual[:12]}")
            continue
        kept.append({"id": record["id"], "kind": "source", "origin": record.get("origin"),
                     "sha256": actual, "path": str(path), "generation": 0, "edit_depth": 0,
                     "caption": "", "caption_source": "", "lineage_ids": [record["id"]],
                     "decision_by": "", "decision_origin": "", "slot": "", "prompt": ""})
        rows.append({"id": record["id"], "kind": "source", "sha256": actual, "file": str(path), "eligible": True,
                     "code": "", "category": "", "reason": "user original", "fix": ""})

    for record in project.candidates():
        state = review.get(record["id"]) or {}
        decision = state.get("decision")
        reason = state.get("reason")
        if decision is None:
            refuse(record, "not_reviewed")
            continue
        if decision == "reject":
            refuse(record, "different_character" if reason == "different_character" else "rejected")
            continue
        if decision == "fix":
            refuse(record, "marked_for_fix")
            continue
        if reason == "different_character":
            refuse(record, "different_character")
            continue
        if record.get("generator") not in SEED_GENERATORS:
            refuse(record, "generator_not_eligible", str(record.get("generator")))
            continue
        generation = int(record.get("train_generation") or 0)
        if generation > caps["seed_max_generation"]:
            refuse(record, "generation_cap", f"generation {generation} > seed_max_generation "
                                             f"{caps['seed_max_generation']}")
            continue
        depth = int(record.get("edit_depth") or 0)
        if depth > caps["max_edit_depth"]:
            refuse(record, "edit_depth_cap", f"depth {depth} > max_edit_depth {caps['max_edit_depth']}")
            continue
        parents = list(record.get("parent_ids") or [])
        loras = [lora for lora in (record.get("models") or {}).get("loras") or []
                 if lora.get("trained_on_snapshot")]
        if not parents and not loras:
            refuse(record, "no_ancestry")
            continue
        ancestors, unknown, cyclic = walk(record, index)
        if cyclic:
            refuse(record, "cyclic_lineage")
            continue
        if unknown:
            refuse(record, "missing_parent", ", ".join(sorted(unknown)))
            continue
        hit = sorted(set(parents) & holdout_ids)
        if hit:
            refuse(record, "holdout_direct", ", ".join(hit))
            continue
        hit = sorted(ancestors & holdout_ids)
        if hit:
            refuse(record, "holdout_indirect", ", ".join(hit))
            continue
        anchor_hits = sorted(ancestors & quarantined)
        if anchor_hits:
            refuse(record, "unverified_ancestor", ", ".join(anchor_hits))
            continue
        tainted = None
        for lora in (record.get("models") or {}).get("loras") or []:
            snapshot_id = lora.get("trained_on_snapshot")
            if not snapshot_id:
                continue
            manifest = manifests.get(snapshot_id)
            if manifest is None:
                tainted = ("lora_snapshot_unverified", str(snapshot_id))
                break
            snap_shas = {entry.get("sha256") for entry in manifest.get("inputs") or []}
            if snap_shas & holdout_shas or set(manifest.get("holdout_ids") or []) & holdout_shas:
                tainted = ("holdout_via_lora", str(snapshot_id))
                break
        if tainted:
            refuse(record, tainted[0], tainted[1])
            continue
        path = project.resolve_file(record.get("file"))
        if not path or not path.is_file():
            refuse(record, "missing_file", str(record.get("file")))
            continue
        actual = lineage.sha256_file(path)
        if actual != record.get("sha256"):
            refuse(record, "tampered_file", f"recorded {str(record.get('sha256'))[:12]}, disk {actual[:12]}")
            continue
        slot = record.get("slot") or {}
        kept.append({"id": record["id"], "kind": "candidate", "origin": "synthetic", "sha256": actual,
                     "path": str(path), "generation": generation, "edit_depth": depth,
                     "caption": record.get("caption", ""),
                     "caption_source": "the candidate's own caption" if record.get("caption") else "",
                     "lineage_ids": [record["id"]], "decision_by": state.get("by", ""),
                     "decision_origin": state.get("origin") or (
                         pj.ORIGIN_M0 if str(state.get("by") or "").lower().startswith(pj.M0_BY_PREFIX)
                         else pj.ORIGIN_USER),
                     "slot": slot.get("slot_id") or slot.get("id") or slot.get("category") or "",
                     "prompt": record.get("prompt", "")})
        rows.append({"id": record["id"], "kind": "candidate", "sha256": actual, "file": str(path),
                     "eligible": True, "code": "", "category": "", "reason": "accepted Qwen edit", "fix": ""})

    # ---- one picture is one training image: keep a representative and name the ids it stands for
    by_sha, order = {}, []
    for entry in kept:
        if entry["sha256"] not in by_sha:
            by_sha[entry["sha256"]] = entry
            order.append(entry["sha256"])
        else:
            first = by_sha[entry["sha256"]]
            first["lineage_ids"] = list(first["lineage_ids"]) + [entry["id"]]
            row = refuse({"id": entry["id"], "sha256": entry["sha256"], "file": entry["path"]},
                         "duplicate_bytes", f"same bytes as {first['id']}")
            row["representative"] = first["id"]
    unique = [by_sha[sha] for sha in order]
    for entry in unique:
        entry["lineage_ids"] = sorted(set(entry["lineage_ids"]))

    # ---- counts
    by_category, by_code = {}, {}
    for row in excluded:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
        by_code[row["code"]] = by_code.get(row["code"], 0) + 1
    synthetic = sum(1 for entry in unique if entry["origin"] == "synthetic")
    accepted = [record for record in project.candidates()
                if (review.get(record["id"]) or {}).get("decision") == "accept"]
    from_m0 = sum(1 for record in accepted
                  if str((review.get(record["id"]) or {}).get("by") or "").lower().startswith(pj.M0_BY_PREFIX))
    quotas = {slot: {"target": int(target),
                     "accepted": sum(1 for entry in kept_candidates(kept, project, slot)),
                     "eligible": sum(1 for entry in unique
                                     if entry["kind"] == "candidate" and entry["slot"] == slot)}
              for slot, target in (project.card.get("slot_quotas") or {}).items()}
    return {
        "policy": SEED_POLICY, "caps": caps,
        "holdout": holdout,
        "kept": unique, "excluded": excluded, "rows": rows,
        "counts": {
            "sources": len(project.source_records()), "candidates": len(project.candidates()),
            "accepted": len(accepted), "accepted_from_m0": from_m0, "accepted_by_review": len(accepted) - from_m0,
            "kept": len(unique), "kept_sources": sum(1 for e in unique if e["kind"] == "source"),
            "kept_candidates": sum(1 for e in unique if e["kind"] == "candidate"),
            "synthetic": synthetic, "user": len(unique) - synthetic,
            "excluded": len(excluded), "by_category": by_category, "by_code": by_code,
        },
        "synthetic_share": (synthetic / len(unique)) if unique else 0.0,
        "duplicates": {entry["sha256"]: entry["lineage_ids"] for entry in unique if len(entry["lineage_ids"]) > 1},
        "quarantined": sorted(quarantined),
        "quotas": quotas,
        "shortfall": {
            "accepted": len(accepted),
            "eligible": sum(1 for entry in unique if entry["kind"] == "candidate"),
            "note": "accepted candidates that cannot be seed inputs, by reason: " + ", ".join(
                f"{code} {count}" for code, count in sorted(by_code.items())),
        },
    }


def kept_candidates(kept, project, slot):
    """Ids of the accepted candidates that occupy one slot (used for the quota shortfall)."""
    review = project.review()
    out = []
    for record in project.candidates():
        if (review.get(record["id"]) or {}).get("decision") != "accept":
            continue
        want = (record.get("slot") or {}).get("slot_id") or (record.get("slot") or {}).get("id") \
            or (record.get("slot") or {}).get("category")
        if want == slot:
            out.append(record["id"])
    return out


def plan_inputs(project, *, log=print):
    """(kept, excluded, holdout entry) for the seed set, under the current eligibility policy."""
    report = audit(project, log=log)
    if log:
        counts = report["counts"]
        log(f"[seed] {report['policy']}: kept {counts['kept']} "
            f"({counts['kept_sources']} original, {counts['kept_candidates']} accepted edit); "
            f"excluded {counts['excluded']}")
        if report["holdout"]["ids"]:
            log(f"[seed] holdout left out: {len(report['holdout']['ids'])} source(s) "
                f"({report['holdout']['reason']})")
        for row in report["excluded"]:
            log(f"  excluded {row['id']} [{row['code']}] {row['reason']}")
    return report["kept"], report["excluded"], report["holdout"]


# ---------------------------------------------------------------------- snapshot
def snapshot_identity(project, entries, *, caps, holdout, policy=SEED_POLICY):
    """The id of a snapshot: the project, the policy, the caps, the holdout, and every image and caption by hash."""
    payload = {
        "project": project.card.get("id"), "card": project.digest(), "policy": policy, "caps": caps,
        "holdout_shas": sorted(holdout["shas"]),
        "inputs": sorted(({"id": e["id"], "lineage_ids": e["lineage_ids"], "image": e["sha256"],
                           "caption": hashlib.sha256(e["caption"].encode("utf-8")).hexdigest()}
                          for e in entries), key=lambda item: item["id"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _caption_for(entry, captions, fallback, project):
    caption = entry["caption"] or captions.get(entry["sha256"] or "") or ""
    source = entry["caption_source"] or ("tool2's caption for the same image" if caption else "")
    if not caption:
        caption, source = fallback, "built from the card"
    return add_traits(caption, project.card), source


def inventory(snapshot):
    """The images a runner would find in a snapshot's dataset folder, in a stable order."""
    dataset = Path(snapshot["dataset"])
    if not dataset.is_dir():
        return []
    return sorted(p for p in dataset.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def validate_snapshot(base):
    """Every way a published snapshot can be wrong, as sentences; empty means it is intact."""
    base = Path(base)
    manifest = pj.read_json(base / "snapshot.json")
    problems = []
    if not manifest:
        return [f"no readable snapshot.json in {base}"]
    entries = manifest.get("inputs") or []
    dataset = base / "dataset"
    if not dataset.is_dir():
        return [f"the dataset folder is gone: {dataset}"]
    shas = [entry.get("sha256") for entry in entries]
    if len(set(shas)) != len(shas):
        problems.append(f"the manifest lists {len(shas) - len(set(shas))} duplicate sha256 value(s)")
    images = inventory({"dataset": str(dataset)})
    if len(images) != len(entries):
        problems.append(f"{len(images)} image(s) on disk against {len(entries)} in the manifest")
    for entry in entries:
        image = dataset / str(entry.get("file") or "")
        if not image.is_file():
            problems.append(f"missing image: {image.name}")
            continue
        sidecar = image.with_suffix(".txt")
        if not sidecar.is_file():
            problems.append(f"missing caption: {sidecar.name}")
            continue
        caption_sha = hashlib.sha256(sidecar.read_text(encoding="utf-8").strip().encode("utf-8")).hexdigest()
        if entry.get("caption_sha256") and caption_sha != entry["caption_sha256"]:
            problems.append(f"caption of {image.name} does not match its recorded hash")
        if lineage.sha256_file(image) != entry.get("sha256"):
            problems.append(f"{image.name} no longer hashes to its recorded sha256")
    return problems


def build_snapshot(project, *, log=print, policy=SEED_POLICY):
    """Write the seed training set and return what describes it (verifying and reusing an existing one)."""
    report = audit(project)
    kept = report["kept"]
    caps = report["caps"]
    holdout = report["holdout"]
    if not kept:
        raise pj.ProjectError(
            "the seed set would be empty: no eligible input is left. Excluded: "
            + ", ".join(f"{code} {count}" for code, count in sorted(report["counts"]["by_code"].items())))
    captions = tool2_captions((project.card.get("tool2") or {}).get("dataset"))
    fallback = card_caption(project.card)

    entries, conflicts = [], []
    for item in kept:
        caption, source = _caption_for(item, captions, fallback, project)
        entries.append({**item, "caption": caption, "caption_source": source,
                        "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest()})
    # Two lineage ids with the same bytes can carry different captions; the chosen one is recorded with its source.
    for entry in entries:
        options = {}
        for lineage_id in entry["lineage_ids"]:
            record = project.candidate_by_id(lineage_id) or {}
            text = (record.get("caption") or "").strip()
            if text:
                options[lineage_id] = text
        if len(set(options.values())) > 1:
            conflicts.append({"sha256": entry["sha256"], "id": entry["id"], "chosen": entry["caption"],
                              "caption_source": entry["caption_source"], "options": options})

    identity = snapshot_identity(project, entries, caps=caps, holdout=holdout, policy=policy)
    base = project.dir / "seed_lora" / identity
    dataset = base / "dataset"
    if base.exists():
        problems = validate_snapshot(base)
        if not problems:
            manifest = pj.read_json(base / "snapshot.json")
            log(f"[seed] snapshot {identity} already published and intact; reusing it "
                f"({len(inventory(manifest))} image(s))")
            return {**manifest, "reused": True}
        raise pj.ProjectError(
            f"the snapshot folder {base} exists but is not intact: " + "; ".join(problems[:5])
            + f". Move it aside (or restore its files) and run again; nothing was overwritten.")

    staging = base.with_name(f"{identity}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    try:
        (staging / "dataset").mkdir(parents=True)
        written = []
        for entry in entries:
            source_path = Path(entry["path"])
            if not source_path.is_file():
                log(f"[seed] missing image, skipped: {entry['id']} ({source_path})")
                continue
            dest = staging / "dataset" / (entry["sha256"][:16] + source_path.suffix.lower())
            shutil.copy2(source_path, dest)
            if lineage.sha256_file(dest) != entry["sha256"]:
                raise pj.ProjectError(f"{source_path} changed while the snapshot was being written")
            dest.with_suffix(".txt").write_text(entry["caption"] + "\n", encoding="utf-8")
            written.append({**entry, "file": dest.name})
        if not written:
            raise pj.ProjectError("none of the seed set's images exist on disk any more")
        synthetic = sum(1 for e in written if e["origin"] == "synthetic")
        share = synthetic / len(written)
        manifest = {
            "kind": "seed", "id": identity, "base": str(base), "dataset": str(dataset),
            "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
            "project": project.card.get("id"), "card_digest": project.digest(), "policy": policy,
            "counts": {"total": len(written), "synthetic": synthetic, "user": len(written) - synthetic,
                       "excluded": len(report["excluded"]), "refused": len(report["excluded"])},
            "synthetic_share": share, "lineage_caps": caps,
            "holdout_ids": sorted(holdout["ids"]), "holdout_files": holdout["files"],
            "holdout_shas": sorted(holdout["shas"]), "holdout_frozen": holdout["frozen"],
            "caption_policy": str(project.card.get("caption_traits") or "omit"),
            "caption_sources": {e["id"]: e["caption_source"] for e in written},
            "caption_conflicts": conflicts,
            "lineage_ids": {e["id"]: e["lineage_ids"] for e in written},
            "excluded": report["excluded"], "excluded_counts": report["counts"]["by_code"],
            "quotas": report["quotas"], "shortfall": report["shortfall"],
            "inputs": written,
        }
        pj.write_json(staging / "snapshot.json", manifest)
        problems = validate_snapshot(staging)
        if problems:
            raise pj.ProjectError("the snapshot did not validate before publishing: " + "; ".join(problems[:5]))
        if base.exists():
            raise pj.ProjectError(f"another run published {base} while this one was building; run again")
        os.replace(staging, base)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log(f"[seed] snapshot {identity}: {len(written)} image(s) "
        f"({len({e['sha256'] for e in written})} unique), synthetic {share:.0%}")
    if share > 0.7:
        log(f"[seed] warning: {synthetic}/{len(written)} inputs are synthetic (spec 7 Q5 warns above 70%)")
    if report["excluded"]:
        log(f"[seed] {len(report['excluded'])} image(s) left out; reasons: "
            + ", ".join(f"{code} {count}" for code, count in sorted(report["counts"]["by_code"].items())))
    return {**manifest, "reused": False}


def runner_config(project, snapshot, *, family="krea2", epochs=30, rank=None, learning_rate=None,
                  megapixels=None, base=None):
    """The runner settings for this snapshot: a project's own config, holding only what this run needs."""
    base = dict(base or models.config())
    card = project.card
    images = len(snapshot["inputs"])
    config = {
        **{k: v for k, v in base.items() if k.startswith("caption_")},
        # The model paths travel with the run: without them the runner falls back to prefs.json, which may point at
        # model folders that no longer exist (that is exactly what M0 fixed).
        "models": base.get("models"),
        "comfy": base.get("comfy"),
        "id": f"{project.card.get('id')}_seed",
        "family": family,
        "kind": str(card.get("kind") or "anime"),
        "dataset_root": snapshot["dataset"],
        "snapshot_id": snapshot.get("id"),
        "seed_policy": snapshot.get("policy"),
        "trigger": card.get("trigger"),
        "target_steps": max(1, images * int(epochs)),
        "save_every_steps_approx": max(1, images),
        "max_images_per_job": max(50, images),
        "rank": int(rank or base.get("rank") or 8),
        "learning_rate": float(learning_rate or base.get("learning_rate") or 2e-4),
        "adaptive_lr": True, "adaptive_lr_min": 2e-4, "adaptive_lr_max": 4e-4,
        "megapixels": float(megapixels or base.get("megapixels") or 0.25),
        # The captions are already written and were curated: this run must not invent or overwrite any.
        "caption_policy": "preserve",
        "sample_prompts": card.get("sample_prompts") or [
            "{trigger}, korean webtoon style, upper body, front view, looking at viewer, white background",
            "{trigger}, korean webtoon style, full body, standing, looking at viewer, white background"],
    }
    return config


def prepare_runner(project, snapshot, *, config=None, log=print):
    """Write the runner's own folder for this snapshot: its settings, its job list and the caption instructions."""
    folder = Path(snapshot["base"]) / "runner"
    folder.mkdir(parents=True, exist_ok=True)
    config = config or runner_config(project, snapshot)
    (folder / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "todo.json").write_text(json.dumps(
        [{"id": config["id"], "enabled": True, "trigger": config["trigger"], "seed": project.card.get("seed", 42)}],
        ensure_ascii=False, indent=1), encoding="utf-8")
    for name in INSTRUCTION_FILES:
        source = AUTO / name
        if source.is_file():
            shutil.copy2(source, folder / name)
    log(f"[seed] runner settings: {folder / 'config.json'}")
    return folder


def train(project, snapshot, *, config=None, python=None, log=print, on_event=None, extra=()):
    """Run the seed training as its own process, in the project's own runs folder."""
    config = config or runner_config(project, snapshot)
    folder = prepare_runner(project, snapshot, config=config, log=log)
    command = [python or sys.executable, str(AUTO / "runner.py"), "run", "--job", config["id"],
               "--config", str(folder / "config.json"), "--todo", str(folder / "todo.json"),
               "--runs-dir", str(Path(snapshot["base"]) / "runs"), "--events", *extra]
    log(f"[seed] {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace", cwd=str(ROOT))
    # stderr is drained on its own thread: a runner that writes more than a pipe holds on stderr would otherwise
    # stop dead while this loop waits for a stdout line. Both handles are closed, whichever way the run ends.
    errors = []
    drain = threading.Thread(target=_drain, args=(process.stderr, errors), daemon=True)
    drain.start()
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@@FIZGIG_EVENT@@"):
                try:
                    event = json.loads(line[len("@@FIZGIG_EVENT@@"):])
                except ValueError:
                    log(line)
                    continue
                if on_event is not None:
                    on_event(event)
                else:
                    log(f"  {event.get('kind')} {event.get('stage', '')}")
            else:
                log(line)
    finally:
        process.stdout.close()
        code = process.wait()
        drain.join(timeout=10)
    if code:
        raise RuntimeError(f"the seed run exited {code}: {''.join(errors).strip()[-2000:]}")
    return Path(snapshot["base"]) / "runs"


def _drain(stream, sink):
    """Read a subprocess stream to the end, so it can neither block on a full pipe nor be left open."""
    try:
        for line in stream:
            sink.append(line)
    finally:
        stream.close()


def lora_dir(runs, job_id):
    """Where the seed run wrote its LoRA checkpoints."""
    runs = Path(runs)
    found = sorted(runs.glob(f"{job_id}-*"))
    if not found:
        raise pj.ProjectError(f"no run folder under {runs} for {job_id}")
    return found[-1] / "output"


# ---------------------------------------------------------------------- command line
def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit and build a project's seed training set (spec 4 M3)")
    parser.add_argument("--project", required=True, help="project folder")
    parser.add_argument("--json", help="write the audit report to this file")
    parser.add_argument("--freeze-holdout", action="store_true", help="write the resolved holdout into the card")
    parser.add_argument("--migrate-review", action="store_true", help="tag review history and move stated reasons")
    parser.add_argument("--apply", action="store_true", help="with --freeze-holdout/--migrate-review: really write")
    parser.add_argument("--build", action="store_true", help="build the snapshot")
    parser.add_argument("--train", action="store_true",
                        help="with --build: train the LoRA on the new snapshot once it is published")
    parser.add_argument("--epochs", type=int, default=30,
                        help="with --train: passes over the snapshot (target_steps = images x epochs)")
    parser.add_argument("--rank", type=int, help="with --train: override the LoRA rank")
    parser.add_argument("--learning-rate", type=float, help="with --train: override the learning rate")
    parser.add_argument("--megapixels", type=float, help="with --train: override the training resolution budget")
    args = parser.parse_args(argv)
    if args.train and not args.build:
        parser.error("--train needs --build: it trains the snapshot that command writes")

    project = pj.Project.open(args.project)
    if args.migrate_review:
        project.migrate_review(apply=args.apply)
    if args.freeze_holdout:
        project.freeze_holdout(apply=args.apply)
    report = audit(project)
    counts = report["counts"]
    print(f"policy {report['policy']}  holdout {len(report['holdout']['ids'])} "
          f"({'frozen' if report['holdout']['frozen'] else 'not frozen'})")
    for key in ("sources", "candidates", "accepted", "accepted_from_m0", "kept", "kept_sources",
                "kept_candidates", "synthetic", "user", "excluded"):
        print(f"  {key:16s} {counts[key]}")
    for code, count in sorted(report["counts"]["by_code"].items()):
        print(f"    {code:24s} {count}")
    if report["kept"]:
        print("  kept images:")
        for entry in report["kept"]:
            print(f"    {entry['id']:22s} {entry['kind']:10s} {Path(entry['path']).name}")
    if args.json:
        pj.write_text(Path(args.json), json.dumps(report, ensure_ascii=False, indent=1))
        print(f"  report -> {args.json}")
    if args.build:
        snapshot = build_snapshot(project)
        print(f"snapshot {snapshot['id']} base={snapshot['base']} reused={snapshot['reused']}")
        if args.train:
            # Training needs the snapshot to exist first, and reuses it when it is already published and intact, so a
            # rerun after a crash trains the same images rather than rebuilding them under a new id.
            config = runner_config(project, snapshot, epochs=args.epochs, rank=args.rank,
                                   learning_rate=args.learning_rate, megapixels=args.megapixels)
            print(f"training {config['id']}: {len(snapshot['inputs'])} image(s) x {args.epochs} epochs "
                  f"= {config['target_steps']} steps, save every ~{config['save_every_steps_approx']}")
            runs = train(project, snapshot, config=config)
            print(f"run folder {runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
