"""M3-2: pick an epoch by looking at what each checkpoint actually draws (spec 3.2, D2).

A run's `output/` folder holds one LoRA per saved epoch. The loss curve says which one fits the training data best;
this module answers the other question - which one draws the character best - by drawing the same fixed prompts with
every checkpoint and putting the pictures side by side for a person to judge.

What it keeps apart, because the spec asks for it explicitly and because confusing them makes a decision unusable:

* the **run** the checkpoints came from (`runs/suamom_seed-8305a4e0160cbfaa`), which is not the project;
* the **checkpoint** itself, by path and SHA-256, since two epochs of one run are different files with the same name;
* the **training snapshot** - which images the run learned from - which is what makes a picture traceable later, and
  which the chosen epoch is recorded against;
* the **evaluation snapshot** - the images and prompts this comparison was made with - so a score can be re-derived.

Only checkpoints that are really usable are enumerated: a file has to carry a readable safetensors header whose
`ss_epoch` agrees with its name. A run interrupted mid-save leaves files that look right by name and are truncated
inside, and drawing 8 prompts with one of those wastes an hour before failing.

It never trains, never picks an epoch, and never decides. It writes the eval folder, the comparison sheet, and - when
a person names a winner - the record of that choice.

  python -m pipeline.epochs list   --project projects/suamom --run <run folder>
  python -m pipeline.epochs plan   --project projects/suamom --run <run folder>
  python -m pipeline.epochs run    --project projects/suamom --run <run folder>
  python -m pipeline.epochs sheet  --project projects/suamom --run <run folder> --serve
  python -m pipeline.epochs choose --project projects/suamom --run <run folder> --epoch 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

from . import comfy as comfy_mod
from . import krea2_gen, lineage, make_cancel, models

AUTO = Path(__file__).resolve().parents[1]          # automation

# The eight prompts every checkpoint is drawn with. They are the compositions the dataset has to cover and the ones a
# character LoRA most often gets wrong, in tool2's own tag vocabulary (`full body`, `from behind`, `three-quarter view`)
# because that is the language the seed was captioned in: a prompt in a different dialect would measure the prompt.
EVAL_PROMPTS = [
    {"id": "e01_face", "category": "framing", "prompt": "portrait, close-up, face focus, front view, looking at viewer"},
    {"id": "e02_upper", "category": "framing", "prompt": "upper body, front view, looking at viewer"},
    {"id": "e03_quarter", "category": "framing", "prompt": "upper body, three-quarter view, looking at viewer"},
    {"id": "e04_profile", "category": "angles", "prompt": "upper body, side view, profile, looking away"},
    {"id": "e05_back", "category": "angles", "prompt": "full body, from behind, standing"},
    {"id": "e06_full", "category": "framing", "prompt": "full body, front view, standing, looking at viewer"},
    {"id": "e07_smile", "category": "expressions", "prompt": "upper body, smile, looking at viewer"},
    {"id": "e08_sitting", "category": "poses", "prompt": "sitting, full body"},
]
# Two seeds: one picture per prompt can be a lucky draw, and the point of the grid is to compare checkpoints, not to
# find the best single picture in the folder.
EVAL_SEEDS = (1003, 1004)

HEADER_LIMIT = 8 * 1024 * 1024      # a safetensors header is a few hundred KB; anything near this is not one


class EpochError(RuntimeError):
    """A checkpoint folder that cannot be evaluated as asked."""


# --------------------------------------------------------------------------------------- which checkpoints exist
def header_of(path, *, limit=HEADER_LIMIT):
    """The safetensors header of `path` as a dict, or None when the file is not a readable one.

    The format is an 8-byte little-endian header length followed by that many bytes of JSON. A truncated file fails
    here, which is the whole point of reading it rather than trusting the name.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return None
            length = struct.unpack("<Q", raw)[0]
            if length <= 0 or length > limit or 8 + length > size:
                return None
            body = handle.read(length)
        if len(body) != length:
            return None
        data = json.loads(body.decode("utf-8"))
    except (OSError, ValueError, struct.error):
        return None
    return data if isinstance(data, dict) else None


def checkpoint_rows(folder):
    """Every LoRA in `folder` with the epoch it says it is, and why an unusable one is unusable.

    Returns a list of `{file, epoch, sha256, size, usable, note}`, sorted by epoch. A file whose name has no epoch
    number is listed with `epoch=None` instead of being silently dropped: it may be the final save, and a person
    looking at the folder needs to know it was seen and skipped.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise EpochError(f"no such folder: {folder}")
    rows = []
    for path in sorted(folder.glob("*.safetensors")):
        name_epoch = _epoch_from_name(path)
        header = header_of(path)
        stated = _epoch_from_header(header)
        row = {"file": str(path), "name": path.name, "epoch": name_epoch, "header_epoch": stated,
               "size": path.stat().st_size, "usable": False, "note": ""}
        if header is None:
            row["note"] = "not a readable safetensors file (truncated, or mid-save when the run stopped)"
        elif stated is None:
            row["note"] = "the header records no ss_epoch, so which epoch this file is cannot be confirmed"
        elif name_epoch is None:
            row["epoch"] = stated
            row["note"] = "the name carries no epoch; using the one in the header"
            row["usable"] = True
        elif int(stated) != int(name_epoch):
            row["note"] = f"the name says epoch {name_epoch}, the header says {stated}"
        else:
            row["usable"] = True
        if row["usable"]:
            row["sha256"] = lineage.sha256_file(path)
        rows.append(row)
    return rows


def _epoch_from_name(path):
    tail = Path(path).stem.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _epoch_from_header(header):
    if not header:
        return None
    value = (header.get("__metadata__") or {}).get("ss_epoch")
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def usable_checkpoints(folder):
    """The checkpoints that can be drawn with, by epoch. Raises when there are none, naming why each was refused."""
    rows = checkpoint_rows(folder)
    good = [row for row in rows if row["usable"]]
    if not good:
        if not rows:
            raise EpochError(f"no .safetensors files in {folder}")
        detail = "; ".join(f"{row['name']}: {row['note']}" for row in rows[:5])
        raise EpochError(f"no usable checkpoint in {folder} ({detail})")
    return good


# --------------------------------------------------------------------------------------------- the evaluation grid
def eval_dir(project, *, run_id=None):
    """Where the grid for this run lives: `experiments/epochs/<run name>/`."""
    name = run_id or "unset"
    return project.dir / "experiments" / "epochs" / name


def plan(project, *, run_folder, epochs=None, seeds=EVAL_SEEDS, prompts=None, size=None, snapshot=None,
         log=print):
    """Write the work list for one run's checkpoints. Returns the manifest, which `run()` then executes.

    `epochs` narrows the grid to named checkpoints; the default is every usable one, because an epoch that looks bad by
    loss can still be the one that draws the character best and leaving it out would hide it.
    """
    run_folder = Path(run_folder)
    output = run_folder / "output"
    candidates = usable_checkpoints(output if output.is_dir() else run_folder)
    if epochs:
        wanted = {int(e) for e in epochs}
        chosen = [row for row in candidates if int(row["epoch"]) in wanted]
        missing = sorted(wanted - {int(row["epoch"]) for row in chosen})
        if missing:
            raise EpochError(f"epoch(s) {', '.join(str(e) for e in missing)} are not usable checkpoints in "
                             f"{output}; available: {', '.join(str(r['epoch']) for r in candidates)}")
    else:
        chosen = candidates

    prompts = list(prompts or EVAL_PROMPTS)
    seeds = tuple(int(s) for s in seeds)
    size = dict(size or _size_from(project))
    manifest_path = eval_dir(project, run_id=run_folder.name) / "epochs.json"
    manifest = {
        "kind": "epoch_grid",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project": project.card.get("id"),
        "run": {"id": run_folder.name, "folder": str(run_folder),
                "state": _run_state(run_folder)},
        # Which images the run learned from. Kept apart from the checkpoints: it does not change between epochs, and
        # it is the thing a picture made with one of them has to be traceable to.
        "training_snapshot": snapshot or _snapshot_of(project, run_folder),
        "size": size,
        "seeds": list(seeds),
        "prompts": prompts,
        "checkpoints": [{k: row.get(k) for k in ("file", "name", "epoch", "sha256", "size")} for row in chosen],
        "tasks": [],
        "status": "planned",
    }
    for row in chosen:
        for prompt in prompts:
            for seed in seeds:
                manifest["tasks"].append({
                    "epoch": int(row["epoch"]),
                    "checkpoint": row["name"],
                    "checkpoint_sha256": row.get("sha256"),
                    "prompt_id": prompt["id"],
                    "category": prompt.get("category"),
                    "prompt": prompt_for(project.card, prompt),
                    "seed": seed,
                    "file": f"images/e{int(row['epoch']):03d}/{prompt['id']}__s{seed}.png",
                    "status": "pending",
                })
    write_manifest(manifest_path, manifest)
    log(f"[epochs] {len(chosen)} checkpoint(s) x {len(prompts)} prompt(s) x {len(seeds)} seed(s) "
        f"= {len(manifest['tasks'])} image(s) -> {manifest_path.parent}")
    return manifest


def _size_from(project):
    """The size M0 settled on, so the grid measures the checkpoints and not a resolution change."""
    inference = project.card.get("krea2_inference") or {}
    return dict(inference.get("size") or {"width": 1024, "height": 1536})


def _run_state(run_folder):
    path = Path(run_folder) / "state.json"
    if not path.is_file():
        return None
    data = _read_json(path)
    return {k: data.get(k) for k in ("complete", "active_stage", "epoch", "step") if k in data} or None


def _snapshot_of(project, run_folder):
    """Which seed snapshot this run was trained from, if the project can say.

    A seed run keeps its settings in `run_config.json` but never writes the snapshot id into it, so the id is read
    from the run's own path first: `seed_lora/<snapshot>/runs/<run>`. That is the layout `seed.train` creates, and
    the id is only accepted when `seed_lora/<id>` really holds a snapshot, so a run placed somewhere else cannot
    borrow a neighbouring name. Otherwise the run's config is read. When neither applies the answer stays null:
    guessing would put a false provenance stamp on every picture the grid produces.
    """
    folder = Path(run_folder)
    if folder.parent.name == "runs" and folder.parent.parent.parent.name == "seed_lora":
        candidate = folder.parent.parent.name
        if (folder.parent.parent / "snapshot.json").is_file():
            return {"id": candidate, "source": "run folder path"}
    for name in ("run_config.json", "config.json", "job.json", "run.json"):
        path = folder / name
        if not path.is_file():
            continue
        data = _read_json(path)
        for key in ("snapshot", "seed_snapshot", "dataset_snapshot"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return {"id": value, "source": path.name}
    return None


def manifest_path_of(project, run_folder):
    return eval_dir(project, run_id=Path(run_folder).name) / "epochs.json"


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise EpochError(f"no grid planned at {path}; run `plan` first")
    return _read_json(path)


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + os.urandom(4).hex())
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def prompt_for(card, prompt):
    """`trigger, caption_base, fixed_traits, outfit, the eval prompt`, in the order spec 3.4 fixes."""
    parts = [card.get("trigger", ""), card.get("caption_base", "")]
    parts += list(card.get("fixed_traits") or [])
    parts.append((card.get("outfits") or {}).get("default", ""))
    parts.append(prompt.get("prompt", ""))
    return ", ".join(p.strip() for p in parts if p and str(p).strip())


def pending(manifest):
    return [task for task in manifest["tasks"] if task["status"] in ("pending", "failed")]


def run(project, manifest, *, client, names, log=print, cancel=None):
    """Draw the pending cells. A cell whose file already exists is skipped, so rerunning adds no GPU work.

    That skip is the whole resumability story here: the grid is a few hundred images, an evaluation is interrupted by
    the same things a training run is, and a rerun has to continue rather than start again.
    """
    manifest_path = manifest_path_of(project, manifest["run"]["folder"])
    exp = manifest_path.parent
    done = skipped = failed = 0
    by_epoch = {int(row["epoch"]): row for row in manifest["checkpoints"]}
    uploaded = {}
    for task in manifest["tasks"]:
        out = exp / task["file"]
        if out.is_file():
            if task["status"] != "done":
                task["status"] = "done"
            skipped += 1
            continue
        if cancel is not None and cancel():
            log("cancelled")
            break
        started = time.time()
        try:
            checkpoint = by_epoch.get(int(task["epoch"]))
            if checkpoint is None:
                task["status"] = "failed"
                task["error"] = "the checkpoint for this epoch is not in the plan"
                failed += 1
                continue
            lora_name = client.name_for("lora", checkpoint["file"])
            if lora_name not in uploaded:
                uploaded[lora_name] = lora_name
            graph = krea2_gen.build(
                "b",
                unet=names["b_unet"], clip=names["clip"], vae=names["vae"], prompt=task["prompt"],
                width=manifest["size"]["width"], height=manifest["size"]["height"], seed=task["seed"],
                character_lora=lora_name, speed_lora=names["speed_lora"],
                filename_prefix=f"pipeline_epochs/e{int(task['epoch']):03d}_{task['prompt_id']}",
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            client.run(graph, out, slot_id=task["prompt_id"], method=f"krea2_b_e{int(task['epoch']):03d}",
                       seed=task["seed"])
            task["status"] = "done"
            done += 1
            log(f"  e{int(task['epoch']):03d} {task['prompt_id']} s{task['seed']}: {time.time() - started:.0f}s")
        except Exception as exc:  # noqa: BLE001 - one failed cell must not end a grid of hundreds
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            log(f"  e{int(task['epoch']):03d} {task['prompt_id']} s{task['seed']}: FAILED {task['error'][:200]}")
        finally:
            task["seconds"] = round(time.time() - started, 1)
            write_manifest(manifest_path, manifest)
    manifest["status"] = "ran"
    write_manifest(manifest_path, manifest)
    log(f"done {done}, skipped {skipped} already present, failed {failed}")
    return done, skipped, failed


# ---------------------------------------------------------------------------------------------------- the decision
def choose(project, *, run_folder, epoch, by="user", note="", log=print):
    """Record the epoch a person picked, keeping run id, checkpoint file+sha and training snapshot apart.

    The spec asks for three distinct things here and the distinction matters later: `epoch` alone says nothing about
    which run it came from, and a run can be retrained. The checkpoint's SHA-256 pins the exact bytes that were judged,
    and the training snapshot says which images that checkpoint learned from - which is what makes every picture the
    generator later produces traceable back to a seed set.
    """
    manifest = read_manifest(manifest_path_of(project, run_folder))
    rows = {int(row["epoch"]): row for row in manifest["checkpoints"]}
    if int(epoch) not in rows:
        raise EpochError(f"epoch {epoch} is not in this grid; the grid has "
                         f"{', '.join(str(e) for e in sorted(rows))}")
    row = rows[int(epoch)]
    path = Path(row["file"])
    if not path.is_file():
        raise EpochError(f"the checkpoint for epoch {epoch} is gone: {path}")
    digest = lineage.sha256_file(path)
    if digest != row.get("sha256"):
        raise EpochError(f"{path.name} changed since the grid was planned (sha256 {digest[:12]} != "
                         f"{str(row.get('sha256'))[:12]}); the pictures in the grid were made with other bytes")

    entry = {
        "epoch": int(epoch),
        "run_id": manifest["run"]["id"],
        "run_folder": manifest["run"]["folder"],
        "checkpoint": row["name"],
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "training_snapshot": (manifest.get("training_snapshot") or {}).get("id"),
        "eval": {"folder": str(manifest_path_of(project, run_folder).parent),
                 "size": manifest["size"], "seeds": manifest["seeds"],
                 "prompts": [p["id"] for p in manifest["prompts"]]},
        "decided": {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": by, "note": note},
    }
    with project.guard():
        inference = dict(project.card.get("krea2_inference") or {})
        inference["epoch"] = int(epoch)
        inference["selected"] = entry
        inference["epoch_note"] = (f"{manifest['run']['id']}의 에포크 {epoch} "
                                   f"({row['name']}, sha256 {digest[:12]}) — 평가 그리드로 선택")
        project.card["krea2_inference"] = inference
        project.card["epoch_note"] = entry
        project.save()
    log(f"[epochs] epoch {epoch} of {manifest['run']['id']} recorded "
        f"(checkpoint {row['name']}, training snapshot {(entry['training_snapshot'] or 'unknown')})")
    return entry


def chosen(project, run_folder=None):
    """The recorded choice, or None. Checks it still matches the file it names."""
    entry = (project.card.get("krea2_inference") or {}).get("selected")
    if not entry:
        return None
    if run_folder is not None and entry.get("run_id") != Path(run_folder).name:
        return None
    path = Path(entry.get("checkpoint_path") or "")
    fresh = lineage.sha256_file(path) if path.is_file() else None
    return {**entry, "checkpoint_present": bool(fresh), "checkpoint_matches": fresh == entry.get("checkpoint_sha256")}


# ------------------------------------------------------------------------------------------------- the comparison
_PAGE_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 16px 20px 60px; background: #14161a; color: #e6e6e6; }}
 h1 {{ font-size: 18px; margin: 0 0 4px; }}
 .hint {{ color: #9aa4b2; margin: 0 0 14px; }}
 .hint b {{ color: #e6e6e6; }}
 table {{ border-collapse: collapse; }}
 th, td {{ border: 1px solid #2a2f37; padding: 6px; vertical-align: top; }}
 th.row {{ text-align: left; min-width: 190px; background: #1b1f25; }}
 th.col {{ background: #1b1f25; }}
 th.col.best {{ background: #1f3a2b; }}
 .meta {{ color: #8b95a3; font-size: 12px; }}
 .prompt {{ color: #b9c2cf; font-size: 12px; margin-top: 4px; }}
 .cell img {{ width: 190px; height: auto; display: block; border-radius: 4px; }}
 .cell.empty, .cell.missing {{ width: 190px; min-height: 120px; color: #6b7480; }}
 .ph {{ padding: 30px 6px; text-align: center; font-size: 12px; }}
 tr.prompts td {{ background: #171a1f; }}
 .missing-flag {{ color: #f0b46b; }}
</style></head><body>
<h1>{title}</h1>
<p class="hint">{summary}</p>
<table><thead><tr><th class="row">프롬프트</th>{head}</tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
"""


def write_sheet(project, run_folder, log=print):
    """Write comparison.html: prompt rows x checkpoint columns, so one composition can be compared down a row.

    Rows are prompts, not checkpoints: the question is "which epoch draws this pose right", and answering it means
    seeing the same composition at every epoch next to each other.
    """
    manifest = read_manifest(manifest_path_of(project, run_folder))
    exp = manifest_path_of(project, run_folder).parent
    by_key = {(t["prompt_id"], int(t["epoch"]), int(t["seed"])): t for t in manifest["tasks"]}
    epochs = sorted({int(row["epoch"]) for row in manifest["checkpoints"]})
    seeds = [int(s) for s in manifest["seeds"]]
    choice = chosen(project) or {}
    picked = choice.get("epoch")
    head = "".join(
        f'<th class="col{" best" if picked == e else ""}">에포크 {e}'
        f'<div class="meta">{html_escape(_short_sha(manifest, e))}'
        f'{" · 선택됨" if picked == e else ""}</div></th>'
        for e in epochs)
    rows = []
    for prompt in manifest["prompts"]:
        cells = []
        for epoch in epochs:
            inner = []
            for seed in seeds:
                task = by_key.get((prompt["id"], epoch, seed))
                if task is None:
                    inner.append('<div class="cell empty">-</div>')
                    continue
                rel = Path(task["file"]).as_posix()
                if not (exp / task["file"]).is_file():
                    inner.append(f'<div class="cell missing"><div class="ph">'
                                 f'{html_escape(task.get("error") or task["status"])}</div></div>')
                    continue
                inner.append(f'<div class="cell"><a href="{rel}" target="_blank">'
                             f'<img loading="lazy" src="{rel}" alt="e{epoch} {prompt["id"]} s{seed}"></a>'
                             f'<div class="meta">s{seed}</div></div>')
            cells.append(f"<td>{''.join(inner)}</td>")
        rows.append(f'<tr><th class="row">{html_escape(prompt["id"])}'
                    f'<div class="meta">{html_escape(prompt.get("category") or "")}</div>'
                    f'<div class="prompt">{html_escape(prompt["prompt"])}</div></th>{"".join(cells)}</tr>')
    missing = sum(1 for t in manifest["tasks"] if not (exp / t["file"]).is_file())
    summary = (f'{html_escape(manifest["run"]["id"])} · 체크포인트 {len(epochs)}개 · '
               f'프롬프트 {len(manifest["prompts"])}개 × 시드 {len(seeds)}개 · '
               f'{manifest["size"]["width"]}×{manifest["size"]["height"]}')
    if missing:
        summary += f' · <span class="missing-flag">아직 없는 칸 {missing}개</span>'
    summary += f'<br>학습 스냅샷: <b>{html_escape((manifest.get("training_snapshot") or {}).get("id") or "기록 없음")}</b>'
    if picked:
        summary += f' · 선택된 에포크: <b>{picked}</b>'
    else:
        summary += (' · 아직 고르지 않음 — CLI: <b>choose --epoch N</b>')
    path = exp / "comparison.html"
    path.write_text(_PAGE_TEMPLATE.format(
        title=html_escape(f"에포크 비교 — {project.card.get('id', '')}"),
        summary=summary, head=head, rows="".join(rows)), encoding="utf-8")
    log(f"[epochs] {path}")
    return path


def _short_sha(manifest, epoch):
    for row in manifest["checkpoints"]:
        if int(row["epoch"]) == int(epoch):
            return (row.get("sha256") or "")[:12]
    return ""


def html_escape(value):
    from html import escape
    return escape(str(value), quote=True)


# ------------------------------------------------------------------------------------------------------- plumbing
def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def find_run(project, run_id=None):
    """The run folder to evaluate: the one named, or the only one the project has.

    The seed snapshot id and the run id are different things and neither can be derived from the other, so this only
    ever returns a folder it actually found.
    """
    base = project.dir / "seed_lora"
    runs = sorted((base).glob("*/runs/*")) if base.is_dir() else []
    runs = [path for path in runs if path.is_dir()]
    if run_id:
        for path in runs:
            if path.name == run_id or path.parent.parent.name == run_id:
                return path
        raise EpochError(f"no run {run_id!r} under {base}")
    if not runs:
        raise EpochError(f"no training run under {base}; train a seed first")
    if len(runs) > 1:
        raise EpochError("more than one run under " + str(base) + "; name one with --run: "
                         + ", ".join(path.name for path in runs))
    return runs[0]


def resolve_names(client, log=print):
    """The ComfyUI names for the (b) graph: RAW DiT + Turbo LoRA + the character LoRA per epoch."""
    resolved = models.resolve("krea2")
    names = {
        "b_unet": client.name_for("unet", resolved["krea2_raw_dit"]),
        "speed_lora": client.name_for("lora", resolved["krea2_turbo_lora"]),
        "clip": client.name_for("clip", resolved["krea2_text_encoder"]),
        "vae": client.name_for("vae", resolved["krea2_vae"]),
    }
    log("[epochs] " + ", ".join(f"{k}={v}" for k, v in names.items()))
    return names


def make_client(url=None):
    """A ComfyUI client that journals before it POSTs, so a crash mid-grid can be picked up rather than guessed at."""
    settings = models.comfy_settings()
    return comfy_mod.Comfy(url or settings["url"], comfy_root=settings["root"],
                           timeout=int(settings.get("request_timeout") or 1800),
                           journal=(AUTO / "state" / "prompt_journal"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pipeline.epochs", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("list", "plan", "run", "sheet", "choose", "status"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", help="run folder name (or the snapshot id holding it); default: the only one")
    parser.add_argument("--epochs", help="comma-separated epochs to draw; default: every usable checkpoint")
    parser.add_argument("--epoch", type=int, help="the epoch to record (choose)")
    parser.add_argument("--seeds", help=f"comma-separated seeds; default: {','.join(map(str, EVAL_SEEDS))}")
    parser.add_argument("--note", default="", help="why this epoch was chosen (choose)")
    parser.add_argument("--serve", action="store_true", help="open the sheet in a browser (sheet)")
    parser.add_argument("--comfy", default=None, help="ComfyUI base URL")
    parser.add_argument("--cancel-file", help="stop before the next cell once this file exists")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from . import project as pj
    project = pj.Project.open(args.project)
    run_folder = find_run(project, args.run)

    if args.command == "list":
        output = run_folder / "output"
        rows = checkpoint_rows(output if output.is_dir() else run_folder)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=1))
            return 0
        for row in rows:
            flag = "ok  " if row["usable"] else "SKIP"
            print(f"{flag} epoch={row['epoch']} {row['name']} {row['size']:,}B"
                  + (f"  {row['note']}" if row["note"] else ""))
        return 0

    if args.command == "status":
        path = manifest_path_of(project, run_folder)
        if not path.is_file():
            print(f"no grid planned for {run_folder.name}")
            return 0
        manifest = read_manifest(path)
        drawn = sum(1 for t in manifest["tasks"] if (path.parent / t["file"]).is_file())
        entry = chosen(project)
        print(f"run {manifest['run']['id']}: {len(manifest['checkpoints'])} checkpoint(s), "
              f"{drawn}/{len(manifest['tasks'])} image(s)")
        print("chosen: " + (json.dumps(entry, ensure_ascii=False, indent=1) if entry else "none"))
        return 0

    if args.command == "plan":
        epochs = [int(v) for v in args.epochs.split(",")] if args.epochs else None
        seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else EVAL_SEEDS
        manifest = plan(project, run_folder=run_folder, epochs=epochs, seeds=seeds)
        if args.json:
            print(json.dumps({"run": manifest["run"], "checkpoints": manifest["checkpoints"],
                              "tasks": len(manifest["tasks"])}, ensure_ascii=False, indent=1))
        return 0

    if args.command == "run":
        manifest = read_manifest(manifest_path_of(project, run_folder))
        client = make_client(args.comfy)
        if not client.ping():
            raise EpochError(f"ComfyUI is not answering at {client.url}; start it (see HANDOFF 8.1 for the UTF-8 "
                             "requirement)")
        problems = models.problems("krea2", require_optional=("krea2_turbo_lora",))
        if problems:
            raise EpochError("model problems: " + "; ".join(problems))
        names = resolve_names(client)
        # A GUI (or a person at a terminal) pauses by creating this file; the check happens between cells, so every
        # picture already drawn stays a complete picture and a rerun picks up from there.
        cancel = make_cancel(args.cancel_file)
        run(project, manifest, client=client, names=names, cancel=cancel)
        if cancel is not None and cancel():
            print("paused: the remaining cells are still pending, so this command continues them")
        return 0

    if args.command == "sheet":
        path = write_sheet(project, run_folder)
        if args.serve:
            import webbrowser
            webbrowser.open(path.as_uri())
        return 0

    if args.command == "choose":
        if args.epoch is None:
            raise EpochError("choose needs --epoch N")
        choose(project, run_folder=run_folder, epoch=args.epoch, note=args.note)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
