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


# The manifest's own identity. Two plans are the same plan only when every one of these agrees (spec N3.1): the run,
# the checkpoint bytes, the images the run learned from, the prompts, the seeds, the size, and the graph/model
# fingerprints. Without this the file's existence was the whole test, so re-planning after a checkpoint was retrained
# re-labelled the old pictures as the new checkpoint's evidence.
PLAN_IDENTITY_KEYS = ("run", "training_snapshot", "size", "seeds", "prompts", "checkpoints", "graph", "models")


def plan_identity(manifest):
    """The canonical JSON of everything that decides what a cell's picture means.

    Deliberately excludes `created_at`, `tasks` and `status`: those describe this manifest's own run, not what a
    picture in it is evidence of.
    """
    payload = {key: manifest.get(key) for key in PLAN_IDENTITY_KEYS}
    payload["checkpoints"] = [{"epoch": int(row.get("epoch")) if row.get("epoch") is not None else None,
                               "name": row.get("name"), "sha256": row.get("sha256"),
                               "size": row.get("size")}
                              for row in (manifest.get("checkpoints") or [])]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def plan_id(manifest):
    """A short, stable name for this plan's identity. Used to keep different plans' pictures apart on disk."""
    return hashlib.sha256(plan_identity(manifest).encode("utf-8")).hexdigest()[:12]


def graph_fingerprint():
    """What the drawing graph is, independent of which checkpoint it loads.

    The builder's own source is the fingerprint: a prompt that draws one way today has to be recognisable as a
    different graph tomorrow, and a version string somebody remembers to bump does not do that.
    """
    source = Path(krea2_gen.__file__).read_bytes()
    return {"builder": Path(krea2_gen.__file__).name,
            "sha256": hashlib.sha256(source).hexdigest()[:16]}


def model_fingerprint(family="krea2"):
    """The model files this grid will load, by path and size, or an empty dict when they cannot be resolved.

    Not a load and not a hash of gigabytes: the resolved paths and their sizes are what a wrong model swap changes,
    and `models.problems()` already checks existence. A resolver failure is recorded as unknown rather than raised -
    `plan` without a configured ComfyUI is a legitimate dry run.
    """
    try:
        rows = models.rows(family)
    except Exception:                      # noqa: BLE001 - an unreadable config must not stop a plan
        return {}
    out = {}
    for row in rows or []:
        name = row.get("role") or row.get("name")
        path = row.get("path")
        if not name or not path:
            continue
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = None
        out[str(name)] = {"path": str(path), "size": size}
    return out


def plan(project, *, run_folder, epochs=None, seeds=EVAL_SEEDS, prompts=None, size=None, snapshot=None,
         log=print):
    """Write the work list for one run's checkpoints. Returns the manifest, which `run()` then executes.

    `epochs` narrows the grid to named checkpoints; the default is every usable one, because an epoch that looks bad by
    loss can still be the one that draws the character best and leaving it out would hide it.

    A re-plan of a *different* identity writes itself beside the old plan instead of over it (spec N3.1): the previous
    manifest and every picture it accounted for stay where they are, under their own `plan-<id>` folder. Only the
    current plan lives at `epochs.json`; the old ones are `epochs-<id>.json` next to it. Without that, re-planning after
    a checkpoint changed made the next `run()` see the old PNGs, call them "already done", and `choose()` then accepted
    bytes that never drew them.
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
    folder = eval_dir(project, run_id=run_folder.name)
    manifest_path = folder / "epochs.json"
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
        # What drew the pictures, beyond which checkpoint: the builder source and the model files (spec N3.1).
        "graph": graph_fingerprint(),
        "models": model_fingerprint(),
        "tasks": [],
        "status": "planned",
    }
    manifest["plan_id"] = plan_id(manifest)
    # Cells land under the plan's own folder, so a picture from one plan can never be read as another's work.
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
                    "file": f"plan-{manifest['plan_id']}/e{int(row['epoch']):03d}/{prompt['id']}__s{seed}.png",
                    "status": "pending",
                })
    folder.mkdir(parents=True, exist_ok=True)
    previous = None
    if manifest_path.is_file():
        try:
            previous = _read_json(manifest_path)
        except (OSError, ValueError):
            previous = None
    if previous and plan_id(previous) != manifest["plan_id"]:
        # Preserve the old plan under its own name before the current one takes `epochs.json`. The old pictures are
        # already in `plan-<old id>/`, which this plan does not read, so they stay with the manifest that names them.
        archive = folder / f"epochs-{plan_id(previous)}.json"
        if not archive.is_file():
            write_manifest(archive, previous)
        log(f"[epochs] an earlier plan ({plan_id(previous)}) is kept at {archive.name}; "
            f"it drew {sum(1 for t in previous.get('tasks') or [] if t.get('status') == 'done')} cell(s)")
        manifest["supersedes"] = {"plan_id": plan_id(previous), "manifest": archive.name}
    write_manifest(manifest_path, manifest)
    log(f"[epochs] {len(chosen)} checkpoint(s) x {len(prompts)} prompt(s) x {len(seeds)} seed(s) "
        f"= {len(manifest['tasks'])} image(s) -> {manifest_path.parent} (plan {manifest['plan_id']})")
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
    """The card's words around the eval prompt, built as batch B builds its own (prompting.build): an epoch is
    judged on the pictures it will actually be asked for."""
    from . import prompting
    return prompting.build(card, prompt.get("prompt", ""))


def pending(manifest):
    return [task for task in manifest["tasks"] if task["status"] in ("pending", "failed")]


def _sha256_of(path):
    import hashlib as _hashlib
    digest = _hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_state(out, task):
    """Is this cell's picture really the plan's picture? (spec N3.2)

    A file's existence was the whole test before, so a truncated write, a decode failure or a PNG left by a different
    plan all counted as "already drawn". The checks here are the cheap ones that catch exactly those: the file decodes,
    its recorded SHA matches when one was recorded, and its checkpoint SHA still equals the plan's. A cell that fails
    any of them is *not* done - it is re-drawn (or reported, never silently adopted).

    Returns `(state, reason)` where state is "done", "redraw" or "adopt" (a decodable file with no recorded SHA, which
    can only be the leftover of an interrupted save of this same plan's own picture).
    """
    if not out.is_file():
        return "pending", None
    try:
        from PIL import Image
        with Image.open(out) as image:
            image.load()
    except Exception as exc:               # noqa: BLE001 - a broken file is the point, not a crash
        return "redraw", f"the file does not decode ({type(exc).__name__})"
    recorded = task.get("sha256")
    if recorded:
        actual = _sha256_of(out)
        if actual != recorded:
            return "redraw", "the bytes are not the ones this plan recorded"
        return "done", None
    # No SHA recorded: the manifest was written after the picture but before the hash (an interrupted save). The file
    # decodes and its checkpoint is checked by the caller, so it can be adopted - but only by recording what it is,
    # never by leaving it unaccounted for.
    return "adopt", None


def validate_manifest(manifest):
    """Refuse a grid manifest that cannot be drawn from, naming what is missing.

    A hand-edited, half-written or older manifest used to fail deep inside the loop with a bare `KeyError: 'epoch'`,
    which says nothing about which file is wrong or what a person should do. Every field the loop reads is checked
    here, once, with its own message. This is not a schema: it is the list of keys `run()` cannot proceed without.
    """
    required = ("run", "tasks", "checkpoints", "size", "seeds", "prompts")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise EpochError(f"the grid manifest is incomplete; missing {', '.join(missing)} - re-run `plan` to write a "
                         "complete one (nothing already drawn is lost: the pictures stay under their plan folder)")
    if not isinstance(manifest["run"], dict) or not manifest["run"].get("folder"):
        raise EpochError("the grid manifest has no run folder; it cannot say which run's pictures these are")
    for index, task in enumerate(manifest["tasks"]):
        absent = [key for key in ("epoch", "file", "prompt_id", "seed") if key not in task]
        if absent:
            raise EpochError(f"task {index} of the grid manifest is incomplete; missing {', '.join(absent)} - "
                             "re-run `plan`")
    for row in manifest["checkpoints"]:
        if "epoch" not in row or "file" not in row:
            raise EpochError("a checkpoint row in the grid manifest has no epoch or file; re-run `plan`")
    if "width" not in manifest["size"] or "height" not in manifest["size"]:
        raise EpochError("the grid manifest has no width/height; re-run `plan`")


def run(project, manifest, *, client, names, log=print, cancel=None):
    """Draw the pending cells. A cell whose file already exists *and checks out* is skipped (spec N3.2).

    That skip is the whole resumability story here: the grid is a few hundred images, an evaluation is interrupted by
    the same things a training run is, and a rerun has to continue rather than start again. What was a bare
    `out.is_file()` is now `cell_state()`: a truncated PNG, a file whose bytes do not match the recorded SHA, or one
    left by a plan with other checkpoints is re-drawn instead of being counted as work already done.

    Returns `(done, skipped, failed)`.
    """
    validate_manifest(manifest)
    manifest_path = manifest_path_of(project, manifest["run"]["folder"])
    exp = manifest_path.parent
    done = skipped = failed = 0
    by_epoch = {int(row["epoch"]): row for row in manifest["checkpoints"]}
    uploaded = {}
    for task in manifest["tasks"]:
        out = exp / task["file"]
        state, reason = cell_state(out, task)
        if state == "redraw":
            log(f"  e{int(task['epoch']):03d} {task['prompt_id']} s{task['seed']}: redrawing ({reason})")
        if state in ("done", "adopt"):
            checkpoint = by_epoch.get(int(task["epoch"]))
            # The picture is only this plan's if the checkpoint it was drawn with still has the planned bytes. A
            # retrained checkpoint under the same name is a different epoch, not a resumable one.
            if checkpoint and checkpoint.get("sha256") and Path(checkpoint["file"]).is_file() \
                    and lineage.sha256_file(checkpoint["file"]) != checkpoint.get("sha256"):
                state = "redraw"
                reason = "the checkpoint's bytes changed since this plan was written"
                log(f"  e{int(task['epoch']):03d} {task['prompt_id']} s{task['seed']}: redrawing ({reason})")
            else:
                if state == "adopt":
                    task["sha256"] = _sha256_of(out)
                    log(f"  e{int(task['epoch']):03d} {task['prompt_id']} s{task['seed']}: adopted an unhashed "
                        "picture from this plan (recorded its sha256)")
                task["status"] = "done"
                task.pop("error", None)
                skipped += 1
                write_manifest(manifest_path, manifest)
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
            # Verify what the client wrote before calling it done: a client that returns without a complete picture
            # (a truncated download) must not leave a cell marked "done" for the next resume to trust.
            check, why = cell_state(out, {k: v for k, v in task.items() if k != "sha256"})
            if check == "redraw":
                raise EpochError(f"the written file is not usable: {why}")
            task["sha256"] = _sha256_of(out)
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
    # The status says what actually happened, so a caller cannot read a half-drawn grid as a finished one (spec N3.3).
    if failed and not done and not skipped:
        manifest["status"] = "failed"
    elif failed or any(t["status"] in ("pending", "failed") for t in manifest["tasks"]):
        manifest["status"] = "partial"
    else:
        manifest["status"] = "ran"
    write_manifest(manifest_path, manifest)
    log(f"done {done}, skipped {skipped} already present, failed {failed} ({manifest['status']})")
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
    # A plan written before _snapshot_of knew where a seed run keeps its snapshot carries null there. Null means the
    # information was absent, not that someone decided it should stay unknown, so it is re-read rather than copied into
    # the card forever. Re-planning would fix it too, but planning rebuilds the manifest and would discard which cells
    # of the grid have already been drawn.
    if not manifest.get("training_snapshot"):
        found = _snapshot_of(project, manifest["run"]["folder"])
        if found:
            manifest["training_snapshot"] = found
            write_manifest(manifest_path_of(project, run_folder), manifest)
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
                 "plan_id": manifest.get("plan_id"),
                 "size": manifest["size"], "seeds": manifest["seeds"],
                 "prompts": [p["id"] for p in manifest["prompts"]]},
        # Which cells of this epoch actually backed the comparison, and which did not (spec N3.3): a choice recorded
        # over a half-drawn grid is still a choice, but it must not read as if every cell had been seen.
        "grid": grid_state(manifest, epoch=int(epoch),
                           folder=manifest_path_of(project, run_folder).parent),
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


def grid_state(manifest, *, epoch=None, folder=None):
    """What the grid actually holds for one epoch (or all of them): drawn, pending, failed, unverified.

    `choose()` records this so a person reading the card later can see whether the epoch was judged on a full grid or
    on three cells out of sixteen - the number the old manifest could not answer, because `status` was just "ran"
    whatever happened.
    """
    tasks = [t for t in manifest["tasks"] if epoch is None or int(t["epoch"]) == int(epoch)]
    drawn = [t for t in tasks if t.get("status") == "done"]
    return {
        "cells": len(tasks),
        "drawn": len(drawn),
        "pending": len([t for t in tasks if t.get("status") == "pending"]),
        "failed": len([t for t in tasks if t.get("status") == "failed"]),
        # A drawn cell whose recorded SHA cannot be re-read is not evidence anyone can check.
        "unverified": len([t for t in drawn if not t.get("sha256")]),
        "status": manifest.get("status"),
        "plan_id": manifest.get("plan_id"),
        "folder": str(folder) if folder is not None else None,
    }


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
 .clip-value {{ color: #9cf; }}
 .meta.best-clip {{ color: #7fe3a0; font-weight: 600; }}
</style></head><body>
<h1>{title}</h1>
<p class="hint">{summary}</p>
<table><thead><tr><th class="row">프롬프트</th>{head}</tr></thead>
<tbody>{rows}</tbody></table>
{clip_tools}
</body></html>
"""


def sheet_clip_path(project, run_folder):
    return manifest_path_of(project, run_folder).parent / "grid_clip.json"


def clip_cache_identity(project, run_folder, manifest):
    """What a cell score is a measurement *of* (spec N3.4).

    A score keyed only by `(prompt_id, epoch, seed)` survives every change that invalidates it: new bytes for the
    picture, an edited original it is compared against, a re-cropped reference, a different scorer. So the cache is
    keyed by this identity instead - the picture's SHA, the reference set's SHAs, the crop policy, and the scorer's
    own name/version - and a mismatched identity means "measure again", never "reuse the old number".
    """
    from . import score as score_mod
    references = []
    for path in score_mod.reference_paths(project):
        try:
            references.append({"path": str(path), "sha256": lineage.sha256_file(path)})
        except OSError:
            references.append({"path": str(path), "sha256": None})
    return {
        "plan_id": manifest.get("plan_id"),
        "references": references,
        # The crop policy in force. There is one today (whole frame); recording it means a future cropped score
        # cannot be mistaken for a whole-frame one.
        "crop": "whole-frame",
        "scorer": clip_vision_descriptor(),
    }


def clip_vision_descriptor():
    """The scorer's identity: its module name, its model id, and the file it loaded from."""
    from .scorers import clip_vision
    path = Path(getattr(clip_vision, "__file__", "") or "")
    return {
        "module": "pipeline.scorers.clip_vision",
        "model": getattr(clip_vision, "MODEL_ID", None) or getattr(clip_vision, "MODEL", None),
        "source_sha256": lineage.sha256_file(path) if path.is_file() else None,
    }


def sheet_clip_scores(project, run_folder, *, embedder=None, log=print):
    """{(prompt_id, epoch, seed): clip_max} for every drawn cell, against the user's originals (spec R7 P2, N3.4).

    This is the auxiliary sort the comparison sheet offers: it says which epoch drew a composition closest to the
    character, per cell, so the human eye starts at the strong candidates. It is a measurement laid over the grid,
    never a decision. The scores are written beside the sheet so it can be re-rendered without touching the GPU.

    The sidecar records the identity it was measured under and the SHA of every cell it scored. A cache whose identity
    does not hold is not reused: it is replaced, because a stale number here is a wrong hint next to a person's
    judgement.
    """
    from . import score as score_mod
    from .scorers import clip_vision
    manifest = read_manifest(manifest_path_of(project, run_folder))
    folder = manifest_path_of(project, run_folder).parent
    refs = score_mod.reference_paths(project)
    if not refs:
        raise EpochError("no user originals to compare against, so the grid cannot be scored")
    cells = [((t["prompt_id"], int(t["epoch"]), int(t["seed"])), folder / t["file"], t)
             for t in manifest["tasks"] if (folder / t["file"]).is_file()]
    if not cells:
        raise EpochError("no drawn cell to score; run the grid first")
    identity = clip_cache_identity(project, run_folder, manifest)
    embedder = embedder or clip_vision.Embedder()
    scores = {}
    batch = 8
    for start in range(0, len(cells), batch):
        chunk = cells[start:start + batch]
        result = clip_vision.identity_scores(
            [path for _key, path, _task in chunk], refs,
            ids=["|".join(str(part) for part in key) for key, _path, _task in chunk], embedder=embedder)
        for row in result["rows"]:
            prompt_id, epoch, seed = str(row["id"]).split("|")
            scores[(prompt_id, int(epoch), int(seed))] = row.get("max_cosine")
    sidecar = sheet_clip_path(project, run_folder)
    payload = {
        "identity": identity,
        "cells": {"|".join(str(part) for part in key):
                  {"max": value,
                   "sha256": lineage.sha256_file(path) if path.is_file() else None}
                  for key, path, _task in cells for value in [scores.get(key)]},
    }
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[epochs] {len(scores)} cell score(s) -> {sidecar}")
    return scores



def _read_clip_sidecar(project, run_folder, *, manifest=None, log=print):
    """The cached CLIP sort, but only when it is still a measurement of *these* cells (spec N3.4).

    The old reader returned whatever numbers the file held, keyed by `(prompt_id, epoch, seed)`. That key does not
    change when the picture behind it is redrawn, when an original it is compared against is edited, or when the
    scorer changes, so a stale hint could sit next to a person's judgement for ever. Now the sidecar carries the
    identity it was measured under and each cell's SHA; a sidecar that cannot prove it measured these bytes is
    discarded with a reason, and the sheet is rendered without the sort rather than with a wrong one.
    """
    sidecar = sheet_clip_path(project, run_folder)
    if not sidecar.is_file():
        return None
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    cells = raw.get("cells")
    if not isinstance(cells, dict):
        # The pre-N3 shape: bare {key: value}. It cannot prove what it measured, so it is not reused.
        log("[epochs] the CLIP sidecar predates the identity check; ignoring it (re-run sheet --clip)")
        return None
    if manifest is None:
        try:
            manifest = read_manifest(manifest_path_of(project, run_folder))
        except EpochError:
            return None
    recorded = raw.get("identity")
    if recorded != clip_cache_identity(project, run_folder, manifest):
        log("[epochs] the CLIP sidecar was measured under a different identity (references, crop policy, scorer or "
            "plan changed); ignoring it (re-run sheet --clip)")
        return None
    folder = manifest_path_of(project, run_folder).parent
    by_key = {"|".join((t["prompt_id"], str(int(t["epoch"])), str(int(t["seed"])))): t
              for t in manifest["tasks"]}
    out = {}
    for key, row in cells.items():
        parts = str(key).split("|")
        if len(parts) != 3 or not isinstance(row, dict) or row.get("max") is None:
            continue
        task = by_key.get(key)
        if task is None:
            continue
        picture = folder / task["file"]
        # The picture scored must still be the picture on disk. A redrawn cell invalidates its own number.
        current = lineage.sha256_file(picture) if picture.is_file() else None
        if row.get("sha256") and current != row.get("sha256"):
            log(f"[epochs] cell {key} changed since it was scored; dropping its cached score")
            continue
        out[(parts[0], int(parts[1]), int(parts[2]))] = float(row["max"])
    return out



_CLIP_SORT_TOOLS = """
<p style="margin: 10px 0 0">
 정렬 <select id="clip-sort">
  <option value="">기본(프롬프트 순)</option>
  <option value="clip-desc">CLIP 높은 순(각 프롬프트의 최고 점수)</option>
  {options}
 </select>
 <span id="clip-note" style="color:#789;font-size:12px"></span>
</p>
<script>
(function () {{
  const sorter = document.getElementById('clip-sort');
  if (!sorter) return;
  const body = document.querySelector('tbody');
  const original = Array.from(body.querySelectorAll('tr')).filter(tr => tr.querySelector('td'));
  const bestOf = tr => Math.max(-1, ...Array.from(tr.querySelectorAll('.cell[data-clip]'))
    .map(cell => parseFloat(cell.dataset.clip)).filter(v => !isNaN(v)));
  sorter.addEventListener('change', () => {{
    const rows = original.slice();
    if (sorter.value === 'clip-desc') {{
      rows.sort((a, b) => bestOf(b) - bestOf(a));
    }} else if (sorter.value.startsWith('clip-desc:')) {{
      const epoch = parseInt(sorter.value.split(':')[1], 10);
      const at = tr => {{
        const index = Array.from(tr.children).findIndex(cell => cell.textContent.includes('에포크 ' + epoch));
        const cell = tr.children[index] && tr.children[index].querySelector('.cell[data-clip]');
        return cell ? parseFloat(cell.dataset.clip) : -1;
      }};
      rows.sort((a, b) => at(b) - at(a));
    }}
    rows.forEach(row => body.appendChild(row));
    document.getElementById('clip-note').textContent = 'CLIP은 눈의 보조일 뿐입니다';
  }});
}})();
</script>
"""


def write_sheet(project, run_folder, log=print, clip_scores=None):
    """Write comparison.html: prompt rows x checkpoint columns, so one composition can be compared down a row.

    Rows are prompts, not checkpoints: the question is "which epoch draws this pose right", and answering it means
    seeing the same composition at every epoch next to each other. When CLIP scores are available - computed by
    `sheet_clip_scores` and cached beside the sheet, or passed in - each cell carries its score, the strongest cell of
    a row is marked, and the rows can be sorted by score (spec R7 P2). The scores are an aid for the eye; the choice
    stays a person's.
    """
    if clip_scores is None:
        clip_scores = _read_clip_sidecar(project, run_folder, log=log)
    manifest = read_manifest(manifest_path_of(project, run_folder))
    exp = manifest_path_of(project, run_folder).parent
    by_key = {(t["prompt_id"], int(t["epoch"]), int(t["seed"])): t for t in manifest["tasks"]}
    epochs = sorted({int(row["epoch"]) for row in manifest["checkpoints"]})
    seeds = [int(s) for s in manifest["seeds"]]
    choice = chosen(project) or {}
    picked = choice.get("epoch")
    def clip_of(prompt_id, epoch, seed):
        if not clip_scores:
            return None
        return clip_scores.get((prompt_id, epoch, seed))

    column_means = {}
    if clip_scores:
        for epoch in epochs:
            values = [clip_of(prompt["id"], epoch, seed) for prompt in manifest["prompts"] for seed in seeds]
            values = [v for v in values if v is not None]
            column_means[epoch] = round(sum(values) / len(values), 4) if values else None
    head = "".join(
        f'<th class="col{" best" if picked == e else ""}">에포크 {e}'
        f'<div class="meta">{html_escape(_short_sha(manifest, e))}'
        f'{" · 선택됨" if picked == e else ""}</div>'
        + (f'<div class="meta">CLIP 평균 {column_means[e]:.3f}</div>' if column_means.get(e) is not None else "")
        + '</th>'
        for e in epochs)
    rows = []
    for prompt in manifest["prompts"]:
        row_clips = {(epoch, seed): clip_of(prompt["id"], epoch, seed) for epoch in epochs for seed in seeds}
        scored = [(key, value) for key, value in row_clips.items() if value is not None]
        best = max(scored, key=lambda pair: pair[1])[0] if scored else None
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
                value = row_clips[(epoch, seed)]
                is_best = best is not None and (epoch, seed) == best
                clip_attr = f' data-clip="{value:.4f}"' if value is not None else ""
                clip_meta = (f'<div class="meta clip-value{" best-clip" if is_best else ""}">'
                             f'{value:.3f}{" · 최고" if is_best else ""}</div>') if value is not None else ""
                inner.append(f'<div class="cell"{clip_attr}><a href="{rel}" target="_blank">'
                             f'<img loading="lazy" src="{rel}" alt="e{epoch} {prompt["id"]} s{seed}"></a>'
                             f'<div class="meta">s{seed}</div>{clip_meta}</div>')
            cells.append(f"<td>{''.join(inner)}</td>")
        rows.append(f'<tr data-order="{len(rows)}"><th class="row">{html_escape(prompt["id"])}'
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
    clip_tools = ""
    if clip_scores:
        summary += ' · <b>CLIP 보조 정렬</b> 사용(점수는 눈의 보조일 뿐, 선택은 사람이 합니다)'
        clip_tools = _CLIP_SORT_TOOLS.format(
            options="".join(f'<option value="clip-desc:{e}">CLIP 높은 순 — 에포크 {e}</option>'
                            for e in reversed(epochs)))
    path = exp / "comparison.html"
    path.write_text(_PAGE_TEMPLATE.format(
        title=html_escape(f"에포크 비교 — {project.card.get('id', '')}"),
        summary=summary, head=head, rows="".join(rows), clip_tools=clip_tools), encoding="utf-8")
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


def run_has_checkpoints(run_folder):
    """Does this run folder hold at least one `.safetensors` a grid could draw with?

    Counting a run by its folder existing made an aborted run - started, stopped during text-encoder caching, zero
    checkpoints - a candidate for "the only run", so a project with one good run and one dead one had to be named by
    hand every time (spec N5 `aborted_run_blocks_find_run`). The folder is still reported, never deleted.
    """
    folder = Path(run_folder)
    for candidate in (folder / "output", folder):
        if candidate.is_dir() and any(candidate.glob("*.safetensors")):
            return True
    return False


def run_rows(project):
    """Every run under the project, with what makes it usable for evaluation (spec N5's run list).

    Returns `[{name, folder, snapshot, checkpoints, usable, state}]`, newest path first. A run with zero checkpoints
    is `usable: False` and carries the reason in `state`; it is listed because a person needs to see that it exists
    and that it will not be offered.
    """
    base = Path(project.dir) / "seed_lora"
    if not base.is_dir():
        return []
    rows = []
    for path in sorted(base.glob("*/runs/*")):
        if not path.is_dir():
            continue
        snapshot = path.parent.parent.name
        output = path / "output"
        count = len(list((output if output.is_dir() else path).glob("*.safetensors")))
        state = _run_state(path) or {}
        rows.append({"name": path.name, "folder": str(path), "snapshot": snapshot,
                     "checkpoints": count, "usable": count > 0,
                     "complete": bool(state.get("complete")),
                     "state": state.get("active_stage") or ("finished" if state.get("complete") else None)})
    return rows


def find_run(project, run_id=None):
    """The run folder to evaluate: the one named, or the only usable one the project has.

    The seed snapshot id and the run id are different things and neither can be derived from the other, so this only
    ever returns a folder it actually found. A run with no checkpoints is not a candidate (spec N5): it cannot be
    evaluated, and counting it made a project with one live run and one aborted one demand a `--run` every time.
    """
    base = project.dir / "seed_lora"
    runs = sorted((base).glob("*/runs/*")) if base.is_dir() else []
    runs = [path for path in runs if path.is_dir()]
    if run_id:
        for path in runs:
            if path.name == run_id or path.parent.parent.name == run_id:
                if not run_has_checkpoints(path):
                    raise EpochError(f"run {path.name} has no checkpoints yet (stopped before the first save); "
                                     "nothing to evaluate - finish or restart the training run")
                return path
        raise EpochError(f"no run {run_id!r} under {base}")
    if not runs:
        raise EpochError(f"no training run under {base}; train a seed first")
    usable = [path for path in runs if run_has_checkpoints(path)]
    skipped = [path for path in runs if not run_has_checkpoints(path)]
    if not usable:
        raise EpochError("none of the " + str(len(runs)) + " run(s) under " + str(base)
                         + " has a checkpoint yet: " + ", ".join(path.name for path in skipped))
    if len(usable) > 1:
        raise EpochError("more than one run under " + str(base) + "; name one with --run: "
                         + ", ".join(path.name for path in usable)
                         + (f" (skipped, no checkpoints: {', '.join(p.name for p in skipped)})" if skipped else ""))
    return usable[0]



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
    parser.add_argument("--clip", action="store_true",
                        help="with sheet: score the drawn cells with CLIP for the auxiliary sort (GPU)")
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
        # The GPU claim covers the whole grid (spec N1): every cell submits to the same card training may be using.
        from . import gpu
        claim = gpu.claim_gpu(what="the epoch grid")
        claim.acquire()
        try:
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
            gpu.vram_gate("krea2", "draw")
            done, skipped, failed = run(project, manifest, client=client, names=names, cancel=cancel)
        finally:
            claim.release()
        paused = cancel is not None and cancel()
        if paused:
            print("paused: the remaining cells are still pending, so this command continues them")
        # The exit code tells the truth about the work (spec N3.3): a grid with any failed cell did not finish, and a
        # caller that read exit 0 as "the plan is done" would call a half-drawn grid complete. A pause is not a
        # failure - the pictures that were drawn are intact and a rerun continues - but it is not a finish either,
        # so it is reported in words while still exiting 0.
        if failed:
            print(f"[epochs] {failed} cell(s) failed; {done} drawn, {skipped} already present", file=sys.stderr)
            return 1
        if any(t["status"] in ("pending", "failed") for t in manifest["tasks"]):
            print(f"[epochs] {len(manifest['tasks']) - done - skipped} cell(s) still pending", file=sys.stderr)
            return 1 if not paused else 0
        return 0

    if args.command == "sheet":
        clip_scores = None
        if args.clip:
            # The auxiliary sort needs scores; they cost one scoring pass over the drawn cells and are cached
            # beside the sheet, so re-rendering costs nothing (spec R7 P2). The CLIP model runs on the same GPU,
            # so the scoring pass takes the common claim too (spec N1). `log` was never defined in this scope -
            # `sheet --clip` died with a NameError before calling the scorer at all (spec N3.4).
            from . import gpu
            claim = gpu.claim_gpu(what="the epoch sheet CLIP scoring")
            claim.acquire()
            try:
                clip_scores = sheet_clip_scores(project, run_folder, log=print)
            finally:
                claim.release()
        path = write_sheet(project, run_folder, clip_scores=clip_scores)
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
