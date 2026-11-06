"""M3-3: batch B - fill the slots the reviewer's accepted pictures do not cover yet (spec 3.3).

The seed LoRA teaches the character. Batch B then asks it for the compositions the dataset still lacks, so the set that
goes to the final training run covers the angles, expressions, backgrounds and poses the shoot actually needs.

Two rules decide what gets drawn, and both exist to stop this stage from wasting GPU time:

* a slot is only short when too few of its pictures were **accepted** - a pending picture might fill it, and a
  correction is still being judged, so the plan counts what is settled and lists the rest;
* the same plan run twice submits nothing new. A cell whose file is already on disk is skipped, and a slot that is
  short only because its picture is waiting in the review queue is reported as waiting rather than drawn again.

Generation is `max(train_generation)` of the snapshot the seed was trained on, plus one: batch B is the first
generation that learned from the seed and did not merely continue a chain of edits. The epoch that produced it is a
separate field, and `trained_on_snapshot` names the seed snapshot - the three facts are kept apart because
"generation 1" alone says nothing about which images taught the model.

  python -m pipeline.batch_gen plan --project projects/suamom
  python -m pipeline.batch_gen run  --project projects/suamom
  python -m pipeline.batch_gen list --project projects/suamom
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import comfy as comfy_mod
from . import epochs as epochs_mod
from . import krea2_gen, lineage, make_cancel, models

AUTO = Path(__file__).resolve().parents[1]          # automation

# The wording each slot is asked for, in tool2's tag vocabulary. A slot the plan asks for but this table does not know
# is refused rather than guessed: a prompt that does not match its slot id would fill the quota with the wrong picture.
SLOT_PROMPTS = {
    "ang_front_close": "portrait, upper body, front view, looking at viewer",
    "ang_front_medium": "upper body, front view, looking at viewer",
    "ang_front_wide": "full body, front view, standing, looking at viewer",
    "ang_quarter_medium": "upper body, three-quarter view, looking at viewer",
    "ang_quarter_close": "portrait, close-up, three-quarter view, looking at viewer",
    "ang_side_medium": "full body, side view, profile, looking away",
    "ang_back_medium": "upper body, from behind, looking back",
    "ang_back_wide": "full body, from behind, standing",
    "frm_upper_body": "upper body, portrait, front view",
    "frm_quarter": "waist-up, three-quarter view",
    "frm_profile": "chest-up, from side, profile",
    "frm_back": "chest-up, from behind, looking back",
    "frm_full_body": "full body, standing, head to shoes",
    "frm_face": "portrait, close-up, face focus, front view",
    "frm_dutch": "dutch angle, upper body",
    "frm_from_above": "from above, full body, looking down",
    "exp_smile": "smile, looking at viewer",
    "exp_angry": "angry expression, clenched teeth, looking at viewer",
    "exp_sad": "crying, tear drop, looking down",
    "exp_surprised": "surprised expression, open mouth",
    "bg_classroom": "classroom background, green chalkboard, wooden desks, indoors, day",
    "bg_street_night": "city street at night, neon lights, wet pavement",
    "bg_park": "cherry blossom background, park, outdoors, day",
    "bg_bedroom": "bedroom background, warm lighting, indoors",
    "pos_sitting": "sitting, full body",
    "pos_walking": "walking, full body, moving toward the viewer",
    "pos_waving": "waving hand, upper body, looking at viewer",
    "pos_phone": "holding phone, upper body, looking at phone",
    "pos_crossed": "arms crossed, leaning against a wall, full body",
    "pos_running": "running, dynamic full body pose",
}
# Two pictures per short slot: one is a single draw, and a slot filled by one bad picture is still a hole. The reviewer
# accepts at most one, so the second is a spare rather than extra work that has to be kept.
SEEDS_PER_SLOT = 2
BASE_SEED = 2000


class BatchError(RuntimeError):
    """A batch that cannot be planned as asked."""


def category_of(slot_id):
    """`ang_front_close` -> `angles`: the first field of a slot id is its category, by the plan's own convention."""
    head = str(slot_id).split("_")[0]
    return {"ang": "angles", "frm": "framing", "exp": "expressions", "bg": "backgrounds", "pos": "poses"}.get(head, head)


def prompt_for(card, slot_id):
    """`trigger, caption_base, fixed_traits, outfit, slot wording`, the order spec 3.4 fixes."""
    parts = [card.get("trigger", ""), card.get("caption_base", "")]
    parts += list(card.get("fixed_traits") or [])
    parts.append((card.get("outfits") or {}).get("default", ""))
    parts.append(SLOT_PROMPTS[slot_id])
    return ", ".join(p.strip() for p in parts if p and str(p).strip())


def training_of(project):
    """Which seed snapshot batch B is being generated *after*, and at which generation.

    Read from the project's own newest snapshot rather than from the card, because a snapshot that was built but never
    used must not raise the generation: the number has to describe the pictures that were learned from.
    """
    folder = project.dir / "seed_lora"
    if not folder.is_dir():
        raise BatchError(f"no seed snapshot under {folder}; build one with `pipeline.seed --build` first")
    snapshots = []
    for child in sorted(folder.iterdir()):
        manifest = child / "snapshot.json"
        if not manifest.is_file():
            continue
        data = _read_json(manifest)
        if data.get("id") and data.get("inputs"):
            snapshots.append((str(data.get("created_at") or ""), data))
    if not snapshots:
        raise BatchError(f"no readable snapshot manifest under {folder}")
    _created, snapshot = max(snapshots, key=lambda pair: pair[0])
    generations = [int(entry.get("generation") or 0) for entry in snapshot["inputs"]]
    return {
        "snapshot_id": snapshot.get("id"),
        "snapshot_file": str(snapshot.get("base") or ""),
        "inputs": len(snapshot["inputs"]),
        # The generation B is *drawn at*: one past the newest thing the seed learned from.
        "generation": (max(generations) if generations else 0) + 1,
        "trained_on_snapshot": snapshot.get("id"),
        "trained_on_generation": max(generations) if generations else 0,
    }


def slot_board(project, *, log=None):
    """(draw, pending): which slots still need pictures, read-only, split by whether review may still fill them.

    A slot is only short when too few of its pictures were **accepted** - a pending picture might fill it, and a
    correction is still being judged, so the board counts what is settled and lists the rest. This is the
    classification both generators consume: batch B draws the draw rows and reports the pending ones, and ph3_edit
    plans the draw rows too - a waiting picture is not a gap to fill, whoever does the filling.

    Read-only on purpose: planning an edit must never rewrite batch B's manifest, and neither generator needs a seed
    snapshot to classify slots.
    """
    quotas = project.card.get("slot_quotas") or {}
    if not quotas:
        raise BatchError("the card has no slot_quotas, so there is nothing to fill")
    board = project.quotas(quotas)
    waiting = {}
    for record in project.undecided():
        slot = (record.get("slot") or {}).get("slot_id")
        if slot:
            waiting[slot] = waiting.get(slot, 0) + 1
    draw, pending = [], []
    for slot, row in sorted(board.items()):
        if row["short"] <= 0:
            continue
        entry = {"slot": slot, "category": category_of(slot), "target": row["target"],
                 "accepted": row["accepted"], "short": row["short"], "waiting": waiting.get(slot, 0)}
        # A waiting picture that is accepted fills the slot, so a slot covered by waiting pictures is not drawn.
        if entry["waiting"] >= row["short"]:
            pending.append(entry)
        else:
            entry["to_draw"] = row["short"] - entry["waiting"]
            draw.append(entry)
    if log is not None:
        log(f"[batch_b] {len(draw)} slot(s) to draw, {len(pending)} waiting on review")
    return draw, pending


def short_slots(project, *, log=None):
    """The slots batch B would draw: the slot board, plus the requirement that every slot have a prompt.

    The wording each slot is asked for is batch B's own vocabulary; a slot the plan asks for but this table does not
    know is refused rather than guessed, because a prompt that does not match its slot id would fill the quota with
    the wrong picture.
    """
    draw, pending = slot_board(project, log=log)
    for entry in draw:
        if entry["slot"] not in SLOT_PROMPTS:
            raise BatchError(f"slot {entry['slot']!r} has no prompt in SLOT_PROMPTS; "
                             "add one rather than guessing its wording")
    return draw, pending


def plan(project, *, seeds_per_slot=SEEDS_PER_SLOT, log=print):
    """Write the work list for the short slots. Returns the manifest, which `run()` then executes.

    The plan pins every input the pictures will be drawn with - the chosen run and checkpoint by sha256, the seed
    snapshot and generation, the graph version, the prompts, the size and the seeds (spec R4): a run or an import may
    happen hours later, and it must draw and record exactly what the plan said, not whatever the card says by then. The
    manifest a plan replaces is archived first, so the manifest that drew already-imported pictures survives.
    """
    training = training_of(project)
    draw, waiting = short_slots(project, log=log)
    prompts = (project.card.get("krea2_inference") or {})
    size = dict(prompts.get("size") or {"width": 1024, "height": 1536})
    chosen = epochs_mod.chosen(project)
    checkpoint = None
    if chosen:
        checkpoint = {"run_id": chosen.get("run_id"), "epoch": chosen.get("epoch"),
                      "checkpoint_path": chosen.get("checkpoint_path"),
                      "checkpoint_sha256": chosen.get("checkpoint_sha256")}
    tasks = []
    for entry in draw:
        for n in range(seeds_per_slot):
            seed = BASE_SEED + n
            tasks.append({
                "slot_id": entry["slot"], "category": entry["category"], "seed": seed,
                "prompt": prompt_for(project.card, entry["slot"]),
                "file": f"images/{entry['slot']}__s{seed}.png",
                "status": "pending", "sha256": None,
            })
    manifest = {
        "kind": "batch_b",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan_id": os.urandom(4).hex(),
        "project": project.card.get("id"),
        "training": training,
        "generation": training["generation"],
        "trained_on_snapshot": training["trained_on_snapshot"],
        "checkpoint": checkpoint,
        "graph_version": krea2_gen.GRAPH_VERSION["b"],
        "inference": {k: (prompts.get(k)) for k in ("variant", "steps", "cfg", "sampler", "scheduler")},
        "size": size,
        "slots": draw,
        "waiting": waiting,
        "seeds_per_slot": int(seeds_per_slot),
        "tasks": tasks,
        "status": "planned",
    }
    write_manifest(manifest_path(project), manifest)
    log(f"[batch_b] generation {manifest['generation']} from snapshot {manifest['trained_on_snapshot']}: "
        f"{len(tasks)} image(s) over {len(draw)} slot(s)"
        + (f", checkpoint {checkpoint['checkpoint_sha256'][:12]}" if checkpoint else ", no epoch chosen yet"))
    for row in waiting:
        log(f"  waiting on review: {row['slot']} (short {row['short']}, {row['waiting']} pending)")
    return manifest


def manifest_path(project):
    return project.dir / "experiments" / "batch_b" / "batch.json"


def read_manifest(project):
    path = manifest_path(project)
    if not path.is_file():
        raise BatchError(f"no batch planned at {path}; run `plan` first")
    return _read_json(path)


def plan_proof(project, record):
    """(proven, detail): does a batch B plan this project holds actually name this picture?

    A batch B candidate has no parent ids - the plan that drew it is its only ancestry - so a reviewer's accept alone
    is not membership, and a slot name alone is never proof: plans get re-planned, and an old picture must not walk
    back in under its successor's slot (spec R2). Two proofs count. The current plan manifest names a task with this
    slot and seed whose output file hashes to the candidate's bytes. Or the plan's own output file -
    `images/<slot>__s<seed>.png`, the one path a B task writes - still hashes to the candidate's bytes while the
    record describes itself as this stage's output, which is how a picture from a plan that was later re-planned stays
    provable when the manifest that drew it no longer holds its task row.
    """
    slot = record.get("slot") or {}
    slot_id = slot.get("slot_id") or slot.get("id") or slot.get("category")
    seed_value = record.get("seed")
    exp = manifest_path(project).parent
    manifest = None
    if manifest_path(project).is_file():
        try:
            manifest = _read_json(manifest_path(project))
        except ValueError:
            manifest = None
    if isinstance(manifest, dict):
        for task in manifest.get("tasks") or []:
            if task.get("slot_id") != slot_id or task.get("seed") != seed_value:
                continue
            out = exp / str(task.get("file") or "")
            if out.is_file() and lineage.sha256_file(out) == record.get("sha256"):
                return True, f"plan task {slot_id} s{seed_value} ({task.get('file')})"
        if not manifest.get("tasks"):
            manifest = None
    loras = [l for l in (record.get("models") or {}).get("loras") or [] if l.get("trained_on_snapshot")]
    if record.get("method") == "krea2_b" and len(loras) == 1 and loras[0].get("trained_on_snapshot"):
        out = exp / "images" / f"{slot_id}__s{seed_value}.png"
        if out.is_file() and lineage.sha256_file(out) == record.get("sha256"):
            return True, f"the plan's own output {out.name} (the manifest was re-planned since it was drawn)"
    return False, f"no batch B task names slot {slot_id} seed {seed_value} with these bytes"


def write_manifest(path, manifest):
    """Write the plan manifest, archiving the one it replaces when that one holds drawn work.

    `batch.json` is the current plan's pointer and is rewritten by every plan; the manifest that actually drew
    already-imported pictures is the lineage's evidence (spec R4), so a plan with drawn or failed tasks is kept beside
    it under its own timestamp instead of being overwritten into silence.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            previous = _read_json(path)
        except ValueError:
            previous = None
        same_plan = previous.get("plan_id") == manifest.get("plan_id")
        drawn = bool(previous) and any(
            task.get("status") in ("done", "failed") for task in previous.get("tasks") or [])
        if drawn and not same_plan:
            stamp = str(previous.get("created_at") or "unknown").replace(":", "").replace("-", "")
            archive = path.with_name(f"batch.json.at-{stamp}.json")
            if not archive.exists():
                archive.write_text(json.dumps(previous, ensure_ascii=False, indent=1), encoding="utf-8")
    temp = path.with_name(path.name + ".tmp-" + os.urandom(4).hex())
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def pinned_checkpoint(manifest):
    """The checkpoint the plan pinned, verified against its recorded sha256.

    A plan may be run or imported hours after it was written; the card's chosen epoch can move in that time, and
    drawing or recording with a different checkpoint than the plan pinned would attach the wrong provenance to every
    picture (spec R4). Returns None for a manifest written before pinning existed - the caller decides the fallback.
    """
    pin = manifest.get("checkpoint") or {}
    if not pin.get("checkpoint_path"):
        return None
    path = Path(pin["checkpoint_path"])
    if not path.is_file():
        raise BatchError(f"the plan's checkpoint is gone: {path}")
    actual = lineage.sha256_file(path)
    if actual != pin.get("checkpoint_sha256"):
        raise BatchError(f"the checkpoint changed since the plan pinned it ({pin['checkpoint_path']}): "
                         f"pinned {str(pin.get('checkpoint_sha256'))[:12]}, disk {actual[:12]}; "
                         "re-plan rather than drawing with a different model than the plan names")
    return path


def run(project, manifest, *, client, names, log=print, cancel=None):
    """Draw the pending cells. Re-running the same plan submits nothing new - and reuses nothing it cannot vouch for.

    A task whose file exists is skipped only when the file is a decodable image and still hashes to the sha the plan
    recorded when it was drawn; a truncated file from a crashed run, or a picture left by a different plan under the
    same name, is drawn again rather than imported (spec R4). The drawn bytes' sha is recorded per task, so a resume
    can tell its own outputs from anything else. The checkpoint the plan pinned is verified before the first submit.
    """
    exp = manifest_path(project).parent
    pinned_checkpoint(manifest)
    assert_variant(manifest)
    done = skipped = failed = 0
    for task in manifest["tasks"]:
        out = exp / task["file"]
        if out.is_file() and _reusable(out, task, log):
            if task["status"] != "done":
                task["status"] = "done"
            skipped += 1
            continue
        if cancel is not None and cancel():
            log("cancelled")
            break
        started = time.time()
        try:
            graph = krea2_gen.build(
                manifest["inference"]["variant"] or "b",
                unet=names["b_unet"], clip=names["clip"], vae=names["vae"], prompt=task["prompt"],
                width=manifest["size"]["width"], height=manifest["size"]["height"], seed=task["seed"],
                character_lora=names["character_lora"], speed_lora=names["speed_lora"],
                filename_prefix=f"pipeline_batch_b/{task['slot_id']}",
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            client.run(graph, out, slot_id=task["slot_id"], method="krea2_b", seed=task["seed"])
            task["status"] = "done"
            task["sha256"] = lineage.sha256_file(out)
            done += 1
            log(f"  {task['slot_id']} s{task['seed']}: {time.time() - started:.0f}s")
        except Exception as exc:  # noqa: BLE001 - one failed picture must not end the batch
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            log(f"  {task['slot_id']} s{task['seed']}: FAILED {task['error'][:200]}")
        finally:
            task["seconds"] = round(time.time() - started, 1)
            write_manifest(manifest_path(project), manifest)
    manifest["status"] = "ran"
    write_manifest(manifest_path(project), manifest)
    log(f"done {done}, skipped {skipped} already present, failed {failed}")
    return done, skipped, failed


def _reusable(path, task, log):
    """Is the file at a task's path the output that task recorded? Anything else is drawn again."""
    if not lineage.image_decodes(path):
        log(f"  {task['slot_id']} s{task['seed']}: the file on disk does not decode; drawing it again")
        return False
    recorded = task.get("sha256")
    if not recorded:
        return False               # no recorded bytes: an earlier plan's file, or a crashed write - redraw
    return lineage.sha256_file(path) == recorded


def assert_variant(manifest):
    """Batch B is the (b) configuration: RAW DiT + Turbo LoRA + the character LoRA, the one M0 chose."""
    variant = (manifest.get("inference") or {}).get("variant") or "b"
    if variant != "b":
        raise BatchError(f"batch B draws with configuration (b), but the card says {variant!r}; "
                         "the M0 evidence for (b) does not transfer to another configuration")


def candidates_of(project, manifest):
    """Lineage records for the batch's finished pictures, ready for `Project.import_candidates`.

    Generation and snapshot are recorded separately on purpose: `train_generation` says how far down the chain this
    picture sits, and `trained_on_snapshot` says which images taught the LoRA that drew it. Collapsing them would make a
    picture from a rebuilt seed look like it came from the old one. The checkpoint is the plan's pin, verified by sha -
    never the card's current choice (spec R4): importing under a checkpoint the plan did not draw with would attach
    provenance to the wrong model.
    """
    exp = manifest_path(project).parent
    pinned = pinned_checkpoint(manifest)
    lora = pinned or character_lora_path(project)
    records = []
    for task in manifest["tasks"]:
        out = exp / task["file"]
        if not out.is_file():
            continue
        if task.get("sha256") and lineage.sha256_file(out) != task["sha256"]:
            continue              # the file on disk is not the bytes the plan recorded: not this plan's output
        slot = {"slot_id": task["slot_id"], "category": task["category"], "prompt": task["prompt"]}
        record = lineage.record(
            out, generator="krea2_lora", parent_ids=[], train_generation=int(manifest["generation"]),
            edit_depth=0, slot=slot,
            models={"dit": {"name": "krea2_raw"},
                    "loras": [lineage.model_entry(
                        lora, trained_on_snapshot=manifest["trained_on_snapshot"],
                        trained_on_generation=(manifest.get("training") or {}).get("trained_on_generation", 0))]},
            prompt=task["prompt"], seed=task["seed"],
            steps=krea2_gen.SAMPLING["steps"], cfg=krea2_gen.SAMPLING["cfg"],
            width=manifest["size"]["width"], height=manifest["size"]["height"],
            graph_version=manifest.get("graph_version") or krea2_gen.GRAPH_VERSION["b"],
            method="krea2_b", batch="batch_b", plan_created_at=manifest.get("created_at"),
        )
        records.append(record)
    return records


def character_lora_path(project):
    """The LoRA the card's chosen epoch points at; batch B draws with it or not at all."""
    entry = epochs_mod.chosen(project)
    if not entry:
        raise BatchError("no epoch has been chosen for this project; run `pipeline.epochs choose --epoch N` first, "
                         "or batch B would draw with whatever checkpoint happened to be loaded")
    if not entry.get("checkpoint_present"):
        raise BatchError(f"the chosen checkpoint is gone: {entry.get('checkpoint_path')}")
    if not entry.get("checkpoint_matches"):
        raise BatchError(f"the chosen checkpoint changed since it was chosen: {entry.get('checkpoint_path')}")
    return Path(entry["checkpoint_path"])


def import_results(project, manifest, log=print):
    """Copy the batch's pictures into the project and record them."""
    records = candidates_of(project, manifest)
    num, skipped = project.import_candidates(records, base=manifest_path(project).parent, log=log)
    return num, skipped


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pipeline.batch_gen", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("list", "plan", "run", "import"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--seeds-per-slot", type=int, default=SEEDS_PER_SLOT)
    parser.add_argument("--comfy", default=None)
    parser.add_argument("--cancel-file", help="stop before the next cell once this file exists")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from . import project as pj
    project = pj.Project.open(args.project)

    if args.command == "list":
        draw, waiting = short_slots(project, log=lambda *_: None)
        if args.json:
            print(json.dumps({"draw": draw, "waiting": waiting}, ensure_ascii=False, indent=1))
            return 0
        print(f"{len(draw)} slot(s) to draw:")
        for row in draw:
            print(f"  {row['slot']:20s} short={row['short']} waiting={row['waiting']} -> draw {row['to_draw']}")
        if waiting:
            print(f"{len(waiting)} slot(s) waiting on review:")
            for row in waiting:
                print(f"  {row['slot']:20s} short={row['short']} pending={row['waiting']}")
        return 0

    if args.command == "plan":
        manifest = plan(project, seeds_per_slot=args.seeds_per_slot)
        if args.json:
            print(json.dumps({"generation": manifest["generation"],
                              "trained_on_snapshot": manifest["trained_on_snapshot"],
                              "slots": [s["slot"] for s in manifest["slots"]],
                              "tasks": len(manifest["tasks"])}, ensure_ascii=False, indent=1))
        return 0

    if args.command == "run":
        manifest = read_manifest(project)
        if not manifest["tasks"]:
            print("[batch_b] nothing to draw: every short slot is waiting on review")
            return 0
        client = epochs_mod.make_client(args.comfy)
        if not client.ping():
            raise BatchError(f"ComfyUI is not answering at {client.url}")
        problems = models.problems("krea2", require_optional=("krea2_turbo_lora",))
        if problems:
            raise BatchError("model problems: " + "; ".join(problems))
        names = epochs_mod.resolve_names(client)
        names["character_lora"] = client.name_for("lora", character_lora_path(project))
        cancel = make_cancel(args.cancel_file)
        run(project, manifest, client=client, names=names, cancel=cancel)
        if cancel is not None and cancel():
            print("paused: the remaining cells are still pending, so this command continues them")
        return 0

    if args.command == "import":
        manifest = read_manifest(project)
        num, _skipped = import_results(project, manifest)
        if args.json:
            print(json.dumps({"imported": num}, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
