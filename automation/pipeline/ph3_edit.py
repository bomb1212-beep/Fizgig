"""Draw acceptance-gap fillers with P-H3 instead of the Qwen editor.

This is the P-H3 counterpart of `batch_gen`: where batch_gen redraws a *missing slot* from scratch with the character
LoRA, this takes an existing accepted picture and asks P-H3 to restage it into the slot that is still short. That is an
edit, not a generation, so it is recorded with `generator="ph3_edit"`, `edit_depth=1`, and the picture it started from
as its parent - which is what keeps the lineage honest and what seed policy v2 checks before such a picture may enter a
seed set.

**Why an edit rather than a new draw.** An accepted picture already has the character right, which is the hard part; a
redraw has to get it right again. An edit only has to move the camera. That is also why the source is a parent: if the
edit drifts, the drift is traceable to one picture.

**Nothing here runs unless a slot is actually short.** `plan` reports zero work when every slot is filled or already
has pictures waiting on review, and `run` then submits nothing. The module deliberately does not invent slots to keep
itself busy.
"""
import json
import os
import time
from pathlib import Path

from pipeline import batch_gen, comfy, epochs as epochs_mod, lineage, ph3_gen, project as pj, seed

# The step budget of one P-H3 edit. Kept here rather than in ph3_gen because it is a budget decision, not a graph fact.
STEPS = 8
# How many seeds to draw per short slot when the caller does not say. One is enough to judge a restaging; two give a
# second chance without doubling the wall clock on a slot that will probably need human review anyway.
SEEDS_PER_SLOT = 2

METHOD = "ph3_edit"
GENERATOR = "ph3_edit"


class EditError(RuntimeError):
    """A batch that cannot be planned or run."""


def manifest_path(project):
    return project.dir / "experiments" / "ph3_edit" / "edit.json"


def read_manifest(project):
    path = manifest_path(project)
    if not path.is_file():
        raise EditError(f"no P-H3 edit planned at {path}; run `plan` first")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(project, manifest):
    path = manifest_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + os.urandom(4).hex())
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def sources(project, *, caps=None, log=None):
    """(usable, unusable) - accepted pictures an edit may start from, with every refusal named.

    Editing puts the source picture into the child's lineage, so the source has to be as clean as a training input,
    checked the same way the audits check one (spec R3): bytes that still hash to what was recorded, a holdout that is
    refused by id **and by sha256** (a renamed copy of the holdout is still the holdout), ancestry that resolves to
    recorded pictures without a cycle and without an unparented synthetic anchor, and a LoRA-trained candidate whose
    training snapshot this project actually holds and that held no holdout. On top of the shared checks sits the child
    shape the edit would produce: a user original edits to generation 1, depth 1; a candidate keeps its parent's
    generation and goes one edit deeper - and a child past the caps is excluded here, before any GPU work, rather than
    becoming a picture no audit can ever admit.
    """
    caps = caps or lineage.caps(project.card.get("lineage"))
    holdout = project.holdout_entry()
    holdout_ids = set(holdout["ids"])
    holdout_shas = set(holdout["shas"])
    index = project.index()
    review = project.review()
    manifests = seed.snapshot_manifests(project)
    # A synthetic source whose own parents were never recorded: anything edited from it inherits an unprovable
    # ancestry and would be refused as unverified_ancestor forever. Skipping it here is the fix.
    quarantined = {r["id"] for r in project.source_records()
                   if r.get("origin") == "synthetic"
                   and not [p for p in (r.get("parent_ids") or []) if p in index]}
    usable, unusable = [], []

    def refuse(record, code, detail=""):
        unusable.append({"id": record["id"], "code": code,
                         "reason": detail or SEED_REASONS.get(code, code)})

    def lineage_problems(record, *, generated):
        """(code, detail) for the first lineage problem, or None: shared with what the audits refuse.

        `no_ancestry` is a candidate's problem only - a user original legitimately has no parents, it is the root the
        whole chain is supposed to reach.
        """
        parents = list(record.get("parent_ids") or [])
        loras = [l for l in (record.get("models") or {}).get("loras") or [] if l.get("trained_on_snapshot")]
        if generated and not parents and not loras:
            return "no_ancestry", "neither parents nor a training snapshot are recorded"
        ancestors, unknown, cyclic = seed.walk(record, index)
        if cyclic:
            return "cyclic_lineage", "the parent chain leads back to the picture itself"
        if unknown:
            return "missing_parent", "a recorded parent is not in the project: " + ", ".join(sorted(unknown))
        hit = sorted(ancestors & holdout_ids)
        if hit:
            return "holdout_indirect", "an ancestor is a holdout original: " + ", ".join(hit)
        sha_hits = sorted({index[a].get("sha256") for a in ancestors if a in index} & holdout_shas)
        if sha_hits:
            return "holdout_indirect", "an ancestor carries a holdout's bytes: " + ", ".join(
                s[:12] for s in sha_hits)
        anchor_hits = sorted(ancestors & quarantined)
        if anchor_hits:
            return "unverified_ancestor", "an ancestor is an unparented synthetic anchor: " + ", ".join(anchor_hits)
        for lora in loras:
            snapshot = manifests.get(lora["trained_on_snapshot"])
            if snapshot is None:
                return "lora_snapshot_unverified", f"the LoRA names snapshot {lora['trained_on_snapshot']}, " \
                                                   "which this project has no record of"
            if {e.get("sha256") for e in snapshot.get("inputs") or []} & holdout_shas:
                return "holdout_via_lora", f"the LoRA was trained on {lora['trained_on_snapshot']}, " \
                                           "which holds a holdout"
        return None

    for record in project.source_records():
        if record["id"] in holdout_ids or record.get("sha256") in holdout_shas:
            refuse(record, "holdout_source", "the holdout itself (or a copy of its bytes) is never an edit source")
            continue
        if record["id"] in quarantined:
            refuse(record, "unverified_ancestor", "synthetic anchor with no recorded parent")
            continue
        path = project.resolve_file(record.get("file") or record.get("src_path"))
        if not path or not path.is_file():
            refuse(record, "missing_file", str(record.get("file") or record.get("src_path")))
            continue
        actual = lineage.sha256_file(path)
        if actual != record.get("sha256"):
            refuse(record, "tampered_file", f"recorded {str(record.get('sha256'))[:12]}, disk {actual[:12]}")
            continue
        problem = lineage_problems(record, generated=False)
        if problem:
            refuse(record, problem[0], problem[1])
            continue
        # A direct edit of a user original: the product's shape for a model-made edit (spec R3).
        usable.append({"id": record["id"], "path": str(path), "origin": "source", "sha256": actual,
                       "generation": 1, "edit_depth": 1})
    for record in project.candidates():
        if (review.get(record["id"]) or {}).get("decision") != "accept":
            continue
        if record["id"] in holdout_ids or record.get("sha256") in holdout_shas:
            refuse(record, "holdout_source", "carries a holdout's id or bytes")
            continue
        path = project.resolve_file(record.get("file"))
        if not path or not path.is_file():
            refuse(record, "missing_file", str(record.get("file")))
            continue
        actual = lineage.sha256_file(path)
        if actual != record.get("sha256"):
            refuse(record, "tampered_file", f"recorded {str(record.get('sha256'))[:12]}, disk {actual[:12]}")
            continue
        problem = lineage_problems(record, generated=True)
        if problem:
            refuse(record, problem[0], problem[1])
            continue
        # An edit of a candidate keeps the parent's generation and goes one edit deeper (spec R3).
        generation = int(record.get("train_generation") or 0)
        depth = int(record.get("edit_depth") or 0) + 1
        if generation > caps["max_train_generation"]:
            refuse(record, "generation_cap", f"the edit would sit at generation {generation}, "
                                             f"past max_train_generation {caps['max_train_generation']}")
            continue
        if depth > caps["max_edit_depth"]:
            refuse(record, "edit_depth_cap", f"the edit would sit at depth {depth}, "
                                             f"past max_edit_depth {caps['max_edit_depth']}")
            continue
        usable.append({"id": record["id"], "path": str(path), "origin": "candidate", "sha256": actual,
                       "generation": generation, "edit_depth": depth})
    if log is not None:
        for row in unusable:
            log(f"[ph3_edit] not editable {row['id']} [{row['code']}] {row['reason']}")
    return usable, unusable


SEED_REASONS = {
    "holdout_source": "the holdout itself, or a copy of its bytes",
    "unverified_ancestor": "synthetic anchor with no recorded parent",
    "no_ancestry": "neither parents nor a training snapshot are recorded",
    "cyclic_lineage": "the parent chain leads back to the picture itself",
    "missing_parent": "a recorded parent is not in the project",
    "holdout_indirect": "an ancestor is a holdout original",
    "holdout_via_lora": "the training snapshot behind the LoRA holds a holdout",
    "lora_snapshot_unverified": "the LoRA names a snapshot this project has no record of",
    "missing_file": "the image file is not on disk",
    "tampered_file": "the bytes on disk no longer hash to what was recorded",
    "generation_cap": "the edit would sit past max_train_generation",
    "edit_depth_cap": "the edit would sit past max_edit_depth",
}


def instruction_for(slot_id, category, card=None):
    """The wording handed to P-H3 for one slot.

    Written out per slot rather than templated from the id: "frm_back" tells a person what it means, and the model
    needs the same thing said in words. When the card names a default outfit it is carried into the instruction, so the
    wording reflects what the card actually dresses the character in (spec R3) instead of a generic "same outfit". A
    slot with no entry is refused by `plan` instead of being sent with a vague instruction that would produce a picture
    nobody asked for.
    """
    text = INSTRUCTIONS.get(slot_id) or INSTRUCTIONS_BY_CATEGORY.get(category)
    if text and card:
        outfit = (card.get("outfits") or {}).get("default", "")
        if outfit:
            text = f"{text}, wearing {outfit}"
    return text


INSTRUCTIONS = {
    "ang_back": "show the character from behind, same clothes and hairstyle",
    "frm_back": "show the character's whole back, standing, same outfit",
    "frm_profile": "show the character standing in profile, side view",
    "frm_dutch": "tilt the camera to a dutch angle, same pose and outfit",
    "frm_full_body": "show the character's whole body from head to feet, standing",
    "ang_front_wide": "pull the camera back to a wide shot, character small in frame, same outfit",
    "ang_front_medium": "medium shot from the front, waist up, same outfit",
    "ang_quarter_close": "close three-quarter view of the face, same hairstyle and expression",
}
INSTRUCTIONS_BY_CATEGORY = {
    "angles": "change the camera angle while keeping the character, outfit and style identical",
    "frames": "change the framing while keeping the character, outfit and style identical",
    "backgrounds": "keep the character exactly as they are and change only the background",
}


def short_slots(project, *, slots=None, force=False, log=None):
    """Short slots with no picture waiting on review, from batch_gen's own read-only classification.

    `batch_gen.slot_board` is the producer of the split this decision depends on, so it is consumed directly instead
    of re-reading a manifest built for someone else (spec R3). Its draw rows are the slots short of an accepted picture
    with nothing pending on review; its pending rows are slots a waiting picture may still fill. Planning against the
    pending rows - what this used to do, and what the old tests encoded - would edit the slots that need it least and
    grow the review backlog instead of shrinking it. Read-only and snapshot-free: nothing is written, and an
    independent edit never draws through the Krea2 LoRA, so no seed snapshot is required to plan one.

    `force` draws the named slots anyway (or every pending slot when none is named). Naming a slot is the deliberate
    override; it exists so the draw path can be exercised on demand, not as a way to quietly grow the backlog.
    """
    try:
        draw, pending = batch_gen.slot_board(project, log=log)
    except batch_gen.BatchError:
        return []                      # a card with no slot plan has no gaps to fill
    rows = [dict(row) for row in draw]
    if force:
        named = set(slots or [])
        rows += [dict(row) for row in pending if not named or row["slot"] in named]
    if slots:
        wanted = set(slots)
        rows = [row for row in rows if row["slot"] in wanted]
    return rows


def plan(project, *, slots=None, seeds=None, source_ids=None, force=False, log=print):
    """{manifest} for the short slots, with the pictures already on disk skipped.

    Returns a manifest with `tasks == []` when there is nothing to do, which is a normal outcome, not an error: a plan
    that is already filled or already waiting on review needs no new pictures. Planning writes only this stage's own
    manifest - the batch B plan is not touched, and no GPU or model upload happens here.
    """
    # Sources are checked first: with nothing editable the plan is impossible whatever the slots say, and that reason
    # is more useful than "no seed snapshot", which a batch plan would raise instead.
    available, unusable = sources(project, log=log)
    if source_ids:
        wanted_sources = set(source_ids)
        available = [s for s in available if s["id"] in wanted_sources]
    if not available:
        raise EditError("no picture is eligible to edit: every source is a holdout, an unparented anchor, or has gone "
                        "missing")

    short = short_slots(project, slots=slots, force=force, log=log)

    seeds = list(seeds or range(7001, 7001 + SEEDS_PER_SLOT))
    tasks, skipped = [], []
    for row in short:
        slot_id, category = row["slot"], row.get("category") or ""
        instruction = instruction_for(slot_id, category, project.card)
        if not instruction:
            skipped.append({"slot_id": slot_id,
                            "reason": "no instruction is written for this slot or its category"})
            continue
        source = available[0]
        for seed_value in seeds:
            tasks.append({
                "slot_id": slot_id, "category": category, "seed": int(seed_value),
                "instruction": instruction,
                "source_id": source["id"], "source_path": source["path"],
                "source_sha256": source["sha256"],
                # The child's shape is decided at plan time from the verified source (spec R3), and `records_of`
                # records exactly this - a direct edit of an original is generation 1, depth 1; an edit of a candidate
                # keeps the parent's generation and goes one deeper.
                "generation": source["generation"], "edit_depth": source["edit_depth"],
                "file": f"{slot_id}_s{seed_value}.png",
                "status": "pending", "seconds": None, "error": "",
            })

    manifest = {
        "kind": "ph3_edit",
        "project": project.card.get("id"),
        "created_at": _now(),
        "status": "planned",
        "steps": STEPS,
        "sigmas": ph3_gen.DEFAULT_SIGMAS,
        "seeds_per_slot": len(seeds),
        "sources": available,
        "unusable_sources": unusable,
        "gaps": short,
        "skipped": skipped,
        "tasks": tasks,
    }
    write_manifest(project, manifest)
    log(f"[ph3_edit] {len(tasks)} picture(s) over {len(short)} short slot(s); "
        f"{len(available)} picture(s) available to edit")
    for row in short:
        log(f"  short: {row['slot']} (needs {row.get('short')}, nothing waiting on review)")
    for row in skipped:
        log(f"  skipped: {row['slot_id']} - {row['reason']}")
    if not tasks:
        log("[ph3_edit] nothing to do: no slot is short of an accepted picture right now")
    return manifest


def run(project, manifest, *, client, log=print, cancel=None):
    """Draw every pending task. Re-running submits nothing new - and reuses nothing it cannot vouch for.

    Before a task is submitted its source is re-checked against the sha the plan pinned: a parent whose bytes changed
    since planning fails that task instead of editing pixels nobody recorded (spec R4). A task whose file exists is
    skipped only when the file decodes and hashes to the sha recorded when it was drawn; anything else - a truncated
    write, a picture left by a plan that named a different parent - is drawn again.
    """
    folder = manifest_path(project).parent
    names = ph3_gen.resolve_names(client)
    # LoadImage cannot open a local path; the reference has to be uploaded first and referred to by the returned name.
    upload_cache = {}
    done = skipped = failed = 0
    for task in manifest["tasks"]:
        out = folder / task["file"]
        if out.is_file() and _reusable(out, task, log):
            task["status"] = "done"
            skipped += 1
            continue
        if cancel is not None and cancel():
            log("cancelled")
            break
        started = time.time()
        try:
            path = task["source_path"]
            pinned_sha = task.get("source_sha256")
            if pinned_sha and lineage.sha256_file(path) != pinned_sha:
                raise EditError(f"the source picture changed since the plan pinned it: {task['source_id']}")
            if path not in upload_cache:
                upload_cache[path] = client.upload(path)
            graph = ph3_gen.build(
                prompt="same character, same outfit and art style",
                instruction=task["instruction"],
                references=[upload_cache[path]],
                seed=task["seed"], steps=manifest.get("steps") or STEPS,
                names=names, filename_prefix=f"pipeline_ph3_edit/{task['slot_id']}",
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            client.run(graph, out, slot_id=task["slot_id"], method=METHOD, seed=task["seed"])
            task["status"] = "done"
            task["sha256"] = lineage.sha256_file(out)
            done += 1
            log(f"  {task['slot_id']} s{task['seed']}: {time.time() - started:.0f}s")
        except Exception as exc:  # noqa: BLE001 - one failed edit must not end the batch
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            log(f"  {task['slot_id']} s{task['seed']}: FAILED {task['error'][:200]}")
        finally:
            task["seconds"] = round(time.time() - started, 1)
            write_manifest(project, manifest)
    manifest["status"] = "ran"
    write_manifest(project, manifest)
    log(f"done {done}, skipped {skipped} already present, failed {failed}")
    return manifest


def _reusable(path, task, log):
    """Is the file at a task's path the output that task recorded? Anything else is drawn again."""
    if not lineage.image_decodes(path):
        log(f"  {task['slot_id']} s{task['seed']}: the file on disk does not decode; drawing it again")
        return False
    recorded = task.get("sha256")
    if not recorded:
        return False               # no recorded bytes: an earlier plan's file, or a crashed write - redraw
    return lineage.sha256_file(path) == recorded


def _child_shape(project, task):
    """(generation, edit_depth) the edit is recorded at, from the plan when it carries them.

    The plan decides the child's shape from the verified source (a direct edit of a user original is generation 1,
    depth 1; an edit of a candidate keeps the parent's generation and goes one deeper - spec R3). A manifest written
    before the plan carried these fields is derived honestly instead: the parent's own record, or the product shape for
    an original when the parent is a source.
    """
    if task.get("generation") is not None and task.get("edit_depth") is not None:
        return int(task["generation"]), int(task["edit_depth"])
    parent = project.candidate_by_id(task.get("source_id") or "")
    if parent is not None:
        return int(parent.get("train_generation") or 0), int(parent.get("edit_depth") or 0) + 1
    return 1, 1


def records_of(project, manifest):
    """Lineage records for the finished edits, ready for `Project.import_candidates`.

    The source as parent is the whole point: an edit of an accepted picture is traceable to the picture it restaged,
    and the depth the plan computed is what the seed caps (`max_edit_depth`) read when such a picture asks to enter a
    training set.
    """
    folder = manifest_path(project).parent
    names = documented_models(manifest)
    records = []
    for task in manifest["tasks"]:
        out = folder / task["file"]
        if not out.is_file():
            continue
        if task.get("sha256") and lineage.sha256_file(out) != task["sha256"]:
            continue               # the file on disk is not the bytes the plan recorded: not this plan's output
        generation, depth = _child_shape(project, task)
        slot = {"slot_id": task["slot_id"], "category": task["category"], "prompt": task["instruction"]}
        records.append(lineage.record(
            out, generator=GENERATOR, parent_ids=[task["source_id"]], train_generation=generation, edit_depth=depth,
            slot=slot, models=names, prompt=task["instruction"], seed=task["seed"],
            steps=int(manifest.get("steps") or STEPS), cfg=0.0,
            graph_version="ph3-edit-v1", method=METHOD,
        ))
    return records


def documented_models(manifest):
    """The four P-H3 files, as the lineage wants them: a name, and for the VAE a real digest.

    Only names are known here; hashing 25 GB to record four checksums is not worth the minutes it would cost on every
    import, and the files are unpinned ComfyUI models rather than trained artefacts this project produced.
    """
    return {"dit": {"name": ph3_gen.UNET, "family": "ph3"},
            "text_encoder": [{"name": ph3_gen.TEXT_ENCODER}, {"name": ph3_gen.UPSAMPLER}],
            "vae": {"name": ph3_gen.VAE}}


def import_results(project, manifest, *, log=print):
    """Register the finished edits as candidates, then let the normal review flow take over."""
    records = records_of(project, manifest)
    if not records:
        log("[ph3_edit] nothing to import")
        return []
    project.import_candidates(records, log=log)
    manifest["status"] = "imported"
    write_manifest(project, manifest)
    log(f"[ph3_edit] imported {len(records)} candidate(s)")
    return records


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="pipeline.ph3_edit",
                                     description="fill short slots by editing a picture P-H3 already got right")
    parser.add_argument("--project", required=True)
    parser.add_argument("--comfy", help="ComfyUI URL, default from prefs.json")
    parser.add_argument("--slot", action="append", help="only this slot (repeatable)")
    parser.add_argument("--seed", action="append", type=int, help="seed to draw (repeatable)")
    parser.add_argument("--source", action="append", help="only edit this picture id (repeatable)")
    parser.add_argument("--import-only", action="store_true", help="import the pictures already on disk")
    parser.add_argument("--plan-only", action="store_true",
                        help="write the plan and stop: no ComfyUI connection, no drawing, no import")
    parser.add_argument("--cancel-file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    project = pj.Project.open(args.project)
    try:
        if args.import_only:
            records = import_results(project, read_manifest(project))
            if args.json:
                print(json.dumps({"imported": len(records)}, ensure_ascii=False))
            return 0
        manifest = plan(project, slots=args.slot, seeds=args.seed, source_ids=args.source)
        if args.plan_only:
            # The plan is the whole deliverable: no GPU is contacted and nothing is imported (spec R3).
            if args.json:
                print(json.dumps({"planned": len(manifest["tasks"]), "gaps": len(manifest["gaps"])},
                                 ensure_ascii=False))
            return 0
        if not manifest["tasks"]:
            if args.json:
                print(json.dumps({"planned": 0, "gaps": len(manifest["gaps"])}, ensure_ascii=False))
            return 0
        client = epochs_mod.make_client(args.comfy)
        cancel = (lambda: Path(args.cancel_file).is_file()) if args.cancel_file else None
        run(project, manifest, client=client, cancel=cancel)
        failed = sum(1 for task in manifest["tasks"] if task.get("status") == "failed")
        if failed:
            # Partial work is not a finished run: the import still recovers what was drawn, and the exit code says
            # the plan did not complete.
            print(f"[ph3_edit] {failed} task(s) failed; importing what was drawn, exit 1")
            import_results(project, manifest)
            return 1
        import_results(project, manifest)
    except (EditError, pj.ProjectError, comfy.ComfyError) as exc:
        print(f"[ph3_edit] {exc}")
        return 1
    if args.json:
        print(json.dumps({"planned": len(manifest["tasks"]), "status": manifest["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
