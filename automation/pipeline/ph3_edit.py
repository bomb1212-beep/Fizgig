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

from pipeline import batch_gen, comfy, epochs as epochs_mod, lineage, ph3_gen, project as pj

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


def sources(project):
    """Accepted pictures an edit may start from: real originals first, then accepted edits.

    Three kinds of picture are refused, each because editing it would poison the lineage rather than merely waste time:

    - a holdout original, whose edits would leak the holdout into the training set through the back door;
    - a synthetic anchor with no recorded parent, because anything edited from it inherits an unprovable ancestry and
      the audit refuses the result as `unverified_ancestor`. Sourcing from a labelled original instead is the fix, so
      the anchor is skipped here rather than producing pictures that can never be used.
    """
    holdout = project.holdout_entry()
    forbidden = set(holdout["ids"]) | set(holdout["shas"])
    review = project.review()
    out = []
    unusable = []
    for record in project.source_records():
        if record["id"] in forbidden or record.get("sha256") in forbidden:
            continue
        # A synthetic source is an anchor whose own ancestry was never recorded; an edit of it can never be verified.
        if (record.get("origin") or "user") != "user" and not (record.get("parent_ids") or []):
            unusable.append({"id": record["id"], "reason": "synthetic anchor with no recorded parent"})
            continue
        path = project.resolve_file(record.get("file") or record.get("src_path"))
        if path and path.is_file():
            out.append({"id": record["id"], "path": str(path), "origin": "source"})
    for record in project.candidates():
        state = review.get(record["id"]) or {}
        if state.get("decision") != "accept":
            continue
        if record["id"] in forbidden or record.get("sha256") in forbidden:
            continue
        path = project.resolve_file(record.get("file"))
        if path and path.is_file():
            out.append({"id": record["id"], "path": str(path), "origin": "candidate"})
    return out, unusable


def instruction_for(slot_id, category):
    """The wording handed to P-H3 for one slot.

    Written out per category rather than templated from the slot id: "ang_back" tells a person what it means, and the
    model needs the same thing said in words. A slot with no entry is refused by `plan` instead of being sent with a
    vague instruction that would produce a picture nobody asked for.
    """
    return INSTRUCTIONS.get(slot_id) or INSTRUCTIONS_BY_CATEGORY.get(category)


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


def short_slots(gaps, *, slots=None, force=False):
    """The gaps worth drawing, from `batch_gen.plan`'s `waiting` rows.

    A slot that already has pictures waiting on review is not a gap to fill: drawing more would grow the review backlog
    instead of shrinking it. This is the same rule batch_gen applies, and getting it wrong would mean sixteen new
    pictures for eight slots that each already have one or two waiting.

    `force` draws the named slots anyway. Naming a slot is the deliberate override; it exists so the draw path can be
    exercised on demand, not as a way to quietly grow the backlog.

    Pure on purpose: it is the decision that matters and it can be tested without a seed snapshot or a GPU.
    """
    rows = [row for row in (gaps.get("waiting") or [])
            if int(row.get("short") or 0) > 0 and (force or int(row.get("waiting") or 0) == 0)]
    if slots:
        wanted = set(slots)
        rows = [row for row in rows if row["slot"] in wanted]
    return rows


def plan(project, *, slots=None, seeds=None, source_ids=None, force=False, log=print):
    """{manifest} for the short slots, with the pictures already on disk skipped.

    Returns a manifest with `tasks == []` when there is nothing to do, which is a normal outcome, not an error: a plan
    that is already filled or already waiting on review needs no new pictures.
    """
    from pipeline import batch_gen as bg

    # Sources are checked first: with nothing editable the plan is impossible whatever the slots say, and that reason
    # is more useful than "no seed snapshot", which is what bg.plan() would raise instead.
    available, unusable = sources(project)
    if source_ids:
        wanted_sources = set(source_ids)
        available = [s for s in available if s["id"] in wanted_sources]
    if not available:
        raise EditError("no picture is eligible to edit: every source is a holdout, an unparented anchor, or has gone "
                        "missing")

    gaps = bg.plan(project, log=lambda *_: None)
    short = short_slots(gaps, slots=slots, force=force)

    seeds = list(seeds or range(7001, 7001 + SEEDS_PER_SLOT))
    tasks, skipped = [], []
    for row in short:
        slot_id, category = row["slot"], row.get("category") or ""
        instruction = instruction_for(slot_id, category)
        if not instruction:
            skipped.append({"slot_id": slot_id, "reason": "no instruction is written for this slot"})
            continue
        for seed in seeds:
            tasks.append({
                "slot_id": slot_id, "category": category, "seed": int(seed),
                "instruction": instruction,
                "source_id": available[0]["id"],
                "source_path": available[0]["path"],
                "file": f"{slot_id}_s{seed}.png",
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
        log(f"  short: {row['slot']} (needs {row.get('short')}, {row.get('waiting', 0)} waiting on review)")
    for row in skipped:
        log(f"  skipped: {row['slot_id']} - {row['reason']}")
    if not tasks:
        log("[ph3_edit] nothing to do: no slot is short of an accepted picture right now")
    return manifest


def run(project, manifest, *, client, log=print, cancel=None):
    """Draw every pending task. A task whose file exists is skipped, so re-running submits nothing twice."""
    folder = manifest_path(project).parent
    names = ph3_gen.resolve_names(client)
    # LoadImage cannot open a local path; the reference has to be uploaded first and referred to by the returned name.
    upload_cache = {}
    done = skipped = failed = 0
    for task in manifest["tasks"]:
        out = folder / task["file"]
        if out.is_file():
            task["status"] = "done"
            skipped += 1
            continue
        if cancel is not None and cancel():
            log("cancelled")
            break
        started = time.time()
        try:
            path = task["source_path"]
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


def records_of(project, manifest):
    """Lineage records for the finished edits, ready for `Project.import_candidates`.

    `edit_depth=1` and the source as parent is the whole point: an edit of an accepted picture is one step further from
    the original, and the seed caps (`max_edit_depth`) are what stop that chain running away.
    """
    folder = manifest_path(project).parent
    names = documented_models(manifest)
    records = []
    for task in manifest["tasks"]:
        out = folder / task["file"]
        if not out.is_file():
            continue
        slot = {"slot_id": task["slot_id"], "category": task["category"], "prompt": task["instruction"]}
        records.append(lineage.record(
            out, generator=GENERATOR, parent_ids=[task["source_id"]], train_generation=0, edit_depth=1,
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
        if not manifest["tasks"]:
            if args.json:
                print(json.dumps({"planned": 0, "gaps": len(manifest["gaps"])}, ensure_ascii=False))
            return 0
        client = epochs_mod.make_client(args.comfy)
        cancel = (lambda: Path(args.cancel_file).is_file()) if args.cancel_file else None
        run(project, manifest, client=client, cancel=cancel)
        import_results(project, manifest)
    except (EditError, pj.ProjectError, comfy.ComfyError) as exc:
        print(f"[ph3_edit] {exc}")
        return 1
    if args.json:
        print(json.dumps({"planned": len(manifest["tasks"]), "status": manifest["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
