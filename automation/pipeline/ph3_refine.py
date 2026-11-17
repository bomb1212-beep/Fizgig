"""Polish batch B's side-profile pictures with P-H3, the originals as reference (2026-11-17, method "C").

Krea2 with the character LoRA gets a side profile's pose, camera and background right but not quite the face: the
originals have few side views to learn one from. P-H3 takes the Krea2 picture as Picture 1 and two training originals
as Pictures 2-3 and is told to keep everything of Picture 1 except the face, hair and line art, which it redraws after
the originals. Of the three ways tried on suamom (Krea2 alone, P-H3 turning an original, this) the user picked this.

* **What is polished** - batch B candidates (`method == "krea2_b"`) of composition cells whose view is in the card's
  `ph3_refine.views` (default: side profile), not rejected in review, and not polished already. Undecided pictures
  count: the point is to have the polished one next to the raw one when the reviewer looks.
* **References** - training originals only (never a holdout, by id or bytes - `ph3_edit.sources` checks them as edit
  sources): one that shows the face large (portrait, close-up, bust), and one side view when there is one.
* **The result** - a new candidate in the same slot, `generator="ph3_edit"`, `method="ph3_refine"`, with the Krea2
  picture and the references as parents (it is made of all three), one edit deeper than the Krea2 picture. Review then
  keeps whichever of the two is better.

  python -m pipeline.ph3_refine --project <folder> [--plan-only] [--import-only] [--view "side profile"]
"""
import json
import os
import time
from pathlib import Path

from pipeline import comfy, composition, epochs as epochs_mod, lineage, ph3_edit, ph3_gen, prompting
from pipeline import project as pj

DEFAULT_VIEWS = ("side profile",)
SEED = 7301
STEPS = ph3_edit.STEPS
METHOD = "ph3_refine"
GENERATOR = "ph3_edit"           # a P-H3 edit as far as lineage and the seed policies are concerned
FACE_SHOTS = ("portrait", "close-up", "bust shot", "upper body")
GRAPH_VERSION = "ph3-refine-v1"


class RefineError(RuntimeError):
    """A refine that cannot be planned or run."""


def manifest_path(project):
    return project.dir / "experiments" / "ph3_refine" / "refine.json"


def read_manifest(project):
    path = manifest_path(project)
    if not path.is_file():
        raise RefineError(f"no P-H3 refine planned at {path}; plan one first")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(project, manifest):
    path = manifest_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + os.urandom(4).hex())
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def views(card):
    block = card.get("ph3_refine") or {}
    chosen = [str(v) for v in (block.get("views") or DEFAULT_VIEWS) if str(v) in composition.VIEWS]
    return tuple(chosen) or DEFAULT_VIEWS


def view_of(slot_id):
    """The view a composition slot id spells out, or None for any other slot."""
    if not composition.is_cell(slot_id):
        return None
    parts = str(slot_id)[len("cmp_"):].split("_")
    decode = {code: word for word, code in composition.CODE.items()}
    return decode.get(parts[1]) if len(parts) > 1 else None


def instruction(card):
    """Keep Picture 1, redraw the woman after the references - with the card's traits named."""
    who = prompting.profile(card)
    person = who["gender"] or "person"
    traits = prompting.traits(card, "face_traits") + prompting.traits(card, "fixed_traits")
    outfit = str((card.get("outfits") or {}).get("default") or "").strip()
    named = "".join(f", {t}" for t in traits)
    return (f"Keep the pose, the camera angle, the framing, the clothing and the background of Picture 1 exactly. "
            f"Redraw only the {person}'s face, hair and line art so the {person} looks exactly like the {person} in "
            f"the other pictures: the same face shape, eyes, nose and jawline from this angle{named}, the same "
            f"hairstyle{' and ' + outfit if outfit else ''}, and the same line art and coloring.")


def references(project, usable):
    """Up to two training originals to copy the face from: one large face, one side view. Holdouts never appear:
    `usable` is ph3_edit.sources' list, which has already refused them."""
    by_id = {row["id"]: row for row in usable if row["origin"] == "source"}
    originals = [o for o in composition.originals(project) if o["id"] in by_id]
    face = [o for o in originals if o["shot"] in FACE_SHOTS and o["view"] != "back view"]
    face.sort(key=lambda o: (FACE_SHOTS.index(o["shot"]), o["view"] == "side profile"))
    side = [o for o in originals if o["view"] == "side profile"]
    side.sort(key=lambda o: (o["shot"] not in FACE_SHOTS, o["shot"]))
    picked = []
    for group in (face, side):
        for o in group:
            if o["id"] not in picked:
                picked.append(o["id"])
                break
    return [{"id": i, "path": by_id[i]["path"], "sha256": by_id[i]["sha256"]} for i in picked]


def targets(project, usable, wanted_views):
    """The batch B pictures to polish: verified sources, in a wanted view, not yet polished."""
    polished = {pid for rec in project.candidates() if rec.get("method") == METHOD
                for pid in (rec.get("parent_ids") or [])[:1]}
    by_id = {rec["id"]: rec for rec in project.candidates()}
    out = []
    for row in usable:
        record = by_id.get(row["id"])
        if row["origin"] != "candidate" or record is None or row["id"] in polished:
            continue
        slot_id = (record.get("slot") or {}).get("slot_id")
        if view_of(slot_id) in wanted_views:
            out.append({**row, "slot": dict(record.get("slot") or {})})
    return out


def plan(project, *, seed=SEED, only=None, log=print):
    wanted = views(project.card)
    usable, unusable = ph3_edit.sources(
        project, log=None,
        candidate_ok=lambda record, decision: record.get("method") == "krea2_b" and decision != "reject")
    refs = references(project, usable)
    if not refs:
        raise RefineError("no training original is usable as a reference (every one is a holdout or missing)")
    text = instruction(project.card)
    tasks = []
    for row in targets(project, usable, wanted):
        if only and row["id"] not in only:
            continue
        slot = row["slot"]
        tasks.append({
            "slot_id": slot.get("slot_id"), "category": slot.get("category") or "composition", "seed": int(seed),
            "instruction": text, "source_id": row["id"], "source_path": row["path"],
            "source_sha256": row["sha256"], "references": refs,
            "generation": row["generation"], "edit_depth": row["edit_depth"],
            "file": f"{slot.get('slot_id')}__{row['id']}_s{seed}.png",
            "status": "pending", "seconds": None, "error": "",
        })
    manifest = {"kind": METHOD, "project": project.card.get("id"), "created_at": _now(), "status": "planned",
                "views": list(wanted), "steps": STEPS, "references": refs, "instruction": text,
                "unusable_sources": [u for u in unusable if u["id"].startswith("c_")], "tasks": tasks}
    write_manifest(project, manifest)
    log(f"[ph3_refine] {len(tasks)} picture(s) to polish in view(s) {', '.join(wanted)}; references "
        + ", ".join(r["id"] for r in refs))
    return manifest


def run(project, manifest, *, client, log=print, cancel=None):
    """Draw every pending task; a source or reference whose bytes changed since planning fails its task."""
    folder = manifest_path(project).parent
    names = ph3_gen.resolve_names(client)
    uploaded = {}
    done = skipped = failed = 0

    def upload(path, sha):
        if sha and lineage.sha256_file(path) != sha:
            raise RefineError(f"{path} changed since the plan pinned it")
        if path not in uploaded:
            uploaded[path] = client.upload(path, subfolder="ph3_refine")
        return uploaded[path]

    for task in manifest["tasks"]:
        out = folder / task["file"]
        if out.is_file() and ph3_edit._reusable(out, task, log):
            task["status"] = "done"
            skipped += 1
            continue
        if cancel is not None and cancel():
            log("cancelled")
            break
        started = time.time()
        try:
            refs = [upload(task["source_path"], task.get("source_sha256"))]
            refs += [upload(r["path"], r.get("sha256")) for r in task.get("references") or []]
            graph = ph3_gen.build(prompt=task["instruction"], instruction=task["instruction"], references=refs,
                                  seed=task["seed"], steps=manifest.get("steps") or STEPS, names=names,
                                  filename_prefix=f"pipeline_ph3_refine/{task['slot_id']}")
            out.parent.mkdir(parents=True, exist_ok=True)
            client.run(graph, out, slot_id=task["slot_id"], method=METHOD, seed=task["seed"],
                       source=task["source_id"])
            task["status"] = "done"
            task["sha256"] = lineage.sha256_file(out)
            done += 1
            log(f"  {task['slot_id']} ({task['source_id']}): {time.time() - started:.0f}s")
        except Exception as exc:  # noqa: BLE001 - one failed picture must not end the batch
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            log(f"  {task['slot_id']} ({task['source_id']}): FAILED {task['error'][:200]}")
        finally:
            task["seconds"] = round(time.time() - started, 1)
            write_manifest(project, manifest)
    manifest["status"] = "ran"
    write_manifest(project, manifest)
    log(f"done {done}, skipped {skipped} already present, failed {failed}")
    return manifest


def records_of(project, manifest):
    folder = manifest_path(project).parent
    records = []
    for task in manifest["tasks"]:
        out = folder / task["file"]
        if not out.is_file() or not task.get("sha256") or lineage.sha256_file(out) != task["sha256"]:
            continue
        parents = [task["source_id"]] + [r["id"] for r in task.get("references") or []]
        slot = {"slot_id": task["slot_id"], "category": task["category"], "prompt": task["instruction"]}
        records.append(lineage.record(
            out, generator=GENERATOR, parent_ids=parents, train_generation=int(task["generation"]),
            edit_depth=int(task["edit_depth"]), slot=slot, models=ph3_edit.documented_models(manifest),
            prompt=task["instruction"], seed=task["seed"], steps=int(manifest.get("steps") or STEPS), cfg=0.0,
            graph_version=GRAPH_VERSION, method=METHOD,
        ))
    return records


def import_results(project, manifest, *, log=print):
    records = records_of(project, manifest)
    if not records:
        log("[ph3_refine] nothing to import")
        return []
    project.import_candidates(records, log=log)
    manifest["status"] = "imported"
    write_manifest(project, manifest)
    log(f"[ph3_refine] imported {len(records)} candidate(s)")
    return records


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="pipeline.ph3_refine", description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--comfy", help="ComfyUI URL, default from prefs.json")
    parser.add_argument("--source", action="append", help="only polish this candidate id (repeatable)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--plan-only", action="store_true", help="write the plan and stop")
    parser.add_argument("--import-only", action="store_true", help="import the pictures already on disk")
    parser.add_argument("--cancel-file")
    args = parser.parse_args(argv)

    project = pj.Project.open(args.project)
    try:
        if args.import_only:
            import_results(project, read_manifest(project))
            return 0
        manifest = plan(project, seed=args.seed, only=set(args.source or []) or None)
        if args.plan_only or not manifest["tasks"]:
            return 0
        from . import gpu
        claim = gpu.claim_generation(what="P-H3 refine")
        try:
            with claim:
                client = epochs_mod.make_client(args.comfy)
                if not client.ping():
                    raise RefineError(f"ComfyUI is not answering at {client.url}")
                cancel = (lambda: Path(args.cancel_file).is_file()) if args.cancel_file else None
                run(project, manifest, client=client, cancel=cancel)
        finally:
            claim.release()
        import_results(project, manifest)
        failed = sum(1 for task in manifest["tasks"] if task.get("status") == "failed")
        if failed:
            print(f"[ph3_refine] {failed} task(s) failed; imported what was drawn, exit 1")
            return 1
    except (RefineError, ph3_edit.EditError, pj.ProjectError, comfy.ComfyError) as exc:
        print(f"[ph3_refine] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
