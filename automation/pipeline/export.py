"""M4: the frozen export - the set that is actually trained on, and the record of why each picture is in it.

The seed snapshot (`seed.py`) is built from accepted candidates and is the *starting* set. This is the final one, and
since R2 it answers a different question from the seed with rules of its own: it takes every valid user original plus
every candidate the reviewer explicitly accepted that the current pipeline plan actually names - a Krea2 batch B
picture, a correction, an edit - applies the **final** generation ceiling (`max_train_generation`) and the edit-depth
cap, and freezes the result under an id that changes the moment anything in it changes. The checks the two sets share
- holdout separation, lineage, bytes, duplicates, the reviewer's verdict - are one code path in `seed.audit`, so the
export cannot quietly disagree with the seed about what is allowed; the seed's own ceiling and generator list stay
what they always were (`seed-v1`/`seed-v2`).

Three properties the spec asks for, and how each is kept:

* **Only eligible pictures.** `seed.audit(purpose="final")` decides that - the final generation cap, the edit-depth
  cap, holdout separation, lineage that reaches a recorded root, a reviewer's accept, bytes that still hash to what
  was recorded, and for a batch B picture a plan manifest that names it (slot, seed and bytes). What this module adds
  is the refusal of anything still awaiting review, which the audit already produces as `not_reviewed`.
* **Immutable.** The id is a hash over every image and caption in the set. The same set always produces the same id; a
  changed caption, a new picture or a removed one produces a different id. Publishing never overwrites: a second export
  with the same id is verified and reused, and a different id gets its own folder.
* **Traceable.** `export_manifest.json` records every image's candidate/source id, image sha256 and caption sha256,
  the exclusion rules that ran, and the distributions a person checks at a glance (generation, slot, synthetic share,
  white background).

The export is what a final training run reads, so a picture that cannot be traced to an original must not be in it -
and the boundary between publishing and training is enforced, not assumed: `validate()` refuses a folder whose disk
does not match its manifest in either direction, `publish()` refuses a partial set, and `training_gate()` stands
between a training job and the frozen files it names.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from . import lineage
from . import project as pj
from . import seed

# The policy tag travels with the report and the manifest, so an export keeps meaning what it meant when it was
# written. It lives in `seed` next to the rules it names; re-defining it here would let the two drift apart.
EXPORT_POLICY = seed.EXPORT_POLICY
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class ExportError(RuntimeError):
    """An export that cannot be made as asked."""


def _caption_of(entry, captions, card):
    """The caption that goes into the export, and where it came from (spec N6).

    A candidate that already has a caption keeps its wording: it describes the picture that was drawn, and replacing it
    with the card's boilerplate would throw away what the pipeline actually measured. The card's `caption_traits`
    policy is applied **to that inherited text as well as to a caption built from the card** - the same rule the seed
    uses, so the two sets caption alike. Before this, `omit` only applied when a caption was built from scratch, so an
    inherited caption kept the fixed traits and the default outfit it was supposed to drop.

    Returns `(caption, source, detail)`; `detail` is the policy preview (original text, what was dropped, and the note
    a person should read when the caption was free prose rather than a tag list).
    """
    text = (entry.get("caption") or captions.get(entry.get("sha256") or "") or "").strip()
    if text:
        source = entry.get("caption_source") or "inherited"
        original = text
    else:
        text = seed.card_caption(card)
        source = "built from the card"
        original = ""
    preview = seed.caption_policy_preview(text, card)
    detail = {"original_caption": original or None, "policy": preview["policy"], "kind": preview["kind"],
              "dropped": preview["dropped"], "note": preview["note"],
              "policy_version": seed.CAPTION_POLICY_VERSION}
    return preview["after"], source, detail



def collect(project, *, caps=None, log=print):
    """Everything that may go into an export, with the refusals that produced it.

    Returns `(entries, report)`: `entries` are the eligible pictures (deduplicated by bytes, each with its lineage ids),
    `report` is the audit that decided it. Nothing here writes. The audit runs with `purpose="final"`: the seed's own
    ceiling and generator list are untouched, and this is the rule set the frozen set is published under.
    """
    caps = caps or lineage.caps(project.card.get("lineage"))
    report = seed.audit(project, caps=caps, purpose="final", policy=EXPORT_POLICY)
    captions = seed.tool2_captions((project.card.get("tool2") or {}).get("dataset"))
    holdout = project.holdout_entry()

    entries = []
    records = {record["id"]: record for record in project.candidates()}
    for row in report["kept"]:
        # Both the user's originals and the accepted candidates go in, exactly as `seed.build_snapshot` does: an export
        # is the final training set, and a set that leaves out the ground-truth photographs would train a different
        # kind of model than the seed did. `kind` keeps them distinguishable, and the holdout is already gone because
        # `audit` refuses it before this point.
        caption, source, caption_detail = _caption_of(row, captions, project.card)
        # `kept` carries no generator, so it is read from the candidate record it names; an original has none.
        generator = str((records.get(row["id"]) or {}).get("generator") or "")
        entries.append({
            "id": row["id"], "kind": row.get("kind") or "candidate",
            "lineage_ids": list(row.get("lineage_ids") or [row["id"]]),
            "sha256": row["sha256"], "path": row["path"],
            "caption": caption, "caption_source": source,
            "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            # spec N6: the policy and the original text travel with the caption, so a surprising edit is auditable
            # instead of being a difference nobody can explain later.
            "caption_policy": caption_detail,
            "slot": row.get("slot") or "", "generation": int(row.get("generation") or 0),
            "edit_depth": int(row.get("edit_depth") or 0),
            "origin": row.get("origin") or "",
            "generator": generator,
            "decision_by": row.get("decision_by") or "",
        })
    return entries, report, holdout, caps


def identity(project, entries, *, caps, holdout, policy=EXPORT_POLICY):
    """The snapshot id: the project, the policy, the caps, the holdout, and every image and caption by hash.

    Computed exactly like the seed's, so the same set exported twice is recognised as the same set, and any change to an
    image, a caption or the eligibility rules makes a different one.
    """
    payload = {
        "project": project.card.get("id"), "card": project.digest(), "policy": policy, "caps": caps,
        "holdout_shas": sorted(holdout["shas"]),
        "inputs": sorted(({"id": e["id"], "lineage_ids": e["lineage_ids"], "image": e["sha256"],
                           "caption": e["caption_sha256"], "slot": e["slot"]} for e in entries),
                         key=lambda item: item["id"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def distributions(entries):
    """The counts a person checks before training: generations, slots, where each image came from, white background.

    `generated` and `original` are counted by `kind`, not by `origin`: a candidate record carries no `origin` field, so
    treating a missing one as "synthetic" labelled every generated picture correctly by luck and would have labelled a
    future one wrongly. The user's originals are the ones whose kind says so.
    """
    by_generation, by_slot, by_kind, by_generator = {}, {}, {}, {}
    for entry in entries:
        by_generation[str(int(entry.get("generation") or 0))] = \
            by_generation.get(str(int(entry.get("generation") or 0)), 0) + 1
        slot = entry.get("slot") or "(none)"
        by_slot[slot] = by_slot.get(slot, 0) + 1
        kind = entry.get("kind") or "candidate"
        by_kind[kind] = by_kind.get(kind, 0) + 1
        generator = entry.get("generator") or "(original)"
        by_generator[generator] = by_generator.get(generator, 0) + 1
    generated = by_kind.get("candidate", 0)
    originals = by_kind.get("source", 0)
    return {
        "total": len(entries),
        "by_generation": dict(sorted(by_generation.items())),
        "by_slot": dict(sorted(by_slot.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_generator": dict(sorted(by_generator.items())),
        "generated": generated,
        "original": originals,
        "generated_share": (generated / len(entries)) if entries else 0.0,
    }


def base_dir(project):
    return project.dir / "exports"


def export_dir(project, identity_value):
    return base_dir(project) / identity_value


def validate(base):
    """Every way a published export can be wrong, as sentences; empty means it is intact.

    The check is bidirectional (spec R2): every manifest entry must be on disk and still hash to what was recorded,
    and every image and caption on disk must be one the manifest declares. An unlisted PNG in the folder is exactly as
    wrong as a missing one - the runner reads the folder, so anything on disk that the manifest does not name would
    quietly become training data.
    """
    base = Path(base)
    manifest = pj.read_json(base / "export_manifest.json")
    if not manifest:
        return [f"no readable export_manifest.json in {base}"]
    entries = manifest.get("images") or []
    problems = []
    shas = [entry.get("sha256") for entry in entries]
    if len(set(shas)) != len(shas):
        problems.append(f"the manifest lists {len(shas) - len(set(shas))} duplicate sha256 value(s)")
    listed_images = {str(entry.get("file") or "") for entry in entries}
    listed_captions = {str(Path(entry.get("file") or "").with_suffix(".txt").name) for entry in entries}
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.name == "export_manifest.json":
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES and path.name not in listed_images:
            problems.append(f"unlisted image: {path.name} is on disk but the manifest does not declare it")
        elif path.suffix.lower() == ".txt" and path.name not in listed_captions:
            problems.append(f"unlisted caption: {path.name} is on disk but no manifest image uses it")
    for entry in entries:
        image = base / str(entry.get("file") or "")
        if not image.is_file():
            problems.append(f"missing image: {image.name}")
            continue
        if lineage.sha256_file(image) != entry.get("sha256"):
            problems.append(f"{image.name} no longer hashes to its recorded sha256")
        sidecar = image.with_suffix(".txt")
        if not sidecar.is_file():
            problems.append(f"missing caption: {sidecar.name}")
            continue
        text = sidecar.read_text(encoding="utf-8").strip()
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != entry.get("caption_sha256"):
            problems.append(f"caption of {image.name} does not match its recorded hash")
    return problems


def publish(project, *, caps=None, families=None, log=print):
    """Write the frozen export and return its manifest.

    An export whose id already exists is verified and reused rather than rewritten, so re-running the stage is safe. An
    export whose id exists but whose files do not match is an error: something changed a frozen set, and silently
    republishing over it would hide that. An eligible picture that vanishes between the audit and the copy refuses the
    whole export (spec R2): a set that is missing a picture it declared eligible is not a smaller export, it is a
    wrong one, and half of it is never published.
    """
    entries, report, holdout, caps = collect(project, caps=caps, log=log)
    if not entries:
        raise ExportError("nothing is eligible for export: " + _why(report))
    value = identity(project, entries, caps=caps, holdout=holdout)
    base = export_dir(project, value)
    if base.exists():
        problems = validate(base)
        if not problems:
            manifest = pj.read_json(base / "export_manifest.json")
            log(f"[export] {value} already published and intact; reusing it ({len(entries)} image(s))")
            return {**manifest, "reused": True}
        raise ExportError(
            f"the export folder {base} exists but is not intact: " + "; ".join(problems[:5])
            + ". Move it aside and run again; a frozen export is not overwritten.")
    if base_dir(project).exists() and any(base_dir(project).iterdir()):
        log(f"[export] a different export already exists under {base_dir(project)}; this one gets its own folder")

    staging = base.with_name(f"{value}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    try:
        staging.mkdir(parents=True)
        written = []
        for entry in entries:
            source = Path(entry["path"])
            if not source.is_file():
                raise ExportError(f"{entry['id']} vanished before it could be copied ({source}); "
                                  "refusing to publish a partial export")
            digest = lineage.sha256_file(source)
            if digest != entry["sha256"]:
                raise ExportError(f"{source.name} changed while the export was being written")
            dest = staging / (digest[:16] + source.suffix.lower())
            shutil.copy2(source, dest)
            dest.with_suffix(".txt").write_text(entry["caption"] + "\n", encoding="utf-8")
            written.append({**{k: v for k, v in entry.items() if k != "path"},
                            "file": dest.name, "source_path": entry["path"]})
        if not written:
            raise ExportError("none of the eligible images exist on disk any more")

        white = _white_background([staging / entry["file"] for entry in written])
        manifest = {
            "kind": "export", "id": value, "base": str(base), "dataset": str(base),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "project": project.card.get("id"), "card_digest": project.digest(), "policy": EXPORT_POLICY,
            "lineage_caps": caps,
            "families": list(families or project.card.get("final_families") or []),
            "holdout_frozen": {"ids": sorted(holdout["ids"]), "shas": sorted(holdout["shas"])},
            "holdout_files": list(holdout.get("files") or []),
            "caption_policy": str(project.card.get("caption_traits") or "omit"),
            "images": written,
            # The rules that ran, and what they refused, so the set can be explained rather than only described.
            "excluded_counts": report["counts"]["by_code"],
            "excluded_by_category": report["counts"]["by_category"],
            "excluded": [{k: row.get(k) for k in ("id", "code", "category", "reason")}
                         for row in report["excluded"]],
            "distributions": distributions(written),
            "white_background": white,
            "quotas": report["quotas"],
            "missing_at_publish": [],
            "counts": {"images": len(written), "refused": len(report["excluded"]),
                       "skipped_missing": 0},
        }
        (staging / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        problems = validate(staging)
        if problems:
            raise ExportError("the export failed its own integrity check: " + "; ".join(problems[:5]))
        os.replace(staging, base)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log(f"[export] {value}: {len(written)} image(s), {len(report['excluded'])} refused -> {base}")
    return manifest


def _white_background(paths):
    """How white the exported pictures' backgrounds are, using the same border measure the scorer uses.

    Reported as a **distribution**, not just a mean: on this dataset generated frames sit at a median border value of
    251-253 rather than pure white while the originals reach 255, so a mean alone reads as "the export is not on white"
    when the truth is "the generator's white is slightly off-white". The caller sees the spread and decides.
    """
    from .scorers import common as scorers_common
    rows = [{"id": str(path), "path": str(path)} for path in paths]
    scores = scorers_common.white_scores(rows)
    ratios = sorted(row["white_ratio"] for row in scores["rows"])
    if not ratios:
        return {"measured": 0, "unreadable": len(scores["coverage"]["unreadable"])}
    n = len(ratios)
    return {
        "measured": n,
        "mean": round(sum(ratios) / n, 4),
        "median": round(ratios[n // 2], 4),
        "min": round(ratios[0], 4),
        "max": round(ratios[-1], 4),
        "share_above_0.9": round(sum(1 for r in ratios if r >= 0.9) / n, 4),
        "share_below_0.5": round(sum(1 for r in ratios if r < 0.5) / n, 4),
        "unreadable": len(scores["coverage"]["unreadable"]),
        "note": "border-only measure; a low ratio means the frame is filled with something other than a plain background",
    }


def _why(report):
    codes = report["counts"]["by_code"]
    if not codes:
        return "the project has no candidates at all"
    return "nothing passed the eligibility rules (" + ", ".join(
        f"{code} {n}" for code, n in sorted(codes.items())) + ")"


def listed(project):
    """Every published export, newest first, with its integrity state."""
    out = []
    root = base_dir(project)
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        manifest = child / "export_manifest.json"
        if not manifest.is_file():
            continue
        data = pj.read_json(manifest)
        problems = validate(child)
        out.append({"id": data.get("id") or child.name, "base": str(child),
                    "created_at": data.get("created_at"), "images": data.get("counts", {}).get("images"),
                    "intact": not problems, "problems": problems[:5]})
    out.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return out


def chosen(project):
    """The export the project records as its final set, or None."""
    return (project.card.get("export") or {}).get("selected")


def todo_path(automation_dir=None):
    """tool1's job list, which is where a training run is started from."""
    base = Path(automation_dir) if automation_dir else Path(__file__).resolve().parents[1]
    return base / "todo.json"


def read_todo(path):
    data = pj.read_json(path)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ExportError(f"{path} is not a list of jobs")
    return data


def plan_todos(project, manifest, *, families=None, automation_dir=None):
    """The job entries that would train on this export, one per family, without writing anything.

    The spec asks after an export whether to add a tool1 todo whose `dataset_root` is this snapshot, per final family.
    This returns them so a caller can show them first; `add_todos` writes. `dataset_root` is left to tool1's
    `config.json` unless a family needs its own, because the exported folder is the same for both.
    """
    families = list(families or manifest.get("families") or [])
    if not families:
        raise ExportError("the export names no final family, so there is nothing to plan a training job for")
    existing = {str(job.get("id") or "") for job in read_todo(todo_path(automation_dir))}
    export_id = manifest["id"]
    out = []
    for family in families:
        # The id carries the export and the family: re-running the same export adds nothing, and a new export is a new
        # job rather than a silent overwrite of the old one's settings.
        job_id = f"{project.card.get('id')}_{export_id}_{family}"
        out.append({
            "id": job_id, "family": family, "enabled": False,
            "trigger": project.card.get("trigger") or project.card.get("id"),
            "dataset_root": manifest.get("dataset") or manifest.get("base"),
            "export": export_id, "export_manifest": str(Path(manifest.get("base") or "") / "export_manifest.json"),
            "present": job_id in existing,
        })
    return out


def training_gate(manifest_path):
    """(manifest) the check between a training job and the frozen set it names.

    A job that trains on an export must train on exactly what the manifest declares: the folder is verified before a
    run reads a single image, so an unlisted picture, a swapped caption or a lost file stops the run with its name
    instead of quietly entering the training set (spec R2). Returns the manifest when the export is intact.
    """
    path = Path(manifest_path)
    manifest = pj.read_json(path)
    if not manifest or not manifest.get("id"):
        raise ExportError(f"no readable export manifest at {path}")
    base = Path(manifest.get("base") or path.parent)
    problems = validate(base)
    if problems:
        raise ExportError(f"the export {manifest['id']} is not intact: " + "; ".join(problems[:5]))
    return manifest


def add_todos(project, manifest, *, families=None, automation_dir=None, enable=False, log=print):
    """Add the planned jobs to `todo.json` without duplicating one that is already there.

    Written disabled by default. Starting a training run is the user's call and consumes hours of GPU time, so this
    adds the entry and leaves it off rather than queuing work nobody asked for. A broken export gets no job at all:
    the todo would carry the folder's path, and a training run that opens it would either fail or train on files the
    manifest does not declare.
    """
    base = Path(manifest.get("base") or "")
    problems = validate(base)
    if problems:
        raise ExportError(f"the export {manifest.get('id')} is not intact, so no training job will be added: "
                          + "; ".join(problems[:5]))
    path = todo_path(automation_dir)
    jobs = read_todo(path)
    known = {str(job.get("id") or "") for job in jobs}
    planned = plan_todos(project, manifest, families=families, automation_dir=automation_dir)
    added = []
    for job in planned:
        if job["id"] in known:
            log(f'[export] todo {job["id"]} already exists; not adding it again')
            continue
        entry = {key: value for key, value in job.items() if key != "present"}
        entry["enabled"] = bool(enable)
        jobs.append(entry)
        added.append(entry)
    if added:
        staging = path.with_suffix(f".tmp-{os.getpid()}")
        staging.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
        staging.replace(path)
        log(f"[export] added {len(added)} job(s) to {path} (disabled: start them when you are ready)")
    return {"added": added, "planned": planned, "path": str(path)}


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="pipeline.export", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("list", "show", "publish", "verify", "todos"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--id", help="the export id to show, verify or plan training jobs for (todos: newest if omitted)")
    parser.add_argument("--families", nargs="*", help="which final families to plan a training job for")
    parser.add_argument("--add", action="store_true", help="write the jobs into todo.json")
    parser.add_argument("--enable", action="store_true", help="add them enabled, so a run would pick them up")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    project = pj.Project.open(args.project)

    if args.command == "list":
        rows = listed(project)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=1))
            return 0
        if not rows:
            print("no export has been published")
            return 0
        for row in rows:
            state = "intact" if row["intact"] else "BROKEN: " + "; ".join(row["problems"])
            print(f'{row["id"]}  {row["images"]} image(s)  {row["created_at"]}  {state}')
        return 0

    if args.command == "show":
        if not args.id:
            raise ExportError("show needs --id")
        base = export_dir(project, args.id)
        data = pj.read_json(base / "export_manifest.json")
        if not data:
            raise ExportError(f"no export {args.id} under {base_dir(project)}")
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=1))
            return 0
        print(f'export {data["id"]}: {data["counts"]["images"]} image(s), '
              f'refused {data["counts"]["refused"]}')
        print("distributions: " + json.dumps(data["distributions"], ensure_ascii=False))
        print("white background: " + json.dumps(data["white_background"], ensure_ascii=False))
        print("refused by rule: " + json.dumps(data["excluded_counts"], ensure_ascii=False))
        return 0

    if args.command == "verify":
        problems = validate(export_dir(project, args.id)) if args.id else []
        if not args.id:
            rows = listed(project)
            bad = [row for row in rows if not row["intact"]]
            if args.json:
                print(json.dumps({"exports": len(rows), "broken": bad}, ensure_ascii=False, indent=1))
                return 1 if bad else 0
            print(f"{len(rows)} export(s), {len(bad)} broken")
            for row in bad:
                print(f'  {row["id"]}: ' + "; ".join(row["problems"]))
            return 1 if bad else 0
        if problems:
            print("BROKEN: " + "; ".join(problems))
            return 1
        print(f"{args.id} is intact")
        return 0

    if args.command == "todos":
        target = args.id or (listed(project) or [{}])[0].get("id")
        if not target:
            raise ExportError("no export has been published, so there is nothing to train on")
        manifest = pj.read_json(export_dir(project, target) / "export_manifest.json")
        if not manifest:
            raise ExportError(f"no export {target} under {base_dir(project)}")
        if args.add:
            report = add_todos(project, manifest, families=args.families, enable=args.enable)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=1))
            return 0
        planned = plan_todos(project, manifest, families=args.families)
        if args.json:
            print(json.dumps(planned, ensure_ascii=False, indent=1))
            return 0
        print(f"export {target}: a training job for each of {len(planned)} family/families")
        for job in planned:
            state = "already in todo.json" if job["present"] else "would be added"
            print(f'  {job["id"]}  dataset_root={job["dataset_root"]}  ({state})')
        print("\nrun with --add to write them; they are added disabled unless --enable is given")
        return 0

    manifest = publish(project)
    if args.json:
        print(json.dumps({"id": manifest["id"], "images": manifest["counts"]["images"],
                          "refused": manifest["counts"]["refused"], "reused": manifest.get("reused", False)},
                         ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
