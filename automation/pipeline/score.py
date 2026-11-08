"""Run the scorers over a project and write one file the review page and the reports can both read.

The scorers are measurements. This module is the step that takes them over the whole project at once - every candidate
and every original - and stores the result where the review page can sort by it and `evaluate` can compare it with a
person's marks.

Two design points worth stating, because they were decided by the data rather than by preference:

* **Both scorers run, and they disagree.** On this project's M0 sheet, CLIP's whole-frame cosine tracked the human
  identity marks at +0.04 and ArcFace's face cosine at +0.15 - and the two correlate with each other at only +0.34.
  They are measuring different things, so the file carries both and never averages them into one "identity score".
* **No score rejects anything.** Nothing here writes a decision. The review page sorts by these numbers; the reviewer
  decides. A threshold applied automatically would be a rule nobody measured the cost of.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import lineage
from . import project as pj
from .scorers import arcface, clip_vision, common


class ScoreError(RuntimeError):
    """A scoring pass that cannot be run as asked."""


def scores_path(project):
    return project.dir / "experiments" / "scores.json"


def reference_paths(project, *, purpose="review"):
    """The originals a score is measured against (spec N4.5).

    `purpose="review"` is the general case - candidate scoring, the crop re-score, the epoch grid's CLIP sort - and it
    **excludes the frozen holdout originals**. The holdout exists to measure what the model never saw; scoring
    candidates against those two pictures as well would make the holdout a reference for selection, which is the
    quiet version of training on it. `purpose="holdout"` returns exactly those two and nothing else, and is the only
    caller that may.

    The old version returned all fifteen user originals for every purpose, so the holdout was in the reference set of
    the ordinary candidate scores (`review_references`: 15 references, 2 of them the holdout).
    """
    holdout_ids = set((project.holdout_entry() or {}).get("ids") or [])
    holdout_files = set()
    out = []
    for record in project.source_records():
        if record.get("origin") != "user":
            continue
        record_id = record.get("id")
        path = project.resolve_file(record.get("file") or record.get("src_path"))
        if not path or not Path(path).is_file():
            continue
        is_holdout = record_id in holdout_ids
        if purpose == "holdout" and not is_holdout:
            continue
        if purpose != "holdout" and is_holdout:
            # Also remembered so the same picture registered twice (a copy with another id) cannot slip back in.
            holdout_files.add(Path(path).resolve())
            continue
        out.append(str(path))
    if purpose != "holdout" and holdout_files:
        out = [path for path in out if Path(path).resolve() not in holdout_files]
    return sorted(out)


def targets(project):
    """Every candidate that has bytes on disk, as `(id, path)`, in a stable order."""
    out = []
    for record in project.candidates():
        path = project.resolve_file(record.get("file"))
        if path and Path(path).is_file():
            slot = record.get("slot") or {}
            out.append({"id": record["id"], "path": str(path),
                        "sha256": record.get("sha256"),
                        # The plan's vocabulary is `slot_id`; the older `id` key is honoured so a record written
                        # before the rename still lands in its slot (spec R7).
                        "slot": slot.get("slot_id") or slot.get("id") or "",
                        "method": record.get("method") or ""})
    out.sort(key=lambda row: row["id"])
    return out


def run(project, *, scorers=("clip_vision", "arcface"), log=print, embedder=None, face_embedder=None):
    """Score every candidate against the originals and write `experiments/scores.json`.

    A scorer that cannot run is recorded as unavailable with its reason rather than stopping the pass: ArcFace cannot
    score a drawing it finds no face in, and that is a fact about the picture, not a failure of the run.
    """
    refs = reference_paths(project)
    if not refs:
        raise ScoreError("this project has no user originals on disk, so there is nothing to score against")
    rows_targets = targets(project)
    if not rows_targets:
        raise ScoreError("this project has no candidates on disk to score")

    ids = [row["id"] for row in rows_targets]
    paths = [row["path"] for row in rows_targets]
    results, unavailable = {}, {}
    log(f"[score] {len(paths)} candidate(s) against {len(refs)} original(s)")

    if "clip_vision" in scorers:
        try:
            results["clip_vision"] = clip_vision.identity_scores(
                paths, refs, ids=ids, embedder=embedder or clip_vision.Embedder())
            log(f'[score] clip_vision: {results["clip_vision"]["coverage"]["scored"]} scored')
        except Exception as exc:  # noqa: BLE001 - one scorer failing must not lose the others
            unavailable["clip_vision"] = f"{type(exc).__name__}: {exc}"
            log(f"[score] clip_vision unavailable: {unavailable['clip_vision']}")

    if "arcface" in scorers:
        try:
            results["arcface"] = arcface.face_scores(paths, refs, ids=ids, embedder=face_embedder)
            scored = results["arcface"]["coverage"]["scored"]
            log(f"[score] arcface: {scored} scored, {len(results['arcface']['coverage']['no_face'])} without a face")
        except Exception as exc:  # noqa: BLE001
            unavailable["arcface"] = f"{type(exc).__name__}: {exc}"
            log(f"[score] arcface unavailable: {unavailable['arcface']}")

    # The measures that need no model, over the same pictures.
    white = common.white_scores([{"id": r["id"], "path": r["path"]} for r in rows_targets])
    hashes = common.sha_groups([{"id": r["id"], "sha256": r["sha256"]} for r in rows_targets])
    near = common.phash_groups([{"id": r["id"], "path": r["path"]} for r in rows_targets])
    # `Project.quotas` already knows how an accepted candidate maps to a planned slot, including the category-prefix
    # fallback for records that predate slot ids, so the slot fill is asked of it rather than recomputed here.
    planned = {slot: int(target) for slot, target in (project.card.get("slot_quotas") or {}).items()}
    fill_rows = []
    for slot, block in (project.quotas(planned) or {}).items():
        fill_rows.append({"slot": slot, "target": block["target"], "accepted": block["accepted"],
                          "short": block["short"], "over": max(0, block["accepted"] - block["target"])})
    fill_rows.sort(key=lambda row: row["slot"])

    payload = {
        "kind": "scores", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project": project.card.get("id"), "card_digest": project.digest(),
        "references": refs,
        "targets": len(rows_targets),
        "scorers": {name: {"family": res["family"], "scorer": res["scorer"],
                           "reference": res["reference"], "notes": res["notes"],
                           "coverage": res["coverage"], "rows": res["rows"]}
                    for name, res in results.items()},
        "unavailable": unavailable,
        "white_ratio": {"rows": white["rows"], "coverage": white["coverage"]},
        "exact_duplicates": hashes["rows"],
        "near_duplicates": near["rows"],
        "slot_fill": fill_rows,
        # One flat row per picture, which is what the review page sorts on. Each field keeps the scorer's name so a
        # reader can tell an ArcFace number from a CLIP one without looking anything up.
        "rows": [
            {"id": row["id"], "slot": row["slot"], "method": row["method"],
             "clip_max_cosine": _pick(results.get("clip_vision"), row["id"], "max_cosine"),
             "clip_mean_cosine": _pick(results.get("clip_vision"), row["id"], "mean_cosine"),
             "arcface_max_cosine": _pick(results.get("arcface"), row["id"], "max_cosine"),
             "arcface_no_face": _pick(results.get("arcface"), row["id"], "no_face"),
             "white_ratio": _pick({"rows": white["rows"]}, row["id"], "white_ratio")}
            for row in rows_targets
        ],
    }
    path = scores_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(f".tmp-{time.time_ns()}")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    staging.replace(path)
    log(f"[score] wrote {path}")
    return payload


def _pick(result, image_id, field):
    """One field of one image's row, or None. A missing row is None, never zero."""
    if not result:
        return None
    for row in result["rows"]:
        if row.get("id") == image_id:
            return row.get(field)
    return None


def read(project):
    """The stored scores, or None when the project has not been scored."""
    data = pj.read_json(scores_path(project))
    return data or None


def table(project):
    """{id: row} from the stored scores, for the review page."""
    data = read(project)
    if not data:
        return {}
    return {row["id"]: row for row in data.get("rows") or []}


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="pipeline.score", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "show"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--only", nargs="*", choices=("clip_vision", "arcface"),
                        help="run only these scorers")
    args = parser.parse_args(argv)
    project = pj.Project.open(args.project)

    if args.command == "show":
        data = read(project)
        if not data:
            print("this project has not been scored yet")
            return 0
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=1))
            return 0
        print(f'scored {data["targets"]} picture(s) at {data["created_at"]} against {len(data["references"])} original(s)')
        for name, block in (data.get("scorers") or {}).items():
            values = [r["max_cosine"] for r in block["rows"] if r.get("max_cosine") is not None]
            if values:
                print(f'  {name}: {len(values)} scored, max {max(values):.4f}, mean {sum(values)/len(values):.4f}')
            else:
                print(f"  {name}: no picture could be scored")
        for name, reason in (data.get("unavailable") or {}).items():
            print(f"  {name}: unavailable - {reason}")
        print(f'  exact duplicate groups: {len(data.get("exact_duplicates") or [])}')
        print(f'  near duplicate groups: {len(data.get("near_duplicates") or [])}')
        short = [row for row in (data.get("slot_fill") or []) if row.get("short")]
        print(f'  slots short: {len(short)}')
        return 0

    # The scorers load CLIP/ArcFace weights onto the same GPU training and generation use, so the CLI takes the
    # common GPU claim for the whole scoring pass (spec N1).
    from . import gpu
    claim = gpu.claim_gpu(what="the scoring pass")
    claim.acquire()
    try:
        gpu.vram_gate("scoring", "score")
        run(project, scorers=tuple(args.only or ("clip_vision", "arcface")))
    finally:
        claim.release()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
