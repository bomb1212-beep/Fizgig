"""M3-4: corrections. The reviewer marks an image `fix`; this hands only those images to tool2's `--fix-jobs`.

Three rules shape the stage:

* **only what a person asked for** - the plan is exactly the candidates whose decision is `fix`. A page refresh, a
  plan run or a `--plan` invocation never starts GPU work, and nothing is corrected because a score looked low.
* **the corrected picture is a new candidate, not a replacement.** It points at the image it corrects (`parent_ids`),
  keeps that image's training generation, and is one edit deeper; the original stays in the project untouched.
* **the holdout stays out.** A correction copies the frame it is correcting, so the references it may add are the
  character's own originals minus the frozen holdout, by name and by content hash - the same rule the seed and the
  edit stage use. With the default `refs="none"` the tool sees one picture, which is also what keeps a correction
  from laying two copies of the character side by side.

The budget lives in tool2's own fix mode (steps, CFG, resolution, the consistency LoRA), so this module decides *what*
to correct and *what it is called*, and records the answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))

from pipeline import gen_qwen_edit as gen, lineage, models, project as pj  # noqa: E402

# What a correction asks for when the reviewer pressed F without writing anything. It says what to keep as clearly as
# what to change: a correction that redraws the character is not a correction, it is another candidate.
DEFAULT_FIX_PROMPT = ("Correct this picture without changing who or what it shows: keep the same character, face, hair, "
                      "pose, outfit, expression and background, and keep the Korean webtoon art style and its line art. "
                      "Draw the character once, in one single frame, not a split screen or a side-by-side comparison, "
                      "and fix drawing errors, extra or duplicated figures, broken hands and smeared line art.")


def corrections_of(project, target_id):
    """The candidates already imported as corrections of `target_id`, oldest first.

    A correction stays on the reviewer's `fix` decision - deciding the original is what asked for it - so without this
    lookup every later run of the stage would draw every correction that already exists a second time.
    """
    return [record for record in project.candidates() if record.get("fix_of") == target_id]


def chosen_engine(project):
    """(config path, engine, graph_version) - the tool2 run settings the project itself chose.

    The card names the config file; whether that config draws on Qwen or on P-H3 decides what a correction made with
    it must be recorded as (spec R4). A missing or unreadable config is a refusal, not a guess.
    """
    path = str(((project.card.get("tool2") or {}).get("config")) or "")
    if not path:
        raise pj.ProjectError("the card's tool2 block names no config, so the correction engine cannot be chosen")
    config = gen.read_json(path)
    if not config:
        raise pj.ProjectError(f"the card's tool2 config is not readable: {path}")
    engine = "ph3" if str(((config.get("models") or {}).get("backend")) or "") == "p_h3" else "qwen"
    graph = gen.tool_graph_version(gen.tool_paths()["batch"], config.get("models"))
    return {"config": path, "engine": engine, "graph_version": graph}


def plan(project, *, refs="none", redo=(), log=print):
    """The corrections to run: {batch, jobs, blocked, refs, prompts, already}.

    `blocked` is not a failure of this stage: it is the list of images the reviewer marked but that cannot be
    corrected as things stand, each with the reason, because a silent skip would look like the stage had done its job.
    An image that already has a correction is skipped the same way and listed under `already`, with the ids it was
    corrected into, unless its id is named in `redo` - a second correction of the same picture is a decision, not a
    side effect of running the stage again.
    """
    review = project.review()
    holdout = project.holdout_entry()
    holdout_ids = set(holdout["ids"])
    holdout_shas = set(holdout["shas"])
    redo = set(redo or ())
    jobs, blocked, already = [], [], []

    reference_files, reference_ids = [], []
    if refs == "originals":
        wanted = (project.card.get("tool2") or {}).get("fix_refs") or 2
        for source in project.source_records():
            if source["id"] in holdout_ids or source.get("origin") != "user":
                continue
            path = project.resolve_file(source.get("file") or source.get("src_path"))
            if path and path.is_file():
                reference_files.append(path)
                reference_ids.append(source["id"])
            if len(reference_files) >= int(wanted):
                break

    for record in project.candidates():
        state = review.get(record["id"]) or {}
        if state.get("decision") != "fix":
            continue
        if state.get("reason") in pj.FIX_BLOCKED_REASONS:
            blocked.append({"id": record["id"], "reason": "it was judged another character, so a correction would "
                                                           "only draw that other character more carefully"})
            continue
        path = project.resolve_file(record.get("file"))
        if not path or not path.is_file():
            blocked.append({"id": record["id"], "reason": "the image is not on disk"})
            continue
        if record.get("sha256") in holdout_shas:
            blocked.append({"id": record["id"], "reason": "it is a holdout original"})
            continue
        note = str(state.get("note") or "").strip()
        existing = corrections_of(project, record["id"]) if record["id"] not in redo else []
        if existing:
            already.append({"id": record["id"], "corrected_as": [row["id"] for row in existing]})
            continue
        jobs.append({
            "id": record["id"],
            "input": str(path),
            "refs": [str(p) for p in reference_files],
            "ref_ids": list(reference_ids),
            "prompt": note or DEFAULT_FIX_PROMPT,
            "prompt_source": "the reviewer's note" if note else "the default correction prompt",
            "slot": (record.get("slot") or {}).get("slot_id"),
            "category": (record.get("slot") or {}).get("category"),
            "generation": int(record.get("train_generation") or 0),
            "edit_depth": int(record.get("edit_depth") or 0),
            "out": "",
        })
    batch = batch_id(jobs, refs)
    folder = project.dir / "fix" / batch
    for job in jobs:
        job["out"] = str(folder / f"{job['id']}.png")
    log(f"[fix] {len(jobs)} correction(s) planned" + (f", {len(blocked)} blocked" if blocked else "")
        + (f", {len(already)} already corrected" if already else "")
        + (f"; references: {len(reference_ids)}" if reference_ids else "; no references"))
    for job in jobs:
        log(f"  {job['id']}  slot={job['slot'] or '-'}  {job['prompt_source']}")
    for row in blocked:
        log(f"  blocked {row['id']}: {row['reason']}")
    for row in already:
        log(f"  already corrected {row['id']}: {', '.join(row['corrected_as'])}")
    chosen = chosen_engine(project)
    log(f"[fix] engine {chosen['engine']} ({chosen['config']}, graph {chosen['graph_version']})")
    return {"batch": batch, "folder": str(folder), "jobs": jobs, "blocked": blocked, "already": already, "refs": refs,
            "reference_ids": reference_ids, "default_prompt": DEFAULT_FIX_PROMPT, **chosen}


def batch_id(jobs, refs):
    """A name for this set of corrections: the same set asked twice is the same folder, so nothing is drawn twice."""
    payload = {"refs": refs, "jobs": sorted((job["id"], job["prompt"], tuple(job["refs"])) for job in jobs)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def run(plan_data, *, python=None, log=print, on_event=None, cancel_file=None, timeout=7200):
    """Run the plan through tool2's fix mode and return its result rows (one per job)."""
    if not plan_data["jobs"]:
        return []
    folder = Path(plan_data["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    paths = gen.tool_paths()
    command = [python or paths.get("python") or sys.executable, str(paths["batch"])]
    jobs_path = folder / "jobs.json"
    result_path = folder / "result.json"
    payload = []
    for job in plan_data["jobs"]:
        row = {"input": job["input"], "refs": list(job["refs"]), "prompt": job["prompt"], "out": job["out"]}
        if job.get("seed"):
            row["seed"] = int(job["seed"])
        payload.append(row)
    jobs_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    command += ["--fix-jobs", str(jobs_path), "--result-json", str(result_path), "--events"]
    if plan_data.get("config"):
        # The correction runs on the settings the project itself chose, not on whatever tool2 defaults to today
        # (spec R4): the engine a record names has to be the engine that actually drew the picture.
        command += ["--config", str(plan_data["config"])]
    if cancel_file:
        command += ["--cancel-file", str(cancel_file)]
    log(f"[fix] {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace", cwd=str(Path(paths["batch"]).parent))
    errors = []
    drain = threading.Thread(target=gen._drain, args=(process.stderr, errors), daemon=True)
    drain.start()
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                log(line)
                continue
            if on_event is not None:
                on_event(event)
            else:
                log(f"  {event.get('kind')} {event.get('status', '')} {event.get('out', '')}")
    finally:
        process.stdout.close()
        code = process.wait()
        drain.join(timeout=10)
    if code:
        raise RuntimeError(f"tool2 exited {code}: {''.join(errors).strip()[-2000:]}")
    return gen.read_json(result_path).get("jobs") or []


def sidecar_verified(out, job, *, graph_version):
    """Does the correction's own sidecar vouch for these bytes being this job's output?

    An interruption between saving the picture and importing it leaves no result row; the re-run reports the picture
    as `skipped - already there, from the same inputs and recipe`. That is recoverable only when the sidecar the fix
    mode wrote still matches the job's input bytes and the graph that drew it - a sidecar naming other bytes, or no
    sidecar at all, is not evidence (spec R4).
    """
    sidecar = out.with_name(out.name + ".fix.json")
    data = gen.read_json(sidecar) if sidecar.is_file() else None
    if not data:
        return False
    source = Path(job["input"])
    if not source.is_file():
        return False
    try:
        if data.get("input") != lineage.quick_hash(source):
            return False
    except OSError:
        return False
    return not graph_version or str(data.get("graph_version") or "") == str(graph_version)


def import_corrections(project, plan_data, results, *, log=print):
    """Record what was corrected as new candidates: the image it corrects, one edit deeper, same generation.

    A row tool2 reported `done` is imported; so is a `skipped - already there` row whose sidecar still verifies, which
    is how an import that was interrupted recovers its pictures without drawing them twice (spec R4). Anything else -
    failed, no row, an unverifiable sidecar - is refused with its reason. The engine is the plan's: the generator and
    family a record carries are the ones that actually drew the picture. The corrected picture is copied into the
    project's own candidates folder by `Project.import_candidates`, so the correction keeps its provenance even if the
    fix folder is cleaned up later.
    """
    engine = plan_data.get("engine") or "qwen"
    generator = gen.ENGINE_GENERATOR[engine]
    family = gen.ENGINE_FAMILY[engine]
    graph_version = plan_data.get("graph_version") or ""
    rows = {Path(row.get("out") or "").name: row for row in results or []}
    records, skipped = [], []
    for job in plan_data["jobs"]:
        out = Path(job["out"])
        row = rows.get(out.name) or {}
        status = row.get("status")
        if status == "done" and out.is_file():
            pass
        elif status == "skipped" and out.is_file() and sidecar_verified(out, job, graph_version=graph_version):
            row = {**row, "graph_version": row.get("graph_version") or graph_version}
        else:
            reason = row.get("error") or row.get("status") or "no result row"
            if status == "skipped" and out.is_file():
                reason = "the output exists but its sidecar does not verify it as this correction"
            skipped.append({"id": job["id"], "reason": reason})
            continue
        # The row's `resolution` is a budget of resolution**2 pixels, not a pixel count: the graph takes the shape
        # from the first input, so 1541 came back as 1216x1952. The record keeps what the file actually is.
        width = height = None
        try:
            width, height = gen.image_size(out)
        except (OSError, ValueError):
            pass
        record = lineage.record(
            out, generator=generator, parent_ids=[job["id"]] + list(job.get("ref_ids") or []),
            train_generation=int(job.get("generation") or 0), edit_depth=int(job.get("edit_depth") or 0) + 1,
            slot={"slot_id": job.get("slot"), "category": job.get("category")},
            prompt=job["prompt"], seed=row.get("seed"), steps=row.get("steps"), cfg=row.get("cfg"),
            width=width, height=height,
            graph_version=f"tool2:{row.get('graph_version') or graph_version}".rstrip(":"),
            fix_of=job["id"], fix_batch=plan_data["batch"], fix_prompt_source=job["prompt_source"],
            models={"family": family},
            result_row_status=status,
            resolution_budget=row.get("resolution"))
        caption = gen._caption_of(out)
        source = "the correction's own caption"
        if not caption:
            # A correction keeps what the picture shows, so it keeps that picture's caption: tool2's fix mode writes
            # no caption of its own, and an empty one would make the image untrainable for no reason.
            parent = project.candidate_by_id(job["id"]) or {}
            caption = str(parent.get("caption") or "").strip()
            source = "the caption of the image it corrects" if caption else ""
        record["caption"] = caption
        if source:
            record["caption_source"] = source
        records.append(record)
    if records:
        project.import_candidates(records, log=log)
    log(f"[fix] {len(records)} correction(s) imported" + (f", {len(skipped)} not" if skipped else ""))
    for row in skipped:
        log(f"  not imported {row['id']}: {row['reason']}")
    return {"records": records, "skipped": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Correct the images a reviewer marked fix (spec 4 M3-4)")
    parser.add_argument("--project", required=True, help="project folder")
    parser.add_argument("--refs", choices=("none", "originals"), default="none",
                        help="references a correction may use: the character's originals (holdout excluded) or none")
    parser.add_argument("--redo", action="append", default=[], metavar="ID",
                        help="correct this image again even though a correction of it already exists (repeatable)")
    parser.add_argument("--plan", action="store_true", help="show the corrections without running anything")
    parser.add_argument("--run", action="store_true", help="run the corrections and import the results")
    parser.add_argument("--cancel-file", help="stop when this file appears")
    args = parser.parse_args(argv)

    project = pj.Project.open(args.project)
    plan_data = plan(project, refs=args.refs, redo=args.redo)
    print(json.dumps({"batch": plan_data["batch"], "jobs": [
        {"id": job["id"], "slot": job["slot"], "prompt_source": job["prompt_source"], "refs": len(job["refs"])}
        for job in plan_data["jobs"]], "blocked": plan_data["blocked"],
        "already": plan_data["already"]}, ensure_ascii=False, indent=1))
    if args.plan or not args.run:
        return 0
    results = run(plan_data, cancel_file=args.cancel_file)
    outcome = import_corrections(project, plan_data, results)
    print(json.dumps({"imported": len(outcome["records"]), "skipped": outcome["skipped"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
