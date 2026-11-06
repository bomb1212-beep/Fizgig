"""Generator A: tool2's Qwen 2.1 edits, driven by the project's settings.

The project does not build Qwen edit graphs itself (spec decision D1): it hands tool2 a snapshot of its settings and of
the character's input folder, runs tool2's CLI, and imports what came back as candidates with their provenance.

Two things make a run reproducible rather than "whatever the settings are today":

* the snapshot pins the tool2 config, the tool version's graph version, and the exact bytes of every input image;
* the input images are copied into the snapshot, so editing or replacing an original later cannot silently change what
  a rerun of this job would do.

Inputs that tool2 generated itself (full-body anchors, refined anchors) are recorded as `origin: synthetic`, never
inferred from the folder they happen to sit in, because tool2 writes anchors straight into `characters/<name>/`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]          # automation
ROOT = AUTO.parent                                  # Fizgig_auto
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))

from pipeline import lineage, models  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# Names tool2's own make_candidates/refine_anchor write into the character folder. Used only as evidence, and recorded
# as such: the card's `synthetic_sources` list is the certain answer.
SYNTHETIC_PREFIXES = ("anchor_refined_", "anchor_", "retired_", "refined_", "sheet_")
DEFAULT_TOOL2 = {"python": "", "batch": ""}
# The character keys that name an input image tool2 needs. If a holdout is one of these, the run cannot show what it
# was asked to show, and that has to be said during the plan rather than discovered in the output.
REFERENCE_KEYS = ("edit_bases", "body_bases", "figure_refs", "profile_refs", "fullbody_source", "sheet_face",
                  "sheet_choice")


def holdout_from_card(card):
    """The holdout the card froze: {ids, files, shas, reason}. Empty when nothing was frozen."""
    holdout = ((card or {}).get("holdout") or {})
    return {"ids": list(holdout.get("ids") or []), "files": list(holdout.get("files") or []),
            "shas": list(holdout.get("shas") or []), "reason": holdout.get("reason", "")}


def named_references(entry):
    """The input file names a character entry points at, whichever of the reference keys it uses."""
    out = []
    for key in REFERENCE_KEYS:
        value = (entry or {}).get(key)
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(str(item) for item in value)
    return sorted(set(out))


def tool_paths(cfg=None):
    """Where tool2's interpreter and CLI are, from `pipeline.tool2` in automation/config.json."""
    block = dict(((cfg if cfg is not None else models.config()).get("pipeline") or {}).get("tool2") or {})
    paths = {**DEFAULT_TOOL2, **block}
    return {key: models.resolve_path(value) if value else "" for key, value in paths.items()}


def tool_graph_version(batch_path):
    """tool2's own graph version, read from its source.

    Reading one constant is what keeps a snapshot honest about which generator wrote its images; the module is plain
    standard library, so it can be loaded without tool2's interpreter. A failure is reported, never guessed.
    """
    try:
        spec = importlib.util.spec_from_file_location("tool2_dataset_batch_for_version", batch_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return str(getattr(module, "GRAPH_VERSION", "")) or "unknown"
    except Exception as exc:                        # noqa: BLE001 - a missing version must not stop a run
        return f"unknown ({type(exc).__name__})"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {} if default is None else default


def character_in(tool2_cfg, folder):
    for char in tool2_cfg.get("characters") or []:
        if char.get("folder") == folder:
            return char
    raise ValueError(f"tool2's config has no character with folder {folder!r}")


def source_origin(path, char_entry):
    """(origin, how it was decided) for one input image.

    The card's `synthetic_sources` list decides when it names the file; otherwise a name that tool2's own candidate
    generator uses is reported as a likely synthetic input, with the naming pattern as the stated reason.
    """
    name = Path(path).name
    listed = set(char_entry.get("synthetic_sources") or [])
    if name in listed:
        return "synthetic", "listed in synthetic_sources"
    if any(name.startswith(prefix) for prefix in SYNTHETIC_PREFIXES):
        return "synthetic", "name matches tool2's own candidate prefixes " + ", ".join(SYNTHETIC_PREFIXES)
    return "user", "assumed an original the user supplied"


def snapshot_id(tool2_cfg, char_folder, sources, graph_version):
    """A stable id for this exact generator, settings and input bytes."""
    payload = {"tool2": {k: v for k, v in tool2_cfg.items() if k != "characters"},
               "character": char_folder,
               "graph_version": graph_version,
               "sources": sorted(sources)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def build_snapshot(project_dir, card, *, exclude=(), holdout=None, tool2_config=None, paths=None, log=print):
    """Write this run's settings and input folder, and return where everything is.

    The holdout is kept out of the run twice over: by the file names the card froze, and by content hash, so a holdout
    that was renamed on its way into the character folder is still left behind. `exclude` adds names for one run;
    `holdout` overrides what the card says (a dict with `files` and `shas`).

    Both the copied folder and the snapshot's own settings record the exclusion, so which images a run could see is
    answerable afterwards from either one.
    """
    project_dir = Path(project_dir)
    char = card.get("tool2") or {}
    folder = char.get("character") or card.get("id")
    if not folder:
        raise ValueError("the card has no tool2.character and no id to fall back to")
    config_path = Path(tool2_config or char.get("config") or "")
    tool2_cfg = read_json(config_path)
    if not tool2_cfg:
        raise ValueError(f"tool2's config could not be read: {config_path}")
    entry = character_in(tool2_cfg, folder)
    paths = paths or tool_paths()
    batch = Path(paths.get("batch") or "")
    if not batch.is_file():
        raise ValueError(f"tool2's CLI is missing: {batch or '(not configured)'}")
    graph_version = tool_graph_version(batch)

    chars_root = Path(char.get("characters") or (config_path.parent / "characters" / folder))
    if not chars_root.is_dir():
        raise ValueError(f"the character's input folder does not exist: {chars_root}")
    holdout = {**holdout_from_card(card), **(holdout or {})}
    excluded_names = set(exclude) | set(holdout.get("files") or [])
    excluded_shas = set(holdout.get("shas") or [])

    kept, dropped = [], []
    for path in sorted(chars_root.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in excluded_names:
            dropped.append({"name": path.name, "reason": "holdout (named in the card's frozen holdout)"})
            continue
        try:
            digest = lineage.sha256_file(path)
        except OSError:
            dropped.append({"name": path.name, "reason": "unreadable"})
            continue
        if digest in excluded_shas:
            dropped.append({"name": path.name, "reason": "holdout (the same bytes as a holdout original)"})
            continue
        kept.append(path)
    if not kept:
        raise ValueError(f"no usable input images left in {chars_root} after excluding "
                         f"{sorted({row['name'] for row in dropped})}")
    missing_references = sorted(set(named_references(entry)) & {row["name"] for row in dropped})
    if missing_references:
        log(f"[gen_qwen_edit] warning: the character needs these excluded inputs as references: "
            f"{', '.join(missing_references)}; the run cannot show what it was asked for without them")

    sources = []
    for path in kept:
        origin, why = source_origin(path, entry)
        record = lineage.source_record(path, origin=origin, note=why)
        record["decided_by"] = why
        sources.append(record)

    ident = snapshot_id(tool2_cfg, folder, [r["sha256"] for r in sources], graph_version)
    base = project_dir / "tool2" / ident
    chars_dir = base / "chars" / folder
    out_dir = base / "out" / folder
    chars_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in kept:
        dest = chars_dir / path.name
        if not dest.exists():
            shutil.copy2(path, dest)               # bytes, not a link: the snapshot must not follow a later edit
    snapshot_cfg = dict(tool2_cfg)
    snapshot_cfg["characters"] = [{**entry, "excluded_inputs": sorted({row["name"] for row in dropped})}]
    (base / "config.json").write_text(json.dumps(snapshot_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "id": ident, "base": str(base), "config": str(base / "config.json"), "chars": str(chars_dir),
        "out": str(out_dir), "result_json": str(base / "result.json"), "character": folder,
        "graph_version": graph_version, "tool2_batch": str(batch), "tool2_python": paths.get("python") or sys.executable,
        "tool2_config": str(config_path), "excluded": sorted({row["name"] for row in dropped}),
        "excluded_detail": dropped, "missing_references": missing_references, "holdout": holdout,
        "sources": sources, "source_hashes": {r["src_path"]: r["sha256"] for r in sources},
    }
    (base / "snapshot.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[gen_qwen_edit] snapshot {ident}: {len(sources)} input(s) -> {base}")
    synthetic = sum(1 for r in sources if r["origin"] == "synthetic")
    if sources and synthetic / len(sources) > 0.7:
        log(f"[gen_qwen_edit] warning: {synthetic}/{len(sources)} inputs are synthetic")
    return result


def run(snapshot, *, only=None, limit=None, cancel_file=None, log=print, on_event=None, dry_run=False):
    """Run tool2's CLI against the snapshot and return its result JSON, or None when nothing was run."""
    command = [snapshot["tool2_python"], snapshot["tool2_batch"],
               "--config", snapshot["config"], "--chars", str(Path(snapshot["chars"]).parent),
               "--out", str(Path(snapshot["out"]).parent), "--character", snapshot["character"]]
    if only:
        command += ["--only", ",".join(only)]
    if limit:
        command += ["--limit", str(limit)]
    if dry_run:
        command += ["--dry-run"]
    else:
        command += ["--result-json", snapshot["result_json"]]
    if cancel_file:
        command += ["--cancel-file", str(cancel_file)]
    if not dry_run:
        command.append("--events")
    log(f"[gen_qwen_edit] {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace", cwd=str(Path(snapshot["tool2_batch"]).parent))
    # With --events every stdout line is one JSON event and the human log is on stderr. stderr is drained on its own
    # thread: tool2 logs a lot, and a full stderr pipe would stop the run dead while this loop waits on stdout.
    errors = []
    drain = threading.Thread(target=_drain, args=(process.stderr, errors), daemon=True)
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
                log(f"  {event.get('kind')} {event.get('stem', event.get('character', ''))}")
    finally:
        process.stdout.close()
        code = process.wait()
        drain.join(timeout=10)
    if code:
        raise RuntimeError(f"tool2 exited {code}: {''.join(errors).strip()[-2000:]}")
    if dry_run:
        return None
    return read_json(snapshot["result_json"])


def _drain(stream, sink):
    """Read a subprocess stream to the end, so it can neither block on a full pipe nor be left open."""
    try:
        for line in stream:
            sink.append(line)
    finally:
        stream.close()


def tool2_manifest(out_dir, chars_dir):
    """{file name: merged record} from tool2's manifest, so a candidate can point at the originals it was edited from."""
    path = Path(out_dir) / "manifest.jsonl"
    merged = {}
    if not path.is_file():
        return merged
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        name = record.get("file")
        if not name:
            continue
        entry = merged.setdefault(name, {"file": name})
        entry.update({k: v for k, v in record.items() if v not in (None, "", [], {}) or k not in entry})
    return merged


def parents_for(image_name, out_dir, chars_dir, snapshot, log=print):
    """Source ids of the images this candidate was edited from, or [] with the reason."""
    record = tool2_manifest(out_dir, chars_dir).get(image_name)
    if not record:
        return [], "tool2 left no manifest record for this image"
    names = list(record.get("refs") or []) or ([record["source"]] if record.get("source") else [])
    ids = []
    for name in names:
        try:
            ids.append(lineage.source_id(lineage.sha256_file(Path(chars_dir) / name)))
        except OSError:
            log(f"  parent not found in the snapshot: {name}")
    if not ids:
        return [], "none of the recorded parents exist in the snapshot"
    return ids, ""


def import_candidates(project_dir, card, snapshot, result, log=print):
    """Write the snapshot's provenance and the run's outputs into the project as candidates."""
    project_dir = Path(project_dir)
    chars_dir, out_dir = snapshot["chars"], snapshot["out"]
    lineage.write_jsonl(project_dir / "sources.jsonl", snapshot["sources"])
    records, skipped = [], []
    for entry in result.get("characters") or []:
        for row in entry.get("generated") or []:
            path = Path(row.get("file") or "")
            if not path.is_file():
                skipped.append({"file": str(path), "reason": "the file the result names is not on disk"})
                continue
            parents, why = parents_for(path.name, out_dir, chars_dir, snapshot, log)
            train_generation, edit_depth = 1, 1     # a tool2 edit works from the originals: generation 1, depth 1
            record = lineage.record(
                path, generator="qwen_edit", parent_ids=parents, train_generation=train_generation,
                edit_depth=edit_depth, slot={"category": row.get("category") or _category_of(row.get("stem", ""))},
                prompt=row.get("prompt", ""), seed=row.get("seed"), graph_version=f"tool2:{snapshot['graph_version']}",
                snapshot=snapshot["id"], stem=row.get("stem"),
                allow_missing_parents=True, orphan_reason=why)
            record["caption"] = _caption_of(path)
            records.append(record)
    written = lineage.append_jsonl(project_dir / "candidates.jsonl", records)
    log(f"[gen_qwen_edit] {written} new candidate(s) in {project_dir / 'candidates.jsonl'}")
    if skipped:
        log(f"[gen_qwen_edit] {len(skipped)} result row(s) named a file that is not on disk")
    return records, skipped


def _category_of(stem):
    return (stem or "").split("_")[0] or None


def _caption_of(path):
    sidecar = Path(path).with_suffix(".txt")
    try:
        return sidecar.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generator A: run tool2 for a project and import the candidates")
    parser.add_argument("--project", required=True, help="project folder (holds project.json or a card json)")
    parser.add_argument("--card", help="card file, default <project>/project.json")
    parser.add_argument("--only", help="comma list of tool2 categories")
    parser.add_argument("--limit", type=int, help="max jobs per category")
    parser.add_argument("--cancel-file", help="stop when this file appears")
    parser.add_argument("--exclude", help="comma list of input file names to keep out of the snapshot (holdouts)")
    parser.add_argument("--dry-run", action="store_true", help="plan with tool2 but do not generate")
    parser.add_argument("--import-only", metavar="RESULT_JSON", help="import an existing result without running")
    args = parser.parse_args(argv)

    project_dir = Path(args.project)
    card = read_json(args.card or project_dir / "project.json")
    if not card:
        raise SystemExit(f"no card found at {args.card or project_dir / 'project.json'}")
    if args.import_only:
        snapshot = read_json(project_dir / "tool2" / "snapshot.json")
        if not snapshot:
            raise SystemExit("import-only needs the snapshot.json a previous run wrote")
        import_candidates(project_dir, card, snapshot, read_json(args.import_only))
        return 0
    snapshot = build_snapshot(project_dir, card, exclude=(args.exclude or "").split(",") if args.exclude else ())
    result = run(snapshot, only=(args.only or "").split(",") if args.only else None, limit=args.limit,
                 cancel_file=args.cancel_file, dry_run=args.dry_run)
    if result is None:
        return 0
    import_candidates(project_dir, card, snapshot, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
