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

from pipeline import lineage, models, project as pj  # noqa: E402

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


def scrub_references(entry, dropped_names):
    """The character entry with the excluded files taken out of every reference it names.

    Removing a holdout from the copied folder is not enough: tool2's entry names its edit bases, figure references and
    sheet face by file name, and `pick()` refuses to plan when a named file is absent. So the snapshot's own copy of
    the entry drops those names too - the real `dataset_jobs.json` is never edited - and a key that loses all of its
    names is removed so tool2 falls back to its documented default instead of being handed an empty list.
    """
    scrubbed = dict(entry)
    removed = []
    for key in REFERENCE_KEYS:
        value = scrubbed.get(key)
        if isinstance(value, str):
            if value in dropped_names:
                scrubbed.pop(key)
                removed.append({"key": key, "removed": [value]})
            continue
        if not isinstance(value, (list, tuple)):
            continue
        gone = [str(item) for item in value if item in dropped_names]
        if not gone:
            continue
        kept = [item for item in value if item not in dropped_names]
        removed.append({"key": key, "removed": gone})
        if kept:
            scrubbed[key] = kept
        else:
            scrubbed.pop(key)
    return scrubbed, removed


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


def snapshot_id(tool2_cfg, char_folder, sources, graph_version, entry=None):
    """A stable id for this exact generator, settings, character entry and input bytes.

    The character entry belongs in the id: it names the edit bases and the references, so two runs that differ only
    there produce different pictures and must not share a snapshot folder.
    """
    payload = {"tool2": {k: v for k, v in tool2_cfg.items() if k != "characters"},
               "character": char_folder,
               "graph_version": graph_version,
               "sources": sorted(sources)}
    if entry is not None:
        payload["entry"] = {k: v for k, v in entry.items() if k != "excluded_inputs"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


SECTION_OVERRIDES = ("one_image", "use_extra_refs", "steps", "cfg", "resolution")


def apply_section_overrides(tool2_cfg, overrides):
    """(config, record) with per-section settings merged into the run's own copy of tool2's config.

    A per-run setting belongs to the snapshot: editing `dataset_jobs.json` would change the tool for every character
    and every later run, and the spec keeps that file behind the user. The merged keys are listed in the snapshot, so a
    run still says exactly what it was asked to do.
    """
    merged = json.loads(json.dumps(tool2_cfg))          # a deep copy: nested sections are edited, not replaced
    applied = []
    for section, settings in (overrides or {}).items():
        if section not in merged or not isinstance(merged[section], dict):
            raise ValueError(f"tool2's config has no section {section!r} to override")
        for key, value in (settings or {}).items():
            merged[section][key] = value
            applied.append({"section": section, "key": key, "value": value})
    return merged, applied


def parse_overrides(values):
    """`--set expressions.use_extra_refs=false` -> {"expressions": {"use_extra_refs": False}}.

    A value written as JSON is parsed as JSON, which is how a list setting (which shots, which items) can be chosen
    for one run without editing the tool's config.
    """
    out = {}
    for item in values or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--set wants section.key=value, not {item!r}")
        path, raw = item.split("=", 1)
        section, key = path.rsplit(".", 1)
        text = raw.strip()
        if text.lower() in ("true", "false"):
            value = text.lower() == "true"
        elif text[:1] in "[{":
            try:
                value = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"--set {path}: the value looks like JSON but does not parse ({exc})") from exc
        else:
            try:
                value = int(text)
            except ValueError:
                try:
                    value = float(text)
                except ValueError:
                    value = raw
        out.setdefault(section, {})[key] = value
    return out


def parse_character_overrides(values):
    """`--set-char edit_bases=a.png,b.png` -> {"edit_bases": ["a.png", "b.png"]}."""
    out = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--set-char wants key=value, not {item!r}")
        key, raw = item.split("=", 1)
        text = raw.strip()
        out[key.strip()] = [part.strip() for part in text.split(",") if part.strip()] if "," in text else text
    return out


def build_snapshot(project_dir, card, *, exclude=(), holdout=None, sections=None, character=None, tool2_config=None,
                   paths=None, log=print):
    """Write this run's settings and input folder, and return where everything is.

    The holdout is kept out of the run twice over: by the file names the card froze, and by content hash, so a holdout
    that was renamed on its way into the character folder is still left behind. `exclude` adds names for one run;
    `holdout` overrides what the card says (a dict with `files` and `shas`).

    Both the copied folder and the snapshot's own settings record the exclusion, so which images a run could see is
    answerable afterwards from either one.
    """
    project_dir = Path(project_dir).resolve()
    char = card.get("tool2") or {}
    folder = char.get("character") or card.get("id")
    if not folder:
        raise ValueError("the card has no tool2.character and no id to fall back to")
    # Every path that travels to tool2 is absolute: tool2 runs with its own folder as the working directory, so a
    # relative `--project` would hand it a config path it cannot find.
    config_path = Path(tool2_config or char.get("config") or "").resolve()
    tool2_cfg = read_json(config_path)
    if not tool2_cfg:
        raise ValueError(f"tool2's config could not be read: {config_path}")
    tool2_cfg, applied_overrides = apply_section_overrides(tool2_cfg, sections)
    for row in applied_overrides:
        log(f"[gen_qwen_edit] {row['section']}.{row['key']} = {row['value']} (this run only)")
    entry = character_in(tool2_cfg, folder)
    paths = paths or tool_paths()
    batch = Path(paths.get("batch") or "").resolve()
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
    snapshot_entry, dropped_references = scrub_references(entry, {row["name"] for row in dropped})
    for row in dropped_references:
        log(f"[gen_qwen_edit] {row['key']}: dropped excluded reference(s) {', '.join(row['removed'])}")
    applied_character = {}
    for key, value in (character or {}).items():
        # The character entry decides which picture each job edits, so a per-run change to it is part of the run.
        snapshot_entry[key] = value
        applied_character[key] = value
        log(f"[gen_qwen_edit] character.{key} = {value} (this run only)")

    sources = []
    for path in kept:
        origin, why = source_origin(path, entry)
        record = lineage.source_record(path, origin=origin, note=why)
        record["decided_by"] = why
        sources.append(record)

    ident = snapshot_id(tool2_cfg, folder, [r["sha256"] for r in sources], graph_version, entry=snapshot_entry)
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
    snapshot_cfg["characters"] = [{**snapshot_entry, "excluded_inputs": sorted({row["name"] for row in dropped})}]
    (base / "config.json").write_text(json.dumps(snapshot_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "id": ident, "base": str(base), "config": str(base / "config.json"), "chars": str(chars_dir),
        "out": str(out_dir), "result_json": str(base / "result.json"), "character": folder,
        "graph_version": graph_version, "tool2_batch": str(batch), "tool2_python": paths.get("python") or sys.executable,
        "tool2_config": str(config_path), "excluded": sorted({row["name"] for row in dropped}),
        "excluded_detail": dropped, "missing_references": missing_references,
        "dropped_references": dropped_references, "holdout": holdout,
        "section_overrides": applied_overrides, "character_overrides": applied_character,
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
    """(parent source ids, the reference file names, the reason when there are no parents)."""
    record = tool2_manifest(out_dir, chars_dir).get(image_name)
    if not record:
        return [], [], "tool2 left no manifest record for this image"
    names = list(record.get("refs") or []) or ([record["source"]] if record.get("source") else [])
    ids = []
    for name in names:
        try:
            ids.append(lineage.source_id(lineage.sha256_file(Path(chars_dir) / name)))
        except OSError:
            log(f"  parent not found in the snapshot: {name}")
    if not ids:
        return [], names, "none of the recorded parents exist in the snapshot"
    return ids, names, ""


def open_project(project_dir):
    """The project behind a folder, or None when the folder is just a working directory (the offline tests)."""
    folder = Path(project_dir)
    return pj.Project.open(folder) if (folder / "project.json").is_file() else None


def merge_sources(project_dir, sources, *, snapshot_id="", log=print):
    """Add this run's sources to the project without touching the ones it already has.

    The project's `sources.jsonl` is the record of the user's own originals, including the holdout reasoning; a
    generator run must not be able to replace it. Rows are matched by id (the content hash), so a source the project
    already knows keeps its own row - with the original path the user's file lives at - and only new bytes are added.
    """
    path = Path(project_dir) / "sources.jsonl"
    known = {r.get("id") for r in lineage.read_jsonl(path)}
    fresh = []
    for record in sources:
        if record.get("id") in known:
            continue
        fresh.append({**record, "snapshot": snapshot_id or record.get("snapshot", "")})
        known.add(record.get("id"))
    written = lineage.append_jsonl(path, fresh)
    report = {"added": written, "reused": len(sources) - written, "total": len(known)}
    if written:
        log(f"[gen_qwen_edit] sources: {written} added, {len(sources) - written} already in the project")
    return report


def model_hashes(snapshot):
    """Quick hashes of the models this run used: once per run, not once per image.

    A DiT is tens of gigabytes, so `lineage.quick_hash` (size, mtime, both ends) is the honest cheap identity; a LoRA
    gets a full sha256. A model that is not on disk is recorded as missing rather than left out silently.
    """
    config = read_json(snapshot.get("config") or "")
    comfy_dir = config.get("comfy_dir") or ""
    out = {}
    for name, value in (config.get("models") or {}).items():
        path = Path(str(value))
        if not path.is_absolute():
            path = (Path(comfy_dir) / "models" / str(value)) if comfy_dir else path
        entry = {"path": str(path)}
        if not path.is_file():
            entry["missing"] = True
            out[name] = entry
            continue
        try:
            entry["quick_hash"] = lineage.quick_hash(path)
        except OSError as exc:
            entry["error"] = str(exc)
        out[name] = entry
    return out


def find_snapshot(project_dir, card=None, *, snapshot_id=None, exclude=(), sections=None, paths=None, log=print):
    """The snapshot an existing result belongs to: its own folder, or the one the card produces today.

    Importing a run that finished after the process died needs the settings and input list of *that* run. The id is a
    hash of exactly those, so picking the newest snapshot folder with the same character and then, if that fails,
    rebuilding the snapshot from the card (which submits no GPU work) recovers it.
    """
    project_dir = Path(project_dir)
    folder = project_dir / "tool2"
    candidates = []
    if folder.is_dir():
        for child in sorted(folder.iterdir()):
            data = read_json(child / "snapshot.json")
            if data.get("id"):
                candidates.append(data)
    if snapshot_id:
        for data in candidates:
            if data["id"] == snapshot_id:
                return data
    character = ((card or {}).get("tool2") or {}).get("character") or (card or {}).get("id")
    same_character = [data for data in candidates if data.get("character") == character]
    if same_character:
        newest = max(same_character, key=lambda data: Path(data.get("base") or "").stat().st_mtime
                     if Path(data.get("base") or "").exists() else 0)
        log(f"[gen_qwen_edit] import-only: using snapshot {newest['id']}")
        return newest
    if not card:
        raise ValueError(f"no snapshot under {folder} and no card to rebuild one from")
    rebuilt = build_snapshot(project_dir, card, exclude=exclude, sections=sections, paths=paths, log=log)
    log(f"[gen_qwen_edit] import-only: rebuilt snapshot {rebuilt['id']} from the card (no GPU work was submitted)")
    return rebuilt


def import_candidates(project_dir, card, snapshot, result, *, log=print):
    """Write the snapshot's provenance and the run's outputs into the project as candidates.

    Three things this refuses to do: replace the project's own source list (it merges), drop an image the project
    already has (records are keyed by content hash), and record a run without saying which snapshot, graph and models
    produced it.
    """
    project_dir = Path(project_dir)
    project = open_project(project_dir)
    chars_dir, out_dir = snapshot["chars"], snapshot["out"]
    sources_report = merge_sources(project_dir, snapshot["sources"], snapshot_id=snapshot["id"], log=log)
    hashes = model_hashes(snapshot)
    manifest_records = tool2_manifest(out_dir, chars_dir)
    records, skipped = [], []
    for entry in result.get("characters") or []:
        for row in entry.get("generated") or []:
            path = Path(row.get("file") or "")
            if not path.is_file():
                skipped.append({"file": str(path), "reason": "the file the result names is not on disk"})
                continue
            parents, refs, why = parents_for(path.name, out_dir, chars_dir, snapshot, log)
            record = lineage.record(
                path, generator="qwen_edit", parent_ids=parents,
                train_generation=1, edit_depth=1,          # a tool2 edit works from the originals
                slot={"category": row.get("category") or _category_of(row.get("stem", ""))},
                prompt=row.get("prompt", ""), seed=row.get("seed"),
                steps=row.get("steps"), cfg=row.get("cfg"),
                width=row.get("resolution") or row.get("width"), height=row.get("resolution") or row.get("height"),
                graph_version=f"tool2:{snapshot['graph_version']}", snapshot=snapshot["id"],
                stem=row.get("stem"), prompt_id=f"{snapshot['id']}:{row.get('stem') or path.stem}",
                refs=refs, models={"family": "qwen_image21", "files": hashes,
                                   "record": manifest_records.get(path.name) or {}},
                allow_missing_parents=True, orphan_reason=why)
            record["caption"] = _caption_of(path)
            records.append(record)
    if project is not None:
        written, project_skipped = project.import_candidates(records, log=log)
        for item in project_skipped:
            skipped.append({"file": item["id"], "reason": item["reason"]})
        records = [r for r in records if r["id"] not in {s.get("id") for s in project_skipped}]
        if records:
            # The run's slot vocabulary is the project's: a new image is counted by the quota board the moment it lands.
            assign_slots(project, [record["id"] for record in records], log=log)
    else:
        written = lineage.append_jsonl(project_dir / "candidates.jsonl", records)
        log(f"[gen_qwen_edit] {written} new candidate(s) in {project_dir / 'candidates.jsonl'}")
    if skipped:
        log(f"[gen_qwen_edit] {len(skipped)} result row(s) were not imported")
    report = {"candidates": written, "skipped": skipped, "sources": sources_report,
              "snapshot": snapshot["id"], "records": records}
    return records, skipped, report


def _category_of(stem):
    return (stem or "").split("_")[0] or None


def _caption_of(path):
    sidecar = Path(path).with_suffix(".txt")
    try:
        return sidecar.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def restore_slots(project, exp_dir, *, apply=True, log=print):
    """Give project candidates the slot id an experiment manifest recorded for the same bytes.

    The M0 harness knew which slot every image was made for, but it only wrote the slot's *category* into
    `candidates.jsonl`, so the project's quota board counted nothing. The manifest still has the answer keyed by the
    image file it produced; matching that file to a candidate by content hash (not by name) recovers the slot, and an
    image whose bytes match more than one slot is left unresolved instead of guessed.
    """
    exp_dir = Path(exp_dir)
    manifest = read_json(exp_dir / "experiment.json")
    if not manifest:
        raise ValueError(f"no experiment.json in {exp_dir}")
    by_file = {task.get("file"): task for task in manifest.get("tasks") or [] if task.get("file")}
    sha_to_slots = {}
    for record in lineage.read_jsonl(exp_dir / "candidates.jsonl"):
        sha = record.get("sha256")
        name = Path(str(record.get("file") or "")).name
        task = by_file.get(record.get("file")) or by_file.get(f"images/{record.get('method')}/{name}")
        if not sha or not task:
            continue
        sha_to_slots.setdefault(sha, set()).add(task.get("slot_id"))
    restored, ambiguous, unresolved = [], [], []
    records = project.candidates()
    for record in records:
        slot = dict(record.get("slot") or {})
        if slot.get("slot_id"):
            continue
        wanted = sha_to_slots.get(record.get("sha256")) or set()
        if len(wanted) > 1:
            ambiguous.append({"id": record["id"], "slots": sorted(str(s) for s in wanted)})
            continue
        if not wanted:
            unresolved.append({"id": record["id"], "reason": "no manifest task for these bytes"})
            continue
        slot["slot_id"] = next(iter(wanted))
        slot["slot_source"] = f"restored from {exp_dir.name}/experiment.json"
        record["slot"] = slot
        restored.append({"id": record["id"], "slot_id": slot["slot_id"]})
    if apply and restored:
        lineage.write_jsonl(project.dir / "candidates.jsonl", records)
    log(f"[gen_qwen_edit] slots: {len(restored)} restored, {len(ambiguous)} ambiguous, {len(unresolved)} unresolved")
    return {"restored": restored, "ambiguous": ambiguous, "unresolved": unresolved}


def slot_for(record, slots):
    """The project slot a candidate was asked for, from the prompt and caption it carries, or None.

    tool2's own job category (`expressions_*`, `backgrounds_*`, ...) is the first word of the stem and its vocabulary is
    exactly the project's slot categories, so it decides which slots are even candidates: a background edit's caption
    mentions "from above" and "upper body" because that is the frame it was asked to keep, and letting those words pick
    an angle slot would file the picture under a job nobody asked for. Within the category the rule is the M0 harness's
    (`experiment.match_slot`): the text has to satisfy every alternative group and the strongest match wins.
    """
    lowered = " ".join(str(part or "") for part in (record.get("prompt"), record.get("caption"),
                                                    record.get("stem"))).lower()
    if not lowered.strip():
        return None
    category = (record.get("slot") or {}).get("category") or _category_of(record.get("stem", "")) or ""
    pool = [slot for slot in slots if slot["category"] == category] or slots
    best, best_score = None, 0
    for slot in pool:
        if not all(any(alt.lower() in lowered for alt in group) for group in slot["match"]):
            continue
        hits = sum(1 for group in slot["match"] for alt in group if alt.lower() in lowered)
        if hits > best_score:
            best, best_score = slot, hits
    return best


def assign_slots(project, candidate_ids=None, *, apply=True, log=print):
    """Give new candidates the project's slot ids, so the quota board counts them like the older ones.

    Old candidates get theirs from the manifest (`restore_slots`); a new run's candidate has to be read from what was
    asked for. Several images can land on one slot (a seed each, a stricter prompt), which is reported, not resolved:
    the reviewer is the one who picks.
    """
    from pipeline import experiment

    slots = experiment.SLOTS
    records = project.candidates()
    wanted = set(candidate_ids) if candidate_ids is not None else None
    assigned, unmatched = [], []
    for record in records:
        if wanted is not None and record["id"] not in wanted:
            continue
        slot = dict(record.get("slot") or {})
        if slot.get("slot_id"):
            continue
        match = slot_for(record, slots)
        if match is None:
            unmatched.append({"id": record["id"], "category": slot.get("category")})
            continue
        record["slot"] = {**slot, "slot_id": match["id"], "category": match.get("category", slot.get("category")),
                          "slot_source": "matched from the prompt against the project's slot table"}
        assigned.append({"id": record["id"], "slot_id": match["id"]})
    if apply and assigned:
        lineage.write_jsonl(project.dir / "candidates.jsonl", records)
    counts = {}
    for row in assigned:
        counts[row["slot_id"]] = counts.get(row["slot_id"], 0) + 1
    shared = {slot: count for slot, count in counts.items() if count > 1}
    log(f"[gen_qwen_edit] slots: {len(assigned)} assigned, {len(unmatched)} without a slot"
        + (f"; slots with more than one candidate: {shared}" if shared else ""))
    return {"assigned": assigned, "unmatched": unmatched, "shared": shared}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generator A: run tool2 for a project and import the candidates")
    parser.add_argument("--project", required=True, help="project folder (holds project.json or a card json)")
    parser.add_argument("--card", help="card file, default <project>/project.json")
    parser.add_argument("--only", help="comma list of tool2 categories")
    parser.add_argument("--limit", type=int, help="max jobs per category")
    parser.add_argument("--cancel-file", help="stop when this file appears")
    parser.add_argument("--exclude", help="comma list of input file names to keep out of the snapshot (holdouts)")
    parser.add_argument("--dry-run", action="store_true", help="plan with tool2 but do not generate")
    parser.add_argument("--import-only", metavar="RESULT_JSON",
                        help="import an existing result without running (no GPU work is submitted)")
    parser.add_argument("--restore-slots", metavar="EXPERIMENT_DIR",
                        help="give candidates the slot ids an experiment manifest recorded for their bytes")
    parser.add_argument("--assign-slots", action="store_true",
                        help="give candidates the project slot id their own prompt asks for (no GPU work)")
    parser.add_argument("--set", dest="overrides", action="append", metavar="SECTION.KEY=VALUE",
                        help="override a setting in the snapshot's own copy of tool2's config for this run only "
                             "(e.g. --set expressions.use_extra_refs=false)")
    args = parser.parse_args(argv)

    project_dir = Path(args.project)
    card = read_json(args.card or project_dir / "project.json")
    if not card:
        raise SystemExit(f"no card found at {args.card or project_dir / 'project.json'}")
    if args.assign_slots:
        project = open_project(project_dir)
        if project is None:
            raise SystemExit("--assign-slots needs a project folder")
        assign_slots(project)
        return 0
    if args.restore_slots:
        project = open_project(project_dir)
        if project is None:
            raise SystemExit("--restore-slots needs a project folder")
        restore_slots(project, args.restore_slots)
        return 0
    overrides = parse_overrides(args.overrides)
    if args.import_only:
        # The snapshot a run belongs to is `tool2/<id>/snapshot.json`; the id is a hash of the settings and the input
        # bytes, so it can be recovered without generating anything.
        snapshot = find_snapshot(project_dir, card,
                                 exclude=(args.exclude or "").split(",") if args.exclude else (),
                                 sections=overrides)
        import_candidates(project_dir, card, snapshot, read_json(args.import_only))
        return 0
    snapshot = build_snapshot(project_dir, card, exclude=(args.exclude or "").split(",") if args.exclude else (),
                              sections=overrides)
    result = run(snapshot, only=(args.only or "").split(",") if args.only else None, limit=args.limit,
                 cancel_file=args.cancel_file, dry_run=args.dry_run)
    if result is None:
        return 0
    import_candidates(project_dir, card, snapshot, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
