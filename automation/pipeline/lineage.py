"""Provenance for every image the pipeline touches.

The folder an image sits in says nothing about where it came from: tool2's `set_anchor()` writes generated full-body
anchors straight into `characters/<name>/`, next to the originals. So origin is recorded, never inferred:

* `train_generation` counts how many LoRA generations separate an image from a real original. 0 is an original, an
  image made by a LoRA is one more than the newest generation its LoRA was trained on, and an edit keeps its input's
  generation (it corrects a picture, it does not make a new one).
* `edit_depth` counts how many times an image has been through an editor (Qwen edit, Qwen fix), so a corrected picture
  can be told from its input and the training cap can refuse a chain that grew too long.

`lineage.max_train_generation`, `seed_max_generation` and `max_edit_depth` are checked here, at the point where a
snapshot is chosen, so an over-generation image is refused with a reason instead of quietly joining a training set.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

HEAD_BYTES = 4 * 1024 * 1024          # a quick hash reads this much at each end of a large model file
SMALL_FILE = 8 * 1024 * 1024          # below this, a plain sha256 of the whole file is cheaper than three reads

# Every engine that can have produced a picture. `ph3_edit` is recorded so a P-H3 edit can be described truthfully;
# whether it may also *enter a seed set* is a separate question answered by seed.POLICY_GENERATORS, because that
# depends on the policy a snapshot was written under.
GENERATORS = ("user", "qwen_edit", "ph3_edit", "krea2_lora", "krea2_lora+qwen_fix")
DEFAULT_LINEAGE = {"max_train_generation": 2, "seed_max_generation": 1, "max_edit_depth": 1}


class LineageError(ValueError):
    """A record that cannot be made honest (unknown parent, impossible generation) or breaks a cap."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quick_hash(path):
    """A model file's identity that is cheap enough to compute every run.

    A 26 GB DiT is not hashed end to end for a resume key: size, modification time and the two ends are enough to
    notice a replaced or truncated file. Small files (LoRAs) get the full sha256, which is both cheap and exact.
    """
    path = Path(path)
    stat = path.stat()
    size = stat.st_size
    if size <= SMALL_FILE:
        return {"mode": "full", "size": size, "sha256": sha256_file(path)}
    with open(path, "rb") as handle:
        head = hashlib.sha256(handle.read(HEAD_BYTES)).hexdigest()
        handle.seek(max(0, size - HEAD_BYTES))
        tail = hashlib.sha256(handle.read(HEAD_BYTES)).hexdigest()
    return {"mode": "edges", "size": size, "mtime_ns": stat.st_mtime_ns, "head_sha256": head, "tail_sha256": tail}


def hash_key(value):
    """A single comparable string for a `quick_hash` dict (or for a bare sha256 string)."""
    if isinstance(value, str):
        return value
    if not value:
        return ""
    if value.get("mode") == "full":
        return "full:" + str(value.get("sha256", ""))
    return "edges:{size}:{mtime_ns}:{head_sha256}:{tail_sha256}".format(
        size=value.get("size"), mtime_ns=value.get("mtime_ns"),
        head_sha256=value.get("head_sha256"), tail_sha256=value.get("tail_sha256"))


def image_id(sha):
    return "c_" + sha[:16]


def source_id(sha):
    return "s_" + sha[:16]


def model_entry(path, strength=1.0, trained_on_snapshot=None, trained_on_generation=0):
    """One LoRA's row in a candidate's `models.loras`."""
    path = Path(path)
    return {
        "name": path.name,
        "path": str(path),
        "sha256": sha256_file(path),
        "strength": float(strength),
        "trained_on_snapshot": trained_on_snapshot,
        "trained_on_generation": int(trained_on_generation),
    }


def dit_entry(path):
    """The base model's row in `models.dit` (quick hashed: these files are tens of gigabytes)."""
    return {"name": Path(path).name, "path": str(path), "quick_hash": quick_hash(path)}


def generation_from_loras(loras):
    """An image made by LoRAs inherits one more generation than the newest snapshot the LoRAs were trained on."""
    if not loras:
        return 0
    return 1 + max(int((lora.get("trained_on_generation") or 0)) for lora in loras)


def generation_from_edit(parent_generation):
    """An edit keeps its input's generation: it corrects the picture, it does not add a training generation."""
    return int(parent_generation)


def record(path, *, generator, parent_ids=(), train_generation=0, edit_depth=0, slot=None, models=None,
           prompt="", seed=None, steps=None, cfg=None, width=None, height=None, graph_version="", **extra):
    """A `candidates.jsonl` line for one image, with its id and sha256 taken from the bytes on disk.

    A generated image must point somewhere: parent images, or a LoRA whose training snapshot is named. `extra` may
    carry `allow_missing_parents=True` for the rare case where neither is knowable, which also records
    `orphan_reason`, because an unexplained orphan is exactly the thing this module exists to prevent.
    """
    path = Path(path)
    sha = sha256_file(path)
    if generator not in GENERATORS:
        raise LineageError(f"unknown generator {generator!r} (known: {', '.join(GENERATORS)})")
    snapshots = [lora for lora in ((models or {}).get("loras") or []) if lora.get("trained_on_snapshot")]
    allow_missing = bool(extra.pop("allow_missing_parents", False))
    if generator != "user" and not parent_ids and not snapshots and not allow_missing:
        raise LineageError(f"{path.name}: a generated image needs a parent id or a LoRA trained on a recorded "
                           "snapshot (pass allow_missing_parents and a reason if neither is knowable)")
    data = {
        "id": image_id(sha),
        "file": str(path),
        "sha256": sha,
        "generator": generator,
        "parent_ids": list(parent_ids),
        "train_generation": int(train_generation),
        "edit_depth": int(edit_depth),
        "slot": slot or {},
        "models": models or {},
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "graph_version": graph_version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    data.update(extra)
    return data


def source_record(path, *, origin="user", parent_ids=(), note=""):
    """A `sources.jsonl` line for an original (or for a synthetic image that a folder happens to hold)."""
    path = Path(path)
    sha = sha256_file(path)
    if origin not in ("user", "synthetic"):
        raise LineageError(f"origin must be 'user' or 'synthetic', not {origin!r}")
    return {"id": source_id(sha), "sha256": sha, "src_path": str(path), "origin": origin,
            "parent_ids": list(parent_ids), "note": note}


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_jsonl(path, records):
    """Append records that are not already there, by id. Returns the number actually written.

    Rerunning a step must not duplicate its own history, so the caller can call this unconditionally.
    """
    path = Path(path)
    known = {r.get("id") for r in read_jsonl(path)}
    fresh = [r for r in records if r.get("id") and r["id"] not in known]
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + os.urandom(4).hex())
    with open(temp, "w", encoding="utf-8") as handle:
        for record in fresh:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(temp.read_text(encoding="utf-8"))
    temp.unlink(missing_ok=True)
    return len(fresh)


def write_jsonl(path, records):
    """Replace a jsonl file with exactly these records (still de-duplicated by id, last one winning)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {}
    for record in records:
        key = record.get("id") or json.dumps(record, sort_keys=True)
        by_id[key] = record
    temp = path.with_name(path.name + ".tmp-" + os.urandom(4).hex())
    with open(temp, "w", encoding="utf-8") as handle:
        for record in by_id.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    return len(by_id)


def caps(lineage=None):
    return {**DEFAULT_LINEAGE, **{k: int(v) for k, v in (lineage or {}).items() if v is not None}}


def check_generation(record, *, limit, what="training"):
    """Raise LineageError when a record is past the generation a snapshot allows."""
    generation = int(record.get("train_generation") or 0)
    if generation > limit:
        raise LineageError(
            f"{Path(record.get('file', '?')).name} is training generation {generation}, past the {what} cap of "
            f"{limit}; it descends from {len(record.get('parent_ids') or [])} parent(s)")
    return True


def check_edit_depth(record, *, limit):
    depth = int(record.get("edit_depth") or 0)
    if depth > limit:
        raise LineageError(f"{Path(record.get('file', '?')).name} has edit depth {depth}, past the cap of {limit}")
    return True


def filter_for_snapshot(records, lineage=None):
    """Split records into (kept, refused) for a training snapshot, applying the generation and edit-depth caps."""
    limits = caps(lineage)
    kept, refused = [], []
    for record in records:
        try:
            check_generation(record, limit=limits["max_train_generation"])
            check_edit_depth(record, limit=limits["max_edit_depth"])
            kept.append(record)
        except LineageError as exc:
            refused.append({"id": record.get("id"), "file": record.get("file"), "reason": str(exc)})
    return kept, refused


def trace_to_origin(record, index):
    """Follow parent ids to the originals, returning the ids visited (missing parents are recorded as unknown)."""
    seen, stack, unknown = [], [record], []
    while stack:
        current = stack.pop()
        for parent in current.get("parent_ids") or []:
            if parent in seen:
                continue
            seen.append(parent)
            if parent in index:
                stack.append(index[parent])
            else:
                unknown.append(parent)
    return seen, unknown
