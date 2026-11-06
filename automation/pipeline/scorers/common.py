"""The measurements that need no model: duplicates, how white a background is, and how full the slots are.

These are the ones that catch the problems a person cannot see at a glance across a thousand pictures. They are also
the ones that run on any machine, which matters because they answer the questions the review page shows first.

None of them deletes anything. The spec is explicit that near-duplicates are **grouped and shown**, not removed: two
pictures can be a byte-identical copy (worth naming) or two near-identical draws (worth seeing next to each other, since
the better one may not be the first), and deciding which to keep is the reviewer's job.
"""
from __future__ import annotations

from pathlib import Path

from . import ScorerError, result


def sha_groups(rows):
    """Group rows by exact bytes. `rows` is `[{"id":..., "sha256":...}, ...]`.

    Exact duplicates are the one case where the answer is unambiguous: the same bytes are the same picture, so the
    group is reported with a representative and the ids that repeat it. It is still not deleted - a candidate record
    that a pipeline step refers to has to keep existing.
    """
    by_sha = {}
    for row in rows:
        sha = row.get("sha256")
        if not sha:
            continue
        by_sha.setdefault(sha, []).append(row["id"])
    groups = [{"sha256": sha, "ids": ids, "representative": ids[0], "count": len(ids)}
              for sha, ids in by_sha.items() if len(ids) > 1]
    return result("sha256_groups", family="common", rows=groups,
                  notes=["exact copies; nothing is removed, the group is only reported"],
                  coverage={"images": len(rows), "groups": len(groups),
                            "redundant": sum(g["count"] - 1 for g in groups)})


def dhash(path, size=8):
    """A difference hash: `size*size` bits of where one pixel is brighter than its right-hand neighbour.

    Chosen over an average hash because it survives brightness and contrast changes, which a re-encoded or re-lit
    draw of the same composition goes through. It is deliberately cheap: the point is to shortlist pairs a person
    should look at, not to be a perceptual metric.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ScorerError(f"hashing needs Pillow: {exc}") from exc
    try:
        image = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except OSError as exc:
        raise ScorerError(f"cannot read {path}: {exc}") from exc
    # One byte per pixel in mode "L", and `tobytes` is stable across Pillow versions where `getdata` is deprecated.
    pixels = image.tobytes()
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def phash_groups(rows, *, max_distance=6, bits=64):
    """Group near-duplicate rows by difference-hash distance, without removing anything.

    `rows` is `[{"id":..., "path":...}, ...]`. Two rows join a group when their hashes are within `max_distance` of a
    row already in it, which is single-link clustering: a chain of similar pictures forms one group, and that is the
    intended behaviour because a chain is exactly what "this slot was drawn ten times" looks like.

    `max_distance=6` of 64 bits is a deliberately loose threshold. Tight enough to separate different poses, loose
    enough to catch a redraw with slightly different lighting; the groups are shown, not acted on.
    """
    entries = []
    for row in rows:
        path = row.get("path")
        if not path or not Path(path).is_file():
            continue
        entries.append({"id": row["id"], "path": str(path), "hash": dhash(path, 8)})
    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if hamming(entries[i]["hash"], entries[j]["hash"]) <= max_distance:
                union(i, j)
    buckets = {}
    for index, entry in enumerate(entries):
        buckets.setdefault(find(index), []).append(entry)
    groups = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        groups.append({
            "ids": [m["id"] for m in members],
            "count": len(members),
            "paths": [m["path"] for m in members],
            "hashes": [f"{m['hash']:016x}" for m in members],
        })
    groups.sort(key=lambda g: -g["count"])
    return result("phash_groups", family="common", rows=groups,
                  notes=[f"near-duplicates within {max_distance} of {bits} bits; grouped for review, never removed"],
                  coverage={"images": len(entries), "groups": len(groups),
                            "unanhashed": len(rows) - len(entries)})


def white_ratio(path, *, threshold=245, sample=256):
    """How much of the border of a picture is near-white, as a share of that border.

    The dataset is meant to be shot on plain backgrounds, so a white border is expected; a low share means the frame is
    filled with something else (a room, a crowd) which changes what a LoRA learns from it. Only the border is measured:
    measuring the whole frame would score a character wearing white as 100% white background.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ScorerError(f"measuring needs Pillow: {exc}") from exc
    try:
        image = Image.open(path).convert("L")
    except OSError as exc:
        raise ScorerError(f"cannot read {path}: {exc}") from exc
    image.thumbnail((sample, sample))
    width, height = image.size
    if width < 3 or height < 3:
        raise ScorerError(f"{Path(path).name} is too small to measure ({width}x{height})")
    pixels = image.load()
    band = max(1, min(width, height) // 20)          # 5% in from each edge
    total = white = 0
    for y in range(height):
        edge_row = y < band or y >= height - band
        for x in range(width):
            if not edge_row and band <= x < width - band:
                continue
            total += 1
            if pixels[x, y] >= threshold:
                white += 1
    if not total:
        raise ScorerError(f"{Path(path).name} has no border pixels to measure")
    return white / total


def white_scores(rows, *, threshold=245):
    """One row per image with its border white ratio; unreadable images are reported, not dropped."""
    out, unreadable = [], []
    for row in rows:
        path = row.get("path")
        try:
            ratio = white_ratio(path, threshold=threshold) if path else None
        except ScorerError as exc:
            unreadable.append({"id": row["id"], "reason": str(exc)})
            continue
        if ratio is None:
            unreadable.append({"id": row["id"], "reason": "no path"})
            continue
        out.append({"id": row["id"], "white_ratio": round(ratio, 4)})
    notes = []
    if unreadable:
        notes.append(f"{len(unreadable)} image(s) could not be measured")
    return result("white_ratio", family="common", rows=out, notes=notes,
                  coverage={"measured": len(out), "unreadable": unreadable})


def slot_fill(quotas, counts):
    """Per slot: target, how many pictures it has, and the shortfall.

    `counts` is `{slot: accepted}`. Reported for every slot in `quotas`, including the ones already full: a slot that is
    over target is also worth seeing, because it means the quota is not describing what the set actually contains.
    """
    rows = []
    for slot, target in sorted(quotas.items()):
        target = int(target)
        have = int(counts.get(slot, 0))
        rows.append({"slot": slot, "target": target, "accepted": have,
                     "short": max(0, target - have), "over": max(0, have - target)})
    short = [row["slot"] for row in rows if row["short"]]
    return result("slot_fill", family="common", rows=rows,
                  notes=[f"{len(short)} slot(s) short"] if short else ["every slot is filled"],
                  coverage={"slots": len(rows), "short": len(short), "over": sum(1 for r in rows if r["over"])})
