"""Identity scoring for photographic sets: ArcFace, through the embedder Fizgig already ships.

The spec says to reuse `face_utils.FaceEmbedder` rather than add another face model, and that is what this does. It is
kept apart from `clip_vision.py` because the two answer the same question with different instruments and their numbers
are not comparable: ArcFace compares **faces** and says nothing about a picture with no detectable face, while CLIP
compares whole frames and always returns something.

That difference is the reason this scorer is honest about coverage instead of returning a low score for a missing face.
A picture where no face was found has **no ArcFace identity score** - not a bad one - and reporting 0.0 would make a
profile shot look like a different person.
"""
from __future__ import annotations

from pathlib import Path

from . import ScorerError, result


def _embedder():
    """The project's own ArcFace embedder, imported from the Fizgig root.

    `face_utils` sits at the repository root, not on the path a pipeline module is normally run from (the pipeline is
    used from `automation/`), so the root is put on `sys.path` before importing rather than requiring the caller to
    arrange it. Imported lazily so this module stays importable (and testable) without insightface installed.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]      # .../Fizgig_auto, three levels above pipeline/scorers/
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from face_utils import FaceEmbedder
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ScorerError(f"face_utils is not importable from {root}: {exc}") from exc
    return FaceEmbedder()


def _embed(embedder, path):
    """(embedding or None, reason). A missing face is a fact to report, not an error to raise."""
    if not Path(path).is_file():
        return None, "file is missing"
    try:
        vector = embedder.embed(str(path))
    except Exception as exc:  # noqa: BLE001 - a broken image must not end a scoring pass
        return None, f"{type(exc).__name__}: {exc}"
    if vector is None:
        return None, "no face was detected"
    return vector, ""


def face_scores(paths, reference_paths, *, embedder=None, ids=None, exclude_self=True):
    """Score each image's face against the faces found in the user's originals.

    Reports `max_cosine` and `mean_cosine` over the originals that had a detectable face, plus `no_face` for the images
    where nothing could be measured. When no original yields a face the whole scorer refuses, because comparing against
    an empty set would silently score everything.

    `exclude_self` keeps an original from being compared with itself, matched by file **content**: an exported copy of
    an original has a different path but identical bytes, and that comparison is always exactly 1.0 and says nothing
    about identity.
    """
    reference_paths = [Path(p) for p in reference_paths]
    if not reference_paths:
        raise ScorerError("face scoring needs at least one original to compare against")
    embedder = embedder or _embedder()
    if not getattr(embedder, "available", True):
        raise ScorerError("InsightFace is not installed, so ArcFace scoring is unavailable; use the clip_vision "
                          "scorer for this set instead")

    ref, ref_skipped = [], []
    for path in reference_paths:
        vector, reason = _embed(embedder, path)
        if vector is None:
            ref_skipped.append({"path": str(path), "reason": reason})
        else:
            ref.append((path, vector))
    if not ref:
        raise ScorerError("no face was detected in any original, so there is nothing to compare against ("
                          + "; ".join(f"{Path(s['path']).name}: {s['reason']}" for s in ref_skipped[:3]) + ")")

    targets = [Path(p) for p in paths]
    ids = list(ids) if ids is not None else [p.stem for p in targets]
    if len(ids) != len(targets):
        raise ScorerError(f"{len(ids)} id(s) for {len(targets)} image(s)")

    rows, skipped, no_reference = [], [], []
    from .clip_vision import _digest
    ref_digests = [_digest(p) for p, _v in ref] if exclude_self else [""] * len(ref)
    for image_id, path in zip(ids, targets):
        vector, reason = _embed(embedder, path)
        if vector is None:
            skipped.append({"id": image_id, "path": str(path), "reason": reason})
            # The row still exists with no score: a caller filtering on `max_cosine` must not mistake "no face" for
            # "zero similarity", and a row that vanishes entirely is worse than a row that says why.
            rows.append({"id": image_id, "path": str(path), "max_cosine": None, "mean_cosine": None,
                         "closest": None, "no_face": reason})
            continue
        mine = _digest(path) if exclude_self else ""
        keep = [i for i, (ref_path, _v) in enumerate(ref)
                if not (exclude_self and mine and ref_digests[i] == mine)]
        pool = [ref[i] for i in keep]
        if not pool:
            no_reference.append(str(path))
            rows.append({"id": image_id, "path": str(path), "max_cosine": None, "mean_cosine": None,
                         "closest": None, "no_face": "no other original to compare with"})
            continue
        import numpy as np
        sims = [float(np.dot(vector, ref_vector)) for _p, ref_vector in pool]
        rows.append({"id": image_id, "path": str(path),
                     "max_cosine": max(sims), "mean_cosine": sum(sims) / len(sims),
                     "closest": pool[sims.index(max(sims))][0].name,
                     "reference_count": len(pool), "no_face": ""})
    notes = [f"ArcFace compares faces only; {len(skipped)} image(s) had none and carry no score"]
    if exclude_self:
        notes.append("an original is compared only with the other originals, never with itself")
    if ref_skipped:
        notes.append(f"{len(ref_skipped)} original(s) had no detectable face and were not used")
    return result("arcface", family="real", rows=rows,
                  reference={"kind": "originals", "count": len(ref),
                             "paths": [str(p) for p, _v in ref]},
                  notes=notes,
                  coverage={"scored": len(rows) - len(skipped) - len(no_reference), "no_face": skipped,
                            "no_reference": no_reference,
                            "references_used": len(ref), "references_skipped": ref_skipped})
