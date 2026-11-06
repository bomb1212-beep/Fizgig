"""Where a scorer's numbers come from, and what they are allowed to claim.

A score here is a **measurement attached to a picture**, not a verdict. The spec is explicit that scoring never rejects
an image: the reviewer does that. So every scorer returns observations with the rule that produced them, and nothing in
this package may delete, exclude or refuse anything.

The two things that make a score usable later are both here:

* `family` - which set of rules produced it. A CLIP cosine between two crops and an ArcFace cosine between two faces
  are not the same number even when both are 0.87, and comparing them as if they were is how a threshold stops meaning
  anything;
* `inputs` - what was measured against what. A score is only re-derivable if the images it compared are named.
"""
from __future__ import annotations


class ScorerError(RuntimeError):
    """A scorer that cannot measure what it was asked to, said plainly instead of returning a guess."""


def result(name, *, family, rows, reference=None, notes=(), coverage=None):
    """One scorer's output.

    `rows` is one dict per image, each carrying at minimum `id` and the numbers that scorer produces. `reference`
    names what the numbers were measured against (the originals, the crops, the holdout), because "identity 0.83" is
    meaningless without it.
    """
    return {
        "scorer": name,
        "family": family,
        "reference": reference,
        "rows": list(rows),
        "notes": list(notes),
        "coverage": coverage or {},
    }


def by_id(rows):
    return {row["id"]: row for row in rows}
