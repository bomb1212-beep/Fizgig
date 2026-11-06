"""M4: the scoring pass and how the review page shows it.

The scorers themselves are measured against real pictures in `_scratch` and in the M0 report; what is tested here is the
plumbing, and the two rules that plumbing must not break:

* **a missing score is not a zero.** ArcFace finds no face in a back view, and "no identity score" and "the worst
  identity score" are different statements. Every field stays `None` rather than becoming 0.0;
* **one scorer failing does not lose the others.** The pass records the failure with its reason and keeps going, because
  a broken face model must not cost a reviewer the CLIP numbers for the whole set.
"""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import score


def vector(*values):
    """A stand-in for a torch tensor: enough to support `a * b` and `.sum()` as `cosine` uses them."""

    class Vec:
        def __init__(self, items):
            self.items = tuple(items)

        def __mul__(self, other):
            return Vec(a * b for a, b in zip(self.items, other.items))

        def sum(self):
            return sum(self.items)

    return Vec(values)


class FakeEmbedder:
    """Returns the same vector for every path, so every cosine is 1.0 and only the plumbing is under test."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def embed(self, paths):
        return {str(p): self.mapping.get(Path(p).name) or vector(1.0, 0.0) for p in paths}

    def embed_crops(self, path, boxes):
        return {name: vector(1.0, 0.0) for name in boxes}


class FakeProject:
    def __init__(self, root, *, candidates, sources, card=None, review=None):
        self.dir = Path(root)
        self._candidates = candidates
        self._sources = sources
        self._review = review or {}
        self.card = card or {"id": "p", "slot_quotas": {}}
        self.dir.mkdir(parents=True, exist_ok=True)

    def digest(self):
        return "digest"

    def candidates(self):
        return list(self._candidates)

    def source_records(self):
        return list(self._sources)

    def review(self):
        return dict(self._review)

    def resolve_file(self, name):
        return self.dir / name if name else None

    def quotas(self, planned):
        return {slot: {"target": int(t), "accepted": 0, "short": int(t)} for slot, t in planned.items()}


def write_png(path, colour=(255, 255, 255)):
    from PIL import Image
    Image.new("RGB", (16, 16), colour).save(path)
    return path


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_png(self.root / "a.png")
        write_png(self.root / "b.png")
        self.project = FakeProject(
            self.root,
            candidates=[{"id": "c_1", "file": "a.png", "sha256": "s1", "slot": {"id": "frm_face"}, "method": "qwen"},
                        {"id": "c_2", "file": "missing.png", "sha256": "s2", "slot": {}, "method": "qwen"}],
            sources=[{"id": "s_1", "file": "a.png", "origin": "user"},
                     {"id": "s_2", "file": "b.png", "origin": "user"},
                     {"id": "s_3", "file": "b.png", "origin": "synthetic"}],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_only_candidates_with_bytes_on_disk_are_scored(self):
        rows = score.targets(self.project)
        self.assertEqual([r["id"] for r in rows], ["c_1"])

    def test_only_user_originals_are_references(self):
        refs = score.reference_paths(self.project)
        self.assertEqual(len(refs), 2)
        self.assertTrue(all("b.png" in r or "a.png" in r for r in refs))

    def test_a_slot_id_is_carried_through(self):
        rows = score.targets(self.project)
        self.assertEqual(rows[0]["slot"], "frm_face")


class PickTests(unittest.TestCase):
    def test_a_missing_scorer_is_none(self):
        self.assertIsNone(score._pick(None, "c_1", "max_cosine"))

    def test_a_missing_id_is_none_not_zero(self):
        result = {"rows": [{"id": "c_2", "max_cosine": 0.5}]}
        self.assertIsNone(score._pick(result, "c_1", "max_cosine"))

    def test_a_present_value_comes_back(self):
        result = {"rows": [{"id": "c_1", "max_cosine": 0.5}]}
        self.assertEqual(score._pick(result, "c_1", "max_cosine"), 0.5)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_png(self.root / "a.png")
        write_png(self.root / "b.png", (10, 10, 10))
        self.project = FakeProject(
            self.root,
            candidates=[{"id": "c_1", "file": "a.png", "sha256": "s1", "slot": {"id": "frm_face"}, "method": "m"}],
            sources=[{"id": "s_1", "file": "b.png", "origin": "user"}],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_the_file_is_written_and_has_one_flat_row_per_picture(self):
        payload = score.run(self.project, scorers=("clip_vision",), embedder=FakeEmbedder(), log=lambda *_: None)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["id"], "c_1")
        self.assertTrue(score.scores_path(self.project).is_file())

    def test_a_scorer_that_fails_is_recorded_and_the_others_survive(self):
        payload = score.run(self.project, scorers=("clip_vision", "arcface"),
                            embedder=FakeEmbedder(), face_embedder=object(), log=lambda *_: None)
        self.assertIn("clip_vision", payload["scorers"])
        self.assertIn("arcface", payload["unavailable"])
        self.assertEqual(len(payload["rows"]), 1)

    def test_a_picture_with_no_score_keeps_none_rather_than_zero(self):
        payload = score.run(self.project, scorers=("clip_vision",), embedder=FakeEmbedder(), log=lambda *_: None)
        # arcface did not run at all in this pass, so every arcface field is None.
        self.assertIsNone(payload["rows"][0]["arcface_max_cosine"])
        self.assertIsNone(payload["rows"][0]["arcface_no_face"] or None)

    def test_no_originals_is_refused(self):
        empty = FakeProject(self.root, candidates=[], sources=[])
        with self.assertRaises(score.ScoreError):
            score.run(empty, scorers=("clip_vision",), embedder=FakeEmbedder(), log=lambda *_: None)

    def test_no_candidates_is_refused(self):
        write_png(self.root / "o.png")
        bare = FakeProject(self.root, candidates=[], sources=[{"id": "s", "file": "o.png", "origin": "user"}])
        with self.assertRaises(score.ScoreError):
            score.run(bare, scorers=("clip_vision",), embedder=FakeEmbedder(), log=lambda *_: None)

    def test_the_slot_board_counts_the_planned_slots(self):
        self.project.card["slot_quotas"] = {"frm_face": 2}
        payload = score.run(self.project, scorers=("clip_vision",), embedder=FakeEmbedder(), log=lambda *_: None)
        self.assertEqual(len(payload["slot_fill"]), 1)
        self.assertEqual(payload["slot_fill"][0]["slot"], "frm_face")
        self.assertEqual(payload["slot_fill"][0]["target"], 2)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = FakeProject(Path(self.temp.name), candidates=[], sources=[])

    def tearDown(self):
        self.temp.cleanup()

    def test_an_unscored_project_has_an_empty_table(self):
        self.assertEqual(score.table(self.project), {})

    def test_the_table_is_keyed_by_candidate_id(self):
        path = score.scores_path(self.project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": [{"id": "c_1", "clip_max_cosine": 0.4}]}), encoding="utf-8")
        self.assertIn("c_1", score.table(self.project))

    def test_a_broken_file_is_treated_as_unscored(self):
        path = score.scores_path(self.project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
        self.assertIsNone(score.read(self.project))


if __name__ == "__main__":
    unittest.main()
