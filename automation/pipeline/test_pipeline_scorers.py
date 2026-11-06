"""The two identity scorers, tested on what can be tested without loading a 1.2 GB model.

`Embedder` needs torch and CLIP Vision weights, so the tests drive `identity_scores` and `face_scores` with a stand-in
that returns fixed vectors. That still checks the parts where a mistake changes a number a person would read: which
references are compared, whether an image is compared with itself, and what happens to a picture with no score.
"""
import tempfile
import unittest
from pathlib import Path

from pipeline.scorers import arcface, clip_vision


def vec(*values):
    class Vec:
        def __init__(self, items):
            self.items = tuple(items)

        def __mul__(self, other):
            return Vec(a * b for a, b in zip(self.items, other.items))

        def sum(self):
            return sum(self.items)

    return Vec(values)


class FixedEmbedder:
    """`{filename: vector}`, so a test can say exactly how similar two pictures are."""

    def __init__(self, vectors, crop=None):
        self.vectors = vectors
        self.crop = crop or {}

    def embed(self, paths):
        out = {}
        for path in paths:
            name = Path(path).name
            out[str(path)] = self.vectors.get(name, vec(0.0, 1.0))
        return out

    def embed_crops(self, path, boxes):
        name = Path(path).name
        return {box: self.crop.get((name, box), vec(0.0, 1.0)) for box in boxes}


class CosineTests(unittest.TestCase):
    def test_identical_vectors_are_one(self):
        self.assertAlmostEqual(clip_vision.cosine(vec(1.0, 0.0), vec(1.0, 0.0)), 1.0)

    def test_orthogonal_vectors_are_zero(self):
        self.assertAlmostEqual(clip_vision.cosine(vec(1.0, 0.0), vec(0.0, 1.0)), 0.0)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make(self, name, colour):
        from PIL import Image
        path = self.root / name
        Image.new("RGB", (8, 8), colour).save(path)
        return path

    def test_max_and_mean_come_from_different_originals(self):
        # Against two references that disagree, the maximum and the mean must not be the same number: one answers
        # "which original does this look like", the other "how like the character in general".
        target = self.make("t.png", (0, 0, 0))
        near = self.make("near.png", (0, 0, 1))
        far = self.make("far.png", (0, 1, 1))
        embedder = FixedEmbedder({"t.png": vec(1.0, 0.0), "near.png": vec(1.0, 0.1), "far.png": vec(0.0, 1.0)})
        result = clip_vision.identity_scores([target], [near, far], embedder=embedder)
        row = result["rows"][0]
        self.assertAlmostEqual(row["max_cosine"], 1.0)
        self.assertAlmostEqual(row["mean_cosine"], 0.5)
        self.assertEqual(row["closest"], "near.png")

    def test_an_exported_copy_of_an_original_is_not_compared_with_itself(self):
        # An export renames every file to `<sha16>.png`, so the copy of an original has a different path and the same
        # bytes. Comparing it with the original scored it at exactly 1.0 and read as perfect identity.
        original = self.make("orig.png", (10, 10, 10))
        copy = self.root / "copy.png"
        copy.write_bytes(original.read_bytes())
        other = self.make("other.png", (90, 90, 90))
        embedder = FixedEmbedder({"orig.png": vec(1.0, 0.0), "copy.png": vec(1.0, 0.0), "other.png": vec(0.0, 1.0)})
        result = clip_vision.identity_scores([copy], [original, other], embedder=embedder)
        row = result["rows"][0]
        self.assertTrue(row["self_excluded"])
        self.assertEqual(row["reference_count"], 1)
        self.assertAlmostEqual(row["max_cosine"], 0.0)

    def test_the_only_original_gets_no_score_rather_than_a_perfect_one(self):
        original = self.make("only.png", (10, 10, 10))
        embedder = FixedEmbedder({"only.png": vec(1.0, 0.0)})
        result = clip_vision.identity_scores([original], [original], embedder=embedder)
        self.assertIsNone(result["rows"][0]["max_cosine"])
        self.assertEqual(result["coverage"]["no_reference"], [str(original)])

    def test_no_reference_at_all_is_refused(self):
        target = self.make("t.png", (0, 0, 0))
        with self.assertRaises(clip_vision.ScorerError):
            clip_vision.identity_scores([target], [], embedder=FixedEmbedder({}))

    def test_a_mismatched_id_count_is_refused(self):
        target = self.make("t.png", (0, 0, 0))
        ref = self.make("r.png", (0, 0, 1))
        with self.assertRaises(clip_vision.ScorerError):
            clip_vision.identity_scores([target], [ref], embedder=FixedEmbedder({}), ids=["a", "b"])

    def test_crops_are_compared_against_crops_of_the_same_name(self):
        target = self.make("t.png", (0, 0, 0))
        ref = self.make("r.png", (0, 0, 1))
        embedder = FixedEmbedder({"t.png": vec(1.0, 0.0), "r.png": vec(1.0, 0.0)},
                                 crop={("t.png", "face"): vec(0.0, 1.0), ("r.png", "face"): vec(1.0, 0.0)})
        result = clip_vision.identity_scores([target], [ref], embedder=embedder,
                                             crops={"t.png": {"face": [0, 0, 4, 4]},
                                                    "r.png": {"face": [0, 0, 4, 4]}})
        row = result["rows"][0]
        self.assertIn("crop_face_max", row)
        self.assertAlmostEqual(row["crop_face_max"], 0.0)


class FaceScorerTests(unittest.TestCase):
    """ArcFace refuses plainly rather than scoring nothing."""

    class FakeFaceEmbedder:
        def __init__(self, available=True, vectors=None):
            self.available = available
            self.vectors = vectors or {}

        def embed(self, path):
            return self.vectors.get(Path(path).name)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = self.root / "a.png"
        self.b = self.root / "b.png"
        for path in (self.a, self.b):
            path.write_bytes(b"x")

    def tearDown(self):
        self.temp.cleanup()

    def test_no_insightface_is_refused_with_a_reason(self):
        with self.assertRaises(arcface.ScorerError) as caught:
            arcface.face_scores([self.a], [self.b], embedder=self.FakeFaceEmbedder(available=False))
        self.assertIn("InsightFace", str(caught.exception))

    def test_no_face_in_any_original_is_refused(self):
        with self.assertRaises(arcface.ScorerError) as caught:
            arcface.face_scores([self.a], [self.b], embedder=self.FakeFaceEmbedder())
        self.assertIn("nothing to compare against", str(caught.exception))

    def test_a_picture_with_no_face_gets_no_score_and_a_reason(self):
        import numpy as np
        embedder = self.FakeFaceEmbedder(vectors={"b.png": np.array([1.0, 0.0])})
        result = arcface.face_scores([self.a, self.b], [self.b], ids=["c_1", "c_2"], embedder=embedder)
        rows = {row["id"]: row for row in result["rows"]}
        self.assertIsNone(rows["c_1"]["max_cosine"])
        self.assertEqual(rows["c_1"]["no_face"], "no face was detected")
        # c_2 has the same bytes as the only reference, so it is the same picture and is not compared with itself.
        self.assertIsNone(rows["c_2"]["max_cosine"])
        self.assertEqual(len(result["coverage"]["no_face"]), 1)

    def test_the_scored_count_excludes_pictures_with_no_face(self):
        import numpy as np
        other = self.root / "c.png"
        other.write_bytes(b"y")
        embedder = self.FakeFaceEmbedder(vectors={"b.png": np.array([1.0, 0.0]), "c.png": np.array([0.6, 0.8])})
        result = arcface.face_scores([other, self.a], [self.b], ids=["c_3", "c_1"], embedder=embedder)
        self.assertEqual(result["coverage"]["scored"], 1)


if __name__ == "__main__":
    unittest.main()
