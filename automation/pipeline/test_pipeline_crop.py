"""N7: a crop is a per-image decision, kept, and never a change to the pictures.

The audit's finding (`review.crop_scores` applying one ratio box to every original) is the first test here: a group
shot and a head-and-shoulders portrait put the face in very different places, so one box for both compares different
regions and reports it as one number.

The rest is what the spec asks of the store: the box and the image's SHA are kept, the original bytes are untouched,
a missing reference crop is reported as unspecified rather than guessed, and a reload restores what was measured.
"""
import json
import sys
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
for entry in (str(AUTO.parent / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import review  # noqa: E402
from PIL import Image  # noqa: E402


def png(path, *, size=(64, 64), box=None):
    """A picture with a bright square at `box` (in 0~1 fractions), so a crop's content is checkable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (10, 10, 10))
    if box:
        x, y, w, h = box
        left, top = int(x * size[0]), int(y * size[1])
        right, bottom = int((x + w) * size[0]), int((y + h) * size[1])
        for px in range(left, max(left + 1, right)):
            for py in range(top, max(top + 1, bottom)):
                image.putpixel((px, py), (240, 240, 240))
    image.save(path)
    return path


class _FakeProject:
    """The parts of `Project` the crop path uses, over a real temporary folder."""

    def __init__(self, root, *, candidates, sources, review_state=None):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._candidates = candidates
        self._sources = sources
        self._review = review_state or {}
        self.card = {"id": "p", "trigger": "t", "slot_quotas": {}, "holdout": {"ids": []}}

    def digest(self):
        return "digest"

    def candidates(self):
        return list(self._candidates)

    def source_records(self):
        return list(self._sources)

    def holdout_entry(self):
        return dict(self.card.get("holdout") or {"ids": []})

    def candidate_by_id(self, candidate_id):
        return next((r for r in self._candidates if r["id"] == candidate_id), None)

    def state_of(self, candidate_id):
        return self._review.get(candidate_id) or {}

    def resolve_file(self, name):
        """An absolute path stays as given; a bare name resolves under the project, like the real `Project`."""
        if not name:
            return None
        path = Path(name)
        return path if path.is_absolute() else self.dir / path

    def index(self):
        return {record["id"]: record for record in self._candidates + self._sources}

    def quotas(self, planned):
        return {slot: {"target": int(t), "accepted": 0, "short": int(t)} for slot, t in planned.items()}

    def source_share(self):
        return 0.0, {"user": len(self._sources), "synthetic": 0}

    def undecided(self):
        return [record["id"] for record in self._candidates if record["id"] not in self._review]

    def review(self):
        return dict(self._review)


class VectorEmbedder:
    """An embedder whose vector is the mean brightness of one crop box, so a box that lands elsewhere scores lower.

    `embed_crops(path, boxes)` is the interface the real `clip_vision.Embedder` offers and `identity_scores` calls, so
    this fake models the same contract: one vector per named box, with the box in **pixels**.
    """

    def embed(self, paths):
        out = {}
        for path in paths:
            path = Path(path)
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                pixels = list(rgb.getdata())
                mean = sum(sum(pixel) / 3 for pixel in pixels) / len(pixels) / 255.0
            # Keyed by the full path string, which is how `identity_scores` looks vectors up.
            out[str(path)] = Vec([mean, 1.0 - mean])
        return out

    def embed_crops(self, path, boxes):
        path = Path(path)
        rows = {}
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            for name, box in boxes.items():
                x, y, w, h = (int(v) for v in box)
                region = rgb.crop((x, y, x + w, y + h))
                pixels = list(region.getdata())
                if not pixels:
                    rows[name] = Vec([0.0, 1.0])
                    continue
                mean = sum(sum(pixel) / 3 for pixel in pixels) / len(pixels) / 255.0
                rows[name] = Vec([mean, 1.0 - mean])
        return rows


class Vec:
    def __init__(self, values):
        self.values = values

    def __mul__(self, other):
        return Vec([a * b for a, b in zip(self.values, other.values)])

    def sum(self):
        return sum(self.values)


class CropStoreTests(unittest.TestCase):
    """The sidecar: what is kept, and what it must never touch."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = png(self.root / "src" / "a.png", box=(0.1, 0.1, 0.2, 0.2))
        self.candidate = png(self.root / "cand.png", box=(0.6, 0.6, 0.2, 0.2))
        self.project = _FakeProject(
            self.root / "proj",
            candidates=[{"id": "c_1", "file": str(self.candidate), "generator": "qwen_edit",
                         "slot": {"slot_id": "frm"}, "train_generation": 1, "edit_depth": 0,
                         "parent_ids": ["s_1"], "caption": ""}],
            sources=[{"id": "s_1", "file": str(self.original), "origin": "user", "sha256": "sa"}])

    def test_a_box_is_stored_with_the_images_sha(self):
        import hashlib
        review.set_crop(self.project, self.original, [0.1, 0.1, 0.2, 0.2], kind="face")
        store = review.read_crop_store(self.project)
        entry = store["s_1"]
        self.assertEqual(entry["box"], [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(entry["kind"], "face")
        self.assertEqual(entry["image_sha256"], hashlib.sha256(self.original.read_bytes()).hexdigest())
        self.assertEqual(entry["by"], "user")

    def test_the_original_bytes_are_never_touched(self):
        before = self.original.read_bytes()
        review.set_crop(self.project, self.original, [0.2, 0.2, 0.3, 0.3])
        review.crop_scores(self.project, "c_1", [0.5, 0.5, 0.3, 0.3], embedder=VectorEmbedder())
        self.assertEqual(self.original.read_bytes(), before, "a crop is a measurement, not an edit")

    def test_a_box_is_read_back_for_its_own_kind_only(self):
        review.set_crop(self.project, self.candidate, [0.6, 0.6, 0.2, 0.2], kind="face")
        self.assertEqual(review.crop_for(self.project, self.candidate, name="face"), [0.6, 0.6, 0.2, 0.2])
        self.assertIsNone(review.crop_for(self.project, self.candidate, name="upper"),
                          "a face crop is not an upper-body crop")

    def test_a_corrupt_store_reads_as_empty_rather_than_raising(self):
        path = review.crop_store_path(self.project)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(review.read_crop_store(self.project), {})


class PerImageCropTests(unittest.TestCase):
    """The comparison the audit found wrong: one box for pictures that frame differently."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Two originals whose bright region sits in different places: a head-and-shoulders shot and a group shot.
        self.close = png(self.root / "src" / "close.png", box=(0.35, 0.10, 0.30, 0.30))
        self.wide = png(self.root / "src" / "wide.png", box=(0.05, 0.60, 0.12, 0.12))
        self.candidate = png(self.root / "cand.png", box=(0.35, 0.10, 0.30, 0.30))
        self.project = _FakeProject(
            self.root / "proj",
            candidates=[{"id": "c_1", "file": str(self.candidate), "generator": "qwen_edit", "slot": {"slot_id": "frm"},
                         "train_generation": 1, "edit_depth": 0, "parent_ids": ["s_1"], "caption": ""}],
            sources=[{"id": "s_1", "file": str(self.close), "origin": "user", "sha256": "sa"},
                     {"id": "s_2", "file": str(self.wide), "origin": "user", "sha256": "sb"}])

    def test_a_per_image_box_lands_on_each_originals_own_subject(self):
        result = review.crop_scores(
            self.project, "c_1", [0.35, 0.10, 0.30, 0.30], name="face", embedder=VectorEmbedder(),
            boxes={str(self.close): [0.35, 0.10, 0.30, 0.30], str(self.wide): [0.05, 0.60, 0.12, 0.12]})
        self.assertTrue(result["per_image"])
        self.assertEqual(result["unmeasured"], [])
        self.assertEqual(result["boxes"]["wide.png"], [0.05, 0.6, 0.12, 0.12])
        # Both crops landed on a bright square, so the candidate matches both originals closely.
        self.assertGreater(result["mean"], 0.8)

    def test_one_box_for_every_picture_is_worse_on_the_framed_differently_original(self):
        # This is the audit's finding, made explicit. The candidate wears the bright patch where the close-up original
        # has it; the wide shot puts its subject somewhere else entirely. One shared ratio box lands on the wide
        # shot's background, so it cannot recognise that original - naming each picture's own region can.
        #
        # The comparison is on `mean`, not `max`: `max` asks "does it look like *any* original" and is already
        # saturated by the close-up match, while `mean` asks "how well does it match the set", which is exactly what a
        # box that lands on a background ruins.
        boxes = {str(self.close): [0.35, 0.10, 0.30, 0.30], str(self.wide): [0.05, 0.60, 0.12, 0.12]}
        shared = review.crop_scores(self.project, "c_1", [0.35, 0.10, 0.30, 0.30], name="face",
                                    embedder=VectorEmbedder(), store=False)
        per_image = review.crop_scores(self.project, "c_1", [0.35, 0.10, 0.30, 0.30], name="face",
                                       embedder=VectorEmbedder(), boxes=boxes)
        self.assertGreater(per_image["mean"], shared["mean"],
                           "naming each picture's own region has to beat one box for every picture")

    def test_a_reference_the_request_does_not_name_is_reported_unmeasured(self):
        result = review.crop_scores(
            self.project, "c_1", [0.35, 0.10, 0.30, 0.30], name="face", embedder=VectorEmbedder(),
            boxes={str(self.close): [0.35, 0.10, 0.30, 0.30]})
        self.assertEqual(result["unmeasured"], ["wide.png"],
                         "a reference with no box is named, not silently scored with the candidate's box")

    def test_a_box_that_would_have_no_area_is_refused(self):
        with self.assertRaises(review.pj.ProjectError):
            review.crop_scores(self.project, "c_1", [0.999, 0.999, 0.001, 0.001], name="face",
                               embedder=VectorEmbedder(), store=False)

    def test_the_boxes_used_are_stored_so_a_reload_can_show_them(self):
        review.crop_scores(self.project, "c_1", [0.35, 0.10, 0.30, 0.30], name="face", embedder=VectorEmbedder())
        store = review.read_crop_store(self.project)
        self.assertEqual(store["s_1"]["box"], [0.35, 0.1, 0.3, 0.3])
        self.assertEqual(store["s_2"]["box"], [0.35, 0.1, 0.3, 0.3])
        self.assertEqual(store["c_1"]["box"], [0.35, 0.1, 0.3, 0.3])


class RouteTests(unittest.TestCase):
    """The HTTP surface: parse the boxes, restore them, clear them."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.close = png(self.root / "src" / "close.png", box=(0.35, 0.10, 0.30, 0.30))
        self.candidate = png(self.root / "cand.png", box=(0.35, 0.10, 0.30, 0.30))
        self.project = _FakeProject(
            self.root / "proj",
            candidates=[{"id": "c_1", "file": str(self.candidate), "generator": "qwen_edit", "slot": {"slot_id": "frm"},
                         "train_generation": 1, "edit_depth": 0, "parent_ids": ["s_1"], "caption": ""}],
            sources=[{"id": "s_1", "file": str(self.close), "origin": "user", "sha256": "sa"}])
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()

    def serve(self):
        server = review.make_server(self.project, port=0, quota_plan=None, embedder_factory=VectorEmbedder,
                                    claim_factory=lambda: _NullClaim(), log=lambda *_: None)
        self.servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_the_page_restores_a_stored_box(self):
        review.set_crop(self.project, self.candidate, [0.35, 0.1, 0.3, 0.3], kind="face")
        server = self.serve()
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            html = response.read().decode("utf-8")
        self.assertIn("crop-score", html)
        self.assertIn("0.35,0.1,0.3,0.3", html)

    def test_the_crops_route_answers_with_the_stored_kinds(self):
        review.set_crop(self.project, self.candidate, [0.35, 0.1, 0.3, 0.3], kind="face")
        server = self.serve()
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/crops/c_1", timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertEqual(data["id"], "c_1")
        self.assertIn("face", data["crops"])
        self.assertEqual(data["crops"]["face"]["box"], [0.35, 0.1, 0.3, 0.3])

    def test_a_per_image_request_scores_and_reports_what_it_measured(self):
        server = self.serve()
        port = server.server_address[1]
        # The page URL-encodes each key once; `parse_qs` decodes it once, and `parse_crop_boxes` decodes the key it
        # finds. Sending a pre-encoded value would test a double encode the browser never produces.
        raw = f"{self.close}:0.35,0.1,0.3,0.3"
        query = urllib.parse.urlencode({"kind": "face", "box": "0.35,0.1,0.3,0.3", "boxes": raw})
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/crop-score/c_1?{query}", timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertTrue(data["per_image"])
        self.assertEqual(data["unmeasured"], [])

    def test_clearing_forgets_the_boxes_and_keeps_the_pictures(self):
        review.set_crop(self.project, self.candidate, [0.35, 0.1, 0.3, 0.3], kind="face")
        before = self.candidate.read_bytes()
        server = self.serve()
        port = server.server_address[1]
        request = urllib.request.Request(f"http://127.0.0.1:{port}/crop-clear/c_1", method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertGreaterEqual(data["cleared"], 1)
        self.assertEqual(review.read_crop_store(self.project).get("c_1"), None)
        self.assertEqual(self.candidate.read_bytes(), before)


class BoxParsingTests(unittest.TestCase):
    def test_per_image_boxes_parse(self):
        boxes = review.parse_crop_boxes("s_1:0.1,0.1,0.2,0.2;s_2:0.5,0.5,0.3,0.3")
        self.assertEqual(boxes["s_1"], [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(boxes["s_2"], [0.5, 0.5, 0.3, 0.3])

    def test_an_empty_string_means_no_per_image_boxes(self):
        self.assertEqual(review.parse_crop_boxes(""), {})

    def test_a_malformed_entry_is_refused_rather_than_skipped(self):
        # Skipping it would score that image with the candidate's box and make the comparison look complete.
        with self.assertRaises(review.pj.ProjectError):
            review.parse_crop_boxes("s_1:0.1,0.1,0.2,0.2;nonsense")

    def test_a_box_outside_the_image_fractions_is_refused(self):
        with self.assertRaises(review.pj.ProjectError):
            review.parse_crop_boxes("s_1:1.2,0.1,0.2,0.2")


class _NullClaim:
    def acquire(self):
        return self

    def release(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    unittest.main()
