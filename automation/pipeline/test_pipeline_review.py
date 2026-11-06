"""The review page's score display and its sorting controls.

What is tested here is what a reviewer relies on: the numbers appear next to the picture they belong to, a picture with
no score does not look like a picture with a score of zero, and the page still renders on a project that was never
scored. The sort itself is JavaScript and is not exercised here; what is tested is that every value it needs is on the
card as data.
"""
import tempfile
import unittest
from pathlib import Path

from pipeline import review


class FakeProject:
    def __init__(self, root, candidates, card=None, review_state=None):
        self.dir = Path(root)
        self._candidates = candidates
        self._review = review_state or {}
        self.card = card or {"id": "p", "trigger": "suamom", "caption_base": "style", "slot_quotas": {}}
        self.dir.mkdir(parents=True, exist_ok=True)

    def digest(self):
        return "digest"

    def candidates(self):
        return list(self._candidates)

    def source_records(self):
        return []

    def source_share(self):
        return 0.0, {"user": 0, "synthetic": 0}

    def review(self):
        return dict(self._review)

    def undecided(self):
        return []

    def index(self):
        return {c["id"]: c for c in self._candidates}

    def state_of(self, candidate_id):
        return self._review.get(candidate_id) or {}

    def resolve_file(self, name):
        return self.dir / name if name else None

    def quotas(self, planned):
        return {slot: {"target": int(t), "accepted": 0, "short": int(t)} for slot, t in planned.items()}


def write_png(path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (200, 200, 200)).save(path)
    return path


def candidate(ident, slot="frm_face"):
    return {"id": ident, "file": f"{ident}.png", "generator": "qwen_edit", "slot": {"id": slot},
            "train_generation": 1, "edit_depth": 0, "parent_ids": [], "caption": ""}


class RowScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_png(self.root / "c_1.png")
        self.project = FakeProject(self.root, [candidate("c_1")])

    def tearDown(self):
        self.temp.cleanup()

    def test_scores_reach_the_row(self):
        scores = {"c_1": {"clip_max_cosine": 0.91, "clip_mean_cosine": 0.8,
                          "arcface_max_cosine": 0.7, "arcface_no_face": "", "white_ratio": 0.95}}
        rows = review.candidate_rows(self.project, scores=scores)
        self.assertAlmostEqual(rows[0]["clip_max"], 0.91)
        self.assertAlmostEqual(rows[0]["arcface_max"], 0.7)
        self.assertAlmostEqual(rows[0]["white_ratio"], 0.95)

    def test_a_picture_with_no_face_has_no_face_score_and_says_why(self):
        scores = {"c_1": {"clip_max_cosine": 0.5, "arcface_max_cosine": None,
                          "arcface_no_face": "no face was detected"}}
        rows = review.candidate_rows(self.project, scores=scores)
        self.assertIsNone(rows[0]["arcface_max"])
        self.assertEqual(rows[0]["arcface_note"], "no face was detected")

    def test_an_unscored_picture_has_none_everywhere(self):
        rows = review.candidate_rows(self.project, scores={})
        self.assertIsNone(rows[0]["clip_max"])
        self.assertIsNone(rows[0]["arcface_max"])
        self.assertIsNone(rows[0]["white_ratio"])

    def test_the_two_scorers_are_never_blended_into_one_number(self):
        # They measured differently on real data (+0.34 between them), so the row keeps them apart.
        scores = {"c_1": {"clip_max_cosine": 0.9, "arcface_max_cosine": 0.2}}
        rows = review.candidate_rows(self.project, scores=scores)
        self.assertAlmostEqual(rows[0]["clip_max"], 0.9)
        self.assertAlmostEqual(rows[0]["arcface_max"], 0.2)
        self.assertNotIn("identity", rows[0])


class PageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_png(self.root / "c_1.png")
        write_png(self.root / "c_2.png")
        self.project = FakeProject(self.root, [candidate("c_1"), candidate("c_2", "pos_sitting")])

    def tearDown(self):
        self.temp.cleanup()

    def render(self, scores=None):
        out = self.root / "review.html"
        review.write_page(self.project, out, log=lambda *_: None, scores=scores or {})
        return out.read_text(encoding="utf-8")

    def test_the_page_renders_without_any_scores(self):
        html = self.render({})
        self.assertIn('id="controls"', html)
        self.assertIn("점수 없음", html)

    def test_every_card_carries_the_values_the_sort_reads(self):
        html = self.render({"c_1": {"clip_max_cosine": 0.9, "arcface_max_cosine": 0.5, "white_ratio": 0.8}})
        self.assertIn('data-clip="0.9"', html)
        self.assertIn('data-face="0.5"', html)
        self.assertIn('data-white="0.8"', html)

    def test_a_card_with_no_score_has_an_empty_sort_value_not_a_zero(self):
        html = self.render({"c_1": {"clip_max_cosine": 0.9}})
        start = html.index('data-id="c_2"')
        card = html[start:html.index('data-keys=', start)]
        self.assertIn('data-clip=""', card)
        self.assertIn('data-face=""', card)

    def test_the_slot_filter_offers_the_slots_that_are_present(self):
        html = self.render({})
        self.assertIn('<option value="frm_face">', html)
        self.assertIn('<option value="pos_sitting">', html)

    def test_the_page_shows_a_score_next_to_the_picture(self):
        html = self.render({"c_1": {"clip_max_cosine": 0.912, "arcface_max_cosine": 0.734, "white_ratio": 0.88}})
        self.assertIn("CLIP 0.912", html)
        self.assertIn("face 0.734", html)
        self.assertIn("흰배경 0.88", html)

    def test_a_missing_face_is_shown_as_a_dash_not_a_zero(self):
        html = self.render({"c_1": {"clip_max_cosine": 0.5, "arcface_no_face": "no face was detected"}})
        self.assertIn("face —", html)

    def test_no_template_placeholder_is_left_in_the_output(self):
        html = self.render({})
        for leftover in ("{cards}", "{quotas}", "{slot_options}", "{summary}", "{keys}"):
            self.assertNotIn(leftover, html)


class DuplicateGroupTests(unittest.TestCase):
    """R7 P2: near-duplicates are grouped with the scorer's own pHash and shown, never removed."""

    def setUp(self):
        import tempfile
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = Image.new("RGB", (64, 64), (120, 120, 120))
        for x in range(0, 64, 8):
            for y in range(0, 64, 8):
                base.putpixel((x, y), (10, 10, 10))
        base.save(self.root / "twin_a.png")
        # A redraw with slightly different lighting: same structure, a few values moved - one pHash group.
        twin = base.copy()
        for x in range(64):
            for y in range(64):
                r, g, b = twin.getpixel((x, y))
                twin.putpixel((x, y), (min(255, r + 6), g, b))
        twin.save(self.root / "twin_b.png")
        # Strongly different structure, so its difference-hash sits far from the flat twins: left half dark,
        # right half light, with a hard vertical edge.
        different = Image.new("RGB", (64, 64), (10, 10, 10))
        for x in range(32, 64):
            for y in range(64):
                different.putpixel((x, y), (250, 250, 250))
        different.save(self.root / "different.png")
        candidates = [
            {"id": "c_t1", "file": "twin_a.png", "generator": "qwen_edit", "slot": {"id": "frm_face"},
             "train_generation": 1, "edit_depth": 0, "parent_ids": [], "caption": ""},
            {"id": "c_t2", "file": "twin_b.png", "generator": "qwen_edit", "slot": {"id": "frm_face"},
             "train_generation": 1, "edit_depth": 0, "parent_ids": [], "caption": ""},
            {"id": "c_d1", "file": "different.png", "generator": "qwen_edit", "slot": {"id": "frm_back"},
             "train_generation": 1, "edit_depth": 0, "parent_ids": [], "caption": ""},
        ]
        self.project = FakeProject(self.root, candidates)

    def tearDown(self):
        self.temp.cleanup()

    def page(self):
        return review.write_page(self.project, self.root / "review.html", log=lambda *_: None).read_text(
            encoding="utf-8")

    def test_the_twins_share_a_group_and_the_odd_one_out_has_none(self):
        import re
        page = self.page()
        twins = re.findall(r'<div class="card" data-id="(c_t[12])"[^>]*data-dup="(\d+)"', page)
        self.assertEqual({tid for tid, _group in twins}, {"c_t1", "c_t2"})
        odd = re.search(r'<div class="card" data-id="c_d1"[^>]*data-dup="([^"]*)"', page)
        self.assertIsNotNone(odd)
        self.assertEqual(odd.group(1), "", "not being a near-duplicate is the normal case, not group 0")

    def test_the_group_is_named_on_the_card(self):
        self.assertIn("근접 중복 G1 (2장)", self.page())

    def test_the_summary_names_the_grouping(self):
        self.assertIn("근접 중복 묶음 1", self.page())

    def test_the_filter_checkbox_is_on_the_page(self):
        page = self.page()
        self.assertIn('id="dup"', page)
        self.assertIn("근접 중복만", page)


class CropScoreTests(unittest.TestCase):
    """R7 P2: a user-typed crop box re-scores the picture against the same crop of every original."""

    def setUp(self):
        import tempfile
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (64, 64), (200, 200, 200)).save(self.root / "a.png")
        Image.new("RGB", (64, 64), (100, 100, 100)).save(self.root / "b.png")

        class Project(FakeProject):
            def __init__(self, root, candidates):
                super().__init__(root, candidates)
                self._sources = [{"id": "s_1", "file": "a.png", "origin": "user", "sha256": "sa"},
                                 {"id": "s_2", "file": "b.png", "origin": "user", "sha256": "sb"}]

            def source_records(self):
                return list(self._sources)

            def candidate_by_id(self, candidate_id):
                for record in self.candidates():
                    if record["id"] == candidate_id:
                        return record
                return None

        self.candidate_file = self.root / "cand.png"
        Image.new("RGB", (64, 64), (210, 210, 210)).save(self.candidate_file)
        self.project = Project(self.root, [{"id": "c_1", "file": "cand.png", "generator": "qwen_edit",
                                            "slot": {"id": "frm_face"}, "train_generation": 1,
                                            "edit_depth": 0, "parent_ids": [], "caption": ""}])

    def tearDown(self):
        self.temp.cleanup()

    def test_a_box_with_the_wrong_shape_is_refused(self):
        for bad in ("1,2,3", "a,b,c,d", "0.5,0.5,-1,0.2", "0.5,0.5,0.2,0"):
            with self.assertRaises(review.pj.ProjectError):
                review.parse_crop_box(bad)

    def test_a_good_box_parses(self):
        self.assertEqual(review.parse_crop_box("0.1, 0.2, 0.3, 0.4"), [0.1, 0.2, 0.3, 0.4])

    def test_a_pixel_box_comes_from_the_image_size(self):
        self.assertEqual(review._pixel_box(self.root / "a.png", [0.25, 0.5, 0.5, 0.25]), [16, 32, 32, 16])

    def test_the_route_scores_the_crop_against_the_same_crop_of_the_originals(self):
        import json as json_mod
        import threading
        import urllib.request
        from PIL import Image

        class Vec:
            """Just enough of a tensor for the scorer's cosine: elementwise multiply and sum."""

            def __init__(self, values):
                self.values = values

            def __mul__(self, other):
                return Vec([a * b for a, b in zip(self.values, other.values)])

            def sum(self):
                return sum(self.values)

        class FakeEmbedder:
            """Brightness vectors, so the bright crop matches the bright original and nothing else."""

            @staticmethod
            def _vector_of(image):
                colours = [image.getpixel((x, y)) for x in (0, 32) for y in (0, 32)]
                brightness = sum(sum(c) / 3 for c in colours) / len(colours)
                return Vec([brightness / 255.0, 1.0 - brightness / 255.0])

            def embed(self, paths):
                out = {}
                for p in paths:
                    with Image.open(p) as image:
                        out[str(p)] = self._vector_of(image)
                return out

            def embed_crops(self, path, boxes):
                out = {}
                with Image.open(path) as image:
                    for name, (x, y, w, h) in boxes.items():
                        colours = [image.getpixel((px, py)) for px, py in
                                   [(x, y), (min(x + w, 63), min(y + h, 63))]]
                        brightness = sum(sum(c) / 3 for c in colours) / len(colours)
                        out[name] = Vec([brightness / 255.0, 1.0 - brightness / 255.0])
                return out

        server = review.make_server(self.project, port=0, quota_plan=None,
                                    embedder_factory=FakeEmbedder)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            url = f"http://127.0.0.1:{port}/crop-score/c_1?kind=face&box=0.1,0.1,0.6,0.6"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json_mod.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(data["id"], "c_1")
        self.assertEqual(data["kind"], "face")
        self.assertIsNotNone(data["max"])
        # The bright crop sits closer to the bright original than to the dark one: max above the mean over both.
        self.assertGreater(data["max"], 0.6)
        self.assertGreater(data["max"], data["mean"])
        self.assertEqual(data["reference_count"], 2)

    def test_the_route_refuses_a_bad_box_without_touching_the_embedder(self):
        import threading
        import urllib.error
        import urllib.request

        def no_embedder():
            raise AssertionError("a refused box must not build an embedder")

        server = review.make_server(self.project, port=0, quota_plan=None, embedder_factory=no_embedder)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            url = f"http://127.0.0.1:{port}/crop-score/c_1?kind=face&box=1,2,3"
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url, timeout=10)
            self.assertEqual(caught.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()


class KeyboardScopeTests(unittest.TestCase):
    """R1: the keyboard decides only what the reviewer can see.

    The page's judgement code is JavaScript and was exercised in a real browser against a throwaway project (the run
    is recorded in PROGRESS); what is pinned here are the invariants a future edit must not silently drop, read from
    the page the server actually renders.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        write_png(cls.root / "c_1.png")
        cls.project = FakeProject(cls.root, [candidate("c_1", "frm_face"), candidate("c_2", "frm_back")])
        cls.html = review.write_page(cls.project, cls.root / "review.html", log=lambda *_: None).read_text(
            encoding="utf-8")
        cls.script = cls.html.split("<script>")[1].split("</script>")[0]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def script_includes(self, *snippets):
        return all(snippet in self.script for snippet in snippets)

    def test_cards_reads_only_the_visible_cards(self):
        # A filter hides a card from the eye, so it must hide it from the keyboard too.
        self.assertTrue(self.script_includes(
            "function cards()",
            "filter(c => c.style.display !== 'none')"))

    def test_the_cursor_and_the_decision_keys_share_one_visible_list(self):
        self.assertTrue(self.script_includes(
            "const list = cards();",
            "const card = list[Math.min(cursor, list.length - 1)];"))

    def test_a_filter_control_that_owns_the_keyboard_silences_the_keys(self):
        self.assertTrue(self.script_includes(
            "document.getElementById('controls').contains(document.activeElement)"))

    def test_zero_visible_cards_means_no_decision(self):
        self.assertTrue(self.script_includes("else if (list.length) {"))

    def test_a_stored_decision_re_applies_the_view(self):
        # Under the undecided filter a decided card must leave the list the moment its decision lands.
        self.assertTrue(self.script_includes("applyView();\n}\nfunction setCursor"))

    def test_the_view_change_re_syncs_the_cursor(self):
        self.assertTrue(self.script_includes("syncCursor();\n}"))

    def test_clicking_a_card_syncs_the_cursor_to_it(self):
        self.assertTrue(self.script_includes(
            "const at = cards().indexOf(card);",
            "if (at >= 0) setCursor(at);"))

    def test_the_decision_write_keeps_the_target_element_alive(self):
        # The decision is written into its own span: overwriting the state bar's textContent would destroy the
        # cursor's target element and break the next re-sync (found in the browser run).
        self.assertTrue(self.script_includes("getElementById('last').textContent"))
        self.assertIn('<span id="target"></span><span id="last"></span>', self.html)


if __name__ == "__main__":
    unittest.main()
