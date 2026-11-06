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
