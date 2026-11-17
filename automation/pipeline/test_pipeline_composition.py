"""The composition plan: half body / half face, spread views, poses and expressions, the originals counted first.

  venv\\Scripts\\python.exe -B -m unittest pipeline.test_pipeline_composition      (from automation\\)
"""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pipeline import batch_gen, composition, project as pj, settings


def original(shot, view="front view", pose="standing", expression="neutral"):
    half = "body" if shot in composition.SHOTS["body"] else "face"
    return {"half": half, "shot": shot, "view": view, "pose": pose if half == "body" else "",
            "expression": expression}


# Close to the real character: almost every original is a body shot, several of them from behind.
ORIGINALS = [original("upper body", expression="smiling"), original("cowboy shot", "back view", "sitting"),
             original("upper body", expression="smiling"), original("upper body", "side profile"),
             original("full body", pose="dynamic pose", expression="sad"), original("full body", "back view"),
             original("full body", "side profile", "dynamic pose"), original("upper body", "back view"),
             original("portrait", "three-quarter view"), original("full body"),
             original("full body", "back view"), original("cowboy shot", "three-quarter view", expression="smiling"),
             original("upper body")]


class PlanCellsTests(unittest.TestCase):
    def setUp(self):
        self.cells = composition.plan_cells(ORIGINALS, 50)
        self.every = ORIGINALS + self.cells

    def test_the_originals_plus_the_cells_make_the_target_half_and_half(self):
        self.assertEqual(len(self.cells), 50 - len(ORIGINALS))
        self.assertEqual(Counter(c["half"] for c in self.every), {"body": 25, "face": 25})

    def test_full_body_leads_the_body_half_and_the_face_half_is_even(self):
        shots = Counter(c["shot"] for c in self.every)
        # 25 body pictures split 2:1:1, the 25 face pictures 1:1:1.
        self.assertEqual([shots[s] for s in composition.SHOTS["body"]], [13, 6, 6], shots)
        self.assertTrue(all(8 <= shots[s] <= 9 for s in composition.SHOTS["face"]), shots)
        self.assertEqual(set(Counter(c["expression"] for c in self.every).values()), {10})

    def test_every_view_is_used_and_side_profiles_are_not_scarce(self):
        views = Counter(c["view"] for c in self.every)
        self.assertEqual(set(views), set(composition.VIEWS))
        self.assertEqual(views.most_common(1)[0][0], "front view")
        # Side profiles get about as much room as the three-quarter views: at least a quarter of the set.
        self.assertGreaterEqual(views["side profile"], 50 // 4, views)
        self.assertGreaterEqual(views["side profile"], views["three-quarter view"] - 2, views)

    def test_full_body_and_side_cells_say_it_plainly(self):
        full = next(c for c in self.cells if c["shot"] == "full body")
        self.assertIn("head to toe", full["tags"])
        side = next(c for c in self.cells if c["view"] == "side profile")
        self.assertIn("facing sideways", side["tags"])
        self.assertNotIn("looking at viewer,", side["tags"].replace("not looking at viewer", ""))

    def test_faces_are_never_lost_to_the_camera(self):
        for cell in self.cells:
            if cell["shot"] == "close-up":
                self.assertNotEqual(cell["view"], "back view", cell)
            if cell["shot"] == "portrait":
                self.assertNotEqual(cell["view"], "back view", cell)
            if cell["half"] == "body" and cell["view"] == "back view":
                self.assertEqual(cell["expression"], "neutral", cell)
                self.assertNotIn("frown", cell["tags"])

    def test_poses_belong_to_body_shots_only(self):
        self.assertTrue(all(c["pose"] for c in self.cells if c["half"] == "body"))
        self.assertTrue(all(not c["pose"] for c in self.cells if c["half"] == "face"))
        poses = Counter(c["pose"] for c in self.every if c["half"] == "body")
        self.assertEqual(set(poses), set(composition.POSES))

    def test_no_combination_is_drawn_twice(self):
        combos = Counter((c["shot"], c["view"], c["pose"], c["expression"]) for c in self.cells)
        self.assertEqual(max(combos.values()), 1, combos.most_common(3))

    def test_slot_ids_are_unique_readable_and_stable(self):
        ids = [c["slot_id"] for c in self.cells]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(i.startswith("cmp_") and composition.is_cell(i) for i in ids))
        self.assertEqual(ids, [c["slot_id"] for c in composition.plan_cells(ORIGINALS, 50)])
        self.assertEqual(batch_gen.category_of(ids[0]), "composition")

    def test_a_smaller_target_still_splits_in_half(self):
        cells = composition.plan_cells([], 10)
        self.assertEqual(Counter(c["half"] for c in cells), {"body": 5, "face": 5})


class ClassifyTests(unittest.TestCase):
    def test_tool2_words_map_to_the_axes(self):
        self.assertEqual(composition.classify({"shot": "cowboy shot, sitting, from behind, looking back",
                                               "expr": "parted lips"}),
                         {"half": "body", "shot": "cowboy shot", "view": "back view", "pose": "sitting",
                          "expression": "neutral"})
        self.assertEqual(composition.classify({"shot": "portrait, close-up, three-quarter view", "expr": "smile"}),
                         {"half": "face", "shot": "portrait", "view": "three-quarter view", "pose": "",
                          "expression": "smiling"})
        self.assertEqual(composition.classify({"shot": "crouching, from side, hand on knee"})["pose"], "dynamic pose")


class ProjectPlanTests(unittest.TestCase):
    """A real project: originals from sources.jsonl, tags from the tool2 config the card names, holdout left out."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        jobs = root / "dataset_jobs.json"
        jobs.write_text(json.dumps({"characters": [{"folder": "x", "original_tags": {
            "a.png": {"shot": "full body, standing"}, "b.png": {"shot": "portrait", "expr": "smile"},
            "hold.png": {"shot": "close-up"}}}]}), encoding="utf-8")
        card = {"id": "p", "trigger": "p", "caption_base": "webtoon", "slot_plan": "composition",
                "slot_quotas": {"ang_front_medium": 1},
                "tool2": {"config": str(jobs), "character": "x"},
                "holdout": {"ids": ["s_hold"], "shas": ["h" * 64], "frozen_at": "x"}}
        self.folder = root / "p"
        self.project = pj.Project.create("p", card, folder=self.folder)
        rows = [{"id": "s_a", "origin": "user", "sha256": "a" * 64, "src_path": "C:/chars/x/a.png"},
                {"id": "s_b", "origin": "user", "sha256": "b" * 64, "src_path": "C:/chars/x/b.png"},
                {"id": "s_hold", "origin": "user", "sha256": "h" * 64, "src_path": "C:/chars/x/hold.png"},
                {"id": "s_syn", "origin": "synthetic", "sha256": "s" * 64, "src_path": "C:/chars/x/anchor.png"}]
        (self.folder / "sources.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def test_only_user_originals_outside_the_holdout_count(self):
        names = sorted(o["name"] for o in composition.originals(self.project))
        self.assertEqual(names, ["a.png", "b.png"])

    def test_the_plan_follows_the_projects_target(self):
        settings.write(self.project, {"target_images": 12})
        result = composition.plan(self.project)
        self.assertEqual(result["target"], 12)
        self.assertEqual(len(result["cells"]), 10)
        self.assertEqual(result["totals"]["half"], {"body": 6, "face": 6})

    def test_the_card_opts_in_and_the_slot_plan_follows(self):
        settings.write(self.project, {"target_images": 12})
        plan = self.project.slot_plan()
        self.assertEqual(len(plan), 10)
        self.assertTrue(all(composition.is_cell(slot) and target == 1 for slot, target in plan.items()))
        self.project.card["slot_plan"] = ""
        self.assertEqual(self.project.slot_plan(), {"ang_front_medium": 1})

    def test_a_target_below_the_originals_is_refused(self):
        with self.assertRaises(composition.CompositionError):
            composition.plan(self.project, target=1)

    def test_batch_b_plans_every_cell_with_its_own_words(self):
        settings.write(self.project, {"target_images": 12})
        snapshot = self.folder / "seed_lora" / "snap"
        snapshot.mkdir(parents=True)
        (snapshot / "snapshot.json").write_text(json.dumps({
            "id": "snap", "created_at": "2026-11-17T00:00:00", "base": str(snapshot),
            "inputs": [{"id": "s_a", "generation": 0}]}), encoding="utf-8")
        self.project.card["krea2_inference"] = {"variant": "a", "dit": "krea2_turbo_dit",
                                                "size": {"width": 1024, "height": 1536}}
        manifest = batch_gen.plan(self.project, seeds_per_slot=1, log=lambda *_: None)
        cells = {c["slot_id"]: c for c in composition.plan(self.project)["cells"]}
        self.assertEqual({t["slot_id"] for t in manifest["tasks"]}, set(cells))
        for task in manifest["tasks"]:
            self.assertTrue(task["prompt"].startswith("p, webtoon, "), task["prompt"])
            self.assertTrue(task["prompt"].endswith(cells[task["slot_id"]]["tags"]), task["prompt"])
            self.assertEqual(task["category"], "composition")


if __name__ == "__main__":
    unittest.main()
