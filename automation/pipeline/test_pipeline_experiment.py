"""Offline checks for the M0 harness: slot matching, the resumable run, the sheet and the report."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import experiment as exp  # noqa: E402

CARD = {"id": "suamom", "trigger": "suamom", "subject": "the woman", "caption_base": "korean webtoon style",
        "fixed_traits": [], "outfits": {"default": ""},
        "seed_lora": {"name": "suamom_v1", "epochs": [28]},
        "lineage": {"max_train_generation": 2, "seed_max_generation": 1, "max_edit_depth": 1}}


class FakeClient:
    """Stands in for Comfy: writes a placeholder PNG where the real one would land."""

    def __init__(self):
        self.calls = []

    def name_for(self, kind, path):
        return Path(path).name

    def run(self, graph, out_path=None, **meta):
        self.calls.append({"out": out_path, "meta": meta, "nodes": len(graph)})
        Path(out_path).write_bytes(b"PNG-" + Path(out_path).name.encode())
        return b"PNG"


def make_tool2(folder, entries):
    folder.mkdir(parents=True, exist_ok=True)
    for name, caption in entries.items():
        (folder / name).write_bytes(b"TOOL2-" + name.encode())
        (folder / f"{Path(name).stem}.txt").write_text(caption, encoding="utf-8")
    return folder


def make_loras(folder, epochs):
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (folder / f"LoraName_TokenName_krea2-{epoch:06d}.safetensors").write_bytes(b"LORA" + str(epoch).encode())
    return folder


class SlotTests(unittest.TestCase):
    def test_the_fixed_set_has_twenty_four_unique_slots(self):
        ids = [s["id"] for s in exp.SLOTS]
        self.assertEqual(len(ids), 24)
        self.assertEqual(len(set(ids)), 24)

    def test_every_category_tool2_uses_is_covered(self):
        categories = {s["category"] for s in exp.SLOTS}
        self.assertEqual(categories, {"angles", "framing", "expressions", "backgrounds", "poses"})

    def test_prompt_follows_the_documented_order(self):
        card = dict(CARD, fixed_traits=["long dark hair"], outfits={"default": "white blouse"})
        prompt = exp.prompt_for(card, {"prompt": "smile"})
        self.assertEqual(prompt, "suamom, korean webtoon style, long dark hair, white blouse, smile")

    def test_limit_takes_the_first_slots(self):
        self.assertEqual([s["id"] for s in exp.slots(2)], ["ang_front_close", "ang_front_medium"])


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.captions = {
            "angles_00_x": "suamom, upper body, front view, looking at viewer",
            "backgrounds_00_x": "suamom, upper body, front view, classroom background",
            "expressions_00_x": "suamom, upper body, front view, smile",
        }

    def test_a_same_category_image_wins_a_tie(self):
        slot = {"id": "exp_smile", "category": "expressions", "match": [["smile"]]}
        name, _caption = exp.match_slot(slot, self.captions, set())
        self.assertEqual(name, "expressions_00_x")

    def test_an_unused_candidate_is_preferred(self):
        slot = {"id": "s", "category": "angles", "match": [["front view"]]}
        name, _caption = exp.match_slot(slot, self.captions, {"angles_00_x"})
        self.assertNotEqual(name, "angles_00_x")

    def test_a_source_image_is_never_borrowed_for_a_second_slot(self):
        captions = {"angles_00_x": "suamom, upper body, front view"}
        slot = {"id": "s", "category": "angles", "match": [["front view"]]}
        name, reason = exp.match_slot(slot, captions, {"angles_00_x"})
        self.assertIsNone(name)
        self.assertIn("no tool2 image", reason)

    def test_no_match_returns_none(self):
        name, _caption = exp.match_slot({"id": "d", "category": "framing", "match": [["dutch angle"]]},
                                        self.captions, set())
        self.assertIsNone(name)


class PlanRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.tool2 = make_tool2(self.dir / "tool2", {
            "angles_00_x.png": "suamom, upper body, front view, looking at viewer",
            "angles_01_x.png": "suamom, upper body, front view, hands on hips",
        })
        self.loras = make_loras(self.dir / "loras", [28, 62])
        self.exp = self.dir / "exp"

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, **kwargs):
        options = dict(epochs=[28], seeds=(1003,), slot_limit=2, tool2_folder=self.tool2, variants=("a",))
        options.update(kwargs)
        return exp.plan(self.exp, CARD, **options)

    def test_plan_counts_one_task_per_method_and_seed(self):
        manifest = self.plan()
        # 2 slots x (1 tool2 + 1 Krea2 configuration) x 1 seed
        self.assertEqual(len(manifest["tasks"]), 4)
        self.assertEqual(manifest["methods"], ["qwen_edit", "krea2_a_e028"])

    def test_plan_records_slots_tool2_cannot_fill(self):
        manifest = self.plan(epochs=[28], variants=("a",), slots=None) if False else self.plan()
        self.assertEqual(manifest["slots_without_tool2"], [])
        manifest = exp.plan(self.exp, CARD, epochs=[28], seeds=(1003,), slot_limit=1, tool2_folder=self.dir / "empty",
                            variants=("a",))
        self.assertEqual(len(manifest["slots_without_tool2"]), 1)
        self.assertTrue(any(t["status"] == "missing" for t in manifest["tasks"]))

    def test_planning_twice_with_the_same_arguments_is_stable(self):
        first = self.plan()
        second = self.plan()
        self.assertEqual([t["file"] for t in first["tasks"]], [t["file"] for t in second["tasks"]])

    def test_run_copies_and_generates_then_a_rerun_adds_nothing(self):
        manifest = self.plan()
        client = FakeClient()
        done, skipped, failed, missing = exp.run(self.exp, manifest, client=client,
                                                 names={"a_unet": "turbo", "b_unet": "raw", "speed_lora": "speed",
                                                        "clip": "te", "vae": "vae"},
                                                 lora_dir=self.loras, log=lambda *_: None)
        # per slot: one tool2 copy, one Krea2 image
        self.assertEqual((done, skipped, failed, missing), (4, 0, 0, 0))
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all((self.exp / t["file"]).is_file() for t in manifest["tasks"]))

        again = FakeClient()
        done, skipped, failed, missing = exp.run(self.exp, exp.read_manifest(self.exp), client=again,
                                                 names={}, lora_dir=self.loras, log=lambda *_: None)
        self.assertEqual((done, failed, missing), (0, 0, 0))
        self.assertEqual(skipped, 4)
        self.assertEqual(again.calls, [])

    def test_run_records_lineage_for_every_image(self):
        manifest = self.plan()
        exp.run(self.exp, manifest, client=FakeClient(),
                names={"a_unet": "turbo", "b_unet": "raw", "speed_lora": "speed", "clip": "te", "vae": "vae"},
                lora_dir=self.loras, log=lambda *_: None)
        records = [json.loads(line) for line in
                   (self.exp / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(records), 4)
        by_method = {r["method"]: r for r in records}
        self.assertEqual(by_method["qwen_edit"]["generator"], "qwen_edit")
        self.assertEqual(by_method["qwen_edit"]["edit_depth"], 1)
        self.assertEqual(by_method["krea2_a_e028"]["generator"], "krea2_lora")
        self.assertEqual(by_method["krea2_a_e028"]["models"]["loras"][0]["trained_on_snapshot"], "suamom_v1@28")

    def test_a_failing_task_does_not_stop_the_others(self):
        manifest = self.plan(epochs=[28, 62], variants=("a",), seeds=(1003,), slot_limit=1)

        class HalfBroken(FakeClient):
            def run(self, graph, out_path=None, **meta):
                if meta.get("method") == "krea2_a_e028":
                    raise RuntimeError("ComfyUI said no")
                return super().run(graph, out_path, **meta)

        done, skipped, failed, _missing = exp.run(self.exp, manifest, client=HalfBroken(),
                                                  names={"a_unet": "t", "b_unet": "r", "speed_lora": "s",
                                                         "clip": "c", "vae": "v"},
                                                  lora_dir=self.loras, log=lambda *_: None)
        self.assertEqual(failed, 1)
        self.assertEqual(done, 2)
        statuses = {t["method"]: t["status"] for t in manifest["tasks"] if t["kind"] == "comfy"}
        self.assertEqual(statuses["krea2_a_e028"], "failed")
        self.assertEqual(statuses["krea2_a_e062"], "done")

    def test_nearest_epoch_stands_in_for_a_missing_checkpoint(self):
        chosen = exp.ensure_character_lora("krea2_a_e030", 30, self.loras)
        self.assertTrue(chosen.stem.endswith("000028"))

    def test_an_exact_epoch_is_used_when_it_exists(self):
        chosen = exp.ensure_character_lora("krea2_a_e062", 62, self.loras)
        self.assertTrue(chosen.stem.endswith("000062"))


class ParentTests(unittest.TestCase):
    """tool2 records which originals an edit came from; that is the link that reaches generation 0."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.characters = self.dir / "characters"
        self.characters.mkdir()
        self.original = self.characters / "front.png"
        self.original.write_bytes(b"ORIGINAL-FRONT")
        self.dataset = self.dir / "dataset"
        self.dataset.mkdir()
        self.write_manifest({"file": "angles_00_x.png", "refs": ["front.png"]})

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, record):
        (self.dataset / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_the_recorded_reference_becomes_the_parent_id(self):
        from pipeline import lineage
        ids, reason = exp.tool2_parents(self.dataset, self.characters, "angles_00_x.png", log=lambda *_: None)
        self.assertEqual(ids, [lineage.source_id(lineage.sha256_file(self.original))])
        self.assertEqual(reason, "")

    def test_a_copied_original_records_its_source(self):
        self.write_manifest({"file": "originals_00.png", "source": "front.png"})
        ids, _reason = exp.tool2_parents(self.dataset, self.characters, "originals_00.png", log=lambda *_: None)
        self.assertEqual(len(ids), 1)

    def test_an_image_tool2_never_mentioned_is_reported_not_guessed(self):
        ids, reason = exp.tool2_parents(self.dataset, self.characters, "no_such_image.png", log=lambda *_: None)
        self.assertEqual(ids, [])
        self.assertIn("manifest", reason)

    def test_a_record_whose_original_vanished_is_reported(self):
        self.write_manifest({"file": "x.png", "refs": ["gone.png"]})
        ids, reason = exp.tool2_parents(self.dataset, self.characters, "x.png", log=lambda *_: None)
        self.assertEqual(ids, [])
        self.assertIn("exist", reason)


class ServerTests(unittest.TestCase):
    """The score form posts to this server, so it must serve the sheet and nothing outside it."""

    def setUp(self):
        import threading
        import time
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "sheet_root"
        (self.root / "images").mkdir(parents=True)
        (self.root / "sheet.html").write_text("<html>sheet</html>", encoding="utf-8")
        (self.root / "images" / "a.png").write_bytes(b"PNG")
        (Path(self.temp.name) / "outside.txt").write_text("secret", encoding="utf-8")
        sibling = Path(str(self.root) + "_sibling")
        sibling.mkdir()
        (sibling / "leak.txt").write_text("secret", encoding="utf-8")
        self.server = exp.make_server(self.root, port=0, log=lambda *_: None)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        time.sleep(0.3)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def get(self, path):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_it_serves_the_sheet_and_its_images(self):
        self.assertEqual(self.get("/sheet.html"), 200)
        self.assertEqual(self.get("/images/a.png"), 200)

    def test_it_refuses_anything_outside_the_experiment_folder(self):
        self.assertEqual(self.get("/%2e%2e/outside.txt"), 404)
        self.assertEqual(self.get("/%2e%2e/sheet_root_sibling/leak.txt"), 404)
        self.assertEqual(self.get("/nope.png"), 404)

    def test_a_posted_score_becomes_the_csv(self):
        import json as _json
        import urllib.request
        body = _json.dumps({"rows": [{"slot": "s", "method": "m", "identity": 5, "quality": 3}]}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/scores", body,
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
        rows = exp.read_scores(self.root)
        self.assertEqual(rows[0]["identity"], "5")
        self.assertEqual(rows[0]["method"], "m")


class SheetReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        tool2 = make_tool2(self.dir / "tool2", {
            "angles_00_x.png": "suamom, upper body, front view, looking at viewer",
            "angles_01_x.png": "suamom, upper body, front view, hands on hips",
        })
        self.loras = make_loras(self.dir / "loras", [28])
        self.exp = self.dir / "exp"
        manifest = exp.plan(self.exp, CARD, epochs=[28], seeds=(1003,), slot_limit=2, tool2_folder=tool2,
                            variants=("a", "b"))
        self.manifest = exp.run(self.exp, manifest, client=FakeClient(),
                                names={"a_unet": "turbo", "b_unet": "raw", "speed_lora": "speed",
                                       "clip": "te", "vae": "vae"},
                                lora_dir=self.loras, log=lambda *_: None) and manifest

    def tearDown(self):
        self.temp.cleanup()

    def test_the_sheet_has_a_cell_for_every_method_and_slot(self):
        path = exp.write_sheet(self.exp, self.manifest)
        page = path.read_text(encoding="utf-8")
        for slot in ("ang_front_close", "ang_front_medium"):
            self.assertIn(f'data-slot="{slot}"', page)
        for method in ("qwen_edit", "krea2_a_e028", "krea2_b_e028"):
            self.assertIn(f"<th>{method}</th>", page)
        self.assertIn("/scores", page)          # the save button posts to the server
        self.assertIn("CSV 저장", page)

    def test_scores_round_trip_through_the_csv(self):
        rows = [{"slot": "ang_front_close", "method": "krea2_a_e028", "seed": 1003, "task": "t",
                 "identity": 5, "instruction": 4, "quality": 4, "error": 0, "note": "good"}]
        path = exp.write_scores(self.exp, rows)
        with open(path, encoding="utf-8", newline="") as handle:
            loaded = list(csv.DictReader(handle))
        self.assertEqual(loaded[0]["identity"], "5")
        self.assertEqual(exp.read_scores(self.exp)[0]["note"], "good")

    def test_the_aggregate_weighs_identity_and_counts_errors(self):
        rows = [{"slot": "ang_front_close", "method": "a", "identity": 5, "instruction": 5, "quality": 5, "error": 0},
                {"slot": "ang_quarter_medium", "method": "a", "identity": 2, "instruction": 4, "quality": 4, "error": 1},
                {"slot": "ang_front_close", "method": "b", "identity": 3, "instruction": 3, "quality": 3, "error": 0}]
        summary = exp.aggregate(rows)
        self.assertEqual(summary["a"]["n"], 2)
        self.assertAlmostEqual(summary["a"]["error_rate"], 0.5)
        self.assertAlmostEqual(summary["b"]["weighted"], 3.0)
        self.assertGreater(summary["a"]["weighted"], summary["b"]["weighted"])

    def test_the_recommendation_names_the_best_method(self):
        rows = [{"slot": "ang_front_close", "method": "weak", "identity": 2, "instruction": 2, "quality": 2},
                {"slot": "ang_front_close", "method": "strong", "identity": 5, "instruction": 4, "quality": 5},
                {"slot": "ang_side_medium", "method": "strong", "identity": 4, "instruction": 4, "quality": 4}]
        best, reasons = exp.recommendation(exp.aggregate(rows))
        self.assertEqual(best, "strong")
        self.assertTrue(any("2위" in reason for reason in reasons))

    def test_no_scores_yet_means_no_recommendation(self):
        best, reasons = exp.recommendation(exp.aggregate([]))
        self.assertIsNone(best)
        self.assertIn("점수", reasons[0])

    def test_the_report_lists_every_method(self):
        exp.write_scores(self.exp, [{"slot": "ang_front_close", "method": "qwen_edit", "identity": 4,
                                     "instruction": 4, "quality": 4, "error": 0}])
        path, best, _reasons = exp.write_report(self.exp, self.manifest, log=lambda *_: None)
        text = path.read_text(encoding="utf-8")
        self.assertIn("| qwen_edit |", text)
        self.assertEqual(best, "qwen_edit")


if __name__ == "__main__":
    unittest.main()

