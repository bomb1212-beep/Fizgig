"""Offline checks for generator A's bridge to tool2: the snapshot, the run, and the import.

The subprocess is replaced by a small script that behaves like tool2's CLI (JSON Lines on stdout, a result file), so
these tests need no ComfyUI and no GPU.
"""
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

from pipeline import gen_qwen_edit as gen  # noqa: E402
from pipeline import lineage, models  # noqa: E402

FAKE_TOOL2 = '''\
import json, sys
from pathlib import Path
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1] if flag in args else None
out = Path(value("--out")) / value("--character")
result_arg = value("--result-json")
if "--dry-run" in args:
    print("   framing_00_abcdefabcdef_s1000 [front.png]: a plan line", flush=True)
    sys.exit(0)
print(json.dumps({"kind": "run", "character": value("--character"), "planned": 2}), flush=True)
print(json.dumps({"kind": "image", "stem": "framing_00_abc_s1000", "status": "done"}), flush=True)
print("a human line on stderr", file=sys.stderr, flush=True)
out.mkdir(parents=True, exist_ok=True)
made = out / "framing_00_abcdefabcdef_s1000.png"
made.write_bytes(b"MADE-BY-TOOL2")
Path(result_arg).write_text(json.dumps({"version": 1, "graph_version": "fake@1", "characters": [
    {"folder": value("--character"), "planned": 1, "done": 1, "generated": [
        {"stem": "framing_00_abcdefabcdef_s1000", "file": str(made), "category": "framing"}]}]}), encoding="utf-8")
print(json.dumps({"kind": "summary", "done": 1, "skipped": 0, "failed": 0}), flush=True)
'''


class OriginTests(unittest.TestCase):
    def test_a_name_the_card_lists_is_synthetic(self):
        origin, why = gen.source_origin("anchor_x.png", {"synthetic_sources": ["anchor_x.png"]})
        self.assertEqual(origin, "synthetic")
        self.assertIn("synthetic_sources", why)

    def test_a_candidate_style_name_is_reported_as_synthetic_with_its_reason(self):
        # The spec's own example: tool2's set_anchor writes this into the character folder.
        origin, why = gen.source_origin("anchor_refined_fullbody_d_00020__e7417d_s1001.png", {})
        self.assertEqual(origin, "synthetic")
        self.assertIn("prefix", why)

    def test_an_ordinary_original_stays_user(self):
        origin, why = gen.source_origin("file_00000000755c820986b9a86181c4345d.png", {})
        self.assertEqual(origin, "user")
        self.assertIn("assumed", why)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.chars = self.dir / "characters" / "tester"
        self.chars.mkdir(parents=True)
        for name in ("front.png", "side.png", "anchor_refined_fullbody_x_s1.png"):
            (self.chars / name).write_bytes(b"IMG-" + name.encode())
        self.batch = self.dir / "fake_tool2.py"
        self.batch.write_text(FAKE_TOOL2, encoding="utf-8")
        self.config = self.dir / "jobs.json"
        self.config.write_text(json.dumps({
            "comfy_url": "http://127.0.0.1:8190", "resolution": 1024, "base_seed": 1000, "seeds_per_job": 1,
            "models": {"unet": "u", "clip": "c", "vae": "v"},
            "characters": [{"folder": "tester", "trigger": "trg", "edit_bases": 1}],
        }, ensure_ascii=False), encoding="utf-8")
        self.card = {"id": "tester", "tool2": {"config": str(self.config), "character": "tester",
                                               "characters": str(self.chars)}}
        self.paths = {"python": sys.executable, "batch": str(self.batch)}
        self.project = self.dir / "project"

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self, **kwargs):
        return gen.build_snapshot(self.project, self.card, paths=self.paths, log=lambda *_: None, **kwargs)

    def test_the_snapshot_pins_the_config_inputs_and_graph_version(self):
        snap = self.snapshot()
        self.assertTrue(Path(snap["config"]).is_file())
        self.assertTrue(Path(snap["chars"]).is_dir())
        pinned = json.loads(Path(snap["config"]).read_text(encoding="utf-8"))
        self.assertEqual([c["folder"] for c in pinned["characters"]], ["tester"])
        self.assertEqual(len(snap["sources"]), 3)
        self.assertTrue(Path(snap["base"]).joinpath("snapshot.json").is_file())

    def test_the_inputs_are_copied_so_a_later_edit_cannot_change_this_run(self):
        snap = self.snapshot()
        copied = Path(snap["chars"]) / "front.png"
        self.assertEqual(copied.read_bytes(), b"IMG-front.png")
        (self.chars / "front.png").write_bytes(b"CHANGED LATER")
        self.assertEqual(copied.read_bytes(), b"IMG-front.png")

    def test_synthetic_inputs_are_marked_and_counted(self):
        snap = self.snapshot()
        by_name = {Path(r["src_path"]).name: r for r in snap["sources"]}
        self.assertEqual(by_name["anchor_refined_fullbody_x_s1.png"]["origin"], "synthetic")
        self.assertEqual(by_name["front.png"]["origin"], "user")
        self.assertEqual(len(snap["sources"]), len({r["id"] for r in snap["sources"]}))

    def test_the_same_inputs_give_the_same_snapshot_id(self):
        first, second = self.snapshot(), self.snapshot()
        self.assertEqual(first["id"], second["id"])

    def test_a_changed_input_changes_the_snapshot_id(self):
        first = self.snapshot()
        (self.chars / "front.png").write_bytes(b"DIFFERENT")
        second = self.snapshot()
        self.assertNotEqual(first["id"], second["id"])

    def test_excluding_a_holdout_keeps_it_out_of_the_snapshot(self):
        snap = self.snapshot(exclude=("front.png",))
        names = {Path(r["src_path"]).name for r in snap["sources"]}
        self.assertNotIn("front.png", names)
        self.assertFalse((Path(snap["chars"]) / "front.png").exists())
        self.assertEqual(snap["excluded"], ["front.png"])

    def test_a_card_without_inputs_is_refused(self):
        empty = self.dir / "empty"
        empty.mkdir()
        card = {"id": "tester", "tool2": {**self.card["tool2"], "characters": str(empty)}}
        with self.assertRaises(ValueError):
            gen.build_snapshot(self.project, card, paths=self.paths, log=lambda *_: None)

    def test_a_missing_tool2_cli_is_refused(self):
        with self.assertRaises(ValueError):
            gen.build_snapshot(self.project, self.card, paths={"python": sys.executable, "batch": "nope.py"},
                               log=lambda *_: None)


class RunTests(SnapshotTests):
    def test_the_run_forwards_events_and_returns_the_result(self):
        snap = self.snapshot()
        seen = []
        result = gen.run(snap, log=lambda *_: None, on_event=seen.append)
        self.assertEqual([e["kind"] for e in seen], ["run", "image", "summary"])
        self.assertEqual(seen[1]["stem"], "framing_00_abc_s1000")
        self.assertEqual(result["characters"][0]["done"], 1)

    def test_a_dry_run_reports_no_result(self):
        snap = self.snapshot()
        self.assertIsNone(gen.run(snap, dry_run=True, log=lambda *_: None))

    def test_a_failing_tool2_is_reported(self):
        snap = self.snapshot()
        Path(snap["tool2_batch"]).write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        with self.assertRaises(RuntimeError) as caught:
            gen.run(snap, log=lambda *_: None)
        self.assertIn("exited 3", str(caught.exception))


class ImportTests(SnapshotTests):
    def _result(self, snap):
        return {"characters": [{"folder": "tester", "generated": [
            {"stem": "framing_00_abcdefabcdef_s1000", "file": str(Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"),
             "category": "framing"}]}]}

    def test_a_candidate_carries_its_parents_and_generation(self):
        snap = self.snapshot()
        made = Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"
        made.write_bytes(b"MADE")
        (Path(snap["out"]) / "manifest.jsonl").write_text(json.dumps(
            {"file": made.name, "refs": ["front.png", "side.png"], "category": "framing"}) + "\n", encoding="utf-8")
        records, skipped = gen.import_candidates(self.project, self.card, snap, self._result(snap), log=lambda *_: None)
        self.assertEqual(skipped, [])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["generator"], "qwen_edit")
        self.assertEqual(record["train_generation"], 1)
        self.assertEqual(record["edit_depth"], 1)
        self.assertEqual(len(record["parent_ids"]), 2)
        self.assertEqual(record["slot"]["category"], "framing")
        self.assertTrue(record["graph_version"].startswith("tool2:"))
        # the parents must be the source ids of the originals, so a trace reaches generation 0
        source_ids = {r["id"] for r in snap["sources"]}
        self.assertTrue(set(record["parent_ids"]) <= source_ids)

    def test_the_sources_are_written_with_their_origin(self):
        snap = self.snapshot()
        made = Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"
        made.write_bytes(b"MADE")
        gen.import_candidates(self.project, self.card, snap, self._result(snap), log=lambda *_: None)
        written = {r["id"]: r for r in lineage.read_jsonl(self.project / "sources.jsonl")}
        self.assertEqual(len(written), 3)
        origins = {Path(r["src_path"]).name: r["origin"] for r in written.values()}
        self.assertEqual(origins["anchor_refined_fullbody_x_s1.png"], "synthetic")

    def test_a_parentless_image_is_recorded_with_the_reason(self):
        snap = self.snapshot()
        made = Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"
        made.write_bytes(b"MADE")
        records, _skipped = gen.import_candidates(self.project, self.card, snap, self._result(snap), log=lambda *_: None)
        self.assertEqual(records[0]["parent_ids"], [])
        self.assertTrue(records[0]["orphan_reason"])

    def test_an_imported_row_carries_the_parameters_tool2_reported(self):
        snap = self.snapshot()
        made = Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"
        made.write_bytes(b"MADE")
        result = {"characters": [{"folder": "tester", "generated": [
            {"stem": "framing_00_abcdefabcdef_s1000", "file": str(made), "category": "framing",
             "seed": 1003, "prompt": "Zoom in on the woman.", "steps": 25, "cfg": 1.0, "resolution": 1024}]}]}
        records, _skipped = gen.import_candidates(self.project, self.card, snap, result, log=lambda *_: None)
        self.assertEqual(records[0]["seed"], 1003)
        self.assertEqual(records[0]["prompt"], "Zoom in on the woman.")

    def test_a_result_row_whose_file_is_gone_is_reported_not_guessed(self):
        snap = self.snapshot()
        records, skipped = gen.import_candidates(self.project, self.card, snap, self._result(snap), log=lambda *_: None)
        self.assertEqual(records, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("not on disk", skipped[0]["reason"])

    def test_importing_twice_does_not_duplicate(self):
        snap = self.snapshot()
        made = Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"
        made.write_bytes(b"MADE")
        gen.import_candidates(self.project, self.card, snap, self._result(snap), log=lambda *_: None)
        gen.import_candidates(self.project, self.card, snap, self._result(snap), log=lambda *_: None)
        records = lineage.read_jsonl(self.project / "candidates.jsonl")
        self.assertEqual(len(records), 1)


class WiringTests(unittest.TestCase):
    def test_the_tool_paths_come_from_the_pipeline_config(self):
        paths = gen.tool_paths({"pipeline": {"tool2": {"python": "X:/python.exe", "batch": "X:/dataset_batch.py"}}})
        self.assertEqual(Path(paths["batch"]).name, "dataset_batch.py")
        self.assertTrue(Path(paths["batch"]).is_absolute())

    def test_the_real_tool2_graph_version_is_readable(self):
        """The snapshot records it, so it must survive being read from the tool's own file."""
        paths = gen.tool_paths()
        if not Path(paths.get("batch") or "").is_file():
            self.skipTest("tool2 is not installed at the configured path")
        version = gen.tool_graph_version(paths["batch"])
        self.assertFalse(version.startswith("unknown"), version)
        self.assertEqual(version, gen.tool_graph_version(paths["batch"]))


if __name__ == "__main__":
    unittest.main()