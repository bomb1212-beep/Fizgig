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
            "expressions": {"steps": 25, "cfg": 1.0, "items": [{"text": "a smile", "tags": "smiling"}]},
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
        records, skipped, _report = gen.import_candidates(self.project, self.card, snap, self._result(snap),
                                                          log=lambda *_: None)
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
        records, _skipped, _report = gen.import_candidates(self.project, self.card, snap, self._result(snap), log=lambda *_: None)
        self.assertEqual(records[0]["parent_ids"], [])
        self.assertTrue(records[0]["orphan_reason"])

    def test_an_imported_row_carries_the_parameters_tool2_reported(self):
        snap = self.snapshot()
        made = Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"
        made.write_bytes(b"MADE")
        result = {"characters": [{"folder": "tester", "generated": [
            {"stem": "framing_00_abcdefabcdef_s1000", "file": str(made), "category": "framing",
             "seed": 1003, "prompt": "Zoom in on the woman.", "steps": 25, "cfg": 1.0, "resolution": 1024}]}]}
        records, _skipped, _report = gen.import_candidates(self.project, self.card, snap, result, log=lambda *_: None)
        self.assertEqual(records[0]["seed"], 1003)
        self.assertEqual(records[0]["prompt"], "Zoom in on the woman.")

    def test_a_result_row_whose_file_is_gone_is_reported_not_guessed(self):
        snap = self.snapshot()
        records, skipped, _report = gen.import_candidates(self.project, self.card, snap, self._result(snap),
                                                          log=lambda *_: None)
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


class OverrideTests(SnapshotTests):
    """A per-run setting lives in the snapshot, never in tool2's own config file."""

    def test_a_section_setting_is_parsed_from_a_command_line_value(self):
        parsed = gen.parse_overrides(["expressions.use_extra_refs=false", "backgrounds.one_image=true",
                                      "poses.steps=30", "angles.cfg=1.5", "framing.resolution=1536"])
        self.assertEqual(parsed["expressions"]["use_extra_refs"], False)
        self.assertEqual(parsed["backgrounds"]["one_image"], True)
        self.assertEqual(parsed["poses"]["steps"], 30)
        self.assertEqual(parsed["angles"]["cfg"], 1.5)
        self.assertEqual(parsed["framing"]["resolution"], 1536)

    def test_a_malformed_override_is_refused(self):
        for bad in ("expressions", "nokey=1", "=1"):
            with self.assertRaises(ValueError):
                gen.parse_overrides([bad])

    def test_the_override_lands_in_the_snapshots_own_config_only(self):
        before = self.config.read_text(encoding="utf-8")
        snap = gen.build_snapshot(self.project, self.card, paths=self.paths, log=lambda *_: None,
                                  sections={"expressions": {"one_image": True}})
        pinned = json.loads(Path(snap["config"]).read_text(encoding="utf-8"))
        self.assertTrue(pinned["expressions"]["one_image"])
        self.assertEqual({row["key"] for row in snap["section_overrides"]}, {"one_image"})
        self.assertEqual(self.config.read_text(encoding="utf-8"), before, "the tool's own config must not change")

    def test_an_unknown_section_is_refused(self):
        with self.assertRaises(ValueError):
            gen.build_snapshot(self.project, self.card, paths=self.paths, log=lambda *_: None,
                               sections={"no_such_section": {"one_image": True}})

    def test_a_different_setting_is_a_different_snapshot(self):
        plain = self.snapshot()
        with_flag = gen.build_snapshot(self.project, self.card, paths=self.paths, log=lambda *_: None,
                                       sections={"expressions": {"one_image": True}})
        self.assertNotEqual(plain["id"], with_flag["id"],
                            "the settings decide the pictures, so they have to decide the snapshot")


class HoldoutTests(SnapshotTests):
    """T1: generator A cannot see the holdout, by name or by bytes, and says so when a reference is missing."""

    def test_a_frozen_holdout_is_kept_out_without_the_caller_saying_so(self):
        card = json.loads(json.dumps(self.card))
        card["holdout"] = {"ids": ["s_testsource"], "files": ["front.png"], "shas": [], "reason": "frozen"}
        snap = gen.build_snapshot(self.project, card, paths=self.paths, log=lambda *_: None)
        names = {Path(r["src_path"]).name for r in snap["sources"]}
        self.assertNotIn("front.png", names)
        self.assertFalse((Path(snap["chars"]) / "front.png").exists())
        self.assertEqual(snap["excluded"], ["front.png"])
        self.assertIn("holdout", snap["excluded_detail"][0]["reason"])
        # the run's own settings record what the run could not see
        pinned = json.loads(Path(snap["config"]).read_text(encoding="utf-8"))
        self.assertEqual(pinned["characters"][0]["excluded_inputs"], ["front.png"])

    def test_a_holdout_under_another_name_is_also_left_out(self):
        import hashlib
        digest = hashlib.sha256(b"IMG-front.png").hexdigest()
        card = json.loads(json.dumps(self.card))
        card["holdout"] = {"ids": [], "files": [], "shas": [digest], "reason": "frozen by hash"}
        (self.chars / "renamed-copy.png").write_bytes(b"IMG-front.png")
        snap = gen.build_snapshot(self.project, card, paths=self.paths, log=lambda *_: None)
        names = {Path(r["src_path"]).name for r in snap["sources"]}
        self.assertNotIn("front.png", names)
        self.assertNotIn("renamed-copy.png", names)
        self.assertEqual(sorted(snap["excluded"]), ["front.png", "renamed-copy.png"])

    def test_an_excluded_reference_is_reported_at_plan_time(self):
        self.config.write_text(json.dumps({
            "comfy_url": "http://127.0.0.1:8190", "resolution": 1024, "base_seed": 1000, "seeds_per_job": 1,
            "models": {"unet": "u", "clip": "c", "vae": "v"},
            "characters": [{"folder": "tester", "trigger": "trg", "edit_bases": ["front.png", "side.png"]}],
        }, ensure_ascii=False), encoding="utf-8")
        card = json.loads(json.dumps(self.card))
        card["holdout"] = {"ids": [], "files": ["front.png"], "shas": [], "reason": "frozen"}
        lines = []
        snap = gen.build_snapshot(self.project, card, paths=self.paths, log=lines.append)
        self.assertEqual(snap["missing_references"], ["front.png"])
        self.assertTrue(any("needs these excluded inputs" in line for line in lines))

    def test_the_snapshot_entry_no_longer_names_the_excluded_file(self):
        """tool2 refuses to plan when a listed reference is absent, so the snapshot's own entry must not list it."""
        self.config.write_text(json.dumps({
            "comfy_url": "http://127.0.0.1:8190", "resolution": 1024, "base_seed": 1000, "seeds_per_job": 1,
            "models": {"unet": "u", "clip": "c", "vae": "v"},
            "characters": [{"folder": "tester", "trigger": "trg",
                            "edit_bases": ["front.png", "side.png"],
                            "figure_refs": ["front.png"],
                            "sheet_face": "front.png"}],
        }, ensure_ascii=False), encoding="utf-8")
        card = json.loads(json.dumps(self.card))
        card["holdout"] = {"ids": [], "files": ["front.png"], "shas": [], "reason": "frozen"}
        snap = gen.build_snapshot(self.project, card, paths=self.paths, log=lambda *_: None)
        entry = json.loads(Path(snap["config"]).read_text(encoding="utf-8"))["characters"][0]
        self.assertEqual(entry["edit_bases"], ["side.png"])
        self.assertNotIn("figure_refs", entry, "a reference key with nothing left is removed, not left empty")
        self.assertNotIn("sheet_face", entry)
        self.assertEqual({row["key"] for row in snap["dropped_references"]},
                         {"edit_bases", "figure_refs", "sheet_face"})
        # the real tool2 config is untouched
        self.assertIn("front.png", json.loads(self.config.read_text(encoding="utf-8"))["characters"][0]["edit_bases"])

    def test_the_snapshot_id_does_not_move_when_references_are_scrubbed(self):
        """Dropping a name from the entry must not split one run into two snapshot folders."""
        before = self.snapshot()
        card = json.loads(json.dumps(self.card))
        card["holdout"] = {"ids": [], "files": ["front.png"], "shas": [], "reason": "frozen"}
        after = gen.build_snapshot(self.project, card, paths=self.paths, log=lambda *_: None)
        self.assertNotEqual(before["id"], after["id"],
                            "the input set really did change here, so a new snapshot is correct")
        # same input set, different entry wording: the id follows the bytes, not the reference lists
        again = gen.build_snapshot(self.project, card, paths=self.paths, log=lambda *_: None)
        self.assertEqual(after["id"], again["id"])


class ProjectImportTests(SnapshotTests):
    """T4: importing into a real project keeps what the project already had, and can be resumed."""

    def setUp(self):
        super().setUp()
        from pipeline import project as pj
        self.pj = pj
        self.proj = pj.Project.create("tester", {"id": "tester", "tool2": dict(self.card["tool2"])},
                                      folder=self.dir / "real-project")
        # the user's own originals, imported before the generator ever runs
        self.proj.import_sources([self.chars / "front.png", self.chars / "side.png"], log=lambda *_: None)
        self.before = {r["id"]: r["src_path"] for r in self.proj.source_records()}

    def _result(self, snap):
        made = Path(snap["out"]) / "framing_00_abcdefabcdef_s1000.png"
        made.write_bytes(b"MADE-BY-TOOL2")
        (Path(snap["out"]) / "manifest.jsonl").write_text(json.dumps(
            {"file": made.name, "refs": ["front.png", "side.png"], "category": "framing"}) + "\n", encoding="utf-8")
        return {"graph_version": "fake@1", "characters": [{"folder": "tester", "generated": [
            {"stem": "framing_00_abcdefabcdef_s1000", "file": str(made), "category": "framing",
             "prompt": "Zoom in.", "seed": 1003, "steps": 25, "cfg": 1.0, "resolution": 1024}]}]}

    def test_the_projects_own_sources_survive_an_import(self):
        snap = self.snapshot()
        gen.import_candidates(self.proj.dir, self.card, snap, self._result(snap), log=lambda *_: None)
        after = {r["id"]: r["src_path"] for r in self.proj.source_records()}
        for source_id, path in self.before.items():
            self.assertEqual(after.get(source_id), path, "an existing source row must not move or vanish")
        self.assertGreaterEqual(len(after), len(self.before))

    def test_the_copy_of_the_run_lands_in_the_projects_own_candidate_folder(self):
        snap = self.snapshot()
        gen.import_candidates(self.proj.dir, self.card, snap, self._result(snap), log=lambda *_: None)
        records = self.proj.candidates()
        self.assertEqual(len(records), 1)
        stored = Path(records[0]["file"])
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.parent.parent.name, "qwen_edit")
        self.assertEqual(stored.parent, self.proj.dir / "candidates" / "qwen_edit" / snap["id"][:24])

    def test_the_candidate_records_the_snapshot_graph_refs_and_models(self):
        snap = self.snapshot()
        records, _skipped, report = gen.import_candidates(self.proj.dir, self.card, snap, self._result(snap),
                                                          log=lambda *_: None)
        record = records[0]
        self.assertEqual(record["snapshot"], snap["id"])
        self.assertEqual(report["snapshot"], snap["id"])
        self.assertEqual(record["refs"], ["front.png", "side.png"])
        self.assertTrue(record["prompt_id"].startswith(snap["id"]))
        self.assertTrue(record["graph_version"].startswith("tool2:"))
        self.assertEqual(record["steps"], 25)
        self.assertEqual(record["width"], 1024)
        self.assertEqual(record["models"]["family"], "qwen_image21")
        # the snapshot's config names models; a run must say which files they were, even when they are absent
        self.assertTrue(record["models"]["files"])
        stored = {r["id"]: r for r in self.proj.candidates()}[record["id"]]
        self.assertEqual(stored["snapshot"], snap["id"])

    def test_importing_the_same_result_twice_adds_no_rows(self):
        snap = self.snapshot()
        result = self._result(snap)
        gen.import_candidates(self.proj.dir, self.card, snap, result, log=lambda *_: None)
        _records, skipped, report = gen.import_candidates(self.proj.dir, self.card, snap, result,
                                                          log=lambda *_: None)
        self.assertEqual(report["candidates"], 0)
        self.assertEqual(len(self.proj.candidates()), 1)
        self.assertTrue(skipped)
        self.assertIn("already in the project", skipped[0]["reason"])

    def test_an_import_can_find_its_own_snapshot_again_without_generating(self):
        snap = self.snapshot()
        found = gen.find_snapshot(self.proj.dir, self.card, snapshot_id=snap["id"], paths=self.paths,
                                  log=lambda *_: None)
        self.assertEqual(found["id"], snap["id"])
        # and by character alone, when the exact id is not given
        self.assertEqual(gen.find_snapshot(self.proj.dir, self.card, paths=self.paths, log=lambda *_: None)["id"],
                         snap["id"])

    def test_import_only_leaves_the_project_alone_when_the_result_is_already_in(self):
        snap = self.snapshot()
        result = self._result(snap)
        result_path = self.dir / "result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        gen.import_candidates(self.proj.dir, self.card, snap, result, log=lambda *_: None)
        before = len(self.proj.candidates())
        code = gen.main(["--project", str(self.proj.dir), "--import-only", str(result_path)])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.proj.candidates()), before)
        self.assertEqual(len(self.proj.candidates()), 1)


class SlotRestoreTests(unittest.TestCase):
    """T4: the M0 harness lost the slot id; the manifest still has it, keyed by the image it produced."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        from pipeline import project as pj
        self.pj = pj
        self.project = pj.Project.create("demo", {"id": "demo"}, folder=self.dir / "project")
        self.exp = self.dir / "m0"
        images = self.exp / "images" / "qwen_edit"
        images.mkdir(parents=True)
        (images / "ang_front_close__s1003.png").write_bytes(b"A-CLOSE")
        (images / "pos_sitting__s1003.png").write_bytes(b"POSED")
        (self.exp / "experiment.json").write_text(json.dumps({"tasks": [
            {"slot_id": "ang_front_close", "category": "angles", "file": "images/qwen_edit/ang_front_close__s1003.png"},
            {"slot_id": "pos_sitting", "category": "poses", "file": "images/qwen_edit/pos_sitting__s1003.png"},
        ]}), encoding="utf-8")
        from pipeline import lineage
        self.lineage = lineage
        rows = []
        for name in ("ang_front_close__s1003.png", "pos_sitting__s1003.png"):
            path = images / name
            rows.append(lineage.record(path, generator="qwen_edit", parent_ids=["s_1"], train_generation=1,
                                       edit_depth=1, slot={"category": name.split("_")[0]},
                                       method="qwen_edit"))
        (self.exp / "candidates.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        for row in rows:
            self.project.import_candidates([row], log=lambda *_: None)

    def tearDown(self):
        self.temp.cleanup()

    def test_the_slot_ids_come_back_from_the_manifest(self):
        from pipeline import gen_qwen_edit as gen
        report = gen.restore_slots(self.project, self.exp, log=lambda *_: None)
        self.assertEqual(len(report["restored"]), 2)
        slots = {r["id"]: r["slot"]["slot_id"] for r in self.project.candidates()}
        self.assertEqual(sorted(slots.values()), ["ang_front_close", "pos_sitting"])
        self.assertTrue(all("restored from" in r["slot"]["slot_source"] for r in self.project.candidates()))

    def test_the_quota_board_counts_the_restored_slots(self):
        from pipeline import gen_qwen_edit as gen
        plan = {"ang_front_close": 1, "pos_sitting": 1}
        before = self.project.quotas(plan)
        self.assertEqual(sum(info["accepted"] for info in before.values()), 0)
        for record in self.project.candidates():
            self.project.decide(record["id"], "accept")
        gen.restore_slots(self.project, self.exp, log=lambda *_: None)
        after = self.project.quotas(plan)
        self.assertEqual(after["ang_front_close"]["accepted"], 1)
        self.assertEqual(after["pos_sitting"]["accepted"], 1)
        self.assertEqual(sum(info["short"] for info in after.values()), 0)

    def test_an_image_that_fits_two_slots_is_left_unresolved(self):
        from pipeline import gen_qwen_edit as gen
        # the manifest says the same bytes were produced for two different slots: it cannot say which one a candidate
        # with those bytes is, so it is reported instead of guessed
        rows = self.lineage.read_jsonl(self.exp / "candidates.jsonl")
        rows[1]["sha256"] = rows[0]["sha256"]
        (self.exp / "candidates.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        report = gen.restore_slots(self.project, self.exp, log=lambda *_: None)
        self.assertTrue(report["ambiguous"])
        self.assertTrue(report["unresolved"])
        for record in self.project.candidates():
            self.assertFalse(record["slot"].get("slot_id"))


class SlotAssignTests(unittest.TestCase):
    """T4: a new run's images must be counted by the same slot vocabulary the project already uses."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        from pipeline import project as pj
        self.project = pj.Project.create("demo", {"id": "demo"}, folder=self.dir / "project")
        self.added = []

    def tearDown(self):
        self.temp.cleanup()

    def candidate(self, name, prompt, category="expressions"):
        from pipeline import lineage
        path = self.dir / name
        path.write_bytes(("IMAGE-" + name).encode())
        record = lineage.record(path, generator="qwen_edit", parent_ids=["s_1"], train_generation=1, edit_depth=1,
                                slot={"category": category}, prompt=prompt, stem=Path(name).stem)
        self.project.import_candidates([record], log=lambda *_: None)
        self.added.append(record["id"])
        return record["id"]

    def test_a_smile_and_a_classroom_get_their_slots(self):
        from pipeline import gen_qwen_edit as gen
        # tool2's own wording: "smiling", "classroom" (the slot table has to recognise both)
        smile = self.candidate("expressions_00.png", "suamom, korean webtoon style, smiling, closed eyes, happy")
        room = self.candidate("backgrounds_00.png", "suamom, korean webtoon style, classroom, indoors, day",
                              category="backgrounds")
        report = gen.assign_slots(self.project, log=lambda *_: None)
        slots = {row["id"]: row["slot_id"] for row in report["assigned"]}
        self.assertEqual(slots[smile], "exp_smile")
        self.assertEqual(slots[room], "bg_classroom")
        stored = {r["id"]: r["slot"]["slot_id"] for r in self.project.candidates()}
        self.assertEqual(stored[smile], "exp_smile")

    def test_an_image_that_fits_no_slot_is_left_without_one(self):
        from pipeline import gen_qwen_edit as gen
        vague = self.candidate("expressions_99.png", "suamom, korean webtoon style")
        report = gen.assign_slots(self.project, log=lambda *_: None)
        self.assertEqual([row["id"] for row in report["unmatched"]], [vague])
        self.assertNotIn("slot_id", self.project.candidate_by_id(vague)["slot"])

    def test_a_frame_word_does_not_move_a_background_into_an_angle_slot(self):
        """A background edit keeps the original frame, so its caption says "from above, upper body"."""
        from pipeline import gen_qwen_edit as gen
        rooftop = self.candidate(
            "backgrounds_04.png",
            "Replace the background with a school rooftop. suamom, korean webtoon style, rooftop, sunset, upper body, "
            "from above, leaning forward, looking at viewer", category="backgrounds")
        room = self.candidate(
            "backgrounds_00.png",
            "Replace the background with a classroom. suamom, korean webtoon style, classroom, upper body, from above",
            category="backgrounds")
        report = gen.assign_slots(self.project, log=lambda *_: None)
        slots = {row["id"]: row["slot_id"] for row in report["assigned"]}
        self.assertEqual(slots[room], "bg_classroom")
        self.assertEqual([row["id"] for row in report["unmatched"]], [rooftop],
                         "a rooftop has no slot in the plan; it must not be filed as an angle")

    def test_two_images_for_one_slot_are_reported_not_resolved(self):
        from pipeline import gen_qwen_edit as gen
        first = self.candidate("expressions_00.png", "suamom, korean webtoon style, smiling")
        second = self.candidate("expressions_01.png", "suamom, korean webtoon style, smiling, closed eyes")
        report = gen.assign_slots(self.project, log=lambda *_: None)
        self.assertEqual(len(report["assigned"]), 2)
        self.assertEqual(report["shared"], {"exp_smile": 2})
        self.assertEqual({self.project.candidate_by_id(first)["slot"]["slot_id"],
                          self.project.candidate_by_id(second)["slot"]["slot_id"]}, {"exp_smile"})

    def test_a_candidate_that_already_has_a_slot_is_left_alone(self):
        from pipeline import gen_qwen_edit as gen
        self.candidate("expressions_00.png", "suamom, korean webtoon style, smiling")
        gen.assign_slots(self.project, log=lambda *_: None)
        record = self.project.candidates()[0]
        record["slot"]["slot_source"] = "kept"
        from pipeline import lineage
        lineage.write_jsonl(self.project.dir / "candidates.jsonl", [record])
        report = gen.assign_slots(self.project, log=lambda *_: None)
        self.assertEqual(report["assigned"], [])
        self.assertEqual(self.project.candidates()[0]["slot"]["slot_source"], "kept")


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