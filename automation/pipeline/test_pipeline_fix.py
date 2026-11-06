"""M3-4: corrections. A `fix` decision is the only thing that authorises one, and the result is a new candidate."""
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

from pipeline import fix, lineage, project as pj, review  # noqa: E402

CARD = {"trigger": "trg", "caption_traits": "omit",
        "holdout": {"min_sources": 3, "count": 1, "pick": []},
        "lineage": {"max_train_generation": 2, "seed_max_generation": 1, "max_edit_depth": 1}}
TINY_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
            b"\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tool2_config = self.root / "jobs.json"
        self.tool2_config.write_text(json.dumps({"models": {}}), encoding="utf-8")
        card = dict(CARD, tool2={"config": str(self.tool2_config)})
        self.project = pj.Project.create("demo", card, folder=self.root / "demo")
        self.chars = self.root / "chars"
        self.chars.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def origins(self, count=3):
        paths = []
        for i in range(count):
            path = self.chars / f"o{i}.png"
            path.write_bytes(TINY_PNG + str(i).encode())
            paths.append(path)
        self.project.import_sources(paths, log=lambda *_: None)
        return {Path(r["src_path"]).name: r["id"] for r in self.project.source_records()}

    def candidate(self, name, *, parents, decision="fix", note="", reason=None, depth=1, generation=1):
        path = self.root / name
        path.write_bytes(TINY_PNG + name.encode())
        record = lineage.record(path, generator="qwen_edit", parent_ids=list(parents),
                                train_generation=generation, edit_depth=depth,
                                slot={"slot_id": "exp_smile", "category": "expressions"}, prompt="a smile")
        self.project.import_candidates([record], log=lambda *_: None)
        if decision:
            self.project.decide(record["id"], decision, note=note, reason=reason)
        return record["id"]


class PlanTests(Base):
    def test_only_a_fix_decision_is_planned(self):
        sources = self.origins()
        fixing = self.candidate("fixme.png", parents=[sources["o1.png"]])
        self.candidate("accepted.png", parents=[sources["o1.png"]], decision="accept")
        self.candidate("rejected.png", parents=[sources["o1.png"]], decision="reject")
        self.candidate("undecided.png", parents=[sources["o1.png"]], decision=None)
        data = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual([job["id"] for job in data["jobs"]], [fixing])

    def test_planning_runs_nothing(self):
        sources = self.origins()
        self.candidate("fixme.png", parents=[sources["o1.png"]])
        fix.plan(self.project, log=lambda *_: None)
        self.assertFalse((self.project.dir / "fix").exists(), "a plan must not create work folders")

    def test_a_correction_carries_the_reviewers_note_when_there_is_one(self):
        sources = self.origins()
        with_note = self.candidate("a.png", parents=[sources["o1.png"]], note="hand is broken")
        without = self.candidate("b.png", parents=[sources["o1.png"]])
        jobs = {job["id"]: job for job in fix.plan(self.project, log=lambda *_: None)["jobs"]}
        self.assertEqual(jobs[with_note]["prompt"], "hand is broken")
        self.assertEqual(jobs[with_note]["prompt_source"], "the reviewer's note")
        self.assertEqual(jobs[without]["prompt"], fix.DEFAULT_FIX_PROMPT)
        self.assertEqual(jobs[without]["prompt_source"], "the default correction prompt")

    def test_the_default_prompt_asks_for_one_frame(self):
        self.assertIn("one single frame", fix.DEFAULT_FIX_PROMPT)
        self.assertIn("side-by-side", fix.DEFAULT_FIX_PROMPT)

    def test_a_different_character_cannot_be_corrected(self):
        sources = self.origins()
        stranger = self.candidate("stranger.png", parents=[sources["o1.png"]], decision="reject",
                                  reason=review.FIX_BLOCKED)
        # the server refuses `fix` for it, but a hand-edited review.json could still say so
        self.project.decide(stranger, "fix", reason=review.FIX_BLOCKED)
        data = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual(data["jobs"], [])
        self.assertIn("another character", data["blocked"][0]["reason"])

    def test_a_missing_file_is_reported_not_planned(self):
        sources = self.origins()
        gone = self.candidate("gone.png", parents=[sources["o1.png"]])
        Path(self.project.candidate_by_id(gone)["file"]).unlink()
        data = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual(data["jobs"], [])
        self.assertIn("not on disk", data["blocked"][0]["reason"])

    def test_no_references_is_the_default_and_the_holdout_is_never_one(self):
        sources = self.origins()
        holdout, _why = self.project.holdout_ids()
        self.candidate("fixme.png", parents=[sources["o1.png"]])
        none = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual(none["jobs"][0]["refs"], [])
        with_refs = fix.plan(self.project, refs="originals", log=lambda *_: None)
        self.assertTrue(with_refs["jobs"][0]["refs"])
        self.assertNotIn(holdout[0], with_refs["jobs"][0]["ref_ids"])
        for ref in with_refs["jobs"][0]["refs"]:
            self.assertTrue(Path(ref).is_file())

    def test_the_same_set_of_corrections_is_the_same_batch(self):
        sources = self.origins()
        self.candidate("fixme.png", parents=[sources["o1.png"]])
        first = fix.plan(self.project, log=lambda *_: None)
        second = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual(first["batch"], second["batch"])
        self.assertEqual(first["folder"], second["folder"])

    def test_a_changed_note_is_a_new_batch(self):
        sources = self.origins()
        target = self.candidate("fixme.png", parents=[sources["o1.png"]])
        first = fix.plan(self.project, log=lambda *_: None)
        self.project.decide(target, "fix", note="different instruction")
        second = fix.plan(self.project, log=lambda *_: None)
        self.assertNotEqual(first["batch"], second["batch"])

    def test_an_image_that_already_has_a_correction_is_not_drawn_again(self):
        # A correction stays on the reviewer's `fix` decision, so the original is still `fix` after it was corrected.
        # Planning again must not redraw it: the reviewer asked for one correction, not for one per run of the stage.
        sources = self.origins()
        target = self.candidate("fixme.png", parents=[sources["o1.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        Path(data["jobs"][0]["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(data["jobs"][0]["out"]).write_bytes(TINY_PNG + b"CORRECTED")
        outcome = fix.import_corrections(
            self.project, data,
            [{"out": data["jobs"][0]["out"], "status": "done", "steps": 25, "cfg": 1.0, "resolution": 1024, "seed": 7}],
            log=lambda *_: None)
        second = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual(second["jobs"], [])
        self.assertEqual([row["id"] for row in second["already"]], [target])
        self.assertEqual(second["already"][0]["corrected_as"], [outcome["records"][0]["id"]])
        self.assertEqual(second["blocked"], [], "an already corrected image is not a failure")

    def test_only_the_image_just_marked_is_planned(self):
        # The real shape of the project: seven corrections already exist, and the reviewer marks an eighth image.
        sources = self.origins()
        done = self.candidate("done.png", parents=[sources["o1.png"]])
        first = fix.plan(self.project, log=lambda *_: None)
        Path(first["jobs"][0]["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(first["jobs"][0]["out"]).write_bytes(TINY_PNG + b"CORRECTED")
        fix.import_corrections(self.project, first,
                               [{"out": first["jobs"][0]["out"], "status": "done", "steps": 25, "cfg": 1.0,
                                 "resolution": 1024, "seed": 7}], log=lambda *_: None)
        fresh = self.candidate("fresh.png", parents=[sources["o2.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual([job["id"] for job in data["jobs"]], [fresh])
        self.assertEqual([row["id"] for row in data["already"]], [done])

    def test_a_named_image_can_be_corrected_again(self):
        sources = self.origins()
        target = self.candidate("fixme.png", parents=[sources["o1.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        Path(data["jobs"][0]["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(data["jobs"][0]["out"]).write_bytes(TINY_PNG + b"CORRECTED")
        fix.import_corrections(self.project, data,
                               [{"out": data["jobs"][0]["out"], "status": "done", "steps": 25, "cfg": 1.0,
                                 "resolution": 1024, "seed": 7}], log=lambda *_: None)
        again = fix.plan(self.project, redo=[target], log=lambda *_: None)
        self.assertEqual([job["id"] for job in again["jobs"]], [target])
        self.assertEqual(again["already"], [])

    def test_a_correction_of_a_correction_is_planned_only_when_named(self):
        # Correcting a corrected picture is how a second pass is asked for, so an undecided correction of a correction
        # is not re-corrected by accident: only an image whose own decision says `fix` is ever planned.
        sources = self.origins()
        target = self.candidate("fixme.png", parents=[sources["o1.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        Path(data["jobs"][0]["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(data["jobs"][0]["out"]).write_bytes(TINY_PNG + b"CORRECTED")
        outcome = fix.import_corrections(
            self.project, data,
            [{"out": data["jobs"][0]["out"], "status": "done", "steps": 25, "cfg": 1.0, "resolution": 1024, "seed": 7}],
            log=lambda *_: None)
        corrected = outcome["records"][0]["id"]
        self.assertEqual(fix.plan(self.project, log=lambda *_: None)["jobs"], [])
        self.project.decide(corrected, "fix")
        jobs = [job["id"] for job in fix.plan(self.project, log=lambda *_: None)["jobs"]]
        self.assertEqual(jobs, [corrected], "a person marking the correction itself is what asks for a second pass")


class ImportTests(Base):
    def _plan_with_out(self):
        sources = self.origins()
        target = self.candidate("fixme.png", parents=[sources["o1.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        out = Path(data["jobs"][0]["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(TINY_PNG + b"CORRECTED")
        results = [{"input": data["jobs"][0]["input"], "out": str(out), "status": "done", "steps": 25, "cfg": 1.0,
                    "resolution": 1024, "seed": 7}]
        return target, data, results

    def test_the_correction_is_a_new_candidate_one_edit_deeper(self):
        target, data, results = self._plan_with_out()
        before = len(self.project.candidates())
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(len(outcome["records"]), 1)
        self.assertEqual(len(self.project.candidates()), before + 1)
        corrected = self.project.candidate_by_id(outcome["records"][0]["id"])
        original = self.project.candidate_by_id(target)
        self.assertEqual(corrected["parent_ids"][0], target)
        self.assertEqual(corrected["edit_depth"], original["edit_depth"] + 1)
        self.assertEqual(corrected["train_generation"], original["train_generation"])
        self.assertEqual(corrected["fix_of"], target)
        self.assertEqual(corrected["slot"]["slot_id"], original["slot"]["slot_id"],
                         "a correction fills the same slot as the image it corrects")
        self.assertTrue(Path(corrected["file"]).is_file())
        self.assertEqual(Path(corrected["file"]).parent.parent.name, "qwen_edit")

    def test_the_original_is_not_touched(self):
        target, data, results = self._plan_with_out()
        before = self.project.candidate_by_id(target)
        stat = Path(before["file"]).stat()
        fix.import_corrections(self.project, data, results, log=lambda *_: None)
        after = self.project.candidate_by_id(target)
        self.assertEqual(after["file"], before["file"])
        self.assertEqual(Path(before["file"]).stat().st_size, stat.st_size)
        self.assertEqual((self.project.state_of(target) or {}).get("decision"), "fix",
                         "importing a correction does not decide anything about the original")

    def test_a_row_that_did_not_finish_is_not_imported(self):
        _target, data, _results = self._plan_with_out()
        results = [{"out": data["jobs"][0]["out"], "status": "failed", "error": "ComfyUI stopped"}]
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(outcome["records"], [])
        self.assertIn("ComfyUI stopped", outcome["skipped"][0]["reason"])

    def test_importing_the_same_correction_twice_adds_nothing(self):
        _target, data, results = self._plan_with_out()
        fix.import_corrections(self.project, data, results, log=lambda *_: None)
        count = len(self.project.candidates())
        fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(len(self.project.candidates()), count)

    def test_the_correction_is_traceable_to_the_original(self):
        target, data, results = self._plan_with_out()
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        visited, unknown = self.project.trace(outcome["records"][0]["id"])
        self.assertIn(target, visited)
        self.assertEqual(unknown, [])

    def test_the_correction_keeps_the_caption_of_the_picture_it_corrects(self):
        target, data, results = self._plan_with_out()
        from pipeline import lineage
        rows = [dict(r) for r in self.project.candidates()]
        for row in rows:
            if row["id"] == target:
                row["caption"] = "trg, korean webtoon style, smiling, classroom"
        lineage.write_jsonl(self.project.dir / "candidates.jsonl", rows)
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        corrected = outcome["records"][0]
        self.assertEqual(corrected["caption"], "trg, korean webtoon style, smiling, classroom")
        self.assertIn("corrects", corrected["caption_source"])

    def test_a_correction_with_its_own_caption_keeps_it(self):
        target, data, results = self._plan_with_out()
        Path(data["jobs"][0]["out"]).with_suffix(".txt").write_text("trg, its own caption\n", encoding="utf-8")
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(outcome["records"][0]["caption"], "trg, its own caption")
        self.assertEqual(outcome["records"][0]["caption_source"], "the correction's own caption")

    def test_the_recorded_size_is_the_file_not_the_resolution_budget(self):
        # tool2 reports `resolution` as a budget of resolution**2 pixels and takes the shape from the input, so a
        # 1541 budget came back as 1216x1952. Recording the budget as both width and height would be a false statement
        # about the file, and the export and the quota board both read it.
        target, data, _results = self._plan_with_out()
        out = Path(data["jobs"][0]["out"])
        out.write_bytes(TINY_PNG)
        row = {"out": str(out), "status": "done", "steps": 25, "cfg": 1.0, "resolution": 1541, "seed": 7}
        outcome = fix.import_corrections(self.project, data, [row], log=lambda *_: None)
        corrected = outcome["records"][0]
        self.assertEqual((corrected["width"], corrected["height"]), (1, 1),
                         "the size comes from the file's own header")
        self.assertEqual(corrected["resolution_budget"], 1541, "the budget is kept, but as the budget it is")

    def test_a_file_whose_size_cannot_be_read_is_still_imported(self):
        _target, data, _results = self._plan_with_out()
        out = Path(data["jobs"][0]["out"])
        out.write_bytes(b"not a picture")
        row = {"out": str(out), "status": "done", "steps": 25, "cfg": 1.0, "resolution": 1541, "seed": 7}
        outcome = fix.import_corrections(self.project, data, [row], log=lambda *_: None)
        self.assertEqual(len(outcome["records"]), 1, "an unreadable size is not a reason to lose the correction")
        self.assertIsNone(outcome["records"][0]["width"])

    def test_the_page_shows_the_picture_it_corrected(self):
        target, data, results = self._plan_with_out()
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        rows = {row["id"]: row for row in review.candidate_rows(self.project)}
        corrected = rows[outcome["records"][0]["id"]]
        self.assertEqual(corrected["fix_of"], target)
        self.assertIsNotNone(corrected["before"])
        self.assertEqual(corrected["before"]["id"], target)
        self.assertTrue(corrected["before"]["url"].endswith(".png"))
        html = review.write_page(self.project, self.root / "review.html", log=lambda *_: None).read_text("utf-8")
        self.assertIn('class="pair"', html)
        self.assertIn("원본", html)


class RecoveryTests(Base):
    """R4: an import that was interrupted recovers its pictures through the sidecar, and the engine is the plan's."""

    def _corrected(self, plan_data):
        out = Path(plan_data["jobs"][0]["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(TINY_PNG + b"CORRECTED")
        return out

    def test_a_skipped_row_with_a_verified_sidecar_is_recovered(self):
        sources = self.origins()
        target = self.candidate("fixme.png", parents=[sources["o1.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        out = self._corrected(data)
        job = data["jobs"][0]
        out.with_name(out.name + ".fix.json").write_text(json.dumps({
            "input": lineage.quick_hash(Path(job["input"])),
            "graph_version": data["graph_version"]}), encoding="utf-8")
        results = [{"input": job["input"], "out": str(out), "status": "skipped"}]
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(len(outcome["records"]), 1, "the interrupted import recovers its picture")
        self.assertEqual(outcome["records"][0]["result_row_status"], "skipped")

    def test_a_skipped_row_without_a_sidecar_is_not_recovered(self):
        sources = self.origins()
        self.candidate("fixme.png", parents=[sources["o1.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        out = self._corrected(data)
        results = [{"input": data["jobs"][0]["input"], "out": str(out), "status": "skipped"}]
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(outcome["records"], [])
        self.assertIn("sidecar", outcome["skipped"][0]["reason"])

    def test_a_sidecar_naming_other_bytes_is_not_evidence(self):
        sources = self.origins()
        self.candidate("fixme.png", parents=[sources["o1.png"]])
        data = fix.plan(self.project, log=lambda *_: None)
        out = self._corrected(data)
        job = data["jobs"][0]
        other = self.root / "other.png"
        other.write_bytes(TINY_PNG + b"OTHER")
        out.with_name(out.name + ".fix.json").write_text(json.dumps({
            "input": lineage.quick_hash(other), "graph_version": data["graph_version"]}), encoding="utf-8")
        results = [{"input": job["input"], "out": str(out), "status": "skipped"}]
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(outcome["records"], [])

    def test_the_plan_records_the_engine_the_project_chose(self):
        self.origins()
        self.candidate("fixme.png", parents=[self.project.source_records()[0]["id"]])
        data = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual(data["engine"], "qwen")
        self.assertTrue(Path(data["config"]).is_file())

    def test_a_ph3_plan_records_a_ph3_correction(self):
        self.tool2_config.write_text(json.dumps({"models": {"backend": "p_h3"}}), encoding="utf-8")
        sources = self.origins()
        self.candidate("fixme.png", parents=[sources["o1.png"]], depth=0, generation=1)
        data = fix.plan(self.project, log=lambda *_: None)
        self.assertEqual(data["engine"], "ph3")
        out = self._corrected(data)
        results = [{"input": data["jobs"][0]["input"], "out": str(out), "status": "done", "steps": 8, "cfg": 1.0,
                    "resolution": 1024, "seed": 7}]
        outcome = fix.import_corrections(self.project, data, results, log=lambda *_: None)
        self.assertEqual(len(outcome["records"]), 1)
        self.assertEqual(outcome["records"][0]["generator"], "ph3_edit")
        self.assertEqual(outcome["records"][0]["models"]["family"], "ph3")


if __name__ == "__main__":
    unittest.main()
