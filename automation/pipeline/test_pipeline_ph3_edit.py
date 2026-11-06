"""P-H3 edit-run tests: which pictures may be edited, and what lands in the lineage.

The draw itself needs ComfyUI and 25 GB of weights, so it is exercised by hand. What is pinned here is the part that
would otherwise fail silently days later: sourcing from a holdout, sourcing from an unparented anchor, and drawing
into a slot that already has pictures waiting on review.
"""
import json
import unittest
from pathlib import Path

from pipeline import batch_gen, lineage, ph3_edit, project as pj, seed

SOURCE = "G:/데이터셋 통합작성툴/Fizgig_auto/automation/pipeline/seed.py"


def _png(path, tag="x"):
    """Enough bytes that the digest differs per tag; the tests never decode it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + tag.encode())
    return path


class _Base(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "proj"
        self.root.mkdir(parents=True)
        (self.root / "project.json").write_text(json.dumps({
            "id": "t", "character": "t", "lineage": {"max_train_generation": 2,
                                                     "seed_max_generation": 1, "max_edit_depth": 1},
        }), encoding="utf-8")
        self.project = pj.Project.open(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def original(self, name):
        return _png(self.root / "sources" / name, name)

    def import_two_originals(self):
        paths = [self.original("o1.png"), self.original("o2.png")]
        self.project.import_sources(paths, log=lambda *_: None)
        return {Path(r["src_path"]).name: r["id"] for r in self.project.source_records()}

    def import_anchor(self):
        """A synthetic anchor: origin=synthetic and no parents, which is what makes it unusable as a source."""
        anchor = self.original("anchor.png")
        self.project.import_sources([anchor], origin_by_name={"anchor.png": "synthetic"}, log=lambda *_: None)
        return [r for r in self.project.source_records() if Path(r["src_path"]).name == "anchor.png"][0]

    @staticmethod
    def refusal(unusable, record_id):
        """The refusal row for one id, so an assertion names the picture it means."""
        row = [u for u in unusable if u["id"] == record_id]
        assert row, f"no refusal for {record_id}: {unusable}"
        return row[0]

    def add_candidate(self, name, *, generator="qwen_edit", parents=("s_1",), depth=1, generation=1):
        path = _png(self.root / "candidates" / name, name)
        record = lineage.record(path, generator=generator, parent_ids=list(parents),
                                train_generation=generation, edit_depth=depth,
                                slot={"slot_id": "frm_back", "category": "framing"},
                                prompt="p", seed=1, steps=8, cfg=0.0,
                                graph_version="ph3-edit-v1", method=generator)
        self.project.import_candidates([record], log=lambda *_: None)
        return record


class SourceTests(_Base):
    def test_a_user_original_is_editable(self):
        self.import_two_originals()
        available, unusable = ph3_edit.sources(self.project)
        self.assertEqual(len(available), 2)
        self.assertEqual(unusable, [])
        # A direct edit of an original is a generation-1, depth-1 picture (the product's shape for a model-made edit).
        self.assertEqual((available[0]["generation"], available[0]["edit_depth"]), (1, 1))

    def test_a_holdout_original_is_never_a_source(self):
        # Editing it would put the holdout back into the training set through the back door. A holdout needs
        # holdout.min_sources originals before it resolves at all, so the card is given one explicitly here.
        self.import_two_originals()
        holdout_id = self.project.source_records()[0]["id"]
        self.project.card["holdout"] = {"ids": [holdout_id], "files": [], "shas": [],
                                        "reason": "frozen for the test", "frozen_at": "2026-01-01T00:00:00"}
        self.project.save()
        self.project = pj.Project.open(self.root)
        available, _unusable = ph3_edit.sources(self.project)
        self.assertNotIn(holdout_id, {s["id"] for s in available})

    def test_a_picture_carrying_the_holdouts_bytes_is_never_a_source(self):
        # The holdout frozen by id keeps its id, but the bytes can travel under another candidate's id: the sha is
        # what identifies a holdout picture, not the name (spec R3).
        self.import_two_originals()
        holdout = self.project.source_records()[0]
        self.project.card["holdout"] = {"ids": [holdout["id"]], "files": [], "shas": [],
                                        "reason": "frozen for the test", "frozen_at": "2026-01-01T00:00:00"}
        self.project.save()
        self.project = pj.Project.open(self.root)
        twin = _png(self.root / "candidates" / "twin.png")
        twin.write_bytes(Path(holdout["src_path"]).read_bytes())
        record = lineage.record(twin, generator="qwen_edit", parent_ids=["s_other"], train_generation=1,
                                edit_depth=1, slot={"slot_id": "frm_x"}, prompt="p")
        self.project.import_candidates([record], log=lambda *_: None)
        self.project.decide(record["id"], "accept")
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(record["id"], {s["id"] for s in available})
        self.assertEqual(self.refusal(unusable, record["id"])["code"], "holdout_source")

    def test_an_unparented_synthetic_anchor_is_refused_with_its_reason(self):
        # This is the case a real run hit: the anchor has origin=synthetic and no parent, so every edit of it would be
        # excluded as unverified_ancestor - a picture that can never be used.
        self.import_two_originals()
        anchor = self.import_anchor()
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(anchor["id"], {s["id"] for s in available})
        self.assertEqual(len(unusable), 1)
        self.assertIn("no recorded parent", unusable[0]["reason"])

    def test_a_holdout_derived_candidate_is_never_a_source(self):
        # Editing it would hand the holdout's pixels to another picture the audit can never admit.
        by_name = self.import_two_originals()
        holdout_id = self.project.source_records()[0]["id"]
        self.project.card["holdout"] = {"ids": [holdout_id], "files": [], "shas": [],
                                        "reason": "frozen for the test", "frozen_at": "2026-01-01T00:00:00"}
        self.project.save()
        self.project = pj.Project.open(self.root)
        made = self.add_candidate("child.png", parents=(holdout_id,))
        self.project.decide(made["id"], "accept")
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(made["id"], {s["id"] for s in available})
        self.assertEqual(self.refusal(unusable, made["id"])["code"], "holdout_indirect")

    def test_a_tampered_source_is_refused(self):
        # The plan must check the parent's bytes, not just its record: an edit of a changed picture would record a
        # parent id for pixels the record never described. The bytes that get edited are the project's own copy.
        self.import_two_originals()
        first = self.project.source_records()[0]
        resolved = self.project.resolve_file(first.get("file") or first.get("src_path"))
        resolved.write_bytes(b"\x89PNG\r\n\x1a\ntampered")
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(first["id"], {s["id"] for s in available})
        self.assertEqual(self.refusal(unusable, first["id"])["code"], "tampered_file")

    def test_a_candidate_with_a_missing_parent_is_refused(self):
        self.import_two_originals()
        made = self.add_candidate("orphan.png", parents=("s_nope",))
        self.project.decide(made["id"], "accept")
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(made["id"], {s["id"] for s in available})
        self.assertEqual(self.refusal(unusable, made["id"])["code"], "missing_parent")

    def test_a_candidate_trained_on_an_unknown_snapshot_is_refused(self):
        self.import_two_originals()
        path = _png(self.root / "candidates" / "lora_child.png")
        record = lineage.record(path, generator="krea2_lora", parent_ids=[], train_generation=2, edit_depth=0,
                                slot={"slot_id": "frm_x"}, prompt="p",
                                models={"loras": [{"name": "l.safetensors", "trained_on_snapshot": "nosnap00"}]},
                                method="krea2_b")
        self.project.import_candidates([record], log=lambda *_: None)
        self.project.decide(record["id"], "accept")
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(record["id"], {s["id"] for s in available})
        self.assertEqual(self.refusal(unusable, record["id"])["code"], "lora_snapshot_unverified")

    def test_a_candidate_whose_lora_snapshot_holds_a_holdout_is_refused(self):
        # The picture was drawn by a LoRA that learned from the holdout: editing it would wash the holdout through
        # another picture (spec R3: "그 편집을 학습한 LoRA와 그 자손도 전부 제외한다").
        self.import_two_originals()
        holdout = self.project.source_records()[0]
        self.project.card["holdout"] = {"ids": [holdout["id"]], "files": [], "shas": [],
                                        "reason": "frozen for the test", "frozen_at": "2026-01-01T00:00:00"}
        self.project.save()
        self.project = pj.Project.open(self.root)
        snap_dir = self.root / "seed_lora" / "snapbad00"
        snap_dir.mkdir(parents=True)
        (snap_dir / "snapshot.json").write_text(json.dumps(
            {"id": "snapbad00", "inputs": [{"sha256": holdout["sha256"]}], "holdout_ids": []}), encoding="utf-8")
        path = _png(self.root / "candidates" / "via_lora.png")
        record = lineage.record(path, generator="krea2_lora", parent_ids=[], train_generation=2, edit_depth=0,
                                slot={"slot_id": "frm_x"}, prompt="p",
                                models={"loras": [{"name": "l.safetensors", "trained_on_snapshot": "snapbad00"}]},
                                method="krea2_b")
        self.project.import_candidates([record], log=lambda *_: None)
        self.project.decide(record["id"], "accept")
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(record["id"], {s["id"] for s in available})
        self.assertEqual(self.refusal(unusable, record["id"])["code"], "holdout_via_lora")

    def test_an_edit_of_a_depth_one_candidate_is_refused_when_the_cap_is_one(self):
        # The child would sit at depth 2: past a cap of 1 no audit can ever admit it, so the plan refuses it before
        # any GPU work instead of drawing a picture that can only be thrown away (spec R3).
        self.import_two_originals()
        made = self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],))
        self.project.decide(made["id"], "accept")
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(made["id"], {s["id"] for s in available})
        self.assertEqual(self.refusal(unusable, made["id"])["code"], "edit_depth_cap")

    def test_a_candidate_edit_keeps_the_parent_generation_and_goes_one_deeper(self):
        # With a cap of 2 the edit is a legitimate depth-2 picture: generation stays at the parent's, depth +1.
        self.import_two_originals()
        made = self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],), depth=1, generation=1)
        self.project.decide(made["id"], "accept")
        self.project.card["lineage"]["max_edit_depth"] = 2
        self.project.save()
        self.project = pj.Project.open(self.root)
        available, _unusable = ph3_edit.sources(self.project)
        row = [s for s in available if s["id"] == made["id"]][0]
        self.assertEqual((row["generation"], row["edit_depth"]), (1, 2))

    def test_an_accepted_edit_is_editable(self):
        # A depth-0 picture is a batch B draw: editing it lands at depth 1, inside the cap.
        self.import_two_originals()
        made = self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],), depth=0, generation=2,
                                  generator="krea2_lora")
        self.project.decide(made["id"], "accept")
        available, _unusable = ph3_edit.sources(self.project)
        self.assertIn(made["id"], {s["id"] for s in available})

    def test_an_undecided_edit_is_not_editable(self):
        self.import_two_originals()
        made = self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],))
        available, _unusable = ph3_edit.sources(self.project)
        self.assertNotIn(made["id"], {s["id"] for s in available})


class InstructionTests(unittest.TestCase):
    def test_every_known_slot_has_an_instruction(self):
        for slot in ("frm_back", "frm_profile", "frm_dutch", "frm_full_body", "ang_back",
                     "ang_front_wide", "ang_front_medium", "ang_quarter_close"):
            self.assertTrue(ph3_edit.instruction_for(slot, "framing"), slot)

    def test_a_slot_without_an_instruction_falls_back_to_its_category(self):
        self.assertTrue(ph3_edit.instruction_for("some_new_slot", "backgrounds"))

    def test_an_unknown_slot_in_an_unknown_category_has_none(self):
        # A vague instruction would produce a picture nobody asked for, so this is a refusal, not a default.
        self.assertIsNone(ph3_edit.instruction_for("mystery", "nonsense"))


class RecordTests(_Base):
    def test_a_finished_edit_becomes_a_depth_one_candidate_of_its_source(self):
        by_name = self.import_two_originals()
        folder = ph3_edit.manifest_path(self.project).parent
        _png(folder / "frm_back_s7001.png", "edit")
        manifest = {
            "steps": 8,
            "tasks": [{"slot_id": "frm_back", "category": "framing", "seed": 7001,
                       "instruction": "show the back", "source_id": by_name["o1.png"],
                       "source_path": "x", "file": "frm_back_s7001.png"}],
        }
        records = ph3_edit.records_of(self.project, manifest)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["generator"], "ph3_edit")
        self.assertEqual(record["edit_depth"], 1)
        self.assertEqual(record["parent_ids"], [by_name["o1.png"]])
        self.assertEqual(record["slot"]["slot_id"], "frm_back")

    def test_a_missing_file_is_not_recorded(self):
        self.import_two_originals()
        manifest = {"steps": 8, "tasks": [{"slot_id": "frm_back", "category": "framing", "seed": 1,
                                           "instruction": "i", "source_id": "s_1", "source_path": "x",
                                           "file": "never_drawn.png"}]}
        self.assertEqual(ph3_edit.records_of(self.project, manifest), [])

    def test_the_edit_is_eligible_only_under_policy_v2(self):
        # The whole point of recording ph3_edit: v2 admits it, v1 keeps its old meaning.
        by_name = self.import_two_originals()
        folder = ph3_edit.manifest_path(self.project).parent
        _png(folder / "frm_back_s7001.png", "edit")
        manifest = {"steps": 8, "tasks": [{"slot_id": "frm_back", "category": "framing", "seed": 7001,
                                           "instruction": "i", "source_id": by_name["o1.png"],
                                           "source_path": "x", "file": "frm_back_s7001.png"}]}
        record = ph3_edit.records_of(self.project, manifest)[0]
        self.project.import_candidates([record], log=lambda *_: None)
        self.project.decide(record["id"], "accept")
        v2 = seed.audit(self.project, policy="seed-v2")
        self.assertIn(record["id"], {k["id"] for k in v2["kept"]})
        v1 = seed.audit(self.project, policy="seed-v1")
        excluded = {x["id"]: x["code"] for x in v1["excluded"]}
        self.assertEqual(excluded.get(record["id"]), "generator_not_eligible")


class ShortSlotTests(_Base):
    """The decision that matters, tested without a snapshot or a GPU.

    The split consumed here is `batch_gen.short_slots` - the producer itself, read-only - not a hand-built manifest:
    planning against the wrong half of that split (the rows a waiting picture may still fill) is exactly how a real
    run once planned sixteen pictures for eight slots that each already had one waiting (spec R3).
    """

    def project_with_quotas(self, quotas):
        self.project.card["slot_quotas"] = dict(quotas)
        self.project.save()
        self.project = pj.Project.open(self.root)
        return self.project

    def test_a_short_slot_with_nothing_pending_is_drawn(self):
        self.import_two_originals()
        self.project_with_quotas({"ang_side": 1})
        rows = ph3_edit.short_slots(self.project)
        self.assertEqual([r["slot"] for r in rows], ["ang_side"])
        self.assertEqual(rows[0]["waiting"], 0)

    def test_a_slot_already_waiting_on_review_is_left_alone(self):
        # Drawing here would grow the backlog instead of shrinking it - the mistake this function exists to prevent.
        self.import_two_originals()
        made = self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],), depth=0, generation=1)
        self.project_with_quotas({"frm_back": 1})     # the undecided candidate sits in frm_back, waiting
        self.assertEqual(ph3_edit.short_slots(self.project), [])

    def test_a_filled_slot_is_not_drawn(self):
        self.import_two_originals()
        made = self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],), depth=0, generation=1)
        self.project.decide(made["id"], "accept")
        self.project_with_quotas({"frm_back": 1})     # accepted: the quota is met
        self.assertEqual(ph3_edit.short_slots(self.project), [])

    def test_force_draws_a_slot_that_is_already_waiting(self):
        self.import_two_originals()
        self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],), depth=0, generation=1)
        self.project_with_quotas({"frm_back": 1})
        self.assertEqual([r["slot"] for r in ph3_edit.short_slots(self.project, force=True)], ["frm_back"])

    def test_naming_slots_narrows_the_choice(self):
        self.import_two_originals()
        self.project_with_quotas({"ang_side": 1, "frm_full_body": 1})
        rows = ph3_edit.short_slots(self.project, slots=["frm_full_body"])
        self.assertEqual([r["slot"] for r in rows], ["frm_full_body"])

    def test_a_card_with_no_slot_plan_has_no_gaps(self):
        self.import_two_originals()
        self.assertEqual(ph3_edit.short_slots(self.project), [])


class PlanTests(_Base):
    def test_planning_with_no_eligible_source_raises(self):
        # Reached before any snapshot lookup, so a project with no editable picture fails for the right reason.
        with self.assertRaises(ph3_edit.EditError) as caught:
            ph3_edit.plan(self.project, log=lambda *_: None)
        self.assertIn("eligible to edit", str(caught.exception))

    def test_a_plan_refuses_an_unparented_anchor_as_its_only_source(self):
        self.import_anchor()
        with self.assertRaises(ph3_edit.EditError):
            ph3_edit.plan(self.project, log=lambda *_: None)

    def _project_with_a_short_slot(self):
        """Two originals, no candidates: `frm_back` is short with nothing waiting on review."""
        self.import_two_originals()
        self.project.card["slot_quotas"] = {"frm_back": 1}
        self.project.save()
        return pj.Project.open(self.root)

    def test_a_plan_draws_the_seeds_asked_for_a_short_slot(self):
        project = self._project_with_a_short_slot()
        manifest = ph3_edit.plan(project, seeds=[7001, 7002, 7003], log=lambda *_: None)
        self.assertEqual(len(manifest["tasks"]), 3)
        self.assertEqual({t["slot_id"] for t in manifest["tasks"]}, {"frm_back"})
        task = manifest["tasks"][0]
        self.assertEqual((task["generation"], task["edit_depth"]), (1, 1))
        self.assertEqual(task["source_sha256"], project.source_records()[0]["sha256"])

    def test_a_plan_skips_a_slot_whose_picture_is_waiting_on_review(self):
        project = self._project_with_a_short_slot()
        made = self.add_candidate("w.png", parents=(project.source_records()[0]["id"],), depth=0, generation=1)
        project.card["slot_quotas"] = {"frm_back": 1}
        project.save()
        project = pj.Project.open(self.root)          # the waiting candidate sits in frm_back
        manifest = ph3_edit.plan(project, log=lambda *_: None)
        self.assertEqual(manifest["tasks"], [])
        self.assertEqual([g["slot"] for g in manifest["gaps"]], [])

    def test_planning_writes_nothing_but_its_own_manifest_and_needs_no_seed_snapshot(self):
        # Planning an edit must not re-plan batch B (its manifest is someone else's deliverable) and must not require
        # a Krea2 training snapshot: an independent edit does not draw through that LoRA (spec R3).
        project = self._project_with_a_short_slot()
        batch_plan = project.dir / "experiments" / "batch_b" / "batch.json"
        batch_plan.parent.mkdir(parents=True)
        batch_plan.write_text('{"kind": "batch_b", "tasks": [{"slot_id": "frm_back", "seed": 2000}]}',
                              encoding="utf-8")
        before = batch_plan.read_bytes()
        manifest = ph3_edit.plan(project, log=lambda *_: None)
        self.assertEqual(len(manifest["tasks"]), 2)   # SEEDS_PER_SLOT for the one short slot
        self.assertEqual(batch_plan.read_bytes(), before)
        self.assertFalse((project.dir / "seed_lora").exists())

    def test_the_plan_records_why_a_source_is_not_editable(self):
        self.import_two_originals()
        self.import_anchor()
        self.project.card["slot_quotas"] = {"frm_back": 1}
        self.project.save()
        project = pj.Project.open(self.root)
        manifest = ph3_edit.plan(project, log=lambda *_: None)
        self.assertEqual(len(manifest["unusable_sources"]), 1)
        self.assertEqual(manifest["unusable_sources"][0]["code"], "unverified_ancestor")


if __name__ == "__main__":
    unittest.main()
