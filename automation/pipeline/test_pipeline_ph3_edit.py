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

    def test_an_unparented_synthetic_anchor_is_refused_with_its_reason(self):
        # This is the case a real run hit: the anchor has origin=synthetic and no parent, so every edit of it would be
        # excluded as unverified_ancestor - a picture that can never be used.
        self.import_two_originals()
        anchor = self.import_anchor()
        available, unusable = ph3_edit.sources(self.project)
        self.assertNotIn(anchor["id"], {s["id"] for s in available})
        self.assertEqual(len(unusable), 1)
        self.assertIn("no recorded parent", unusable[0]["reason"])

    def test_an_accepted_edit_is_editable(self):
        self.import_two_originals()
        made = self.add_candidate("c1.png", parents=(self.project.source_records()[0]["id"],))
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


class ShortSlotTests(unittest.TestCase):
    """The decision that matters, tested without a snapshot or a GPU."""

    def _gaps(self, *rows):
        return {"waiting": list(rows)}

    def test_a_short_slot_with_nothing_pending_is_drawn(self):
        gaps = self._gaps({"slot": "a", "short": 1, "waiting": 0})
        self.assertEqual([r["slot"] for r in ph3_edit.short_slots(gaps)], ["a"])

    def test_a_slot_already_waiting_on_review_is_left_alone(self):
        # Drawing here would grow the backlog instead of shrinking it - the mistake this function exists to prevent.
        gaps = self._gaps({"slot": "a", "short": 1, "waiting": 2})
        self.assertEqual(ph3_edit.short_slots(gaps), [])

    def test_a_filled_slot_is_not_drawn(self):
        gaps = self._gaps({"slot": "a", "short": 0, "waiting": 0})
        self.assertEqual(ph3_edit.short_slots(gaps), [])

    def test_force_draws_a_slot_that_is_already_waiting(self):
        gaps = self._gaps({"slot": "a", "short": 1, "waiting": 2})
        self.assertEqual([r["slot"] for r in ph3_edit.short_slots(gaps, force=True)], ["a"])

    def test_naming_slots_narrows_the_choice(self):
        gaps = self._gaps({"slot": "a", "short": 1, "waiting": 0}, {"slot": "b", "short": 1, "waiting": 0})
        self.assertEqual([r["slot"] for r in ph3_edit.short_slots(gaps, slots=["b"])], ["b"])

    def test_an_empty_plan_stays_empty(self):
        self.assertEqual(ph3_edit.short_slots({"waiting": []}), [])
        self.assertEqual(ph3_edit.short_slots({}), [])


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


if __name__ == "__main__":
    unittest.main()
