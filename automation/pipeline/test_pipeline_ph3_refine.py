"""P-H3 refine ("측면 다듬기"): which batch B pictures are polished, with which originals, and what lands in lineage.

  venv\\Scripts\\python.exe -B -m unittest pipeline.test_pipeline_ph3_refine      (from automation\\)
"""
import json
import unittest
from pathlib import Path
from unittest import mock

from pipeline import composition, ph3_edit, ph3_refine
from pipeline.test_pipeline_ph3_edit import _Base, _png


def original(oid, shot, view):
    half = "body" if shot in composition.SHOTS["body"] else "face"
    return {"id": oid, "name": f"{oid}.png", "tagged": True, "half": half, "shot": shot, "view": view,
            "pose": "", "expression": "neutral"}


class FakeClient:
    def __init__(self):
        self.uploads, self.runs = [], []

    def name_for(self, kind, name):
        return name

    def upload(self, path, subfolder="x"):
        self.uploads.append(Path(path).name)
        return Path(path).name

    def run(self, graph, out, **meta):
        self.runs.append(meta)
        from PIL import Image
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (5, 10, 15)).save(out)
        return out


class WordsTests(unittest.TestCase):
    def test_the_view_comes_from_the_cell_id(self):
        self.assertEqual(ph3_refine.view_of("cmp_portrait_side_smiling"), "side profile")
        self.assertEqual(ph3_refine.view_of("cmp_full_quarter_walking_angry"), "three-quarter view")
        self.assertIsNone(ph3_refine.view_of("frm_back"))

    def test_the_card_picks_the_views_and_unknown_words_fall_back(self):
        self.assertEqual(ph3_refine.views({}), ("side profile",))
        self.assertEqual(ph3_refine.views({"ph3_refine": {"views": ["back view", "nonsense"]}}), ("back view",))
        self.assertEqual(ph3_refine.views({"ph3_refine": {"views": ["nonsense"]}}), ("side profile",))

    def test_the_instruction_keeps_picture_one_and_names_the_traits(self):
        text = ph3_refine.instruction({"subject": "the woman", "face_traits": ["mole under her left eye"],
                                       "fixed_traits": ["black scrunchie"]})
        self.assertTrue(text.startswith("Keep the pose, the camera angle, the framing, the clothing and the "
                                        "background of Picture 1 exactly."))
        self.assertIn("the woman's face", text)
        self.assertIn("mole under her left eye", text)
        self.assertIn("black scrunchie", text)


class PlanTests(_Base):
    """sources() is patched: its lineage checks are ph3_edit's and tested there; this is about the choice."""

    def usable(self):
        rows = []
        for oid in ("s_face", "s_side", "s_full"):
            rows.append({"id": oid, "path": str(_png(self.root / "sources" / f"{oid}.png", oid)), "origin": "source",
                         "sha256": oid * 4, "generation": 1, "edit_depth": 1})
        for cid in ("c_side", "c_front", "c_done"):
            rows.append({"id": cid, "path": str(_png(self.root / "candidates" / f"{cid}.png", cid)),
                         "origin": "candidate", "sha256": cid * 4, "generation": 1, "edit_depth": 1})
        return rows

    def candidates(self):
        return [{"id": "c_side", "method": "krea2_b", "slot": {"slot_id": "cmp_bust_side_sad", "category": "composition"}},
                {"id": "c_front", "method": "krea2_b",
                 "slot": {"slot_id": "cmp_bust_front_sad", "category": "composition"}},
                {"id": "c_done", "method": "krea2_b",
                 "slot": {"slot_id": "cmp_portrait_side_angry", "category": "composition"}},
                {"id": "c_polished", "method": "ph3_refine", "parent_ids": ["c_done", "s_face"]}]

    def plan(self, originals):
        with mock.patch.object(ph3_edit, "sources", return_value=(self.usable(), [])), \
                mock.patch.object(composition, "originals", return_value=originals), \
                mock.patch.object(type(self.project), "candidates", lambda _self: self.candidates()):
            return ph3_refine.plan(self.project, log=lambda *_: None)

    def test_only_side_pictures_not_yet_polished_are_planned(self):
        manifest = self.plan([original("s_face", "portrait", "three-quarter view"),
                              original("s_side", "upper body", "side profile")])
        self.assertEqual([t["source_id"] for t in manifest["tasks"]], ["c_side"])
        task = manifest["tasks"][0]
        self.assertEqual(task["slot_id"], "cmp_bust_side_sad")
        self.assertEqual([r["id"] for r in task["references"]], ["s_face", "s_side"])
        self.assertTrue(ph3_refine.manifest_path(self.project).is_file())

    def test_the_references_are_only_originals_the_sources_check_let_through(self):
        # s_hold is classified as an original but is not in the usable list (a holdout): it is never picked.
        manifest = self.plan([original("s_hold", "close-up", "front view"),
                              original("s_side", "upper body", "side profile")])
        self.assertEqual([r["id"] for r in manifest["references"]], ["s_side"])

    def test_no_usable_original_refuses_the_plan(self):
        with self.assertRaises(ph3_refine.RefineError):
            self.plan([original("s_hold", "close-up", "front view")])


class CandidateFilterTests(_Base):
    def test_undecided_pictures_are_sources_only_when_the_caller_says_so(self):
        ids = self.import_two_originals()
        record = self.add_candidate("c.png", parents=(ids["o1.png"],), depth=0)   # child depth 1: within the cap
        usable, _ = ph3_edit.sources(self.project)
        self.assertNotIn(record["id"], [u["id"] for u in usable])
        usable, _ = ph3_edit.sources(self.project, candidate_ok=lambda rec, decision: decision != "reject")
        self.assertIn(record["id"], [u["id"] for u in usable])


class RunTests(_Base):
    def test_picture_one_is_the_source_and_the_record_names_every_picture_it_came_from(self):
        ids = self.import_two_originals()
        source = self.add_candidate("krea.png", parents=(ids["o1.png"],))
        src_path = self.project.resolve_file(source["file"])
        refs = []
        for name in ("o1.png", "o2.png"):
            rec = [r for r in self.project.source_records() if Path(r["src_path"]).name == name][0]
            refs.append({"id": rec["id"], "path": str(self.project.resolve_file(rec.get("file") or rec["src_path"])),
                         "sha256": rec["sha256"]})
        manifest = {"kind": "ph3_refine", "steps": 8, "status": "planned", "tasks": [{
            "slot_id": "cmp_bust_side_sad", "category": "composition", "seed": 7301, "instruction": "keep it",
            "source_id": source["id"], "source_path": str(src_path), "source_sha256": source["sha256"],
            "references": refs, "generation": 1, "edit_depth": 2, "file": "x.png", "status": "pending"}]}
        client = FakeClient()
        ph3_refine.run(self.project, manifest, client=client, log=lambda *_: None)
        self.assertEqual(client.uploads, [src_path.name] + [Path(r["path"]).name for r in refs])
        self.assertEqual(manifest["tasks"][0]["status"], "done")
        records = ph3_refine.records_of(self.project, manifest)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["parent_ids"], [source["id"]] + [r["id"] for r in refs])
        self.assertEqual((record["generator"], record["method"]), ("ph3_edit", "ph3_refine"))
        self.assertEqual(record["slot"]["slot_id"], "cmp_bust_side_sad")
        self.assertEqual(record["edit_depth"], 2)

    def test_a_reference_that_changed_fails_the_task(self):
        ids = self.import_two_originals()
        source = self.add_candidate("krea.png", parents=(ids["o1.png"],))
        manifest = {"kind": "ph3_refine", "steps": 8, "tasks": [{
            "slot_id": "cmp_bust_side_sad", "category": "composition", "seed": 1, "instruction": "x",
            "source_id": source["id"], "source_path": str(self.project.resolve_file(source["file"])),
            "source_sha256": source["sha256"],
            "references": [{"id": "s_x", "path": str(_png(self.root / "sources" / "r.png", "r")), "sha256": "0" * 64}],
            "generation": 1, "edit_depth": 2, "file": "y.png", "status": "pending"}]}
        client = FakeClient()
        ph3_refine.run(self.project, manifest, client=client, log=lambda *_: None)
        self.assertEqual(manifest["tasks"][0]["status"], "failed")
        self.assertEqual(client.runs, [])


if __name__ == "__main__":
    unittest.main()
