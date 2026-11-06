"""R2: the seed and the final export answer different questions, so they get different rules.

The fixture is a real `Project` folder holding every kind of picture the two audits must sort apart, with the batch B
plan manifest a final export needs for proof. What is pinned here:

* **the seed keeps its own meaning** - `seed.audit()` with no purpose keeps the seed-v2 generator list and the
  `seed_max_generation` ceiling, so a Krea2 batch B picture stays a seed non-input;
* **the final export widens exactly that** - the same shared checks (holdout, lineage, bytes, duplicates, the
  reviewer's verdict) plus the seed generators, plus a Krea2 batch B picture whose plan manifest names it, under the
  final `max_train_generation` ceiling: 14 originals + 13 kept edits (one edit is a byte-duplicate of a source) =
  **27 seed**, + 6 plan-proven batch B pictures = **33 final**;
* **membership is proven, not assumed** - a slot name alone never lets a picture in: an old plan's picture, and a
  picture that merely reuses a planned slot and seed, are refused with a reason;
* **the holdout is sealed** - a direct holdout edit, its descendants, a LoRA trained on a snapshot holding a holdout
  and that LoRA's output are all outside the final set.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline import batch_gen, export, lineage, seed
from pipeline import project as pj

SNAP_OK = "111111111111"      # the seed snapshot batch B drew with: the 13 non-holdout originals
SNAP_BAD = "222222222222"     # a snapshot that holds a holdout: its LoRA's output is refused
SNAP_MISSING = "deadbeefdead"  # a snapshot this project has no record of


def bytes_of(marker):
    """Distinct, stable bytes per picture. The audit hashes; it does not decode, so these need not be real PNGs."""
    return bytes([marker % 256, (marker // 256) % 256]) * 64


class Fixture:
    """A real project with 17 sources and 36 candidates, and the batch B plan manifest several of them need."""

    def __init__(self, root):
        self.root = Path(root)
        self.lora_path = self.root / "lora.safetensors"
        self.lora_path.write_bytes(bytes_of(1))
        self.project = pj.Project.create("fx", self.card(), folder=self.root / "proj")
        self.build_sources()
        self.build_candidates()
        self.build_plans()

    def card(self):
        return {"id": "fx", "trigger": "fx", "caption_base": "style", "caption_traits": "omit",
                "fixed_traits": [], "outfits": {},
                "lineage": {"max_train_generation": 2, "seed_max_generation": 1, "max_edit_depth": 1},
                "final_families": ["krea2"], "slot_quotas": {"frm_back": 1},
                "holdout": {"ids": []}}  # the real ids are frozen once the sources exist

    # ------------------------------------------------------------------ sources
    def build_sources(self):
        self.shas = {}
        incoming = self.root / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        for n in range(1, 16):                      # 15 user originals; the first two become the frozen holdout
            path = incoming / f"orig_{n:02d}.png"
            path.write_bytes(bytes_of(n))
            self.shas[path.name] = lineage.sha256_file(path)
        anchor = incoming / "anchor_extra.png"      # a synthetic anchor with no recorded parent
        anchor.write_bytes(bytes_of(99))
        self.shas[anchor.name] = lineage.sha256_file(anchor)
        # The same bytes as the first Qwen edit will have, arriving as a source: in the real project that is how
        # duplicates appear (a candidate and a source holding one picture), and one of the two must give way.
        dup = incoming / "orig_dup_of_edit.png"
        dup.write_bytes(bytes_of(100))
        self.project.import_sources(sorted(incoming.iterdir()), log=lambda *_: None)
        self.source_ids = {r["sha256"]: r["id"] for r in self.project.source_records()}
        self.holdout_ids = [self.source_ids[self.shas["orig_01.png"]], self.source_ids[self.shas["orig_02.png"]]]
        self.user_ids = [self.source_ids[self.shas[f"orig_{n:02d}.png"]] for n in range(3, 16)]
        # Freeze the holdout the way the pipeline does, by writing it into the card.
        self.project.card["holdout"] = {"ids": self.holdout_ids}
        self.project.save()

    # ------------------------------------------------------------------ candidates
    def source_png(self, name, marker):
        path = self.root / "drawn" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes_of(marker))
        return path

    def add_candidate(self, path, **kwargs):
        record = lineage.record(path, **kwargs)
        self.project.import_candidates([record], base=path.parent, log=lambda *_: None)
        return record["id"]

    def accept(self, candidate_id, decision="accept", **kwargs):
        self.project.decide(candidate_id, decision, **kwargs)

    def build_candidates(self):
        self.a_ids = []
        for n in range(14):                      # 14 accepted Qwen edits, generation 1, depth 1 - the seed candidates
            parent = self.user_ids[n % len(self.user_ids)]
            self.a_ids.append(self.add_candidate(
                self.source_png(f"a_{n:02d}.png", 100 + n), generator="qwen_edit", parent_ids=[parent],
                train_generation=1, edit_depth=1, slot={"slot_id": f"frm_face_{n}"}, prompt="p"))
        for candidate_id in self.a_ids:
            self.accept(candidate_id)

        self.rejected_id = self.add_candidate(
            self.source_png("rejected.png", 130), generator="qwen_edit", parent_ids=[self.user_ids[0]],
            train_generation=1, edit_depth=1, slot={"slot_id": "frm_x"}, prompt="p")
        self.accept(self.rejected_id, "reject")
        self.diff_char_id = self.add_candidate(
            self.source_png("diff_char.png", 131), generator="qwen_edit", parent_ids=[self.user_ids[0]],
            train_generation=1, edit_depth=1, slot={"slot_id": "frm_y"}, prompt="p")
        self.accept(self.diff_char_id, "reject", reason="different_character")
        self.fix_id = self.add_candidate(
            self.source_png("fix_pending.png", 132), generator="qwen_edit", parent_ids=[self.user_ids[0]],
            train_generation=1, edit_depth=1, slot={"slot_id": "frm_z"}, prompt="p")
        self.accept(self.fix_id, "fix")
        self.unreviewed_id = self.add_candidate(
            self.source_png("unreviewed.png", 133), generator="qwen_edit", parent_ids=[self.user_ids[0]],
            train_generation=1, edit_depth=1, slot={"slot_id": "frm_w"}, prompt="p")

        self.depth2_id = self.add_candidate(     # a fix of a fix: past max_edit_depth for both purposes
            self.source_png("depth2.png", 134), generator="qwen_edit", parent_ids=[self.a_ids[1]],
            train_generation=1, edit_depth=2, slot={"slot_id": "frm_d2"}, prompt="p")
        self.accept(self.depth2_id)
        self.holdout_direct_id = self.add_candidate(  # drawn directly from a holdout original
            self.source_png("holdout_direct.png", 135), generator="qwen_edit", parent_ids=[self.holdout_ids[0]],
            train_generation=1, edit_depth=1, slot={"slot_id": "frm_hd"}, prompt="p")
        self.accept(self.holdout_direct_id)
        self.holdout_indirect_id = self.add_candidate(  # a child of that edit: still holdout-derived
            self.source_png("holdout_indirect.png", 136), generator="qwen_edit",
            parent_ids=[self.holdout_direct_id], train_generation=1, edit_depth=1,
            slot={"slot_id": "frm_hi"}, prompt="p")
        self.accept(self.holdout_indirect_id)

        first = self.add_candidate(              # two pictures that name each other as parents: a cycle
            self.source_png("cyc_a.png", 137), generator="qwen_edit", parent_ids=[], train_generation=1,
            edit_depth=1, slot={"slot_id": "cyc_a"}, prompt="p", allow_missing_parents=True,
            orphan_reason="fixture cycle")
        second = self.add_candidate(
            self.source_png("cyc_b.png", 138), generator="qwen_edit", parent_ids=[first], train_generation=1,
            edit_depth=1, slot={"slot_id": "cyc_b"}, prompt="p")
        # Close the cycle by rewriting the first record in place; `import_candidates` would skip a known id.
        records = self.project.candidates()
        for record in records:
            if record["id"] == first:
                record["parent_ids"] = [second]
        lineage.write_jsonl(self.project.dir / "candidates.jsonl", records)
        self.cyclic_ids = [first, second]
        self.accept(first)
        self.accept(second)

        self.missing_parent_id = self.add_candidate(
            self.source_png("missing_parent.png", 139), generator="qwen_edit", parent_ids=["c_nope"],
            train_generation=1, edit_depth=1, slot={"slot_id": "frm_mp"}, prompt="p")
        self.accept(self.missing_parent_id)

        # An orphan built by hand: `lineage.record` would refuse a generated picture with neither parents nor a
        # LoRA snapshot, which is exactly the rule the audit is being asked to enforce.
        self.no_ancestry_id = self.add_candidate(
            self.source_png("no_ancestry.png", 140), generator="qwen_edit", parent_ids=[], train_generation=1,
            edit_depth=1, slot={"slot_id": "frm_na"}, prompt="p", allow_missing_parents=True,
            orphan_reason="fixture orphan")
        records = self.project.candidates()
        for record in records:
            if record["id"] == self.no_ancestry_id:
                record["models"] = {}
        lineage.write_jsonl(self.project.dir / "candidates.jsonl", records)
        self.accept(self.no_ancestry_id)

    # ------------------------------------------------------------------ batch B and its plan
    def b_kwargs(self, slot_id, category, seed_value, generation, snapshot):
        return dict(generator="krea2_lora", parent_ids=[], train_generation=generation, edit_depth=0,
                    slot={"slot_id": slot_id, "category": category, "prompt": f"prompt {slot_id}"},
                    models={"dit": {"name": "krea2_raw"},
                            "loras": [lineage.model_entry(self.lora_path, trained_on_snapshot=snapshot,
                                                          trained_on_generation=1)]},
                    prompt=f"prompt {slot_id}", seed=seed_value, steps=28, cfg=1.0,
                    width=1024, height=1536, graph_version="b-v3", method="krea2_b", batch="batch_b")

    def build_plans(self):
        exp = self.project.dir / "experiments" / "batch_b"
        images = exp / "images"
        images.mkdir(parents=True, exist_ok=True)
        self.b_ids = []
        tasks = []
        marker = iter(range(300, 399))           # one distinct byte pattern per picture: no two share a hash
        for slot_id in ("ang_front_medium", "ang_quarter_close", "frm_back"):
            for seed_value in (2000, 2001):
                path = images / f"{slot_id}__s{seed_value}.png"
                path.write_bytes(bytes_of(next(marker)))
                candidate_id = self.add_candidate(path, **self.b_kwargs(slot_id, "framing", seed_value, 2, SNAP_OK))
                self.b_ids.append(candidate_id)
                self.accept(candidate_id)
                tasks.append({"slot_id": slot_id, "category": "framing", "seed": seed_value,
                              "prompt": f"prompt {slot_id}", "file": f"images/{slot_id}__s{seed_value}.png",
                              "status": "done"})
        # Two accepted batch B pictures the shared checks must refuse for their own reasons: one drew with a LoRA
        # whose snapshot holds a holdout, one names a snapshot this project has no record of.
        for slot_id, snapshot in (("via_lora", SNAP_BAD), ("unknown_snap", SNAP_MISSING)):
            path = images / f"{slot_id}__s4000.png"
            path.write_bytes(bytes_of(next(marker)))
            candidate_id = self.add_candidate(path, **self.b_kwargs(slot_id, "poses", 4000, 2, snapshot))
            self.accept(candidate_id)
            tasks.append({"slot_id": slot_id, "category": "poses", "seed": 4000,
                          "prompt": f"prompt {slot_id}", "file": f"images/{slot_id}__s4000.png", "status": "done"})
        # Generation 3 from the next cycle: the plan names it, the final ceiling refuses it.
        path = images / "pos_gen3__s3000.png"
        path.write_bytes(bytes_of(next(marker)))
        self.gen3_id = self.add_candidate(path, **self.b_kwargs("pos_gen3", "poses", 3000, 3, SNAP_OK))
        self.accept(self.gen3_id)
        tasks.append({"slot_id": "pos_gen3", "category": "poses", "seed": 3000, "prompt": "prompt pos_gen3",
                      "file": "images/pos_gen3__s3000.png", "status": "done"})
        manifest = {
            "kind": "batch_b", "created_at": "2026-11-07T00:00:00", "project": "fx", "generation": 2,
            "training": {"trained_on_snapshot": SNAP_OK, "trained_on_generation": 1},
            "trained_on_snapshot": SNAP_OK, "inference": {"variant": "b"}, "size": {"width": 1024, "height": 1536},
            "slots": [], "waiting": [], "seeds_per_slot": 2, "tasks": tasks, "status": "ran",
        }
        (exp / "batch.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

        # A picture from an older plan: its slot exists in the plan, its task does not, and no plan output file
        # carries its bytes. Membership would have to come from the slot name alone, which is exactly what R2 forbids.
        old = self.source_png("old_plan.png", 220)
        self.old_plan_id = self.add_candidate(old, **self.b_kwargs("frm_back", "framing", 9999, 2, SNAP_OK))
        self.accept(self.old_plan_id)
        # A picture that reuses a planned slot and seed but is not the bytes that task produced.
        stolen = self.source_png("stolen_slot.png", 221)
        self.stolen_id = self.add_candidate(stolen, **self.b_kwargs("ang_front_medium", "framing", 2000, 2, SNAP_OK))
        self.accept(self.stolen_id)

        # The two training snapshots the LoRA checks read: one clean, one holding a holdout.
        for snap_id, inputs in ((SNAP_OK, [self.shas[f"orig_{n:02d}.png"] for n in range(3, 16)]),
                                (SNAP_BAD, [self.shas["orig_01.png"]])):
            base = self.project.dir / "seed_lora" / snap_id
            base.mkdir(parents=True, exist_ok=True)
            (base / "snapshot.json").write_text(json.dumps(
                {"id": snap_id, "inputs": [{"sha256": sha} for sha in inputs], "holdout_ids": []}), encoding="utf-8")


class FinalPolicyTests(unittest.TestCase):
    """The two audits over one fixture: the seed stays 27, the final set is 33, and every refusal is named."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fx = Fixture(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def audit(self, **kwargs):
        return seed.audit(self.fx.project, **kwargs)

    def test_the_seed_keeps_its_own_meaning(self):
        report = self.audit()
        self.assertEqual(report["policy"], "seed-v2")
        self.assertEqual(report["counts"]["kept"], 27)
        self.assertEqual(report["counts"]["kept_sources"], 14)
        self.assertEqual(report["counts"]["kept_candidates"], 13)
        self.assertEqual(report["counts"]["by_code"].get("generator_not_eligible"), 11)

    def test_the_final_set_adds_the_plan_proven_batch_b_pictures(self):
        report = self.audit(purpose="final")
        self.assertEqual(report["policy"], "export-v2")
        self.assertEqual(report["counts"]["kept"], 33)
        self.assertEqual(report["counts"]["kept_sources"], 14)
        self.assertEqual(report["counts"]["kept_candidates"], 19)
        kept_ids = {entry["id"] for entry in report["kept"]}
        for candidate_id in self.fx.b_ids:
            self.assertIn(candidate_id, kept_ids)

    def test_the_final_refusals_are_named(self):
        codes = self.audit(purpose="final")["counts"]["by_code"]
        self.assertEqual(codes.get("plan_unproven"), 2)            # the old plan's picture + the stolen slot+seed
        self.assertEqual(codes.get("generation_cap"), 1)           # generation 3, past the final ceiling
        self.assertEqual(codes.get("holdout_via_lora"), 1)
        self.assertEqual(codes.get("lora_snapshot_unverified"), 1)
        self.assertEqual(codes.get("holdout_direct"), 1)
        self.assertEqual(codes.get("holdout_indirect"), 1)
        self.assertEqual(codes.get("cyclic_lineage"), 2)
        self.assertEqual(codes.get("missing_parent"), 1)
        self.assertEqual(codes.get("no_ancestry"), 1)
        self.assertEqual(codes.get("edit_depth_cap"), 1)
        self.assertEqual(codes.get("duplicate_bytes"), 1)
        self.assertEqual(codes.get("not_reviewed"), 1)
        self.assertEqual(codes.get("rejected"), 1)
        self.assertEqual(codes.get("different_character"), 1)
        self.assertEqual(codes.get("marked_for_fix"), 1)
        self.assertNotIn("generator_not_eligible", codes)          # the final list is wider, not narrower

    def test_a_slot_name_alone_is_never_membership(self):
        excluded = {row["id"]: row["code"] for row in self.audit(purpose="final")["excluded"]}
        self.assertEqual(excluded[self.fx.old_plan_id], "plan_unproven")
        self.assertEqual(excluded[self.fx.stolen_id], "plan_unproven")
        for row in self.audit(purpose="final")["excluded"]:
            if row["id"] == self.fx.old_plan_id:
                self.assertIn("no batch B task names slot", row["reason"])

    def test_the_seed_and_the_final_disagree_only_where_the_rules_differ(self):
        seed_kept = {entry["id"] for entry in self.audit()["kept"]}
        final_kept = {entry["id"] for entry in self.audit(purpose="final")["kept"]}
        self.assertEqual(final_kept - seed_kept, set(self.fx.b_ids))
        self.assertEqual(seed_kept - final_kept, set())

    def test_an_unknown_purpose_is_refused(self):
        with self.assertRaises(pj.ProjectError):
            self.audit(purpose="later")

    def test_a_bad_export_policy_is_refused(self):
        with self.assertRaises(pj.ProjectError):
            self.audit(purpose="final", policy="export-v1")

    def test_the_plan_proof_reads_the_real_manifest(self):
        record = self.fx.project.candidate_by_id(self.fx.b_ids[0])
        proven, detail = batch_gen.plan_proof(self.fx.project, record)
        self.assertTrue(proven, detail)
        self.assertIn("plan task", detail)


class CollectAndPublishTests(unittest.TestCase):
    """The frozen set: 33 pictures, complete or refused, and the boundary to training enforced."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fx = Fixture(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_collect_returns_the_final_set(self):
        entries, report, holdout, caps = export.collect(self.fx.project, log=lambda *_: None)
        self.assertEqual(len(entries), 33)
        self.assertEqual({entry["generator"] for entry in entries if entry["kind"] == "candidate"},
                         {"qwen_edit", "krea2_lora"})
        # The person's accept count and the exportable count are different numbers (spec R2): the quota board says
        # 3 accepts sit in frm_back, but only the 2 plan-proven ones can actually be exported.
        quotas = report["quotas"]
        self.assertEqual(quotas["frm_back"]["accepted"], 3)
        self.assertEqual(quotas["frm_back"]["eligible"], 2)

    def test_publish_writes_33_and_the_manifest_matches_the_disk(self):
        manifest = export.publish(self.fx.project, log=lambda *_: None)
        self.assertEqual(manifest["counts"]["images"], 33)
        self.assertEqual(manifest["policy"], "export-v2")
        self.assertEqual(export.validate(manifest["base"]), [])
        base = Path(manifest["base"])
        disk = sorted(p.name for p in base.iterdir() if p.suffix.lower() in export.IMAGE_SUFFIXES)
        self.assertEqual(disk, sorted(entry["file"] for entry in manifest["images"]))
        self.assertEqual(manifest["counts"]["refused"], 20)

    def test_publishing_twice_reuses_the_id(self):
        first = export.publish(self.fx.project, log=lambda *_: None)
        second = export.publish(self.fx.project, log=lambda *_: None)
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["reused"])

    def test_the_training_gate_passes_the_published_export(self):
        manifest = export.publish(self.fx.project, log=lambda *_: None)
        gate = export.training_gate(Path(manifest["base"]) / "export_manifest.json")
        self.assertEqual(gate["id"], manifest["id"])

    def tampered_copy(self, name):
        """A copy of the published export whose manifest names the copy itself, plus one change."""
        manifest = export.publish(self.fx.project, log=lambda *_: None)
        copy = Path(self.temp.name) / name
        shutil.copytree(manifest["base"], copy)
        data = json.loads((copy / "export_manifest.json").read_text(encoding="utf-8"))
        data["base"] = str(copy)
        data["dataset"] = str(copy)
        (copy / "export_manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return copy, data

    def test_a_tampered_published_copy_is_refused_by_the_gate(self):
        copy, _data = self.tampered_copy("copy")
        (copy / "unregistered.png").write_bytes(bytes_of(250))
        with self.assertRaises(export.ExportError) as caught:
            export.training_gate(copy / "export_manifest.json")
        self.assertIn("unlisted image", str(caught.exception))

    def test_a_tampered_copy_gets_no_training_job(self):
        copy, data = self.tampered_copy("copy2")
        (copy.parent / "todo.json").write_text("[]", encoding="utf-8")
        image = next(p for p in sorted(copy.iterdir()) if p.suffix.lower() in export.IMAGE_SUFFIXES)
        image.with_suffix(".txt").write_text("a swapped caption\n", encoding="utf-8")
        with self.assertRaises(export.ExportError) as caught:
            export.add_todos(self.fx.project, data, automation_dir=copy.parent, log=lambda *_: None)
        self.assertIn("not intact", str(caught.exception))
        self.assertEqual(json.loads((copy.parent / "todo.json").read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
