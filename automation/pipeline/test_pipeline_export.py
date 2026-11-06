"""M4: the frozen export and the reports that judge it. What is tested is what makes the set trustworthy.

The properties the spec names, each with its own group:

* **only eligible pictures** - the export calls the same audit the seed does, so it cannot disagree with it about
  generation caps, holdout separation or lineage, and a picture still awaiting review never gets in;
* **immutable** - the same set always produces the same id, a changed caption produces a different one, and a published
  export is verified and reused rather than rewritten;
* **traceable** - the manifest carries each image's id, sha256 and caption sha256, the rules that refused the others,
  and the distributions a person checks before training.
"""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import export, evaluate
from pipeline.scorers import common

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


class FakeProject:
    """The parts of `Project` the export uses: a folder, a card, a digest, and a guard."""

    def __init__(self, root, card=None):
        self.dir = Path(root)
        self.card = card or {
            "id": "suamom", "trigger": "suamom", "caption_base": "korean webtoon style",
            "caption_traits": "omit", "fixed_traits": [], "outfits": {"default": "dress"},
            "lineage": {"max_train_generation": 2, "seed_max_generation": 1, "max_edit_depth": 1},
            "final_families": ["qwen_image21", "krea2"], "slot_quotas": {},
        }
        self.dir.mkdir(parents=True, exist_ok=True)

    def digest(self):
        return "card-digest"

    def holdout_entry(self):
        return {"ids": [], "shas": [], "files": []}


def fake_audit(kept, excluded=(), by_code=None):
    return {
        "kept": list(kept), "excluded": list(excluded),
        "counts": {"by_code": by_code or {}, "by_category": {}},
        "quotas": {},
    }


class CollectTests(unittest.TestCase):
    """What goes into an export: the originals as well as the generated pictures."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = FakeProject(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def patch(self, kept, records=(), captions=None):
        """Run `collect` against a fixed audit, so the eligibility rules are not what is under test here."""
        orig_audit, orig_tool2, orig_records = export.seed.audit, export.seed.tool2_captions, None
        export.seed.audit = lambda p, caps=None: fake_audit(kept)
        export.seed.tool2_captions = lambda folder: (captions or {})
        self.project.candidates = lambda: list(records)
        try:
            return export.collect(self.project, log=lambda *_: None)
        finally:
            export.seed.audit, export.seed.tool2_captions = orig_audit, orig_tool2

    def test_the_user_originals_are_included_alongside_generated_pictures(self):
        # A set built only from generated pictures teaches the character from its own output; the spec lists
        # candidate *and* source ids for a reason.
        kept = [
            {"id": "s_aaa", "kind": "source", "origin": "user", "sha256": "a" * 64, "path": "/x/a.png",
             "generation": 0, "edit_depth": 0, "slot": "", "lineage_ids": ["s_aaa"]},
            {"id": "c_bbb", "kind": "candidate", "origin": "synthetic", "sha256": "b" * 64, "path": "/x/b.png",
             "generation": 1, "edit_depth": 0, "slot": "frm_face", "lineage_ids": ["c_bbb"]},
        ]
        entries, _report, _holdout, _caps = self.patch(kept, records=[{"id": "c_bbb", "generator": "qwen_edit"}])
        self.assertEqual({e["id"] for e in entries}, {"s_aaa", "c_bbb"})
        self.assertEqual({e["kind"] for e in entries}, {"source", "candidate"})

    def test_a_generated_picture_records_which_generator_made_it(self):
        kept = [{"id": "c_bbb", "kind": "candidate", "sha256": "b" * 64, "path": "/x/b.png",
                 "generation": 1, "edit_depth": 0, "slot": "frm_face", "lineage_ids": ["c_bbb"]}]
        entries, _r, _h, _c = self.patch(kept, records=[{"id": "c_bbb", "generator": "qwen_edit"}])
        self.assertEqual(entries[0]["generator"], "qwen_edit")

    def test_an_original_has_no_generator(self):
        kept = [{"id": "s_aaa", "kind": "source", "sha256": "a" * 64, "path": "/x/a.png",
                 "generation": 0, "edit_depth": 0, "slot": "", "lineage_ids": ["s_aaa"]}]
        entries, _r, _h, _c = self.patch(kept)
        self.assertEqual(entries[0]["generator"], "")

    def test_every_entry_carries_its_caption_hash(self):
        kept = [{"id": "c_bbb", "kind": "candidate", "sha256": "b" * 64, "path": "/x/b.png",
                 "generation": 1, "edit_depth": 0, "slot": "s", "lineage_ids": ["c_bbb"],
                 "caption": "a caption", "caption_source": "own"}]
        entries, _r, _h, _c = self.patch(kept)
        self.assertEqual(len(entries[0]["caption_sha256"]), 64)
        self.assertEqual(entries[0]["caption"], "a caption")

    def test_a_picture_without_a_caption_gets_the_cards_under_the_omit_policy(self):
        kept = [{"id": "c_bbb", "kind": "candidate", "sha256": "b" * 64, "path": "/x/b.png",
                 "generation": 1, "edit_depth": 0, "slot": "s", "lineage_ids": ["c_bbb"]}]
        entries, _r, _h, _c = self.patch(kept)
        self.assertEqual(entries[0]["caption"], "suamom, korean webtoon style")
        self.assertEqual(entries[0]["caption_source"], "built from the card")


class IdentityTests(unittest.TestCase):
    """The id: same set, same id; any change, a new id."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = FakeProject(self.temp.name)
        self.caps = {"max_train_generation": 2, "seed_max_generation": 1, "max_edit_depth": 1}
        self.holdout = {"ids": [], "shas": []}

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, image="a" * 64, caption="b" * 64, slot="frm_face", ident="c_1"):
        return {"id": ident, "lineage_ids": [ident], "sha256": image, "caption_sha256": caption, "slot": slot}

    def test_the_same_set_gives_the_same_id(self):
        one = [self.entry()]
        self.assertEqual(export.identity(self.project, one, caps=self.caps, holdout=self.holdout),
                         export.identity(self.project, one, caps=self.caps, holdout=self.holdout))

    def test_a_different_image_gives_a_different_id(self):
        self.assertNotEqual(
            export.identity(self.project, [self.entry(image="a" * 64)], caps=self.caps, holdout=self.holdout),
            export.identity(self.project, [self.entry(image="d" * 64)], caps=self.caps, holdout=self.holdout))

    def test_a_changed_caption_gives_a_different_id(self):
        self.assertNotEqual(
            export.identity(self.project, [self.entry(caption="b" * 64)], caps=self.caps, holdout=self.holdout),
            export.identity(self.project, [self.entry(caption="e" * 64)], caps=self.caps, holdout=self.holdout))

    def test_a_changed_slot_gives_a_different_id(self):
        self.assertNotEqual(
            export.identity(self.project, [self.entry(slot="frm_face")], caps=self.caps, holdout=self.holdout),
            export.identity(self.project, [self.entry(slot="pos_sitting")], caps=self.caps, holdout=self.holdout))

    def test_an_added_picture_gives_a_different_id(self):
        one = [self.entry()]
        two = one + [self.entry(ident="c_2", image="f" * 64)]
        self.assertNotEqual(export.identity(self.project, one, caps=self.caps, holdout=self.holdout),
                            export.identity(self.project, two, caps=self.caps, holdout=self.holdout))

    def test_a_changed_holdout_gives_a_different_id(self):
        one = [self.entry()]
        self.assertNotEqual(
            export.identity(self.project, one, caps=self.caps, holdout={"ids": [], "shas": []}),
            export.identity(self.project, one, caps=self.caps, holdout={"ids": [], "shas": ["f" * 64]}))


class DistributionTests(unittest.TestCase):
    """The counts a person checks before training."""

    def test_generated_and_original_are_counted_by_kind(self):
        entries = [{"kind": "source", "generation": 0, "slot": "", "generator": ""},
                   {"kind": "candidate", "generation": 1, "slot": "frm_face", "generator": "qwen_edit"},
                   {"kind": "candidate", "generation": 1, "slot": "frm_face", "generator": "krea2_lora"}]
        result = export.distributions(entries)
        self.assertEqual(result["generated"], 2)
        self.assertEqual(result["original"], 1)
        self.assertAlmostEqual(result["generated_share"], 2 / 3)

    def test_a_missing_kind_is_not_treated_as_an_original(self):
        result = export.distributions([{"generation": 1, "slot": "s"}])
        self.assertEqual(result["original"], 0)
        self.assertEqual(result["generated"], 1)

    def test_generations_and_slots_are_counted(self):
        entries = [{"kind": "candidate", "generation": 1, "slot": "a"}, {"kind": "candidate", "generation": 1, "slot": "a"},
                   {"kind": "candidate", "generation": 2, "slot": "b"}]
        result = export.distributions(entries)
        self.assertEqual(result["by_generation"], {"1": 2, "2": 1})
        self.assertEqual(result["by_slot"], {"a": 2, "b": 1})

    def test_generators_are_counted(self):
        entries = [{"kind": "candidate", "generation": 1, "slot": "a", "generator": "qwen_edit"},
                   {"kind": "candidate", "generation": 1, "slot": "a", "generator": "qwen_edit"},
                   {"kind": "source", "generation": 0, "slot": "", "generator": ""}]
        result = export.distributions(entries)
        self.assertEqual(result["by_generator"]["qwen_edit"], 2)
        self.assertEqual(result["by_generator"]["(original)"], 1)


class ValidateTests(unittest.TestCase):
    """An export that was changed after it was frozen must be reported, not silently accepted."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "exp"
        self.base.mkdir()
        image = self.base / "abcdef.png"
        image.write_bytes(PNG)
        image.with_suffix(".txt").write_text("a caption\n", encoding="utf-8")
        import hashlib
        self.manifest = {
            "id": "x", "images": [{
                "id": "c_1", "file": "abcdef.png", "sha256": hashlib.sha256(PNG).hexdigest(),
                "caption_sha256": hashlib.sha256(b"a caption").hexdigest(),
            }],
        }
        (self.base / "export_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_an_intact_export_has_no_problems(self):
        self.assertEqual(export.validate(self.base), [])

    def test_a_changed_image_is_reported(self):
        (self.base / "abcdef.png").write_bytes(PNG + b"tampered")
        problems = export.validate(self.base)
        self.assertTrue(any("no longer hashes" in p for p in problems), problems)

    def test_a_changed_caption_is_reported(self):
        (self.base / "abcdef.txt").write_text("a different caption\n", encoding="utf-8")
        problems = export.validate(self.base)
        self.assertTrue(any("caption of" in p for p in problems), problems)

    def test_a_deleted_image_is_reported(self):
        (self.base / "abcdef.png").unlink()
        problems = export.validate(self.base)
        self.assertTrue(any("missing image" in p for p in problems), problems)

    def test_a_deleted_caption_is_reported(self):
        (self.base / "abcdef.txt").unlink()
        problems = export.validate(self.base)
        self.assertTrue(any("missing caption" in p for p in problems), problems)

    def test_no_manifest_is_reported(self):
        (self.base / "export_manifest.json").unlink()
        self.assertTrue(export.validate(self.base))


class ScorerTests(unittest.TestCase):
    """The measurements that need no model."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, colour):
        from PIL import Image
        path = self.root / name
        Image.new("RGB", (64, 64), colour).save(path)
        return path

    def test_exact_duplicates_are_grouped_by_hash(self):
        rows = [{"id": "a", "sha256": "x"}, {"id": "b", "sha256": "x"}, {"id": "c", "sha256": "y"}]
        result = common.sha_groups(rows)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["ids"], ["a", "b"])
        self.assertEqual(result["coverage"]["redundant"], 1)

    def test_identical_pictures_land_in_one_near_duplicate_group(self):
        first = self.image("a.png", (120, 80, 60))
        second = self.image("b.png", (120, 80, 60))
        result = common.phash_groups([{"id": "a", "path": str(first)}, {"id": "b", "path": str(second)}])
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["count"], 2)

    def test_very_different_pictures_do_not_group(self):
        plain = self.image("plain.png", (255, 255, 255))
        banded = self.root / "banded.png"
        from PIL import Image
        im = Image.new("RGB", (64, 64), (0, 0, 0))
        for x in range(0, 64, 8):
            for y in range(64):
                for dx in range(4):
                    im.putpixel((min(63, x + dx), y), (255, 255, 255))
        im.save(banded)
        result = common.phash_groups([{"id": "a", "path": str(plain)}, {"id": "b", "path": str(banded)}])
        self.assertEqual(result["rows"], [])

    def test_a_white_frame_measures_high_and_a_dark_one_low(self):
        white = self.image("w.png", (255, 255, 255))
        dark = self.image("d.png", (10, 10, 10))
        self.assertGreater(common.white_ratio(white), 0.95)
        self.assertLess(common.white_ratio(dark), 0.05)

    def test_an_unreadable_image_is_reported_not_dropped(self):
        results = common.white_scores([{"id": "x", "path": str(self.root / "nope.png")}])
        self.assertEqual(results["rows"], [])
        self.assertEqual(len(results["coverage"]["unreadable"]), 1)

    def test_slot_fill_counts_short_and_over(self):
        result = common.slot_fill({"a": 2, "b": 1}, {"a": 1, "b": 3})
        rows = {r["slot"]: r for r in result["rows"]}
        self.assertEqual(rows["a"]["short"], 1)
        self.assertEqual(rows["b"]["over"], 2)
        self.assertEqual(result["coverage"]["short"], 1)


class EvaluateTests(unittest.TestCase):
    """Agreement with a person, and the holdout report."""

    def test_the_weighted_mark_matches_m0(self):
        row = {"identity": "5", "instruction": "4", "quality": "3"}
        self.assertAlmostEqual(evaluate.weighted_human(row), 5 * 0.45 + 4 * 0.30 + 3 * 0.25)

    def test_an_incomplete_row_has_no_mark(self):
        self.assertIsNone(evaluate.weighted_human({"identity": "5", "instruction": "", "quality": "3"}))

    def test_side_and_back_slots_are_recognised_as_hard_views(self):
        self.assertTrue(evaluate.is_hard_view("ang_back_medium"))
        self.assertTrue(evaluate.is_hard_view("frm_profile"))
        self.assertFalse(evaluate.is_hard_view("frm_face"))
        self.assertFalse(evaluate.is_hard_view("exp_smile"))

    def test_spearman_is_one_for_identical_orders(self):
        self.assertAlmostEqual(evaluate.spearman([(1, 10), (2, 20), (3, 30), (4, 40)]), 1.0)

    def test_spearman_is_minus_one_for_reversed_orders(self):
        self.assertAlmostEqual(evaluate.spearman([(1, 40), (2, 30), (3, 20), (4, 10)]), -1.0)

    def test_quartiles_need_at_least_four_points(self):
        self.assertIsNone(evaluate.quartiles([1, 2, 3]))
        self.assertIsNotNone(evaluate.quartiles([1, 2, 3, 4]))

    def test_agreement_separates_hard_views_from_easy_ones(self):
        scores = [
            {"task": "t1", "slot": "frm_face", "human": 5.0, "rejected": False},
            {"task": "t2", "slot": "frm_face", "human": 4.0, "rejected": False},
            {"task": "t3", "slot": "frm_face", "human": 3.0, "rejected": False},
            {"task": "t4", "slot": "frm_face", "human": 2.0, "rejected": False},
            {"task": "t5", "slot": "ang_back_medium", "human": 5.0, "rejected": False},
            {"task": "t6", "slot": "ang_back_medium", "human": 4.0, "rejected": False},
            {"task": "t7", "slot": "ang_back_medium", "human": 3.0, "rejected": False},
            {"task": "t8", "slot": "ang_back_medium", "human": 2.0, "rejected": False},
        ]
        auto = {"t1": 0.9, "t2": 0.8, "t3": 0.5, "t4": 0.2, "t5": 0.4, "t6": 0.3, "t7": 0.2, "t8": 0.1}
        report = evaluate.agreement(scores, auto)
        self.assertEqual(report["paired"], 8)
        self.assertEqual(report["easy_views"]["pairs"], 4)
        self.assertEqual(report["hard_views"]["pairs"], 4)

    def test_a_row_without_an_automatic_score_is_counted_as_unpaired(self):
        scores = [{"task": "t1", "slot": "frm_face", "human": 5.0, "rejected": False},
                  {"task": "t2", "slot": "frm_face", "human": 4.0, "rejected": False}]
        report = evaluate.agreement(scores, {"t1": 0.9})
        self.assertEqual(report["paired"], 1)
        self.assertEqual(report["unpaired_count"], 1)

    def test_no_pairs_at_all_is_refused_rather_than_reported_as_zero(self):
        with self.assertRaises(evaluate.EvalError):
            evaluate.agreement([{"task": "t1", "slot": "x", "human": None}], {})

    def test_a_rejected_picture_scoring_high_is_named(self):
        scores = [{"task": f"t{i}", "slot": "frm_face", "human": 2.0, "rejected": True} for i in range(4)]
        auto = {f"t{i}": 0.9 - i * 0.01 for i in range(4)}
        report = evaluate.agreement(scores, auto)
        self.assertTrue(report["overall"]["rejected_but_scored_high"])

    def test_the_holdout_report_names_its_references(self):
        scores = {"scorer": "clip_vision", "rows": [{"id": "a", "max_cosine": 0.7}, {"id": "b", "max_cosine": 0.9}]}
        report = evaluate.holdout_family_report("krea2", scores, references=["/ref/one.png"])
        self.assertEqual(report["family"], "krea2")
        self.assertEqual(report["images"], 2)
        self.assertAlmostEqual(report["max"], 0.9)
        self.assertEqual(report["references"], ["/ref/one.png"])

    def test_an_unscored_image_is_not_counted_as_zero(self):
        scores = {"scorer": "arcface", "rows": [{"id": "a", "max_cosine": None}, {"id": "b", "max_cosine": 0.8}]}
        report = evaluate.holdout_family_report("krea2", scores, references=["/r.png"])
        self.assertEqual(report["images"], 1)
        self.assertAlmostEqual(report["mean"], 0.8)
        self.assertEqual(report["unscored"], ["a"])
        self.assertIn("no score", report["note"])

    def test_a_holdout_report_with_no_references_is_refused(self):
        with self.assertRaises(evaluate.EvalError):
            evaluate.holdout_family_report("krea2", {"scorer": "x", "rows": []}, references=[])

    def test_comparing_families_warns_when_the_scorers_differ(self):
        reports = [{"family": "krea2", "scorer": "clip_vision", "images": 2, "max": .9, "mean": .8, "median": .8},
                   {"family": "qwen_image21", "scorer": "arcface", "images": 2, "max": .5, "mean": .5, "median": .5}]
        table = evaluate.compare_families(reports)
        self.assertEqual(len(table["families"]), 2)
        self.assertIn("not directly comparable", table["note"])

    def test_one_scorer_across_families_needs_no_warning(self):
        reports = [{"family": "a", "scorer": "clip_vision", "images": 1, "max": .9, "mean": .9, "median": .9},
                   {"family": "b", "scorer": "clip_vision", "images": 1, "max": .8, "mean": .8, "median": .8}]
        self.assertIsNone(evaluate.compare_families(reports)["note"])


class TodoTests(unittest.TestCase):
    """The step after an export: a training job per final family, added disabled.

    Adding it enabled would start a run that costs hours of GPU time on the strength of an export the user may still be
    reviewing, so "added but off" is the only safe default.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.automation = self.root / "automation"
        self.automation.mkdir()
        (self.automation / "todo.json").write_text("[]", encoding="utf-8")
        self.project = FakeProject(self.root / "projects" / "p", card={"id": "p", "trigger": "suamom"})
        self.manifest = {"id": "abc123", "families": ["krea2", "qwen_image21"],
                         "base": str(self.root / "exports" / "abc123"),
                         "dataset": str(self.root / "exports" / "abc123")}

    def tearDown(self):
        self.temp.cleanup()

    def test_one_job_per_family_and_no_dataset_root_guessing(self):
        jobs = export.plan_todos(self.project, self.manifest, automation_dir=self.automation)
        self.assertEqual(len(jobs), 2)
        self.assertEqual({job["family"] for job in jobs}, {"krea2", "qwen_image21"})
        self.assertTrue(all(job["dataset_root"].endswith("abc123") for job in jobs))

    def test_the_job_id_carries_the_export_so_a_new_export_is_a_new_job(self):
        jobs = export.plan_todos(self.project, self.manifest, automation_dir=self.automation)
        self.assertIn("abc123", jobs[0]["id"])
        other = dict(self.manifest, id="def456")
        self.assertNotEqual(jobs[0]["id"],
                            export.plan_todos(self.project, other, automation_dir=self.automation)[0]["id"])

    def test_adding_writes_them_disabled(self):
        report = export.add_todos(self.project, self.manifest, automation_dir=self.automation, log=lambda *_: None)
        self.assertEqual(len(report["added"]), 2)
        written = json.loads((self.automation / "todo.json").read_text(encoding="utf-8"))
        self.assertEqual(len(written), 2)
        self.assertTrue(all(job["enabled"] is False for job in written))

    def test_adding_the_same_export_twice_adds_nothing(self):
        export.add_todos(self.project, self.manifest, automation_dir=self.automation, log=lambda *_: None)
        again = export.add_todos(self.project, self.manifest, automation_dir=self.automation, log=lambda *_: None)
        self.assertEqual(again["added"], [])
        written = json.loads((self.automation / "todo.json").read_text(encoding="utf-8"))
        self.assertEqual(len(written), 2)

    def test_an_existing_unrelated_job_is_kept(self):
        (self.automation / "todo.json").write_text(
            json.dumps([{"id": "eunbi_waterbomb", "enabled": True, "trigger": "eunbiwb", "seed": 42}]),
            encoding="utf-8")
        export.add_todos(self.project, self.manifest, automation_dir=self.automation, log=lambda *_: None)
        written = json.loads((self.automation / "todo.json").read_text(encoding="utf-8"))
        self.assertEqual(len(written), 3)
        self.assertEqual(written[0]["id"], "eunbi_waterbomb")

    def test_enable_is_opt_in(self):
        export.add_todos(self.project, self.manifest, automation_dir=self.automation, enable=True,
                         log=lambda *_: None)
        written = json.loads((self.automation / "todo.json").read_text(encoding="utf-8"))
        self.assertTrue(all(job["enabled"] is True for job in written))

    def test_no_family_is_refused(self):
        with self.assertRaises(export.ExportError):
            export.plan_todos(self.project, dict(self.manifest, families=[]), automation_dir=self.automation)


if __name__ == "__main__":
    unittest.main()
