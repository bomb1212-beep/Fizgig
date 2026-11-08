"""N4: a holdout measurement is complete, separated, and honest about which model it judges.

The audit left four reproductions here, and each one is a test below:

* `holdout_integrity` - a plan of four cells with two recorded, one of them unhashed, and two byte-different PNGs with
  identical pixels: the old reader accepted all of it and returned `missing=[]`, `duplicate_pixels=[]`.
* `holdout_orphan_resume` - a picture in the folder with no manifest entry crashed the resume path with
  `AttributeError: 'NoneType' object has no attribute 'get'`.
* `holdout_qwen` - `family=qwen_image21` was an unconditional "not wired" error, and nothing named which final LoRA a
  measurement was supposed to judge.
* `review_references` - the ordinary candidate score used fifteen originals, two of which were the frozen holdout.

Nothing here draws a picture with a model: the clients are fakes, and the reference-set tests use a temporary project.
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AUTO = Path(__file__).resolve().parents[1]
for entry in (str(AUTO.parent / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import evaluate, score  # noqa: E402


def png(path, colour=(10, 20, 30), *, size=(8, 8)):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Client:
    """A ComfyUI stand-in: writes a real picture and counts the submissions."""

    url = "http://fake"

    def __init__(self, *, identical=False, fail=False):
        self.runs = 0
        self.identical = identical
        self.fail = fail

    def ping(self):
        return True

    def name_for(self, kind, path):
        return kind

    def run(self, graph, out, slot_id=None, method=None, seed=None, **_meta):
        if self.fail:
            raise RuntimeError("ComfyUI said no")
        self.runs += 1
        colour = (9, 9, 9) if self.identical else (9, 9, int(seed) % 200)
        png(out, colour)
        return out


class _Project:
    """A real `Project` with a frozen holdout, a chosen checkpoint and `final_families`."""

    def __init__(self, root, *, final_families=("krea2", "qwen_image21"), choose=True, holdouts=2):
        from pipeline import project as pj
        self.root = Path(root)
        self.checkpoint = self.root / "char-000030.safetensors"
        self.checkpoint.write_bytes(b"safetensors-" * 8)
        card = {"id": "fx", "trigger": "fx", "caption_base": "style", "caption_traits": "omit",
                "final_families": list(final_families), "holdout": {"ids": []},
                "krea2_inference": {"size": {"width": 64, "height": 96}}}
        if choose:
            card["krea2_inference"]["selected"] = {
                "run_id": "fx_seed-r1", "epoch": 30, "checkpoint_path": str(self.checkpoint),
                "checkpoint_sha256": sha(self.checkpoint)}
        self.project = pj.Project.create("fx", card, folder=self.root / "proj")
        incoming = self.root / "incoming"
        incoming.mkdir(parents=True)
        names = [f"hold_{i}.png" for i in range(holdouts)] + [f"plain_{i}.png" for i in range(3)]
        for index, name in enumerate(names):
            # A distinct colour per file: `import_sources` identity is the bytes, so two files with one picture would
            # collapse into a single source and the fixture would not have the holdout it claims.
            png(incoming / name, (index * 30 + 5, 60, 90 - index * 10))
        self.project.import_sources(sorted(incoming.iterdir()), log=lambda *_: None)
        records = {r["id"]: r for r in self.project.source_records()}
        ids = [r["id"] for r in records.values() if Path(r["src_path"]).name.startswith("hold_")]
        self.project.card["holdout"] = {"ids": ids}
        self.project.save()

    def __getattr__(self, name):
        return getattr(self.project, name)


class ReferenceSeparationTests(unittest.TestCase):
    """spec N4.5 `review_references`: the holdout is not a reference for candidate selection."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = _Project(self.temp.name)

    def test_the_general_reference_set_excludes_the_frozen_holdout(self):
        refs = score.reference_paths(self.fixture.project)
        holdout_ids = set(self.fixture.project.holdout_entry()["ids"])
        self.assertEqual(len(holdout_ids), 2)
        holdout_files = {Path(self.fixture.project.resolve_file(
            (self.fixture.project.index()[rid] or {}).get("file"))).resolve() for rid in holdout_ids}
        self.assertEqual(len(refs), 3, "three user originals are not the holdout")
        for path in refs:
            self.assertNotIn(Path(path).resolve(), holdout_files)

    def test_the_holdout_purpose_returns_exactly_those_two(self):
        refs = score.reference_paths(self.fixture.project, purpose="holdout")
        holdout = self.fixture.project.holdout_entry()
        self.assertEqual(len(refs), 2)
        # `import_sources` stores a copy named for its content hash, so the record's `file` is the authority, not the
        # name of the file that was fed in.
        expected = {Path(self.fixture.project.resolve_file(
            (self.fixture.project.index()[rid] or {}).get("file"))).resolve() for rid in holdout["ids"]}
        self.assertEqual({Path(p).resolve() for p in refs}, expected)

    def test_a_copied_holdout_under_another_id_cannot_come_back(self):
        # A second record pointing at the same bytes is the same picture; excluding by id alone would let it through.
        import shutil
        holdout = self.fixture.project.holdout_entry()
        record = self.fixture.project.index()[holdout["ids"][0]]
        source = Path(record["src_path"])
        copy = source.parent / "copy_of_holdout.png"
        shutil.copy2(source, copy)
        self.fixture.project.import_sources([copy], log=lambda *_: None)
        refs = {Path(p).resolve() for p in score.reference_paths(self.fixture.project)}
        self.assertNotIn(copy.resolve(), refs)


class CellAuditTests(unittest.TestCase):
    """spec N4.1/N4.2: the plan's cells versus what the folder can vouch for."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = _Project(self.temp.name)
        self.project = self.fixture.project
        self.folder = evaluate.holdout_dir(self.project, "krea2")
        # `draw_holdout` reads the prompts back from the manifest on a resume, and needs a real prompt string.
        self.prompts = [{"id": "s_a", "prompt": "fx, style, standing"},
                        {"id": "s_b", "prompt": "fx, style, sitting"}]
        self.manifest = {
            "kind": "holdout", "family": "krea2", "purpose": "seed",
            "pin": {"purpose": "seed", "checkpoint_sha256": sha(self.fixture.checkpoint)},
            "prompts": self.prompts, "seeds": [7001, 7002],
            "size": {"width": 8, "height": 8},
            "cells": [],
        }

    def write_manifest(self):
        evaluate.write_holdout_manifest(self.project, "krea2", self.manifest, purpose="seed")

    def test_two_recorded_cells_of_four_are_reported_as_a_partial_measurement(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        one = png(self.folder / "s_a__s7001.png", (1, 2, 3))
        two = png(self.folder / "s_b__s7001.png", (4, 5, 6))
        self.manifest["cells"] = [{"id": "s_a", "seed": 7001, "file": one.name, "sha256": sha(one)},
                                  {"id": "s_b", "seed": 7001, "file": two.name, "sha256": sha(two)}]
        self.write_manifest()
        pairs, audit = evaluate.holdout_cells(self.project, "krea2", purpose="seed")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(audit["expected"], 4)
        self.assertEqual(audit["scored"], 2)
        self.assertFalse(audit["complete"])
        self.assertEqual(len(audit["missing"]), 2, "the two cells the plan never got are named")
        self.assertTrue(all("never recorded" in name for name in audit["missing"]))

    def test_a_cell_with_no_recorded_sha_is_not_scored(self):
        # `missing_sha_accepted` in the audit: a file with no hash cannot be told from one somebody dropped in.
        self.folder.mkdir(parents=True, exist_ok=True)
        one = png(self.folder / "s_a__s7001.png", (7, 8, 9))
        self.manifest["cells"] = [{"id": "s_a", "seed": 7001, "file": one.name}]
        self.write_manifest()
        pairs, audit = evaluate.holdout_cells(self.project, "krea2", purpose="seed")
        self.assertEqual(pairs, [])
        self.assertEqual(audit["no_sha"], [one.name])
        self.assertFalse(audit["complete"])

    def test_two_files_with_one_picture_are_counted_once(self):
        # `holdout_integrity`: two PNGs with different bytes and identical pixels.
        self.folder.mkdir(parents=True, exist_ok=True)
        one = png(self.folder / "s_a__s7001.png", (11, 12, 13))
        twin = self.folder / "s_b__s7001.png"
        # Same pixels, different bytes: re-encode through the same RGB grid with different compression settings.
        from PIL import Image
        with Image.open(one) as image:
            image.save(twin, compress_level=0)
        self.assertNotEqual(sha(one), sha(twin), "the bytes differ, so the old byte-hash saw two pictures")
        self.manifest["cells"] = [{"id": "s_a", "seed": 7001, "file": one.name, "sha256": sha(one)},
                                  {"id": "s_b", "seed": 7001, "file": twin.name, "sha256": sha(twin)}]
        self.write_manifest()
        pairs, audit = evaluate.holdout_cells(self.project, "krea2", purpose="seed")
        self.assertEqual(len(pairs), 1, "one picture is one measurement")
        self.assertEqual(len(audit["duplicate_pixels"]), 1, "and the twin is named as the duplicate")

    def test_a_file_no_manifest_records_is_never_trusted(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        png(self.folder / "someone-elses.png", (1, 1, 1))
        self.write_manifest()
        _pairs, audit = evaluate.holdout_cells(self.project, "krea2", purpose="seed")
        self.assertEqual(audit["unlisted"], ["someone-elses.png"])
        self.assertFalse(audit["complete"])

    def test_the_report_says_a_partial_grid_is_partial(self):
        scores = {"scorer": "clip_vision", "rows": [{"id": "s_a__s7001", "max_cosine": 0.7}]}
        report = evaluate.holdout_family_report("krea2", scores, references=["r.png"],
                                                cells_audit={"expected": 4, "scored": 1, "complete": False},
                                                purpose="seed")
        self.assertEqual(report["purpose"], "seed")
        self.assertTrue(report["partial"])
        self.assertIn("partial measurement", report["note"])


class OrphanResumeTests(unittest.TestCase):
    """spec N4.2 `holdout_orphan_resume`: an unrecorded picture must not crash the resume path."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = _Project(self.temp.name)
        self.project = self.fixture.project
        self.folder = evaluate.holdout_dir(self.project, "krea2")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest = {"kind": "holdout", "family": "krea2", "purpose": "seed",
                         "pin": {"purpose": "seed", "checkpoint_sha256": sha(self.fixture.checkpoint)},
                         "prompts": [{"id": "s_a", "prompt": "fx, style, standing"},
                                     {"id": "s_b", "prompt": "fx, style, sitting"}],
                         "seeds": [7001],
                         "size": {"width": 8, "height": 8}, "cells": []}
        evaluate.write_holdout_manifest(self.project, "krea2", self.manifest, purpose="seed")

    def test_an_unrecorded_picture_is_redrawn_not_adopted(self):
        orphan = png(self.folder / "s_a__s7001.png", (3, 3, 3))
        self.assertTrue(orphan.is_file())
        client = _Client()
        drawn, skipped, failed = evaluate.draw_holdout(self.project, family="krea2", seeds=(7001,),
                                                       client=client, log=lambda *_: None)
        self.assertEqual(failed, [])
        self.assertEqual(skipped, 0, "a file no manifest records is not a cell this plan vouches for")
        self.assertEqual(len(drawn), 2)
        self.assertEqual(client.runs, 2)
        manifest = evaluate.read_holdout_manifest(self.project, "krea2", purpose="seed")
        self.assertTrue(all(cell["sha256"] for cell in manifest["cells"]))

    def test_a_cell_recorded_without_a_sha_is_drawn_again(self):
        png(self.folder / "s_a__s7001.png", (3, 3, 3))
        self.manifest["cells"] = [{"id": "s_a", "seed": 7001, "file": "s_a__s7001.png"}]
        evaluate.write_holdout_manifest(self.project, "krea2", self.manifest, purpose="seed")
        client = _Client()
        drawn, skipped, _failed = evaluate.draw_holdout(self.project, family="krea2", seeds=(7001,),
                                                       client=client, log=lambda *_: None)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(drawn), 2)


class PurposeTests(unittest.TestCase):
    """spec N4.3: seed, final and baseline are separate experiments, and none substitutes for another."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = _Project(self.temp.name)

    def test_the_seed_pin_comes_from_the_card_choice(self):
        pin = evaluate.holdout_pin(self.fixture.project, purpose="seed", family="krea2")
        self.assertEqual(pin["purpose"], "seed")
        self.assertEqual(pin["checkpoint_sha256"], sha(self.fixture.checkpoint))

    def test_a_final_measurement_needs_that_familys_own_checkpoint(self):
        # A seed checkpoint is not a final LoRA: substituting one would report a seed number as a final number.
        with self.assertRaises(evaluate.EvalError) as caught:
            evaluate.draw_holdout(self.fixture.project, family="qwen_image21", purpose="final",
                                  log=lambda *_: None)
        self.assertIn("--run", str(caught.exception))
        self.assertIn("--checkpoint", str(caught.exception))

    def test_a_final_measurement_pins_the_checkpoint_it_was_given(self):
        final = self.fixture.root / "final_qwen.safetensors"
        final.write_bytes(b"final-lora" * 4)
        pin = evaluate.holdout_pin(self.fixture.project, purpose="final", family="qwen_image21",
                                   run="qwen_runs/r1", checkpoint=str(final))
        self.assertEqual(pin["purpose"], "final")
        self.assertEqual(pin["family"], "qwen_image21")
        self.assertEqual(pin["run_id"], "r1")
        self.assertEqual(pin["checkpoint_sha256"], sha(final))

    def test_the_baseline_pin_has_no_character_lora(self):
        pin = evaluate.holdout_pin(self.fixture.project, purpose="baseline", family="krea2")
        self.assertIsNone(pin["character_lora"])
        self.assertEqual(pin["purpose"], "baseline")

    def test_each_purpose_gets_its_own_folder(self):
        seed = evaluate.holdout_dir(self.fixture.project, "krea2", purpose="seed")
        final = evaluate.holdout_dir(self.fixture.project, "krea2", purpose="final")
        baseline = evaluate.holdout_dir(self.fixture.project, "krea2", purpose="baseline")
        self.assertEqual(len({seed, final, baseline}), 3)
        self.assertEqual(seed, self.fixture.project.dir / "experiments" / "holdout" / "krea2",
                         "the historical seed folder keeps its path so the existing measurement stays put")

    def test_a_seed_measurement_cannot_be_requested_for_qwen(self):
        reason = evaluate.holdout_waiting_reason(self.fixture.project, "seed", "qwen_image21")
        self.assertIn("krea2", reason)

    def test_a_family_outside_final_families_is_named(self):
        fixture = _Project(tempfile.mkdtemp(dir=self.temp.name), final_families=("krea2",))
        reason = evaluate.holdout_waiting_reason(fixture.project, "final", "qwen_image21")
        self.assertIn("final families", reason)
        self.assertIn("qwen_image21", reason)

    def test_a_folder_drawn_for_another_purpose_is_refused(self):
        folder = evaluate.holdout_dir(self.fixture.project, "krea2", purpose="seed")
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"kind": "holdout", "family": "krea2", "purpose": "seed",
                    "pin": {"purpose": "seed", "checkpoint_sha256": sha(self.fixture.checkpoint)},
                    "prompts": [{"id": "s_a"}], "seeds": [7001], "cells": []}
        evaluate.write_holdout_manifest(self.fixture.project, "krea2", manifest, purpose="seed")
        # The final purpose looks in its own folder, so it starts clean rather than reading the seed cells.
        self.assertIsNone(evaluate.read_holdout_manifest(self.fixture.project, "krea2", purpose="final"))


class MeasurementTests(unittest.TestCase):
    """The number finally printed, over a fake embedder: the cells and the model's attribution must be exact."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = _Project(self.temp.name)
        self.project = self.fixture.project

    def test_a_full_two_by_two_grid_is_measured_and_every_cell_is_attributed(self):
        folder = evaluate.holdout_dir(self.project, "krea2")
        folder.mkdir(parents=True, exist_ok=True)
        prompts = [{"id": p["id"], "prompt": p["prompt"]} for p in evaluate.holdout_prompts(self.project)]
        seeds = [7001, 7002]
        cells = []
        for index, (row, seed) in enumerate((r, s) for r in prompts for s in seeds):
            path = png(folder / f'{row["id"]}__s{seed}.png', (10 * index, 40, 40))
            cells.append({"id": row["id"], "seed": seed, "file": path.name, "sha256": sha(path)})
        manifest = {"kind": "holdout", "family": "krea2", "purpose": "seed",
                    "pin": {"purpose": "seed", "checkpoint_sha256": sha(self.fixture.checkpoint)},
                    "prompts": prompts, "seeds": seeds, "cells": cells}
        evaluate.write_holdout_manifest(self.project, "krea2", manifest, purpose="seed")
        pairs, audit = evaluate.holdout_cells(self.project, "krea2", purpose="seed")
        self.assertEqual(len(pairs), 4)
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["scored"], audit["expected"])

        from pipeline.scorers import clip_vision
        with mock.patch.object(clip_vision, "identity_scores",
                               side_effect=lambda paths, refs, ids=None, embedder=None: {
                                   "scorer": "clip_vision",
                                   "rows": [{"id": i, "max_cosine": 0.5 + n / 100}
                                            for n, i in enumerate(ids or [])]}):
            scores, refs, _meta = evaluate.holdout_scores(self.project, pairs, family="krea2")
        self.assertEqual(len(refs), 2, "the holdout originals, and only they, are the references")
        report = evaluate.holdout_family_report("krea2", scores, references=refs, cells_audit=audit,
                                               purpose="seed")
        self.assertEqual(report["images"], 4)
        self.assertNotIn("partial", report)
        self.assertEqual(report["purpose"], "seed")


class PinDecidesTheModelTests(unittest.TestCase):
    """The pin has to decide which LoRA is loaded, not just what the record says (spec N4.3).

    Found while measuring the real final LoRA: `names_for_holdout` ignored the pin and asked
    `batch_gen.character_lora_path()`, which returns the **seed run's** LoRA. A `purpose="final"` draw therefore
    produced pictures from the seed checkpoint while its manifest recorded the final one - the record named one
    model and the pixels came from another.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def names(self, pin):
        from pipeline import evaluate as ev
        client = _Client()

        class Records:
            url = "http://fake"

            def __init__(self):
                self.loras = []

            def ping(self):
                return True

            def name_for(self, kind, path):
                if kind == "lora":
                    self.loras.append(str(path))
                return Path(path).name

        recorder = Records()
        with mock.patch.object(ev, "_client", return_value=recorder):
            out = ev.names_for_holdout(_Minimal(self.root), recorder, "krea2", pin=pin)
        return out, recorder.loras

    def test_a_final_pin_loads_that_checkpoint(self):
        checkpoint = self.root / "final.safetensors"
        checkpoint.write_bytes(b"lora")
        names, loras = self.names({"purpose": "final", "checkpoint_path": str(checkpoint)})
        self.assertEqual(names["character_lora"], "final.safetensors")
        self.assertIn(str(checkpoint), loras)

    def test_a_baseline_pin_loads_no_character_lora(self):
        names, loras = self.names({"purpose": "baseline", "character_lora": None})
        self.assertIsNone(names["character_lora"])
        self.assertTrue(all("character" not in path for path in loras))

    def test_a_pin_whose_checkpoint_is_gone_is_refused(self):
        with self.assertRaises(evaluate.EvalError) as caught:
            self.names({"purpose": "final", "checkpoint_path": str(self.root / "missing.safetensors")})
        self.assertIn("gone", str(caught.exception))

    def test_a_draw_with_no_pin_refuses_rather_than_guessing(self):
        with self.assertRaises(evaluate.EvalError) as caught:
            self.names(None)
        self.assertIn("no pinned checkpoint", str(caught.exception))


class _Minimal:
    """Just enough of a project for `names_for_holdout`: it reads nothing else."""

    def __init__(self, root):
        self.dir = Path(root)


if __name__ == "__main__":
    unittest.main()
