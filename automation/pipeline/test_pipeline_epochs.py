"""M3-2: the epoch grid. What is tested here is what makes the grid trustworthy rather than merely produced.

Three things decide that, and each has its own group below:

* a checkpoint counts as usable only when its own header says so - a run that was interrupted leaves files whose names
  look right and whose contents are truncated;
* the eval folder records run, checkpoint bytes and training snapshot as separate facts, because the spec asks for the
  distinction and because "epoch 20" alone means nothing if a run can be retrained;
* re-running the grid adds no GPU work, and choosing an epoch refuses a checkpoint whose bytes changed under it.
"""
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import epochs


def safetensors_bytes(epoch=None, *, payload=b"x" * 64, extra_meta=None):
    """A byte-accurate safetensors file: 8-byte length, JSON header, then the tensor payload."""
    meta = {"modelspec.architecture": "Krea-2/lora"}
    if epoch is not None:
        meta["ss_epoch"] = str(epoch)
    meta.update(extra_meta or {})
    header = {"__metadata__": meta, "lora.a": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(payload)]}}
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + payload


def tiny_card(**over):
    card = {"id": "suamom", "trigger": "suamom", "caption_base": "korean webtoon style",
            "fixed_traits": [], "outfits": {"default": "beige dress"},
            "krea2_inference": {"size": {"width": 1024, "height": 1536}}}
    card.update(over)
    return card


class FakeProject:
    """The parts of `Project` this module actually uses: a folder, a card, and the write guard."""

    def __init__(self, root, card=None):
        self.dir = Path(root)
        self.card = card or tiny_card()
        self.saved = 0

    def guard(self, *_a, **_k):
        import contextlib
        return contextlib.nullcontext()

    def save(self):
        self.saved += 1


class CheckpointTests(unittest.TestCase):
    """Which files in `output/` may be drawn with."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "output"
        self.out.mkdir()
        self.project = FakeProject(self.root / "proj")
        self.project.dir.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, data):
        path = self.out / name
        path.write_bytes(data)
        return path

    def test_a_saved_checkpoint_is_usable(self):
        self.write("char-000002.safetensors", safetensors_bytes(2))
        rows = epochs.checkpoint_rows(self.out)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["usable"])
        self.assertEqual(rows[0]["epoch"], 2)
        self.assertEqual(len(rows[0]["sha256"]), 64)

    def test_a_truncated_checkpoint_is_refused_with_the_reason(self):
        # What an interrupted run leaves behind: the right name, a header that starts, and no payload.
        good = safetensors_bytes(4)
        self.write("char-000004.safetensors", good[: len(good) // 2])
        rows = epochs.checkpoint_rows(self.out)
        self.assertFalse(rows[0]["usable"])
        self.assertIn("not a readable safetensors", rows[0]["note"])
        self.assertNotIn("sha256", rows[0])

    def test_a_name_that_disagrees_with_the_header_is_refused(self):
        # A file renamed by hand says one thing and contains another; drawing with it would mislabel every picture.
        self.write("char-000006.safetensors", safetensors_bytes(8))
        rows = epochs.checkpoint_rows(self.out)
        self.assertFalse(rows[0]["usable"])
        self.assertIn("the name says epoch 6, the header says 8", rows[0]["note"])

    def test_a_header_without_an_epoch_is_refused(self):
        self.write("char-000010.safetensors", safetensors_bytes(None))
        rows = epochs.checkpoint_rows(self.out)
        self.assertFalse(rows[0]["usable"])
        self.assertIn("no ss_epoch", rows[0]["note"])

    def test_a_file_named_without_an_epoch_uses_the_header(self):
        self.write("final.safetensors", safetensors_bytes(30))
        rows = epochs.checkpoint_rows(self.out)
        self.assertTrue(rows[0]["usable"])
        self.assertEqual(rows[0]["epoch"], 30)
        self.assertIn("name carries no epoch", rows[0]["note"])

    def test_usable_checkpoints_names_why_each_one_was_refused(self):
        self.write("a-000002.safetensors", b"")
        self.write("b-000004.safetensors", safetensors_bytes(6))
        with self.assertRaises(epochs.EpochError) as caught:
            epochs.usable_checkpoints(self.out)
        message = str(caught.exception)
        self.assertIn("no usable checkpoint", message)
        self.assertIn("a-000002", message)

    def test_a_missing_folder_is_an_error_not_an_empty_list(self):
        with self.assertRaises(epochs.EpochError):
            epochs.checkpoint_rows(self.root / "nowhere")


class PlanTests(unittest.TestCase):
    """The work list: every usable checkpoint, the same prompts, the same seeds."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "runs" / "char-abc123"
        (self.run / "output").mkdir(parents=True)
        self.project = FakeProject(self.root / "proj")
        self.project.dir.mkdir(parents=True)
        for epoch in (2, 4, 6):
            (self.run / "output" / f"char-{epoch:06d}.safetensors").write_bytes(safetensors_bytes(epoch))

    def tearDown(self):
        self.temp.cleanup()

    def test_every_usable_checkpoint_is_drawn_with_every_prompt_and_seed(self):
        manifest = epochs.plan(self.project, run_folder=self.run, log=lambda *_: None)
        self.assertEqual([c["epoch"] for c in manifest["checkpoints"]], [2, 4, 6])
        self.assertEqual(len(manifest["tasks"]),
                         len(epochs.EVAL_PROMPTS) * len(epochs.EVAL_SEEDS) * 3)
        self.assertEqual({t["prompt_id"] for t in manifest["tasks"]},
                         {p["id"] for p in epochs.EVAL_PROMPTS})

    def test_the_prompt_carries_the_character_card_first(self):
        manifest = epochs.plan(self.project, run_folder=self.run, log=lambda *_: None)
        task = manifest["tasks"][0]
        self.assertTrue(task["prompt"].startswith("suamom, korean webtoon style, beige dress,"),
                        task["prompt"])
        self.assertIn(task["prompt_id"] and "", task["prompt"])

    def test_naming_epochs_narrows_the_grid(self):
        manifest = epochs.plan(self.project, run_folder=self.run, epochs=[4], log=lambda *_: None)
        self.assertEqual([c["epoch"] for c in manifest["checkpoints"]], [4])
        self.assertEqual({t["epoch"] for t in manifest["tasks"]}, {4})

    def test_naming_an_epoch_that_is_not_usable_names_the_ones_that_are(self):
        with self.assertRaises(epochs.EpochError) as caught:
            epochs.plan(self.project, run_folder=self.run, epochs=[3], log=lambda *_: None)
        self.assertIn("available: 2, 4, 6", str(caught.exception))

    def test_the_size_comes_from_the_card_so_the_grid_measures_the_checkpoints(self):
        self.project.card["krea2_inference"]["size"] = {"width": 832, "height": 1216}
        manifest = epochs.plan(self.project, run_folder=self.run, log=lambda *_: None)
        self.assertEqual(manifest["size"], {"width": 832, "height": 1216})

    def test_the_plan_is_written_where_the_sheet_will_look_for_it(self):
        epochs.plan(self.project, run_folder=self.run, log=lambda *_: None)
        path = epochs.manifest_path_of(self.project, self.run)
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text("utf-8"))["run"]["id"], "char-abc123")

    def test_the_run_state_is_recorded_when_the_run_has_one(self):
        (self.run / "state.json").write_text(json.dumps({"complete": False, "active_stage": "Training",
                                                         "epoch": 20, "step": 620}), encoding="utf-8")
        manifest = epochs.plan(self.project, run_folder=self.run, log=lambda *_: None)
        self.assertEqual(manifest["run"]["state"]["complete"], False)
        self.assertEqual(manifest["run"]["state"]["epoch"], 20)


class RunTests(unittest.TestCase):
    """Drawing the grid: resumable, and honest about what it drew."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "runs" / "char-abc123"
        (self.run / "output").mkdir(parents=True)
        self.project = FakeProject(self.root / "proj")
        self.project.dir.mkdir(parents=True)
        (self.run / "output" / "char-000002.safetensors").write_bytes(safetensors_bytes(2))
        (self.run / "output" / "char-000004.safetensors").write_bytes(safetensors_bytes(4))
        self.manifest = epochs.plan(self.project, run_folder=self.run, seeds=[1], log=lambda *_: None)

    def tearDown(self):
        self.temp.cleanup()

    def client(self, fail_on=None):
        class Fake:
            def __init__(self):
                self.runs = []
                self.names = {}

            def name_for(self, kind, path):
                return Path(path).name

            def run(self, graph, out, slot_id=None, method=None, seed=None):
                if fail_on and fail_on in str(slot_id):
                    raise RuntimeError("ComfyUI said no")
                self.runs.append({"graph": graph, "out": out, "method": method, "seed": seed})
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
                return out

        return Fake()

    def names(self):
        return {"b_unet": "raw.safetensors", "speed_lora": "turbo.safetensors",
                "clip": "te.safetensors", "vae": "vae.safetensors"}

    def test_every_cell_is_drawn(self):
        client = self.client()
        done, skipped, failed = epochs.run(self.project, self.manifest, client=client, names=self.names(),
                                           log=lambda *_: None)
        self.assertEqual((done, skipped, failed), (len(epochs.EVAL_PROMPTS) * 2, 0, 0))
        self.assertEqual(len(client.runs), len(epochs.EVAL_PROMPTS) * 2)

    def test_each_cell_loads_the_checkpoint_of_its_own_epoch(self):
        client = self.client()
        epochs.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        for row in client.runs:
            epoch = int(row["method"].rsplit("e", 1)[-1])
            loras = [node["inputs"]["lora_name"] for node in row["graph"].values()
                     if node["class_type"] == "LoraLoaderModelOnly"]
            self.assertIn(f"char-{epoch:06d}.safetensors", loras,
                          f"{row['method']} used {loras}")

    def test_a_second_run_draws_nothing(self):
        client = self.client()
        epochs.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        before = len(client.runs)
        done, skipped, failed = epochs.run(self.project, self.manifest, client=client, names=self.names(),
                                           log=lambda *_: None)
        self.assertEqual((done, failed), (0, 0))
        self.assertEqual(skipped, before)
        self.assertEqual(len(client.runs), before)

    def test_one_failed_cell_does_not_end_the_grid(self):
        client = self.client(fail_on="e02_upper")
        done, _skipped, failed = epochs.run(self.project, self.manifest, client=client, names=self.names(),
                                            log=lambda *_: None)
        self.assertEqual(failed, 2, "e02 fails once per checkpoint")
        self.assertEqual(done, len(epochs.EVAL_PROMPTS) * 2 - 2)

    def test_cancelling_stops_between_cells_and_keeps_what_was_drawn(self):
        # A pause has to leave complete pictures and the rest still pending; that is what makes a rerun continue.
        client = self.client()
        drawn = []

        def cancel():
            return len(drawn) >= 3

        original = client.run

        def counting(graph, out, slot_id=None, method=None, seed=None):
            result = original(graph, out, slot_id=slot_id, method=method, seed=seed)
            drawn.append(out)
            return result

        client.run = counting
        done, _skipped, failed = epochs.run(self.project, self.manifest, client=client, names=self.names(),
                                            log=lambda *_: None, cancel=cancel)
        self.assertEqual(failed, 0)
        self.assertEqual(done, 3, "the cell in flight finishes, then the loop stops")
        self.assertTrue(all(Path(path).is_file() for path in drawn))
        again = epochs.run(self.project, self.manifest, client=self.client(), names=self.names(),
                           log=lambda *_: None)
        self.assertEqual(again[0], len(epochs.EVAL_PROMPTS) * 2 - 3, "the rerun draws only what is missing")

    def test_a_failed_cell_is_retried_on_the_next_run_and_the_rest_is_not(self):
        client = self.client(fail_on="e02_upper")
        epochs.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        client2 = self.client()
        done, skipped, _failed = epochs.run(self.project, self.manifest, client=client2, names=self.names(),
                                            log=lambda *_: None)
        self.assertEqual(done, 2, "the two failed cells are drawn, nothing else")
        self.assertEqual(skipped, len(epochs.EVAL_PROMPTS) * 2 - 2)
        self.assertEqual(len(client2.runs), 2)


class ChooseTests(unittest.TestCase):
    """Recording a person's choice: run, checkpoint bytes and training snapshot kept apart."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "runs" / "char-abc123"
        (self.run / "output").mkdir(parents=True)
        self.project = FakeProject(self.root / "proj")
        self.project.dir.mkdir(parents=True)
        self.checkpoints = {}
        for epoch in (2, 4):
            path = self.run / "output" / f"char-{epoch:06d}.safetensors"
            path.write_bytes(safetensors_bytes(epoch))
            self.checkpoints[epoch] = path
        self.manifest = epochs.plan(self.project, run_folder=self.run, seeds=[1], log=lambda *_: None)

    def tearDown(self):
        self.temp.cleanup()

    def test_the_choice_records_the_three_things_separately(self):
        entry = epochs.choose(self.project, run_folder=self.run, epoch=4, log=lambda *_: None)
        self.assertEqual(entry["epoch"], 4)
        self.assertEqual(entry["run_id"], "char-abc123")
        self.assertEqual(entry["checkpoint"], "char-000004.safetensors")
        self.assertEqual(len(entry["checkpoint_sha256"]), 64)
        self.assertIn("training_snapshot", entry)
        self.assertEqual(self.project.card["krea2_inference"]["epoch"], 4)
        self.assertEqual(self.project.card["epoch_note"]["run_id"], "char-abc123")

    def test_choosing_repairs_a_plan_that_never_recorded_the_snapshot(self):
        # A grid planned before the seed-run layout was understood has null there. The choice must not inherit the
        # hole, and the repair must not cost the drawn cells.
        seed_base = self.project.dir / "seed_lora" / "abc123def456"
        (seed_base / "runs").mkdir(parents=True)
        (seed_base / "snapshot.json").write_text("{}", encoding="utf-8")
        inner = seed_base / "runs" / self.run.name
        shutil.move(str(self.run), str(inner))

        manifest_path = epochs.manifest_path_of(self.project, inner)
        data = epochs.read_manifest(manifest_path)
        data["run"]["folder"] = str(inner)
        data["training_snapshot"] = None
        data["checkpoints"] = [{**row, "file": str(inner / "output" / Path(row["file"]).name)}
                               for row in data["checkpoints"]]
        for task in data["tasks"]:
            task["status"] = "done"
        epochs.write_manifest(manifest_path, data)

        entry = epochs.choose(self.project, run_folder=inner, epoch=4, log=lambda *_: None)
        self.assertEqual(entry["training_snapshot"], "abc123def456")
        after = epochs.read_manifest(manifest_path)
        self.assertEqual(after["training_snapshot"]["id"], "abc123def456")
        self.assertEqual(sum(1 for t in after["tasks"] if t["status"] == "done"), len(after["tasks"]))

    def test_choosing_leaves_a_recorded_snapshot_alone(self):
        # A snapshot somebody did record is a decision; the repair must not overwrite it with a folder guess.
        manifest_path = epochs.manifest_path_of(self.project, self.run)
        data = epochs.read_manifest(manifest_path)
        data["training_snapshot"] = {"id": "keptbyhand1", "source": "config.json"}
        epochs.write_manifest(manifest_path, data)
        entry = epochs.choose(self.project, run_folder=self.run, epoch=2, log=lambda *_: None)
        self.assertEqual(entry["training_snapshot"], "keptbyhand1")

    def test_choosing_an_epoch_the_grid_did_not_draw_is_refused(self):
        with self.assertRaises(epochs.EpochError) as caught:
            epochs.choose(self.project, run_folder=self.run, epoch=99, log=lambda *_: None)
        self.assertIn("not in this grid", str(caught.exception))

    def test_a_checkpoint_that_changed_since_the_grid_was_drawn_is_refused(self):
        # The pictures in the grid were made with other bytes: recording the choice would attach a judgement to a
        # file nobody looked at.
        self.checkpoints[4].write_bytes(safetensors_bytes(4, payload=b"different" * 20))
        with self.assertRaises(epochs.EpochError) as caught:
            epochs.choose(self.project, run_folder=self.run, epoch=4, log=lambda *_: None)
        self.assertIn("changed since the grid was planned", str(caught.exception))

    def test_a_deleted_checkpoint_is_refused(self):
        self.checkpoints[2].unlink()
        with self.assertRaises(epochs.EpochError):
            epochs.choose(self.project, run_folder=self.run, epoch=2, log=lambda *_: None)

    def test_the_recorded_choice_can_be_read_back_and_verified(self):
        epochs.choose(self.project, run_folder=self.run, epoch=2, log=lambda *_: None)
        entry = epochs.chosen(self.project)
        self.assertEqual(entry["epoch"], 2)
        self.assertTrue(entry["checkpoint_present"])
        self.assertTrue(entry["checkpoint_matches"])
        self.checkpoints[2].write_bytes(safetensors_bytes(2, payload=b"y" * 64))
        self.assertFalse(epochs.chosen(self.project)["checkpoint_matches"])

    def test_a_choice_from_another_run_is_not_returned_for_this_one(self):
        epochs.choose(self.project, run_folder=self.run, epoch=2, log=lambda *_: None)
        other = self.root / "runs" / "other-run"
        other.mkdir(parents=True)
        self.assertIsNone(epochs.chosen(self.project, other))


class SheetTests(unittest.TestCase):
    """The comparison page: one row per prompt, one column per checkpoint."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "runs" / "char-abc123"
        (self.run / "output").mkdir(parents=True)
        self.project = FakeProject(self.root / "proj")
        self.project.dir.mkdir(parents=True)
        for epoch in (2, 4):
            (self.run / "output" / f"char-{epoch:06d}.safetensors").write_bytes(safetensors_bytes(epoch))
        self.manifest = epochs.plan(self.project, run_folder=self.run, seeds=[1], log=lambda *_: None)
        exp = epochs.manifest_path_of(self.project, self.run).parent
        for task in self.manifest["tasks"]:
            path = exp / task["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

    def tearDown(self):
        self.temp.cleanup()

    def test_every_prompt_and_checkpoint_appears(self):
        path = epochs.write_sheet(self.project, self.run, log=lambda *_: None)
        html = path.read_text("utf-8")
        for prompt in epochs.EVAL_PROMPTS:
            self.assertIn(prompt["id"], html)
        for epoch in (2, 4):
            self.assertIn(f"에포크 {epoch}", html)
        self.assertIn("comparison.html", str(path))

    def test_a_cell_that_was_not_drawn_is_shown_as_a_gap_not_as_a_picture(self):
        exp = epochs.manifest_path_of(self.project, self.run).parent
        (exp / self.manifest["tasks"][0]["file"]).unlink()
        html = epochs.write_sheet(self.project, self.run, log=lambda *_: None).read_text("utf-8")
        self.assertIn("아직 없는 칸 1개", html)

    def test_the_training_snapshot_is_shown(self):
        self.manifest["training_snapshot"] = {"id": "abc123def456", "source": "config.json"}
        epochs.write_manifest(epochs.manifest_path_of(self.project, self.run), self.manifest)
        html = epochs.write_sheet(self.project, self.run, log=lambda *_: None).read_text("utf-8")
        self.assertIn("abc123def456", html)

    def test_an_unrecorded_snapshot_says_so_instead_of_inventing_one(self):
        html = epochs.write_sheet(self.project, self.run, log=lambda *_: None).read_text("utf-8")
        self.assertIn("기록 없음", html)

    def test_the_chosen_epoch_is_marked(self):
        epochs.choose(self.project, run_folder=self.run, epoch=4, log=lambda *_: None)
        html = epochs.write_sheet(self.project, self.run, log=lambda *_: None).read_text("utf-8")
        self.assertIn("선택됨", html)
        self.assertIn("best", html)


class SnapshotOfTests(unittest.TestCase):
    """A seed run keeps its snapshot only in the folder layout, so reading it must not be a guess."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = FakeProject(self.root / "proj")
        self.project.dir.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def _seed_run(self, snapshot="abc123def456", *, snapshot_json=True):
        base = self.project.dir / "seed_lora" / snapshot
        base.mkdir(parents=True)
        if snapshot_json:
            (base / "snapshot.json").write_text("{}", encoding="utf-8")
        run = base / "runs" / "demo_seed-0123456789abcdef"
        run.mkdir(parents=True)
        return run

    def test_the_snapshot_is_read_from_the_seed_run_layout(self):
        run = self._seed_run()
        self.assertEqual(epochs._snapshot_of(self.project, run)["id"], "abc123def456")

    def test_a_folder_that_only_looks_like_a_seed_run_is_not_trusted(self):
        # Without snapshot.json the name is just a folder name; a stray directory must not become provenance.
        run = self._seed_run(snapshot_json=False)
        self.assertIsNone(epochs._snapshot_of(self.project, run))

    def test_a_run_config_still_wins_when_the_path_says_nothing(self):
        run = self.root / "elsewhere" / "demo_seed-0123456789abcdef"
        run.mkdir(parents=True)
        (run / "run_config.json").write_text('{"snapshot": "fromconfig12"}', encoding="utf-8")
        found = epochs._snapshot_of(self.project, run)
        self.assertEqual(found["id"], "fromconfig12")
        self.assertEqual(found["source"], "run_config.json")


class FindRunTests(unittest.TestCase):
    """Which run to evaluate: only ever one that was actually found."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = FakeProject(self.root / "proj")
        self.project.dir.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, snapshot, name):
        path = self.project.dir / "seed_lora" / snapshot / "runs" / name
        path.mkdir(parents=True)
        return path

    def test_the_only_run_is_used_without_being_named(self):
        made = self.make_run("snap1", "char-aaa")
        self.assertEqual(epochs.find_run(self.project), made)

    def test_more_than_one_run_must_be_named(self):
        self.make_run("snap1", "char-aaa")
        self.make_run("snap2", "char-bbb")
        with self.assertRaises(epochs.EpochError) as caught:
            epochs.find_run(self.project)
        self.assertIn("more than one run", str(caught.exception))
        self.assertIn("char-aaa", str(caught.exception))

    def test_a_run_is_found_by_its_name_or_its_snapshot_id(self):
        made = self.make_run("snap1", "char-aaa")
        self.make_run("snap2", "char-bbb")
        self.assertEqual(epochs.find_run(self.project, "char-bbb").name, "char-bbb")
        self.assertEqual(epochs.find_run(self.project, "snap1"), made)

    def test_no_run_at_all_says_to_train_one_first(self):
        with self.assertRaises(epochs.EpochError) as caught:
            epochs.find_run(self.project)
        self.assertIn("train a seed first", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
