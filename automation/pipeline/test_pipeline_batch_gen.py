"""M3-3: batch B. What is tested is the reasoning that stops this stage from wasting GPU time and from lying.

Three rules carry the module, and each has its own group:

* a slot is drawn only when its pictures were *refused*, never when one is merely waiting to be judged - otherwise
  every run of the stage draws the same pictures again;
* generation is the newest generation the seed actually learned from, plus one, and the snapshot it came from is a
  separate field - a picture has to stay traceable to the images that taught the model that drew it;
* the (b) graph is used or nothing is, because the M0 evidence for it does not transfer to another configuration.
"""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from pipeline import batch_gen, epochs


def tiny_card(**over):
    card = {"id": "suamom", "trigger": "suamom", "caption_base": "korean webtoon style",
            "fixed_traits": [], "outfits": {"default": "beige dress"},
            "slot_quotas": {"ang_front_medium": 1, "bg_classroom": 1, "frm_back": 1},
            "krea2_inference": {"variant": "b", "size": {"width": 1024, "height": 1536}}}
    card.update(over)
    return card


class FakeProject:
    """The parts of `Project` batch_gen uses: the folder, the card, the quota board and the review queue."""

    def __init__(self, root, card=None, accepted=None, undecided=None):
        self.dir = Path(root)
        self.card = card or tiny_card()
        self._accepted = list(accepted or [])
        self._undecided = list(undecided or [])
        self.dir.mkdir(parents=True, exist_ok=True)

    def guard(self, *_a, **_k):
        import contextlib
        return contextlib.nullcontext()

    def save(self):
        pass

    def quotas(self, planned_slots):
        counts = {}
        for record in self._accepted:
            slot = (record.get("slot") or {}).get("slot_id")
            counts[slot] = counts.get(slot, 0) + 1
        return {slot: {"target": int(target), "accepted": counts.get(slot, 0),
                       "short": max(0, int(target) - counts.get(slot, 0))}
                for slot, target in planned_slots.items()}

    def accepted(self):
        return list(self._accepted)

    def undecided(self):
        return list(self._undecided)

    def import_candidates(self, records, base=None, log=print):
        return len(records), []


def accepted_in(slot):
    return {"id": "c_" + slot, "slot": {"slot_id": slot, "category": "x"}}


def pending_in(slot):
    return {"id": "c_pending_" + slot, "slot": {"slot_id": slot, "category": "x"}}


def write_snapshot(project, *, snapshot_id="snap1", generations=(0, 0, 1), created="2026-11-06T12:00:00"):
    folder = project.dir / "seed_lora" / snapshot_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "seed", "id": snapshot_id, "created_at": created,
        "base": str(folder), "dataset": str(folder / "dataset"),
        "inputs": [{"id": f"s_{n}", "generation": g, "edit_depth": 0} for n, g in enumerate(generations)],
    }
    (folder / "snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def safetensors_bytes(epoch):
    header = {"__metadata__": {"ss_epoch": str(epoch)},
              "lora.a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + b"xxxx"


class ShortSlotTests(unittest.TestCase):
    """Which slots need a picture drawn, and which only need the reviewer."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = FakeProject(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_met_slot_is_not_short(self):
        self.project._accepted = [accepted_in("ang_front_medium")]
        draw, waiting = batch_gen.short_slots(self.project)
        self.assertEqual([r["slot"] for r in draw], ["bg_classroom", "frm_back"])
        self.assertEqual(waiting, [])

    def test_a_short_slot_with_nothing_waiting_is_drawn(self):
        draw, _waiting = batch_gen.short_slots(self.project)
        self.assertEqual(sorted(r["slot"] for r in draw), ["ang_front_medium", "bg_classroom", "frm_back"])
        self.assertTrue(all(r["to_draw"] == 1 for r in draw))

    def test_a_short_slot_whose_picture_is_waiting_is_not_drawn_again(self):
        # This is the rule that keeps a second run from submitting the same six pictures.
        self.project._undecided = [pending_in("bg_classroom")]
        draw, waiting = batch_gen.short_slots(self.project)
        self.assertNotIn("bg_classroom", [r["slot"] for r in draw])
        self.assertEqual([r["slot"] for r in waiting], ["bg_classroom"])
        self.assertEqual(waiting[0]["waiting"], 1)

    def test_a_slot_with_more_waiting_than_it_needs_is_still_only_waiting(self):
        self.project._undecided = [pending_in("bg_classroom")] * 3
        draw, waiting = batch_gen.short_slots(self.project)
        self.assertNotIn("bg_classroom", [r["slot"] for r in draw])
        self.assertEqual(waiting[0]["slot"], "bg_classroom")

    def test_a_slot_that_needs_two_and_has_one_waiting_draws_one(self):
        self.project.card["slot_quotas"]["bg_classroom"] = 2
        self.project._undecided = [pending_in("bg_classroom")]
        draw, _waiting = batch_gen.short_slots(self.project)
        row = next(r for r in draw if r["slot"] == "bg_classroom")
        self.assertEqual(row["to_draw"], 1)

    def test_a_slot_with_no_known_prompt_is_refused_rather_than_guessed(self):
        self.project.card["slot_quotas"]["pos_cartwheel"] = 1
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.short_slots(self.project)
        self.assertIn("no prompt in SLOT_PROMPTS", str(caught.exception))

    def test_a_card_without_quotas_says_there_is_nothing_to_fill(self):
        self.project.card["slot_quotas"] = {}
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.short_slots(self.project)
        self.assertIn("no slot_quotas", str(caught.exception))

    def test_the_category_comes_from_the_slot_id(self):
        self.assertEqual(batch_gen.category_of("ang_quarter_close"), "angles")
        self.assertEqual(batch_gen.category_of("bg_park"), "backgrounds")
        self.assertEqual(batch_gen.category_of("pos_running"), "poses")
        self.assertEqual(batch_gen.category_of("exp_sad"), "expressions")
        self.assertEqual(batch_gen.category_of("frm_dutch"), "framing")


class TrainingTests(unittest.TestCase):
    """Which generation batch B is drawn at, and what the record says it learned from."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = FakeProject(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_the_generation_is_one_past_the_newest_the_seed_learned_from(self):
        write_snapshot(self.project, generations=(0, 0, 1))
        training = batch_gen.training_of(self.project)
        self.assertEqual(training["generation"], 2)
        self.assertEqual(training["trained_on_generation"], 1)

    def test_an_all_original_seed_is_drawn_at_generation_one(self):
        write_snapshot(self.project, generations=(0, 0, 0))
        self.assertEqual(batch_gen.training_of(self.project)["generation"], 1)

    def test_the_snapshot_id_is_recorded_separately_from_the_generation(self):
        write_snapshot(self.project, snapshot_id="abc123", generations=(1,))
        training = batch_gen.training_of(self.project)
        self.assertEqual(training["snapshot_id"], "abc123")
        self.assertEqual(training["trained_on_snapshot"], "abc123")
        self.assertEqual(training["generation"], 2)

    def test_the_newest_snapshot_is_the_one_used(self):
        write_snapshot(self.project, snapshot_id="old", generations=(0,), created="2026-01-01T00:00:00")
        write_snapshot(self.project, snapshot_id="new", generations=(0, 1), created="2026-11-01T00:00:00")
        training = batch_gen.training_of(self.project)
        self.assertEqual(training["snapshot_id"], "new")
        self.assertEqual(training["generation"], 2)

    def test_no_snapshot_at_all_says_to_build_one_first(self):
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.training_of(self.project)
        self.assertIn("no seed snapshot", str(caught.exception))


class PlanTests(unittest.TestCase):
    """The work list written for the short slots."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = FakeProject(self.temp.name)
        self.project._accepted = [accepted_in("ang_front_medium")]
        write_snapshot(self.project, snapshot_id="snap1", generations=(0, 1))

    def tearDown(self):
        self.temp.cleanup()

    def test_only_the_short_slots_get_images(self):
        manifest = batch_gen.plan(self.project, log=lambda *_: None)
        self.assertEqual(sorted({t["slot_id"] for t in manifest["tasks"]}), ["bg_classroom", "frm_back"])
        self.assertEqual(len(manifest["tasks"]), 4, "two slots x two seeds")

    def test_the_prompt_carries_the_card_and_the_slot_wording(self):
        manifest = batch_gen.plan(self.project, log=lambda *_: None)
        task = next(t for t in manifest["tasks"] if t["slot_id"] == "bg_classroom")
        self.assertTrue(task["prompt"].startswith("suamom, korean webtoon style, beige dress,"), task["prompt"])
        self.assertIn("classroom background", task["prompt"])

    def test_the_generation_and_snapshot_are_written_into_the_plan(self):
        manifest = batch_gen.plan(self.project, log=lambda *_: None)
        self.assertEqual(manifest["generation"], 2)
        self.assertEqual(manifest["trained_on_snapshot"], "snap1")
        self.assertEqual(manifest["training"]["trained_on_generation"], 1)

    def test_the_size_and_configuration_come_from_the_card(self):
        manifest = batch_gen.plan(self.project, log=lambda *_: None)
        self.assertEqual(manifest["size"], {"width": 1024, "height": 1536})
        self.assertEqual(manifest["inference"]["variant"], "b")

    def test_a_plan_with_every_slot_waiting_is_written_empty_rather_than_refused(self):
        self.project._undecided = [pending_in("bg_classroom"), pending_in("frm_back")]
        manifest = batch_gen.plan(self.project, log=lambda *_: None)
        self.assertEqual(manifest["tasks"], [])
        self.assertEqual(len(manifest["waiting"]), 2)

    def test_planning_twice_writes_the_same_work(self):
        first = batch_gen.plan(self.project, log=lambda *_: None)
        second = batch_gen.plan(self.project, log=lambda *_: None)
        self.assertEqual([t["file"] for t in first["tasks"]], [t["file"] for t in second["tasks"]])


class RunTests(unittest.TestCase):
    """Drawing the batch: resumable, and it uses the (b) graph or refuses."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project = FakeProject(self.temp.name)
        write_snapshot(self.project, snapshot_id="snap1", generations=(0, 1))
        self.project.card["slot_quotas"] = {"bg_classroom": 1}
        self.manifest = batch_gen.plan(self.project, log=lambda *_: None)

    def tearDown(self):
        self.temp.cleanup()

    def client(self):
        class Fake:
            def __init__(self):
                self.runs = []

            def name_for(self, kind, path):
                return Path(path).name

            def run(self, graph, out, slot_id=None, method=None, seed=None):
                self.runs.append({"graph": graph, "method": method})
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                Image.new("RGB", (8, 8), (10, 20, 30)).save(out)
                return out

        return Fake()

    def names(self):
        return {"b_unet": "raw.safetensors", "speed_lora": "turbo.safetensors",
                "clip": "te.safetensors", "vae": "vae.safetensors", "character_lora": "char.safetensors"}

    def test_every_cell_is_drawn(self):
        client = self.client()
        done, _skipped, failed = batch_gen.run(self.project, self.manifest, client=client, names=self.names(),
                                               log=lambda *_: None)
        self.assertEqual((done, failed), (2, 0))

    def test_a_second_run_submits_nothing(self):
        client = self.client()
        batch_gen.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        before = len(client.runs)
        done, skipped, failed = batch_gen.run(self.project, self.manifest, client=client, names=self.names(),
                                              log=lambda *_: None)
        self.assertEqual((done, failed), (0, 0))
        self.assertEqual(skipped, before)
        self.assertEqual(len(client.runs), before)

    def test_the_raw_dit_and_the_speed_lora_are_both_in_the_graph(self):
        client = self.client()
        batch_gen.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        loras = [node["inputs"]["lora_name"] for node in client.runs[0]["graph"].values()
                 if node["class_type"] == "LoraLoaderModelOnly"]
        self.assertIn("turbo.safetensors", loras)
        self.assertIn("char.safetensors", loras)
        unets = [node["inputs"]["unet_name"] for node in client.runs[0]["graph"].values()
                 if node["class_type"] == "UNETLoader"]
        self.assertEqual(unets, ["raw.safetensors"])

    def test_cancelling_stops_between_cells_and_the_rest_stays_pending(self):
        client = self.client()
        done, _skipped, failed = batch_gen.run(self.project, self.manifest, client=client, names=self.names(),
                                               log=lambda *_: None, cancel=lambda: True)
        self.assertEqual((done, failed), (0, 0))
        self.assertEqual(len(client.runs), 0, "the stop is checked before the first cell here")
        after = batch_gen.run(self.project, self.manifest, client=self.client(), names=self.names(),
                              log=lambda *_: None)
        self.assertEqual(after[0], 2, "nothing was drawn, so the whole plan is still pending")

    def test_a_cancel_after_one_cell_keeps_that_cell(self):
        client = self.client()
        drawn = []
        original = client.run

        def counting(graph, out, slot_id=None, method=None, seed=None):
            result = original(graph, out, slot_id=slot_id, method=method, seed=seed)
            drawn.append(Path(out))
            return result

        client.run = counting
        done, _skipped, failed = batch_gen.run(self.project, self.manifest, client=client, names=self.names(),
                                               log=lambda *_: None, cancel=lambda: len(drawn) >= 1)
        self.assertEqual((done, failed), (1, 0))
        self.assertTrue(drawn[0].is_file())
        after = batch_gen.run(self.project, self.manifest, client=self.client(), names=self.names(),
                              log=lambda *_: None)
        self.assertEqual(after[0], 1, "the drawn cell is skipped and the other is drawn")

    def test_a_file_that_no_longer_decodes_is_drawn_again(self):
        client = self.client()
        batch_gen.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        exp = batch_gen.manifest_path(self.project).parent
        (exp / self.manifest["tasks"][0]["file"]).write_bytes(b"\x89PNG\r\n\x1a\ntruncated")
        done, skipped, _failed = batch_gen.run(self.project, self.manifest, client=client, names=self.names(),
                                               log=lambda *_: None)
        self.assertEqual(done, 1, "a file that does not decode is not an output, whatever its name says")
        self.assertEqual(skipped, 1)
        self.assertEqual(len(client.runs), 3, "two drawn up front, one redrawn for the file that stopped decoding")

    def test_a_file_from_another_plan_is_drawn_again(self):
        # A task with no recorded sha has no output this plan can vouch for: the file at its path could be anything.
        client = self.client()
        exp = batch_gen.manifest_path(self.project).parent
        for task in self.manifest["tasks"]:
            (exp / task["file"]).parent.mkdir(parents=True, exist_ok=True)
            (exp / task["file"]).write_bytes(b"\x89PNG\r\n\x1a\nold-plan-bytes")
        done, _skipped, failed = batch_gen.run(self.project, self.manifest, client=client, names=self.names(),
                                               log=lambda *_: None)
        self.assertEqual(done, len(self.manifest["tasks"]))
        self.assertEqual(len(client.runs), len(self.manifest["tasks"]))

    def test_a_replan_archives_the_manifest_that_drew_work(self):
        # The manifest that drew already-imported pictures is the lineage's evidence (spec R4): a new plan keeps it
        # beside itself instead of overwriting it into silence.
        client = self.client()
        batch_gen.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        batch_gen.plan(self.project, log=lambda *_: None)
        archives = list(batch_gen.manifest_path(self.project).parent.glob("batch.json.at-*.json"))
        self.assertEqual(len(archives), 1)
        archived = json.loads(archives[0].read_text(encoding="utf-8"))
        self.assertTrue(all(task["status"] == "done" for task in archived["tasks"]))
        self.assertTrue(all(task.get("sha256") for task in archived["tasks"]))

    def test_another_configuration_is_refused_before_anything_is_drawn(self):
        self.manifest["inference"]["variant"] = "a"
        client = self.client()
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.run(self.project, self.manifest, client=client, names=self.names(), log=lambda *_: None)
        self.assertIn("configuration (b)", str(caught.exception))
        self.assertEqual(client.runs, [])


class CandidatesTests(unittest.TestCase):
    """The lineage of a batch B picture: generation, snapshot and epoch kept apart."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = FakeProject(self.root / "proj")
        write_snapshot(self.project, snapshot_id="snap1", generations=(0, 1))
        self.project.card["slot_quotas"] = {"bg_classroom": 1}
        self.run_folder = self.root / "runs" / "char-abc"
        (self.run_folder / "output").mkdir(parents=True)
        self.checkpoint = self.run_folder / "output" / "char-000004.safetensors"
        self.checkpoint.write_bytes(safetensors_bytes(4))
        epochs.plan(self.project, run_folder=self.run_folder, seeds=[1], log=lambda *_: None)
        epochs.choose(self.project, run_folder=self.run_folder, epoch=4, log=lambda *_: None)
        self.manifest = batch_gen.plan(self.project, log=lambda *_: None)
        exp = batch_gen.manifest_path(self.project).parent
        for task in self.manifest["tasks"]:
            path = exp / task["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

    def tearDown(self):
        self.temp.cleanup()

    def test_every_drawn_picture_gets_a_record(self):
        records = batch_gen.candidates_of(self.project, self.manifest)
        self.assertEqual(len(records), len(self.manifest["tasks"]))
        self.assertTrue(all(r["slot"]["slot_id"] == "bg_classroom" for r in records))

    def test_the_generation_is_the_batch_generation_not_the_epoch(self):
        records = batch_gen.candidates_of(self.project, self.manifest)
        self.assertEqual({r["train_generation"] for r in records}, {2})
        self.assertEqual({r["edit_depth"] for r in records}, {0})

    def test_the_lora_records_which_snapshot_taught_it(self):
        records = batch_gen.candidates_of(self.project, self.manifest)
        lora = records[0]["models"]["loras"][0]
        self.assertEqual(lora["trained_on_snapshot"], "snap1")
        self.assertEqual(lora["trained_on_generation"], 1)

    def test_the_chosen_epoch_is_the_checkpoint_that_was_used(self):
        self.assertEqual(batch_gen.character_lora_path(self.project), self.checkpoint)
        records = batch_gen.candidates_of(self.project, self.manifest)
        lora = records[0]["models"]["loras"][0]
        self.assertEqual(lora["name"], "char-000004.safetensors")
        self.assertEqual(lora["sha256"][:12], epochs.chosen(self.project)["checkpoint_sha256"][:12])

    def test_an_unchosen_epoch_refuses_to_draw_at_all(self):
        del self.project.card["krea2_inference"]["selected"]
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.character_lora_path(self.project)
        self.assertIn("no epoch has been chosen", str(caught.exception))

    def test_a_chosen_checkpoint_that_changed_is_refused(self):
        self.checkpoint.write_bytes(safetensors_bytes(9))
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.character_lora_path(self.project)
        self.assertIn("changed since it was chosen", str(caught.exception))

    def test_a_chosen_checkpoint_that_was_deleted_is_refused(self):
        self.checkpoint.unlink()
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.character_lora_path(self.project)
        self.assertIn("is gone", str(caught.exception))

    def test_a_file_whose_bytes_changed_is_not_recorded_as_the_plans_output(self):
        records = batch_gen.candidates_of(self.project, self.manifest)
        self.assertEqual(len(records), len(self.manifest["tasks"]))
        for record, task in zip(records, self.manifest["tasks"]):
            task["sha256"] = record["sha256"]
        exp = batch_gen.manifest_path(self.project).parent
        task = self.manifest["tasks"][0]
        (exp / task["file"]).write_bytes(b"\x89PNG\r\n\x1a\nreplaced-bytes")
        self.assertEqual(len(batch_gen.candidates_of(self.project, self.manifest)), len(self.manifest["tasks"]) - 1,
                         "the plan records the bytes it drew; a changed file is not that output")

    def test_a_checkpoint_changed_after_planning_refuses_the_run(self):
        self.manifest["checkpoint"]["checkpoint_sha256"] = "0" * 64
        with self.assertRaises(batch_gen.BatchError) as caught:
            batch_gen.pinned_checkpoint(self.manifest)
        self.assertIn("changed since the plan pinned it", str(caught.exception))

    def test_the_plan_pins_the_chosen_checkpoint(self):
        chosen = epochs.chosen(self.project)
        self.assertEqual(self.manifest["checkpoint"]["checkpoint_sha256"], chosen["checkpoint_sha256"])
        self.assertEqual(self.manifest["checkpoint"]["run_id"], chosen["run_id"])

    def test_an_older_manifest_without_a_pin_reads_the_card_as_before(self):
        del self.manifest["checkpoint"]
        records = batch_gen.candidates_of(self.project, self.manifest)
        self.assertEqual(len(records), len(self.manifest["tasks"]))


if __name__ == "__main__":
    unittest.main()
