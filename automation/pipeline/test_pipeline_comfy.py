"""Offline checks for the ComfyUI client: the prompt journal, atomic writes and filename resolution."""
import json
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import comfy  # noqa: E402


class Response:
    def __init__(self, payload=b"{}"):
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeComfy:
    """A ComfyUI that answers from a dict of path -> payload, and records what was asked."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.requests = []
        self.object_info = {
            "UNETLoader": {"input": {"required": {"unet_name": [["raw.safetensors"], {}]}}},
            "VAELoader": {"input": {"required": {"vae_name": [["vae.safetensors"], {}]}}},
            "CLIPLoader": {"input": {"required": {"clip_name": [["te.safetensors"], {}],
                                                  "type": [["qwen_image", "krea2"], {}]}}},
            "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [["2. krea2\\suamom.safetensors",
                                                                           "krea2_turbo_lora_rank_64_bf16.safetensors"],
                                                                  {}]}}},
        }

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        if isinstance(req, str):
            url = req
        else:
            url = req.full_url
        path = urllib.parse.urlsplit(url).path
        if path == "/object_info":
            return Response(self.object_info)
        if path.startswith("/object_info/"):
            node = path.rsplit("/", 1)[-1]
            if node in self.object_info:
                return Response({node: self.object_info[node]})
            return Response(b"{}")
        if path in self.answers:
            return Response(self.answers[path])
        if path.startswith("/history/"):
            return Response({})
        return Response({})

    def posted(self):
        return [json.loads(r.data.decode()) for r in self.requests if getattr(r, "data", None)]


def make(tmp, fake):
    return comfy.Comfy("http://127.0.0.1:9", journal=Path(tmp) / "journal", comfy_root="G:/Lakis/ComfyUI",
                       opener=fake, timeout=1)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_record_is_on_disk_with_status_submitted(self):
        journal = comfy.Journal(self.dir / "journal")
        journal.record("pid1", method="krea2_a")
        data = json.loads((self.dir / "journal" / "pid1.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "submitted")
        self.assertEqual(data["method"], "krea2_a")

    def test_unresolved_lists_only_open_prompts(self):
        journal = comfy.Journal(self.dir / "journal")
        journal.record("a")
        journal.update("a", comfy.Journal.DONE)
        journal.record("b")
        self.assertEqual([e["prompt_id"] for e in journal.unresolved()], ["b"])

    def test_submit_writes_the_id_before_returning(self):
        fake = FakeComfy({"/prompt": {"prompt_id": "p-123"}})
        client = make(self.dir, fake)
        prompt_id = client.submit({"1": {"class_type": "SaveImage", "inputs": {}}}, method="m")
        self.assertEqual(prompt_id, "p-123")
        self.assertTrue((self.dir / "journal" / "p-123.json").is_file())

    def test_a_refused_graph_leaves_no_unresolved_record(self):
        # The server refused the graph, so nothing was submitted; the closed intent says so instead of dangling
        # (spec R6) - and an unresolved scan must not mistake it for a submission in flight.
        def boom(_req, timeout=None):
            raise comfy.urllib.error.HTTPError("http://x/prompt", 400, "bad", {}, None)
        fake = FakeComfy({})
        client = make(self.dir, fake)
        with patch.object(client, "_open", side_effect=boom):
            with self.assertRaises(comfy.ComfyError):
                client.submit({"1": {}})
        self.assertEqual(client.journal.unresolved(), [])
        intents = [e for e in client.journal.entries() if e.get("status") == "failed"]
        self.assertEqual(len(intents), 1)
        self.assertIn("refused", intents[0]["error"])

    def test_wait_marks_done_and_returns_outputs(self):
        fake = FakeComfy({"/history/p1": {"p1": {"status": {"status_str": "success"},
                                                "outputs": {"9": {"images": [{"filename": "x.png",
                                                                              "subfolder": "", "type": "output"}]}}}}})
        client = make(self.dir, fake)
        client.journal.record("p1")
        outputs = client.wait("p1", timeout=5)
        self.assertIn("9", outputs)
        self.assertEqual(client.journal.read("p1")["status"], comfy.Journal.DONE)
        self.assertEqual(client.journal.read("p1")["images"][0]["filename"], "x.png")

    def test_wait_raises_and_records_a_failed_prompt(self):
        fake = FakeComfy({"/history/p1": {"p1": {"status": {"status_str": "error", "messages": ["boom"]}}}})
        client = make(self.dir, fake)
        client.journal.record("p1")
        with self.assertRaises(comfy.ComfyError):
            client.wait("p1", timeout=5)
        self.assertEqual(client.journal.read("p1")["status"], comfy.Journal.FAILED)

    def test_cancel_stops_the_wait_and_records_it(self):
        fake = FakeComfy({})
        client = make(self.dir, fake)
        client.journal.record("p1")
        with self.assertRaises(comfy.Cancelled):
            client.wait("p1", timeout=25, cancel=lambda: True)
        self.assertEqual(client.journal.read("p1")["status"], comfy.Journal.CANCELLED)
        paths = [urllib.parse.urlsplit(r.full_url).path for r in fake.requests if getattr(r, "data", None)]
        self.assertIn("/queue", paths)

    def test_recover_settles_a_prompt_the_server_finished(self):
        fake = FakeComfy({"/history/p1": {"p1": {"status": {"status_str": "success"},
                                                "outputs": {"9": {"images": [{"filename": "x.png"}]}}}}})
        client = make(self.dir, fake)
        client.journal.record("p1")
        settled = client.recover()
        self.assertIn("p1", settled)
        self.assertEqual(client.journal.read("p1")["status"], comfy.Journal.DONE)

    def test_recover_marks_a_prompt_the_server_forgot_as_failed(self):
        fake = FakeComfy({"/queue": {"queue_running": [], "queue_pending": []}})
        client = make(self.dir, fake)
        client.journal.record("p1")
        client.recover()
        record = client.journal.read("p1")
        self.assertEqual(record["status"], comfy.Journal.FAILED)
        self.assertIn("unknown to ComfyUI", record["error"])

    def test_recover_leaves_a_still_queued_prompt_alone(self):
        fake = FakeComfy({"/queue": {"queue_running": [[0, "p1", {}, {}, []]], "queue_pending": []}})
        client = make(self.dir, fake)
        client.journal.record("p1")
        client.recover()
        self.assertEqual(client.journal.read("p1")["status"], comfy.Journal.SUBMITTED)


class RecoveryTests(unittest.TestCase):
    """R6: a lost answer, a failed download and a crash mid-manifest recover without submitting twice."""

    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make(self, fake):
        return comfy.Comfy("http://127.0.0.1:9", journal=self.dir / "journal", opener=fake, timeout=1)

    def test_the_intent_is_on_disk_before_the_post(self):
        seen = []
        journal_dir = self.dir / "journal"

        class Watching(FakeComfy):
            def __call__(self, req, timeout=None):
                if getattr(req, "data", None) and not seen:
                    seen.extend(journal_dir.glob("intent-*.json"))
                return super().__call__(req, timeout)

        client = self.make(Watching({"/prompt": {"prompt_id": "p-123"}}))
        client.submit({"1": {}}, slot_id="frm_x", seed=1)
        self.assertEqual(len(seen), 1, "the intent exists while the POST is in flight")
        self.assertFalse(list((self.dir / "journal").glob("intent-*.json")),
                         "the answered intent became the prompt's record")
        record = client.journal.entries()[0]
        self.assertEqual(record["status"], "submitted")
        self.assertEqual(record["slot_id"], "frm_x")

    def test_a_lost_answer_is_adopted_from_the_queue_by_client_id(self):
        client_id = "abc123"
        fake = FakeComfy({"/queue": {"queue_running": [[0, "p-lost", {}, {"client_id": client_id}, []]],
                                     "queue_pending": []},
                          "/history/p-lost": {"p-lost": {"status": {"status_str": "success"},
                                                         "outputs": {"1": {"images": [
                                                             {"filename": "x.png", "subfolder": "", "type": "output"}]}}}}})
        client = self.make(fake)
        client.journal.record_intent(client_id, slot_id="frm_x", seed=7)
        settled = client.recover()
        self.assertIn("p-lost", settled)
        record = client.journal.read("p-lost")
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["slot_id"], "frm_x")
        self.assertEqual(client.journal.read_intent(client_id), None)

    def test_a_lost_answer_is_adopted_from_history_when_the_queue_forgot_it(self):
        client_id = "abc123"
        fake = FakeComfy({
            "/queue": {"queue_running": [], "queue_pending": []},
            "/history": {"p-done": {"status": {"status_str": "success"},
                                    "outputs": {"1": {"images": [
                                        {"filename": "x.png", "subfolder": "", "type": "output"}]}},
                                    "prompt": [0, "p-done", {}, {"client_id": client_id}]}},
            "/history/p-done": {"p-done": {"status": {"status_str": "success"},
                                           "outputs": {"1": {"images": [
                                               {"filename": "x.png", "subfolder": "", "type": "output"}]}}}}})
        client = self.make(fake)
        client.journal.record_intent(client_id, slot_id="frm_x", seed=7)
        settled = client.recover()
        self.assertIn("p-done", settled)
        self.assertEqual(client.journal.read("p-done")["status"], "done")

    def test_an_untraceable_intent_is_recorded_unconfirmed_not_resubmitted(self):
        fake = FakeComfy({"/queue": {"queue_running": [], "queue_pending": []}})
        client = self.make(fake)
        client.journal.record_intent("gone", slot_id="frm_x", seed=7)
        client.recover()
        intents = [e for e in client.journal.entries() if e.get("client_id") == "gone"]
        self.assertEqual(intents[0]["status"], "failed")
        self.assertIn("unconfirmed", intents[0]["error"])
        self.assertEqual(fake.posted(), [], "recovery never submits anything")

    def test_a_done_cell_whose_image_never_landed_is_fetched_not_resubmitted(self):
        from PIL import Image
        import io as io_mod
        buffer = io_mod.BytesIO()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
        png = buffer.getvalue()
        fake = FakeComfy({"/view": png})
        client = self.make(fake)
        client.journal.record("p1", slot_id="frm_back", method="krea2_b", seed=2000)
        client.journal.update("p1", "done", images=[{"filename": "f.png", "subfolder": "", "type": "output"}])
        out = self.dir / "out.png"
        data = client.run({"1": {}}, out, slot_id="frm_back", method="krea2_b", seed=2000)
        self.assertEqual(data, png)
        self.assertTrue(out.is_file())
        self.assertEqual(fake.posted(), [], "the recorded output is fetched, not generated again")

    def test_a_done_record_for_a_present_file_is_ignored(self):
        fake = FakeComfy({"/prompt": {"prompt_id": "p2"},
                          "/history/p2": {"p2": {"status": {"status_str": "success"},
                                                 "outputs": {"1": {"images": [
                                                     {"filename": "f.png", "subfolder": "",
                                                      "type": "output"}]}}}}})
        client = self.make(fake)
        client.journal.record("p1", slot_id="frm_back", method="krea2_b", seed=2000)
        client.journal.update("p1", "done", images=[{"filename": "f.png", "subfolder": "", "type": "output"}])
        out = self.dir / "out.png"
        out.write_bytes(b"already here")
        client.run({"1": {}}, out, slot_id="frm_back", method="krea2_b", seed=2000)
        self.assertEqual(len(fake.posted()), 1, "the file exists: this is a normal draw, not a recovery")


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_image_appears_only_after_the_rename(self):
        target = self.dir / "out.png"
        comfy.save_image(target, b"PNGDATA")
        self.assertEqual(target.read_bytes(), b"PNGDATA")
        self.assertEqual(list(self.dir.iterdir()), [target])

    def test_a_failure_while_renaming_leaves_no_file(self):
        target = self.dir / "out.png"
        with patch.object(comfy.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                comfy.save_image(target, b"PNGDATA")
        self.assertFalse(target.exists())
        self.assertEqual([p.name for p in self.dir.iterdir() if not p.name.startswith("out.png.tmp-")], [])


class NameResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = comfy.Comfy("http://127.0.0.1:9", comfy_root="G:/Lakis/ComfyUI", opener=FakeComfy())

    def tearDown(self):
        self.temp.cleanup()

    def test_a_path_under_the_loras_folder_resolves_to_its_relative_name(self):
        name = self.client.name_for("lora", "G:/Lakis/ComfyUI/models/loras/2. krea2/suamom.safetensors")
        self.assertEqual(name, "2. krea2\\suamom.safetensors")

    def test_a_file_outside_the_folders_resolves_by_unique_name(self):
        name = self.client.name_for(
            "lora", "G:/Lakis/ComfyUI/models/diffusion_models/krea2/krea2_turbo_lora_rank_64_bf16.safetensors")
        self.assertEqual(name, "krea2_turbo_lora_rank_64_bf16.safetensors")

    def test_an_invisible_file_names_the_folder_it_belongs_in(self):
        with self.assertRaises(comfy.ComfyError) as caught:
            self.client.name_for("lora", "G:/Lakis/ComfyUI/models/loras/missing.safetensors")
        self.assertIn("models\\loras", str(caught.exception).replace("/", "\\"))

    def test_an_ambiguous_name_is_refused(self):
        fake = FakeComfy()
        fake.object_info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"] = [
            ["a\\same.safetensors", "b\\same.safetensors"], {}]
        client = comfy.Comfy("http://127.0.0.1:9", comfy_root="G:/Lakis/ComfyUI", opener=fake)
        with self.assertRaises(comfy.ComfyError) as caught:
            client.name_for("lora", "G:/elsewhere/same.safetensors")
        self.assertIn("ambiguous", str(caught.exception))

    def test_krea2_is_reported_as_an_installed_clip_type(self):
        self.assertIn("krea2", self.client.clip_types())

    def test_missing_nodes_are_listed(self):
        self.assertEqual(self.client.missing_nodes(["UNETLoader", "NoSuchNode"]), ["NoSuchNode"])


class GpuTests(unittest.TestCase):
    def test_free_asks_the_server_to_unload_without_killing_it(self):
        fake = FakeComfy({"/free": b"{}"})
        client = comfy.Comfy("http://127.0.0.1:9", opener=fake)
        client.free()
        body = json.loads([r for r in fake.requests if getattr(r, "data", None)][0].data.decode())
        self.assertEqual(body, {"unload_models": True, "free_memory": True})

    def test_idle_reads_the_queue(self):
        fake = FakeComfy({"/queue": {"queue_running": [], "queue_pending": []}})
        client = comfy.Comfy("http://127.0.0.1:9", opener=fake)
        self.assertTrue(client.idle())


class RecoveryScopeTests(unittest.TestCase):
    """`run(recover=...)`: what a done journal record is allowed to stand in for.

    Found while measuring the real final LoRA. The recovery matches a done record by whatever `meta` the caller
    passed - for the holdout that is slot and seed - and the journal does not record the prompt. Two holdout prompts
    share a seed, so the second cell was served the first cell's picture from a previous run and both prompts came
    back pixel-identical. A holdout cell must always be its own submission.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_done(self, journal, prompt_id, **meta):
        journal.record(prompt_id, **meta)
        journal.update(prompt_id, comfy.Journal.DONE,
                       images=[{"filename": "old.png", "subfolder": "", "type": "output"}])

    def history(self, prompt_id, filename):
        return {f"/history/{prompt_id}": {prompt_id: {
            "status": {"status_str": "success"},
            "outputs": {"9": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}}}}}

    def test_recovery_serves_a_recorded_cell_whose_file_never_landed(self):
        journal = comfy.Journal(self.dir / "journal")
        self.write_done(journal, "p-old", slot_id="s_a", seed=1)
        fake = FakeComfy({"/view": b"OLD"})
        client = make(self.dir, fake)
        out = self.dir / "cell.png"
        data = client.run({"1": {}}, out, slot_id="s_a", seed=1)
        self.assertEqual(data, b"OLD")
        self.assertTrue(out.is_file())
        self.assertEqual(fake.posted(), [], "the recovery must not submit")

    def test_recover_false_forces_a_fresh_submission(self):
        journal = comfy.Journal(self.dir / "journal")
        self.write_done(journal, "p-old", slot_id="s_a", seed=1)
        fake = FakeComfy({"/prompt": {"prompt_id": "p-new"}, "/view": b"NEW",
                          **self.history("p-new", "new.png")})
        client = make(self.dir, fake)
        out = self.dir / "cell.png"
        data = client.run({"1": {}}, out, slot_id="s_a", seed=1, recover=False)
        self.assertEqual(data, b"NEW", "recover=False must ask for a new picture, not adopt the recorded one")

    def test_two_prompts_with_one_seed_do_not_collapse_to_one_picture(self):
        # The defect, in miniature: two cells, same slot/seed, so the second would find the first's record.
        journal = comfy.Journal(self.dir / "journal")
        self.write_done(journal, "p-first", slot_id="s_a", seed=1)
        fake = FakeComfy({"/prompt": {"prompt_id": "p-second"}, "/view": b"SECOND",
                          **self.history("p-second", "b.png")})
        client = make(self.dir, fake)
        out = self.dir / "second.png"
        data = client.run({"1": {}}, out, slot_id="s_a", seed=1, recover=False)
        self.assertEqual(data, b"SECOND")
        self.assertNotEqual(data, b"FIRST")


if __name__ == "__main__":
    unittest.main()
