"""Offline regression checks for dataset selection, plans and queue recovery."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import sys

SPEC = importlib.util.spec_from_file_location("lora_runner", Path(__file__).with_name("runner.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.AUTO)
        self.root = Path(self.temp.name)
        self.folder = self.root / "source"
        self.folder.mkdir()
        from PIL import Image
        Image.new("RGB", (128, 128), "red").save(self.folder / "one.png")
        (self.folder / "two.png").write_bytes((self.folder / "one.png").read_bytes())
        (self.folder / "bad.jpg").write_text("invalid image")
        (self.folder / "ignored.txt").write_text("not a sidecar")
        self.cfg = runner.settings({"id": "test_job", "trigger": "testtrigger", "dataset_root": str(self.folder)})

    def tearDown(self):
        self.temp.cleanup()

    def test_dedup_and_corrupt_files(self):
        images, rejected = runner.inventory(self.cfg)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(rejected), 2)

    def test_source_changes_version(self):
        before = runner.context(self.cfg)[0]
        (self.folder / "one.txt").write_text("a red image", encoding="utf-8")
        after = runner.context(self.cfg)[0]
        self.assertNotEqual(before, after)

    def test_invalid_job_path_and_limit(self):
        with self.assertRaises(ValueError):
            runner.settings({"id": "../bad", "trigger": "test"})
        self.cfg["max_images_per_job"] = 0
        with self.assertRaises(ValueError):
            runner.inventory(self.cfg)

    def test_prepare_preserves_original_and_prefixes_caption(self):
        (self.folder / "one.txt").write_text("a red image", encoding="utf-8")
        original = (self.folder / "one.png").read_bytes()
        auto = self.root / "automation"
        auto.mkdir()
        (auto / "caption_instructions.txt").write_text("describe image")
        with patch.object(runner, "AUTO", auto):
            run, images, _ = runner.prepare(self.cfg)
            cap = next((run / "images").glob("*.txt")).read_text()
            self.assertEqual(cap.strip(), "testtrigger, a red image")
            self.assertEqual((self.folder / "one.png").read_bytes(), original)

    def test_native_plan_steps_and_adapter(self):
        from fizgig.families.registry import get
        from fizgig.families.launch import plan as native_plan
        image_dir = self.root / "images"
        image_dir.mkdir()
        (image_dir / "one.png").write_bytes((self.folder / "one.png").read_bytes())
        (image_dir / "one.txt").write_text("testtrigger, a red image")
        model = self.root / "placeholder.safetensors"
        model.touch()
        # The runner now resolves model paths through pipeline.models (config.json over prefs.json), so the placeholder
        # stands in for that resolution rather than for prefs.json alone.
        resolved = {key: str(model) for key in ("qwen21_dit", "qwen21_vae", "qwen21_text_encoder",
                                                "qwen21_training_adapter")}
        with patch.object(runner.model_paths, "resolve", return_value=resolved):
            result = runner.launch_plan(self.root, self.cfg, 32)
        self.assertEqual([s.name for s in result.stages], ["Cache Preparation", "Text Encoder Caching", "Training"])
        cmd = result.stages[-1].cmd
        self.assertEqual(cmd[cmd.index("--max_train_epochs") + 1], "94")
        self.assertEqual(cmd[cmd.index("--training_adapter") + 1], str(model))
        self.assertIn("--save_state", cmd)
        self.assertIn("--sample_prompts", cmd)
        self.assertIn("--sample_every_n_epochs", cmd)

    def test_local_caption_needs_no_key(self):
        images_dir = self.root / "images"
        images_dir.mkdir()
        (images_dir / "one.png").write_bytes((self.folder / "one.png").read_bytes())
        client = MagicMock()
        client.caption.return_value = ("testtrigger, a red image", {"total_tokens": 30})
        with patch.dict(runner.os.environ, {"OPENAI_API_KEY": ""}), patch.object(runner, "GemmaServer") as server:
            server.return_value.__enter__.return_value = client
            runner.captions(self.root, self.cfg, "describe")
            client.caption.assert_called_once()
        self.assertEqual((images_dir / "one.txt").read_text().strip(), "testtrigger, a red image")

    def test_nonlocal_endpoint_rejected(self):
        cfg = {**self.cfg, "caption_base_url": "https://api.openai.com/v1"}
        with self.assertRaises(ValueError):
            runner.LocalClient(cfg)

    def test_local_payload_and_reasoning_exclusion(self):
        client = runner.LocalClient(self.cfg)
        answer = {"choices": [{"finish_reason": "stop", "message": {
            "content": "<think>private analysis</think>A red image.", "reasoning_content": "unused"}}],
            "usage": {"total_tokens": 30}}
        with patch.object(client, "json_request", return_value=answer) as call:
            text, usage = client.caption(self.folder / "one.png", "Describe only visible colors.")
            self.assertEqual(text, "testtrigger, A red image.")
            url, payload = call.call_args.args[:2]
            self.assertEqual(url, "http://127.0.0.1:18080/v1/chat/completions")
            self.assertEqual(payload["reasoning_effort"], "none")
            self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")
            self.assertNotIn("Authorization", payload)

    def test_truncated_caption_not_saved(self):
        client = runner.LocalClient(self.cfg)
        with patch.object(client, "json_request", return_value={"choices": [
                {"finish_reason": "length", "message": {"content": "partial"}}]}):
            with self.assertRaises(RuntimeError):
                client.caption(self.folder / "one.png", "Describe image.")

    def test_existing_server_is_never_stopped(self):
        server = runner.GemmaServer(self.cfg, self.root)
        with patch.object(server.client, "ready", return_value=True), \
             patch.object(server.client, "verify_model"), patch("local_caption.subprocess.run") as taskkill:
            with server:
                self.assertTrue(server.external)
            taskkill.assert_not_called()

    def test_missing_captions_load_once_and_existing_skipped(self):
        images_dir = self.root / "images"
        images_dir.mkdir()
        (images_dir / "one.png").write_bytes((self.folder / "one.png").read_bytes())
        (images_dir / "one.txt").write_text("existing caption")
        with patch.object(runner, "GemmaServer") as server:
            runner.captions(self.root, self.cfg, "describe")
            server.assert_not_called()

    def make_auto(self):
        auto = self.root / "automation"
        auto.mkdir()
        (auto / "caption_instructions.txt").write_text("describe image", encoding="utf-8")
        return auto

    def test_training_setting_change_reuses_matching_captions(self):
        auto = self.make_auto()
        with patch.object(runner, "AUTO", auto):
            before, _, _ = runner.prepare(self.cfg)
            cap = next((before / "images").glob("*.png")).with_suffix(".txt")
            cap.write_text("testtrigger, a reviewed caption", encoding="utf-8")
            changed = {**self.cfg, "target_steps": 32, "rank": 16, "caption_startup_timeout": 999}
            after, _, instructions = runner.prepare(changed)
            self.assertNotEqual(before, after)
            self.assertEqual(next((after / "images").glob("*.txt")).read_text().strip(), "testtrigger, a reviewed caption")
            with patch.object(runner, "GemmaServer") as server:
                runner.captions(after, changed, instructions)
                server.assert_not_called()

    def test_changed_instructions_do_not_reuse_old_captions(self):
        auto = self.make_auto()
        with patch.object(runner, "AUTO", auto):
            before, _, _ = runner.prepare(self.cfg)
            next((before / "images").glob("*.png")).with_suffix(".txt").write_text("old caption")
            (auto / "caption_instructions.txt").write_text("a different instruction")
            after, _, _ = runner.prepare(self.cfg)
            self.assertNotEqual(before, after)
            self.assertFalse(list((after / "images").glob("*.txt")))

    def test_working_caption_changes_fingerprint(self):
        auto = self.make_auto()
        with patch.object(runner, "AUTO", auto):
            run, _, _ = runner.prepare(self.cfg)
            before = runner.working_fingerprint(run)
            next((run / "images").glob("*.png")).with_suffix(".txt").write_text("a changed caption")
            self.assertNotEqual(before, runner.working_fingerprint(run))

    def test_completed_flag_cannot_hide_missing_final_lora(self):
        auto = self.make_auto()
        with patch.object(runner, "AUTO", auto):
            run, _, _ = runner.prepare(self.cfg)
            runner.write_json(run / "state.json", {"complete": True, "completed_stages": ["Training"],
                              "dataset_fingerprint": runner.working_fingerprint(run)})
            result = SimpleNamespace(dirs=[], files=[], stages=[SimpleNamespace(name="Training", cmd=["fixture"])])
            group = MagicMock()
            group.__len__.return_value = 1
            with patch.object(runner.LocalClient, "listening", return_value=False), \
                 patch.object(runner, "launch_plan", return_value=result), \
                 patch.object(runner, "caches_valid", return_value=True), \
                 patch("fizgig.dataset.config.BlueprintGenerator"), \
                 patch("fizgig.dataset.config.load_user_config"), \
                 patch("fizgig.dataset.config.generate_dataset_group_by_blueprint", return_value=group), \
                 patch.object(runner.subprocess, "call", return_value=0) as child:
                with self.assertRaisesRegex(RuntimeError, "without the final LoRA"):
                    runner.execute(run, self.cfg, 1)
                child.assert_called_once()
                self.assertFalse(runner.read_json(run / "state.json")["complete"])

    def test_changed_dataset_forces_latent_rebuild(self):
        auto = self.make_auto()
        with patch.object(runner, "AUTO", auto):
            run, _, _ = runner.prepare(self.cfg)
            runner.write_json(run / "state.json", {"completed_stages": ["Cache Preparation"], "dataset_fingerprint": "outdated"})
            result = SimpleNamespace(dirs=[], files=[], stages=[SimpleNamespace(name="Cache Preparation", cmd=["fixture", "--skip_existing"])])
            with patch.object(runner.LocalClient, "listening", return_value=False), \
                 patch.object(runner, "launch_plan", return_value=result), \
                 patch.object(runner.subprocess, "call", return_value=0) as child:
                runner.execute(run, self.cfg, 1)
                self.assertEqual(child.call_args.args[0], ["fixture"])
                self.assertFalse(runner.read_json(run / "state.json")["rebuild_latents"])

    def test_legacy_matching_cache_preserves_saved_resume_state(self):
        auto = self.make_auto()
        with patch.object(runner, "AUTO", auto):
            run, _, _ = runner.prepare(self.cfg)
            runner.write_json(run / "state.json", {"completed_stages": ["Cache Preparation", "Text Encoder Caching"]})
            saved = run / "output" / "test_job-000016-state"
            saved.mkdir(parents=True)
            for name in ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt"):
                (saved / name).write_text("fixture")
            result = SimpleNamespace(dirs=[], files=[], stages=[SimpleNamespace(name="Training", cmd=["fixture"])])
            group = MagicMock()
            group.__len__.return_value = 1
            with patch.object(runner.LocalClient, "listening", return_value=False), \
                 patch.object(runner, "launch_plan", return_value=result), \
                 patch.object(runner, "caches_valid", return_value=True), \
                 patch("fizgig.dataset.config.BlueprintGenerator"), \
                 patch("fizgig.dataset.config.load_user_config"), \
                 patch("fizgig.dataset.config.generate_dataset_group_by_blueprint", return_value=group), \
                 patch.object(runner.subprocess, "call", return_value=1) as child:
                with self.assertRaises(RuntimeError):
                    runner.execute(run, self.cfg, 1)
                self.assertEqual(child.call_args.args[0], ["fixture", "--resume", str(saved)])

    def test_doctor_fails_when_cuda_or_models_are_missing(self):
        def read(path):
            return [{"id": "test_job", "trigger": "testtrigger"}] if Path(path).name == "todo.json" else {}
        with patch.object(sys, "argv", ["runner.py", "doctor"]), \
             patch.object(runner, "read_json", side_effect=read), \
             patch.object(runner, "settings", return_value=self.cfg), \
             patch("torch.cuda.is_available", return_value=False), \
             patch.object(runner.LocalClient, "ready", return_value=False):
            with self.assertRaisesRegex(ValueError, "Environment check failed"):
                runner.main()

if __name__ == "__main__":
    unittest.main()


class FreeBeforeTrainingTests(unittest.TestCase):
    """R6: training asks the generation server for its VRAM back, but only when nothing is running on it."""

    def setUp(self):
        self.calls = []

    def client(self, *, answers_ping=True, answers_idle=True):
        class FakeComfy:
            def ping(self):
                return answers_ping

            def idle(self):
                return answers_idle

            def free(self):
                self.calls.append("free")
                return {}

        fake = FakeComfy()
        fake.calls = self.calls
        return fake

    def cfg(self, url="http://127.0.0.1:8190"):
        return {"comfy": {"url": url}}

    def test_an_idle_server_frees_its_vram(self):
        fake = self.client()
        self.assertTrue(runner.free_comfy_before_training(self.cfg(), client=fake, log=lambda *_: None))
        self.assertEqual(fake.calls, ["free"])

    def test_a_busy_server_is_not_interrupted(self):
        fake = self.client(answers_idle=False)
        self.assertFalse(runner.free_comfy_before_training(self.cfg(), client=fake, log=lambda *_: None))
        self.assertEqual(fake.calls, [], "freeing under a live request would yank its models")

    def test_an_unreachable_server_is_skipped(self):
        fake = self.client(answers_ping=False)
        self.assertFalse(runner.free_comfy_before_training(self.cfg(), client=fake, log=lambda *_: None))
        self.assertEqual(fake.calls, [])

    def test_no_server_in_the_config_is_skipped(self):
        self.assertFalse(runner.free_comfy_before_training({"comfy": {}}, client=self.client(),
                                                          log=lambda *_: None))
        self.assertEqual(self.calls, [])
