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

    def test_doctor_only_fails_on_families_the_jobs_use(self):
        problems = {"krea2": [], "qwen_image21": ["Qwen Image 2.1 DiT file does not exist: x"]}
        with patch.object(runner.model_paths, "families", return_value=["qwen_image21", "krea2"]), \
             patch.object(runner.model_paths, "rows", return_value=[]), \
             patch.object(runner.model_paths, "problems", side_effect=lambda f: problems[f]), \
             patch("builtins.print"):
            failures = []
            self.assertEqual(runner.doctor_models(failures, used={"krea2"}), ["krea2"])
            self.assertEqual(failures, [])
            failures = []
            runner.doctor_models(failures, used={"qwen_image21"})
            self.assertEqual(failures, ["[qwen_image21] Qwen Image 2.1 DiT file does not exist: x"])
            failures = []
            runner.doctor_models(failures)
            self.assertEqual(len(failures), 1)

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
        self.assertEqual(runner.free_comfy_before_training(self.cfg(), client=fake, log=lambda *_: None), "freed")
        self.assertEqual(fake.calls, ["free"])

    def test_a_busy_server_is_not_interrupted_and_reports_busy(self):
        fake = self.client(answers_idle=False)
        # spec N1.4: "busy" is the caller's signal to refuse the training start, not to free under it.
        self.assertEqual(runner.free_comfy_before_training(self.cfg(), client=fake, log=lambda *_: None), "busy")
        self.assertEqual(fake.calls, [], "freeing under a live request would yank its models")

    def test_an_unreachable_server_is_skipped(self):
        fake = self.client(answers_ping=False)
        self.assertEqual(runner.free_comfy_before_training(self.cfg(), client=fake, log=lambda *_: None),
                         "unreachable")
        self.assertEqual(fake.calls, [])

    def test_no_server_in_the_config_is_skipped(self):
        self.assertEqual(runner.free_comfy_before_training({"comfy": {}}, client=self.client(),
                                                           log=lambda *_: None), "no-server")
        self.assertEqual(self.calls, [])

    def test_every_configured_address_is_checked_for_busyness(self):
        # spec N1.4: another address of the same-GPU server holds the VRAM too, so the 8190/8192 pair must both
        # be consulted before a training start.
        fake = self.client()
        cfg = {"comfy": {"url": "http://127.0.0.1:8190", "extra_urls": ["http://127.0.0.1:8192"]}}
        self.assertEqual(runner.free_comfy_before_training(cfg, client=fake, log=lambda *_: None), "freed")
        self.assertEqual(fake.calls, ["free"])

    def test_a_busy_second_address_refuses_even_when_the_first_is_idle(self):
        fake = self.client(answers_idle=True)
        cfg = {"comfy": {"url": "http://127.0.0.1:8190", "extra_urls": ["http://127.0.0.1:8192"]}}
        # The client argument only covers the first address; the extra address builds its own client, so simulate
        # the second one being busy through a patched Comfy constructor.
        import pipeline.comfy as comfy_mod
        original = comfy_mod.Comfy
        busy = {"second": False}

        class SecondBusy:
            def __init__(self, url, **_kwargs):
                pass

            def ping(self):
                return True

            def idle(self):
                return not busy["second"]

            def free(self):
                fake.calls.append("free-extra")

        comfy_mod.Comfy = SecondBusy
        try:
            busy["second"] = False
            self.assertEqual(runner.free_comfy_before_training(cfg, client=fake, log=lambda *_: None), "freed")
            self.assertEqual(fake.calls, ["free", "free-extra"],
                             "every reachable idle address releases its VRAM")
            busy["second"] = True
            self.assertEqual(runner.free_comfy_before_training(cfg, client=fake, log=lambda *_: None), "busy")
        finally:
            comfy_mod.Comfy = original


class CloudQueueTests(unittest.TestCase):
    """`run`/`caption` take the queue lock first; local segments claim gpu.lock per job; cloud jobs never claim it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.auto = self.root / "auto"
        self.auto.mkdir()
        self.root.joinpath("caption_instructions.txt").write_bytes(
            (runner.AUTO / "caption_instructions.txt").read_bytes())
        self.root.joinpath("caption_instructions_anime.txt").write_bytes(
            (runner.AUTO / "caption_instructions_anime.txt").read_bytes())
        self._old = {key: runner._PATH_OVERRIDES.get(key) for key in ("config", "todo", "runs")}
        runner._PATH_OVERRIDES["runs"] = self.runs
        runner._PATH_OVERRIDES["todo"] = self.root / "todo.json"
        runner._PATH_OVERRIDES["config"] = self.root / "config.json"
        (self.root / "config.json").write_text(json.dumps(
            {"target_steps": 2, "max_images_per_job": 10, "rank": 8, "save_every_steps_approx": 1,
             "learning_rate": 0.0002, "adaptive_lr": True, "adaptive_lr_min": 0.0002, "adaptive_lr_max": 0.0004,
             "megapixels": 0.25, "caption_base_url": "http://127.0.0.1:18080/v1",
             "caption_model": "gemma", "caption_max_output_tokens": 300, "sample_prompts": ["{trigger}, portrait"]}), encoding="utf-8")
        self.datasets = {}
        from PIL import Image
        for name, color in (("a", "red"), ("b", "blue"), ("c", "green")):
            ds = self.root / f"ds_{name}"
            ds.mkdir()
            Image.new("RGB", (16, 16), color).save(ds / "one.png")
            self.datasets[name] = ds
        # The cloud job has captions already; the local jobs rely on the fake Gemma.
        (self.datasets["b"] / "one.txt").write_text("trg, a caption", encoding="utf-8")
        self.todo = self.root / "todo.json"

    def tearDown(self):
        for key, value in self._old.items():
            if value is None:
                runner._PATH_OVERRIDES.pop(key, None)
            else:
                runner._PATH_OVERRIDES[key] = value
        self.temp.cleanup()

    def write_todo(self, jobs):
        self.todo.write_text(json.dumps(jobs), encoding="utf-8")

    def job(self, name, *, backend=None):
        job = {"id": f"job{name}", "enabled": True, "trigger": "trg",
               "dataset_root": str(self.datasets[name])}
        if backend == "modal":
            job["backend"] = "modal"
            job["modal_gpu"] = "L40S"
            job["family"] = "krea2"
        return job

    class FakeClaim:
        def __init__(self, recorder, what):
            self.recorder, self.what = recorder, what

        def acquire(self):
            self.recorder.append(("acquire", self.what))

        def release(self):
            self.recorder.append(("release", self.what))

    def run_main(self, argv, execute_recorder, claim_recorder):
        runner_locks = []

        def fake_claim_gpu(automation_dir=None, *, what="generation"):
            return CloudQueueTests.FakeClaim(claim_recorder, what)

        class FakeGemma:
            def __init__(self, cfg, run, cancelled=None):
                self.cfg = cfg

            def __enter__(self):
                self = self
                return self

            def __exit__(self, *exc):
                return False

            def caption(self, path, instructions):
                return "trg, a caption", {"total": 1}

        with patch.object(runner, "AUTO", self.auto), \
             patch.object(runner, "execute", side_effect=lambda run, cfg, n: execute_recorder.append(cfg["id"])), \
             patch.object(runner, "GemmaServer", FakeGemma), \
             patch.object(runner.model_paths, "problems", return_value=[]), \
             patch.object(runner, "free_comfy_before_training", return_value="no-server"), \
             patch("pipeline.gpu.vram_gate", lambda *a, **k: None), \
             patch("pipeline.gpu.take_queue_lock", lambda *a, **k: None), \
             patch("pipeline.gpu.claim_gpu", side_effect=fake_claim_gpu), \
             patch.object(sys, "argv", argv):
            runner.main()
        return runner_locks

    def test_mixed_queue_runs_in_order_and_claims_gpu_only_for_local_jobs(self):
        self.write_todo([self.job("a"), self.job("b", backend="modal"), self.job("c")])
        executed, claims = [], []
        self.run_main(["runner.py", "run"], executed, claims)
        self.assertEqual(executed, ["joba", "jobb", "jobc"])
        acquires = [what for action, what in claims if action == "acquire"]
        releases = [what for action, what in claims if action == "release"]
        self.assertEqual(acquires, ["run joba", "run jobc"])   # the Modal job never claims the GPU
        self.assertEqual(releases, acquires)

    def test_cloud_job_with_missing_caption_refuses_before_any_work(self):
        (self.datasets["b"] / "one.txt").unlink()
        self.write_todo([self.job("b", backend="modal")])
        executed, claims = [], []
        with self.assertRaisesRegex(ValueError, "cannot caption"):
            self.run_main(["runner.py", "run"], executed, claims)
        self.assertEqual(executed, [])
        self.assertEqual(claims, [])

    def test_plan_for_a_cloud_job_is_offline_and_posix(self):
        self.write_todo([self.job("b", backend="modal")])
        self.run_main(["runner.py", "plan"], [], [])
        plans = list(self.runs.glob("jobb-*/launch_plan.json"))
        self.assertEqual(len(plans), 1)
        doc = json.loads(plans[0].read_text(encoding="utf-8"))
        self.assertEqual(doc["family"], "krea2")
        for stage in doc["stages"]:
            for arg in stage["cmd"]:
                self.assertNotIn("\\", str(arg))
                self.assertNotRegex(str(arg), r"^[A-Za-z]:")
        for path, _content in doc["files"]:
            self.assertTrue(path.startswith("/runs/"))
            self.assertNotIn("\\", path)


class DoctorCloudTests(unittest.TestCase):
    """`doctor` checks the Modal side for cloud jobs and never fails cloud-only runs on local GPU/model gaps."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.auto = self.root / "auto"
        self.auto.mkdir()
        self._old = {key: runner._PATH_OVERRIDES.get(key) for key in ("config", "todo", "runs")}
        runner._PATH_OVERRIDES["runs"] = self.runs
        runner._PATH_OVERRIDES["todo"] = self.root / "todo.json"
        runner._PATH_OVERRIDES["config"] = self.root / "config.json"
        ds = self.root / "ds"
        ds.mkdir()
        from PIL import Image
        Image.new("RGB", (16, 16), "red").save(ds / "one.png")
        (ds / "one.txt").write_text("trg, a caption", encoding="utf-8")
        self.root.joinpath("caption_instructions.txt").write_bytes(
            (runner.AUTO / "caption_instructions.txt").read_bytes())
        (self.root / "config.json").write_text(json.dumps(
            {"target_steps": 2, "max_images_per_job": 10, "rank": 8, "save_every_steps_approx": 1,
             "learning_rate": 0.0002, "adaptive_lr": True, "adaptive_lr_min": 0.0002,
             "adaptive_lr_max": 0.0004, "megapixels": 0.25,
             "caption_base_url": "http://127.0.0.1:18080/v1",
             "caption_model": "gemma", "caption_max_output_tokens": 300, "sample_prompts": ["{trigger}, portrait"]}), encoding="utf-8")
        self.ds = ds

    def tearDown(self):
        for key, value in self._old.items():
            if value is None:
                runner._PATH_OVERRIDES.pop(key, None)
            else:
                runner._PATH_OVERRIDES[key] = value
        self.temp.cleanup()

    def todo(self, jobs):
        (self.root / "todo.json").write_text(json.dumps(jobs), encoding="utf-8")

    def modal_job(self):
        return {"id": "cloudjob", "enabled": True, "trigger": "trg", "dataset_root": str(self.ds),
                "backend": "modal", "family": "krea2"}

    def fake_modal_module(self, **behavior):
        import types

        class NotFoundError(Exception):
            pass

        class Function:
            @staticmethod
            def from_name(app_name, name):
                if behavior.get("deployed") is None:
                    raise NotFoundError(f"NotFoundError: {app_name}.{name}")
                return types.SimpleNamespace(remote=lambda: behavior["deployed"])

        class Volume:
            @staticmethod
            def from_name(name, create_if_missing=False):
                raise AssertionError("doctor must not touch volumes")

        return types.SimpleNamespace(Function=Function, Volume=Volume,
                                     exception=types.SimpleNamespace(NotFoundError=NotFoundError))

    def test_doctor_for_modal_jobs_passes_without_local_cuda(self):
        self.todo([self.modal_job()])
        saved = sys.modules.get("modal")
        sys.modules["modal"] = self.fake_modal_module(deployed=None)   # app not deployed yet is not a failure
        try:
            with patch.object(sys, "argv", ["runner.py", "doctor"]):
                runner.main()   # must not raise: CUDA absence is not a cloud-job failure (torch is never imported)
        finally:
            if saved is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = saved

    def test_doctor_reports_a_missing_modal_package(self):
        self.todo([self.modal_job()])
        saved = sys.modules.get("modal")
        sys.modules["modal"] = None     # `import modal` raises ImportError
        try:
            with patch.object(sys, "argv", ["runner.py", "doctor"]):
                with self.assertRaisesRegex(ValueError, "modal"):
                    runner.main()
        finally:
            if saved is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = saved

    def test_doctor_fails_on_a_modal_job_with_missing_captions(self):
        (self.ds / "one.txt").unlink()
        self.todo([self.modal_job()])
        saved = sys.modules.get("modal")
        sys.modules["modal"] = self.fake_modal_module(deployed=None)
        try:
            with patch.object(sys, "argv", ["runner.py", "doctor"]), \
                 patch("torch.cuda.is_available", return_value=True):
                with self.assertRaisesRegex(ValueError, "caption"):
                    runner.main()
        finally:
            if saved is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = saved


class SyncCancelTests(unittest.TestCase):
    """`sync` does nothing on an empty queue; `cancel --job` finds the remote run to cancel."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir(parents=True)
        self._old = {key: runner._PATH_OVERRIDES.get(key) for key in ("runs", "config", "todo")}
        runner._PATH_OVERRIDES["runs"] = self.runs
        # modal_backend imports the sys.path runner, which owns its own override table: set it too, or the
        # default runs root points at the real automation/runs (and its project seed folders).
        import runner as sys_runner
        self.sys_runner = sys_runner
        self._old_sys = sys_runner._PATH_OVERRIDES.get("runs")
        sys_runner._PATH_OVERRIDES["runs"] = self.runs

    def tearDown(self):
        for key, value in self._old.items():
            if value is None:
                runner._PATH_OVERRIDES.pop(key, None)
            else:
                runner._PATH_OVERRIDES[key] = value
        if self._old_sys is None:
            self.sys_runner._PATH_OVERRIDES.pop("runs", None)
        else:
            self.sys_runner._PATH_OVERRIDES["runs"] = self._old_sys
        self.temp.cleanup()

    def test_sync_on_an_empty_queue_is_a_clean_exit(self):
        with patch.object(sys, "argv", ["runner.py", "sync"]):
            runner.main()   # no exception

    def test_cancel_finds_the_matching_run_and_reports_ids(self):
        run = self.runs / "job-abc123"
        run.mkdir()
        (run / "state.json").write_text(json.dumps(
            {"remote": {"call_id": "fc-1", "run_key": "job-abc123", "status": "submitted",
                        "gpu": "L40S", "submitted_at": "2026-11-10T21:00:00",
                        "dataset_fingerprint": "f" * 64}}), encoding="utf-8")
        from cloud import modal_backend
        calls = []
        with patch.object(modal_backend, "cancel", side_effect=lambda r, **k: calls.append(str(r))), \
             patch.object(sys, "argv", ["runner.py", "cancel", "--job", "job"]):
            runner.main()
        self.assertEqual(calls, [str(run)])

    def test_cancel_with_several_matches_asks_for_a_narrower_scope(self):
        for suffix in ("aaa111", "bbb222"):
            run = self.runs / f"job-{suffix}"
            run.mkdir()
            (run / "state.json").write_text(json.dumps({"remote": {"call_id": "fc-1", "status": "submitted"}}),
                                            encoding="utf-8")
        with patch.object(sys, "argv", ["runner.py", "cancel", "--job", "job"]):
            with self.assertRaisesRegex(ValueError, "several runs"):
                runner.main()

    def test_cancel_with_nothing_found_says_so_and_does_not_raise(self):
        with patch.object(sys, "argv", ["runner.py", "cancel", "--job", "nosuchjob"]):
            runner.main()   # prints "no remote run found"; not an error
