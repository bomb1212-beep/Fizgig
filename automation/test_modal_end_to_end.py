"""End-to-end: a Modal run driven against the file-backed stub SDK, PC death included.

Nothing here touches the network, a GPU or real Modal. The stub (automation/testdata/modal_stub) implements exactly the
surface `cloud.client.ModalClient` and the runner use, and shares one JSON world between processes, so the runner, a
later `runner.py sync`, and this test all see the same "remote" state.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

AUTO = Path(__file__).resolve().parent
ROOT = AUTO.parent
STUB = AUTO / "testdata" / "modal_stub"


def _local_version():
    spec_path = AUTO / "cloud" / "spec.py"
    spec_mod = importlib.util.spec_from_file_location("fizgig_spec_e2e", spec_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    return mod.local_deployment_version()


class ModalEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.world = self.root / "fake-modal-world.json"
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        from PIL import Image
        for i, color in enumerate(("red", "blue")):
            Image.new("RGB", (16, 16), color).save(self.dataset / f"img{i}.png")
            (self.dataset / f"img{i}.txt").write_text(f"trg, photo {i}", encoding="utf-8")
        self.config = self.root / "config.json"
        import shutil
        for name in ("caption_instructions.txt", "caption_instructions_anime.txt"):
            source = AUTO / name
            if source.is_file():
                shutil.copy2(source, self.root / name)
        self.config.write_text(json.dumps({
            "target_steps": 2, "max_images_per_job": 10, "rank": 8, "save_every_steps_approx": 1,
            "learning_rate": 0.0002, "adaptive_lr": True, "adaptive_lr_min": 0.0002, "adaptive_lr_max": 0.0004,
            "megapixels": 0.25, "caption_base_url": "http://127.0.0.1:18080/v1",
            "caption_model": "gemma", "caption_max_output_tokens": 300,
            "sample_prompts": ["{trigger}, portrait"],
            "cloud": {"modal": {"poll_seconds": 0.1, "overhead_minutes": 0,
                                "sec_per_step": {"L40S": 0.01}, "gpu_rate": {"L40S": 1.95},
                                "cpu_rate": 0.047, "memory_rate": 0.008}}}),
            encoding="utf-8")
        self.todo = self.root / "todo.json"
        self.todo.write_text(json.dumps(
            [{"id": "e2ejob", "enabled": True, "trigger": "trg", "dataset_root": str(self.dataset),
              "family": "krea2", "backend": "modal", "modal_gpu": "L40S"}]),
            encoding="utf-8")
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = os.pathsep.join([str(STUB), str(ROOT / "src"), str(AUTO)])
        self.env["FIZGIG_FAKE_MODAL_WORLD"] = str(self.world)
        self.env["FIZGIG_FAKE_MODAL_VERSION"] = _local_version()
        self.env["FIZGIG_FAKE_MODAL_SCRATCH"] = str(self.root / "scratch")
        self.env["PYTHONUNBUFFERED"] = "1"
        # The runner prints this repo's (Korean) paths; both ends agree on UTF-8 whatever the console code page is.
        self.env["PYTHONIOENCODING"] = "utf-8"
        self.runner = str(AUTO / "runner.py")
        # Never the user's automation/queue.lock: a killed runner (the PC going off) leaves its lock behind.
        self.own_lock = ("--queue-lock", str(self.root / "queue.lock"))

    def run_runner(self, command, *extra, env=None, timeout=240, check=True):
        return subprocess.run([sys.executable, "-B", self.runner, command, *extra, *self.own_lock],
                              env=env or self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, check=False)

    def spawn_runner(self, command, *extra, env=None):
        return subprocess.Popen([sys.executable, "-B", self.runner, command, *extra, *self.own_lock],
                                env=env or self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def world_now(self):
        deadline = time.monotonic() + 10
        while True:
            try:
                return json.loads(self.world.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {"calls": {}, "volume": {}}
            except (PermissionError, ValueError):   # the runner is swapping the file in (Windows)
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)

    def wait_for_call(self, proc=None, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            world = self.world_now()
            if world.get("calls"):
                return next(iter(world["calls"]))
            if proc is not None and proc.poll() is not None:
                out, _ = proc.communicate()
                raise AssertionError(f"the runner exited ({proc.returncode}) without submitting:\n{out[-3000:]}")
            time.sleep(0.1)
        out = ""
        if proc is not None:
            proc.kill()
            out, _ = proc.communicate()
        raise AssertionError(f"the runner never submitted to the fake Modal world:\n{out[-3000:]}")

    def test_submit_to_fetch_in_one_run(self):
        proc = self.spawn_runner("run", "--job", "e2ejob", "--config", str(self.config),
                                 "--todo", str(self.todo), "--runs-dir", str(self.runs), "--events")
        out, _ = proc.communicate(timeout=240)
        self.assertEqual(proc.returncode, 0, out)
        # The GUI-visible events carry the backend, which is what pause/close wiring relies on.
        self.assertIn('"backend": "modal"', out)
        run_dirs = sorted(self.runs.glob("e2ejob-*"))
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue((run_dirs[0] / "output" / "e2ejob.safetensors").exists())
        state = json.loads((run_dirs[0] / "state.json").read_text(encoding="utf-8"))
        self.assertTrue(state["complete"])
        self.assertEqual(state["remote"]["status"], "fetched")
        # The remote folder was cleaned up after a verified fetch.
        key = state["remote"]["run_key"]
        self.assertNotIn(f"/runs/{key}/plan.json", self.world_now().get("volume", {}))

    def test_pc_off_then_sync_fetches_the_result(self):
        env = dict(self.env, FIZGIG_FAKE_MODAL_POLLS_NEEDED="1000")   # stays "running" until we say otherwise
        proc = self.spawn_runner("run", "--job", "e2ejob", "--config", str(self.config),
                                 "--todo", str(self.todo), "--runs-dir", str(self.runs), "--events",
                                 env=env)
        call_id = self.wait_for_call(proc)
        proc.kill()                       # the PC is off
        proc.wait()
        # The remote world still holds the unfinished run.
        self.assertEqual(self.world_now()["calls"][call_id]["state"], "running")
        sync_env = dict(self.env, FIZGIG_FAKE_MODAL_POLLS_NEEDED="1")  # the next poll flips it done
        result = self.run_runner("sync", "--runs-dir", str(self.runs), env=sync_env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout)
        run_dirs = sorted(self.runs.glob("e2ejob-*"))
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue((run_dirs[0] / "output" / "e2ejob.safetensors").exists())
        state = json.loads((run_dirs[0] / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["remote"]["status"], "fetched")

    def real_queue_lock_state(self):
        lock = AUTO / "queue.lock"
        try:
            return lock.read_bytes(), lock.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def test_a_killed_runner_leaves_the_real_queue_lock_alone(self):
        # Killing the runner (the PC going off) cannot clean up its queue lock. The test's runner must hold its own
        # lock, never automation/queue.lock: a stale file there with a reused pid blocks the user's real queue and
        # the GUIs read it as "a runner is active".
        before = self.real_queue_lock_state()
        env = dict(self.env, FIZGIG_FAKE_MODAL_POLLS_NEEDED="1000")
        proc = self.spawn_runner("run", "--job", "e2ejob", "--config", str(self.config),
                                 "--todo", str(self.todo), "--runs-dir", str(self.runs), "--events",
                                 env=env)
        self.wait_for_call(proc)
        proc.kill()
        proc.wait()
        self.assertEqual(self.real_queue_lock_state(), before)

    def test_explicit_cancel_reaches_the_remote_call(self):
        cancel_path = self.root / "cancel_remote.request"
        env = dict(self.env, FIZGIG_REMOTE_CANCEL_REQUEST=str(cancel_path),
                   FIZGIG_FAKE_MODAL_POLLS_NEEDED="1000")
        proc = self.spawn_runner("run", "--job", "e2ejob", "--config", str(self.config),
                                 "--todo", str(self.todo), "--runs-dir", str(self.runs),
                                 "--events", env=env)
        call_id = self.wait_for_call(proc)
        cancel_path.write_text("cancel\n", encoding="utf-8")   # what the GUI's cancel button writes
        out, _ = proc.communicate(timeout=120)
        self.assertEqual(proc.returncode, 0, out)
        self.assertEqual(self.world_now()["calls"][call_id]["state"], "cancelled")
        self.assertIn('"kind": "run"', out)
        self.assertIn('"backend": "modal"', out)


class StubWorldFileTests(unittest.TestCase):
    """The stub's world file is shared by the runner (writer) and the test (reader) at the same time.

    On Windows `os.replace` onto a file another process has open fails with PermissionError, and a read during
    the swap can fail the same way; neither may crash the runner or be mistaken for an empty world."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.world = Path(self.temp.name) / "world.json"
        old = os.environ.get("FIZGIG_FAKE_MODAL_WORLD")
        os.environ["FIZGIG_FAKE_MODAL_WORLD"] = str(self.world)
        try:
            spec_mod = importlib.util.spec_from_file_location(
                "fizgig_modal_stub_world_test", STUB / "modal" / "__init__.py")
            self.stub = importlib.util.module_from_spec(spec_mod)
            spec_mod.loader.exec_module(self.stub)
        finally:
            if old is None:
                os.environ.pop("FIZGIG_FAKE_MODAL_WORLD", None)
            else:
                os.environ["FIZGIG_FAKE_MODAL_WORLD"] = old

    def hold_open_for(self, seconds):
        import threading
        handle = open(self.world, encoding="utf-8")
        timer = threading.Timer(seconds, handle.close)
        timer.start()
        self.addCleanup(timer.cancel)
        self.addCleanup(handle.close)

    def test_store_waits_out_a_reader_instead_of_crashing(self):
        self.stub._store({"version": "v", "volume": {}, "calls": {"fc-1": {}}, "next_call": 2})
        self.hold_open_for(0.5)
        self.stub._store({"version": "v", "volume": {}, "calls": {"fc-1": {}, "fc-2": {}}, "next_call": 3})
        self.assertEqual(sorted(json.loads(self.world.read_text(encoding="utf-8"))["calls"]), ["fc-1", "fc-2"])

    def test_load_of_a_busy_file_is_not_an_empty_world(self):
        self.stub._store({"version": "v", "volume": {"k/plan.json": "{}"}, "calls": {"fc-1": {}}, "next_call": 2})
        real_read = Path.read_text
        failures = [PermissionError(13, "busy")] * 3

        def flaky_read(path, *args, **kwargs):
            if path == self.world and failures:
                raise failures.pop()
            return real_read(path, *args, **kwargs)

        from unittest.mock import patch
        with patch.object(Path, "read_text", flaky_read):
            world = self.stub._load()
        self.assertEqual(world["calls"], {"fc-1": {}})
        self.assertIn("k/plan.json", world["volume"])

    def test_a_missing_file_is_a_fresh_world(self):
        world = self.stub._load()
        self.assertEqual((world["calls"], world["volume"], world["next_call"]), ({}, {}, 1))


if __name__ == "__main__":
    unittest.main()
