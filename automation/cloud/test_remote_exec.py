"""Offline tests for the remote executor: fake stage commands only, never a real GPU or Modal."""
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "fizgig_remote_exec", Path(__file__).with_name("remote_exec.py"))
remote_exec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote_exec)


def stage_script(path, body):
    path.write_text(body, encoding="utf-8")
    return str(path)


def build_tarball(src_root, tarball):
    with tarfile.open(tarball, "w:gz") as tar:
        for file in sorted(Path(src_root).rglob("*")):
            if file.is_file():
                arcname = str(file.relative_to(src_root)).replace(os.sep, "/")
                if arcname.split("/")[0] != "src":
                    raise AssertionError("fixture tree must be rooted at src/")
                tar.add(str(file), arcname=arcname)
    return hashlib.sha256(Path(tarball).read_bytes()).hexdigest()


def training_outputs_script(run_folder, job_id, epochs):
    state = f"{job_id}-{epochs:06d}-state"
    return f"""
import json, os
from safetensors.torch import save_file
import torch
output = os.path.join({run_folder!r}, 'output')
os.makedirs(output, exist_ok=True)
save_file({{"lora_unet_blocks_0": torch.zeros(1)}}, os.path.join(output, {job_id!r} + '.safetensors'),
           metadata={{"ss_epoch": {str(epochs)!r}}})
state_dir = os.path.join(output, {state!r})
os.makedirs(state_dir, exist_ok=True)
with open(os.path.join(state_dir, 'training_state.json'), 'w', encoding='utf-8') as fh:
    json.dump({{"epoch": {epochs}}}, fh)
for name in ('lora.safetensors', 'optimizer.pt', 'rng.pt'):
    with open(os.path.join(state_dir, name), 'wb') as fh:
        fh.write(b'x')
print('training outputs written')
"""


def crash_resume_script(run_folder, job_id, epochs, marker):
    state = f"{job_id}-{epochs:06d}-state"
    return f"""
import json, os, sys
from safetensors.torch import save_file
import torch
marker = {marker!r}
crashed_flag = marker + '.crashed'
output = os.path.join({run_folder!r}, 'output')
os.makedirs(output, exist_ok=True)
save_file({{"lora_unet_blocks_0": torch.zeros(1)}}, os.path.join(output, {job_id!r} + '.safetensors'),
           metadata={{"ss_epoch": {str(epochs)!r}}})
state_dir = os.path.join(output, {state!r})
os.makedirs(state_dir, exist_ok=True)
with open(os.path.join(state_dir, 'training_state.json'), 'w', encoding='utf-8') as fh:
    json.dump({{"epoch": {epochs}}}, fh)
for name in ('lora.safetensors', 'optimizer.pt', 'rng.pt'):
    with open(os.path.join(state_dir, name), 'wb') as fh:
        fh.write(b'x')
with open(marker, 'w', encoding='utf-8') as fh:
    fh.write(' '.join(sys.argv))
print('training outputs written')
if not os.path.exists(crashed_flag):
    with open(crashed_flag, 'w') as fh:
        fh.write('1')
    print('simulating a crash after the checkpoint was saved')
    sys.exit(1)
print('resumed and finished')
"""


class RemoteExecBase(unittest.TestCase):
    """The fixture and helpers; no tests of its own."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # The submitted source tree: a marker module that only exists in the tarball.
        self.repo = self.root / "repo"
        self.src = self.repo / "src"
        self.src.mkdir(parents=True)
        (self.src / "marker.py").write_text("MARKER = 'submitted-src'\n", encoding="utf-8")
        self.run_root = self.root / "runs" / "job-testkey123"
        (self.run_root / "code").mkdir(parents=True)
        self.run_folder = self.run_root / "run"
        self.run_folder.mkdir()
        (self.run_folder / "images").mkdir()
        (self.run_folder / "images" / "a.png").write_bytes(b"png")
        (self.run_folder / "images" / "a.txt").write_text("a caption", encoding="utf-8")
        self.tarball = self.run_root / "code" / "fizgig_src.tar.gz"
        self.sha = build_tarball(self.repo, self.tarball)
        (self.run_root / "code" / "sha256.txt").write_text(self.sha + "  fizgig_src.tar.gz\n", encoding="utf-8")
        self.code_dir = self.root / "tmp_code"
        self.commits = []
        self.log_lines = []

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, *, attempt_id="attempt-1", stages=None, epochs=3, images=1,
             fingerprint="fp" + "1" * 12, code_sha256=None, files=None, model_files=None,
             arch_id="krea2drv"):
        if stages is None:
            stages = self.default_stages()
        return {"attempt_id": attempt_id, "job_id": "job", "family": "krea2",
                "arch_id": arch_id, "images": images, "epochs": epochs,
                "dataset_fingerprint": fingerprint,
                "code_sha256": code_sha256 if code_sha256 is not None else self.sha,
                "stages": [{"name": name, "cmd": list(cmd)} for name, cmd in stages],
                "files": files if files is not None else [[str(self.run_folder / "dataset.toml"), "[general]\n"]],
                "model_files": model_files or []}

    def default_stages(self):
        ok = stage_script(self.root / "ok.py",
                          "import sys\nprint('ok', sys.argv[1:])\n")
        training = stage_script(self.root / "train.py",
                                training_outputs_script(str(self.run_folder), "job", 3))
        return [("Cache Preparation", [sys.executable, ok, "cache"]),
                ("Text Encoder Caching", [sys.executable, ok, "te"]),
                ("Training", [sys.executable, training])]

    def write_plan(self, **kwargs):
        plan = self.plan(**kwargs)
        (self.run_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        return plan

    def attempt(self, **kwargs):
        opts = dict(code_dir=str(self.code_dir), commit=lambda: self.commits.append(1),
                    log=self.log_lines.append, verify_steps=lambda: None)
        opts.update(kwargs)
        return remote_exec.run(str(self.run_root), **opts)

    def read_status(self):
        return json.loads((self.run_root / "status.json").read_text(encoding="utf-8"))


class RemoteExecTests(RemoteExecBase):
    def test_stages_run_in_order_and_finish_done(self):
        self.write_plan()
        status = self.attempt()
        self.assertEqual(status["state"], "done")
        self.assertEqual(status["attempt_id"], "attempt-1")
        self.assertEqual(status["exit_code"], 0)
        log_cache = (self.run_folder / "Cache_Preparation.log").read_text(encoding="utf-8")
        log_te = (self.run_folder / "Text_Encoder_Caching.log").read_text(encoding="utf-8")
        log_train = (self.run_folder / "Training.log").read_text(encoding="utf-8")
        self.assertIn("cache", log_cache)
        self.assertIn("te", log_te)
        self.assertIn("training outputs written", log_train)
        state = json.loads((self.run_folder / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["completed_stages"], ["Cache Preparation", "Text Encoder Caching", "Training"])
        self.assertTrue(state["complete"])
        self.assertGreaterEqual(len(self.commits), 4)   # one per stage end + the final commit

    def test_submit_source_is_the_one_on_the_pythonpath(self):
        probe = stage_script(self.root / "probe.py",
                             "import os, marker\nprint(os.environ['PYTHONPATH'])\nprint(marker.MARKER)\n")
        stages = [("Cache Preparation", [sys.executable, probe]),
                  ("Text Encoder Caching", [sys.executable, probe]),
                  ("Training", [sys.executable, stage_script(self.root / "train.py",
                                                             training_outputs_script(str(self.run_folder), "job", 3))])]
        self.write_plan(stages=stages)
        self.attempt()
        log = (self.run_folder / "Cache_Preparation.log").read_text(encoding="utf-8")
        self.assertIn(str(self.code_dir) + os.sep + "src" if os.name == "nt" else str(self.code_dir) + "/src",
                      log)
        self.assertIn("submitted-src", log)

    def test_completed_cache_stages_are_skipped_and_resume_uses_a_complete_checkpoint(self):
        # One crash script for both attempts: identical stages, so the plan fingerprint matches and the cache
        # stages are skipped the second time. The first call saves a checkpoint then crashes; the second finishes.
        marker = self.root / "training_argv.txt"
        crash = stage_script(self.root / "crash_after_checkpoint.py",
                             crash_resume_script(str(self.run_folder), "job", 3, str(marker)))
        cache = stage_script(self.root / "cache.py", "print('cache ran')\n")
        te = stage_script(self.root / "te.py", "print('te ran')\n")
        stages = [("Cache Preparation", [sys.executable, cache]),
                  ("Text Encoder Caching", [sys.executable, te]),
                  ("Training", [sys.executable, crash])]
        self.write_plan(attempt_id="attempt-1", stages=stages)
        with self.assertRaises(remote_exec.RemoteError):
            self.attempt()
        status = self.read_status()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["stage"], "Training")
        self.assertEqual(status["exit_code"], 1)
        self.assertIn("training outputs written", status["log_tail"])
        self.assertNotIn("--resume", marker.read_text(encoding="utf-8"))
        # Second attempt: the same submission, same files. Cache stages must not run; Training must resume.
        self.write_plan(attempt_id="attempt-1", stages=stages)
        self.attempt()
        argv = marker.read_text(encoding="utf-8")
        self.assertIn("--resume", argv)
        cache_log = (self.run_folder / "Cache_Preparation.log").read_text(encoding="utf-8")
        self.assertEqual(cache_log.count("cache ran"), 1)
        status = self.read_status()
        self.assertEqual(status["state"], "done")
        state = json.loads((self.run_folder / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["completed_stages"], ["Cache Preparation", "Text Encoder Caching", "Training"])

    def test_a_checkpoint_with_a_missing_file_is_never_resumed(self):
        output = self.run_folder / "output"
        saved = output / "job-000003-state"
        saved.mkdir(parents=True)
        (saved / "training_state.json").write_text(json.dumps({"epoch": 3}), encoding="utf-8")
        (saved / "lora.safetensors").write_bytes(b"x")
        (saved / "optimizer.pt").write_bytes(b"x")
        # rng.pt missing on purpose
        marker = self.root / "resume_argv.txt"
        probe = stage_script(self.root / "probe.py",
                             "import sys\nopen(sys.argv[-1], 'w', encoding='utf-8').write(' '.join(sys.argv))\n"
                             + training_outputs_script(str(self.run_folder), "job", 3))
        stages = [("Cache Preparation", [sys.executable, str(self.root / "cache.py")]),
                  ("Text Encoder Caching", [sys.executable, str(self.root / "te.py")]),
                  ("Training", [sys.executable, probe, str(marker)])]
        state = {"completed_stages": ["Cache Preparation", "Text Encoder Caching"],
                 "dataset_fingerprint": "fp" + "1" * 12, "code_sha256": self.sha,
                 "plan_fingerprint": None, "resume_after_ns": 0}
        plan = self.plan(stages=stages)
        plan_fp = remote_exec.plan_fingerprint(plan)
        state["plan_fingerprint"] = plan_fp
        (self.run_folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
        self.write_plan(stages=stages)
        self.attempt()
        self.assertNotIn("--resume", marker.read_text(encoding="utf-8"))

    def test_failure_status_carries_stage_code_and_log_tail(self):
        noisy = stage_script(self.root / "noisy.py",
                             "".join(f"print('line {i}')\n" for i in range(1, 101)) + "import sys\nsys.exit(3)\n")
        stages = [("Cache Preparation", [sys.executable, noisy]),
                  ("Text Encoder Caching", [sys.executable, noisy]),
                  ("Training", [sys.executable, noisy])]
        self.write_plan(stages=stages)
        with self.assertRaises(remote_exec.RemoteError):
            self.attempt()
        status = self.read_status()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["stage"], "Cache Preparation")
        self.assertEqual(status["exit_code"], 3)
        self.assertIn("line 100", status["log_tail"])
        self.assertNotIn("line 1\n", status["log_tail"])
        self.assertLessEqual(len(status["log_tail"].splitlines()), 40)

    def test_a_done_attempt_reexecuting_uses_the_committed_result(self):
        self.write_plan()
        self.attempt()
        committed = self.read_status()
        # The platform re-runs the same call with the same inputs; the stages would fail this time.
        sabotage = stage_script(self.root / "sabotage.py", "import sys\nsys.exit(9)\n")
        stages = [("Cache Preparation", [sys.executable, sabotage]),
                  ("Text Encoder Caching", [sys.executable, sabotage]),
                  ("Training", [sys.executable, sabotage])]
        self.write_plan(attempt_id="attempt-1", stages=stages)
        status = self.attempt()
        self.assertEqual(status["state"], "done")
        self.assertEqual(status, committed)
        self.assertNotIn("sabotage ran", (self.run_folder / "Training.log").read_text(encoding="utf-8") or "")

    def test_bad_tarball_members_are_rejected(self):
        for bad_name in ("evil.txt", "/abs/path.py", "src/../evil.py", "src/./evil.py"):
            with self.subTest(member=bad_name):
                sub = tempfile.TemporaryDirectory()
                self.addCleanup(sub.cleanup)
                repo = Path(sub.name) / "repo"
                staging = self.root / f"bad_{abs(hash(bad_name))}"
                staging.mkdir(exist_ok=True)
                tarball = staging / "fizgig_src.tar.gz"
                with tarfile.open(tarball, "w:gz") as tar:
                    payload = Path(sub.name) / "payload.txt"
                    payload.write_text("x", encoding="utf-8")
                    tar.add(str(payload), arcname=bad_name)
                (self.run_root / "code" / "fizgig_src.tar.gz").unlink()
                (self.run_root / "code" / "fizgig_src.tar.gz").write_bytes(tarball.read_bytes())
                sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
                (self.run_root / "code" / "sha256.txt").write_text(sha, encoding="utf-8")
                self.write_plan(code_sha256=sha)
                with self.assertRaises(remote_exec.RemoteError):
                    self.attempt()
                self.assertEqual(self.read_status()["state"], "failed")

    def test_a_symlink_member_is_rejected(self):
        staging = self.root / "sym"
        staging.mkdir()
        payload = self.root / "target.txt"
        payload.write_text("x", encoding="utf-8")
        tarball = staging / "fizgig_src.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            info = tarfile.TarInfo("src/link.txt")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        (self.run_root / "code" / "fizgig_src.tar.gz").write_bytes(tarball.read_bytes())
        sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
        (self.run_root / "code" / "sha256.txt").write_text(sha, encoding="utf-8")
        self.write_plan(code_sha256=sha)
        with self.assertRaises(remote_exec.RemoteError):
            self.attempt()

    def test_tampered_tarball_is_rejected(self):
        self.write_plan()
        with (self.run_root / "code" / "fizgig_src.tar.gz").open("ab") as fh:
            fh.write(b"tampered")
        with self.assertRaises(remote_exec.RemoteError):
            self.attempt()
        self.assertEqual(self.read_status()["state"], "failed")

    def test_plan_files_must_stay_inside_the_run_folder(self):
        self.write_plan(files=[[str(self.root / "elsewhere.txt"), "nope\n"]])
        with self.assertRaises(remote_exec.RemoteError):
            self.attempt()

    def test_model_files_are_size_checked_before_any_stage_runs(self):
        models = [{"path": str(self.root / "model.safetensors"), "bytes": 10}]
        Path(self.root / "model.safetensors").write_bytes(b"short")
        cache_marker = self.root / "cache_marker.txt"
        cache = stage_script(self.root / "cache.py", f"open({str(cache_marker)!r}, 'w').write('ran')\n")
        stages = [("Cache Preparation", [sys.executable, cache]),
                  ("Text Encoder Caching", [sys.executable, cache]),
                  ("Training", [sys.executable, cache])]
        self.write_plan(stages=stages, model_files=models)
        with self.assertRaises(remote_exec.RemoteError):
            self.attempt()
        self.assertFalse(cache_marker.exists())
        self.assertEqual(self.read_status()["state"], "failed")

    def test_commits_happen_during_long_stages_and_at_the_end(self):
        class FakeProcess:
            def __init__(self):
                self.polls = iter([None, None, None, 0])
                self.returncode = 0

            def poll(self):
                return next(self.polls)

        class FakeMonotonic:
            def __init__(self):
                self.now = 0

            def __call__(self):
                self.now += 100
                return self.now

        popen = lambda *args, **kwargs: FakeProcess()  # noqa: E731
        monotonic = FakeMonotonic()
        self.write_plan()
        self.attempt(popen=popen, monotonic=monotonic, sleep=lambda _s: None, commit_interval=60,
                     outputs_valid=lambda *a: True)
        self.assertGreaterEqual(len(self.commits), 3 + 1 + 3 * 2)   # stage ends + final + in-stage intervals

    def test_training_outputs_are_the_done_gate(self):
        silent = stage_script(self.root / "silent.py", "print('nothing written')\n")
        stages = [("Cache Preparation", [sys.executable, silent]),
                  ("Text Encoder Caching", [sys.executable, silent]),
                  ("Training", [sys.executable, silent])]
        self.write_plan(stages=stages)
        with self.assertRaises(remote_exec.RemoteError):
            self.attempt()
        status = self.read_status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("final LoRA", status["message"])


# A stand-in for the real fizgig.dataset.config: dataset.toml holds `fake_steps = N` and the "dataset group" is N long.
FAKE_FIZGIG_CONFIG = """
class ConfigSanitizer:
    pass


class _Blueprint:
    def __init__(self, dataset_group):
        self.dataset_group = dataset_group


class BlueprintGenerator:
    def __init__(self, sanitizer):
        pass

    def generate(self, user_config, args, architecture=None):
        return _Blueprint(user_config)


def load_user_config(path):
    with open(path, encoding="utf-8") as fh:
        return int(fh.read().split("=")[1])


def generate_dataset_group_by_blueprint(dataset_group, training=True, shared_epoch=None):
    return [None] * dataset_group
"""


def write_fake_fizgig(src_root):
    package = Path(src_root) / "fizgig" / "dataset"
    package.mkdir(parents=True, exist_ok=True)
    (Path(src_root) / "fizgig" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "config.py").write_text(FAKE_FIZGIG_CONFIG, encoding="utf-8")


class SubmittedSourceStepCheckTests(RemoteExecBase):
    """The step-count check before Training runs inside the Modal function's own process, where only remote_exec
    and spec are importable - fizgig exists only in the submitted tarball. Proven in a fresh, isolated interpreter
    because this test process may already have the real fizgig imported."""

    CLOUD = str(Path(__file__).resolve().parent)

    def isolated(self, code):
        driver = self.root / "driver.py"
        driver.write_text("import sys\nsys.path.insert(0, %r)\nimport remote_exec\n" % self.CLOUD + code,
                          encoding="utf-8")
        # -I: no PYTHONPATH, no script folder on sys.path - the container's function process sees no fizgig either.
        return subprocess.run([sys.executable, "-I", str(driver)], cwd=str(self.root), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=300)

    def test_verify_cache_steps_imports_fizgig_from_the_given_source(self):
        fake_src = self.root / "fake_src"
        write_fake_fizgig(fake_src)
        (self.run_folder / "dataset.toml").write_text("fake_steps = 2\n", encoding="utf-8")
        check = "remote_exec.verify_cache_steps('krea2drv', %d, %r, src_root=%r)\nprint('CHECK OK')\n"
        matching = self.isolated(check % (2, str(self.run_folder), str(fake_src)))
        self.assertEqual(matching.returncode, 0, matching.stdout + matching.stderr)
        self.assertIn("CHECK OK", matching.stdout)
        mismatched = self.isolated(check % (3, str(self.run_folder), str(fake_src)))
        self.assertNotEqual(mismatched.returncode, 0)
        self.assertIn("Cache contains 2 steps/epoch, expected 3", mismatched.stderr)

    def run_with_default_check(self, images):
        write_fake_fizgig(self.src)
        self.sha = build_tarball(self.repo, self.tarball)
        (self.run_root / "code" / "sha256.txt").write_text(self.sha + "  fizgig_src.tar.gz\n", encoding="utf-8")
        self.write_plan(images=images, files=[[str(self.run_folder / "dataset.toml"), "fake_steps = 1\n"]])
        return self.isolated(
            "try:\n"
            "    status = remote_exec.run(%r, code_dir=%r, commit=lambda: None, log=lambda *a: None)\n"
            "except Exception as exc:\n"
            "    print('RAISED', type(exc).__name__)\n"
            "import json\n"
            "with open(%r, encoding='utf-8') as fh:\n"
            "    status = json.load(fh)\n"
            "print('STATE=' + status['state'])\n"
            "print('MESSAGE=' + str(status.get('message')))\n"
            % (str(self.run_root), str(self.code_dir), str(self.run_root / "status.json")))

    def test_run_checks_steps_with_the_submitted_source_and_finishes(self):
        result = self.run_with_default_check(images=1)
        self.assertIn("STATE=done", result.stdout, result.stdout + result.stderr)
        self.assertIn("training outputs written",
                      (self.run_folder / "Training.log").read_text(encoding="utf-8"))

    def test_run_step_mismatch_fails_before_training(self):
        result = self.run_with_default_check(images=2)
        self.assertIn("STATE=failed", result.stdout, result.stdout + result.stderr)
        self.assertIn("Cache contains 1 steps/epoch, expected 2", result.stdout)
        self.assertFalse((self.run_folder / "Training.log").exists())


if __name__ == "__main__":
    unittest.main()
