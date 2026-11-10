"""Offline tests for the modal backend: a fake client stands in for Modal; nothing touches the network."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "lora_runner_modal", Path(__file__).resolve().parents[1] / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # automation/

from cloud import modal_backend, remote_exec, spec  # noqa: E402


RUNS = "/runs"


def remote_path(run_key, *parts):
    return f"{RUNS}/{run_key}/" + "/".join(parts)


class FakeClient:
    """The ModalClient surface, scripted. Nothing reaches out."""

    def __init__(self, *, deployed="match"):
        self.local_version = spec.local_deployment_version()
        self.deployed = self.local_version if deployed == "match" else deployed
        self.deploys = 0
        self.models_ready = 0
        self.spawned = []          # (gpu, run_key)
        self.uploads = []          # (local_dir, remote_dir)
        self.uploaded_files = []   # remote paths, one per uploaded file
        self.files = {}            # remote path -> bytes
        self.poll_script = {}      # call_id -> list of poll results (consumed one per call, last one sticks)
        self.next_call = 0
        self.deleted = []
        self.downloads = []
        self.cancelled = []
        self.cancel_accepted = True
        self.delete_fails = False

    # deployment
    def deployed_version(self):
        return self.deployed

    def deploy(self):
        self.deploys += 1
        self.deployed = self.local_version
        return True

    def ensure_models(self):
        self.models_ready += 1
        return {"cached": True}

    # submission
    def upload(self, local_dir, remote_dir):
        self.uploads.append((str(local_dir), remote_dir))
        count = 0
        prefix = remote_dir.rstrip("/")
        for root, _dirs, files in os.walk(local_dir):
            for name in files:
                rel = Path(root, name).relative_to(local_dir).as_posix()
                self.files[f"{prefix}/{rel}"] = Path(root, name).read_bytes()
                self.uploaded_files.append(f"{prefix}/{rel}")
                count += 1
        return count

    def spawn(self, gpu, run_key):
        self.next_call += 1
        call_id = f"fc-{self.next_call}"
        self.spawned.append((gpu, run_key, call_id))
        self.poll_script.setdefault(call_id, ["done"])
        return call_id

    # tracking
    def poll(self, call_id):
        script = self.poll_script.setdefault(call_id, ["done"])
        if len(script) > 1:
            return script.pop(0)
        return script[0]

    def read_bytes(self, path):
        return self.files.get(path)

    def read_text(self, path):
        data = self.files.get(path)
        return data.decode("utf-8") if data is not None else None

    def download(self, remote_dir, local_dir, exclude=None):
        prefix = remote_dir.rstrip("/") + "/"
        count = 0
        for path, data in self.files.items():
            if not path.startswith(prefix):
                continue
            target = Path(local_dir) / path[len(prefix):]
            if exclude and exclude(str(target)):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            count += 1
        self.downloads.append((remote_dir, local_dir, count))
        return count

    def delete(self, remote_dir):
        if self.delete_fails:
            raise RuntimeError("delete blocked")
        prefix = remote_dir.rstrip("/") + "/"
        gone = [p for p in list(self.files) if p == remote_dir or p.startswith(prefix)]
        for p in gone:
            del self.files[p]
        self.deleted.append(remote_dir)
        return True

    def cancel(self, call_id):
        self.cancelled.append(call_id)
        if not self.cancel_accepted:
            return False
        return True


def write_training_outputs(directory, job_id, epochs):
    """A valid local output folder (safetensors + last state), as Fizgig/the remote job would write."""
    from safetensors.torch import save_file
    import torch
    out = Path(directory) / "output"
    out.mkdir(parents=True, exist_ok=True)
    save_file({"lora_unet_blocks_0": torch.zeros(1)}, str(out / f"{job_id}.safetensors"),
              metadata={"ss_epoch": str(epochs)})
    state = out / f"{job_id}-{epochs:06d}-state"
    state.mkdir(exist_ok=True)
    (state / "training_state.json").write_text(json.dumps({"epoch": epochs}), encoding="utf-8")
    for name in ("lora.safetensors", "optimizer.pt", "rng.pt"):
        (state / name).write_bytes(b"x")


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self._old_runs = runner._PATH_OVERRIDES.get("runs")
        runner._PATH_OVERRIDES["runs"] = self.runs
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        from PIL import Image
        for name, color in (("a.png", "red"), ("b.png", "blue")):
            Image.new("RGB", (16, 16), color).save(self.dataset / name)
            (self.dataset / (name[:-4] + ".txt")).write_text("trg, a photo", encoding="utf-8")
        self.cfg = runner.settings({"id": "job", "trigger": "trg", "dataset_root": str(self.dataset),
                                    "family": "krea2", "backend": "modal", "modal_gpu": "L40S",
                                    "target_steps": 6, "sample_prompts": ["{trigger}, portrait"]})
        self.run, _images, _instructions = runner.prepare(self.cfg)
        self.n = 2
        self.epochs = runner.epochs_for(self.cfg["target_steps"], self.n)   # 3
        self.client = FakeClient()

    def tearDown(self):
        if self._old_runs is None:
            runner._PATH_OVERRIDES.pop("runs", None)
        else:
            runner._PATH_OVERRIDES["runs"] = self._old_runs
        self.temp.cleanup()

    def fingerprint(self):
        return runner.working_fingerprint(self.run)

    def remote_block(self, **kwargs):
        base = {"provider": "modal", "app_version": self.client.local_version, "gpu": "L40S",
                "call_id": "fc-old", "run_key": "job-" + self.fingerprint()[:12],
                "submitted_at": "2026-11-10T21:00:00", "code_sha256": "c0de" + "0" * 60,
                "plan_fingerprint": "plan" + "0" * 60,
                "dataset_fingerprint": self.fingerprint(), "images": self.n,
                "attempt_id": "att-old", "status": None, "message": ""}
        base.update(kwargs)
        state = modal_backend._read_state(self.run)
        state["remote"] = base
        modal_backend._write_state(self.run, state)
        return base

    def plant_remote_output(self, run_key=None):
        run_key = run_key or ("job-" + self.fingerprint()[:12])
        out = self.root / "staged_output"
        write_training_outputs(out, "job", self.epochs)
        prefix = f"{RUNS}/{run_key}/run/output"
        for root, _dirs, files in os.walk(out / "output"):
            for name in files:
                path = Path(root, name)
                self.client.files[f"{prefix}/{path.relative_to(out / 'output').as_posix()}"] = path.read_bytes()
        return run_key

    def plant_bundle(self, run_key=None):
        run_key = run_key or ("job-" + self.fingerprint()[:12])
        fp = self.fingerprint()
        plan = modal_backend.remote_plan(self.run, self.cfg, self.n, run_key=run_key, attempt_id="att-old",
                                         fingerprint=fp, inventory_fingerprint="inv" + "0" * 61,
                                         code_sha256="c0de" + "0" * 60, epochs=self.epochs)
        self.plan_fingerprint = modal_backend.remote_exec.plan_fingerprint(plan)
        self.client.files[remote_path(run_key, "plan.json")] = json.dumps(plan).encode()
        self.client.files[remote_path(run_key, "code", "sha256.txt")] = ("c0de" + "0" * 60 + "  fizgig_src.tar.gz\n").encode()
        return run_key


class RemotePlanTests(BackendTests):
    def test_plan_has_stages_files_and_model_bytes(self):
        plan = modal_backend.remote_plan(self.run, self.cfg, self.n,
                                         run_key="job-x", attempt_id="att-1",
                                         fingerprint=self.fingerprint(),
                                         inventory_fingerprint="inv" + "0" * 61,
                                         code_sha256="a" * 64, epochs=self.epochs)
        self.assertEqual(plan["attempt_id"], "att-1")
        self.assertEqual(plan["job_id"], "job")
        self.assertEqual(plan["arch_id"], "krea2drv")
        self.assertEqual(plan["images"], self.n)
        self.assertEqual(plan["epochs"], self.epochs)
        self.assertEqual([s["name"] for s in plan["stages"]],
                         ["Cache Preparation", "Text Encoder Caching", "Training"])
        for stage in plan["stages"]:
            for arg in stage["cmd"]:
                self.assertNotIn("\\", str(arg))
                self.assertNotRegex(str(arg), r"^[A-Za-z]:")
        roles = {m["role"]: m for m in plan["model_files"]}
        self.assertEqual(roles["dit"]["bytes"], 26_283_332_608)
        self.assertTrue(roles["speed_lora"]["path"].startswith("/models/"))
        self.assertTrue(all(isinstance(cmd, list) for cmd in (s["cmd"] for s in plan["stages"])))

    def test_estimate_matches_the_documented_formula(self):
        est = modal_backend.estimate(self.cfg, 36, 2)
        steps = 72
        hours = 0.25 + steps * 1.6 / 3600
        rate = 1.95 + 4 * 0.047 + 64 * 0.008
        self.assertAlmostEqual(est["hours"], round(hours, 3), places=3)
        self.assertAlmostEqual(est["usd"], round(hours * rate, 2), places=2)

    def test_code_tarball_is_deterministic(self):
        first = self.root / "a.tgz"
        second = self.root / "b.tgz"
        sha1 = modal_backend.build_source_tarball(Path(runner.ROOT) / "src", first)
        sha2 = modal_backend.build_source_tarball(Path(runner.ROOT) / "src", second)
        self.assertEqual(sha1, sha2)


class SubmitTests(BackendTests):
    def test_submit_deploys_once_then_reuses_the_deployed_version(self):
        self.client.deployed = None   # nothing deployed yet
        result = modal_backend.submit(self.run, self.cfg, self.n, client=self.client, log=lambda *a: None)
        self.assertEqual(self.client.deploys, 1)
        self.assertEqual(result["run_key"], "job-" + self.fingerprint()[:12])
        self.assertTrue(self.client.spawned)
        state = modal_backend._read_state(self.run)
        self.assertEqual(state["remote"]["status"], "submitted")
        self.assertEqual(state["remote"]["attempt_id"], result["attempt_id"])
        self.assertEqual(state["remote"]["dataset_fingerprint"], self.fingerprint())
        # A second submit uses the already-deployed version.
        modal_backend.submit(self.run, self.cfg, self.n, client=self.client, log=lambda *a: None)
        self.assertEqual(self.client.deploys, 1)

    def test_submit_reuses_the_stored_bundle_on_same_input_retry(self):
        fp = self.fingerprint()
        self.plant_bundle()
        self.remote_block(status="failed", dataset_fingerprint=fp,
                          code_sha256="c0de" + "0" * 60, attempt_id="att-old",
                          plan_fingerprint=self.plan_fingerprint)
        result = modal_backend.submit(self.run, self.cfg, self.n, client=self.client, log=lambda *a: None)
        # No bundle re-upload on a same-input retry: only the fresh status record goes up.
        self.assertEqual(self.client.uploaded_files, [remote_path("job-" + fp[:12], "status.json")])
        self.assertEqual(len(self.client.spawned), 1)
        self.assertEqual(result["attempt_id"], "att-old")   # same logical attempt resumed

    def test_submit_uploads_a_fresh_bundle_when_settings_changed(self):
        fp = self.fingerprint()
        self.plant_bundle()
        self.remote_block(status="failed", dataset_fingerprint=fp, code_sha256="c0de" + "0" * 60,
                          plan_fingerprint="deadbeef", attempt_id="att-old")
        modal_backend.submit(self.run, self.cfg, self.n, client=self.client, log=lambda *a: None)
        self.assertEqual(len(self.client.uploads), 1)      # a fresh bundle goes up

    def test_missing_captions_refuse_before_any_upload(self):
        (self.dataset / "a.txt").unlink()
        # Rebuild the run from scratch so the manifest records the missing caption.
        import shutil
        shutil.rmtree(self.run)
        self.run, _i, _ins = runner.prepare(self.cfg)
        before_uploads, before_spawns = len(self.client.uploads), len(self.client.spawned)
        with self.assertRaises(ValueError):
            modal_backend.submit(self.run, self.cfg, self.n, client=self.client, log=lambda *a: None)
        self.assertEqual((len(self.client.uploads), len(self.client.spawned)), (before_uploads, before_spawns))


class DispatchTests(BackendTests):
    def test_no_remote_submits_and_follows(self):
        self.plant_remote_output()
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        state = modal_backend._read_state(self.run)
        self.assertEqual(state["remote"]["status"], "fetched")
        self.assertTrue(state["complete"])
        self.assertGreaterEqual(len(self.client.spawned), 1)
        self.assertIn(f"{RUNS}/job-" + self.fingerprint()[:12], self.client.deleted)

    def test_running_never_resubmits(self):
        fp = self.fingerprint()
        key = self.plant_remote_output("job-" + fp[:12])
        self.client.poll_script["fc-old"] = ["running", "done"]
        self.remote_block(status="running", call_id="fc-old")
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(self.client.spawned, [])   # nothing resubmitted
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "fetched")

    def test_done_fetches_without_spawning(self):
        fp = self.fingerprint()
        key = self.plant_remote_output()
        self.remote_block(status="done")
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(self.client.spawned, [])
        state = modal_backend._read_state(self.run)
        self.assertEqual(state["remote"]["status"], "fetched")
        self.assertTrue(state["complete"])

    def test_fetched_marks_completed_without_touching_the_network(self):
        write_training_outputs(self.run, "job", self.epochs)
        self.remote_block(status="fetched")
        before = len(self.client.spawned) + self.client.deploys
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(len(self.client.spawned) + self.client.deploys, before)
        self.assertTrue(runner.output_valid(self.run, self.cfg, self.epochs))

    def test_fetched_with_damaged_outputs_says_so(self):
        self.remote_block(status="fetched")
        out = modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                                    log=lambda *a: None, sleep=lambda s: None)
        self.assertFalse(out.get("complete"))

    def test_failed_same_input_resubmits_with_stored_bundle(self):
        fp = self.fingerprint()
        self.plant_bundle("job-" + fp[:12])
        self.remote_block(status="failed", dataset_fingerprint=fp,
                          code_sha256="c0de" + "0" * 60, attempt_id="att-old",
                          plan_fingerprint=self.plan_fingerprint)
        self.plant_remote_output()
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        # stored bundle reused: only the fresh status record is uploaded
        self.assertEqual(self.client.uploaded_files, [remote_path("job-" + fp[:12], "status.json")])
        self.assertEqual(len(self.client.spawned), 1)
        state = modal_backend._read_state(self.run)
        self.assertEqual(state["remote"]["status"], "fetched")

    def test_stale_submits_new_run_and_deletes_the_old_folder(self):
        old_fp = "0" * 64
        self.remote_block(status="stale", dataset_fingerprint=old_fp, call_id="fc-old",
                          run_key="job-000000000000")
        self.client.poll_script["fc-old"] = ["failed"]
        self.plant_remote_output()   # planted under the NEW key (current fingerprint)
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        self.assertGreaterEqual(len(self.client.spawned), 1)
        self.assertIn("/runs/job-000000000000", self.client.deleted)

    def test_unknown_refuses_until_the_status_record_exists(self):
        self.remote_block(status="unknown", call_id="fc-old")
        self.client.poll_script["fc-old"] = ["unknown"]
        with self.assertRaises(RuntimeError):
            modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                                  log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(self.client.spawned, [])

    def test_unknown_with_a_finished_record_fetches(self):
        fp = self.fingerprint()
        key = self.plant_remote_output()
        run_key = "job-" + fp[:12]
        self.client.files[remote_path(run_key, "status.json")] = json.dumps(
            {"state": "done", "attempt_id": "att-old", "stage": "Training",
             "code_sha256": "c0de" + "0" * 60, "dataset_fingerprint": fp,
             "exit_code": 0, "message": "", "updated_at": "2026-11-10T22:00:00"}).encode()
        self.remote_block(status="unknown", call_id="fc-old")
        self.client.poll_script["fc-old"] = ["unknown"]
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "fetched")


class FetchTests(BackendTests):
    def test_fetch_gates_on_the_working_fingerprint(self):
        self.plant_remote_output()
        self.remote_block(status="done", dataset_fingerprint="f" * 64)
        with self.assertRaises(RuntimeError):
            modal_backend.fetch(self.run, self.cfg, client=self.client, log=lambda *a: None)
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "stale")
        self.assertEqual(self.client.downloads, [])
        self.assertEqual(self.client.deleted, [])

    def test_fetch_validation_failure_keeps_the_remote_folder(self):
        run_key = self.plant_remote_output()
        # Corrupt the planted LoRA (ss_epoch mismatch).
        prefix = f"{RUNS}/{run_key}/run/output"
        self.client.files[f"{prefix}/job.safetensors"] = b"not a safetensors"
        self.remote_block(status="done")
        with self.assertRaises(RuntimeError):
            modal_backend.fetch(self.run, self.cfg, client=self.client, log=lambda *a: None)
        self.assertEqual(self.client.deleted, [])
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "done")

    def test_fetch_publishes_logs_and_marks_complete(self):
        run_key = self.plant_remote_output()
        self.client.files[remote_path(run_key, "run", "Training.log")] = b"loss 0.5\nloss 0.4\n"
        self.remote_block(status="done")
        modal_backend.fetch(self.run, self.cfg, client=self.client, log=lambda *a: None)
        self.assertTrue((self.run / "Training.log").exists())
        self.assertTrue((self.run / "output" / "job.safetensors").exists())
        state = modal_backend._read_state(self.run)
        self.assertTrue(state["complete"])
        self.assertEqual(state["remote"]["status"], "fetched")
        self.assertEqual(self.client.deleted, [f"{RUNS}/{run_key}"])

    def test_fetch_logs_delete_failure_is_left_for_sync(self):
        self.plant_remote_output()
        self.remote_block(status="done")
        self.client.delete_fails = True
        modal_backend.fetch(self.run, self.cfg, client=self.client, log=lambda *a: None)
        state = modal_backend._read_state(self.run)
        self.assertTrue(state["complete"])
        self.assertTrue(state["remote"]["cleanup_pending"])


class CancelAndSyncTests(BackendTests):
    def test_cancel_request_file_stops_follow_and_cancels(self):
        self.remote_block(status="running", call_id="fc-old")
        self.client.poll_script["fc-old"] = ["running"]
        (self.run / "cancel_remote.request").write_text("", encoding="utf-8")
        result = modal_backend.follow(self.run, self.cfg, client=self.client,
                                      log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(result, {"status": "cancelled"})
        self.assertIn("fc-old", self.client.cancelled)
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "cancelled")
        self.assertFalse((self.run / "cancel_remote.request").exists())

    def test_cancel_when_the_call_is_gone_fetches_the_result(self):
        run_key = self.plant_remote_output()
        self.remote_block(status="running", call_id="fc-old")
        self.client.cancel_accepted = False
        modal_backend.cancel(self.run, client=self.client, log=lambda *a: None)
        state = modal_backend._read_state(self.run)
        self.assertEqual(state["remote"]["status"], "fetched")

    def test_cancel_marks_unknown_when_the_call_is_missing_and_cannot_fetch(self):
        self.remote_block(status="unknown", call_id="fc-old")
        self.client.cancel_accepted = False
        self.client.poll_script["fc-old"] = ["unknown"]
        with self.assertRaises(RuntimeError):
            modal_backend.cancel(self.run, client=self.client, log=lambda *a: None)

    def test_sync_empty_queue_never_touches_the_client(self):
        class ExplodingClient:
            def __getattr__(self, name):
                raise AssertionError("sync must not use the client when there is nothing to do")

        summary = modal_backend.sync(roots=[self.runs], client=ExplodingClient(), log=lambda *a: None)
        self.assertEqual(summary["checked"], 0)

    def test_sync_fetches_finished_and_reports_running(self):
        self.plant_remote_output()      # done remote exists for this run
        self.remote_block(status="done", call_id="fc-1")
        self.client.poll_script["fc-1"] = ["done"]
        summary = modal_backend.sync(roots=[self.runs], client=self.client, log=lambda *a: None)
        self.assertEqual(summary["fetched"], [self.run.name])
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "fetched")

    def test_sync_skips_a_locked_run(self):
        self.remote_block(status="running", call_id="fc-old")
        (self.run / ".remote.lock").write_text(str(os.getpid()), encoding="utf-8")
        summary = modal_backend.sync(roots=[self.runs], client=self.client, log=lambda *a: None)
        self.assertEqual(summary["skipped"], [self.run.name])


class ResubmitOverLeftoverStatusTests(BackendTests):
    """A failed attempt leaves `status.json` (state=failed) on the volume. A re-run must not read it as the new call's verdict."""

    def leave_failed_status(self, run_key):
        self.client.files[remote_path(run_key, "status.json")] = json.dumps(
            {"state": "failed", "attempt_id": "att-old", "stage": "Training", "exit_code": 1,
             "message": "Training exited 1; see the remote run folder", "updated_at": "2026-11-10T22:00:00",
             "code_sha256": "c0de" + "0" * 60, "dataset_fingerprint": self.fingerprint(),
             "log_tail": "CUDA OOM"}).encode()

    def test_stored_bundle_resubmit_follows_the_new_call_to_fetched(self):
        fp = self.fingerprint()
        key = self.plant_bundle("job-" + fp[:12])
        self.remote_block(status="failed", dataset_fingerprint=fp, code_sha256="c0de" + "0" * 60,
                          attempt_id="att-old", plan_fingerprint=self.plan_fingerprint)
        self.leave_failed_status(key)
        self.plant_remote_output(key)
        self.client.poll_script["fc-1"] = ["running", "running", "done"]
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(len(self.client.spawned), 1)
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "fetched")

    def test_fresh_upload_resubmit_follows_the_new_call_to_fetched(self):
        fp = self.fingerprint()
        key = "job-" + fp[:12]
        self.remote_block(status="failed", dataset_fingerprint=fp, attempt_id="att-old",
                          plan_fingerprint="different" + "0" * 55)
        self.leave_failed_status(key)
        self.plant_remote_output(key)
        self.client.poll_script["fc-1"] = ["running", "running", "done"]
        modal_backend.execute(self.run, self.cfg, self.n, client=self.client,
                              log=lambda *a: None, sleep=lambda s: None)
        self.assertEqual(len(self.client.uploads), 1)
        self.assertEqual(len(self.client.spawned), 1)
        self.assertEqual(modal_backend._read_state(self.run)["remote"]["status"], "fetched")

    def test_status_is_reset_to_submitted_before_spawn_on_both_paths(self):
        seen = []
        fake = self.client

        class Recording(FakeClient):
            def spawn(self, gpu, run_key):
                seen.append(json.loads(self.files[remote_path(run_key, "status.json")]))
                return super().spawn(gpu, run_key)

        for reuse in (True, False):
            with self.subTest(reuse=reuse):
                self.client = Recording()
                fp = self.fingerprint()
                key = "job-" + fp[:12]
                if reuse:
                    self.plant_bundle(key)
                    plan_fp = self.plan_fingerprint
                else:
                    plan_fp = "different" + "0" * 55
                self.remote_block(status="failed", dataset_fingerprint=fp, code_sha256="c0de" + "0" * 60,
                                  attempt_id="att-old", plan_fingerprint=plan_fp)
                self.leave_failed_status(key)
                out = modal_backend.submit(self.run, self.cfg, self.n, client=self.client, log=lambda *a: None)
                status = seen[-1]
                self.assertEqual(status["state"], "submitted")
                self.assertEqual(status["attempt_id"], out["attempt_id"])
                self.assertEqual(status["dataset_fingerprint"], fp)
                self.assertIsNone(status["exit_code"])
        self.client = fake

    def test_reuse_path_does_not_spawn_when_the_status_reset_cannot_be_uploaded(self):
        fp = self.fingerprint()
        key = self.plant_bundle("job-" + fp[:12])
        self.remote_block(status="failed", dataset_fingerprint=fp, code_sha256="c0de" + "0" * 60,
                          attempt_id="att-old", plan_fingerprint=self.plan_fingerprint)
        self.leave_failed_status(key)

        def broken_upload(local_dir, remote_dir):
            raise ConnectionError("upload interrupted")

        self.client.upload = broken_upload
        with self.assertRaises(ConnectionError):
            modal_backend.submit(self.run, self.cfg, self.n, client=self.client, log=lambda *a: None)
        self.assertEqual(self.client.spawned, [])


class ContainerVanishedTests(BackendTests):
    """Modal reports the call failed, but the volume holds no failure record from the container (spec 7.2)."""

    def follow_a_call_that_modal_killed(self, volume_state):
        key = "job-" + self.fingerprint()[:12]
        self.remote_block(status="submitted", call_id="fc-dead")
        self.client.poll_script["fc-dead"] = ["failed"]
        self.client.files[remote_path(key, "status.json")] = json.dumps(
            {"state": volume_state, "attempt_id": "att-old", "stage": None, "exit_code": None, "message": "",
             "updated_at": "2026-11-10T22:00:00", "code_sha256": "c0de" + "0" * 60,
             "dataset_fingerprint": self.fingerprint()}).encode()
        with self.assertRaises(RuntimeError):
            modal_backend.follow(self.run, self.cfg, client=self.client, log=lambda *a: None,
                                 sleep=lambda s: None)
        return modal_backend._read_state(self.run)["remote"]

    def test_a_call_killed_mid_stage_says_modal_stopped_it(self):
        remote = self.follow_a_call_that_modal_killed("running")
        self.assertEqual(remote["status"], "failed")
        self.assertIn("Modal이 작업을 중단했다", remote["message"])

    def test_a_call_killed_before_the_container_wrote_anything_says_modal_stopped_it(self):
        remote = self.follow_a_call_that_modal_killed("submitted")
        self.assertEqual(remote["status"], "failed")
        self.assertIn("Modal이 작업을 중단했다", remote["message"])
        self.assertNotIn("exited None", remote["message"])


if __name__ == "__main__":
    unittest.main()
