"""N1: the one-GPU gate, reproduced with real processes.

One GPU resource lock (gpu.lock) now carries the whole mutual exclusion: training, generation, scoring and the
epoch grid all take it atomically, so no two-file pre-check is trusted to keep them apart. The queue lock keeps
its queue-serialization role and the same dead-owner rule. The tests spawn real sleeper processes because a pid
that is not running is the whole point, and the race tests drive real product entry points in real subprocesses.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import gpu  # noqa: E402


class GpuGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.children = []

    def tearDown(self):
        for child in self.children:
            child.kill()
            child.wait()
        self.temp.cleanup()

    def sleeper(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.children.append(child)
        return child

    def dead_pid(self):
        child = self.sleeper()
        child.kill()
        child.wait()
        return child.pid

    def write_lock(self, name, pid, *, what="", heir=None):
        record = {"pid": pid, "what": what, "at": 0.0}
        if heir is not None:
            record["heir"] = heir
        (self.base / name).write_text(json.dumps(record), encoding="utf-8")

    def test_a_claim_is_taken_and_released(self):
        claim = gpu.claim_gpu(automation_dir=self.base, what="test")
        with claim:
            self.assertTrue(gpu.gpu_claim_active(self.base))
            record = json.loads((self.base / gpu.GPU_LOCK).read_text(encoding="utf-8"))
            self.assertEqual(record["pid"], gpu.os.getpid())
        self.assertFalse(gpu.gpu_claim_active(self.base))
        self.assertFalse((self.base / gpu.GPU_LOCK).exists())

    def test_a_legacy_bare_pid_record_still_means_busy(self):
        (self.base / gpu.GPU_LOCK).write_text(str(self.sleeper().pid), encoding="utf-8")
        self.assertTrue(gpu.gpu_claim_active(self.base))

    def test_a_live_training_claim_refuses_generation(self):
        # One lock, both directions: what the job calls itself does not matter, the card is single.
        self.write_lock(gpu.GPU_LOCK, self.sleeper().pid, what="training run")
        with self.assertRaises(gpu.GpuBusy) as caught:
            gpu.claim_gpu(automation_dir=self.base, what="batch B").acquire()
        self.assertIn("training run", str(caught.exception))

    def test_a_live_generation_claim_refuses_another(self):
        with gpu.claim_gpu(automation_dir=self.base, what="first"):
            with self.assertRaises(gpu.GpuBusy) as caught:
                gpu.claim_gpu(automation_dir=self.base, what="second").acquire()
            self.assertIn("already claimed", str(caught.exception))

    def test_a_stale_claim_from_a_dead_run_is_replaced(self):
        self.write_lock(gpu.GPU_LOCK, self.dead_pid())
        with gpu.claim_gpu(automation_dir=self.base, what="test"):
            self.assertTrue(gpu.gpu_claim_active(self.base))

    def test_an_empty_lock_is_a_writer_in_progress_not_a_dead_owner(self):
        # The span between create and write must not be mugged: within the grace window the acquirer refuses
        # instead of stealing, and the empty file survives untouched.
        (self.base / gpu.GPU_LOCK).write_text("", encoding="utf-8")
        original_grace, original_timeout = gpu.EMPTY_GRACE_SECONDS, gpu.ACQUIRE_TIMEOUT_SECONDS
        gpu.EMPTY_GRACE_SECONDS, gpu.ACQUIRE_TIMEOUT_SECONDS = 30.0, 0.4
        try:
            with self.assertRaises(gpu.GpuBusy):
                gpu.claim_gpu(automation_dir=self.base, what="impatient").acquire()
        finally:
            gpu.EMPTY_GRACE_SECONDS, gpu.ACQUIRE_TIMEOUT_SECONDS = original_grace, original_timeout
        self.assertEqual((self.base / gpu.GPU_LOCK).read_text(encoding="utf-8"), "")
        self.assertFalse(gpu.gpu_claim_active(self.base))

    def test_an_empty_lock_past_the_grace_window_is_replaced(self):
        # A crash between create and write leaves the file empty forever; past the grace window it is replaced,
        # which is not the same as obeying it.
        (self.base / gpu.GPU_LOCK).write_text("", encoding="utf-8")
        original_grace = gpu.EMPTY_GRACE_SECONDS
        gpu.EMPTY_GRACE_SECONDS = 0.1
        try:
            with gpu.claim_gpu(automation_dir=self.base, what="test"):
                self.assertTrue(gpu.gpu_claim_active(self.base))
        finally:
            gpu.EMPTY_GRACE_SECONDS = original_grace

    def test_release_only_removes_your_own_lock(self):
        first = gpu.claim_gpu(automation_dir=self.base, what="first").acquire()
        second = gpu.claim_gpu(automation_dir=self.base, what="second")
        self.assertFalse(second.release())                      # not acquired: removes nothing
        self.assertTrue((self.base / gpu.GPU_LOCK).exists())
        self.assertTrue(gpu.gpu_claim_active(self.base))
        self.assertTrue(first.release())
        self.assertFalse((self.base / gpu.GPU_LOCK).exists())

    def test_a_live_owner_is_alive_and_a_dead_one_is_not(self):
        child = self.sleeper()
        self.assertTrue(gpu.owner_alive(child.pid))
        self.assertFalse(gpu.owner_alive(self.dead_pid()))
        self.assertFalse(gpu.owner_alive(None))
        self.assertFalse(gpu.owner_alive(0))

    def test_no_lock_at_all_means_nothing_is_active(self):
        self.assertFalse(gpu.gpu_claim_active(self.base))
        self.assertFalse(gpu.training_queue_active(self.base))


class HandoffTests(unittest.TestCase):
    """The parent-to-child ownership contract (spec N1): the lock follows the GPU work, not the CLI that wrote it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.children = []

    def tearDown(self):
        for child in self.children:
            child.kill()
            child.wait()
        self.temp.cleanup()

    def write_lock(self, name, pid, *, what="", heir=None):
        record = {"pid": pid, "what": what, "at": 0.0}
        if heir is not None:
            record["heir"] = heir
        (self.base / name).write_text(json.dumps(record), encoding="utf-8")

    def sleeper(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.children.append(child)
        return child

    def dead_pid(self):
        child = self.sleeper()
        child.kill()
        child.wait()
        return child.pid

    def test_hand_off_records_the_child_so_an_orphan_stays_busy(self):
        claim = gpu.claim_gpu(automation_dir=self.base, what="generator A").acquire()
        child = self.sleeper()
        self.assertTrue(claim.hand_off(child.pid))
        record = json.loads((self.base / gpu.GPU_LOCK).read_text(encoding="utf-8"))
        self.assertEqual(record["heir"], child.pid)
        # The parent is still alive here, so liveness is trivially true; the contract question is the record.
        claim.release()
        self.assertFalse((self.base / gpu.GPU_LOCK).exists())

    def test_a_dead_owner_with_a_live_heir_still_means_busy(self):
        # A killed parent must not turn the orphan child's GPU work into a stale claim (spec N1.2).
        self.write_lock(gpu.GPU_LOCK, self.dead_pid(), heir=self.sleeper().pid)
        self.assertTrue(gpu.gpu_claim_active(self.base))
        with self.assertRaises(gpu.GpuBusy):
            gpu.claim_gpu(automation_dir=self.base, what="next").acquire()

    def test_a_dead_owner_and_a_dead_heir_are_replaced(self):
        self.write_lock(gpu.GPU_LOCK, self.dead_pid(), heir=self.dead_pid())
        with gpu.claim_gpu(automation_dir=self.base, what="next"):
            self.assertTrue(gpu.gpu_claim_active(self.base))

    def test_hand_off_needs_the_claim(self):
        claim = gpu.claim_gpu(automation_dir=self.base, what="never acquired")
        self.assertFalse(claim.hand_off(self.sleeper().pid))
        self.assertFalse((self.base / gpu.GPU_LOCK).exists())


class QueueLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.children = []

    def tearDown(self):
        for child in self.children:
            child.kill()
            child.wait()
        self.temp.cleanup()

    def sleeper(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.children.append(child)
        return child

    def dead_pid(self):
        child = self.sleeper()
        child.kill()
        child.wait()
        return child.pid

    def test_a_live_owner_refuses(self):
        (self.base / gpu.TRAINING_LOCK).write_text(str(self.sleeper().pid), encoding="utf-8")
        with self.assertRaises(gpu.GpuBusy) as caught:
            gpu.take_queue_lock(self.base / gpu.TRAINING_LOCK)
        self.assertIn("Queue lock exists", str(caught.exception))

    def test_a_dead_owner_is_replaced_not_obeyed(self):
        # The unified rule (spec N1): runner.py used to refuse on mere file existence, even for a dead pid.
        (self.base / gpu.TRAINING_LOCK).write_text(str(self.dead_pid()), encoding="utf-8")
        path = gpu.take_queue_lock(self.base / gpu.TRAINING_LOCK)
        self.assertEqual(int(path.read_text(encoding="utf-8")), gpu.os.getpid())
        path.unlink(missing_ok=True)

    def test_an_empty_file_gets_the_grace_window(self):
        (self.base / gpu.TRAINING_LOCK).write_text("", encoding="utf-8")
        original_grace, original_timeout = gpu.EMPTY_GRACE_SECONDS, gpu.ACQUIRE_TIMEOUT_SECONDS
        gpu.EMPTY_GRACE_SECONDS, gpu.ACQUIRE_TIMEOUT_SECONDS = 30.0, 0.4
        try:
            with self.assertRaises(gpu.GpuBusy):
                gpu.take_queue_lock(self.base / gpu.TRAINING_LOCK)
        finally:
            gpu.EMPTY_GRACE_SECONDS, gpu.ACQUIRE_TIMEOUT_SECONDS = original_grace, original_timeout
        self.assertFalse(gpu.owner_alive(None))


class VramGateTests(unittest.TestCase):
    def setUp(self):
        self.log = []

    def logs(self):
        return " | ".join(self.log)

    def test_below_the_requirement_is_refused_with_both_numbers(self):
        with self.assertRaises(gpu.VramShort) as caught:
            gpu.vram_gate("krea2", "train", query=lambda: 1024, log=self.log.append)
        self.assertIn("17,300", str(caught.exception))
        self.assertIn("1,024", str(caught.exception))

    def test_enough_free_memory_proceeds(self):
        self.assertEqual(gpu.vram_gate("krea2", "train", query=lambda: 20000, log=self.log.append), "allow")

    def test_a_failed_query_warns_and_proceeds(self):
        self.assertEqual(gpu.vram_gate("krea2", "train", query=lambda: None, log=self.log.append), "warn")
        self.assertIn("could not read free VRAM", self.logs())

    def test_an_unmeasured_family_warns_and_proceeds(self):
        self.assertEqual(gpu.vram_gate("some_family", "train", query=lambda: 10, log=self.log.append), "warn")
        self.assertIn("no VRAM requirement", self.logs())

    def test_the_config_can_set_or_override_thresholds(self):
        config = {"gpu": {"vram_required": {"some_family.train": 500}}}
        with self.assertRaises(gpu.VramShort):
            gpu.vram_gate("some_family", "train", config=config, query=lambda: 400, log=self.log.append)
        config["gpu"]["vram_required"]["some_family.train"] = 300
        self.assertEqual(gpu.vram_gate("some_family", "train", config=config,
                                       query=lambda: 400, log=self.log.append), "allow")


class _FakeTool2Fixture:
    """The card, config and fake tool2 CLI the gen_qwen_edit entry-point race test needs."""

    FAKE_TOOL2 = '''import json, sys, time
from pathlib import Path
GRAPH_VERSION = "fake@1"
def value(flag, args):
    return args[args.index(flag) + 1] if flag in args else None
def _cli():
    args = sys.argv[1:]
    out = Path(value("--out", args)) / value("--character", args)
    result_arg = value("--result-json", args)
    if "--dry-run" in args:
        sys.exit(0)
    # The parent CLI runs this file with tool2's folder as the cwd, so cwd is the one shared place both race
    # processes can agree on. Hold the GPU claim open until the observer writes the release file, so the losing
    # process is guaranteed to attempt its claim while the winner's GPU work is still in flight.
    release = Path.cwd() / "release"
    deadline = time.time() + 60
    while not release.exists() and time.time() < deadline:
        time.sleep(0.05)
    with (Path.cwd() / "tool2_invocations.txt").open("a", encoding="utf-8") as handle:
        handle.write("invoked\\n")
    print(json.dumps({"kind": "run", "character": value("--character", args), "planned": 1}), flush=True)
    print(json.dumps({"kind": "image", "stem": "framing_00_abc_s1000", "status": "done"}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    made = out / "framing_00_abcdefabcdef_s1000.png"
    made.write_bytes(b"MADE-BY-TOOL2")
    Path(result_arg).write_text(json.dumps({"version": 1, "graph_version": "fake@1", "characters": [
        {"folder": value("--character", args), "planned": 1, "done": 1, "generated": [
            {"stem": "framing_00_abcdefabcdef_s1000", "file": str(made), "category": "framing"}]}]}),
        encoding="utf-8")
    print(json.dumps({"kind": "summary", "done": 1, "skipped": 0, "failed": 0}), flush=True)
if __name__ == "__main__":
    _cli()
'''


class EntryPointRaceTests(_FakeTool2Fixture, unittest.TestCase):
    """Two real processes leaving a real barrier through the real `pipeline.gen_qwen_edit` entry point.

    Spec N1's completion condition: of the two processes that start together, exactly one reaches GPU work; the
    other submits nothing (the fake tool2 is invoked once) and fails with GpuBusy. The winner's tool2 holds the
    claim until the test releases it, so the loser always contends.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.batch = self.dir / "fake_tool2.py"
        self.batch.write_text(self.FAKE_TOOL2, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def project_fixture(self, name):
        chars = self.dir / f"chars_{name}" / "tester"
        chars.mkdir(parents=True)
        (chars / "front.png").write_bytes(b"IMG-" + name.encode())
        config = self.dir / f"jobs_{name}.json"
        config.write_text(json.dumps({
            "comfy_url": "http://127.0.0.1:8190", "resolution": 1024, "base_seed": 1000, "seeds_per_job": 1,
            "models": {"unet": "u", "clip": "c", "vae": "v"},
            "expressions": {"steps": 25, "cfg": 1.0, "items": [{"text": "a smile", "tags": "smiling"}]},
            "characters": [{"folder": "tester", "trigger": "trg", "edit_bases": 1}],
        }, ensure_ascii=False), encoding="utf-8")
        project = self.dir / f"project_{name}"
        project.mkdir(exist_ok=True)
        # gen.main reads the card from <project>/project.json (args.card default).
        (project / "project.json").write_text(
            json.dumps({"id": name, "tool2": {"config": str(config), "character": "tester",
                                              "characters": str(chars)}}, ensure_ascii=False),
            encoding="utf-8")
        return project

    def test_two_processes_racing_at_a_barrier_submit_once(self):
        go = self.dir / "go"
        release = self.dir / "release"
        invocations = self.dir / "tool2_invocations.txt"
        runner = (
            "import sys, time\n"
            "from pathlib import Path\n"
            f"while not Path({str(go)!r}).exists():\n"
            "    time.sleep(0.02)\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            f"sys.path.insert(0, {str(AUTO)!r})\n"
            "from pipeline import gen_qwen_edit as gen\n"
            "fake = {'python': sys.executable, 'batch': sys.argv[1]}\n"
            "gen.tool_paths = lambda cfg=None: fake\n"
            "raise SystemExit(gen.main(['--project', sys.argv[2]]))\n"
        )
        script = self.dir / "race_runner.py"
        script.write_text(runner, encoding="utf-8")
        projects = [self.project_fixture("one"), self.project_fixture("two")]
        children = [subprocess.Popen([sys.executable, str(script), str(self.batch), str(project)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     encoding="utf-8", errors="replace") for project in projects]
        try:
            # The barrier: both runners sit in their wait loop until this file appears, then claim together.
            go.write_text("now", encoding="utf-8")
            outcomes = [child.communicate(timeout=120) for child in children]
        finally:
            release.write_text("go", encoding="utf-8")
            for child in children:
                if child.poll() is None:
                    child.wait(timeout=120)
        codes = [child.returncode for child in children]
        self.assertEqual(sorted(code == 0 for code in codes), [False, True],
                         f"exactly one entry point may reach GPU work; stderr: {outcomes}")
        busy_messages = sum(1 for (_out, err) in outcomes if "already claimed" in err)
        self.assertEqual(busy_messages, 1, f"the loser must be refused with GpuBusy; stderr: {outcomes}")
        self.assertEqual(invocations.read_text(encoding="utf-8").count("invoked"), 1,
                         "the refused process must submit nothing")


if __name__ == "__main__":
    unittest.main()
