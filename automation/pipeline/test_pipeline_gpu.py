"""R6: the one-GPU gate, reproduced with real processes.

Two claims and two directions: a live training run (queue.lock, the runner's own claim) refuses every generation
CLI, and a live generation claim refuses a training start. The claims are pid records checked for liveness, so a
crashed run's stale file is replaced rather than obeyed, and the tests spawn real sleeper processes because a pid
that is not running is the whole point.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline import gpu


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

    def write_lock(self, name, pid):
        (self.base / name).write_text(str(pid), encoding="utf-8")

    def test_a_claim_is_taken_and_released(self):
        claim = gpu.claim_generation(automation_dir=self.base, what="test")
        with claim:
            self.assertTrue(gpu.generation_claim_active(self.base))
            self.assertEqual((self.base / gpu.GENERATION_LOCK).read_text(encoding="utf-8").strip(),
                             str(gpu.os.getpid()))
        self.assertFalse(gpu.generation_claim_active(self.base))
        self.assertFalse((self.base / gpu.GENERATION_LOCK).exists())

    def test_a_live_training_run_refuses_the_claim(self):
        self.write_lock(gpu.TRAINING_LOCK, self.sleeper().pid)
        with self.assertRaises(gpu.GpuBusy) as caught:
            gpu.claim_generation(automation_dir=self.base, what="batch B").acquire()
        self.assertIn("training run holds the GPU queue", str(caught.exception))

    def test_a_live_generation_claim_refuses_another(self):
        with gpu.claim_generation(automation_dir=self.base, what="first"):
            with self.assertRaises(gpu.GpuBusy) as caught:
                gpu.claim_generation(automation_dir=self.base, what="second").acquire()
            self.assertIn("another generation run", str(caught.exception))

    def test_a_stale_claim_from_a_dead_run_is_replaced(self):
        self.write_lock(gpu.GENERATION_LOCK, self.dead_pid())
        with gpu.claim_generation(automation_dir=self.base, what="test"):
            self.assertTrue(gpu.generation_claim_active(self.base))

    def test_the_training_side_sees_a_live_claim(self):
        # This predicate is what runner.main checks before it takes the queue lock (spec R6, the other direction).
        self.write_lock(gpu.GENERATION_LOCK, self.sleeper().pid)
        self.assertTrue(gpu.generation_claim_active(self.base))
        self.assertFalse(gpu.training_queue_active(self.base))

    def test_a_live_owner_is_alive_and_a_dead_one_is_not(self):
        child = self.sleeper()
        self.assertTrue(gpu.owner_alive(child.pid))
        self.assertFalse(gpu.owner_alive(self.dead_pid()))
        self.assertFalse(gpu.owner_alive(None))
        self.assertFalse(gpu.owner_alive(0))

    def test_no_lock_at_all_means_nothing_is_active(self):
        self.assertFalse(gpu.training_queue_active(self.base))
        self.assertFalse(gpu.generation_claim_active(self.base))


if __name__ == "__main__":
    unittest.main()
