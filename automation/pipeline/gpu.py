"""The one-GPU gate between training and generation (spec R6).

This machine trains and generates on the same GPU, and the two run from different CLIs whose only shared state is this
folder. Two claims keep the later job from contending with the earlier one:

* **training** is claimed by `runner.py`'s `queue.lock` (the runner creates it for its whole run);
* **generation** is claimed here, by `pipeline.gpu.claim_generation`, for the span in which a generation CLI actually
  submits work.

Every generation CLI checks both directions before its first submit - a live training owner refuses generation, and a
live generation claim refuses a training start (the runner checks before it takes the queue). The claims are pid
records with a liveness check, so a crashed run leaves a stale file that is replaced rather than a permanent refusal.
Nothing here ever kills a process or stops a server: a claim only says "someone is using the GPU".
"""
from __future__ import annotations

import os
import time
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
TRAINING_LOCK = "queue.lock"
GENERATION_LOCK = "gpu.lock"


class GpuBusy(RuntimeError):
    """The GPU is claimed by the other side; the caller submits nothing."""


def _pid_of(path):
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def owner_alive(pid):
    """Is this pid a live process? Windows access-denied counts as alive, not as proof of death."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def training_queue_active(automation_dir=None):
    """True when a live training runner holds the queue lock."""
    base = Path(automation_dir) if automation_dir else AUTO
    return owner_alive(_pid_of(base / TRAINING_LOCK))


def generation_claim_active(automation_dir=None):
    """True when a live generation run holds the generation claim."""
    base = Path(automation_dir) if automation_dir else AUTO
    return owner_alive(_pid_of(base / GENERATION_LOCK))


class GenerationClaim:
    """The machine-wide claim a generation CLI holds while it submits work.

    One GPU, so one claim: two configs pointing at the same card are the same contention. Taking it refuses while a
    training run holds the queue, and the runner refuses while this claim is live - whichever job came second submits
    nothing (spec R6).
    """

    def __init__(self, automation_dir=None, *, what="generation"):
        self.base = Path(automation_dir) if automation_dir else AUTO
        self.what = what
        self.path = self.base / GENERATION_LOCK
        self.acquired = False

    def acquire(self):
        base = self.base
        if training_queue_active(base):
            pid = _pid_of(base / TRAINING_LOCK)
            raise GpuBusy(f"a training run holds the GPU queue (pid {pid}); {self.what} would contend for "
                          "the same GPU - wait for training to finish")
        if generation_claim_active(base):
            pid = _pid_of(base / GENERATION_LOCK)
            raise GpuBusy(f"another generation run is already using the GPU (pid {pid}); {self.what} waits")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                with self.path.open("x", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                self.acquired = True
                return self
            except FileExistsError:
                pid = _pid_of(self.path)
                if owner_alive(pid):
                    raise GpuBusy(f"another generation run is already using the GPU (pid {pid}); "
                                  f"{self.what} waits")
                self.path.unlink(missing_ok=True)      # a stale claim from a dead run: replaced, not obeyed
        raise GpuBusy(f"could not take the GPU claim at {self.path}")

    def release(self):
        if not self.acquired:
            return False
        self.path.unlink(missing_ok=True)
        self.acquired = False
        return True

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_exc):
        self.release()
        return False


def claim_generation(automation_dir=None, *, what="generation"):
    """Take the generation claim: `with claim_generation(): ...` around everything that submits GPU work."""
    return GenerationClaim(automation_dir=automation_dir, what=what)


def wait_for_queue(seconds=0.0):
    """Test helper: sleep long enough for another process to observe a claim."""
    time.sleep(seconds)
