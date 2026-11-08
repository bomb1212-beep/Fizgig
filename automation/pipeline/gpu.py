"""The one-GPU gate between training, generation and scoring (spec R6, reworked by N1).

This machine trains and generates on the same GPU, and the jobs run from different CLIs whose only shared state is
this folder. Spec N1 replaced the two-lock pre-check dance (a generation claim that checked queue.lock before
creating gpu.lock, a runner that checked gpu.lock before creating queue.lock): between a check and a create the
other side could walk in, and both locks ended up "valid" at once. Now there is **one GPU resource lock**:

* **gpu.lock** — the machine-wide claim every GPU job (training, generation, scoring, the epoch grid, crop scoring)
  takes *atomically* (create-exclusive) before its first submit and holds until its GPU work ends. Whoever arrives
  second gets `GpuBusy` and submits nothing. A parent CLI that runs a GPU-submitting child (tool2) hands ownership
  to the child's pid, so a killed parent never turns a live child's work into a "stale" claim.
* **queue.lock** — the runner's own queue-serialization claim, unchanged in role. Its dead-owner rule is now the
  same one gpu.lock uses: a dead owner's file is replaced, never obeyed.

Lock records are JSON `{"pid": ..., "what": ..., "at": ..., "heir": ...}`; a bare pid (the old format) still
parses. Liveness is the owner pid **or** the heir pid. A file that exists but is empty is the tiny span between
create and write, not a dead owner: the acquirer waits out a short grace window before it is allowed to replace
it, so a slow writer is never mugged.

Nothing here ever kills a process or stops a server: a claim only says "someone is using the GPU". External
ComfyUI/llama servers are never touched - `vram_gate` reads free VRAM and refuses a start that would not fit,
and `free_comfy_before_training` (runner.py) only *asks* the configured server to release memory when it is idle.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
GPU_LOCK = "gpu.lock"
TRAINING_LOCK = "queue.lock"
# How long an existing-but-unreadable lock file is treated as a writer-in-progress rather than a crashed owner.
EMPTY_GRACE_SECONDS = 5.0
# Ceiling on the acquire loop itself: bounded retries, never an unbounded spin.
ACQUIRE_TIMEOUT_SECONDS = 10.0

# Measured peaks from PROGRESS (krea2 seed training 16.9 GB; P-H3 w4a8+bf16 inference 25.2 GB). A family/kind the
# table does not name has no reliable number, so the gate warns and proceeds rather than guessing a threshold.
DEFAULT_VRAM_REQUIRED_MB = {"krea2.train": 17300, "ph3.draw": 25800, "ph3.edit": 25800, "scoring.score": 1600}


class GpuBusy(RuntimeError):
    """The GPU is claimed by another job; the caller submits nothing."""


class VramShort(RuntimeError):
    """Free VRAM is below the configured requirement; the caller starts nothing."""


def _pid_of(path):
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_record(path):
    """Parse a lock file: JSON record, or the legacy bare-pid text. None when empty or unreadable."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            pid = data.get("pid")
            heir = data.get("heir")
            return {"pid": int(pid) if pid else None, "heir": int(heir) if heir else None,
                    "what": str(data.get("what") or "")}
    except ValueError:
        pass
    try:
        return {"pid": int(text), "heir": None, "what": ""}
    except ValueError:
        return None


def _atomic_write_text(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


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


def _lock_path(automation_dir, name):
    base = Path(automation_dir) if automation_dir else AUTO
    return base / name


def gpu_claim_active(automation_dir=None):
    """True when a live process (owner or heir) holds the GPU resource lock."""
    record = _read_record(_lock_path(automation_dir, GPU_LOCK))
    if record is None:
        return False
    return owner_alive(record["pid"]) or owner_alive(record["heir"])


def generation_claim_active(automation_dir=None):
    """Older name for `gpu_claim_active`: a live generation claim means the GPU lock is held."""
    return gpu_claim_active(automation_dir)


def training_queue_active(automation_dir=None):
    """True when a live training runner holds the queue lock (queue serialization, not the GPU claim)."""
    return owner_alive(_pid_of(_lock_path(automation_dir, TRAINING_LOCK)))


def _replace_stale(path, expected_bytes):
    """Remove a dead owner's lock file, but only if it still holds the bytes we judged stale.

    A racer that created a fresh lock between our read and our unlink writes different content, so the comparison
    fails and the file survives; the caller's next loop iteration sees the live owner and refuses.
    """
    try:
        if path.read_bytes() != expected_bytes:
            return False
    except OSError:
        return False
    path.unlink(missing_ok=True)
    return True


class GpuClaim:
    """The machine-wide GPU claim one GPU job holds for the whole span of its GPU work (spec N1).

    One lock, one direction: acquire is a single create-exclusive on gpu.lock, so no two-file pre-check is ever
    trusted to keep training and generation apart. Taking it refuses while *any* live job holds the lock, whatever
    it is - training, another generation, scoring - and the runner refuses while this claim is live. Whichever job
    came second submits nothing.
    """

    def __init__(self, automation_dir=None, *, what="generation"):
        self.base = Path(automation_dir) if automation_dir else AUTO
        self.what = what
        self.path = self.base / GPU_LOCK
        self.acquired = False
        self.heir = None

    def _record_text(self):
        return json.dumps({"pid": os.getpid(), "what": self.what, "at": time.time()}, ensure_ascii=False)

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        empty_deadline = time.monotonic() + EMPTY_GRACE_SECONDS
        gave_up = time.monotonic() + ACQUIRE_TIMEOUT_SECONDS
        while True:
            if time.monotonic() > gave_up:
                raise GpuBusy(f"could not take the GPU claim at {self.path}")
            try:
                with self.path.open("x", encoding="utf-8") as handle:
                    handle.write(self._record_text())
                self.acquired = True
                return self
            except FileExistsError:
                content = self.path.read_bytes()
                record = _read_record(self.path)
                if record is None or (record["pid"] is None and record["heir"] is None):
                    # Empty or unreadable: a writer between create and write, not a dead owner. Wait out the grace
                    # window, then treat it as a crash between those two statements and replace it. An acquirer
                    # that cannot wait that long gives up instead of mugging the writer.
                    if time.monotonic() >= empty_deadline:
                        _replace_stale(self.path, content)
                    else:
                        time.sleep(0.05)
                    continue
                if owner_alive(record["pid"]) or owner_alive(record["heir"]):
                    pid = record["pid"]
                    role = record["what"] or "another GPU job"
                    raise GpuBusy(f"the GPU is already claimed by {role} (pid {pid}); {self.what} would contend "
                                  "for the same card - wait for it to finish")
                _replace_stale(self.path, content)      # a stale claim from a dead run: replaced, not obeyed
                continue

    def hand_off(self, child_pid):
        """Record the child process that now does this job's GPU work (spec N1 ownership handoff).

        A parent CLI that runs a GPU-submitting child (tool2) stays the lock's writer, but liveness follows the
        child too: a parent that dies mid-run leaves the claim saying "busy" until the child itself is gone, so
        the next job neither contends with the orphan nor steals its lock.
        """
        if not self.acquired:
            return False
        record = _read_record(self.path)
        if record is None or record.get("pid") != os.getpid():
            return False
        record["heir"] = int(child_pid)
        _atomic_write_text(self.path, json.dumps(record, ensure_ascii=False))
        self.heir = int(child_pid)
        return True

    def release(self):
        if not self.acquired:
            return False
        record = _read_record(self.path)
        if record is not None and record.get("pid") not in (None, os.getpid()):
            # The file no longer records us (a stale replacement took it over): never delete someone else's lock.
            self.acquired = False
            return False
        self.path.unlink(missing_ok=True)
        self.acquired = False
        return True

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_exc):
        self.release()
        return False


def claim_gpu(automation_dir=None, *, what="generation"):
    """Take the GPU claim: `with claim_gpu(): ...` around everything that starts GPU work (spec N1)."""
    return GpuClaim(automation_dir=automation_dir, what=what)


def claim_generation(automation_dir=None, *, what="generation"):
    """Older name for `claim_gpu`, kept for the existing generation CLIs."""
    return claim_gpu(automation_dir=automation_dir, what=what)


def take_queue_lock(path, *, what="training"):
    """Create the runner's queue lock with the unified dead-owner rule (spec N1).

    pid text, as before - `runner.queue_owner_running` and the GUIs read it. A live owner refuses; a dead owner's
    file is replaced; an empty file gets the same grace window as the GPU lock, because the writer may still be
    between create and write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    empty_deadline = time.monotonic() + EMPTY_GRACE_SECONDS
    gave_up = time.monotonic() + ACQUIRE_TIMEOUT_SECONDS
    while True:
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            return path
        except FileExistsError:
            content = path.read_bytes()
            pid = _pid_of(path)
            if pid is None:
                if time.monotonic() < empty_deadline:
                    time.sleep(0.05)
                    continue
                if time.monotonic() > gave_up:
                    raise GpuBusy(f"could not take the queue lock at {path}")
                _replace_stale(path, content)
                continue
            if owner_alive(pid):
                raise GpuBusy(f"Queue lock exists and its owner (pid {pid}) is running; if no runner is active, "
                              "remove automation/queue.lock.")
            if time.monotonic() > gave_up:
                raise GpuBusy(f"could not take the queue lock at {path}")
            _replace_stale(path, content)


# --- the read-only VRAM gate (spec N1.6) -----------------------------------------------------------

def vram_required_mb(family, kind, config=None):
    """The configured requirement for this family/kind, or None when no measured number exists for it."""
    table = dict(DEFAULT_VRAM_REQUIRED_MB)
    configured = ((config or {}).get("gpu") or {}).get("vram_required")
    if isinstance(configured, dict):
        table.update(configured)
    for key in (f"{family}.{kind}", kind, family):
        if key in table:
            return table[key]
    return None


def free_vram_mb(*, nvidia_smi="nvidia-smi", query=None):
    """Free VRAM of the first GPU, read-only. None when it cannot be read - the gate must not block on that."""
    if query is not None:
        return query()
    try:
        out = subprocess.run([nvidia_smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def vram_gate(family, kind, *, config=None, log=print, query=None, nvidia_smi="nvidia-smi"):
    """Refuse a GPU start that would not fit, warn-and-proceed when there is no number to compare (spec N1.6).

    Returns "allow" or "warn"; raises `VramShort` with the required and the free amount when the measured free VRAM
    is below the configured requirement. A failed query or a family/kind without a configured threshold never
    blocks. Nothing is killed and no release is requested here.
    """
    required = vram_required_mb(family, kind, config)
    if required is None:
        log(f"[gpu] no VRAM requirement configured for {family}.{kind}; skipping the VRAM gate")
        return "warn"
    smi = ((config or {}).get("gpu") or {}).get("nvidia_smi") or nvidia_smi
    free = free_vram_mb(nvidia_smi=smi, query=query)
    if free is None:
        log(f"[gpu] could not read free VRAM; proceeding without the VRAM gate for {family}.{kind}")
        return "warn"
    if free < required:
        raise VramShort(f"{family}.{kind} needs about {required:,} MB of free VRAM but only {free:,} MB is free; "
                        "not starting - close the job that holds the memory yourself (nothing is killed here)")
    log(f"[gpu] VRAM gate passed: {free:,} MB free >= {required:,} MB needed for {family}.{kind}")
    return "allow"


def wait_for_queue(seconds=0.0):
    """Test helper: sleep long enough for another process to observe a claim."""
    time.sleep(seconds)
