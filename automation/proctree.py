"""Ending a stage means ending everything it started (spec N2).

A pipeline stage is a subprocess, and that subprocess starts more of them: `seed --train` runs `runner.py`, the runner
runs the trainer, and generator A / the correction run start tool2 which may in turn start a ComfyUI request. Calling
`terminate()` on the first child leaves the grandchildren drawing on the GPU after the window is gone - the audit
reproduced exactly that (`gui_cancel_child`: parent exited, owned child still alive).

So cancellation here has two steps, in this order:

1. **Ask politely.** The stages that take `--cancel-file` stop between cells, which leaves every finished picture on
   disk; that is what makes a rerun continue instead of restarting. This module only writes that marker.
2. **Then end the tree.** Only after the grace period does anything get killed, and only the process tree rooted at
   the child this program started: `taskkill /T /F` on Windows, per-pid `SIGKILL` elsewhere. A server the user started
   themselves is not in that tree, so it is never a target - that is the whole reason the kill is scoped by parentage
   instead of by image name or port.

Nothing here is Windows-only in its interface: the process table falls back to `ps` off Windows, so the same contract
is exercised by the tests on either platform.
"""
from __future__ import annotations

import os
import subprocess
import time

# taskkill's tree kill is the only dependency-free way to end a whole tree on Windows; it ships with the OS.
TASKKILL = "taskkill"
# How long a politely-asked stage gets before its tree is ended anyway.
DEFAULT_GRACE = 5.0
# How long after the kill to keep checking that nothing in the tree survived.
KILL_SETTLE = 10.0


def _process_table():
    """{pid: parent_pid} for every process the OS will tell us about. Empty when it cannot be read."""
    if os.name != "nt":
        return _process_table_posix()
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:                     # pragma: no cover - ctypes is part of CPython
        return {}

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(Entry))
    kernel.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(Entry))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    snapshot = kernel.CreateToolhelp32Snapshot(0x00000002, 0)      # TH32CS_SNAPPROCESS
    if not snapshot:
        return {}
    table = {}
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        if kernel.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                table[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                if not kernel.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel.CloseHandle(snapshot)
    return table


def _process_table_posix():
    table = {}
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return table
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = int(parts[1])
    return table


def descendants(pid, *, table=None):
    """Every live pid below `pid`, nearest first. Excludes `pid` itself."""
    if not pid or int(pid) <= 0:
        return []
    pid = int(pid)
    table = _process_table() if table is None else table
    children = {}
    for child, parent in table.items():
        children.setdefault(parent, []).append(child)
    found, frontier = [], [pid]
    while frontier:
        current = frontier.pop(0)
        for child in children.get(current, ()):
            if child in found or child == pid:
                continue                       # a pid cycle cannot happen, but a repeat must not loop forever
            found.append(child)
            frontier.append(child)
    return found


def alive(pid):
    """Is this pid a running process? Access-denied counts as alive (a live process we may not inspect)."""
    if not pid or int(pid) <= 0:
        return False
    pid = int(pid)
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
        except ImportError:                 # pragma: no cover
            return False
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)                 # QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                         # access denied: not proof of death
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def wait_gone(pids, timeout, *, interval=0.05, is_alive=None):
    """Wait until none of these pids is alive, or the timeout expires. Returns the pids still alive."""
    probe = is_alive or alive
    deadline = time.monotonic() + max(0.0, timeout)
    remaining = [int(pid) for pid in pids if pid]
    while True:
        remaining = [pid for pid in remaining if probe(pid)]
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(interval)


def kill_tree(pid, *, log=print, is_alive=None):
    """Kill `pid` and every process below it. Returns the pids it covered and those still alive.

    The tree is read *before* the kill so the result can say which pids were covered, and re-checked afterwards so a
    caller can tell "nothing was left running" from "something survived" instead of assuming either. Only the tree
    rooted at `pid` is touched: a process that merely shares an image name or a port is not a target.
    """
    probe = is_alive or alive
    if not pid or int(pid) <= 0:
        return {"root": None, "targets": [], "alive": [], "error": "no pid"}
    pid = int(pid)
    targets = [pid] + descendants(pid)
    if os.name == "nt":
        try:
            subprocess.run([TASKKILL, "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            log(f"[proctree] taskkill failed for pid {pid}: {exc}")
    else:
        # Killing deepest-first means a parent cannot respawn a child while the loop is still running.
        for target in reversed(targets):
            try:
                os.kill(target, 9)
            except OSError:
                pass
    still = wait_gone(targets, KILL_SETTLE, is_alive=probe)
    if still:
        log(f"[proctree] still alive after the tree kill: {still}")
    return {"root": pid, "targets": targets, "alive": still, "error": None}


def stop_tree(pid, *, cancel_file=None, grace=DEFAULT_GRACE, log=print, is_alive=None):
    """Ask nicely, then end the tree. The shared cancel path for both GUIs (spec N2).

    `cancel_file` is the marker the stage polls between cells; writing it is the graceful request. After `grace`
    seconds the tree is ended whether or not the stage co-operated, because a stage that ignores the marker (or is
    stuck in one long cell) must still stop when the window is closing.

    Returns `{"graceful", "killed", "targets", "alive", "cancel_file"}`. A non-empty `alive` means the cancel could
    not be completed, and the caller has to say so rather than report a clean stop.
    """
    probe = is_alive or alive
    report = {"graceful": False, "killed": False, "targets": [], "alive": [], "cancel_file": None}
    if not pid or int(pid) <= 0:
        return report
    pid = int(pid)
    # Read the tree *before* anything is asked or killed: once the root is gone the OS can reuse its pid, and a child
    # orphaned by a parent that exited politely would look unrelated. These pids were provably the stage's own.
    tree = [pid] + descendants(pid)
    report["targets"] = tree
    if cancel_file:
        from pathlib import Path
        marker = Path(cancel_file)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("cancel\n", encoding="utf-8")
            report["cancel_file"] = str(marker)
        except OSError as exc:
            log(f"[proctree] could not write the cancel marker {marker}: {exc}")
    if grace > 0:
        still = wait_gone(tree, grace, is_alive=probe)
        if not still:
            # The whole tree ended on its own after the marker. No kill is sent.
            report["graceful"] = True
            return report
        pid_alive = probe(pid)
        orphans = [target for target in tree if target != pid and probe(target)]
        if not pid_alive:
            # The stage exited politely but something it started is still running (a trainer, a tool2 run, a ComfyUI
            # request). Those pids were collected while the root was alive, so ending them is still scoped to this
            # stage - and leaving them would be the exact defect N2 exists to fix.
            report["graceful"] = True
            for orphan in orphans:
                outcome = kill_tree(orphan, log=log, is_alive=probe)
                report["killed"] = True
                report["alive"] = sorted(set(report["alive"]) | set(outcome["alive"]))
            return report
    outcome = kill_tree(pid, log=log, is_alive=probe)
    report["killed"] = True
    report["targets"] = sorted(set(report["targets"]) | set(outcome["targets"]))
    report["alive"] = outcome["alive"]
    return report
