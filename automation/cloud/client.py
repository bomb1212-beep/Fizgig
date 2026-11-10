"""The ONLY local entry point to the Modal SDK.

Everything the local side needs - deploy, the two volumes, function calls, poll/cancel - goes through this
adapter; the rest of the code (runner, `cloud.modal_backend`, tests) never does `import modal`. Call it with a
*fake* standing in for the SDK in tests.

Environment: the standard Modal token is used as-is (`%USERPROFILE%\\.modal.toml`, or MODAL_TOKEN_ID /
MODAL_TOKEN_SECRET). The tool never creates, prints or stores tokens; `deploy()` uses the very Python that
imports this file, with the same environment and profile, so the `modal` CLI and the SDK always agree.

`poll()` maps a call to one word:

* `done`      - the call finished successfully (result is available for 7 days)
* `running`   - still pending (Modal polls with `FunctionCall.get(timeout=0)`)
* `failed`    - the remote run died (function exception, or the 6 h limit: FunctionTimeoutError)
* `cancelled` - the call was cancelled (via `cancel()`)
* `expired`   - the result window (7 days) is gone; inspect the attempt's Volume status before judging
* `unknown`   - Modal no longer knows the call id, or the communication failed; never auto-resubmit

`read_text`/`download`/`delete` operate on the *Volume* filesystem (e.g. `/runs/<key>/status.json`), translating
container-absolute paths to the volume's own relative layout.
"""
import os
import posixpath
import subprocess
import sys
from pathlib import Path

RUN_LOG_TAIL = 200


def _exc_name(exc):
    return type(exc).__name__


def _is_not_found(exc):
    # Volume.read_file/remove_file turn the server's NotFoundError into the builtin FileNotFoundError.
    return (isinstance(exc, FileNotFoundError) or _exc_name(exc) == "NotFoundError"
            or "not found" in str(exc).lower())


_FILE_ENTRY = 1   # modal.types.FileEntryType.FILE (an IntEnum); compared by value so this module never imports modal


def _is_file_entry(entry):
    """`Volume.iterdir(recursive=True)` lists directories too; only files can be read. No `type` = a file."""
    kind = getattr(entry, "type", None)
    if kind is None:
        return True
    try:
        return int(kind) == _FILE_ENTRY
    except (TypeError, ValueError):
        return False


class ModalClient:
    """The adapter the backend talks to. Tests substitute a fake with the same methods."""

    APP_NAME = None   # bound from spec at first use so importing this module is cheap

    def __init__(self, *, app_name=None, log=print):
        from cloud import spec
        self.spec = spec
        self.app_name = app_name or spec.APP_NAME
        self.log = log
        self._modal = None   # lazy: importing this module never requires the SDK

    # ------------------------------------------------------------------ SDK plumbing
    def _sdk(self):
        if self._modal is None:
            try:
                import modal
            except ImportError as exc:
                raise RuntimeError(
                    "the `modal` package is not installed. Install it with:\n"
                    "  .\\venv\\Scripts\\python.exe -m pip install -r automation\\cloud\\requirements.txt\n"
                    "then create a token with: .\\venv\\Scripts\\python.exe -m modal token new") from exc
            self._modal = modal
        return self._modal

    def _function(self, name):
        modal = self._sdk()
        return modal.Function.from_name(self.app_name, name)

    def _volume(self, which):
        modal = self._sdk()
        return modal.Volume.from_name(which, create_if_missing=(which == self.spec.VOLUME_RUNS))

    def _volume_path(self, absolute):
        """Container-absolute (`/runs/<key>/...`) -> the path inside the named volume (`<key>/...`)."""
        for prefix in (self.spec.RUNS_ROOT, self.spec.MODELS_ROOT):
            if absolute == prefix:
                return ""
            if absolute.startswith(prefix + "/"):
                return absolute[len(prefix) + 1:]
        return absolute.lstrip("/")

    # ------------------------------------------------------------------ deployment
    def deployed_version(self):
        """The deployed app's version string, or None when the app is not deployed (or unreachable as 'missing')."""
        try:
            return self._function("version").remote()
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise

    def deploy(self):
        """Run `modal deploy automation/cloud/modal_app.py` with this very Python; raise on failure."""
        root = Path(__file__).resolve().parents[2]
        app_path = root / "automation" / "cloud" / "modal_app.py"
        self.log("deploying fizgig-train (the first build can take 10-20 minutes)...")
        cmd = [sys.executable, "-m", "modal", "deploy", str(app_path)]
        result = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-RUN_LOG_TAIL:]
            raise RuntimeError(f"modal deploy failed ({result.returncode}): {tail}")
        return True

    def ensure_models(self):
        """Wait for the models volume to hold the four Krea 2 files; returns the per-file report."""
        return self._function("ensure_models").remote()

    # ------------------------------------------------------------------ inputs
    def upload(self, local_dir, remote_dir):
        """Upload a folder (files only) into the runs volume under remote_dir. Returns the file count."""
        volume = self._volume(self.spec.VOLUME_RUNS)
        prefix = self._volume_path(remote_dir).rstrip("/")
        files = [p for p in sorted(Path(local_dir).rglob("*")) if p.is_file()]
        with volume.batch_upload(force=True) as batch:
            for file in files:
                rel = file.relative_to(local_dir).as_posix()
                batch.put_file(str(file), posixpath.join(prefix, rel))
        return len(files)

    def spawn(self, gpu, run_key):
        """Spawn the GPU training function detached; returns the call id. Raises ValueError for an unknown GPU."""
        try:
            function_name = self.spec.TRAIN_GPU_FUNCTIONS[gpu]
        except KeyError:
            raise ValueError(f"unknown modal_gpu {gpu!r}; known: {', '.join(self.spec.GPUS)}") from None
        call = self._function(function_name).spawn(run_key)
        return call.object_id

    # ------------------------------------------------------------------ tracking
    def poll(self, call_id, *, log_communication_errors=True):
        try:
            call = self._sdk().FunctionCall.from_id(call_id)
        except Exception:
            return "unknown"
        try:
            call.get(timeout=0)
            return "done"
        except Exception as exc:
            name = _exc_name(exc)
            message = str(exc).lower()
            if name == "OutputExpiredError":
                return "expired"
            if "cancel" in name.lower() or "cancel" in message:
                return "cancelled"
            if name in ("FunctionTimeoutError", "ExecTimeoutError", "InteractiveTimeoutError",
                        "RemoteError", "ImageBuildError", "ExecutionError", "SandboxTimeoutError",
                        "SandboxTerminatedError", "InternalFailure"):
                return "failed"
            if name == "TimeoutError":     # Modal's pending sentinel (and/or the builtin one)
                return "running"
            if log_communication_errors:
                self.log(f"modal poll failed: {exc!r}")
            return "unknown"

    def cancel(self, call_id):
        """Ask Modal to cancel the call (and terminate its container). False when the call is already gone."""
        try:
            call = self._sdk().FunctionCall.from_id(call_id)
            call.cancel(terminate_containers=True)
            return True
        except Exception as exc:
            if _is_not_found(exc):
                return False
            raise

    # ------------------------------------------------------------------ outputs
    def read_bytes(self, remote_path):
        path = self._volume_path(remote_path)
        try:
            chunks = self._volume(self.spec.VOLUME_RUNS).read_file(path)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        return b"".join(chunks)

    def read_text(self, remote_path):
        data = self.read_bytes(remote_path)
        return data.decode("utf-8", errors="replace") if data is not None else None

    def download(self, remote_dir, local_dir, exclude=None):
        """Download a remote folder. `exclude` is an optional predicate on the local target path."""
        volume = self._volume(self.spec.VOLUME_RUNS)
        prefix = self._volume_path(remote_dir).rstrip("/")
        count = 0
        for entry in volume.iterdir(prefix, recursive=True):
            if not _is_file_entry(entry):
                continue
            rel = getattr(entry, "path", None) or str(entry)
            rel = str(rel).replace("\\", "/").lstrip("/")
            if rel.endswith("/"):
                continue
            target = os.path.join(local_dir, os.path.relpath(rel, prefix) if rel.startswith(prefix) else os.path.basename(rel))
            if exclude and exclude(target):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as handle:
                for chunk in volume.read_file(rel):
                    handle.write(chunk)
            count += 1
        return count

    def delete(self, remote_dir):
        path = self._volume_path(remote_dir)
        try:
            self._volume(self.spec.VOLUME_RUNS).remove_file(path, recursive=True)
            return True
        except Exception as exc:
            if _is_not_found(exc):
                return False
            raise
