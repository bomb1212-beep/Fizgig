"""A file-backed fake of the Modal SDK, for the end-to-end test.

Every bit of "Modal state" lives in one JSON file pointed at by FIZGIG_FAKE_MODAL_WORLD, so separate processes
(the runner under test, a later `runner.py sync`, the test itself) share the same world. It implements exactly the
surface `cloud.client.ModalClient` and `runner.py` use — no sleeps, no clock games: a call finishes after a fixed
number of polls (FIZGIG_FAKE_MODAL_POLLS_NEEDED, default 2), and the poll count is per call.
"""
import json
import os
from pathlib import Path

WORLD = Path(os.environ.get("FIZGIG_FAKE_MODAL_WORLD", "fake-modal-world.json"))


# The runner writes the world while the test (another process) reads it. On Windows a read during the swap, and the
# swap itself while a reader holds the file open, fail with PermissionError for a few milliseconds; wait those out.
_BUSY_RETRY_SECONDS = 10.0


def _fresh_world():
    return {"version": os.environ.get("FIZGIG_FAKE_MODAL_VERSION", "fake"),
            "volume": {}, "calls": {}, "next_call": 1}


def _load():
    import time
    deadline = time.monotonic() + _BUSY_RETRY_SECONDS
    while True:
        try:
            return json.loads(WORLD.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _fresh_world()
        except (PermissionError, ValueError):
            # Busy, or caught mid-swap: never mistake it for an empty world (the next store would wipe the state).
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _store(world):
    import time
    WORLD.parent.mkdir(parents=True, exist_ok=True)
    tmp = WORLD.with_suffix(".tmp")
    tmp.write_text(json.dumps(world, ensure_ascii=False), encoding="utf-8")
    deadline = time.monotonic() + _BUSY_RETRY_SECONDS
    while True:
        try:
            os.replace(tmp, WORLD)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


class _Error(Exception):
    pass


class NotFoundError(_Error):
    pass


class OutputExpiredError(_Error):
    pass


class FunctionTimeoutError(_Error):
    pass


class RemoteError(_Error):
    pass


class TimeoutError(_Error):
    pass


class ConnectionError(_Error):
    pass


class AuthError(_Error):
    pass


exception = type("exception", (), {
    "NotFoundError": NotFoundError, "OutputExpiredError": OutputExpiredError,
    "FunctionTimeoutError": FunctionTimeoutError, "RemoteError": RemoteError,
    "TimeoutError": TimeoutError, "ConnectionError": ConnectionError, "AuthError": AuthError,
})


def _next_call_id(world):
    call_id = f"fc-{world['next_call']}"
    world["next_call"] += 1
    return call_id


class App:
    def __init__(self, *args, **kwargs):
        self.name = args[0] if args else kwargs.get("name", "app")

    def function(self, *args, **kwargs):
        def wrap(fn):
            return fn
        return wrap


class _Image:
    def debian_slim(self, **kwargs):
        return self

    def apt_install(self, *args, **kwargs):
        return self

    def add_local_file(self, *args, **kwargs):
        return self

    def add_local_dir(self, *args, **kwargs):
        return self

    def run_commands(self, *args, **kwargs):
        return self

    def env(self, *args, **kwargs):
        return self


class Image:
    @staticmethod
    def debian_slim(**kwargs):
        return _Image()


def _save_file_bytes(path):
    data = Path(path).read_bytes()

    class _Chunks:
        def __iter__(self):
            yield data

    return _Chunks()


class _UploadBatch:
    def __init__(self, volume, force=False):
        self.volume = volume
        self.force = force

    def put_file(self, local_path, remote_path):
        world = _load()
        world["volume"][remote_path] = Path(local_path).read_bytes().decode("latin1")
        _store(world)

    def put_directory(self, local_path, remote_path):
        world = _load()
        prefix = remote_path.rstrip("/")
        for file in sorted(Path(local_path).rglob("*")):
            if file.is_file():
                rel = file.relative_to(local_path).as_posix()
                world["volume"][f"{prefix}/{rel}"] = file.read_bytes().decode("latin1")
        _store(world)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


FILE_ENTRY, DIRECTORY_ENTRY = 1, 2   # modal.types.FileEntryType.FILE / .DIRECTORY (an IntEnum)


class _VolumeEntry:
    """Like modal.types.FileEntry: `path` relative to the volume root, `type` FILE or DIRECTORY."""

    def __init__(self, path, type, size=0):
        self.path, self.type, self.mtime, self.size = path, type, 0, size

    def __repr__(self):
        return f"FileEntry(path={self.path!r}, type={self.type})"


def _missing(path):
    # The real Volume.read_file/remove_file raise the builtin FileNotFoundError with the server's message.
    return FileNotFoundError(f"File not found: {path}")


class Volume:
    @staticmethod
    def from_name(name, create_if_missing=False, **kwargs):
        return Volume()

    def commit(self):
        return None

    def reload(self):
        return None

    def batch_upload(self, force=False):
        return _UploadBatch(self, force=force)

    def read_file(self, path):
        world = _load()
        data = world["volume"].get(path)
        if data is None:      # a missing path, or a directory (only files can be read)
            raise _missing(path)

        class _Chunks:
            def __iter__(self):
                yield data.encode("latin1")

        return _Chunks()

    def iterdir(self, path, recursive=True):
        world = _load()
        prefix = path.rstrip("/")
        prefix_dir = prefix + "/"
        # The real recursive listing yields the directories under the path as well as the files.
        entries = {name: (FILE_ENTRY, len(data)) for name, data in world["volume"].items()}
        for name in list(entries):
            parts = name.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                entries.setdefault("/".join(parts[:i]), (DIRECTORY_ENTRY, 0))
        for name in sorted(entries):
            if name.startswith(prefix_dir) or (name == path and entries[name][0] == FILE_ENTRY):
                kind, size = entries[name]
                yield _VolumeEntry(name, kind, size)

    def listdir(self, path, recursive=False):
        return list(self.iterdir(path, recursive=recursive))

    def remove_file(self, path, recursive=False):
        world = _load()
        prefix_dir = path.rstrip("/") + "/"
        doomed = [name for name in list(world["volume"]) if name == path or name.startswith(prefix_dir)]
        if not doomed:
            raise _missing(path)
        for name in doomed:
            del world["volume"][name]
        _store(world)


class Function:
    def __init__(self, app_name, name):
        self.app_name, self.name = app_name, name

    @classmethod
    def from_name(cls, app_name, name):
        known = ("version", "ensure_models", "train_run_l40s", "train_run_a100_80gb",
                 "train_run_rtx_pro_6000", "train_run_h100")
        if app_name == "fizgig-train" and name in known:
            return cls(app_name, name)
        raise NotFoundError(f"NotFoundError: {app_name}.{name}")

    def remote(self):
        world = _load()
        if self.name == "version":
            return os.environ.get("FIZGIG_FAKE_MODAL_VERSION", world.get("version", "fake"))
        if self.name == "ensure_models":
            return {"cached": True}
        raise _Error(f"unknown remote call {self.name}")

    def spawn(self, run_key):
        world = _load()
        call_id = _next_call_id(world)
        world["calls"][call_id] = {"function": self.name, "run_key": run_key, "state": "running", "polls": 0}
        _store(world)
        return FunctionCall(call_id)


def _fabricate_outputs(world, run_key):
    """Write what a finished remote run would leave behind, so `fetch()` has something valid to pull."""
    try:
        plan = json.loads(world["volume"][f"{run_key}/plan.json"].encode("latin1").decode("utf-8-sig"))
    except Exception:
        return
    job_id = plan.get("job_id", "job")
    epochs = int(plan.get("epochs", 1))
    prefix = f"{run_key}/run/output"
    from safetensors.torch import save_file
    import torch
    scratch = Path(os.environ.get("FIZGIG_FAKE_MODAL_SCRATCH",
                                  str(Path(os.environ.get("FIZGIG_FAKE_MODAL_WORLD", "world.json")).with_suffix(".scratch"))))
    scratch.mkdir(parents=True, exist_ok=True)
    lora = scratch / f"{job_id}.safetensors"
    save_file({"lora_unet_blocks_0": torch.zeros(1)}, str(lora), metadata={"ss_epoch": str(epochs)})
    lora_bytes = lora.read_bytes().decode("latin1")
    world["volume"][f"{prefix}/{job_id}.safetensors"] = lora_bytes
    state_dir = f"{prefix}/{job_id}-{epochs:06d}-state"
    world["volume"][f"{state_dir}/training_state.json"] = json.dumps({"epoch": epochs})
    world["volume"][f"{state_dir}/lora.safetensors"] = lora_bytes
    world["volume"][f"{state_dir}/optimizer.pt"] = "optimizer-bytes"
    world["volume"][f"{state_dir}/rng.pt"] = "rng-bytes"
    world["volume"][f"{run_key}/run/Training.log"] = "loss 0.5\nloss 0.4\n"
    world["volume"][f"{run_key}/status.json"] = json.dumps(
        {"state": "done", "attempt_id": "att-e2e", "stage": "Training",
         "code_sha256": "fake", "dataset_fingerprint": plan.get("dataset_fingerprint", ""),
         "exit_code": 0, "message": "", "updated_at": "2026-11-10T00:00:00"})


class FunctionCall:
    def __init__(self, call_id):
        self._id = call_id

    @classmethod
    def from_id(cls, call_id):
        world = _load()
        if call_id not in world["calls"]:
            raise NotFoundError(f"NotFoundError: {call_id}")
        return cls(call_id)

    @property
    def object_id(self):
        return self._id

    def get(self, timeout=None, index=0):
        world = _load()
        call = world["calls"][self._id]
        call["polls"] += 1
        needed = int(os.environ.get("FIZGIG_FAKE_MODAL_POLLS_NEEDED", "2"))
        if call["polls"] >= needed:
            call["state"] = "done"
            _fabricate_outputs(world, call["run_key"])
            _store(world)
            return {"state": "done"}
        _store(world)
        raise TimeoutError()

    def cancel(self, terminate_containers=False):
        world = _load()
        if self._id in world["calls"]:
            world["calls"][self._id]["state"] = "cancelled"
        world["cancelled"] = world.get("cancelled", []) + [self._id]
        _store(world)
        return True
