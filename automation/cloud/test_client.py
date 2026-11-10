"""SDK-boundary tests for client.py: a fake Modal stands in for every call; nothing touches the network."""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # automation/

from cloud import client, spec  # noqa: E402


class TimeoutError(Exception):
    pass


class FunctionTimeoutError(Exception):
    pass


class OutputExpiredError(Exception):
    pass


class RemoteError(Exception):
    pass


class ConnectionError(Exception):
    pass


class NotFoundError(Exception):
    pass


FILE_ENTRY, DIRECTORY_ENTRY = 1, 2      # modal.types.FileEntryType.FILE / .DIRECTORY (an IntEnum)


class FakeFunction:
    def __init__(self, modal, app_name, name):
        self.modal, self.app_name, self.name = modal, app_name, name

    def remote(self):
        if self.name == "version":
            return self.modal.version
        if self.name == "ensure_models":
            return self.modal.models_report
        raise AssertionError(f"unexpected remote call {self.name}")

    def spawn(self, run_key):
        self.modal.spawned.append((self.name, run_key))
        return FakeCall(self.modal, self.modal.next_call_id)


class FakeCall:
    def __init__(self, modal, object_id):
        self.modal, self.object_id = modal, object_id

    def get(self, timeout=None):
        behavior = self.modal.poll_behavior.get(self.object_id, "done")
        if behavior == "done":
            return {"state": "done"}
        if behavior == "running":
            raise TimeoutError()
        if behavior == "pending":
            raise TimeoutError()
        if behavior == "expired":
            raise OutputExpiredError()
        if behavior == "failed":
            raise FunctionTimeoutError()
        if behavior == "remote_error":
            raise RemoteError("training exploded")
        if behavior == "cancelled":
            raise RemoteError("input was cancelled")
        if behavior == "unreachable":
            raise ConnectionError("connection reset")
        raise AssertionError(behavior)

    def cancel(self, terminate_containers=False):
        if self.modal.cancel_raises is not None:
            raise self.modal.cancel_raises
        self.modal.cancelled.append((self.object_id, terminate_containers))


class FakeVolume:
    def __init__(self, modal, name):
        self.modal, self.name = modal, name

    @property
    def files(self):
        return self.modal.volume_files.setdefault(self.name, {})

    def batch_upload(self, force=False):
        modal_self = self.modal

        class Batch:
            def put_file(self, local, remote):
                modal_self.uploaded.append((str(local), remote))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Batch()

    # Like the real SDK: a missing path (or a directory) is the *builtin* FileNotFoundError carrying the
    # server's message, and a recursive listing yields directory entries next to the files.
    def _missing(self, path):
        return FileNotFoundError(self.modal.missing_message.format(path=path))

    def _directories(self):
        dirs = set()
        for name in self.files:
            parts = name.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:i]))
        return dirs

    def read_file(self, path):
        data = self.files.get(path)
        if data is None:
            raise self._missing(path)
        self.modal.read_paths.append(path)
        return iter([data])

    def iterdir(self, path, recursive=True):
        prefix = path.rstrip("/") + "/"
        entries = [(name, FILE_ENTRY) for name in self.files]
        entries += [(name, DIRECTORY_ENTRY) for name in self._directories()]
        for name, kind in sorted(entries):
            if name == path or name.startswith(prefix):
                yield SimpleNamespace(path=name, type=kind, mtime=0, size=0)

    def remove_file(self, path, recursive=False):
        files = self.files
        removed = [name for name in list(files) if name == path or name.startswith(path.rstrip("/") + "/")]
        if not removed:
            raise self._missing(path)
        for name in removed:
            del files[name]
        self.modal.deleted.append(path)


class FakeModal:
    def __init__(self, **kwargs):
        self.version = kwargs.get("version", "abc123def456")
        self.models_report = kwargs.get("models_report", {"diffusion_models/krea2_raw_bf16.safetensors": "cached"})
        self.function_missing = kwargs.get("function_missing", False)
        self.next_call_id = kwargs.get("next_call_id", "fc-fake-1")
        self.spawned = []
        self.cancelled = []
        self.uploaded = []
        self.poll_behavior = kwargs.get("poll_behavior", {})
        self.volume_files = kwargs.get("volume_files", {})
        self.missing_message = kwargs.get("missing_message", "File not found: {path}")
        self.read_paths = []
        self.deleted = []
        self.cancel_raises = kwargs.get("cancel_raises")

        modal_self = self

        class Function:
            @staticmethod
            def from_name(app_name, name):
                if modal_self.function_missing:
                    raise NotFoundError(f"NotFoundError: {app_name}.{name}")
                if name not in ("version", "ensure_models") and not name.startswith("train_run_"):
                    raise NotFoundError(f"NotFoundError: {name}")
                return FakeFunction(modal_self, app_name, name)

        class FunctionCall:
            @staticmethod
            def from_id(call_id):
                if modal_self.poll_behavior.get(call_id) == "gone":
                    raise NotFoundError(f"NotFoundError: {call_id}")
                return FakeCall(modal_self, call_id)

        class Volume:
            @staticmethod
            def from_name(name, create_if_missing=False):
                return FakeVolume(modal_self, name)

        self.Function = Function
        self.FunctionCall = FunctionCall
        self.Volume = Volume
        self.exception = SimpleNamespace(
            OutputExpiredError=OutputExpiredError,
            TimeoutError=TimeoutError,
            FunctionTimeoutError=FunctionTimeoutError,
            RemoteError=RemoteError,
            NotFoundError=NotFoundError,
            ConnectionError=ConnectionError,
        )


def fake_client(modal, **kwargs):
    c = client.ModalClient(**kwargs)
    c._modal = modal
    return c


class ClientTests(unittest.TestCase):
    def test_deployed_version_roundtrip(self):
        c = fake_client(FakeModal(version="deadbeef1234"))
        self.assertEqual(c.deployed_version(), "deadbeef1234")

    def test_deployed_version_none_when_app_missing(self):
        c = fake_client(FakeModal(function_missing=True))
        self.assertIsNone(c.deployed_version())

    def test_deployed_version_raises_on_genuine_failures(self):
        class Broken(FakeModal):
            pass

        modal = FakeModal()

        def boom(app_name, name):
            raise ConnectionError("ws boom")

        modal.Function.from_name = staticmethod(boom)
        c = fake_client(modal)
        with self.assertRaises(ConnectionError):
            c.deployed_version()

    def test_ensure_models_returns_report(self):
        report = {"loras/krea2_turbo_lora_rank_64_bf16.safetensors": "cached"}
        c = fake_client(FakeModal(models_report=report))
        self.assertEqual(c.ensure_models(), report)

    def test_upload_puts_every_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir()
            (root / "images" / "a.png").write_bytes(b"png")
            (root / "dataset.toml").write_text("[general]\n", encoding="utf-8")
            (root / "output").mkdir()
            (root / "output" / "pin.bin").write_bytes(b"pin")
            modal = FakeModal()
            c = fake_client(modal)
            count = c.upload(str(root), "/runs/key-1")
            self.assertEqual(count, 3)
            paths = [remote for _local, remote in modal.uploaded]
            self.assertIn("key-1/images/a.png", paths)
            self.assertIn("key-1/dataset.toml", paths)
            self.assertIn("key-1/output/pin.bin", paths)

    def test_spawn_returns_object_id_and_validates_gpu(self):
        modal = FakeModal(next_call_id="fc-online-9")
        c = fake_client(modal)
        self.assertEqual(c.spawn("L40S", "job-abc"), "fc-online-9")
        self.assertEqual(modal.spawned, [("train_run_l40s", "job-abc")])
        with self.assertRaises(ValueError):
            c.spawn("A100-40GB", "job-abc")

    def test_poll_maps_everything(self):
        cases = {"done": "done", "running": "running", "pending": "running",
                 "expired": "expired", "failed": "failed", "remote_error": "failed",
                 "cancelled": "cancelled", "unreachable": "unknown", "gone": "unknown"}
        for behavior, expected in cases.items():
            with self.subTest(behavior=behavior):
                modal = FakeModal(poll_behavior={"fc-1": behavior})
                c = fake_client(modal, log=lambda *a: None)
                self.assertEqual(c.poll("fc-1"), expected)

    def test_cancel_terminates_containers(self):
        modal = FakeModal()
        c = fake_client(modal)
        self.assertTrue(c.cancel("fc-1"))
        self.assertEqual(modal.cancelled, [("fc-1", True)])

    def test_cancel_missing_call_is_false(self):
        modal = FakeModal(cancel_raises=NotFoundError("NotFoundError"))
        c = fake_client(modal)
        self.assertFalse(c.cancel("fc-gone"))

    def test_read_text_returns_none_for_missing(self):
        modal = FakeModal(volume_files={"fizgig-runs": {"key-1/status.json": b"{}"}})
        c = fake_client(modal)
        self.assertEqual(c.read_text("/runs/key-1/status.json"), "{}")
        self.assertIsNone(c.read_text("/runs/key-1/missing.json"))

    def test_download_writes_and_can_exclude(self):
        modal = FakeModal(volume_files={"fizgig-runs": {
            "key-1/run/output/job.safetensors": b"lora",
            "key-1/run/output/cache/inner.safetensors": b"cache",
        }})
        c = fake_client(modal)
        with tempfile.TemporaryDirectory() as td:
            count = c.download("/runs/key-1/run/output", td)
            self.assertEqual(count, 2)
            self.assertEqual((Path(td) / "job.safetensors").read_bytes(), b"lora")
            # Excluded files are skipped.
            modal2 = FakeModal(volume_files={"fizgig-runs": {
                "key-1/run/output/job.safetensors": b"lora",
                "key-1/run/cache/inner.safetensors": b"cache",
            }})
            c2 = fake_client(modal2)
            with tempfile.TemporaryDirectory() as td2:
                count = c2.download("/runs/key-1/run/output", td2,
                                    exclude=lambda p: "cache" in p)
                self.assertEqual(count, 1)
                self.assertFalse((Path(td2) / "cache").exists())

    def test_read_text_is_none_whatever_the_missing_file_message_says(self):
        for message in ("File not found: {path}", "No such file or directory: {path}"):
            c = fake_client(FakeModal(missing_message=message))
            self.assertIsNone(c.read_text("/runs/key-1/status.json"), message)
            self.assertIsNone(c.read_bytes("/runs/key-1/run/Training.log"), message)

    def test_download_of_a_nested_output_skips_directory_entries(self):
        files = {"key-1/run/output/job.safetensors": b"lora"}
        for name in ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt"):
            files[f"key-1/run/output/job-000003-state/{name}"] = name.encode()
        modal = FakeModal(volume_files={"fizgig-runs": files})
        c = fake_client(modal)
        with tempfile.TemporaryDirectory() as td:
            count = c.download("/runs/key-1/run/output", td)
            self.assertEqual(count, 5)
            self.assertEqual((Path(td) / "job.safetensors").read_bytes(), b"lora")
            for name in ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt"):
                self.assertEqual((Path(td) / "job-000003-state" / name).read_bytes(), name.encode())
        self.assertEqual(sorted(modal.read_paths), sorted(files))   # only files were read, never a folder

    def test_delete_of_a_missing_folder_is_false_whatever_the_message(self):
        c = fake_client(FakeModal(missing_message="No such file or directory: {path}"))
        self.assertFalse(c.delete("/runs/key-gone"))

    def test_delete_is_recursive(self):
        modal = FakeModal(volume_files={"fizgig-runs": {"key-1/plan.json": b"{}", "key-1/run/output/o.bin": b"x"}})
        c = fake_client(modal)
        self.assertTrue(c.delete("/runs/key-1"))
        self.assertEqual(modal.deleted, ["key-1"])
        self.assertFalse(c.delete("/runs/key-1"))   # already gone

    def test_missing_modal_package_prints_install_command(self):
        saved = sys.modules.get("modal")
        sys.modules["modal"] = None
        try:
            c = client.ModalClient()
            with self.assertRaises(RuntimeError) as ctx:
                c._sdk()
            self.assertIn("automation\\cloud\\requirements.txt", str(ctx.exception))
        finally:
            if saved is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = saved


if __name__ == "__main__":
    unittest.main()
