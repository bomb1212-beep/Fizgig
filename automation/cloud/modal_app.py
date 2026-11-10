"""The deployed Modal app for Fizgig cloud training.

Deployed with `modal deploy automation/cloud/modal_app.py` (see client.deploy - first deploy builds the image,
10-20 minutes). The image installs the repo's requirements.txt into /opt/venv with the same uv_install_deps.py the
official Dockerfile uses, so the container runtime matches a Fizgig Linux box. Only this file and spec.py are
deployment inputs; remote_exec.py is baked in and shared with the local side for testing.

Functions (CPU unless noted):
* version()        - the deployment version (sha256 of the baked files); the local side redeploys only on mismatch
* ensure_models()  - downloads the four Krea 2 training files from Comfy-Org/Krea-2 at the pinned revision into
                     the fizgig-models volume; re-downloads only files whose size differs; ~30 GB on first run
* train_run_<gpu>  - one per allowlisted GPU; cpu=4, memory=64 GiB, timeout=6 h; runs remote_exec.run(run_key)

No retries are configured: a call that dies is a call the user re-submits, with the same run_key and the same
checkpoints. Modal may re-run a preempted call; remote_exec is idempotent for a finished attempt.
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import spec  # noqa: E402  (the baked spec.py - both sides hash the same bytes)

import modal  # noqa: E402

_MODELS_VOLUME = modal.Volume.from_name(spec.VOLUME_MODELS, create_if_missing=True)
_RUNS_VOLUME = modal.Volume.from_name(spec.VOLUME_RUNS, create_if_missing=True)

_REPO_ROOT = _HERE.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "libgl1", "libglib2.0-0")
    .add_local_file(str(_REPO_ROOT / "requirements.txt"),
                    f"{spec.BUILD_INPUTS_DIR}/requirements.txt", copy=True)
    .add_local_file(str(_REPO_ROOT / "uv_install_deps.py"),
                    f"{spec.BUILD_INPUTS_DIR}/uv_install_deps.py", copy=True)
    .add_local_file(str(_HERE / "remote_exec.py"), "/opt/fizgig/remote_exec.py", copy=True)
    .add_local_file(str(_HERE / "spec.py"), "/opt/fizgig/spec.py", copy=True)
    .add_local_file(str(_HERE / "modal_app.py"), "/opt/fizgig/modal_app.py", copy=True)
    .run_commands(
        "python3 -m venv /opt/venv"
        " && /opt/venv/bin/pip install --upgrade pip uv"
        f" && DISABLE_CUDA=1 /opt/venv/bin/python {spec.BUILD_INPUTS_DIR}/uv_install_deps.py "
        f"{spec.BUILD_INPUTS_DIR}/requirements.txt /opt/venv",
        # torch.compile needs a compiler and Python headers for 3.12 (same reason the Dockerfile ships them).
        "test -x /usr/bin/gcc",
        "/opt/venv/bin/python -c \"import sys, sysconfig, os; "
        "p = sysconfig.get_path('include'); "
        "assert sys.version_info[:2] == (3, 12), sys.version; "
        "assert os.path.isfile(os.path.join(p, 'Python.h')), 'Python.h not found in ' + p\"",
        # The cu128 wheels for torch/torchvision must have been picked (Linux cu128, not the CPU build).
        "/opt/venv/bin/python -c \"import torch; assert torch.version.cuda, 'CPU-only torch was installed'; "
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)\"",
        "/opt/venv/bin/python -c \"import os, sysconfig; "
        "assert sysconfig.get_path('purelib').startswith('/opt/venv'), 'venv not on sys.path'; "
        "print('PYTHONPATH ok:', os.environ.get('PYTHONPATH'))\"",
    )
    .env({"PYTHONPATH": "/opt/fizgig",
          "VIRTUAL_ENV": "/opt/venv",
          "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
          # The trainer writes hf caches only if something resolves from the hub; models live on the volume.
          "HF_HOME": "/tmp/hf-cache",
          "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App(spec.APP_NAME, image=image)


def _model_path(entry):
    return os.path.join(spec.MODELS_ROOT, entry["path"])


@app.function(cpu=1, memory=512, timeout=60)
def version():
    """The deployment version: short sha256 of the files that decide what runs remotely."""
    mapping = {
        "modal_app.py": "/opt/fizgig/modal_app.py",
        "remote_exec.py": "/opt/fizgig/remote_exec.py",
        "spec.py": "/opt/fizgig/spec.py",
        "requirements.txt": f"{spec.BUILD_INPUTS_DIR}/requirements.txt",
        "uv_install_deps.py": f"{spec.BUILD_INPUTS_DIR}/uv_install_deps.py",
    }
    return spec.deployment_version(lambda name: Path(mapping[name]).read_bytes())


@app.function(cpu=2, memory=2048, timeout=spec.DOWNLOAD_TIMEOUT_SECONDS,
              volumes={spec.MODELS_ROOT: _MODELS_VOLUME})
def ensure_models():
    """Download the Krea 2 training files into the models volume (skipping exact matches). Returns a per-file report."""
    import shutil
    import time
    from huggingface_hub import hf_hub_download

    # A warmed container may have stale data; see what is actually on the volume now.
    _MODELS_VOLUME.reload()
    results = {}
    for entry in spec.MODEL_FILES:
        dest = _model_path(entry)
        if os.path.isfile(dest) and os.path.getsize(dest) == entry["bytes"]:
            results[entry["path"]] = "cached"
            continue
        if os.path.isfile(dest):
            os.unlink(dest)   # wrong size - re-download it
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        started = time.monotonic()
        hf_hub_download(repo_id=spec.HF_REPO, filename=entry["path"], revision=spec.HF_REVISION,
                        local_dir=spec.MODELS_ROOT)
        if not os.path.isfile(dest) or os.path.getsize(dest) != entry["bytes"]:
            actual = os.path.getsize(dest) if os.path.isfile(dest) else -1
            if os.path.isfile(dest):
                os.unlink(dest)
            raise RuntimeError(
                f"{entry['path']} downloaded as {actual} bytes, expected {entry['bytes']}")
        results[entry["path"]] = f"downloaded in {time.monotonic() - started:.0f}s"
    # Drop any internal cache the downloader may have left in the volume before committing.
    shutil.rmtree(os.path.join(spec.MODELS_ROOT, ".cache"), ignore_errors=True)
    _MODELS_VOLUME.commit()
    return results


def _train(run_key, runs_volume):
    import remote_exec
    runs_volume.reload()          # see the upload this call was spawned for
    _MODELS_VOLUME.reload()
    remote_exec.run(os.path.join(spec.RUNS_ROOT, run_key),
                    code_dir=spec.CODE_DIR,
                    commit=runs_volume.commit,
                    commit_interval=spec.COMMIT_INTERVAL_SECONDS)
    return True


_TRAIN_KWARGS = dict(cpu=spec.CPU_CORES, memory=spec.MEMORY_MIB, timeout=spec.TRAIN_TIMEOUT_SECONDS,
                     volumes={spec.MODELS_ROOT: _MODELS_VOLUME, spec.RUNS_ROOT: _RUNS_VOLUME})


@app.function(gpu="L40S", **_TRAIN_KWARGS)
def train_run_l40s(run_key: str):
    return _train(run_key, _RUNS_VOLUME)


@app.function(gpu="A100-80GB", **_TRAIN_KWARGS)
def train_run_a100_80gb(run_key: str):
    return _train(run_key, _RUNS_VOLUME)


@app.function(gpu="RTX-PRO-6000", **_TRAIN_KWARGS)
def train_run_rtx_pro_6000(run_key: str):
    return _train(run_key, _RUNS_VOLUME)


@app.function(gpu="H100", **_TRAIN_KWARGS)
def train_run_h100(run_key: str):
    return _train(run_key, _RUNS_VOLUME)
