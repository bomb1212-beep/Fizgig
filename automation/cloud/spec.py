"""Shared constants for the Modal cloud-training backend.

Both the deployed app (``modal_app.py``) and the local side (``modal_backend.py``, and the
runner's ``settings()`` validation) read this file. The values here are baked into the
deployment image and the function declarations, so changing any of them requires a redeploy;
they are deliberately NOT configurable through ``config.json`` (``cloud.modal`` in
``config.json`` only carries the polling interval and the cost-estimate assumptions).

Do not ``import modal`` here - this file must stay importable in environments without the
Modal SDK.
"""
import hashlib
from pathlib import Path

APP_NAME = "fizgig-train"
VOLUME_MODELS = "fizgig-models"
VOLUME_RUNS = "fizgig-runs"

# Remote paths inside the container / volumes.
MODELS_ROOT = "/models"
RUNS_ROOT = "/runs"
CODE_DIR = "/tmp/code"                     # the code tarball is extracted here
CODE_SRC = CODE_DIR + "/src"               # the PYTHONPATH the submitted source runs with
REMOTE_PYTHON = "/opt/venv/bin/python"
BUILD_INPUTS_DIR = "/opt/fizgig/build_inputs"   # requirements.txt + uv_install_deps.py inside the image

# GPU allowlist. The training function for each GPU is registered as TRAIN_GPU_FUNCTIONS[gpu].
GPUS = ("L40S", "A100-80GB", "RTX-PRO-6000", "H100")
DEFAULT_GPU = "L40S"
TRAIN_GPU_FUNCTIONS = {
    "L40S": "train_run_l40s",
    "A100-80GB": "train_run_a100_80gb",
    "RTX-PRO-6000": "train_run_rtx_pro_6000",
    "H100": "train_run_h100",
}

# Fixed resources for every training function (not configurable per job).
CPU_CORES = 4
MEMORY_GIB = 64
MEMORY_MIB = MEMORY_GIB * 1024
TRAIN_TIMEOUT_SECONDS = 21600        # 6 hours
DOWNLOAD_TIMEOUT_SECONDS = 14400     # 4 hours, for ensure_models() pulling ~30 GiB over HTTP
COMMIT_INTERVAL_SECONDS = 60         # how often the remote runner commits the runs volume

# Model files for Krea 2 training, from the public HuggingFace repo Comfy-Org/Krea-2, pinned to a
# fixed revision with exact byte sizes (verified against the HF API on 2026-11-10). The sizes are
# a hard contract: ensure_models() re-downloads only files whose size differs, and the remote
# runner checks the same before training starts. "약 30GB" is the download-size guidance only.
HF_REPO = "Comfy-Org/Krea-2"
HF_REVISION = "588feaf955b08b9ac4723eeb751752ab045dba71"
MODEL_FILES = (
    {"role": "dit", "path": "diffusion_models/krea2_raw_bf16.safetensors", "bytes": 26_283_332_608},
    {"role": "vae", "path": "vae/qwen_image_vae.safetensors", "bytes": 253_806_246},
    {"role": "text_encoder", "path": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "bytes": 5_242_467_968},
    {"role": "speed_lora", "path": "loras/krea2_turbo_lora_rank_64_bf16.safetensors", "bytes": 469_423_778},
)
MODEL_BYTES = sum(entry["bytes"] for entry in MODEL_FILES)   # 32,249,030,600 (~30 GiB)
MODEL_DOWNLOAD_GUIDE = "약 30GB"
# Roles the Krea 2 training command cannot run without (speed_lora only feeds the previews).
REQUIRED_MODEL_ROLES = ("dit", "vae", "text_encoder")

# The family the Modal backend supports.
SUPPORTED_FAMILY = "krea2"

# Deployment version: short sha256 over the files that decide what runs remotely. Both sides
# compute it from the same relative names - the local side from the repo, the deployed app from
# its baked image contents - so a mismatch means "redeploy".
SOURCE_FILES = ("modal_app.py", "remote_exec.py", "spec.py")
BUILD_INPUT_FILES = ("requirements.txt", "uv_install_deps.py")


def model_remote_path(entry):
    return MODELS_ROOT + "/" + entry["path"]


def role_model_map(desc):
    """{pref_key: remote /models path} for the family roles the remote plan must know."""
    return {desc.pref_for(entry["role"]): model_remote_path(entry) for entry in MODEL_FILES}


def remote_run_root(run_key):
    return f"{RUNS_ROOT}/{run_key}"


def remote_run_folder(run_key):
    return f"{RUNS_ROOT}/{run_key}/run"


def deployment_version(read_file):
    """`read_file(name) -> bytes`; pass local paths locally, container paths remotely."""
    digest = hashlib.sha256()
    for name in SOURCE_FILES + BUILD_INPUT_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(read_file(name))
    return digest.hexdigest()[:12]


def local_deployment_version():
    here = Path(__file__).resolve().parent
    root = here.parent.parent

    def read(name):
        if name in SOURCE_FILES:
            return (here / name).read_bytes()
        return (root / name).read_bytes()

    return deployment_version(read)


def estimate_inputs(cfg):
    """Pull the cloud.modal cost/polling assumptions out of a (merged) config, with defaults."""
    cloud = (cfg.get("cloud") or {}).get("modal") or {}
    defaults = {
        "poll_seconds": 30,
        "overhead_minutes": 15,
        "sec_per_step": {"L40S": 1.6, "A100-80GB": 1.3, "RTX-PRO-6000": 0.9, "H100": 0.8},
        "gpu_rate": {"L40S": 1.95, "A100-80GB": 2.50, "RTX-PRO-6000": 3.03, "H100": 3.95},
        "cpu_rate": 0.047,
        "memory_rate": 0.008,
    }
    merged = dict(defaults)
    for key, value in cloud.items():
        if key in ("sec_per_step", "gpu_rate") and isinstance(value, dict):
            merged[key] = {**defaults[key], **value}
        else:
            merged[key] = value
    return merged
