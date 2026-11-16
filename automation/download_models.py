"""Download Qwen's official checkpoint and the frozen Fizgig training adapter."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "automation"))
from pipeline.models import models_dir, prefs_path  # noqa: E402

MODELS = models_dir()
os.environ.setdefault("HF_HOME", str(MODELS / ".hf-cache"))
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def main():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from safetensors import safe_open
    official = MODELS / "Qwen-Image-2.1"
    info = HfApi().model_info("Qwen/Qwen-Image-2.1", files_metadata=True)
    print(f"Downloading Qwen/Qwen-Image-2.1 @ {info.sha}", flush=True)
    patterns = ["model_index.json", "transformer/*", "text_encoder/*", "vae/*", "processor/*", "scheduler/*", "README.md", "LICENSE*"]
    snapshot_download(info.id, revision=info.sha, local_dir=str(official), allow_patterns=patterns, max_workers=2)
    # The trainer resolves its tokenizer by repo id; warm that small cache too.
    snapshot_download(info.id, revision=info.sha, allow_patterns=["processor/*"], max_workers=2)
    adapter = hf_hub_download("ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter",
                              "fizgig_qwen_image_2.1_training_adapter.safetensors",
                              local_dir=str(MODELS / "fizgig_adapter"))
    weights = list(official.rglob("*.safetensors")) + [Path(adapter)]
    for weight in weights:
        with safe_open(str(weight), framework="pt") as handle:
            if not list(handle.keys()):
                raise RuntimeError(f"Empty checkpoint: {weight}")
    prefs_file = prefs_path()
    prefs = json.loads(prefs_file.read_text(encoding="utf-8-sig"))
    prefs.update(qwen21_dit=str(official / "transformer"),
                 qwen21_vae=str(official / "vae" / "diffusion_pytorch_model.safetensors"),
                 qwen21_text_encoder=str(official / "text_encoder" / "model-00001-of-00004.safetensors"),
                 qwen21_training_adapter=adapter)
    temp = prefs_file.with_suffix(".json.download.tmp")
    temp.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(prefs_file)
    (ROOT / "automation" / "download_manifest.json").write_text(
        json.dumps({"official_repo": info.id, "revision": info.sha,
                    "weights": [str(p) for p in weights]}, indent=2), encoding="utf-8")
    print("DOWNLOAD COMPLETE: weights verified, Qwen preferences saved.", flush=True)

if __name__ == "__main__":
    main()
