"""Check downloaded official tensor keys/shapes against Fizgig without allocating model weights."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "automation"))
from pipeline.models import models_dir, prefs_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(models_dir() / ".hf-cache"))

def shapes(folder):
    from safetensors import safe_open
    result = {}
    for path in sorted(folder.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                if key in result:
                    raise RuntimeError(f"Duplicate tensor {key}")
                result[key] = tuple(f.get_slice(key).get_shape())
    return result

def check(label, actual, model):
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    mismatch = [key for key in expected.keys() & actual.keys() if expected[key] != actual[key]]
    if missing or extra or mismatch:
        raise RuntimeError(f"{label}: missing={missing[:6]}, extra={extra[:6]}, shape mismatch={mismatch[:6]}")
    print(label, "OK:", len(expected), "tensor shapes match", flush=True)

def main():
    import torch
    from fizgig.qwen_image21.model import QwenImage21DiT, QWEN21_CONFIG
    from fizgig.qwen_image21.embedder import _text_encoder_config
    from fizgig.krea2.embedder import _convert_comfyui_qwen3vl_state_dict
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, AutoTokenizer
    official = models_dir() / "Qwen-Image-2.1"
    with torch.device("meta"):
        dit = QwenImage21DiT(**QWEN21_CONFIG)
        te = Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(_text_encoder_config()))
    check("Official Qwen DiT", shapes(official / "transformer"), dit)
    check("Official Qwen text encoder", _convert_comfyui_qwen3vl_state_dict(shapes(official / "text_encoder")), te)
    tokenizer = AutoTokenizer.from_pretrained(str(official / "processor"), local_files_only=True)
    print("Local tokenizer OK:", len(tokenizer("a photographic portrait").input_ids), "tokens")
    prefs = json.loads(prefs_path().read_text(encoding="utf-8-sig"))
    if not Path(prefs["qwen21_text_encoder"]).is_file():
        raise RuntimeError("Text encoder preference must point to its first shard file.")
    print("Model verification complete.")

if __name__ == "__main__":
    main()
