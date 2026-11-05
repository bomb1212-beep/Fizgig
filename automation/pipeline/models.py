"""Where a family's model files live, and what doctor says about them.

`automation/config.json` may carry a `models` map keyed first by family and then by the family's Preferences key:

    "models": {"krea2": {"krea2_raw_dit": "G:/Lakis/ComfyUI/models/diffusion_models/krea2_raw_bf16.safetensors"}}

A path given there wins; every key it leaves out falls back to `prefs.json`, the file the Fizgig GUI owns. This module
only ever reads `prefs.json`: a path typed into the GUI and a path in `config.json` are the same thing to the trainer.

Relative paths are resolved against the Fizgig_auto root so the local Qwen files do not depend on how the folder is
mounted. ComfyUI's own model files stay absolute: pointing at them is the whole reason 26 GB is not copied.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # Fizgig_auto
AUTO = ROOT / "automation"
DEFAULT_FAMILY = "qwen_image21"


def read_json(path, default=None):
    """The parsed file, or `default` when it is missing or unreadable (a broken prefs.json must not crash doctor)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {} if default is None else default


def config():
    return read_json(AUTO / "config.json", {})


def prefs():
    return read_json(ROOT / "prefs.json", {})


def description(family):
    """The family's FamilyDescription. Imported lazily so this module stays importable without torch."""
    from fizgig.families.registry import get
    desc = get(family)
    if desc is None:
        known = ", ".join(sorted(_registry_keys()))
        raise ValueError(f"unknown family {family!r} (known: {known})")
    return desc


def _registry_keys():
    from fizgig.families.registry import FAMILIES
    return FAMILIES


def families():
    """Every family the trainer knows, in registry order."""
    return list(_registry_keys())


def overrides(family, cfg=None):
    """Only the paths this config.json pins for the family, already absolute."""
    block = ((cfg if cfg is not None else config()).get("models") or {}).get(family) or {}
    return {key: resolve_path(value) for key, value in block.items() if isinstance(value, str) and value.strip()}


def resolve_path(value):
    """Absolute form of a configured path: relative paths hang off the Fizgig_auto root, `~` is expanded."""
    path = Path(os.path.expanduser(str(value).strip()))
    return str(path if path.is_absolute() else (ROOT / path))


def resolve(family, cfg=None, base=None):
    """{pref_key: absolute path} for every model file the family declares.

    `base` replaces prefs.json (the tests use it); a key found in neither place stays "" so `plan()` can report it
    with the family's own wording instead of a KeyError.
    """
    desc = description(family)
    source = prefs() if base is None else dict(base)
    merged = {key: value for key, value in source.items() if isinstance(value, str) and value.strip()}
    merged = {key: resolve_path(value) for key, value in merged.items()}
    merged.update(overrides(family, cfg))
    return {f.pref_key: merged.get(f.pref_key, "") for f in desc.model_files}


def rows(family, cfg=None, base=None):
    """One row per declared model file, for doctor: what it is, where it points, and whether that exists."""
    desc = description(family)
    pinned = overrides(family, cfg)
    paths = resolve(family, cfg, base)
    out = []
    for f in desc.model_files:
        path = paths[f.pref_key]
        out.append({
            "family": family,
            "pref_key": f.pref_key,
            "label": f.label,
            "role": f.role,
            "required": bool(f.required),
            "source": "config.json" if f.pref_key in pinned else "prefs.json",
            "path": path,
            "exists": bool(path) and Path(path).exists(),
        })
    return out


def problems(family, cfg=None, base=None, require_optional=()):
    """Reasons training cannot start, in the trainer's wording, plus any optional key the caller needs.

    `require_optional` names pref keys that this run needs even though training alone would not (the experiment's
    preview DiT, for instance).
    """
    missing = []
    wanted = set(require_optional)
    for row in rows(family, cfg, base):
        if not row["required"] and row["pref_key"] not in wanted:
            continue
        if not row["path"]:
            missing.append(f"{row['label']} path is empty (set models.{family}.{row['pref_key']} "
                           f"in automation/config.json or {row['pref_key']} in prefs.json)")
        elif not row["exists"]:
            missing.append(f"{row['label']} file does not exist: {row['path']}")
    return missing


def comfy_settings(cfg=None):
    """The `comfy` block with defaults filled in: the address, ComfyUI's own root, and how it is started."""
    block = dict((cfg if cfg is not None else config()).get("comfy") or {})
    root = resolve_path(block.get("root") or "G:/Lakis/ComfyUI")
    block["root"] = root
    block.setdefault("url", "http://127.0.0.1:8190")
    block.setdefault("workdir", str(Path(root).parent))
    block.setdefault("python", str(Path(root).parent / "python_embeded" / "python.exe"))
    block.setdefault("args", ["-s", "ComfyUI/main.py", "--windows-standalone-build",
                              "--port", str(_port(block["url"])), "--disable-auto-launch"])
    block.setdefault("request_timeout", 1800)
    return block


def _port(url):
    from urllib.parse import urlparse
    return urlparse(url).port or 8188


def model_dir(comfy_root, kind):
    """ComfyUI's model folders a filename category is read from (text encoders also live under the legacy `clip`)."""
    root = Path(comfy_root) / "models"
    return {
        "unet": [root / "diffusion_models"],
        "lora": [root / "loras"],
        "vae": [root / "vae"],
        "clip": [root / "text_encoders", root / "clip"],
    }[kind]
