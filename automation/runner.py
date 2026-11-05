"""Local serial runner: prepared copies -> local Gemma captions -> Fizgig caches -> training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "automation"
os.environ.setdefault("HF_HOME", str(ROOT / "models" / ".hf-cache"))
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AUTO))
from local_caption import GemmaServer, LocalClient
from pipeline import models as model_paths
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EVENT_PREFIX = "@@FIZGIG_EVENT@@"
EVENTS = False
DEFAULT_FAMILY = "qwen_image21"
# Settings that describe the pipeline plumbing rather than the training run: which files the runner resolves them to
# is checked on disk before anything loads, and hashing them would move every existing run to a new version directory
# the moment the block is added. The family is hashed only when it is not the default, for the same reason.
PIPELINE_SETTINGS = ("comfy", "models", "pipeline", "pipeline_model_paths")

def emit_event(kind, **values):
    if EVENTS:
        print(EVENT_PREFIX + json.dumps({"kind": kind, **values}, ensure_ascii=False), flush=True)

def check_pause():
    if (AUTO / "stop_requested").exists():
        raise RuntimeError("Automation paused. Resume from the GUI or remove automation/stop_requested.")

def queue_owner_running():
    """Read-only liveness check; never signal an existing Windows training process."""
    try:
        pid = int((AUTO / "queue.lock").read_text(encoding="utf-8").strip())
        if pid <= 0:
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
                return ctypes.get_last_error() == 5  # access denied is not proof the owner died
            try:
                code = wintypes.DWORD()
                return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
            finally:
                kernel.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def settings(job):
    cfg = read_json(AUTO / "config.json")
    cfg.update(job)
    required = ("id", "dataset_root", "target_steps", "max_images_per_job", "rank", "save_every_steps_approx",
                "caption_base_url", "caption_model", "caption_max_output_tokens")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError("Missing settings in automation/config.json: " + ", ".join(missing))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", cfg["id"]):
        raise ValueError("Job id must be 1-80 letters/numbers, underscores or hyphens.")
    if not cfg.get("trigger", "").strip():
        raise ValueError("A trigger is required.")
    for key in ("target_steps", "max_images_per_job", "rank", "save_every_steps_approx"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    LocalClient(cfg)  # Validate the local destination before copying or reading image content.
    return cfg

def inventory(cfg):
    from PIL import Image
    folder = Path(cfg["dataset_root"])
    if not folder.is_dir():
        raise ValueError(f"Dataset folder missing: {folder}")
    valid, rejected, seen = [], [], set()
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha in seen:
                rejected.append({"path": str(path), "reason": "exact duplicate"})
                continue
            seen.add(sha)
            sidecar = path.with_suffix(".txt")
            cap = sidecar.read_text(encoding="utf-8-sig").strip() if sidecar.is_file() else ""
            valid.append({"source": str(path), "sha256": sha, "caption": cap})
        except (OSError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": type(exc).__name__})
    if not valid:
        raise ValueError("No readable images in the dataset's top level.")
    if len(valid) > cfg["max_images_per_job"]:
        raise ValueError(f"{len(valid)} images exceed max_images_per_job; change config to allow them.")
    return valid, rejected

def family_of(cfg):
    """The training family this run uses: `family` in the job or config.json, qwen_image21 when neither says."""
    family = str(cfg.get("family") or DEFAULT_FAMILY).strip()
    if family not in model_paths.families():
        raise ValueError(f"unknown family {family!r}; known: " + ", ".join(model_paths.families()))
    return family

def version_settings(cfg):
    """The part of the settings that decides the run version.

    The pipeline blocks are plumbing (where ComfyUI is, where the models are); a run does not become a different run
    because they were added, and existing version directories must keep resuming. A non-default family does decide it,
    so Krea 2 and Qwen 2.1 runs never share caches or a saved state.
    """
    out = {k: v for k, v in cfg.items() if k not in PIPELINE_SETTINGS and k != "family"}
    family = family_of(cfg)
    if family != DEFAULT_FAMILY:
        out["family"] = family
    return out

def context(cfg):
    images, rejected = inventory(cfg)
    instructions = (AUTO / "caption_instructions.txt").read_text(encoding="utf-8-sig")
    version = digest({"settings": version_settings(cfg), "images": images, "instructions": instructions})[:16]
    run = AUTO / "runs" / (cfg["id"] + "-" + version)
    return run, images, rejected, instructions

def caption_identity(cfg, instructions):
    transport = {"caption_base_url", "caption_auto_start", "caption_startup_timeout", "caption_request_timeout",
                 "caption_launch_mode", "caption_server_exe"}
    return digest({"caption": {k: v for k, v in cfg.items() if k.startswith("caption_") and k not in transport},
                   "trigger": cfg["trigger"], "instructions": instructions})

def save_run_config(run, cfg, instructions):
    write_json(Path(run) / "run_config.json", {"settings": cfg, "caption_identity": caption_identity(cfg, instructions)})

def caption_sources(cfg, instructions, exclude=None):
    identity = caption_identity(cfg, instructions)
    sources = []
    for run in sorted((AUTO / "runs").glob(cfg["id"] + "-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if run == exclude or not run.is_dir():
            continue
        try:
            saved = read_json(run / "run_config.json")
            manifest = read_json(run / "dataset_manifest.json")
            if saved.get("caption_identity") == identity:
                sources.append((run, {i["sha256"]: i for i in manifest["images"]}))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sources

def find_caption(item, sources):
    for run, images in sources:
        previous = images.get(item["sha256"])
        if previous is None or previous.get("caption", "") != item.get("caption", ""):
            continue
        image = run / "images" / (item["sha256"] + Path(previous["source"]).suffix.lower())
        try:
            if hashlib.sha256(image.read_bytes()).hexdigest() != item["sha256"]:
                continue
            text = image.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
            if text:
                return text
        except (OSError, ValueError):
            continue
    return None

def working_fingerprint(run):
    folder = Path(run) / "images"
    return digest([(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(folder.iterdir())
                   if p.is_file() and (p.suffix in EXTENSIONS or p.suffix == ".txt")])

def caches_valid(run):
    from safetensors import safe_open
    from fizgig.families.launch import cache_dir_for
    folder = Path(cache_dir_for(str(run / "cache"), str(run / "images")))
    try:
        for image in (run / "images").iterdir():
            if image.suffix not in EXTENSIONS:
                continue
            latents = list(folder.glob(image.stem + "_*_qwenimage21.safetensors"))
            if len(latents) != 1:
                return False
            with safe_open(str(latents[0]), framework="pt") as f:
                if not any(k.startswith("latent_") for k in f.keys()):
                    return False
            with safe_open(str(folder / (image.stem + "_qwenimage21_te.safetensors")), framework="pt") as f:
                if not any(k.startswith("cond__") for k in f.keys()):
                    return False
                if (f.metadata() or {}).get("caption1") != image.with_suffix(".txt").read_text(encoding="utf-8-sig").strip():
                    return False
        return True
    except Exception:
        # Safetensors raises its own exception type for a damaged header.
        return False

def output_valid(run, cfg, epochs):
    from safetensors import safe_open
    output = run / "output"
    try:
        with safe_open(str(output / (cfg["id"] + ".safetensors")), framework="pt") as f:
            if not f.keys() or (f.metadata() or {}).get("ss_epoch") != str(epochs):
                return False
        saved = output / f"{cfg['id']}-{epochs:06d}-state"
        if not all((saved / name).is_file() for name in ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt")):
            return False
        if read_json(saved / "training_state.json").get("epoch") != epochs:
            return False
        if cfg.get("sample_prompts") and not list((output / "sample").glob(f"{cfg['id']}_e{epochs:06d}_*.png")):
            return False
        return True
    except Exception:
        return False

def prepare(cfg):
    run, images, rejected, instructions = context(cfg)
    folder = run / "images"
    folder.mkdir(parents=True, exist_ok=True)
    sources = caption_sources(cfg, instructions, exclude=run)
    reused = 0
    for item in images:
        source = Path(item["source"])
        dest = folder / (item["sha256"] + source.suffix.lower())
        if not dest.exists():
            shutil.copy2(source, dest)
        cap = dest.with_suffix(".txt")
        if not cap.exists():
            text = item["caption"] or find_caption(item, sources)
            if not text:
                continue
            if not item["caption"]:
                reused += 1
            if not text.startswith(cfg["trigger"]):
                text = cfg["trigger"] + ", " + text
            cap.write_text(text + "\n", encoding="utf-8")
    write_json(run / "dataset_manifest.json", {"images": images, "rejected": rejected})
    save_run_config(run, cfg, instructions)
    print(f"{cfg['id']}: {len(images)} images, {len(rejected)} rejected -> {folder}", flush=True)
    if reused:
        print(f"Reused {reused} captions from matching earlier settings.", flush=True)
    gemma_log = run / "gemma_server.log"
    emit_event("run", path=str(run), job=cfg["id"], images=len(images),
               gemma_offset=gemma_log.stat().st_size if gemma_log.exists() else 0)
    return run, images, instructions

def captions(run, cfg, instructions, limit=None):
    check_pause()
    files = sorted(p for p in (run / "images").iterdir() if p.suffix in EXTENSIONS)
    pending = [p for p in files if not p.with_suffix(".txt").exists()
               or not p.with_suffix(".txt").read_text(encoding="utf-8").strip()]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print("All captions already present; Gemma load skipped.", flush=True)
        return
    with GemmaServer(cfg, run, cancelled=lambda: (AUTO / "stop_requested").exists()) as client:
        for i, path in enumerate(pending, 1):
            check_pause()
            text, usage = client.caption(path, instructions)
            cap = path.with_suffix(".txt")
            temp = cap.with_suffix(".txt.tmp")
            temp.write_text(text + "\n", encoding="utf-8")
            temp.replace(cap)
            with (run / "local_caption_usage.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"image": path.name, "model": cfg["caption_model"], "usage": usage}) + "\n")
            print(f"Caption {i}/{len(pending)}: {path.name[:16]}", flush=True)
            emit_event("caption", done=i, total=len(pending))

def launch_plan(run, cfg, n):
    from fizgig.families.registry import get
    from fizgig.families.launch import plan
    family = family_of(cfg)
    if family != DEFAULT_FAMILY:
        # M1 generalizes the rest of this file (cache names, architecture, sample defaults); until then a run must not
        # quietly plan a Krea 2 job with the Qwen 2.1 descriptor.
        raise ValueError(f"family {family!r} is not wired into the runner yet; only {DEFAULT_FAMILY} training runs")
    desc = get(family)
    # Batch=1, accumulation=1, repeat=1: one image is one step per epoch.
    epochs = max(1, math.floor(cfg["target_steps"] / n + 0.5))
    every = max(1, math.ceil(cfg["save_every_steps_approx"] / n))
    inputs = dict(desc.presets[0][1])
    inputs.update(python=sys.executable, repo_dir=str(ROOT), models=model_paths.resolve(family, cfg),
        image_folder=str(run / "images"), caption_ext=".txt", batch_size=1,
        megapixels=cfg["megapixels"], enable_bucket=True, no_upscale=True,
        cache_root=str(run / "cache"), enable_cache=True, blocks_swap="Auto (detect from GPU)",
        DATASET_CONFIG=str(run / "dataset.toml"), LORA_NAME=cfg["id"],
        LORA_OUTPUT_DIR=str(run / "output"), NETWORK_DIM=cfg["rank"], NETWORK_ALPHA=cfg["rank"],
        LEARNING_RATE=cfg["learning_rate"], MAX_TRAIN_EPOCHS=epochs, SAVE_EVERY_N_EPOCHS=every,
        SEED=cfg.get("seed", 42), ADAPTIVE_LR=cfg["adaptive_lr"],
        ADAPTIVE_LR_MIN=cfg["adaptive_lr_min"], ADAPTIVE_LR_MAX=cfg["adaptive_lr_max"],
        LORA_LR_RATIO=1, GRADIENT_ACCUMULATION=1, MAX_GRAD_NORM=1.0, NETWORK_DROPOUT=0,
        SAVE_STATE=True, SAVE_STATE_ON_TRAIN_END=True, KEEP_LAST_N_STATES=2,
        caption_trigger=cfg["trigger"], loss_watch={"detect": True},
        samples={"enabled": True, "every": every, "width": 768, "height": 768,
                 "steps": 25, "cfg": 1.0, "seed": 42,
                 "prompts": [p.replace("{trigger}", cfg["trigger"]) for p in cfg["sample_prompts"]]})
    result = plan(desc, inputs)
    print(f"Target {cfg['target_steps']}; {n} steps/epoch x {epochs} epochs = ~{n * epochs} steps.")
    if result.problems:
        raise ValueError("\n".join(result.problems))
    return result

def execute(run, cfg, n):
    if LocalClient(cfg).listening():
        raise RuntimeError("Gemma server is still running. Close it before Qwen GPU training, then rerun.")
    result = launch_plan(run, cfg, n)
    for directory in result.dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    for path, content in result.files:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
    state_path = run / "state.json"
    state = read_json(state_path) if state_path.exists() else {"completed_stages": []}
    fingerprint = working_fingerprint(run)
    epochs = max(1, math.floor(cfg["target_steps"] / n + 0.5))
    cached_ok = caches_valid(run)
    # Earlier runners did not save fingerprints. Adopt a legacy run only when its
    # cache captions match and every content-addressed photo still matches its name.
    legacy_matches = (state.get("dataset_fingerprint") is None and cached_ok and
        all(hashlib.sha256(p.read_bytes()).hexdigest() == p.stem
            for p in (run / "images").iterdir() if p.suffix in EXTENSIONS))
    if legacy_matches:
        state["dataset_fingerprint"] = fingerprint
    if state.get("dataset_fingerprint") != fingerprint:
        state["completed_stages"] = []
        state["complete"] = False
        # States from before a caption/image edit cannot silently restart an already finished run.
        state["resume_after_ns"] = time.time_ns()
        state["rebuild_latents"] = True
        state["dataset_fingerprint"] = fingerprint
    final_ok = output_valid(run, cfg, epochs)
    if state.get("complete") and final_ok:
        print("Already completed; skipped.")
        return
    state["complete"] = False
    if not final_ok:
        state["completed_stages"] = [s for s in state["completed_stages"] if s != "Training"]
    if not cached_ok:
        state["completed_stages"] = [s for s in state["completed_stages"]
                                     if s not in ("Cache Preparation", "Text Encoder Caching")]
    write_json(state_path, state)
    for stage in result.stages:
        check_pause()
        if stage.name in state["completed_stages"]:
            continue
        cmd = list(stage.cmd)
        if stage.name == "Cache Preparation" and state.get("rebuild_latents") and "--skip_existing" in cmd:
            cmd.remove("--skip_existing")
        if stage.name == "Training":
            from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                generate_dataset_group_by_blueprint, load_user_config)
            from multiprocessing import Value
            blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
                load_user_config(str(run / "dataset.toml")), argparse.Namespace(), architecture="qwenimage21")
            group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True,
                                                        shared_epoch=Value("i", 0))
            actual_steps = len(group)
            del group
            if actual_steps != n:
                raise ValueError(f"Cache contains {actual_steps} steps/epoch, expected {n}; inspect cache logs.")
            output = run / "output"
            output.mkdir(exist_ok=True)
            if (output / ".pause_requested").exists():
                raise ValueError("Pause marker exists; remove it explicitly before resuming.")
            states = sorted((p for p in output.glob(cfg["id"] + "-*-state")
                             if all((p / name).is_file() for name in ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt"))
                             and (p / "training_state.json").stat().st_mtime_ns >= state.get("resume_after_ns", 0)),
                             key=lambda p: (p / "training_state.json").stat().st_mtime_ns)
            if states:
                cmd += ["--resume", str(states[-1])]
        state["active_stage"] = stage.name
        write_json(state_path, state)
        print(f"Starting {stage.name}", flush=True)
        stage_log = run / (stage.name.replace(" ", "_") + ".log")
        emit_event("stage", stage=stage.name, log_offset=stage_log.stat().st_size if stage_log.exists() else 0)
        with stage_log.open("a", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        if code:
            raise RuntimeError(f"{stage.name} exited {code}; see {run}")
        if stage.name == "Training" and (run / "output" / ".pause_requested").exists():
            raise RuntimeError("Training paused; queue stopped. Remove pause marker to resume.")
        if stage.name == "Training" and not (run / "output" / (cfg["id"] + ".safetensors")).exists():
            raise RuntimeError("Training exited without the final LoRA; inspect Training.log.")
        if stage.name == "Training" and not output_valid(run, cfg, epochs):
            raise RuntimeError("Final LoRA, saved training state or final-epoch samples are missing/invalid; inspect Training.log.")
        state["completed_stages"].append(stage.name)
        if stage.name == "Cache Preparation":
            state["rebuild_latents"] = False
        write_json(state_path, state)
    state["complete"] = True
    write_json(state_path, state)
    print(f"DONE: {run / 'output'}", flush=True)

def doctor_models(failures):
    """Report every family's model files, marking broken paths, and return the families that are fully ready."""
    ready = []
    for family in model_paths.families():
        print(f"[{family}]")
        rows = model_paths.rows(family)
        for row in rows:
            needed = "required" if row["required"] else "optional"
            state = "OK" if row["exists"] else ("MISSING" if row["required"] or row["path"] else "unset")
            print(f"  {row['pref_key']:<26} {state:<8} {needed:<9} [{row['source']}] {row['path']}")
        blocked = model_paths.problems(family)
        if blocked:
            failures.extend(f"[{family}] {p}" for p in blocked)
        else:
            ready.append(family)
    return ready

def doctor_comfy(failures):
    """Connect to ComfyUI and prove the Krea 2 path works before any generation is attempted."""
    from pipeline import comfy as comfy_client
    from pipeline import krea2_gen
    cfg = model_paths.comfy_settings()
    client = comfy_client.Comfy(cfg["url"], comfy_root=cfg["root"])
    print("ComfyUI:", cfg["url"], "OK" if client.ping() else "NOT ANSWERING")
    if not client.ping():
        failures.append(f"ComfyUI is not answering at {cfg['url']}; start it or set comfy.url in automation/config.json")
        return
    try:
        stats = client.system_stats()
        device = (stats.get("devices") or [{}])[0]
        print("  device:", device.get("name"), "VRAM",
              f"{device.get('vram_free', 0) / 2**30:.1f}/{device.get('vram_total', 0) / 2**30:.1f} GB free")
    except comfy_client.ComfyError as exc:
        failures.append(f"ComfyUI /system_stats failed: {exc}")
    nodes = set()
    for variant in krea2_gen.VARIANTS:
        nodes.update(_variant_nodes(variant))
    missing = client.missing_nodes(sorted(nodes))
    print("  nodes:", "OK" if not missing else "MISSING " + ", ".join(missing))
    if missing:
        failures.append("ComfyUI is missing node types: " + ", ".join(missing))
    types = client.clip_types()
    print("  CLIPLoader type krea2:", "OK" if "krea2" in types else "MISSING")
    if "krea2" not in types:
        failures.append("this ComfyUI does not register a krea2 text encoder type")
    for key, kind in (("krea2_raw_dit", "unet"), ("krea2_turbo_dit", "unet"), ("krea2_turbo_lora", "lora"),
                      ("krea2_text_encoder", "clip"), ("krea2_vae", "vae")):
        path = model_paths.resolve("krea2").get(key, "")
        try:
            print(f"  {key}: ComfyUI sees {client.name_for(kind, path)!r}")
        except comfy_client.ComfyError as exc:
            print(f"  {key}: NOT VISIBLE")
            failures.append(str(exc))

def _variant_nodes(variant):
    """ComfyUI node classes each Krea 2 configuration needs."""
    common = {"UNETLoader", "CLIPLoader", "VAELoader", "VAEDecode", "SaveImage", "EmptyLatentImage",
              "KSampler", "LoraLoaderModelOnly", "CLIPTextEncode", "ConditioningZeroOut"}
    if variant == "c":
        common |= {"TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod", "CFGGuider",
                   "SamplerCustomAdvanced", "KSamplerSelect", "BasicScheduler", "RandomNoise", "LoadImage",
                   "ModelSamplingFlux"}
    return common

def main():
    global EVENTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "prepare", "caption", "plan", "run", "status"])
    parser.add_argument("--job", action="append")
    parser.add_argument("--events", action="store_true", help="Structured progress for the desktop GUI")
    parser.add_argument("--limit", type=int, help="Caption at most N pending photos (caption command only)")
    args = parser.parse_args()
    EVENTS = args.events
    if args.limit is not None and (args.command != "caption" or args.limit <= 0):
        parser.error("--limit must be positive and can only be used with caption")
    jobs = [j for j in read_json(AUTO / "todo.json") if j.get("enabled", True)
            and (not args.job or j["id"] in args.job)]
    if not jobs or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError("No selected jobs or duplicate job ids.")
    if args.command == "doctor":
        import torch
        cuda = torch.cuda.is_available()
        failures = [] if cuda else ["CUDA is not available."]
        print("Python:", sys.executable, "CUDA:", cuda)
        for job in jobs:
            cfg = settings(job)
            client = LocalClient(cfg)
            print("Local caption model:", cfg["caption_model"], "at", client.base)
            print("Gemma server:", "ready" if client.ready() else "not running; auto-start on caption")
            if client.ready():
                client.verify_model()
            executable_key = "caption_server_exe" if cfg.get("caption_launch_mode") == "direct" else "caption_start_script"
            for key in (executable_key, "caption_model_file", "caption_mmproj_file"):
                path = cfg.get(key, "")
                exists = bool(path) and Path(path).is_file()
                print(key, "OK" if exists else "MISSING", path)
                if not exists and cfg.get("caption_auto_start", True):
                    failures.append("Missing " + key)
        ready = doctor_models(failures)
        print("Families ready to train:", ", ".join(ready) if ready else "none")
        doctor_comfy(failures)
        for job in jobs:
            cfg = settings(job)
            images, rejected = inventory(cfg)
            print(cfg["id"], family_of(cfg), len(images), "readable unique images;", len(rejected), "rejected")
        if failures:
            raise ValueError("Environment check failed: " + "; ".join(failures))
        return
    lock = AUTO / "queue.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise ValueError("Queue lock exists. If no runner is active, remove automation/queue.lock.") from None
    try:
        for job in jobs:
            cfg = settings(job)
            if args.command == "status":
                run, _, _, _ = context(cfg)
                print(cfg["id"], read_json(run / "state.json") if (run / "state.json").exists() else "not started")
                continue
            run, images, instructions = prepare(cfg)
            if args.command in ("caption", "run"):
                # Check training model readiness before loading the caption model.
                if args.command == "run":
                    blocked = model_paths.problems(family_of(cfg))
                    if blocked:
                        raise ValueError("Models not ready: " + "; ".join(blocked))
                captions(run, cfg, instructions, limit=args.limit)
            if args.command == "plan":
                result = launch_plan(run, cfg, len(images))
                write_json(run / "launch_plan.json", {"stages": [{"name": s.name, "cmd": s.cmd} for s in result.stages],
                                                       "files": result.files})
                print("Plan saved:", run / "launch_plan.json")
            if args.command == "run":
                execute(run, cfg, len(images))
    finally:
        lock.unlink(missing_ok=True)

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
