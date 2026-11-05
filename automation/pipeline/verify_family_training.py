"""M1 acceptance run: a small, real Krea 2 training on a scratch dataset.

This is the GPU half of the M1 acceptance criteria (spec section 4, M1): cache, train, save checkpoints, pause and
resume, and then let ComfyUI load the resulting LoRA. It is deliberately small (a handful of images, a few epochs) and
it works in its own `automation` directory, so the real `config.json` and `todo.json` are never touched.

  python -m pipeline.verify_family_training --source "G:/.../dataset/수아맘" --images 10 --epochs 3
  python -m pipeline.verify_family_training --source ... --pause-at 2        # pause, then resume

The pause/resume check runs the queue twice: the first run stops itself at the given epoch (the same `.pause_requested`
marker the GUI uses), the second must pick up from the checkpoint rather than from scratch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]          # automation
ROOT = AUTO.parent                                  # Fizgig_auto
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))

import runner  # noqa: E402
from pipeline import models  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def pick_sources(folder, count):
    """Images that already have a caption beside them, so the run never needs the caption model."""
    folder = Path(folder)
    out = []
    for image in sorted(folder.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        caption = image.with_suffix(".txt")
        if caption.is_file() and caption.read_text(encoding="utf-8-sig").strip():
            out.append((image, caption))
        if len(out) >= count:
            break
    return out


def scratch_workspace(sources, *, family, trigger, epochs, megapixels, rank, policy, kind):
    """A temporary automation folder holding a config.json and todo.json for this one run."""
    workspace = Path(tempfile.mkdtemp(prefix="m1-verify-", dir=AUTO))
    dataset = workspace / "dataset"
    dataset.mkdir()
    for image, caption in sources:
        shutil.copy2(image, dataset / image.name)
        shutil.copy2(caption, dataset / caption.name)

    # Start from the real settings (caption server, sample prompts) so this is the same runner users get.
    base = models.read_json(AUTO / "config.json", {})
    base.update(dataset_root=str(dataset), family=family, trigger=trigger, kind=kind,
                caption_policy=policy, megapixels=megapixels, rank=rank, adaptive_lr=True,
                adaptive_lr_min=0.0002, adaptive_lr_max=0.0004, learning_rate=0.0002,
                target_steps=len(sources) * epochs, save_every_steps_approx=len(sources),
                max_images_per_job=max(50, len(sources)))
    (workspace / "config.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    (workspace / "todo.json").write_text(json.dumps(
        [{"id": "m1_verify", "enabled": True, "trigger": trigger, "seed": 42}], ensure_ascii=False, indent=2),
        encoding="utf-8")
    (workspace / "caption_instructions.txt").write_text("describe the image", encoding="utf-8")
    (workspace / "caption_instructions_anime.txt").write_text("describe the illustration", encoding="utf-8")
    return workspace


def pause_when_checkpointed(workspace, stop_event):
    """Write the trainer's pause marker once the first checkpoint exists, the way the GUI's pause button does.

    Stopping between stages only proves the queue can stop. Resuming a *training* needs a checkpoint saved mid-run:
    the trainer finishes the epoch it is on, writes `<id>-00000N-state`, then stops, and the next run finds that state
    and passes `--resume` to the trainer instead of starting the epochs again.
    """
    import threading
    import time

    def watch():
        deadline = time.monotonic() + 3600
        while not stop_event.is_set() and time.monotonic() < deadline:
            runs = sorted((workspace / "runs").glob("m1_verify-*"))
            if runs:
                output = runs[-1] / "output"
                states = sorted(output.glob("*-state")) if output.is_dir() else []
                if states:
                    (output / ".pause_requested").write_text("1", encoding="utf-8")
                    print(f"pause marker written after {states[0].name}", flush=True)
                    return
            time.sleep(2)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread


def run_once(workspace, argv, *, pause_at_first_checkpoint=False, record_commands=None):
    """Run the real runner against the scratch folder, with the same entry point the CLI uses."""
    import threading
    from unittest.mock import patch
    original = runner.AUTO
    runner.AUTO = workspace
    stop_event = threading.Event()
    watcher = None
    real_call = runner.subprocess.call

    def logging_call(*a, **k):
        cmd = a[0] if a else k.get("args")
        if record_commands is not None and isinstance(cmd, (list, tuple)):
            record_commands.append(list(cmd))
        return real_call(*a, **k)

    try:
        if pause_at_first_checkpoint:
            watcher = pause_when_checkpointed(workspace, stop_event)
        with patch.object(runner.subprocess, "call", side_effect=logging_call):
            with patch.object(sys, "argv", ["runner.py", *argv]):
                runner.main()
    except (ValueError, RuntimeError) as exc:
        print(f"runner said: {exc}", file=sys.stderr)
        return False
    finally:
        stop_event.set()
        if watcher is not None:
            watcher.join(timeout=5)
        runner.AUTO = original
    return True


def report(workspace):
    """What the run actually produced, read back from the run folder."""
    runs = sorted((workspace / "runs").glob("m1_verify-*"))
    if not runs:
        print("no run folder was created")
        return None
    run = runs[-1]
    state = models.read_json(run / "state.json", {})
    output = run / "output"
    loras = sorted(p.name for p in output.glob("*.safetensors")) if output.is_dir() else []
    states = sorted(p.name for p in output.glob("*-state")) if output.is_dir() else []
    samples = sorted(p.name for p in (output / "sample").glob("*.png")) if (output / "sample").is_dir() else []
    caches = sorted(p.name for p in (run / "cache").rglob("*.safetensors")) if (run / "cache").is_dir() else []
    print(json.dumps({
        "run": str(run), "family": state.get("family"), "arch_id": state.get("arch_id"),
        "complete": state.get("complete"), "preview_missing": state.get("preview_missing"),
        "stages": state.get("completed_stages"), "loras": loras, "states": states,
        "samples": samples, "cache_files": len(caches),
        "cache_names": [c for c in caches if "_te.safetensors" in c][:2],
    }, ensure_ascii=False, indent=1))
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1 Krea 2 training acceptance run")
    parser.add_argument("--source", required=True, help="folder of images that already have captions")
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--family", default="krea2")
    parser.add_argument("--trigger", default="suamom")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--megapixels", type=float, default=0.25)
    parser.add_argument("--policy", default="preserve")
    parser.add_argument("--kind", default="anime")
    parser.add_argument("--pause-at", type=int, help="stop after this epoch, then resume in a second run")
    parser.add_argument("--keep", action="store_true", help="leave the scratch folder behind for inspection")
    args = parser.parse_args(argv)

    sources = pick_sources(args.source, args.images)
    if len(sources) < 3:
        raise SystemExit(f"need at least 3 captioned images in {args.source}, found {len(sources)}")
    workspace = scratch_workspace(sources, family=args.family, trigger=args.trigger, epochs=args.epochs,
                                  megapixels=args.megapixels, rank=args.rank, policy=args.policy, kind=args.kind)
    print(f"scratch workspace: {workspace} ({len(sources)} images, {args.epochs} epochs, "
          f"{args.megapixels} MP, rank {args.rank}, policy {args.policy})")
    try:
        if args.pause_at:
            # One run, paused by its own pause marker at the first checkpoint; then a second run that must resume.
            print("--- run 1: cache, then train until the first checkpoint and pause there ---")
            run_once(workspace, ["run", "--job", "m1_verify"], pause_at_first_checkpoint=True)
            runs = sorted((workspace / "runs").glob("m1_verify-*"))
            marker = runs[-1] / "output" / ".pause_requested" if runs else None
            states = sorted(p.name for p in (runs[-1] / "output").glob("*-state")) if runs else []
            print(f"--- paused with {len(states)} checkpoint(s): {states} ---")
            if marker is not None:
                marker.unlink(missing_ok=True)
            print("--- run 2: resume ---")
            commands = []
            ok = run_once(workspace, ["run", "--job", "m1_verify"], record_commands=commands)
            resumed = [c for c in commands if "--resume" in c]
            print(f"training commands: {len(commands)}, with --resume: {len(resumed)}")
            for cmd in resumed:
                print("  resume arg:", cmd[cmd.index("--resume") + 1])
        else:
            commands = []
            ok = run_once(workspace, ["run", "--job", "m1_verify"], record_commands=commands)
        print(f"runner exit: {'ok' if ok else 'failed'}")
        report(workspace)
    finally:
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)
        else:
            print(f"kept: {workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())