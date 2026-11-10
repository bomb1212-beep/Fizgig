"""The remote executor ??runs inside the Modal container, but is pure Python and locally testable.

Reads `<run_root>/plan.json`, extracts the submitted `src/` tree, and runs the stages in the same order and with
the same rules as the local `runner.execute()`:

* the three stages run in order: `Cache Preparation` ??`Text Encoder Caching` ??`Training`;
* a stage is skipped only when the remote `run/state.json` records it complete **for the same data, code and plan**;
* before `Training`, the cache's actual step count is checked with the dataset blueprint, like the local runner;
* `--resume` is attached only if the newest `output/<id>-*-state` holds all four required files (and a readable
  `training_state.json`) and its mtime is not older than the state's `resume_after_ns`;
* stage output is appended to `run/<Stage_Name>.log`; `status.json` is replaced atomically;
* the runs Volume is committed every `commit_interval` seconds while a stage runs, and at each stage end and at
  the end of the attempt;
* `status.json` carries the submission attempt id so a finished attempt can be told apart from an older one;
* if the same attempt already finished (its `status.json` says `done`), a re-execution uses the committed result
  and runs nothing (Modal may re-run a successful call on preemption).

The `commit` callback is the only side channel; everything else is files.
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time

STAGES = ("Cache Preparation", "Text Encoder Caching", "Training")
CHECKPOINT_FILES = ("training_state.json", "lora.safetensors", "optimizer.pt", "rng.pt")
LOG_TAIL_LINES = 40


class RemoteError(RuntimeError):
    """A fatal problem with the submitted attempt itself (bad tarball, bad plan). The call must fail."""


class _AlreadyRecorded(RemoteError):
    """The failure status has already been written and committed; do not overwrite it."""


# --------------------------------------------------------------------------------------------- json helpers
def read_json(path):
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def atomic_write_json(path, obj):
    path = str(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def plan_fingerprint(plan):
    """Identity of the plan for skip decisions: the stage commands and the files to write, nothing else."""
    return digest({"stages": [{"name": s.get("name"), "cmd": list(s.get("cmd") or [])} for s in plan.get("stages", [])],
                   "files": [list(item) for item in plan.get("files", [])]})


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def write_status(run_root, *, state, attempt_id, stage=None, exit_code=None, message="",
                 code_sha256="", dataset_fingerprint="", log_tail=None):
    status = {"state": state, "stage": stage, "exit_code": exit_code, "message": message,
              "updated_at": utc_now(), "code_sha256": code_sha256,
              "dataset_fingerprint": dataset_fingerprint, "attempt_id": attempt_id}
    if log_tail is not None:
        status["log_tail"] = log_tail
    atomic_write_json(os.path.join(str(run_root), "status.json"), status)
    return status


# --------------------------------------------------------------------------------------------- tarball
def _check_member(member):
    name = member.name
    if not name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise RemoteError(f"code tarball member with an absolute path: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RemoteError(f"code tarball member with a parent-directory segment: {name!r}")
    if parts[0] != "src":
        raise RemoteError(f"code tarball member outside src/: {name!r}")
    if member.issym() or member.islnk():
        raise RemoteError(f"code tarball member is a link: {name!r}")
    if not (member.isfile() or member.isdir()):
        raise RemoteError(f"code tarball member is not a regular file or directory: {name!r}")


def verify_and_extract_code(run_root, code_dir):
    """Verify the tarball against the submitted digest and unpack only src/ into code_dir."""
    tarball = os.path.join(str(run_root), "code", "fizgig_src.tar.gz")
    if not os.path.isfile(tarball):
        raise RemoteError(f"code tarball missing: {tarball}")
    digest_file = os.path.join(str(run_root), "code", "sha256.txt")
    if not os.path.isfile(digest_file):
        raise RemoteError(f"code digest missing: {digest_file}")
    with open(digest_file, "r", encoding="utf-8") as digest_fh:
        expected = digest_fh.read().split()[0].strip()
    actual = hashlib.sha256()
    with open(tarball, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            actual.update(chunk)
    if actual.hexdigest() != expected:
        raise RemoteError("code tarball sha256 does not match code/sha256.txt")
    if os.path.isdir(code_dir):
        import shutil
        shutil.rmtree(code_dir)
    os.makedirs(code_dir, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            _check_member(member)
        tar.extractall(code_dir, members=members, filter="data")
    return expected


# --------------------------------------------------------------------------------------------- plan & files
def load_plan(run_root):
    plan_path = os.path.join(str(run_root), "plan.json")
    if not os.path.isfile(plan_path):
        raise RemoteError(f"plan.json missing: {plan_path}")
    plan = read_json(plan_path)
    for key in ("attempt_id", "stages", "files", "job_id", "epochs"):
        if key not in plan:
            raise RemoteError(f"plan.json is missing {key!r}")
    for stage in plan["stages"]:
        if not isinstance(stage.get("cmd"), list) or not stage.get("cmd"):
            raise RemoteError(f"plan.json stage {stage.get('name')!r} has no argv array")
    return plan


def write_plan_files(plan, run_folder):
    """Write the files the plan generated (dataset.toml, preview prompts) ??only inside the run folder."""
    prefix = os.path.normpath(run_folder).replace("\\", "/").rstrip("/") + "/"
    for item in plan.get("files", []):
        path, content = item[0], item[1]
        normalized = os.path.normpath(path).replace("\\", "/")
        if ".." in normalized.split("/") or not normalized.startswith(prefix):
            raise RemoteError(f"plan.json wants to write outside the run folder: {path!r}")
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp, normalized)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise


def verify_model_files(plan):
    """The container checks the declared models before any GPU time is spent (the local side cannot)."""
    problems = []
    for entry in plan.get("model_files") or []:
        path, expected = entry.get("path"), entry.get("bytes")
        try:
            actual = os.path.getsize(path)
        except OSError:
            problems.append(f"model file missing: {path}")
            continue
        if expected is not None and actual != int(expected):
            problems.append(f"model file size mismatch: {path} ({actual} bytes, expected {expected})")
    if problems:
        raise RemoteError("; ".join(problems))


def verify_cache_steps(desc_arch, image_count, run_folder, src_root=None):
    """The same blueprint check the local runner does before Training (injectable for tests).

    It runs in this process, not in a stage subprocess, so `fizgig` must be importable here: the Modal function
    process has only remote_exec/spec on its path, and the code exists only as the submitted source. `src_root`
    (the extracted `<code_dir>/src`) goes first on sys.path so the check reads the same code the stages run.
    """
    import argparse
    import sys
    from multiprocessing import Value
    if src_root:
        src_root = str(src_root)
        if src_root in sys.path:
            sys.path.remove(src_root)
        sys.path.insert(0, src_root)
    from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                                       generate_dataset_group_by_blueprint, load_user_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        load_user_config(os.path.join(run_folder, "dataset.toml")), argparse.Namespace(), architecture=desc_arch)
    group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True,
                                                shared_epoch=Value("i", 0))
    actual_steps = len(group)
    del group
    if actual_steps != image_count:
        raise ValueError(f"Cache contains {actual_steps} steps/epoch, expected {image_count}; inspect cache logs.")


def latest_checkpoint(output_folder, lora_id, resume_after_ns):
    """The newest complete `<id>-*-state` newer than `resume_after_ns`, or None (never a partial folder)."""
    best = None
    for entry in os.listdir(output_folder) if os.path.isdir(output_folder) else []:
        if not entry.startswith(lora_id + "-") or not entry.endswith("-state"):
            continue
        folder = os.path.join(output_folder, entry)
        if not all(os.path.isfile(os.path.join(folder, name)) for name in CHECKPOINT_FILES):
            continue
        state_path = os.path.join(folder, "training_state.json")
        try:
            with open(state_path, "rb") as fh:
                json.load(fh)
            mtime = os.stat(state_path).st_mtime_ns
        except (OSError, ValueError):
            continue
        if mtime < (resume_after_ns or 0):
            continue
        if best is None or mtime > best[1]:
            best = (folder, mtime)
    return best[0] if best else None


def training_outputs_valid(run_folder, lora_id, epochs):
    """Final LoRA + the last-epoch state, checked the way the local runner checks a finished run."""
    from safetensors import safe_open
    try:
        with safe_open(os.path.join(run_folder, "output", lora_id + ".safetensors"), framework="pt") as handle:
            if not handle.keys() or (handle.metadata() or {}).get("ss_epoch") != str(epochs):
                return False
        saved = os.path.join(run_folder, "output", f"{lora_id}-{int(epochs):06d}-state")
        if not all(os.path.isfile(os.path.join(saved, name)) for name in CHECKPOINT_FILES):
            return False
        with open(os.path.join(saved, "training_state.json"), encoding="utf-8") as fh:
            if int(json.load(fh).get("epoch")) != int(epochs):
                return False
        return True
    except Exception:
        return False


def stage_log_tail(log_path, lines=LOG_TAIL_LINES):
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            text = fh.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except OSError:
        return ""


def _run_stage_command(cmd, *, log_path, env, cwd, popen, monotonic, sleep, commit, commit_interval, log):
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = popen(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        last_commit = monotonic()
        while process.poll() is None:
            sleep(0.25)
            if monotonic() - last_commit >= commit_interval:
                log(f"committing the volume after {commit_interval}s of progress")
                commit()
                last_commit = monotonic()
        code = process.returncode
    commit()
    return code


def run(run_root, *, code_dir, commit, log=print, verify_steps=True, outputs_valid=training_outputs_valid,
        monotonic=time.monotonic, sleep=time.sleep, popen=subprocess.Popen,
        commit_interval=60, local_src=None):
    """Run the submitted attempt once. Returns the final status dict.

    `run_root` is the `/runs/<run_key>` folder; `code_dir` is where the tarball is extracted; `commit` commits the
    runs volume (no arguments); `verify_steps` is a bool (use the blueprint check) or a callable (fake in tests);
    `outputs_valid` gates the `done` state on the final LoRA/last state being readable (injectable in tests);
    `local_src` forces the PYTHONPATH source (defaults to <code_dir>/src).
    """
    run_root = str(run_root)
    run_folder = os.path.join(run_root, "run")
    plan = None
    try:
        plan = load_plan(run_root)
    except RemoteError as exc:
        write_status(run_root, state="failed", attempt_id="unknown", exit_code=None,
                     message=str(exc), code_sha256="", dataset_fingerprint="")
        commit()
        raise

    attempt_id = plan["attempt_id"]
    code_sha = plan.get("code_sha256", "")
    fingerprint = plan.get("dataset_fingerprint", "")
    job_id = plan["job_id"]
    epochs = int(plan["epochs"])
    try:
        tarball_sha = verify_and_extract_code(run_root, code_dir)
        if code_sha and code_sha != tarball_sha:
            raise RemoteError("plan.json's code_sha256 does not match the tarball (plan.json and code/ disagree)")
        code_sha = tarball_sha
    except RemoteError as exc:
        write_status(run_root, state="failed", attempt_id=attempt_id, exit_code=None,
                     message=str(exc), code_sha256=code_sha, dataset_fingerprint=fingerprint)
        commit()
        raise

    # A re-execution of an attempt that already finished uses the committed result.
    status_path = os.path.join(run_root, "status.json")
    if os.path.isfile(status_path):
        try:
            prior = read_json(status_path)
        except (OSError, ValueError):
            prior = None
        if prior and prior.get("state") == "done" and prior.get("attempt_id") == attempt_id:
            log("this attempt already finished; using the committed result")
            return prior

    try:
        write_plan_files(plan, run_folder)
        verify_model_files(plan)
        # The code comes from the submitted snapshot, never from the host checkout. The rest of the host
        # environment is inherited (Windows needs SystemRoot for sockets; Linux needs standard locale vars).
        src_root = local_src or os.path.join(code_dir, "src")
        env = dict(os.environ)
        env["PYTHONPATH"] = src_root
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        state_path = os.path.join(run_folder, "state.json")
        state = read_json(state_path) if os.path.isfile(state_path) else {"completed_stages": []}
        plan_fp = plan_fingerprint(plan)
        same_run = (state.get("dataset_fingerprint") == fingerprint
                    and (not code_sha or state.get("code_sha256", code_sha) == code_sha)
                    and state.get("plan_fingerprint", plan_fp) == plan_fp)
        if not same_run:
            log("the data, code or plan changed; starting the attempt from the top")
            state = {"completed_stages": [], "resume_after_ns": time.time_ns(), "rebuild_latents": True}
        state["dataset_fingerprint"] = fingerprint
        state["code_sha256"] = code_sha
        state["plan_fingerprint"] = plan_fp
        state["attempt_id"] = attempt_id
        state.setdefault("completed_stages", [])
        atomic_write_json(state_path, state)

        image_count = plan.get("images")
        for stage in plan["stages"]:
            name = stage["name"]
            cmd = list(stage["cmd"])
            if name in state["completed_stages"]:
                log(f"skipping {name}; already completed for this data/code/plan")
                continue
            if name == "Cache Preparation" and state.get("rebuild_latents") and "--skip_existing" in cmd:
                cmd.remove("--skip_existing")
            if name == "Training":
                if verify_steps:
                    if callable(verify_steps):
                        verify_steps()
                    else:
                        arch = plan.get("arch_id")
                        if not arch:
                            raise ValueError("plan.json is missing arch_id (needed for the cache step check)")
                        verify_cache_steps(arch, image_count, run_folder, src_root=src_root)
                output_folder = os.path.join(run_folder, "output")
                os.makedirs(output_folder, exist_ok=True)
                if os.path.isfile(os.path.join(output_folder, ".pause_requested")):
                    raise ValueError("Pause marker exists; remove it explicitly before resuming.")
                checkpoint = latest_checkpoint(output_folder, job_id, state.get("resume_after_ns", 0))
                if checkpoint:
                    cmd += ["--resume", checkpoint]
            state["active_stage"] = name
            atomic_write_json(state_path, state)
            write_status(run_root, state="running", attempt_id=attempt_id, stage=name,
                         code_sha256=code_sha, dataset_fingerprint=fingerprint,
                         message=f"running {name}")
            log(f"Starting {name}")
            stage_log = os.path.join(run_folder, name.replace(" ", "_") + ".log")
            code = _run_stage_command(cmd, log_path=stage_log, env=env, cwd=code_dir,
                                      popen=popen, monotonic=monotonic, sleep=sleep,
                                      commit=commit, commit_interval=commit_interval, log=log)
            if code:
                tail = stage_log_tail(stage_log)
                write_status(run_root, state="failed", attempt_id=attempt_id, stage=name,
                             exit_code=code, message=f"{name} exited {code}; see the remote run folder",
                             code_sha256=code_sha, dataset_fingerprint=fingerprint, log_tail=tail)
                commit()
                raise _AlreadyRecorded(f"{name} exited {code}; see {run_root}")
            if name == "Training":
                if os.path.isfile(os.path.join(run_folder, "output", ".pause_requested")):
                    write_status(run_root, state="failed", attempt_id=attempt_id, stage=name, exit_code=None,
                                 message="Training paused; queue stopped. Remove the pause marker to resume.",
                                 code_sha256=code_sha, dataset_fingerprint=fingerprint,
                                 log_tail=stage_log_tail(stage_log))
                    commit()
                    raise _AlreadyRecorded("Training paused; queue stopped. Remove pause marker to resume.")
                if not outputs_valid(run_folder, job_id, epochs):
                    write_status(run_root, state="failed", attempt_id=attempt_id, stage=name, exit_code=None,
                                 message="final LoRA or the saved training state is missing/invalid",
                                 code_sha256=code_sha, dataset_fingerprint=fingerprint,
                                 log_tail=stage_log_tail(stage_log))
                    commit()
                    raise _AlreadyRecorded("final LoRA or the saved training state is missing/invalid; inspect Training.log")
            state["completed_stages"].append(name)
            if name == "Cache Preparation":
                state["rebuild_latents"] = False
            atomic_write_json(state_path, state)
            commit()
        state["complete"] = True
        state["active_stage"] = None
        atomic_write_json(state_path, state)
        status = write_status(run_root, state="done", attempt_id=attempt_id, stage="Training",
                              exit_code=0, message="", code_sha256=code_sha,
                              dataset_fingerprint=fingerprint)
        commit()
        log("DONE")
        return status
    except _AlreadyRecorded:
        raise
    except Exception as exc:
        try:
            write_status(run_root, state="failed", attempt_id=attempt_id, exit_code=None,
                         stage=None,
                         message=str(exc), code_sha256=code_sha, dataset_fingerprint=fingerprint)
        except Exception:
            pass
        commit()
        raise
