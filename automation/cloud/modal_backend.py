"""Local side of the Modal backend.

`submit`, `follow`, `fetch`, `sync`, `cancel`, `estimate` and `remote_plan` - the flows from the Modal cloud-training
design (docs/superpowers/specs/2026-11-10-modal-cloud-training-design.md Section 5-Section 9). Everything talks to Modal through
`cloud.client.ModalClient`; tests substitute a fake with the same surface.

Layout invariants:

* the local run folder keeps the local layout (images/, state.json, logs, output/); the remote folder mirrors it at
  `/runs/<run_key>/run` in the `fizgig-runs` Volume;
* local `state.json` carries a `remote` block: provider/app_version/call_id/run_key/gpu/submitted_at/code_sha256/
  plan_fingerprint/dataset_fingerprint/inventory_fingerprint/images/estimate/status/stage/message/attempt_id;
* `remote.status` machine: (absent) -> submitted -> running -> done -> fetched | failed | cancelled | stale | unknown;
* a **submitted/running** remote never resubmits - `execute()` follows only (Section 7.5);
* failed/cancelled resubmits the *same* run_key with the *stored* bundle when the stored plan still matches (Section 7.2);
* `unknown` refuses to resubmit until the user cancels (Section 7.2);
* fingerprint drift marks `stale` and never adopts a remote result (Section 5.4);
* the local complete marker (`complete`, `completed_stages`) is written only after `output_valid` passes (Section 5.4);
  the remote folder is deleted only after that, and deletion failure is retried by `sync`.
"""
import contextlib
import datetime
import json
import os
import shutil
import tarfile
import tempfile
import time
import uuid
import hashlib
from pathlib import Path

import runner
from cloud import remote_exec, spec

STAGE_NAMES = ("Cache Preparation", "Text Encoder Caching", "Training")
REMOTE_STATUSES = ("submitted", "running", "done", "fetched", "failed", "cancelled", "stale", "unknown")


class RemoteBusy(RuntimeError):
    """Another local process holds this run's update lock."""


# --------------------------------------------------------------------------- helpers
def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _new_attempt_id():
    return uuid.uuid4().hex[:12]


def _run_key(cfg, fingerprint):
    return f"{cfg['id']}-{fingerprint[:12]}"


def _pid_alive(pid):
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
        handle = kernel.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5   # access denied is not proof the owner died
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259   # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def _run_lock(run, wait_seconds=30.0):
    """Serialize remote-state writes for one run (the same run can be touched by run/sync/cancel)."""
    path = Path(run) / ".remote.lock"
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = -1
            if _pid_alive(pid):
                if wait_seconds <= 0 or time.monotonic() > deadline:
                    raise RemoteBusy(f"{run} is being updated by another process (pid {pid})")
                time.sleep(0.2)
                continue
            try:
                path.unlink()   # stale lock: the owner died without cleaning up
            except OSError:
                pass
    try:
        yield
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def cancel_request_path(run):
    """The file the GUI writes to ask for a remote cancel; the GUI wires it up before submit (spec Section 7.3)."""
    env = os.environ.get("FIZGIG_REMOTE_CANCEL_REQUEST")
    return Path(env) if env else Path(run) / "cancel_remote.request"


def _read_json_safe(path):
    try:
        return runner.read_json(path)
    except Exception:
        return {}


def _read_state(run):
    return _read_json_safe(Path(run) / "state.json")


def _write_state(run, state):
    runner.write_json(Path(run) / "state.json", state)


def _update_remote(run, state, **fields):
    remote = state.setdefault("remote", {})
    remote.update(fields)
    remote["updated_at"] = _now_iso()
    _write_state(run, state)
    return remote


def _count_images(run):
    folder = Path(run) / "images"
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in runner.EXTENSIONS)


def _current_inventory_fingerprint(cfg):
    """Fingerprint of the *current original* inventory (re-read from dataset_root now, not the manifest)."""
    try:
        images, _ = runner.inventory(cfg)
        return runner.digest([(i["sha256"], str(i.get("caption") or "")) for i in images])
    except Exception:
        return None


def _cfg_from_run_config(run):
    saved = _read_json_safe(Path(run) / "run_config.json")
    cfg = saved.get("settings")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"run_config.json missing or unreadable in {run}")
    return cfg


def _missing_captions(run):
    manifest = _read_json_safe(Path(run) / "dataset_manifest.json")
    return [item for item in manifest.get("images", []) if not str(item.get("caption") or "").strip()]


def _elapsed_since(iso):
    try:
        then = datetime.datetime.fromisoformat(str(iso))
        seconds = (datetime.datetime.now() - then).total_seconds()
    except ValueError:
        return "?"
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{int(seconds / 60)}m"
    return f"{seconds / 3600:.1f}h"


def default_runs_roots():
    roots = [runner.runs_root()]
    try:
        from pipeline import project as pipeline_project
        projects = pipeline_project.projects_dir()
        if projects.is_dir():
            for project in sorted(projects.iterdir()):
                seed_lora = project / "seed_lora"
                if not seed_lora.is_dir():
                    continue
                for snapshot in sorted(seed_lora.iterdir()):
                    runs = snapshot / "runs"
                    if runs.is_dir():
                        roots.append(runs)
    except Exception:
        pass
    return roots


def find_runs(job_id, roots=None):
    """Every run folder whose job id matches (id-version folders, default and per-project seed roots)."""
    out = []
    for root in roots or default_runs_roots():
        root = Path(root)
        if not root.is_dir():
            continue
        for run in sorted(root.glob(f"{job_id}-*")):
            if run.is_dir() and (run / "state.json").exists():
                out.append(run)
    return out


# --------------------------------------------------------------------------- plan & estimate
def estimate(cfg, n, epochs):
    inputs = spec.estimate_inputs(cfg)
    gpu = str(cfg.get("modal_gpu") or spec.DEFAULT_GPU)
    sec_per_step = float(inputs["sec_per_step"].get(gpu, 1.6))
    steps = int(n) * int(epochs)
    hours = inputs["overhead_minutes"] / 60 + steps * sec_per_step / 3600
    rate = (float(inputs["gpu_rate"].get(gpu, 1.95))
            + spec.CPU_CORES * float(inputs["cpu_rate"])
            + spec.MEMORY_GIB * float(inputs["memory_rate"]))
    return {"hours": round(hours, 3), "usd": round(hours * rate, 2),
            "steps": steps, "sec_per_step": sec_per_step, "gpu": gpu}


def build_source_tarball(src_root, dst):
    """Deterministic tar.gz of src/ (no caches), for a stable code sha across submissions of unchanged code."""
    src_root = Path(src_root)
    files = []
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root).as_posix()
        parts = rel.split("/")
        if any(part == "__pycache__" or part.endswith((".pyc", ".pyo")) for part in parts):
            continue
        files.append((path, "src/" + rel))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    import gzip
    with open(dst, "wb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for path, arcname in files:
                ti = tar.gettarinfo(str(path), arcname=arcname)
                ti.mtime = 0
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mode = 0o644
                with open(path, "rb") as handle:
                    tar.addfile(ti, handle)
    return hashlib.sha256(Path(dst).read_bytes()).hexdigest()


def remote_plan(run, cfg, n, *, run_key, attempt_id, fingerprint, inventory_fingerprint,
                code_sha256, epochs=None):
    """The remote plan document (plan.json). Pure - writes nothing by itself."""
    from fizgig.families.launch import PlanTarget
    epochs = epochs or runner.epochs_for(cfg["target_steps"], n)
    family = runner.family_of(cfg)
    target = PlanTarget(posix=True, python=spec.REMOTE_PYTHON, repo_dir=spec.CODE_DIR,
                        image_folder=f"{spec.RUNS_ROOT}/{run_key}/run/images",
                        cache_root=f"{spec.RUNS_ROOT}/{run_key}/run/cache",
                        dataset_config=f"{spec.RUNS_ROOT}/{run_key}/run/dataset.toml",
                        output_dir=f"{spec.RUNS_ROOT}/{run_key}/run/output",
                        check_image_folder=str(Path(run) / "images"))
    result = runner.launch_plan(run, cfg, n, target=target)
    arch_id = runner.model_paths.description(family).arch_id
    model_files = [{"role": entry["role"], "path": spec.model_remote_path(entry), "bytes": entry["bytes"]}
                   for entry in spec.MODEL_FILES]
    return {"format": 1, "attempt_id": attempt_id, "job_id": cfg["id"], "family": family, "arch_id": arch_id,
            "images": int(n), "epochs": int(epochs), "target_steps": int(cfg["target_steps"]),
            "dataset_fingerprint": fingerprint, "inventory_fingerprint": inventory_fingerprint,
            "code_sha256": code_sha256, "created_at": _now_iso(),
            "stages": [{"name": stage.name, "cmd": list(stage.cmd)} for stage in result.stages],
            "files": [[path, content] for path, content in result.files],
            "dirs": list(result.dirs),
            "model_files": model_files}


# --------------------------------------------------------------------------- submit
def _stored_bundle_usable(client, remote):
    """The stored submission is still there and self-consistent - a re-submit can spawn without re-uploading."""
    run_key = remote.get("run_key")
    if not run_key or not remote.get("code_sha256"):
        return False
    plan_text = client.read_text(f"{spec.RUNS_ROOT}/{run_key}/plan.json")
    sha_text = client.read_text(f"{spec.RUNS_ROOT}/{run_key}/code/sha256.txt")
    if not plan_text or not sha_text:
        return False
    try:
        doc = json.loads(plan_text)
    except ValueError:
        return False
    if doc.get("dataset_fingerprint") != remote.get("dataset_fingerprint"):
        return False
    if sha_text.split()[0].strip() != remote["code_sha256"]:
        return False
    return True


def _submitted_status(attempt_id, code_sha256, dataset_fingerprint):
    """The volume status.json written just before spawn, replacing whatever an earlier attempt left there.

    Without it a re-run reads the old attempt's `failed` record before the new container has written anything and
    reports the new (still running, still billing) call as failed. Same keys as `remote_exec.write_status`."""
    return {"state": "submitted", "stage": None, "exit_code": None, "message": "",
            "updated_at": remote_exec.utc_now(), "code_sha256": code_sha256 or "",
            "dataset_fingerprint": dataset_fingerprint or "", "attempt_id": attempt_id}


def _write_status_file(folder, status):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "status.json").write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")


def submit(run, cfg, n, *, epochs=None, fingerprint=None, inventory_fingerprint=None,
           client=None, log=print):
    """Section 5.1: validate, deploy if the version drifted, ensure models, upload the bundle, spawn once."""
    run = Path(run)
    epochs = epochs or runner.epochs_for(cfg["target_steps"], n)
    fingerprint = fingerprint or runner.working_fingerprint(run)
    inv = inventory_fingerprint or runner.inventory_fingerprint(run)
    client = client or _default_client(log)
    missing = _missing_captions(run)
    if missing:
        names = ", ".join(Path(item.get("source", "?")).name for item in missing[:5])
        raise ValueError(f"{cfg['id']}: {len(missing)} adopted image(s) have no caption (e.g. {names}); "
                         f"Modal runs cannot caption - caption them first")
    state = _read_state(run)
    remote = state.get("remote") or {}

    local_version = spec.local_deployment_version()
    deployed = client.deployed_version()
    if deployed != local_version:
        log(f"deployed version {deployed!r} != local {local_version!r} - deploying first "
            f"(the first build can take 10-20 minutes)")
        client.deploy()
        deployed = client.deployed_version()
        if deployed != local_version:
            raise RuntimeError(f"the deployed app reports {deployed!r}, expected {local_version!r}; "
                               f"check `modal deploy automation/cloud/modal_app.py`")
    client.ensure_models()

    # The fresh plan fingerprint (offline, cheap) decides between reusing the stored bundle and uploading anew.
    attempt_id = _new_attempt_id()
    run_key = _run_key(cfg, fingerprint)
    fresh_plan = remote_plan(run, cfg, n, run_key=run_key, attempt_id=attempt_id,
                             fingerprint=fingerprint, inventory_fingerprint=inv,
                             code_sha256="", epochs=epochs)
    fresh_plan_fp = remote_exec.plan_fingerprint(fresh_plan)
    reuse = (remote.get("status") in ("failed", "cancelled", "unknown")
             and remote.get("dataset_fingerprint") == fingerprint
             and remote.get("plan_fingerprint") == fresh_plan_fp
             and _stored_bundle_usable(client, remote))
    if reuse:
        run_key = remote["run_key"]
        attempt_id = remote.get("attempt_id") or _new_attempt_id()
        _update_remote(run, _read_state(run), status="submitting", attempt_id=attempt_id,
                       message="re-submitting the stored bundle (no re-upload; checkpoints kept)")
        # Only the status record goes up: replace the earlier attempt's verdict before the new call exists.
        # An upload failure raises here, before anything is spawned.
        status_staging = Path(tempfile.mkdtemp(prefix="fizgig-status-"))
        try:
            _write_status_file(status_staging, _submitted_status(
                attempt_id, remote.get("code_sha256"), remote.get("dataset_fingerprint")))
            client.upload(str(status_staging), f"{spec.RUNS_ROOT}/{run_key}")
        finally:
            shutil.rmtree(status_staging, ignore_errors=True)
        try:
            call_id = client.spawn(cfg["modal_gpu"], run_key)
        except Exception as exc:
            _update_remote(run, _read_state(run), status="unknown",
                           message=f"spawn result unknown ({exc}); the call may exist anyway - "
                                   f"check the Modal dashboard before re-running")
            raise
        _update_remote(run, _read_state(run), call_id=call_id, status="submitted",
                       attempt_id=attempt_id, stage=None, message="")
        log(f"re-submitted {run_key} as call {call_id} (stored code and plan; checkpoints kept)")
        return {"run_key": run_key, "call_id": call_id, "attempt_id": attempt_id}

    run_key = _run_key(cfg, fingerprint)
    staging = Path(tempfile.mkdtemp(prefix="fizgig-submit-"))
    try:
        src_root = Path(runner.ROOT) / "src"
        tarball = staging / "code" / "fizgig_src.tar.gz"
        code_sha = build_source_tarball(src_root, tarball)
        fresh_plan["code_sha256"] = code_sha
        plan = fresh_plan
        plan_fp = fresh_plan_fp
        (staging / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        (staging / "code" / "sha256.txt").write_text(code_sha + "  fizgig_src.tar.gz\n", encoding="utf-8")
        # Uploaded with the bundle: overwrites any status.json an earlier attempt left under the same run_key.
        _write_status_file(staging, _submitted_status(attempt_id, code_sha, fingerprint))
        shutil.copytree(run / "images", staging / "run" / "images")
        est = estimate(cfg, n, epochs)
        _update_remote(run, state, provider="modal", app_version=local_version, run_key=run_key,
                       gpu=cfg["modal_gpu"], submitted_at=_now_iso(), code_sha256=code_sha,
                       plan_fingerprint=plan_fp, dataset_fingerprint=fingerprint,
                       inventory_fingerprint=inv, images=int(n), attempt_id=attempt_id,
                       estimate=est, status="submitting", stage=None, message="uploading the bundle")
        uploaded = client.upload(str(staging), f"{spec.RUNS_ROOT}/{run_key}")
        try:
            call_id = client.spawn(cfg["modal_gpu"], run_key)
        except Exception as exc:
            _update_remote(run, _read_state(run), status="unknown",
                           message=f"spawn result unknown ({exc}); the call may exist anyway - check the "
                                   f"Modal dashboard; `runner cancel --job` clears the block once you know")
            raise
        _update_remote(run, _read_state(run), call_id=call_id, status="submitted",
                       stage=None, message="")
        log(f"submitted {run_key} as call {call_id} ({uploaded} files)")
        log(f"estimate: ~{est['steps']} steps, ~{est['hours']} h on {est['gpu']}, about ${est['usd']} "
            f"(the first image build and the ~30 GB model download are one-time costs)")
        return {"run_key": run_key, "call_id": call_id, "attempt_id": attempt_id, "estimate": est}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _default_client(log):
    from cloud.client import ModalClient
    return ModalClient(log=log)


# --------------------------------------------------------------------------- logs
def _log_offsets(run):
    offsets = {}
    for name in STAGE_NAMES:
        path = Path(run) / (name.replace(" ", "_") + ".log")
        offsets[name] = path.stat().st_size if path.exists() else 0
    return offsets


def _pump_logs(run, client, run_key, offsets):
    """Append only the bytes the local copies have not seen yet (the remote commits every 60 s)."""
    for name in STAGE_NAMES:
        data = client.read_bytes(f"{spec.RUNS_ROOT}/{run_key}/run/{name.replace(' ', '_')}.log")
        if data is None:
            continue
        path = Path(run) / (name.replace(" ", "_") + ".log")
        current = offsets[name]
        if len(data) > current:
            mode = "ab" if path.exists() else "wb"
            with open(path, mode) as handle:
                handle.write(data[current:])
            offsets[name] = len(data)


# --------------------------------------------------------------------------- follow
def follow(run, cfg=None, *, client=None, once=False, log=print, sleep=time.sleep):
    """Section 5.2: poll the call + the remote status.json; tail the stage logs; fetch on success; Section 7.2/Section 7.3 otherwise."""
    run = Path(run)
    cfg = cfg or _cfg_from_run_config(run)
    state = _read_state(run)
    remote = state.get("remote") or {}
    run_key, call_id = remote.get("run_key"), remote.get("call_id")
    if not run_key or not call_id:
        raise RuntimeError(f"{run.name}: no submitted call recorded; run `runner run` to submit")
    client = client or _default_client(log)
    poll_seconds = float(spec.estimate_inputs(cfg).get("poll_seconds") or 30)
    offsets = _log_offsets(run)

    def _vstatus():
        text = client.read_text(f"{spec.RUNS_ROOT}/{run_key}/status.json")
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    while True:
        poll = client.poll(call_id)
        vol = _vstatus()
        vstate = (vol or {}).get("state")
        vstage = (vol or {}).get("stage")
        state = _read_state(run)
        remote = state.get("remote") or {}
        if vstage and vstage != remote.get("stage"):
            log(f"remote stage: {vstage}")
            runner.emit_event("stage", stage=vstage,
                              log_offset=offsets.get(vstage, 0))
            _update_remote(run, state, status="running", stage=vstage)
        elif poll == "running" and remote.get("status") == "submitted":
            _update_remote(run, state, status="running")
        _pump_logs(run, client, run_key, offsets)

        cancel_path = cancel_request_path(run)
        if cancel_path.exists():
            client.cancel(call_id)
            _update_remote(run, _read_state(run), status="cancelled", message="cancelled by user")
            try:
                cancel_path.unlink()
            except OSError:
                pass
            log("cancelled; the remote folder is kept for a later resume")
            return {"status": "cancelled"}

        if vstate == "done" or poll == "done":
            _update_remote(run, _read_state(run), status="done")
            log("remote run finished; fetching the result")
            return fetch(run, cfg, client=client, log=log)

        if vstate == "failed" or poll == "failed":
            _remote_failed(run, poll, vol, client, log)
            raise AssertionError("unreachable")

        if vstate == "cancelled" or poll == "cancelled":
            message = (vol or {}).get("message") or "cancelled"
            _update_remote(run, _read_state(run), status="cancelled", message=message)
            raise RuntimeError("the remote run was cancelled")

        if poll == "expired":
            if vstate == "done":
                _update_remote(run, _read_state(run), status="done")
                return fetch(run, cfg, client=client, log=log)
            if vstate == "failed":
                _remote_failed(run, poll, vol, client, log)
                raise AssertionError("unreachable")
            message = ("the call result expired and there is no finished record in the volume; "
                       "check the Modal dashboard - nothing is resubmitted automatically")
            _update_remote(run, _read_state(run), status="unknown", message=message)
            raise RuntimeError(message)

        if poll == "unknown":
            if vstate == "done":
                _update_remote(run, _read_state(run), status="done")
                return fetch(run, cfg, client=client, log=log)
            message = ("Modal no longer knows this call; check the Modal dashboard, then run "
                       "`runner cancel --job ...` to mark it cancelled before re-running")
            _update_remote(run, _read_state(run), status="unknown", message=message)
            raise RuntimeError(message)

        # still running
        state = _read_state(run)
        remote = state.get("remote") or {}
        if remote.get("status") != "running":
            _update_remote(run, state, status="running")
        if once:
            log(f"running on {remote.get('gpu', '?')} (stage {vstage or '?'}, "
                f"{_elapsed_since(remote.get('submitted_at'))} elapsed)")
            return {"status": "running"}
        sleep(poll_seconds)


def _remote_failed(run, poll, vol, client, log):
    message = (vol or {}).get("message") or ""
    stage = (vol or {}).get("stage")
    code = (vol or {}).get("exit_code")
    tail = (vol or {}).get("log_tail") or ""
    # "submitted" is the record submit() leaves before spawn: the container died before it wrote anything.
    if poll == "failed" and (vol is None or vol.get("state") in ("submitted", "running")):
        message = ("Modal이 작업을 중단했다 - 크레딧, 지출 한도 소진이나 6시간 제한일 수 있다 "
                   "(the remote container vanished mid-run; the checkpoint folder is preserved)")
    if not message:
        message = f"{stage or 'the remote run'} exited {code}; see {run}"
    _update_remote(run, _read_state(run), status="failed", stage=stage, message=message)
    if tail:
        print("--- last remote log lines ---", flush=True)
        print(tail, flush=True)
    raise RuntimeError(message)


# --------------------------------------------------------------------------- fetch
def fetch(run, cfg=None, *, epochs=None, client=None, log=print):
    """Section 5.4: compare fingerprints, download into a temp folder, validate with the local gate, publish, then delete the remote folder."""
    run = Path(run)
    cfg = cfg or _cfg_from_run_config(run)
    state = _read_state(run)
    remote = state.get("remote") or {}
    run_key = remote.get("run_key")
    if not run_key:
        raise RuntimeError(f"{run.name}: no remote run recorded; nothing to fetch")
    fingerprint = remote.get("dataset_fingerprint")
    client = client or _default_client(log)

    current = runner.working_fingerprint(run) if (run / "images").is_dir() else None
    if current != fingerprint:
        _update_remote(run, _read_state(run), status="stale",
                       message="제출 후 데이터가 바뀌었다 - 원격 결과를 채택하지 않는다")
        raise RuntimeError("the working images changed after submission; not fetching the remote result")
    current_inventory = _current_inventory_fingerprint(cfg)
    if remote.get("inventory_fingerprint") and current_inventory and current_inventory != remote["inventory_fingerprint"]:
        _update_remote(run, _read_state(run), status="stale",
                       message="원본 이미지, 캡션 인벤토리가 바뀌었다 - 이전 원격 결과를 채택하지 않는다")
        raise RuntimeError("the original image captions changed after submission; not fetching the remote result")

    n = remote.get("images") or _count_images(run)
    epochs = epochs or runner.epochs_for(cfg["target_steps"], n)
    tmp = Path(tempfile.mkdtemp(prefix="fizgig-fetch-"))
    try:
        client.download(f"{spec.RUNS_ROOT}/{run_key}/run/output", str(tmp / "output"))
        logs_dir = tmp / "logs"
        logs_dir.mkdir()
        for name in STAGE_NAMES:
            data = client.read_bytes(f"{spec.RUNS_ROOT}/{run_key}/run/{name.replace(' ', '_')}.log")
            if data is not None:
                (logs_dir / (name.replace(" ", "_") + ".log")).write_bytes(data)

        training_ok, _preview_ok = runner.output_state(tmp, cfg, epochs)
        if not training_ok:
            _update_remote(run, _read_state(run), status="done",
                           message="fetch validation failed: final LoRA or the last state folder is missing/invalid")
            raise RuntimeError("fetch validation failed; the remote run folder is preserved - "
                               "inspect the remote output, then retry with the same job")

        # Gate the publish on the fingerprint one last time.
        if runner.working_fingerprint(run) != fingerprint:
            raise RuntimeError("the dataset changed while fetching; the remote result is preserved, not published")

        _publish_output(run, tmp / "output")
        for log_file in logs_dir.iterdir():
            shutil.copy2(log_file, run / log_file.name)

        state = _read_state(run)
        state["complete"] = True
        state["completed_stages"] = list(STAGE_NAMES)
        state["active_stage"] = None
        warning = runner.preview_missing(run, cfg, epochs)
        if warning:
            state["preview_missing"] = True
        else:
            state.pop("preview_missing", None)
        _update_remote(run, state, status="fetched", message="", cleanup_pending=False)
        print(f"Fetched {run_key} -> {run / 'output'}", flush=True)
        if warning:
            print(warning, flush=True)
        try:
            client.delete(f"{spec.RUNS_ROOT}/{run_key}")
        except Exception as exc:
            _update_remote(run, _read_state(run), cleanup_pending=True,
                           message=f"fetched, but the remote cleanup failed: {exc}")
            log(f"warning: remote cleanup failed ({exc}); sync will retry it")
        return _read_state(run)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _publish_output(run, source_output):
    dest = Path(run) / "output"
    dest.mkdir(exist_ok=True)
    for source in Path(source_output).rglob("*"):
        if source.is_dir():
            continue
        target = dest / source.relative_to(source_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


# --------------------------------------------------------------------------- sync
def default_runs_roots_for_sync(runs_dirs=None):
    roots = default_runs_roots()
    for extra in runs_dirs or []:
        path = Path(extra)
        if path.is_dir() and path not in roots:
            roots.append(path)
    return roots


def sync(roots=None, *, client=None, log=print):
    """Section 5.3: poll in-flight remotes once; fetch finished ones; never submit anything new. Cheap when nothing is pending."""
    roots = roots
    if roots is None:
        roots = default_runs_roots()
    summary = {"checked": 0, "fetched": [], "running": [], "failed": [], "skipped": []}
    candidates = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for run in sorted(root.iterdir()):
            if not run.is_dir() or not (run / "state.json").exists():
                continue
            remote = _read_state(run).get("remote") or {}
            status = remote.get("status")
            if status in ("submitted", "running", "done") or (status == "fetched" and remote.get("cleanup_pending")):
                candidates.append(run)
    if not candidates:
        log("sync: nothing to do")
        return summary
    client = client or _default_client(log)
    for run in candidates:
        summary["checked"] += 1
        try:
            with _run_lock(run, wait_seconds=0):
                state = _read_state(run)
                remote = state.get("remote") or {}
                status = remote.get("status")
                if status == "fetched":
                    if client.delete(f"{spec.RUNS_ROOT}/{remote.get('run_key')}"):
                        _update_remote(run, _read_state(run), cleanup_pending=False, message="")
                        log(f"{run.name}: remote cleanup done")
                    continue
                poll = client.poll(remote.get("call_id"))
                vol = None
                text = client.read_text(f"{spec.RUNS_ROOT}/{remote.get('run_key')}/status.json")
                try:
                    vol = json.loads(text) if text else None
                except ValueError:
                    vol = None
                vstate = (vol or {}).get("state")
                if poll == "running" and vstate not in ("done", "failed", "cancelled"):
                    log(f"{run.name}: running on {remote.get('gpu', '?')} "
                        f"({_elapsed_since(remote.get('submitted_at'))} elapsed)")
                    summary["running"].append(run.name)
                    continue
                if vstate == "done" or poll == "done":
                    fetch(run, client=client, log=log)
                    summary["fetched"].append(run.name)
                    continue
                if vstate in ("failed", "cancelled") or poll in ("failed", "cancelled"):
                    try:
                        _remote_failed(run, poll, vol, client, log)
                    except RuntimeError as exc:
                        summary["failed"].append((run.name, str(exc)))
                    continue
                if poll in ("expired", "unknown"):
                    if vstate == "done":
                        _update_remote(run, _read_state(run), status="done")
                        fetch(run, client=client, log=log)
                        summary["fetched"].append(run.name)
                        continue
                    message = ("Modal no longer knows this call; check the Modal dashboard, then run "
                               "`runner cancel --job ...` to mark it cancelled before re-running")
                    _update_remote(run, _read_state(run), status="unknown", message=message)
                    log(f"{run.name}: {message}")
                    summary["failed"].append((run.name, message))
                    continue
                log(f"{run.name}: unhandled remote state (poll={poll}, volume={vstate})")
                summary["failed"].append((run.name, f"poll={poll} volume={vstate}"))
        except RemoteBusy as exc:
            summary["skipped"].append(run.name)
            log(str(exc))
        except Exception as exc:
            summary["failed"].append((run.name, str(exc)))
            log(f"{run.name}: {exc}")
    log(f"sync: {summary['checked']} checked, {len(summary['fetched'])} fetched, "
        f"{len(summary['running'])} running, {len(summary['failed'])} failed, "
        f"{len(summary['skipped'])} skipped")
    return summary


# --------------------------------------------------------------------------- cancel
def cancel(run, *, client=None, log=print):
    """Section 7.3: cancel the call; if it already ended, fetch the result instead."""
    run = Path(run)
    state = _read_state(run)
    remote = state.get("remote") or {}
    call_id = remote.get("call_id")
    if not call_id:
        raise RuntimeError(f"{run.name}: no submitted call to cancel")
    client = client or _default_client(log)
    try:
        accepted = client.cancel(call_id)
    except Exception as exc:
        raise RuntimeError(
            f"cancel request failed: {exc}; local state is unchanged - check the Modal dashboard") from exc
    if accepted:
        _update_remote(run, _read_state(run), status="cancelled", message="cancelled by user")
        log("cancel requested; the remote folder is kept for a later resume")
        return True
    # The call is already gone - treat it as finished and fetch normally (Section 7.3).
    log("the call was already finished; fetching the result instead")
    _update_remote(run, _read_state(run), status="done")
    fetch(run, client=client, log=log)
    return True


# --------------------------------------------------------------------------- execute (runner dispatch)
def execute(run, cfg, n, *, epochs=None, fingerprint=None, inventory_fingerprint=None,
            client=None, log=print, sleep=time.sleep):
    """The modal branch of runner.execute() - the Section 3.5 dispatch table."""
    run = Path(run)
    epochs = epochs or runner.epochs_for(cfg["target_steps"], n)
    fingerprint = fingerprint or runner.working_fingerprint(run)
    inv = inventory_fingerprint or runner.inventory_fingerprint(run)
    client = client or _default_client(log)
    with _run_lock(run):
        family = runner.family_of(cfg)
        state = _read_state(run)
        state["family"] = family
        state["arch_id"] = runner.model_paths.description(family).arch_id
        if inv:
            state["inventory_fingerprint"] = inv
        _write_state(run, state)
        remote = state.get("remote") or {}
        status = remote.get("status")

        if status == "fetched":
            if runner.output_valid(run, cfg, epochs):
                warning = runner.preview_missing(run, cfg, epochs)
                if warning:
                    state["preview_missing"] = True
                    _write_state(run, state)
                    print(f"Already completed; skipped. {warning}", flush=True)
                else:
                    print("Already completed; skipped.", flush=True)
            else:
                print("warning: the previously fetched outputs are damaged; nothing to refetch - "
                      "the remote folder was already deleted when it was fetched", flush=True)
            return _read_state(run)
        if status == "done":
            return fetch(run, cfg, epochs=epochs, client=client, log=log)
        if status in ("submitted", "running"):
            # Section 7.5: never resubmit an in-flight run; follow it.
            return follow(run, cfg, client=client, log=log, sleep=sleep)
        if status == "unknown":
            vol = client.read_text(f"{spec.RUNS_ROOT}/{remote.get('run_key')}/status.json")
            try:
                vstate = json.loads(vol).get("state") if vol else None
            except ValueError:
                vstate = None
            if vstate == "done":
                _update_remote(run, _read_state(run), status="done")
                return fetch(run, cfg, epochs=epochs, client=client, log=log)
            message = ("Modal does not know this call anymore; check the Modal dashboard, then run "
                       "`runner cancel --job ...` to mark it cancelled before re-running")
            _update_remote(run, _read_state(run), status="unknown", message=message)
            raise RuntimeError(message)
        if cancel_request_path(run).exists():
            try:
                cancel_request_path(run).unlink()
            except OSError:
                pass
            raise RuntimeError("cancelled by user before the run was submitted")

        stale = None
        if status == "stale":
            stale = remote
        elif status in ("failed", "cancelled") and remote.get("dataset_fingerprint") != fingerprint:
            stale = remote
        if stale is None and status in ("failed", "cancelled"):
            # Same input - submit() decides between reusing the stored bundle and a fresh upload.
            submit(run, cfg, n, epochs=epochs, fingerprint=fingerprint,
                   inventory_fingerprint=inv, client=client, log=log)
            return follow(run, cfg, client=client, log=log, sleep=sleep)
        submit(run, cfg, n, epochs=epochs, fingerprint=fingerprint,
               inventory_fingerprint=inv, client=client, log=log)
        if stale:
            _cleanup_stale_remote(run, cfg, stale, client, log, new_key=_run_key(cfg, fingerprint))
        return follow(run, cfg, client=client, log=log, sleep=sleep)


def _cleanup_stale_remote(run, cfg, stale_remote, client, log, new_key=""):
    """Section 3.5: the old remote folder is deleted only once the old call is confirmed terminated."""
    old_key = stale_remote.get("run_key")
    old_call = stale_remote.get("call_id")
    if not old_key:
        return
    # The fingerprint is part of the key, so a new submit never shares it; guard against hand-fabricated state anyway.
    if new_key and old_key == new_key:
        log(f"old remote key {old_key} is the current key; keeping it")
        return
    alive = True
    if old_call:
        try:
            alive = client.poll(old_call) == "running"
        except Exception:
            alive = True   # cannot confirm it stopped - keep the folder
    if alive:
        log(f"old remote run {old_key} may still be running; keeping its folder")
        return
    try:
        client.delete(f"{spec.RUNS_ROOT}/{old_key}")
        log(f"removed stale remote run {old_key}")
    except Exception as exc:
        log(f"warning: could not remove stale remote run {old_key} ({exc}); it lingers on the Volume")


__all__ = ["RemoteBusy", "default_runs_roots", "default_runs_roots_for_sync", "find_runs",
           "estimate", "build_source_tarball", "remote_plan", "submit", "follow", "fetch",
           "sync", "cancel", "execute", "STAGE_NAMES", "REMOTE_STATUSES"]
