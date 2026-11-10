"""The dataset pipeline as a tab in the Fizgig Auto window.

Spec M5 asks for a "데이터셋 생성" tab with project selection, a button per stage, a log, and pause/cancel, without
changing the existing tabs. So this is one self-contained tab that calls the same CLIs and reads the same project files
as the command line - it holds no rules of its own about what may be drawn, scored or exported. Every gate (holdout
exclusion, lineage caps, the chosen epoch, the export policy) lives in the pipeline modules; if this tab were to
reimplement any of them the two entry points would drift apart, and the GUI would become a way around a rule.

Two behaviours are deliberately not shared with the existing queue tab:

* **A stage is a subprocess, not a job.** The existing tab drives `runner.py` over `todo.json` entries; these stages are
  `pipeline.<module>` commands over a project folder. Pause therefore means "stop between cells" for the stages that can
  honour it (`epochs`, `batch_gen`, `fix` all take a cancel callback or file) and Cancel means terminate.
  Nothing here pretends a stage stopped cleanly when it was killed.
* **Nothing starts on its own.** No stage runs on open, on a timer, or because a previous one finished. Every draw is a
  click, because each one spends GPU time and the user is the one who knows whether ComfyUI is free.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import proctree


# Ordered as the pipeline runs. The third field is the module and its arguments, without `--project`, which is added
# from the selected folder. Planning and drawing are separate buttons because the CLIs keep them separate: `run` reads
# the manifest `plan` wrote, and does not plan for you. A label that said "에포크 평가" over a `plan` command would be
# wrong about whether any GPU time had been spent.
STAGES = (
    ("audit", "감사·스냅샷", ("seed", "--build"), "검수 이력과 계보를 검사하고 학습 스냅샷을 만듭니다"),
    ("seed_train", "시드 학습", ("seed", "--build", "--train"), "스냅샷을 만들고 시드 LoRA를 학습합니다(GPU, 몇 시간)"),
    ("gen_a", "생성 A", ("gen_qwen_edit",), "툴2로 편집을 실행하고 결과를 후보로 등록합니다(GPU)"),
    ("gen_ph3", "P-H3 편집", ("ph3_edit",), "P-H3로 부족 슬롯을 편집해 등록합니다(GPU)"),
    ("batch_import", "배치 B 등록", ("batch_gen", "import"), "그려진 배치 B를 후보로 등록합니다(제출 없음)"),
    ("holdout_draw", "홀드아웃 그리기", ("evaluate", "holdout", "--draw"), "홀드아웃 프롬프트를 선택 LoRA로 그립니다(GPU)"),
    ("holdout_score", "홀드아웃 평가", ("evaluate", "holdout", "--score"), "그려진 셀을 채점해 family별 보고서를 냅니다"),
    ("review", "검수 열기", ("review",), "브라우저에서 검수 페이지를 엽니다(서버는 계속 실행)"),
    ("fix", "보정 C 실행", ("fix", "--run"), "보정 표시된 후보를 툴2로 다시 그립니다"),
    ("epochs_list", "에포크 목록", ("epochs", "list"), "학습된 체크포인트를 파일 헤더로 검증해 나열합니다"),
    ("epochs_plan", "에포크 계획", ("epochs", "plan"), "고정 프롬프트×시드 그리드를 계획합니다(그리지 않음)"),
    ("epochs_run", "에포크 그리기", ("epochs", "run"), "계획된 칸을 실제로 그립니다(GPU·중단 가능)"),
    ("epochs_sheet", "비교 시트", ("epochs", "sheet"), "에포크를 나란히 볼 HTML을 만듭니다"),
    ("batch_plan", "배치 B 계획", ("batch_gen", "plan"), "쿼터가 모자란 슬롯만 계획합니다(그리지 않음)"),
    ("batch_run", "배치 B 그리기", ("batch_gen", "run"), "계획된 슬롯을 그립니다(GPU·중단 가능)"),
    ("score", "점수", ("score", "run"), "CLIP·ArcFace로 채점합니다(정렬용, 자동 판정 없음)"),
    ("export", "내보내기", ("export", "publish"), "확정 가능한 것만 불변 스냅샷으로 묶습니다"),
    ("todos", "최종 학습 준비", ("export", "todos", "--add"), "패밀리별 학습 작업을 꺼진 상태로 추가합니다"),
)


def runs_for_project(folder):
    """Every seed run under a project with what makes it usable, read from disk (spec N5).

    `epochs` has the authority over which run may be evaluated (`find_run` refuses a run with no checkpoint), so this
    list is a display: a person has to see that an aborted run exists and that it will not be offered. Anything this
    function gets wrong is caught by the CLI, which is the point of keeping the rule there.
    """
    base = Path(folder) / "seed_lora"
    if not base.is_dir():
        return []
    rows = []
    for path in sorted(base.glob("*/runs/*")):
        if not path.is_dir():
            continue
        output = path / "output"
        count = len(list((output if output.is_dir() else path).glob("*.safetensors")))
        state = {}
        state_file = path / "state.json"
        if state_file.is_file():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = {}
        rows.append({"name": path.name, "folder": str(path), "snapshot": path.parent.parent.name,
                     "checkpoints": count, "usable": count > 0,
                     "complete": bool(state.get("complete")),
                     "stage": state.get("active_stage") or ""})
    return rows


def epochs_for_run(run_folder):
    """`(epoch, usable, note)` for every checkpoint in a run, read without loading any weights (spec N5)."""
    folder = Path(run_folder)
    output = folder / "output"
    base = output if output.is_dir() else folder
    from pipeline import epochs as epochs_mod
    try:
        rows = epochs_mod.checkpoint_rows(base)
    except Exception:  # noqa: BLE001 - a header read must never take the tab down
        return []
    return [{"epoch": row.get("epoch"), "name": row["name"], "usable": bool(row["usable"]),
             "note": row.get("note") or ""} for row in rows]


def exports_for_project(folder):
    """Published exports, newest first, with the counts a person checks before choosing one (spec N5)."""
    base = Path(folder) / "exports"
    if not base.is_dir():
        return []
    rows = []
    for child in sorted(base.iterdir()):
        manifest = child / "export_manifest.json"
        if not child.is_dir() or not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        counts = data.get("counts") or {}
        rows.append({"id": data.get("id") or child.name, "created_at": str(data.get("created_at") or ""),
                     "images": counts.get("images"), "refused": counts.get("refused"),
                     "policy": data.get("policy")})
    rows.sort(key=lambda row: row["created_at"], reverse=True)
    return rows


def project_roots(automation_dir):
    """Every project folder under `automation/projects`, newest first.

    Read from disk rather than from a settings list: a project is a folder with a `project.json`, so one copied in by
    hand must appear here too, and one whose folder was deleted must not.
    """
    base = Path(automation_dir) / "projects"
    if not base.is_dir():
        return []
    found = []
    for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if (child / "project.json").is_file():
            found.append(child)
    found.sort(key=lambda p: (p / "project.json").stat().st_mtime, reverse=True)
    return found


def project_summary(folder):
    """The few facts the tab shows for a project, read from its own files.

    Every value is what is on disk; a missing file leaves its entries None rather than guessing, so the header can say
    "아직 없음" instead of a zero that looks like a real measurement.
    """
    folder = Path(folder)
    card = {}
    try:
        card = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        card = {}
    out = {
        "id": card.get("id") or folder.name,
        "trigger": card.get("trigger") or "",
        "sources": None, "candidates": None, "accepted": None, "undecided": None,
        "holdsout": len((card.get("holdout") or {}).get("ids") or []),
        "epoch": (card.get("krea2_inference") or {}).get("epoch"),
        "families": list(card.get("final_families") or []),
        "export": (card.get("export") or {}).get("selected"),
        "scores": None, "export_count": None,
    }
    try:
        rows = [json.loads(line) for line in
                (folder / "sources.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        out["sources"] = len([r for r in rows if r.get("origin") == "user"])
    except (OSError, ValueError):
        pass
    try:
        rows = [json.loads(line) for line in
                (folder / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        out["candidates"] = len(rows)
    except (OSError, ValueError):
        pass
    try:
        review = json.loads((folder / "review.json").read_text(encoding="utf-8"))
        # `Project.decide` writes {candidate_id: {decision, ...}}; honour the older {"decisions": {...}} shape too,
        # so a file written before that format existed still counts (spec R7).
        decisions = review
        if isinstance(review, dict) and isinstance(review.get("decisions"), dict):
            decisions = review["decisions"]
        if isinstance(decisions, dict):
            out["accepted"] = sum(1 for state in decisions.values()
                                  if isinstance(state, dict) and state.get("decision") == "accept")
            out["undecided"] = max(0, (out["candidates"] or 0) - len(decisions))
    except (OSError, ValueError):
        pass
    try:
        scores = json.loads((folder / "experiments" / "scores.json").read_text(encoding="utf-8"))
        out["scores"] = scores.get("targets")
    except (OSError, ValueError):
        pass
    exports = folder / "exports"
    if exports.is_dir():
        out["export_count"] = len([p for p in exports.iterdir() if p.is_dir() and not p.name.startswith(".")])
    chosen = out.get("export")
    if not chosen and out.get("export_count"):
        # Nothing is recorded as chosen yet; the newest published export is what a re-run would compare against,
        # so its counts are what the header shows (spec R7).
        newest = None
        for child in sorted(exports.iterdir()):
            manifest = child / "export_manifest.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if newest is None or str(data.get("created_at") or "") > str(newest[0]):
                newest = (str(data.get("created_at") or ""), data)
        if newest:
            chosen = newest[1].get("id")
            counts = newest[1].get("counts") or {}
            out["export_counts"] = {"images": counts.get("images"), "refused": counts.get("refused"),
                                    "policy": newest[1].get("policy"), "id": chosen}
    if chosen:
        manifest = exports / str(chosen) / "export_manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            counts = data.get("counts") or {}
            # What the reviewer can actually export, and what the rules refused - the two numbers a person checks
            # against the quota board before publishing (spec R7).
            out["export_counts"] = {"images": counts.get("images"), "refused": counts.get("refused"),
                                    "policy": data.get("policy")}
        except (OSError, ValueError):
            out["export_counts"] = None
    return out


def stage_command(automation_dir, project_dir, spec):
    """The exact argv for one stage, so a test can assert what a click would run.

    `spec` is `(module, *arguments)`. `--project` always goes last and always comes from the selected folder, so no
    stage can be pointed at a project other than the one the header shows.
    """
    python = Path(automation_dir).parent / "venv" / "Scripts" / "python.exe"
    if not python.is_file():
        python = Path(sys.executable)
    return [str(python), "-B", "-u", "-m", "pipeline." + spec[0], *spec[1:], "--project", str(project_dir)]


# The stages whose CLI takes `--cancel-file`, mapped to whether a pause there leaves resumable work.
#
# The audience for this table is the pause button: a module named here gets the marker and is offered "일시정지"; one
# not named is terminated outright, because claiming a killed process paused would be a lie about the data on disk.
# `gen_qwen_edit` and `ph3_edit` were added by spec N2 - both already took `--cancel-file` and poll it at a slot
# boundary, and both leave every finished picture imported-able, so a pause there is a real pause. The resume contract
# holds because each records what it drew before the next slot; a rerun skips what is on disk.
# The stages whose CLI takes `--cancel-file`, mapped to whether a pause there leaves resumable work.
#
# The audience for this table is the pause button: a module named here gets the marker and is offered "일시정지"; one
# not named is terminated outright, because claiming a killed process paused would be a lie about the data on disk.
# `gen_qwen_edit` and `ph3_edit` were added by spec N2 - both already took `--cancel-file` and poll it at a slot
# boundary, and both leave every finished picture imported-able, so a pause there is a real pause. The resume contract
# holds because each records what it drew before the next slot; a rerun skips what is on disk.
CANCELABLE = {"epochs", "batch_gen", "fix", "gen_qwen_edit", "ph3_edit"}

# The two buttons the Modal choice applies to. Everything else stays local by construction.
MODAL_STAGES = ("seed_train", "todos")

# The GUI always asks for this card; it is the L40S price/performance recommendation in the design.
MODAL_GPU = "L40S"

# Where the tab passes the remote-cancel hook to the subprocess (the runner reads it at follow() time).
REMOTE_CANCEL_ENV = "FIZGIG_REMOTE_CANCEL_REQUEST"


# How long a cancelled stage gets to stop on its own before its process tree is ended. A stage that polls its marker
# stops at the next cell boundary; one stuck inside a single long cell needs the tree kill. Kept as a module constant
# rather than a literal so a test can shorten it and a future caller can lengthen it.
CANCEL_GRACE = proctree.DEFAULT_GRACE


class DatasetTab:
    """The "데이터셋 생성" tab. Owns its widgets, its worker and its log; the App owns only the frame."""

    # One background sync per process at window open, shared by the two hosts (a stand-alone window and the main
    # window) so they never schedule two. Resettable in tests.
    _sync_lock = threading.Lock()
    _sync_started = False

    @classmethod
    def _try_claim_initial_sync(cls):
        with cls._sync_lock:
            if cls._sync_started:
                return False
            cls._sync_started = True
            return True

    @classmethod
    def _reset_sync_claim(cls):
        with cls._sync_lock:
            cls._sync_started = False

    def __init__(self, app, parent, *, automation_dir=None):
        self.app = app
        self.parent = parent
        self.auto = Path(automation_dir or getattr(app, "AUTO", None) or Path(__file__).parent)
        self.events = queue.Queue()
        self.process = None
        self.worker = None
        self.running = None            # the stage key currently running
        self.running_backend = None    # "modal" or None for the current stage (Modal stages only)
        self.cancel_requested = False
        self.pause_requested = False
        self.cancel_file = None
        self.remote_cancel_file = None
        self.cancel_report = None      # what the tree kill reported, once the cancel worker has finished
        self.pause_noted = False       # set when the stage's own output says it stopped for a pause
        self.started_at = None
        self.last_log_path = None      # the file this run's output was also written to
        self.log_lines = []
        self.project = tk.StringVar()
        self.location = tk.StringVar(value="로컬")
        self.run = tk.StringVar()           # the seed run the epoch stages work on (spec N5)
        self.epoch = tk.StringVar()         # the epoch a person picked in this tab (spec N5)
        self.export_choice = tk.StringVar()  # the published export the final todo step would use (spec N5)
        self.status = tk.StringVar(value="프로젝트를 고르세요")
        self.info = tk.StringVar(value="")
        self._buttons = {}
        self._runs = []
        self._epochs = []
        self._exports = []
        self._run_labels = {}
        self._epoch_labels = {}
        self._export_labels = {}
        self.run_combo = None          # set by `build()`; a test that skips it also skips the pick refresh
        self.build()
        self.refresh_projects()
        self.after_ids = []
        self._schedule_poll()
        self._start_initial_sync()

    # ---------------------------------------------------------------- widgets

    def build(self):
        tab = self.parent
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(5, weight=1)

        header = ttk.Frame(tab)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="프로젝트", style="Heading.TLabel").pack(side="left")
        self.combo = ttk.Combobox(header, textvariable=self.project, state="readonly", width=34)
        self.combo.pack(side="left", padx=10)
        self.combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_info())
        self.app.controls.append((self.combo, "readonly"))
        self.app.button(header, "새로고침", self.refresh_projects).pack(side="left")
        ttk.Label(header, text="학습 위치", style="Muted.TLabel").pack(side="right", padx=(8, 0))
        self.location_combo = ttk.Combobox(header, textvariable=self.location, state="readonly", width=9,
                                           values=("로컬", "Modal"))
        self.location_combo.pack(side="right", padx=(4, 12))
        self.app.controls.append((self.location_combo, "readonly"))
        ttk.Label(header, textvariable=self.status, style="Muted.TLabel").pack(side="right")

        ttk.Label(tab, textvariable=self.info, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 4))

        # Run / epoch / export selection (spec N5). These are choices a person makes once and every stage after them
        # uses: the epoch buttons all take the same run, so a project with more than one run never has to be typed at
        # a terminal, and an epoch is only ever recorded by an explicit click here.
        picks = ttk.Frame(tab)
        picks.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(picks, text="학습 런", style="Heading.TLabel").pack(side="left")
        self.run_combo = ttk.Combobox(picks, textvariable=self.run, state="readonly", width=40)
        self.run_combo.pack(side="left", padx=(8, 4))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_run_details())
        self.app.controls.append((self.run_combo, "readonly"))
        ttk.Label(picks, text="에포크", style="Heading.TLabel").pack(side="left", padx=(16, 0))
        self.epoch_combo = ttk.Combobox(picks, textvariable=self.epoch, state="readonly", width=14)
        self.epoch_combo.pack(side="left", padx=(8, 4))
        self.app.controls.append((self.epoch_combo, "readonly"))
        self.choose_button = self.app.button(picks, "이 에포크 선택", self.choose_epoch)
        self.choose_button.pack(side="left", padx=4)
        self.app.controls.append((self.choose_button, "normal"))
        ttk.Label(picks, text="게시본", style="Heading.TLabel").pack(side="left", padx=(16, 0))
        self.export_combo = ttk.Combobox(picks, textvariable=self.export_choice, state="readonly", width=22)
        self.export_combo.pack(side="left", padx=(8, 4))
        self.app.controls.append((self.export_combo, "readonly"))

        ttk.Label(tab, text="단계", style="Heading.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 6))
        grid = ttk.Frame(tab)
        grid.grid(row=4, column=0, sticky="ew")
        columns = 4
        for index, (key, label, _spec, hint) in enumerate(STAGES):
            button = self.app.button(grid, label, lambda k=key: self.start(k))
            button.grid(row=index // columns, column=index % columns, sticky="ew", padx=(0, 8), pady=3)
            self._buttons[key] = button
            button.bind("<Enter>", lambda _e, text=hint: self.status.set(text))
            button.bind("<Leave>", lambda _e: self.status.set(self._idle_status()))
        for column in range(columns):
            grid.columnconfigure(column, weight=1)

        actions = ttk.Frame(tab)
        actions.grid(row=6, column=0, sticky="ew", pady=(10, 6))
        self.pause_button = ttk.Button(actions, text="일시정지 요청", command=self.request_pause, state="disabled")
        self.pause_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="취소", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        ttk.Label(actions, text="일시정지는 다음 항목부터 멈춥니다(칸 단위). 취소는 즉시 종료하고, "
                               "이미 그린 그림은 남습니다.",
                  style="Muted.TLabel").pack(side="left", padx=8)

        self.log = tk.Text(tab, height=12, bg="#0e1116", fg="#8ea", insertbackground="#8ea",
                           relief="flat", font=("Consolas", 9), state="disabled", wrap="none")
        self.log.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        scroll = ttk.Scrollbar(tab, orient="vertical", command=self.log.yview)
        scroll.grid(row=5, column=1, sticky="ns", pady=(6, 0))
        self.log.configure(yscrollcommand=scroll.set)
        tab.rowconfigure(5, weight=1)

    def _schedule_poll(self):
        # A callback that fires after the app is torn down is the "invalid command name ..._poll" warning; stop
        # scheduling once the interpreter is going away (spec R7).
        try:
            if not self.parent.winfo_exists():
                return
        except tk.TclError:
            return
        job = self.parent.after(150, self._poll)
        self.after_ids.append(job)

    def _poll(self):
        try:
            alive = self.parent.winfo_exists()
        except tk.TclError:
            return
        if not alive:
            return
        self._drain_events()
        self._schedule_poll()

    def _drain_events(self):
        finished = None
        for _ in range(200):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self._append(payload)
            elif kind == "sync_line":
                self._append(payload)
            elif kind == "sync_done":
                if payload == 0:
                    self._append("— sync 완료: 회수된 런은 목록에 반영됩니다\n")
                else:
                    self._append(f"— sync 실패 (종료 코드 {payload}) · 위 로그에서 원인을 확인하세요\n")
                self.refresh_projects()
                self.refresh_info()
            elif kind == "process":
                self.process = payload
            elif kind == "cancel_report":
                # The tree kill finished; `_finish` reports from it. Recorded rather than acted on, because the
                # process's own exit is what ends the stage.
                self.cancel_report = payload
            elif kind == "finished":
                finished = payload
        if finished is not None:
            self._finish(finished)

    def destroy(self):
        for job in self.after_ids:
            try:
                self.parent.after_cancel(job)
            except tk.TclError:
                pass
        self.after_ids = []

    # ---------------------------------------------------------------- projects

    def refresh_projects(self):
        roots = project_roots(self.auto)
        names = [p.name for p in roots]
        self._roots = dict(zip(names, roots))
        self.combo.configure(values=names)
        if names and self.project.get() not in names:
            self.project.set(names[0])
        self.refresh_info()

    def project_dir(self):
        """The folder the selected project names, or None.

        Returns None rather than a default folder: a stage that silently ran against the wrong project would write
        candidates into someone else's dataset.
        """
        name = self.project.get()
        if not name:
            return None
        return self._roots.get(name)

    def refresh_info(self):
        folder = self.project_dir()
        if folder is None:
            self.info.set("프로젝트가 없습니다. automation/projects/<이름>/project.json 이 필요합니다.")
            self.status.set("프로젝트 없음")
            self._set_enabled({key: False for key, *_ in STAGES})
            return
        summary = project_summary(folder)
        bits = [f'{summary["id"]}']
        if summary["trigger"]:
            bits.append(f'트리거 {summary["trigger"]}')
        if summary["sources"] is not None:
            bits.append(f'원본 {summary["sources"]}')
        if summary["candidates"] is not None:
            bits.append(f'후보 {summary["candidates"]}')
        if summary["accepted"] is not None:
            bits.append(f'채택 {summary["accepted"]}')
        if summary["undecided"]:
            bits.append(f'미검수 {summary["undecided"]}')
        bits.append(f'홀드아웃 {summary["holdsout"]}')
        bits.append(f'에포크 {summary["epoch"] if summary["epoch"] is not None else "미선택"}')
        bits.append(f'scores.json {summary["scores"] if summary["scores"] is not None else "없음"}')
        bits.append(f'export {summary["export_count"] or 0}')
        if summary["families"]:
            bits.append("패밀리 " + ",".join(summary["families"]))
        self.info.set(" · ".join(bits))
        self.status.set(self._idle_status())
        self.refresh_picks(folder)
        self._apply_stage_rules(summary)

    # ------------------------------------------------------------- run/epoch/export picks

    def refresh_picks(self, folder=None):
        """Fill the run, epoch and export drop-downs from what is on disk (spec N5).

        The run list shows a run with zero checkpoints and marks it, because a person needs to see that it exists and
        that it will not be offered - the audit's `aborted_run_blocks_find_run` was exactly that run forcing a `--run`
        on every command. The selection is preserved when it is still valid, and falls back to the only usable run.

        A tab built without the pick widgets (a test that only exercises the stage rules) skips this: the rules are
        computed from the project's card, not from the drop-downs.
        """
        if getattr(self, "run_combo", None) is None:
            return
        folder = folder or self.project_dir()
        if folder is None:
            self._runs, self._epochs, self._exports = [], [], []
            self._set_pick_values(self.run_combo, self.run, [])
            self._set_pick_values(self.epoch_combo, self.epoch, [])
            self._set_pick_values(self.export_combo, self.export_choice, [])
            return
        self._runs = runs_for_project(folder)
        labels = []
        for row in self._runs:
            mark = "" if row["usable"] else " (체크포인트 없음)"
            labels.append(f'{row["name"]}{mark}')
        self._run_labels = {label: row for label, row in zip(labels, self._runs)}
        current = self.run.get()
        values = labels
        if current not in labels:
            usable = [label for label, row in zip(labels, self._runs) if row["usable"]]
            current = usable[0] if usable else (labels[0] if labels else "")
            self.run.set(current)
        self.run_combo.configure(values=values)
        self.refresh_run_details()

        exports = exports_for_project(folder)
        self._exports = exports
        export_labels = [f'{row["id"]} · {row["images"]}장' for row in exports]
        self._export_labels = dict(zip(export_labels, exports))
        if self.export_choice.get() not in export_labels:
            chosen = project_summary(folder).get("export")
            prefs = [label for label, row in zip(export_labels, exports) if row["id"] == chosen]
            self.export_choice.set(prefs[0] if prefs else (export_labels[0] if export_labels else ""))
        self.export_combo.configure(values=export_labels)

    def refresh_run_details(self):
        """The epoch drop-down for the chosen run, from header-checked checkpoints only."""
        row = self._run_labels.get(self.run.get()) if getattr(self, "_run_labels", None) else None
        if row is None:
            self._epochs = []
            self._set_pick_values(self.epoch_combo, self.epoch, [])
            return
        self._epochs = epochs_for_run(row["folder"])
        labels = [f'{item["epoch"]} ({item["name"]})' if item["usable"] else
                  f'{item["epoch"] if item["epoch"] is not None else "?"} (사용 불가: {item["note"][:40]})'
                  for item in self._epochs]
        self._epoch_labels = {label: item for label, item in zip(labels, self._epochs)}
        if self.epoch.get() not in labels:
            usable = [label for label, item in zip(labels, self._epochs) if item["usable"]]
            self.epoch.set(usable[-1] if usable else (labels[-1] if labels else ""))
        self.epoch_combo.configure(values=labels)

    def _set_pick_values(self, combo, variable, values):
        combo.configure(values=values)
        if variable.get() not in values:
            variable.set(values[0] if values else "")

    def chosen_epoch(self):
        """The epoch number this tab would record, or None when the drop-down does not name a usable checkpoint."""
        item = getattr(self, "_epoch_labels", {}).get(self.epoch.get())
        if item is None or not item["usable"] or item["epoch"] is None:
            return None
        return int(item["epoch"])

    def choose_epoch(self):
        """Record the selected epoch through the CLI's own `choose` (spec N5).

        A grading decision is a person's and only an explicit click here records one: opening the sheet, sorting it or
        changing the drop-down never writes. The CLI is what writes the card, so the rule (checkpoint bytes unchanged,
        training snapshot, run id) is the same one the command line enforces.
        """
        if self.running is not None:
            return
        folder = self.project_dir()
        row = getattr(self, "_run_labels", {}).get(self.run.get()) if getattr(self, "_run_labels", None) else None
        epoch = self.chosen_epoch()
        if folder is None or row is None or epoch is None:
            messagebox.showinfo("에포크 선택", "쓸 수 있는 체크포인트를 먼저 고르세요.", parent=self.parent)
            return
        argv = stage_command(self.auto, folder, ("epochs", "choose", "--run", row["name"], "--epoch", str(epoch)))
        self._run_stage("epochs_choose", "에포크 선택", argv, cancelable=False)

    def _export_selected(self):
        row = getattr(self, "_export_labels", {}).get(self.export_choice.get())
        return row["id"] if row else None


    def _idle_status(self):
        return "준비됨" if self.running is None else f"{self._label(self.running)} 실행 중"

    def _label(self, key):
        """The button's own label, or the key itself for an action that is not one of the stage buttons."""
        return next((label for name, label, _s, _h in STAGES if name == key), key or "")

    def _apply_stage_rules(self, summary):
        """Disable what cannot run yet, so the reason is visible before a GPU job is queued.

        These mirror gates the CLIs already enforce - batch B refuses without a chosen epoch, drawing refuses without a
        checksum-verified checkpoint. They only grey out a button; the CLI still refuses, which is what the tests check.
        """
        states = {key: True for key, *_ in STAGES}
        if summary.get("epoch") is None:
            # `batch_gen.character_lora_path` raises without a selected epoch; failing here saves queueing a job that
            # would have to be thrown away.
            states["batch_plan"] = False
            states["batch_run"] = False
        if self.running is not None:
            states = {key: False for key in states}
        self._set_enabled(states)
        if self.running is None:
            if summary.get("epoch") is None:
                self.status.set("준비됨 · 배치 B는 에포크를 고른 뒤에 쓸 수 있습니다")
            else:
                self.status.set("준비됨")

    def _set_enabled(self, states):
        for key, button in self._buttons.items():
            button.configure(state="normal" if states.get(key, True) else "disabled")

    # ---------------------------------------------------------------- running

    def start(self, key):
        """Run one stage from the button row. The picked run is passed to every epoch stage (spec N5)."""
        if self.running is not None:
            return
        folder = self.project_dir()
        if folder is None:
            messagebox.showinfo("프로젝트 선택", "먼저 프로젝트를 고르세요.", parent=self.parent)
            return
        spec = next((s for name, _label, s, _hint in STAGES if name == key), None)
        if spec is None:
            return
        spec = self._with_picked_export(self._with_picked_run(spec))
        location = getattr(self, "location", None)
        modal = bool(location is not None and location.get() == "Modal" and key in MODAL_STAGES)
        if modal:
            spec = spec + ("--backend", "modal", "--modal-gpu", MODAL_GPU)
            if key == "seed_train":
                self._append("학습 위치: Modal (krea2, L40S) - 제출 후 이 창을 닫아도 원격에서 계속 진행합니다.\n")
            else:
                self._append("학습 위치: Modal - todo가 backend=modal로 추가됩니다. 학습은 실행할 때 원격에서 진행됩니다.\n")
        argv = stage_command(self.auto, folder, spec)
        self.running_backend = "modal" if (modal and key == "seed_train") else None
        self.remote_cancel_file = (folder / "cancel_remote.request") if self.running_backend == "modal" else None
        self._run_stage(key, self._label(key), argv, cancelable=spec[0] in CANCELABLE)

    def _with_picked_run(self, spec):
        """Append `--run <name>` to the epoch stages.

        Every epoch button goes through this, so the whole run -> plan -> draw -> sheet path works on the run the
        person picked. The audit's `gui_epoch_default` was this: the buttons passed no `--run`, and on the real project
        `find_run()` refused with "more than one run ... name one with --run" - so the entire epoch path was unusable
        from the GUI even though the CLI would have worked.
        """
        if not spec or spec[0] != "epochs":
            return spec
        if len(spec) > 1 and spec[1] == "choose":
            return spec                      # `choose_epoch` builds its own arguments
        row = getattr(self, "_run_labels", {}).get(self.run.get()) if getattr(self, "_run_labels", None) else None
        if row is None:
            return spec
        return spec + ("--run", row["name"])

    def _with_picked_export(self, spec):
        """Append `--id <export>` to the final-training step, so it plans jobs for the export the drop-down shows.

        Without it `export todos` falls back to the newest published export: picking an older one and clicking
        "최종 학습 준비" added a training job for a different set than the one on screen. With no export picked the
        CLI's own fallback (and its "nothing published" error) still applies.
        """
        if tuple(spec[:2]) != ("export", "todos"):
            return spec
        export_id = self._export_selected()
        return spec + ("--id", export_id) if export_id else spec

    def _run_stage(self, key, label, argv, *, cancelable):
        """Launch one subprocess and mark the tab busy. Shared by the buttons and the epoch chooser."""
        folder = self.project_dir()
        self.cancel_requested = False
        self.pause_requested = False
        self.pause_noted = False
        self.last_log_path = None
        # A stop file per run, named for the stage and the moment (spec N2). A fixed `<project>/stop_requested` was
        # left behind whenever the window closed after a pause request, and the next run of that stage saw it
        # immediately, drew nothing and exited 0 - which the tab then reported as "완료". A path no earlier run could
        # have written cannot be inherited that way, and clearing the stage's own previous marker keeps a single
        # abort from poisoning the next attempt either.
        self.cancel_file = folder / f"stop_requested.{key}" if cancelable and folder is not None else None
        if self.cancel_file is not None:
            try:
                self.cancel_file.unlink(missing_ok=True)

            except OSError:
                pass
        self.running = key
        self.started_at = time.time()
        self._append(f"\n▶ {label}\n$ {' '.join(argv)}\n")
        self.status.set(f"{label} 실행 중")
        self._set_enabled({name: False for name, *_ in STAGES})
        # Only a stage that reads the stop file can be paused honestly; the rest offer Cancel alone.
        pausable = self.cancel_file is not None
        self.pause_button.configure(state="normal" if pausable else "disabled")
        self.cancel_button.configure(state="normal")
        self.app.set_busy(True)

        events = self.events
        cancel_file = self.cancel_file
        if pausable:
            argv = list(argv)
            argv[-2:-2] = ["--cancel-file", str(cancel_file)]

        def work():
            try:
                # The two packages the stages need live in different folders: `pipeline` in `automation/`, `fizgig` in
                # `src/`. So the working directory is the repo root and `automation/` goes on PYTHONPATH; without that,
                # `pipeline.epochs run` finds its own module but then fails importing `fizgig.families.registry`.
                root = Path(self.auto).parent
                search = os.pathsep.join((str(self.auto), str(root / "src")))
                inherited = os.environ.get("PYTHONPATH")
                environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                               "PYTHONPATH": search + (os.pathsep + inherited if inherited else "")}
                remote_cancel = getattr(self, "remote_cancel_file", None)
                if remote_cancel is not None:
                    try:
                        Path(remote_cancel).unlink(missing_ok=True)
                    except OSError:
                        pass
                    environment[REMOTE_CANCEL_ENV] = str(remote_cancel)
                with subprocess.Popen(argv, cwd=str(root), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                                      bufsize=1, stdin=subprocess.DEVNULL, env=environment,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as proc:
                    events.put(("process", proc))
                    if proc.stdout is not None:
                        for line in proc.stdout:
                            events.put(("line", line))
                    events.put(("finished", proc.wait()))
            except Exception as exc:  # noqa: BLE001 - a launch failure must reach the log, not vanish in a thread
                events.put(("line", f"실행 오류: {exc}\n"))
                events.put(("finished", 1))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def request_pause(self):
        """Ask the stage to stop between cells.

        Written as a file the stage can see rather than as a signal: the pipeline modules check for it at a cell
        boundary, so a pause leaves a complete picture behind instead of a half-drawn one.
        """
        if self.running is None or self.cancel_file is None:
            return
        try:
            self.cancel_file.write_text("pause\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("일시정지", str(exc), parent=self.parent)
            return
        # Remembered so the verdict can say "일시정지됨" instead of "완료". The stage exits 0 after a pause, so the exit
        # code alone cannot tell a finished grid from a stopped one.
        self.pause_requested = True
        self.status.set("일시정지 요청됨 · 다음 항목에서 멈춥니다")
        self._append("… 일시정지 요청: 다음 항목부터 멈춥니다\n")
        self.pause_button.configure(state="disabled")

    def cancel(self):
        """Stop the stage and everything it started (spec N2).

        The stage's own children are the point: `seed --train` runs the runner which runs the trainer, and generator A
        runs tool2. Terminating only the child this tab started left those drawing on the card - the audit's
        `gui_cancel_child`. So the marker is written first (a stage that polls it stops between cells and keeps what it
        drew), and after a short grace the whole tree is ended. The report is put in the log, and a survivor is
        reported as a survivor rather than as a clean stop.
        """
        if self.running is None:
            return
        remote_cancel = getattr(self, "remote_cancel_file", None)
        if remote_cancel is not None:
            # A cloud stage is cancelled by asking the runner to cancel the remote call (spec Section 7.3):
            # the local tree is left alone, the remote checkpoints stay, and the order of events does the rest.
            try:
                Path(remote_cancel).write_text("cancel\n", encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("취소", str(exc), parent=self.parent)
                return
            self.cancel_requested = True
            self.cancel_report = {"remote": True}
            self._append("… 원격 취소 요청 전송: 러너가 확인하면 Modal 작업을 취소합니다\n")
            self.status.set("원격 취소 대기 중")
            self.cancel_button.configure(state="disabled")
            return
        self.cancel_requested = True
        self._append("… 취소 요청: 정상 중단을 요청하고, 남으면 소유한 프로세스 트리를 종료합니다\n")
        self.status.set("취소 중")
        self.cancel_button.configure(state="disabled")
        proc = self.process
        pid = getattr(proc, "pid", None)
        marker = self.cancel_file
        self.cancel_report = {"in_flight": True, "graceful": False, "killed": False, "alive": [],
                              "cancel_file": str(marker) if marker else None}

        def work():
            # Run the graceful-then-tree stop off the Tk thread: the wait is up to the grace period, and freezing the
            # window during it would read as a hang. A pid that never arrived (the click landed before the worker
            # posted the handle) falls back to the marker alone, which the stage still honours at its next cell.
            report = proctree.stop_tree(pid, cancel_file=marker, grace=CANCEL_GRACE,
                                        log=lambda text: self.events.put(("line", text + "\n")))
            report["in_flight"] = False
            # Assigned here rather than queued: the stage's exit event can be drained before a queued report, and
            # `_finish` would then describe the cancel from the "still running" placeholder.
            self.cancel_report = report

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _finish(self, code):
        key = self.running
        self.running = None
        self.process = None
        self.pause_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        # The run's own marker is removed so a later run cannot find it. It is this run's file by construction (the
        # stage name and the moment are in the path), so nothing another run wrote is touched here.
        if self.cancel_file is not None:
            try:
                self.cancel_file.unlink(missing_ok=True)
            except OSError:
                pass
        self.app.set_busy(False)
        self.running_backend = None
        self.remote_cancel_file = None
        report = self.cancel_report or {}
        if self.cancel_requested:
            if report.get("remote"):
                verdict = f"{self._label(key)} 원격 취소됨 · 체크포인트는 남아 있습니다"
            else:
                verdict = f"{self._label(key)} 취소됨 · 이미 그린 항목은 남아 있습니다"
            if report.get("alive"):
                # Saying "취소됨" over a surviving process would be the same lie as calling a crash a pause.
                verdict += f' · 경고: {len(report["alive"])}개 프로세스가 남아 있습니다 {report["alive"]}'
        elif code == 0 and (self.pause_requested or self.pause_noted):
            verdict = f"{self._label(key)} 일시정지됨 · 다시 누르면 이어서 그립니다"
        elif code == 0:
            verdict = f"{self._label(key)} 완료"
        else:
            # A non-zero exit is only called a pause when exit 0 followed a pause request; anything else is a failure,
            # because reporting a crash as a clean stop is how a broken stage looks finished.
            verdict = f"{self._label(key)} 실패 (종료 코드 {code}) · 위 로그에서 원인을 확인하세요"
        self.cancel_report = None
        # The verdict is the last-run state and must survive the refresh (spec N5): `refresh_info()` recomputes the
        # header and its own idle status, so it runs first and the verdict is set afterwards. Before this the failure
        # string was overwritten with "준비됨" one line later.
        self.refresh_info()
        self.status.set(verdict)
        self._append(f"■ {verdict}\n")


    # ---------------------------------------------------------------- initial sync
    def _start_initial_sync(self):
        """Schedule the one background `runner sync` per process at window open (spec Section 6/Phase 6)."""
        if DatasetTab._try_claim_initial_sync():
            self._launch_initial_sync()

    def _launch_initial_sync(self):
        """Spawn `runner.py sync` in the background; its lines and the finish come back through the event queue."""
        root = Path(self.auto).parent
        python = root / "venv" / "Scripts" / "python.exe"
        if not Path(python).is_file():
            python = Path(sys.executable)
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                       "PYTHONPATH": os.pathsep.join((str(self.auto), str(root / "src")))}

        def work():
            try:
                with subprocess.Popen([str(python), "-B", "-u", str(Path(self.auto) / "runner.py"), "sync"],
                                      cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1,
                                      stdin=subprocess.DEVNULL, env=environment,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as proc:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.events.put(("sync_line", line))
                    code = proc.wait()
            except Exception as exc:  # noqa: BLE001 - a broken sync must never take the tab down
                self.events.put(("sync_line", f"sync 실행 오류: {exc}\n"))
                code = 1
            self.events.put(("sync_done", code))

        threading.Thread(target=work, daemon=True, name="fizgig-initial-sync").start()

    def terminate_local(self):
        """End only this window's tracking of a cloud stage; the remote run keeps going (spec Section 7.4)."""
        if self.running is None:
            return False
        self.cancel_requested = False
        self.running_backend = None
        self.remote_cancel_file = None
        self._append("로컬 추적만 종료합니다 - 원격에서 계속 진행 중이며, 다음에 열 때 결과를 받습니다\n")
        self.status.set("로컬 추적 종료됨 · 원격은 계속 진행 중")
        proc = self.process
        pid = getattr(proc, "pid", None)
        proctree.stop_tree(pid, grace=1, log=lambda text: self._append(text + "\n"))
        return True

    def _append(self, text):
        if not text:
            return
        self.log.configure(state="normal")
        self.log.insert("end", text)
        # Bound the widget the same way the existing log does; an overnight draw prints a lot of lines.
        if int(self.log.index("end-1c").split(".")[0]) > 4000:
            self.log.delete("1.0", "1001.0")
        self.log.see("end")
        self.log.configure(state="disabled")
        self._write_log(text)
        # A stage that stopped because of the pause marker says so in its own output. Reading that here is what lets
        # the verdict say "일시정지" for a stage whose marker was written by a previous window (spec N2): the exit code
        # is 0 either way, so the word in the log is the only evidence that this run stopped early rather than finished.
        if not self.pause_noted and _looks_paused(text):
            self.pause_noted = True

    def _write_log(self, text):
        """Append this run's output to a file under the project (spec N5).

        The tab's log lives in a Tk widget, so closing the window erased the only record of a stage's output - the
        third seed run died and there was no file to say why. Each run writes `logs/<stage>-<stamp>.log` beside the
        project's other products, named so two runs of the same stage cannot collide.
        """
        if self.running is None:
            return
        path = self.log_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text if text.endswith("\n") else text + "\n")
        except OSError:
            pass                        # a read-only project must not take the stage down over a log line

    def log_path(self):
        """This run's log file, or None before a stage starts. Created lazily by `_write_log`."""
        folder = self.project_dir()
        if folder is None or self.running is None:
            return None
        if self.last_log_path is not None:
            return self.last_log_path
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started_at or time.time()))
        self.last_log_path = folder / "logs" / f"{self.running}-{stamp}.log"
        return self.last_log_path


def _looks_paused(text):
    """Does this line say the stage stopped for a pause?

    The pipeline's own wording is the contract: `epochs`, `batch_gen` and `ph3_edit` print "paused"/"cancelled" when
    their cancel hook fires. Matching it is looser than passing a flag, but the alternative - inferring a pause from
    exit 0 - is exactly what made a half-drawn grid look finished.
    """
    lowered = text.lower()
    return "paused:" in lowered or "cancelled" in lowered or "취소됨" in text

