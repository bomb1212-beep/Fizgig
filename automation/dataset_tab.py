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
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


# Ordered as the pipeline runs. The third field is the module and its arguments, without `--project`, which is added
# from the selected folder. Planning and drawing are separate buttons because the CLIs keep them separate: `run` reads
# the manifest `plan` wrote, and does not plan for you. A label that said "에포크 평가" over a `plan` command would be
# wrong about whether any GPU time had been spent.
STAGES = (
    ("audit", "감사·스냅샷", ("seed", "--build"), "검수 이력과 계보를 검사하고 학습 스냅샷을 만듭니다"),
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
        decisions = review.get("decisions") if isinstance(review, dict) else None
        if isinstance(decisions, dict):
            out["accepted"] = sum(1 for state in decisions.values() if (state or {}).get("decision") == "accept")
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


# The stages whose CLI takes `--cancel-file`. A stage not listed here runs to completion or is terminated; claiming it
# paused would report a killed process as a clean stop.
CANCELABLE = {"epochs", "batch_gen", "fix"}


class DatasetTab:
    """The "데이터셋 생성" tab. Owns its widgets, its worker and its log; the App owns only the frame."""

    def __init__(self, app, parent, *, automation_dir=None):
        self.app = app
        self.parent = parent
        self.auto = Path(automation_dir or getattr(app, "AUTO", None) or Path(__file__).parent)
        self.events = queue.Queue()
        self.process = None
        self.worker = None
        self.running = None            # the stage key currently running
        self.cancel_requested = False
        self.pause_requested = False
        self.cancel_file = None
        self.log_lines = []
        self.project = tk.StringVar()
        self.status = tk.StringVar(value="프로젝트를 고르세요")
        self.info = tk.StringVar(value="")
        self._buttons = {}
        self.build()
        self.refresh_projects()
        self.after_ids = []
        self._schedule_poll()

    # ---------------------------------------------------------------- widgets

    def build(self):
        tab = self.parent
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

        header = ttk.Frame(tab)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="프로젝트", style="Heading.TLabel").pack(side="left")
        self.combo = ttk.Combobox(header, textvariable=self.project, state="readonly", width=34)
        self.combo.pack(side="left", padx=10)
        self.combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_info())
        self.app.controls.append((self.combo, "readonly"))
        self.app.button(header, "새로고침", self.refresh_projects).pack(side="left")
        ttk.Label(header, textvariable=self.status, style="Muted.TLabel").pack(side="right")

        ttk.Label(tab, textvariable=self.info, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 4))

        ttk.Label(tab, text="단계", style="Heading.TLabel").grid(row=2, column=0, sticky="w", pady=(10, 6))
        grid = ttk.Frame(tab)
        grid.grid(row=3, column=0, sticky="ew")
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
        actions.grid(row=5, column=0, sticky="ew", pady=(10, 6))
        self.pause_button = ttk.Button(actions, text="일시정지 요청", command=self.request_pause, state="disabled")
        self.pause_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="취소", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        ttk.Label(actions, text="일시정지는 다음 항목부터 멈춥니다(칸 단위). 취소는 즉시 종료하고, "
                               "이미 그린 그림은 남습니다.",
                  style="Muted.TLabel").pack(side="left", padx=8)

        self.log = tk.Text(tab, height=12, bg="#0e1116", fg="#8ea", insertbackground="#8ea",
                           relief="flat", font=("Consolas", 9), state="disabled", wrap="none")
        self.log.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        scroll = ttk.Scrollbar(tab, orient="vertical", command=self.log.yview)
        scroll.grid(row=4, column=1, sticky="ns", pady=(6, 0))
        self.log.configure(yscrollcommand=scroll.set)

    def _schedule_poll(self):
        job = self.parent.after(150, self._poll)
        self.after_ids.append(job)

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
        self._apply_stage_rules(summary)

    def _idle_status(self):
        return "준비됨" if self.running is None else f"{self._label(self.running)} 실행 중"

    def _label(self, key):
        return next((label for name, label, _s, _h in STAGES if name == key), key)

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
        if self.running is not None:
            return
        folder = self.project_dir()
        if folder is None:
            messagebox.showinfo("프로젝트 선택", "먼저 프로젝트를 고르세요.", parent=self.parent)
            return
        spec = next((s for name, _label, s, _hint in STAGES if name == key), None)
        if spec is None:
            return
        argv = stage_command(self.auto, folder, spec)
        self.cancel_requested = False
        self.pause_requested = False
        self.cancel_file = folder / "stop_requested"
        self.running = key
        self._append(f"\n▶ {self._label(key)}\n$ {' '.join(argv)}\n")
        self.status.set(f"{self._label(key)} 실행 중")
        self._set_enabled({name: False for name, *_ in STAGES})
        # Only a stage that reads the stop file can be paused honestly; the rest offer Cancel alone.
        cancelable = spec[0] in CANCELABLE
        self.pause_button.configure(state="normal" if cancelable else "disabled")
        self.cancel_button.configure(state="normal")
        self.app.set_busy(True)

        events = self.events
        cancel_file = self.cancel_file
        if cancelable:
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
        if self.running is None:
            return
        self.cancel_requested = True
        proc = self.process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        self._append("… 취소 요청: 즉시 종료합니다\n")
        self.status.set("취소 중")
        self.cancel_button.configure(state="disabled")

    def _poll(self):
        finished = None
        for _ in range(200):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self._append(payload)
            elif kind == "process":
                self.process = payload
            elif kind == "finished":
                finished = payload
        if finished is not None:
            self._finish(finished)
        self._schedule_poll()

    def _finish(self, code):
        key = self.running
        self.running = None
        self.process = None
        self.pause_button.configure(state="disabled")
        self.cancel_button.configure(state="disabled")
        if self.cancel_file is not None:
            self.cancel_file.unlink(missing_ok=True)
        self.app.set_busy(False)
        if self.cancel_requested:
            verdict = f"{self._label(key)} 취소됨 · 이미 그린 항목은 남아 있습니다"
        elif code == 0 and self.pause_requested:
            verdict = f"{self._label(key)} 일시정지됨 · 다시 누르면 이어서 그립니다"
        elif code == 0:
            verdict = f"{self._label(key)} 완료"
        else:
            # A non-zero exit is only called a pause when exit 0 followed a pause request; anything else is a failure,
            # because reporting a crash as a clean stop is how a broken stage looks finished.
            verdict = f"{self._label(key)} 실패 (종료 코드 {code}) · 위 로그에서 원인을 확인하세요"
        self.status.set(verdict)
        self._append(f"■ {verdict}\n")
        self.refresh_info()

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
