"""Fizgig Auto desktop: local vision captions, editable LLM profiles, and serial LoRA jobs."""
import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

import proctree
import runner
import dataset_tab
from local_caption import LocalClient

ROOT, AUTO = runner.ROOT, runner.AUTO
BG, CARD, INPUT = "#10151e", "#1b2330", "#111923"
TEXT, MUTED, ACCENT, BORDER = "#edf2fa", "#97a8bd", "#63d5b1", "#304055"
DEFAULT_EXE = "C:/LLM/models/Qwen3.8-Flash-Next/runtime/b11020/llama-server.exe"

def llm_values(original, raw, auto_start):
    """Validate form data before changing any file; preserve an unchanged batch profile."""
    values = dict(original)
    values.update({key: str(raw[key]).strip() for key in (
        "caption_model", "caption_base_url", "caption_start_script", "caption_model_file", "caption_mmproj_file")})
    if not values["caption_model"]:
        raise ValueError("모델 이름을 입력하세요. 서버의 모델 별칭과 같아야 합니다.")
    LocalClient(values)
    for key, minimum in (("caption_max_output_tokens", 1), ("caption_startup_timeout", 1),
                         ("caption_request_timeout", 1)):
        values[key] = int(raw[key])
        if values[key] < minimum:
            raise ValueError("토큰 수와 대기 시간은 1 이상의 정수여야 합니다.")
    mode = raw["caption_launch_mode"]
    if mode not in ("batch", "direct"):
        raise ValueError("실행 방식을 선택하세요.")
    values["caption_auto_start"] = bool(auto_start)
    if mode != "batch" or "caption_launch_mode" in original:
        values["caption_launch_mode"] = mode
    if mode == "direct":
        values["caption_server_exe"] = raw["caption_server_exe"].strip()
        values["caption_context_size"] = int(raw["caption_context_size"])
        if values["caption_context_size"] <= 0:
            raise ValueError("컨텍스트 크기는 1 이상이어야 합니다.")
    else:
        values.pop("caption_server_exe", None)
        values.pop("caption_context_size", None)
    if auto_start:
        required = ["caption_model_file", "caption_mmproj_file",
                    "caption_server_exe" if mode == "direct" else "caption_start_script"]
        for key in required:
            if not Path(values[key]).is_file():
                raise ValueError(f"파일을 찾을 수 없습니다:\n{values[key] or key}")
    return values

def apply_llm(config, profile):
    return {**{k: v for k, v in config.items() if not k.startswith("caption_")}, **profile}

class LLMDialog(tk.Toplevel):
    MODES = {"지정 배치 파일 실행": "batch", "모델 파일로 직접 실행": "direct"}

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("캡션 LLM 설정")
        self.configure(bg=BG)
        self.geometry("930x690")
        self.minsize(820, 650)
        self.transient(app)
        self.grab_set()
        self.original = {k: v for k, v in app.cfg.items() if k.startswith("caption_")}
        self.profiles = runner.read_json(AUTO / "llm_profiles.json") if (AUTO / "llm_profiles.json").exists() else {}
        self.vars = {}
        body = ttk.Frame(self, padding=22)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text="사진을 읽을 로컬 LLM", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(body, text="배치 파일을 바꾸거나, 다른 GGUF 모델과 이미지 프로젝터를 선택할 수 있습니다.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 16))
        ttk.Label(body, text="저장된 프로필").grid(row=2, column=0, sticky="w", pady=6)
        self.profile = tk.StringVar()
        self.profile_box = ttk.Combobox(body, textvariable=self.profile, values=list(self.profiles), state="readonly")
        self.profile_box.grid(row=2, column=1, sticky="ew", padx=12)
        self.profile_box.bind("<<ComboboxSelected>>", self.load_profile)
        ttk.Button(body, text="프로필로 저장", command=self.save_profile).grid(row=2, column=2)
        self.mode = tk.StringVar(value=next(label for label, value in self.MODES.items()
                                           if value == self.original.get("caption_launch_mode", "batch")))
        ttk.Label(body, text="실행 방식").grid(row=3, column=0, sticky="w", pady=6)
        self.mode_box = ttk.Combobox(body, textvariable=self.mode, values=list(self.MODES), state="readonly")
        self.mode_box.grid(row=3, column=1, sticky="ew", padx=12)
        self.mode_box.bind("<<ComboboxSelected>>", lambda _: self.update_mode())
        fields = [
            ("모델 이름 / 별칭", "caption_model", None),
            ("서버 주소", "caption_base_url", None),
            ("실행 배치 파일", "caption_start_script", "bat"),
            ("llama-server 실행 파일", "caption_server_exe", "exe"),
            ("LLM 모델 파일", "caption_model_file", "gguf"),
            ("이미지 프로젝터", "caption_mmproj_file", "gguf"),
            ("컨텍스트 크기", "caption_context_size", None),
            ("캡션 최대 토큰", "caption_max_output_tokens", None),
            ("모델 로딩 제한 (초)", "caption_startup_timeout", None),
            ("사진당 제한 (초)", "caption_request_timeout", None),
        ]
        defaults = {"caption_server_exe": DEFAULT_EXE, "caption_context_size": 65536,
                    "caption_max_output_tokens": 300, "caption_startup_timeout": 300, "caption_request_timeout": 180}
        self.entries, self.browse_buttons = {}, {}
        for row, (label, key, extension) in enumerate(fields, 4):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=6)
            var = self.vars[key] = tk.StringVar(value=self.original.get(key, defaults.get(key, "")))
            entry = self.entries[key] = ttk.Entry(body, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=12)
            if extension:
                button = ttk.Button(body, text="찾기…", command=lambda k=key, e=extension: self.browse(k, e))
                button.grid(row=row, column=2, sticky="ew")
                self.browse_buttons[key] = button
        self.auto_start = tk.BooleanVar(value=self.original.get("caption_auto_start", True))
        ttk.Checkbutton(body, text="캡션 시작 시 서버를 자동 실행하고, 끝나면 종료", variable=self.auto_start).grid(
            row=14, column=0, columnspan=3, sticky="w", pady=(12, 6))
        ttk.Label(body, text="이미지 프로젝터는 선택한 모델에 맞는 mmproj 파일이어야 합니다.\n"
                  "자동 실행을 끄면 이미 켜진 서버에 연결합니다. 사용자가 켠 서버는 자동 종료하지 않습니다.",
                  style="Muted.TLabel", wraplength=830).grid(row=15, column=0, columnspan=3, sticky="w", pady=8)
        footer = ttk.Frame(body)
        footer.grid(row=16, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Button(footer, text="연결 확인", command=self.probe).pack(side="left")
        self.probe_label = ttk.Label(footer, text="", style="Muted.TLabel")
        self.probe_label.pack(side="left", padx=12)
        ttk.Button(footer, text="취소", command=self.destroy).pack(side="right")
        ttk.Button(footer, text="적용", style="Accent.TButton", command=self.apply).pack(side="right", padx=8)
        self.update_mode()

    def update_mode(self):
        direct = self.MODES[self.mode.get()] == "direct"
        for key, enabled in (("caption_start_script", not direct), ("caption_server_exe", direct),
                             ("caption_context_size", direct)):
            self.entries[key].configure(state="normal" if enabled else "disabled")
            if key in self.browse_buttons:
                self.browse_buttons[key].configure(state="normal" if enabled else "disabled")

    def browse(self, key, extension):
        path = filedialog.askopenfilename(parent=self, title="파일 선택", filetypes=[(extension.upper(), "*." + extension), ("모든 파일", "*.*")])
        if path:
            self.vars[key].set(Path(path).as_posix())
            if key == "caption_model_file":
                self.mode.set("모델 파일로 직접 실행")
                self.update_mode()

    def collect(self):
        raw = {key: var.get() for key, var in self.vars.items()}
        raw["caption_launch_mode"] = self.MODES[self.mode.get()]
        return llm_values(self.original, raw, self.auto_start.get())

    def load_profile(self, _=None):
        profile = self.profiles[self.profile.get()]
        self.original = dict(profile)
        defaults = {"caption_server_exe": DEFAULT_EXE, "caption_context_size": 65536,
                    "caption_max_output_tokens": 300, "caption_startup_timeout": 300, "caption_request_timeout": 180}
        for key, var in self.vars.items():
            var.set(profile.get(key, defaults.get(key, "")))
        self.mode.set(next(label for label, value in self.MODES.items() if value == profile.get("caption_launch_mode", "batch")))
        self.auto_start.set(profile.get("caption_auto_start", True))
        self.update_mode()

    def save_profile(self):
        try:
            profile = self.collect()
        except (ValueError, KeyError) as exc:
            messagebox.showerror("설정 확인", str(exc), parent=self)
            return
        name = simpledialog.askstring("프로필 저장", "프로필 이름", parent=self)
        if name and name.strip():
            self.profiles[name.strip()] = profile
            runner.write_json(AUTO / "llm_profiles.json", self.profiles)
            self.profile_box.configure(values=list(self.profiles))
            self.profile.set(name.strip())

    def apply(self):
        try:
            if not self.app.ensure_idle():
                return
            cfg = apply_llm(self.app.collect_config(), self.collect())
            self.app.remember_runs()
            runner.write_json(AUTO / "config.json", cfg)
        except (ValueError, KeyError, OSError) as exc:
            messagebox.showerror("설정 확인", str(exc), parent=self)
            return
        self.app.cfg = cfg
        self.app.update_model_label()
        self.app.refresh_review()
        self.app.log("캡션 LLM 설정을 저장했습니다: " + cfg["caption_model"])
        self.destroy()

    def probe(self):
        try:
            cfg = self.collect()
        except (ValueError, KeyError) as exc:
            messagebox.showerror("설정 확인", str(exc), parent=self)
            return
        self.probe_label.configure(text="연결 확인 중…")
        events, dialog_id = self.app.events, str(self)
        def work():
            try:
                client = LocalClient(cfg)
                if not client.ready():
                    raise ValueError("서버가 꺼져 있습니다. 캡션 테스트를 누르면 자동 실행됩니다.")
                client.verify_model()
                result = "연결됨 · " + cfg["caption_model"]
            except Exception as exc:
                result = str(exc)
            events.put(("probe", dialog_id, result))
        self.app.start_worker(work)

class JobDialog(tk.Toplevel):
    def __init__(self, app, index=None):
        super().__init__(app)
        self.app, self.index = app, index
        self.title("작업 추가" if index is None else "작업 편집")
        self.configure(bg=BG)
        self.geometry("760x375")
        self.transient(app)
        self.grab_set()
        original = {} if index is None else app.jobs[index]
        self.original = dict(original)
        body = ttk.Frame(self, padding=22)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        self.vars = {}
        rows = [("작업 이름 (영문 / 숫자)", "id", ""), ("트리거", "trigger", ""),
                ("데이터셋 (비우면 기본 폴더)", "dataset_root", ""),
                ("목표 스텝 (비우면 기본값)", "target_steps", ""),
                ("Rank (비우면 기본값)", "rank", ""), ("시드", "seed", 42)]
        for row, (label, key, default) in enumerate(rows):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=8)
            var = self.vars[key] = tk.StringVar(value=original.get(key, default))
            ttk.Entry(body, textvariable=var).grid(row=row, column=1, sticky="ew", padx=12)
            if key == "dataset_root":
                ttk.Button(body, text="찾기…", command=self.browse).grid(row=row, column=2)
        self.enabled = tk.BooleanVar(value=original.get("enabled", True))
        ttk.Checkbutton(body, text="이 작업 활성화", variable=self.enabled).grid(row=6, column=0, sticky="w", pady=8)
        ttk.Button(body, text="저장", style="Accent.TButton", command=self.save).grid(row=7, column=1, sticky="e", pady=12)

    def browse(self):
        path = filedialog.askdirectory(parent=self)
        if path:
            self.vars["dataset_root"].set(Path(path).as_posix())

    def save(self):
        try:
            if not self.app.ensure_idle():
                return
            job = dict(self.original)
            for key in ("id", "trigger"):
                job[key] = self.vars[key].get().strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job["id"]):
                raise ValueError("작업 이름은 영문·숫자·밑줄·하이픈으로 입력하세요.")
            if not job["trigger"]:
                raise ValueError("트리거를 입력하세요.")
            if any(j["id"] == job["id"] for i, j in enumerate(self.app.jobs) if i != self.index):
                raise ValueError("같은 작업 이름이 이미 있습니다.")
            for key in ("target_steps", "rank"):
                raw = self.vars[key].get().strip()
                if raw:
                    job[key] = int(raw)
                    if job[key] <= 0:
                        raise ValueError("스텝과 Rank는 1 이상이어야 합니다.")
                else:
                    job.pop(key, None)
            folder = self.vars["dataset_root"].get().strip()
            if folder:
                job["dataset_root"] = folder
            else:
                job.pop("dataset_root", None)
            job["seed"] = int(self.vars["seed"].get())
            job["enabled"] = self.enabled.get()
            jobs = list(self.app.jobs)
            if self.index is None:
                jobs.append(job)
            else:
                jobs[self.index] = job
            self.app.remember_runs()
            runner.write_json(AUTO / "todo.json", jobs)
        except (ValueError, OSError) as exc:
            messagebox.showerror("작업 확인", str(exc), parent=self)
            return
        self.app.jobs = jobs
        self.app.refresh_jobs(select=job["id"])
        self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fizgig Auto · 로컬 LLM 캡션 & LoRA")
        self.configure(bg=BG)
        width, height = min(1220, self.winfo_screenwidth() - 70), min(900, self.winfo_screenheight() - 90)
        self.geometry(f"{width}x{height}")
        self.minsize(1000, 700)
        self.cfg = runner.read_json(AUTO / "config.json")
        self.jobs = runner.read_json(AUTO / "todo.json")
        self.events = queue.Queue()
        self.busy = False
        self.close_after = False
        self.process = None
        self.current_run = None
        self.current_backend = "local"
        self.stage = None
        self.tail_offsets = {}
        self.review_run = None
        self.review_files = {}
        self.review_generation = 0
        self.current_image = None
        self.caption_dirty = False
        self.caption_loading = False
        self.selected_jobs = []
        self.pause_requested = False
        self.after_id = None
        self.stage_buffers = {}
        self.operation = None
        self.review_busy = False
        self.review_cfg = None
        self.review_job_id = None
        self.skip_job_selection = None
        self.external_monitor = False
        self.external_pid = None
        self.external_last_check = 0
        self.workers = []
        self.controls = []
        self.setup_style()
        self.build()
        self.refresh_jobs()
        self.update_model_label()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after_id = self.after(100, self.poll)
        self.attach_existing()

    def setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self.option_add("*Font", ("맑은 고딕", 10))
        style.configure(".", background=BG, foreground=TEXT, font=("맑은 고딕", 10))
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("맑은 고딕", 18, "bold"))
        style.configure("Heading.TLabel", font=("맑은 고딕", 11, "bold"))
        style.configure("TEntry", fieldbackground=INPUT, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER, padding=7)
        style.configure("TCombobox", fieldbackground=INPUT, foreground=TEXT, background=CARD, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", INPUT)], foreground=[("readonly", TEXT)])
        style.configure("TButton", background=CARD, foreground=TEXT, bordercolor=BORDER, padding=(12, 7))
        style.map("TButton", background=[("active", "#2b3b50")], foreground=[("disabled", MUTED)])
        style.configure("Accent.TButton", background=ACCENT, foreground=BG, font=("맑은 고딕", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#8ee7c9"), ("disabled", CARD)], foreground=[("disabled", MUTED)])
        style.configure("Treeview", background=INPUT, fieldbackground=INPUT, foreground=TEXT, rowheight=31, bordercolor=BORDER)
        style.configure("Treeview.Heading", background=CARD, foreground=MUTED, padding=7)
        style.map("Treeview", background=[("selected", "#244b48")], foreground=[("selected", TEXT)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=MUTED, padding=(18, 9))
        style.map("TNotebook.Tab", background=[("selected", "#244b48")], foreground=[("selected", TEXT)])
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=INPUT, bordercolor=BORDER)
        self.option_add("*TCombobox*Listbox.background", INPUT)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)

    def button(self, parent, text, command, accent=False):
        button = ttk.Button(parent, text=text, command=command, style="Accent.TButton" if accent else "TButton")
        self.controls.append((button, "normal"))
        return button

    def build(self):
        main = ttk.Frame(self, padding=(24, 20))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        left = ttk.Frame(header)
        left.pack(side="left")
        ttk.Label(left, text="FIZGIG AUTO", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="로컬 LLM으로 캡션을 만들고, Qwen Image 2.1로 학습하세요.", style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        self.button(header, "LLM 설정…", self.open_llm, accent=True).pack(side="right")
        self.model_label = ttk.Label(header, text="", style="Muted.TLabel")
        self.model_label.pack(side="right", padx=16)
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.work_tab = ttk.Frame(self.notebook, padding=18)
        self.review_tab = ttk.Frame(self.notebook, padding=18)
        self.instructions_tab = ttk.Frame(self.notebook, padding=18)
        self.dataset_pane = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(self.work_tab, text="작업 & 학습")
        self.notebook.add(self.review_tab, text="사진 · 캡션 검수")
        self.notebook.add(self.instructions_tab, text="캡션 지침")
        self.notebook.add(self.dataset_pane, text="데이터셋 생성")
        self.build_work()
        self.build_review()
        self.build_instructions()
        # The dataset tab is a separate object so the existing tabs keep their own code path; it only shares the window.
        # Named `dataset_view` because `self.dataset` is already the 기본 데이터셋 StringVar of the work tab.
        self.dataset_view = dataset_tab.DatasetTab(self, self.dataset_pane, automation_dir=AUTO)
        footer = ttk.Frame(main)
        footer.grid(row=2, column=0, sticky="ew", pady=(15, 8))
        self.status = tk.StringVar(value="준비됨 · 캡션과 학습을 순서대로 실행합니다.")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        self.pause_button = ttk.Button(footer, text="일시정지 요청", command=self.pause, state="disabled")
        self.pause_button.pack(side="right")
        self.button(footer, "실행 잠금 정리", self.clear_stale_lock).pack(side="right", padx=8)
        self.progress = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.logs = ScrolledText(main, height=6, bg=INPUT, fg=MUTED, insertbackground=TEXT,
                                 relief="flat", font=("Consolas", 9), state="disabled")
        self.logs.grid(row=4, column=0, sticky="ew")

    def build_work(self):
        tab = self.work_tab
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="기본 데이터셋", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.dataset = tk.StringVar(value=self.cfg["dataset_root"])
        entry = ttk.Entry(tab, textvariable=self.dataset)
        entry.grid(row=0, column=1, sticky="ew", padx=12)
        self.controls.append((entry, "normal"))
        self.button(tab, "폴더 선택…", self.browse_dataset).grid(row=0, column=2)
        row = ttk.Frame(tab)
        row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(14, 10))
        self.steps = tk.StringVar(value=self.cfg["target_steps"])
        self.rank = tk.StringVar(value=self.cfg["rank"])
        self.megapixels = tk.StringVar(value=self.cfg["megapixels"])
        for label, var, choices in (("목표 스텝", self.steps, None), ("Rank", self.rank, [4, 8, 16, 32, 64]),
                                    ("해상도 (MP)", self.megapixels, [0.25, 0.5, 1.0, 2.0])):
            ttk.Label(row, text=label).pack(side="left", padx=(0, 8))
            widget = ttk.Combobox(row, textvariable=var, values=choices, width=7, state="readonly") if choices else ttk.Entry(row, textvariable=var, width=9)
            widget.pack(side="left", padx=(0, 22))
            self.controls.append((widget, "readonly" if choices else "normal"))
        self.button(row, "학습률 설정…", self.learning_rate_dialog).pack(side="right")
        self.button(row, "기본 설정 저장", self.save_config).pack(side="right", padx=8)
        ttk.Label(tab, text="TODO 작업", style="Heading.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 8))
        tree_frame = ttk.Frame(tab)
        tree_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")
        tab.rowconfigure(3, weight=1)
        self.job_tree = ttk.Treeview(tree_frame, columns=("enabled", "name", "trigger", "dataset", "steps"), show="headings", height=5, selectmode="browse")
        for key, label, width in (("enabled", "활성", 55), ("name", "작업 이름", 180), ("trigger", "트리거", 120),
                                   ("dataset", "데이터셋", 460), ("steps", "스텝", 70)):
            self.job_tree.heading(key, text=label)
            self.job_tree.column(key, width=width, stretch=key == "dataset")
        self.job_tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.job_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.job_tree.configure(yscrollcommand=scrollbar.set)
        self.job_tree.bind("<<TreeviewSelect>>", self.job_selected)
        self.job_tree.bind("<Double-1>", self.job_double_click)
        buttons = ttk.Frame(tab)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        for label, command in (("추가", lambda: JobDialog(self)), ("편집", self.edit_job),
                               ("활성 / 비활성", self.toggle_job), ("삭제", self.delete_job)):
            self.button(buttons, label, command).pack(side="left", padx=(0, 7))
        self.scope = tk.StringVar(value="선택한 작업")
        self.scope_box = ttk.Combobox(buttons, textvariable=self.scope, values=["선택한 작업", "활성 작업 전체"], state="readonly", width=14)
        self.scope_box.pack(side="right")
        self.controls.append((self.scope_box, "readonly"))
        ttk.Label(buttons, text="실행 대상", style="Muted.TLabel").pack(side="right", padx=10)
        actions = ttk.Frame(tab)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(6, 5))
        for label, command, accent in (("환경 검사", lambda: self.start("doctor"), False),
                ("사진 준비", lambda: self.start("prepare"), False),
                ("캡션 1장 테스트", lambda: self.start("caption", limit=1), False),
                ("전체 캡션", lambda: self.start("caption"), False),
                ("캡션 + LoRA 학습 / 재개", lambda: self.start("run"), True)):
            self.button(actions, label, command, accent).pack(side="left", padx=(0, 8))
        ttk.Label(tab, text="완료된 캡션은 재사용합니다. 학습은 가까운 에포크 단위로 맞추고 중간 LoRA와 샘플을 저장합니다.",
                  style="Muted.TLabel").grid(row=6, column=0, columnspan=3, sticky="w", pady=(9, 0))

    def build_review(self):
        tab = self.review_tab
        header = ttk.Frame(tab)
        header.pack(fill="x", pady=(0, 12))
        self.review_status = tk.StringVar(value="작업을 선택하면 사진과 캡션을 표시합니다.")
        ttk.Label(header, textvariable=self.review_status, style="Muted.TLabel").pack(side="left")
        ttk.Button(header, text="새로고침", command=self.refresh_review).pack(side="right")
        panes = ttk.Panedwindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left, right = ttk.Frame(panes), ttk.Frame(panes)
        panes.add(left, weight=1)
        panes.add(right, weight=3)
        self.photo_tree = ttk.Treeview(left, columns=("caption",), show="tree headings", height=12, selectmode="browse")
        self.photo_tree.heading("#0", text="사진")
        self.photo_tree.column("#0", width=220)
        self.photo_tree.heading("caption", text="캡션")
        self.photo_tree.column("caption", width=70, stretch=False)
        self.photo_tree.pack(fill="both", expand=True)
        self.photo_tree.bind("<<TreeviewSelect>>", self.select_photo)
        self.preview = tk.Label(right, bg=INPUT, fg=MUTED, text="사진 미리보기", height=10)
        self.preview.pack(fill="both", expand=True, padx=(16, 0))
        self.caption_text = ScrolledText(right, height=6, wrap="word", bg=INPUT, fg=TEXT, insertbackground=TEXT,
                                        relief="flat", padx=12, pady=10, font=("맑은 고딕", 10))
        self.caption_text.pack(fill="x", padx=(16, 0), pady=12)
        self.caption_text.bind("<<Modified>>", self.caption_modified)
        self.preview.bind("<Configure>", self.resize_preview)
        row = ttk.Frame(right)
        row.pack(fill="x", padx=(16, 0))
        self.button(row, "캡션 저장", self.save_caption, accent=True).pack(side="right")
        ttk.Button(row, text="작업 폴더 열기", command=self.open_folder).pack(side="left")
        ttk.Label(row, text="학습 사본의 캡션을 편집합니다.", style="Muted.TLabel").pack(side="left", padx=12)

    def build_instructions(self):
        ttk.Label(self.instructions_tab, text="사진을 어떤 방식으로 설명할지 지시하세요.", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(self.instructions_tab, text="트리거는 실행기가 앞에 붙입니다. 지침을 바꾸면 새 캡션 작업 버전으로 처리합니다.",
                  style="Muted.TLabel").pack(anchor="w", pady=(8, 15))
        self.instructions = ScrolledText(self.instructions_tab, wrap="word", bg=INPUT, fg=TEXT,
                                         insertbackground=TEXT, relief="flat", padx=16, pady=14, font=("맑은 고딕", 11))
        self.instructions.pack(fill="both", expand=True)
        self.instructions.insert("1.0", (AUTO / "caption_instructions.txt").read_text(encoding="utf-8-sig"))
        self.button(self.instructions_tab, "지침 저장", self.save_instructions, accent=True).pack(anchor="e", pady=(14, 0))

    def log(self, text):
        self.logs.configure(state="normal")
        self.logs.insert("end", text.rstrip() + "\n")
        # Keep a long overnight run from growing the widget without a bound.
        if int(self.logs.index("end-1c").split(".")[0]) > 2500:
            self.logs.delete("1.0", "501.0")
        self.logs.see("end")
        self.logs.configure(state="disabled")

    def report_callback_exception(self, kind, value, tb):
        import traceback
        error = "".join(traceback.format_exception(kind, value, tb))
        self.log(error)
        self.status.set("화면 처리 중 오류가 발생했습니다. 아래 로그를 확인하세요.")
        try:
            with (AUTO / "gui_error.log").open("a", encoding="utf-8") as stream:
                stream.write(error + "\n")
        except OSError:
            pass

    def start_worker(self, target):
        self.workers = [thread for thread in self.workers if thread.is_alive()]
        thread = threading.Thread(target=target, daemon=True)
        self.workers.append(thread)
        thread.start()

    def ensure_idle(self):
        if self.busy or (AUTO / "queue.lock").exists():
            messagebox.showinfo("실행 중", "실행 중인 작업이 끝나거나 일시정지한 뒤 변경하세요.", parent=self)
            return False
        return True

    def attach_existing(self):
        if not (AUTO / "queue.lock").exists():
            return
        if not runner.queue_owner_running():
            self.status.set("이전 실행의 잠금 파일이 남아 있습니다. ‘실행 잠금 정리’를 눌러 확인하세요.")
            self.log(self.status.get())
            return
        self.external_pid = (AUTO / "queue.lock").read_text(encoding="utf-8").strip()
        self.external_monitor = True
        self.operation = "run"
        self.set_busy(True)
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.status.set("진행 중인 기존 자동화에 연결했습니다.")
        self.log(self.status.get())
        self.monitor_existing()

    def monitor_existing(self):
        if not self.external_monitor or time.monotonic() - self.external_last_check < 1:
            return
        self.external_last_check = time.monotonic()
        lock = AUTO / "queue.lock"
        if not runner.queue_owner_running():
            self.external_monitor = False
            complete = False
            if self.current_run and (self.current_run / "state.json").is_file():
                complete = runner.read_json(self.current_run / "state.json").get("complete", False)
            self.events.put(("finished", 0 if complete else 1))
            return
        try:
            if lock.read_text(encoding="utf-8").strip() != self.external_pid:
                # A different queue is not the execution to which this window attached.
                self.external_monitor = False
                self.events.put(("finished", 1))
                return
            candidates = sorted((AUTO / "runs").glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            # The runner can unlink its lock between the liveness check and this read.
            return
        if not candidates:
            return
        try:
            state = runner.read_json(candidates[0])
            run, stage = candidates[0].parent, state.get("active_stage")
            if run != self.current_run or stage != self.stage:
                self.current_run, self.stage = run, stage
                if stage:
                    path = run / (stage.replace(" ", "_") + ".log")
                    self.tail_offsets[path] = max(0, path.stat().st_size - 16384) if path.is_file() else 0
                self.log("기존 작업: " + run.name)
        except (OSError, ValueError):
            return

    def clear_stale_lock(self):
        if self.busy or runner.queue_owner_running():
            messagebox.showinfo("실행 중", "자동화가 실행 중이므로 잠금을 지우지 않습니다.", parent=self)
            return
        try:
            (AUTO / "queue.lock").unlink(missing_ok=True)
            self.status.set("남은 실행 잠금을 정리했습니다.")
            self.log(self.status.get())
        except OSError as exc:
            messagebox.showerror("잠금 정리", str(exc), parent=self)

    def collect_config(self):
        cfg = dict(self.cfg)
        cfg["dataset_root"] = self.dataset.get().strip()
        if not cfg["dataset_root"] or not Path(cfg["dataset_root"]).is_dir():
            raise ValueError("읽을 수 있는 기본 데이터셋 폴더를 선택하세요.")
        for key, var in (("target_steps", self.steps), ("rank", self.rank)):
            cfg[key] = int(var.get())
            if cfg[key] <= 0:
                raise ValueError("스텝과 Rank는 1 이상이어야 합니다.")
        cfg["megapixels"] = float(self.megapixels.get())
        if not math.isfinite(cfg["megapixels"]) or cfg["megapixels"] <= 0:
            raise ValueError("해상도는 0보다 큰 수여야 합니다.")
        return cfg

    def remember_runs(self):
        """Capture old, exactly matching settings before a form changes its run version."""
        for job in self.jobs:
            cfg = {**self.cfg, **job}
            try:
                run, _, _, instructions = runner.context(cfg)
                if run.is_dir() and not (run / "run_config.json").exists():
                    runner.save_run_config(run, cfg, instructions)
            except (OSError, ValueError, KeyError):
                continue

    def save_config(self, quiet=False):
        if not self.ensure_idle():
            return False
        try:
            cfg = self.collect_config()
            self.remember_runs()
            runner.write_json(AUTO / "config.json", cfg)
            self.cfg = cfg
        except (ValueError, OSError, KeyError) as exc:
            messagebox.showerror("설정 확인", str(exc), parent=self)
            return False
        if not quiet:
            self.log("기본 데이터·학습 설정을 저장했습니다.")
        self.refresh_jobs()
        return True

    def browse_dataset(self):
        path = filedialog.askdirectory(parent=self, initialdir=self.dataset.get())
        if path:
            self.dataset.set(Path(path).as_posix())

    def learning_rate_dialog(self):
        if not self.ensure_idle():
            return
        dialog = tk.Toplevel(self)
        dialog.title("학습률 설정")
        dialog.configure(bg=BG)
        dialog.transient(self)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=22)
        body.pack(fill="both", expand=True)
        adaptive = tk.BooleanVar(value=self.cfg.get("adaptive_lr", True))
        ttk.Checkbutton(body, text="Adaptive LR 사용", variable=adaptive).grid(row=0, columnspan=2, sticky="w")
        fields = {}
        for row, (key, label) in enumerate((("learning_rate", "기본 학습률"), ("adaptive_lr_min", "최소 학습률"),
                                            ("adaptive_lr_max", "최대 학습률")), 1):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=8)
            fields[key] = tk.StringVar(value=self.cfg[key])
            ttk.Entry(body, textvariable=fields[key]).grid(row=row, column=1, padx=12)
        def save():
            try:
                cfg = self.collect_config()
                values = {key: float(var.get()) for key, var in fields.items()}
                if any(not math.isfinite(v) or v <= 0 for v in values.values()):
                    raise ValueError("학습률은 0보다 큰 수여야 합니다.")
                if adaptive.get() and values["adaptive_lr_min"] >= values["adaptive_lr_max"]:
                    raise ValueError("최소 학습률은 최대 학습률보다 작아야 합니다.")
                self.remember_runs()
                cfg.update(values, adaptive_lr=adaptive.get())
                runner.write_json(AUTO / "config.json", cfg)
                self.cfg = cfg
                self.refresh_jobs()
                self.log("학습률 설정을 저장했습니다.")
                dialog.destroy()
            except (ValueError, OSError, KeyError) as exc:
                messagebox.showerror("학습률 확인", str(exc), parent=dialog)
        ttk.Button(body, text="저장", style="Accent.TButton", command=save).grid(row=4, column=1, sticky="e", pady=(12, 0))

    def update_model_label(self):
        self.model_label.configure(text=self.cfg.get("caption_model", "LLM 설정 필요"))

    def open_llm(self):
        if self.ensure_idle():
            LLMDialog(self)

    def selected_index(self):
        selected = self.job_tree.selection()
        return int(selected[0]) if selected else None

    def refresh_jobs(self, select=None):
        if select is None:
            index = self.selected_index()
            select = self.jobs[index]["id"] if index is not None and index < len(self.jobs) else None
        self.job_tree.delete(*self.job_tree.get_children())
        pick = None
        for i, job in enumerate(self.jobs):
            self.job_tree.insert("", "end", iid=str(i), values=("●" if job.get("enabled", True) else "—",
                job["id"], job.get("trigger", ""), job.get("dataset_root", self.cfg["dataset_root"]),
                job.get("target_steps", self.cfg["target_steps"])))
            if job["id"] == select:
                pick = str(i)
        if self.jobs:
            self.job_tree.selection_set(pick or "0")
        self.refresh_review()

    def edit_job(self):
        index = self.selected_index()
        if index is not None and self.ensure_idle():
            JobDialog(self, index)

    def job_double_click(self, event):
        if self.job_tree.identify_row(event.y):
            self.edit_job()

    def job_selected(self, _=None):
        selected = self.job_tree.selection()
        if selected and selected[0] == self.skip_job_selection:
            self.skip_job_selection = None
            return
        self.refresh_review()

    def toggle_job(self):
        index = self.selected_index()
        if index is None or not self.ensure_idle():
            return
        jobs = [dict(j) for j in self.jobs]
        jobs[index]["enabled"] = not jobs[index].get("enabled", True)
        try:
            self.remember_runs()
            runner.write_json(AUTO / "todo.json", jobs)
            self.jobs = jobs
            self.refresh_jobs(select=jobs[index]["id"])
        except OSError as exc:
            messagebox.showerror("작업 저장", str(exc), parent=self)

    def delete_job(self):
        index = self.selected_index()
        if index is None or not self.ensure_idle():
            return
        job = self.jobs[index]
        if not messagebox.askyesno("작업 삭제", f"TODO에서 {job['id']} 작업을 제거할까요?\n사진·캡션·학습 결과는 보존됩니다.", parent=self):
            return
        try:
            jobs = self.jobs[:index] + self.jobs[index + 1:]
            runner.write_json(AUTO / "todo.json", jobs)
            self.jobs = jobs
            self.refresh_jobs()
        except OSError as exc:
            messagebox.showerror("작업 삭제", str(exc), parent=self)

    def caption_modified(self, _=None):
        if self.caption_text.edit_modified():
            if not self.caption_loading:
                self.caption_dirty = True
                self.review_status.set("캡션 수정 중 · 저장 버튼을 눌러 반영하세요.")
            self.caption_text.edit_modified(False)

    def resolve_caption_edit(self):
        if not self.caption_dirty:
            return True
        answer = messagebox.askyesnocancel("캡션 저장", "수정한 캡션을 저장할까요?", parent=self)
        if answer is None:
            return False
        if answer:
            return self.save_caption()
        self.caption_dirty = False
        return True

    def refresh_review(self):
        if not self.resolve_caption_edit():
            for i, job in enumerate(self.jobs):
                if job["id"] == self.review_job_id:
                    self.skip_job_selection = str(i)
                    self.job_tree.selection_set(str(i))
                    break
            return
        self.review_generation += 1
        generation = self.review_generation
        index = self.selected_index()
        self.current_image = None
        self.review_files = {}
        self.review_run = None
        self.photo_tree.delete(*self.photo_tree.get_children())
        self.caption_loading = True
        self.caption_text.configure(state="normal")
        self.caption_text.delete("1.0", "end")
        self.caption_text.edit_modified(False)
        self.caption_loading = False
        self.caption_text.configure(state="disabled" if self.busy else "normal")
        self.preview.configure(image="", text="사진 미리보기")
        self.preview.image = None
        if index is None:
            self.review_status.set("작업을 추가하거나 선택하세요.")
            return
        cfg = {**self.cfg, **self.jobs[index]}
        self.review_cfg = cfg
        self.review_job_id = cfg["id"]
        self.review_busy = True
        self.review_status.set("사진과 기존 캡션을 확인하는 중…")
        events = self.events
        def work():
            try:
                run, images, rejected, instructions = runner.context(cfg)
                # Preview a matching older caption set without copying or changing source files.
                candidates = runner.caption_sources(cfg, instructions, exclude=run)
                files = []
                for item in images:
                    source = Path(item["source"])
                    image = run / "images" / (item["sha256"] + source.suffix.lower())
                    cap = image.with_suffix(".txt")
                    inherited = None
                    if not cap.is_file():
                        inherited = runner.find_caption(item, candidates)
                    text = cap.read_text(encoding="utf-8-sig").strip() if cap.is_file() else (inherited or item["caption"])
                    files.append((image if image.is_file() else source, image, text))
                state = runner.read_json(run / "state.json") if (run / "state.json").is_file() else {}
                events.put(("review", generation, run, files, len(rejected), state))
            except Exception as exc:
                events.put(("review_error", generation, str(exc)))
        self.start_worker(work)

    def select_photo(self, _=None):
        selected = self.photo_tree.selection()
        if not selected:
            return
        if selected[0] == self.current_image:
            return
        if not self.resolve_caption_edit():
            if self.current_image is not None:
                self.photo_tree.selection_set(self.current_image)
            return
        self.current_image = selected[0]
        image, _, text = self.review_files[self.current_image]
        self.caption_loading = True
        self.caption_text.configure(state="normal")
        self.caption_text.delete("1.0", "end")
        self.caption_text.insert("1.0", text)
        self.caption_text.edit_modified(False)
        self.caption_dirty = False
        self.caption_loading = False
        self.caption_text.configure(state="disabled" if self.busy else "normal")
        self.render_preview(image)

    def render_preview(self, path):
        from PIL import Image, ImageOps, ImageTk
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((max(100, self.preview.winfo_width() - 20), max(100, self.preview.winfo_height() - 20)))
                photo = ImageTk.PhotoImage(image, master=self)
            self.preview.configure(image=photo, text="")
            self.preview.image = photo
        except (OSError, ValueError) as exc:
            self.preview.configure(image="", text=f"사진을 읽지 못했습니다.\n{exc}")

    def resize_preview(self, _=None):
        if self.current_image in self.review_files:
            self.render_preview(self.review_files[self.current_image][0])

    def save_caption(self):
        if self.current_image is None or not self.ensure_idle():
            return False
        try:
            text = self.caption_text.get("1.0", "end").strip()
            if not text:
                raise ValueError("빈 캡션은 저장할 수 없습니다.")
            cfg = self.review_cfg
            run, _, _ = runner.prepare(cfg)
            source, target, _ = self.review_files[self.current_image]
            if run != self.review_run:
                raise ValueError("데이터 또는 설정이 바뀌었습니다. 목록을 새로고침하세요.")
            if not text.startswith(cfg["trigger"] + ","):
                text = cfg["trigger"] + ", " + text
            cap = target.with_suffix(".txt")
            temp = cap.with_suffix(".txt.tmp")
            temp.write_text(text + "\n", encoding="utf-8")
            temp.replace(cap)
            self.review_files[self.current_image] = (target, target, text)
            self.photo_tree.set(self.current_image, "caption", "있음")
            self.caption_loading = True
            self.caption_text.delete("1.0", "end")
            self.caption_text.insert("1.0", text)
            self.caption_text.edit_modified(False)
            self.caption_loading = False
            self.caption_dirty = False
            self.review_status.set("캡션 저장 완료 · 변경된 캡션은 학습 캐시에 반영됩니다.")
            self.log("캡션 저장: " + cap.name[:20])
            return True
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("캡션 저장", str(exc), parent=self)
            return False

    def save_instructions(self):
        if not self.ensure_idle():
            return
        text = self.instructions.get("1.0", "end").strip()
        if not text:
            messagebox.showerror("지침 확인", "캡션 지침을 입력하세요.", parent=self)
            return
        try:
            self.remember_runs()
            path = AUTO / "caption_instructions.txt"
            temp = path.with_suffix(".txt.tmp")
            temp.write_text(text + "\n", encoding="utf-8")
            temp.replace(path)
            self.log("캡션 지침을 저장했습니다. 새 지침은 새 작업 버전에 적용됩니다.")
            self.refresh_review()
        except OSError as exc:
            messagebox.showerror("지침 저장", str(exc), parent=self)

    def open_folder(self):
        path = self.review_run
        if path is None or not path.is_dir():
            messagebox.showinfo("작업 폴더", "사진 준비를 실행하면 작업 폴더가 만들어집니다.", parent=self)
            return
        if os.name == "nt":
            os.startfile(str(path))

    def set_busy(self, busy):
        self.busy = busy
        for widget, idle_state in self.controls:
            widget.configure(state="disabled" if busy else idle_state)
        self.caption_text.configure(state="disabled" if busy else "normal")
        self.instructions.configure(state="disabled" if busy else "normal")
        self.pause_button.configure(state="normal" if busy and self.operation != "doctor" else "disabled")

    def start(self, command, limit=None):
        if not self.ensure_idle() or not self.resolve_caption_edit():
            return
        index = self.selected_index()
        if self.scope.get() == "활성 작업 전체":
            jobs = [j for j in self.jobs if j.get("enabled", True)]
        elif index is not None and self.jobs[index].get("enabled", True):
            jobs = [self.jobs[index]]
        else:
            jobs = []
        if not jobs:
            messagebox.showinfo("작업 선택", "활성 작업을 선택하거나 추가하세요.", parent=self)
            return
        if not self.save_config(quiet=True):
            return
        # Only clear markers for the jobs selected in this GUI; never clear another runner's lock.
        try:
            if command != "doctor":
                (AUTO / "stop_requested").unlink(missing_ok=True)
                for job in jobs:
                    run = runner.context({**self.cfg, **job})[0]
                    (run / "output" / ".pause_requested").unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            messagebox.showerror("재개 확인", str(exc), parent=self)
            return
        self.operation = command
        self.selected_jobs = [j["id"] for j in jobs]
        self.pause_requested = False
        self.current_run = self.stage = None
        self.current_backend = "local"
        self.pause_button.configure(text="일시정지")
        self.tail_offsets.clear()
        self.stage_buffers.clear()
        self.set_busy(True)
        self.status.set({"doctor": "환경 검사 중", "prepare": "사진 준비 중", "caption": "캡션 생성 중", "run": "자동화 실행 중"}[command])
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.log("\n▶ " + self.status.get() + " · " + ", ".join(self.selected_jobs))
        python = ROOT / "venv" / "Scripts" / "python.exe"
        cmd = [str(python if python.is_file() else Path(sys.executable)), str(AUTO / "runner.py"), command, "--events"]
        for job in jobs:
            cmd.extend(("--job", job["id"]))
        if limit is not None:
            cmd.extend(("--limit", str(limit)))
        events = self.events
        def work():
            try:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding="utf-8", errors="replace", bufsize=1, stdin=subprocess.DEVNULL,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                events.put(("process", proc))
                with proc.stdout:
                    for line in proc.stdout:
                        events.put(("line", line))
                events.put(("finished", proc.wait()))
            except Exception as exc:
                events.put(("line", "실행 오류: " + str(exc)))
                events.put(("finished", 1))
        self.start_worker(work)

    def pause(self):
        if not self.busy or self.operation == "doctor":
            return
        if getattr(self, "current_backend", "local") == "modal":
            # A Modal run has no local checkpoint to pause at; the button is "원격 취소".
            if self.current_run is not None:
                try:
                    (self.current_run / "cancel_remote.request").write_text("GUI cancel\n", encoding="utf-8")
                except OSError as exc:
                    messagebox.showerror("원격 취소", str(exc), parent=self)
                    return
                self.pause_button.configure(state="disabled")
                self.status.set("원격 취소 요청됨 - 원격 작업을 취소합니다(체크포인트는 남습니다)")
                self.log(self.status.get())
            else:
                self.log("아직 원격 작업이 제출되지 않았습니다 - 제출 전에는 취소할 수 없습니다")
            return
        try:
            (AUTO / "stop_requested").write_text("GUI pause\n", encoding="utf-8")
            if self.current_run is not None:
                output = self.current_run / "output"
                output.mkdir(parents=True, exist_ok=True)
                (output / ".pause_requested").write_text("GUI pause\n", encoding="utf-8")
            self.pause_requested = True
            self.pause_button.configure(state="disabled")
            self.status.set("일시정지 요청됨 · 캡션 요청/캐시는 완료 후, 학습은 에포크 끝에 저장하고 멈춥니다.")
            self.log(self.status.get())
        except OSError as exc:
            messagebox.showerror("일시정지", str(exc), parent=self)

    def handle_line(self, line):
        if line.startswith(runner.EVENT_PREFIX):
            try:
                data = json.loads(line[len(runner.EVENT_PREFIX):])
            except (ValueError, TypeError):
                self.log(line)
                return
            if data["kind"] == "run":
                self.current_run = Path(data["path"])
                self.tail_offsets[self.current_run / "gemma_server.log"] = data.get("gemma_offset", 0)
                self.current_backend = str(data.get("backend") or "local")
                if self.current_backend == "modal":
                    self.pause_button.configure(text="원격 취소")
                else:
                    self.pause_button.configure(text="일시정지")
                if self.pause_requested:
                    self.pause()
            elif data["kind"] == "stage":
                self.stage = data["stage"]
                path = self.current_run / (self.stage.replace(" ", "_") + ".log")
                self.tail_offsets[path] = data.get("log_offset", 0)
                names = {"Cache Preparation": "VAE 캐시 생성", "Text Encoder Caching": "텍스트 캐시 생성", "Training": "LoRA 학습"}
                if not self.pause_requested:
                    self.status.set(names.get(self.stage, self.stage))
            elif data["kind"] == "caption":
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=data["total"], value=data["done"])
                if not self.pause_requested:
                    self.status.set(f"캡션 생성 · {data['done']}/{data['total']}장")
            return
        self.log(line)
        epoch = re.search(r"epoch (\d+)/(\d+)\s+avr_loss=.*step=(\d+)", line)
        if epoch:
            done, total, step = map(int, epoch.groups())
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=total, value=done)
            if not self.pause_requested:
                self.status.set(f"LoRA 학습 · {done}/{total}에포크 · {step}스텝")
        step_match = re.search(r"steps:\s+\d+%.*?\|\s*(\d+)/(\d+)", line)
        if step_match:
            done, total = map(int, step_match.groups())
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=total, value=done)
            if not self.pause_requested:
                self.status.set(f"LoRA 학습 · {done:,}/{total:,}스텝")
        if "Gemma server is still running" in line:
            self.log("캡션 서버가 이미 켜져 있습니다. 사용자가 켠 서버를 종료한 뒤 학습/재개 버튼을 누르세요. 기존 캡션은 재사용합니다.")

    def tail_logs(self):
        if self.current_run is None:
            return
        paths = list(self.tail_offsets)
        if self.operation in ("caption", "run") and not self.stage:
            paths.append(self.current_run / "gemma_server.log")
        for path in paths:
            try:
                if not path.is_file():
                    continue
                offset = self.tail_offsets.get(path)
                if offset is None:
                    self.tail_offsets[path] = path.stat().st_size
                    continue
                if path.stat().st_size < offset:
                    offset = 0
                with path.open("rb") as stream:
                    stream.seek(offset)
                    chunk = stream.read(65536)
                    self.tail_offsets[path] = stream.tell()
                if not chunk:
                    continue
                # Decode only complete lines (UTF-8 characters can straddle read boundaries).
                buffered = self.stage_buffers.get(path, b"") + chunk
                split = re.split(b"[\r\n]", buffered)
                self.stage_buffers[path] = split.pop()
                for raw in split:
                    if raw.strip():
                        self.handle_line(raw.decode("utf-8", errors="replace"))
            except OSError:
                continue

    def poll(self):
        if self.external_monitor:
            self.monitor_existing()
        for _ in range(300):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "line":
                self.handle_line(event[1])
            elif kind == "process":
                self.process = event[1]
            elif kind == "probe":
                try:
                    dialog = self.nametowidget(event[1])
                    dialog.probe_label.configure(text=event[2])
                except (KeyError, tk.TclError):
                    pass
            elif kind == "review_error" and event[1] == self.review_generation:
                self.review_busy = False
                self.review_status.set("사진 확인 실패 · " + event[2])
            elif kind == "review" and event[1] == self.review_generation:
                _, _, self.review_run, files, rejected, state = event
                self.review_busy = False
                for i, item in enumerate(files):
                    self.review_files[str(i)] = item
                    self.photo_tree.insert("", "end", iid=str(i), text=f"{i + 1:02d} · {item[0].name}", values=("있음" if item[2] else "없음",))
                count = sum(bool(item[2]) for item in files)
                label = "학습 완료" if state.get("complete") else "사진·캡션 준비"
                self.review_status.set(f"{len(files)}장 · 캡션 {count}/{len(files)} · 제외 {rejected} · {label}")
                if files:
                    self.photo_tree.selection_set("0")
                    self.select_photo()
            elif kind == "finished":
                self.tail_logs()
                self.process = None
                self.set_busy(False)
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=100, value=100 if event[1] == 0 else 0)
                result = "일시정지됨 · 학습/재개 버튼으로 이어갈 수 있습니다." if self.pause_requested else (
                    "환경 검사 완료" if event[1] == 0 and self.operation == "doctor" else
                    "작업 완료" if event[1] == 0 else "작업이 중단됐습니다. 아래 로그에서 원인을 확인하세요.")
                self.status.set(result)
                self.log(result)
                self.refresh_review()
                if self.close_after:
                    self.destroy()
                    return
        if self.busy:
            self.tail_logs()
        elif self.close_after:
            # A dataset stage owns its own worker and reports completion through `set_busy(False)`, not through this
            # event queue's "finished". Checking the tab itself is what lets a close-during-a-stage actually close:
            # before this the window waited for a queue event that a dataset stage never sends (spec N2).
            self.close_after = False
            self.destroy()
            return
        self.after_id = self.after(200, self.poll)

    def close(self):
        """Closing with work in flight ends the work first (spec N2).

        The old path only wrote `automation/stop_requested` and waited for a `poll()` that never looked at
        `close_after` for a dataset stage: a runner checks that file between stages, but a dataset stage polls its own
        project marker, so the window could sit there while the stage kept drawing - and the leftover
        `stop_requested` then made the next seed training refuse with "Automation paused".

        Both kinds of work now go through the same stop: the dataset tab cancels its own stage through
        `DatasetTab.cancel` (marker, then tree), and a runner operation is paused through its own marker as before.
        The window closes once the dataset stage's process has actually exited; a runner is given the same polite
        treatment and its own child tree is ended if it outlives the grace window.
        """
        if self.busy:
            self.close_after = True
            if getattr(self, "dataset_view", None) is not None and self.dataset_view.running is not None:
                if getattr(self.dataset_view, "running_backend", None) == "modal":
                    self.dataset_view.terminate_local()
                    self.status.set("원격은 계속 진행합니다. 로컬 추적만 종료했으며, 다음에 열 때 결과를 받습니다.")
                    return
                self.dataset_view.cancel()
                self.status.set("데이터셋 단계를 중단한 뒤 창을 닫습니다. 저장이 끝날 때까지 기다려 주세요.")
            elif getattr(self, "current_backend", "local") == "modal":
                # A Modal run keeps going on Modal: only the local tracking ends. No pause markers, no remote cancel.
                self.log("원격에서 계속 진행 중 - 다음에 열 때 결과를 받습니다")
                if self.process is not None and self.process.poll() is None:
                    pid = getattr(self.process, "pid", None)
                    proctree.stop_tree(pid, grace=1, log=lambda text: self.log(text))
                self.close_after = False
                self.destroy()
                return
            else:
                self.pause()
                self.status.set("작업을 안전하게 중단한 뒤 창을 닫습니다. 저장이 끝날 때까지 기다려 주세요.")
            return
        if not self.resolve_caption_edit():
            return
        if self.after_id:
            self.after_cancel(self.after_id)
        self.destroy()

    def destroy(self):
        # The dataset tab owns scheduled callbacks of its own; they must be cancelled before this interpreter dies,
        # or Tk fires them against a destroyed widget and logs "invalid command name" (spec R7).
        try:
            self.dataset_view.destroy()
        except tk.TclError:
            pass
        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        # Finish short file readers before Tk variables are destroyed. Workers only
        # retain plain data and the event queue, never a Tk widget/interpreter.
        deadline = time.monotonic() + 2
        for thread in self.workers:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        super().destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", action="store_true", help="Open the GUI and run the environment check")
    args = parser.parse_args()
    try:
        app = App()
        if args.doctor and not app.busy:
            app.after(300, lambda: app.start("doctor"))
        app.mainloop()
    except Exception:
        import traceback
        error = traceback.format_exc()
        try:
            (AUTO / "gui_error.log").write_text(error, encoding="utf-8")
        except OSError:
            pass
        root = tk._default_root or tk.Tk()
        root.withdraw()
        messagebox.showerror("Fizgig Auto 실행 오류", "GUI를 시작하지 못했습니다.\n" + error[-1600:], parent=root)
        root.destroy()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
