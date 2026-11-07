"""데이터셋 파이프라인 전용 실행기 — DatasetTab을 그대로 호스트하는 얇은 창.

automation_gui.py의 "데이터셋 생성" 탭이 하는 일을 미리보기·캡션·LLM 탭 없이 띄운다. 이 파일이 상위 객체에게
공급하는 것은 DatasetTab이 요구하는 세 가지뿐이다: 버튼을 등록해 두었다가 busy에 함께 잠그는 것, busy 전환,
그리고 자동화 폴더 위치. 규칙(홀드아웃·계보·에포크·export)은 전부 CLI 안에 있고, 이 창은 CLI를 부르는 다리다.

    pythonw automation/dataset_gui.py [--automation-dir 경로]
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

AUTO = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTO))

from dataset_tab import DatasetTab  # noqa: E402


class Host:
    """DatasetTab의 요구사항: 버튼을 등록해 두었다가 busy일 때 함께 잠근다."""

    def __init__(self, root):
        self.root = root
        self.controls = []
        self.busy = False

    def button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.controls.append((button, "normal"))
        return button

    def set_busy(self, busy):
        self.busy = busy
        for widget, idle_state in self.controls:
            try:
                widget.configure(state="disabled" if busy else idle_state)
            except tk.TclError:
                pass                    # a widget destroyed with its window is not a state problem


class PipelineApp(tk.Tk):
    def __init__(self, automation_dir=None):
        super().__init__()
        self.title("데이터셋 파이프라인 — Fizgig 자동화")
        self.geometry("1060x840")
        self.host = Host(self)
        style = ttk.Style(self)
        style.configure("Heading.TLabel", font=("맑은 고딕", 11, "bold"))
        style.configure("Muted.TLabel", foreground="#6a7480")
        style.configure("Accent.TButton", font=("맑은 고딕", 10, "bold"))

        main = ttk.Frame(self, padding=(14, 12))
        main.pack(fill="both", expand=True)
        header = ttk.Frame(main)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="데이터셋 파이프라인",
                  font=("맑은 고딕", 14, "bold")).pack(side="left")
        ttk.Label(header, text="버튼 하나가 CLI 한 단계를 실행합니다. 규칙(홀드아웃·계보·에포크·export)은 CLI 안에 "
                               "있고, 이 창은 실행·로그·취소만 책집니다.",
                  style="Muted.TLabel").pack(side="left", padx=12)
        # DatasetTab builds straight into its parent frame.
        tab = ttk.Frame(main)
        tab.pack(fill="both", expand=True)
        self.dataset_view = DatasetTab(self.host, tab, automation_dir=automation_dir or AUTO)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        if self.host.busy:
            if not messagebox.askyesno("실행 중", "단계가 실행 중입니다. 종료하고 닫을까요?\n"
                                                 "(이미 그려진 그림은 남습니다)"):
                return
            self.dataset_view.cancel()
        self.dataset_view.destroy()
        self.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--automation-dir", default=str(AUTO), help="automation folder (projects/ lives in it)")
    args = parser.parse_args(argv)
    app = PipelineApp(args.automation_dir)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
