"""The standalone dataset-pipeline window: it hosts the tested tab and nothing else.

`DatasetTab` carries its own tests (commands, summary, cancel, close); what is checked here is the thin host — the
window builds with the real stage buttons, busy locks every control, and closing while busy terminates the run and
destroys cleanly.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import tkinter as tk  # noqa: E402

import dataset_gui  # noqa: E402
import dataset_tab  # noqa: E402
from test_dataset_tab import make_project  # noqa: E402


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dataset-gui-test-", dir=AUTO)
        self.root = Path(self.temp.name)
        self.auto = self.root / "automation"
        self.auto.mkdir()
        make_project(self.auto, "fixture", candidates=3,
                     decisions={"c_0": {"decision": "accept"}, "c_1": {"decision": "reject"}})
        self.app = dataset_gui.PipelineApp(automation_dir=str(self.auto))
        self.app.withdraw()

    def tearDown(self):
        try:
            self.app.destroy()
        except tk.TclError:
            pass
        self.temp.cleanup()

    def test_the_window_builds_with_every_stage_button(self):
        self.assertEqual(self.app.title(), "데이터셋 파이프라인 — Fizgig 자동화")
        self.assertEqual(len(self.app.dataset_view._buttons), len(dataset_tab.STAGES))
        self.assertIn("fixture", self.app.dataset_view.project.get(),
                      "the fixture project is discovered and selected")

    def test_busy_locks_every_registered_control(self):
        self.app.host.set_busy(True)
        self.assertTrue(all(str(widget.cget("state")) == "disabled"
                            for widget, _idle in self.app.host.controls))
        self.app.host.set_busy(False)
        self.assertTrue(all(str(widget.cget("state")) != "disabled"
                            for widget, _idle in self.app.host.controls))

    def test_the_header_reads_the_project_summary(self):
        # The header's numbers come from the project's own files (R7): 3 candidates, 1 accepted, 1 decided reject.
        summary = dataset_tab.project_summary(self.app.dataset_view.auto / "projects" / "fixture")
        self.assertEqual((summary["candidates"], summary["accepted"], summary["undecided"]), (3, 1, 1))
        self.assertIn("fixture", self.app.dataset_view.info.get())

    def test_closing_while_busy_asks_then_closes(self):
        self.app.host.busy = True
        with patch.object(dataset_gui.messagebox, "askyesno", return_value=False) as ask:
            self.app.close()
            ask.assert_called_once()
            self.assertTrue(self.app.winfo_exists(), "a refused close leaves the window up")
        with patch.object(dataset_gui.messagebox, "askyesno", return_value=True):
            self.app.close()
        self.assertEqual(self.app.dataset_view.after_ids, [],
                         "closing cancels the tab's callbacks before the interpreter dies")


if __name__ == "__main__":
    unittest.main()
