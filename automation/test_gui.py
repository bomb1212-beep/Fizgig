"""Real Tk widgets and a lightweight subprocess fixture; no model loading or GPU training."""
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import automation_gui as gui
import runner


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gui-test-", dir=runner.AUTO)
        self.root = Path(self.temp.name)
        self.auto = self.root / "automation"
        self.auto.mkdir()
        self.source = self.root / "source"
        self.source.mkdir()
        from PIL import Image
        Image.new("RGB", (160, 100), "red").save(self.source / "one.png")
        (self.source / "one.txt").write_text("a red image", encoding="utf-8")
        self.cfg = runner.read_json(runner.AUTO / "config.json")
        self.cfg["dataset_root"] = str(self.source)
        runner.write_json(self.auto / "config.json", self.cfg)
        runner.write_json(self.auto / "todo.json", [{"id": "fixture", "trigger": "test", "enabled": True}])
        runner.write_json(self.auto / "llm_profiles.json", {})
        (self.auto / "caption_instructions.txt").write_text("Describe the colors.", encoding="utf-8")
        # Exercise the actual GUI reader thread and Popen lifecycle with an inexpensive child.
        (self.auto / "runner.py").write_text('''import json,sys,time
from pathlib import Path
auto=Path(__file__).parent
print('fixture arguments: '+json.dumps(sys.argv[1:]), flush=True)
command=sys.argv[1]
if command=='caption':
    print('fixture failure', flush=True)
    sys.exit(3)
if command=='run':
    run=auto/'fixture-run'
    run.mkdir(exist_ok=True)
    print('@@FIZGIG_EVENT@@'+json.dumps({'kind':'run','path':str(run)}),flush=True)
    for i in range(100):
        if (auto/'stop_requested').exists():
            print('fixture paused',flush=True)
            sys.exit(1)
        time.sleep(.03)
print('fixture completed',flush=True)
''', encoding="utf-8")
        self.patches = [patch.object(gui, "AUTO", self.auto), patch.object(runner, "AUTO", self.auto)]
        for item in self.patches:
            item.start()
        self.app = gui.App()
        self.app.withdraw()
        self.wait_for(lambda: not self.app.review_busy)

    def wait_for(self, predicate, timeout=10):
        end = time.monotonic() + timeout
        while not predicate() and time.monotonic() < end:
            self.app.update()
            time.sleep(.01)
        self.app.update()
        self.assertTrue(predicate(), "GUI operation did not finish")

    def tearDown(self):
        if self.app.busy:
            self.app.pause()
            self.wait_for(lambda: not self.app.busy)
        try:
            self.app.destroy()
        except tk.TclError:
            pass                    # a test that closed the app itself has nothing left to destroy
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_closing_the_app_cancels_the_dataset_tabs_callbacks(self):
        # The tab schedules a poll every 150ms; a callback that fires after the interpreter dies is the
        # "invalid command name ..._poll" warning on close (spec R7). The app's close path runs the tab's destroy
        # before the interpreter goes away, and nothing is scheduled after it.
        tab = self.app.dataset_view
        self.assertTrue(tab.after_ids, "the poll is scheduled while the app lives")
        observed = []
        original = tab.destroy

        def spy():
            original()
            observed.append(list(tab.after_ids))

        tab.destroy = spy
        self.app.destroy()
        self.assertEqual(observed, [[]], "the app's close must cancel the tab's callbacks first")
        tab._schedule_poll()
        self.assertEqual(tab.after_ids, [], "no callback may be scheduled on a dead interpreter")

    def test_the_dataset_tab_poll_drains_events_before_the_app_closes(self):
        tab = self.app.dataset_view
        tab.events.put(("line", "drain me"))
        tab._poll()
        self.assertEqual(tab.events.empty(), True)

    def test_tabs_preview_and_caption_save_preserve_original(self):
        # Four tabs since M5 added "데이터셋 생성"; the three original tabs and their behaviour are unchanged.
        self.assertEqual(len(self.app.notebook.tabs()), 4)
        self.assertEqual(len(self.app.review_files), 1)
        original = (self.source / "one.txt").read_text(encoding="utf-8")
        self.app.caption_text.delete("1.0", "end")
        self.app.caption_text.insert("1.0", "a changed caption")
        self.assertTrue(self.app.save_caption())
        target = self.app.review_files[self.app.current_image][1].with_suffix(".txt")
        self.assertEqual(target.read_text(encoding="utf-8").strip(), "test, a changed caption")
        self.assertEqual((self.source / "one.txt").read_text(encoding="utf-8"), original)

    def test_unchanged_llm_profile_preserves_version(self):
        dialog = gui.LLMDialog(self.app)
        dialog.withdraw()
        expected = {k: v for k, v in self.cfg.items() if k.startswith("caption_")}
        self.assertEqual(dialog.collect(), expected)
        dialog.destroy()

    def test_log_area_stays_visible_in_small_window(self):
        self.app.geometry("1000x700")
        self.app.deiconify()
        self.app.update()
        self.assertGreater(self.app.logs.winfo_height(), 60)
        self.assertLessEqual(self.app.logs.winfo_y() + self.app.logs.winfo_height(), self.app.winfo_height())

    def test_cancel_job_switch_preserves_unsaved_caption(self):
        dialog = gui.JobDialog(self.app)
        dialog.withdraw()
        dialog.vars["id"].set("fixture_two")
        dialog.vars["trigger"].set("second")
        dialog.save()
        self.wait_for(lambda: not self.app.review_busy)
        self.app.job_tree.selection_set("0")
        self.app.update()
        self.wait_for(lambda: not self.app.review_busy)
        self.app.caption_text.delete("1.0", "end")
        self.app.caption_text.insert("1.0", "unsaved caption")
        self.app.update()
        with patch.object(gui.messagebox, "askyesnocancel", return_value=None):
            self.app.job_tree.selection_set("1")
            self.app.update()
        self.assertEqual(self.app.job_tree.selection(), ("0",))
        self.assertEqual(self.app.caption_text.get("1.0", "end").strip(), "unsaved caption")
        self.assertEqual(self.app.review_cfg["id"], "fixture")
        self.app.caption_dirty = False

    def test_job_dialog_saves_todo(self):
        dialog = gui.JobDialog(self.app)
        dialog.withdraw()
        dialog.vars["id"].set("fixture_two")
        dialog.vars["trigger"].set("second")
        dialog.save()
        self.assertEqual([j["id"] for j in runner.read_json(self.auto / "todo.json")], ["fixture", "fixture_two"])

    def test_environment_check_keeps_window_and_result(self):
        self.app.start("doctor")
        self.wait_for(lambda: not self.app.busy)
        self.assertTrue(self.app.winfo_exists())
        self.assertEqual(self.app.status.get(), "환경 검사 완료")
        self.assertIn("fixture completed", self.app.logs.get("1.0", "end"))

    def test_failure_is_visible_and_controls_are_restored(self):
        self.app.start("caption")
        self.wait_for(lambda: not self.app.busy)
        self.assertIn("중단", self.app.status.get())
        self.assertIn("fixture failure", self.app.logs.get("1.0", "end"))
        self.assertEqual(str(self.app.scope_box.cget("state")), "readonly")

    def test_pause_and_resume_manage_both_markers(self):
        self.app.start("run")
        self.wait_for(lambda: self.app.current_run is not None)
        run = self.app.current_run
        self.app.pause()
        self.assertTrue((self.auto / "stop_requested").is_file())
        self.assertTrue((run / "output" / ".pause_requested").is_file())
        self.wait_for(lambda: not self.app.busy)
        self.assertIn("일시정지됨", self.app.status.get())
        # The marker cleared on resume is the selected job's actual context, not every run.
        actual_run = runner.context({**self.app.cfg, **self.app.jobs[0]})[0]
        (actual_run / "output").mkdir(parents=True, exist_ok=True)
        marker = actual_run / "output" / ".pause_requested"
        marker.touch()
        self.app.start("prepare")
        self.assertFalse((self.auto / "stop_requested").exists())
        self.assertFalse(marker.exists())
        self.wait_for(lambda: not self.app.busy)

    def test_existing_queue_attaches_without_starting_another_process(self):
        cfg = {**self.app.cfg, **self.app.jobs[0]}
        run, _, _ = runner.prepare(cfg)
        runner.write_json(run / "state.json", {"active_stage": "Training", "completed_stages": []})
        (run / "Training.log").write_text("epoch 2/94  avr_loss=0.3  step=64\n", encoding="utf-8")
        (self.auto / "queue.lock").write_text("12345", encoding="utf-8")
        with patch.object(runner, "queue_owner_running", return_value=True):
            self.app.attach_existing()
            self.app.tail_logs()
            self.assertTrue(self.app.external_monitor)
            self.assertIsNone(self.app.process)
            self.assertIn("2/94", self.app.status.get())
        (self.auto / "queue.lock").unlink()
        self.app.external_last_check = 0
        self.wait_for(lambda: not self.app.busy)


if __name__ == "__main__":
    unittest.main()
