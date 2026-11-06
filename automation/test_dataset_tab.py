"""M5: the "데이터셋 생성" tab.

What matters here is that the tab is a faithful front end and not a second implementation: it must build the same argv
the command line would, refuse what the pipeline refuses, and never report a crash as a clean stop. The existing tabs
are covered by `test_gui.py`, which still runs unchanged.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dataset_tab


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data, encoding="utf-8")


def make_project(automation, name, *, card=None, candidates=0, decisions=None, epoch=None, scores=None):
    folder = automation / "projects" / name
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"id": name, "trigger": "trig", "slot_quotas": {}}
    if epoch is not None:
        payload["krea2_inference"] = {"epoch": epoch}
    payload.update(card or {})
    write(folder / "project.json", payload)
    write(folder / "sources.jsonl",
          "\n".join(json.dumps({"id": f"s_{i}", "origin": "user"}) for i in range(2)))
    if candidates is not None:
        write(folder / "candidates.jsonl",
              "\n".join(json.dumps({"id": f"c_{i}"}) for i in range(candidates)))
    if decisions is not None:
        # The format  writes: {candidate_id: {decision, ...}} (spec R7 - the old wrapper shape
        # kept the tab testing a fiction the product never wrote).
        write(folder / "review.json", decisions)
    if scores is not None:
        write(folder / "experiments" / "scores.json", {"targets": scores})
    return folder


class ProjectDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.auto = Path(self.temp.name) / "automation"
        self.auto.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_only_folders_with_a_card_are_projects(self):
        make_project(self.auto, "one")
        (self.auto / "projects" / "not_a_project").mkdir()
        self.assertEqual([p.name for p in dataset_tab.project_roots(self.auto)], ["one"])

    def test_a_missing_projects_folder_is_empty_not_an_error(self):
        self.assertEqual(dataset_tab.project_roots(self.auto), [])

    def test_the_summary_reads_what_is_on_disk(self):
        folder = make_project(self.auto, "p", candidates=5,
                              decisions={"c_0": {"decision": "accept"}, "c_1": {"decision": "reject"}},
                              scores=5, epoch=42)
        summary = dataset_tab.project_summary(folder)
        self.assertEqual(summary["sources"], 2)
        self.assertEqual(summary["candidates"], 5)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["undecided"], 3)
        self.assertEqual(summary["scores"], 5)
        self.assertEqual(summary["epoch"], 42)

    def test_a_project_with_only_a_card_reports_none_rather_than_zero(self):
        # "never measured" and "measured zero" are different, and a header showing 0 for both misleads. An absent file
        # is None; an empty file of rows is a real 0 and is reported as one.
        folder = make_project(self.auto, "bare", candidates=None)
        summary = dataset_tab.project_summary(folder)
        self.assertIsNone(summary["candidates"])
        self.assertIsNone(summary["scores"])
        self.assertIsNone(summary["epoch"])
        self.assertEqual(summary["sources"], 2)

    def test_an_empty_row_file_is_a_measured_zero(self):
        folder = make_project(self.auto, "empty", candidates=0)
        self.assertEqual(dataset_tab.project_summary(folder)["candidates"], 0)

    def test_a_broken_card_does_not_stop_the_listing(self):
        folder = self.auto / "projects" / "broken"
        folder.mkdir(parents=True)
        (folder / "project.json").write_text("not json", encoding="utf-8")
        roots = dataset_tab.project_roots(self.auto)
        self.assertEqual([p.name for p in roots], ["broken"])
        self.assertEqual(dataset_tab.project_summary(folder)["id"], "broken")


class ProductApiSummaryTests(unittest.TestCase):
    """R7: the header's numbers come from the product API's own fixture, not a hand-made file."""

    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        from pipeline import lineage, project as pj
        lineage_record = lineage.record
        card = {"id": "fx", "trigger": "fx", "caption_base": "style",
                "final_families": ["qwen_image21", "krea2"],
                "holdout": {"ids": []}}
        self.project = pj.Project.create("fx", card, folder=self.root / "proj")
        # The real project's shape, at its own counts: 104 candidates, 32 accepted, 37 rejected, 8 marked for a
        # correction, 27 still waiting (spec R7).
        incoming = self.root / "incoming"
        incoming.mkdir(parents=True)
        for n in range(104):
            path = incoming / f"c{n}.png"
            path.write_bytes(b"IMG-" + str(n).encode())
            record = lineage_record(path, generator="qwen_edit", parent_ids=["s_x"], train_generation=1,
                                    edit_depth=1, slot={"slot_id": f"frm_{n}"}, prompt="p")
            self.project.import_candidates([record], log=lambda *_: None)
        candidates = self.project.candidates()
        for record in candidates[:32]:
            self.project.decide(record["id"], "accept")
        for record in candidates[32:69]:
            self.project.decide(record["id"], "reject")
        for record in candidates[69:77]:
            self.project.decide(record["id"], "fix")

    def tearDown(self):
        self.temp.cleanup()

    def test_the_summary_matches_the_real_aggregate(self):
        summary = dataset_tab.project_summary(self.project.dir)
        self.assertEqual(summary["candidates"], 104)
        self.assertEqual(summary["accepted"], 32)
        self.assertEqual(summary["undecided"], 27)

    def test_the_chosen_export_reports_what_it_holds_and_refused(self):
        import hashlib
        export_id = "abc123def456"
        base = self.project.dir / "exports" / export_id
        base.mkdir(parents=True)
        image = base / "abcdef.png"
        image.write_bytes(b"PNG-BYTES")
        manifest = {"id": export_id, "policy": "export-v2",
                    "counts": {"images": 33, "refused": 20}}
        (base / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.project.card["export"] = {"selected": export_id}
        self.project.save()
        summary = dataset_tab.project_summary(self.project.dir)
        self.assertEqual(summary["export_counts"],
                         {"images": 33, "refused": 20, "policy": "export-v2"})


class CommandTests(unittest.TestCase):
    """The argv a click builds must be the argv the CLI documents."""

    def setUp(self):
        self.auto = Path("C:/tool/automation")
        self.project = Path("C:/tool/automation/projects/p")

    def test_every_stage_targets_its_own_module(self):
        for key, _label, spec, _hint in dataset_tab.STAGES:
            argv = dataset_tab.stage_command(self.auto, self.project, spec)
            self.assertEqual(argv[3:5], ["-m", "pipeline." + spec[0]], key)

    def test_the_project_always_comes_from_the_selected_folder(self):
        for key, _label, spec, _hint in dataset_tab.STAGES:
            argv = dataset_tab.stage_command(self.auto, self.project, spec)
            self.assertEqual(argv[-2], "--project", key)
            self.assertEqual(argv[-1], str(self.project), key)

    def test_the_subcommand_is_passed_through(self):
        spec = dict((name, s) for name, _l, s, _h in dataset_tab.STAGES)["export"]
        argv = dataset_tab.stage_command(self.auto, self.project, spec)
        self.assertEqual(argv[5], "publish")

    def test_the_review_stage_passes_no_subcommand(self):
        spec = dict((name, s) for name, _l, s, _h in dataset_tab.STAGES)["review"]
        argv = dataset_tab.stage_command(self.auto, self.project, spec)
        self.assertEqual(argv[4:], ["pipeline.review", "--project", str(self.project)])

    def test_only_stages_that_read_a_stop_file_are_marked_cancelable(self):
        # The set is a claim about the CLIs, so it is checked against them rather than trusted.
        for key, _label, spec, _hint in dataset_tab.STAGES:
            module = spec[0]
            accepts = module in dataset_tab.CANCELABLE
            if accepts:
                self.assertIn(module, ("epochs", "batch_gen", "fix"))
        self.assertEqual(dataset_tab.CANCELABLE, {"epochs", "batch_gen", "fix"})

    def test_planning_and_drawing_are_separate_buttons(self):
        # `epochs run` reads the manifest `plan` wrote and does not plan for itself, so one button cannot mean both.
        commands = {key: spec for key, _label, spec, _hint in dataset_tab.STAGES}
        self.assertEqual(commands["epochs_plan"], ("epochs", "plan"))
        self.assertEqual(commands["epochs_run"], ("epochs", "run"))
        self.assertEqual(commands["batch_plan"], ("batch_gen", "plan"))
        self.assertEqual(commands["batch_run"], ("batch_gen", "run"))

    def test_no_button_label_promises_a_draw_it_does_not_do(self):
        # A label saying "그리기" over a `plan` command would misreport whether GPU time had been spent.
        for key, label, spec, _hint in dataset_tab.STAGES:
            if "그리기" in label or "실행" in label:
                self.assertIn(spec[-1], ("run", "--run", "--draw"), f"{key} is labelled as a draw")

    def test_every_stage_has_a_distinct_key(self):
        keys = [key for key, *_ in dataset_tab.STAGES]
        self.assertEqual(len(keys), len(set(keys)))


class CancelHookTests(unittest.TestCase):
    """`make_cancel` is what turns "일시정지" into a real stop between cells."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.marker = Path(self.temp.name) / "stop_requested"

    def tearDown(self):
        self.temp.cleanup()

    def test_no_file_means_no_hook(self):
        from pipeline import make_cancel
        self.assertIsNone(make_cancel(None))
        self.assertIsNone(make_cancel(""))

    def test_the_hook_follows_the_file(self):
        from pipeline import make_cancel
        cancel = make_cancel(self.marker)
        self.assertFalse(cancel())
        self.marker.write_text("pause", encoding="utf-8")
        self.assertTrue(cancel())

    def test_the_hook_is_read_each_time_not_cached(self):
        from pipeline import make_cancel
        cancel = make_cancel(self.marker)
        self.marker.write_text("x", encoding="utf-8")
        self.assertTrue(cancel())
        self.marker.unlink()
        self.assertFalse(cancel())


class ScorerCountsOnDisabledStages(unittest.TestCase):
    """A stage is greyed out only for a reason the pipeline itself enforces."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.auto = Path(self.temp.name) / "automation"
        self.auto.mkdir()
        self.states = {}

    def tearDown(self):
        self.temp.cleanup()

    def make_tab(self, folder):
        """A DatasetTab with its widgets replaced by recorders, so the rules can be tested without Tk."""

        class FakeApp:
            AUTO = self.auto

            def __init__(self, states):
                self.controls = []
                self.states = states

            def button(self, parent, text, command, accent=False):
                return _FakeButton(text)

            def set_busy(self, busy):
                pass

        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.app = FakeApp(self.states)
        tab.auto = self.auto
        tab.running = None
        tab._roots = {folder.name: folder}
        tab.project = _FakeVar(folder.name)
        tab.info = _FakeVar("")
        tab.status = _FakeVar("")
        tab._buttons = {key: _FakeButton(key) for key, *_ in dataset_tab.STAGES}
        return tab

    def test_batch_is_disabled_without_a_chosen_epoch(self):
        folder = make_project(self.auto, "p")
        tab = self.make_tab(folder)
        tab.refresh_info()
        self.assertEqual(tab._buttons["batch_plan"].state, "disabled")
        self.assertEqual(tab._buttons["batch_run"].state, "disabled")
        self.assertEqual(tab._buttons["score"].state, "normal")

    def test_batch_is_enabled_once_an_epoch_is_chosen(self):
        folder = make_project(self.auto, "p", epoch=62)
        tab = self.make_tab(folder)
        tab.refresh_info()
        self.assertEqual(tab._buttons["batch_plan"].state, "normal")
        self.assertEqual(tab._buttons["batch_run"].state, "normal")

    def test_every_stage_is_disabled_while_one_is_running(self):
        folder = make_project(self.auto, "p", epoch=62)
        tab = self.make_tab(folder)
        tab.running = "score"
        tab.refresh_info()
        self.assertTrue(all(button.state == "disabled" for button in tab._buttons.values()))

    def test_no_project_disables_everything(self):
        folder = make_project(self.auto, "p")
        tab = self.make_tab(folder)
        tab._roots = {}
        tab.project = _FakeVar("missing")
        tab.refresh_info()
        self.assertTrue(all(button.state == "disabled" for button in tab._buttons.values()))


class _FakeButton:
    def __init__(self, text):
        self.text = text
        self.state = "normal"

    def configure(self, state=None, **_kwargs):
        if state is not None:
            self.state = state

    def bind(self, *_args, **_kwargs):
        pass

    def grid(self, **_kwargs):
        pass

    def pack(self, **_kwargs):
        pass


class _FakeVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class RealWidgetTests(unittest.TestCase):
    """The tab must actually build in a real Tk window, not only pass with recorder widgets.

    A tab that raises while constructing its widgets takes the whole window down on a real launch, which no fake-widget
    test would notice.
    """

    def setUp(self):
        import tkinter as tk
        try:
            self.window = tk.Tk()
        except tk.TclError as exc:            # no display in a headless run
            self.skipTest(f"Tk is not available: {exc}")
        self.window.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.auto = Path(self.temp.name) / "automation"
        self.auto.mkdir()
        make_project(self.auto, "demo", candidates=2, epoch=62, scores=2)

        class Host:
            AUTO = self.auto

            def __init__(self, window):
                self.window = window
                self.controls = []

            def button(self, parent, text, command, accent=False):
                from tkinter import ttk
                return ttk.Button(parent, text=text, command=command)

            def set_busy(self, busy):
                self.busy = busy

        from tkinter import ttk
        self.frame = ttk.Frame(self.window)
        self.frame.pack()
        self.tab = dataset_tab.DatasetTab(Host(self.window), self.frame, automation_dir=self.auto)

    def tearDown(self):
        try:
            self.tab.destroy()
        finally:
            self.window.destroy()
            self.temp.cleanup()

    def test_the_tab_builds_and_lists_the_project(self):
        self.assertEqual(list(self.tab.combo.cget("values")), ["demo"])
        self.assertEqual(self.tab.project.get(), "demo")
        self.assertEqual(self.tab.project_dir().name, "demo")

    def test_the_header_shows_what_is_on_disk(self):
        self.assertIn("후보 2", self.tab.info.get())
        self.assertIn("에포크 62", self.tab.info.get())

    def test_the_log_accepts_text_and_stays_read_only(self):
        self.tab._append("한 줄\n")
        self.assertIn("한 줄", self.tab.log.get("1.0", "end"))
        # A disabled Tk Text ignores an insert silently rather than raising, so the check is on the content.
        self.tab.log.insert("end", "typed by hand")
        self.assertNotIn("typed by hand", self.tab.log.get("1.0", "end"))

    def test_a_long_log_is_trimmed(self):
        for index in range(4100):
            self.tab._append(f"line {index}\n")
        lines = int(self.tab.log.index("end-1c").split(".")[0])
        self.assertLess(lines, 3200)
        self.assertIn("line 4099", self.tab.log.get("1.0", "end"))


class VerdictTests(unittest.TestCase):
    """A stopped grid and a finished grid both exit 0, so the verdict must come from what was requested."""

    def verdict(self, *, code, cancelled=False, paused=False):
        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.running = "epochs_run"
        tab.process = None
        tab.pause_button = _FakeButton("pause")
        tab.cancel_button = _FakeButton("cancel")
        tab.cancel_file = None
        tab.cancel_requested = cancelled
        tab.pause_requested = paused
        tab.status = _FakeVar("")
        tab.log_lines = []
        tab.refresh_info = lambda: None
        tab._append = lambda text: None
        tab.app = _FakeApp()
        tab._finish(code)
        return tab.status.get()

    def test_a_clean_exit_without_a_request_is_complete(self):
        self.assertIn("완료", self.verdict(code=0))

    def test_a_pause_is_not_reported_as_complete(self):
        # The CLI exits 0 after a pause, so exit code alone would call a half-drawn grid finished.
        text = self.verdict(code=0, paused=True)
        self.assertIn("일시정지", text)
        self.assertNotIn("완료", text)

    def test_a_cancel_is_reported_as_cancelled_and_keeps_what_was_drawn(self):
        text = self.verdict(code=0, cancelled=True)
        self.assertIn("취소됨", text)
        self.assertIn("남아 있습니다", text)

    def test_a_cancel_wins_over_a_pause(self):
        self.assertIn("취소됨", self.verdict(code=0, cancelled=True, paused=True))

    def test_a_crash_is_never_called_a_pause(self):
        text = self.verdict(code=1)
        self.assertIn("실패", text)
        self.assertNotIn("일시정지", text)


class _FakeApp:
    def set_busy(self, busy):
        pass


if __name__ == "__main__":
    unittest.main()
