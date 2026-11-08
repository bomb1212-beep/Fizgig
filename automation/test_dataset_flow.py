"""N5: the GUI path a real project can actually walk, and the state it must not lose.

The audit's two reproductions drive this file:

* `gui_epoch_default` - the epoch buttons passed no `--run`, and `find_run()` refuses a project with more than one run,
  so the whole epoch path was dead from the GUI. `suamom` has three runs.
* `gui_failure_status` - `_finish()` set the verdict and then called `refresh_info()`, which recomputed the status, so
  a failed stage ended up displayed as "준비됨".

Plus the run list itself: an aborted run with zero checkpoints was a candidate for "the only run", which made every
command demand `--run`.

Everything here reads temporary projects and builds argv. Nothing starts a GPU job or touches the real `suamom`.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AUTO = Path(__file__).resolve().parents[1]
for entry in (str(AUTO.parent / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import dataset_tab  # noqa: E402


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def safetensors_bytes(epoch):
    """A byte-accurate safetensors file whose header states the epoch."""
    import struct
    payload = b"x" * 32
    header = {"__metadata__": {"ss_epoch": str(epoch)},
              "lora.a": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(payload)]}}
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + payload


def make_project(root, name="p", *, runs=(), epoch=None, exports=()):
    """A project with the runs/epochs/exports a person would see on disk.

    `runs` is `[(run_name, snapshot, [epochs])]`; an entry with an empty epoch list is the aborted run of the audit.
    """
    folder = Path(root) / "projects" / name
    card = {"id": name, "trigger": "trg", "slot_quotas": {}, "final_families": ["krea2"]}
    if epoch is not None:
        card["krea2_inference"] = {"epoch": epoch}
    write_json(folder / "project.json", card)
    (folder / "sources.jsonl").write_text("", encoding="utf-8")
    (folder / "candidates.jsonl").write_text("", encoding="utf-8")
    (folder / "review.json").write_text("{}", encoding="utf-8")
    for run_name, snapshot, epochs in runs:
        output = folder / "seed_lora" / snapshot / "runs" / run_name / "output"
        output.mkdir(parents=True, exist_ok=True)
        write_json(output.parent / "snapshot.json", {"id": snapshot})
        for value in (epochs or []):
            (output / f"char-{value:06d}.safetensors").write_bytes(safetensors_bytes(value))
        if not epochs:
            write_json(output.parent / "state.json", {"complete": False, "active_stage": "Text Encoder Caching"})
    for export_id in exports:
        base = folder / "exports" / export_id
        base.mkdir(parents=True, exist_ok=True)
        write_json(base / "export_manifest.json",
                   {"id": export_id, "created_at": f"2026-11-0{len(export_id) % 9}T10:00:00",
                    "counts": {"images": 27, "refused": 88}, "policy": "export-v2"})
    return folder


class RunListTests(unittest.TestCase):
    """What the tab shows about runs, from disk, without loading a model."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_an_aborted_run_is_listed_but_not_offered(self):
        folder = make_project(self.temp.name, runs=[("char-good", "snap1", [10, 20]),
                                                    ("char-dead", "snap2", [])])
        rows = {row["name"]: row for row in dataset_tab.runs_for_project(folder)}
        self.assertTrue(rows["char-good"]["usable"])
        self.assertEqual(rows["char-good"]["checkpoints"], 2)
        self.assertFalse(rows["char-dead"]["usable"], "a run with no checkpoint cannot be evaluated")
        self.assertEqual(rows["char-dead"]["checkpoints"], 0)
        self.assertEqual(rows["char-dead"]["stage"], "Text Encoder Caching")
        self.assertTrue((folder / "seed_lora" / "snap2" / "runs" / "char-dead").is_dir(),
                        "the dead run's folder is preserved")

    def test_the_epochs_of_a_run_come_from_header_checked_checkpoints(self):
        folder = make_project(self.temp.name, runs=[("char-good", "snap1", [10, 20])])
        run = dataset_tab.runs_for_project(folder)[0]
        rows = dataset_tab.epochs_for_run(run["folder"])
        self.assertEqual([row["epoch"] for row in rows], [10, 20])
        self.assertTrue(all(row["usable"] for row in rows))

    def test_a_truncated_checkpoint_is_listed_as_unusable(self):
        folder = make_project(self.temp.name, runs=[("char-good", "snap1", [10])])
        run = dataset_tab.runs_for_project(folder)[0]
        (Path(run["folder"]) / "output" / "char-000011.safetensors").write_bytes(b"truncated")
        rows = {row["epoch"]: row for row in dataset_tab.epochs_for_run(run["folder"])}
        self.assertTrue(rows[10]["usable"])
        self.assertFalse(rows[11]["usable"])
        self.assertIn("not a readable", rows[11]["note"])

    def test_exports_are_listed_newest_first_with_their_counts(self):
        folder = make_project(self.temp.name, runs=[("char-good", "snap1", [10])],
                              exports=["aaa111222333", "bbb444555666"])
        rows = dataset_tab.exports_for_project(folder)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["images"] for row in rows}, {27})
        self.assertEqual(rows[0]["created_at"] >= rows[1]["created_at"], True)


class _Widget:
    """A recorder standing in for a Tk widget."""

    def __init__(self, text=""):
        self.text = text
        self.state = "normal"
        self.values = []

    def configure(self, state=None, values=None, **_kwargs):
        if state is not None:
            self.state = state
        if values is not None:
            self.values = list(values)

    def bind(self, *_a, **_k):
        pass

    def pack(self, **_k):
        pass

    def grid(self, **_k):
        pass

    def cget(self, key):
        return self.values if key == "values" else self.text

    def insert(self, _index, text):
        pass

    def see(self, _index):
        pass

    def index(self, _index):
        return "1.0"

    def get(self, *_a):
        return self.text

    def set(self, value):
        self.text = value


class _App:
    def __init__(self, automation):
        self.AUTO = automation
        self.controls = []
        self.busy = False

    def button(self, parent, text, command, accent=False):
        return _Widget(text)

    def set_busy(self, busy):
        self.busy = busy


class PickTests(unittest.TestCase):
    """The run/epoch/export picks, and what every epoch button does with them."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.auto = Path(self.temp.name) / "automation"
        self.auto.mkdir()
        self.folder = make_project(self.auto, "p",
                                   runs=[("char-good", "snap1", [10, 20]), ("char-dead", "snap2", [])],
                                   epoch=20, exports=["aaa111222333"])

    def tab(self):
        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.app = _App(self.auto)
        tab.auto = self.auto
        tab.running = None
        tab._roots = {"p": self.folder}
        tab.project = _Widget("p")
        tab.run = _Widget("")
        tab.epoch = _Widget("")
        tab.export_choice = _Widget("")
        tab.status = _Widget("")
        tab.info = _Widget("")
        tab.log = _Widget("")
        tab._buttons = {key: _Widget(key) for key, *_ in dataset_tab.STAGES}
        tab.run_combo = _Widget()
        tab.epoch_combo = _Widget()
        tab.export_combo = _Widget()
        tab.pause_button = _Widget("pause")
        tab.cancel_button = _Widget("cancel")
        tab.events = __import__("queue").Queue()
        tab.last_log_path = None
        tab.started_at = None
        tab.pause_noted = False
        tab.cancel_report = None
        tab.cancel_requested = False
        tab.pause_requested = False
        tab.cancel_file = None
        tab.process = None
        tab.worker = None
        return tab

    def test_the_only_usable_run_is_picked_and_the_dead_one_is_labelled(self):
        tab = self.tab()
        tab.refresh_info()
        labels = tab.run_combo.values
        self.assertEqual(len(labels), 2)
        self.assertTrue(any("체크포인트 없음" in label for label in labels),
                        "an aborted run has to be visible and marked, not hidden")
        self.assertIn("char-good", tab.run.get())

    def test_the_epoch_combo_offers_the_runs_checkpoints(self):
        tab = self.tab()
        tab.refresh_info()
        self.assertEqual(len(tab.epoch_combo.values), 2)
        self.assertIn("20", tab.epoch.get())
        self.assertEqual(tab.chosen_epoch(), 20)

    def test_every_epoch_button_passes_the_picked_run(self):
        # `gui_epoch_default`: without `--run` every epoch stage died on a project with more than one run.
        tab = self.tab()
        tab.refresh_info()
        for key in ("epochs_list", "epochs_plan", "epochs_run", "epochs_sheet"):
            spec = next(s for name, _l, s, _h in dataset_tab.STAGES if name == key)
            argv = dataset_tab.stage_command(self.auto, self.folder, tab._with_picked_run(spec))
            self.assertIn("--run", argv, key)
            self.assertEqual(argv[argv.index("--run") + 1], "char-good", key)

    def test_a_stage_that_is_not_an_epoch_stage_gets_no_run_argument(self):
        tab = self.tab()
        tab.refresh_info()
        spec = next(s for name, _l, s, _h in dataset_tab.STAGES if name == "score")
        self.assertNotIn("--run", tab._with_picked_run(spec))

    def test_the_export_combo_defaults_to_the_projects_own_choice(self):
        write_json(self.folder / "project.json",
                   {"id": "p", "trigger": "trg", "slot_quotas": {}, "final_families": ["krea2"],
                    "krea2_inference": {"epoch": 20}, "export": {"selected": "aaa111222333"}})
        tab = self.tab()
        tab.refresh_info()
        self.assertIn("aaa111222333", tab.export_choice.get())
        self.assertEqual(tab._export_selected(), "aaa111222333")


class ChooseEpochTests(unittest.TestCase):
    """Recording an epoch happens only on an explicit click, through the CLI's own `choose`."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.auto = Path(self.temp.name) / "automation"
        self.auto.mkdir()
        self.folder = make_project(self.auto, "p", runs=[("char-good", "snap1", [10, 20])])

    def make_tab(self):
        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.app = _App(self.auto)
        tab.auto = self.auto
        tab.running = None
        tab.parent = None
        tab._roots = {"p": self.folder}
        tab.project = _Widget("p")
        tab.run = _Widget("")
        tab.epoch = _Widget("")
        tab.export_choice = _Widget("")
        tab.status = _Widget("")
        tab.info = _Widget("")
        tab.log = _Widget("")
        tab._buttons = {key: _Widget(key) for key, *_ in dataset_tab.STAGES}
        tab.run_combo = _Widget()
        tab.epoch_combo = _Widget()
        tab.export_combo = _Widget()
        tab.pause_button = _Widget("pause")
        tab.cancel_button = _Widget("cancel")
        tab.events = __import__("queue").Queue()
        tab.last_log_path = None
        tab.started_at = None
        tab.pause_noted = False
        tab.cancel_report = None
        tab.cancel_requested = False
        tab.pause_requested = False
        tab.cancel_file = None
        tab.process = None
        tab.worker = None
        return tab

    def test_choosing_builds_the_cli_command_with_the_run_and_the_epoch(self):
        tab = self.make_tab()
        tab.refresh_info()
        launched = []

        def fake_run_stage(key, label, argv, *, cancelable):
            launched.append({"key": key, "label": label, "argv": argv, "cancelable": cancelable})

        tab._run_stage = fake_run_stage
        tab.choose_epoch()
        self.assertEqual(len(launched), 1)
        argv = launched[0]["argv"]
        self.assertEqual(argv[3:5], ["-m", "pipeline.epochs"])
        self.assertEqual(argv[5], "choose")
        self.assertEqual(argv[argv.index("--run") + 1], "char-good")
        self.assertEqual(argv[argv.index("--epoch") + 1], "10" if tab.chosen_epoch() == 10 else "20")
        self.assertFalse(launched[0]["cancelable"], "choosing an epoch is a decision, not a long draw")

    def test_refreshing_never_records_a_choice(self):
        # Opening the tab, switching projects and changing a drop-down must not write the card.
        before = (self.folder / "project.json").read_text(encoding="utf-8")
        tab = self.make_tab()
        tab.refresh_info()
        tab.refresh_info()
        tab.epoch.set(tab.epoch_combo.values[0])
        tab.refresh_run_details()
        after = (self.folder / "project.json").read_text(encoding="utf-8")
        self.assertEqual(before, after, "no view action may record an epoch")

    def test_choosing_without_a_usable_checkpoint_explains_itself(self):
        folder = make_project(self.auto, "dead", runs=[("char-dead", "snap2", [])])
        tab = self.make_tab()
        tab._roots = {"dead": folder}
        tab.project = _Widget("dead")
        tab.refresh_info()
        shown = []
        with patch.object(dataset_tab.messagebox, "showinfo", side_effect=lambda *a, **k: shown.append(a)):
            tab.choose_epoch()
        self.assertTrue(shown, "a person who clicked with nothing to choose must be told")


class StatusTests(unittest.TestCase):
    """`gui_failure_status`: the verdict must outlive the header refresh."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.auto = Path(self.temp.name) / "automation"
        self.auto.mkdir()
        self.folder = make_project(self.auto, "p", runs=[("char-good", "snap1", [10])])

    def tab(self):
        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.app = _App(self.auto)
        tab.auto = self.auto
        tab.running = "score"
        tab._roots = {"p": self.folder}
        tab.project = _Widget("p")
        tab.run = _Widget("")
        tab.epoch = _Widget("")
        tab.export_choice = _Widget("")
        tab.status = _Widget("")
        tab.info = _Widget("")
        tab.log = _Widget("")
        tab._buttons = {key: _Widget(key) for key, *_ in dataset_tab.STAGES}
        tab.run_combo = None            # the stage-rule path runs without pick widgets
        tab.pause_button = _Widget("pause")
        tab.cancel_button = _Widget("cancel")
        tab.cancel_file = None
        tab.cancel_requested = False
        tab.pause_requested = False
        tab.pause_noted = False
        tab.cancel_report = None
        tab.last_log_path = None
        tab.process = None
        return tab

    def test_a_failed_stage_stays_failed(self):
        tab = self.tab()
        tab._finish(1)
        self.assertIn("실패", tab.status.get())
        self.assertNotIn("준비됨", tab.status.get())

    def test_a_completed_stage_says_so(self):
        tab = self.tab()
        tab._finish(0)
        self.assertIn("완료", tab.status.get())

    def test_a_stage_whose_own_output_says_paused_is_not_called_complete(self):
        # A pause marker left by an earlier window can stop a stage before this process asked for it.
        tab = self.tab()
        tab.pause_noted = True
        tab._finish(0)
        self.assertIn("일시정지", tab.status.get())
        self.assertNotIn("완료", tab.status.get())


if __name__ == "__main__":
    unittest.main()
