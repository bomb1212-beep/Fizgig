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

    def clicked(self, tab, key):
        """The argv the button `key` would launch, captured instead of started."""
        seen = []
        with patch.object(tab, "_run_stage", lambda _key, _label, argv, cancelable: seen.append(argv)):
            tab.start(key)
        self.assertEqual(len(seen), 1, key)
        return seen[0]

    def test_the_final_training_button_uses_the_picked_export_not_the_newest(self):
        # The todos button passed no `--id`, so `export todos` fell back to the newest export whatever the drop-down
        # showed: picking an older export and clicking "최종 학습 준비" added a job for a different training set.
        write_json(self.folder / "exports" / "bbb444555666" / "export_manifest.json",
                   {"id": "bbb444555666", "created_at": "2026-11-09T10:00:00",
                    "counts": {"images": 36, "refused": 72}, "policy": "export-v2"})
        tab = self.tab()
        tab.refresh_info()
        self.assertTrue(tab.export_choice.get().startswith("bbb444555666"), "the newest is the default")
        older = next(label for label in tab.export_combo.values if label.startswith("aaa111222333"))
        tab.export_choice.set(older)
        argv = self.clicked(tab, "todos")
        self.assertIn("--id", argv)
        self.assertEqual(argv[argv.index("--id") + 1], "aaa111222333")
        self.assertEqual(argv[argv.index("pipeline.export"):argv.index("pipeline.export") + 3],
                         ["pipeline.export", "todos", "--add"])

    def test_without_a_published_export_the_final_training_button_adds_no_id(self):
        folder = make_project(self.auto, "q", runs=[("char-good", "snap1", [10])])
        tab = self.tab()
        tab._roots = {"q": folder}
        tab.project = _Widget("q")
        tab.refresh_info()
        self.assertNotIn("--id", self.clicked(tab, "todos"))

    def test_only_the_final_training_button_takes_the_export(self):
        tab = self.tab()
        tab.refresh_info()
        for key in ("export", "score", "audit"):
            self.assertNotIn("--id", self.clicked(tab, key), key)


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


class ModalLocationTests(unittest.TestCase):
    """The 학습 위치 dropdown: Modal adds --backend/--modal-gpu to the two training buttons; the rest stays local."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.auto = Path(self.temp.name) / "automation"
        self.auto.mkdir()
        self.folder = make_project(self.auto, "p", runs=[("char-good", "snap1", [10])],
                                   epoch=10, exports=["aaa111222333"])
        dataset_tab.DatasetTab._reset_sync_claim()

    def tearDown(self):
        dataset_tab.DatasetTab._reset_sync_claim()

    def tab(self, location="로컬"):
        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.app = _App(self.auto)
        tab.auto = self.auto
        tab.running = None
        tab.running_backend = None
        tab.remote_cancel_file = None
        tab._roots = {"p": self.folder}
        tab.project = _Widget("p")
        tab.location = _Widget(location)
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
        tab.log_lines = []
        return tab

    def clicked(self, tab, key):
        seen = []
        with patch.object(tab, "_run_stage", lambda _key, _label, argv, cancelable: seen.append(argv)):
            tab.start(key)
        self.assertEqual(len(seen), 1, key)
        return seen[0]

    def test_local_seed_train_argv_is_unchanged(self):
        tab = self.tab("로컬")
        tab.refresh_info()
        argv = self.clicked(tab, "seed_train")
        self.assertNotIn("--backend", argv)
        self.assertNotIn("--modal-gpu", argv)
        self.assertEqual(argv[argv.index("pipeline.seed"):argv.index("pipeline.seed") + 2],
                         ["pipeline.seed", "--build"])
        self.assertTrue(argv[-2] == "--project")

    def test_modal_seed_train_appends_backend_and_gpu(self):
        tab = self.tab("Modal")
        tab.refresh_info()
        argv = self.clicked(tab, "seed_train")
        self.assertIn("--backend", argv)
        self.assertEqual(argv[argv.index("--backend") + 1], "modal")
        self.assertEqual(argv[argv.index("--modal-gpu") + 1], "L40S")
        self.assertTrue(argv[-2] == "--project")
        self.assertEqual(tab.running_backend, "modal")
        self.assertTrue(str(tab.remote_cancel_file).endswith("cancel_remote.request"))

    def test_modal_todos_keeps_the_picked_export_and_adds_the_backend(self):
        tab = self.tab("Modal")
        tab.refresh_info()
        argv = self.clicked(tab, "todos")
        self.assertIn("--backend", argv)
        self.assertEqual(argv[argv.index("--backend") + 1], "modal")
        self.assertEqual(argv[argv.index("--modal-gpu") + 1], "L40S")
        self.assertIn("--id", argv)
        self.assertEqual(argv[argv.index("--id") + 1], "aaa111222333")
        self.assertEqual(tab.running_backend, None)   # todos is not a streaming stage
        self.assertIsNone(tab.remote_cancel_file)

    def test_modal_guidance_is_shown_for_both_training_buttons(self):
        tab = self.tab("Modal")
        tab.refresh_info()
        with patch.object(tab, "_append") as append:
            self.clicked(tab, "seed_train")
        text = "".join(str(call.args[0]) for call in append.call_args_list)
        self.assertIn("Modal", text)
        with patch.object(tab, "_append") as append:
            self.clicked(tab, "todos")
        text = "".join(str(call.args[0]) for call in append.call_args_list)
        self.assertIn("Modal", text)
        self.assertIn("todo", text)

    def test_explicit_cancel_writes_the_remote_request_and_kills_nothing(self):
        tab = self.tab("Modal")
        tab.refresh_info()
        self.clicked(tab, "seed_train")
        tab.running = "seed_train"   # the real _run_stage sets this; the fake above does not
        with patch.object(dataset_tab.proctree, "stop_tree") as stop:
            tab.cancel()
            stop.assert_not_called()
        request = self.folder / "cancel_remote.request"
        self.assertTrue(request.is_file())
        self.assertEqual(request.read_text(encoding="utf-8"), "cancel\n")
        self.assertEqual(tab.cancel_report, {"remote": True})
        self.assertEqual(tab.status.get(), "원격 취소 대기 중")

    def test_a_modal_stage_ends_the_local_tracking_only(self):
        tab = self.tab("Modal")
        tab.refresh_info()
        self.clicked(tab, "seed_train")
        tab.running = "seed_train"
        with patch.object(dataset_tab.proctree, "stop_tree", return_value={"graceful": False, "killed": True,
                                                                            "targets": [], "alive": []}) as stop:
            self.assertTrue(tab.terminate_local())
            stop.assert_called_once()
        self.assertIsNone(tab.running_backend)
        self.assertIsNone(tab.remote_cancel_file)
        self.assertFalse((self.folder / "cancel_remote.request").exists(),
                         "ending local tracking must not cancel the remote run")

    def test_initial_sync_is_claimed_once_per_process(self):
        tab = self.tab()
        with patch.object(dataset_tab.DatasetTab, "_launch_initial_sync") as launch:
            tab._start_initial_sync()
            tab._start_initial_sync()
            launch.assert_called_once()

    def test_two_hosts_share_the_single_initial_sync(self):
        first = self.tab()
        second = self.tab()
        second.app = _App(self.auto)
        with patch.object(dataset_tab.DatasetTab, "_launch_initial_sync") as launch:
            first._start_initial_sync()
            second._start_initial_sync()
            launch.assert_called_once()


class SettingsFieldTests(unittest.TestCase):
    """목표 장수 / 시드 에포크 / 최종 스텝: loaded per project, saved by the button, and saved before training."""

    setUp = ModalLocationTests.setUp
    tearDown = ModalLocationTests.tearDown
    clicked = ModalLocationTests.clicked

    def tab(self):
        tab = ModalLocationTests.tab(self, "로컬")
        tab.target_images = _Widget("")
        tab.seed_epochs = _Widget("")
        tab.final_steps = _Widget("")
        tab._settings_for = None
        tab.logged = []
        tab._append = tab.logged.append
        return tab

    def test_a_project_without_settings_shows_the_defaults(self):
        tab = self.tab()
        tab.refresh_info()
        self.assertEqual((tab.target_images.get(), tab.seed_epochs.get(), tab.final_steps.get()),
                         ("50", "30", "2000"))

    def test_the_fields_show_what_the_project_saved(self):
        from pipeline import settings
        settings.write(self.folder, {"target_images": 60, "seed_epochs": 25, "final_steps": 2400})
        tab = self.tab()
        tab.refresh_info()
        self.assertEqual((tab.target_images.get(), tab.seed_epochs.get(), tab.final_steps.get()),
                         ("60", "25", "2400"))

    def test_a_refresh_keeps_numbers_typed_but_not_saved(self):
        tab = self.tab()
        tab.refresh_info()
        tab.final_steps.set("2500")
        tab.refresh_info()                     # e.g. after a stage finishes
        self.assertEqual(tab.final_steps.get(), "2500")

    def test_the_button_saves_the_fields(self):
        from pipeline import settings
        tab = self.tab()
        tab.refresh_info()
        tab.target_images.set("40")
        tab.final_steps.set("1800")
        self.assertIsNotNone(tab.save_settings())
        self.assertEqual(settings.read(self.folder), {"target_images": 40, "seed_epochs": 30, "final_steps": 1800})
        self.assertTrue(any("기본값 저장" in line for line in tab.logged))

    def test_seed_training_saves_and_passes_the_epochs(self):
        from pipeline import settings
        tab = self.tab()
        tab.refresh_info()
        tab.seed_epochs.set("35")
        argv = self.clicked(tab, "seed_train")
        self.assertEqual(argv[argv.index("--epochs") + 1], "35")
        self.assertEqual(settings.read(self.folder)["seed_epochs"], 35)
        self.assertEqual(argv[-2], "--project")

    def test_the_seed_set_picks_the_policy(self):
        for label, policy in dataset_tab.SEED_SETS:
            tab = self.tab()
            tab.seed_set = _Widget(label)
            tab.refresh_info()
            argv = self.clicked(tab, "seed_train")
            self.assertEqual(argv[argv.index("--policy") + 1], policy)
        self.assertEqual(dataset_tab.SEED_SETS[0], ("원본만", "seed-originals"))

    def test_final_todo_saves_the_steps_it_will_write(self):
        from pipeline import settings
        tab = self.tab()
        tab.refresh_info()
        tab.final_steps.set("2200")
        argv = self.clicked(tab, "todos")
        self.assertNotIn("--epochs", argv)
        self.assertEqual(settings.read(self.folder)["final_steps"], 2200)

    def test_a_bad_number_stops_the_stage_before_it_runs(self):
        from pipeline import settings
        tab = self.tab()
        tab.refresh_info()
        tab.seed_epochs.set("thirty")
        seen = []
        with patch.object(dataset_tab.messagebox, "showerror") as shown, \
                patch.object(tab, "_run_stage", lambda *args, **kwargs: seen.append(args)):
            tab.start("seed_train")
        self.assertEqual(seen, [])
        shown.assert_called_once()
        self.assertFalse(settings.path(self.folder).exists())


class CharacterFieldTests(unittest.TestCase):
    """나이 / 고정 특징 / 얼굴 특징: loaded from the card with the project, saved into the card by the button."""

    setUp = ModalLocationTests.setUp
    tearDown = ModalLocationTests.tearDown

    def tab(self):
        tab = SettingsFieldTests.tab(self)
        tab.age = _Widget("")
        tab.fixed_traits = _Widget("")
        tab.face_traits = _Widget("")
        return tab

    def card(self):
        return json.loads((self.folder / "project.json").read_text(encoding="utf-8"))

    def test_the_fields_show_the_card(self):
        card = self.card()
        card.update(profile={"age": 28}, fixed_traits=["low ponytail"], face_traits=["mole under her eye"])
        (self.folder / "project.json").write_text(json.dumps(card), encoding="utf-8")
        tab = self.tab()
        tab.refresh_info()
        self.assertEqual((tab.age.get(), tab.fixed_traits.get(), tab.face_traits.get()),
                         ("28", "low ponytail", "mole under her eye"))

    def test_the_button_writes_the_card_and_keeps_the_rest_of_it(self):
        tab = self.tab()
        tab.refresh_info()
        tab.age.set("28")
        tab.fixed_traits.set("low ponytail, black scrunchie")
        tab.face_traits.set("mole under her eye")
        self.assertIsNotNone(tab.save_character())
        card = self.card()
        self.assertEqual(card["profile"], {"age": 28})
        self.assertEqual(card["fixed_traits"], ["low ponytail", "black scrunchie"])
        self.assertEqual(card["face_traits"], ["mole under her eye"])
        self.assertEqual(card["trigger"], "trg")
        self.assertTrue(any("캐릭터 저장" in line for line in tab.logged))

    def test_a_bad_age_writes_nothing(self):
        before = self.card()
        tab = self.tab()
        tab.refresh_info()
        tab.age.set("스물여덟")
        with patch.object(dataset_tab.messagebox, "showerror") as shown:
            self.assertIsNone(tab.save_character())
        shown.assert_called_once()
        self.assertEqual(self.card(), before)


if __name__ == "__main__":
    unittest.main()
