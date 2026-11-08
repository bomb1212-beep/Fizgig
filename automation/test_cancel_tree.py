"""N2: cancelling and closing end the work they own, and nothing else.

The audit's `gui_cancel_child` reproduced the defect this file pins: the tab called `terminate()` on its own child, the
child's children kept running (a trainer, a tool2 run, a ComfyUI request), and the window closed as if the GPU were
free. `stale_stop_file` was the other half: a marker written by a paused run that was never cleaned up made the *next*
run draw nothing and exit 0, which the tab then reported as "완료".

Everything here uses real subprocesses and a dummy server of its own. No GPU work is started, and the dummy server
stands in for the user's ComfyUI: the point is that it is still running afterwards.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

AUTO = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTO))

import dataset_tab  # noqa: E402
import proctree  # noqa: E402


def sleeper(*, children=0):
    """A real process that starts `children` sleepers of its own, so a tree exists to be cancelled."""
    body = ("import subprocess, sys, time\n"
            "for _ in range(int(sys.argv[1])):\n"
            "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "time.sleep(120)\n")
    return subprocess.Popen([sys.executable, "-c", body, str(children)])


class ProcessTreeTests(unittest.TestCase):
    """The primitive the cancel path is built on."""

    def setUp(self):
        self.spawned = []

    def tearDown(self):
        for child in self.spawned:
            if child.poll() is None:
                child.kill()
                child.wait()

    def spawn(self, *, children=0):
        process = sleeper(children=children)
        self.spawned.append(process)
        return process

    def test_a_grandchild_is_part_of_the_tree_not_a_stranger(self):
        root = self.spawn(children=1)
        deadline = time.time() + 15
        while time.time() < deadline and not proctree.descendants(root.pid):
            time.sleep(0.05)
        kids = proctree.descendants(root.pid)
        self.assertTrue(kids, "the child never appeared in the process table")
        self.assertTrue(all(proctree.alive(pid) for pid in kids))

    def test_kill_tree_ends_the_whole_tree_and_says_so(self):
        root = self.spawn(children=2)
        deadline = time.time() + 15
        while time.time() < deadline and len(proctree.descendants(root.pid)) < 2:
            time.sleep(0.05)
        kids = proctree.descendants(root.pid)
        self.assertGreaterEqual(len(kids), 2)
        report = proctree.kill_tree(root.pid, log=lambda *_: None)
        root.wait(timeout=30)
        self.assertEqual(report["alive"], [], "a killed tree must leave nothing alive")
        self.assertTrue(set(kids).issubset(set(report["targets"])),
                        "the pids read before the kill must be covered by it")
        for pid in kids:
            self.assertFalse(proctree.alive(pid), f"pid {pid} outlived the tree kill")

    def test_stop_tree_asks_nicely_and_a_polite_stage_needs_no_kill(self):
        # The stage exits by itself once the marker appears: that is the path that keeps finished pictures.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        marker = Path(temp.name) / "stop_requested"
        polite = ("import pathlib, sys, time\n"
                  "marker = pathlib.Path(sys.argv[1])\n"
                  "deadline = time.time() + 20\n"
                  "while not marker.exists() and time.time() < deadline:\n"
                  "    time.sleep(0.05)\n")
        process = subprocess.Popen([sys.executable, "-c", polite, str(marker)])
        self.spawned.append(process)
        time.sleep(0.4)
        report = proctree.stop_tree(process.pid, cancel_file=marker, grace=10.0, log=lambda *_: None)
        self.assertTrue(report["graceful"])
        self.assertFalse(report["killed"], "a stage that stopped on its own must not be killed as well")
        self.assertEqual(report["alive"], [])
        self.assertTrue(marker.is_file(), "the marker is the request; it stays for the record and is removed by the caller")

    def test_stop_tree_ends_a_stage_that_ignores_the_marker(self):
        root = self.spawn(children=1)
        deadline = time.time() + 15
        while time.time() < deadline and not proctree.descendants(root.pid):
            time.sleep(0.05)
        kids = proctree.descendants(root.pid)
        report = proctree.stop_tree(root.pid, grace=0.3, log=lambda *_: None)
        root.wait(timeout=30)
        self.assertTrue(report["killed"], "a stage that never reads the marker has to be ended")
        self.assertEqual(report["alive"], [])
        for pid in kids:
            self.assertFalse(proctree.alive(pid), f"the ignored marker left pid {pid} running")

    def test_the_kill_is_scoped_to_the_tree_so_a_users_server_survives(self):
        # The dummy server is not this program's child, so it must be alive after the cancel. This is the constraint
        # the whole design exists for: the user's ComfyUI and LLM servers are never the product's to stop.
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"alive")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        root = self.spawn(children=1)
        report = proctree.stop_tree(root.pid, grace=0.2, log=lambda *_: None)
        root.wait(timeout=30)
        self.assertEqual(report["alive"], [])
        self.assertTrue(proctree.alive(os.getpid()))
        # And the server still answers: "the tree was killed" is not the same claim as "nothing else was".
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/", timeout=5) as response:
            self.assertEqual(response.read(), b"alive")


class _NeverStarted:
    """A `threading.Thread` stand-in: `start()` records nothing and runs nothing.

    `DatasetTab.start` builds its argv, writes the marker path and hands the launch to a thread. For the argv-level
    tests the launch itself is not the subject, so the thread must not run a real subprocess.
    """

    def __init__(self, target=None, daemon=None, **_kwargs):
        self.target = target

    def start(self):
        pass

    def join(self, timeout=None):
        pass


class _Recorder:
    """The bits of the Tk widgets `DatasetTab._finish` and `cancel` touch."""


    def __init__(self, text=""):
        self.text = text
        self.state = "normal"
        self.inserted = ""

    def configure(self, state=None, **_kwargs):
        if state is not None:
            self.state = state

    def get(self, *_args):
        return self.text + self.inserted

    def set(self, value):
        self.text = value

    def insert(self, _index, text):
        self.inserted += text

    def see(self, _index):
        pass

    def index(self, _index):
        return "1.0"


class _Host:
    AUTO = AUTO

    def __init__(self, automation):
        self.AUTO = automation
        self.controls = []
        self.busy = False

    def button(self, parent, text, command, accent=False):
        return _Recorder(text)

    def set_busy(self, busy):
        self.busy = busy


class CancelWiringTests(unittest.TestCase):
    """The tab's cancel path, driven with a real process tree and a recorder for the widgets."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp.name)
        self.auto = self.root_dir / "automation"
        self.project = self.auto / "projects" / "p"
        self.project.mkdir(parents=True)
        (self.project / "project.json").write_text(json.dumps({"id": "p", "trigger": "trg"}), encoding="utf-8")
        self.spawned = []

    def tearDown(self):
        for child in self.spawned:
            if child.poll() is None:
                child.kill()
                child.wait()
        self.temp.cleanup()

    def make_tab(self):
        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.app = _Host(self.auto)
        tab.auto = self.auto
        tab.running = "epochs_run"
        tab._roots = {"p": self.project}
        tab.project = _Recorder("p")
        tab.info = _Recorder("")
        tab.status = _Recorder("")
        tab._buttons = {}
        tab.events = __import__("queue").Queue()
        tab.log_lines = []
        tab.log = _Recorder("")
        tab.process = None
        tab.worker = None
        tab.cancel_requested = False
        tab.pause_requested = False
        tab.pause_noted = False
        tab.cancel_report = None
        tab.cancel_file = self.project / "stop_requested.epochs_run"
        tab.started_at = time.time()
        tab.last_log_path = None
        tab.pause_button = _Recorder("pause")
        tab.cancel_button = _Recorder("cancel")
        tab.refresh_info = lambda: tab.status.set("준비됨")
        # `_append` is the product's own, so the log file and the pause wording are exercised rather than stubbed.
        return tab

    def test_the_marker_stops_the_stage_and_the_tree_when_it_ignores_it(self):
        root = sleeper(children=1)
        self.spawned.append(root)
        deadline = time.time() + 15
        while time.time() < deadline and not proctree.descendants(root.pid):
            time.sleep(0.05)
        kids = proctree.descendants(root.pid)
        self.assertTrue(kids)
        tab = self.make_tab()
        tab.process = root
        with patch.object(dataset_tab, "CANCEL_GRACE", 0.3):
            tab.cancel()
            # The cancel worker runs off the Tk thread; wait for it the way the poll loop would. The placeholder
            # exists immediately, so the wait is on `in_flight` going false, not on the dict appearing.
            deadline = time.time() + 40
            while time.time() < deadline and (tab.cancel_report or {}).get("in_flight"):
                time.sleep(0.05)
        root.wait(timeout=30)
        self.assertIsNotNone(tab.cancel_report)
        self.assertTrue(tab.cancel_report["killed"])
        self.assertEqual(tab.cancel_report["alive"], [], "no process this stage started may outlive the cancel")
        for pid in kids:
            self.assertFalse(proctree.alive(pid))
        self.assertTrue(tab.cancel_file.is_file(), "the marker is written before anything is killed")

    def test_the_run_removes_its_own_marker_when_it_finishes(self):
        tab = self.make_tab()
        tab.cancel_file.write_text("pause\n", encoding="utf-8")
        tab._finish(0)
        self.assertFalse(tab.cancel_file.exists(), "a leftover marker would make the next run draw nothing")

    def test_a_stage_gets_its_own_marker_so_an_old_one_is_never_inherited(self):
        # spec N2 `stale_stop_file`: a fixed `<project>/stop_requested` left behind by a closed window made the next
        # `epochs run` draw 0 cells and exit 0, which the tab reported as "완료".
        stale = self.project / "stop_requested"
        stale.write_text("pause\n", encoding="utf-8")
        tab = self.startable_tab()
        with patch.object(dataset_tab.threading, "Thread", _NeverStarted):
            tab.start("epochs_run")
        self.assertIsNotNone(tab.cancel_file)
        self.assertNotEqual(tab.cancel_file, stale, "the run must not reuse a marker an earlier run may have left")
        self.assertIn("epochs_run", tab.cancel_file.name)
        tab.running = None

    def test_a_pause_written_by_a_previous_window_does_not_poison_the_next_run(self):
        # The marker the new run passes is deleted before launch, so the stage starts with a clean slate.
        marker_path = self.project / "stop_requested.epochs_run"
        marker_path.write_text("pause\n", encoding="utf-8")
        tab = self.startable_tab()
        with patch.object(dataset_tab.threading, "Thread", _NeverStarted):
            tab.start("epochs_run")
        self.assertFalse(marker_path.exists(), "the stage must not find a marker before it even starts")
        tab.running = None

    def test_a_stage_that_takes_no_marker_is_cancelled_by_the_tree_alone(self):
        # `score` reads no stop file, so the tab must not offer it a pause; the cancel still has to end it.
        tab = self.startable_tab()
        with patch.object(dataset_tab.threading, "Thread", _NeverStarted):
            tab.start("score")
        self.assertIsNone(tab.cancel_file)
        self.assertEqual(tab.pause_button.state, "disabled")
        self.assertEqual(tab.cancel_button.state, "normal")
        tab.running = None

    def test_the_editing_stages_are_pauseable_now(self):
        # spec N2: both already took `--cancel-file` and poll it at a slot boundary, so the button was hidden for no
        # reason - the GUI went straight to terminate where a clean stop was available.
        for key in ("gen_a", "gen_ph3"):
            tab = self.startable_tab()
            with patch.object(dataset_tab.threading, "Thread", _NeverStarted):
                tab.start(key)
            self.assertIsNotNone(tab.cancel_file, key)
            self.assertEqual(tab.pause_button.state, "normal", key)
            tab.running = None

    def startable_tab(self):
        """A tab bare enough for `start()` but with every attribute it reads."""
        tab = dataset_tab.DatasetTab.__new__(dataset_tab.DatasetTab)
        tab.app = _Host(self.auto)
        tab.auto = self.auto
        tab.running = None
        tab._roots = {"p": self.project}
        tab.project = _Recorder("p")
        tab.info = _Recorder("")
        tab.status = _Recorder("")
        tab._buttons = {key: _Recorder(key) for key, *_ in dataset_tab.STAGES}
        tab.log = _Recorder("")
        tab.pause_button = _Recorder("pause")
        tab.cancel_button = _Recorder("cancel")
        tab.events = __import__("queue").Queue()
        tab.refresh_info = lambda: None
        tab.last_log_path = None
        tab.started_at = None
        tab.pause_noted = False
        return tab

    def test_a_paused_stage_is_recognised_from_its_own_output(self):
        # A marker left by an earlier window can stop a stage before this process asked; exit 0 alone would call that
        # a finished grid. The CLI's own wording is the evidence.
        tab = self.make_tab()
        tab._append("paused: the remaining cells are still pending, so this command continues them\n")
        self.assertTrue(tab.pause_noted)

    def test_output_is_kept_in_a_file_under_the_project(self):
        # spec N5: the third seed run died and there was no file to say why, because the log lived only in a widget.
        tab = self.make_tab()
        tab._append("hello\n")
        path = tab.log_path()
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())
        self.assertIn("hello", path.read_text(encoding="utf-8"))
        self.assertEqual(path.parent, self.project / "logs")
        # A second write appends rather than replacing, so a long stage keeps its whole record.
        tab._append("world\n")
        self.assertIn("world", path.read_text(encoding="utf-8"))
        self.assertIn("hello", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
