"""The per-project numbers (목표 장수 / 시드 에포크 / 최종 스텝) and the two commands that read them.

  venv\\Scripts\\python.exe -B -m unittest pipeline.test_pipeline_settings      (from automation\\)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AUTO = Path(__file__).resolve().parents[1]
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))

from pipeline import export, project as pj, seed, settings  # noqa: E402
from pipeline.test_pipeline_export import FakeProject, write_minimal_export  # noqa: E402


class SettingsFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_an_old_project_reads_the_defaults_without_being_written(self):
        self.assertEqual(settings.read(self.folder), {"target_images": 50, "seed_epochs": 30, "final_steps": 2000})
        self.assertFalse(settings.path(self.folder).exists())

    def test_a_write_merges_over_what_is_stored(self):
        settings.write(self.folder, {"final_steps": "3000"})
        settings.write(self.folder, {"seed_epochs": 40})
        self.assertEqual(settings.read(self.folder), {"target_images": 50, "seed_epochs": 40, "final_steps": 3000})
        stored = json.loads(settings.path(self.folder).read_text(encoding="utf-8"))
        self.assertEqual(stored["final_steps"], 3000)

    def test_bad_values_are_refused_and_nothing_is_written(self):
        for values in ({"target_images": 0}, {"seed_epochs": "abc"}, {"final_steps": 1.5},
                       {"final_steps": True}, {"final_steps": 10 ** 7}, {"rank": 8}):
            with self.assertRaises(settings.SettingsError, msg=values):
                settings.write(self.folder, values)
        self.assertFalse(settings.path(self.folder).exists())

    def test_a_damaged_file_or_value_falls_back_to_the_default(self):
        settings.path(self.folder).write_text("{not json", encoding="utf-8")
        self.assertEqual(settings.read(self.folder)["final_steps"], 2000)
        settings.path(self.folder).write_text(json.dumps({"final_steps": -5, "seed_epochs": 12}), encoding="utf-8")
        self.assertEqual(settings.read(self.folder), {"target_images": 50, "seed_epochs": 12, "final_steps": 2000})

    def test_a_project_object_and_its_folder_name_the_same_file(self):
        project = pj.Project(self.folder, {"id": "p"})
        self.assertEqual(settings.path(project), settings.path(self.folder))

    def test_settings_stay_out_of_the_card_so_identities_do_not_move(self):
        project = pj.Project(self.folder, {"id": "p"})
        before = project.digest()
        settings.write(project, {"final_steps": 2500})
        self.assertEqual(project.digest(), before)


class FinalStepsTodoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.automation = root / "automation"
        self.automation.mkdir()
        (self.automation / "todo.json").write_text("[]", encoding="utf-8")
        self.project = FakeProject(root / "projects" / "p", card={"id": "p", "trigger": "suamom"})
        self.manifest = write_minimal_export(root / "exports" / "abc123")

    def test_the_todo_carries_the_default_final_steps(self):
        jobs = export.plan_todos(self.project, self.manifest, automation_dir=self.automation)
        self.assertTrue(jobs)
        self.assertTrue(all(job["target_steps"] == 2000 for job in jobs))

    def test_the_todo_carries_the_projects_own_final_steps(self):
        settings.write(self.project, {"final_steps": 2600})
        export.add_todos(self.project, self.manifest, automation_dir=self.automation, log=lambda *_: None)
        written = json.loads((self.automation / "todo.json").read_text(encoding="utf-8"))
        self.assertTrue(written)
        self.assertTrue(all(job["target_steps"] == 2600 for job in written))


class SeedEpochsTests(unittest.TestCase):
    """`pipeline.seed --build --train` without --epochs trains for the project's seed_epochs."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "p"
        self.folder.mkdir()
        (self.folder / "project.json").write_text(json.dumps({"id": "p", "trigger": "t"}), encoding="utf-8")

    def epochs_used(self, argv):
        seen = {}
        report = {"policy": "seed-v2", "holdout": {"ids": [], "frozen": True}, "kept": [],
                  "counts": {key: 0 for key in ("sources", "candidates", "accepted", "accepted_from_m0", "kept",
                                                "kept_sources", "kept_candidates", "synthetic", "user", "excluded")}}
        report["counts"]["by_code"] = {}

        def fake_config(_project, _snapshot, *, epochs, **_kwargs):
            seen["epochs"] = epochs
            return {"id": "p_seed", "target_steps": epochs, "save_every_steps_approx": 1, "backend": "local",
                    "modal_gpu": "L40S"}

        with patch.object(seed, "audit", lambda _project, **_k: report), \
                patch.object(seed, "build_snapshot", lambda _project, **_k: {"id": "s", "base": "b", "reused": True,
                                                                       "inputs": []}), \
                patch.object(seed, "runner_config", fake_config), \
                patch.object(seed, "train", lambda *_a, **_k: "runs"), \
                patch("builtins.print"):
            seed.main(["--project", str(self.folder), "--build", "--train", *argv])
        return seen["epochs"]

    def test_the_default_is_thirty(self):
        self.assertEqual(self.epochs_used([]), 30)

    def test_the_project_setting_is_used(self):
        settings.write(self.folder, {"seed_epochs": 45})
        self.assertEqual(self.epochs_used([]), 45)

    def test_an_explicit_flag_still_wins(self):
        settings.write(self.folder, {"seed_epochs": 45})
        self.assertEqual(self.epochs_used(["--epochs", "12"]), 12)


if __name__ == "__main__":
    unittest.main()
