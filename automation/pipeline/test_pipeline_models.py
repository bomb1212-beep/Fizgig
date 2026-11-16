"""Offline checks for model path resolution (config.json first, prefs.json second)."""
import sys
import unittest
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import models  # noqa: E402


class ModelPathTests(unittest.TestCase):
    def test_config_wins_and_prefs_fills_the_rest(self):
        cfg = {"models": {"krea2": {"krea2_raw_dit": "G:/comfy/raw.safetensors"}}}
        prefs = {"krea2_raw_dit": "G:/old/raw.safetensors", "krea2_vae": "G:/comfy/vae.safetensors"}
        resolved = models.resolve("krea2", cfg, base=prefs)
        self.assertEqual(Path(resolved["krea2_raw_dit"]), Path("G:/comfy/raw.safetensors"))
        self.assertEqual(Path(resolved["krea2_vae"]), Path("G:/comfy/vae.safetensors"))

    def test_relative_paths_hang_off_the_fizgig_root(self):
        cfg = {"models": {"qwen_image21": {"qwen21_vae": "models/Qwen-Image-2.1/vae/x.safetensors"}}}
        resolved = models.resolve("qwen_image21", cfg, base={})
        self.assertEqual(Path(resolved["qwen21_vae"]), ROOT / "models/Qwen-Image-2.1/vae/x.safetensors")
        self.assertTrue(Path(resolved["qwen21_vae"]).is_absolute())

    def test_every_declared_file_gets_a_row(self):
        from fizgig.families.registry import get
        rows = models.rows("krea2", {"models": {}}, base={})
        self.assertEqual([r["pref_key"] for r in rows], [f.pref_key for f in get("krea2").model_files])
        self.assertTrue(all(r["exists"] is False for r in rows))

    def test_row_reports_where_the_path_came_from(self):
        cfg = {"models": {"krea2": {"krea2_raw_dit": "G:/comfy/raw.safetensors"}}}
        rows = {r["pref_key"]: r for r in models.rows("krea2", cfg, base={"krea2_raw_dit": "G:/old/raw"})}
        self.assertEqual(rows["krea2_raw_dit"]["source"], "config.json")
        self.assertEqual(rows["krea2_vae"]["source"], "prefs.json")

    def test_prefs_follow_fizgig_into_user_data(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            new = root / "user_data" / "prefs.json"
            with mock.patch.object(models, "ROOT", root), mock.patch("fizgig.user_data.PREFS_FILE", str(new)):
                self.assertEqual(models.prefs_path(), new)            # neither yet: where Fizgig 8 will write it
                (root / "prefs.json").write_text('{"a": "old"}', encoding="utf-8")
                self.assertEqual(models.prefs_path(), root / "prefs.json")   # not moved yet: the old one
                self.assertEqual(models.prefs(), {"a": "old"})
                new.parent.mkdir()
                new.write_text('{"a": "new"}', encoding="utf-8")
                self.assertEqual(models.prefs(), {"a": "new"})        # moved (or both): user_data wins
                (root / "models").mkdir()
                self.assertEqual(models.models_dir(), root / "models")
                (root / "user_data" / "models").mkdir()
                self.assertEqual(models.models_dir(), root / "user_data" / "models")

    def test_problems_name_the_family_key_to_fill(self):
        problems = models.problems("krea2", {"models": {}}, base={})
        self.assertTrue(problems)
        self.assertIn("models.krea2.krea2_raw_dit", " ".join(problems))

    def test_optional_file_only_checked_when_asked_for(self):
        base = {}
        from fizgig.families.registry import get
        required = {f.pref_key for f in get("krea2").model_files if f.required}
        for row in models.rows("krea2", {"models": {}}, base=base):
            if row["required"]:
                continue
            self.assertNotIn(f"{row['label']} path is empty", " ".join(models.problems("krea2", {}, base)))
        blocked = " ".join(models.problems("krea2", {}, base, require_optional=("krea2_turbo_lora",)))
        self.assertIn("krea2_turbo_lora", blocked)
        self.assertEqual(len(required), sum(1 for r in models.rows("krea2", {}, base) if r["required"]))

    def test_unknown_family_is_refused(self):
        with self.assertRaises(ValueError):
            models.resolve("not_a_family", {}, base={})

    def test_comfy_settings_fill_defaults(self):
        block = models.comfy_settings({"comfy": {"root": "G:/Lakis/ComfyUI"}})
        self.assertEqual(block["url"], "http://127.0.0.1:8190")
        self.assertEqual(Path(block["root"]), Path("G:/Lakis/ComfyUI"))
        self.assertIn("8190", block["args"])

    def test_model_dir_maps_kinds_to_folders(self):
        dirs = models.model_dir("G:/Lakis/ComfyUI", "lora")
        self.assertEqual([Path(d).name for d in dirs], ["loras"])
        self.assertEqual([Path(d).name for d in models.model_dir("G:/Lakis/ComfyUI", "clip")],
                         ["text_encoders", "clip"])


if __name__ == "__main__":
    unittest.main()
