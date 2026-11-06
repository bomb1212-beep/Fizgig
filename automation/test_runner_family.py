"""Offline checks for family support: version identity, cache separation, plan shape and completion verdicts."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("lora_runner_family", Path(__file__).with_name("runner.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

# The version this configuration produced before family support existed. A Qwen 2.1 job must keep landing in the same
# run directory, or every existing run would be recaptioned, recached and retrained from scratch.
LEGACY_VERSION = "107be407f043e0a7"
LEGACY_CFG = {
    "dataset_root": "D:/ds", "target_steps": 3000, "megapixels": 0.5, "rank": 8, "learning_rate": 0.0002,
    "adaptive_lr": True, "adaptive_lr_min": 0.0002, "adaptive_lr_max": 0.0004, "caption_model": "gemma",
    "caption_base_url": "http://127.0.0.1:18080/v1", "caption_auto_start": True, "max_images_per_job": 200,
    "save_every_steps_approx": 500, "sample_prompts": ["{trigger}, portrait"], "id": "job1", "trigger": "trg",
    "seed": 42, "caption_max_output_tokens": 300,
}
LEGACY_IMAGES = [{"source": "D:/ds/a.png", "sha256": "aa" * 32, "caption": "a red image"}]


def version_of(cfg, images=None):
    return runner.digest({"settings": runner.version_settings(cfg),
                          "images": images or LEGACY_IMAGES, "instructions": "describe"})[:16]


class VersionTests(unittest.TestCase):
    def test_a_qwen_run_keeps_the_version_it_had_before_family_support(self):
        self.assertEqual(version_of(LEGACY_CFG), LEGACY_VERSION)

    def test_the_pipeline_blocks_do_not_move_a_run(self):
        noisy = dict(LEGACY_CFG, comfy={"url": "http://x"}, models={"krea2": {}},
                     pipeline={"tool2": {}})
        self.assertEqual(version_of(noisy), LEGACY_VERSION)

    def test_a_second_seed_is_a_different_run(self):
        self.assertNotEqual(version_of(dict(LEGACY_CFG, seed=43)), LEGACY_VERSION)

    def test_another_family_never_shares_a_run_directory(self):
        krea2 = dict(LEGACY_CFG, family="krea2")
        self.assertNotEqual(version_of(krea2), LEGACY_VERSION)
        self.assertNotEqual(version_of(dict(LEGACY_CFG, family="qwen_image21")), version_of(krea2))

    def test_an_unknown_family_is_refused(self):
        with self.assertRaises(ValueError):
            runner.family_of(dict(LEGACY_CFG, family="nope"))


class CaptionProfileTests(unittest.TestCase):
    def test_photo_is_the_default_profile(self):
        self.assertEqual(runner.caption_profile(LEGACY_CFG), "photo")
        self.assertIn("photograph", runner.instructions_for(LEGACY_CFG))

    def test_a_project_kind_of_2d_means_the_anime_profile(self):
        # The card says what the character is ("2d"); the profile says what the local model is told.
        self.assertEqual(runner.caption_profile(dict(LEGACY_CFG, kind="2d")), "anime")
        self.assertEqual(runner.caption_profile(dict(LEGACY_CFG, kind="photo")), "photo")

    def test_anime_selects_its_own_instructions(self):
        cfg = dict(LEGACY_CFG, kind="anime")
        self.assertEqual(runner.caption_profile(cfg), "anime")
        text = runner.instructions_for(cfg)
        self.assertIn("Danbooru", text)
        # The two profiles must not be the same file, or switching kind would change nothing.
        self.assertNotEqual(text, runner.instructions_for(LEGACY_CFG))

    def test_switching_kind_changes_the_caption_identity(self):
        photo = runner.caption_identity(LEGACY_CFG, runner.instructions_for(LEGACY_CFG))
        anime_cfg = dict(LEGACY_CFG, kind="anime")
        anime = runner.caption_identity(anime_cfg, runner.instructions_for(anime_cfg))
        self.assertNotEqual(photo, anime)

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            runner.caption_profile(dict(LEGACY_CFG, kind="sketch"))

    def test_the_default_policy_leaves_existing_behaviour_alone(self):
        self.assertEqual(runner.caption_policy(LEGACY_CFG), "missing-only")

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError):
            runner.caption_policy(dict(LEGACY_CFG, caption_policy="whatever"))


class CaptionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        (self.run / "images").mkdir()
        from PIL import Image
        self.image = self.run / "images" / "a.png"
        Image.new("RGB", (32, 32), "red").save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_only_captions_the_gap_and_keeps_a_written_caption(self):
        self.image.with_suffix(".txt").write_text("hand written", encoding="utf-8")
        client = unittest.mock.MagicMock()
        client.caption.return_value = ("machine", {})
        with patch.object(runner, "GemmaServer") as server:
            server.return_value.__enter__.return_value = client
            runner.captions(self.run, dict(LEGACY_CFG, caption_policy="missing-only"), "describe")
        self.assertEqual(self.image.with_suffix(".txt").read_text(encoding="utf-8").strip(), "hand written")
        client.caption.assert_not_called()

    def test_explicit_recaption_rewrites_every_caption(self):
        self.image.with_suffix(".txt").write_text("hand written", encoding="utf-8")
        client = unittest.mock.MagicMock()
        client.caption.return_value = ("machine", {})
        with patch.object(runner, "GemmaServer") as server:
            server.return_value.__enter__.return_value = client
            runner.captions(self.run, dict(LEGACY_CFG, caption_policy="explicit-recaption"), "describe")
        self.assertEqual(self.image.with_suffix(".txt").read_text(encoding="utf-8").strip(), "machine")

    def test_preserve_refuses_a_dataset_with_a_gap_instead_of_inventing_one(self):
        with patch.object(runner, "GemmaServer") as server:
            with self.assertRaises(ValueError) as caught:
                runner.captions(self.run, dict(LEGACY_CFG, caption_policy="preserve"), "describe")
            server.assert_not_called()
        self.assertIn("preserve", str(caught.exception))
        self.assertFalse(self.image.with_suffix(".txt").exists())

    def test_preserve_accepts_a_fully_captioned_dataset(self):
        self.image.with_suffix(".txt").write_text("hand written", encoding="utf-8")
        with patch.object(runner, "GemmaServer") as server:
            runner.captions(self.run, dict(LEGACY_CFG, caption_policy="preserve"), "describe")
            server.assert_not_called()
        self.assertEqual(self.image.with_suffix(".txt").read_text(encoding="utf-8").strip(), "hand written")


class CacheTests(unittest.TestCase):
    """A cache file is named for its architecture, and the metadata says so too."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        (self.run / "images").mkdir()
        from PIL import Image
        Image.new("RGB", (32, 32), "red").save(self.run / "images" / "a.png")
        (self.run / "images" / "a.txt").write_text("caption", encoding="utf-8")
        from fizgig.families.launch import cache_dir_for
        self.cache = Path(cache_dir_for(str(self.run / "cache"), str(self.run / "images")))
        self.cache.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_cache(self, arch, caption="caption"):
        from safetensors.torch import save_file
        import torch
        save_file({"latent_8x8": torch.zeros(1)},
                  str(self.cache / f"a_512x512_{arch}.safetensors"),
                  metadata={"architecture": arch})
        save_file({"cond__k": torch.zeros(1)},
                  str(self.cache / f"a_{arch}_te.safetensors"),
                  metadata={"architecture": arch, "caption1": caption})

    def test_a_complete_cache_for_this_family_is_accepted(self):
        self.write_cache("krea2drv")
        self.assertTrue(runner.caches_valid(self.run, "krea2"))

    def test_another_family_cache_is_never_reused(self):
        self.write_cache("qwenimage21")
        self.assertFalse(runner.caches_valid(self.run, "krea2"))
        self.assertTrue(runner.caches_valid(self.run, "qwen_image21"))

    def test_a_cache_whose_metadata_names_another_family_is_refused(self):
        from safetensors.torch import save_file
        import torch
        save_file({"latent_8x8": torch.zeros(1)}, str(self.cache / "a_512x512_krea2drv.safetensors"),
                  metadata={"architecture": "qwenimage21"})
        save_file({"cond__k": torch.zeros(1)}, str(self.cache / "a_krea2drv_te.safetensors"),
                  metadata={"architecture": "krea2drv", "caption1": "caption"})
        self.assertFalse(runner.caches_valid(self.run, "krea2"))

    def test_a_stale_caption_invalidates_the_text_cache(self):
        self.write_cache("krea2drv", caption="an older caption")
        self.assertFalse(runner.caches_valid(self.run, "krea2"))

    def test_a_missing_cache_is_not_valid(self):
        self.assertFalse(runner.caches_valid(self.run, "krea2"))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "images" / "a.png").write_bytes(b"png")
        (self.root / "images" / "a.txt").write_text("suamom, a red image", encoding="utf-8")
        from PIL import Image
        Image.new("RGB", (32, 32), "red").save(self.root / "images" / "a.png")
        self.cfg = runner.settings({"id": "familyjob", "trigger": "suamom",
                                    "dataset_root": str(self.root / "images")})

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, family):
        model = self.root / "placeholder.safetensors"
        model.touch()
        resolved = {key: str(model) for key in (
            "qwen21_dit", "qwen21_vae", "qwen21_text_encoder", "qwen21_training_adapter",
            "krea2_raw_dit", "krea2_vae", "krea2_text_encoder", "krea2_turbo_lora", "krea2_turbo_dit")}
        with patch.object(runner.model_paths, "resolve", return_value=resolved):
            return runner.launch_plan(self.root, dict(self.cfg, family=family), 4)

    def test_a_krea2_plan_trains_the_raw_base_with_its_own_family(self):
        result = self.plan("krea2")
        self.assertEqual([s.name for s in result.stages],
                         ["Cache Preparation", "Text Encoder Caching", "Training"])
        cache = result.stages[0].cmd
        self.assertEqual(cache[cache.index("--family") + 1], "krea2")
        train = result.stages[-1].cmd
        self.assertEqual(train[train.index("--family") + 1], "krea2")
        self.assertTrue(train[train.index("--dit") + 1].endswith("placeholder.safetensors"))
        self.assertNotIn("--training_adapter", train)      # Krea 2 has no adapter

    def test_a_krea2_plan_previews_at_the_family_defaults(self):
        train = self.plan("krea2").stages[-1].cmd
        self.assertEqual(train[train.index("--sample_width") + 1], "1024")
        self.assertEqual(train[train.index("--sample_height") + 1], "1024")
        self.assertEqual(train[train.index("--sample_steps") + 1], "8")
        self.assertEqual(float(train[train.index("--sample_cfg_scale") + 1]), 1.0)

    def test_a_qwen_plan_is_unchanged(self):
        train = self.plan("qwen_image21").stages[-1].cmd
        self.assertEqual(train[train.index("--family") + 1], "qwen_image21")
        self.assertEqual(train[train.index("--sample_width") + 1], "1024")
        self.assertIn("--training_adapter", train)

    def test_a_job_setting_overrides_the_family_preset(self):
        # The spec allows a todo item to replace the preset's rank and learning rate (its old 64 / 1e-4 recipe).
        model = self.root / "placeholder.safetensors"
        model.touch()
        resolved = {f.pref_key: str(model) for f in runner.model_paths.description("krea2").model_files}
        heavy = dict(self.cfg, family="krea2", rank=64, learning_rate=0.0001, adaptive_lr=False)
        with patch.object(runner.model_paths, "resolve", return_value=resolved):
            result = runner.launch_plan(self.root, heavy, 4)
        train = result.stages[-1].cmd
        self.assertEqual(train[train.index("--network_dim") + 1], "64")
        self.assertEqual(train[train.index("--network_alpha") + 1], "64")
        self.assertEqual(float(train[train.index("--learning_rate") + 1]), 0.0001)
        self.assertNotIn("--adaptive_lr", train)

    def test_the_family_preset_supplies_what_the_job_does_not_name(self):
        train = self.plan("krea2").stages[-1].cmd
        # The Ultra Fast preset's optimizer and precision reach the trainer without being in config.json.
        self.assertEqual(train[train.index("--optimizer_type") + 1], "adamw8bit")
        self.assertIn("--precision", train)


class CompletionTests(unittest.TestCase):
    """Training finished and previews missing is a finished run, not an unfinished one."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.cfg = dict(LEGACY_CFG, id="job1", sample_prompts=["a portrait"])
        self.epochs = 3

    def tearDown(self):
        self.temp.cleanup()

    def write_training(self, epoch=3, previews=0, keys=True, state=True):
        from safetensors.torch import save_file
        import torch
        output = self.run / "output"
        output.mkdir(parents=True, exist_ok=True)
        save_file({"lora_unet_blocks_0": torch.zeros(1)}, str(output / "job1.safetensors"),
                  metadata={"ss_epoch": str(epoch)} if keys else {})
        if state:
            saved = output / f"job1-{epoch:06d}-state"
            saved.mkdir(exist_ok=True)
            (saved / "training_state.json").write_text(json.dumps({"epoch": epoch}), encoding="utf-8")
            for name in ("lora.safetensors", "optimizer.pt", "rng.pt"):
                (saved / name).write_bytes(b"x")
        if previews:
            sample = output / "sample"
            sample.mkdir(exist_ok=True)
            for i in range(previews):
                (sample / f"job1_e{epoch:06d}_{i:02d}.png").write_bytes(b"png")

    def test_training_alone_counts_as_finished(self):
        self.write_training(previews=0)
        self.assertEqual(runner.output_state(self.run, self.cfg, self.epochs), (True, False))

    def test_a_run_with_previews_reports_both(self):
        self.write_training(previews=2)
        self.assertEqual(runner.output_state(self.run, self.cfg, self.epochs), (True, True))

    def test_the_missing_preview_becomes_a_warning_not_a_retrain(self):
        self.write_training(previews=0)
        warning = runner.preview_missing(self.run, self.cfg, self.epochs)
        self.assertIn("preview_missing", warning)
        self.assertIn("Not retraining", warning)

    def test_no_warning_when_the_previews_are_there(self):
        self.write_training(previews=1)
        self.assertEqual(runner.preview_missing(self.run, self.cfg, self.epochs), "")

    def test_a_missing_final_lora_is_still_unfinished(self):
        self.write_training(state=False)
        self.assertEqual(runner.output_state(self.run, self.cfg, self.epochs), (False, False))
        self.assertEqual(runner.preview_missing(self.run, self.cfg, self.epochs), "")

    def test_a_lora_from_another_epoch_is_not_this_run(self):
        self.write_training(epoch=2, previews=1)
        self.assertEqual(runner.output_state(self.run, self.cfg, self.epochs), (False, False))


if __name__ == "__main__":
    unittest.main()