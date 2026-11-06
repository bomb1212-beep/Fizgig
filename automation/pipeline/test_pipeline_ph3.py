"""P-H3 graph tests: the shape of the graph, not its pixels.

Pixels need ComfyUI and 25 GB of weights, so these tests pin the things that are cheap to get wrong and expensive to
notice: a sigma schedule of the wrong length, a reference handed over as a local path, a negative prompt that the
CFG-free sampler cannot honour.
"""
import unittest
from pathlib import Path

from pipeline import ph3_gen


class SigmaTests(unittest.TestCase):
    def test_the_shipped_length_is_returned(self):
        self.assertEqual(ph3_gen.sigmas_for(8), ph3_gen.DEFAULT_SIGMAS)

    def test_another_length_is_refused_rather_than_interpolated(self):
        # Interpolating would silently draw a different curve and still look like a plausible picture.
        with self.assertRaises(ph3_gen.GraphError) as caught:
            ph3_gen.sigmas_for(20)
        self.assertIn("8-step", str(caught.exception))

    def test_the_schedule_ends_at_zero(self):
        values = [float(v) for v in ph3_gen.DEFAULT_SIGMAS.split(",")]
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[-1], 0.0)

    def test_the_schedule_has_one_more_sigma_than_steps(self):
        values = [float(v) for v in ph3_gen.DEFAULT_SIGMAS.split(",")]
        self.assertEqual(len(values), ph3_gen.DEFAULT_STEPS + 1)


class ReferenceTests(unittest.TestCase):
    def test_a_local_path_is_refused_with_the_reason(self):
        # ComfyUI rejects a path here too, but several steps later and in terms that do not say what to do.
        with self.assertRaises(ph3_gen.GraphError) as caught:
            ph3_gen.check_references([__file__])
        self.assertIn("Comfy.upload", str(caught.exception))

    def test_an_uploaded_name_is_accepted(self):
        names = ["dataset_pipeline/a.png", "dataset_pipeline/b.png"]
        self.assertEqual(ph3_gen.check_references(names), names)

    def test_more_than_the_autogrow_limit_is_refused(self):
        with self.assertRaises(ph3_gen.GraphError) as caught:
            ph3_gen.check_references([f"n{i}.png" for i in range(ph3_gen.MAX_REFERENCES + 1)])
        self.assertIn(str(ph3_gen.MAX_REFERENCES), str(caught.exception))

    def test_an_empty_name_is_refused(self):
        with self.assertRaises(ph3_gen.GraphError):
            ph3_gen.check_references(["  "])


class GraphShapeTests(unittest.TestCase):
    def test_the_text_to_image_graph_carries_the_upsampler(self):
        graph = ph3_gen.build(prompt="a character")
        kinds = {node["class_type"] for node in graph.values()}
        self.assertIn("TextGenerate", kinds)
        self.assertIn("PH3ImageConditioning", kinds)
        self.assertNotIn("LoadImage", kinds)

    def test_the_prompt_reaches_the_conditioning_node(self):
        # The upsample path is a StringFormat -> TextGenerate -> PH3ImageConditioning chain; a wiring bug that fed the
        # raw brief straight in would still produce a picture, so the chain is what gets asserted.
        graph = ph3_gen.build(prompt="a character")
        by_kind = {}
        for node_id, node in graph.items():
            by_kind.setdefault(node["class_type"], []).append((node_id, node))
        textgen_id, textgen = by_kind["TextGenerate"][0]
        cond_id, cond = by_kind["PH3ImageConditioning"][0]
        self.assertEqual(cond["inputs"]["prompt"], [textgen_id, 0])
        self.assertIn("clip", textgen["inputs"])
        self.assertIn("prompt", textgen["inputs"])

    def test_the_latent_comes_from_the_conditioning_node(self):
        graph = ph3_gen.build(prompt="a character")
        by_kind = {node["class_type"]: (nid, node) for nid, node in graph.items()}
        cond_id, _cond = by_kind["PH3ImageConditioning"]
        _sid, sampler = by_kind["SamplerCustomAdvanced"]
        self.assertEqual(sampler["inputs"]["latent_image"], [cond_id, 1])

    def test_a_requested_size_reaches_the_conditioning_node(self):
        graph = ph3_gen.build(prompt="a character", width=832, height=1216)
        cond = next(node for node in graph.values() if node["class_type"] == "PH3ImageConditioning")
        self.assertEqual(cond["inputs"]["width"], 832)
        self.assertEqual(cond["inputs"]["height"], 1216)

    def test_the_seed_reaches_the_noise_node(self):
        graph = ph3_gen.build(prompt="a character", seed=4242)
        noise = next(node for node in graph.values() if node["class_type"] == "RandomNoise")
        self.assertEqual(noise["inputs"]["noise_seed"], 4242)

    def test_a_picture_to_edit_sets_the_size_instead_of_the_caller(self):
        # With a reference, width/height 0 means "follow image_1"; passing a number would crop the edit.
        graph = ph3_gen.build(prompt="a character", references=["uploaded.png"], width=1024, height=1536)
        cond = next(node for node in graph.values() if node["class_type"] == "PH3ImageConditioning")
        self.assertEqual(cond["inputs"]["width"], 0)
        self.assertEqual(cond["inputs"]["height"], 0)
        self.assertIn("images.image_1", cond["inputs"])

    def test_references_become_conditioning_inputs_in_order(self):
        graph = ph3_gen.build(prompt="a character", references=["a.png", "b.png"])
        cond = next(node for node in graph.values() if node["class_type"] == "PH3ImageConditioning")
        self.assertIn("images.image_1", cond["inputs"])
        self.assertIn("images.image_2", cond["inputs"])
        self.assertNotIn("images.image_3", cond["inputs"])
        # a and b must not be swapped: image_1 is the picture being edited
        self.assertNotEqual(cond["inputs"]["images.image_1"], cond["inputs"]["images.image_2"])

    def test_no_negative_prompt_is_offered(self):
        # BasicGuider has no negative input; accepting one would be a control that does nothing.
        with self.assertRaises(TypeError):
            ph3_gen.build(prompt="a character", negative="blurry")

    def test_a_precomputed_prompt_skips_the_upsampler(self):
        graph = ph3_gen.build(prompt="a character", upsampled_prompt="already rewritten")
        kinds = {node["class_type"] for node in graph.values()}
        self.assertNotIn("TextGenerate", kinds)
        cond = next(node for node in graph.values() if node["class_type"] == "PH3ImageConditioning")
        self.assertEqual(cond["inputs"]["prompt"], "already rewritten")

    def test_a_lora_sits_between_the_unet_and_the_shift(self):
        graph = ph3_gen.build(prompt="a character", lora="my_character.safetensors", lora_strength=0.8)
        by_kind = {node["class_type"]: (nid, node) for nid, node in graph.items()}
        lora_id, lora = by_kind["LoraLoaderModelOnly"]
        unet_id, _unet = by_kind["UNETLoader"]
        _shift_id, shift = by_kind["MiniMaxH3SigmaShift"]
        self.assertEqual(lora["inputs"]["model"], [unet_id, 0])
        self.assertEqual(shift["inputs"]["model"], [lora_id, 0])
        self.assertEqual(lora["inputs"]["strength_model"], 0.8)

    def test_every_link_points_at_a_node_that_exists(self):
        graph = ph3_gen.build(prompt="a character", references=["a.png", "b.png"])
        for node_id, node in graph.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], graph, f"{node_id}.{name} points at missing node {value[0]}")

    def test_node_ids_are_dense_and_start_at_one(self):
        graph = ph3_gen.build(prompt="a character")
        self.assertEqual(sorted(int(k) for k in graph), list(range(1, len(graph) + 1)))

    def test_the_filename_prefix_is_carried(self):
        graph = ph3_gen.build(prompt="a character", filename_prefix="jobs/one")
        save = next(node for node in graph.values() if node["class_type"] == "SaveImage")
        self.assertEqual(save["inputs"]["filename_prefix"], "jobs/one")


class NameResolutionTests(unittest.TestCase):
    def test_resolve_names_asks_comfy_for_each_file(self):
        class _Client:
            def __init__(self):
                self.asked = []

            def name_for(self, kind, name):
                self.asked.append((kind, str(name)))
                return f"{kind}::{name}"

        client = _Client()
        names = ph3_gen.resolve_names(client)
        self.assertEqual(set(names), {"ph3_unet", "ph3_text_encoder", "ph3_upsampler", "ph3_vae"})
        self.assertEqual(len(client.asked), 4)

    def test_a_configured_name_wins_over_the_constant(self):
        class _Client:
            def name_for(self, kind, name):
                return f"{kind}::{name}"

        names = ph3_gen.resolve_names(_Client(), {"ph3_unet": "custom.safetensors"})
        self.assertEqual(names["ph3_unet"], "unet::custom.safetensors")


if __name__ == "__main__":
    unittest.main()
