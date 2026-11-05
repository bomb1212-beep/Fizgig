"""Offline checks for the Krea 2 graph builders: what each configuration loads, and what it refuses."""
import sys
import unittest
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import krea2_gen as gen  # noqa: E402

BASE = dict(unet="turbo.safetensors", clip="te.safetensors", vae="vae.safetensors", prompt="suamom, smile")


def classes(graph):
    return {node["class_type"] for node in graph.values()}


def by_class(graph, name):
    return [node for node in graph.values() if node["class_type"] == name]


class VariantATests(unittest.TestCase):
    def setUp(self):
        self.graph = gen.build("a", **BASE, character_lora="char.safetensors", seed=7, width=1024, height=1536)

    def test_it_is_the_turbo_checkpoint_with_one_lora(self):
        self.assertEqual(by_class(self.graph, "UNETLoader")[0]["inputs"]["unet_name"], "turbo.safetensors")
        loras = by_class(self.graph, "LoraLoaderModelOnly")
        self.assertEqual([l["inputs"]["lora_name"] for l in loras], ["char.safetensors"])

    def test_it_samples_in_eight_cfg_free_steps(self):
        sampler = by_class(self.graph, "KSampler")[0]["inputs"]
        self.assertEqual((sampler["steps"], sampler["cfg"]), (8, 1.0))
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"]), ("euler", "simple"))
        self.assertEqual(sampler["seed"], 7)

    def test_the_negative_branch_is_the_zeroed_positive(self):
        positive = by_class(self.graph, "CLIPTextEncode")[0]
        zero = by_class(self.graph, "ConditioningZeroOut")[0]
        self.assertEqual(zero["inputs"]["conditioning"][0],
                         next(k for k, v in self.graph.items() if v is positive))

    def test_the_text_encoder_is_loaded_as_krea2(self):
        self.assertEqual(by_class(self.graph, "CLIPLoader")[0]["inputs"]["type"], "krea2")

    def test_it_refuses_the_turbo_lora_on_the_turbo_checkpoint(self):
        with self.assertRaises(gen.GraphError):
            gen.build("a", **BASE, speed_lora="krea2_turbo_lora.safetensors")

    def test_it_refuses_reference_images(self):
        with self.assertRaises(gen.GraphError):
            gen.build("a", **BASE, reference_images=["ref.png"])


class VariantBTests(unittest.TestCase):
    def setUp(self):
        self.graph = gen.build("b", **BASE, character_lora="char.safetensors",
                               speed_lora="krea2_turbo_lora.safetensors")

    def test_the_turbo_lora_comes_before_the_character_lora(self):
        chain = [l["inputs"]["lora_name"] for l in by_class(self.graph, "LoraLoaderModelOnly")]
        self.assertEqual(chain, ["krea2_turbo_lora.safetensors", "char.safetensors"])
        # each LoRA feeds the next: the graph must be a chain, not two LoRAs on the base model
        first = by_class(self.graph, "LoraLoaderModelOnly")[0]
        second = by_class(self.graph, "LoraLoaderModelOnly")[1]
        self.assertEqual(second["inputs"]["model"][0],
                         next(k for k, v in self.graph.items() if v is first))

    def test_it_needs_the_speed_lora(self):
        with self.assertRaises(gen.GraphError):
            gen.build("b", **BASE)

    def test_the_character_strength_is_carried(self):
        graph = gen.build("b", **BASE, character_lora="char.safetensors",
                          speed_lora="speed.safetensors", character_strength=0.75)
        strengths = [l["inputs"]["strength_model"] for l in by_class(graph, "LoraLoaderModelOnly")]
        self.assertEqual(strengths, [1.0, 0.75])


class VariantCTests(unittest.TestCase):
    def test_it_encodes_the_references_and_pins_the_shift(self):
        graph = gen.build("c", **BASE, character_lora="char.safetensors",
                          reference_images=["ref1.png", "ref2.png"])
        encode = by_class(graph, "TextEncodeQwenImageEditPlus")[0]["inputs"]
        loaded = [k for k in graph if graph[k]["class_type"] == "LoadImage"]
        self.assertEqual(encode["image1"], [loaded[0], 0])
        self.assertEqual(encode["image2"], [loaded[1], 0])
        self.assertEqual(graph[loaded[0]]["inputs"]["image"], "ref1.png")
        method = by_class(graph, "FluxKontextMultiReferenceLatentMethod")[0]["inputs"]
        self.assertEqual(method["reference_latents_method"], "index_timestep_zero")
        shift = by_class(graph, "ModelSamplingFlux")[0]["inputs"]
        self.assertEqual(shift["max_shift"], 1.15)

    def test_it_extends_the_blueprint_with_the_advanced_sampler(self):
        graph = gen.build("c", **BASE, reference_images=["ref1.png"])
        self.assertTrue({"SamplerCustomAdvanced", "CFGGuider", "BasicScheduler", "RandomNoise", "KSamplerSelect"}
                        <= classes(graph))
        self.assertEqual(by_class(graph, "CFGGuider")[0]["inputs"]["cfg"], 1.0)

    def test_it_needs_a_reference_image(self):
        with self.assertRaises(gen.GraphError):
            gen.build("c", **BASE)

    def test_it_takes_at_most_three_references(self):
        with self.assertRaises(gen.GraphError):
            gen.build("c", **BASE, reference_images=["a.png", "b.png", "c.png", "d.png"])

    def test_a_style_lora_goes_in_front_of_the_character_lora(self):
        graph = gen.build("c", **BASE, character_lora="char.safetensors", reference_images=["ref.png"],
                          style_lora="krea2_style_reference.safetensors")
        chain = [l["inputs"]["lora_name"] for l in by_class(graph, "LoraLoaderModelOnly")]
        self.assertEqual(chain, ["krea2_style_reference.safetensors", "char.safetensors"])


class SharedTests(unittest.TestCase):
    def test_the_graph_is_json_serialisable(self):
        import json
        graph = gen.build("a", **BASE)
        self.assertIsInstance(json.loads(json.dumps(graph)), dict)

    def test_every_variant_declares_which_files_it_needs(self):
        for variant in gen.VARIANTS:
            self.assertIn("krea2_text_encoder", gen.NEEDS[variant])
            self.assertIn("krea2_vae", gen.NEEDS[variant])
        self.assertIn("krea2_turbo_lora", gen.NEEDS["b"])
        self.assertNotIn("krea2_turbo_lora", gen.NEEDS["a"])

    def test_unknown_variant_is_refused(self):
        with self.assertRaises(gen.GraphError):
            gen.build("z", **BASE)

    def test_an_empty_prompt_is_refused(self):
        with self.assertRaises(gen.GraphError):
            gen.build("a", **{**BASE, "prompt": "  "})

    def test_the_graph_version_names_the_shape(self):
        self.assertEqual(gen.GRAPH_VERSION["a"], "krea2_t2i_turbo@1")
        self.assertEqual(gen.GRAPH_VERSION["b"], "krea2_t2i_raw_turbo@1")

    def test_doctor_describes_a_variant(self):
        line = gen.graph_for_doctor("b", {"krea2_raw_dit": "raw.safetensors"})
        self.assertIn("raw.safetensors", line)
        self.assertIn("krea2_t2i_raw_turbo", line)


if __name__ == "__main__":
    unittest.main()
