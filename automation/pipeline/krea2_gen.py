"""Krea 2 text-to-image graphs for the ComfyUI API.

Three configurations, all 8-step and CFG-free, as the Turbo checkpoint and the Turbo LoRA both are:

a  Turbo DiT + the character LoRA.                        the cheapest shape; one UNETLoader, one LoRA
b  RAW DiT + the Turbo LoRA + the character LoRA.         the RAW base trains, so this is the shape a Fizgig-trained
                                                          LoRA was actually fitted to
c  Turbo DiT + the Krea 2 style-reference LoRA + the character LoRA, with 1-3 reference images encoded by
   TextEncodeQwenImageEditPlus.                           the Image Style Reference blueprint's graph; whether a
                                                          character LoRA survives it is unproven, so the experiment
                                                          tries it and records the outcome

Turbo DiT stacked with the Turbo LoRA is refused: the LoRA was distilled from the RAW base, and the Turbo checkpoint
already contains it. Sampling settings come from the family description (`src/fizgig/families/krea2.py`): 8 steps,
CFG 1.0, euler/simple, with the model's own `shift` 1.15 (ComfyUI's Krea2 class sets it, so no ModelSamplingFlux node
is needed on the Turbo shapes).

The builders are pure: names in, a JSON-serialisable graph out. That keeps every rule unit-testable without a server.
"""
from __future__ import annotations

GRAPH_VERSION = {
    "a": "krea2_t2i_turbo@1",
    "b": "krea2_t2i_raw_turbo@1",
    "c": "krea2_ref_turbo@1",
}

VARIANTS = ("a", "b", "c")

LABELS = {
    "a": "Krea2 (a) Turbo DiT + character LoRA",
    "b": "Krea2 (b) RAW DiT + Turbo LoRA + character LoRA",
    "c": "Krea2 (c) Turbo DiT + style reference + character LoRA",
}

# Which model files each configuration loads, by the family's Preferences keys.
NEEDS = {
    "a": ("krea2_turbo_dit", "krea2_text_encoder", "krea2_vae"),
    "b": ("krea2_raw_dit", "krea2_turbo_lora", "krea2_text_encoder", "krea2_vae"),
    "c": ("krea2_turbo_dit", "krea2_text_encoder", "krea2_vae"),
}

SAMPLING = {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"}

# ModelSamplingFlux carries the Turbo checkpoint's shift in the style-reference blueprint; the plain text-to-image
# blueprint leaves it to the model, so only (c) sets it, to stay faithful to the graph that was shipped.
STYLE_REFERENCE_SHIFT = {"max_shift": 1.15, "base_shift": 0.5}
REFERENCE_LATENTS_METHOD = "index_timestep_zero"

STYLE_LORA = "krea2_style_reference.safetensors"


class GraphError(ValueError):
    """A configuration that cannot be built, or not as asked."""


def check(variant):
    if variant not in GRAPH_VERSION:
        raise GraphError(f"unknown Krea2 variant {variant!r} (known: {', '.join(VARIANTS)})")
    return variant


def build(variant, *, unet, clip, vae, prompt, negative="", width=1024, height=1024, seed=0,
          steps=None, cfg=None, sampler=None, scheduler=None, character_lora=None, character_strength=1.0,
          speed_lora=None, style_lora=None, reference_images=(), batch_size=1,
          filename_prefix="krea2_gen/img"):
    """The API-format graph for one image.

    `unet`, `clip`, `vae`, the LoRAs and the reference images are the names ComfyUI lists (see Comfy.name_for);
    `reference_images` are names LoadImage can open, i.e. uploaded first.
    """
    check(variant)
    if not prompt or not str(prompt).strip():
        raise GraphError("a Krea2 prompt is required")
    if variant != "c" and reference_images:
        raise GraphError(f"variant {variant!r} takes no reference images; use variant 'c'")
    if variant == "c" and not reference_images:
        raise GraphError("variant 'c' conditions on reference images; pass at least one")
    if variant == "c" and len(reference_images) > 3:
        raise GraphError("TextEncodeQwenImageEditPlus takes at most three reference images")
    if variant == "a" and speed_lora:
        # The Turbo checkpoint already contains the distillation the speed LoRA adds, so stacking them is not a
        # configuration anyone measured; refuse instead of silently producing something worse than either alone.
        raise GraphError("the Turbo DiT and the Turbo LoRA cannot be stacked; use variant 'b' for the RAW base")
    if variant == "b" and not speed_lora:
        raise GraphError("variant 'b' trains on the RAW base, which needs the Turbo LoRA to sample in 8 steps")

    sampling = {**SAMPLING, "steps": steps or SAMPLING["steps"], "cfg": SAMPLING["cfg"] if cfg is None else cfg,
                "sampler": sampler or SAMPLING["sampler"], "scheduler": scheduler or SAMPLING["scheduler"]}
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "20": {"class_type": "EmptyLatentImage", "inputs": {"width": int(width), "height": int(height),
                                                            "batch_size": int(batch_size)}},
        "30": {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["3", 0]}},
        "50": {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": filename_prefix}},
    }
    positive = _positive_conditioning(variant, graph, prompt, reference_images)
    negative = _negative_conditioning(graph, positive, negative)
    model = _model_chain(variant, graph, character_lora, character_strength, speed_lora, style_lora,
                         width, height)
    if variant == "c":
        graph["40"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["41", 0], "guider": ["43", 0], "sampler": ["44", 0], "sigmas": ["45", 0],
                       "latent_image": ["20", 0]},
        }
        graph["41"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
        graph["43"] = {"class_type": "CFGGuider",
                       "inputs": {"model": model, "positive": [positive, 0], "negative": [negative, 0],
                                  "cfg": float(sampling["cfg"])}}
        graph["44"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampling["sampler"]}}
        graph["45"] = {"class_type": "BasicScheduler",
                       "inputs": {"model": model, "scheduler": sampling["scheduler"],
                                  "steps": int(sampling["steps"]), "denoise": 1.0}}
    else:
        graph["40"] = {
            "class_type": "KSampler",
            "inputs": {"model": model, "seed": int(seed), "steps": int(sampling["steps"]),
                       "cfg": float(sampling["cfg"]), "sampler_name": sampling["sampler"],
                       "scheduler": sampling["scheduler"], "positive": [positive, 0], "negative": [negative, 0],
                       "latent_image": ["20", 0], "denoise": 1.0},
        }
    return graph


def _positive_conditioning(variant, graph, prompt, reference_images):
    if variant != "c":
        graph["10"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": str(prompt)}}
        return "10"
    graph["10"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": str(prompt)},
    }
    for i, name in enumerate(reference_images, 1):
        graph[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["10"]["inputs"][f"image{i}"] = [f"ref{i}", 0]
    graph["11"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                   "inputs": {"conditioning": ["10", 0], "reference_latents_method": REFERENCE_LATENTS_METHOD}}
    return "11"


def _negative_conditioning(graph, positive, negative):
    """Both shapes are CFG-free (cfg 1.0), so the negative branch is a zeroed positive by default.

    A caller's own negative text is encoded beside it and zeroed the same way, which is what the shipped blueprints
    do: with cfg 1.0 the branch is skipped entirely, and raising cfg later uses it.
    """
    source = positive
    if negative:
        graph["13"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": str(negative)}}
        source = "13"
    graph["12"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [source, 0]}}
    return "12"


def _model_chain(variant, graph, character_lora, character_strength, speed_lora, style_lora, width, height):
    """UNETLoader → (style LoRA) → (Turbo LoRA) → (character LoRA) → (ModelSamplingFlux), returning the last node id."""
    model = ["1", 0]
    n = 60
    for name, strength in ((style_lora if variant == "c" else None, 1.0),
                           (speed_lora if variant == "b" else None, 1.0),
                           (character_lora, float(character_strength))):
        if not name:
            continue
        node = str(n)
        graph[node] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [node, 0]
        n += 1
    if variant == "c":
        graph["70"] = {"class_type": "ModelSamplingFlux",
                       "inputs": {"model": model, "max_shift": STYLE_REFERENCE_SHIFT["max_shift"],
                                  "base_shift": STYLE_REFERENCE_SHIFT["base_shift"],
                                  "width": int(width), "height": int(height)}}
        return ["70", 0]
    return model


def graph_for_doctor(variant, names):
    """A one-sentence description of what this variant will load, for doctor and the experiment's log."""
    files = ", ".join(f"{k}={names.get(k) or '?'}" for k in NEEDS[check(variant)])
    return f"{LABELS[variant]} [{GRAPH_VERSION[variant]}] {files}"
