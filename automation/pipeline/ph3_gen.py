"""P-H3 image generation and editing, as a ComfyUI graph built on this machine's measured behaviour.

P-H3 is a MiniMax-H3-derived image model. It is *not* a Krea2 drop-in: it samples with an explicit sigma schedule
rather than a step count, needs no CFG, upsamples the prompt through a second language model, and takes up to nine
reference pictures for edits. So the graph is written out here instead of being parameterised into krea2_gen.

**Measured on this box** (w4a8 weights, 1024x1536, one 24 GB-class card): 14.1 s for a text-to-image picture once the
weights were resident. A cold first call is slower because 25 GB has to come off disk.

**Weights.** The published workflows name bf16 files; this machine holds the w4a8 set, which is what fits in the
31.8 GB of VRAM available. The choice is a module constant rather than a parameter because picking the wrong one is
not a quality trade-off, it is an out-of-memory crash.

**The prompt upsampler is not optional.** PH3ImageConditioning reads a brief and the shipped graph always feeds it
TextGenerate's output. `upsampled_prompt=` exists so a caller can pass a brief it already upsampled; bypassing the
upsampler entirely is not offered, because it would change the wording the model is conditioned on.
"""
from pathlib import Path

# The w4a8 set: ~25 GB resident, which leaves headroom. The int8 set is ~36 GB and does not fit.
UNET = "p-h3-image_w4a8.safetensors"
TEXT_ENCODER = "p-h3-image_text_encoder_w4a8.safetensors"
UPSAMPLER = "p-h3-image_prompt_upsampler_w4a8.safetensors"
VAE = "p-h3-image_vae_bf16.safetensors"

# The schedule P-H3 ships with: 8 sampling steps expressed as sigmas. Handing the sampler a step count would silently
# change the curve, so a different length is refused rather than interpolated.
DEFAULT_SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"
DEFAULT_STEPS = 8

# The upsampler was trained on this chat framing; the picture slots are part of the format, not decoration.
UPSCALER_FORMAT = (
    "<|im_start|>user\n<Picture 1>: <|vision_start|><|image_pad|><|vision_end|>"
    "<Picture 2>: <|vision_start|><|image_pad|><|vision_end|>Edit <Picture 1>. {a}<|im_end|>\n"
    "<|im_start|>assistant\n thinking\n"
)
# Without reference pictures the same framing still applies; only the second picture slot goes away.
UPSCALER_FORMAT_NO_IMAGES = (
    "<|im_start|>user\n<Picture 1>: <|vision_start|><|image_pad|><|vision_end|>{a}<|im_end|>\n"
    "<|im_start|>assistant\n</think>\n"
)

# Reference pictures are scaled to this many megapixels before they reach the model. This is the shipped workflow's
# value; a different target changes how much of the reference survives the trip through the VAE.
REFERENCE_MEGAPIXELS = 4

# PH3ImageConditioning's autogrow input declares image_1..image_9 and nothing more, so a longer list is refused rather
# than silently truncated.
MAX_REFERENCES = 9

SAMPLER = "euler"
SHIFT_VIDEO = 1
SHIFT_AUDIO = 1

_CONFIG_KEYS = ("ph3_unet", "ph3_text_encoder", "ph3_upsampler", "ph3_vae")


class GraphError(ValueError):
    """A graph that cannot be built, or not as asked."""


def sigmas_for(steps):
    """The shipped 8-step schedule, refusing any other length rather than inventing one."""
    if int(steps) != DEFAULT_STEPS:
        raise GraphError(f"P-H3 ships an 8-step sigma schedule; asked for {steps} steps. "
                         f"Pass sigma_values explicitly if you really mean a different schedule")
    return DEFAULT_SIGMAS


def check_references(references):
    """Reference names, validated before a graph is built so a mistake fails here and not inside ComfyUI.

    These are names LoadImage can open, i.e. what `Comfy.upload` returned - not local paths. Handing a local path to
    LoadImage is refused by ComfyUI itself, so accepting one here would only move the error somewhere less obvious.
    """
    references = [str(r) for r in references]
    if len(references) > MAX_REFERENCES:
        raise GraphError(f"P-H3 takes at most {MAX_REFERENCES} reference pictures; got {len(references)}")
    for ref in references:
        if Path(ref).is_file():
            raise GraphError(f"{ref!r} is a local path; LoadImage needs a name from Comfy.upload() instead. "
                             f"Upload it first, then pass the returned name")
        if not ref.strip():
            raise GraphError("a reference name is empty")
    return references


def upload_references(client, paths, subfolder="dataset_pipeline"):
    """Upload local pictures and return the names LoadImage accepts, in the order given."""
    return [client.upload(path, subfolder=subfolder) for path in paths]


def build(*, prompt, width=1024, height=1536, seed=0, steps=None, sigma_values=None,
          references=(), instruction=None, upsampled_prompt=None,
          lora=None, lora_strength=1.0, filename_prefix="pipeline_ph3/img",
          names=None):
    """The API-format graph for one P-H3 picture.

    With no `references` this is text-to-image. With references it is an edit: the first is the picture to change and
    the rest are context, which is how the shipped edit workflow is wired.

    There is no `negative` argument on purpose: P-H3 runs CFG-free (BasicGuider has no negative input), so accepting
    one would mean pretending to a control this graph does not have.

    `names` maps the four config keys to concrete ComfyUI filenames; without it the module constants are used.
    """
    references = check_references(references)
    names = dict(names or {})
    unet = names.get("ph3_unet") or UNET
    text_encoder = names.get("ph3_text_encoder") or TEXT_ENCODER
    upsampler = names.get("ph3_upsampler") or UPSAMPLER
    vae = names.get("ph3_vae") or VAE

    # Every node is written by name and numbered once at the end, so inserting a node (a LoRA, a reference) cannot
    # leave a stale numeric id behind.
    nodes = {}

    nodes["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
    model = ["unet", 0]
    if lora:
        nodes["lora"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": model, "lora_name": str(lora),
                                    "strength_model": float(lora_strength)}}
        model = ["lora", 0]
    nodes["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                      "inputs": {"model": model, "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO}}
    nodes["text_encoder"] = {"class_type": "CLIPLoader",
                             "inputs": {"clip_name": text_encoder, "type": "minimax", "device": "default"}}
    nodes["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}

    # --- the brief the diffusion model is conditioned on ---------------------------------------------------------
    if upsampled_prompt is not None:
        prompt_link = upsampled_prompt
    else:
        nodes["upsampler"] = {"class_type": "CLIPLoader",
                              "inputs": {"clip_name": upsampler, "type": "minimax", "device": "default"}}
        nodes["brief"] = {"class_type": "PrimitiveStringMultiline",
                          "inputs": {"value": instruction or prompt}}
        nodes["format"] = {"class_type": "StringFormat",
                           "inputs": {"values.a": ["brief", 0],
                                      "f_string": UPSCALER_FORMAT if references else UPSCALER_FORMAT_NO_IMAGES}}
        textgen = {"class_type": "TextGenerate",
                   "inputs": {"clip": ["upsampler", 0], "prompt": ["format", 0],
                              "max_length": 3072, "sampling_mode": "off",
                              "thinking": False, "use_default_template": True, "mtp": "auto"}}
        if references:
            # Let the upsampler look at the pictures: it is what makes the rewritten brief about *these* references.
            loader_ids = []
            for index in range(len(references)):
                key = f"load{index}"
                nodes[key] = {"class_type": "LoadImage",
                              "inputs": {"image": references[index], "upload": "image"}}
                loader_ids.append(key)
            nodes["refs_for_text"] = {
                "class_type": "BatchImagesNode",
                "inputs": {f"images.image{i}": [key, 0] for i, key in enumerate(loader_ids)},
            }
            textgen["inputs"]["image"] = ["refs_for_text", 0]
        nodes["textgen"] = textgen
        prompt_link = ["textgen", 0]

    # --- conditioning ---------------------------------------------------------------------------------------------
    condition = {"class_type": "PH3ImageConditioning",
                 "inputs": {"clip": ["text_encoder", 0], "vae": ["vae", 0], "prompt": prompt_link,
                            "width": 0 if references else int(width),
                            "height": 0 if references else int(height)}}
    if references:
        # image_1 sets the output size (width/height 0 means "follow it"), so it is scaled but never padded.
        for index, ref in enumerate(references):
            key = f"load{index}"
            if key not in nodes:
                nodes[key] = {"class_type": "LoadImage", "inputs": {"image": ref, "upload": "image"}}
            nodes[f"scale{index}"] = {"class_type": "ImageScaleToTotalPixels",
                                      "inputs": {"image": [key, 0], "upscale_method": "lanczos",
                                                 "megapixels": REFERENCE_MEGAPIXELS, "resolution_steps": 32}}
            condition["inputs"][f"images.image_{index + 1}"] = [f"scale{index}", 0]
    nodes["conditioning"] = condition

    # --- sampling -------------------------------------------------------------------------------------------------
    nodes["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
    nodes["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}}
    nodes["sigmas"] = {"class_type": "ManualSigmas",
                       "inputs": {"sigmas": sigma_values or sigmas_for(DEFAULT_STEPS if steps is None else steps)}}
    nodes["guider"] = {"class_type": "BasicGuider",
                       "inputs": {"model": ["shift", 0], "conditioning": ["conditioning", 0]}}
    nodes["sample"] = {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                  "sigmas": ["sigmas", 0], "latent_image": ["conditioning", 1]}}
    nodes["decode"] = {"class_type": "VAEDecode",
                       "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    nodes["save"] = {"class_type": "SaveImage",
                     "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix}}

    return _numbered(nodes)


def _numbered(nodes):
    """Number the nodes in dependency-safe order and replace name links with ComfyUI's ["id", slot] form."""
    order = _order(nodes)
    ids = {key: str(index + 1) for index, key in enumerate(order)}
    out = {}
    for key in order:
        node = nodes[key]
        inputs = {}
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in ids:
                inputs[name] = [ids[value[0]], value[1]]
            else:
                inputs[name] = value
        out[ids[key]] = {"class_type": node["class_type"], "inputs": inputs}
    return out


def _order(nodes):
    """Every node, with link targets before the nodes that read them.

    Written out rather than discovered because the graph is small and fixed, and an explicit order makes an accidental
    cycle impossible instead of something to debug.
    """
    order = ["unet"]
    if "lora" in nodes:
        order.append("lora")
    order.append("shift")
    order += ["text_encoder", "vae"]
    if "upsampler" in nodes:
        order.append("upsampler")
    # reference loaders must precede whatever reads them
    order += sorted((k for k in nodes if k.startswith("load")), key=_index_of)
    if "refs_for_text" in nodes:
        order.append("refs_for_text")
    if "brief" in nodes:
        order.append("brief")
    if "format" in nodes:
        order.append("format")
    if "textgen" in nodes:
        order.append("textgen")
    order += sorted((k for k in nodes if k.startswith("scale")), key=_index_of)
    order.append("conditioning")
    order += ["noise", "sampler", "sigmas", "guider", "sample", "decode", "save"]
    missing = [k for k in nodes if k not in order]
    if missing:
        raise GraphError(f"internal: node(s) {missing} would have no id")
    return order


def _index_of(key):
    digits = "".join(ch for ch in key if ch.isdigit())
    return int(digits) if digits else 0


def resolve_names(client, cfg=None):
    """Ask ComfyUI for the real filename of each P-H3 file, so a renamed file fails loudly instead of silently."""
    cfg = cfg or {}
    out = {}
    for kind, key, default in (("unet", "ph3_unet", UNET), ("clip", "ph3_text_encoder", TEXT_ENCODER),
                               ("clip", "ph3_upsampler", UPSAMPLER), ("vae", "ph3_vae", VAE)):
        wanted = cfg.get(key) or default
        out[key] = client.name_for(kind, wanted)
    return out
