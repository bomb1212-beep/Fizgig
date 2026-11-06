"""Identity scoring for 2D sets: how close a picture's CLIP embedding is to the user's originals.

The spec asks for two numbers - the **maximum** and the **mean** cosine against the originals - and for the option of
comparing **crops** when the user marks a face or upper-body region. Both are here.

Why maximum as well as mean: a character can be photographed at many angles and in many outfits, so the originals do
not form one tight cluster. A picture that matches one original closely is probably the same person seen the same way;
averaging over originals that look nothing like it drags that signal down. Reporting both lets a person see which
happened.

The model is ComfyUI's own `clip_vision_h.safetensors`, loaded through transformers rather than through ComfyUI: the
CLIP Vision config is a CLIP ViT-H/14 and the file's tensors line up with it exactly (checked: 0 missing, 1 unused
`position_ids`). Nothing is downloaded and no new dependency is added - both were the spec's conditions.

Crops are given by the user as boxes, because the spec rules out a face detector: `{"face": [x, y, w, h]}` in pixels
relative to the image. A crop is compared only against crops of the same name, so a face box is never compared with a
full frame.
"""
from __future__ import annotations

from pathlib import Path

from . import ScorerError, result

DEFAULT_MODEL = Path(r"G:\Lakis\ComfyUI\models\clip_vision\clip_vision_h.safetensors")
# The CLIP ViT-H/14 shape, read off the checkpoint rather than guessed: projection 1024x1280, patch 14, 32 layers.
HIDDEN, LAYERS, HEADS, PATCH, PROJECTION = 1280, 32, 16, 14, 1024
IMAGE_SIZE = 224
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


class Embedder:
    """Loads the CLIP Vision model once and embeds images in batches.

    Lazy on purpose: importing this module must not pull in torch, so the scorers that do not need a model (duplicates,
    white ratio, slot fill) stay usable on a machine without a GPU.
    """

    def __init__(self, model_path=None, device=None, batch=8):
        self.model_path = Path(model_path or DEFAULT_MODEL)
        self.device = device
        self.batch = int(batch)
        self._model = None
        self._proc = None

    def _ensure(self):
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise ScorerError(f"CLIP Vision weights are not at {self.model_path}; point clip_vision.model at the "
                              "clip_vision_h.safetensors ComfyUI already has")
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ScorerError(f"scoring needs torch/transformers/safetensors in the Fizgig venv: {exc}") from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = load_file(str(self.model_path))
        config = CLIPVisionConfig(hidden_size=HIDDEN, intermediate_size=5120, num_hidden_layers=LAYERS,
                                  num_attention_heads=HEADS, image_size=IMAGE_SIZE, patch_size=PATCH,
                                  projection_dim=PROJECTION)
        model = CLIPVisionModelWithProjection(config)
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self._model = model
        self._proc = CLIPImageProcessor(size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
                                        crop_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
                                        image_mean=MEAN, image_std=STD)

    def embed(self, paths):
        """{path: unit-length embedding tensor} for every path, in one or more batches."""
        self._ensure()
        import torch
        from PIL import Image
        out = {}
        paths = [Path(p) for p in paths]
        for start in range(0, len(paths), self.batch):
            chunk = paths[start:start + self.batch]
            images = []
            for path in chunk:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except OSError as exc:
                    raise ScorerError(f"cannot read {path}: {exc}") from exc
            inputs = self._proc(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeds = self._model(pixel_values=inputs["pixel_values"].to(self._model.dtype)).image_embeds
            embeds = torch.nn.functional.normalize(embeds.float(), dim=-1).cpu()
            for path, vector in zip(chunk, embeds):
                out[str(path)] = vector
        return out

    def embed_crops(self, path, boxes):
        """{box name: unit embedding} for the named boxes of one image."""
        self._ensure()
        import torch
        from PIL import Image
        image = Image.open(path).convert("RGB")
        crops, names = [], []
        for name, box in boxes.items():
            x, y, w, h = (int(v) for v in box)
            if w <= 0 or h <= 0:
                raise ScorerError(f"crop {name!r} for {Path(path).name} has no area: {box}")
            crops.append(image.crop((x, y, x + w, y + h)))
            names.append(name)
        if not crops:
            return {}
        inputs = self._proc(images=crops, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds = self._model(pixel_values=inputs["pixel_values"].to(self._model.dtype)).image_embeds
        embeds = torch.nn.functional.normalize(embeds.float(), dim=-1).cpu()
        return {name: vector for name, vector in zip(names, embeds)}


def cosine(a, b):
    """Cosine between two unit vectors, as a plain float."""
    return float((a * b).sum())


def _digest(path):
    """The sha256 of a file's bytes, or "" when it cannot be read.

    Self-exclusion has to work on **content**, not on path: an export copies every picture into its own folder under a
    name like `<sha16>.png`, so the exported copy of an original is a different file with the same bytes. Comparing it
    with the original scored it at exactly 1.0 and read as perfect identity rather than "this is the same picture".
    """
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def identity_scores(paths, reference_paths, *, embedder=None, crops=None, ids=None, exclude_self=True):
    """Score each image against the user's originals.

    `paths` is the pictures to score, `reference_paths` the originals to compare against. Returns one row per image with
    `max_cosine` and `mean_cosine`, plus `crop_max` / `crop_mean` per crop name when `crops` gives boxes.

    Both are reported because neither answers the question alone: the maximum finds "looks like this one original",
    the mean finds "looks like the character in general", and a picture can be high on one and low on the other.

    `exclude_self` drops an image from **its own** reference list, matched by file **content**: an exported copy of an
    original has different path but identical bytes, and comparing it with itself scored every original at exactly 1.0.
    An original still gets scored against the *other* originals, which is the meaningful number - "how typical is this
    picture of the set".
    """
    reference_paths = [Path(p) for p in reference_paths]
    if not reference_paths:
        raise ScorerError("identity scoring needs at least one original to compare against")
    embedder = embedder or Embedder()
    targets = [Path(p) for p in paths]
    ids = list(ids) if ids is not None else [p.stem for p in targets]
    if len(ids) != len(targets):
        raise ScorerError(f"{len(ids)} id(s) for {len(targets)} image(s)")

    everything = targets + reference_paths
    vecs = embedder.embed(everything)
    ref_digests = [_digest(p) for p in reference_paths] if exclude_self else [""] * len(reference_paths)

    crop_ref = {}
    if crops:
        for path in reference_paths:
            boxes = crops.get(Path(path).name) or crops.get(str(path))
            if boxes:
                for name, vector in embedder.embed_crops(path, boxes).items():
                    crop_ref.setdefault(name, []).append(vector)

    rows, no_reference = [], []
    for image_id, path in zip(ids, targets):
        vector = vecs[str(path)]
        mine = _digest(path) if exclude_self else ""
        # An image that is itself one of the originals is compared with the others, never with its own bytes.
        keep = [i for i, ref in enumerate(reference_paths) if not (exclude_self and mine and ref_digests[i] == mine)]
        dropped = len(keep) < len(reference_paths)
        pool = [reference_paths[i] for i in keep]
        if not pool:
            # The only original in the set: there is nothing to compare it with, and 1.0 would be a lie.
            no_reference.append(str(path))
            rows.append({"id": image_id, "path": str(path), "max_cosine": None, "mean_cosine": None,
                         "closest": None, "reference_count": 0, "self_excluded": dropped})
            continue
        sims = [cosine(vector, vecs[str(p)]) for p in pool]
        row = {"id": image_id, "path": str(path),
               "max_cosine": max(sims), "mean_cosine": sum(sims) / len(sims),
               "closest": pool[sims.index(max(sims))].name,
               "reference_count": len(pool), "self_excluded": dropped}
        boxes = (crops or {}).get(Path(path).name) or (crops or {}).get(str(path))
        if boxes and crop_ref:
            mine_crops = embedder.embed_crops(path, boxes)
            for name, vector_c in mine_crops.items():
                others = crop_ref.get(name) or []
                if not others:
                    continue
                sims_c = [cosine(vector_c, r) for r in others]
                row[f"crop_{name}_max"] = max(sims_c)
                row[f"crop_{name}_mean"] = sum(sims_c) / len(sims_c)
        rows.append(row)
    notes = ["a score is a measurement, not a decision: nothing is rejected on it"]
    if exclude_self:
        notes.append("an image whose bytes are one of the originals is compared only with the other originals")
    if no_reference:
        notes.append(f"{len(no_reference)} image(s) had no other original to compare with and carry no score")
    return result("clip_vision", family="2d", rows=rows,
                  reference={"kind": "originals", "count": len(reference_paths),
                             "paths": [str(p) for p in reference_paths]},
                  notes=notes,
                  coverage={"scored": len(rows) - len(no_reference), "references": len(reference_paths),
                            "no_reference": no_reference})
