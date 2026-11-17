"""The three numbers a person sets per project: how many pictures the dataset aims for, how many epochs the seed LoRA
trains, and how many steps the final training runs.

They live in `<project>/settings.json`, not in `project.json`, on purpose: the card is hashed into every seed snapshot
and export id, so keeping training lengths there would make the same pictures a "different" dataset each time someone
changed how long to train. A missing file, a missing key or a value that is not a positive whole number reads as the
default, so an old project opens with 50 / 30 / 2000 without being rewritten.

* `target_images` - the size of the planned dataset, originals included (half body shots, half face shots).
* `seed_epochs`   - passes over the seed snapshot when `pipeline.seed --train` is run without `--epochs`.
* `final_steps`   - the `target_steps` written into each final training job by `pipeline.export todos`.
"""
from __future__ import annotations

import json
from pathlib import Path

FILE = "settings.json"
DEFAULTS = {"target_images": 50, "seed_epochs": 30, "final_steps": 2000}
# Bounds that catch a typo (an extra zero, a pasted id) without second-guessing a real choice.
LIMITS = {"target_images": (1, 1000), "seed_epochs": (1, 1000), "final_steps": (1, 100000)}


class SettingsError(ValueError):
    """A value that cannot be saved: not a whole number, or outside its bounds."""


def path(project):
    """`project` is a Project or its folder."""
    return Path(getattr(project, "dir", project)) / FILE


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def read(project):
    """Every setting, with the default for anything absent or unusable. Never raises for a bad file."""
    values = dict(DEFAULTS)
    try:
        stored = json.loads(path(project).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return values
    if not isinstance(stored, dict):
        return values
    for key, (low, high) in LIMITS.items():
        number = _number(stored.get(key))
        if number is not None and low <= number <= high:
            values[key] = number
    return values


def validate(values):
    """The settings as whole numbers, or SettingsError naming the first bad one. Unknown keys are refused."""
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise SettingsError("unknown setting(s): " + ", ".join(unknown))
    out = {}
    for key, value in values.items():
        number = _number(value)
        low, high = LIMITS[key]
        if number is None or not low <= number <= high:
            raise SettingsError(f"{key} must be a whole number from {low} to {high}, got {value!r}")
        out[key] = number
    return out


def write(project, values):
    """Merge `values` over what is stored and save; returns the full settings now in effect."""
    from . import project as pj

    merged = read(project)
    merged.update(validate(values))
    pj.write_json(path(project), merged)
    return merged
