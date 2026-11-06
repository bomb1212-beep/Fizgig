"""Dataset pipeline: generation, selection and export around the Fizgig trainer.

Modules keep the runner's conventions: standard library only where practical, JSON state written atomically, and
no import-time side effects, so the offline unit tests can import any of them without a ComfyUI, a GPU or a model.
"""
from __future__ import annotations

from pathlib import Path


def make_cancel(stop_file):
    """A `cancel()` for a long draw, driven by the existence of one file.

    Every long stage here already takes a `cancel` callback and checks it between cells, so a caller - a GUI, or a
    person who wants to stop after the current picture - only needs a way to flip it. A file is used rather than a
    signal because the check happens at a cell boundary: the picture being drawn finishes and stays on disk, which is
    what makes a rerun continue the grid instead of starting it again.

    Returns None when no stop file was asked for, so the stages keep their default "no cancel hook" behaviour.
    """
    if not stop_file:
        return None
    marker = Path(stop_file)
    return lambda: marker.exists()

