"""Dataset pipeline: generation, selection and export around the Fizgig trainer.

Modules keep the runner's conventions: standard library only where practical, JSON state written atomically, and
no import-time side effects, so the offline unit tests can import any of them without a ComfyUI, a GPU or a model.
"""
