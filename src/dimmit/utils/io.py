"""Utilidades compartidas."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path):
    with open(REPO_ROOT / path) as f:
        return yaml.safe_load(f)


def patch_torch_load():
    """torch>=2.6 usa weights_only=True por defecto y rompe la carga de
    checkpoints de yolov10 (ultralytics 8.1). Se fuerza el comportamiento previo."""
    import torch

    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _load
