"""Utilidades compartidas: rutas, configs, tablas y parches de compatibilidad."""
import json
import os
from pathlib import Path

import yaml

# DIMMIT_ROOT permite correr el mismo código fuera del repo (p. ej. en el kernel de Kaggle)
REPO_ROOT = Path(os.environ.get("DIMMIT_ROOT", Path(__file__).resolve().parents[3]))

DATA = REPO_ROOT / "data"
INDEX = DATA / "index"
FEATURES = DATA / "features"
EXTERNAL = DATA / "external"
REPORTS = REPO_ROOT / "reports"
OUTPUT = REPORTS / "output"
EVAL = REPORTS / "evaluation"

CLASS_KEYS = ["D00", "D10", "D20", "D40"]
CLASS_SLUGS = {
    "D00": "grieta_longitudinal",
    "D10": "grieta_transversal",
    "D20": "piel_cocodrilo",
    "D40": "baches",
}


def load_yaml(path):
    with open(REPO_ROOT / path) as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_compat_patches():
    """Parches para cargar checkpoints antiguos con torch/numpy actuales (inofensivos con
    ultralytics 8.3):
    - torch>=2.6 usa weights_only=True por defecto y rompe la carga de checkpoints.
    - numpy 2 eliminó np.trapz."""
    import numpy as np
    import torch

    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _load
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
