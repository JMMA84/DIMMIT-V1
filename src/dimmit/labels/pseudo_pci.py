"""Pseudo-PCI (ASTM D6433 simplificado) a partir de cajas de daño.

Se usa con las cajas anotadas por humanos (verdad de referencia) y, con las mismas fórmulas,
con las detecciones del modelo (features). Todas las funciones son vectorizadas sobre pandas.
"""
import json

import numpy as np
import pandas as pd

from dimmit.utils.io import CLASS_KEYS, INDEX, load_yaml


def load_cfg():
    return load_yaml("configs/labels.yaml")


def box_density(boxes: pd.DataFrame, cfg) -> np.ndarray:
    """Densidad (%) aportada por cada caja según su clase."""
    out = np.zeros(len(boxes))
    for cls, spec in cfg["density"].items():
        m = (boxes["cls"] == cls).to_numpy()
        var = {
            "h": boxes["h"],
            "w": boxes["w"],
            "wh": boxes["w"] * boxes["h"],
            "one": pd.Series(1.0, index=boxes.index),
        }[spec["var"]]
        out[m] = spec["k"] * var.to_numpy()[m]
    return out


def size_proxy(boxes: pd.DataFrame, cfg) -> np.ndarray:
    """Tamaño aparente corregido por perspectiva (las cajas lejanas se ven más pequeñas)."""
    hz = cfg["horizon_y"]
    dy = (boxes["cy"] - hz).to_numpy()
    delta = np.where(dy > 0, np.clip(0.6 / np.maximum(dy, 1e-6), 1, 3), 3.0)
    w, h = boxes["w"].to_numpy(), boxes["h"].to_numpy()
    cls = boxes["cls"].to_numpy()
    base = np.where(cls == "D00", h, np.where(cls == "D10", w, np.sqrt(w * h)))
    return delta * base


def severity_cutoffs(boxes: pd.DataFrame, cfg) -> dict:
    """Percentiles (L/M | M/H) del tamaño aparente por clase, sobre cajas de referencia."""
    s = size_proxy(boxes, cfg)
    lo, hi = cfg["severity_percentiles"]
    cuts = {}
    for cls in CLASS_KEYS:
        v = s[(boxes["cls"] == cls).to_numpy()]
        cuts[cls] = [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [1.0, 2.0]
    return cuts


def box_weighted_density(boxes: pd.DataFrame, cfg, cuts: dict, weights=None) -> np.ndarray:
    """Densidad ponderada por severidad (kappa) y opcionalmente por confianza."""
    s = size_proxy(boxes, cfg)
    kappa = np.full(len(boxes), cfg["severity_kappa"]["M"])
    cls = boxes["cls"].to_numpy()
    for c, (lo, hi) in cuts.items():
        m = cls == c
        kappa[m & (s < lo)] = cfg["severity_kappa"]["L"]
        kappa[m & (s >= hi)] = cfg["severity_kappa"]["H"]
    rho = kappa * box_density(boxes, cfg)
    if weights is not None:
        rho = rho * np.asarray(weights)
    return rho


def frame_rho(boxes: pd.DataFrame, image_ids, cfg, cuts, weights=None) -> pd.DataFrame:
    """Densidad ponderada por clase y fotograma (0 si no hay cajas)."""
    cols = [f"rho_{c}" for c in CLASS_KEYS]
    if len(boxes) == 0:
        return pd.DataFrame(0.0, index=pd.Index(image_ids, name="image_id"), columns=cols)
    b = boxes.copy()
    b["rho"] = box_weighted_density(b, cfg, cuts, weights)
    piv = b.pivot_table(index="image_id", columns="cls", values="rho", aggfunc="sum")
    piv = piv.reindex(columns=CLASS_KEYS).fillna(0.0)
    piv.columns = cols
    return piv.reindex(pd.Index(image_ids, name="image_id")).fillna(0.0)


def deduct_values(rho: pd.DataFrame, cfg) -> pd.DataFrame:
    out = pd.DataFrame(index=rho.index)
    for c in CLASS_KEYS:
        p = cfg["deduct"][c]
        r = rho[f"rho_{c}"].to_numpy()
        dv = np.clip(p["a"] + p["b"] * np.log10(np.maximum(r, 1e-9)), 0, p["dv_max"])
        out[f"dv_{c}"] = np.where(r > 0, dv, 0.0)
    return out


def combine_deducts(dv: np.ndarray, rest_weight: float) -> np.ndarray:
    """CDV = min(100, DVmax + w*(suma - DVmax)) por fila."""
    dv = np.asarray(dv, dtype=float)
    mx = dv.max(axis=1)
    return np.minimum(100.0, mx + rest_weight * (dv.sum(axis=1) - mx))


def pci_from_rho(rho: pd.DataFrame, cfg) -> pd.DataFrame:
    """Sub-scores por clase (100 - DV) y PCI visual desde densidades ponderadas."""
    dv = deduct_values(rho, cfg)
    out = rho.copy()
    for c in CLASS_KEYS:
        out[f"dv_{c}"] = dv[f"dv_{c}"]
        out[f"sub_{c}"] = 100.0 - dv[f"dv_{c}"]
    out["pci_vis"] = 100.0 - combine_deducts(dv.to_numpy(), cfg["cdv_rest_weight"])
    return out


def ride_index(iri, cfg):
    p = cfg["ride"]
    return 100.0 * np.exp(-p["k"] * np.maximum(0.0, np.asarray(iri, dtype=float) - p["iri0"]))


def icv(pci_vis, ride, cfg):
    w = cfg["icv_weights"]
    return w["visual"] * np.asarray(pci_vis) + w["ride"] * np.asarray(ride)


def state_of(score, cfg):
    """Estado por umbrales (límite inferior); vectorizado."""
    items = sorted(cfg["states"].items(), key=lambda kv: -kv[1])
    score = np.asarray(score, dtype=float)
    out = np.full(score.shape, items[-1][0], dtype=object)
    for name, lower in reversed(items):
        out[score >= lower] = name
    return out


STATE_ORDER = ["muy_malo", "malo", "regular", "satisfactorio", "bueno"]


def state_index(states):
    return np.array([STATE_ORDER.index(s) for s in states])


def save_cutoffs(cuts):
    (INDEX / "severity_cutoffs.json").write_text(json.dumps(cuts, indent=2))


def load_cutoffs():
    return json.loads((INDEX / "severity_cutoffs.json").read_text())
