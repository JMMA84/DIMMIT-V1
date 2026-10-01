"""Métricas del detector en el split de test (global y por país) -> detector_metrics.json.

Por clase: precisión, recall, F1, AP50, AP50-95; matriz de confusión (incluye fondo).
"""
import argparse
import json

import numpy as np

from dimmit.utils.io import REPO_ROOT, ensure_dir, load_yaml


def _metrics(model, data, imgsz, batch, split="test"):
    m = model.val(data=str(data), split=split, imgsz=imgsz, batch=batch, conf=0.001, plots=False, verbose=False)
    names = load_yaml("configs/data_rdd2020.yaml")["names"]
    box = m.box
    per_class = {}
    for j, c in enumerate(box.ap_class_index):
        p, r = float(box.p[j]), float(box.r[j])
        per_class[names[int(c)].split("_")[0]] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "ap50": float(box.ap50[j]),
            "ap50_95": float(box.ap[j]),
        }
    out = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "por_clase": per_class,
        "velocidad_ms": {k: float(v) for k, v in m.speed.items()},
    }
    cm = getattr(m, "confusion_matrix", None)
    if cm is not None:
        out["matriz_confusion"] = np.asarray(cm.matrix).astype(int).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="data/features/detector")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    model = YOLO(str(REPO_ROOT / args.weights))
    cfg = load_yaml("configs/data_rdd2020.yaml")
    proc = REPO_ROOT / "data/processed"
    res = {"test": _metrics(model, proc / "data.yaml", args.imgsz, args.batch)}
    for country in cfg["countries"]:
        res[f"test_{country}"] = _metrics(model, proc / f"data_{country}.yaml", args.imgsz, args.batch)
    out = ensure_dir(REPO_ROOT / args.out) / "detector_metrics.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[ok] mAP50 test = {res['test']['map50']:.3f} -> {out}")


if __name__ == "__main__":
    main()
