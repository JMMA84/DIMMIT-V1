"""Umbrales de confianza de campo derivados de RDD2020: precisión por caja frente a confianza.

Para cada clase, tau_campo = la menor confianza (de una rejilla) a la que la precisión por caja
(IoU >= 0.5 con una anotación humana de la misma clase, emparejamiento voraz por confianza) en el
TEST de RDD2020 alcanza `precision_objetivo`. Escribe data/campo/features/umbrales_campo.json con la
tabla completa para que el umbral sea auditable, y sugiere el bloque `filtros.tau_campo`.
"""
import argparse
import json

import numpy as np
import pandas as pd

from dimmit.campo.filtros import iou_matrix
from dimmit.campo.scores import FEAT
from dimmit.utils.io import CLASS_KEYS, INDEX, REPO_ROOT, ensure_dir

GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def box_tp(det, gt, iou_min=0.5):
    det = det.sort_values("conf", ascending=False).reset_index(drop=True)
    tp = np.zeros(len(det), bool)
    g_by = {k: v for k, v in gt.groupby("image_id")}
    for img, grp in det.groupby("image_id"):
        g = g_by.get(img)
        if g is None:
            continue
        for c in CLASS_KEYS:
            gi = g[g["cls"] == c][["cx", "cy", "w", "h"]].to_numpy()
            di = grp[grp["cls"] == c]
            if not len(gi) or not len(di):
                continue
            m = iou_matrix(di[["cx", "cy", "w", "h"]].to_numpy(float), gi)
            used = set()
            for r, idx in enumerate(di.index):
                j = int(m[r].argmax())
                if m[r, j] >= iou_min and j not in used:
                    used.add(j)
                    tp[idx] = True
    return det.assign(tp=tp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det", default="models/kaggle/dimmit-yolov10-rdd2020/outputs/detections.parquet")
    ap.add_argument("--precision-objetivo", type=float, default=0.5)
    args = ap.parse_args()
    splits = pd.read_csv(INDEX / "splits.csv")
    test = set(splits.loc[splits["split"] == "test", "image_id"])
    det = pd.read_parquet(REPO_ROOT / args.det)
    det = det[det["image_id"].isin(test)].copy()
    det["cls"] = det["class_id"].map(dict(enumerate(CLASS_KEYS)))
    gt = pd.read_parquet(INDEX / "boxes.parquet")
    gt = gt[gt["image_id"].isin(test) & gt["cls"].isin(CLASS_KEYS)]
    det = box_tp(det, gt)
    tabla, tau = {}, {}
    for c in CLASS_KEYS:
        d, n_gt = det[det["cls"] == c], int((gt["cls"] == c).sum())
        tabla[c] = []
        for th in GRID:
            s = d[d["conf"] >= th]
            prec = float(s["tp"].mean()) if len(s) else float("nan")
            tabla[c].append({"conf": th, "precision": prec, "recall": float(s["tp"].sum() / max(n_gt, 1)), "n_cajas": int(len(s))})
            if c not in tau and len(s) and prec >= args.precision_objetivo:
                tau[c] = th
    out = {"fuente": "test de RDD2020 (3 folds T), IoU >= 0.5, emparejamiento voraz por confianza", "precision_objetivo": args.precision_objetivo,
           "tau_campo": tau, "tabla": tabla}
    ensure_dir(FEAT)
    (FEAT / "umbrales_campo.json").write_text(json.dumps(out, indent=1))
    print(f"[ok] tau_campo = {tau} -> {FEAT / 'umbrales_campo.json'}")
    for c in CLASS_KEYS:
        print(c, " ".join(f"@{r['conf']:.2f} P={r['precision']:.2f} R={r['recall']:.2f}" for r in tabla[c]))


if __name__ == "__main__":
    main()
