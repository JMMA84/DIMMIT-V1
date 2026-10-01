"""Features de visión por fotograma y por segmento a partir de las salidas de YOLOv10.

- Umbral por clase tau_c: maximiza el F1 de presencia de la clase por imagen en el conjunto F.
- Por fotograma: conteos duros (>= tau_c) y blandos (suma de confianzas), confianza máxima,
  detecciones bajo umbral, área dañada (unión en rejilla 64x64), tamaños, densidad/DV/PCI con las
  mismas fórmulas del pseudo-PCI (PCI_det), posición vertical, franja de rodada, entropía de
  clases, calidad de imagen (brillo, contraste, nitidez, tamaño JPEG) y PCA-32 del embedding.
- Por segmento: media/máximo, fracción de fotogramas con cada clase y persistencia entre
  fotogramas consecutivos de la pseudo-ruta.
Salidas: data/features/vision_frame.parquet, vision_segment.parquet, vision_thresholds.json
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from dimmit.labels import pseudo_pci as pp
from dimmit.utils.io import CLASS_KEYS, FEATURES, INDEX, ensure_dir

GRID = 64


def image_quality(path):
    with Image.open(path) as im:
        g = im.convert("L").resize((256, 256))
    a = np.asarray(g, dtype=np.float32)
    lap = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return float(a.mean()), float(a.std()), float(lap.var())


def quality_table(images):
    out = INDEX / "image_quality.parquet"
    if out.exists():
        q = pd.read_parquet(out)
        if set(images["image_id"]) <= set(q["image_id"]):
            return q
    with ProcessPoolExecutor(4) as ex:
        vals = list(ex.map(image_quality, images["path"], chunksize=128))
    q = pd.DataFrame(vals, columns=["brillo", "contraste", "nitidez"])
    q.insert(0, "image_id", images["image_id"].to_numpy())
    q["jpeg_kb"] = images["jpeg_kb"].to_numpy()
    q.to_parquet(out, index=False)
    return q


def presence_thresholds(det, gt_images, boxes):
    """tau_c por clase: máximo F1 de presencia por imagen (sobre las imágenes de F)."""
    taus, report = {}, {}
    grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
    for k, c in enumerate(CLASS_KEYS):
        truth = gt_images.isin(boxes.loc[boxes["cls"] == c, "image_id"])
        maxc = det[det["class_id"] == k].groupby("image_id")["conf"].max().reindex(gt_images).fillna(0).to_numpy()
        best = (0.0, 0.25, 0.0, 0.0)
        for t in grid:
            pred = maxc >= t
            tp = (pred & truth.to_numpy()).sum()
            p = tp / max(pred.sum(), 1)
            r = tp / max(truth.sum(), 1)
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if f1 > best[0]:
                best = (f1, float(t), p, r)
        taus[c] = best[1]
        report[c] = {"tau": best[1], "f1": best[0], "precision": best[2], "recall": best[3]}
    return taus, report


def union_area(b):
    """Fracción de la imagen cubierta por la unión de cajas (rejilla GRIDxGRID)."""
    if len(b) == 0:
        return 0.0
    m = np.zeros((GRID, GRID), dtype=bool)
    for cx, cy, w, h in b:
        x0, x1 = int(max(0, (cx - w / 2) * GRID)), int(min(GRID, np.ceil((cx + w / 2) * GRID)))
        y0, y1 = int(max(0, (cy - h / 2) * GRID)), int(min(GRID, np.ceil((cy + h / 2) * GRID)))
        m[y0:y1, x0:x1] = True
    return float(m.mean())


def _union_by_image(boxes):
    if len(boxes) == 0:
        return pd.Series(dtype=float)
    return pd.Series({i: union_area(g[["cx", "cy", "w", "h"]].to_numpy()) for i, g in boxes.groupby("image_id")})


def frame_features(det, image_ids, taus, cfg_lab, cuts):
    det = det.assign(cls=det["class_id"].map(dict(enumerate(CLASS_KEYS))))
    det["tau"] = det["cls"].map(taus)
    hard = det[det["conf"] >= det["tau"]]
    feats = pd.DataFrame(index=pd.Index(image_ids, name="image_id"))
    for c in CLASS_KEYS:
        dc, hc = det[det["cls"] == c], hard[hard["cls"] == c]
        feats[f"n_{c}"] = hc.groupby("image_id").size()
        feats[f"soft_{c}"] = dc.groupby("image_id")["conf"].sum()
        feats[f"maxconf_{c}"] = dc.groupby("image_id")["conf"].max()
        feats[f"nsub_{c}"] = dc[(dc["conf"] >= 0.05) & (dc["conf"] < dc["tau"])].groupby("image_id").size()
        area = hc["w"] * hc["h"]
        feats[f"area_mean_{c}"] = area.groupby(hc["image_id"]).mean()
        feats[f"area_max_{c}"] = area.groupby(hc["image_id"]).max()
        feats[f"cy_mean_{c}"] = hc.groupby("image_id")["cy"].mean()
        feats[f"union_{c}"] = _union_by_image(hc)
    feats = feats.fillna({c: 0.0 for c in feats.columns if not c.startswith("cy_mean")})
    for c in CLASS_KEYS:
        feats[f"cy_mean_{c}"] = feats[f"cy_mean_{c}"].fillna(0.0)
    feats["union_total"] = _union_by_image(hard)
    feats["union_total"] = feats["union_total"].fillna(0.0)
    wp = hard.assign(a=hard["w"] * hard["h"], wp=(hard["cx"] - 0.5).abs() <= 0.3)
    feats["rodada_share"] = (wp["a"] * wp["wp"]).groupby(wp["image_id"]).sum() / wp.groupby("image_id")["a"].sum()
    feats["rodada_share"] = feats["rodada_share"].fillna(0.0)
    counts = feats[[f"n_{c}" for c in CLASS_KEYS]].to_numpy()
    p = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    feats["entropia_clases"] = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0)).sum(1)
    feats["max_conf"] = det.groupby("image_id")["conf"].max().reindex(feats.index).fillna(0.0)
    # densidades y PCI con las fórmulas del pseudo-PCI: duras y blandas (ponderadas por confianza)
    rho_h = pp.frame_rho(hard[["image_id", "cls", "cx", "cy", "w", "h"]], feats.index, cfg_lab, cuts)
    pci_h = pp.pci_from_rho(rho_h, cfg_lab)
    for c in CLASS_KEYS:
        feats[f"rho_det_{c}"] = pci_h[f"rho_{c}"]
        feats[f"dv_det_{c}"] = pci_h[f"dv_{c}"]
    feats["pci_det"] = pci_h["pci_vis"]
    soft = det[det["conf"] >= 0.05]
    rho_s = pp.frame_rho(soft[["image_id", "cls", "cx", "cy", "w", "h"]], feats.index, cfg_lab, cuts, weights=soft["conf"].to_numpy())
    for c in CLASS_KEYS:
        feats[f"rho_soft_{c}"] = rho_s[f"rho_{c}"]
    feats["pci_soft"] = pp.pci_from_rho(rho_s, cfg_lab)["pci_vis"]
    return feats.reset_index()


def embedding_pca(det_dir, splits, n=32):
    emb = np.load(det_dir / "embeddings.npy").astype(np.float32)
    ids = pd.read_csv(det_dir / "embeddings_index.csv")["image_id"].to_numpy()
    sp = splits.set_index("image_id").loc[ids, "split"].to_numpy()
    fit = sp == "train" if (sp == "train").sum() > 1000 else sp == "fusion"
    mu = emb[fit].mean(0)
    sd = emb[fit].std(0) + 1e-6
    z = (emb - mu) / sd
    _, _, vt = np.linalg.svd(z[fit], full_matrices=False)
    comps = vt[:n]
    pcs = z @ comps.T
    np.savez(FEATURES / "embedding_pca.npz", mu=mu, sd=sd, comps=comps, fit_split="train" if (sp == "train").sum() > 1000 else "fusion")
    out = pd.DataFrame(pcs, columns=[f"emb_pc{k:02d}" for k in range(n)])
    out.insert(0, "image_id", ids)
    return out


def segment_features(frame, splits):
    df = frame.merge(splits[["image_id", "segment_id", "chain_pos"]], on="image_id").sort_values(["segment_id", "chain_pos"])
    num = [c for c in frame.columns if c != "image_id" and not c.startswith("emb_pc")]
    g = df.groupby("segment_id")
    seg = pd.concat([g[num].mean().add_suffix("_mean"), g[num].max().add_suffix("_max")], axis=1)
    emb_cols = [c for c in frame.columns if c.startswith("emb_pc")]
    if emb_cols:
        seg = seg.join(g[emb_cols].mean())
    extra = {}
    for c in CLASS_KEYS:
        has = (df[f"n_{c}"] > 0).astype(float)
        extra[f"frac_frames_{c}"] = has.groupby(df["segment_id"]).mean()
        both = has * has.groupby(df["segment_id"]).shift(1)
        extra[f"persistencia_{c}"] = both.groupby(df["segment_id"]).mean().fillna(0.0)
    extra["n_fotogramas"] = g.size()
    return pd.concat([seg, pd.DataFrame(extra)], axis=1).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det", default="data/features/detector", help="carpeta con detections.parquet y embeddings")
    args = ap.parse_args()
    from dimmit.utils.io import REPO_ROOT

    det_dir = REPO_ROOT / args.det
    ensure_dir(FEATURES)
    splits = pd.read_csv(INDEX / "splits.csv")
    images = pd.read_parquet(INDEX / "images.parquet")
    boxes = pd.read_parquet(INDEX / "boxes.parquet")
    det = pd.read_parquet(det_dir / "detections.parquet")
    ids = pd.read_csv(det_dir / "embeddings_index.csv")["image_id"] if (det_dir / "embeddings_index.csv").exists() else det["image_id"].unique()
    cfg_lab = pp.load_cfg()
    cuts = pp.load_cutoffs()

    fus = splits.loc[splits["split"] == "fusion", "image_id"]
    taus, rep = presence_thresholds(det[det["image_id"].isin(fus)], fus, boxes)
    (FEATURES / "vision_thresholds.json").write_text(json.dumps({"tau": taus, "f1_presencia_F": rep}, indent=2))
    print("[ok] umbrales:", taus)

    frame = frame_features(det, pd.Index(ids), taus, cfg_lab, cuts)
    q = quality_table(images[images["image_id"].isin(frame["image_id"])])
    frame = frame.merge(q, on="image_id", how="left")
    if (det_dir / "embeddings.npy").exists():
        frame = frame.merge(embedding_pca(det_dir, splits), on="image_id", how="left")
    frame.to_parquet(FEATURES / "vision_frame.parquet", index=False)
    seg = segment_features(frame, splits)
    seg.to_parquet(FEATURES / "vision_segment.parquet", index=False)
    (FEATURES / "detector_source.txt").write_text(str(det_dir))
    print(f"[ok] {len(frame)} fotogramas, {len(seg)} segmentos, {frame.shape[1]} columnas -> {FEATURES}")


if __name__ == "__main__":
    main()
