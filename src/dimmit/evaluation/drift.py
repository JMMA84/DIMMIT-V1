"""Control positivo de deriva: lote sintético degradado (desenfoque + oscuridad + JPEG-20).

Toma N imágenes de test, las degrada, corre el MISMO detector y calcula las mismas features de
visión (incluida la PCA del embedding). El monitor PSI de run_eval debe marcarlo en ALERTA/FALLA;
si no lo hace, el monitor no sirve. Salida: data/features/degradado_frame.parquet
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from dimmit.labels import pseudo_pci as pp
from dimmit.models.detector.predict import run as predict_run
from dimmit.models.score.vision import frame_features, image_quality
from dimmit.utils.io import DATA, FEATURES, INDEX, REPO_ROOT, ensure_dir, load_yaml


def degrade(src, dst, rng):
    with Image.open(src) as im:
        im = im.convert("RGB").filter(ImageFilter.GaussianBlur(radius=rng.uniform(2, 3.5)))
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.35, 0.55))
        im.save(dst, quality=20)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    cfg = load_yaml("configs/evaluation.yaml")
    rng = np.random.default_rng(7)
    splits = pd.read_csv(INDEX / "splits.csv")
    images = pd.read_parquet(INDEX / "images.parquet").set_index("image_id")
    ids = rng.choice(splits.loc[splits["split"] == "test", "image_id"].to_numpy(), cfg["deriva"]["degradacion_n"], replace=False)
    out_dir = ensure_dir(DATA / "interim" / "degradado" / "images")
    paths = []
    for i in ids:
        dst = out_dir / f"{i}.jpg"
        if not dst.exists():
            degrade(images.loc[i, "path"], dst, rng)
        paths.append(str(dst))
    det_dir = ensure_dir(FEATURES / "degradado_det")
    predict_run(REPO_ROOT / args.weights, paths, det_dir, imgsz=args.imgsz, batch=16)
    det = pd.read_parquet(det_dir / "detections.parquet")
    taus = pd.read_json(FEATURES / "vision_thresholds.json")["tau"].to_dict()
    f = frame_features(det, pd.Index(ids), taus, pp.load_cfg(), pp.load_cutoffs())
    q = pd.DataFrame([image_quality(p) for p in paths], columns=["brillo", "contraste", "nitidez"])
    q["image_id"] = ids
    q["jpeg_kb"] = [(out_dir / f"{i}.jpg").stat().st_size / 1024 for i in ids]
    f = f.merge(q, on="image_id")
    pca = np.load(FEATURES / "embedding_pca.npz")
    emb = np.load(det_dir / "embeddings.npy").astype(np.float32)
    pcs = ((emb - pca["mu"]) / pca["sd"]) @ pca["comps"].T
    for k in range(pcs.shape[1]):
        f[f"emb_pc{k:02d}"] = pcs[:, k]
    f.to_parquet(FEATURES / "degradado_frame.parquet", index=False)
    print(f"[ok] lote degradado: {len(f)} imágenes -> {FEATURES / 'degradado_frame.parquet'}")


if __name__ == "__main__":
    main()
