"""Inferencia del detector sobre una lista de imágenes -> detecciones + embeddings.

- Procesa por lotes propios (no acumula Results en memoria: el v0 se quedaba sin RAM).
- Embeddings: hooks en las capas que alimentan la cabeza (head.f, P3/P4/P5); cada mapa se
  promedia globalmente y se concatenan (64+128+256 = 448 dims en YOLOv10n).
Salidas en --out: detections.parquet, embeddings.npy (float16), embeddings_index.csv,
latency.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dimmit.utils.io import REPO_ROOT, ensure_dir

DET_COLS = ["image_id", "class_id", "conf", "cx", "cy", "w", "h"]


class EmbeddingHooks:
    """Captura la salida de las capas de entrada de la cabeza en el último forward."""

    def __init__(self, det_model):
        head = det_model.model[-1]
        self.idx = list(head.f) if isinstance(head.f, (list, tuple)) else [head.f]
        self.buf = {}
        self.handles = [det_model.model[i].register_forward_hook(self._hook(i)) for i in self.idx]

    def _hook(self, i):
        def fn(_m, _inp, out):
            self.buf[i] = out.detach().float().mean(dim=(2, 3)).cpu().numpy()

        return fn

    def take(self):
        return np.concatenate([self.buf[i] for i in self.idx], axis=1)

    def close(self):
        for h in self.handles:
            h.remove()


def run(weights, paths, out_dir, conf=0.01, max_det=100, imgsz=640, batch=16, embeddings=True, device=None):
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    out_dir = ensure_dir(out_dir)
    model = YOLO(str(weights))
    hooks = None
    rows, embs, ids = [], [], []
    t_inf, n_done = 0.0, 0
    kw = dict(conf=conf, max_det=max_det, imgsz=imgsz, verbose=False)
    if device is not None:
        kw["device"] = device
    model.predict(paths[:1], **kw)  # calentamiento (antes de registrar hooks)
    if embeddings:
        hooks = EmbeddingHooks(model.model)
    for s in range(0, len(paths), batch):
        chunk = paths[s : s + batch]
        t = time.time()
        results = model.predict(chunk, batch=len(chunk), **kw)
        t_inf += time.time() - t
        if hooks:
            e = hooks.take()
            assert len(e) == len(results), "lote de embeddings desalineado"
            embs.append(e.astype(np.float16))
        for res in results:
            image_id = Path(res.path).stem
            ids.append(image_id)
            b = res.boxes
            if len(b):
                xywhn = b.xywhn.cpu().numpy()
                for c, p, (x, y, w, h) in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), xywhn):
                    rows.append((image_id, int(c), float(p), float(x), float(y), float(w), float(h)))
        n_done += len(chunk)
        if (s // batch) % 50 == 0:
            print(f"  {n_done}/{len(paths)} imágenes ({1000 * t_inf / n_done:.0f} ms/img)", flush=True)
    if hooks:
        hooks.close()
    pd.DataFrame(rows, columns=DET_COLS).to_parquet(out_dir / "detections.parquet", index=False)
    if embs:
        np.save(out_dir / "embeddings.npy", np.concatenate(embs))
        pd.DataFrame({"image_id": ids}).to_csv(out_dir / "embeddings_index.csv", index=False)
    # latencia con lote 1 (despliegue en el vehículo procesa imagen a imagen)
    lat = []
    for p in paths[: min(50, len(paths))]:
        t = time.time()
        model.predict(p, **kw)
        lat.append(1000 * (time.time() - t))
    info = {
        "imagenes": len(ids),
        "detecciones": len(rows),
        "ms_por_imagen_lote": 1000 * t_inf / max(1, n_done),
        "latencia_lote1_ms_p50": float(np.percentile(lat, 50)),
        "latencia_lote1_ms_p95": float(np.percentile(lat, 95)),
        "dispositivo": str(model.device),
        "imgsz": imgsz,
    }
    (out_dir / "latency.json").write_text(json.dumps(info, indent=2))
    print(f"[ok] {len(rows)} detecciones en {len(ids)} imágenes -> {out_dir}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", default="data/processed/all.txt", help="archivo con rutas de imágenes")
    ap.add_argument("--out", default="data/features/detector")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()
    paths = [p for p in (REPO_ROOT / args.list).read_text().split() if p]
    if args.limit:
        paths = paths[: args.limit]
    run(REPO_ROOT / args.weights, paths, REPO_ROOT / args.out, args.conf, imgsz=args.imgsz, batch=args.batch, embeddings=not args.no_embeddings)


if __name__ == "__main__":
    main()
