"""Inferencia del detector sobre el split de validación -> reports/detections.csv."""
import argparse

import pandas as pd

from dimmit.utils.io import REPO_ROOT, apply_compat_patches, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolov10n_small/weights/best.pt")
    ap.add_argument("--source", default="data/processed/images/val")
    ap.add_argument("--out", default="reports/detections.csv")
    ap.add_argument("--conf", type=float, default=0.01, help="umbral bajo: el modelo v0 apenas entrena 3 épocas")
    args = ap.parse_args()

    apply_compat_patches()
    from ultralytics import YOLOv10

    names = load_yaml("configs/data_rdd2020.yaml")["names"]
    model = YOLOv10(str(REPO_ROOT / args.weights))
    results = model.predict(source=str(REPO_ROOT / args.source), conf=args.conf, verbose=False)

    rows = []
    for res in results:
        image = res.path.split("/")[-1]
        for box in res.boxes:
            cls = int(box.cls)
            x, y, w, h = box.xywhn[0].tolist()
            rows.append(
                {
                    "image": image,
                    "class_id": cls,
                    "class_name": names[cls],
                    "conf": float(box.conf),
                    "cx": x,
                    "cy": y,
                    "w": w,
                    "h": h,
                }
            )
    out = REPO_ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["image", "class_id", "class_name", "conf", "cx", "cy", "w", "h"]).to_csv(
        out, index=False
    )
    print(f"[ok] {len(rows)} detecciones en {len(results)} imágenes -> {out}")


if __name__ == "__main__":
    main()
