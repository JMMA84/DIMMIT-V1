"""Entrena YOLOv10 (fork THU-MIG, PyTorch) sobre el subset de RDD2020."""
import argparse
import urllib.request
from pathlib import Path

from dimmit.utils.io import REPO_ROOT, apply_compat_patches, load_yaml

WEIGHTS_URL = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/{name}"


def ensure_weights(name: str) -> Path:
    dest = REPO_ROOT / "models" / name
    if not dest.exists():
        print(f"[down] pesos preentrenados {name}...")
        urllib.request.urlretrieve(WEIGHTS_URL.format(name=name), dest)
    return dest


def pick_device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_small.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    apply_compat_patches()
    from ultralytics import YOLOv10

    model = YOLOv10(str(ensure_weights(cfg["model"])))
    model.train(
        data=str(REPO_ROOT / cfg["data"]),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        device=pick_device(),
        project=str(REPO_ROOT / cfg["project"]),
        name=cfg["name"],
        exist_ok=True,
        plots=False,
    )
    print(f"[ok] pesos en {REPO_ROOT / cfg['project'] / cfg['name'] / 'weights'}")


if __name__ == "__main__":
    main()
