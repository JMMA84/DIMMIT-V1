"""Entrena YOLOv10 (paquete oficial ultralytics, PyTorch) sobre RDD2020.

- Reanuda desde last.pt si existe una corrida sin marcador DONE.
- max_minutes (opcional, CPU): sale con código 75 al final de la época en que ya no cabe otra
  dentro del presupuesto, sin finalizar (así last.pt conserva el estado para reanudar).
"""
import argparse
import os
import time
import urllib.request
from pathlib import Path

from dimmit.utils.io import REPO_ROOT, load_yaml

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"


def ensure_weights(name: str) -> Path:
    dest = REPO_ROOT / "models" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[down] pesos preentrenados {name}...")
        urllib.request.urlretrieve(WEIGHTS_URL.format(name=name), dest)
    return dest


def pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_kaggle.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    run_dir = REPO_ROOT / cfg["project"] / cfg["name"]
    last = run_dir / "weights" / "last.pt"
    done = run_dir / "DONE"
    if done.exists():
        print(f"[skip] {run_dir} ya terminó")
        return
    resume = last.exists()
    model = YOLO(str(last if resume else ensure_weights(cfg["model"])))

    t0 = time.time()
    max_min = cfg.get("max_minutes")

    def on_fit_epoch_end(trainer):
        if not max_min:
            return
        elapsed = (time.time() - t0) / 60
        epoch_min = getattr(trainer, "epoch_time", 0) / 60
        if elapsed + 1.15 * epoch_min > max_min and last.exists() and last.stat().st_mtime > t0:
            print(f"[chunk] {elapsed:.0f} min: presupuesto agotado, reanudar con el mismo comando", flush=True)
            os._exit(75)

    def on_train_end(trainer):
        done.write_text("ok")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    if resume:
        print(f"[resume] {last}")
        model.train(resume=True)
        return
    keys = ["epochs", "imgsz", "batch", "workers", "patience", "cache", "cos_lr", "close_mosaic", "fraction", "seed", "time", "val", "amp"]
    extra = {k: cfg[k] for k in keys if k in cfg and cfg[k] is not None}
    model.train(
        data=str(REPO_ROOT / cfg["data"]),
        device=pick_device(),
        project=str(REPO_ROOT / cfg["project"]),
        name=cfg["name"],
        exist_ok=True,
        plots=False,
        deterministic=True,
        fliplr=0.5,
        flipud=0.0,
        **extra,
    )
    print(f"[ok] pesos en {run_dir / 'weights'}")


if __name__ == "__main__":
    main()
