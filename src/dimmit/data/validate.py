"""Valida las salidas del pipeline de datos. Sale con código != 0 si algo falla."""
import sys

import pandas as pd

from dimmit.utils.io import REPO_ROOT

PROCESSED = REPO_ROOT / "data/processed"
SIMULATED = REPO_ROOT / "data/simulated"

SENSOR_COLS = ["segment_id", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed_kmh"]


def check(ok, msg, errors):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        errors.append(msg)


def main():
    errors = []

    print("Imágenes y labels (formato YOLO):")
    for split in ("train", "val"):
        imgs = {p.stem for p in (PROCESSED / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (PROCESSED / "labels" / split).glob("*.txt")}
        check(len(imgs) > 0, f"{split}: {len(imgs)} imágenes", errors)
        check(imgs == lbls, f"{split}: imágenes y labels emparejados 1 a 1", errors)
        bad = 0
        for lbl in (PROCESSED / "labels" / split).glob("*.txt"):
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    bad += 1
                    continue
                cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
                if cls not in range(4) or any(not 0 <= v <= 1 for v in vals):
                    bad += 1
        check(bad == 0, f"{split}: bboxes normalizados y clases 0-3 ({bad} inválidos)", errors)
    check((PROCESSED / "data.yaml").exists(), "data.yaml generado", errors)

    print("Sensores simulados:")
    sensors_path = SIMULATED / "sensors.parquet"
    segments_path = SIMULATED / "segments.csv"
    check(sensors_path.exists(), "sensors.parquet existe", errors)
    check(segments_path.exists(), "segments.csv existe", errors)
    if sensors_path.exists() and segments_path.exists():
        sensors = pd.read_parquet(sensors_path)
        segments = pd.read_csv(segments_path)
        check(list(sensors.columns) == SENSOR_COLS, "esquema de sensores correcto", errors)
        check(int(sensors.isna().sum().sum()) == 0, "sensores sin nulos", errors)
        check((sensors["az"] > 0).all(), "aceleración vertical plausible (>0)", errors)
        covered = set(segments["segment_id"]) == set(sensors["segment_id"])
        check(covered, "todos los segmentos tienen series de sensores", errors)
        n_imgs = sum(1 for _ in (PROCESSED / "images").rglob("*.jpg"))
        check(len(segments) == n_imgs, f"cada imagen mapeada a un segmento ({len(segments)}/{n_imgs})", errors)

    if errors:
        print(f"\nValidación FALLÓ: {len(errors)} problema(s)")
        sys.exit(1)
    print("\nValidación OK")


if __name__ == "__main__":
    main()
