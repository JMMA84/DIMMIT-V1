"""Valida las salidas del pipeline de datos v1. Sale con código != 0 si algo falla."""
import sys

import pandas as pd

from dimmit.utils.io import DATA, INDEX, load_yaml

SENSOR_COLS = ["segment_id", "obs_id", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed_kmh"]
EXPECTED = {"train": 21041, "test1": 2631, "test2": 2664}


def check(ok, msg, errors):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        errors.append(msg)


def main():
    errors = []
    cfg = load_yaml("configs/data_rdd2020.yaml")
    print("Índice RDD2020:")
    images = pd.read_parquet(INDEX / "images.parquet")
    for src, n in EXPECTED.items():
        got = int((images["source"] == src).sum())
        check(got == n, f"{src}: {got} imágenes (esperadas {n})", errors)
    boxes = pd.read_parquet(INDEX / "boxes.parquet")
    bad = ((boxes[["cx", "cy", "w", "h"]] < 0) | (boxes[["cx", "cy", "w", "h"]] > 1)).any(axis=1).sum()
    check(bad == 0, f"cajas normalizadas en [0,1] ({bad} inválidas)", errors)
    check(boxes["cls"].isin(cfg["voc_classes"]).all(), "clases de cajas en D00/D10/D20/D40", errors)
    lbl_missing = sum(not (p.parent.parent / "labels" / (p.stem + ".txt")).exists() for p in map(lambda s: __import__("pathlib").Path(s), images.loc[images["labeled"], "path"]))
    check(lbl_missing == 0, f"cada imagen anotada tiene etiqueta YOLO ({lbl_missing} faltantes)", errors)

    print("Splits:")
    s = pd.read_csv(INDEX / "splits.csv")
    lab = s[s["split"] != "despliegue"]
    check(s["image_id"].is_unique and len(s) == len(images), "cada imagen en exactamente un split", errors)
    check((s.groupby("segment_id")["split"].nunique() == 1).all(), "ningún segmento cruza splits", errors)
    check((lab.groupby("group_id")["split"].nunique() == 1).all(), "ningún bloque cruza splits", errors)
    check((lab.groupby("dup_cluster")["split"].nunique() == 1).all(), "ningún clúster de casi-duplicados cruza splits", errors)

    print("Sensores simulados:")
    path = DATA / "simulated" / "sensors.parquet"
    if path.exists():
        sensors = pd.read_parquet(path)
        check(list(sensors.columns) == SENSOR_COLS, "esquema de sensores v1", errors)
        check(int(sensors.drop(columns=["obs_id"]).isna().sum().sum()) == 0, "sensores sin nulos", errors)
        check((sensors["az"].abs() <= 2 * 9.81 + 1e-6).all(), "acelerómetro dentro del rango ±2 g", errors)
        check(sensors["lat"].between(4.4, 4.9).all() and sensors["lon"].between(-74.3, -73.9).all(), "GPS dentro de Bogotá", errors)
        segs = set(s.loc[s["split"].isin(["fusion", "test", "despliegue"]), "segment_id"])
        check(segs <= set(sensors["segment_id"]), "todos los segmentos de fusión/test/despliegue tienen sensores", errors)
    else:
        check(False, "sensors.parquet existe", errors)

    if errors:
        print(f"\nValidación FALLÓ: {len(errors)} problema(s)")
        sys.exit(1)
    print("\nValidación OK")


if __name__ == "__main__":
    main()
