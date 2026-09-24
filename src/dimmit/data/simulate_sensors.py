"""Genera datos dummy de los sensores descritos en el Reporte Técnico V1.

Agrupa las imágenes procesadas en "segmentos de vía" y simula, por segmento,
series de acelerómetro triaxial, giroscopio y GPS a 50 Hz durante 10 s.
La vibración vertical se correlaciona con la severidad del daño anotado en las
imágenes del segmento, para que el modelo de scores tenga señal realista.

Salidas:
  data/simulated/sensors.parquet   series de tiempo por segmento
  data/simulated/segments.csv      mapeo imagen -> segmento + severidad anotada
"""
import numpy as np
import pandas as pd

from dimmit.utils.io import REPO_ROOT, load_yaml

PROCESSED = REPO_ROOT / "data/processed"
OUT = REPO_ROOT / "data/simulated"

HZ = 50
SECONDS = 10
IMAGES_PER_SEGMENT = 10
G = 9.81
BASE_LAT, BASE_LON = 4.6482, -74.0648  # Bogotá como origen de las rutas simuladas


def image_severity(label_path, class_weights):
    """Severidad anotada de una imagen: suma de w_clase * (1 + 10*área_bbox)."""
    sev = 0.0
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, w, h = int(parts[0]), float(parts[3]), float(parts[4])
        sev += class_weights[cls] * (1 + 10 * w * h)
    return sev


def simulate_segment(rng, segment_id, severity_norm, t0):
    """Serie de 10 s a 50 Hz para un segmento; más daño => más vibración."""
    n = HZ * SECONDS
    t = np.arange(n) / HZ
    noise = 0.15 + 1.2 * severity_norm
    az = G + rng.normal(0, noise, n)
    # baches: picos esporádicos proporcionales a la severidad
    for _ in range(rng.poisson(6 * severity_norm)):
        az[rng.integers(0, n)] += rng.uniform(3, 8) * severity_norm
    speed = np.clip(rng.normal(40, 8), 15, 80)  # km/h aprox constante en el segmento
    dist_deg = (speed / 3600 * t) / 111  # grados aprox recorridos
    heading = rng.uniform(0, 2 * np.pi)
    return pd.DataFrame(
        {
            "segment_id": segment_id,
            "timestamp": pd.Timestamp("2025-09-23T08:00:00") + pd.to_timedelta(t0 + t, unit="s"),
            "ax": rng.normal(0, 0.3 * (1 + severity_norm), n),
            "ay": rng.normal(0, 0.3 * (1 + severity_norm), n),
            "az": az,
            "gx": rng.normal(0, 0.05 * (1 + 2 * severity_norm), n),
            "gy": rng.normal(0, 0.05 * (1 + 2 * severity_norm), n),
            "gz": rng.normal(0, 0.02, n),
            "lat": BASE_LAT + dist_deg * np.cos(heading),
            "lon": BASE_LON + dist_deg * np.sin(heading),
            "speed_kmh": speed + rng.normal(0, 1.5, n),
        }
    )


def main():
    score_cfg = load_yaml("configs/score.yaml")
    weights = list(score_cfg["class_weights"].values())  # indexados por clase 0-3

    labels = sorted((PROCESSED / "labels").rglob("*.txt"))
    if not labels:
        raise SystemExit("No hay labels en data/processed; corre `make prepare-data` primero")

    rows = []
    for i, lbl in enumerate(labels):
        rows.append(
            {
                "image": lbl.stem + ".jpg",
                "segment_id": f"SEG-{i // IMAGES_PER_SEGMENT:04d}",
                "severity": image_severity(lbl, weights),
            }
        )
    segments = pd.DataFrame(rows)
    seg_sev = segments.groupby("segment_id")["severity"].mean()
    sev_norm = (seg_sev / seg_sev.max()).fillna(0)

    rng = np.random.default_rng(42)
    series = [
        simulate_segment(rng, seg_id, sev_norm[seg_id], t0=k * SECONDS)
        for k, seg_id in enumerate(seg_sev.index)
    ]
    sensors = pd.concat(series, ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    sensors.to_parquet(OUT / "sensors.parquet", index=False)
    segments.to_csv(OUT / "segments.csv", index=False)
    print(f"[ok] {len(seg_sev)} segmentos, {len(sensors)} muestras -> {OUT}")


if __name__ == "__main__":
    main()
