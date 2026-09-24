"""Une detecciones del modelo de visión con los sensores por segmento de vía."""
import numpy as np
import pandas as pd

from dimmit.utils.io import REPO_ROOT

G = 9.81


def detection_features(detections: pd.DataFrame, segments: pd.DataFrame, class_names, conf_threshold):
    """Severidad visual por segmento: por clase, n_detecciones * (1 + 10*área media)."""
    det = detections[detections["conf"] >= conf_threshold].copy()
    det["area"] = det["w"] * det["h"]
    det = det.merge(segments[["image", "segment_id"]], on="image", how="inner")

    rows = []
    for seg_id, g in det.groupby("segment_id"):
        row = {"segment_id": seg_id}
        for cls_id, name in class_names.items():
            sub = g[g["class_id"] == cls_id]
            row[f"sev_{name}"] = len(sub) * (1 + 10 * (sub["area"].mean() if len(sub) else 0))
        rows.append(row)
    cols = ["segment_id"] + [f"sev_{n}" for n in class_names.values()]
    feats = pd.DataFrame(rows, columns=cols)
    # segmentos sin ninguna detección también cuentan (severidad 0)
    all_segs = segments[["segment_id"]].drop_duplicates()
    return all_segs.merge(feats, on="segment_id", how="left").fillna(0.0)


def sensor_features(sensors: pd.DataFrame):
    """Rugosidad por segmento: RMS de la aceleración vertical sin gravedad, normalizada 0-1."""
    rms = (
        sensors.groupby("segment_id")["az"]
        .apply(lambda s: float(np.sqrt(np.mean((s - G) ** 2))))
        .rename("roughness_rms")
        .reset_index()
    )
    max_rms = rms["roughness_rms"].max()
    rms["roughness_norm"] = rms["roughness_rms"] / max_rms if max_rms > 0 else 0.0
    return rms


def build_features(detections, segments, sensors, class_names, conf_threshold):
    vis = detection_features(detections, segments, class_names, conf_threshold)
    sen = sensor_features(sensors)
    return vis.merge(sen, on="segment_id", how="left").fillna(0.0)


def load_inputs():
    detections = pd.read_csv(REPO_ROOT / "reports/detections.csv")
    segments = pd.read_csv(REPO_ROOT / "data/simulated/segments.csv")
    sensors = pd.read_parquet(REPO_ROOT / "data/simulated/sensors.parquet")
    return detections, segments, sensors
