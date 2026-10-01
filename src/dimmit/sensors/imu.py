"""Features de IMU/GPS por fotograma (ventana alrededor de la captura) y por segmento.

No depende de la orientación del sensor: el eje vertical se estima con el vector de gravedad
(media del acelerómetro) y la aceleración vertical se filtra en 0.5-20 Hz. Solo lee
sensors.parquet y sensor_frames.parquet (nunca sensors_truth.parquet).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from dimmit.geo.geohash import haversine_m
from dimmit.utils.io import DATA, FEATURES, ensure_dir, load_yaml

FS = 50
BANDS = [(0.5, 2), (2, 5), (5, 9), (9, 15), (15, 25)]
SOS = butter(2, [0.5, 20], btype="band", fs=FS, output="sos")
SOS_G = butter(2, [0.2, 15], btype="band", fs=FS, output="sos")


def vertical_accel(acc):
    """Proyección de la aceleración sobre la gravedad estimada, filtrada 0.5-20 Hz."""
    g = acc.mean(axis=0)
    gn = g / np.linalg.norm(g)
    av = acc @ gn
    return sosfiltfilt(SOS, av - av.mean()) if len(av) > 15 else av - av.mean(), gn


def window_features(w, gn=None):
    """Features de una ventana (DataFrame con columnas del parquet de sensores)."""
    acc = w[["ax", "ay", "az"]].to_numpy()
    gyr = w[["gx", "gy", "gz"]].to_numpy()
    if len(w) < 20:
        return {}
    av, gvec = vertical_accel(acc)
    gn = gvec if gn is None else gn
    v = np.maximum(w["speed_kmh"].to_numpy() / 3.6, 0.5)
    vm = float(v.mean())
    rms = float(np.sqrt(np.mean(av**2)))
    f = {
        "imu_rms": rms,
        "imu_std": float(av.std()),
        "imu_p2p": float(av.max() - av.min()),
        "imu_max": float(np.abs(av).max()),
        "imu_p95": float(np.percentile(np.abs(av), 95)),
        "imu_p99": float(np.percentile(np.abs(av), 99)),
        "imu_kurtosis": float(kurtosis(av)),
        "imu_skew": float(skew(av)),
        "imu_crest": float(np.abs(av).max() / max(rms, 1e-6)),
        "imu_rms_sqrtv": rms / np.sqrt(vm),
        "imu_energia_m": float(np.sum(av**2) / FS / max(vm * len(av) / FS, 1e-3)),
        "imu_jerk_rms": float(np.sqrt(np.mean((np.diff(av) * FS) ** 2))),
        "imu_choques_2": int((np.abs(av) > 2).sum()),
        "imu_choques_4": int((np.abs(av) > 4).sum()),
        "imu_max_sqrtv": float(np.abs(av).max() / np.sqrt(vm)),
    }
    fr, p = welch(av, fs=FS, nperseg=min(128, len(av)))
    tot = np.trapezoid(p, fr) + 1e-12
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        f[f"imu_banda_{lo:g}_{hi:g}"] = float(np.trapezoid(p[m], fr[m]) / tot) if m.sum() > 1 else 0.0
    f["imu_centroide_hz"] = float((fr * p).sum() / (p.sum() + 1e-12))
    f["imu_frec_dominante"] = float(fr[np.argmax(p)])
    # giróscopo: componentes perpendiculares a la gravedad ~ alabeo/cabeceo; paralela ~ guiñada
    gz_v = gyr @ gn
    g_perp = gyr - np.outer(gz_v, gn)
    gp = sosfiltfilt(SOS_G, g_perp, axis=0) if len(gyr) > 15 else g_perp
    f["gyro_tilt_rms"] = float(np.sqrt(np.mean(np.sum(gp**2, axis=1))))
    f["gyro_yaw_rms"] = float(np.sqrt(np.mean(gz_v**2)))
    # velocidad y GPS
    f["vel_media_kmh"] = vm * 3.6
    f["vel_std_kmh"] = float(w["speed_kmh"].std())
    f["vel_min_kmh"] = float(w["speed_kmh"].min())
    f["frac_detenido"] = float((w["speed_kmh"] < 3).mean())
    lat, lon = w["lat"].to_numpy(), w["lon"].to_numpy()
    f["gps_distancia_m"] = float(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
    f["gps_jitter_m"] = float(np.std(haversine_m(lat, lon, lat.mean(), lon.mean())))
    # calidad de datos
    ts = w["timestamp"].astype("int64").to_numpy() / 1e9
    f["dq_muestras"] = len(w)
    f["dq_jitter_ms"] = float(np.std(np.diff(ts)) * 1000) if len(ts) > 2 else 0.0
    f["dq_saturacion"] = float((np.abs(acc) > 2 * 9.81 * 0.98).mean())
    return f


def build(sim_dir, cfg):
    sensors = pd.read_parquet(sim_dir / "sensors.parquet")
    frames = pd.read_parquet(sim_dir / "sensor_frames.parquet")
    w0, w1 = cfg["imu_window_s"]
    frame_rows, seg_rows = [], []
    for seg_id, s in sensors.groupby("segment_id", sort=False):
        s = s.sort_values("timestamp")
        sf = window_features(s)
        sf["segment_id"] = seg_id
        seg_rows.append(sf)
        t = s["timestamp"].to_numpy()
        for _, fr in frames[frames["segment_id"] == seg_id].iterrows():
            t0 = fr["timestamp_captura"]
            m = (t >= t0 + pd.Timedelta(seconds=w0)) & (t < t0 + pd.Timedelta(seconds=w1))
            ff = window_features(s[m])
            ff["image_id"] = fr["image_id"]
            frame_rows.append(ff)
    frame_df = pd.DataFrame(frame_rows)
    frame_df = frame_df.merge(frames[["image_id", "segment_id", "timestamp_captura", "lat", "lon"]], on="image_id", how="left")
    return frame_df, pd.DataFrame(seg_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--null-world", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml("configs/sensors.yaml")
    sim_dir = DATA / ("simulated_null" if args.null_world else "simulated")
    frame_df, seg_df = build(sim_dir, cfg)
    out = ensure_dir(FEATURES / ("null_world" if args.null_world else ""))
    frame_df.to_parquet(out / "imu_frame.parquet", index=False)
    seg_df.to_parquet(out / "imu_segment.parquet", index=False)
    print(f"[ok] IMU: {len(frame_df)} fotogramas, {len(seg_df)} segmentos -> {out}")


if __name__ == "__main__":
    main()
