"""Lectura de trazas GPX (BasicAirData GPS Logger) e interpolación de posición por timestamp."""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from dimmit.geo.geohash import haversine_m


def read_gpx(path) -> pd.DataFrame:
    """Fixes del track: timestamp (UTC), lat, lon, ele, speed (m/s), sat."""
    root = ET.parse(path).getroot()
    ns = {"g": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"g:{t}") if ns else (lambda t: t)
    rows = []
    for pt in root.iter(f"{{{ns['g']}}}trkpt" if ns else "trkpt"):
        get = lambda tag: (pt.find(q(tag), ns).text if pt.find(q(tag), ns) is not None else None)
        rows.append({
            "timestamp": pd.Timestamp(get("time")),
            "lat": float(pt.get("lat")),
            "lon": float(pt.get("lon")),
            "ele": float(get("ele")) if get("ele") else np.nan,
            "speed": float(get("speed")) if get("speed") else np.nan,
            "sat": int(get("sat")) if get("sat") else -1,
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def smooth_track(track: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Mediana móvil centrada (ventana en segundos) sobre lat/lon: quita el jitter del teléfono."""
    t = track.set_index("timestamp")
    win = f"{int(round(window_s))}s"
    sm = t[["lat", "lon"]].rolling(win, center=True, min_periods=1).median()
    out = track.copy()
    out["lat_s"], out["lon_s"] = sm["lat"].to_numpy(), sm["lon"].to_numpy()
    d = np.r_[0.0, haversine_m(out["lat_s"].to_numpy()[:-1], out["lon_s"].to_numpy()[:-1], out["lat_s"].to_numpy()[1:], out["lon_s"].to_numpy()[1:])]
    out["dist_acum_m"] = np.cumsum(d)
    return out


def interpolate(track_s: pd.DataFrame, timestamps) -> pd.DataFrame:
    """Posición suavizada, distancia acumulada, satélites y velocidad en cada timestamp dado."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    epoch = pd.Timestamp(0, tz="UTC")  # segundos desde época, sin depender de la resolución (s/ms/ns) de la serie
    x = (track_s["timestamp"] - epoch).dt.total_seconds().to_numpy()
    xq = (ts - epoch).dt.total_seconds().to_numpy()
    out = pd.DataFrame({"timestamp": ts.to_numpy()})
    for col in ("lat_s", "lon_s", "dist_acum_m", "speed"):
        out[col.replace("_s", "")] = np.interp(xq, x, track_s[col].to_numpy())
    nearest = np.abs(xq[:, None] - x[None, :]).argmin(axis=1)
    out["sat"] = track_s["sat"].to_numpy()[nearest]
    out["gps_dt_s"] = np.abs(xq - x[nearest])
    out["dentro_de_traza"] = (xq >= x.min()) & (xq <= x.max())
    return out
