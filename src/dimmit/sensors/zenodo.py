"""Validación con datos REALES del estimador de rugosidad (Zenodo 4386256, CC0).

Botshekan et al. (MIT/UMass): perfiles láser de 3 vías de Massachusetts (Mass Ave norte/sur y
Route 2 en Concord) y aceleraciones de 2 smartphones en carros, con GPS y velocidad.
  1. IRI de referencia (Golden Car) por tramo de 200 m desde el perfil láser.
  2. Cada viaje se ubica sobre la vía por GPS (punto de vía más cercano a <= 15 m, sentido
     coherente con el avance del índice).
  3. Por tramo se calculan las MISMAS features de IMU del pipeline (sensors/imu.py) y el
     estimador IRI_hat = k * rms/sqrt(v), con k calibrado dejando una vía afuera.
Es la única métrica de sensores sobre datos reales; el resto está condicionado al simulador.
Salida: data/features/zenodo_validacion.json
"""
import io
import json

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.stats import spearmanr
from sklearn.neighbors import BallTree

from dimmit.geo.geohash import EARTH_R, haversine_m
from dimmit.sensors.imu import window_features
from dimmit.sensors.iri import golden_car_slope
from dimmit.utils.io import EXTERNAL, FEATURES

Z = EXTERNAL / "zenodo"
SECTION_M = 200.0


def geopoints(name):
    """Extrae lat/lon de un objeto MATLAB geopoint (MCOS) guardado en el .mat."""
    d = sio.loadmat(Z / f"{name}.mat")
    ss = io.BytesIO(d["__function_workspace__"].tobytes())
    ss.seek(8)
    r = MatFile5Reader(ss, byte_order="<")
    r.mat_stream.seek(8)
    r.initialize_read()
    arrays = []
    try:
        while True:
            hdr, nxt = r.read_var_header()
            res = r.read_var_array(hdr)

            def walk(x):
                if isinstance(x, np.ndarray):
                    if x.dtype == object:
                        for e in x.ravel():
                            walk(e)
                    elif x.dtype.names:
                        for n in x.dtype.names:
                            for e in x[n].ravel():
                                walk(e)
                    elif x.dtype.kind == "f" and x.size > 100:
                        arrays.append(x.ravel())

            walk(res)
            r.mat_stream.seek(nxt)
    except Exception:  # noqa: BLE001 - fin del flujo MCOS
        pass
    lat = next(a for a in arrays if 40 < a.mean() < 45)
    lon = next(a for a in arrays if -75 < a.mean() < -70)
    return lat, lon


def laser_sections(dist_km, elev_cm):
    dx = float(np.median(np.diff(dist_km))) * 1000
    xs, s = golden_car_slope(np.asarray(elev_cm) / 100.0, dx)
    sec = (xs // SECTION_M).astype(int)
    df = pd.DataFrame({"sec": sec, "s": s, "x": xs})
    df = df[df["x"] >= 11]
    g = df.groupby("sec")
    out = 1000 * g["s"].mean()
    return out[g.size() * 0.25 >= 0.8 * SECTION_M]  # tramos completos


def tracks():
    nbsb = sio.loadmat(Z / "road_elevation_NB_SB.mat", squeeze_me=True)
    conc = sio.loadmat(Z / "road_elevation_concord.mat", squeeze_me=True)
    half = len(nbsb["dist"]) // 2
    t = {
        "mass_ave_norte": (geopoints("geopoint_Massachusetts_NB_INSIDE"), nbsb["dist"][:half], nbsb["elevation"][:half]),
        "mass_ave_sur": (geopoints("geopoint_Massachusetts_SB_INSIDE"), nbsb["dist"][half:] - nbsb["dist"][half], nbsb["elevation"][half:]),
        "concord_ruta2": (geopoints("geopoint_ConcordTurnpike_LongLoop_MIDDLE"), conc["dist"], conc["elevation"]),
    }
    out = {}
    for name, ((lat, lon), dist, elev) in t.items():
        length = (dist[-1] - dist[0]) * 1000
        out[name] = {"lat": lat, "lon": lon, "spacing": length / (len(lat) - 1), "iri": laser_sections(dist, elev)}
    return out


def trip_sections(path, trk):
    d = sio.loadmat(path, squeeze_me=True)
    df = pd.DataFrame({k: d[k] for k in ("t", "x", "y", "z", "speedMS", "LAT", "LON")})
    df = df.iloc[::2].reset_index(drop=True)  # 100 Hz -> 50 Hz (mismo muestreo del pipeline)
    q = np.radians(df[["LAT", "LON"]].to_numpy())
    rows = []
    for name, tr in trk.items():
        tree = BallTree(np.radians(np.c_[tr["lat"], tr["lon"]]), metric="haversine")
        dist, idx = tree.query(q, k=1)
        dm = dist[:, 0] * EARTH_R
        on = dm <= 15
        if on.mean() < 0.2:
            continue
        idx = idx[:, 0].astype(float)
        di = np.diff(idx[on])
        if np.median(di[di != 0]) <= 0 if (di != 0).any() else True:  # sentido contrario
            continue
        sec = np.where(on, (idx * tr["spacing"] // SECTION_M), -1).astype(int)
        w = pd.DataFrame(
            {
                "ax": df["x"], "ay": df["y"], "az": df["z"], "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "lat": df["LAT"], "lon": df["LON"], "speed_kmh": df["speedMS"] * 3.6,
                "timestamp": pd.to_datetime(df["t"], unit="s"),
            }
        )
        for s in np.unique(sec[sec >= 0]):
            m = (sec == s) & (df["speedMS"].to_numpy() > 3)
            if m.sum() < 50 * 5:
                continue
            f = window_features(w[m])
            rows.append({"via": name, "sec": int(s), "viaje": path.stem, **{k: f[k] for k in ("imu_rms_sqrtv", "imu_rms", "vel_media_kmh")}})
    return rows


def main():
    trk = tracks()
    rows = []
    for p in sorted(Z.glob("*-2019-*.mat")):
        rows.extend(trip_sections(p, trk))
    ph = pd.DataFrame(rows)
    agg = ph.groupby(["via", "sec"]).agg(rms_sqrtv=("imu_rms_sqrtv", "median"), n_pasadas=("viaje", "nunique")).reset_index()
    agg["iri_laser"] = [trk[v]["iri"].get(s, np.nan) for v, s in zip(agg["via"], agg["sec"])]
    agg = agg.dropna()
    # calibración k dejando una vía afuera (escala log)
    pred = np.zeros(len(agg))
    for v in agg["via"].unique():
        tr = agg["via"] != v
        k = np.exp(np.median(np.log(agg.loc[tr, "iri_laser"]) - np.log(agg.loc[tr, "rms_sqrtv"])))
        pred[~tr.to_numpy()] = k * agg.loc[~tr, "rms_sqrtv"]
    y = np.log(agg["iri_laser"])
    p = np.log(pred)
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    res = {
        "spearman": float(spearmanr(agg["iri_laser"], agg["rms_sqrtv"]).statistic),
        "r2": float(r2),
        "mae_iri_m_km": float(np.abs(agg["iri_laser"] - pred).mean()),
        "n": int(len(agg)),
        "por_via": {
            v: {
                "n_tramos": int((agg["via"] == v).sum()),
                "iri_laser_medio": float(agg.loc[agg["via"] == v, "iri_laser"].mean()),
                "spearman_dentro_de_via": float(spearmanr(agg.loc[agg["via"] == v, "iri_laser"], agg.loc[agg["via"] == v, "rms_sqrtv"]).statistic),
            }
            for v in agg["via"].unique()
        },
        "notas": "tramos de 200 m; R2 en log(IRI) con calibración dejando una vía afuera",
    }
    agg.to_csv(FEATURES / "zenodo_tramos.csv", index=False)
    (FEATURES / "zenodo_validacion.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
