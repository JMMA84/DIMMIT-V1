"""Features de contexto por segmento (Bogotá): geohash, clima, elevación/pendiente, colegios,
salud, siniestralidad y atributos oficiales de la calzada (UMV/IDU).

El contexto es real pero la imagen asignada no es de esa calle, así que estas columnas NO entran
al modelo de condición; alimentan el modelo de importancia (etiquetas reales UMV), la prioridad
y las descripciones. Salida: data/features/context_segment.parquet
"""
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from dimmit.geo import geohash as gh
from dimmit.geo.roads import assign_carriageways
from dimmit.geo.sources import fetch_json, load_crashes, load_health, load_idu_state, load_schools
from dimmit.utils.io import DATA, EXTERNAL, FEATURES, INDEX, ensure_dir, load_yaml

BOGOTA_CENTER = (4.6097, -74.0817)  # Plaza de Bolívar


def _tree(lat, lon):
    return BallTree(np.radians(np.c_[np.asarray(lat, float), np.asarray(lon, float)]), metric="haversine")


def proximity(points, ref, radii, prefix, weights=None):
    """Distancia al más cercano y conteos (o suma de pesos) dentro de cada radio (m)."""
    tree = _tree(ref["lat"], ref["lon"])
    q = np.radians(points[["lat", "lon"]].to_numpy(dtype=float))
    d, _ = tree.query(q, k=1)
    out = pd.DataFrame(index=points.index)
    out[f"dist_{prefix}_m"] = d[:, 0] * gh.EARTH_R
    for r in radii:
        idx = tree.query_radius(q, r=r / gh.EARTH_R)
        if weights is None:
            out[f"{prefix}_{r}m"] = [len(i) for i in idx]
        else:
            w = np.asarray(weights)
            out[f"{prefix}_{r}m"] = [float(w[i].sum()) for i in idx]
    return out


def point_features(points, cfg):
    """Features de proximidad para un DataFrame con lat/lon (segmentos o calzadas UMV)."""
    schools, health, crashes = load_schools(cfg), load_health(cfg), load_crashes(cfg)
    r = cfg["radii_m"]
    recent = crashes[crashes["anio"] >= crashes["anio"].max() - cfg["crash_years"] + 1]
    parts = [
        proximity(points, schools, r["colegios"], "colegio"),
        proximity(points, health, r["salud"], "ips"),
        proximity(points, health[health["hospital"]], [1000], "hospital"),
        proximity(points, recent, r["siniestros"], "siniestros"),
        proximity(points, recent, r["siniestros"], "siniestros_pond", weights=recent["peso"]),
    ]
    out = pd.concat(parts, axis=1).drop(columns=["dist_siniestros_pond_m"])
    out["dist_centro_m"] = gh.haversine_m(points["lat"], points["lon"], *BOGOTA_CENTER)
    return out


def weather(cells_dates, cfg):
    """Clima diario por celda geohash (Open-Meteo, reanálisis) -> agregados por fecha."""
    cache = ensure_dir(EXTERNAL / "meteo")
    start = (pd.Timestamp(cfg["survey_start"]) - pd.Timedelta(days=cfg["weather_history_days"])).date()
    end = pd.Timestamp(cfg["survey_end"]).date()
    rows = []
    for cell in sorted(cells_dates["cell"].unique()):
        f = cache / f"{cell}_{start}_{end}.json"
        if f.exists():
            d = pd.read_json(f)
        else:
            lat, lon = gh.decode(cell)
            url = (
                f"{cfg['urls']['meteo_archive']}?latitude={lat:.4f}&longitude={lon:.4f}&start_date={start}&end_date={end}"
                "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min&timezone=America%2FBogota"
            )
            d = pd.DataFrame(fetch_json(url)["daily"])
            d.to_json(f)
            time.sleep(0.3)
        d["time"] = pd.to_datetime(d["time"])
        d = d.set_index("time").sort_index()
        p = d["precipitation_sum"].fillna(0)
        agg = pd.DataFrame(
            {
                "lluvia_1d_mm": p,
                "lluvia_7d_mm": p.rolling(7, min_periods=1).sum(),
                "lluvia_30d_mm": p.rolling(30, min_periods=1).sum(),
                "lluvia_365d_mm": p.rolling(365, min_periods=1).sum(),
                "dias_lluvia_30d": (p >= 1).rolling(30, min_periods=1).sum(),
                "temp_max_7d": d["temperature_2m_max"].rolling(7, min_periods=1).mean(),
                "temp_min_7d": d["temperature_2m_min"].rolling(7, min_periods=1).mean(),
            }
        )
        agg["cell"] = cell
        rows.append(agg.reset_index().rename(columns={"time": "fecha"}))
    met = pd.concat(rows)
    q = cells_dates.assign(fecha=pd.to_datetime(cells_dates["fecha"]).dt.normalize())
    return q.merge(met, on=["cell", "fecha"], how="left").drop(columns=["cell", "fecha"])


def elevation(lat, lon, cfg):
    cache = EXTERNAL / "elevacion.parquet"
    pts = pd.DataFrame({"lat": np.round(lat, 5), "lon": np.round(lon, 5)})
    known = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["lat", "lon", "elev_m"])
    todo = pts.merge(known, on=["lat", "lon"], how="left")
    missing = todo[todo["elev_m"].isna()][["lat", "lon"]].drop_duplicates()
    new = []
    for s in range(0, len(missing), 100):
        chunk = missing.iloc[s : s + 100]
        url = f"{cfg['urls']['elevation']}?latitude={','.join(map(str, chunk['lat']))}&longitude={','.join(map(str, chunk['lon']))}"
        try:
            new.append(chunk.assign(elev_m=fetch_json(url, retries=3, timeout=30)["elevation"]))
        except RuntimeError as e:  # servicio limitado: se deja NaN y se completa en otra corrida
            print(f"[warn] elevación no disponible ({e}); pendiente queda NaN para {len(missing) - s} puntos")
            break
        time.sleep(0.3)
    if new:
        known = pd.concat([known, *new], ignore_index=True)
        known.to_parquet(cache, index=False)
    return pts.merge(known, on=["lat", "lon"], how="left")["elev_m"].to_numpy(dtype=float)


def main():
    cfg = load_yaml("configs/context.yaml")
    splits = pd.read_csv(INDEX / "splits.csv")
    roads = assign_carriageways(splits["segment_id"])
    seg = roads.copy()
    seg["lat"], seg["lon"] = (seg["lat_a"] + seg["lat_b"]) / 2, (seg["lon_a"] + seg["lon_b"]) / 2
    for p in (5, 6, 7):
        seg[f"geohash{p}"] = gh.encode_many(seg["lat"], seg["lon"], p)
    # fecha de captura: la del sensor si existe; si no, la misma regla de la simulación
    fr = DATA / "simulated" / "sensor_frames.parquet"
    from dimmit.data.simulate_sensors import survey_start

    sensor_cfg = load_yaml("configs/sensors.yaml")
    dates = pd.read_parquet(fr).groupby("segment_id")["timestamp_captura"].min() if fr.exists() else pd.Series(dtype="datetime64[ns]")
    seg["fecha_captura"] = [dates.get(s, survey_start(s, cfg, sensor_cfg["seed"])) for s in seg["segment_id"]]
    seg = pd.concat([seg, point_features(seg[["lat", "lon"]], cfg)], axis=1)
    cells = seg[[f"geohash{cfg['weather_geohash_precision']}"]].rename(columns=lambda _: "cell").assign(fecha=seg["fecha_captura"])
    seg = pd.concat([seg, weather(cells, cfg).set_index(seg.index)], axis=1)
    ea = elevation(seg["lat_a"], seg["lon_a"], cfg)
    eb = elevation(seg["lat_b"], seg["lon_b"], cfg)
    seg["elevacion_m"] = (ea + eb) / 2
    seg["pendiente_pct"] = 100 * np.abs(eb - ea) / np.maximum(seg["axis_m"], 1)
    try:
        idu = load_idu_state(cfg)[["calcodigo", "estado_superficial"]].drop_duplicates("calcodigo")
        seg = seg.merge(idu.rename(columns={"calcodigo": "pk_calzada", "estado_superficial": "estado_idu_codigo"}), on="pk_calzada", how="left")
    except Exception as e:  # noqa: BLE001 - fuente informativa
        print(f"[warn] sin estado IDU: {e}")
    ensure_dir(FEATURES)
    seg.to_parquet(FEATURES / "context_segment.parquet", index=False)
    print(f"[ok] contexto de {len(seg)} segmentos, {seg.shape[1]} columnas -> {FEATURES / 'context_segment.parquet'}")
    print(seg[["dist_colegio_m", "colegio_300m", "dist_hospital_m", "siniestros_250m", "lluvia_30d_mm", "pendiente_pct"]].describe().round(1))


if __name__ == "__main__":
    main()
