"""Asignación de segmentos a calzadas reales de Bogotá (UMV) para la ruta simulada.

Cada segmento de fotogramas se ubica sobre una calzada distinta, recorrida a lo largo de su eje
largo (del rectángulo mínimo). La asignación es aleatoria y con semilla: la imagen NO es de esa
calle (RDD2020 es de Japón/India/Chequia), por eso el contexto no entra al modelo de condición.
"""
import numpy as np
import pandas as pd

from dimmit.geo.geohash import haversine_m
from dimmit.geo.sources import load_umv
from dimmit.utils.io import INDEX, load_yaml

MAX_SEGMENT_M = 150.0


def assign_carriageways(segments: pd.Series, cfg=None) -> pd.DataFrame:
    cfg = cfg or load_yaml("configs/context.yaml")
    out = INDEX / "segment_roads.parquet"
    if out.exists():
        cached = pd.read_parquet(out)
        if set(segments) <= set(cached["segment_id"]):
            return cached
    umv = load_umv(cfg)
    umv = umv.assign(axis_m=haversine_m(umv["lat_a"], umv["lon_a"], umv["lat_b"], umv["lon_b"]))
    pool = umv[(umv["axis_m"] >= cfg["min_carriageway_length_m"]) & umv["civ"].notna()]
    seg_ids = sorted(set(segments))
    rng = np.random.default_rng(cfg["seed"])
    pick = pool.iloc[rng.choice(len(pool), size=len(seg_ids), replace=False)].reset_index(drop=True)
    # sentido de recorrido aleatorio
    flip = rng.random(len(pick)) < 0.5
    for a, b in (("lat_a", "lat_b"), ("lon_a", "lon_b")):
        pick.loc[flip, [a, b]] = pick.loc[flip, [b, a]].to_numpy()
    pick["segment_id"] = seg_ids
    pick["longitud_m"] = np.minimum(pick["axis_m"], MAX_SEGMENT_M)
    pick.to_parquet(out, index=False)
    return pick
