"""Modelo de importancia vial con etiquetas REALES: puntajes de priorización de la UMV.

Para las 93 680 calzadas de la UMV se predicen sus dimensiones social, económica y técnica (y el
índice IP = suma) a partir de features abiertas e INDEPENDIENTES de las que usó la UMV
(proximidad a colegios, IPS/hospitales, siniestros, centro, geometría de la calzada). Se
comparan una red neuronal (MLP) y gradient boosting con validación cruzada espacial por bloques
geohash-5 (sin fuga espacial). El mejor modelo se reentrena con todo y puntúa los segmentos.
Salidas: data/features/importance_segment.parquet, data/features/importance_cv.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dimmit.geo import geohash as gh
from dimmit.geo.context import point_features
from dimmit.geo.sources import load_umv
from dimmit.utils.io import EXTERNAL, FEATURES, ensure_dir, load_yaml

TARGETS = ["dim_social", "dim_economica", "dim_tecnica", "ip_umv"]


def umv_features(umv, cfg):
    cache = EXTERNAL / "umv_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pts = umv[["lat", "lon"]].astype(float)
    f = point_features(pts, cfg)
    f["ancho_m"] = umv["ancho_m"].to_numpy()
    f["area_m2"] = umv["area_m2"].to_numpy()
    f["longitud_m"] = umv["longitud_umv_m"].to_numpy()
    f["tipo_superficie"] = umv["tipo_superficie"].fillna(-1).to_numpy()
    tree_q = np.radians(pts.to_numpy())
    from sklearn.neighbors import BallTree

    tree = BallTree(tree_q, metric="haversine")
    f["densidad_calzadas_300m"] = [len(i) for i in tree.query_radius(tree_q, r=300 / gh.EARTH_R)]
    f.to_parquet(cache, index=False)
    return f


def feature_cols(df):
    return [c for c in df.columns if c not in ("lat", "lon")]


def models():
    return {
        "red_neuronal_mlp": make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=42)
        ),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42),
    }


def main():
    cfg = load_yaml("configs/context.yaml")
    umv = load_umv(cfg)
    X = umv_features(umv, cfg)
    X = X.assign(log_area=np.log1p(X["area_m2"]))
    cols = feature_cols(X)
    groups = gh.encode_many(umv["lat"], umv["lon"], 5)
    cv = GroupKFold(n_splits=5)
    report = {}
    for target in TARGETS:
        y = umv[target].to_numpy(dtype=float)
        report[target] = {}
        for name, model in models().items():
            pred = np.zeros_like(y)
            for tr, te in cv.split(X, y, groups):
                m = model.fit(X.iloc[tr][cols], y[tr])
                pred[te] = m.predict(X.iloc[te][cols])
            report[target][name] = {"r2": float(r2_score(y, pred)), "spearman": float(spearmanr(y, pred).statistic)}
        print(target, report[target])
    best = {t: max(report[t], key=lambda n: report[t][n]["spearman"]) for t in TARGETS}
    # cada segmento ES una calzada UMV: se reutiliza su fila de features
    seg = pd.read_parquet(FEATURES / "context_segment.parquet")
    row_of = pd.Series(np.arange(len(umv)), index=umv["pk_calzada"].to_numpy())
    Xs = X.iloc[row_of.loc[seg["pk_calzada"].to_numpy()].to_numpy()].reset_index(drop=True)
    out = seg[["segment_id"]].copy()
    for t in TARGETS:
        m = models()[best[t]].fit(X[cols], umv[t].to_numpy(dtype=float))
        out[f"{t}_predicho"] = m.predict(Xs[cols])
        out[f"{t}_oficial"] = seg[t].to_numpy()
    ensure_dir(FEATURES)
    out.to_parquet(FEATURES / "importance_segment.parquet", index=False)
    (FEATURES / "importance_cv.json").write_text(json.dumps({"cv_espacial_geohash5": report, "mejor_modelo": best}, indent=2))
    print(f"[ok] importancia de {len(out)} segmentos; mejor modelo por dimensión: {best}")


if __name__ == "__main__":
    main()
