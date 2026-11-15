"""Calibración de la cámara contra el estado superficial de calzada del IDU.

Problema: el detector (RDD2020) no vio Bogotá; en las fotos de campo su confianza es baja (mediana ~0,08) y el
umbral por clase elegido en RDD2020 (precisión >= 50 %) borra casi todo, así que el score queda "bueno" por
defecto. Aquí se ajusta UN parámetro: un multiplicador m de los umbrales de confianza (tau = m · tau_campo),
para que el score de los tramos con estado IDU conocido caiga en el rango PCI de ese estado
(IDU: bueno PCI >= 90, regular 56-89, malo < 56).

Tramos con etiqueta IDU:
  * con GPS (toma del 5-oct, solo cámara): calzada = polígono IDU que contiene la mayoría de sus fotos
    (o el más cercano a <= 15 m si el punto cae en el andén);
  * sin GPS: calzada declarada en configs/campo.yaml (`sitios`).
Solo cuentan calzadas con estado publicado (bueno, regular o malo) y superficie compatible.

Validación fuera de muestra: se calibra con una toma y se evalúa en la otra (otro día y otra cámara).
Salida: data/campo/features/calibracion_idu.json y calibracion_idu.csv
"""
import json

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from dimmit.campo import CAMPO, load_cfg
from dimmit.campo.scores import CALIB, FEAT, compute, prepare
from dimmit.utils.io import ensure_dir

IDU = CAMPO / "idu"
ESTADOS = ("bueno", "regular", "malo")


def rangos(cfg):
    r = cfg["calibracion"]["rangos_idu"]
    return {"bueno": (r["bueno"], 100.0), "regular": (r["regular"], r["bueno"]), "malo": (0.0, r["regular"])}


def estado_por_score(score, cfg):
    r = cfg["calibracion"]["rangos_idu"]
    return "bueno" if score >= r["bueno"] else ("regular" if score >= r["regular"] else "malo")


def calzadas_de_tramos(ctx, cfg):
    """id_via -> calcodigo IDU: GPS por mayoría de fotos dentro del polígono; sin GPS, la calzada declarada."""
    g = json.loads((IDU / "calzadas.geojson").read_text())
    polys = [shape(f["geometry"]) for f in g["features"]]
    codes = [f["properties"]["CALCODIGO"] for f in g["features"]]
    tree = STRtree(polys)
    fr = ctx["frames"]
    rows = []
    for vid, f in fr.groupby("id_via"):
        if f["ubicacion_fuente"].iloc[0] == "gps":
            hits = []
            for lat, lon in zip(f["lat"], f["lon"]):
                if not np.isfinite(lat):
                    continue
                pt = Point(lon, lat)
                inside = [codes[i] for i in tree.query(pt, predicate="within")]
                if inside:
                    hits.append(inside[0])
                    continue
                i = int(tree.nearest(pt))
                if polys[i].distance(pt) * 111_000 <= 15:  # andén o error de GPS: calzada más cercana
                    hits.append(codes[i])
            cod = pd.Series(hits).mode().iloc[0] if hits else None
            frac = (pd.Series(hits) == cod).mean() if hits else 0.0
        else:
            cod, frac = f["calcodigo_idu"].iloc[0], 1.0
        rows.append({"id_via": vid, "calcodigo": None if cod is None or pd.isna(cod) else int(cod), "frac_fotos_en_calzada": float(frac)})
    out = pd.DataFrame(rows)
    est = pd.read_csv(IDU / "calzadas_estado.csv")[["calcodigo", "estado_idu", "superficie", "civ", "fecha_idu"]]
    return out.merge(est, on="calcodigo", how="left")


def main():
    cfg = load_cfg()
    ctx = prepare(cfg)
    lab = calzadas_de_tramos(ctx, cfg)
    tr = ctx["tramos"].set_index("id_via")
    lab["prueba"], lab["fecha"], lab["conjunto"], lab["superficie_tramo"] = (lab["id_via"].map(tr[c]) for c in ("prueba", "fecha", "conjunto", "superficie"))
    compatible = lab["superficie"].isin(["flexible", "mixto", "rigido"]) == (lab["superficie_tramo"] == "asfalto")
    usable = lab[lab["estado_idu"].isin(ESTADOS) & compatible].copy()
    print("tramos con calzada IDU:")
    print(lab[["id_via", "conjunto", "prueba", "fecha", "calcodigo", "civ", "superficie", "estado_idu", "frac_fotos_en_calzada"]].to_string(index=False))
    if usable.empty:
        raise SystemExit("ningún tramo con estado IDU utilizable")

    mc = cfg["calibracion"]["multiplicador"]
    grid = np.round(np.arange(mc["min"], mc["max"] + 1e-9, mc["paso"]), 3)
    rg = rangos(cfg)
    base = cfg["filtros"]
    scores = {}
    for m in grid:
        f = {**base, "tau_campo": {k: v * m for k, v in base["tau_campo"].items()}}
        _, vias, _, _ = compute(ctx, cfg, f)
        scores[m] = vias[vias["nivel"] == "tramo"].set_index("id_via")["score_calidad"]
    S = pd.DataFrame(scores).loc[usable["id_via"]]  # tramos x m

    def dist(sc, est):
        """Puntos de score fuera del rango PCI del estado IDU (0 si cae dentro; el tope de bueno es 100 inclusive)."""
        lo, hi = rg[est]
        return max(lo - sc, 0.0) + (max(sc - hi, 0.0) if est != "bueno" else 0.0)

    def evaluate(ids, m):
        e = usable.set_index("id_via").loc[ids, "estado_idu"]
        sc = S.loc[ids, m]
        d = np.array([dist(s, x) for s, x in zip(sc, e)])
        acc = float(np.mean([estado_por_score(s, cfg) == x for s, x in zip(sc, e)]))
        return float(d.mean()), acc

    def fit(ids):
        res = [(m, *evaluate(ids, m)) for m in grid]
        best_loss = min(r[1] for r in res)
        cand = [r for r in res if r[1] <= best_loss + 1e-9]
        # empate (varios m dentro del rango): el que deja el score medio más cerca del centro del rango
        est = usable.set_index("id_via").loc[ids, "estado_idu"]
        centro = float(np.mean([np.mean(rg[x]) for x in est]))
        return min(cand, key=lambda r: abs(S.loc[ids, r[0]].mean() - centro))[0], res

    m_opt, curva = fit(list(usable["id_via"]))
    # validación cruzada por toma (fecha): calibrar en una, evaluar en la otra
    cv = []
    for fecha in sorted(usable["fecha"].unique()):
        test = list(usable.loc[usable["fecha"] == fecha, "id_via"])
        train = list(usable.loc[usable["fecha"] != fecha, "id_via"])
        if not train:
            continue
        m_tr, _ = fit(train)
        l_te, a_te = evaluate(test, m_tr)
        l_b, a_b = evaluate(test, 1.0) if 1.0 in S.columns else (np.nan, np.nan)
        cv.append({"evaluada": fecha, "calibrada_con": sorted(set(usable["fecha"]) - {fecha}), "multiplicador": float(m_tr),
                   "n_tramos": len(test), "acierto_estado_idu": a_te, "distancia_media_al_rango": l_te,
                   "acierto_sin_calibrar": a_b, "distancia_sin_calibrar": l_b})
    l_opt, a_opt = evaluate(list(usable["id_via"]), m_opt)
    l_1, a_1 = evaluate(list(usable["id_via"]), 1.0)
    res = {
        "multiplicador": float(m_opt),
        "tau_campo_rdd2020": base["tau_campo"],
        "tau_campo_calibrado": {k: round(v * m_opt, 4) for k, v in base["tau_campo"].items()},
        "rangos_idu_pci": {k: list(v) for k, v in rg.items()},
        "n_tramos": int(len(usable)), "n_calzadas": int(usable["calcodigo"].nunique()),
        "calzadas": usable.groupby("calcodigo").agg(estado_idu=("estado_idu", "first"), civ=("civ", "first"), superficie=("superficie", "first"),
                                                    n_tramos=("id_via", "size"), fecha_idu=("fecha_idu", "first")).reset_index().to_dict("records"),
        "ajuste": {"acierto_estado_idu": a_opt, "distancia_media_al_rango": l_opt, "acierto_sin_calibrar": a_1, "distancia_sin_calibrar": l_1},
        "validacion_por_toma": cv,
        "limitacion": ("Todos los tramos etiquetados caen en calzadas con el mismo estado IDU, así que el ajuste ancla el "
                       "umbral a ese estado pero no prueba que el score separe bueno/regular/malo. Para una calibración "
                       "completa hacen falta 20-30 calzadas con estados distintos y fotos con GPS.") if usable["estado_idu"].nunique() == 1 else "",
    }
    ensure_dir(FEAT)
    CALIB.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    out = usable.copy()
    out["score_sin_calibrar"] = S[1.0].reindex(out["id_via"]).to_numpy() if 1.0 in S.columns else np.nan
    out["score_calibrado"] = S[m_opt].reindex(out["id_via"]).to_numpy()
    out["estado_dimmit_calibrado"] = [estado_por_score(s, cfg) for s in out["score_calibrado"]]
    out.to_csv(FEAT / "calibracion_idu.csv", index=False)
    pd.DataFrame([{"multiplicador": m, "distancia_media_al_rango": l, "acierto_estado_idu": a, "score_medio": float(S[m].mean())} for m, l, a in curva]).to_csv(
        FEAT / "calibracion_idu_curva.csv", index=False)
    print(f"[ok] m = {m_opt:g} -> tau {res['tau_campo_calibrado']}; acierto {a_1:.0%} -> {a_opt:.0%} en {len(usable)} tramos / {res['n_calzadas']} calzadas")
    print(pd.DataFrame(cv).to_string(index=False))
    print(out[["prueba", "fecha", "estado_idu", "score_sin_calibrar", "score_calibrado", "estado_dimmit_calibrado"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
