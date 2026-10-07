"""Calidad de imagen, confianza del modelo, deriva frente a RDD2020 y correlación sensor <-> YOLO.

Salida: data/campo/features/calidad_largo.parquet (formato largo: id_via | nivel | bloque | metrica |
valor | ic95_inf | ic95_sup | n | nota) y columnas de calidad/confianza agregadas a vias.parquet.
Referencia de deriva: fotogramas de F ∪ T de RDD2020 (data/features/vision_frame.parquet).
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from dimmit.campo import INDEX_CAMPO, load_cfg
from dimmit.campo.scores import FEAT
from dimmit.evaluation.metrics import psi
from dimmit.utils.io import CLASS_KEYS, CLASS_SLUGS, FEATURES, INDEX

Q_COLS = ["brillo", "contraste", "nitidez", "jpeg_kb"]


def reference_frames():
    ref = pd.read_parquet(FEATURES / "vision_frame.parquet")
    splits = pd.read_csv(INDEX / "splits.csv")
    ids = splits.loc[splits["split"].isin(["fusion", "test"]), "image_id"]
    return ref[ref["image_id"].isin(ids)].reset_index(drop=True)


def quality_flags(fr, ref):
    p05 = ref["nitidez"].quantile(0.05)
    lo, hi = ref["brillo"].quantile(0.02), ref["brillo"].quantile(0.98)
    out = fr.copy()
    out["borrosa"] = out["nitidez"] < p05
    out["oscura"] = out["brillo"] < lo
    out["sobreexpuesta"] = out["brillo"] > hi
    return out, {"nitidez_p05_ref": float(p05), "brillo_p02_ref": float(lo), "brillo_p98_ref": float(hi)}


def boot_ci(x, y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            vals.append(fn(x[i], y[i])[0])
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))) if vals else (np.nan, np.nan)


def domain_auroc(fr, ref, cols):
    """Qué tan separables son las fotos de campo de las de RDD2020 (0.5 = indistinguibles)."""
    a = ref[cols].sample(min(len(ref), 3000), random_state=0)
    b = fr[cols]
    x = pd.concat([a, b]).fillna(0).to_numpy(float)
    x = np.log1p(np.maximum(x, 0))
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), (x - x.mean(0)) / (x.std(0) + 1e-9), y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def build(fr, vias, cfg):
    ref = reference_frames()
    fr, cuts = quality_flags(fr, ref)
    rows = []
    add = lambda via, nivel, bloque, metrica, valor, n, nota="", lo=np.nan, hi=np.nan: rows.append(
        {"id_via": via, "nivel": nivel, "bloque": bloque, "metrica": metrica, "valor": valor, "ic95_inf": lo, "ic95_sup": hi, "n": n, "nota": nota})
    # --- por tramo y por prueba: calidad de imagen y confianza ---------------------------
    for nivel, col in (("tramo", "id_via"), ("prueba", "id_prueba")):
        for key, g in fr.groupby(col):
            for q in Q_COLS:
                add(key, nivel, "calidad_imagen", f"{q}_media", float(g[q].mean()), len(g))
            for flag in ("borrosa", "oscura", "sobreexpuesta"):
                add(key, nivel, "calidad_imagen", f"frac_fotos_{flag}", float(g[flag].mean()), len(g), f"umbral de RDD2020 F∪T: {cuts}")
            add(key, nivel, "confianza_modelo", "max_conf_media", float(g["max_conf"].mean()), len(g))
            add(key, nivel, "confianza_modelo", "max_conf_max", float(g["max_conf"].max()), len(g))
            add(key, nivel, "confianza_modelo", "conf_media_detecciones_tau", float(g["conf_media_tau"].mean()), len(g), "cajas con conf >= tau por clase")
            add(key, nivel, "confianza_modelo", "frac_fotos_con_deteccion_tau", float((g["n_detecciones_tau"] > 0).mean()), len(g))
            add(key, nivel, "confianza_modelo", "frac_fotos_con_deteccion_conf25", float((g[[f"n25_{c}" for c in CLASS_KEYS]].sum(axis=1) > 0).mean()), len(g))
            add(key, nivel, "confianza_modelo", "n_detecciones_por_foto_tau", float(g["n_detecciones_tau"].mean()), len(g))
            add(key, nivel, "confianza_modelo", "frac_fotos_con_deteccion_campo", float((g["n_detecciones_campo"] > 0).mean()), len(g), "cajas que pasan las 4 capas de filtros")
            add(key, nivel, "confianza_modelo", "n_detecciones_por_foto_campo", float(g["n_detecciones_campo"].mean()), len(g))
            add(key, nivel, "confianza_modelo", "conf_media_detecciones_campo", float(g.loc[g["max_conf_campo"] > 0, "max_conf_campo"].mean()) if (g["max_conf_campo"] > 0).any() else float("nan"), len(g))
    # --- deriva vs RDD2020 (lote completo y por prueba) --------------------------------------
    dv = cfg["deriva"]["variables"]
    for key, g in [("LOTE", fr)] + list(fr.groupby("id_prueba")):
        nivel = "lote" if key == "LOTE" else "prueba"
        for v in dv:
            val = psi(ref[v], g[v])
            est = "OK" if val <= cfg["deriva"]["psi_ok"] else ("ALERTA" if val <= cfg["deriva"]["psi_alerta"] else "FALLA")
            add(key, nivel, "deriva", f"psi_{v}", val, len(g), f"{est}; referencia F∪T RDD2020 (n={len(ref)})")
        if len(g) >= 20:
            add(key, nivel, "deriva", "dominio_auroc", domain_auroc(g, ref, Q_COLS + ["max_conf", "pci_det"]), len(g), "0.5 = indistinguible de RDD2020; 1.0 = otro dominio")
    # --- correlación sensor ultrasónico <-> YOLO ------------------------------------------------
    t = vias[(vias["nivel"] == "tramo") & vias["profundidad_max_cm"].notna()]
    pairs = [("profundidad_max_cm", "n_baches"), ("profundidad_max_cm", "maxconf_baches"), ("profundidad_media_cm", "n_baches"),
             ("profundidad_frac_bache", "frac_fotos_baches"), ("profundidad_max_cm", "score_calidad"),
             ("profundidad_max_cm", "n_grieta_longitudinal"),  # control negativo
             ("profundidad_max_cm_base40", "n_baches")]  # con la línea base fija del firmware (antes)
    for a, b in pairs:
        if len(t) >= 4:
            rs, lo, hi = spearmanr(t[a], t[b])[0], *boot_ci(t[a], t[b], spearmanr)
            nota = "control negativo: no debería correlacionar" if "longitudinal" in b else ("línea base fija 40 cm (antes)" if "base40" in a else "por tramo; línea base autocalibrada; n pequeño, IC bootstrap")
            add("LOTE", "tramo", "correlacion_sensor", f"spearman_{a}__{b}", float(rs), len(t), nota, lo, hi)
            add("LOTE", "tramo", "correlacion_sensor", f"pearson_{a}__{b}", float(pearsonr(t[a], t[b])[0]), len(t), nota)
    p = vias[(vias["nivel"] == "prueba") & vias["profundidad_max_cm"].notna()]
    if len(p) >= 3:
        for a, b in (("profundidad_max_cm", "n_baches"), ("profundidad_media_cm", "score_calidad")):
            add("LOTE", "prueba", "correlacion_sensor", f"spearman_{a}__{b}", float(spearmanr(p[a], p[b])[0]), len(p), "n = 3 pruebas: solo descriptivo")
    # por foto, dentro de cada prueba, con la profundidad alineada por tiempo normalizado
    for key, g in fr.groupby("id_prueba"):
        g = g[g["profundidad_cm"].notna()]
        if len(g) >= 10:
            for b in ("maxconfcampo_D40", "ncampo_D40", "maxconf_D40", "maxconf_D00"):
                nota = "supuesto: alineación por tiempo normalizado 0-1" + (" (control negativo)" if b == "maxconf_D00" else "")
                add(key, "prueba", "correlacion_sensor", f"spearman_foto_profundidad_cm__{b}", float(spearmanr(g["profundidad_cm"], g[b])[0]), len(g), nota)
    # --- validación mínima: revisión visual de una muestra de fotos (presencia por clase) ---------
    rev_path = INDEX_CAMPO / "revision_visual_31.csv"
    if rev_path.exists():
        rev = pd.read_csv(rev_path).merge(fr, on="image_id")
        nota = "revisión visual rápida de una muestra estratificada (no anotación experta); presencia a nivel foto"
        for modo, cols in (("tau_rdd2020", {"piel_cocodrilo": "n_D20", "bache": "n_D40", "cualquier_dano": None}),
                           ("campo", {"piel_cocodrilo": "ncampo_D20", "bache": "ncampo_D40", "cualquier_dano": None})):
            for clase, col in cols.items():
                if col is None:
                    pred = (rev[[f"{'ncampo' if modo == 'campo' else 'n'}_{c}" for c in CLASS_KEYS]].sum(axis=1) > 0)
                    truth = (rev[["piel_cocodrilo", "bache", "grieta"]].sum(axis=1) > 0)
                else:
                    pred, truth = rev[col] > 0, rev[clase] == 1
                tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
                add("MUESTRA", "foto", "validacion_visual", f"precision_{clase}_{modo}", tp / max(tp + fp, 1), len(rev), f"{nota}; TP={tp} FP={fp} FN={fn}")
                add("MUESTRA", "foto", "validacion_visual", f"recall_{clase}_{modo}", tp / max(tp + fn, 1), len(rev), f"{nota}; TP={tp} FP={fp} FN={fn}")
        parches = rev[rev["parche"] == 1]
        add("MUESTRA", "foto", "validacion_visual", "parches_marcados_como_piel_cocodrilo_campo", float((parches["ncampo_D20"] > 0).mean()) if len(parches) else np.nan, len(parches),
            "fracción de parches de asfalto (reparaciones) que el modo campo clasifica como piel de cocodrilo: confusión conocida del detector")
    largo = pd.DataFrame(rows)
    # columnas resumidas para vias.parquet
    piv = largo[largo["bloque"].isin(["calidad_imagen", "confianza_modelo"])].pivot_table(index="id_via", columns="metrica", values="valor")
    keep = ["brillo_media", "nitidez_media", "frac_fotos_borrosa", "frac_fotos_sobreexpuesta", "max_conf_media", "conf_media_detecciones_tau", "frac_fotos_con_deteccion_conf25", "conf_media_detecciones_campo"]
    vias = vias.drop(columns=[c for c in keep if c in vias], errors="ignore").merge(piv[keep], left_on="id_via", right_index=True, how="left")
    return fr, vias, largo


def main():
    cfg = load_cfg()
    fr = pd.read_parquet(FEAT / "fotogramas_features.parquet")
    vias = pd.read_parquet(FEAT / "vias.parquet")
    fr, vias, largo = build(fr, vias, cfg)
    fr.to_parquet(FEAT / "fotogramas_features.parquet", index=False)
    vias.to_parquet(FEAT / "vias.parquet", index=False)
    largo.to_parquet(FEAT / "calidad_largo.parquet", index=False)
    print(f"[ok] {len(largo)} métricas -> {FEAT / 'calidad_largo.parquet'}")
    print(largo[largo["bloque"].isin(["deriva", "correlacion_sensor"]) & (largo["id_via"] == "LOTE")][["nivel", "metrica", "valor", "ic95_inf", "ic95_sup", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
