"""De las detecciones de YOLO en campo al score de calidad, estado y etiqueta por tramo.

Features acotadas: de YOLO solo se usan clase, confianza y geometría de cada caja (el embedding
de 448-d no aportó en la ablación v1). Dos lecturas del mismo tramo:
  * modo RDD2020 (`*_tau_rdd2020`): umbrales por clase elegidos en RDD2020 (0.04-0.09), sin filtros,
    profundidad del firmware (distancia − 40 cm, sin depurar). Es el "antes".
  * modo campo (principal): cajas que sobreviven las cuatro capas de `campo/filtros.py`, profundidad
    con el sensor depurado (sin lecturas nulas ni > 35 cm; distancia − 8 cm) y, cuando hay sensor, severidad del bache por profundidad (ASTM
    D6433) en vez del tamaño aparente. Es el "después".
Salidas (data/campo/features/): fotogramas_features.parquet, vias.parquet, detecciones_filtradas.parquet,
filtros_resumen.csv
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from dimmit.campo import CAMPO, INDEX_CAMPO, OUT_CAMPO, RAW, load_cfg
from dimmit.campo.filtros import apply as apply_filters
from dimmit.campo.ultrasonico import align_normalized, astm_severity, depth_features, read_depth
from dimmit.geo.geohash import encode
from dimmit.geo.priority import rule_action
from dimmit.labels import pseudo_pci as pp
from dimmit.models.score.vision import _union_by_image, frame_features, image_quality
from dimmit.utils.io import CLASS_KEYS, CLASS_SLUGS, FEATURES, REPO_ROOT, ensure_dir

FEAT = CAMPO / "features"


def detector_dir(arg=None):
    """Detecciones de Kaggle si ya se descargaron; si no, la corrida local."""
    if arg:
        return REPO_ROOT / arg
    k = OUT_CAMPO / "kaggle" / "outputs"
    return k if (k / "detections.parquet").exists() else CAMPO / f"det_local_{load_cfg()['inferencia']['imgsz']}"


def etiqueta(estado, prof_max_cm, maxconf_bache, cfg_et):
    inter, mant = cfg_et["intervencion"], cfg_et["mantenimiento"]
    prof = prof_max_cm if np.isfinite(prof_max_cm) else -np.inf
    if estado in inter["estados"] or prof >= inter["profundidad_cm"] or maxconf_bache >= inter["bache_conf_min"]:
        return "intervencion"
    if estado in mant["estados"] or prof >= mant["profundidad_cm"]:
        return "mantenimiento"
    return "prevencion"


def estado_idu(score, cfg):
    """Estado en la escala de 3 clases del IDU (bueno PCI >= 90, regular 56-89, malo < 56)."""
    r = (cfg.get("calibracion") or {}).get("rangos_idu", {"bueno": 90, "regular": 56})
    return "bueno" if score >= r["bueno"] else ("regular" if score >= r["regular"] else "malo")


def load_depth(cfg):
    """{prueba: (serie, features)} del sensor ultrasónico según `pruebas[*].sensor` en configs/campo.yaml."""
    out = {}
    u = cfg["ultrasonico"]
    for prueba, spec in cfg["pruebas"].items():
        path = RAW / spec["sensor"] if spec.get("sensor") else None
        if path is None or not path.exists():
            print(f"[aviso] {prueba}: sin archivo de sensor ({spec.get('sensor')})")
            continue
        s = read_depth(path, u["linea_base_cm"], u["rango_max_cm"], u.get("firmware_base_cm", 40.0))
        out[prueba] = (s, depth_features(s, u["umbral_bache_cm"]))
    return out


def agg_level(fr):
    """Nivel agregado: "sitio" (tomas repetidas del mismo lugar) si las pruebas lo declaran; si no, "prueba"."""
    if "id_sitio" in fr and fr["id_sitio"].notna().all():
        return "sitio", "id_sitio"
    return "prueba", "id_prueba"


def astm_weights(boxes, cfg_lab, cuts, severity_by_image):
    """Pesos que reemplazan la severidad por tamaño de los baches por la severidad ASTM medida."""
    kappa = cfg_lab["severity_kappa"]
    s = pp.size_proxy(boxes, cfg_lab)
    k_size = np.full(len(boxes), kappa["M"])
    lo, hi = cuts["D40"]
    k_size[s < lo], k_size[s >= hi] = kappa["L"], kappa["H"]
    w = np.ones(len(boxes))
    is_pot = (boxes["cls"] == "D40").to_numpy()
    sev = boxes["image_id"].map(severity_by_image).to_numpy()
    for i in np.where(is_pot)[0]:
        if isinstance(sev[i], str):
            w[i] = kappa[sev[i]] / k_size[i]
    return w


def image_quality_table(frames):
    q = pd.DataFrame([image_quality(p) for p in frames["path"]], columns=["brillo", "contraste", "nitidez"])
    q["image_id"] = frames["image_id"].to_numpy()
    q["jpeg_kb"] = [__import__("os").stat(p).st_size / 1024 for p in frames["path"]]
    return q


def frame_table(det, det_f, frames, cfg, severity_by_image, quality=None):
    cfg_lab, cuts = pp.load_cfg(), pp.load_cutoffs()
    taus = pd.read_json(FEATURES / "vision_thresholds.json")["tau"].to_dict()
    ids = pd.Index(frames["image_id"])
    # --- modo RDD2020: features del pipeline v1 tal cual ------------------------------------
    f = frame_features(det, ids, taus, cfg_lab, cuts)
    q = quality if quality is not None else image_quality_table(frames)
    f = f.merge(q, on="image_id")
    d = det.assign(cls=det["class_id"].map(dict(enumerate(CLASS_KEYS))))
    d["tau"] = d["cls"].map(taus)
    f["n_detecciones_tau"] = f[[f"n_{c}" for c in CLASS_KEYS]].sum(axis=1)
    f["conf_media_tau"] = d[d["conf"] >= d["tau"]].groupby("image_id")["conf"].mean().reindex(f["image_id"]).to_numpy()
    for c in CLASS_KEYS:
        f[f"n25_{c}"] = d[(d["cls"] == c) & (d["conf"] >= 0.25)].groupby("image_id").size().reindex(f["image_id"]).fillna(0).astype(int).to_numpy()
    # --- modo campo: cajas conservadas por los filtros + severidad ASTM ----------------------------
    keep = det_f[det_f["conservada"]]
    cols = ["image_id", "cls", "cx", "cy", "w", "h"]
    rho_c = pp.frame_rho(keep[cols], ids, cfg_lab, cuts, weights=astm_weights(keep, cfg_lab, cuts, severity_by_image) if len(keep) else None)
    pci_c = pp.pci_from_rho(rho_c, cfg_lab)
    for c in CLASS_KEYS:
        kc = keep[keep["cls"] == c]
        f[f"ncampo_{c}"] = kc.groupby("image_id").size().reindex(f["image_id"]).fillna(0).astype(int).to_numpy()
        f[f"maxconfcampo_{c}"] = kc.groupby("image_id")["conf"].max().reindex(f["image_id"]).fillna(0.0).to_numpy()
        f[f"rhocampo_{c}"] = rho_c[f"rho_{c}"].reindex(f["image_id"]).to_numpy()
    f["pci_campo"] = pci_c["pci_vis"].reindex(f["image_id"]).to_numpy()
    f["union_campo"] = _union_by_image(keep).reindex(f["image_id"]).fillna(0.0).to_numpy() if len(keep) else 0.0
    f["n_detecciones_campo"] = f[[f"ncampo_{c}" for c in CLASS_KEYS]].sum(axis=1)
    f["max_conf_campo"] = keep.groupby("image_id")["conf"].max().reindex(f["image_id"]).fillna(0.0).to_numpy()
    f["severidad_bache_astm"] = f["image_id"].map(severity_by_image)
    base = ["image_id", "path", "prueba", "id_prueba", "id_via", "tramo", "fecha", "timestamp", "lat", "lon", "sat", "vel_ms", "gps_dt_s",
            "gps_incertidumbre_m", "dist_acum_m", "t_norm", "hash_imagen"]
    base += [c for c in ("sitio", "id_sitio", "superficie", "con_gps", "archivo_original") if c in frames]
    return frames[base].merge(f, on="image_id")


def _score(g, prefix, cfg_lab):
    rho = g[[f"{prefix}_{c}" for c in CLASS_KEYS]].mean().to_frame().T
    rho.columns = [f"rho_{c}" for c in CLASS_KEYS]
    pci = pp.pci_from_rho(rho, cfg_lab).iloc[0]
    return float(pci["pci_vis"]), pci


def _depth(g, nivel, prueba, depth, cfg, col):
    """Profundidad asociada: por prueba o sitio las series completas, por tramo las lecturas alineadas a sus fotos."""
    out = {"max": np.nan, "media": np.nan, "frac": np.nan, "fuente": "sin_sensor"}
    pruebas = [q for q in g["prueba"].unique() if q in depth]
    if not pruebas:
        return out
    serie, feats = depth[pruebas[0]]
    completa = nivel != "tramo" or (g["t_norm"].min() <= 0 and g["t_norm"].max() >= 1)  # tramo = prueba entera (sin GPS)
    if completa:
        p = np.concatenate([depth[q][0][col].to_numpy(float) for q in pruebas])
        p = p[np.isfinite(p)]
        if not len(p):
            return {**out, "fuente": "sin_lecturas_validas"}
        return {"max": float(p.max()), "media": float(p.mean()), "frac": float((p >= cfg["ultrasonico"]["umbral_bache_cm"]).mean()), "fuente": "ultrasonico_prueba_completa"}
    al = g[col].to_numpy(float)
    al = al[np.isfinite(al)]
    if not len(al) and col == "profundidad_cm" and feats.get("n_validas", 1) == 0:
        out["fuente"] = "sin_lecturas_validas"
    if len(al):
        out = {"max": float(al.max()), "media": float(al.mean()), "frac": float((al >= cfg["ultrasonico"]["umbral_bache_cm"]).mean()), "fuente": "ultrasonico_alineado_tiempo_normalizado"}
    return out


def via_table(fr, tramos, depth, cfg):
    cfg_lab = pp.load_cfg()
    rows = []
    con_sitio = fr["id_sitio"].notna() if "id_sitio" in fr else pd.Series(False, index=fr.index)
    groups = [("tramo", k, g) for k, g in fr.groupby("id_via")]
    groups += [("sitio", k, g) for k, g in fr[con_sitio].groupby("id_sitio")] + [("prueba", k, g) for k, g in fr[~con_sitio].groupby("id_prueba")]
    for nivel, key, g in groups:
        prueba = g["prueba"].iloc[0] if nivel != "sitio" else f"Sitio_{int(g['sitio'].iloc[0])}"
        # modo campo (principal)
        score, pci = _score(g, "rhocampo", cfg_lab)
        estado = str(pp.state_of([score], cfg_lab)[0])
        dv = {c: float(pci[f"dv_{c}"]) for c in CLASS_KEYS}
        dominante = max(dv, key=dv.get) if max(dv.values()) > 0 else None
        presentes = [c for c in CLASS_KEYS if (g[f"ncampo_{c}"] > 0).mean() >= 0.2]
        prof = _depth(g, nivel, prueba, depth, cfg, "profundidad_cm")
        et = etiqueta(estado, prof["max"], float(g["maxconfcampo_D40"].max()), cfg["etiqueta"])
        # modo RDD2020 (antes)
        score_r, pci_r = _score(g, "rho_det", cfg_lab)
        estado_r = str(pp.state_of([score_r], cfg_lab)[0])
        prof_r = _depth(g, nivel, prueba, depth, cfg, "profundidad_cm_base40")
        et_r = etiqueta(estado_r, prof_r["max"], float(g["maxconf_D40"].max()), cfg["etiqueta"])
        fs = [depth[q][1] for q in g["prueba"].unique() if q in depth]
        feats = {"linea_base_cm": fs[0]["linea_base_cm"], "n_validas": sum(f["n_validas"] for f in fs), "n_lecturas": sum(f["n_lecturas"] for f in fs)} if fs else {}
        row = {
            "id_via": key, "nivel": nivel, "id_prueba": g["id_prueba"].iloc[0] if nivel != "sitio" else "", "prueba": prueba,
            "conjunto": g["conjunto"].iloc[0] if "conjunto" in g else "evaluacion",
            "ubicacion_fuente": g["ubicacion_fuente"].iloc[0] if "ubicacion_fuente" in g else "",
            "calcodigo_idu": g["calcodigo_idu"].iloc[0] if "calcodigo_idu" in g else None,
            "sitio": g["sitio"].iloc[0] if "sitio" in g else np.nan, "superficie": ";".join(sorted(g["superficie"].unique())) if "superficie" in g else "asfalto", "id_sitio": g["id_sitio"].iloc[0] if "id_sitio" in g else None,
            "tramo": int(g["tramo"].iloc[0]) if nivel == "tramo" else -1, "fecha": g["fecha"].iloc[0],
            "n_fotogramas": int(len(g)), "lat": float(g["lat"].mean()), "lon": float(g["lon"].mean()),
            "danos_detectados": ";".join(CLASS_SLUGS[c] for c in presentes), "dano_dominante": CLASS_SLUGS[dominante] if dominante else "",
            **{f"n_{CLASS_SLUGS[c]}": int(g[f"ncampo_{c}"].sum()) for c in CLASS_KEYS},
            **{f"frac_fotos_{CLASS_SLUGS[c]}": float((g[f"ncampo_{c}"] > 0).mean()) for c in CLASS_KEYS},
            **{f"maxconf_{CLASS_SLUGS[c]}": float(g[f"maxconfcampo_{c}"].max()) for c in CLASS_KEYS},
            "area_danada_pct": float(100 * g["union_campo"].mean()),
            **{f"sub_score_{CLASS_SLUGS[c]}": float(pci[f"sub_{c}"]) for c in CLASS_KEYS},
            "score_calidad": score, "estado": estado, "estado_idu_equivalente": estado_idu(score, cfg), "etiqueta": et, "accion_recomendada": rule_action(estado, dominante or "D00"),
            "profundidad_max_cm": prof["max"], "profundidad_media_cm": prof["media"], "profundidad_frac_bache": prof["frac"], "fuente_profundidad": prof["fuente"],
            "linea_base_cm": feats.get("linea_base_cm", np.nan), "lecturas_validas": feats.get("n_validas", 0), "lecturas_totales": feats.get("n_lecturas", 0),
            "severidad_bache_astm": astm_severity(prof["max"]) or "",
            "confianza_modelo": float(g.loc[g[f"maxconfcampo_{dominante}"] > 0, f"maxconfcampo_{dominante}"].mean()) if dominante else float("nan"),
            "max_conf": float(g["max_conf"].max()), "frac_fotos_con_deteccion": float((g["n_detecciones_campo"] > 0).mean()),
            # antes: modo RDD2020
            **{f"n_tau_rdd2020_{CLASS_SLUGS[c]}": int(g[f"n_{c}"].sum()) for c in CLASS_KEYS},
            "score_calidad_tau_rdd2020": score_r, "estado_tau_rdd2020": estado_r, "etiqueta_tau_rdd2020": et_r,
            "profundidad_max_cm_base40": prof_r["max"], "frac_fotos_con_deteccion_tau_rdd2020": float((g["n_detecciones_tau"] > 0).mean()),
        }
        rows.append(row)
    vias = pd.DataFrame(rows)
    t = tramos.set_index("id_via")
    for col in ("geohash7", "longitud_m", "hora_ini", "hora_fin", "gps_incertidumbre_m", "hash_imagenes", "lat_ini", "lon_ini", "lat_fin", "lon_fin"):
        vias[col] = vias["id_via"].map(t[col])
    ts = tramos["id_sitio"].notna() if "id_sitio" in tramos else pd.Series(False, index=tramos.index)
    for k, g in list(tramos[ts].groupby("id_sitio")) + list(tramos[~ts].groupby("id_prueba")):  # filas agregadas: agregados de sus tramos
        m = vias["id_via"] == k
        vias.loc[m, "longitud_m"] = float(g["longitud_m"].sum(min_count=1))
        vias.loc[m, "hora_ini"], vias.loc[m, "hora_fin"] = g["hora_ini"].min(), g["hora_fin"].max()
        vias.loc[m, "gps_incertidumbre_m"] = float(g["gps_incertidumbre_m"].max())
        lat0, lon0 = float(vias.loc[m, "lat"].iloc[0]), float(vias.loc[m, "lon"].iloc[0])
        vias.loc[m, "geohash7"] = encode(lat0, lon0, 7) if np.isfinite(lat0) else None
        vias.loc[m, "hash_imagenes"] = hashlib.sha1("".join(sorted(g["hash_imagenes"])).encode()).hexdigest()[:12]
    return vias


def fusion_scores(fr, cfg_lab):
    """Score secundario: red de fusión v1 en modo solo visión (grupos imu y embedding apagados)."""
    try:
        from dimmit.models.score.fusion import load_release, predict_ensemble

        members, _ = load_release(REPO_ROOT / "models/release/fusion")
    except Exception as e:  # sin pesos de fusión: se omite la columna
        print(f"[aviso] red de fusión no disponible: {e}")
        return None
    frame = fr.rename(columns={"id_via": "segment_id"})
    missing = sorted({c for m in members for g in ("imu", "embedding") for c in m["pres"][g].cols if c not in frame})
    frame = pd.concat([frame, pd.DataFrame(0.0, index=frame.index, columns=missing)], axis=1)
    _, seg = predict_ensemble(members, frame, cfg_lab, drop_groups=("imu", "embedding"))
    return seg["pci_vis"].rename("score_red_fusion_vision")


CALIB = FEAT / "calibracion_idu.json"


def filtros_calibrados(cfg):
    """Filtros de configs/campo.yaml con el umbral de confianza calibrado contra el IDU (si existe y se usa)."""
    f = {**cfg["filtros"], "tau_campo": dict(cfg["filtros"]["tau_campo"])}
    m = 1.0
    if (cfg.get("calibracion") or {}).get("usar") and CALIB.exists():
        m = float(json.loads(CALIB.read_text())["multiplicador"])
        f["tau_campo"] = {k: round(v * m, 4) for k, v in f["tau_campo"].items()}
    return f, m


def prepare(cfg, det_arg=None):
    """Todo lo que no depende de los filtros: detecciones, fotos, tramos, sensor, severidad, escalas y calidad de imagen."""
    det_dir = detector_dir(det_arg)
    det = pd.read_parquet(det_dir / "detections.parquet")
    frames = pd.read_parquet(INDEX_CAMPO / "fotogramas.parquet")
    tramos = pd.read_parquet(INDEX_CAMPO / "tramos.parquet")
    depth = load_depth(cfg)
    # profundidad alineada por foto (tiempo normalizado), con las dos líneas base
    frames["profundidad_cm"], frames["profundidad_cm_base40"] = np.nan, np.nan
    for prueba, (serie, _) in depth.items():
        m = frames["prueba"] == prueba
        frames.loc[m, "profundidad_cm"] = align_normalized(serie, frames.loc[m, "t_norm"].to_numpy())
        frames.loc[m, "profundidad_cm_base40"] = align_normalized(serie, frames.loc[m, "t_norm"].to_numpy(), col="profundidad_cm_base40")
    # severidad ASTM del bache por foto (profundidad depurada y alineada); sin sensor -> tamaño aparente
    sev = {r.image_id: astm_severity(r.profundidad_cm) for r in frames.itertuples() if np.isfinite(r.profundidad_cm)}
    # corridas a otras escalas para el consenso multi-escala
    scales = {}
    for s in cfg["filtros"]["escalas"]["imgsz"]:
        p = CAMPO / f"det_local_{s}" / "detections.parquet"
        if p.exists():
            scales[str(s)] = pd.read_parquet(p)
    return {"det_dir": det_dir, "det": det, "frames": frames, "tramos": tramos, "depth": depth, "sev": sev, "scales": scales,
            "quality": image_quality_table(frames)}


def compute(ctx, cfg, filtros):
    """Filtros -> features por foto -> score, estado y etiqueta por tramo/sitio/prueba."""
    det, frames = ctx["det"], ctx["frames"]
    det_f, resumen = apply_filters(det, frames, filtros, ctx["scales"] or None)
    fr = frame_table(det, det_f, frames, cfg, ctx["sev"], ctx["quality"])
    for col in ("profundidad_cm", "profundidad_cm_base40"):
        fr[col] = fr["image_id"].map(frames.set_index("image_id")[col])
    for col in ("conjunto", "ubicacion_fuente", "calcodigo_idu"):
        if col in frames:
            fr[col] = fr["image_id"].map(frames.set_index("image_id")[col])
    vias = via_table(fr, ctx["tramos"], ctx["depth"], cfg)
    return fr, vias, det_f, resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det", default=None, help="carpeta con detections.parquet (por defecto reports/campo/kaggle/outputs o data/campo/det_local)")
    args = ap.parse_args()
    cfg = load_cfg()
    ctx = prepare(cfg, args.det)
    det_dir, depth, scales = ctx["det_dir"], ctx["depth"], ctx["scales"]
    filtros, m = filtros_calibrados(cfg)
    fr, vias, det_f, resumen = compute(ctx, cfg, filtros)
    sec = fusion_scores(fr, pp.load_cfg())
    if sec is not None:
        vias["score_red_fusion_vision"] = vias["id_via"].map(sec)
    ensure_dir(FEAT)
    fr.to_parquet(FEAT / "fotogramas_features.parquet", index=False)
    vias.to_parquet(FEAT / "vias.parquet", index=False)
    det_f.to_parquet(FEAT / "detecciones_filtradas.parquet", index=False)
    resumen.to_csv(FEAT / "filtros_resumen.csv")
    (FEAT / "fuente_detecciones.json").write_text(json.dumps({
        "detector_dir": str(det_dir.relative_to(REPO_ROOT)), "escalas_consenso": list(scales),
        "calibracion_idu": {"multiplicador": m, "tau_campo_usado": filtros["tau_campo"]},
        "latencia": json.loads((det_dir / "latency.json").read_text()) if (det_dir / "latency.json").exists() else None,
        "sensor": {p: {k: f[k] for k in ("linea_base_cm", "n_lecturas", "n_validas", "n_fuera_de_rango", "n_nulas", "prof_max_cm", "severidad_astm")} for p, (_, f) in depth.items()}},
        indent=2, default=lambda x: None if isinstance(x, float) and x != x else x))
    print(f"[ok] detecciones de {det_dir.relative_to(REPO_ROOT)}; escalas {list(scales)}; tau = {m:g} · tau_campo {filtros['tau_campo']}; {len(fr)} fotos, {len(vias)} filas -> {FEAT}")
    print(resumen.to_string())
    print(vias[["id_via", "nivel", "prueba", "tramo", "score_calidad", "estado", "etiqueta", "score_calidad_tau_rdd2020", "etiqueta_tau_rdd2020",
                                           "profundidad_max_cm", "profundidad_max_cm_base40", "danos_detectados", "confianza_modelo"]].to_string(index=False))


if __name__ == "__main__":
    main()
