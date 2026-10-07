"""De las detecciones de YOLO en campo al score de calidad, estado y etiqueta por tramo.

Features acotadas: de YOLO solo se usan clase, confianza y geometría de cada caja (el embedding
de 448-d no aportó en la ablación v1). Score = pseudo-PCI visual con las mismas fórmulas y
umbrales por clase (tau, elegidos en el conjunto F de RDD2020) del pipeline v1. La profundidad
ultrasónica no entra al score: se reporta y puede subir la etiqueta operativa.
Salidas (data/campo/features/): fotogramas_features.parquet, vias.parquet
"""
import argparse
import json

import numpy as np
import pandas as pd

from dimmit.campo import CAMPO, INDEX_CAMPO, OUT_CAMPO, RAW, load_cfg
from dimmit.campo.ultrasonico import align_normalized, depth_features, read_depth
from dimmit.geo.priority import rule_action
from dimmit.labels import pseudo_pci as pp
from dimmit.models.score.vision import frame_features, image_quality
from dimmit.utils.io import CLASS_KEYS, CLASS_SLUGS, FEATURES, REPO_ROOT, ensure_dir

FEAT = CAMPO / "features"


def detector_dir(arg=None):
    """Detecciones de Kaggle si ya se descargaron; si no, la corrida local."""
    if arg:
        return REPO_ROOT / arg
    k = OUT_CAMPO / "kaggle" / "outputs"
    return k if (k / "detections.parquet").exists() else CAMPO / "det_local"


def etiqueta(estado, prof_max_cm, maxconf_bache, cfg_et):
    inter, mant = cfg_et["intervencion"], cfg_et["mantenimiento"]
    prof = prof_max_cm if np.isfinite(prof_max_cm) else -np.inf
    if estado in inter["estados"] or prof >= inter["profundidad_cm"] or maxconf_bache >= inter["bache_conf_min"]:
        return "intervencion"
    if estado in mant["estados"] or prof >= mant["profundidad_cm"]:
        return "mantenimiento"
    return "prevencion"


def load_depth(cfg):
    """{prueba: (serie, features)} del sensor ultrasónico según el mapa de configs/campo.yaml."""
    out = {}
    folder = RAW / "tomadedatos"
    for prueba, tag in cfg["ultrasonico"]["mapa"].items():
        hits = sorted(folder.glob(f"*{tag}*.txt"))
        if hits:
            s = read_depth(hits[0], cfg["ultrasonico"]["linea_base_cm"])
            out[prueba] = (s, depth_features(s, cfg["ultrasonico"]["umbral_bache_cm"]))
    return out


def frame_table(det, frames, cfg):
    taus = pd.read_json(FEATURES / "vision_thresholds.json")["tau"].to_dict()
    f = frame_features(det, pd.Index(frames["image_id"]), taus, pp.load_cfg(), pp.load_cutoffs())
    q = pd.DataFrame([image_quality(p) for p in frames["path"]], columns=["brillo", "contraste", "nitidez"])
    q["image_id"] = frames["image_id"].to_numpy()
    q["jpeg_kb"] = [REPO_ROOT.joinpath(p).stat().st_size / 1024 if not p.startswith("/") else __import__("os").stat(p).st_size / 1024 for p in frames["path"]]
    f = f.merge(q, on="image_id")
    # conteos con confianza >= 0.25 (sensibilidad: los tau de RDD2020 son bajos, 0.04-0.09)
    d = det.assign(cls=det["class_id"].map(dict(enumerate(CLASS_KEYS))))
    for c in CLASS_KEYS:
        n25 = d[(d["cls"] == c) & (d["conf"] >= 0.25)].groupby("image_id").size()
        f[f"n25_{c}"] = f["image_id"].map(n25).fillna(0).astype(int)
    # score alternativo con cajas de confianza >= 0.25 (sensibilidad al umbral fuera de dominio)
    d25 = d[d["conf"] >= 0.25]
    rho25 = pp.frame_rho(d25[["image_id", "cls", "cx", "cy", "w", "h"]], pd.Index(f["image_id"]), pp.load_cfg(), pp.load_cutoffs())
    for c in CLASS_KEYS:
        f[f"rho25_{c}"] = rho25[f"rho_{c}"].to_numpy()
    f["pci25"] = pp.pci_from_rho(rho25, pp.load_cfg())["pci_vis"].to_numpy()
    f["n_detecciones_tau"] = f[[f"n_{c}" for c in CLASS_KEYS]].sum(axis=1)
    f["conf_media_tau"] = d.merge(f[["image_id"]], on="image_id").assign(tau=lambda x: x["cls"].map(taus)).query("conf >= tau").groupby("image_id")["conf"].mean().reindex(f["image_id"]).to_numpy()
    cols = ["image_id", "path", "prueba", "id_prueba", "id_via", "tramo", "fecha", "timestamp", "lat", "lon", "sat", "vel_ms", "gps_dt_s",
            "gps_incertidumbre_m", "dist_acum_m", "t_norm", "hash_imagen"]
    return frames[cols].merge(f, on="image_id")


def via_table(fr, tramos, depth, cfg):
    cfg_lab = pp.load_cfg()
    rows = []
    groups = [("tramo", k, g) for k, g in fr.groupby("id_via")] + [("prueba", k, g) for k, g in fr.groupby("id_prueba")]
    for nivel, key, g in groups:
        prueba = g["prueba"].iloc[0]
        rho = g[[f"rho_det_{c}" for c in CLASS_KEYS]].mean().to_frame().T
        rho.columns = [f"rho_{c}" for c in CLASS_KEYS]
        pci = pp.pci_from_rho(rho, cfg_lab).iloc[0]
        score = float(pci["pci_vis"])
        rho25 = g[[f"rho25_{c}" for c in CLASS_KEYS]].mean().to_frame().T
        rho25.columns = [f"rho_{c}" for c in CLASS_KEYS]
        score25 = float(pp.pci_from_rho(rho25, cfg_lab)["pci_vis"].iloc[0])
        estado = str(pp.state_of([score], cfg_lab)[0])
        dv = {c: float(pci[f"dv_{c}"]) for c in CLASS_KEYS}
        presentes = [c for c in CLASS_KEYS if (g[f"n_{c}"] > 0).mean() >= 0.2]
        dominante = max(dv, key=dv.get) if max(dv.values()) > 0 else None
        maxconf_bache = float(g["maxconf_D40"].max())
        # profundidad: por prueba, la serie completa; por tramo, lecturas alineadas por tiempo normalizado
        prof = {"profundidad_max_cm": np.nan, "profundidad_media_cm": np.nan, "profundidad_frac_bache": np.nan, "fuente_profundidad": "sin_sensor"}
        if prueba in depth:
            serie, feats = depth[prueba]
            if nivel == "prueba":
                prof = {"profundidad_max_cm": feats["prof_max_cm"], "profundidad_media_cm": feats["prof_media_cm"],
                        "profundidad_frac_bache": feats["frac_mayor_umbral"], "fuente_profundidad": "ultrasonico_prueba_completa"}
            else:
                al = g["profundidad_cm"].to_numpy(float)
                al = al[np.isfinite(al)]
                if len(al):
                    prof = {"profundidad_max_cm": float(al.max()), "profundidad_media_cm": float(al.mean()),
                            "profundidad_frac_bache": float((al >= cfg["ultrasonico"]["umbral_bache_cm"]).mean()),
                            "fuente_profundidad": "ultrasonico_alineado_tiempo_normalizado"}
        et = etiqueta(estado, prof["profundidad_max_cm"], maxconf_bache, cfg["etiqueta"])
        row = {
            "id_via": key if nivel == "tramo" else g["id_prueba"].iloc[0], "nivel": nivel, "id_prueba": g["id_prueba"].iloc[0], "prueba": prueba,
            "tramo": int(g["tramo"].iloc[0]) if nivel == "tramo" else -1, "fecha": g["fecha"].iloc[0],
            "n_fotogramas": int(len(g)), "lat": float(g["lat"].mean()), "lon": float(g["lon"].mean()),
            "danos_detectados": ";".join(CLASS_SLUGS[c] for c in presentes), "dano_dominante": CLASS_SLUGS[dominante] if dominante else "",
            **{f"n_{CLASS_SLUGS[c]}": int(g[f"n_{c}"].sum()) for c in CLASS_KEYS},
            **{f"n25_{CLASS_SLUGS[c]}": int(g[f"n25_{c}"].sum()) for c in CLASS_KEYS},
            **{f"frac_fotos_{CLASS_SLUGS[c]}": float((g[f"n_{c}"] > 0).mean()) for c in CLASS_KEYS},
            **{f"maxconf_{CLASS_SLUGS[c]}": float(g[f"maxconf_{c}"].max()) for c in CLASS_KEYS},
            "area_danada_pct": float(100 * g["union_total"].mean()),
            **{f"sub_score_{CLASS_SLUGS[c]}": float(pci[f"sub_{c}"]) for c in CLASS_KEYS},
            "score_calidad": score, "estado": estado, "etiqueta": et,
            "score_calidad_conf25": score25, "estado_conf25": str(pp.state_of([score25], cfg_lab)[0]),
            "accion_recomendada": rule_action(estado, dominante or "D00"),
            **prof,
            "confianza_modelo": float(g[f"maxconf_{dominante}"].mean()) if dominante else float(g["max_conf"].mean()),
            "max_conf": float(g["max_conf"].max()), "frac_fotos_con_deteccion": float((g["n_detecciones_tau"] > 0).mean()),
        }
        rows.append(row)
    vias = pd.DataFrame(rows)
    t = tramos.set_index("id_via")
    for col in ("geohash7", "longitud_m", "hora_ini", "hora_fin", "gps_incertidumbre_m", "hash_imagenes", "lat_ini", "lon_ini", "lat_fin", "lon_fin"):
        vias[col] = vias["id_via"].map(t[col])
    # filas por prueba: agregados de sus tramos
    for k, g in tramos.groupby("id_prueba"):
        m = vias["id_via"] == k
        vias.loc[m, "longitud_m"] = float(g["longitud_m"].sum())
        vias.loc[m, "hora_ini"], vias.loc[m, "hora_fin"] = g["hora_ini"].min(), g["hora_fin"].max()
        vias.loc[m, "gps_incertidumbre_m"] = float(g["gps_incertidumbre_m"].max())
        from dimmit.geo.geohash import encode

        vias.loc[m, "geohash7"] = encode(float(vias.loc[m, "lat"].iloc[0]), float(vias.loc[m, "lon"].iloc[0]), 7)
        vias.loc[m, "hash_imagenes"] = __import__("hashlib").sha1("".join(sorted(g["hash_imagenes"])).encode()).hexdigest()[:12]
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
    # columnas de grupos apagados: ceros (la red las anula por el flag de modalidad)
    missing = sorted({c for m in members for g in ("imu", "embedding") for c in m["pres"][g].cols if c not in frame})
    frame = pd.concat([frame, pd.DataFrame(0.0, index=frame.index, columns=missing)], axis=1)
    _, seg = predict_ensemble(members, frame, cfg_lab, drop_groups=("imu", "embedding"))
    return seg["pci_vis"].rename("score_red_fusion_vision")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det", default=None, help="carpeta con detections.parquet (por defecto reports/campo/kaggle o data/campo/det_local)")
    args = ap.parse_args()
    cfg = load_cfg()
    det_dir = detector_dir(args.det)
    det = pd.read_parquet(det_dir / "detections.parquet")
    frames = pd.read_parquet(INDEX_CAMPO / "fotogramas.parquet")
    tramos = pd.read_parquet(INDEX_CAMPO / "tramos.parquet")
    depth = load_depth(cfg)
    fr = frame_table(det, frames, cfg)
    fr["profundidad_cm"] = np.nan
    for prueba, (serie, _) in depth.items():
        m = fr["prueba"] == prueba
        fr.loc[m, "profundidad_cm"] = align_normalized(serie, fr.loc[m, "t_norm"].to_numpy())
    vias = via_table(fr, tramos, depth, cfg)
    sec = fusion_scores(fr, pp.load_cfg())
    if sec is not None:
        vias["score_red_fusion_vision"] = vias["id_via"].map(sec)
    ensure_dir(FEAT)
    fr.to_parquet(FEAT / "fotogramas_features.parquet", index=False)
    vias.to_parquet(FEAT / "vias.parquet", index=False)
    (FEAT / "fuente_detecciones.json").write_text(json.dumps({"detector_dir": str(det_dir.relative_to(REPO_ROOT)),
                                                             "latencia": json.loads((det_dir / "latency.json").read_text()) if (det_dir / "latency.json").exists() else None}, indent=2))
    print(f"[ok] detecciones de {det_dir.relative_to(REPO_ROOT)}; {len(fr)} fotos, {len(vias)} filas (tramos + pruebas) -> {FEAT}")
    print(vias[vias["nivel"] == "tramo"][["id_via", "prueba", "tramo", "n_fotogramas", "score_calidad", "estado", "etiqueta", "danos_detectados", "profundidad_max_cm", "confianza_modelo"]].to_string(index=False))


if __name__ == "__main__":
    main()
