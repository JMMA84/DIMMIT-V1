"""Arma los entregables tabulares (fuera de muestra: test T + despliegue):

  reports/output/segmentos_scores.csv      una fila por segmento: índice, scores, estado,
                                           contexto, prioridad y descripción
  reports/output/observaciones_scores.csv  una fila por fotograma
  reports/output/diccionario_datos.csv     significado, unidad y fuente de cada columna
"""
import json

import numpy as np
import pandas as pd

from dimmit.geo.priority import priority
from dimmit.labels import pseudo_pci as pp
from dimmit.llm.describe import describe_all
from dimmit.models.score.dataset import build_tables, segment_truth
from dimmit.utils.io import CLASS_KEYS, CLASS_SLUGS, DATA, EVAL, FEATURES, OUTPUT, ensure_dir, load_yaml

PRED = FEATURES / "predicciones"

DICCIONARIO = [
    ("ranking_prioridad", "Índice del segmento: posición en el orden de intervención (1 = más prioritario)", "rango", "geo/priority.py"),
    ("segmento_id", "Identificador del segmento (pseudo-ruta de hasta 10 fotogramas)", "", "data/splits.py"),
    ("conjunto", "test (con verdad de referencia) o despliegue (test1/test2 de RDD2020, sin anotaciones)", "", "data/splits.py"),
    ("pais_imagenes", "País de origen de las imágenes RDD2020", "", "RDD2020"),
    ("n_fotogramas", "Fotogramas del segmento", "conteo", "data/splits.py"),
    ("civ", "Código de identificación vial de la calzada de Bogotá asignada (ruta simulada)", "", "UMV"),
    ("lat", "Latitud del centro del segmento", "grados", "UMV"),
    ("lon", "Longitud del centro del segmento", "grados", "UMV"),
    ("geohash7", "Celda geohash de 7 caracteres (~150 m)", "", "geo/geohash.py"),
    ("longitud_m", "Longitud recorrida del segmento", "m", "UMV"),
    ("fecha_captura", "Fecha y hora de captura (simulada)", "", "simulador de sensores"),
    ("fuente_sensor", "sim_gt: sensores simulados desde anotaciones; sim_pred: desde el detector", "", "simulador de sensores"),
    ("n_grieta_longitudinal", "Detecciones D00 (grieta longitudinal) sobre el umbral por clase", "conteo", "YOLOv10"),
    ("n_grieta_transversal", "Detecciones D10 (grieta transversal)", "conteo", "YOLOv10"),
    ("n_piel_cocodrilo", "Detecciones D20 (piel de cocodrilo)", "conteo", "YOLOv10"),
    ("n_baches", "Detecciones D40 (baches)", "conteo", "YOLOv10"),
    ("area_danada_pct", "Porcentaje medio de la imagen cubierto por daños detectados", "%", "YOLOv10"),
    ("score_grieta_longitudinal", "Sub-score por grietas longitudinales (100 - valor deducido)", "0-100", "red de fusión"),
    ("score_grieta_transversal", "Sub-score por grietas transversales", "0-100", "red de fusión"),
    ("score_piel_cocodrilo", "Sub-score por piel de cocodrilo", "0-100", "red de fusión"),
    ("score_baches", "Sub-score por baches", "0-100", "red de fusión"),
    ("score_visual_pci", "Índice visual tipo PCI (ASTM D6433 simplificado)", "0-100", "red de fusión"),
    ("iri_m_km", "Rugosidad estimada (Índice de Rugosidad Internacional)", "m/km", "red de fusión"),
    ("score_rugosidad", "Índice de rodadura desde el IRI", "0-100", "red de fusión"),
    ("score_condicion", "ÍNDICE DE CONDICIÓN VIAL FINAL (ICV) = 0.65 visual + 0.35 rodadura", "0-100", "red de fusión"),
    ("score_condicion_p05", "Límite inferior del intervalo conformal 90 %", "0-100", "red de fusión"),
    ("score_condicion_p95", "Límite superior del intervalo conformal 90 %", "0-100", "red de fusión"),
    ("estado", "bueno (86-100) / satisfactorio (71-85) / regular (56-70) / malo (41-55) / muy_malo (0-40)", "", "red de fusión"),
    ("confianza_estado", "Probabilidad del estado entregado", "0-1", "red de fusión"),
    ("prob_malo_o_peor", "Probabilidad de que el segmento esté malo o muy malo", "0-1", "red de fusión"),
    ("incertidumbre", "Desviación estándar del ensamble de redes", "puntos ICV", "red de fusión"),
    ("score_social", "Percentil de la dimensión social UMV (población, peticiones, colegios/salud, uso)", "0-100", "UMV"),
    ("score_movilidad", "Percentil de malla vial + rutas de transporte UMV", "0-100", "UMV"),
    ("score_seguridad_vial", "Percentil de siniestros ponderados por gravedad a 250 m (3 años)", "0-100", "SDM"),
    ("score_clima", "Percentil de lluvia acumulada 30 días", "0-100", "Open-Meteo"),
    ("score_importancia", "Importancia ponderada (social, movilidad, seguridad vial)", "0-100", "geo/priority.py"),
    ("score_prioridad", "Prioridad de intervención = 0.6 necesidad + 0.4 importancia + bonificaciones", "0-100", "geo/priority.py"),
    ("nivel_prioridad", "alta (>= 60) / media (>= 40) / baja", "", "geo/priority.py"),
    ("dist_colegio_m", "Distancia al colegio más cercano", "m", "SED Bogotá"),
    ("colegios_500m", "Colegios a 500 m", "conteo", "SED Bogotá"),
    ("dist_hospital_m", "Distancia al hospital/ESE más cercano", "m", "SDS Bogotá"),
    ("siniestros_250m", "Siniestros viales a 250 m (3 años)", "conteo", "SDM Bogotá"),
    ("lluvia_30d_mm", "Lluvia acumulada 30 días antes de la captura", "mm", "Open-Meteo"),
    ("pendiente_pct", "Pendiente longitudinal aproximada", "%", "Open-Meteo elevación"),
    ("ip_umv", "Índice de priorización oficial UMV 2020 de la calzada", "puntos", "UMV"),
    ("descripcion", "Descripción en español del segmento", "texto", "plantilla o Claude"),
    ("accion_recomendada", "Acción de mantenimiento sugerida", "", "reglas + plantilla/Claude"),
    ("urgencia", "Ventana de intervención sugerida", "", "reglas"),
    ("fuente_descripcion", "plantilla (determinista) o modelo de Claude usado", "", "llm/describe.py"),
    ("version_modelo", "Detector y red usados en la corrida", "", "pipeline"),
    ("run_id", "Identificador de la corrida", "", "pipeline"),
    ("score_condicion_real", "ICV de referencia (anotaciones humanas + IRI del simulador); solo test", "0-100", "labels"),
    ("estado_real", "Estado de referencia; solo test", "", "labels"),
]


def run_id():
    return pd.Timestamp.now(tz="America/Bogota").strftime("%Y%m%d-%H%M")


def build(run=None, use_llm=None):
    cfg_lab = pp.load_cfg()
    run = run or run_id()
    frame = build_tables()
    ctx = pd.read_parquet(FEATURES / "context_segment.parquet")
    truth = segment_truth(frame)
    det_src = (FEATURES / "detector_source.txt").read_text().strip()
    version = f"det={det_src.split('/')[-1]};fusion=ensamble{len(json.loads((DATA.parent / 'models/release/fusion/fusion_meta.json').read_text())['miembros'])}"
    parts = []
    for split in ("test", "despliegue"):
        seg = pd.read_parquet(PRED / f"principal__{split}_segmento.parquet")
        seg["conjunto"] = split
        parts.append(seg)
    seg = pd.concat(parts)
    seg["estado"] = pp.state_of(seg["icv"], cfg_lab)
    fr = frame[frame["segment_id"].isin(seg.index)]
    g = fr.groupby("segment_id")
    out = pd.DataFrame(index=seg.index)
    out["segmento_id"] = seg.index
    out["conjunto"] = seg["conjunto"]
    out["pais_imagenes"] = g["country"].first()
    out["n_fotogramas"] = g.size()
    c = ctx.set_index("segment_id").loc[seg.index]
    out["civ"] = c["civ"].astype("Int64")
    out["lat"], out["lon"], out["geohash7"] = c["lat"].round(6), c["lon"].round(6), c["geohash7"]
    out["longitud_m"] = c["longitud_m"].round(1)
    out["fecha_captura"] = pd.to_datetime(c["fecha_captura"]).dt.strftime("%Y-%m-%d %H:%M")
    out["fuente_sensor"] = np.where(seg["conjunto"] == "test", "sim_gt", "sim_pred")
    for k, s in CLASS_SLUGS.items():
        out[f"n_{s}"] = g[f"n_{k}"].sum().astype(int)
    out["area_danada_pct"] = (100 * g["union_total"].mean()).round(2)
    for k, s in CLASS_SLUGS.items():
        out[f"score_{s}"] = seg[f"sub_{k}"].round(1)
    out["score_visual_pci"] = seg["pci_vis"].round(1)
    out["iri_m_km"] = seg["iri"].round(2)
    out["score_rugosidad"] = seg["ride"].round(1)
    out["score_condicion"] = seg["icv"].round(1)
    out["score_condicion_p05"] = seg["icv_p05"].round(1)
    out["score_condicion_p95"] = seg["icv_p95"].round(1)
    out["estado"] = seg["estado"]
    probs = seg[[f"p_{s}" for s in pp.STATE_ORDER]]
    out["confianza_estado"] = [round(float(seg.loc[i, f"p_{e}"]), 3) for i, e in zip(seg.index, seg["estado"])]
    out["prob_malo_o_peor"] = (probs["p_malo"] + probs["p_muy_malo"]).round(3)
    out["incertidumbre"] = seg["icv_std_ensamble"].round(2)
    pr = priority(seg[["icv", "estado"]], ctx)
    out = out.join(pr)
    out["dist_colegio_m"] = c["dist_colegio_m"].round(0)
    out["colegios_500m"] = c["colegio_500m"]
    out["dist_hospital_m"] = c["dist_hospital_m"].round(0)
    out["siniestros_250m"] = c["siniestros_250m"]
    out["lluvia_30d_mm"] = c["lluvia_30d_mm"].round(1)
    out["pendiente_pct"] = c["pendiente_pct"].round(1)
    out["ip_umv"] = c["ip_umv"].round(2)
    texts, llm_stats = describe_all(out.reset_index(drop=True), use_llm=use_llm)
    out = out.merge(texts[["segmento_id", "descripcion", "accion_recomendada", "urgencia", "fuente_descripcion", "justificacion"]], on="segmento_id")
    out["version_modelo"] = version
    out["run_id"] = run
    t = truth.reindex(out["segmento_id"])
    out["score_condicion_real"] = np.where(out["conjunto"] == "test", t["icv"].round(1).to_numpy(), np.nan)
    out["estado_real"] = np.where(out["conjunto"] == "test", t["estado"].to_numpy(), None)
    out = out.sort_values("score_prioridad", ascending=False).reset_index(drop=True)
    out.insert(0, "ranking_prioridad", np.arange(1, len(out) + 1))
    cols = [c for c, *_ in DICCIONARIO]
    out = out[cols + ["justificacion"]]
    ensure_dir(OUTPUT)
    out.to_csv(OUTPUT / "segmentos_scores.csv", index=False)
    (EVAL / "llm_stats.json").write_text(json.dumps(llm_stats, indent=2, ensure_ascii=False))

    # observaciones (fotogramas)
    obs = []
    for split in ("test", "despliegue"):
        f = pd.read_parquet(PRED / f"principal__{split}_fotograma.parquet")
        obs.append(f.assign(conjunto=split))
    obs = pd.concat(obs)
    meta = pd.read_parquet(DATA / "simulated" / "sensor_frames.parquet")
    o = obs[["image_id", "segment_id", "conjunto", *[f"sub_{k}" for k in CLASS_KEYS], "pci_vis", "iri", "ride", "icv", "icv_directo"]].merge(
        meta[["image_id", "timestamp_captura", "lat", "lon"]], on="image_id", how="left"
    )
    o = o.merge(frame[["image_id", *[f"n_{k}" for k in CLASS_KEYS], "max_conf", "pci_det", "icv"]].rename(columns={"icv": "icv_real"}), on="image_id", how="left")
    o["estado"] = pp.state_of(o["icv"], cfg_lab)
    ren = {f"sub_{k}": f"score_{s}" for k, s in CLASS_SLUGS.items()} | {f"n_{k}": f"n_{s}" for k, s in CLASS_SLUGS.items()}
    ren |= {"pci_vis": "score_visual_pci", "iri": "iri_m_km", "ride": "score_rugosidad", "icv": "score_condicion", "icv_directo": "score_condicion_cabeza_directa",
            "segment_id": "segmento_id", "image_id": "imagen", "pci_det": "pci_detector", "icv_real": "score_condicion_real"}
    o = o.rename(columns=ren).round(3)
    o.to_csv(OUTPUT / "observaciones_scores.csv", index=False)
    pd.DataFrame(DICCIONARIO, columns=["columna", "descripcion", "unidad", "fuente"]).to_csv(OUTPUT / "diccionario_datos.csv", index=False)
    print(f"[ok] {len(out)} segmentos y {len(o)} fotogramas -> {OUTPUT}")
    return out, llm_stats, run


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", action="store_true", help="usar Claude (requiere ANTHROPIC_API_KEY)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    build(args.run_id, use_llm=True if args.llm else None)


if __name__ == "__main__":
    main()
