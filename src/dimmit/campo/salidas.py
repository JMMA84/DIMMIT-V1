"""Las dos salidas finales de campo (CSV + JSON) más el detalle por foto y el diccionario.

reports/campo/vias_clasificacion.{csv,json}            una fila por tramo (id_via) y por prueba
reports/campo/calidad_confianza_correlacion.{csv,json} formato largo: calidad, confianza, deriva, correlación
reports/campo/fotogramas_detalle.csv                   una fila por foto (trazabilidad)
reports/campo/diccionario_campo.csv
"""
import json
import time

import pandas as pd

from dimmit.campo import OUT_CAMPO
from dimmit.campo.scores import FEAT
from dimmit.utils.io import CLASS_SLUGS, ensure_dir

VERSION_DETECTOR = "yolov10n_rdd2020 (Kaggle CPU, 12 épocas, 416 px)"

COLS_VIAS = [
    ("id_via", "ID único del tramo (VIA-<sha1 de fecha|prueba|tramo|centroide>) o de la prueba (PRU-…)"),
    ("nivel", "tramo (~10 m) o prueba (recorrido completo)"),
    ("id_prueba", "ID de la prueba a la que pertenece"), ("prueba", "nombre de la carpeta de grabación"), ("tramo", "índice del tramo dentro de la prueba (-1 = fila de prueba)"),
    ("fecha", "fecha de captura"), ("hora_ini", "hora local de la primera foto"), ("hora_fin", "hora local de la última foto"),
    ("lat", "latitud del centroide (WGS84)"), ("lon", "longitud del centroide"), ("geohash7", "geohash de 7 caracteres (~150 m)"),
    ("longitud_m", "longitud recorrida según GPS suavizado"), ("gps_incertidumbre_m", "incertidumbre nominal de posición"), ("n_fotogramas", "fotos del tramo"),
    ("danos_detectados", "clases presentes en ≥ 20 % de las fotos (cajas que pasan los filtros del modo campo)"), ("dano_dominante", "clase con mayor valor deducido"),
    *[(f"n_{s}", f"cajas de {s} conservadas por el modo campo (suma sobre fotos)") for s in CLASS_SLUGS.values()],
    ("area_danada_pct", "porcentaje medio de la imagen cubierto por cajas conservadas"),
    *[(f"sub_score_{s}", f"100 − valor deducido de {s}") for s in CLASS_SLUGS.values()],
    ("score_calidad", "pseudo-PCI visual 0–100 (100 = sin daño) en modo campo: cajas que pasan las 4 capas de filtros, severidad de bache por profundidad ASTM"),
    ("estado", "bueno/satisfactorio/regular/malo/muy_malo según score_calidad"),
    ("score_calidad_tau_rdd2020", "antes: mismo score con el umbral de RDD2020 y sin filtros"), ("estado_tau_rdd2020", "estado del score anterior"),
    ("etiqueta_tau_rdd2020", "antes: etiqueta con el umbral de RDD2020 y la profundidad del firmware (distancia − 40 cm, sin depurar)"),
    ("etiqueta", "prevencion / mantenimiento / intervencion (regla en configs/campo.yaml)"), ("accion_recomendada", "acción por regla v1 según estado y daño dominante"),
    ("profundidad_max_cm", "profundidad máxima del sensor: Distancia − 8 cm (altura del sensor sobre pavimento plano), solo lecturas válidas (0 < distancia ≤ 35 cm)"),
    ("profundidad_media_cm", "profundidad media de las lecturas válidas; ≈ 0 o levemente negativa = pavimento sano"),
    ("profundidad_frac_bache", "fracción de lecturas válidas ≥ umbral de bache (3 cm)"), ("profundidad_max_cm_base40", "antes: profundidad máxima del firmware (distancia − 40 cm, sin depurar)"),
    ("linea_base_cm", "altura del sensor sobre pavimento plano (cm)"), ("lecturas_validas", "lecturas del sensor tras descartar nulas y > 35 cm (posicionamiento y apagado)"),
    ("lecturas_totales", "lecturas crudas del sensor en la prueba"), ("severidad_bache_astm", "severidad del bache por profundidad (ASTM D6433): L < 2.5 cm, M 2.5–5, H > 5"),
    ("fuente_profundidad", "cómo se asoció la profundidad (prueba completa / alineación por tiempo normalizado / sin sensor)"),
    ("confianza_modelo", "confianza media de las cajas conservadas de la clase dominante"), ("frac_fotos_con_deteccion", "fotos con alguna caja conservada"),
    ("brillo_media", "brillo medio (0–255)"), ("nitidez_media", "varianza de bordes (mayor = más nítida)"), ("frac_fotos_borrosa", "fotos bajo el p05 de nitidez de RDD2020"),
    ("frac_fotos_sobreexpuesta", "fotos sobre el p98 de brillo de RDD2020"),
    ("conf_media_detecciones_campo", "confianza máxima media de las fotos con alguna caja conservada"),
    ("hash_imagenes", "sha1 (12) de los hashes ordenados de sus fotos"), ("version_detector", "detector usado"), ("run_id", "identificador de la corrida"),
]


def main():
    run_id = time.strftime("campo-%Y%m%d-%H%M")
    vias = pd.read_parquet(FEAT / "vias.parquet")
    vias["version_detector"], vias["run_id"] = VERSION_DETECTOR, run_id
    cols = [c for c, _ in COLS_VIAS if c in vias]
    vias = vias.sort_values(["nivel", "prueba", "tramo"], ascending=[False, True, True])[cols]
    ensure_dir(OUT_CAMPO)
    vias.to_csv(OUT_CAMPO / "vias_clasificacion.csv", index=False, float_format="%.4f")
    vias.to_json(OUT_CAMPO / "vias_clasificacion.json", orient="records", force_ascii=False, indent=1)
    largo = pd.read_parquet(FEAT / "calidad_largo.parquet")
    largo.insert(0, "run_id", run_id)
    largo.to_csv(OUT_CAMPO / "calidad_confianza_correlacion.csv", index=False, float_format="%.4f")
    largo.to_json(OUT_CAMPO / "calidad_confianza_correlacion.json", orient="records", force_ascii=False, indent=1)
    fr = pd.read_parquet(FEAT / "fotogramas_features.parquet")
    # detalle por foto: solo lo que usa el modo campo (clase, confianza y geometría de las cajas conservadas)
    det_cols = ["image_id", "prueba", "id_via", "tramo", "timestamp", "lat", "lon", "gps_dt_s", "t_norm", "profundidad_cm", "severidad_bache_astm",
                "brillo", "nitidez", "borrosa", "sobreexpuesta", "n_detecciones_campo", "max_conf_campo", "pci_campo", "hash_imagen"]
    det_cols += [c for c in fr.columns if c.startswith(("ncampo_", "maxconfcampo_"))]
    fr[det_cols].to_csv(OUT_CAMPO / "fotogramas_detalle.csv", index=False, float_format="%.4f")
    dic = pd.DataFrame(COLS_VIAS, columns=["columna", "descripcion"])
    dic = dic[dic["columna"].isin(cols)]
    dic.to_csv(OUT_CAMPO / "diccionario_campo.csv", index=False)
    det_f = pd.read_parquet(FEAT / "detecciones_filtradas.parquet")
    keep_cols = ["image_id", "cls", "conf", "cx", "cy", "w", "h", "conservada", "motivo_descarte"]
    det_f[[c for c in keep_cols if c in det_f]].to_csv(OUT_CAMPO / "detecciones_filtradas.csv", index=False, float_format="%.4f")
    pd.read_csv(FEAT / "filtros_resumen.csv").to_csv(OUT_CAMPO / "filtros_resumen.csv", index=False)
    assert set(cols) == set(dic["columna"]), "diccionario incompleto"
    print(f"[ok] {run_id}: {len(vias)} filas en vias_clasificacion, {len(largo)} métricas, {len(fr)} fotos -> {OUT_CAMPO}")


if __name__ == "__main__":
    main()
