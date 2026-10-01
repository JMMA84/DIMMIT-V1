"""Índice de prioridad de intervención: necesidad (100 - ICV) x importancia del segmento.

Importancia con las dimensiones OFICIALES de la UMV de la calzada (percentil sobre las 93 680
calzadas de Bogotá) y la exposición local:
  score_social          percentil de la dimensión social UMV (población, peticiones, sitios
                        sociales como colegios/salud, uso social del suelo)
  score_movilidad       percentil de malla vial + rutas de transporte (UMV)
  score_seguridad_vial  percentil de siniestros ponderados por gravedad a 250 m (3 años)
  score_clima           percentil de lluvia acumulada 30 días (riesgo de infiltración en fisuras)
Bonificaciones: colegio a <= 300 m, hospital a <= 1 km, lluvia alta con estado malo/muy malo.
Reglas de acción y urgencia compartidas con las descripciones (llm/describe.py).
"""
import numpy as np
import pandas as pd

from dimmit.geo.sources import load_umv
from dimmit.utils.io import load_yaml

ACTIONS = ["mantenimiento_rutinario", "sellado_de_fisuras", "parcheo", "bacheo", "rehabilitacion", "reconstruccion"]
ALLOWED_ACTIONS = {
    "bueno": {"mantenimiento_rutinario"},
    "satisfactorio": {"mantenimiento_rutinario", "sellado_de_fisuras"},
    "regular": {"sellado_de_fisuras", "parcheo", "bacheo"},
    "malo": {"parcheo", "bacheo", "rehabilitacion"},
    "muy_malo": {"bacheo", "rehabilitacion", "reconstruccion"},
}
URGENCY = {"alta": "inmediata_30_dias", "media": "programada_6_meses", "baja": "rutinaria_12_meses"}


def rule_action(estado, dominant):
    """Acción por regla según estado y daño dominante."""
    if estado == "bueno":
        return "mantenimiento_rutinario"
    if estado == "satisfactorio":
        return "sellado_de_fisuras"
    if estado == "regular":
        return "bacheo" if dominant == "D40" else ("parcheo" if dominant == "D20" else "sellado_de_fisuras")
    if estado == "malo":
        return "bacheo" if dominant == "D40" else ("rehabilitacion" if dominant == "D20" else "parcheo")
    return "reconstruccion" if dominant == "D20" else "rehabilitacion"


def _pct(values, ref):
    ref = np.sort(np.asarray(ref, dtype=float))
    return 100 * np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)


def priority(seg: pd.DataFrame, ctx: pd.DataFrame, cfg_prio=None) -> pd.DataFrame:
    cfg_prio = cfg_prio or load_yaml("configs/evaluation.yaml")["prioridad"]
    umv = load_umv(load_yaml("configs/context.yaml"))
    d = seg.join(ctx.set_index("segment_id"), how="left")
    out = pd.DataFrame(index=seg.index)
    out["score_social"] = _pct(d["dim_social"].fillna(umv["dim_social"].median()), umv["dim_social"])
    mov_ref = umv["p_malla"] + umv["p_rutas_transporte"]
    out["score_movilidad"] = _pct((d["p_malla"] + d["p_rutas_transporte"]).fillna(mov_ref.median()), mov_ref)
    out["score_seguridad_vial"] = _pct(d["siniestros_pond_250m"].fillna(0), ctx["siniestros_pond_250m"])
    out["score_clima"] = _pct(d["lluvia_30d_mm"].fillna(ctx["lluvia_30d_mm"].median()), ctx["lluvia_30d_mm"])
    w = cfg_prio["pesos_importancia"]
    out["score_importancia"] = (w["social"] * out["score_social"] + w["movilidad"] * out["score_movilidad"] + w["seguridad_vial"] * out["score_seguridad_vial"]) / sum(w.values())
    need = 100 - d["icv"]
    bonus = (
        cfg_prio["bono_colegio_300m"] * (d["dist_colegio_m"] <= 300)
        + cfg_prio["bono_hospital_1km"] * (d["dist_hospital_m"] <= 1000)
        + cfg_prio["bono_lluvia"] * ((out["score_clima"] >= 75) & d["estado"].isin(["malo", "muy_malo"]))
    )
    out["score_prioridad"] = np.clip(cfg_prio["peso_necesidad"] * need + (1 - cfg_prio["peso_necesidad"]) * out["score_importancia"] + bonus, 0, 100)
    lv = cfg_prio["niveles"]
    out["nivel_prioridad"] = np.where(out["score_prioridad"] >= lv["alta"], "alta", np.where(out["score_prioridad"] >= lv["media"], "media", "baja"))
    return out.round(2)
