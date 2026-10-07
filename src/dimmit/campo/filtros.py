"""Defensa contra falsos positivos del detector en el encuadre de campo (a pie, cámara baja).

Cuatro capas, cada una marca las cajas que descarta en `motivo_descarte` (la primera capa que la
descarta gana) y cuenta cuántas quitó:
  1. umbral de confianza de campo (tau_campo, todas las clases);
  2. geometría: bandas anchas y bajas de D10 (juntas, sombra del bordillo) y centro fuera del
     trapecio de la calzada (el andén en el encuadre a pie);
  3. persistencia temporal: a ~2 Hz un daño real aparece en fotos consecutivas; la caja necesita una
     pareja de la misma clase (IoU tras compensar el avance) en las `vecinos` fotos anteriores o
     siguientes;
  4. consenso multi-escala: la caja debe aparecer también en la inferencia a otra escala (imgsz).
"""
import numpy as np
import pandas as pd

from dimmit.utils.io import CLASS_KEYS


def iou_matrix(a, b):
    """IoU entre dos tablas de cajas (cx, cy, w, h normalizadas)."""
    ax0, ay0 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax1, ay1 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx1, by1 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:4].prod(1)[:, None] + b[:, 2:4].prod(1)[None] - inter + 1e-9)


def in_road(cx, cy, cfg=None):
    cfg = cfg or {"cy_min": 0.35, "half_w0": 0.38, "half_w_slope": 0.12}
    return (cy > cfg["cy_min"]) & (np.abs(cx - 0.5) < cfg["half_w0"] + cfg["half_w_slope"] * cy)


def _best_iou_with_shift(box, cand, dy_tol):
    """IoU máximo contra las candidatas probando un desplazamiento vertical (avance de la cámara)."""
    best = 0.0
    for dy in (0.0, dy_tol, -dy_tol):
        shifted = cand.copy()
        shifted[:, 1] += dy
        best = max(best, float(iou_matrix(box[None], shifted).max()))
    return best


def apply(det: pd.DataFrame, frames: pd.DataFrame, cfg, det_scales: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (det con `conservada` y `motivo_descarte`, resumen por capa y clase)."""
    d = det.copy()
    if "cls" not in d:
        d["cls"] = d["class_id"].map(dict(enumerate(CLASS_KEYS)))
    d["motivo_descarte"] = ""
    # --- capa 1: umbral de campo ------------------------------------------------------------
    tau = cfg["tau_campo"]
    d["tau_campo"] = d["cls"].map(tau) if isinstance(tau, dict) else float(tau)
    m = d["conf"] < d["tau_campo"]
    d.loc[m, "motivo_descarte"] = "1_confianza"
    # --- capa 2: geometría -------------------------------------------------------------------
    b = cfg["banda"]
    m = (d["motivo_descarte"] == "") & (d["cls"] == b["clase"]) & (d["w"] > b["w_min"]) & (d["h"] < b["h_max"])
    d.loc[m, "motivo_descarte"] = "2_banda"
    m = (d["motivo_descarte"] == "") & ~in_road(d["cx"].to_numpy(), d["cy"].to_numpy(), cfg["via"])
    d.loc[m, "motivo_descarte"] = "2_fuera_de_via"
    # --- capa 3: persistencia temporal -----------------------------------------------------------
    order = frames.sort_values(["prueba", "timestamp"]).reset_index(drop=True)
    pos = {iid: (p, k) for k, (iid, p) in enumerate(zip(order["image_id"], order["prueba"]))}
    seq = {p: g["image_id"].tolist() for p, g in order.groupby("prueba", sort=False)}
    alive = d[d["motivo_descarte"] == ""]
    by_img = {k: g for k, g in alive.groupby("image_id")}
    pc = cfg["persistencia"]
    keep = pd.Series(True, index=d.index)
    for idx, r in alive.iterrows():
        if r["image_id"] not in pos:
            continue
        p, _ = pos[r["image_id"]]
        ids = seq[p]
        k = ids.index(r["image_id"])
        neigh = ids[max(0, k - pc["vecinos"]) : k] + ids[k + 1 : k + 1 + pc["vecinos"]]
        ok = False
        for n in neigh:
            g = by_img.get(n)
            if g is None:
                continue
            cand = g[g["cls"] == r["cls"]][["cx", "cy", "w", "h"]].to_numpy()
            if len(cand) and _best_iou_with_shift(r[["cx", "cy", "w", "h"]].to_numpy(float), cand, pc["dy_tol"]) >= pc["iou_min"]:
                ok = True
                break
        keep[idx] = ok
    d.loc[(d["motivo_descarte"] == "") & ~keep, "motivo_descarte"] = "3_sin_persistencia"
    # --- capa 4: consenso multi-escala -----------------------------------------------------------
    if det_scales:
        sc = cfg["escalas"]
        alive = d[d["motivo_descarte"] == ""]
        hits = pd.Series(0, index=d.index)
        for name, other in det_scales.items():
            o = other.copy()
            if "cls" not in o:
                o["cls"] = o["class_id"].map(dict(enumerate(CLASS_KEYS)))
            o["tau_campo"] = o["cls"].map(tau) if isinstance(tau, dict) else float(tau)
            o = o[o["conf"] >= o["tau_campo"] * 0.6]  # a otra escala la confianza cambia: tolerancia
            o_by = {k: g for k, g in o.groupby("image_id")}
            for idx, r in alive.iterrows():
                g = o_by.get(r["image_id"])
                cand = g[g["cls"] == r["cls"]][["cx", "cy", "w", "h"]].to_numpy() if g is not None else np.zeros((0, 4))
                if len(cand) and iou_matrix(r[["cx", "cy", "w", "h"]].to_numpy(float)[None], cand).max() >= sc["iou_min"]:
                    hits[idx] += 1
        d.loc[(d["motivo_descarte"] == "") & (hits < sc.get("minimo", 1)), "motivo_descarte"] = "4_sin_consenso_escala"
    d["conservada"] = d["motivo_descarte"] == ""
    resumen = d.assign(motivo=d["motivo_descarte"].replace("", "conservada")).pivot_table(index="cls", columns="motivo", values="conf", aggfunc="size", fill_value=0)
    return d, resumen
