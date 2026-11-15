"""Sensor ultrasónico de profundidad de bache (Arduino): depuración, profundidad y resumen por serie.

Archivo: `Tiempo_ms,Distancia_cm,ProfundidadBache_cm` a ~4 Hz; el tiempo es relativo al arranque
del Arduino (sin reloj absoluto). Metodología del equipo de medición (sensor en un bastón manual):
  1. Depuración: al posicionar el bastón y al levantarlo para apagar el sistema salen lecturas fuera
     de rango (distancia > 35 cm) o nulas (0,00 cm). Se descartan; quedan las lecturas estables a ras
     de pavimento.
  2. Profundidad = Distancia − H_base, con H_base = 8,0 cm (altura real del sensor sobre pavimento
     plano). El firmware restaba 40 cm, lo que daba valores negativos sin sentido físico; esa versión
     se conserva solo como referencia del "antes" (`profundidad_cm_base40`).
  3. Lectura: > 0 cm es la profundidad de la depresión o bache; ≈ 0 o levemente negativa es pavimento
     sano y su rugosidad superficial.
"""
import numpy as np
import pandas as pd

H_BASE_CM = 8.0
RANGO_MAX_CM = 35.0
FIRMWARE_BASE_CM = 40.0


def read_depth(path, linea_base_cm=H_BASE_CM, rango_max_cm=RANGO_MAX_CM, firmware_base_cm=FIRMWARE_BASE_CM) -> pd.DataFrame:
    """Serie completa con la marca `valida`; `profundidad_cm` es NaN en las lecturas descartadas."""
    df = pd.read_csv(path)
    df.columns = ["t_ms", "distancia_cm", "profundidad_firmware_cm"][: len(df.columns)]
    df = df.dropna(subset=["t_ms", "distancia_cm"]).astype({"t_ms": float, "distancia_cm": float}).reset_index(drop=True)
    d = df["distancia_cm"]
    df["valida"] = (d > 0) & (d <= rango_max_cm)
    df["motivo_descarte"] = np.where(d <= 0, "lectura_nula", np.where(d > rango_max_cm, "fuera_de_rango", ""))
    df["profundidad_cm"] = np.where(df["valida"], d - linea_base_cm, np.nan)
    df["profundidad_cm_base40"] = d - firmware_base_cm  # referencia del "antes", sin depurar
    df.attrs.update(linea_base_cm=float(linea_base_cm), rango_max_cm=float(rango_max_cm))
    # tiempo normalizado sobre la serie completa, para alinear con las fotos de toda la prueba
    df["t_s"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0
    df["t_norm"] = df["t_s"] / max(df["t_s"].iloc[-1], 1e-9)
    return df


# severidad de bache por profundidad (ASTM D6433, baches de diámetro medio): L < 2.5 cm, M 2.5-5 cm, H > 5 cm
ASTM_POTHOLE_CM = {"L": 2.5, "M": 5.0}


def astm_severity(depth_cm):
    if depth_cm is None or not np.isfinite(depth_cm) or depth_cm < 0:
        return None
    return "L" if depth_cm < ASTM_POTHOLE_CM["L"] else ("M" if depth_cm < ASTM_POTHOLE_CM["M"] else "H")


def depth_features(df: pd.DataFrame, umbral_cm=3.0) -> dict:
    v = df[df["valida"]] if "valida" in df else df
    p = v["profundidad_cm"].to_numpy(float)
    n = len(p)
    base = {
        "n_lecturas": int(len(df)), "n_validas": n,
        "n_fuera_de_rango": int((df.get("motivo_descarte", pd.Series(dtype=str)) == "fuera_de_rango").sum()),
        "n_nulas": int((df.get("motivo_descarte", pd.Series(dtype=str)) == "lectura_nula").sum()),
        "duracion_s": float(df["t_s"].iloc[-1]), "linea_base_cm": float(df.attrs.get("linea_base_cm", H_BASE_CM)),
        "prof_max_cm_base40": float(df["profundidad_cm_base40"].max()) if "profundidad_cm_base40" in df else np.nan,
    }
    if not n:
        return {**base, "prof_max_cm": np.nan, "prof_p95_cm": np.nan, "prof_media_cm": np.nan, "prof_std_cm": np.nan,
                "frac_mayor_umbral": np.nan, "severidad_astm": None}
    return {**base, "prof_max_cm": float(p.max()), "prof_p95_cm": float(np.percentile(p, 95)), "prof_media_cm": float(p.mean()),
            "prof_std_cm": float(p.std()), "frac_mayor_umbral": float((p >= umbral_cm).mean()),
            "severidad_astm": astm_severity(float(np.percentile(p, 95)))}


def align_normalized(df: pd.DataFrame, t_norm_query, half_window=0.03, col="profundidad_cm") -> np.ndarray:
    """Profundidad máxima válida en una ventana de tiempo normalizado alrededor de cada consulta (NaN si vacía)."""
    tn, p = df["t_norm"].to_numpy(), df[col].to_numpy(float)
    ok = np.isfinite(p)
    out = np.full(len(t_norm_query), np.nan)
    for i, q in enumerate(np.asarray(t_norm_query, dtype=float)):
        m = ok & (np.abs(tn - q) <= half_window)
        if m.any():
            out[i] = p[m].max()
    return out
