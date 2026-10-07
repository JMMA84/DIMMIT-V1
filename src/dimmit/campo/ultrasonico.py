"""Sensor ultrasónico de profundidad de bache (Arduino): lectura y resumen por serie.

Archivo: `Tiempo_ms,Distancia_cm,ProfundidadBache_cm` a ~4 Hz; el tiempo es relativo al arranque
del Arduino (sin reloj absoluto). Profundidad = Distancia - línea base (40 cm por defecto).
"""
import numpy as np
import pandas as pd


def baseline_mode(distancia_cm, min_n=20, fallback=40.0):
    """Línea base autocalibrada: la distancia más frecuente (redondeada a 1 cm) es el pavimento sano.

    No requiere medir el montaje: con menos de min_n lecturas se usa el valor fijo de configuración.
    """
    d = np.asarray(distancia_cm, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < min_n:
        return float(fallback), "fija"
    vals, counts = np.unique(np.round(d).astype(int), return_counts=True)
    return float(vals[counts.argmax()]), "moda"


def read_depth(path, linea_base_cm=40.0, auto=True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["t_ms", "distancia_cm", "profundidad_cm"][: len(df.columns)]
    df = df.dropna().astype({"t_ms": float, "distancia_cm": float, "profundidad_cm": float})
    # recalculamos desde la distancia para no depender de la línea base que usó el firmware
    base, fuente = (baseline_mode(df["distancia_cm"], fallback=linea_base_cm) if auto else (float(linea_base_cm), "fija"))
    df["profundidad_cm_base40"] = df["distancia_cm"] - linea_base_cm
    df["profundidad_cm"] = df["distancia_cm"] - base
    df.attrs["linea_base_cm"], df.attrs["fuente_linea_base"] = base, fuente
    df["t_s"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0
    df["t_norm"] = df["t_s"] / max(df["t_s"].iloc[-1], 1e-9)
    return df.reset_index(drop=True)


# severidad de bache por profundidad (ASTM D6433, baches de diámetro medio): L < 2.5 cm, M 2.5-5 cm, H > 5 cm
ASTM_POTHOLE_CM = {"L": 2.5, "M": 5.0}


def astm_severity(depth_cm):
    if depth_cm is None or not np.isfinite(depth_cm) or depth_cm < 0:
        return None
    return "L" if depth_cm < ASTM_POTHOLE_CM["L"] else ("M" if depth_cm < ASTM_POTHOLE_CM["M"] else "H")


def depth_features(df: pd.DataFrame, umbral_cm=3.0) -> dict:
    p = df["profundidad_cm"].to_numpy()
    return {
        "prof_max_cm": float(p.max()),
        "prof_p95_cm": float(np.percentile(p, 95)),
        "prof_media_cm": float(p.mean()),
        "prof_std_cm": float(p.std()),
        "frac_mayor_umbral": float((p >= umbral_cm).mean()),
        "n_lecturas": int(len(p)),
        "duracion_s": float(df["t_s"].iloc[-1]),
        "hz": float(len(p) / max(df["t_s"].iloc[-1], 1e-9)),
        "linea_base_cm": float(df.attrs.get("linea_base_cm", np.nan)),
        "fuente_linea_base": df.attrs.get("fuente_linea_base", "fija"),
        "prof_max_cm_base40": float(df["profundidad_cm_base40"].max()) if "profundidad_cm_base40" in df else np.nan,
        "severidad_astm": astm_severity(float(np.percentile(p, 95))),
    }


def align_normalized(df: pd.DataFrame, t_norm_query, half_window=0.03) -> np.ndarray:
    """Profundidad máxima en una ventana de tiempo normalizado alrededor de cada consulta (NaN si vacía)."""
    tn, p = df["t_norm"].to_numpy(), df["profundidad_cm"].to_numpy()
    out = np.full(len(t_norm_query), np.nan)
    for i, q in enumerate(np.asarray(t_norm_query, dtype=float)):
        m = np.abs(tn - q) <= half_window
        if m.any():
            out[i] = p[m].max()
    return out
