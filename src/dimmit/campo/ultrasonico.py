"""Sensor ultrasónico de profundidad de bache (Arduino): lectura y resumen por serie.

Archivo: `Tiempo_ms,Distancia_cm,ProfundidadBache_cm` a ~4 Hz; el tiempo es relativo al arranque
del Arduino (sin reloj absoluto). Profundidad = Distancia - línea base (40 cm por defecto).
"""
import numpy as np
import pandas as pd


def read_depth(path, linea_base_cm=40.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["t_ms", "distancia_cm", "profundidad_cm"][: len(df.columns)]
    df = df.dropna().astype({"t_ms": float, "distancia_cm": float, "profundidad_cm": float})
    # recalculamos desde la distancia para no depender de la línea base que usó el firmware
    df["profundidad_cm"] = df["distancia_cm"] - linea_base_cm
    df["t_s"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0
    df["t_norm"] = df["t_s"] / max(df["t_s"].iloc[-1], 1e-9)
    return df.reset_index(drop=True)


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
