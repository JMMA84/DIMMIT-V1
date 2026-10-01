"""Métricas del marco de evaluación (regresión, clasificación ordinal, calibración, deriva).

Los intervalos de confianza usan bootstrap por conglomerados (bloques del split), porque los
segmentos de un mismo bloque no son independientes.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dimmit.labels.pseudo_pci import STATE_ORDER


def regression(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    ss = ((y - y.mean()) ** 2).sum()
    return {
        "mae": float(np.abs(y - p).mean()),
        "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "r2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan"),
        "spearman": float(spearmanr(y, p).statistic) if len(y) > 2 else float("nan"),
        "pearson": float(pearsonr(y, p).statistic) if len(y) > 2 else float("nan"),
        "sesgo": float((p - y).mean()),
    }


def ordinal(y_true, y_pred):
    yt = np.array([STATE_ORDER.index(s) for s in y_true])
    yp = np.array([STATE_ORDER.index(s) for s in y_pred])
    bad = yt <= STATE_ORDER.index("malo")
    out = {
        "exactitud": float((yt == yp).mean()),
        "exactitud_balanceada": float(balanced_accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", labels=range(len(STATE_ORDER)), zero_division=0)),
        "kappa_ponderada_cuadratica": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "exactitud_mas_menos_1": float((np.abs(yt - yp) <= 1).mean()),
        "recall_malo_o_peor": float((yp[bad] <= STATE_ORDER.index("malo")).mean()) if bad.any() else float("nan"),
        "precision_malo_o_peor": float((yt[yp <= 1] <= 1).mean()) if (yp <= 1).any() else float("nan"),
    }
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=range(len(STATE_ORDER)), zero_division=0)
    per = {STATE_ORDER[k]: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "soporte": int(s[k])} for k in range(len(STATE_ORDER))}
    return out, per


def confusion(y_true, y_pred):
    return pd.crosstab(pd.Categorical(y_true, STATE_ORDER), pd.Categorical(y_pred, STATE_ORDER), rownames=["real"], colnames=["predicho"], dropna=False)


def ece(probs: np.ndarray, y_true, n_bins=10):
    """Error de calibración esperado (confianza de la clase predicha)."""
    yt = np.array([STATE_ORDER.index(s) for s in y_true])
    conf = probs.max(1)
    pred = probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == yt[m]).mean() - conf[m].mean())
    return float(e)


def coverage(y, lo, hi):
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    return float(((y >= lo) & (y <= hi)).mean()), float((hi - lo).mean())


def detection_presence(y_true, score, pred):
    y_true = np.asarray(y_true, bool)
    p = (pred & y_true).sum() / max(pred.sum(), 1)
    r = (pred & y_true).sum() / max(y_true.sum(), 1)
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "ap": float(average_precision_score(y_true, score)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(2 * p * r / (p + r)) if p + r else 0.0,
    }


def block_bootstrap(df, groups, stat, n=1000, seed=0):
    """IC 95 % de stat(df) re-muestreando grupos (bloques) con reemplazo."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by = {u: np.flatnonzero(g == u) for u in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx_by[u] for u in pick])
        try:
            vals.append(stat(df.iloc[rows]))
        except Exception:  # noqa: BLE001 - réplicas degeneradas (p. ej. una sola clase)
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def psi(ref, cur, bins=10):
    """Population Stability Index con cuantiles de la referencia."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


def status(value, ok, warn, direction):
    """OK / ALERTA / FALLA según umbrales; direction = 'mayor' (mejor alto) o 'menor'."""
    if value is None or not np.isfinite(value) or ok is None:
        return "INFO"
    if direction == "mayor":
        return "OK" if value >= ok else ("ALERTA" if value >= warn else "FALLA")
    if direction == "menor":
        return "OK" if value <= ok else ("ALERTA" if value <= warn else "FALLA")
    if direction == "rango":  # ok y warn son (lo, hi)
        return "OK" if ok[0] <= value <= ok[1] else ("ALERTA" if warn[0] <= value <= warn[1] else "FALLA")
    return "INFO"
