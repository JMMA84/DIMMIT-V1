"""Modelo de scores v0: índice de condición de vía 0-100 por segmento.

Implementa el índice basado en reglas propuesto en el Reporte Técnico V1:
    score = 100 - sum(w_clase * severidad_clase) - w_sensor * rugosidad_norm
La featurización queda lista para reemplazar el scorer por uno aprendido
(RandomForest/XGBoost) cuando existan etiquetas reales de condición.
"""
import numpy as np
import pandas as pd

from dimmit.models.score.features import build_features, load_inputs
from dimmit.utils.io import REPO_ROOT, load_yaml


class RuleBasedScorer:
    def __init__(self, cfg):
        self.class_weights = cfg["class_weights"]
        self.sensor_weight = cfg["sensor_weight"]
        self.categories = cfg["categories"]

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features[["segment_id"]].copy()
        penalty = np.zeros(len(features))
        for name, w in self.class_weights.items():
            penalty += w * features[f"sev_{name}"].to_numpy()
        penalty += self.sensor_weight * features["roughness_norm"].to_numpy()
        out["score"] = np.clip(100 - penalty, 0, 100).round(1)
        out["category"] = out["score"].apply(self._categorize)
        return out

    def _categorize(self, score):
        for name, lower in sorted(self.categories.items(), key=lambda kv: -kv[1]):
            if score >= lower:
                return name
        return min(self.categories, key=self.categories.get)


class LearnedScorer:
    """Placeholder: entrenar sobre las mismas features cuando haya etiquetas
    reales de condición (p. ej. PCI medido en campo)."""

    def __init__(self):
        raise NotImplementedError("Pendiente de etiquetas reales de condición de vía")


def main():
    cfg = load_yaml("configs/score.yaml")
    detections, segments, sensors = load_inputs()
    names = load_yaml("configs/data_rdd2020.yaml")["names"]

    features = build_features(detections, segments, sensors, names, cfg["conf_threshold"])
    scores = RuleBasedScorer(cfg).score(features)

    out = REPO_ROOT / "reports/scores.csv"
    scores.merge(features, on="segment_id").to_csv(out, index=False)

    sev_sim = segments.groupby("segment_id")["severity"].mean().reset_index()
    sanity = scores.merge(sev_sim, on="segment_id")
    corr = sanity["score"].corr(sanity["severity"])
    print(scores["category"].value_counts().rename("segmentos").to_string())
    print(f"\nCorrelación score vs severidad simulada: {corr:.2f} (se espera negativa)")
    print(f"[ok] {len(scores)} segmentos -> {out}")


if __name__ == "__main__":
    main()
