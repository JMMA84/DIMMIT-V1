"""Líneas base contra las que se mide la red de fusión (mismos conjuntos F -> T).

  B0  score v0 por reglas (RuleBasedScorer de models/score/model.py, sin cambios).
  B1  regla física calibrada: PCI_det (fórmulas del pseudo-PCI sobre detecciones) + índice de
      rodadura con IRI_hat = k * rms/sqrt(v) (k ajustado en F). Es la barra a superar.
  B2  gradient boosting (HistGradientBoosting) sobre las mismas features por fotograma
      (implementa el antiguo placeholder LearnedScorer).
  C1  canario de contexto: gradient boosting SOLO con contexto de Bogotá; debe dar R2 ~ 0.
Salida: data/features/predicciones/<baseline>__{test,despliegue}_segmento.parquet
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from dimmit.labels import pseudo_pci as pp
from dimmit.models.score.dataset import build_tables, feature_groups, segment_truth
from dimmit.models.score.features import build_features
from dimmit.models.score.model import RuleBasedScorer
from dimmit.utils.io import CLASS_KEYS, DATA, FEATURES, REPO_ROOT, ensure_dir, load_yaml

OUT = FEATURES / "predicciones"


def b0_rule_v0(frame):
    """Score v0 tal cual (umbral de confianza y pesos de configs/score.yaml)."""
    cfg = load_yaml("configs/score.yaml")
    names = load_yaml("configs/data_rdd2020.yaml")["names"]
    det_dir = REPO_ROOT / (FEATURES / "detector_source.txt").read_text().strip()
    det = pd.read_parquet(det_dir / "detections.parquet").rename(columns={"image_id": "image"})
    segments = frame[["image_id", "segment_id"]].rename(columns={"image_id": "image"})
    sensors = pd.read_parquet(DATA / "simulated" / "sensors.parquet", columns=["segment_id", "az"])
    sensors = sensors[sensors["segment_id"].isin(segments["segment_id"])]
    feats = build_features(det, segments, sensors, names, cfg["conf_threshold"])
    sc = RuleBasedScorer(cfg).score(feats).set_index("segment_id")
    return pd.DataFrame({"icv": sc["score"]})


def b1_physics(frame, cfg_lab):
    imu_seg = pd.read_parquet(FEATURES / "imu_segment.parquet").set_index("segment_id")
    truth = segment_truth(frame)
    f_ids = frame.loc[frame["split"] == "fusion", "segment_id"].unique()
    x = imu_seg.loc[f_ids, "imu_rms_sqrtv"]
    y = truth.loc[f_ids, "iri"]
    k = float(np.exp(np.median(np.log(y) - np.log(x))))  # ajuste robusto en escala log
    g = frame.groupby("segment_id")
    rho = g[[f"rho_det_{c}" for c in CLASS_KEYS]].mean()
    rho.columns = [f"rho_{c}" for c in CLASS_KEYS]
    seg = pp.pci_from_rho(rho, cfg_lab)
    seg["iri"] = k * imu_seg.loc[seg.index, "imu_rms_sqrtv"]
    seg["ride"] = pp.ride_index(seg["iri"], cfg_lab)
    seg["icv"] = pp.icv(seg["pci_vis"], seg["ride"], cfg_lab)
    return seg, k


def gbm_frame(frame, cols, target="icv"):
    f = frame[frame["split"] == "fusion"]
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42).fit(f[cols], f[target])
    out = frame[["segment_id", "split"]].copy()
    out["pred"] = m.predict(frame[cols])
    return out.groupby("segment_id").agg(icv=("pred", "mean"), split=("split", "first"))


def main():
    cfg_lab = pp.load_cfg()
    frame = build_tables()
    frame = frame[frame["split"] != "train"]  # sin sensores ni objetivos de fusión en el train del detector
    groups = feature_groups(frame)
    ensure_dir(OUT)
    res = {}
    res["B0_reglas_v0"] = b0_rule_v0(frame)
    b1, k = b1_physics(frame, cfg_lab)
    res["B1_fisico_calibrado"] = b1
    print(f"[ok] B1: IRI_hat = {k:.3f} * rms/sqrt(v)")
    res["B2_gradient_boosting"] = gbm_frame(frame, groups["vision"] + groups["embedding"] + groups["imu"])
    ctx = pd.read_parquet(FEATURES / "context_segment.parquet")
    ctx_cols = ["dist_colegio_m", "colegio_300m", "dist_ips_m", "dist_hospital_m", "siniestros_250m", "lluvia_30d_mm", "pendiente_pct", "ancho_m", "dim_social", "dim_tecnica", "ip_umv", "elevacion_m"]
    fc = frame.merge(ctx[["segment_id", *ctx_cols]], on="segment_id", how="left")
    res["C1_canario_contexto"] = gbm_frame(fc, ctx_cols)
    split_of = frame.groupby("segment_id")["split"].first()
    for name, seg in res.items():
        for split in ("test", "despliegue"):
            ids = split_of.index[split_of == split]
            sub = seg.loc[seg.index.intersection(ids)]
            if len(sub):
                sub.drop(columns=["split"], errors="ignore").to_parquet(OUT / f"{name}__{split}_segmento.parquet")
    print(f"[ok] líneas base -> {OUT}")


if __name__ == "__main__":
    main()
