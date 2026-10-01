"""Tablas de modelado por fotograma y segmento: features (visión + IMU) y objetivos.

Objetivos (solo fusión/test): densidad ponderada por clase (de las cajas humanas), IRI real del
simulador por fotograma, y derivados con las fórmulas de labels/pseudo_pci.py (sub-scores, PCI
visual, índice de rodadura, ICV y estado). Las columnas de contexto se agregan aparte y nunca
forman parte de los grupos de entrada del modelo de condición.
"""
import numpy as np
import pandas as pd

from dimmit.labels import pseudo_pci as pp
from dimmit.utils.io import CLASS_KEYS, DATA, FEATURES, INDEX

ID_COLS = ["image_id", "segment_id", "split", "country", "chain_pos"]


def feature_groups(frame: pd.DataFrame) -> dict:
    """Columnas por grupo de entrada del modelo de condición."""
    vis = [
        c
        for c in frame.columns
        if c.startswith(("n_D", "soft_", "maxconf_", "nsub_", "area_", "cy_mean_", "union_", "rho_det_", "dv_det_", "rho_soft_"))
        or c in ("rodada_share", "entropia_clases", "max_conf", "pci_det", "pci_soft", "brillo", "contraste", "nitidez", "jpeg_kb")
    ]
    emb = [c for c in frame.columns if c.startswith("emb_pc")]
    imu = [c for c in frame.columns if c.startswith(("imu_", "gyro_", "vel_", "frac_detenido", "gps_"))]
    return {"vision": vis, "embedding": emb, "imu": imu}


def segment_targets(rho_frame: pd.DataFrame, iri_frame: pd.Series, segment_ids: pd.Series, cfg):
    """Objetivos de segmento: densidades promediadas -> PCI; IRI promedio -> rodadura -> ICV."""
    df = rho_frame.copy()
    df["iri"] = iri_frame.to_numpy()
    df["segment_id"] = segment_ids.to_numpy()
    g = df.groupby("segment_id").mean()
    pci = pp.pci_from_rho(g[[f"rho_{c}" for c in CLASS_KEYS]], cfg)
    out = pci[[f"sub_{c}" for c in CLASS_KEYS] + ["pci_vis"]].copy()
    out["iri"] = g["iri"]
    out["ride"] = pp.ride_index(g["iri"], cfg)
    out["icv"] = pp.icv(out["pci_vis"], out["ride"], cfg)
    out["estado"] = pp.state_of(out["icv"], cfg)
    for c in CLASS_KEYS:
        out[f"rho_{c}"] = g[f"rho_{c}"]
    return out


def build_tables(sim_dir=None, imu_dir=None):
    """Tabla por fotograma (features + objetivos) para fusión/test/despliegue."""
    cfg = pp.load_cfg()
    sim_dir = sim_dir or DATA / "simulated"
    imu_dir = imu_dir or FEATURES
    splits = pd.read_csv(INDEX / "splits.csv")
    vision = pd.read_parquet(FEATURES / "vision_frame.parquet")
    imu = pd.read_parquet(imu_dir / "imu_frame.parquet").drop(columns=["segment_id", "timestamp_captura", "lat", "lon"], errors="ignore")
    frame = splits[ID_COLS].merge(vision, on="image_id").merge(imu, on="image_id", how="left")
    targets = pd.read_parquet(INDEX / "targets_frame.parquet")
    truth = pd.read_parquet(sim_dir / "sensors_truth.parquet")[["image_id", "iri_real_fotograma"]]
    frame = frame.merge(targets, on="image_id", how="left").merge(truth, on="image_id", how="left")
    frame = frame.rename(columns={"iri_real_fotograma": "iri_real"})
    lab = frame["split"].isin(["fusion", "test"])
    frame.loc[lab, "ride"] = pp.ride_index(frame.loc[lab, "iri_real"], cfg)
    frame.loc[lab, "icv"] = pp.icv(frame.loc[lab, "pci_vis"], frame.loc[lab, "ride"], cfg)
    frame.loc[lab, "estado"] = pp.state_of(frame.loc[lab, "icv"], cfg)
    frame = frame.sort_values(["segment_id", "chain_pos"]).reset_index(drop=True)
    return frame


def segment_truth(frame):
    cfg = pp.load_cfg()
    lab = frame[frame["split"].isin(["fusion", "test"])]
    return segment_targets(lab[[f"rho_{c}" for c in CLASS_KEYS]], lab["iri_real"], lab["segment_id"], cfg)


def finite(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
