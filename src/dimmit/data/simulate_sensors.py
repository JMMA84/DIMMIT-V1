"""Genera los datos de sensores (simulados, v2 físico) de los segmentos de fusión, test y
despliegue sobre calzadas reales de Bogotá.

Mientras no existan los sensores físicos (Arduino + MPU-6050 + GPS en Raspberry Pi), este módulo
produce series con el esquema definitivo. El PCI que guía la rugosidad latente es:
  - fusión/test: el pseudo-PCI de las anotaciones humanas (acoplamiento parcial, ver config);
  - despliegue (sin anotaciones): PCI del detector + residuo muestreado de F en el mismo decil.
Salidas en data/simulated/ (o data/simulated_null/ con --null-world, control sin acoplamiento):
  sensors.parquet        series a 50 Hz (segment_id, obs_id, timestamp, ax..gz, lat, lon, speed_kmh)
  sensor_frames.parquet  hora y GPS de captura de cada fotograma
  sensors_truth.parquet  VERDAD (IRI latente/real, eventos); ninguna feature debe leerlo
"""
import argparse
import zlib

import numpy as np
import pandas as pd

from dimmit.geo.roads import assign_carriageways
from dimmit.labels import pseudo_pci as pp
from dimmit.sensors.simulator import severity_labels, simulate_segment
from dimmit.utils.io import CLASS_KEYS, DATA, FEATURES, INDEX, REPO_ROOT, ensure_dir, load_yaml


def survey_start(segment_id, cfg_ctx, seed):
    rng = np.random.default_rng([seed, zlib.crc32(segment_id.encode())])
    days = (pd.Timestamp(cfg_ctx["survey_end"]) - pd.Timestamp(cfg_ctx["survey_start"])).days
    day = pd.Timestamp(cfg_ctx["survey_start"]) + pd.Timedelta(days=int(rng.integers(0, days + 1)))
    return day + pd.Timedelta(seconds=int(rng.uniform(7, 17) * 3600))


def deployment_pci(splits, vision, targets, seed):
    """PCI de referencia para despliegue: PCI_det + residuo (PCI_vis - PCI_det) de F por decil."""
    fus = splits.loc[splits["split"] == "fusion", "image_id"]
    f = vision[vision["image_id"].isin(fus)].merge(targets[["image_id", "pci_vis"]], on="image_id")
    edges = np.unique(np.quantile(f["pci_det"], np.linspace(0, 1, 11)))
    f["dec"] = np.clip(np.searchsorted(edges, f["pci_det"], side="right") - 1, 0, len(edges) - 2)
    resid = {d: (g["pci_vis"] - g["pci_det"]).to_numpy() for d, g in f.groupby("dec")}
    dep = vision[vision["image_id"].isin(splits.loc[splits["split"] == "despliegue", "image_id"])].copy()
    dep["dec"] = np.clip(np.searchsorted(edges, dep["pci_det"], side="right") - 1, 0, len(edges) - 2)
    rng = np.random.default_rng(seed)
    dep["pci"] = [np.clip(p + rng.choice(resid.get(d, np.zeros(1))), 0, 100) for p, d in zip(dep["pci_det"], dep["dec"])]
    return dep.set_index("image_id")["pci"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--null-world", action="store_true", help="control: rugosidad sin relación con el daño visual")
    ap.add_argument("--splits", default="fusion,test,despliegue")
    args = ap.parse_args()
    cfg = load_yaml("configs/sensors.yaml")
    cfg_ctx = load_yaml("configs/context.yaml")
    cfg_lab = pp.load_cfg()
    cuts = pp.load_cutoffs()
    out_dir = ensure_dir(DATA / ("simulated_null" if args.null_world else "simulated"))
    wanted = args.splits.split(",")
    if args.null_world:
        wanted = [w for w in wanted if w != "despliegue"]

    splits = pd.read_csv(INDEX / "splits.csv")
    splits = splits[splits["split"].isin(wanted)]
    targets = pd.read_parquet(INDEX / "targets_frame.parquet")
    gt = pd.read_parquet(INDEX / "boxes.parquet")
    gt["sev"] = severity_labels(gt, cfg_lab, cuts)
    pci = targets.set_index("image_id")["pci_vis"]
    boxes = gt
    if "despliegue" in wanted:
        vf = FEATURES / "vision_frame.parquet"
        if not vf.exists():
            raise SystemExit("faltan features de visión (dimmit.models.score.vision) para simular despliegue")
        vision = pd.read_parquet(vf)
        pci = pd.concat([pci, deployment_pci(pd.read_csv(INDEX / "splits.csv"), vision, targets, cfg["seed"])])
        det_dir = REPO_ROOT / (FEATURES / "detector_source.txt").read_text().strip()
        taus = pd.read_json(FEATURES / "vision_thresholds.json")["tau"].to_dict()
        det = pd.read_parquet(det_dir / "detections.parquet")
        det["cls"] = det["class_id"].map(dict(enumerate(CLASS_KEYS)))
        det = det[det["conf"] >= det["cls"].map(taus)]
        det = det[det["image_id"].isin(splits.loc[splits["split"] == "despliegue", "image_id"])]
        det["sev"] = severity_labels(det, cfg_lab, cuts)
        boxes = pd.concat([gt, det[gt.columns]], ignore_index=True)
    boxes = boxes[boxes["cls"].isin(["D20", "D40"])]
    by_img = {k: g for k, g in boxes.groupby("image_id")}

    roads = assign_carriageways(pd.read_csv(INDEX / "splits.csv")["segment_id"]).set_index("segment_id")
    df = splits.sort_values(["segment_id", "chain_pos"])
    df["pci"] = df["image_id"].map(pci)
    df = df[df["pci"].notna()]
    all_rows, all_meta, all_truth = [], [], []
    coupling = 0.0 if args.null_world else 1.0
    for k, (seg_id, fr) in enumerate(df.groupby("segment_id", sort=True)):
        road = roads.loc[seg_id].copy()
        road["segment_id"] = seg_id
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(seg_id.encode()), int(args.null_world)])
        rows, meta, truth = simulate_segment(road, fr, by_img, cfg, rng, survey_start(seg_id, cfg_ctx, cfg["seed"]), coupling)
        all_rows.append(rows)
        all_meta.append(meta)
        all_truth.append(truth.assign(split=fr["split"].iloc[0]))
        if k % 250 == 0:
            print(f"  {k} segmentos simulados", flush=True)
    sensors = pd.concat(all_rows, ignore_index=True)
    meta = pd.concat(all_meta, ignore_index=True)
    truth = pd.concat(all_truth, ignore_index=True)
    sensors.to_parquet(out_dir / "sensors.parquet", index=False)
    meta.to_parquet(out_dir / "sensor_frames.parquet", index=False)
    truth.to_parquet(out_dir / "sensors_truth.parquet", index=False)
    lab = truth[truth["split"] != "despliegue"].groupby("segment_id")[["pci_referencia", "iri_real_segmento", "iri_latente"]].mean()
    print(f"[ok] {truth['segment_id'].nunique()} segmentos, {len(sensors)} muestras -> {out_dir}")
    print(f"     corr(PCI visual, IRI real) por segmento = {lab['pci_referencia'].corr(lab['iri_real_segmento']):.2f}")
    print(f"     IRI real por segmento: {lab['iri_real_segmento'].describe().round(2).to_dict()}")


if __name__ == "__main__":
    main()
