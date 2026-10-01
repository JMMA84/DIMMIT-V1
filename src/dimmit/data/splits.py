"""Segmentos, bloques y splits sin fuga + objetivos visuales por fotograma.

1. Pseudo-PCI por fotograma desde las cajas anotadas (labels/pseudo_pci.py); los cortes de
   severidad se fijan con todas las cajas anotadas y se guardan en data/index.
2. Pseudo-rutas: dentro de cada (fuente, país) se encadenan fotogramas por vecino visual más
   cercano (dHash 256 bits, recorrido voraz) y la cadena se corta cada 10 -> segmentos.
3. Bloques de 5 segmentos consecutivos de la cadena; los bloques que comparten un clúster de
   casi-duplicados se fusionan. El bloque es la unidad de agrupación del split.
4. StratifiedGroupKFold(20) estratificado por país x tercil de condición del bloque:
   13 folds -> train (detector), 4 -> fusion (red de fusión, F), 3 -> test (T).
   test1/test2 (sin anotaciones) -> despliegue.
Salidas: data/index/splits.csv, data/index/targets_frame.parquet, data/index/split_report.json
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dimmit.data.audit import HASH_COLS, UnionFind, hamming
from dimmit.labels import pseudo_pci as pp
from dimmit.utils.io import INDEX, load_yaml, save_json


def greedy_chain(hashes: np.ndarray, start: int = 0) -> np.ndarray:
    """Orden voraz por vecino más cercano (aprox. de una ruta continua)."""
    n = len(hashes)
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    cur = start
    for k in range(n):
        order[k] = cur
        visited[cur] = True
        if k == n - 1:
            break
        d = hamming(hashes, hashes[cur][None, :]).astype(np.int32)
        d[visited] = 1 << 20
        cur = int(np.argmin(d))
    return order


def frame_targets(images, boxes, cfg_lab):
    labeled = images[images["labeled"]]
    cuts = pp.severity_cutoffs(boxes, cfg_lab)
    pp.save_cutoffs(cuts)
    rho = pp.frame_rho(boxes, labeled["image_id"], cfg_lab, cuts)
    tgt = pp.pci_from_rho(rho, cfg_lab).reset_index()
    tgt.to_parquet(INDEX / "targets_frame.parquet", index=False)
    return tgt


def build_segments(df, cfg):
    rows = []
    for (source, country), g in df.groupby(["source", "country"], sort=False):
        g = g.sort_values("frame_num").reset_index(drop=True)
        order = greedy_chain(g[HASH_COLS].to_numpy(dtype=np.uint64))
        g = g.iloc[order].reset_index(drop=True)
        seg_size = cfg["segment_max_frames"]
        prefix = f"{source[:2].upper()}-{country[:2].upper()}"
        g["chain_pos"] = np.arange(len(g))
        g["segment_id"] = [f"{prefix}-{k // seg_size:04d}" for k in range(len(g))]
        n_seg = (len(g) + seg_size - 1) // seg_size
        seg_num = np.arange(len(g)) // seg_size
        g["block_id"] = [f"{prefix}-B{s // cfg['block_segments']:03d}" for s in seg_num]
        rows.append(g)
        print(f"[ok] {source}/{country}: {len(g)} fotogramas, {n_seg} segmentos")
    return pd.concat(rows, ignore_index=True)


def merge_blocks_by_duplicates(df):
    blocks = df["block_id"].unique()
    pos = {b: i for i, b in enumerate(blocks)}
    uf = UnionFind(len(blocks))
    for _, g in df.groupby("dup_cluster"):
        bs = g["block_id"].unique()
        for b in bs[1:]:
            uf.union(pos[bs[0]], pos[b])
    return df["block_id"].map(lambda b: blocks[uf.find(pos[b])])


def main():
    cfg = load_yaml("configs/data_rdd2020.yaml")
    cfg_lab = pp.load_cfg()
    images = pd.read_parquet(INDEX / "images.parquet")
    boxes = pd.read_parquet(INDEX / "boxes.parquet")
    hashes = pd.read_parquet(INDEX / "dhash.parquet")
    dups = pd.read_parquet(INDEX / "dup_clusters.parquet")

    tgt = frame_targets(images, boxes, cfg_lab)
    df = images.merge(hashes, on="image_id").merge(dups, on="image_id")
    df = build_segments(df, cfg)
    df["group_id"] = merge_blocks_by_duplicates(df)
    df = df.merge(tgt[["image_id", "pci_vis"]], on="image_id", how="left")

    lab = df[df["source"] == cfg["labeled_source"]].copy()
    grp_pci = lab.groupby("group_id")["pci_vis"].mean()
    terc = pd.qcut(grp_pci, 3, labels=False, duplicates="drop")
    lab["strata"] = lab["country"] + "_" + lab["group_id"].map(terc).astype(str)
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    lab["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(lab, lab["strata"], groups=lab["group_id"])):
        lab.iloc[te, lab.columns.get_loc("fold")] = k
    a, b = cfg["folds_train"], cfg["folds_train"] + cfg["folds_fusion"]
    lab["split"] = np.where(lab["fold"] < a, "train", np.where(lab["fold"] < b, "fusion", "test"))

    dep = df[df["source"] != cfg["labeled_source"]].copy()
    dep["fold"], dep["split"], dep["strata"] = -1, "despliegue", ""
    out = pd.concat([lab, dep], ignore_index=True)
    cols = ["image_id", "source", "country", "segment_id", "block_id", "group_id", "dup_cluster", "chain_pos", "fold", "split"]
    out[cols].to_csv(INDEX / "splits.csv", index=False)

    # reporte de balance y fugas
    rep = {"frames": out["split"].value_counts().to_dict()}
    rep["segmentos"] = out.groupby("split")["segment_id"].nunique().to_dict()
    rep["fuga_grupos"] = int((out[out.split != "despliegue"].groupby("group_id")["split"].nunique() > 1).sum())
    rep["fuga_duplicados"] = int((lab.groupby("dup_cluster")["split"].nunique() > 1).sum())
    rep["fuga_segmentos"] = int((out.groupby("segment_id")["split"].nunique() > 1).sum())
    share = lab.groupby("split")["strata"].value_counts(normalize=True).unstack(0).fillna(0)
    rep["desbalance_estratos_pp"] = float(100 * (share.max(axis=1) - share.min(axis=1)).max())
    seg_pci = lab.groupby(["split", "segment_id"])["pci_vis"].mean()
    rep["pci_vis_segmento"] = {s: seg_pci[s].describe().round(1).to_dict() for s in ("train", "fusion", "test")}
    states = pd.Series(pp.state_of(seg_pci.to_numpy(), cfg_lab)).value_counts(normalize=True).round(3)
    rep["estados_pci_vis_segmento"] = states.to_dict()
    save_json(rep, INDEX / "split_report.json")
    for k, v in rep.items():
        print(k, v)


if __name__ == "__main__":
    main()
