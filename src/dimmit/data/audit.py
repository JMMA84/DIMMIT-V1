"""Auditoría del dataset: dHash por imagen, test de secuencialidad y casi-duplicados.

- dHash de 256 bits (17x16 en grises) por imagen -> data/index/dhash.parquet (h0..h3)
- Secuencialidad: distancia Hamming entre fotogramas consecutivos (orden de nombre) vs pares
  aleatorios. En RDD2020 son iguales: los nombres NO siguen el recorrido del vehículo.
- Casi-duplicados (Hamming <= dup_hamming dentro de cada país) agrupados con union-find:
  cada clúster se fuerza a un único split. Con 64 bits el umbral encadenaba miles de vías
  parecidas en un solo clúster; con 256 bits y umbral 16 quedan solo duplicados reales.
- Perfil del dataset -> reports/evaluation/perfil_dataset.csv
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from dimmit.utils.io import EVAL, INDEX, ensure_dir, load_yaml, save_json


HASH_COLS = ["h0", "h1", "h2", "h3"]


def dhash(path) -> np.ndarray:
    """dHash de 256 bits como 4 enteros uint64."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits).view(">u8").astype(np.uint64)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancia Hamming entre arreglos de hashes (..., 4) con broadcasting."""
    return np.bitwise_count(np.bitwise_xor(a, b)).sum(-1).astype(np.int16)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, i):
        root = i
        while self.p[root] != root:
            root = self.p[root]
        while self.p[i] != root:
            self.p[i], i = root, self.p[i]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_clusters(hashes: np.ndarray, thr: int, chunk=2000) -> np.ndarray:
    """Etiqueta de clúster por imagen (comparación todos-contra-todos por bloques)."""
    n = len(hashes)
    uf = UnionFind(n)
    for s in range(0, n, chunk):
        d = hamming(hashes[s : s + chunk, None, :], hashes[None, :, :])
        ii, jj = np.nonzero(d <= thr)
        for i, j in zip(ii + s, jj):
            if i < j:
                uf.union(i, j)
    return np.array([uf.find(i) for i in range(n)])


def main():
    cfg = load_yaml("configs/data_rdd2020.yaml")
    images = pd.read_parquet(INDEX / "images.parquet")
    out = INDEX / "dhash.parquet"
    if out.exists():
        hashes = pd.read_parquet(out)
        print("[skip] dhash ya calculado")
    else:
        with ProcessPoolExecutor(4) as ex:
            hs = list(ex.map(dhash, images["path"], chunksize=256))
        hashes = pd.DataFrame(np.stack(hs), columns=HASH_COLS)
        hashes.insert(0, "image_id", images["image_id"].to_numpy())
        hashes.to_parquet(out, index=False)
    df = images.merge(hashes, on="image_id")

    rng = np.random.default_rng(cfg["seed"])
    stats, clusters = {}, []
    for (source, country), g in df.sort_values("frame_num").groupby(["source", "country"]):
        h = g[HASH_COLS].to_numpy(dtype=np.uint64)
        consec = hamming(h[1:], h[:-1])
        rand = hamming(h[rng.integers(0, len(h), 4000)], h[rng.integers(0, len(h), 4000)])
        stats[f"{source}/{country}"] = {
            "n": int(len(h)),
            "hamming_consecutivo_mediana": float(np.median(consec)),
            "hamming_aleatorio_mediana": float(np.median(rand)),
            "secuencial": bool(np.median(consec) < 0.5 * np.median(rand)),
        }
    # clústeres de casi-duplicados por país (train + test juntos: comparten secuencias)
    for country, g in df.groupby("country"):
        lab = duplicate_clusters(g[HASH_COLS].to_numpy(dtype=np.uint64), cfg["dup_hamming"])
        clusters.append(pd.DataFrame({"image_id": g["image_id"].to_numpy(), "dup_cluster": [f"{country}-{x}" for x in lab]}))
    clusters = pd.concat(clusters)
    sizes = clusters["dup_cluster"].value_counts()
    clusters.to_parquet(INDEX / "dup_clusters.parquet", index=False)
    stats["duplicados"] = {
        "clusters_multi": int((sizes > 1).sum()),
        "imagenes_en_clusters_multi": int(sizes[sizes > 1].sum()),
        "max_cluster": int(sizes.max()),
    }
    save_json(stats, INDEX / "audit.json")
    for k, v in stats.items():
        print(k, v)

    # perfil del dataset
    rows = []
    for (source, country), g in images.groupby(["source", "country"]):
        r = {"fuente": source, "pais": country, "imagenes": len(g), "con_dano": int(g["has_damage"].sum()) if source == "train" else None}
        for k in ("D00", "D10", "D20", "D40"):
            r[f"cajas_{k}"] = int(g[f"n_{k}"].sum()) if source == "train" else None
        r["cajas_otras_clases"] = int(g["n_otros"].sum()) if source == "train" else None
        r["ancho_mediano"] = int(g["width"].median())
        rows.append(r)
    ensure_dir(EVAL)
    pd.DataFrame(rows).to_csv(EVAL / "perfil_dataset.csv", index=False)
    print(f"[ok] {EVAL / 'perfil_dataset.csv'}")


if __name__ == "__main__":
    main()
