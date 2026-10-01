"""Prepara RDD2020 completo en formato YOLO (v1).

Etapas (idempotentes):
  extract  extrae una sola vez cada archivo (train/test1/test2, tar.gz o zip del espejo HF)
           a data/interim/rdd2020/{fuente}/{país}/{images,xmls}
  index    parsea todos los XML -> data/index/images.parquet y boxes.parquet, y escribe las
           etiquetas YOLO junto a las imágenes ({país}/labels/*.txt). Las imágenes sin clases
           objetivo quedan como negativas (etiqueta vacía).
  lists    con data/index/splits.csv escribe listas de imágenes por split y data.yaml
           (ultralytics acepta listas .txt, así no se copian imágenes).
"""
import argparse
import random
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from dimmit.utils.io import DATA, INDEX, ensure_dir, load_yaml

RAW = DATA / "raw"
INTERIM = DATA / "interim" / "rdd2020"
PROCESSED = DATA / "processed"
MEMBER_RE = re.compile(r"(?:^|/)(train|test1|test2)?/?(Japan|India|Czech)/(images|annotations/xmls)/([^/]+\.(?:jpg|xml))$")


def voc_objects(xml_path):
    """Lee un XML Pascal VOC -> (ancho, alto, [(clase, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    iw = float(size.findtext("width") or 0) if size is not None else 0.0
    ih = float(size.findtext("height") or 0) if size is not None else 0.0
    objs = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().upper()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        objs.append((name, *box))
    return iw, ih, objs


def voc_to_yolo_lines(xml_path, class_map, image_size=None):
    """Convierte un XML Pascal VOC a líneas YOLO (cls cx cy w h normalizados).

    Ignora clases fuera de class_map (RDD2020 trae otras como D01/D44). Si el XML declara
    tamaño 0, usa image_size=(ancho, alto) cuando se entrega.
    """
    iw, ih, objs = voc_objects(xml_path)
    if (iw <= 0 or ih <= 0) and image_size:
        iw, ih = image_size
    lines = []
    if iw <= 0 or ih <= 0:
        return lines
    for name, xmin, ymin, xmax, ymax in objs:
        if name not in class_map:
            continue
        xmin, ymin = max(0.0, xmin), max(0.0, ymin)
        xmax, ymax = min(iw, xmax), min(ih, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        cx, cy = (xmin + xmax) / 2 / iw, (ymin + ymax) / 2 / ih
        w, h = (xmax - xmin) / iw, (ymax - ymin) / ih
        lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _archives():
    found = []
    for src in ("train", "test1", "test2"):
        for ext in (".tar.gz", ".zip"):
            p = RAW / f"{src}{ext}"
            if p.exists():
                found.append((src, p))
                break
    return found


def extract_all():
    """Extrae cada archivo en una sola pasada, solo imágenes y XML de los 3 países."""
    for source, path in _archives():
        marker = INTERIM / source / ".done"
        if marker.exists():
            print(f"[skip] {source} ya extraído")
            continue
        print(f"[extract] {path.name} ...", flush=True)
        n = 0
        opener = tarfile.open(path, "r:*") if path.suffix != ".zip" else zipfile.ZipFile(path)
        with opener as arc:
            members = arc.getmembers() if isinstance(arc, tarfile.TarFile) else arc.infolist()
            for m in members:
                name = m.name if isinstance(arc, tarfile.TarFile) else m.filename
                hit = MEMBER_RE.search(name)
                if not hit:
                    continue
                _, country, kind, fname = hit.groups()
                sub = "images" if kind == "images" else "xmls"
                dest = INTERIM / source / country / sub / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                f = arc.extractfile(m) if isinstance(arc, tarfile.TarFile) else arc.open(m)
                if f is None:
                    continue
                dest.write_bytes(f.read())
                n += 1
        marker.write_text(str(n))
        print(f"[ok] {source}: {n} archivos")


def build_index(cfg):
    """Índice de imágenes y cajas + etiquetas YOLO junto a las imágenes."""
    class_map = cfg["voc_classes"]
    img_rows, box_rows = [], []
    for source in cfg["sources"]:
        for country in cfg["countries"]:
            img_dir = INTERIM / source / country / "images"
            if not img_dir.exists():
                continue
            lbl_dir = ensure_dir(INTERIM / source / country / "labels")
            for img in sorted(img_dir.glob("*.jpg")):
                xml = INTERIM / source / country / "xmls" / (img.stem + ".xml")
                with Image.open(img) as im:
                    w_img, h_img = im.size
                row = {
                    "image_id": img.stem,
                    "source": source,
                    "country": country,
                    "path": str(img),
                    "width": w_img,
                    "height": h_img,
                    "frame_num": int(re.sub(r"\D", "", img.stem) or -1),
                    "labeled": xml.exists(),
                    "xml_zero_size": False,
                    "jpeg_kb": img.stat().st_size / 1024,
                }
                counts = {}
                if xml.exists():
                    iw, ih, objs = voc_objects(xml)
                    row["xml_zero_size"] = iw <= 0 or ih <= 0
                    for name, *_ in objs:
                        counts[name] = counts.get(name, 0) + 1
                    lines = voc_to_yolo_lines(xml, class_map, image_size=(w_img, h_img))
                    (lbl_dir / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
                    for ln in lines:
                        c, cx, cy, w, h = ln.split()
                        key = cfg["names"][int(c)].split("_")[0]
                        box_rows.append(
                            {"image_id": img.stem, "cls": key, "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)}
                        )
                for k in ("D00", "D10", "D20", "D40"):
                    row[f"n_{k}"] = counts.get(k, 0)
                row["n_otros"] = sum(v for k, v in counts.items() if k not in class_map)
                row["otros_clases"] = ",".join(sorted(k for k in counts if k not in class_map))
                row["has_damage"] = any(row[f"n_{k}"] > 0 for k in ("D00", "D10", "D20", "D40"))
                img_rows.append(row)
        print(f"[ok] índice {source}: {sum(r['source'] == source for r in img_rows)} imágenes", flush=True)
    ensure_dir(INDEX)
    images = pd.DataFrame(img_rows)
    boxes = pd.DataFrame(box_rows, columns=["image_id", "cls", "cx", "cy", "w", "h"])
    images.to_parquet(INDEX / "images.parquet", index=False)
    boxes.to_parquet(INDEX / "boxes.parquet", index=False)
    print(f"[ok] {len(images)} imágenes, {len(boxes)} cajas -> {INDEX}")
    return images, boxes


def write_lists(cfg, splits_path=None, smoke=0):
    """Listas de imágenes por split (train/val/fusion/test) y data.yaml para ultralytics."""
    splits = pd.read_csv(splits_path or INDEX / "splits.csv")
    images = pd.read_parquet(INDEX / "images.parquet")[["image_id", "path", "has_damage"]]
    df = splits.merge(images, on="image_id", how="inner")
    rng = random.Random(cfg["seed"])
    ensure_dir(PROCESSED)
    lists = {}
    tr = df[df["split"] == "train"]
    pos, neg = tr[tr["has_damage"]], tr[~tr["has_damage"]]
    max_neg = int(cfg["max_negative_share"] * len(pos) / (1 - cfg["max_negative_share"]))
    neg = neg.sample(n=min(len(neg), max_neg), random_state=cfg["seed"])
    lists["train"] = sorted(pd.concat([pos, neg])["path"])
    fus = sorted(df.loc[df["split"] == "fusion", "path"])
    lists["fusion"] = fus
    lists["val"] = sorted(rng.sample(fus, min(cfg["val_subset_frames"], len(fus))))
    lists["test"] = sorted(df.loc[df["split"] == "test", "path"])
    lists["despliegue"] = sorted(df.loc[df["split"] == "despliegue", "path"])
    lists["all"] = sorted(df["path"])
    if smoke:
        lists = {k: sorted(rng.sample(v, min(smoke, len(v)))) for k, v in lists.items()}
    for k, v in lists.items():
        (PROCESSED / f"{k}.txt").write_text("\n".join(v) + "\n")
    data_yaml = {"path": str(PROCESSED), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": cfg["names"]}
    with open(PROCESSED / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    # una data.yaml por país para métricas de test por país
    for country in cfg["countries"]:
        sub = [p for p in lists["test"] if f"/{country}/" in p]
        (PROCESSED / f"test_{country}.txt").write_text("\n".join(sub) + "\n")
        with open(PROCESSED / f"data_{country}.yaml", "w") as f:
            yaml.safe_dump({**data_yaml, "test": f"test_{country}.txt", "val": f"test_{country}.txt"}, f, sort_keys=False)
    print("[ok] listas: " + ", ".join(f"{k}={len(v)}" for k, v in lists.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="extract,index")
    ap.add_argument("--splits", default=None, help="splits.csv (por defecto data/index/splits.csv)")
    ap.add_argument("--smoke", type=int, default=0, help="limita cada lista a N imágenes")
    args = ap.parse_args()
    cfg = load_yaml("configs/data_rdd2020.yaml")
    stages = args.stages.split(",")
    if "extract" in stages:
        extract_all()
    if "index" in stages:
        build_index(cfg)
    if "lists" in stages:
        write_lists(cfg, args.splits, args.smoke)


if __name__ == "__main__":
    main()
