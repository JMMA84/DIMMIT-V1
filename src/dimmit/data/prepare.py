"""Prepara el subset pequeño de RDD2020 en formato YOLO.

Extrae solo el país configurado de train.tar.gz, muestrea n_images con al menos
una anotación válida, convierte Pascal VOC XML -> txt YOLO y hace split train/val.
Salida: data/processed/{images,labels}/{train,val} + data/processed/data.yaml
"""
import random
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from dimmit.utils.io import REPO_ROOT, load_yaml

RAW_TAR = REPO_ROOT / "data/raw/train.tar.gz"
INTERIM = REPO_ROOT / "data/interim"
PROCESSED = REPO_ROOT / "data/processed"


def voc_to_yolo_lines(xml_path, class_map):
    """Convierte un XML Pascal VOC a líneas YOLO (cls cx cy w h normalizados).

    Ignora clases fuera de class_map (RDD2020 trae otras como D01/D44).
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    iw, ih = float(size.findtext("width")), float(size.findtext("height"))
    lines = []
    if iw <= 0 or ih <= 0:
        return lines
    for obj in root.iter("object"):
        name = obj.findtext("name")
        if name not in class_map:
            continue
        bb = obj.find("bndbox")
        xmin = max(0.0, float(bb.findtext("xmin")))
        ymin = max(0.0, float(bb.findtext("ymin")))
        xmax = min(iw, float(bb.findtext("xmax")))
        ymax = min(ih, float(bb.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx, cy = (xmin + xmax) / 2 / iw, (ymin + ymax) / 2 / ih
        w, h = (xmax - xmin) / iw, (ymax - ymin) / ih
        lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def extract_country(country):
    """Extrae imágenes y XML del país indicado a data/interim (idempotente)."""
    dest = INTERIM / country
    if dest.exists() and any(dest.rglob("*.xml")):
        print(f"[skip] {dest} ya extraído")
        return dest
    if not RAW_TAR.exists():
        raise SystemExit(f"No existe {RAW_TAR}; corre primero `make download-data`")
    print(f"[extract] {country} desde {RAW_TAR.name} (tarda unos minutos)...")
    prefix = f"train/{country}/"
    with tarfile.open(RAW_TAR, "r:gz") as tar:
        members = [m for m in tar if m.name.startswith(prefix)]
        tar.extractall(INTERIM, members=members)
    extracted = INTERIM / "train" / country
    if extracted.exists():
        extracted.rename(dest)
    shutil.rmtree(INTERIM / "train", ignore_errors=True)
    return dest


def main():
    cfg = load_yaml("configs/data_rdd2020.yaml")
    class_map, names = cfg["voc_classes"], cfg["names"]
    country_dir = extract_country(cfg["country"])

    xml_dir = country_dir / "annotations" / "xmls"
    img_dir = country_dir / "images"
    candidates = []
    for xml in sorted(xml_dir.glob("*.xml")):
        img = img_dir / (xml.stem + ".jpg")
        if img.exists() and voc_to_yolo_lines(xml, class_map):
            candidates.append((img, xml))
    print(f"[info] {len(candidates)} imágenes con anotaciones válidas en {cfg['country']}")

    rng = random.Random(cfg["seed"])
    sample = rng.sample(candidates, min(cfg["n_images"], len(candidates)))
    n_val = max(1, int(len(sample) * cfg["val_fraction"]))
    splits = {"val": sample[:n_val], "train": sample[n_val:]}

    for split, items in splits.items():
        for sub in ("images", "labels"):
            d = PROCESSED / sub / split
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        for img, xml in items:
            shutil.copy2(img, PROCESSED / "images" / split / img.name)
            lines = voc_to_yolo_lines(xml, class_map)
            (PROCESSED / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines) + "\n")
        print(f"[ok] {split}: {len(items)} imágenes")

    data_yaml = {
        "path": str(PROCESSED),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with open(PROCESSED / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"[ok] {PROCESSED / 'data.yaml'}")


if __name__ == "__main__":
    main()
