"""Descarga la carpeta pública de Drive con la toma de campo y deja un manifiesto con hash.

No usa la API de Google: lista cada carpeta con `embeddedfolderview` y baja los archivos con
`uc?export=download`. Estructura esperada:
    grabacion_<fecha>_<hora>_Prueba_N/   captura_*.jpg + <track>.gpx + <track>.kml
    tomadedatos/                          Tiempo_ms,Distancia_cm,Profundidad*.txt
Salida: data/campo/raw/<carpeta>/... y data/campo/index/manifiesto.csv
"""
import argparse
import hashlib
import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

from dimmit.campo import INDEX_CAMPO, RAW, load_cfg
from dimmit.utils.io import ensure_dir

LIST_URL = "https://drive.google.com/embeddedfolderview?id={id}#list"
FILE_URL = "https://drive.google.com/uc?export=download&id={id}"
ENTRY_RE = re.compile(r'<div class="flip-entry" id="entry-([^"]+)".*?<div class="flip-entry-title">(.*?)</div>', re.S)
HEADERS = {"User-Agent": "Mozilla/5.0 (DIMMIT campo ingest)"}


def _get(url, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r
            msg = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # red: reintento con backoff
            msg = str(e)
        time.sleep(2**k)
    raise RuntimeError(f"no pude descargar {url}: {msg}")


def list_folder(folder_id):
    """[(id, nombre)] de una carpeta pública; las subcarpetas no traen extensión."""
    page = _get(LIST_URL.format(id=folder_id)).text
    return [(i, html.unescape(n).strip()) for i, n in ENTRY_RE.findall(page)]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_tree(folder_id, dest: Path, rows, prefix=""):
    ensure_dir(dest)
    for fid, name in list_folder(folder_id):
        if "." not in name:  # subcarpeta
            download_tree(fid, dest / name, rows, prefix=f"{prefix}{name}/")
            continue
        target = dest / name
        if not target.exists() or target.stat().st_size == 0:
            target.write_bytes(_get(FILE_URL.format(id=fid)).content)
        rows.append({"archivo": f"{prefix}{name}", "id_drive": fid, "sha1": sha1(target), "bytes": target.stat().st_size})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default=None, help="id de la carpeta de Drive (por defecto configs/campo.yaml)")
    args = ap.parse_args()
    cfg = load_cfg()
    rows = []
    download_tree(args.folder or cfg["drive_folder_id"], RAW, rows)
    man = pd.DataFrame(rows).sort_values("archivo")
    ensure_dir(INDEX_CAMPO)
    man.to_csv(INDEX_CAMPO / "manifiesto.csv", index=False)
    n_img = man["archivo"].str.endswith(".jpg").sum()
    print(f"[ok] {len(man)} archivos ({n_img} imágenes, {man['bytes'].sum() / 1e6:.1f} MB) -> {RAW}")


if __name__ == "__main__":
    main()
