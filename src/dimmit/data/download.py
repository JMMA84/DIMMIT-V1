"""Descarga el dataset RDD2020 (Mendeley 5ty2wb6gvg, v1) a data/raw/.

Usa solo la librería estándar para poder ejecutarse antes de crear el venv.
Por defecto descarga train.tar.gz (el único con anotaciones) y FileStructure.txt;
con --all descarga también test1/test2.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

API_URL = "https://data.mendeley.com/public-api/datasets/5ty2wb6gvg/files?folder_id=root&version=1"
DEFAULT_FILES = ["train.tar.gz", "FileStructure.txt"]
CHUNK = 1 << 20  # 1 MB
# Mendeley devuelve 403 al user-agent por defecto de urllib
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _open(url, timeout):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def fetch_listing():
    with _open(API_URL, 60) as r:
        return json.load(r)


def download_file(entry, out_dir: Path):
    dest = out_dir / entry["filename"]
    size = entry["size"]
    if dest.exists() and dest.stat().st_size == size:
        print(f"[skip] {dest.name} ya está completo ({size / 1e6:.0f} MB)")
        return
    url = entry["content_details"]["download_url"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[down] {dest.name} ({size / 1e6:.0f} MB)")
    done = 0
    next_report = 0.05
    with _open(url, 120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if size and done / size >= next_report:
                print(f"       {dest.name}: {100 * done / size:.0f}%", flush=True)
                next_report += 0.05
    if done != size:
        raise RuntimeError(f"{dest.name}: se esperaban {size} bytes, llegaron {done}")
    tmp.rename(dest)
    print(f"[ok]   {dest.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw", type=Path)
    ap.add_argument("--all", action="store_true", help="incluye test1/test2 (sin anotaciones)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    listing = fetch_listing()
    wanted = None if args.all else DEFAULT_FILES
    for entry in listing:
        if wanted is None or entry["filename"] in wanted:
            download_file(entry, args.out)
    print("Descarga completa.")


if __name__ == "__main__":
    sys.exit(main())
