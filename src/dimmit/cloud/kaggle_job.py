"""Orquesta el entrenamiento del detector en Kaggle (GPU) desde este contenedor.

  bundle  crea/actualiza el dataset privado <usuario>/dimmit-bundle (src + configs + splits.csv)
  push    sube el kernel <usuario>/dimmit-yolov10-rdd2020 (GPU T4, internet) y lo ejecuta
  status  estado del kernel
  pull    descarga las salidas del kernel a models/kaggle/<kernel>/

Credencial: ~/.kaggle/access_token (o KAGGLE_API_TOKEN); nunca se escribe en el repo.
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from dimmit.utils.io import INDEX, REPO_ROOT, ensure_dir

BUILD = REPO_ROOT / "build" / "kaggle"
KERNEL_SLUG = "dimmit-yolov10-rdd2020"
BUNDLE_SLUG = "dimmit-bundle"
RAW_SLUG = "rdd2020-raw"
CAMPO_SLUG = "dimmit-campo"                 # fotos de campo (privado)
INFER_SLUG = "dimmit-campo-inferencia"      # kernel de solo inferencia
KAGGLE = [str(Path(sys.executable).parent / "kaggle")]


def kaggle(*args, capture=False):
    cmd = KAGGLE + list(args)
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    return subprocess.run(cmd).returncode


def username():
    out = kaggle("config", "view", capture=True)
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("no pude resolver el usuario de Kaggle (¿token en ~/.kaggle/access_token?)")


def build_bundle():
    """Arma build/kaggle/bundle/dimmit_bundle.zip (código + configs + splits)."""
    d = BUILD / "bundle"
    shutil.rmtree(d, ignore_errors=True)
    ensure_dir(d)
    with zipfile.ZipFile(d / "dimmit_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for base in ("src", "configs"):
            for p in (REPO_ROOT / base).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(REPO_ROOT))
        z.write(INDEX / "splits.csv", "data/index/splits.csv")
        # sin internet en el kernel: wheels de ultralytics y pesos preentrenados van en el bundle
        for whl in (BUILD / "wheels").glob("*.whl"):
            z.write(whl, f"wheels/{whl.name}")
        for w in ("yolov10n.pt", "yolo11n.pt"):
            if (REPO_ROOT / "models" / w).exists():
                z.write(REPO_ROOT / "models" / w, f"models/{w}")
        rel = REPO_ROOT / "models/release/yolov10n_rdd2020.pt"  # detector entrenado: lo usa el modo infer
        if rel.exists():
            z.write(rel, "models/release/yolov10n_rdd2020.pt")
    return d


def upload_dataset(slug, d, message):
    user = username()
    meta = {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = slug in kaggle("datasets", "list", "--mine", capture=True)
    if exists:
        rc = kaggle("datasets", "version", "-p", str(d), "-m", message, "-q", "--dir-mode", "zip")
    else:
        rc = kaggle("datasets", "create", "-p", str(d), "-q", "--dir-mode", "zip")
    print(kaggle("datasets", "status", f"{user}/{slug}", capture=True))
    return rc


def bundle(message="actualización"):
    return upload_dataset(BUNDLE_SLUG, build_bundle(), message)


def campo_dataset(message="toma de campo"):
    """Sube data/campo/raw (fotos, GPX, KML y lecturas) como dataset privado dimmit-campo."""
    src = REPO_ROOT / "data/campo/raw"
    if not src.exists():
        raise SystemExit("primero: make campo-ingest")
    d = BUILD / "campo"
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d)
    return upload_dataset(CAMPO_SLUG, d, message)


def push(infer=False):
    user = username()
    slug = INFER_SLUG if infer else KERNEL_SLUG
    d = ensure_dir(BUILD / ("kernel_infer" if infer else "kernel"))
    code = (REPO_ROOT / "src/dimmit/cloud/kaggle_kernel.py").read_text()
    if infer:  # el kernel no recibe variables de entorno: se fija el modo al inicio del script
        code = code.replace("import glob\n", "import os\nos.environ['DIMMIT_MODE'] = 'infer'\nimport glob\n", 1)
    (d / "kaggle_kernel.py").write_text(code)
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "kaggle_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{user}/{BUNDLE_SLUG}", f"{user}/{CAMPO_SLUG}" if infer else f"{user}/{RAW_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return kaggle("kernels", "push", "-p", str(d))


def status(infer=False):
    out = kaggle("kernels", "status", f"{username()}/{INFER_SLUG if infer else KERNEL_SLUG}", capture=True)
    print(out.strip())
    return out


def pull(infer=False):
    if infer:
        dest = ensure_dir(REPO_ROOT / "reports" / "campo" / "kaggle")
        return kaggle("kernels", "output", f"{username()}/{INFER_SLUG}", "-p", str(dest), "-o")
    dest = ensure_dir(REPO_ROOT / "models" / "kaggle" / KERNEL_SLUG)
    return kaggle("kernels", "output", f"{username()}/{KERNEL_SLUG}", "-p", str(dest), "-o")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "bundle", "push", "status", "pull", "campo-dataset", "infer-push", "infer-status", "infer-pull"])
    ap.add_argument("-m", "--message", default="actualización")
    args = ap.parse_args()
    fn = {"build": build_bundle, "bundle": lambda: bundle(args.message), "push": push, "status": status, "pull": pull,
          "campo-dataset": lambda: campo_dataset(args.message), "infer-push": lambda: push(infer=True),
          "infer-status": lambda: status(infer=True), "infer-pull": lambda: pull(infer=True)}[args.cmd]
    rc = fn()
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
