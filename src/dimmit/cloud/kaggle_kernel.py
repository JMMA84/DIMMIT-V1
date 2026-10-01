"""Script que corre DENTRO del kernel de Kaggle (GPU). Autocontenido salvo el bundle.

1. Ubica el bundle (dataset privado <usuario>/dimmit-bundle: código + configs + splits.csv).
2. Descarga RDD2020 (Mendeley; espejo Hugging Face si falla) y lo prepara con los mismos splits.
3. Entrena YOLOv10n, evalúa en test (global y por país), infiere TODAS las imágenes
   (detecciones + embeddings) y deja todo en /kaggle/working/outputs.
Con --smoke (o DIMMIT_SMOKE=N) corre la misma ruta con N imágenes por lista y 1 época, para
probarlo localmente antes de subirlo.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path(os.environ.get("DIMMIT_WORK", "/kaggle/working"))
ROOT = WORK / "dimmit"
OUT = WORK / "outputs"
SMOKE = int(os.environ.get("DIMMIT_SMOKE") or (40 if "--smoke" in sys.argv else 0))
LOG = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def sh(args, check=True, env=None):
    log("$ " + " ".join(map(str, args)))
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line:  # barras de progreso: solo al log
            LOG.write(line)
            continue
        print(line, end="", flush=True)
        LOG.write(line)
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f"falló ({p.returncode}): {' '.join(map(str, args))}")
    return p.returncode


def find_bundle():
    for pat in ("/kaggle/input/**/dimmit_bundle.zip", str(WORK.parent / "input/**/dimmit_bundle.zip")):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return Path(hits[0])
    for pat in ("/kaggle/input/**/src/dimmit/__init__.py",):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return Path(hits[0]).parents[2]
    raise FileNotFoundError("no encontré el bundle de DIMMIT en /kaggle/input")


def unpack(bundle):
    if ROOT.exists():
        shutil.rmtree(ROOT)
    if bundle.suffix == ".zip":
        with zipfile.ZipFile(bundle) as z:
            z.extractall(ROOT)
    else:
        shutil.copytree(bundle, ROOT)


def download(env):
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    if sh([sys.executable, "src/dimmit/data/download.py", "--all"], check=False, env=env) == 0:
        return
    log("Mendeley falló; uso el espejo de Hugging Face ShixuanAn/RDD_2020")
    from huggingface_hub import hf_hub_download

    for name in ("train.zip", "test1.zip", "test2.zip"):
        p = hf_hub_download("ShixuanAn/RDD_2020", name, repo_type="dataset")
        shutil.copy(p, raw / name)


def main():
    global LOG
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = open(OUT / "kernel_log.txt", "a")
    t0 = time.time()
    bundle = find_bundle() if not os.environ.get("DIMMIT_BUNDLE") else Path(os.environ["DIMMIT_BUNDLE"])
    log(f"bundle: {bundle}  smoke={SMOKE}")
    unpack(bundle)
    if not SMOKE:
        sh([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.253"], check=False)
    env = {**os.environ, "DIMMIT_ROOT": str(ROOT), "PYTHONPATH": str(ROOT / "src")}
    py = sys.executable
    if os.environ.get("DIMMIT_RAW"):  # prueba local: reutiliza los archivos ya descargados
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        os.symlink(os.environ["DIMMIT_RAW"], ROOT / "data/raw")
    else:
        download(env)
    sh([py, "-m", "dimmit.data.prepare", "--stages", "extract,index,lists", "--smoke", str(SMOKE)], env=env)
    cfg = "configs/train_kaggle.yaml"
    if SMOKE:
        smoke_cfg = (ROOT / cfg).read_text().replace("epochs: 120", "epochs: 1").replace("cache: ram", "cache: false")
        smoke_cfg = smoke_cfg.replace("batch: 32", "batch: 8").replace("imgsz: 640", "imgsz: 320").replace("workers: 4", "workers: 1")
        (ROOT / "configs/train_smoke.yaml").write_text(smoke_cfg)
        cfg = "configs/train_smoke.yaml"
    sh([py, "-m", "dimmit.models.detector.train", "--config", cfg], env=env)
    run = ROOT / "models/yolov10n_rdd2020"
    best = run / "weights/best.pt"
    log(f"entrenamiento listo en {(time.time() - t0) / 3600:.2f} h")
    imgsz = "320" if SMOKE else "640"
    sh([py, "-m", "dimmit.models.detector.evaluate", "--weights", str(best), "--out", str(OUT), "--imgsz", imgsz], check=False, env=env)
    sh([py, "-m", "dimmit.models.detector.predict", "--weights", str(best), "--list", "data/processed/all.txt", "--out", str(OUT), "--imgsz", imgsz, "--batch", "32"], env=env)
    for f in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"):
        if (run / f).exists():
            shutil.copy(run / f, OUT / Path(f).name)
    log(f"listo en {(time.time() - t0) / 3600:.2f} h")
    LOG.close()
    shutil.rmtree(ROOT, ignore_errors=True)  # no publicar datos como salida del kernel


if __name__ == "__main__":
    main()
