"""Script que corre DENTRO del kernel de Kaggle (GPU). Autocontenido salvo el bundle.

1. Ubica el bundle (dataset privado <usuario>/dimmit-bundle: código, configs, splits.csv,
   wheels de ultralytics y pesos preentrenados), así funciona aunque el kernel no tenga internet.
2. Toma RDD2020 del dataset privado <usuario>/rdd2020-raw (o lo descarga de Mendeley / espejo
   Hugging Face si hay internet) y lo prepara con los mismos splits.
   Sin GPU (cuenta sin verificar) usa una configuración de CPU con tope de tiempo.
3. Entrena YOLOv10n, evalúa en test (global y por país), infiere TODAS las imágenes
   (detecciones + embeddings) y deja todo en /kaggle/working/outputs.
Con --smoke (o DIMMIT_SMOKE=N) corre la misma ruta con N imágenes por lista y 1 época, para
probarlo localmente antes de subirlo.
Con DIMMIT_MODE=infer (kernel dimmit-campo-inferencia) NO entrena: toma los pesos de
models/release del bundle y las imágenes del dataset dimmit-campo, y deja solo detecciones.
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


def link_input_data():
    """Enlaza los archivos (o carpetas ya extraídas por Kaggle) del dataset rdd2020-raw."""
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    found = False
    for src in ("train", "test1", "test2"):
        hits = [h for h in glob.glob(f"/kaggle/input/**/{src}.tar.gz", recursive=True) if os.path.isfile(h)]
        if hits:
            os.symlink(hits[0], raw / f"{src}.tar.gz")
            found = True
            continue
        dirs = [d for d in glob.glob(f"/kaggle/input/**/{src}/Japan/images", recursive=True)]
        if dirs:
            base = Path(dirs[0]).parents[1]
            dest = ROOT / "data/interim/rdd2020" / src
            dest.mkdir(parents=True, exist_ok=True)
            for country_dir in base.iterdir():
                if country_dir.is_dir():
                    (dest / country_dir.name).mkdir(exist_ok=True)
                    os.symlink(country_dir / "images", dest / country_dir.name / "images")
                    xmls = country_dir / "annotations" / "xmls"
                    if xmls.exists():
                        os.symlink(xmls, dest / country_dir.name / "xmls")
            (dest / ".done").write_text("kaggle-input")
            found = True
    log(f"datos desde /kaggle/input: {found}")
    return found


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


def find_campo_images():
    """Carpeta con las fotos de campo: DIMMIT_CAMPO (prueba local) o el dataset dimmit-campo."""
    if os.environ.get("DIMMIT_CAMPO"):
        return Path(os.environ["DIMMIT_CAMPO"])
    hits = glob.glob("/kaggle/input/**/grabacion_*/", recursive=True)
    if not hits:
        raise FileNotFoundError("no encontré carpetas grabacion_* en /kaggle/input")
    return Path(hits[0]).parent


def infer(env, t0):
    """Modo inferencia: detecciones del detector ya entrenado sobre las fotos de campo."""
    import json

    weights = ROOT / "models/release/yolov10n_rdd2020.pt"
    campo = find_campo_images()
    paths = sorted(str(p) for p in campo.glob("grabacion_*/*.jpg"))
    if SMOKE:
        paths = paths[:SMOKE]
    (ROOT / "campo.txt").write_text("\n".join(paths) + "\n")
    log(f"inferencia: {len(paths)} fotos de {campo} con {weights.name}")
    try:
        import torch

        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    log(f"GPU disponible: {gpu}")
    sh([sys.executable, "-m", "dimmit.models.detector.predict", "--weights", str(weights), "--list", "campo.txt", "--out", str(OUT),
        "--imgsz", os.environ.get("DIMMIT_IMGSZ", "416"), "--batch", "16", "--conf", "0.01", "--no-embeddings"], env=env)
    (OUT / "resumen.json").write_text(json.dumps({"modo": "infer", "fotos": len(paths), "pesos": weights.name, "gpu": gpu,
                                                  "imgsz": int(os.environ.get("DIMMIT_IMGSZ", "416")), "segundos": round(time.time() - t0, 1)}, indent=2))
    log(f"inferencia lista en {time.time() - t0:.0f} s")


def main():
    global LOG
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = open(OUT / "kernel_log.txt", "a")
    t0 = time.time()
    bundle = find_bundle() if not os.environ.get("DIMMIT_BUNDLE") else Path(os.environ["DIMMIT_BUNDLE"])
    mode = os.environ.get("DIMMIT_MODE", "train")
    log(f"bundle: {bundle}  smoke={SMOKE}  modo={mode}")
    unpack(bundle)
    if not SMOKE:
        wheels = sorted(glob.glob(str(ROOT / "wheels/*.whl")))
        sh([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels], check=False)
    if mode == "infer":
        infer({**os.environ, "DIMMIT_ROOT": str(ROOT), "PYTHONPATH": str(ROOT / "src")}, t0)
        LOG.close()
        shutil.rmtree(ROOT, ignore_errors=True)
        return
    for w in ("yolo11n.pt", "yolov10n.pt"):  # chequeo AMP de ultralytics y pesos base, sin red
        if (ROOT / "models" / w).exists():
            shutil.copy(ROOT / "models" / w, ROOT / w)
    env = {**os.environ, "DIMMIT_ROOT": str(ROOT), "PYTHONPATH": str(ROOT / "src")}
    py = sys.executable
    if os.environ.get("DIMMIT_RAW"):  # prueba local: reutiliza los archivos ya descargados
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        os.symlink(os.environ["DIMMIT_RAW"], ROOT / "data/raw")
    elif not link_input_data():
        download(env)
    sh([py, "-m", "dimmit.data.prepare", "--stages", "extract,index,lists", "--smoke", str(SMOKE)], env=env)
    cfg = "configs/train_kaggle.yaml"
    try:
        import torch

        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    log(f"GPU disponible: {gpu}")
    imgsz = "640"
    if not gpu and not SMOKE:
        cpu_cfg = (ROOT / "configs/train_cpu.yaml").read_text()
        for a, b in (("epochs: 2", "epochs: 12"), ("fraction: 0.25", "fraction: 1.0"), ("max_minutes: 105", "max_minutes: 560"),
                     ("workers: 3", "workers: 4"), ("name: yolov10n_cpu_smoke", "name: yolov10n_rdd2020"), ("close_mosaic: 1", "close_mosaic: 2")):
            cpu_cfg = cpu_cfg.replace(a, b)
        (ROOT / "configs/train_kaggle_cpu.yaml").write_text(cpu_cfg)
        cfg, imgsz = "configs/train_kaggle_cpu.yaml", "416"
    if SMOKE:
        smoke_cfg = (ROOT / cfg).read_text().replace("epochs: 120", "epochs: 1").replace("cache: ram", "cache: false")
        smoke_cfg = smoke_cfg.replace("batch: 32", "batch: 8").replace("imgsz: 640", "imgsz: 320").replace("workers: 4", "workers: 1")
        (ROOT / "configs/train_smoke.yaml").write_text(smoke_cfg)
        cfg = "configs/train_smoke.yaml"
    sh([py, "-m", "dimmit.models.detector.train", "--config", cfg], check=False, env=env)  # 75 = tope de tiempo
    run = ROOT / "models/yolov10n_rdd2020"
    best = run / "weights/best.pt"
    log(f"entrenamiento listo en {(time.time() - t0) / 3600:.2f} h")
    imgsz = "320" if SMOKE else imgsz
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
