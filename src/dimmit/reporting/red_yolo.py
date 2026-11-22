"""La red YOLOv10n de DIMMIT por dentro, con las herramientas estándar (sin dibujos propios):

  arquitectura  tabla de capas que imprime Ultralytics a partir del archivo de pesos
  traza         una pasada real de una foto: forma de la salida de cada capa (hooks de PyTorch), las 3 cuadrículas
                de la cabeza y la salida final; luego el perfilador de Ultralytics (tiempo y GFLOPs por capa)
  onnx          exporta el modelo a ONNX con el exportador de Ultralytics (para verlo en Netron)
  visualizar    predicción de Ultralytics con visualize=True: guarda los mapas de activación de cada etapa y la
                foto anotada por la propia librería

Uso: python -m dimmit.reporting.red_yolo {arquitectura|traza|onnx|visualizar} [--foto Prueba_6/Prueba_6_007] [--conf 0.16]
Salidas en build/red/ (no se versionan: se regeneran en cada corrida).
"""
import argparse
import logging
import shutil

import numpy as np
import torch
from PIL import Image

from dimmit.utils.io import REPO_ROOT, ensure_dir

PESOS = REPO_ROOT / "models/release/yolov10n_rdd2020.pt"
OUT = REPO_ROOT / "build" / "red"


def cargar():
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.INFO)
    return YOLO(str(PESOS))


def tensor(foto):
    from ultralytics.data.augment import LetterBox

    img = np.array(Image.open(REPO_ROOT / "data/campo/fotos" / f"{foto}.jpg").convert("RGB"))
    lb = LetterBox((640, 640), auto=False, stride=32)(image=img[:, :, ::-1])[:, :, ::-1].copy()
    print(f"foto {foto}: {img.shape[1]}×{img.shape[0]} px -> letterbox 640×640 -> tensor (1, 3, 640, 640) con valores de 0 a 1")
    return torch.from_numpy(lb).permute(2, 0, 1)[None].float() / 255


def arquitectura(a):
    from ultralytics.nn.tasks import DetectionModel

    y = cargar()
    print(f"Pesos: {PESOS.relative_to(REPO_ROOT)} · clases: {y.model.names}")
    DetectionModel(y.model.yaml, nc=len(y.model.names), verbose=True)


def traza(a):
    net = cargar().model.eval()
    x = tensor(a.foto)

    def hook(m, _inp, o):
        if torch.is_tensor(o):
            print(f"  capa {m.i:2d} · {m.type.split('.')[-1]:9s} · entra desde {str(m.f):12s} -> salida {tuple(o.shape)}", flush=True)

    hs = [m.register_forward_hook(hook) for m in net.model[:-1]]
    print("Pasada hacia adelante (forward) de la foto, capa por capa:")
    with torch.no_grad():
        out, aux = net(x)
    for h in hs:
        h.remove()
    print("  capa 23 · v10Detect (cabeza): una cuadrícula por escala, 68 números por celda (64 de caja + 4 de clase)")
    tot = 0
    for t, s in zip(aux["one2one"], (8, 16, 32)):
        n = t.shape[2] * t.shape[3]
        tot += n
        print(f"    P{int(np.log2(s))}: tensor {tuple(t.shape)} -> cuadrícula {t.shape[2]}×{t.shape[3]} = {n} celdas (stride {s} px)")
    print(f"    total: {tot} celdas candidatas -> la cabeza uno a uno elige las 300 mejores (sin NMS)")
    print(f"  salida final {tuple(out.shape)}: [x1, y1, x2, y2, confianza, clase]; las 5 primeras filas:")
    for r in out[0][:5].numpy():
        print(f"    x1={r[0]:6.1f} y1={r[1]:6.1f} x2={r[2]:6.1f} y2={r[3]:6.1f} conf={r[4]:.3f} clase={int(r[5])} ({net.names[int(r[5])]})")
    print("\nPerfilador de Ultralytics (cada capa se mide 10 veces):")
    with torch.no_grad():
        net._predict_once(x, profile=True)


def onnx(a):
    y = cargar()
    p = y.export(format="onnx", imgsz=640, opset=17, simplify=True)
    dst = ensure_dir(OUT) / "yolov10n_rdd2020.onnx"
    shutil.move(p, dst)
    print(f"[ok] {dst.relative_to(REPO_ROOT)} ({dst.stat().st_size / 1e6:.1f} MB) -> se abre en Netron")


def visualizar(a):
    y = cargar()
    dst = OUT / "prediccion"
    shutil.rmtree(dst, ignore_errors=True)
    r = y.predict(str(REPO_ROOT / "data/campo/fotos" / f"{a.foto}.jpg"), imgsz=640, conf=a.conf, visualize=True, save=True,
                  project=str(OUT), name="prediccion", exist_ok=True)[0]
    for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        print(f"[ok] detección: {y.model.names[int(c)]} · confianza {cf:.3f}")
    print(f"[ok] mapas de activación y foto anotada en {r.save_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("accion", choices=["arquitectura", "traza", "onnx", "visualizar"])
    ap.add_argument("--foto", default="Prueba_6/Prueba_6_007")
    ap.add_argument("--conf", type=float, default=0.16, help="umbral de confianza (0,16 = umbral calibrado con el IDU para piel de cocodrilo)")
    a = ap.parse_args()
    {"arquitectura": arquitectura, "traza": traza, "onnx": onnx, "visualizar": visualizar}[a.accion](a)


if __name__ == "__main__":
    main()
