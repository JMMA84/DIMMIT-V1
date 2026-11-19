"""Imágenes sueltas para armar en Canva las diapositivas de YOLO y de progreso dentro de la plantilla del equipo.

No son diapositivas completas: cada imagen va dentro de un marco de la plantilla, con su proporción:
  * tarjetas de imagen del layout "Julio / Agosto / Septiembre" (página 9): 2,14 : 1
  * marco orgánico del layout "Abril / Mayo / Junio" (página 7): 1,43 : 1
Fondo blanco o foto a sangre, tipografía Boriboon (--fuentes) y la paleta de la plantilla.
Escribe reports/presentacion/canva/*.png|jpg
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from dimmit.campo.scores import CALIB
from dimmit.reporting.slides_yolo_progreso import F, NAVY, PERI, TEAL, ejemplo, fonts
from dimmit.utils.io import REPO_ROOT, ensure_dir

OUT = REPO_ROOT / "reports" / "presentacion" / "canva"
CARD = (1712, 800)  # 2,14 : 1
BLOB = (1430, 1000)  # 1,43 : 1
INK = NAVY


def crop_ratio(arr, ratio, fy=0.5):
    """Recorte centrado (fy = posición vertical 0-1) con la proporción pedida."""
    h, w = arr.shape[:2]
    if w / h > ratio:
        nw = int(h * ratio)
        x0 = (w - nw) // 2
        return arr[:, x0:x0 + nw], (x0, 0)
    nh = int(w / ratio)
    y0 = int((h - nh) * fy)
    return arr[y0:y0 + nh], (0, y0)


def save_arr(arr, name, size):
    p = ensure_dir(OUT) / name
    Image.fromarray(arr).resize(size, Image.LANCZOS).convert("RGB").save(p, quality=92)
    return p


def fig_card(size=CARD):
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def t(ax, x, y, s, size, color=INK, f="reg", **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    ax.text(x, y, s, fontsize=size, color=color, fontfamily=F[f]["family"], fontweight=F[f]["weight"], **kw)


def save_fig(fig, name):
    p = ensure_dir(OUT) / name
    fig.savefig(p, dpi=100, facecolor="white")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fuentes", default=None, help="carpeta con Boriboon-*.otf (extraídas del .pptx de la plantilla)")
    args = ap.parse_args()
    fonts(args.fuentes)
    e = ejemplo()
    lb = e["lb"]
    real = lb[140:500]  # zona con imagen dentro del letterbox 640 × 640 (1280 × 720 -> 640 × 360)
    out = []

    # YOLO 1 · tarjeta 1: la foto tal cual
    foto, _ = crop_ratio(e["img"], CARD[0] / CARD[1], fy=0.75)
    out.append(save_arr(foto, "yolo1_1_foto.jpg", CARD))

    # YOLO 1 · tarjeta 2: el tensor (3 canales apilados)
    fig, ax = fig_card()
    sq = 560
    for j, (cm, off, lab) in enumerate((("Blues", 120, "B"), ("Greens", 60, "G"), ("Reds", 0, "R"))):
        ch = lb[:, :, 2 - j].astype(float) / 255
        x0, y0 = 250 + off, 60 + (120 - off)
        ax.imshow(plt.get_cmap(cm)(0.25 + 0.7 * ch), extent=(x0, x0 + sq, y0 + sq, y0), zorder=2 + j)
        ax.add_patch(Rectangle((x0, y0), sq, sq, fill=False, ec=INK, lw=2, zorder=2 + j))
        t(ax, x0 + 16, y0 + 8, lab, 44, "white", "heavy", zorder=5)
    t(ax, 1050, 290, "3 × 640", 78, INK, "heavy")
    t(ax, 1050, 400, "× 640", 78, INK, "heavy")
    t(ax, 1050, 530, "canales × alto × ancho", 34, PERI, "bold")
    out.append(save_fig(fig, "yolo1_2_tensor.png"))

    # YOLO 1 · tarjeta 3: los 3 mapas de rasgos (activación real de esta foto)
    fig, ax = fig_card()
    labs = [(16, "P3", "80 × 80", "64 rasgos"), (19, "P4", "40 × 40", "128 rasgos"), (22, "P5", "20 × 20", "256 rasgos")]
    for k, (i, p, res, ch) in enumerate(labs):
        fm = e["feats"][i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-9)
        x0 = 96 + k * 540
        ax.imshow(plt.get_cmap("viridis")(fm), extent=(x0, x0 + 440, 40 + 440, 40), interpolation="nearest", zorder=2)
        ax.add_patch(Rectangle((x0, 40), 440, 440, fill=False, ec=TEAL, lw=5, zorder=3))
        t(ax, x0, 510, p, 80, INK, "heavy")
        t(ax, x0 + 160, 522, res, 46, INK, "bold")
        t(ax, x0 + 160, 600, ch, 40, PERI, "bold")
    out.append(save_fig(fig, "yolo1_3_rasgos.png"))

    # YOLO 2 · tarjeta 1: la cuadrícula de 20 × 20 con la celda que detecta
    fig, ax = fig_card()
    ratio = CARD[0] / CARD[1]
    y_top = 500 - int(640 / ratio)  # recorte de la parte baja de la zona real, donde está la celda
    sub = lb[y_top:500]
    ax.imshow(sub, extent=(0, CARD[0], CARD[1], 0), zorder=1)
    sx = CARD[0] / 640
    for k in range(0, 641, 32):
        ax.plot([k * sx] * 2, [0, CARD[1]], color="white", lw=1.4, alpha=0.7, zorder=2)
    for yy in range(0, 641, 32):
        ys = (yy - y_top) * sx
        if 0 <= ys <= CARD[1]:
            ax.plot([0, CARD[0]], [ys, ys], color="white", lw=1.4, alpha=0.7, zorder=2)
    cx, cy = e["col"] * 32 * sx, (e["row"] * 32 - y_top) * sx
    ax.add_patch(Rectangle((cx, cy), 32 * sx, 32 * sx, fc=TEAL, ec="white", lw=3, alpha=0.9, zorder=3))
    out.append(save_fig(fig, "yolo2_1_cuadricula.png"))

    # YOLO 2 · tarjeta 2: el vector de 68 números de esa celda
    fig, ax = fig_card()
    t(ax, 70, 50, "DÓNDE", 54, INK, "heavy")
    names = ("izq.", "arriba", "der.", "abajo")
    for k in range(4):
        x0, y0 = 70 + k * 205, 175
        d = e["dist"][k]
        for b in range(16):
            hh = 200 * d[b] / d.max()
            ax.add_patch(Rectangle((x0 + b * 10, y0 + 200 - hh), 8, hh, fc=PERI, ec="none"))
        exp = float((d * np.arange(16)).sum())
        ax.plot([x0 + exp * 10 + 4] * 2, [y0 - 6, y0 + 204], color=TEAL, lw=4)
        t(ax, x0, y0 + 225, names[k], 36, INK, "reg")
        t(ax, x0, y0 + 285, f"{exp * e['stride']:.0f} px", 46, INK, "xbold")
    t(ax, 930, 50, "QUÉ", 54, INK, "heavy")
    clases = ("Grieta long.", "Grieta transv.", "Piel de cocodrilo", "Bache")
    for k, (c, pr) in enumerate(zip(clases, e["probs"])):
        yy = 165 + k * 150
        hi = k == 2
        t(ax, 930, yy, c, 38, INK, "xbold" if hi else "reg")
        ax.add_patch(Rectangle((930, yy + 62), 560, 46, fc="#E6E8F5", ec="none"))
        ax.add_patch(Rectangle((930, yy + 62), max(560 * float(pr), 5), 46, fc=TEAL if hi else PERI, ec="none"))
        t(ax, 1510, yy + 56, f"{float(pr):.2f}".replace(".", ","), 44, INK, "xbold" if hi else "reg")
    out.append(save_fig(fig, "yolo2_2_vector.png"))

    # YOLO 2 · tarjeta 3: el resultado (recorte con la caja final)
    x1, y1, x2, y2, conf, _ = e["det"]
    cxm, cym = (x1 + x2) / 2, (y1 + y2) / 2
    hh = min(360, (y2 - y1) * 1.9)
    ww = hh * ratio
    X0 = int(np.clip(cxm - ww / 2, 0, 640 - ww))
    Y0 = int(np.clip(cym - hh / 2, 140, 500 - hh))
    sub = lb[Y0:int(Y0 + hh), X0:int(X0 + ww)]
    fig, ax = fig_card()
    ax.imshow(sub, extent=(0, CARD[0], CARD[1], 0), zorder=1)
    s = CARD[0] / sub.shape[1]
    bx, by = (x1 - X0) * s, (y1 - Y0) * s
    ax.add_patch(Rectangle((bx, by), (x2 - x1) * s, (y2 - y1) * s, fill=False, ec=TEAL, lw=7, zorder=3))
    ax.add_patch(Rectangle((bx, by - 96), 760, 92, fc=TEAL, ec="none", zorder=3))
    t(ax, bx + 24, by - 50, f"piel de cocodrilo · {conf:.2f}".replace(".", ","), 44, INK, "heavy", va="center", zorder=4)
    out.append(save_fig(fig, "yolo2_3_resultado.png"))

    # Progreso · foto para el marco orgánico: la Carrera 5 con el bache relleno y el CAI al fondo
    p = REPO_ROOT / "data" / "campo" / "fotos" / "Prueba_5" / "Prueba_5_020.jpg"
    arr = np.array(Image.open(p).convert("RGB"))
    arr, _ = crop_ratio(arr, BLOB[0] / BLOB[1], fy=0.6)
    out.append(save_arr(arr, "progreso_foto_carrera5.jpg", BLOB))
    for o in out:
        print(f"[ok] {o}")


if __name__ == "__main__":
    main()
