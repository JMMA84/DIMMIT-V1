"""Diapositivas (PNG 16:9) con el estilo de la plantilla de Canva del equipo ("marketing moderno azul"):
dos que explican YOLOv10 paso a paso con un ejemplo real de la toma y dos del progreso del proyecto.

Usa datos reales: el modelo de models/release sobre una foto de campo (activaciones, celda que detecta,
vector de 68 números), el historial de git y las salidas de la corrida calibrada.

Fuente: Boriboon (la de la plantilla). Se pasa con --fuentes (carpeta con Boriboon-*.otf extraídas del
.pptx de Canva); si no está, se usa DejaVu Sans.
Escribe reports/presentacion/<nombre>.png (con título) y <nombre>_sin_titulo.png (para poner el título en Canva).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from dimmit.campo import load_cfg
from dimmit.campo.scores import CALIB, FEAT
from dimmit.utils.io import REPO_ROOT, ensure_dir

NAVY, DARK, TEAL, TEAL_L, PERI, PERI_D, WHITE = "#0F0E47", "#040414", "#5AB7BB", "#ACD9DC", "#3A46A0", "#262C6E", "#FFFFFF"
MUTED = "#B9BCE0"
OUT = REPO_ROOT / "reports" / "presentacion"
EJEMPLO = "Prueba_6/Prueba_6_007"
W, H = 1920, 1080
F = {}  # pesos de la fuente


def fonts(folder):
    fam = "DejaVu Sans"
    if folder and Path(folder).exists():
        for p in sorted(Path(folder).glob("Boriboon-*.otf")):
            font_manager.fontManager.addfont(str(p))
        fam = "Boriboon"
    for k, w in (("heavy", "black"), ("xbold", "heavy"), ("bold", "bold"), ("semi", "semibold"), ("med", "medium"), ("reg", "regular"), ("light", "light")):
        F[k] = {"family": [fam, "DejaVu Sans"], "weight": w}


# --- utilidades de dibujo (coordenadas en px, y hacia abajo) ----------------------------------------------
def canvas():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    g = np.linspace(0, 1, 256)[:, None]
    c0, c1 = np.array(matplotlib.colors.to_rgb(NAVY)), np.array(matplotlib.colors.to_rgb(DARK))
    ax.imshow((c0 * (1 - g) + c1 * g)[:, :, None].transpose(0, 2, 1).repeat(2, 1), extent=(0, W, H, 0), aspect="auto", zorder=0)
    return fig, ax


def text(ax, x, y, s, size, color=WHITE, f="reg", ha="left", va="top", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontfamily=F[f]["family"], fontweight=F[f]["weight"], zorder=6, **kw)


def title(ax, kicker, t, show):
    if not show:
        return
    text(ax, 110, 60, kicker, 17, TEAL, "semi")
    text(ax, 110, 92, t, 42, WHITE, "heavy")


def card(ax, x, y, w, h, fc=WHITE, ec=None, r=18, alpha=1.0, z=2, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=lw, alpha=alpha, zorder=z))


def badge(ax, x, y, n, label):
    ax.add_patch(plt.Circle((x + 22, y + 22), 22, color=TEAL, zorder=5))
    text(ax, x + 22, y + 23, str(n), 17, NAVY, "heavy", ha="center", va="center")
    text(ax, x + 56, y + 23, label, 17.5, WHITE, "xbold", va="center")


def arrow(ax, x0, y0, x1, y1, color=TEAL, lw=3.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=26, lw=lw, color=color, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def image(ax, arr, x, y, w, h, z=3, **kw):
    ax.imshow(arr, extent=(x, x + w, y + h, y), zorder=z, aspect="auto", **kw)


def save(fig, name):
    p = ensure_dir(OUT) / f"{name}.png"
    fig.savefig(p, dpi=200, facecolor=NAVY)
    plt.close(fig)
    return p


# --- datos reales del ejemplo ---------------------------------------------------------------------------------
def ejemplo():
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    path = REPO_ROOT / "data" / "campo" / "fotos" / f"{EJEMPLO}.jpg"
    net = YOLO(str(REPO_ROOT / "models/release/yolov10n_rdd2020.pt")).model.eval()
    img = np.array(Image.open(path).convert("RGB"))
    lb = LetterBox((640, 640), auto=False, stride=32)(image=img[:, :, ::-1])[:, :, ::-1].copy()
    x = torch.from_numpy(lb).permute(2, 0, 1)[None].float() / 255
    feats = {}
    for i in (16, 19, 22):
        net.model[i].register_forward_hook(lambda m, a, o, i=i: feats.__setitem__(i, o.detach()[0].abs().mean(0).numpy()))
    with torch.no_grad():
        out, aux = net(x)
    cat = torch.cat([t.flatten(2) for t in aux["one2one"]], 2)[0]
    probs = torch.sigmoid(cat[64:])
    best = int(probs[2].argmax())
    sizes, grids, strides = [6400, 1600, 400], [80, 40, 20], [8, 16, 32]
    k, s = best, 0
    while k >= sizes[s]:
        k -= sizes[s]
        s += 1
    row, col = divmod(k, grids[s])
    dist = torch.softmax(cat[:64, best].view(4, 16), 1).numpy()
    det = out[0][0].numpy()  # x1, y1, x2, y2, conf, clase (la mejor de las 300)
    n_params = sum(p.numel() for p in net.parameters())
    return {"img": img, "lb": lb, "feats": feats, "probs": probs[:, best].numpy(), "dist": dist, "scale": s, "row": row, "col": col,
            "stride": strides[s], "grid": grids[s], "det": det, "n_params": n_params}


# --- diapositiva 1: de la foto al mapa de características --------------------------------------------------------
def caption(ax, x, y, head, body):
    text(ax, x, y, head, 17.5, WHITE, "semi")
    text(ax, x, y + 34, body, 15.5, MUTED, "reg", linespacing=1.5)


def slide_yolo_1(e, show_title=True):
    fig, ax = canvas()
    title(ax, "CÓMO FUNCIONA YOLO · 1 DE 2", "DE LA FOTO AL MAPA DE RASGOS", show_title)
    top = 240 if show_title else 150
    xs, cw = [110, 560, 1010, 1460], 380
    cap = top + 470
    # 1. foto
    badge(ax, xs[0], top, 1, "LA FOTO")
    ih = cw * 720 / 1280
    image(ax, e["img"], xs[0], top + 90 + (330 - ih) / 2, cw, ih)
    caption(ax, xs[0], cap, "Foto de celular en la vía", "1280 × 720 píxeles\n3 colores por píxel (RGB)\nvalores de 0 a 255")
    # 2. tensor: los 3 canales de la imagen ya escalada a 640 × 640 (con relleno gris arriba y abajo)
    badge(ax, xs[1], top, 2, "EL TENSOR")
    sq = 270
    for j, (cm, off) in enumerate((("Blues", 56), ("Greens", 28), ("Reds", 0))):
        ch = e["lb"][:, :, 2 - j].astype(float) / 255
        image(ax, plt.get_cmap(cm)(0.25 + 0.7 * ch), xs[1] + off, top + 90 + (56 - off), sq, sq, z=3 + j)
        ax.add_patch(Rectangle((xs[1] + off, top + 90 + (56 - off)), sq, sq, fill=False, ec=WHITE, lw=1.2, zorder=3 + j))
    for lab, off in (("B", 56), ("G", 28), ("R", 0)):  # letra en la esquina visible de cada canal
        text(ax, xs[1] + off + 8, top + 90 + (56 - off) + 4, lab, 14, WHITE, "heavy")
    caption(ax, xs[1], cap, "Se escala a 640 px y se rellena", "Tensor de 3 × 640 × 640\n= 1,2 millones de números\nentre 0 y 1")
    # 3. backbone: bloques con menos píxeles y más canales
    badge(ax, xs[2], top, 3, "BACKBONE")
    blocks = [("320", 16), ("160", 32), ("80", 64), ("40", 128), ("20", 256)]
    by, x = top + 110, xs[2] + 6
    text(ax, xs[2] + 6, by - 40, "lado del mapa (px)", 13.5, MUTED, "reg", va="bottom")
    for k, (res, ch) in enumerate(blocks):
        hgt, wid = 250 - k * 42, 26 + k * 16
        y0 = by + (250 - hgt) / 2
        col = [PERI_D, PERI, "#4D5BC0", "#4E9FB5", TEAL][k]
        ax.add_patch(FancyBboxPatch((x, y0), wid, hgt, boxstyle="round,pad=0,rounding_size=5", fc=col, ec=WHITE, lw=0.8, zorder=3))
        text(ax, x + wid / 2, y0 - 6, res, 12.5, WHITE, "bold", ha="center", va="bottom")
        text(ax, x + wid / 2, by + 262, str(ch), 13.5, TEAL_L, "bold", ha="center")
        x += wid + 12
    text(ax, xs[2] + 6, by + 290, "canales (rasgos por píxel)", 13.5, MUTED, "reg")
    caption(ax, xs[2], cap, "Convoluciones que resumen", "Menos píxeles, más canales:\nbordes → texturas → patrones\nde daño (+ atención PSA)")
    # 4. neck: 3 mapas de rasgos (activación real de esta foto)
    badge(ax, xs[3], top, 4, "NECK (FPN + PAN)")
    labs = [(16, "P3 · 80×80 · 64 canales", "fisuras finas"), (19, "P4 · 40×40 · 128 canales", "daños medianos"),
            (22, "P5 · 20×20 · 256 canales", "baches, zonas grandes")]
    cmap = plt.get_cmap("viridis")
    for k, (i, l1, l2) in enumerate(labs):
        fm = e["feats"][i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-9)
        yy = top + 86 + k * 118
        image(ax, cmap(fm), xs[3], yy, 104, 104, interpolation="nearest")
        ax.add_patch(Rectangle((xs[3], yy), 104, 104, fill=False, ec=TEAL, lw=1.5, zorder=4))
        text(ax, xs[3] + 120, yy + 30, l1, 15, WHITE, "bold")
        text(ax, xs[3] + 120, yy + 60, l2, 14.5, MUTED, "reg")
    caption(ax, xs[3], cap, "Mezcla los niveles", "Une detalle fino y contexto:\n3 mapas de rasgos\n(activación real de esta foto)")
    for k in range(3):
        arrow(ax, xs[k] + cw + 14, top + 255, xs[k + 1] - 22, top + 255)
    card(ax, 110, 960, 1700, 70, fc=PERI_D, r=14, alpha=0.9)
    text(ax, 140, 995, f"YOLOv10n · {e['n_params'] / 1e6:.1f} M parámetros · entrenado con RDD2020 (Japón, India, Chequia) · ".replace(".", ",", 1)
         + "4 clases: grieta longitudinal, grieta transversal, piel de cocodrilo, bache", 15.5, WHITE, "med", va="center")
    return fig


# --- diapositiva 2: de la cuadrícula a la etiqueta ----------------------------------------------------------
def slide_yolo_2(e, tau, show_title=True):
    fig, ax = canvas()
    title(ax, "CÓMO FUNCIONA YOLO · 2 DE 2", "DE LA CUADRÍCULA A LA ETIQUETA", show_title)
    top = 240 if show_title else 150
    xs = [110, 520, 1030, 1475]
    cap = top + 470
    # 5. cuadrícula (la de 20 × 20, donde detecta esta foto)
    badge(ax, xs[0], top, 5, "LA CUADRÍCULA")
    s, g = 340, e["grid"]
    image(ax, e["lb"], xs[0], top + 86, s, s)
    for k in range(g + 1):
        ax.plot([xs[0] + k * s / g] * 2, [top + 86, top + 86 + s], color=WHITE, lw=0.5, alpha=0.55, zorder=4)
        ax.plot([xs[0], xs[0] + s], [top + 86 + k * s / g] * 2, color=WHITE, lw=0.5, alpha=0.55, zorder=4)
    ax.add_patch(Rectangle((xs[0] + e["col"] * s / g, top + 86 + e["row"] * s / g), s / g, s / g, fc=TEAL, ec=WHITE, lw=1.5, zorder=5))
    caption(ax, xs[0], cap, "Cada celda propone un daño", "80×80 + 40×40 + 20×20\n= 8.400 celdas candidatas\n(resaltada: la que detecta aquí)")
    # 6. vector de 68 números de esa celda
    badge(ax, xs[1], top, 6, "EL VECTOR DE LA CELDA")
    cx0, cwid = xs[1], 470
    card(ax, cx0, top + 86, cwid, 500, fc=WHITE, r=18)
    text(ax, cx0 + 24, top + 108, "68 números por celda", 18, NAVY, "xbold")
    text(ax, cx0 + 24, top + 146, "64 · DÓNDE: 4 lados × 16 probabilidades", 14, PERI, "bold")
    names = ("izq.", "arriba", "der.", "abajo")
    for k in range(4):
        x0, y0 = cx0 + 24 + k * 108, top + 180
        d = e["dist"][k]
        for bb in range(16):
            hh = 90 * d[bb] / d.max()
            ax.add_patch(Rectangle((x0 + bb * 6, y0 + 90 - hh), 5, hh, fc=PERI, ec="none", zorder=4))
        exp = float((d * np.arange(16)).sum())
        ax.plot([x0 + exp * 6 + 2.5] * 2, [y0 - 4, y0 + 92], color=TEAL, lw=2.2, zorder=5)
        text(ax, x0, y0 + 100, names[k], 13, NAVY, "reg")
        text(ax, x0, y0 + 124, f"{exp * e['stride']:.0f} px", 14, NAVY, "bold")
    text(ax, cx0 + 24, top + 352, "4 · QUÉ: probabilidad por clase", 14, PERI, "bold")
    clases = ("Grieta longitudinal", "Grieta transversal", "Piel de cocodrilo", "Bache")
    for k, (c, pr) in enumerate(zip(clases, e["probs"])):
        yy = top + 388 + k * 46
        hi = k == 2
        text(ax, cx0 + 24, yy + 13, c, 14, NAVY, "bold" if hi else "reg", va="center")
        ax.add_patch(Rectangle((cx0 + 215, yy + 1), 170, 24, fc="#E6E8F5", ec="none", zorder=4))
        ax.add_patch(Rectangle((cx0 + 215, yy + 1), max(170 * float(pr), 2), 24, fc=TEAL if hi else PERI, ec="none", zorder=5))
        text(ax, cx0 + 398, yy + 13, f"{float(pr):.2f}".replace(".", ","), 14, NAVY, "bold" if hi else "reg", va="center")
    # 7. decisión
    badge(ax, xs[2], top, 7, "LA DECISIÓN")
    pasos = [("8.400 candidatos", "uno por celda"), ("Cabeza uno a uno (YOLOv10)", "sin NMS: quedan las 300 mejores"),
             ("Umbral calibrado con el IDU", f"piel de cocodrilo ≥ {tau['D20']:.2f}".replace(".", ",")),
             ("Filtros de campo", "vía, superficie, persistencia, escalas")]
    for k, (a_, b_) in enumerate(pasos):
        yy = top + 86 + k * 126
        card(ax, xs[2], yy, 400, 96, fc=PERI_D, r=14)
        text(ax, xs[2] + 22, yy + 20, a_, 16, WHITE, "bold")
        text(ax, xs[2] + 22, yy + 56, b_, 14.5, TEAL_L, "reg")
        if k < 3:
            arrow(ax, xs[2] + 200, yy + 98, xs[2] + 200, yy + 124, lw=2.5)
    # 8. resultado: recorte de la imagen real (sin el relleno gris) con la caja final
    badge(ax, xs[3], top, 8, "EL RESULTADO")
    x1, y1, x2, y2, conf, _ = e["det"]
    pad, ytop, ybot = 70, 140, 500  # filas con imagen dentro del letterbox (720 px -> 360 px, centrado)
    cx1, cy1, cx2, cy2 = int(max(x1 - pad, 0)), int(max(y1 - pad, ytop)), int(min(x2 + pad, 640)), int(min(y2 + pad, ybot))
    crop = e["lb"][cy1:cy2, cx1:cx2]
    dw = 340
    sx = dw / crop.shape[1]
    dh = crop.shape[0] * sx
    image(ax, crop, xs[3], top + 86, dw, dh)
    bx0, by0 = xs[3] + (x1 - cx1) * sx, top + 86 + (y1 - cy1) * sx
    ax.add_patch(Rectangle((bx0, by0), (x2 - x1) * sx, (y2 - y1) * sx, fill=False, ec=TEAL, lw=3.5, zorder=5))
    card(ax, bx0, by0 - 36, 258, 34, fc=TEAL, r=6, z=5)
    text(ax, bx0 + 12, by0 - 19, f"piel de cocodrilo · {conf:.2f}".replace(".", ","), 14, NAVY, "bold", va="center")
    caption(ax, xs[3], cap, "Caja + clase + confianza", "Alimenta el score 0–100\ndel tramo (pseudo-PCI,\nASTM D6433)")
    for k in range(3):
        arrow(ax, xs[k + 1] - 62, top + 255, xs[k + 1] - 16, top + 255)
    card(ax, 110, 960, 1700, 70, fc=PERI_D, r=14, alpha=0.9)
    text(ax, 140, 995, "Ejemplo real: foto de la Carrera 5 (toma del 15-nov-2026). Todas las cifras salen del modelo de DIMMIT sobre esta foto.",
         15.5, WHITE, "med", va="center")
    return fig


# --- diapositiva 3: línea de tiempo del repositorio ---------------------------------------------------------
NB = " "
HITOS = [
    ("23 SEP", "Arranque", f"Estructura del repositorio, pipeline de datos, YOLOv10 y score{NB}v0", "Prueba de humo\nde punta a punta", "a4e75a2"),
    ("1 OCT", "Modelo v1", f"RDD2020 completo (9.420 fotos), particiones sin fuga y YOLOv10n entrenado en Kaggle (12{NB}épocas)",
     f"mAP50 0,03 → 0,28\nρ PCI visual 0,80", "6948369"),
    ("7–8 OCT", "Primera toma", f"Drive → tramos con ID → Kaggle; filtros contra falsos positivos y severidad ASTM", f"191 fotos · 9 tramos", "3b31e0f"),
    ("15 NOV", "Datos correctos", f"5 recorridos en 2 sitios; profundidad = distancia − 8{NB}cm; filtro de adoquín", f"131 fotos · 1.177 lecturas", "12ba676"),
    ("15 NOV", "Validado con el IDU", "Sitios anclados a la calzada oficial; cámara calibrada contra el estado del IDU",
     f"83{NB}% dentro del rango IDU", "004a723"),
]


def slide_progreso_1(n_commits, n_merges, n_tests, show_title=True):
    import textwrap

    fig, ax = canvas()
    title(ax, "NUESTRO PROGRESO · 1 DE 2", "DEL REPOSITORIO VACÍO A RESULTADOS VALIDADOS", show_title)
    top = 250 if show_title else 150
    cw, gap = 322, 22
    xs = [110 + k * (cw + gap) for k in range(5)]
    ly = top + 70
    ax.plot([xs[0] + 20, xs[-1] + cw - 20], [ly, ly], color=TEAL, lw=4, zorder=2, solid_capstyle="round")
    for k, (fecha, t, desc, kpi, sha) in enumerate(HITOS):
        x = xs[k]
        last = k == len(HITOS) - 1
        ax.add_patch(plt.Circle((x + cw / 2, ly), 15, color=WHITE if last else TEAL, ec=TEAL, lw=3, zorder=4))
        text(ax, x + cw / 2, ly - 30, fecha, 22, WHITE if last else TEAL, "heavy", ha="center", va="bottom")
        card(ax, x, ly + 42, cw, 500, fc=WHITE, r=18)
        text(ax, x + 24, ly + 66, f"{k + 1:02d}", 15, PERI, "bold")
        text(ax, x + 24, ly + 94, t, 21, NAVY, "xbold")
        text(ax, x + 24, ly + 150, textwrap.fill(desc, 27), 16.5, NAVY, "reg", linespacing=1.5)
        text(ax, x + 24, ly + 365, f"commit {sha}", 13.5, PERI, "semi")
        card(ax, x + 16, ly + 400, cw - 32, 120, fc=TEAL_L, r=12, z=3)
        text(ax, x + cw / 2, ly + 460, kpi, 16, NAVY, "bold", ha="center", va="center", linespacing=1.35)
    text(ax, 110, 1010, f"{n_commits} commits · {n_merges} ramas fusionadas · {n_tests} pruebas automáticas · cada resultado se reproduce con make",
         16.5, TEAL_L, "semi", va="center")
    return fig


# --- diapositiva 4: de "todo en intervención" a validado ----------------------------------------------------
def slide_progreso_2(sitio1, cal, prof, n_cajas, show_title=True):
    fig, ax = canvas()
    title(ax, "NUESTRO PROGRESO · 2 DE 2", "CÓMO LLEGAMOS A RESULTADOS CONFIABLES", show_title)
    top = 240 if show_title else 150
    text(ax, 110, top, "Sitio 1 (Carrera 5): score de los 3 recorridos en cada etapa", 17.5, WHITE, "semi")
    gx, gy, gw, gh = 190, top + 90, 900, 470
    y2p = lambda v: gy + gh - v / 100 * gh
    r = cal["rangos_idu_pci"]["regular"]
    ax.add_patch(Rectangle((gx, y2p(r[1])), gw, y2p(r[0]) - y2p(r[1]), fc=TEAL, alpha=0.18, zorder=1))
    text(ax, gx + gw - 14, y2p(r[0]) - 10, "rango «regular» del IDU para esta calle (PCI 56–89)", 13.5, TEAL_L, "semi", ha="right", va="bottom")
    for v in (0, 25, 50, 75, 100):
        ax.plot([gx, gx + gw], [y2p(v)] * 2, color=WHITE, alpha=0.12, lw=1, zorder=1)
        text(ax, gx - 14, y2p(v), str(v), 13, MUTED, "reg", ha="right", va="center")
    text(ax, gx - 66, gy + gh / 2, "score visual 0–100", 13.5, MUTED, "reg", ha="center", va="center", rotation=90)
    etapas = ["Umbral RDD2020\nsin filtros", "Filtros de campo\nsin calibrar", "Calibrado\ncon el IDU"]
    notas = ["«intervención»\nen todo", "«bueno» en todo:\ndemasiado optimista", "«regular»,\nigual que el IDU"]
    ex = [gx + 150 + k * 300 for k in range(3)]
    cols = [WHITE, TEAL_L, TEAL]
    for serie in sitio1:
        ax.plot(ex, [y2p(v) for v in serie], color=WHITE, alpha=0.35, lw=2, zorder=2)
    for k in range(3):
        vals = [s_[k] for s_ in sitio1]
        for v in vals:
            ax.add_patch(plt.Circle((ex[k], y2p(v)), 13, color=cols[k], ec=NAVY, lw=2, zorder=4))
        text(ax, ex[k], gy + gh + 20, etapas[k], 15, WHITE, "semi", ha="center", linespacing=1.3)
        text(ax, ex[k], gy + gh + 78, f"dispersión: {max(vals) - min(vals):.0f} pts", 13.5, MUTED, "reg", ha="center")
        if k == 2:
            text(ax, ex[k] + 26, y2p(max(vals)) - 4, notas[k], 14, cols[k], "bold", ha="left", va="bottom", linespacing=1.25)
        else:
            text(ax, ex[k], y2p(max(vals)) - 24, notas[k], 14, cols[k], "bold", ha="center", va="bottom", linespacing=1.25)
    kx, kw = 1200, 610
    a0, a1 = cal["ajuste"]["acierto_sin_calibrar"], cal["ajuste"]["acierto_estado_idu"]
    pct = lambda v: f"{v:.0%}".replace("%", " %")
    d_cal = max(s_[2] for s_ in sitio1) - min(s_[2] for s_ in sitio1)
    d_sin = max(s_[1] for s_ in sitio1) - min(s_[1] for s_ in sitio1)
    kpis = [(pct(a1), f"de los tramos dentro del rango\ndel IDU (sin calibrar: {pct(a0)})"),
            (f"{d_cal:.0f} pts", f"de diferencia entre recorridos\ndel mismo sitio (antes {d_sin:.0f})"),
            (f"{prof[0]:.0f}–{prof[1]:.0f} cm", "de profundidad medida por el\nsensor en baches que la cámara\ncasi no ve"),
            (f"{n_cajas:,}".replace(",", "."), "cajas idénticas entre Kaggle y la\ncorrida local (|Δ| < 0,000002)")]
    for k, (big, small) in enumerate(kpis):
        yy = top + 40 + k * 148
        card(ax, kx, yy, kw, 130, fc=WHITE, r=16)
        text(ax, kx + 26, yy + 65, big, 34, NAVY, "heavy", va="center")
        text(ax, kx + 262, yy + 65, small, 14.5, NAVY, "reg", va="center", linespacing=1.35)
    card(ax, 110, 960, 1700, 70, fc=PERI_D, r=14, alpha=0.9)
    text(ax, 140, 995, f"Límite honesto: validado en {cal['n_calzadas']} calzada ({cal['n_tramos']} tramos, 2 tomas). "
         "Siguiente paso: 20–30 calzadas con estados distintos y fotos con GPS.", 15.5, WHITE, "med", va="center")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fuentes", default=None, help="carpeta con Boriboon-*.otf (extraídas del .pptx de la plantilla)")
    args = ap.parse_args()
    fonts(args.fuentes)
    cfg = load_cfg()
    cal = json.loads(CALIB.read_text())
    tau = cal["tau_campo_calibrado"]
    e = ejemplo()
    v = pd.read_parquet(FEAT / "vias.parquet")
    t = v[(v["nivel"] == "tramo") & (v["conjunto"] == "evaluacion")]
    s1 = t[t["sitio"] == 1].sort_values("prueba")
    # score de cada recorrido del sitio 1 en las 3 etapas (sin filtros, filtros sin calibrar, calibrado)
    sin_cal = pd.read_csv(FEAT / "calibracion_idu.csv").set_index("prueba")
    sitio1 = [(r.score_calidad_tau_rdd2020, float(sin_cal.loc[r.prueba, "score_sin_calibrar"]), r.score_calidad) for r in s1.itertuples()]
    prof = (float(t["profundidad_max_cm"].min()), float(t["profundidad_max_cm"].max()))
    n_cajas = len(pd.read_parquet(REPO_ROOT / "reports/campo/kaggle/outputs/detections.parquet"))
    import subprocess

    log = subprocess.run(["git", "log", "origin/main", "--format=%s"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.splitlines()
    n_commits, n_merges = len(log), sum(s.startswith("Merge") for s in log)
    n_tests = 27
    for name, fn in (("yolo_1_de_la_foto_al_mapa", lambda st: slide_yolo_1(e, st)), ("yolo_2_de_la_cuadricula_a_la_etiqueta", lambda st: slide_yolo_2(e, tau, st)),
                     ("progreso_1_linea_de_tiempo", lambda st: slide_progreso_1(n_commits, n_merges, n_tests, st)),
                     ("progreso_2_resultados_confiables", lambda st: slide_progreso_2(sitio1, cal, prof, n_cajas, st))):
        for st in (True, False):
            p = save(fn(st), name + ("" if st else "_sin_titulo"))
        print(f"[ok] {p.parent / name}.png")


if __name__ == "__main__":
    main()
