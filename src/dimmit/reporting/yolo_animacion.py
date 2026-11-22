"""Animación (MP4) de cómo YOLOv10 procesa una foto real de la toma, con los números reales del modelo de DIMMIT:
foto -> tensor -> mapas de rasgos -> cuadrículas 80/40/20 -> barrido celda por celda (los 68 números de cada una)
-> la celda ganadora arma su caja -> de 300 candidatas a 1 caja (umbral calibrado con el IDU) -> resultado.

Uso: python -m dimmit.reporting.yolo_animacion [--foto Prueba_6/Prueba_6_007] [--fuentes carpeta_Boriboon]
Salida: reports/grabacion/yolo_paso_a_paso.mp4 (lo reproduce el paso 7 de scripts/proceso_completo.sh).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from dimmit.utils.io import REPO_ROOT, ensure_dir

W, H, FPS = 1600, 900, 15
NAVY, DARK, TEAL, TEAL_L, PERI, WHITE, MUTED = (15, 14, 71), (4, 4, 20), (90, 183, 187), (172, 217, 220), (58, 70, 160), (255, 255, 255), (185, 188, 224)
AMBER, RED = (224, 165, 38), (208, 59, 59)
CLASES = ["Grieta longitudinal", "Grieta transversal", "Piel de cocodrilo", "Bache"]
COL_CLS = [(127, 168, 232), (179, 157, 219), TEAL, AMBER]
PX, PY, PS = 60, 170, 680  # panel de imagen (cuadrado)
RX = 800  # panel derecho
OUT = REPO_ROOT / "reports" / "grabacion"


class Fuentes:
    def __init__(self, carpeta=None):
        dv = Path("/usr/share/fonts/truetype/dejavu")
        def f(nombre, alt):
            p = Path(carpeta) / nombre if carpeta else None
            return str(p) if p and p.exists() else str(dv / alt)
        self.heavy, self.bold, self.reg = f("Boriboon-Heavy.otf", "DejaVuSans-Bold.ttf"), f("Boriboon-Bold.otf", "DejaVuSans-Bold.ttf"), f("Boriboon-Regular.otf", "DejaVuSans.ttf")
        self.cache = {}

    def __call__(self, peso, tam):
        k = (peso, tam)
        if k not in self.cache:
            self.cache[k] = ImageFont.truetype(getattr(self, peso), tam)
        return self.cache[k]


# --- datos reales del modelo sobre la foto -------------------------------------------------------------------
def analizar(foto):
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    net = YOLO(str(REPO_ROOT / "models/release/yolov10n_rdd2020.pt")).model.eval()
    img = np.array(Image.open(REPO_ROOT / "data/campo/fotos" / f"{foto}.jpg").convert("RGB"))
    lb = LetterBox((640, 640), auto=False, stride=32)(image=img[:, :, ::-1])[:, :, ::-1].copy()
    x = torch.from_numpy(lb).permute(2, 0, 1)[None].float() / 255
    feats = {}
    for i in (2, 4, 6, 8, 16, 19, 22):
        net.model[i].register_forward_hook(lambda m, a, o, i=i: feats.__setitem__(i, o.detach()[0].abs().mean(0).numpy()))
    with torch.no_grad():
        out, aux = net(x)
    escalas = []
    for t, stride in zip(aux["one2one"], (8, 16, 32)):
        t = t[0]
        g = t.shape[-1]
        escalas.append({"stride": stride, "g": g, "probs": torch.sigmoid(t[64:]).numpy(),  # 4 x g x g
                        "dist": torch.softmax(t[:64].view(4, 16, g, g), 1).numpy()})  # 4 x 16 x g x g
    # celda ganadora (máxima probabilidad de cualquier clase en las 3 escalas)
    best = max(((s, int(e["probs"].max(0).argmax())) for s, e in enumerate(escalas)), key=lambda p: escalas[p[0]]["probs"].max(0).flat[p[1]])
    s, k = best
    e = escalas[s]
    row, col = divmod(k, e["g"])
    exp = (e["dist"][:, :, row, col] * np.arange(16)[None]).sum(1) * e["stride"]
    cx, cy = (col + 0.5) * e["stride"], (row + 0.5) * e["stride"]
    return {"img": img, "lb": lb, "feats": feats, "escalas": escalas, "best": (s, row, col), "exp": exp, "centro": (cx, cy),
            "caja": (cx - exp[0], cy - exp[1], cx + exp[2], cy + exp[3]), "top300": out[0].numpy(),
            "n_params": sum(p.numel() for p in net.parameters())}


# --- dibujo ------------------------------------------------------------------------------------------------------
def fondo():
    g = np.linspace(0, 1, H)[:, None, None]
    a = np.array(NAVY)[None, None] * (1 - g) + np.array(DARK)[None, None] * g
    return Image.fromarray(np.repeat(a, W, 1).astype(np.uint8))


def lienzo(base, F, paso, titulo, sub=""):
    im = base.copy()
    d = ImageDraw.Draw(im)
    d.text((PX, 34), paso, font=F("bold", 22), fill=TEAL)
    d.text((PX, 64), titulo, font=F("heavy", 44), fill=WHITE)
    if sub:
        d.text((PX, 122), sub, font=F("reg", 22), fill=MUTED)
    return im, d


def a_panel(v):
    return v * PS / 640


def pegar_lb(im, lb, alpha=1.0):
    p = Image.fromarray(lb).resize((PS, PS), Image.BILINEAR)
    if alpha < 1:
        base = im.crop((PX, PY, PX + PS, PY + PS))
        p = Image.blend(base, p, alpha)
    im.paste(p, (PX, PY))


def overlay(im, dibujar):
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dibujar(ImageDraw.Draw(ov))
    im.alpha_composite(ov) if im.mode == "RGBA" else im.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"))


def texto(d, F, x, y, s, tam=24, peso="reg", fill=WHITE, esp=8):
    d.multiline_text((x, y), s, font=F(peso, tam), fill=fill, spacing=esp)


def flecha(d, p0, p1, color, w=4):
    d.line([p0, p1], fill=color, width=w)
    ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    a, L = 0.5, 14
    pts = [p1, (p1[0] - L * np.cos(ang - a), p1[1] - L * np.sin(ang - a)), (p1[0] - L * np.cos(ang + a), p1[1] - L * np.sin(ang + a))]
    d.polygon(pts, fill=color)


def barras_clase(d, F, x, y, probs, resaltar=None, ancho=330):
    for k, (c, p) in enumerate(zip(CLASES, probs)):
        yy = y + k * 52
        hi = k == resaltar
        d.text((x, yy), c, font=F("bold" if hi else "reg", 22), fill=WHITE)
        d.rectangle([x + 250, yy + 4, x + 250 + ancho, yy + 30], fill=(40, 44, 100))
        d.rectangle([x + 250, yy + 4, x + 250 + max(int(ancho * float(p)), 2), yy + 30], fill=TEAL if hi else COL_CLS[k])
        d.text((x + 260 + ancho, yy), f"{float(p):.2f}".replace(".", ","), font=F("bold" if hi else "reg", 22), fill=WHITE)


def histos(d, F, x, y, dist, stride, alto=70, paso=6, marcar=True):
    nombres = ("izq.", "arriba", "der.", "abajo")
    for k in range(4):
        x0 = x + k * 120
        m = dist[k].max() + 1e-9
        for b in range(16):
            hh = alto * dist[k, b] / m
            d.rectangle([x0 + b * paso, y + alto - hh, x0 + b * paso + paso - 2, y + alto], fill=PERI)
        e = float((dist[k] * np.arange(16)).sum())
        if marcar:
            d.line([x0 + e * paso + 2, y - 4, x0 + e * paso + 2, y + alto + 2], fill=TEAL, width=3)
        d.text((x0, y + alto + 8), nombres[k], font=F("reg", 18), fill=MUTED)
        d.text((x0, y + alto + 32), f"{e * stride:.0f} px", font=F("bold", 20), fill=WHITE)


def cmap(v):
    """0-1 -> color (azul -> turquesa -> amarillo -> rojo)."""
    stops = np.array([(40, 44, 120), TEAL, (240, 200, 60), RED], float)
    t = np.clip(v, 0, 1) * 3
    i = np.minimum(t.astype(int), 2)
    f = (t - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def mapa(fm, tam):
    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-9)
    return Image.fromarray(cmap(fm)).resize((tam, tam), Image.NEAREST)


# --- escenas ---------------------------------------------------------------------------------------------------
def escenas(A, F, ctx):
    base = fondo()
    lb, img = A["lb"], A["img"]
    total = 9
    T = lambda n: f"CÓMO VE YOLO UNA FOTO · PASO {n}/{total}"
    # 1. la foto
    for f in range(60):
        im, d = lienzo(base, F, T(1), "LA FOTO", f"Foto real de la toma del 15-nov · {ctx['foto_txt']}")
        p = Image.fromarray(img).resize((PS, int(PS * img.shape[0] / img.shape[1])))
        im.paste(p, (PX, PY + (PS - p.size[1]) // 2))
        texto(d, F, RX, PY + 20, "Foto de celular en la vía", 30, "bold")
        texto(d, F, RX, PY + 70, f"{img.shape[1]} × {img.shape[0]} píxeles\n3 colores por píxel (rojo, verde, azul)\nvalores de 0 a 255", 24, fill=MUTED)
        yield im
    # 2. el tensor
    for f in range(75):
        im, d = lienzo(base, F, T(2), "EL TENSOR", "Se escala a 640 × 640 y se rellena con gris (letterbox)")
        pegar_lb(im, lb)
        texto(d, F, RX, PY + 20, "3 × 640 × 640 = 1.228.800 números", 30, "bold")
        texto(d, F, RX, PY + 70, "cada uno entre 0 y 1 (valor del píxel / 255)", 24, fill=MUTED)
        n = min(3, f // 15 + 1)
        for j in range(n):
            ch = lb[:, :, j].astype(float) / 255
            tint = np.zeros((640, 640, 3))
            tint[:, :, j] = 1
            arr = (40 + 215 * ch[..., None] * (0.35 + 0.65 * tint)).astype(np.uint8)
            t = Image.fromarray(arr).resize((200, 200))
            im.paste(t, (RX + j * 230, PY + 150))
            d.text((RX + j * 230, PY + 360), ("R · rojo", "G · verde", "B · azul")[j], font=F("bold", 22), fill=WHITE)
        yield im
    # 3. backbone y neck
    capas = [(2, "160×160", "32"), (4, "80×80", "64"), (6, "40×40", "128"), (8, "20×20", "256")]
    for f in range(105):
        im, d = lienzo(base, F, T(3), "BACKBONE Y NECK", "Convoluciones que resumen la imagen: menos píxeles, más rasgos")
        pegar_lb(im, lb)
        n = min(4, f // 18 + 1)
        texto(d, F, RX, PY, "Backbone: activación real de esta foto", 26, "bold")
        for j in range(n):
            i, res, ch = capas[j]
            im.paste(mapa(A["feats"][i], 150), (RX + j * 175, PY + 50))
            d.text((RX + j * 175, PY + 210), res, font=F("bold", 20), fill=WHITE)
            d.text((RX + j * 175, PY + 236), f"{ch} canales", font=F("reg", 18), fill=MUTED)
        if f >= 75:
            texto(d, F, RX, PY + 300, "Neck (FPN + PAN): 3 mapas para detectar", 26, "bold")
            for j, (i, lab) in enumerate(((16, "P3 · 80×80 · fisuras finas"), (19, "P4 · 40×40 · daños medianos"), (22, "P5 · 20×20 · baches"))):
                im.paste(mapa(A["feats"][i], 110), (RX + j * 245, PY + 350))
                d.text((RX + j * 245, PY + 470), lab, font=F("reg", 18), fill=MUTED)
        yield im
    # 4. cuadrículas
    grids = [(80, "P3", 6400), (40, "P4", 1600), (20, "P5", 400)]
    for gi, (g, nom, n_c) in enumerate(grids):
        for f in range(40):
            im, d = lienzo(base, F, T(4), "LAS CUADRÍCULAS", "Cada mapa divide la imagen en celdas; cada celda propone un daño")
            pegar_lb(im, lb)
            k = min(g + 1, int((f + 1) / 25 * (g + 1)) + 1)
            def dib(o, g=g, k=k):
                for t in range(k):
                    v = PX + t * PS / g
                    o.line([(v, PY), (v, PY + PS)], fill=(255, 255, 255, 110), width=1)
                    h_ = PY + t * PS / g
                    o.line([(PX, h_), (PX + PS, h_)], fill=(255, 255, 255, 110), width=1)
            overlay(im, dib)
            acum = sum(c for _, _, c in grids[:gi]) + n_c * min(1, (f + 1) / 25)
            for j, (gg, nn, cc) in enumerate(grids):
                col = WHITE if j <= gi else MUTED
                texto(d, F, RX, PY + 20 + j * 60, f"{nn}: {gg} × {gg} = {cc:,} celdas".replace(",", "."), 28, "bold" if j == gi else "reg", col)
            texto(d, F, RX, PY + 240, f"Total: {int(acum):,} celdas candidatas".replace(",", "."), 34, "heavy", TEAL)
            yield im
    # 5. barrido de P5: cada celda calcula 68 números
    s5 = A["escalas"][2]
    g = s5["g"]
    mx = s5["probs"].max(0)
    n_fr = 120
    for f in range(n_fr + 30):
        im, d = lienzo(base, F, T(5), "CADA CELDA CALCULA 68 NÚMEROS", "Barrido de la cuadrícula P5 (20 × 20): 64 para la caja + 4 para la clase")
        pegar_lb(im, lb)
        hasta = min(g * g, int((f + 1) / n_fr * g * g))
        cel = max(hasta - 1, 0)
        r, c = divmod(cel, g)
        def dib(o):
            for q in range(hasta):
                rr, cc = divmod(q, g)
                v = mx[rr, cc] / 0.35
                color = tuple(int(u) for u in cmap(np.array([v]))[0])
                o.rectangle([PX + cc * PS / g, PY + rr * PS / g, PX + (cc + 1) * PS / g, PY + (rr + 1) * PS / g], fill=color + (int(25 + 175 * min(v, 1)),))
            if hasta < g * g:
                o.rectangle([PX + c * PS / g, PY + r * PS / g, PX + (c + 1) * PS / g, PY + (r + 1) * PS / g], outline=(255, 255, 255, 255), width=3)
        overlay(im, dib)
        texto(d, F, RX, PY, f"Celda fila {r + 1}, columna {c + 1}" if hasta < g * g else "Las 400 celdas de P5 calculadas", 28, "bold")
        texto(d, F, RX, PY + 52, "DÓNDE · 4 lados × 16 probabilidades", 22, "bold", TEAL_L)
        histos(d, F, RX, PY + 92, s5["dist"][:, :, r, c], s5["stride"])
        texto(d, F, RX, PY + 250, "QUÉ · probabilidad de cada daño", 22, "bold", TEAL_L)
        barras_clase(d, F, RX, PY + 290, s5["probs"][:, r, c], resaltar=int(s5["probs"][:, r, c].argmax()))
        texto(d, F, RX, PY + 510, f"color = probabilidad más alta de la celda · {hasta}/400", 20, fill=MUTED)
        yield im
    # 6. la celda ganadora arma su caja
    s, r, c = A["best"]
    e = A["escalas"][s]
    cx, cy = (a_panel(v) for v in A["centro"])
    x1, y1, x2, y2 = (a_panel(v) for v in A["caja"])
    cls = int(e["probs"][:, r, c].argmax())
    for f in range(105):
        im, d = lienzo(base, F, T(6), "LA CELDA GANADORA ARMA SU CAJA", f"Celda de P{s + 3} (fila {r + 1}, columna {c + 1}): la de mayor probabilidad")
        pegar_lb(im, lb)
        gs = PS / e["g"]
        cxp, cyp = PX + cx, PY + cy
        t = min(1, f / 45)
        def dib(o):
            o.rectangle([PX + c * gs, PY + r * gs, PX + (c + 1) * gs, PY + (r + 1) * gs], fill=TEAL + (200,), outline=WHITE, width=2)
            if f > 50:
                a = min(255, int((f - 50) * 12))
                o.rectangle([PX + x1, PY + y1, PX + x2, PY + y2], outline=TEAL + (a,), width=5)
        overlay(im, dib)
        for (dx, dy) in ((-(cx - x1), 0), (0, -(cy - y1)), (x2 - cx, 0), (0, y2 - cy)):
            flecha(d, (cxp, cyp), (cxp + dx * t, cyp + dy * t), WHITE, 4)
        texto(d, F, RX, PY, "Las 4 distancias desde el centro de la celda", 26, "bold")
        histos(d, F, RX, PY + 50, e["dist"][:, :, r, c], e["stride"])
        texto(d, F, RX, PY + 210, "distancia = promedio ponderado de las 16 probabilidades", 20, fill=MUTED)
        texto(d, F, RX, PY + 270, "Y la clase:", 26, "bold")
        barras_clase(d, F, RX, PY + 315, e["probs"][:, r, c], resaltar=cls)
        yield im
    # 7. de 300 candidatas a 1
    top = A["top300"]
    tau = ctx["tau"]
    etapas = [("300 mejores celdas (cabeza uno a uno, sin NMS)", top[:, 4] >= -1),
              ("confianza de al menos 0,05 (inferencia)", top[:, 4] >= 0.05),
              ("umbral calibrado con el IDU (0,12–0,20)", np.array([cf >= tau[int(k)] for cf, k in zip(top[:, 4], top[:, 5])])),
              ("filtros de campo: vía, superficie, persistencia, escalas", np.array([cf >= tau[int(k)] for cf, k in zip(top[:, 4], top[:, 5])]))]
    for f in range(150):
        k = min(3, f // 35)
        m = etapas[k][1]
        im, d = lienzo(base, F, T(7), "DE 300 CANDIDATAS A 1 CAJA", "Cada caja trae su confianza; los umbrales y filtros deciden cuáles se quedan")
        pegar_lb(im, lb)
        def dib(o):
            for (bx1, by1, bx2, by2, cf, kk), keep in zip(top, m):
                if not keep:
                    continue
                a = int(110 + 145 * min(1, cf / 0.35))
                o.rectangle([PX + a_panel(bx1), PY + a_panel(by1), PX + a_panel(bx2), PY + a_panel(by2)], outline=COL_CLS[int(kk)] + (a,), width=4 if cf > 0.15 else 2)
        overlay(im, dib)
        for j, (txt, mm) in enumerate(etapas):
            col = WHITE if j <= k else MUTED
            cy_ = PY + j * 70 + 14
            if j < k:
                d.ellipse([RX, cy_ - 8, RX + 16, cy_ + 8], fill=TEAL)
            elif j == k:
                d.ellipse([RX, cy_ - 8, RX + 16, cy_ + 8], outline=TEAL, width=3)
            else:
                d.ellipse([RX + 4, cy_ - 4, RX + 12, cy_ + 4], fill=MUTED)
            texto(d, F, RX + 30, PY + j * 70, txt, 22, "bold" if j == k else "reg", col)
            if j <= k:
                nq = int(mm.sum())
                texto(d, F, RX + 30, PY + 30 + j * 70, f"queda{'' if nq == 1 else 'n'} {nq} caja{'' if nq == 1 else 's'}", 20, "reg", TEAL_L)
        texto(d, F, RX, PY + 320, f"{int(m.sum())}", 120, "heavy", TEAL)
        texto(d, F, RX + 10, PY + 460, "caja en pantalla" if int(m.sum()) == 1 else "cajas en pantalla", 24, fill=MUTED)
        yield im
    # 8. resultado
    for f in range(75):
        im, d = lienzo(base, F, T(8), "EL RESULTADO", "Caja + clase + confianza: lo que entra al score del tramo")
        pegar_lb(im, lb)
        bx1, by1, bx2, by2, cf, kk = top[0]
        def dib(o):
            o.rectangle([PX + a_panel(bx1), PY + a_panel(by1), PX + a_panel(bx2), PY + a_panel(by2)], outline=TEAL + (255,), width=6)
            o.rectangle([PX + a_panel(bx1), PY + a_panel(by1) - 44, PX + a_panel(bx1) + 330, PY + a_panel(by1) - 4], fill=TEAL + (255,))
        overlay(im, dib)
        d.text((PX + a_panel(bx1) + 12, PY + a_panel(by1) - 40), f"{CLASES[int(kk)].lower()} · {cf:.2f}".replace(".", ","), font=F("heavy", 24), fill=NAVY)
        texto(d, F, RX, PY, f"{CLASES[int(kk)]} · confianza {cf:.2f}".replace(".", ","), 30, "bold")
        texto(d, F, RX, PY + 60, ctx["resultado"], 24, fill=MUTED, esp=12)
        yield im
    # 9. montaje: lo mismo foto por foto
    for j, (pth, cajas) in enumerate(ctx["montaje"]):
        foto = Image.open(pth).convert("RGB")
        ph = int(PS * foto.size[1] / foto.size[0])
        p = foto.resize((PS, ph))
        for f in range(22):
            im, d = lienzo(base, F, T(9), "LO MISMO EN CADA FOTO", f"{ctx['n_fotos']} fotos de la toma · {ctx['n_cajas']} cajas conservadas tras umbral y filtros")
            im.paste(p, (PX, PY + (PS - ph) // 2))
            def dib(o):
                for cx_, cy_, w_, h_, cl, cf in cajas:
                    k_ = ["D00", "D10", "D20", "D40"].index(cl)
                    o.rectangle([PX + (cx_ - w_ / 2) * PS, PY + (PS - ph) // 2 + (cy_ - h_ / 2) * ph, PX + (cx_ + w_ / 2) * PS,
                                 PY + (PS - ph) // 2 + (cy_ + h_ / 2) * ph], outline=COL_CLS[k_] + (255,), width=5)
            overlay(im, dib)
            texto(d, F, RX, PY, Path(pth).stem.replace("_", " "), 30, "bold")
            for q, (cx_, cy_, w_, h_, cl, cf) in enumerate(cajas):
                k_ = ["D00", "D10", "D20", "D40"].index(cl)
                texto(d, F, RX, PY + 60 + q * 40, f"{CLASES[k_].lower()} · {cf:.2f}".replace(".", ","), 24, "reg", COL_CLS[k_])
            yield im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--foto", default="Prueba_6/Prueba_6_007")
    ap.add_argument("--fuentes", default=None)
    args = ap.parse_args()
    F = Fuentes(args.fuentes)
    print(f"[animación] corriendo YOLOv10n sobre {args.foto} y extrayendo activaciones, cuadrículas y vectores…", flush=True)
    A = analizar(args.foto)
    feat = REPO_ROOT / "data/campo/features"
    cal = json.loads((feat / "calibracion_idu.json").read_text())
    tau = [cal["tau_campo_calibrado"][k] for k in ("D00", "D10", "D20", "D40")]
    v = pd.read_parquet(feat / "vias.parquet")
    prueba = args.foto.split("/")[0]
    row = v[(v["nivel"] == "tramo") & (v["prueba"] == prueba)].iloc[0]
    resultado = (f"Esta caja suma a la densidad del tramo ({prueba.replace('_', ' ')}).\n"
                 f"Score del tramo: {row['score_calidad']:.1f}, «{row['estado_idu_equivalente']}» en la escala del IDU.\n"
                 f"El sensor mide un bache de {row['profundidad_max_cm']:.1f} cm.\n"
                 f"Etiqueta: {row['etiqueta'].upper().replace('INTERVENCION', 'INTERVENCIÓN')}").replace(".", ",").replace(",\n", ".\n")
    det = pd.read_parquet(feat / "detecciones_filtradas.parquet")
    fr = pd.read_parquet(REPO_ROOT / "data/campo/index/fotogramas.parquet")
    ev = fr[fr["conjunto"] == "evaluacion"]
    keep = det[det["conservada"] & det["image_id"].isin(ev["image_id"])]
    montaje = []
    for iid, g in keep.groupby("image_id"):
        if iid != args.foto.split("/")[1]:
            montaje.append((ev.set_index("image_id").loc[iid, "path"], list(zip(g["cx"], g["cy"], g["w"], g["h"], g["cls"], g["conf"]))))
    ctx = {"tau": tau, "resultado": resultado, "montaje": montaje[:7], "n_fotos": len(ev), "n_cajas": len(keep),
           "foto_txt": f"{prueba.replace('_', ' ')}, Carrera 5 detrás del CAI Rosario"}
    out = ensure_dir(OUT) / "yolo_paso_a_paso.mp4"
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    n = 0
    for n, fr_ in enumerate(escenas(A, F, ctx), 1):
        ff.stdin.write(np.asarray(fr_.convert("RGB")).tobytes())
        if n % 150 == 0:
            print(f"[animación] {n} cuadros ({n / FPS:.0f} s de video)", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"[ok] {out.relative_to(REPO_ROOT)} · {n} cuadros · {n / FPS:.0f} s · {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
