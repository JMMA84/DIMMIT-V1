"""Video de cómo la red YOLOv10n de DIMMIT procesa una foto, hecho solo con tensores reales de una pasada.

Nada se dibuja "a mano": cada número, mapa y caja sale de la red al procesar la foto.
  * PyTorch register_forward_hook   captura la salida de cada una de las 23 capas y de la cabeza v10Detect
  * Matplotlib                      convierte tensores (filtros, mapas de activación, vector de 68 números,
                                    distribuciones DFL, puntajes) en imágenes de colores
  * Captum (LayerGradCam)           mapa de calor Grad-CAM de la clase detectada
  * OpenCV                          dibuja la ventana 3×3 de la convolución, las cuadrículas, las flechas y las
                                    cajas (cv2.rectangle, cv2.line, cv2.putText) y une los fotogramas con
                                    cv2.VideoWriter; ffmpeg (vía imageio-ffmpeg) lo pasa a H.264
Comprobaciones dentro del video: la convolución recalculada a mano coincide con el hook, y las cajas
decodificadas desde la cuadrícula coinciden con la salida de la red y con YOLO.predict() de Ultralytics.

Nota: YOLO no usa ventana deslizante sobre la imagen (mira la imagen una sola vez); la única ventana deslizante
real es el kernel 3×3 de cada convolución, que es la que se anima aquí con los pesos reales de la capa 0.

Uso: python -m dimmit.reporting.video_yolo [--foto Prueba_6/Prueba_6_007] [--conf 0.16] [--fps 15]
Salida: reports/grabacion/yolo_por_dentro_<foto>.mp4
"""
import argparse
import shutil
import subprocess
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from dimmit.utils.io import REPO_ROOT, ensure_dir  # noqa: E402

PESOS = REPO_ROOT / "models/release/yolov10n_rdd2020.pt"
FOTOS = REPO_ROOT / "data/campo/fotos"
OUT = REPO_ROOT / "reports/grabacion"

W, H = 1600, 900
X0, Y0, S = 30, 100, 1.5  # imagen letterbox 640 px a escala 1.5 en el panel izquierdo
PX = 1020  # inicio del panel derecho
FONDO, NAVY, TEAL, CLARO, BLANCO, AMARILLO, ROJO = ((4, 4, 20), (15, 14, 71), (90, 183, 187), (172, 217, 220),
                                                    (240, 240, 245), (250, 205, 70), (235, 80, 80))
_TTF = font_manager.findfont("DejaVu Sans")
_TTFB = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
_FONTS = {}


def fuente(tam, negrita=False):
    k = (tam, negrita)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(_TTFB if negrita else _TTF, tam)
    return _FONTS[k]


def texto(img, lineas, x, y, tam=22, color=BLANCO, negrita=False, sep=1.35):
    """Texto con tildes (Pillow); las etiquetas de las cajas van con cv2.putText."""
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for i, ln in enumerate([lineas] if isinstance(lineas, str) else lineas):
        d.text((x, y + i * tam * sep), ln, font=fuente(tam, negrita), fill=color)
    img[:] = np.asarray(pil)
    return img


def lienzo(paso, titulo, sub=""):
    c = np.zeros((H, W, 3), np.uint8)
    c[:] = FONDO
    c[:78] = NAVY
    c[78:82] = TEAL
    texto(c, paso, 30, 12, 18, CLARO, True)
    texto(c, titulo, 30, 38, 28, BLANCO, True)
    if sub:
        texto(c, sub, W - 30 - int(fuente(17).getlength(sub)), 14, 17, CLARO)
    return c


def pegar(c, img, x, y, w=None, h=None):
    if w or h:
        hh, ww = img.shape[:2]
        if w and h:
            r = min(w / ww, h / hh)
        else:
            r = w / ww if w else h / hh
        img = cv2.resize(img, (max(1, int(ww * r)), max(1, int(hh * r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_NEAREST)
    hh, ww = img.shape[:2]
    c[y:y + hh, x:x + ww] = img[: H - y, : W - x]
    return ww, hh


def fig_a_img(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return a


def estilo(ax):
    ax.set_facecolor("#040414")
    for s in ax.spines.values():
        s.set_color("#5AB7BB")
    ax.tick_params(colors="#ACD9DC", labelsize=8)
    ax.title.set_color("white")
    ax.xaxis.label.set_color("#ACD9DC")
    ax.yaxis.label.set_color("#ACD9DC")


def figura(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#040414")
    return fig


def a_disp(x, y):
    """Coordenadas de la imagen letterbox (px) -> coordenadas del lienzo."""
    return int(round(X0 + x * S)), int(round(Y0 + y * S))


def mapa_color(m, cmap=cv2.COLORMAP_JET, lo=None, hi=None):
    lo = float(m.min()) if lo is None else lo
    hi = float(m.max()) if hi is None else hi
    u = np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.applyColorMap((u * 255).astype(np.uint8), cmap)[:, :, ::-1], u


class Video:
    def __init__(self, ruta, fps):
        self.ruta, self.fps, self.n = ruta, fps, 0
        self.w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not self.w.isOpened():
            raise SystemExit(f"cv2.VideoWriter no pudo abrir {ruta}")

    def add(self, frame, seg=None, n=None):
        k = n if n is not None else max(1, int(round((seg or 1 / self.fps) * self.fps)))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for _ in range(k):
            self.w.write(bgr)
        self.n += k

    def close(self):
        self.w.release()


# ------------------------------------------------------------------------------------------------ pasada real
def pasada(foto, conf):
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.modules.head import Detect

    y = YOLO(str(PESOS))
    net = y.model
    net.fuse(verbose=False)
    net.eval()
    bgr = cv2.imread(str(FOTOS / f"{foto}.jpg"))
    lb_bgr = LetterBox((640, 640), auto=True, stride=int(net.stride.max()))(image=bgr)  # igual que YOLO.predict
    lb = lb_bgr[:, :, ::-1].copy()
    x = torch.from_numpy(lb).permute(2, 0, 1)[None].float() / 255

    feats = {}
    hs = [m.register_forward_hook(lambda m, i, o: feats.__setitem__(m.i, o.detach())) for m in net.model[:-1]]
    with torch.no_grad():
        out, aux = net(x)
    for h in hs:
        h.remove()
    head = net.model[-1]
    o2o = [t.detach() for t in aux["one2one"]]

    # decodificación propia de la cuadrícula (DFL + anclas) -> debe coincidir con la salida de la red
    filas = []
    for s, (t, st) in enumerate(zip(o2o, head.stride.tolist())):
        _, _, gh, gw = t.shape
        dist_p = t[0, :64].reshape(4, 16, gh, gw).softmax(1)
        dist = (dist_p * torch.arange(16.0)[None, :, None, None]).sum(1)  # (4,gh,gw) en celdas
        cls = t[0, 64:].sigmoid()
        gy, gx = torch.meshgrid(torch.arange(gh) + 0.5, torch.arange(gw) + 0.5, indexing="ij")
        caja = torch.stack([gx - dist[0], gy - dist[1], gx + dist[2], gy + dist[3]]) * st
        filas.append(torch.cat([caja, cls]).reshape(8, -1).T)
    pred = torch.cat(filas)[None]  # (1, N, 8)
    mio = Detect.postprocess(pred.clone(), head.max_det, head.nc)
    dif = float((mio - out).abs().max())

    r = y.predict(str(FOTOS / f"{foto}.jpg"), imgsz=640, conf=conf, verbose=False)[0]
    return dict(y=y, net=net, head=head, x=x, lb=lb, bgr=bgr, feats=feats, out=out[0], o2o=o2o, pred=pred[0],
                dif=dif, pred_ultra=r, names=net.names, strides=head.stride.tolist())


def celda_de(P, det):
    """Índice de celda (escala, fila, columna) cuya caja y puntaje producen la detección det."""
    N = P["pred"]
    k = int(((N[:, :4] - det[:4]).abs().sum(1) + (N[:, 4 + int(det[5])] - det[4]).abs()).argmin())
    acc = 0
    for s, t in enumerate(P["o2o"]):
        n = t.shape[2] * t.shape[3]
        if k < acc + n:
            i = k - acc
            return s, i // t.shape[3], i % t.shape[3], k
        acc += n


# ------------------------------------------------------------------------------------------------ escenas
def img_base(P, c, oscurecer=0.0):
    im = cv2.resize(P["lb"], None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
    if oscurecer:
        im = (im * (1 - oscurecer)).astype(np.uint8)
    pegar(c, im, X0, Y0)
    return c


def escena_titulo(v, P, foto):
    c = lienzo("DIMMIT · YOLOv10n entrenado con RDD2020", "Cómo la red procesa una foto, con sus tensores reales")
    texto(c, ["Herramientas: PyTorch (register_forward_hook) · OpenCV (dibujo y VideoWriter) ·",
              "Matplotlib (tensores -> colores) · Captum (Grad-CAM) · Ultralytics (pesos y comprobación)"], 30, 120, 26, CLARO)
    rgb = P["bgr"][:, :, ::-1]
    pegar(c, rgb, 30, 230, h=560)
    texto(c, [f"Foto de campo: {foto}.jpg", f"{rgb.shape[1]} × {rgb.shape[0]} px",
              "Calle de la Candelaria, detrás del CAI,", "cerca a la Universidad del Rosario (toma 15-nov-2026)"], 1060, 260, 24)
    texto(c, ["Todo lo que aparece sale de una sola pasada", "de esta foto por la red: no hay valores", "inventados ni animaciones de ejemplo."], 1060, 470, 24, AMARILLO)
    v.add(c, 5)


def escena_entrada(v, P):
    x, lb = P["x"], P["lb"]
    h0, w0 = P["bgr"].shape[:2]
    c = lienzo("Paso 1 · Entrada", "La foto se convierte en un tensor", "LetterBox de Ultralytics, igual que YOLO.predict")
    img_base(P, c)
    pad = (lb.shape[0] - round(h0 * 640 / w0)) / 2
    for yy in (0, lb.shape[0] - pad):
        cv2.rectangle(c, a_disp(0, yy), a_disp(640, yy + pad), AMARILLO, 2)
    texto(c, [f"{w0}×{h0} px -> se reduce a 640 de ancho", f"y se rellena a {lb.shape[1]}×{lb.shape[0]} (múltiplo de 32)",
              "(bandas amarillas = relleno gris 114)", "", f"tensor x: {tuple(x.shape)}", "= (lote, canales RGB, alto, ancho)",
              "valores de 0 a 1 (píxel / 255)"], PX, 110, 24)
    fig = figura(540, 250)
    for k, (nombre, col) in enumerate(zip("RGB", ("Reds", "Greens", "Blues"))):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.imshow(x[0, k, 100:108, 300:308].numpy(), cmap=col, vmin=0, vmax=1)
        for (i, j), val in np.ndenumerate(x[0, k, 100:108, 300:308].numpy()):
            ax.text(j, i, f"{val:.2f}"[1:], ha="center", va="center", fontsize=5.5, color="black")
        ax.set_title(f"canal {nombre}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        estilo(ax)
    fig.suptitle("x[0, :, 100:108, 300:308] (8×8 píxeles reales)", color="#ACD9DC", fontsize=9)
    pegar(c, fig_a_img(fig), PX, 440)
    cv2.rectangle(c, a_disp(300, 100), a_disp(308, 108), TEAL, 2)
    texto(c, "La red solo ve números: 3 matrices (rojo, verde, azul) del tamaño de la imagen.", 30, 700, 24, CLARO)
    v.add(c, 5)


def escena_convolucion(v, P):
    x, feats, net = P["x"], P["feats"], P["net"]
    conv0 = net.model[0].conv
    Wt, b = conv0.weight.detach(), conv0.bias.detach()
    f0 = feats[0][0]  # (16, h/2, w/2)
    ch = int(f0.flatten(1).std(1).argmax())
    oh, ow = f0.shape[1:]
    xp = F.pad(x, (1, 1, 1, 1))
    titulo = "La única ventana deslizante de YOLO: el kernel 3×3 de una convolución"
    sub = "capa 0 · Conv 3×3, stride 2, 16 filtros (BN fusionada)"

    # 1) los 16 filtros reales de la capa 0
    c = lienzo("Paso 2 · Convolución", titulo, sub)
    fig = figura(1540, 330)
    for k in range(16):
        ax = fig.add_subplot(2, 8, k + 1)
        w = Wt[k].permute(1, 2, 0).numpy()
        ax.imshow((w - w.min()) / (w.max() - w.min() + 1e-9), interpolation="nearest")
        ax.set_title(f"filtro {k}" + (" ◀" if k == ch else ""), fontsize=9, color="#FACD46" if k == ch else "white")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("conv0.weight: 16 filtros de 3×3×3 (pesos aprendidos en RDD2020, normalizados a color RGB)", color="#ACD9DC", fontsize=11)
    fig.subplots_adjust(top=0.84)
    pegar(c, fig_a_img(fig), 30, 110)
    fig = figura(1540, 330)
    for k in range(16):
        ax = fig.add_subplot(2, 8, k + 1)
        ax.imshow(f0[k].numpy(), cmap="viridis")
        ax.set_title(f"salida {k}", fontsize=9, color="#FACD46" if k == ch else "white")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"salida de la capa 0 capturada con register_forward_hook: {tuple(feats[0].shape)}", color="#ACD9DC", fontsize=11)
    fig.subplots_adjust(top=0.84)
    pegar(c, fig_a_img(fig), 30, 450)
    texto(c, f"Cada filtro recorre la imagen y produce un mapa; se sigue el filtro {ch} (el de mayor variación en esta foto).", 30, 800, 22, CLARO)
    v.add(c, 5)

    # 2) la ventana recorriendo la imagen; el mapa de salida se llena con los valores del hook
    mapa, _ = mapa_color(f0[ch].numpy(), cv2.COLORMAP_VIRIDIS)
    filas_i = np.linspace(4, oh - 5, 7).astype(int)
    cols_j = np.linspace(4, ow - 5, 11).astype(int)
    pos = [(i, j) for i in filas_i for j in cols_j]
    for n, (i, j) in enumerate(pos):
        c = lienzo("Paso 2 · Convolución", titulo, sub)
        img_base(P, c, 0.15)
        x1, y1 = a_disp(2 * j - 1, 2 * i - 1)
        x2, y2 = a_disp(2 * j + 2, 2 * i + 2)
        cv2.rectangle(c, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), AMARILLO, 2)
        cv2.rectangle(c, (x1, y1), (x2, y2), ROJO, -1)
        # lupa 21×21 px alrededor de la ventana
        cy, cx = 2 * i, 2 * j
        crop = np.pad(P["lb"], ((10, 10), (10, 10), (0, 0)))[cy:cy + 21, cx:cx + 21]
        lupa = cv2.resize(crop, (210, 210), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(lupa, (90, 90), (119, 119), AMARILLO, 2)
        for t in (100, 110):
            cv2.line(lupa, (90, t), (119, t), AMARILLO, 1)
            cv2.line(lupa, (t, 90), (t, 119), AMARILLO, 1)
        pegar(c, lupa, PX, 100)
        patch = xp[0, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3]
        z = float((Wt[ch] * patch).sum() + b[ch])
        val = z / (1 + np.exp(-z))
        texto(c, ["ventana 3×3×3 (lupa)", f"posición salida ({i}, {j})", f"z = Σ w·x + b = {z:+.3f}", f"SiLU(z) = {val:+.4f}",
                  f"hook capa 0 = {float(f0[ch, i, j]):+.4f}"], PX + 225, 105, 21, BLANCO)
        revel = np.zeros_like(mapa)
        revel[:i] = mapa[:i]
        revel[i, :j + 1] = mapa[i, :j + 1]
        texto(c, f"mapa de salida del filtro {ch} ({oh}×{ow}), valores del hook", PX, 330, 18, CLARO)
        pegar(c, revel, PX, 360, w=560)
        texto(c, ["La ventana se mueve de 2 en 2 píxeles (stride 2): en cada posición multiplica 27 valores por 27 pesos,",
                  f"suma, aplica SiLU y escribe un número. En total {f'{oh * ow:,}'.replace(',', '.')} posiciones × 16 filtros (aquí se muestran {len(pos)})."],
              30, 700, 21, CLARO)
        v.add(c, n=3 if n else 15)
    c[360:360 + 336, PX:PX + 560] = 0
    pegar(c, mapa, PX, 360, w=560)
    v.add(c, 3)


def escena_capas(v, P):
    feats, net = P["feats"], P["net"]
    for i in range(len(net.model) - 1):
        m = net.model[i]
        f = feats[i][0]
        tipo = m.type.split(".")[-1]
        c = lienzo(f"Paso 3 · Mapas de activación · capa {i} de 22", f"capa {i}: {tipo} · entra desde {m.f} · salida {tuple(feats[i].shape)}",
                   "PyTorch register_forward_hook + Matplotlib")
        media = f.mean(0).numpy()
        col, _ = mapa_color(media)
        col = cv2.resize(col, (P["lb"].shape[1], P["lb"].shape[0]), interpolation=cv2.INTER_NEAREST)
        mezcla = cv2.addWeighted(P["lb"], 0.45, col, 0.55, 0)
        pegar(c, cv2.resize(mezcla, None, fx=S, fy=S), X0, Y0)
        top = f.flatten(1).mean(1).argsort(descending=True)[:16].tolist()
        fig = figura(560, 560)
        for k, cc in enumerate(top):
            ax = fig.add_subplot(4, 4, k + 1)
            ax.imshow(f[cc].numpy(), cmap="viridis")
            ax.set_title(f"canal {cc}", fontsize=8, color="white")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"16 de {f.shape[0]} canales (los más activos)", color="#ACD9DC", fontsize=10)
        pegar(c, fig_a_img(fig), PX, 100)
        etapa = "backbone: extrae bordes, texturas y formas" if i <= 10 else ("neck: mezcla escalas (sube y concatena)" if i <= 19 else "neck: baja de nuevo para objetos grandes")
        texto(c, [f"Izquierda: promedio de los {f.shape[0]} canales sobre la foto. Cada canal es un detector aprendido.",
                  f"Resolución {f.shape[1]}×{f.shape[2]} = 1 valor cada {P['lb'].shape[0] // f.shape[1]} px · {etapa}"], 30, 700, 21, CLARO)
        v.add(c, 1.0 if i not in (0, 10, 16, 19, 22) else 1.8)


def dibujar_rejilla(c, gh, gw, st, color=(30, 30, 50), filas=None):
    for r in range(gh + 1 if filas is None else filas):
        cv2.line(c, a_disp(0, r * st), a_disp(gw * st, r * st), color, 1)
    for k in range(gw + 1):
        cv2.line(c, a_disp(k * st, 0), a_disp(k * st, (gh if filas is None else min(filas, gh)) * st), color, 1)


def escena_cuadriculas(v, P):
    o2o, st, names = P["o2o"], P["strides"], P["names"]
    vmax = max(float(t[0, 64:].sigmoid().max()) for t in o2o)
    tot = 0
    for s, t in enumerate(o2o):
        _, _, gh, gw = t.shape
        tot += gh * gw
        prob = t[0, 64:].sigmoid()
        pmax, cls = prob.max(0)
        col, u = mapa_color(pmax.numpy(), cv2.COLORMAP_INFERNO, 0, vmax)
        cy, cx = np.unravel_index(int(pmax.argmax()), pmax.shape)
        tit = f"Cabeza v10Detect · escala P{int(np.log2(st[s]))}: cuadrícula {gh}×{gw} = {gh * gw:,} celdas".replace(",", ".")
        sub = f"stride {int(st[s])} px · cada celda = vector de 68 números"
        for fase in range(26):
            c = lienzo("Paso 4 · Cuadrículas", tit, sub)
            img_base(P, c, 0.25)
            if fase >= 12:
                a = min(1.0, (fase - 11) / 8)
                over = cv2.resize(P["lb"], None, fx=S, fy=S)
                capa = cv2.resize(col, (over.shape[1], over.shape[0]), interpolation=cv2.INTER_NEAREST)
                peso = (cv2.resize(u, (over.shape[1], over.shape[0]), interpolation=cv2.INTER_NEAREST)[..., None] * 0.85 + 0.15) * a
                c[Y0:Y0 + over.shape[0], X0:X0 + over.shape[1]] = (over * (1 - peso) + capa * peso).astype(np.uint8)
            dibujar_rejilla(c, gh, gw, st[s], (80, 80, 110) if s == 0 else (120, 120, 160), filas=int(gh * min(1, (fase + 1) / 10)))
            if fase >= 20:
                cv2.rectangle(c, a_disp(cx * st[s], cy * st[s]), a_disp((cx + 1) * st[s], (cy + 1) * st[s]), (80, 255, 120), 3)
            fig = figura(560, 330)
            ax = fig.add_subplot(111)
            im = ax.imshow(pmax.numpy(), cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"probabilidad máxima de clase por celda (sigmoide), P{int(np.log2(st[s]))}", fontsize=10)
            cb = fig.colorbar(im, ax=ax, fraction=0.03)
            cb.ax.tick_params(colors="#ACD9DC", labelsize=8)
            estilo(ax)
            pegar(c, fig_a_img(fig), PX, 100)
            texto(c, [f"tensor de la cabeza: {tuple(t.shape)}", "68 = 64 números de caja (4 lados × 16) + 4 clases",
                      f"celda más alta: fila {cy}, columna {cx}", f"p = {float(pmax[cy, cx]):.3f} · {names[int(cls[cy, cx])]}"], PX, 450, 21)
            texto(c, ["YOLO divide la imagen en 3 cuadrículas a la vez (finas para daños pequeños, gruesas para grandes) y cada",
                      "celda predice, de una sola pasada, una caja y la probabilidad de cada clase. Color = probabilidad real."], 30, 700, 21, CLARO)
            v.add(c, n=2 if fase < 25 else 30)
    return tot


def escena_vector(v, P, det, rank):
    s, cy, cx, k = celda_de(P, det)
    t, st, names = P["o2o"][s], P["strides"][s], P["names"]
    vec = t[0, :, cy, cx].numpy()
    dist_p = torch.from_numpy(vec[:64]).reshape(4, 16).softmax(1).numpy()
    dist = (dist_p * np.arange(16)).sum(1)
    prob = 1 / (1 + np.exp(-vec[64:]))
    ax_, ay_ = (cx + 0.5) * st, (cy + 0.5) * st
    caja = [ax_ - dist[0] * st, ay_ - dist[1] * st, ax_ + dist[2] * st, ay_ + dist[3] * st]
    cls = int(prob.argmax())
    ok = prob[cls] >= P["conf"]
    tit = f"Detección #{rank}: de la celda (fila {cy}, col {cx}) de P{int(np.log2(st))} a una caja"
    sub = "vector real de 68 números de esa celda"

    def base(fase):
        c = lienzo("Paso 5 · Del vector a la caja", tit, sub)
        img_base(P, c, 0.2)
        gh, gw = t.shape[2:]
        dibujar_rejilla(c, gh, gw, st, (70, 70, 100))
        cv2.rectangle(c, a_disp(cx * st, cy * st), a_disp((cx + 1) * st, (cy + 1) * st), (80, 255, 120), 3)
        cv2.circle(c, a_disp(ax_, ay_), 6, (80, 255, 120), -1)
        return c

    def panel(c, nivel):
        fig = figura(560, 580)
        ax = fig.add_axes([0.06, 0.86, 0.9, 0.07])
        ax.imshow(vec[None], cmap="coolwarm", aspect="auto")
        ax.set_yticks([])
        ax.set_xticks([0, 16, 32, 48, 64, 67])
        ax.set_title("los 68 números de la celda (salida cruda de la cabeza)", fontsize=10)
        estilo(ax)
        if nivel >= 1:
            for q, lado in enumerate(("izquierda", "arriba", "derecha", "abajo")):
                a = fig.add_axes([0.08 + (q % 2) * 0.47, 0.55 - (q // 2) * 0.235, 0.4, 0.15])
                a.bar(range(16), dist_p[q], color="#5AB7BB")
                a.axvline(dist[q], color="#FACD46", lw=2)
                a.set_title(f"{lado}: {dist[q]:.2f} celdas = {dist[q] * st:.0f} px", fontsize=9)
                a.set_ylim(0, 1)
                estilo(a)
            fig.text(0.06, 0.765, "DFL: 16 números por lado -> softmax -> distancia esperada", color="#ACD9DC", fontsize=9)
        if nivel >= 2:
            a = fig.add_axes([0.36, 0.05, 0.6, 0.13])
            a.barh(range(4), prob, color=["#FACD46" if q == cls else "#3A46A0" for q in range(4)])
            a.set_yticks(range(4))
            a.set_yticklabels([names[q] for q in range(4)], fontsize=8)
            a.set_xlim(0, 1)
            a.axvline(P["conf"], color="#EB5050", lw=1.5, ls="--")
            a.set_title(f"4 clases: sigmoide · umbral {P['conf']:.2f} (calibrado con el IDU)", fontsize=9)
            estilo(a)
        pegar(c, fig_a_img(fig), PX, 95)

    c = base(0)
    panel(c, 0)
    texto(c, [f"Celda de la cuadrícula P{int(np.log2(st))} (stride {int(st)} px) que generó esta detección.",
              f"Su ancla es el centro de la celda: ({ax_:.0f}, {ay_:.0f}) px."], 30, 700, 21, CLARO)
    v.add(c, 2.5)
    c = base(1)
    panel(c, 1)
    texto(c, ["Los primeros 64 números son 4 distribuciones de 16: la red no dice una distancia, dice qué tan probable",
              "es cada distancia (0 a 15 celdas). La distancia es el promedio ponderado (línea amarilla)."], 30, 700, 21, CLARO)
    v.add(c, 3)
    for f in range(1, 16):
        c = base(2)
        panel(c, 2)
        a = f / 15
        p0 = a_disp(ax_, ay_)
        for (dx, dy), d, col in zip(((-1, 0), (0, -1), (1, 0), (0, 1)), dist * st, (ROJO, AMARILLO, ROJO, AMARILLO)):
            cv2.arrowedLine(c, p0, (int(p0[0] + dx * d * a * S), int(p0[1] + dy * d * a * S)), col, 3, tipLength=0.12)
        texto(c, ["caja = ancla − (izq, arriba) y ancla + (der, abajo), por el stride.",
                  "Los últimos 4 números, tras la sigmoide, son la probabilidad de cada clase."], 30, 700, 21, CLARO)
        v.add(c, n=2)
    p1, p2 = a_disp(caja[0], caja[1]), a_disp(caja[2], caja[3])
    cv2.rectangle(c, p1, p2, (80, 255, 120) if ok else (160, 160, 160), 3)
    et = f"{names[cls]} {prob[cls]:.2f}"
    (tw, th), _ = cv2.getTextSize(et, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(c, (p1[0], p1[1] - th - 12), (p1[0] + tw + 8, p1[1]), (80, 255, 120) if ok else (160, 160, 160), -1)
    cv2.putText(c, et, (p1[0] + 4, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    c[700:H] = FONDO
    texto(c, [f"caja (px): x1={caja[0]:.1f} y1={caja[1]:.1f} x2={caja[2]:.1f} y2={caja[3]:.1f}  ·  salida de la red: "
              f"x1={float(det[0]):.1f} y1={float(det[1]):.1f} x2={float(det[2]):.1f} y2={float(det[3]):.1f}",
              f"{names[cls]} con probabilidad {prob[cls]:.3f} -> " + ("SE REPORTA (≥ umbral)" if ok else "se descarta (< umbral)")],
          30, 700, 21, (120, 255, 150) if ok else CLARO)
    v.add(c, 4)


def escena_topk(v, P, tot):
    out, names, conf = P["out"], P["names"], P["conf"]
    pm = P["pred"][:, 4:].max(1).values.sort(descending=True).values.numpy()
    sel = out[out[:, 4] >= conf]
    tit = f"{tot:,} celdas -> TopK 300 -> {len(sel)} con confianza ≥ {conf:.2f}".replace(",", ".")
    sub = "YOLOv10 elige uno a uno: sin NMS"
    fig = figura(560, 380)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(pm) + 1), pm, color="#5AB7BB", lw=2)
    ax.axvspan(1, 300, color="#3A46A0", alpha=0.35, label="TopK 300")
    ax.axhline(conf, color="#EB5050", ls="--", label=f"umbral {conf:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("celdas ordenadas por probabilidad (escala log)")
    ax.set_ylabel("probabilidad máxima de clase")
    ax.legend(facecolor="#040414", labelcolor="white", fontsize=8)
    fig.subplots_adjust(bottom=0.16, left=0.14)
    estilo(ax)
    graf = fig_a_img(fig)
    for fase in range(3):
        c = lienzo("Paso 6 · Selección final", tit, sub)
        img_base(P, c, 0.2)
        if fase == 0:
            for r in out.numpy()[::-1]:
                g = int(60 + 195 * min(1, r[4] / max(conf, 1e-6)))
                cv2.rectangle(c, a_disp(r[0], r[1]), a_disp(r[2], r[3]), (g, g // 2, 255 - g), 1)
            msg = ["Las 300 cajas con mayor probabilidad (salida (1, 300, 6) de la red). Color = probabilidad.",
                   "La cabeza uno a uno se entrenó para dar una sola caja por objeto, así que no hace falta NMS."]
        else:
            for r in sel.numpy():
                p1, p2 = a_disp(r[0], r[1]), a_disp(r[2], r[3])
                cv2.rectangle(c, p1, p2, (80, 255, 120), 3)
                et = f"{names[int(r[5])]} {r[4]:.2f}"
                (tw, th), _ = cv2.getTextSize(et, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(c, (p1[0], p1[1] - th - 12), (p1[0] + tw + 8, p1[1]), (80, 255, 120), -1)
                cv2.putText(c, et, (p1[0] + 4, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            msg = [f"Quedan las cajas con confianza ≥ {conf:.2f} (umbral de campo calibrado contra el estado de calzada del IDU).",
                   "Estas detecciones alimentan el score pseudo-PCI y la etiqueta del tramo."]
        pegar(c, graf, PX, 100)
        if fase == 2:
            r = P["pred_ultra"]
            lin = ["Comprobación con YOLO.predict() de Ultralytics:"]
            for b_, cf, cl in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                lin.append(f"{names[int(cl)]} · conf {cf:.3f}")
                lin.append(f"caja en la foto original: {', '.join(f'{z:.0f}' for z in b_)}")
            lin.append(f"decodificación propia vs red: dif. máx {P['dif']:.1e}")
            texto(c, lin, PX, 500, 20, (120, 255, 150))
        texto(c, msg, 30, 700, 21, CLARO)
        v.add(c, 4)


def escena_gradcam(v, P, det):
    from captum.attr import LayerAttribution, LayerGradCam

    s, cy, cx, _ = celda_de(P, det)
    net, head, names = P["net"], P["head"], P["names"]
    cls = int(det[5])
    for p in net.parameters():
        p.requires_grad_(True)
    capa = head.one2one_cv3[s][1]  # últimas convoluciones antes del clasificador de esa escala

    def f(inp):
        _, a = net(inp)
        return a["one2one"][s][:, 64 + cls, cy, cx]

    attr = LayerGradCam(f, capa).attribute(P["x"].clone().requires_grad_(True), relu_attributions=True)
    lh, lw = P["lb"].shape[:2]
    up = LayerAttribution.interpolate(attr, (lh, lw), interpolate_mode="bilinear")[0, 0].detach().numpy()
    col, u = mapa_color(up)
    mezcla = (P["lb"] * (1 - 0.6 * u[..., None]) + col * 0.6 * u[..., None]).astype(np.uint8)
    c = lienzo("Paso 7 · ¿En qué se fijó la red?", f"Grad-CAM (Captum) de la clase {names[cls]}", "captum.attr.LayerGradCam")
    pegar(c, cv2.resize(mezcla, None, fx=S, fy=S), X0, Y0)
    p1, p2 = a_disp(float(det[0]), float(det[1])), a_disp(float(det[2]), float(det[3]))
    cv2.rectangle(c, p1, p2, (80, 255, 120), 2)
    fig = figura(560, 330)
    ax = fig.add_subplot(111)
    im = ax.imshow(attr[0, 0].detach().numpy(), cmap="jet")
    fig.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors="#ACD9DC", labelsize=8)
    ax.set_title(f"Grad-CAM nativo: {tuple(attr.shape[2:])} (capa one2one_cv3[{s}][1])", fontsize=10)
    estilo(ax)
    pegar(c, fig_a_img(fig), PX, 100)
    texto(c, ["Se deriva el logit de la clase en la celda ganadora", "respecto a las últimas activaciones de la cabeza;", "las zonas rojas son las que más empujaron", "la decisión."], PX, 460, 21)
    texto(c, ["Grad-CAM: gradiente de la salida respecto a los mapas de activación, promediado por canal y usado como peso.",
              "Sirve para revisar que la detección se apoya en el pavimento dañado y no en el entorno."], 30, 700, 21, CLARO)
    v.add(c, 6)


def escena_cierre(v, P, dur):
    c = lienzo("DIMMIT · YOLOv10n", "Resumen de la pasada de la foto")
    lin = [f"1. Entrada: foto -> tensor {tuple(P['x'].shape)}",
           f"2. Convolución: kernel 3×3 deslizante, 16 filtros -> {tuple(P['feats'][0].shape)}",
           "3. 23 capas (backbone + neck): mapas de activación capturados con register_forward_hook",
           "4. 3 cuadrículas (strides 8/16/32) · 68 números por celda",
           "5. DFL + ancla -> caja; sigmoide -> probabilidad de las 4 clases",
           "6. TopK 300 sin NMS -> umbral calibrado con el IDU -> detecciones",
           "7. Grad-CAM (Captum) para revisar en qué se fijó la red",
           "", f"Reconstrucción idéntica a la red (dif. máx {P['dif']:.1e}) y a YOLO.predict() de Ultralytics.",
           f"Video generado con OpenCV (cv2.VideoWriter) en {dur:.0f} s de cómputo · {time.strftime('%d-%m-%Y %H:%M')}"]
    texto(c, lin, 60, 130, 27, BLANCO)
    v.add(c, 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--foto", default="Prueba_6/Prueba_6_007")
    ap.add_argument("--conf", type=float, default=0.16, help="umbral de confianza (0,16 = umbral calibrado con el IDU para piel de cocodrilo)")
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    print(f"[video] pasada real de {a.foto} por YOLOv10n (hooks en las 23 capas y la cabeza)…", flush=True)
    P = pasada(a.foto, a.conf)
    P["conf"] = a.conf
    print(f"[video] decodificación propia vs salida de la red: diferencia máxima {P['dif']:.2e}", flush=True)
    ensure_dir(OUT)
    nombre = f"yolo_por_dentro_{a.foto.split('/')[-1]}"
    tmp = OUT / f".{nombre}_mp4v.mp4"
    v = Video(tmp, a.fps)
    dets = [r for r in P["out"] if float(r[4]) >= a.conf] or [P["out"][0]]
    escenas = [("título", lambda: escena_titulo(v, P, a.foto)), ("entrada", lambda: escena_entrada(v, P)),
               ("convolución 3×3", lambda: escena_convolucion(v, P)), ("mapas de activación", lambda: escena_capas(v, P))]
    for nom, fn in escenas:
        fn()
        print(f"[video] escena {nom} lista · {v.n} fotogramas", flush=True)
    tot = escena_cuadriculas(v, P)
    print(f"[video] escena cuadrículas lista · {v.n} fotogramas", flush=True)
    for k, d in enumerate(dets[:2] + [P["out"][len(dets)]], 1):
        escena_vector(v, P, d, k)
    print(f"[video] escena vector -> caja lista · {v.n} fotogramas", flush=True)
    escena_topk(v, P, tot)
    escena_gradcam(v, P, dets[0])
    print(f"[video] escenas TopK y Grad-CAM listas · {v.n} fotogramas", flush=True)
    escena_cierre(v, P, time.time() - t0)
    v.close()
    final = OUT / f"{nombre}.mp4"
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg")
    if ff:
        subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)], check=True)
        tmp.unlink()
    else:
        tmp.rename(final)
    print(f"[ok] {final.relative_to(REPO_ROOT)} · {v.n / a.fps:.0f} s · {final.stat().st_size / 1e6:.1f} MB "
          f"({v.n} fotogramas a {a.fps} fps, {time.time() - t0:.0f} s de cómputo)")


if __name__ == "__main__":
    main()
