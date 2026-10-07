"""Presentación HTML por diapositivas de la toma de campo (asignatura), con mapa de Bogotá de fondo.

Autocontenida: SVG en línea, miniaturas base64 con las cajas reales del detector, JS propio sin
dependencias. Lee reports/campo/*.csv y data/campo/features/*.parquet. Escribe
reports/campo/dimmit_presentacion.html y build/artifact/dimmit_presentacion.html (mismo contenido).
"""
import base64
import html as H
import io
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from dimmit.campo import OUT_CAMPO
from dimmit.campo.scores import FEAT
from dimmit.utils.io import CLASS_KEYS, CLASS_SLUGS, REPO_ROOT, ensure_dir

SLUG_TXT = {"grieta_longitudinal": "Grieta longitudinal", "grieta_transversal": "Grieta transversal", "piel_cocodrilo": "Piel de cocodrilo", "baches": "Bache"}
ESTADO_TXT = {"bueno": "Bueno", "satisfactorio": "Satisfactorio", "regular": "Regular", "malo": "Malo", "muy_malo": "Muy malo"}
ETIQ_TXT = {"prevencion": "Prevención", "mantenimiento": "Mantenimiento", "intervencion": "Intervención"}
CLS_COLOR = {"D00": "var(--c1)", "D10": "var(--c2)", "D20": "var(--c3)", "D40": "var(--c4)"}
TAUS = {"D00": 0.07, "D10": 0.04, "D20": 0.09, "D40": 0.07}

# --- mapa de Bogotá (lat/lon -> unidades del SVG) ---------------------------------------------
K = 4000.0
LAT0, LON0 = 4.85, -74.25


def P(lat, lon):
    return (lon - LON0) * K, (LAT0 - lat) * K


LANDMARKS = {
    "candelaria": (4.5995, -74.0735, "Plaza de Bolívar · La Candelaria"),
    "centro": (4.6133, -74.0699, "Torre Colpatria"),
    "monserrate": (4.6057, -74.0557, "Monserrate"),
    "sur": (4.52, -74.13, "Usme · Ciudad Bolívar"),
    "norte": (4.74, -74.047, "Autopista Norte · TransMilenio"),
    "occidente": (4.658, -74.094, "Parque Simón Bolívar"),
    "rosario": (4.6015, -74.0729, "Universidad del Rosario"),
    "ciudad": (4.65, -74.11, "Bogotá"),
}
# contorno urbano aproximado (lat, lon), sentido horario desde el norte
OUTLINE = [(4.835, -74.04), (4.80, -74.02), (4.72, -74.03), (4.66, -74.05), (4.60, -74.055), (4.55, -74.08), (4.50, -74.10),
           (4.47, -74.13), (4.49, -74.17), (4.56, -74.19), (4.62, -74.20), (4.68, -74.17), (4.74, -74.12), (4.80, -74.08), (4.835, -74.04)]
CERROS = [(4.835, -74.015), (4.76, -74.02), (4.70, -74.035), (4.64, -74.05), (4.60, -74.055), (4.56, -74.07), (4.52, -74.09), (4.47, -74.11)]
RIO = [(4.83, -74.09), (4.78, -74.12), (4.72, -74.15), (4.66, -74.185), (4.60, -74.21), (4.55, -74.21), (4.50, -74.19)]


def _poly(points, close=False):
    pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in (P(a, b) for a, b in points))
    return pts


def icon(kind, x, y):
    """Íconos lineales tipo cartoon (un trazo), 64 px, centrados en (x, y)."""
    g = f'<g class="lm" data-lm="{kind}" transform="translate({x:.0f} {y:.0f})">'
    if kind == "monserrate":  # cerro con la iglesia arriba
        g += '<path d="M-60 40 L-20 -20 L0 -5 L25 -30 L60 40 Z"/><path d="M14 -30 v-12 h8 v12 M18 -46 v-8 M14 -50 h8"/>'
    elif kind == "centro":  # torre Colpatria
        g += '<path d="M-14 50 V-40 h28 V50 M-6 -40 V-60 h12 V-40 M0 -60 V-72"/><path d="M-14 -20 h28 M-14 0 h28 M-14 20 h28"/>'
    elif kind == "candelaria":  # catedral primada: dos torres y frontón
        g += '<path d="M-50 40 V-10 h18 V40 M32 40 V-10 h18 V40 M-32 40 V0 L0 -28 L32 0 V40"/><path d="M-41 -10 V-34 M41 -10 V-34 M-47 -34 h12 M35 -34 h12 M0 -28 V-44 M-6 -38 h12"/>'
    elif kind == "sur":  # casas en ladera
        g += '<path d="M-60 44 L-30 20 L0 32 L30 10 L60 30"/><path d="M-44 36 v-14 l8 -8 l8 8 v8 M-6 30 v-12 l8 -8 l8 8 v10 M28 20 v-14 l8 -8 l8 8 v12"/>'
    elif kind == "norte":  # bus articulado de TransMilenio
        g += '<path d="M-64 10 h50 a4 4 0 0 1 4 4 v14 h-54 a4 4 0 0 1 -4 -4 v-10 a4 4 0 0 1 4 -4 Z M10 10 h50 a4 4 0 0 1 4 4 v14 h-54 Z M-6 16 h12"/><path d="M-50 36 a5 5 0 1 0 10 0 a5 5 0 1 0 -10 0 M-30 36 a5 5 0 1 0 10 0 a5 5 0 1 0 -10 0 M20 36 a5 5 0 1 0 10 0 a5 5 0 1 0 -10 0 M44 36 a5 5 0 1 0 10 0 a5 5 0 1 0 -10 0 M-56 16 h10 M-40 16 h10 M20 16 h10 M36 16 h10"/>'
    elif kind == "occidente":  # árbol y lago del Simón Bolívar
        g += '<path d="M-20 40 V10 M-20 10 a22 22 0 1 1 1 0 M10 44 q20 -12 50 -4 q-20 10 -50 4 Z"/>'
    elif kind == "rosario":  # claustro: arco y campanario
        g += '<path d="M-50 40 V-10 h100 V40 M-30 40 V10 a30 30 0 0 1 60 0 V40 M30 -10 V-40 h20 V-10 M40 -40 V-52"/>'
    else:  # ciudad: skyline
        g += '<path d="M-70 40 V0 h16 V-20 h12 V40 M-30 40 V-40 h20 V40 M4 40 V-10 h14 V-30 h10 V40 M40 40 V-20 h24 V40"/>'
    return g + "</g>"


def bogota_svg():
    out = ['<svg id="mapa" viewBox="0 0 1000 1600" preserveAspectRatio="xMidYMid slice" aria-hidden="true">', '<g id="mapaG">']
    out.append(f'<polygon class="urb" points="{_poly(OUTLINE)}"/>')
    out.append(f'<clipPath id="clipUrb"><polygon points="{_poly(OUTLINE)}"/></clipPath>')
    grid = []
    for k in range(0, 40):  # calles (E-O, ligeramente inclinadas como la grilla real)
        lat = 4.46 + 0.01 * k
        grid.append(f'<line x1="{P(lat, -74.25)[0]:.0f}" y1="{P(lat, -74.25)[1]:.0f}" x2="{P(lat - 0.02, -74.0)[0]:.0f}" y2="{P(lat - 0.02, -74.0)[1]:.0f}"/>')
    for k in range(0, 26):  # carreras (N-S)
        lon = -74.25 + 0.01 * k
        grid.append(f'<line x1="{P(4.85, lon)[0]:.0f}" y1="{P(4.85, lon)[1]:.0f}" x2="{P(4.45, lon + 0.015)[0]:.0f}" y2="{P(4.45, lon + 0.015)[1]:.0f}"/>')
    out.append('<g class="grid" clip-path="url(#clipUrb)">' + "".join(grid) + "</g>")
    # vías principales: Autopista Norte / Caracas (N-S), Calle 26 y Calle 80 (E-O)
    out.append(f'<polyline class="via" points="{_poly([(4.83, -74.045), (4.70, -74.055), (4.62, -74.07), (4.57, -74.09)])}"/>')
    out.append(f'<polyline class="via" points="{_poly([(4.615, -74.07), (4.66, -74.10), (4.70, -74.15)])}"/>')
    out.append(f'<polyline class="via" points="{_poly([(4.66, -74.06), (4.69, -74.10), (4.72, -74.14)])}"/>')
    out.append(f'<polyline class="rio" points="{_poly(RIO)}"/>')
    out.append(f'<polyline class="cerros" points="{_poly(CERROS)}"/>')
    for k in range(6):  # sombreado de cerros
        out.append(f'<polyline class="cerros2" points="{_poly([(a, b + 0.006 * (k + 1)) for a, b in CERROS])}"/>')
    for kind, (lat, lon, _) in LANDMARKS.items():
        x, y = P(lat, lon)
        out.append(icon(kind, x, y))
    out.append("</g></svg>")
    return "".join(out)


# --- utilidades -------------------------------------------------------------------------------
def thumb(path, boxes, width=560, quality=72):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w0, h0 = im.size
        im = im.resize((width, int(h0 * width / w0)))
        d = ImageDraw.Draw(im)
        col = {"D00": (42, 120, 214), "D10": (235, 104, 52), "D20": (27, 175, 122), "D40": (237, 161, 0)}
        for _, b in boxes.iterrows():
            x0, y0 = (b.cx - b.w / 2) * im.width, (b.cy - b.h / 2) * im.height
            x1, y1 = (b.cx + b.w / 2) * im.width, (b.cy + b.h / 2) * im.height
            d.rectangle([x0, y0, x1, y1], outline=col[b.cls], width=3)
            d.rectangle([x0, y0 - 16, x0 + 110, y0], fill=col[b.cls])
            d.text((x0 + 4, y0 - 15), f"{SLUG_TXT[CLASS_SLUGS[b.cls]][:14]} {b.conf:.2f}", fill=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def tile(v, l, sub=""):
    return f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div>{f"<div class=s>{sub}</div>" if sub else ""}</div>'


# --- gráficas SVG ---------------------------------------------------------------------------
def bars_scores(t):
    """Barras horizontales por tramo: score con umbral RDD2020 (tau) vs score con conf >= 0.25."""
    t = t.sort_values("score_calidad")
    n = len(t)
    rh, gap, top, left, w = 20, 10, 24, 150, 500
    hgt = top + n * (2 * rh + gap) + 40
    sx = lambda v: left + v / 100 * w
    o = [f'<svg viewBox="0 0 {left + w + 230} {hgt}" class="chart" role="img" aria-label="Score por tramo, antes y después">']
    for v in (0, 25, 50, 75, 100):
        o.append(f'<line x1="{sx(v):.0f}" y1="{top - 6}" x2="{sx(v):.0f}" y2="{hgt - 36}" class="grid"/><text x="{sx(v):.0f}" y="{hgt - 18}" text-anchor="middle">{v}</text>')
    for lim, name in ((41, "malo"), (56, "regular"), (71, "satisf."), (86, "bueno")):
        o.append(f'<line x1="{sx(lim):.0f}" y1="{top - 6}" x2="{sx(lim):.0f}" y2="{hgt - 36}" class="lim"/><text x="{sx(lim) + 3:.0f}" y="{top - 10}" class="lim-t">{name} ≥{lim}</text>')
    y = top
    for _, r in t.iterrows():
        o.append(f'<text x="{left - 10}" y="{y + rh + 4}" text-anchor="end" class="t-strong">{r.prueba.replace("_", " ")} · T{int(r.tramo)}</text>')
        o.append(f'<text x="{left - 10}" y="{y + rh + 18}" text-anchor="end" class="mono small">{r.id_via}</text>')
        o.append(f'<rect x="{left}" y="{y}" width="{max(r.score_calidad_tau_rdd2020 / 100 * w, 2):.0f}" height="{rh - 3}" rx="3" class="b-tau"/>')
        o.append(f'<text x="{sx(r.score_calidad_tau_rdd2020) + 6:.0f}" y="{y + rh - 7}">{fmt(r.score_calidad_tau_rdd2020)} · {ESTADO_TXT[r.estado_tau_rdd2020]} · {ETIQ_TXT[r.etiqueta_tau_rdd2020]}</text>')
        o.append(f'<rect x="{left}" y="{y + rh}" width="{max(r.score_calidad / 100 * w, 2):.0f}" height="{rh - 3}" rx="3" class="b-25"/>')
        o.append(f'<text x="{sx(r.score_calidad) + 6:.0f}" y="{y + 2 * rh - 7}" class="t-strong">{fmt(r.score_calidad)} · {ESTADO_TXT[r.estado]} · {ETIQ_TXT[r.etiqueta]}</text>')
        y += 2 * rh + gap
    o.append("</svg>")
    return "".join(o)


def bars_damage(t):
    t = t.sort_values(["prueba", "tramo"])
    n = len(t)
    rh, gap, top, left, w = 20, 8, 16, 150, 560
    hgt = top + n * (rh + gap) + 36
    mx = max(t[[f"n_{s}" for s in CLASS_SLUGS.values()]].sum(axis=1).max(), 1)
    o = [f'<svg viewBox="0 0 {left + w + 60} {hgt}" class="chart" role="img" aria-label="Cajas detectadas por tramo y clase">']
    y = top
    for _, r in t.iterrows():
        o.append(f'<text x="{left - 10}" y="{y + rh - 6}" text-anchor="end" class="t-strong">{r.prueba.replace("_", " ")} · T{int(r.tramo)}</text>')
        x = left
        for c in CLASS_KEYS:
            v = r[f"n_{CLASS_SLUGS[c]}"]
            if v > 0:
                bw = v / mx * w
                o.append(f'<rect x="{x:.0f}" y="{y}" width="{max(bw - 2, 1):.0f}" height="{rh}" rx="2" fill="{CLS_COLOR[c]}"/>')
                if bw > 26:
                    o.append(f'<text x="{x + bw / 2:.0f}" y="{y + rh - 6}" text-anchor="middle" class="on-bar">{int(v)}</text>')
                x += bw
        o.append(f'<text x="{x + 6:.0f}" y="{y + rh - 6}">{int(r[[f"n_{s}" for s in CLASS_SLUGS.values()]].sum())}</text>')
        y += rh + gap
    o.append("</svg>")
    return "".join(o)


def scatter_depth(t, rs, lo, hi, ctrl):
    left, top, w, h = 70, 20, 520, 300
    xs, ys = t["profundidad_max_cm"].to_numpy(float), t["n_baches"].to_numpy(float)
    xmax = max(math.ceil(np.nanmax(xs) * 1.1 / 5) * 5, 10)
    ystep = 10 if ys.max() > 20 else 5
    ymax = max(math.ceil(ys.max() * 1.15 / ystep) * ystep, ystep)
    sx = lambda v: left + v / xmax * w
    sy = lambda v: top + h - v / ymax * h
    o = [f'<svg viewBox="0 0 {left + w + 30} {top + h + 70}" class="chart" role="img" aria-label="Profundidad medida frente a baches detectados">']
    for v in np.arange(0, xmax + 1, 5 if xmax <= 30 else 10):
        if v > xmax:
            break
        o.append(f'<line x1="{sx(v):.0f}" y1="{top}" x2="{sx(v):.0f}" y2="{top + h}" class="grid"/><text x="{sx(v):.0f}" y="{top + h + 18}" text-anchor="middle">{fmt(v)}</text>')
    for v in np.arange(0, ymax + 1, ystep):
        o.append(f'<line x1="{left}" y1="{sy(v):.0f}" x2="{left + w}" y2="{sy(v):.0f}" class="grid"/><text x="{left - 8}" y="{sy(v) + 4:.0f}" text-anchor="end">{fmt(v)}</text>')
    o.append(f'<text x="{left + w / 2:.0f}" y="{top + h + 40}" text-anchor="middle">profundidad máxima del sensor, línea base autocalibrada (cm)</text>')
    o.append(f'<text transform="translate(14 {top + h / 2:.0f}) rotate(-90)" text-anchor="middle">cajas de bache conservadas (modo campo)</text>')
    shapes = {"Prueba_2": "circle", "Prueba_3": "rect", "Prueba_4": "tri"}
    for _, r in t.iterrows():
        x, y = sx(r.profundidad_max_cm), sy(r.n_baches)
        s = shapes.get(r.prueba, "circle")
        if s == "circle":
            o.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" class="pt"/>')
        elif s == "rect":
            o.append(f'<rect x="{x - 6:.0f}" y="{y - 6:.0f}" width="12" height="12" class="pt"/>')
        else:
            o.append(f'<polygon points="{x:.0f},{y - 8:.0f} {x - 8:.0f},{y + 6:.0f} {x + 8:.0f},{y + 6:.0f}" class="pt"/>')
        o.append(f'<text x="{x + 10:.0f}" y="{y - 8:.0f}" class="small">T{int(r.tramo)}</text>')
    o.append(f'<text x="{left}" y="{top + h + 60}" class="t-strong">Spearman {fmt(rs, 2)} · IC 95 % [{fmt(lo, 2)}, {fmt(hi, 2)}] · n = {len(t)} tramos</text>')
    o.append("</svg>")
    return "".join(o)


def street_map(fr, t, depth_frames):
    """Mapa SVG de la calle: trazas por prueba, tramos coloreados por score, marcadores de profundidad."""
    lat0, lon0 = fr["lat"].mean(), fr["lon"].mean()
    mx = lambda lon: (lon - lon0) * 111320 * math.cos(math.radians(lat0))
    my = lambda lat: (lat - lat0) * 110540
    xs, ys = fr["lon"].map(mx), fr["lat"].map(my)
    pad, W, Hh = 14, 640, 420
    sx = lambda x: pad + (x - xs.min()) / (xs.max() - xs.min() + 1e-9) * (W - 2 * pad)
    sy = lambda y: Hh - pad - (y - ys.min()) / (ys.max() - ys.min() + 1e-9) * (Hh - 2 * pad)
    o = [f'<svg viewBox="0 0 {W} {Hh}" class="chart street" role="img" aria-label="Recorridos y tramos sobre la calle">']
    # escala de 10 m
    m10 = 10 / (xs.max() - xs.min() + 1e-9) * (W - 2 * pad)
    o.append(f'<line x1="{W - pad - m10:.0f}" y1="{Hh - 6}" x2="{W - pad:.0f}" y2="{Hh - 6}" class="scale"/><text x="{W - pad - m10 / 2:.0f}" y="{Hh - 10}" text-anchor="middle" class="small">10 m</text>')
    dash = {"Prueba_2": "", "Prueba_3": "6 4", "Prueba_4": "2 4"}
    for prueba, g in fr.sort_values("timestamp").groupby("prueba"):
        pts = " ".join(f"{sx(mx(r.lon)):.1f},{sy(my(r.lat)):.1f}" for _, r in g.iterrows())
        o.append(f'<polyline points="{pts}" class="traza" stroke-dasharray="{dash.get(prueba, "")}"/>')
    for _, r in t.iterrows():
        g = fr[fr["id_via"] == r.id_via].sort_values("timestamp")
        pts = " ".join(f"{sx(mx(q.lon)):.1f},{sy(my(q.lat)):.1f}" for _, q in g.iterrows())
        o.append(f'<polyline points="{pts}" class="tramo" style="stroke:{seq_color(r.score_calidad)}"><title>{r.id_via}: score {fmt(r.score_calidad)}</title></polyline>')
        cx, cy = sx(mx(r.lon)), sy(my(r.lat))
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" class="nodo"/><text x="{cx + 8:.0f}" y="{cy - 6:.0f}" class="small mono">T{int(r.tramo)}</text>')
    for _, q in depth_frames.iterrows():
        o.append(f'<circle cx="{sx(mx(q.lon)):.1f}" cy="{sy(my(q.lat)):.1f}" r="{min(3 + q.profundidad_cm / 4, 12):.1f}" class="prof"><title>{fmt(q.profundidad_cm, 1)} cm</title></circle>')
    o.append("</svg>")
    return "".join(o)


def seq_color(score):
    """Escala ordinal azul validada (misma del reporte v1): peor = más oscuro."""
    if score >= 86:
        return "var(--st-bueno)"
    if score >= 71:
        return "var(--st-satisfactorio)"
    if score >= 56:
        return "var(--st-regular)"
    if score >= 41:
        return "var(--st-malo)"
    return "var(--st-muy_malo)"


# --- construcción ----------------------------------------------------------------------------
def build():
    vias = pd.read_csv(OUT_CAMPO / "vias_clasificacion.csv")
    largo = pd.read_csv(OUT_CAMPO / "calidad_confianza_correlacion.csv")
    fr = pd.read_parquet(FEAT / "fotogramas_features.parquet")
    det_dir = REPO_ROOT / json.loads((FEAT / "fuente_detecciones.json").read_text())["detector_dir"]
    det = pd.read_parquet(det_dir / "detections.parquet")
    det["cls"] = det["class_id"].map(dict(enumerate(CLASS_KEYS)))
    det["tau"] = det["cls"].map(TAUS)
    lat = json.loads((det_dir / "latency.json").read_text())
    t = vias[vias["nivel"] == "tramo"].copy()
    pr = vias[vias["nivel"] == "prueba"].copy()
    L = lambda m, via="LOTE": float(largo[(largo["metrica"] == m) & (largo["id_via"] == via)]["valor"].iloc[0]) if ((largo["metrica"] == m) & (largo["id_via"] == via)).any() else float("nan")
    row = largo[(largo["metrica"] == "spearman_profundidad_max_cm__n_baches") & (largo["nivel"] == "tramo")].iloc[0]
    row40 = largo[(largo["metrica"] == "spearman_profundidad_max_cm_base40__n_baches") & (largo["nivel"] == "tramo")].iloc[0]
    tau_campo = json.loads((FEAT / "umbrales_campo.json").read_text())["tau_campo"]
    # resumen de filtros (cajas descartadas por capa) y validación visual
    res = pd.read_csv(FEAT / "filtros_resumen.csv").set_index("cls")
    cols_f = [c for c in ("1_confianza", "2_banda", "2_fuera_de_via", "3_sin_persistencia", "4_sin_consenso_escala", "conservada") if c in res]
    names_f = {"1_confianza": "1 · confianza < umbral de campo", "2_banda": "2 · banda ancha y baja (junta/bordillo)", "2_fuera_de_via": "2 · fuera de la calzada",
               "3_sin_persistencia": "3 · sin persistencia entre fotos", "4_sin_consenso_escala": "4 · sin consenso a otra escala", "conservada": "conservadas"}
    filtros_html = "".join(f'<tr><td>{names_f[c]}</td>' + "".join(f'<td class="n">{int(res.loc[k, c]) if k in res.index else 0}</td>' for k in CLASS_KEYS) + f'<td class="n">{int(res[c].sum())}</td></tr>' for c in cols_f)
    V = lambda m: L(m, "MUESTRA")
    valid_html = "".join(f'<tr><td>{n}</td><td class="n">{fmt(100 * V(f"precision_{k}_tau_rdd2020"))} %</td><td class="n">{fmt(100 * V(f"recall_{k}_tau_rdd2020"))} %</td>'
                         f'<td class="n us">{fmt(100 * V(f"precision_{k}_campo"))} %</td><td class="n us">{fmt(100 * V(f"recall_{k}_campo"))} %</td></tr>'
                         for k, n in (("piel_cocodrilo", "Piel de cocodrilo"), ("bache", "Bache"), ("cualquier_dano", "Cualquier daño")))
    parches_pct = fmt(100 * V("parches_marcados_como_piel_cocodrilo_campo"))
    ctrl = L("spearman_profundidad_max_cm__n_grieta_longitudinal")

    det_f = pd.read_parquet(FEAT / "detecciones_filtradas.parquet")
    keep = det_f[det_f["conservada"]]
    # miniaturas: la foto con la caja conservada de mayor confianza por tramo (o la de mayor confianza bruta si no hay)
    thumbs = {}
    for _, r in t.iterrows():
        g = fr[fr["id_via"] == r.id_via]
        best = g.loc[g["max_conf_campo"].idxmax()] if g["max_conf_campo"].max() > 0 else g.loc[g["max_conf"].idxmax()]
        b = keep[keep["image_id"] == best.image_id]
        n_antes = int(((det["image_id"] == best.image_id) & (det["conf"] >= det["tau"])).sum())
        thumbs[r.id_via] = (thumb(best.path, b), best.image_id, int(len(b)), n_antes)
    # foto para la animación de YOLO: la de mayor confianza conservada
    best_all = fr.loc[fr["max_conf_campo"].idxmax()]
    yolo_boxes = det[det["image_id"] == best_all.image_id].sort_values("conf", ascending=False)
    yolo_keep = keep[keep["image_id"] == best_all.image_id]
    yolo_img = thumb(best_all.path, yolo_boxes.iloc[0:0], width=800, quality=78)
    yolo_json = json.dumps([{"c": r.cls, "p": round(float(r.conf), 3), "x": round(float(r.cx), 4), "y": round(float(r.cy), 4), "w": round(float(r.w), 4), "h": round(float(r.h), 4),
                             "k": bool(((yolo_keep["cx"] - r.cx).abs() < 1e-6).any())} for _, r in yolo_boxes.head(60).iterrows()])
    # tramo de ejemplo para el flujo de score (el de la foto de YOLO)
    ex = t[t["id_via"] == best_all.id_via].iloc[0]

    n_fotos, n_tramos = len(fr), len(t)
    etq = t["etiqueta"].value_counts().to_dict()
    etq_antes = t["etiqueta_tau_rdd2020"].value_counts().to_dict()
    kpis = (tile(n_tramos, "tramos evaluados", "3 recorridos · 5-oct-2026") + tile(n_fotos, "fotos procesadas", f"{fmt(lat['ms_por_imagen_lote'])} ms por foto en Kaggle (CPU)")
            + tile(fmt(t["score_calidad"].mean()), "score medio (modo campo)", f"antes, con el umbral de RDD2020: {fmt(t['score_calidad_tau_rdd2020'].mean())}")
            + tile(f"{etq.get('intervencion', 0)} / {etq.get('mantenimiento', 0)} / {etq.get('prevencion', 0)}", "intervención / mantenimiento / prevención", f"antes: {etq_antes.get('intervencion', 0)} / {etq_antes.get('mantenimiento', 0)} / {etq_antes.get('prevencion', 0)}")
            + tile(fmt(fr.loc[fr["max_conf_campo"] > 0, "max_conf_campo"].mean(), 2), "confianza media de las cajas conservadas", f"{int((fr['n_detecciones_campo'] > 0).sum())} de {n_fotos} fotos con alguna caja que pasa los filtros"))

    gallery = "".join(
        f'<figure><img src="{thumbs[r.id_via][0]}" alt="Foto de mayor confianza del tramo {r.id_via}" loading="lazy">'
        f'<figcaption><b class="mono">{r.id_via}</b> · {r.prueba.replace("_", " ")} T{int(r.tramo)}<br>score {fmt(r.score_calidad)} ({ESTADO_TXT[r.estado]}) · antes: {fmt(r.score_calidad_tau_rdd2020)} ({ESTADO_TXT[r.estado_tau_rdd2020]})<br>'
        f'<span class="pill et-{r.etiqueta}">{ETIQ_TXT[r.etiqueta]}</span> · {thumbs[r.id_via][2]} cajas conservadas de {thumbs[r.id_via][3]} · {fmt(r.lat, 5)}, {fmt(r.lon, 5)}</figcaption></figure>'
        for _, r in t.sort_values(["prueba", "tramo"]).iterrows())

    table_rows = "".join(
        f'<tr><td class="mono">{r.id_via}</td><td>{r.prueba.replace("_", " ")}</td><td class="n">{int(r.tramo)}</td><td class="n">{int(r.n_fotogramas)}</td>'
        f'<td class="n">{fmt(r.lat, 5)}</td><td class="n">{fmt(r.lon, 5)}</td><td>{", ".join(SLUG_TXT[s] for s in str(r.danos_detectados).split(";") if s and s != "nan")}</td>'
        f'<td class="n">{fmt(r.score_calidad_tau_rdd2020)}</td><td>{ESTADO_TXT[r.estado_tau_rdd2020]}</td><td><span class="pill et-{r.etiqueta_tau_rdd2020}">{ETIQ_TXT[r.etiqueta_tau_rdd2020]}</span></td>'
        f'<td class="n">{fmt(r.score_calidad)}</td><td>{ESTADO_TXT[r.estado]}</td><td><span class="pill et-{r.etiqueta}">{ETIQ_TXT[r.etiqueta]}</span></td>'
        f'<td class="n">{fmt(r.profundidad_max_cm_base40, 1)}</td><td class="n">{fmt(r.profundidad_max_cm, 1)}</td><td class="n">{fmt(r.confianza_modelo, 2)}</td></tr>'
        for _, r in t.sort_values(["prueba", "tramo"]).iterrows())

    cal_rows = "".join(
        f'<tr><td class="mono">{r.id_via}</td><td class="n">{fmt(r.brillo_media)}</td><td class="n">{fmt(r.nitidez_media)}</td><td class="n">{fmt(100 * r.frac_fotos_borrosa)} %</td>'
        f'<td class="n">{fmt(100 * r.frac_fotos_sobreexpuesta)} %</td><td class="n">{fmt(r.max_conf_media, 2)}</td><td class="n">{fmt(100 * r.frac_fotos_con_deteccion)} %</td><td class="n">{fmt(100 * r.frac_fotos_con_deteccion_conf25)} %</td></tr>'
        for _, r in t.sort_values(["prueba", "tramo"]).iterrows())
    corr_rows = ""
    for _, p in pr.iterrows():
        corr_rows += f'<tr><td>{p.prueba.replace("_", " ")}</td><td class="n">{fmt(p.linea_base_cm)}</td><td class="n">{fmt(p.profundidad_max_cm_base40, 1)}</td><td class="n">{fmt(p.profundidad_max_cm, 1)}</td><td>{p.severidad_bache_astm if isinstance(p.severidad_bache_astm, str) else "—"}</td><td class="n">{int(p.n_baches)}</td></tr>'

    depth_frames = fr[fr["profundidad_cm"] >= 3]
    psi_vals = {v: L(f"psi_{v}") for v in ("brillo", "contraste", "nitidez", "jpeg_kb", "max_conf", "pci_det")}
    psi_html = "".join(f'<div class="psi"><span>{k}</span><b>{fmt(v, 2)}</b><i class="{"ok" if v <= 0.1 else ("al" if v <= 0.25 else "fa")}"></i></div>' for k, v in psi_vals.items())

    ctx = {
        "BOGOTA": bogota_svg(), "KPIS": kpis, "GALERIA": gallery, "TABLA": table_rows, "CALIDAD": cal_rows, "CORR": corr_rows,
        "BARRAS": bars_scores(t), "DANOS": bars_damage(t), "DISPERSION": scatter_depth(t, row["valor"], row["ic95_inf"], row["ic95_sup"], ctrl),
        "CALLE": street_map(fr, t, depth_frames), "PSI": psi_html,
        "YOLO_IMG": yolo_img, "YOLO_JSON": yolo_json, "YOLO_ID": best_all.image_id, "YOLO_N": str(int(len(yolo_keep))), "YOLO_N_TAU": str(int((yolo_boxes["conf"] >= yolo_boxes["tau"]).sum())), "PARCHES": parches_pct,
        "EX_ID": ex.id_via, "EX_N": str(int(ex.n_fotogramas)), "EX_SCORE": fmt(ex.score_calidad), "EX_ESTADO": ESTADO_TXT[ex.estado], "EX_ETIQ": ETIQ_TXT[ex.etiqueta],
        "EX_PROF": fmt(ex.profundidad_max_cm, 1), "EX_LAT": fmt(ex.lat, 5), "EX_LON": fmt(ex.lon, 5),
        "EX_RHO": " · ".join(f"{SLUG_TXT[CLASS_SLUGS[c]].split()[0]} {fmt(100 - ex[f'sub_score_{CLASS_SLUGS[c]}'])}" for c in CLASS_KEYS),
        "EX_N_CAJAS": " · ".join(f"{int(ex[f'n_{CLASS_SLUGS[c]}'])} {SLUG_TXT[CLASS_SLUGS[c]].lower()}" for c in CLASS_KEYS),
        "N_FOTOS": str(n_fotos), "N_TRAMOS": str(n_tramos), "MS": fmt(lat["ms_por_imagen_lote"]), "AUROC": fmt(L("dominio_auroc"), 2),
        "CONF_MED": fmt(fr["max_conf"].mean(), 2), "PCT_BORR": fmt(100 * fr["borrosa"].mean()),
        "PCT_SOBRE": fmt(100 * fr["sobreexpuesta"].mean()), "PCT_DET": fmt(100 * (fr["n_detecciones_tau"] > 0).mean()),
        "RS": fmt(row["valor"], 2), "RS_LO": fmt(row["ic95_inf"], 2), "RS_HI": fmt(row["ic95_sup"], 2),
        "RS40": fmt(row40["valor"], 2), "RS40_LO": fmt(row40["ic95_inf"], 2), "RS40_HI": fmt(row40["ic95_sup"], 2),
        "SCORE_TAU": fmt(t["score_calidad_tau_rdd2020"].mean()), "SCORE_CAMPO": fmt(t["score_calidad"].mean()),
        "N_INT": str(etq.get("intervencion", 0)), "N_INT_ANTES": str(etq_antes.get("intervencion", 0)), "LONG": fmt(t["longitud_m"].sum()),
        "N_CONSERVADAS": str(int(len(keep))), "N_BRUTAS": str(int(len(det))), "N_TAU": str(int((det["conf"] >= det["tau"]).sum())),
        "FILTROS": filtros_html, "VALID": valid_html, "PCT_CAMPO": fmt(100 * (fr["n_detecciones_campo"] > 0).mean()), "CONF_CAMPO": fmt(fr.loc[fr["max_conf_campo"] > 0, "max_conf_campo"].mean(), 2),
        "TAU_CAMPO": " · ".join(f"{ {'D00': 'long.', 'D10': 'transv.', 'D20': 'piel', 'D40': 'bache'}[c]} {fmt(v, 2)}" for c, v in tau_campo.items()),
    }
    page = TEMPLATE
    for k, v in ctx.items():
        page = page.replace("{{" + k + "}}", str(v))
    assert "{{" not in page, "placeholder sin reemplazar"
    out = OUT_CAMPO / "dimmit_presentacion_v2.html"
    out.write_text(page)
    art = ensure_dir(REPO_ROOT / "build/artifact") / "dimmit_presentacion_v2.html"
    art.write_text(page)
    print(f"[ok] presentación {len(page) / 1e6:.1f} MB -> {out}")
    return out


TEMPLATE = r"""<title>DIMMIT en La Candelaria v2</title>
<meta name="description" content="Presentación de la primera toma de datos de campo del prototipo DIMMIT V1 en Bogotá.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;700&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
/* Plano urbano: un mapa estilizado de Bogotá fijo al fondo, las diapositivas encima como láminas. */
:root {
  --bg: #eef0ea; --paper: rgba(252,252,250,.88); --ink: #15191c; --ink-2: #434a4e; --muted: #767d81; --ring: rgba(21,25,28,.12);
  --accent: #b5563a; --accent-ink: #8f3f27; --plano: #2a4a6e; --plano-2: rgba(42,74,110,.22); --plano-3: rgba(42,74,110,.10); --agua: #6c9ec9; --cerro: rgba(42,74,110,.35);
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100;
  --st-bueno: #86b6ef; --st-satisfactorio: #5598e7; --st-regular: #2a78d6; --st-malo: #1c5cab; --st-muy_malo: #0d366b;
  --ok: #0ca30c; --warn: #fab219; --crit: #d03b3b;
  --et-prev: #2a78d6; --et-mant: #c98500; --et-int: #d03b3b;
  --f-display: "Barlow Condensed", "Arial Narrow", system-ui, sans-serif; --f-body: "IBM Plex Sans", system-ui, sans-serif; --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e141b; --paper: rgba(20,27,36,.90); --ink: #eef2f5; --ink-2: #c2cad1; --muted: #8a949c; --ring: rgba(238,242,245,.14);
  --accent: #e08a6c; --accent-ink: #f0a98f; --plano: #9fc0e0; --plano-2: rgba(159,192,224,.22); --plano-3: rgba(159,192,224,.10); --agua: #5a8fc0; --cerro: rgba(159,192,224,.35);
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
  --st-bueno: #184f95; --st-satisfactorio: #256abf; --st-regular: #3987e5; --st-malo: #86b6ef; --st-muy_malo: #cde2fb;
  --et-prev: #3987e5; --et-mant: #c98500; --et-int: #e05a5a; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e141b; --paper: rgba(20,27,36,.90); --ink: #eef2f5; --ink-2: #c2cad1; --muted: #8a949c; --ring: rgba(238,242,245,.14);
  --accent: #e08a6c; --accent-ink: #f0a98f; --plano: #9fc0e0; --plano-2: rgba(159,192,224,.22); --plano-3: rgba(159,192,224,.10); --agua: #5a8fc0; --cerro: rgba(159,192,224,.35);
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
  --st-bueno: #184f95; --st-satisfactorio: #256abf; --st-regular: #3987e5; --st-malo: #86b6ef; --st-muy_malo: #cde2fb;
  --et-prev: #3987e5; --et-mant: #c98500; --et-int: #e05a5a; color-scheme: dark; }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.5 var(--f-body); overflow: hidden; }
#mapa { position: fixed; inset: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }
#mapaG { transform-origin: 0 0; transition: transform 1100ms cubic-bezier(.4,.0,.2,1); }
#mapa .urb { fill: var(--plano-3); stroke: var(--plano-2); stroke-width: 3; }
#mapa line, #mapa polyline, #mapa polygon, #mapa .lm path { vector-effect: non-scaling-stroke; }
#mapa .lm { transition: transform 1100ms cubic-bezier(.4,.0,.2,1); }
#mapa .grid line { stroke: var(--plano-2); stroke-width: 1.2; }
#mapa .via { fill: none; stroke: var(--plano); stroke-width: 4; opacity: .45; stroke-linecap: round; }
#mapa .rio { fill: none; stroke: var(--agua); stroke-width: 7; opacity: .55; stroke-linecap: round; stroke-linejoin: round; }
#mapa .cerros { fill: none; stroke: var(--cerro); stroke-width: 10; stroke-linecap: round; stroke-linejoin: round; }
#mapa .cerros2 { fill: none; stroke: var(--cerro); stroke-width: 2; opacity: .5; }
#mapa .lm path { fill: none; stroke: var(--plano); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; opacity: 0; transition: opacity 700ms 500ms; }
#mapa .lm.on path { opacity: .75; }
@media (prefers-reduced-motion: reduce) { #mapaG, #mapa .lm path { transition: none; } }
#deck { position: relative; z-index: 1; height: 100%; overflow-y: auto; scroll-snap-type: y proximity; scroll-behavior: smooth; }
.slide { min-height: 100%; scroll-snap-align: start; display: grid; place-items: center; padding: 56px 16px 72px; box-sizing: border-box; }
.sheet { width: min(1100px, 100%); background: var(--paper); backdrop-filter: blur(6px); border: 1px solid var(--ring); border-radius: 10px; padding: clamp(18px, 3vw, 40px); box-sizing: border-box; display: grid; gap: 18px; min-width: 0; }
.eyebrow { font: 600 .74rem/1.2 var(--f-mono); letter-spacing: .1em; text-transform: uppercase; color: var(--accent-ink); }
h1, h2, h3 { font-family: var(--f-display); margin: 0; line-height: 1.05; text-wrap: balance; }
h1 { font-size: clamp(2.6rem, 7vw, 5.2rem); font-weight: 700; letter-spacing: -.01em; }
h2 { font-size: clamp(1.9rem, 4vw, 3rem); font-weight: 700; }
h3 { font-size: 1.25rem; font-weight: 600; color: var(--ink-2); }
p { margin: 0; max-width: 68ch; color: var(--ink-2); }
.lead { font-size: 1.15rem; color: var(--ink); }
.cols { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); align-items: start; }
.cols > * { min-width: 0; }
.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
.tile { border-top: 3px solid var(--accent); padding: 10px 2px 0; display: grid; gap: 2px; }
.tile .v { font: 700 2.4rem/1 var(--f-display); font-variant-numeric: tabular-nums; }
.tile .l { font-size: .85rem; color: var(--ink-2); }
.tile .s { font: .76rem var(--f-mono); color: var(--muted); }
.big { font: 700 clamp(2.4rem, 5.5vw, 4.6rem)/1 var(--f-display); overflow-wrap: anywhere; color: var(--accent); font-variant-numeric: tabular-nums; }
.big small { font: 500 1.1rem/1.2 var(--f-body); color: var(--ink-2); display: block; max-width: 24ch; margin-top: 6px; }
.cards { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.card { border: 1px solid var(--ring); border-radius: 8px; padding: 14px 16px; display: grid; gap: 6px; min-width: 0; }
.card h3 { font-size: 1.1rem; color: var(--ink); }
.card .k { font: 600 .72rem var(--f-mono); color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
.flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; }
.flow .step { flex: 1 1 150px; border: 1px solid var(--ring); border-radius: 8px; padding: 10px 12px; font-size: .86rem; display: grid; gap: 4px; min-width: 0; position: relative; }
.flow .step b { font-family: var(--f-display); font-size: 1.1rem; font-weight: 600; }
.flow .step .val { font: 600 .82rem var(--f-mono); color: var(--accent-ink); }
.flow .step.out { border-color: var(--accent); border-width: 2px; }
.flow .arrow { align-self: center; color: var(--muted); font-size: 1.4rem; }
.chart { width: 100%; height: auto; display: block; }
.chartbox { overflow-x: auto; } .chartbox .chart { min-width: 560px; }
.chart text { fill: var(--ink-2); font: 12px var(--f-mono); }
.chart .t-strong { fill: var(--ink); font-weight: 600; }
.chart .small { font-size: 10px; } .chart .mono { font-family: var(--f-mono); }
.chart .grid { stroke: var(--ring); stroke-width: 1; } .chart .lim { stroke: var(--muted); stroke-dasharray: 3 3; } .chart .lim-t { fill: var(--muted); font-size: 10px; }
.chart .b-tau { fill: var(--st-muy_malo); } .chart .b-25 { fill: var(--st-bueno); }
.chart .on-bar { fill: #fff; font-weight: 600; font-size: 11px; }
.chart .pt { fill: var(--accent); stroke: var(--paper); stroke-width: 2; }
.street .traza { fill: none; stroke: var(--muted); stroke-width: 1.5; opacity: .6; }
.street .tramo { fill: none; stroke-width: 9; stroke-linecap: round; opacity: .9; }
.street .nodo { fill: var(--paper); stroke: var(--ink); stroke-width: 1.5; }
.street .prof { fill: var(--accent); opacity: .55; }
.street .scale { stroke: var(--ink); stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .82rem; color: var(--ink-2); align-items: center; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 5px; }
.gallery { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); }
figure { margin: 0; display: grid; gap: 6px; min-width: 0; }
figure img { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--ring); }
figcaption { font-size: .8rem; color: var(--ink-2); line-height: 1.4; }
.pill { display: inline-block; font: 600 .7rem var(--f-mono); padding: 1px 8px; border-radius: 99px; color: #fff; }
.et-prevencion { background: var(--et-prev); } .et-mantenimiento { background: var(--et-mant); } .et-intervencion { background: var(--et-int); }
.tablebox { overflow: auto; max-height: 52vh; border: 1px solid var(--ring); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--ring); text-align: left; vertical-align: top; white-space: nowrap; }
th { position: sticky; top: 0; background: var(--paper); font: 600 .7rem var(--f-mono); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); cursor: pointer; }
th:focus-visible { outline: 2px solid var(--accent); }
td.n { text-align: right; font-family: var(--f-mono); }
.cmp th:first-child, .cmp td:first-child { white-space: normal; min-width: 150px; }
.cmp td { white-space: normal; } .cmp th { white-space: normal; }
.compact table { font-size: .76rem; } .compact th, .compact td { padding: 4px 8px; }
.stack { grid-template-columns: 1fr; }
.cmp td.us { color: var(--ink); font-weight: 600; background: color-mix(in oklab, var(--accent) 10%, transparent); }
.psis { display: grid; gap: 6px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.psi { display: flex; justify-content: space-between; align-items: center; gap: 8px; border: 1px solid var(--ring); border-radius: 6px; padding: 6px 10px; font: .8rem var(--f-mono); }
.psi i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; } .psi .ok { background: var(--ok); } .psi .al { background: var(--warn); } .psi .fa { background: var(--crit); }
/* YOLO paso a paso */
.yolo { display: grid; gap: 16px; grid-template-columns: minmax(0, 1.3fr) minmax(240px, 1fr); align-items: start; }
@media (max-width: 760px) { .yolo { grid-template-columns: 1fr; } }
.yolo-stage { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--ring); }
.yolo-stage img { display: block; width: 100%; height: auto; }
.yolo-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.yolo-stage .cell { fill: none; stroke: rgba(255,255,255,.55); stroke-width: .6; }
.yolo-stage .vec rect { fill: var(--c4); } .yolo-stage .vec .bg { fill: rgba(0,0,0,.45); }
.yolo-stage .cand { fill: none; stroke: #fff; stroke-width: 1; opacity: .5; }
.yolo-stage .keep { fill: none; stroke-width: 2.5; }
.yolo-stage .lbl { fill: #fff; font: 600 11px var(--f-mono); paint-order: stroke; stroke: rgba(0,0,0,.6); stroke-width: 3; }
.steps { display: grid; gap: 8px; counter-reset: s; }
.steps li { list-style: none; border-left: 3px solid var(--ring); padding: 6px 12px; font-size: .92rem; color: var(--ink-2); cursor: pointer; }
.steps li.on { border-color: var(--accent); color: var(--ink); }
.steps li b { font-family: var(--f-display); font-size: 1.1rem; display: block; color: var(--ink); }
.ctrl { display: flex; gap: 8px; align-items: center; font-size: .82rem; color: var(--muted); }
button { font: 600 .82rem var(--f-mono); padding: 6px 12px; border-radius: 6px; border: 1px solid var(--ring); background: var(--paper); color: var(--ink); cursor: pointer; }
button:focus-visible { outline: 2px solid var(--accent); }
/* navegación */
#nav { position: fixed; left: 0; right: 0; bottom: 0; z-index: 2; display: flex; justify-content: center; align-items: center; gap: 14px; padding: 10px 16px calc(10px + env(safe-area-inset-bottom, 0px)); pointer-events: none; }
#nav > * { pointer-events: auto; }
#dots { display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; }
#dots a { width: 8px; height: 8px; border-radius: 50%; background: var(--ring); display: block; border: 1px solid var(--muted); }
#dots a.on { background: var(--accent); border-color: var(--accent); }
#place { position: fixed; top: calc(10px + env(safe-area-inset-top, 0px)); right: 16px; z-index: 2; font: 600 .72rem var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); background: var(--paper); border: 1px solid var(--ring); border-radius: 99px; padding: 4px 10px; }
footer.src { font-size: .72rem; color: var(--muted); line-height: 1.5; }
ul.plain { margin: 0; padding-left: 18px; color: var(--ink-2); display: grid; gap: 6px; }
.note { border-left: 3px solid var(--warn); padding: 8px 12px; font-size: .9rem; color: var(--ink-2); background: color-mix(in oklab, var(--warn) 8%, transparent); border-radius: 0 6px 6px 0; }
</style>

{{BOGOTA}}
<div id="place" aria-live="polite">Bogotá</div>

<main id="deck">

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="2.2" data-lm="candelaria" data-place="La Candelaria">
 <div class="sheet">
  <div class="eyebrow">Universidad del Rosario · Escuela de Ciencias e Ingeniería · 2026</div>
  <h1>DIMMIT V1<br>sale a la calle</h1>
  <p class="lead">Dispositivo de Instancia y Modelado Vial para el Mantenimiento de la Infraestructura de Tránsito. Primera toma de datos propios: {{N_FOTOS}} fotos, 3 recorridos y un sensor de profundidad en La Candelaria, Bogotá.</p>
  <p>William Esteban Angarita González · María Paula Díaz Cano · Ana Karina Pulido · Juan Manuel Moreno Agudelo</p>
  <div class="ctrl">→ o espacio para avanzar · ← para volver · también funciona con scroll</div>
 </div>
</section>

<section class="slide" data-lat="4.6133" data-lon="-74.0699" data-z="1.6" data-lm="centro" data-place="Centro">
 <div class="sheet">
  <div class="eyebrow">Qué es</div>
  <h2>Un prototipo portátil que convierte cada recorrido en un registro georreferenciado del estado de la vía</h2>
  <p class="lead">Una cámara ve el pavimento, un detector de daños lo clasifica, un sensor mide la profundidad de los baches y el GPS ubica cada tramo. El resultado es un índice de 0 a 100 por tramo, en la misma escala del PCI que usa el IDU, con una etiqueta operativa: prevención, mantenimiento o intervención.</p>
 </div>
</section>

<section class="slide" data-lat="4.52" data-lon="-74.13" data-z="1.5" data-lm="sur" data-place="Usme · Ciudad Bolívar">
 <div class="sheet">
  <div class="eyebrow">La problemática en tres números</div>
  <div class="cols">
   <div class="big">15.186<small>km-carril de malla vial en Bogotá (IDU, dic-2025)</small></div>
   <div class="big">22 %<small>en estado pobre, muy pobre, grave o fallado; Usme 17,8 % y Ciudad Bolívar 13,5 % de vías falladas</small></div>
   <div class="big">3–5×<small>más caro restaurar una vía que haberla mantenido a tiempo (Harral y Faiz, Banco Mundial)</small></div>
  </div>
  <svg viewBox="0 0 640 170" class="chart" role="img" aria-label="Curva de deterioro del pavimento">
   <path d="M40 30 C 220 36, 360 48, 430 70 C 500 92, 540 130, 600 150" fill="none" stroke="var(--plano)" stroke-width="3"/>
   <line x1="40" y1="150" x2="600" y2="150" class="grid"/><line x1="40" y1="20" x2="40" y2="150" class="grid"/>
   <line x1="430" y1="70" x2="430" y2="150" class="lim"/>
   <text x="40" y="166">tiempo</text><text x="600" y="166" text-anchor="end">falla</text>
   <text x="235" y="22" text-anchor="middle" class="t-strong">dos tercios de la vida con poco daño visible</text>
   <text x="424" y="120" text-anchor="end" class="t-strong">aquí la fisura se vuelve bache</text>
   <text x="424" y="136" text-anchor="end" class="small">y el costo se multiplica</text>
  </svg>
 </div>
</section>

<section class="slide" data-lat="4.74" data-lon="-74.047" data-z="1.6" data-lm="norte" data-place="Autopista Norte">
 <div class="sheet">
  <div class="eyebrow">Cómo se diagnostica hoy</div>
  <div class="cards">
   <div class="card"><div class="k">01 · Campaña láser del IDU</div><h3>Vehículos especializados</h3><p>Objetiva y de alta calidad, pero una vez cada varios años y sobre una parte de la red.</p></div>
   <div class="card"><div class="k">02 · Recorridos de la UMV</div><h3>Inspección visual</h3><p>Depende del inspector: lenta, costosa y poco consistente entre evaluadores.</p></div>
   <div class="card"><div class="k">03 · Reportes ciudadanos</div><h3>"Hay un hueco"</h3><p>Llegan de donde más se reporta, no de donde más se necesita; muchos duplicados y direcciones imprecisas.</p></div>
  </div>
  <p>Entre campaña y campaña, la programación del mantenimiento local depende de los ojos y de las quejas.</p>
 </div>
</section>

<section class="slide" data-lat="4.6015" data-lon="-74.0729" data-z="2.4" data-lm="rosario" data-place="Universidad del Rosario">
 <div class="sheet">
  <div class="eyebrow">La propuesta</div>
  <h2>Complemento de las campañas del IDU, no su reemplazo</h2>
  <div class="flow">
   <div class="step"><b>Cámara + GPS</b>fotos cada 0,5 s con hora y posición</div><div class="arrow">→</div>
   <div class="step"><b>YOLOv10</b>grieta longitudinal · transversal · piel de cocodrilo · bache</div><div class="arrow">→</div>
   <div class="step"><b>Sensor ultrasónico</b>profundidad del bache en cm</div><div class="arrow">→</div>
   <div class="step out"><b>Índice 0–100 por tramo</b>+ etiqueta: prevención · mantenimiento · intervención</div>
  </div>
  <p>Más frecuencia de observación donde hoy solo hay inspección visual y quejas, y registro de las fisuras antes de que sean baches: la ventana en la que mantener es más barato.</p>
 </div>
</section>

<section class="slide" data-lat="4.6133" data-lon="-74.0699" data-z="1.3" data-lm="centro" data-place="Centro">
 <div class="sheet">
  <div class="eyebrow">Metodología</div>
  <div class="flow">
   <div class="step"><b>1 · Datos</b>RDD2020 completo<span class="val">21.041 imágenes anotadas · Japón, India, Chequia</span></div><div class="arrow">→</div>
   <div class="step"><b>2 · Detector</b>YOLOv10n entrenado en Kaggle<span class="val">mAP50 0,28 · piel de cocodrilo 0,50 · 88 ms/foto en CPU</span></div><div class="arrow">→</div>
   <div class="step"><b>3 · Score</b>pseudo-PCI estilo ASTM D6433<span class="val">densidad → valor deducido → 100 − CDV</span></div><div class="arrow">→</div>
   <div class="step"><b>4 · Validación</b>contra anotaciones humanas<span class="val">Spearman 0,80 por tramo</span></div>
  </div>
  <p>Todo el flujo corre en la nube (Kaggle) y es reproducible: mismos datos, mismos umbrales, mismo resultado.</p>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="2.0" data-lm="candelaria" data-place="La Candelaria">
 <div class="sheet">
  <div class="eyebrow">Cómo "ve" YOLOv10 · foto real {{YOLO_ID}}</div>
  <div class="yolo">
   <div class="yolo-stage" id="stage"><img src="{{YOLO_IMG}}" alt="Foto de campo usada para ilustrar la detección"><svg id="ov" viewBox="0 0 1000 562" preserveAspectRatio="none"></svg></div>
   <div>
    <ol class="steps" id="steps">
     <li data-s="0"><b>La imagen entra completa</b>Se reduce a 416 px; la red la mira toda de una vez, no por pedazos.</li>
     <li data-s="1"><b>Se divide en una cuadrícula</b>Cada celda es responsable de lo que cae en ella.</li>
     <li data-s="2"><b>Cada celda produce un vector</b>Posición y tamaño de una caja + probabilidad de cada una de las 4 clases.</li>
     <li data-s="3"><b>Los vectores se vuelven cajas candidatas</b>Muchas, cada una con su confianza. YOLOv10 aprende a dar una sola caja por daño, sin el filtro NMS clásico.</li>
     <li data-s="4"><b>Quedan las que superan el umbral y los filtros</b>Umbral de campo por clase ({{TAU_CAMPO}}: la confianza a la que el 50 % de las cajas acierta en RDD2020) más geometría, persistencia entre fotos y consenso de escala. Con el umbral de RDD2020 quedaban {{YOLO_N_TAU}}; aquí quedan {{YOLO_N}}.</li>
    </ol>
    <div class="ctrl"><button id="yprev">‹ paso</button><button id="ynext">paso ›</button><span id="ylab"></span></div>
    <p style="font-size:.8rem">YOLOv10n: 2,3 millones de parámetros, 12 épocas sobre RDD2020. Las cajas y confianzas de esta animación son las detecciones reales de la foto.</p>
   </div>
  </div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="2.0" data-lm="candelaria" data-place="La Candelaria">
 <div class="sheet">
  <div class="eyebrow">De las cajas al score y a la etiqueta · tramo de ejemplo {{EX_ID}}</div>
  <div class="flow">
   <div class="step"><b>Cajas de YOLO</b>clase · confianza · tamaño<span class="val">{{EX_N_CAJAS}}</span></div><div class="arrow">→</div>
   <div class="step"><b>Conteo y área por clase</b>en las {{EX_N}} fotos del tramo</div><div class="arrow">→</div>
   <div class="step"><b>Densidad ρ por clase</b>% del área visible (~143 m² por foto)</div><div class="arrow">→</div>
   <div class="step"><b>Valor deducido DV</b>curva estilo ASTM D6433<span class="val">{{EX_RHO}}</span></div><div class="arrow">→</div>
   <div class="step"><b>CDV</b>DVmax + 0,35 · (ΣDV − DVmax)</div><div class="arrow">→</div>
   <div class="step out"><b>score = 100 − CDV</b><span class="val">{{EX_SCORE}} → {{EX_ESTADO}}</span></div>
  </div>
  <div class="cols">
   <div class="card"><div class="k">entrada lateral · GPS</div><h3>id_via + lat/lon</h3><p>id_via = hash(fecha, prueba, tramo, centroide). Mismo dato, mismo ID. Ejemplo: {{EX_ID}} en {{EX_LAT}}, {{EX_LON}}.</p></div>
   <div class="card"><div class="k">entrada lateral · ultrasónico</div><h3>profundidad_max_cm</h3><p>No entra al score; puede subir la etiqueta: ≥ 3 cm → al menos mantenimiento, ≥ 8 cm → intervención. Ejemplo: {{EX_PROF}} cm.</p></div>
   <div class="card"><div class="k">regla de etiqueta</div><h3><span class="pill et-prevencion">Prevención</span> <span class="pill et-mantenimiento">Mantenimiento</span> <span class="pill et-intervencion">Intervención</span></h3><p>bueno/satisfactorio → prevención · regular o profundidad ≥ 3 cm → mantenimiento · malo/muy malo, profundidad ≥ 8 cm o bache con confianza ≥ 0,5 → intervención. Ejemplo: {{EX_ETIQ}}.</p></div>
  </div>
  <p style="font-size:.82rem">Qué se descartó de YOLO: el embedding interno de 448 valores no mejoró el score en la evaluación (Δ MAE +0,07, IC incluye 0). Se usan solo clase, confianza y geometría de cada caja.</p>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.2" data-lm="candelaria" data-place="La Candelaria · 5-oct-2026">
 <div class="sheet">
  <div class="eyebrow">La toma de datos · 5 de octubre de 2026, 9:38–9:48</div>
  <div class="tiles">{{KPIS}}</div>
  <div class="cols">
   <div>{{CALLE}}
    <div class="legend"><span><span class="sw" style="background:var(--st-bueno)"></span>bueno</span><span><span class="sw" style="background:var(--st-satisfactorio)"></span>satisfactorio</span><span><span class="sw" style="background:var(--st-regular)"></span>regular</span><span><span class="sw" style="background:var(--st-malo)"></span>malo</span><span><span class="sw" style="background:var(--st-muy_malo)"></span>muy malo</span><span><span class="sw" style="background:var(--accent);border-radius:50%;opacity:.6"></span>lectura ultrasónica ≥ 3 cm</span><span>— Prueba 2 · - - Prueba 3 · ··· Prueba 4</span></div>
   </div>
   <p>Tres recorridos a pie por la misma cuadra, {{LONG}} m en total, cortados en {{N_TRAMOS}} tramos de ~10 m por distancia GPS. Cada tramo tiene su ID, su centroide y sus fotos; la posición tiene ±5 m de incertidumbre (GPS de teléfono).</p>
  </div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.2" data-lm="candelaria" data-place="La Candelaria · resultados">
 <div class="sheet">
  <div class="eyebrow">Qué encontró el detector</div>
  <h2>Antes y después: umbral de RDD2020 frente al modo campo</h2>
  <div class="legend"><span><span class="sw" style="background:var(--st-muy_malo)"></span>antes: umbral de RDD2020 (0,04–0,09), sin filtros, sensor con base fija de 40 cm</span><span><span class="sw" style="background:var(--st-bueno)"></span>modo campo: umbral de precisión 50 %, 4 filtros, sensor autocalibrado, severidad ASTM</span></div>
  <div class="chartbox">{{BARRAS}}</div>
  <div class="note">Antes, los 9 tramos salían en intervención ({{N_INT_ANTES}}/9); ahora {{N_INT}}/9, concentrados en la Prueba 3, donde el sensor sí midió un hueco real. De {{N_BRUTAS}} cajas brutas, {{N_TAU}} pasaban el umbral de RDD2020 y {{N_CONSERVADAS}} pasan los cuatro filtros del modo campo. El precio: el modo campo es conservador y deja pasar daño real (ver validación).</div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.0" data-lm="candelaria" data-place="La Candelaria · diagnóstico">
 <div class="sheet">
  <div class="eyebrow">Diagnóstico: ¿modelo, sensores o contexto? Las tres cosas, y se puede medir cuánto cada una</div>
  <div class="tablebox" style="max-height:none"><table class="cmp"><thead><tr><th>Evidencia</th><th>Causa</th><th>Corrección aplicada</th><th>Estado</th></tr></thead><tbody>
   <tr><td>La Prueba 2 marcaba 10 cm de profundidad <b>constantes</b> durante 35 s; la Prueba 4, 17–25 cm. La distancia más frecuente del sensor era 50 y 61 cm, no los 40 cm del firmware.</td><td><b>Sensor: línea base mal puesta</b> (altura del montaje). Fabricó baches en 2 de 3 pruebas y disparó "intervención" en los 9 tramos.</td><td>Línea base autocalibrada = distancia más frecuente de cada serie (el pavimento sano). Severidad del bache por profundidad según ASTM D6433.</td><td><span class="pill et-prevencion">Corregido</span></td></tr>
   <tr><td>183 "grietas transversales" con ancho mediano 0,52 de la imagen y altura 0,11 (área 4× la de RDD2020): juntas, sombra del bordillo y andén. Áreas de piel de cocodrilo y bache 1,4–2× las de RDD2020.</td><td><b>Dominio / encuadre</b>: cámara a 1,3 m del suelo mirando hacia abajo, a pie, frente a dashcam sobre la vía. El pseudo-PCI usa % de área: densidades infladas → "muy malo".</td><td>Umbral de campo por clase (precisión ≥ 50 % en RDD2020), filtro de bandas y de región de vía, persistencia entre fotos consecutivas, consenso a dos escalas.</td><td><span class="pill et-mantenimiento">Mitigado</span></td></tr>
   <tr><td>En el propio test de RDD2020 la confianza máxima media por foto es 0,17 y solo el 23 % de las fotos tiene una caja ≥ 0,25. En campo: 0,13 y 10 %.</td><td><b>Modelo</b>: YOLOv10n de 12 épocas en CPU a 416 px (mAP50 0,28). La caída por dominio existe pero es moderada; la base ya era débil.</td><td>Inferencia a 640 px (confianza media 0,13 → 0,15). Lo demás requiere datos: fotos propias anotadas desde el vehículo y GPU.</td><td><span class="pill et-intervencion">Pendiente</span></td></tr>
   <tr><td>Los 6 parches rectangulares de asfalto de la muestra: el detector marca la mitad como piel de cocodrilo con confianza 0,29–0,36.</td><td><b>Modelo</b>: confusión parche ↔ piel de cocodrilo (textura). RDD2020 casi no tiene parches anotados.</td><td>Se reporta; no se filtra a mano (sería inventar una regla). Entra al plan de anotación propia.</td><td><span class="pill et-intervencion">Pendiente</span></td></tr>
   <tr><td>"Bogotá tiene más huecos": los daños reales de la cuadra (piel de cocodrilo junto al bordillo, el hueco de la Prueba 3) el modelo sí los ve.</td><td><b>No es el contexto</b> lo que falla; es el encuadre y la calibración.</td><td>—</td><td><span class="pill et-prevencion">Descartado</span></td></tr>
  </tbody></table></div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.0" data-lm="candelaria" data-place="La Candelaria · filtros">
 <div class="sheet">
  <div class="eyebrow">Cómo no dejarnos meter falsos positivos: cuatro capas y una validación</div>
  <div class="cols compact">
   <div>
    <h3>Cajas descartadas por capa (640 px; {{N_BRUTAS}} brutas)</h3>
    <div class="tablebox" style="max-height:none"><table class="cmp"><thead><tr><th>Capa</th><th>grieta long.</th><th>grieta transv.</th><th>piel coc.</th><th>bache</th><th>total</th></tr></thead><tbody>{{FILTROS}}</tbody></table></div>
    <p style="font-size:.82rem">Umbral de campo = confianza a la que la precisión por caja en el test de RDD2020 llega al 50 %: {{TAU_CAMPO}}. Persistencia: a 2 Hz un daño real sale en fotos consecutivas; una sombra parpadea.</p>
   </div>
   <div>
    <h3>Validación en 31 fotos revisadas a ojo (presencia por clase)</h3>
    <div class="tablebox" style="max-height:none"><table class="cmp"><thead><tr><th>Clase</th><th>precisión antes</th><th>recall antes</th><th>precisión campo</th><th>recall campo</th></tr></thead><tbody>{{VALID}}</tbody></table></div>
    <p style="font-size:.82rem">Revisión rápida del equipo, no anotación experta. El modo campo gana precisión y pierde recall: deja pasar piel de cocodrilo de baja confianza y las juntas transversales deterioradas que el filtro de bandas elimina. El {{PARCHES}} % de los parches de asfalto siguen marcados como piel de cocodrilo.</p>
   </div>
  </div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.2" data-lm="candelaria" data-place="La Candelaria · daños">
 <div class="sheet">
  <div class="eyebrow">Daños detectados por tramo (cajas que pasan los cuatro filtros)</div>
  <div class="legend"><span><span class="sw" style="background:var(--c1)"></span>grieta longitudinal</span><span><span class="sw" style="background:var(--c2)"></span>grieta transversal</span><span><span class="sw" style="background:var(--c3)"></span>piel de cocodrilo</span><span><span class="sw" style="background:var(--c4)"></span>bache</span></div>
  <div class="chartbox">{{DANOS}}</div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.2" data-lm="candelaria" data-place="La Candelaria · fotos">
 <div class="sheet">
  <div class="eyebrow">Una foto por tramo: la de mayor confianza conservada, con las cajas que pasaron los filtros</div>
  <div class="gallery">{{GALERIA}}</div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.2" data-lm="candelaria" data-place="La Candelaria · sensor">
 <div class="sheet">
  <div class="eyebrow">¿Coincide el sensor con lo que ve YOLO?</div>
  <div class="cols">
   <div>{{DISPERSION}}<div class="legend"><span>● Prueba 2</span><span>■ Prueba 3</span><span>▲ Prueba 4</span></div></div>
   <div style="display:grid;gap:12px;min-width:0">
    <p class="lead">Débil y sin significancia: Spearman {{RS}} con IC [{{RS_LO}}, {{RS_HI}}] sobre 9 tramos. Con la línea base fija del firmware salía {{RS40}} [{{RS40_LO}}, {{RS40_HI}}], una correlación fabricada por el montaje, no por los baches.</p>
    <div class="tablebox compact" style="max-height:none"><table class="cmp"><thead><tr><th>Prueba</th><th>línea base (moda, cm)</th><th>prof. máx con base 40</th><th>prof. máx autocalibrada</th><th>severidad ASTM</th><th>baches (modo campo)</th></tr></thead><tbody>{{CORR}}</tbody></table></div>
    <p style="font-size:.85rem">La línea base se autocalibra con la distancia más frecuente de cada serie (el pavimento sano): 50, 37 y 61 cm, no los 40 cm del firmware. Con eso la Prueba 2 y la Prueba 4 dejan de "tener" baches de 10–25 cm y la Prueba 3 conserva su evento real (p95 11,5 cm, severidad alta según ASTM D6433). El Arduino no tiene reloj: la alineación foto–lectura es por tiempo normalizado (supuesto). Con 3 pruebas, descriptivo.</p>
   </div>
  </div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.2" data-lm="candelaria" data-place="La Candelaria · calidad">
 <div class="sheet">
  <div class="eyebrow">Calidad de las fotos y confianza del modelo</div>
  <div class="tiles">
   <div class="tile"><div class="v">{{CONF_MED}}</div><div class="l">confianza máxima media por foto</div><div class="s">en RDD2020 test el detector llega a 0,5 en piel de cocodrilo</div></div>
   <div class="tile"><div class="v">{{PCT_CAMPO}} %</div><div class="l">fotos con alguna caja que pasa los filtros</div><div class="s">{{PCT_DET}} % tenían alguna caja con el umbral de RDD2020</div></div>
   <div class="tile"><div class="v">{{PCT_BORR}} %</div><div class="l">fotos borrosas</div><div class="s">bajo el p05 de nitidez de RDD2020</div></div>
   <div class="tile"><div class="v">{{PCT_SOBRE}} %</div><div class="l">fotos sobreexpuestas</div><div class="s">sobre el p98 de brillo de RDD2020</div></div>
   <div class="tile"><div class="v">{{AUROC}}</div><div class="l">AUROC de dominio</div><div class="s">0,5 = igual a RDD2020 · 1,0 = otro mundo</div></div>
   <div class="tile"><div class="v">{{MS}} ms</div><div class="l">por foto en Kaggle (CPU)</div><div class="s">{{N_FOTOS}} fotos en menos de 2 minutos</div></div>
  </div>
  <h3>Deriva frente a RDD2020 (PSI; ≤ 0,10 OK · ≤ 0,25 alerta · mayor falla)</h3>
  <div class="psis">{{PSI}}</div>
  <div class="tablebox"><table id="tcal"><thead><tr><th>id_via</th><th>brillo</th><th>nitidez</th><th>borrosas</th><th>sobreexp.</th><th>conf. máx media</th><th>fotos con caja ≥ τ</th><th>≥ 0,25</th></tr></thead><tbody>{{CALIDAD}}</tbody></table></div>
 </div>
</section>

<section class="slide" data-lat="4.5995" data-lon="-74.0735" data-z="3.2" data-lm="candelaria" data-place="La Candelaria · tabla">
 <div class="sheet">
  <div class="eyebrow">Salida 1 · vias_clasificacion.csv (clic en un encabezado para ordenar)</div>
  <div class="tablebox"><table id="tvias"><thead><tr><th>id_via</th><th>prueba</th><th>tramo</th><th>fotos</th><th>lat</th><th>lon</th><th>daños (modo campo)</th><th>score antes</th><th>estado antes</th><th>etiqueta antes</th><th>score</th><th>estado</th><th>etiqueta</th><th>prof. base 40</th><th>prof. autocal.</th><th>conf. modelo</th></tr></thead><tbody>{{TABLA}}</tbody></table></div>
  <p style="font-size:.82rem">La salida 2 (calidad, confianza, deriva y correlación) está en <span class="mono">calidad_confianza_correlacion.csv</span>, en formato largo, con sus intervalos de confianza.</p>
 </div>
</section>

<section class="slide" data-lat="4.65" data-lon="-74.11" data-z="1.0" data-lm="ciudad" data-place="Bogotá">
 <div class="sheet">
  <div class="eyebrow">Insights · DIMMIT frente al diagnóstico oficial</div>
  <div class="tablebox" style="max-height:none"><table class="cmp"><thead><tr><th>KPI</th><th>Hoy en Bogotá</th><th>DIMMIT (medido en esta entrega)</th></tr></thead><tbody>
   <tr><td>Frecuencia de diagnóstico objetivo</td><td>Campaña láser del IDU: 2016 → 2020, unos 4 años entre campañas</td><td class="us">Cada recorrido</td></tr>
   <tr><td>Tiempo del diagnóstico</td><td>6,5 meses para 8.617 km-carril (56 % de la red)</td><td class="us">{{MS}} ms por foto; {{N_FOTOS}} fotos → resultado en minutos</td></tr>
   <tr><td>Costo del diagnóstico</td><td>≈ $2.440 millones la campaña 2020</td><td class="us">Teléfono + Arduino + sensor ultrasónico (costo unitario por medir)</td></tr>
   <tr><td>Reactivo vs. preventivo</td><td>UMV: ~7.000 huecos/mes, 105.157 en 2025, 211.471 desde 2024</td><td class="us">Registra fisuras y piel de cocodrilo antes del bache: la ventana 3–5× más barata</td></tr>
   <tr><td>Objetividad</td><td>Inspección visual: subjetiva y poco consistente entre evaluadores</td><td class="us">Índice reproducible: mismo dato → mismo ID y mismo score</td></tr>
  </tbody></table></div>
 </div>
</section>

<section class="slide" data-lat="4.658" data-lon="-74.094" data-z="1.4" data-lm="occidente" data-place="Bogotá">
 <div class="sheet">
  <div class="eyebrow">Insights · DIMMIT frente a los reportes ciudadanos</div>
  <div class="tablebox" style="max-height:none"><table class="cmp"><thead><tr><th>KPI</th><th>Reportes ciudadanos</th><th>DIMMIT</th></tr></thead><tbody>
   <tr><td>Reportes útiles</td><td>Medellín, ReportesMED (cuatrienio): de 41.287 reportes de huecos, 12.219 duplicados (30 %) y 10.405 descartados por no competencia, inexistencia o dirección imprecisa (25 %) → ~45 % accionables</td><td class="us">0 duplicados por diseño (hash por foto y por tramo), GPS ±5 m, cada tramo con evidencia fotográfica y score</td></tr>
   <tr><td>Cobertura</td><td>Depende de quién reporta; menos reportes donde más se necesita</td><td class="us">Toda la calle recorrida, no solo lo que alguien decide reportar</td></tr>
   <tr><td>Información por reporte</td><td>"Hay un hueco"</td><td class="us">Tipo de daño (4 clases), score 0–100, estado, etiqueta, profundidad, lat/lon y hora</td></tr>
   <tr><td>Detección de daño por imagen</td><td>—</td><td class="us">AUROC 0,85 · F1 0,82 en el test de RDD2020</td></tr>
   <tr><td>Acuerdo con anotación humana</td><td>—</td><td class="us">Spearman 0,80 a nivel de tramo (solo visión)</td></tr>
  </tbody></table></div>
  <div class="note">Lo que aún no sale a favor: el F1 por caja del detector (0,32) está lejos del mejor del reto GRDDC 2020 (0,67 con ensambles en GPU), y en campo la confianza cae por el cambio de dominio. Siguiente paso: GPU a 640 px y fotos desde el vehículo.</div>
 </div>
</section>

<section class="slide" data-lat="4.65" data-lon="-74.11" data-z="1.0" data-lm="ciudad" data-place="Bogotá">
 <div class="sheet">
  <div class="eyebrow">Qué aprendimos y qué sigue</div>
  <div class="cards">
   <div class="card"><div class="k">funciona</div><h3>El flujo completo, en la nube</h3><p>Drive → Kaggle → ID por tramo → score, etiqueta y GPS, con paridad exacta entre la inferencia en Kaggle y la local, y un diagnóstico que separa modelo, sensor y encuadre con números.</p></div>
   <div class="card"><div class="k">corregido esta semana</div><h3>Sensor autocalibrado y filtros</h3><p>La línea base sale de los datos, la severidad del bache sigue la norma ASTM y cuatro capas frenan los falsos positivos. El precio es recall: el modo campo es conservador.</p></div>
   <div class="card"><div class="k">hay que anotar</div><h3>200–300 fotos propias</h3><p>Desde el vehículo, a la altura del dashcam. Con eso se fijan umbrales y escala del score en campo y se reentrena el detector con parches y juntas.</p></div>
   <div class="card"><div class="k">siguiente</div><h3>Más recorridos y GPU</h3><p>Decenas de cuadras en varias localidades, detector a 640 px en GPU y calibración frente al PCI del IDU.</p></div>
  </div>
  <h2>Un hueco tapado cuesta lo que cuesta. Una fisura vista a tiempo, mucho menos.</h2>
  <footer class="src">Fuentes de las cifras: IDU vía El Tiempo (25-may-2026) y boletín técnico IDU 2021-I; Harral y Faiz, Banco Mundial 1988; IDU, boletín de prensa 17-dic-2020 (campaña láser); bogota.gov.co, "Bogotá supera 200.000 huecos tapados" (12-mar-2026); Portafolio, "Bogotá invertirá 4,1 billones hasta 2027" (29-jul-2026); Alcaldía de Medellín, balance ReportesMED (16-nov-2023); Arya et al., GRDDC 2020; Ragnoli et al. 2018; Bianchini et al. 2010; Kontokosta y Hong 2021; Liu et al. 2024. Resultados propios: reports/campo/ en el repositorio DIMMIT-V1.</footer>
 </div>
</section>

</main>

<nav id="nav" aria-label="Diapositivas"><button id="prev" aria-label="Anterior">‹</button><div id="dots"></div><button id="next" aria-label="Siguiente">›</button></nav>

<script>
(function () {
  const K = 4000, LAT0 = 4.85, LON0 = -74.25;
  const deck = document.getElementById('deck'), slides = [...document.querySelectorAll('.slide')];
  const g = document.getElementById('mapaG'), svg = document.getElementById('mapa'), place = document.getElementById('place');
  const dots = document.getElementById('dots');
  slides.forEach((s, i) => { const a = document.createElement('a'); a.href = '#s' + (i + 1); a.setAttribute('aria-label', 'Diapositiva ' + (i + 1)); a.addEventListener('click', e => { e.preventDefault(); go(i); }); dots.appendChild(a); s.id = 's' + (i + 1); });
  let cur = -1;
  function focusMap(s) {
    const lat = +s.dataset.lat, lon = +s.dataset.lon, z = +s.dataset.z || 1;
    const x = (lon - LON0) * K, y = (LAT0 - lat) * K;
    // el SVG usa slice sobre 1000x1600: calculamos la escala real del viewBox en pantalla
    const W = svg.clientWidth, Hh = svg.clientHeight, sc = Math.max(W / 1000, Hh / 1600);
    const vx = (1000 * sc - W) / 2 / sc, vy = (1600 * sc - Hh) / 2 / sc; // recorte del slice
    const cx = vx + W / sc / 2, cy = vy + Hh / sc / 2;
    g.style.transform = `translate(${cx - x * z}px, ${cy - y * z}px) scale(${z})`;
    document.querySelectorAll('#mapa .lm').forEach(l => { l.classList.toggle('on', l.dataset.lm === s.dataset.lm); const m = l.getAttribute('transform').match(/translate\(([-\d.]+) ([-\d.]+)\)/); if (m) l.setAttribute('transform', `translate(${m[1]} ${m[2]}) scale(${(1.1 / z).toFixed(3)})`); });
    place.textContent = s.dataset.place || 'Bogotá';
  }
  function setCur(i) { if (i === cur) return; cur = i; focusMap(slides[i]); [...dots.children].forEach((d, k) => d.classList.toggle('on', k === i)); history.replaceState(null, '', '#s' + (i + 1)); }
  function go(i) { i = Math.max(0, Math.min(slides.length - 1, i)); slides[i].scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }); setCur(i); }
  const io = new IntersectionObserver(es => { es.forEach(e => { if (e.isIntersecting && e.intersectionRatio > .5) setCur(slides.indexOf(e.target)); }); }, { root: deck, threshold: [.55] });
  slides.forEach(s => io.observe(s));
  document.getElementById('prev').onclick = () => go(cur - 1); document.getElementById('next').onclick = () => go(cur + 1);
  addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); if (slides[cur] && slides[cur].querySelector('#stage') && yoloStep(1)) return; go(cur + 1); }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); go(cur - 1); }
  });
  addEventListener('resize', () => { if (cur >= 0) focusMap(slides[cur]); });
  const start = Math.max(0, parseInt((location.hash.match(/^#s(\d+)$/) || [0, 1])[1], 10) - 1);
  setCur(start); if (start) slides[start].scrollIntoView();

  // --- YOLO paso a paso ----------------------------------------------------------------------
  const boxes = {{YOLO_JSON}}, TAU = { D00: .07, D10: .04, D20: .09, D40: .07 };
  const COL = { D00: 'var(--c1)', D10: 'var(--c2)', D20: 'var(--c3)', D40: 'var(--c4)' }, NAME = { D00: 'grieta long.', D10: 'grieta transv.', D20: 'piel cocodrilo', D40: 'bache' };
  const ov = document.getElementById('ov'), steps = [...document.querySelectorAll('#steps li')], ylab = document.getElementById('ylab');
  let ys = 0;
  const NS = 'http://www.w3.org/2000/svg';
  function el(t, a) { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; }
  function render() {
    ov.replaceChildren(); steps.forEach((l, k) => l.classList.toggle('on', k === ys)); ylab.textContent = 'paso ' + (ys + 1) + ' de 5';
    if (ys >= 1) { for (let i = 1; i < 13; i++) ov.appendChild(el('line', { x1: i * 1000 / 13, y1: 0, x2: i * 1000 / 13, y2: 562, class: 'cell' })); for (let j = 1; j < 8; j++) ov.appendChild(el('line', { x1: 0, y1: j * 562 / 8, x2: 1000, y2: j * 562 / 8, class: 'cell' })); }
    if (ys === 2) { boxes.slice(0, 8).forEach((b, i) => { const gx = Math.floor(b.x * 13) * 1000 / 13 + 4, gy = Math.floor(b.y * 8) * 562 / 8 + 4; const grp = el('g', { class: 'vec' }); grp.appendChild(el('rect', { x: gx, y: gy, width: 62, height: 40, class: 'bg', rx: 2 })); const vals = [b.x, b.y, b.w, b.h, b.c === 'D00' ? b.p : b.p * .3, b.c === 'D10' ? b.p : b.p * .3, b.c === 'D20' ? b.p : b.p * .3, b.c === 'D40' ? b.p : b.p * .3]; vals.forEach((v, k) => { const h = Math.max(2, Math.min(1, v) * 32); const r = el('rect', { x: gx + 3 + k * 7.4, y: gy + 36 - h, width: 5.5, height: h, fill: k < 4 ? 'var(--c1)' : COL[['D00', 'D10', 'D20', 'D40'][k - 4]] }); r.style.animation = `grow .6s ${i * 60}ms both`; grp.appendChild(r); }); ov.appendChild(grp); }); }
    if (ys === 3) boxes.forEach(b => ov.appendChild(el('rect', { x: (b.x - b.w / 2) * 1000, y: (b.y - b.h / 2) * 562, width: b.w * 1000, height: b.h * 562, class: 'cand', style: 'opacity:' + (0.25 + b.p) })));
    if (ys === 4) boxes.filter(b => b.k).forEach(b => { ov.appendChild(el('rect', { x: (b.x - b.w / 2) * 1000, y: (b.y - b.h / 2) * 562, width: b.w * 1000, height: b.h * 562, class: 'keep', stroke: COL[b.c], rx: 3 })); const t = el('text', { x: (b.x - b.w / 2) * 1000 + 4, y: Math.max(12, (b.y - b.h / 2) * 562 - 4), class: 'lbl' }); t.textContent = NAME[b.c] + ' ' + b.p.toFixed(2); ov.appendChild(t); });
  }
  function yoloStep(d) { const n = ys + d; if (n < 0 || n > 4) return false; ys = n; render(); return true; }
  document.getElementById('ynext').onclick = () => yoloStep(1); document.getElementById('yprev').onclick = () => yoloStep(-1);
  steps.forEach(l => l.addEventListener('click', () => { ys = +l.dataset.s; render(); }));
  const st = document.createElement('style'); st.textContent = '@keyframes grow{from{transform:scaleY(0);transform-origin:bottom}to{transform:scaleY(1);transform-origin:bottom}} .vec rect{transform-box:fill-box}'; document.head.appendChild(st);
  render();

  // --- tablas ordenables ---------------------------------------------------------------------
  document.querySelectorAll('#tvias th, #tcal th').forEach((th, i) => { th.tabIndex = 0; const sort = () => { const tb = th.closest('table').tBodies[0], rows = [...tb.rows]; const asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0'; const num = s => parseFloat(s.replace(/\./g, '').replace(',', '.').replace('%', '')); rows.sort((a, b) => { const x = a.cells[i].textContent.trim(), y = b.cells[i].textContent.trim(); const nx = num(x), ny = num(y); const c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y, 'es'); return asc ? c : -c; }); rows.forEach(r => tb.appendChild(r)); }; th.onclick = sort; th.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sort(); } }; });
})();
</script>
"""


if __name__ == "__main__":
    build()
