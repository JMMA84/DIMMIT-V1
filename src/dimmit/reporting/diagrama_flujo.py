"""Diagrama de funcionamiento para el póster (Figura 2b): PNG 300 dpi, casi cuadrado (ancho:alto ~0,9).

Dos columnas que se juntan en la etiqueta: visión (cámara -> YOLOv10 -> filtros -> score) y profundidad
(sensor -> depuración -> profundidad). Umbrales de configs/campo.yaml y conteos de la corrida vigente.
Escribe reports/campo/poster/diagrama_flujo.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dimmit.campo import INDEX_CAMPO, OUT_CAMPO, load_cfg
from dimmit.campo.scores import FEAT, load_depth
from dimmit.utils.io import ensure_dir

INK, MUTED = "#15191c", "#4f575c"
VISION, SENSOR, COMUN = "#2a78d6", "#c98500", "#3d4549"
FIG_W, FIG_H = 6.0, 6.6


def box(ax, x, y, w, h, title, sub, color, fill=True):
    """Caja redondeada; (x, y) = centro, coordenadas 0-1 del lienzo."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.018",
                                facecolor=color if fill else "white", edgecolor=color, linewidth=1.8, zorder=2))
    tc = "white" if fill else INK
    if sub:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc, zorder=3)
        ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=8.2, color=tc if fill else MUTED, zorder=3, linespacing=1.15)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc, zorder=3)
    return (x, y - h / 2), (x, y + h / 2)


def arrow(ax, a, b, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=1, shrinkB=1))


def main():
    cfg = load_cfg()
    fr = pd.read_parquet(INDEX_CAMPO / "fotogramas.parquet")
    if "conjunto" in fr:
        fr = fr[fr["conjunto"] == "evaluacion"]
    det_f = pd.read_parquet(FEAT / "detecciones_filtradas.parquet")
    det_f = det_f[det_f["image_id"].isin(fr["image_id"])]
    depth = load_depth(cfg)
    n_lect = sum(f["n_lecturas"] for _, f in depth.values())
    n_val = sum(f["n_validas"] for _, f in depth.values())
    n_cajas, n_keep = len(det_f), int(det_f["conservada"].sum())
    u, et = cfg["ultrasonico"], cfg["etiqueta"]
    dec = lambda v: f"{v:g}".replace(".", ",")

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    xl, xr, w, h = 0.27, 0.73, 0.42, 0.105
    ys = [0.885, 0.74, 0.595, 0.45]
    ax.text(xl, 0.975, "VISIÓN", ha="center", va="center", fontsize=9.5, fontweight="bold", color=VISION)
    ax.text(xr, 0.975, "PROFUNDIDAD", ha="center", va="center", fontsize=9.5, fontweight="bold", color=SENSOR)
    L = [
        ("1. Cámara (celular)", f"{len(fr)} fotos a pie, {fr['prueba'].nunique()} recorridos"),
        ("2. YOLOv10 (RDD2020)", "grietas, piel de cocodrilo, bache\ninferencia en Kaggle"),
        ("3. Filtros de campo", f"umbral calibrado con el IDU, vía,\nsuperficie, persistencia: {n_cajas} → {n_keep} cajas"),
        ("4. Score 0–100", "pseudo-PCI (ASTM D6433)\nescala IDU: bueno · regular · malo"),
    ]
    R = [
        ("1. Sensor ultrasónico", f"en bastón, ~4 lecturas/s\n{n_lect} lecturas"),
        ("2. Depuración", f"fuera > {dec(u['rango_max_cm'])} cm y 0,00 cm\n{n_val} de {n_lect} válidas"),
        ("3. Profundidad", f"distancia − {dec(u['linea_base_cm'])} cm\n> 0 bache · ≈ 0 sano"),
        ("4. Profundidad del tramo", "máxima; la severidad ASTM del\nbache (L/M/H) pondera el score"),
    ]
    prev_l = prev_r = None
    for y, (tl, sl), (tr, sr) in zip(ys, L, R):
        bl = box(ax, xl, y, w, h, tl, sl, VISION)
        br = box(ax, xr, y, w, h, tr, sr, SENSOR)
        if prev_l:
            arrow(ax, prev_l, bl[1], VISION)
            arrow(ax, prev_r, br[1], SENSOR)
        prev_l, prev_r = bl[0], br[0]
    # la severidad del bache pondera el score (cruce entre columnas)
    arrow(ax, (xr - w / 2, ys[3] + 0.012), (xl + w / 2, ys[3] + 0.012), SENSOR)

    # etiqueta
    ye, he = 0.235, 0.215
    ax.add_patch(FancyBboxPatch((0.06, ye - he / 2), 0.88, he, boxstyle="round,pad=0,rounding_size=0.018",
                                facecolor="white", edgecolor=COMUN, linewidth=1.8, zorder=2))
    ax.text(0.5, ye + he / 2 - 0.028, "5. Etiqueta por tramo (primera regla que se cumple)", ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK)
    reglas = [
        ("Intervención", "#d03b3b", f"estado malo, bache ≥ {dec(et['intervencion']['profundidad_cm'])} cm\no bache visto con conf ≥ {dec(et['intervencion']['bache_conf_min'])}"),
        ("Mantenimiento", "#c98500", f"estado regular\no bache ≥ {dec(et['mantenimiento']['profundidad_cm'])} cm"),
        ("Prevención", "#2a78d6", "el resto"),
    ]
    xs3 = [0.2, 0.5, 0.8]
    for x, (name, col, rule) in zip(xs3, reglas):
        ax.add_patch(FancyBboxPatch((x - 0.13, ye - 0.003), 0.26, 0.05, boxstyle="round,pad=0,rounding_size=0.012", facecolor=col, edgecolor=col, zorder=3))
        ax.text(x, ye + 0.022, name, ha="center", va="center", fontsize=9.2, fontweight="bold", color="white", zorder=4)
        ax.text(x, ye - 0.047, rule, ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.15, zorder=4)
    arrow(ax, (xl, ys[3] - h / 2), (xl, ye + he / 2), VISION)
    arrow(ax, (xr, ys[3] - h / 2), (xr, ye + he / 2), SENSOR)

    # salidas
    yo, ho = 0.055, 0.075
    box(ax, 0.5, yo, 0.88, ho, "6. Salidas: score, etiqueta y profundidad por tramo", "", COMUN)
    arrow(ax, (0.5, ye - he / 2), (0.5, yo + ho / 2), COMUN)

    out = ensure_dir(OUT_CAMPO / "poster") / "diagrama_flujo.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
