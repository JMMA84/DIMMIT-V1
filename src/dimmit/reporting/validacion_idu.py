"""Figura 3 del póster (validación): arriba lo que mide DIMMIT en la toma vigente, abajo lo que reporta el IDU
para la misma zona y el mismo tipo de superficie. PNG 300 dpi, ancho:alto ~1,2.

Lee data/campo/features/vias.parquet y data/campo/idu/calzadas_estado.csv (make campo-idu).
Escribe reports/campo/poster/validacion_idu.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimmit.campo import CAMPO, OUT_CAMPO, load_cfg
from dimmit.campo.scores import FEAT
from dimmit.utils.io import ensure_dir

INK, MUTED, GRID = "#15191c", "#5f676c", "#d9dcd5"
# mismos tonos para el estado en ambos paneles
EST_COL = {"bueno": "#3a9a5b", "regular": "#e0a526", "malo": "#d03b3b", "sin_clasificar": "#b9bdb5"}
EST_NOM = {"bueno": "Bueno", "regular": "Regular", "malo": "Malo", "sin_clasificar": "Sin clasificar"}
ETIQ_COL = {"intervencion": "#d03b3b", "mantenimiento": "#c98500", "prevencion": "#2a78d6"}
ETIQ_NOM = {"intervencion": "Intervención", "mantenimiento": "Mantenimiento", "prevencion": "Prevención"}
GRUPO_NOM = {"asfalto": "asfalto", "adoquin": "adoquín"}


def main():
    cfg = load_cfg()
    v = pd.read_parquet(FEAT / "vias.parquet")
    t = v[v["nivel"] == "tramo"]
    idu = pd.read_csv(CAMPO / "idu" / "calzadas_estado.csv")
    sitios = sorted(t["sitio"].dropna().unique())
    grupos = [("asfalto" if "adoquin" not in t.loc[t["sitio"] == s, "superficie"].iloc[0] else "adoquin") for s in sitios]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 6.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.25, 0.585, 0.5, 0.27])
    bx = fig.add_axes([0.25, 0.165, 0.5, 0.22])

    # --- arriba: DIMMIT -----------------------------------------------------------------------------
    fig.text(0.03, 0.965, "DIMMIT · lo que medimos", fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.918, f"{cfg.get('toma', {}).get('titulo', '')} · score visual de cada recorrido", fontsize=9, color=MUTED, va="top")
    # bandas: bueno/satisf. (verde), regular (ámbar), malo/muy malo (rojo), alineadas con las clases del IDU
    for lo, hi, k in ((0, 56, "malo"), (56, 71, "regular"), (71, 100, "bueno")):
        ax.axvspan(lo, hi, color=EST_COL[k], alpha=0.16, zorder=0)
        ax.text((lo + hi) / 2, len(sitios) - 0.45, EST_NOM[k] + (" (≥ 71)" if k == "bueno" else ""), ha="center", va="bottom", fontsize=8.5, color=MUTED)
    y = np.arange(len(sitios))[::-1]
    labels = []
    for yi, s, g in zip(y, sitios, grupos):
        ts = t[t["sitio"] == s].sort_values("prueba")
        sc = ts["score_calidad"].to_numpy(float)
        off = np.linspace(-0.16, 0.16, len(sc)) if len(sc) > 1 else [0.0]
        for x, o, et in zip(sc, off, ts["etiqueta"]):
            ax.scatter(x, yi + o, s=95, color="#3d4549", edgecolor="white", linewidth=1.2, zorder=3, clip_on=False)
        labels.append(f"Sitio {int(s)} · {GRUPO_NOM[g]}\n{len(sc)} recorridos")
        et = ts["etiqueta"].mode().iloc[0]
        pmin, pmax = np.nanmin(ts["profundidad_max_cm"]), np.nanmax(ts["profundidad_max_cm"])
        prof = f"{pmin:.0f}–{pmax:.0f} cm" if np.isfinite(pmax) else "sin sensor"
        tr = ax.get_yaxis_transform()
        ax.text(1.05, yi + 0.13, ETIQ_NOM[et], transform=tr, fontsize=10.5, fontweight="bold", color=ETIQ_COL[et], va="center", clip_on=False)
        ax.text(1.05, yi - 0.17, f"bache {prof}", transform=tr, fontsize=9, color=INK, va="center", clip_on=False)
    ax.set_yticks(y, labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(sitios) - 0.45 + 0.35)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Score visual 0–100 (YOLOv10 + filtros de campo)", fontsize=9, color=INK, labelpad=4)
    ax.text(1.05, 1.0, "Etiqueta (cámara\n+ sensor)", transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom", clip_on=False)

    # --- abajo: IDU ---------------------------------------------------------------------------------
    ref = cfg["idu"]
    fecha = pd.to_datetime(idu["fecha_idu"]).max()
    fig.text(0.03, 0.49, "IDU · lo que reporta el inventario oficial", fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.443, f"Estado superficial de calzada ({fecha:%d-%m-%Y}) a ≤ {ref['radio_m']} m de la {ref['referencia']}", fontsize=9, color=MUTED, va="top")
    orden = ["bueno", "regular", "malo", "sin_clasificar"]
    blab = []
    for yi, g in zip(y, grupos):
        sub = idu[idu["grupo"] == g]
        pct = sub["estado_idu"].value_counts(normalize=True).reindex(orden).fillna(0) * 100
        left = 0.0
        for k in orden:
            if pct[k] > 0:
                bx.barh(yi, pct[k], left=left, height=0.55, color=EST_COL[k], edgecolor="white", linewidth=1)
                if pct[k] >= 9:
                    bx.text(left + pct[k] / 2, yi, f"{pct[k]:.0f} %", ha="center", va="center", fontsize=8.8, color="white" if k != "sin_clasificar" else INK, fontweight="bold")
                left += pct[k]
        blab.append(f"Calzadas en\n{GRUPO_NOM[g]} (n = {len(sub)})")
        dom = pct.idxmax()
        bx.text(1.05, yi, f"mayoría:\n{EST_NOM[dom].lower()}", transform=bx.get_yaxis_transform(), fontsize=9.5, color=EST_COL[dom] if dom != "sin_clasificar" else MUTED,
                va="center", fontweight="bold", clip_on=False)
    bx.set_yticks(y, blab, fontsize=9.5, color=INK)
    bx.set_xlim(0, 100)
    bx.set_ylim(-0.5, len(sitios) - 0.5)
    bx.set_xticks([0, 25, 50, 75, 100], [f"{k} %" for k in (0, 25, 50, 75, 100)])
    bx.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    bx.tick_params(axis="y", length=0)
    for a in (ax, bx):
        for s_ in ("top", "right", "left"):
            a.spines[s_].set_visible(False)
        a.spines["bottom"].set_color("#c3c6bd")
    # leyenda común de estados y nota
    hs = [plt.Rectangle((0, 0), 1, 1, color=EST_COL[k]) for k in orden]
    fig.legend(hs, [EST_NOM[k] for k in orden], loc="lower left", bbox_to_anchor=(0.23, 0.06), ncol=4, frameon=False, fontsize=8.8,
               handlelength=1.1, columnspacing=1.2)
    fig.text(0.03, 0.012, "Sin GPS en las fotos: comparación por zona y tipo de superficie, no tramo a tramo. El IDU no mide profundidad;\n"
             "DIMMIT la mide con el sensor y por eso etiqueta intervención aunque la superficie se vea bien.",
             fontsize=7.6, color=MUTED, va="bottom", linespacing=1.35)
    out = ensure_dir(OUT_CAMPO / "poster") / "validacion_idu.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
