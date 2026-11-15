"""Figura 3 del póster (validación): arriba lo que mide DIMMIT en la toma vigente (score calibrado contra el IDU),
abajo el estado que publica el IDU para la MISMA calzada, en el mismo eje PCI 0-100. PNG 300 dpi, ancho:alto ~1,2.

Lee data/campo/features/vias.parquet, calibracion_idu.json y data/campo/idu/calzadas_estado.csv.
Escribe reports/campo/poster/validacion_idu.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimmit.campo import CAMPO, OUT_CAMPO, load_cfg
from dimmit.campo.scores import CALIB, FEAT
from dimmit.utils.io import ensure_dir

INK, MUTED = "#15191c", "#5f676c"
EST_COL = {"bueno": "#3a9a5b", "regular": "#e0a526", "malo": "#d03b3b"}
EST_NOM = {"bueno": "Bueno", "regular": "Regular", "malo": "Malo"}
ETIQ_COL = {"intervencion": "#d03b3b", "mantenimiento": "#c98500", "prevencion": "#2a78d6"}
ETIQ_NOM = {"intervencion": "Intervención", "mantenimiento": "Mantenimiento", "prevencion": "Prevención"}
SUP_NOM = {"flexible": "asfalto", "adoquin_arcilla": "adoquín de arcilla", "adoquin_concreto": "adoquín de concreto", "rigido": "concreto", "mixto": "mixto"}


def bands(ax, rg, n, labels=True):
    for k in ("malo", "regular", "bueno"):
        lo, hi = rg[k]
        ax.axvspan(lo, hi, color=EST_COL[k], alpha=0.15, zorder=0)
        if labels:
            ax.text((lo + hi) / 2, n - 0.42, f"{EST_NOM[k]}", ha="center", va="bottom", fontsize=8.5, color=MUTED)


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c6bd")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 56, 75, 90, 100])
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.tick_params(axis="y", length=0)


def main():
    cfg = load_cfg()
    v = pd.read_parquet(FEAT / "vias.parquet")
    t = v[(v["nivel"] == "tramo") & (v.get("conjunto", "evaluacion") == "evaluacion")]
    idu = pd.read_csv(CAMPO / "idu" / "calzadas_estado.csv").set_index("calcodigo")
    cal = json.loads(CALIB.read_text()) if CALIB.exists() else None
    r = cfg["calibracion"]["rangos_idu"]
    rg = {"malo": (0, r["regular"]), "regular": (r["regular"], r["bueno"]), "bueno": (r["bueno"], 100)}
    sitios = sorted(t["sitio"].dropna().unique())
    n = len(sitios)
    y = np.arange(n)[::-1]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 6.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.27, 0.6, 0.47, 0.255])
    bx = fig.add_axes([0.27, 0.19, 0.47, 0.2])

    # --- arriba: DIMMIT -------------------------------------------------------------------------------
    fig.text(0.03, 0.97, "DIMMIT · lo que medimos", fontsize=13.5, fontweight="bold", color=INK, va="top")
    m = f" · cámara calibrada con el IDU (m = {cal['multiplicador']:g})".replace(".", ",") if cal else ""
    fig.text(0.03, 0.925, f"{cfg.get('toma', {}).get('titulo', '')}{m}", fontsize=8.6, color=MUTED, va="top")
    bands(ax, rg, n)
    labels = []
    for yi, s in zip(y, sitios):
        ts = t[t["sitio"] == s].sort_values("prueba")
        sc = ts["score_calidad"].to_numpy(float)
        off = np.linspace(-0.17, 0.17, len(sc)) if len(sc) > 1 else [0.0]
        ax.scatter(sc, yi + np.asarray(off), s=90, color="#3d4549", edgecolor="white", linewidth=1.2, zorder=3, clip_on=False)
        sup = "asfalto" if (ts["superficie"] == "asfalto").all() else "adoquín"
        labels.append(f"Sitio {int(s)} · {sup}\n{len(sc)} recorridos")
        et = ts["etiqueta"].mode().iloc[0]
        pmin, pmax = np.nanmin(ts["profundidad_max_cm"]), np.nanmax(ts["profundidad_max_cm"])
        tr = ax.get_yaxis_transform()
        ax.text(1.04, yi + 0.13, ETIQ_NOM[et], transform=tr, fontsize=10, fontweight="bold", color=ETIQ_COL[et], va="center", clip_on=False)
        ax.text(1.04, yi - 0.17, f"bache {pmin:.0f}–{pmax:.0f} cm", transform=tr, fontsize=8.6, color=INK, va="center", clip_on=False)
    ax.text(1.04, 1.0, "Etiqueta (cámara\n+ sensor)", transform=ax.transAxes, fontsize=7.8, color=MUTED, va="bottom", clip_on=False)
    ax.set_yticks(y, labels, fontsize=9.3, color=INK)
    ax.set_ylim(-0.6, n - 0.42 + 0.33)
    style(ax)
    ax.set_xlabel("Score visual 0–100 (pseudo-PCI)", fontsize=8.8, color=INK, labelpad=3)

    # --- abajo: IDU, misma calzada ---------------------------------------------------------------------
    fecha = pd.to_datetime(idu["fecha_idu"]).max()
    fig.text(0.03, 0.495, "IDU · la misma calzada en el inventario oficial", fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.45, f"Estado superficial de calzada, evaluación del {fecha:%d-%m-%Y} (rango PCI de su estado)", fontsize=8.6, color=MUTED, va="top")
    bands(bx, rg, n, labels=False)
    blab = []
    for yi, s in zip(y, sitios):
        cod = int(cfg["sitios"][int(s)]["calcodigo_idu"])
        row = idu.loc[cod] if cod in idu.index else None
        est = row["estado_idu"] if row is not None else "sin_dato"
        sup = SUP_NOM.get(row["superficie"], row["superficie"]) if row is not None else ""
        blab.append(f"Calzada {cod}\n{sup}")
        if est in rg:
            lo, hi = rg[est]
            bx.barh(yi, hi - lo, left=lo, height=0.42, color=EST_COL[est], zorder=2)
            bx.text((lo + hi) / 2, yi, EST_NOM[est], ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3)
            hi_txt = hi if est == "bueno" else hi - 1
            bx.text(1.04, yi + 0.12, EST_NOM[est].lower(), transform=bx.get_yaxis_transform(), fontsize=9.5, color=EST_COL[est], fontweight="bold", va="center", clip_on=False)
            bx.text(1.04, yi - 0.18, f"PCI {lo:g}–{hi_txt:g}", transform=bx.get_yaxis_transform(), fontsize=8.6, color=INK, va="center", clip_on=False)
        else:
            bx.text(50, yi, f"sin estado publicado (código {int(row['estado_codigo']) if row is not None else '—'})", ha="center", va="center", fontsize=8.8,
                    color=MUTED, style="italic", zorder=3)
            bx.text(1.04, yi, "sin dato", transform=bx.get_yaxis_transform(), fontsize=9.5, color=MUTED, fontweight="bold", va="center", clip_on=False)
    bx.set_yticks(y, blab, fontsize=9.3, color=INK)
    bx.set_ylim(-0.55, n - 0.45)
    style(bx)
    bx.set_xlabel("PCI 0–100 · IDU: bueno ≥ 90 · regular 56–89 · malo < 56", fontsize=8.8, color=INK, labelpad=3)

    nota = "Ubicación: Carrera 5 detrás del CAI Rosario (declarada por el equipo; la toma del 5-oct con GPS cae en la misma calzada)."
    if cal:
        a0, a1 = cal["ajuste"]["acierto_sin_calibrar"], cal["ajuste"]["acierto_estado_idu"]
        nota += (f"\nCalibración: {cal['n_tramos']} tramos de la calzada 91032823 (5-oct y 15-nov) → {a0:.0%} → {a1:.0%} dentro del rango IDU. "
                 "Entre tomas no transfiere:\nel umbral depende de la cámara (las fotos de WhatsApp pierden confianza). Falta calibrar con 20–30 calzadas de estados distintos.")
    fig.text(0.03, 0.012, nota.replace("%", " %"), fontsize=6.9, color=MUTED, va="bottom", linespacing=1.4)
    out = ensure_dir(OUT_CAMPO / "poster") / "validacion_idu.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
