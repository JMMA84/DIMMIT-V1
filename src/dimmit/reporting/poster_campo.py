"""Gráfica para póster (PNG 300 dpi): score por tramo antes y después del modo campo, con la etiqueta.

Lee data/campo/features/vias.parquet. Escribe reports/campo/poster/score_tramos_antes_despues.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimmit.campo import OUT_CAMPO
from dimmit.campo.scores import FEAT
from dimmit.utils.io import ensure_dir

INK, MUTED, GRID = "#15191c", "#6b7378", "#e3e5e0"
ANTES = "#9aa1a4"
ETIQ = {"prevencion": ("Prevención", "#2a78d6"), "mantenimiento": ("Mantenimiento", "#c98500"), "intervencion": ("Intervención", "#d03b3b")}
ESTADOS = [(0, 41, "muy malo"), (41, 56, "malo"), (56, 71, "regular"), (71, 86, "satisfactorio"), (86, 100, "bueno")]
BANDAS = ["#f4e6e3", "#f7ede4", "#f7f3e6", "#eef3ea", "#e7f0f7"]


def main():
    v = pd.read_parquet(FEAT / "vias.parquet")
    t = v[v["nivel"] == "tramo"].sort_values(["prueba", "tramo"], ascending=[False, False]).reset_index(drop=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    fig.patch.set_facecolor("white")
    for (lo, hi, name), col in zip(ESTADOS, BANDAS):
        ax.axvspan(lo, hi, color=col, zorder=0)
        ax.text((lo + hi) / 2, len(t) - 0.35, name, ha="center", va="bottom", color=MUTED, fontsize=13)
    y = np.arange(len(t))
    for i, r in t.iterrows():
        a, d = r["score_calidad_tau_rdd2020"], r["score_calidad"]
        nombre, col = ETIQ[r["etiqueta"]]
        ax.annotate("", xy=(d, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color="#c3c6bd", lw=2.2, mutation_scale=18), zorder=2)
        ax.scatter(a, i, s=170, color=ANTES, zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter(d, i, s=330, color=col, zorder=4, edgecolor="white", linewidth=2, clip_on=False)
        prof = r["profundidad_max_cm"]
        sub = f"bache de {prof:.0f} cm (sensor)" if pd.notna(prof) else ""
        ax.text(103, i + (0.13 if sub else 0), f"{d:.0f}  {nombre}", va="center", fontsize=14, color=col, fontweight="bold", clip_on=False)
        if sub:
            ax.text(103, i - 0.22, sub, va="center", fontsize=11.5, color=MUTED, clip_on=False)
    ax.set_yticks(y, [f"{p.replace('_', ' ')} · tramo {int(k)}" for p, k in zip(t["prueba"], t["tramo"])], fontsize=15, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(t) + 0.25)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", colors=MUTED, labelsize=13)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Score de calidad de la vía (0 = fallada · 100 = sin daño)", color=INK, fontsize=15, labelpad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c6bd")
    fig.text(0.02, 0.945, "De “todo en intervención” a un diagnóstico por tramo", fontsize=26, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.905, "DIMMIT V1 · 9 tramos de ~10 m en La Candelaria, Bogotá (5-oct-2026) · YOLOv10 + sensor ultrasónico",
             fontsize=15, color=MUTED, ha="left")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, color=ANTES, label="Antes: umbral de RDD2020, sin filtros")]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=14, color=c, label=f"Modo campo · {n}") for n, c in ETIQ.values()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=13.5, columnspacing=2.5)
    n_antes = int((t["etiqueta_tau_rdd2020"] == "intervencion").sum())
    n_ahora = int((t["etiqueta"] == "intervencion").sum())
    fig.text(0.02, 0.02, f"Tramos en intervención: {n_antes} → {n_ahora}. Score medio: {t['score_calidad_tau_rdd2020'].mean():.0f} → {t['score_calidad'].mean():.0f}. "
             "Profundidad = distancia − 8 cm con lecturas depuradas (solo la Prueba 3 tiene lecturas válidas).",
             fontsize=12.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.17, right=0.77, top=0.84, bottom=0.23)
    out = ensure_dir(OUT_CAMPO / "poster") / "score_tramos_antes_despues.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
