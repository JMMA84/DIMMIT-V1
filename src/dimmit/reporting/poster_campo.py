"""Gráfica para póster (PNG 300 dpi) con los resultados de la corrida vigente: por recorrido, el score
visual (modo campo) y la profundidad máxima del sensor, coloreados por la etiqueta final.

Lee data/campo/features/vias.parquet y configs/campo.yaml (toma).
Escribe reports/campo/poster/resultados_toma_<fecha>.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimmit.campo import OUT_CAMPO, load_cfg
from dimmit.campo.scores import FEAT
from dimmit.utils.io import ensure_dir

INK, MUTED = "#15191c", "#6b7378"
ETIQ = {"prevencion": ("Prevención", "#2a78d6"), "mantenimiento": ("Mantenimiento", "#c98500"), "intervencion": ("Intervención", "#d03b3b")}
ESTADOS = [(0, 41, "muy malo"), (41, 56, "malo"), (56, 71, "regular"), (71, 86, "satisf."), (86, 100, "bueno")]
BANDAS = ["#f4e6e3", "#f7ede4", "#f7f3e6", "#eef3ea", "#e7f0f7"]


def _row_label(r):
    base = r["prueba"].replace("_", " ")
    if pd.notna(r.get("sitio")):
        base += f" · sitio {int(r['sitio'])}"
    if r.get("superficie") and r["superficie"] != "asfalto":
        base += f" ({r['superficie'].replace('adoquin', 'adoquín')})"
    return base


def main():
    cfg = load_cfg()
    et_cfg = cfg["etiqueta"]
    v = pd.read_parquet(FEAT / "vias.parquet")
    t = v[v["nivel"] == "tramo"].copy()
    t["_s"] = t["sitio"].fillna(0) if "sitio" in t else 0
    t = t.sort_values(["_s", "prueba", "tramo"], ascending=False).reset_index(drop=True)
    n = len(t)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 9), dpi=300, sharey=True, gridspec_kw={"width_ratios": [2.1, 1], "wspace": 0.06})
    fig.patch.set_facecolor("white")
    y = np.arange(n)
    # --- panel 1: score visual de la corrida (modo campo) ------------------------------------------
    for (lo, hi, name), col in zip(ESTADOS, BANDAS):
        ax.axvspan(lo, hi, color=col, zorder=0)
        ax.text((lo + hi) / 2, n - 0.45, name, ha="center", va="bottom", color=MUTED, fontsize=12.5)
    for i, r in t.iterrows():
        d = r["score_calidad"]
        col = ETIQ[r["etiqueta"]][1]
        ax.plot([0, d], [i, i], color=col, lw=3, alpha=0.55, zorder=2, solid_capstyle="butt")
        ax.scatter(d, i, s=330, color=col, zorder=4, edgecolor="white", linewidth=2, clip_on=False)
        ax.text(d - 2.4, i + 0.3, f"{d:.0f} · {r['estado'].replace('_', ' ')}", ha="right", va="bottom", fontsize=13, color=col, fontweight="bold")
    ax.set_yticks(y, [_row_label(r) for _, r in t.iterrows()], fontsize=14.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n + 0.15)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", colors=MUTED, labelsize=12.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Score visual de la vía (YOLOv10 + filtros de campo)\n0 = fallada · 100 = sin daño", color=INK, fontsize=14, labelpad=10)
    ax.set_title("Lo que ve la cámara", loc="left", fontsize=17, fontweight="bold", color=INK, pad=12)
    # --- panel 2: profundidad del sensor ---------------------------------------------------------
    pmax = t["profundidad_max_cm"].to_numpy(float)
    xmax = max(10.0, np.nanmax(pmax) if np.isfinite(pmax).any() else 10.0) * 1.22
    bx.axvspan(0, et_cfg["mantenimiento"]["profundidad_cm"], color="#eef3ea", zorder=0)
    bx.axvspan(et_cfg["mantenimiento"]["profundidad_cm"], et_cfg["intervencion"]["profundidad_cm"], color="#f7f3e6", zorder=0)
    bx.axvspan(et_cfg["intervencion"]["profundidad_cm"], xmax, color="#f4e6e3", zorder=0)
    bx.text(et_cfg["intervencion"]["profundidad_cm"] + 0.3, n - 0.45, f"≥ {et_cfg['intervencion']['profundidad_cm']:.0f} cm: intervención", va="bottom", fontsize=12, color=MUTED)
    for i, r in t.iterrows():
        p = r["profundidad_max_cm"]
        col = ETIQ[r["etiqueta"]][1]
        if np.isfinite(p):
            bx.barh(i, p, height=0.5, color=col, zorder=2)
            pm = r.get("profundidad_media_cm", np.nan)
            bx.text(p + 0.4, i, f"{p:.0f} cm".replace(".", ","), va="center", fontsize=13.5, color=col, fontweight="bold")
        else:
            bx.text(0.4, i, "sin lectura válida", va="center", fontsize=12, color=MUTED)
    bx.set_xlim(0, xmax)
    bx.tick_params(axis="x", colors=MUTED, labelsize=12.5)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Profundidad máxima del bache (cm)\nsensor ultrasónico: distancia − 8 cm", color=INK, fontsize=14, labelpad=10)
    bx.set_title("Lo que mide el sensor", loc="left", fontsize=17, fontweight="bold", color=INK, pad=12)
    for a_ in (ax, bx):
        for s in ("top", "right", "left"):
            a_.spines[s].set_visible(False)
        a_.spines["bottom"].set_color("#c3c6bd")
    # --- textos ------------------------------------------------------------------------------------
    n_int = int((t["etiqueta"] == "intervencion").sum())
    titulo = f"{n_int} de {n} recorridos requieren intervención" if n_int else "Ningún recorrido requiere intervención"
    fig.text(0.02, 0.975, titulo, fontsize=26, fontweight="bold", color=INK, ha="left", va="top")
    rango = f"{np.nanmin(pmax):.0f} a {np.nanmax(pmax):.0f} cm" if np.isfinite(pmax).any() else "sin lecturas válidas"
    fig.text(0.02, 0.918, f"Con los filtros de campo, la cámara ve la superficie en buen estado; el sensor mide depresiones de {rango}.",
             fontsize=15.5, color=INK, ha="left", va="top")
    toma = cfg.get("toma", {}).get("titulo", "")
    fig.text(0.02, 0.878, f"DIMMIT V1 · {toma} · YOLOv10 (RDD2020) + sensor ultrasónico en bastón", fontsize=13, color=MUTED, ha="left", va="top")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=13, color=c, label=f"Etiqueta: {nm}") for k, (nm, c) in ETIQ.items() if (t["etiqueta"] == k).any()]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.095), ncol=len(handles), frameon=False, fontsize=13)
    run = pd.read_csv(OUT_CAMPO / "vias_clasificacion.csv", usecols=["run_id"])["run_id"].iloc[0] if (OUT_CAMPO / "vias_clasificacion.csv").exists() else ""
    nota = (f"Corrida {run} · {int(t['n_fotogramas'].sum())} fotos. Score = pseudo-PCI (ASTM D6433) con las cajas que pasan los filtros de campo\n"
            "(confianza, geometría, superficie, persistencia y consenso multi-escala); en adoquín solo cuentan los baches.\n"
            f"Etiqueta: estado malo, bache ≥ {et_cfg['intervencion']['profundidad_cm']:.0f} cm o bache visto con confianza ≥ {et_cfg['intervencion']['bache_conf_min']:.1f} → intervención; "
            f"estado regular o bache ≥ {et_cfg['mantenimiento']['profundidad_cm']:.0f} cm → mantenimiento.").replace("0.5", "0,5")
    fig.text(0.02, 0.012, nota, fontsize=11, color=MUTED, ha="left", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.27)
    fecha = cfg.get("toma", {}).get("fecha", "")
    out = ensure_dir(OUT_CAMPO / "poster") / f"resultados_toma_{fecha}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
