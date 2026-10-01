"""Reporte HTML autocontenido (español) con evaluación, monitoreo y resultados por segmento.

Gráficos en SVG en línea dibujados con los tokens de color del tema (claro/oscuro), sin
librerías externas. Escribe reports/dimmit_reporte_v1.html (documento completo) y
build/artifact/dimmit_reporte.html (fragmento para publicar como Artifact).
"""
import html
import json
import math

import numpy as np
import pandas as pd

from dimmit.labels.pseudo_pci import STATE_ORDER
from dimmit.utils.io import EVAL, EXTERNAL, FEATURES, OUTPUT, REPO_ROOT, REPORTS, ensure_dir

ESTADO_TXT = {"bueno": "Bueno", "satisfactorio": "Satisfactorio", "regular": "Regular", "malo": "Malo", "muy_malo": "Muy malo"}
ICON = {"OK": "✓", "ALERTA": "!", "FALLA": "✕", "INFO": "i"}

CSS = """
/* Tablero de ingeniería: resumen arriba, evidencia abajo; una columna que se abre a dos en pantallas anchas. */
:root {
  --bg: #f6f7f5; --surface: #fcfcfb; --ink: #14181a; --ink-2: #4c5356; --muted: #7d8487;
  --grid: #e1e3dd; --axis: #c3c6bd; --accent: #2a5d8f; --ring: rgba(20,24,26,.10);
  --st-bueno: #86b6ef; --st-satisfactorio: #5598e7; --st-regular: #2a78d6; --st-malo: #1c5cab; --st-muy_malo: #0d366b;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b; --info: #7d8487;
  --bar: #2a78d6; --bar-2: #9aa1a4; --dot: #c9ccc4;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1213; --surface: #171b1c; --ink: #f2f4f3; --ink-2: #c3c8c6; --muted: #8b9294;
  --grid: #262c2d; --axis: #384042; --accent: #7fb2e5; --ring: rgba(255,255,255,.10);
  --st-bueno: #184f95; --st-satisfactorio: #256abf; --st-regular: #3987e5; --st-malo: #86b6ef; --st-muy_malo: #cde2fb;
  --bar: #3987e5; --bar-2: #6b7275; --dot: #2c3335; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f1213; --surface: #171b1c; --ink: #f2f4f3; --ink-2: #c3c8c6; --muted: #8b9294;
  --grid: #262c2d; --axis: #384042; --accent: #7fb2e5; --ring: rgba(255,255,255,.10);
  --st-bueno: #184f95; --st-satisfactorio: #256abf; --st-regular: #3987e5; --st-malo: #86b6ef; --st-muy_malo: #cde2fb;
  --bar: #3987e5; --bar-2: #6b7275; --dot: #2c3335; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--f-body); margin: 0; }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; display: grid; gap: 34px; }
h1, h2, h3 { font-family: var(--f-display); text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: 2.1rem; font-weight: 600; }
h2 { font-size: 1.35rem; font-weight: 600; }
h3 { font-size: 1rem; font-weight: 600; color: var(--ink-2); }
p { margin: 0; max-width: 72ch; }
.eyebrow { font: 600 .72rem/1.2 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
header { display: grid; gap: 10px; border-bottom: 1px solid var(--grid); padding-bottom: 20px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; font: .82rem var(--f-mono); color: var(--ink-2); }
.note { background: var(--surface); border: 1px solid var(--ring); border-left: 3px solid var(--warning); padding: 12px 14px; border-radius: 6px; font-size: .92rem; }
section { display: grid; gap: 14px; min-width: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px; display: grid; gap: 4px; }
.tile .v { font: 600 1.7rem/1.1 var(--f-display); }
.tile .l { font-size: .8rem; color: var(--ink-2); }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 22px; }
.two > * { min-width: 0; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { font: 600 .72rem var(--f-mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); white-space: nowrap; }
td.n { text-align: right; font-family: var(--f-mono); white-space: nowrap; }
.pill { display: inline-flex; align-items: center; gap: 5px; font: 600 .72rem var(--f-mono); padding: 2px 8px; border-radius: 99px; border: 1px solid var(--ring); white-space: nowrap; }
.pill i { font-style: normal; display: inline-grid; place-items: center; width: 14px; height: 14px; border-radius: 50%; color: #fff; font-size: .62rem; }
.s-OK i { background: var(--good); } .s-ALERTA i { background: var(--warning); color: #14181a; } .s-FALLA i { background: var(--critical); } .s-INFO i { background: var(--info); }
.st { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.st b { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
svg text { fill: var(--ink-2); font: 11px var(--f-mono); }
svg .t-strong { fill: var(--ink); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: .8rem; color: var(--ink-2); }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.controls input, .controls select { font: inherit; padding: 6px 8px; border: 1px solid var(--axis); border-radius: 6px; background: var(--surface); color: var(--ink); }
.controls input:focus-visible, .controls select:focus-visible, th button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
th button { all: unset; cursor: pointer; } th button:hover { color: var(--ink); }
.tablebox { max-height: 640px; overflow: auto; }
.tablebox thead th { position: sticky; top: 0; background: var(--surface); z-index: 1; }
.desc { min-width: 260px; max-width: 420px; color: var(--ink-2); }
footer { font-size: .8rem; color: var(--muted); display: grid; gap: 6px; border-top: 1px solid var(--grid); padding-top: 16px; }
.hover:hover { opacity: .75; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def esc(x):
    return html.escape(str(x))


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer() and abs(v) > 50):
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.{nd}f}"


def pill(s):
    return f'<span class="pill s-{esc(s)}"><i>{ICON.get(s, "i")}</i>{esc(s)}</span>'


def state_chip(s):
    return f'<span class="st"><b style="background:var(--st-{s})"></b>{ESTADO_TXT.get(s, s)}</span>'


def bar_ci(rows, title, xlab, highlight=None, w=560):
    """Barras horizontales con IC (rows: nombre, valor, ic_inf, ic_sup)."""
    rows = [r for r in rows if r[1] is not None and math.isfinite(r[1])]
    h = 26 * len(rows) + 46
    left, right = 190, 24
    mx = max(max(r[3] if r[3] is not None and math.isfinite(r[3]) else r[1] for r in rows), 1e-9) * 1.08
    sx = lambda v: left + (w - left - right) * v / mx  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(title)}">']
    for t in np.linspace(0, mx, 5)[:-1] if mx > 0 else []:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="10" x2="{x:.1f}" y2="{h - 30}" stroke="var(--grid)"/><text x="{x:.1f}" y="{h - 14}" text-anchor="middle">{t:.1f}</text>')
    out.append(f'<text x="{w - right}" y="{h - 2}" text-anchor="end">{esc(xlab)}</text>')
    for i, (name, v, lo, hi) in enumerate(rows):
        y = 14 + 26 * i
        fill = "var(--bar)" if name == highlight else "var(--bar-2)"
        x0, x1 = sx(0), sx(v)
        out.append(f'<g class="hover"><title>{esc(name)}: {v:.2f}' + (f" (IC95 {lo:.2f}–{hi:.2f})" if lo is not None and math.isfinite(lo) else "") + "</title>")
        out.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end" class="{"t-strong" if name == highlight else ""}">{esc(name)}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{y + 3}" width="{max(x1 - x0, 1):.1f}" height="14" rx="3" fill="{fill}"/>')
        if lo is not None and math.isfinite(lo):
            out.append(f'<line x1="{sx(lo):.1f}" y1="{y + 10}" x2="{sx(hi):.1f}" y2="{y + 10}" stroke="var(--ink)" stroke-width="1.5"/>')
        out.append(f'<text x="{x1 + 6:.1f}" y="{y + 14}">{v:.2f}</text></g>')
    out.append("</svg>")
    return "".join(out)


def confusion_svg(cm: pd.DataFrame, w=420):
    labels = STATE_ORDER
    n = len(labels)
    cell = 52
    left, top = 104, 30
    hgt = top + n * cell + 44
    mx = max(cm.values.max(), 1)
    out = [f'<svg viewBox="0 0 {left + n * cell + 10} {hgt}" width="100%" style="max-width:{w}px" role="img" aria-label="Matriz de confusión de estados">']
    for j, lab in enumerate(labels):
        out.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 8}" text-anchor="middle">{ESTADO_TXT[lab][:9]}</text>')
    for i, lab in enumerate(labels):
        out.append(f'<text x="{left - 6}" y="{top + i * cell + cell / 2 + 4}" text-anchor="end">{ESTADO_TXT[lab]}</text>')
        row_total = max(cm.loc[lab].sum(), 1)
        for j, lab2 in enumerate(labels):
            v = int(cm.loc[lab, lab2])
            a = 0.08 + 0.92 * v / mx
            out.append(
                f'<g class="hover"><title>real {ESTADO_TXT[lab]} → predicho {ESTADO_TXT[lab2]}: {v} ({100 * v / row_total:.0f} % de la fila)</title>'
                f'<rect x="{left + j * cell + 1}" y="{top + i * cell + 1}" width="{cell - 2}" height="{cell - 2}" rx="3" fill="var(--bar)" fill-opacity="{a:.2f}"/>'
                f'<text x="{left + j * cell + cell / 2}" y="{top + i * cell + cell / 2 + 4}" text-anchor="middle" class="{"t-strong" if i == j else ""}">{v}</text></g>'
            )
    out.append(f'<text x="{left + n * cell / 2}" y="{hgt - 8}" text-anchor="middle">predicho por la red →  (filas: estado de referencia)</text></svg>')
    return "".join(out)


def histogram_svg(seg: pd.DataFrame, w=560, h=200):
    bins = np.arange(0, 105, 5)
    left, bottom = 34, 34
    counts = {s: np.histogram(seg.loc[seg["estado"] == s, "score_condicion"], bins)[0] for s in STATE_ORDER}
    tot = sum(counts.values())
    mx = max(tot.max(), 1)
    bw = (w - left - 10) / (len(bins) - 1)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Distribución del índice de condición">']
    for t in np.linspace(0, mx, 4):
        y = h - bottom - (h - bottom - 10) * t / mx
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w - 10}" y2="{y:.1f}" stroke="var(--grid)"/><text x="{left - 4}" y="{y + 4:.1f}" text-anchor="end">{int(t)}</text>')
    for k in range(len(bins) - 1):
        y0 = h - bottom
        for s in STATE_ORDER:
            c = counts[s][k]
            if not c:
                continue
            hh = (h - bottom - 10) * c / mx
            out.append(f'<rect x="{left + k * bw + 1:.1f}" y="{y0 - hh:.1f}" width="{bw - 2:.1f}" height="{max(hh - 1, 0.5):.1f}" fill="var(--st-{s})"><title>{bins[k]}–{bins[k + 1]}: {c} segmentos {ESTADO_TXT[s]}</title></rect>')
            y0 -= hh
    for t in range(0, 101, 20):
        out.append(f'<text x="{left + t / 5 * bw:.1f}" y="{h - bottom + 16}" text-anchor="middle">{t}</text>')
    out.append(f'<text x="{w - 10}" y="{h - 4}" text-anchor="end">índice de condición vial (0–100)</text></svg>')
    return "".join(out)


def map_svg(seg: pd.DataFrame, backdrop: pd.DataFrame, w=560, h=640):
    lat0, lat1 = 4.46, 4.84
    lon0, lon1 = -74.23, -74.0
    kx = math.cos(math.radians(4.65))
    sx = lambda lon: 10 + (w - 20) * (lon - lon0) / (lon1 - lon0)  # noqa: E731
    sy = lambda lat: 10 + (h - 20) * (lat1 - lat) / (lat1 - lat0)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" aria-label="Mapa de segmentos sobre calzadas de Bogotá">']
    out.append("".join(f'<circle cx="{sx(lo):.1f}" cy="{sy(la):.1f}" r="0.9" fill="var(--dot)"/>' for la, lo in zip(backdrop["lat"], backdrop["lon"])))
    for _, r in seg.sort_values("score_condicion", ascending=False).iterrows():
        out.append(
            f'<circle class="hover" cx="{sx(r["lon"]):.1f}" cy="{sy(r["lat"]):.1f}" r="4" fill="var(--st-{r["estado"]})" stroke="var(--surface)" stroke-width="1.2">'
            f'<title>{esc(r["segmento_id"])} · ICV {r["score_condicion"]:.0f} ({ESTADO_TXT[r["estado"]]}) · prioridad {r["nivel_prioridad"]} {r["score_prioridad"]:.0f}</title></circle>'
        )
    _ = kx
    out.append(f'<text x="{w - 12}" y="{h - 12}" text-anchor="end">puntos grises: calzadas UMV (muestra)</text></svg>')
    return "".join(out)


def psi_svg(drift: pd.DataFrame, w=560):
    agg = drift.groupby("lote")["psi"].max().sort_values()
    rows = [(k, float(v), None, None) for k, v in agg.items()]
    svg = bar_ci(rows, "PSI máximo por lote", "PSI máximo (0.10 alerta · 0.25 falla)", w=w)
    return svg


def table(df, cols, nd=3):
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                tds.append("<td></td>")
            elif c == "estado" and v in ICON:
                tds.append(f"<td>{pill(v)}</td>")
            elif isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool):
                tds.append(f'<td class="n">{fmt(float(v), nd)}</td>')
            else:
                tds.append(f"<td>{esc('' if v is None or (isinstance(v, float) and np.isnan(v)) else v)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def build():
    met = pd.read_csv(EVAL / "metricas_evaluacion.csv")
    seg = pd.read_csv(OUTPUT / "segmentos_scores.csv")
    conf = pd.read_csv(EVAL / "matrices_confusion.csv")
    drift = pd.read_csv(EVAL / "drift_psi.csv")
    hist = pd.read_csv(EVAL / "historial_corridas.csv")
    run = seg["run_id"].iloc[0]
    version = seg["version_modelo"].iloc[0]
    det = met.loc[met["etapa"] == "detector", "modelo"].iloc[0] if (met["etapa"] == "detector").any() else "—"
    provisional = "cpu" in str(version) or "cpu" in str(det)
    g = lambda e, m, k, obj=None, sub="todos": met.query("etapa==@e and modelo==@m and metrica==@k and subgrupo==@sub" + (" and objetivo==@obj" if obj else ""))["valor"]  # noqa: E731
    val = lambda *a, **k: (float(g(*a, **k).iloc[0]) if len(g(*a, **k)) else float("nan"))  # noqa: E731
    test = seg[seg["conjunto"] == "test"]
    counts = met["estado"].value_counts()

    tiles = [
        (f"{len(seg)}", f"segmentos puntuados ({len(test)} test · {len(seg) - len(test)} despliegue)"),
        (fmt(val("fusion", "principal", "mae", "icv"), 1), "error absoluto medio del índice (test, puntos)"),
        (fmt(val("fusion", "principal", "kappa_ponderada_cuadratica", "estado"), 2), "kappa ponderada del estado (test)"),
        (fmt(val("fusion", "principal", "recall_malo_o_peor", "estado"), 2), "recall de vías malas o muy malas"),
        (fmt(val("detector", det, "map50"), 3), "mAP50 del detector YOLOv10 (test)"),
        (f"{counts.get('OK', 0)} · {counts.get('ALERTA', 0)} · {counts.get('FALLA', 0)}", "métricas OK · ALERTA · FALLA"),
    ]
    tiles_html = "".join(f'<div class="tile"><span class="v">{esc(v)}</span><span class="l">{esc(l)}</span></div>' for v, l in tiles)

    health = met[met["umbral_ok"].notna()].copy()
    health["IC 95 %"] = [f"{a:.3f}–{b:.3f}" if pd.notna(a) else "" for a, b in zip(health["ic95_inf"], health["ic95_sup"])]
    health = health.rename(columns={"etapa": "área", "metrica": "métrica", "subgrupo": "subgrupo", "valor": "valor", "umbral_ok": "umbral OK"})
    health_html = table(health, ["área", "modelo", "métrica", "objetivo", "subgrupo", "conjunto", "valor", "IC 95 %", "umbral OK", "estado"])

    comp = met[(met["metrica"] == "mae") & (met["objetivo"] == "icv") & (met["etapa"].isin(["fusion", "linea_base", "ablacion"]))]
    rows = [(r["modelo"], r["valor"], r["ic95_inf"] if pd.notna(r["ic95_inf"]) else None, r["ic95_sup"] if pd.notna(r["ic95_sup"]) else None) for _, r in comp.sort_values("valor").iterrows()]
    comp_svg = bar_ci(rows, "MAE del índice por modelo", "MAE del índice de condición (puntos, menor es mejor)", highlight="principal")

    cm = conf[conf["modelo"] == "principal"].pivot_table(index="real", columns="predicho", values="n", aggfunc="sum").reindex(index=STATE_ORDER, columns=STATE_ORDER).fillna(0)
    per_class = met[(met["etapa"] == "detector") & (met["nivel"] == "caja") & met["subgrupo"].isin(["D00", "D10", "D20", "D40"])].pivot_table(index="subgrupo", columns="metrica", values="valor").reset_index()
    per_class = per_class.rename(columns={"subgrupo": "clase"})
    det_cols = [c for c in ["clase", "precision", "recall", "f1", "ap50", "ap50_95"] if c in per_class]
    country = met[(met["etapa"] == "detector") & (met["metrica"] == "map50")][["subgrupo", "valor"]].rename(columns={"subgrupo": "país", "valor": "mAP50"})

    controls = met[met["etapa"].isin(["control", "sensores", "importancia", "lenguaje", "vision"])][["etapa", "modelo", "metrica", "objetivo", "conjunto", "subgrupo", "valor", "estado", "notas"]]

    umv = pd.read_parquet(EXTERNAL / "umv_calzadas.parquet", columns=["lat", "lon"]).sample(6000, random_state=1)
    legend = "".join(f'<span class="st"><b style="background:var(--st-{s})"></b>{ESTADO_TXT[s]}</span>' for s in reversed(STATE_ORDER))

    st_cols = ["ranking_prioridad", "segmento_id", "conjunto", "pais_imagenes", "score_condicion", "estado", "score_visual_pci", "iri_m_km", "score_baches", "score_piel_cocodrilo",
               "score_prioridad", "nivel_prioridad", "accion_recomendada", "descripcion"]
    data = seg[st_cols].copy()
    data["estado_txt"] = data["estado"].map(ESTADO_TXT)
    data_json = json.dumps(data.to_dict(orient="records"), ensure_ascii=False).replace("</", "<\\/")

    hist_html = table(hist.tail(10), [c for c in hist.columns if c != "fecha"])

    body = f"""
<title>DIMMIT Condición Vial</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <span class="eyebrow">DIMMIT v1 · Evaluación y resultados</span>
  <h1>Condición de la malla vial: YOLOv10 + sensores + contexto de Bogotá</h1>
  <div class="meta"><span>corrida {esc(run)}</span><span>{esc(version)}</span><span>RDD2020 completo: 26 336 imágenes</span></div>
  {'<p class="note"><b>Corrida provisional.</b> El detector es la versión de humo entrenada en CPU (2 épocas); el entrenamiento completo en GPU reemplaza estos números. Las métricas de la red y de las líneas base son comparables entre sí.</p>' if provisional else ''}
</header>

<section aria-labelledby="s-resumen">
  <h2 id="s-resumen">Resumen</h2>
  <div class="tiles">{tiles_html}</div>
  <p>El <b>índice de condición vial (ICV)</b> combina un índice visual tipo PCI (ASTM D6433 simplificado, 65 %) calculado
  sobre los daños detectados y un índice de rodadura a partir del IRI estimado con el acelerómetro (35 %). El estado sale del
  ICV: muy malo &lt; 41 ≤ malo &lt; 56 ≤ regular &lt; 71 ≤ satisfactorio &lt; 86 ≤ bueno.</p>
</section>

<section aria-labelledby="s-salud">
  <h2 id="s-salud">Salud del modelo</h2>
  <p>Cada métrica con umbral se compara con el objetivo definido en <code>configs/evaluation.yaml</code>. Los intervalos son bootstrap por bloques (los segmentos de un bloque no son independientes).</p>
  <div class="card">{health_html}</div>
</section>

<section aria-labelledby="s-modelos">
  <h2 id="s-modelos">Red de fusión frente a líneas base y ablaciones</h2>
  <div class="two">
    <div class="card"><h3>Error del índice en test (IC 95 %)</h3>{comp_svg}
      <p style="font-size:.85rem;color:var(--ink-2)">B0 = score v0 por reglas · B1 = regla física calibrada · B2 = gradient boosting · oráculo = densidades anotadas en vez de detecciones (techo atribuible al detector) · etiquetas permutadas = azar.</p></div>
    <div class="card"><h3>Estados: referencia frente a la red (test)</h3>{confusion_svg(cm)}</div>
  </div>
  <div class="card"><h3>Distribución del índice por estado (test + despliegue)</h3>{histogram_svg(seg)}<div class="legend">{legend}</div></div>
</section>

<section aria-labelledby="s-detector">
  <h2 id="s-detector">Detector YOLOv10</h2>
  <div class="two">
    <div class="card"><h3>Por clase (test)</h3>{table(per_class, det_cols) if len(per_class) else '<p>Sin métricas por clase en esta corrida.</p>'}</div>
    <div class="card"><h3>Por país (test)</h3>{table(country, ["país", "mAP50"])}</div>
  </div>
</section>

<section aria-labelledby="s-controles">
  <h2 id="s-controles">Controles, sensores, importancia y lenguaje</h2>
  <p>Los controles prueban que la evaluación es honesta: el contexto de Bogotá no debe predecir la condición de imágenes de otros
  países, unos sensores sin relación con el daño no deben predecir el índice visual, y etiquetas barajadas deben rendir como azar.
  La validación con datos reales (Zenodo 4386256) compara el estimador de rugosidad con perfiles láser.</p>
  <div class="card tablebox">{table(controls, ["etapa", "modelo", "metrica", "objetivo", "conjunto", "subgrupo", "valor", "estado", "notas"])}</div>
</section>

<section aria-labelledby="s-deriva">
  <h2 id="s-deriva">Monitoreo de deriva</h2>
  <div class="two">
    <div class="card"><h3>PSI máximo por lote</h3>{psi_svg(drift)}<p style="font-size:.85rem;color:var(--ink-2)">test1/test2 son control negativo (mismas secuencias que el entrenamiento); Chequia sola y el lote degradado son controles positivos que deben alertar.</p></div>
    <div class="card"><h3>Historial de corridas</h3>{hist_html}</div>
  </div>
</section>

<section aria-labelledby="s-mapa">
  <h2 id="s-mapa">Segmentos sobre la malla vial de Bogotá</h2>
  <p>Cada segmento se ubicó sobre una calzada real de la UMV para cruzar contexto (colegios, salud, siniestros, lluvia). Las imágenes
  no son de esas calles, así que el mapa muestra la mecánica del sistema, no el estado real de Bogotá.</p>
  <div class="two"><div class="card">{map_svg(seg, umv)}<div class="legend">{legend}</div></div>
  <div class="card"><h3>Prioridad de intervención</h3>{table(seg.groupby("nivel_prioridad").agg(segmentos=("segmento_id", "count"), icv_medio=("score_condicion", "mean"), cerca_colegio=("dist_colegio_m", lambda s: int((s <= 300).sum()))).reset_index(), ["nivel_prioridad", "segmentos", "icv_medio", "cerca_colegio"], 1)}
  <p style="font-size:.85rem;color:var(--ink-2)">Prioridad = 0.6 × necesidad (100 − ICV) + 0.4 × importancia (dimensiones social y técnica de la UMV, siniestralidad) + bonificaciones por colegio a 300 m, hospital a 1 km y lluvia alta.</p></div></div>
</section>

<section aria-labelledby="s-tabla">
  <h2 id="s-tabla">Segmentos</h2>
  <div class="controls">
    <label for="q">Buscar</label><input id="q" type="search" placeholder="segmento, acción, texto…">
    <label for="fe">Estado</label><select id="fe"><option value="">todos</option>{''.join(f'<option value="{s}">{ESTADO_TXT[s]}</option>' for s in reversed(STATE_ORDER))}</select>
    <label for="fc">Conjunto</label><select id="fc"><option value="">todos</option><option>test</option><option>despliegue</option></select>
    <span id="cnt" style="color:var(--muted);font-size:.85rem"></span>
  </div>
  <div class="card tablebox"><table id="tbl"><thead><tr>
    <th><button data-k="ranking_prioridad">#</button></th><th><button data-k="segmento_id">segmento</button></th><th><button data-k="conjunto">conjunto</button></th>
    <th><button data-k="score_condicion">ICV</button></th><th><button data-k="estado_txt">estado</button></th><th><button data-k="score_visual_pci">PCI visual</button></th>
    <th><button data-k="iri_m_km">IRI m/km</button></th><th><button data-k="score_baches">baches</button></th><th><button data-k="score_prioridad">prioridad</button></th>
    <th>acción</th><th>descripción</th></tr></thead><tbody></tbody></table></div>
</section>

<footer>
  <span>Datos: RDD2020 (Arya et al., CC BY-NC 3.0) · Bogotá datos abiertos: UMV, IDU, SED, SDS, SDM (CC BY 4.0 / CC BY-SA 4.0) · Open-Meteo · Zenodo 4386256 (CC0).</span>
  <span>Sensores simulados (Arduino + MPU-6050 + GPS aún no instalados): las métricas que dependen de ellos están condicionadas al simulador. El titular honesto es la parte visual contra anotaciones humanas.</span>
</footer>
</div>
<script>
const DATA = {data_json};
const tb = document.querySelector('#tbl tbody');
let key = 'ranking_prioridad', asc = true;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
function render() {{
  const q = document.getElementById('q').value.toLowerCase();
  const fe = document.getElementById('fe').value, fc = document.getElementById('fc').value;
  let rows = DATA.filter(r => (!fe || r.estado === fe) && (!fc || r.conjunto === fc) &&
    (!q || (r.segmento_id + ' ' + r.accion_recomendada + ' ' + r.descripcion).toLowerCase().includes(q)));
  rows.sort((a, b) => (a[key] > b[key] ? 1 : a[key] < b[key] ? -1 : 0) * (asc ? 1 : -1));
  document.getElementById('cnt').textContent = rows.length + ' segmentos';
  tb.innerHTML = rows.slice(0, 400).map(r => `<tr><td class="n">${{r.ranking_prioridad}}</td><td>${{esc(r.segmento_id)}}</td><td>${{r.conjunto}}</td>
    <td class="n">${{r.score_condicion.toFixed(1)}}</td><td><span class="st"><b style="background:var(--st-${{r.estado}})"></b>${{r.estado_txt}}</span></td>
    <td class="n">${{r.score_visual_pci.toFixed(1)}}</td><td class="n">${{r.iri_m_km.toFixed(2)}}</td><td class="n">${{r.score_baches.toFixed(1)}}</td>
    <td class="n">${{r.score_prioridad.toFixed(1)}} · ${{r.nivel_prioridad}}</td><td>${{esc(r.accion_recomendada).replaceAll('_',' ')}}</td><td class="desc">${{esc(r.descripcion)}}</td></tr>`).join('');
}}
document.querySelectorAll('th button').forEach(b => b.addEventListener('click', () => {{ const k = b.dataset.k; asc = key === k ? !asc : true; key = k; render(); }}));
['q','fe','fc'].forEach(id => document.getElementById(id).addEventListener('input', render));
render();
</script>
"""
    ensure_dir(REPORTS)
    full = '<!doctype html>\n<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>' + body + "</body></html>"
    (REPORTS / "dimmit_reporte_v1.html").write_text(full, encoding="utf-8")
    art = ensure_dir(REPO_ROOT / "build" / "artifact")
    (art / "dimmit_reporte.html").write_text(body, encoding="utf-8")
    print(f"[ok] reporte -> {REPORTS / 'dimmit_reporte_v1.html'} ({len(full) / 1e6:.2f} MB)")


if __name__ == "__main__":
    _ = FEATURES
    build()
