"""Descripciones en español por segmento: plantilla determinista o Claude (Message Batches).

El modelo de lenguaje solo redacta a partir de los hechos calculados (JSON por segmento); no
calcula el score. Toda salida de Claude se valida:
  - esquema JSON (salida estructurada),
  - fidelidad numérica: cada número del texto debe existir en los hechos (coma o punto decimal),
  - el estado mencionado y la acción deben coincidir con los hechos y la tabla de reglas.
Si algo falla (o no hay clave de API) se usa la plantilla y se marca fuente_descripcion.
"""
import json
import os
import re
import time

from dimmit.geo.priority import ACTIONS, ALLOWED_ACTIONS, URGENCY, rule_action
from dimmit.utils.io import CLASS_SLUGS, load_yaml

ESTADO_TXT = {"bueno": "bueno", "satisfactorio": "satisfactorio", "regular": "regular", "malo": "malo", "muy_malo": "muy malo"}
DANO_TXT = {"D00": "grietas longitudinales", "D10": "grietas transversales", "D20": "piel de cocodrilo", "D40": "baches"}
ACCION_TXT = {
    "mantenimiento_rutinario": "mantenimiento rutinario e inspección",
    "sellado_de_fisuras": "sellado de fisuras",
    "parcheo": "parcheo superficial",
    "bacheo": "bacheo localizado",
    "rehabilitacion": "rehabilitación de la carpeta",
    "reconstruccion": "reconstrucción de la estructura",
}
URG_TXT = {"inmediata_30_dias": "inmediata (≤ 30 días)", "programada_6_meses": "programada (≤ 6 meses)", "rutinaria_12_meses": "rutinaria (≤ 12 meses)"}

SYSTEM_PROMPT = """Eres un ingeniero de pavimentos que redacta fichas de diagnóstico vial para la ciudad de Bogotá.
Recibirás los HECHOS de un segmento de vía en JSON: índice de condición (0-100, mayor es mejor), estado, \
sub-scores por tipo de daño, rugosidad (IRI en m/km), contexto (colegios, hospitales, siniestros, lluvia) y \
prioridad. Redacta en español claro para un equipo de mantenimiento:
- "descripcion": máximo {max_words} palabras. Usa SOLO números que aparezcan en los hechos (puedes redondearlos \
a entero). No inventes daños, distancias ni causas.
- "accion_recomendada": una de las acciones permitidas para el estado (campo acciones_permitidas).
- "urgencia": coherente con el nivel de prioridad (campo urgencia_sugerida).
- "justificacion": una frase que conecte el daño dominante, la rugosidad y el contexto con la acción.
Menciona el estado exactamente como aparece en los hechos."""

SCHEMA = {
    "type": "object",
    "properties": {
        "descripcion": {"type": "string"},
        "accion_recomendada": {"type": "string", "enum": ACTIONS},
        "urgencia": {"type": "string", "enum": list(URG_TXT)},
        "justificacion": {"type": "string"},
    },
    "required": ["descripcion", "accion_recomendada", "urgencia", "justificacion"],
    "additionalProperties": False,
}


def dominant_damage(row):
    subs = {c: row.get(f"score_{slug}", 100) for c, slug in CLASS_SLUGS.items()}
    c, v = min(subs.items(), key=lambda kv: kv[1])
    return c if v < 95 else None


def facts(row) -> dict:
    """Hechos por segmento (lo único que ve el modelo de lenguaje)."""
    dom = dominant_damage(row)
    f = {
        "segmento": row["segmento_id"],
        "longitud_m": round(float(row["longitud_m"])),
        "indice_condicion": round(float(row["score_condicion"]), 1),
        "intervalo_90": [round(float(row["score_condicion_p05"])), round(float(row["score_condicion_p95"]))],
        "estado": ESTADO_TXT[row["estado"]],
        "sub_scores": {DANO_TXT[c]: round(float(row[f"score_{s}"]), 1) for c, s in CLASS_SLUGS.items()},
        "detecciones": {DANO_TXT[c]: int(row[f"n_{s}"]) for c, s in CLASS_SLUGS.items()},
        "dano_dominante": DANO_TXT.get(dom, "ninguno"),
        "iri_m_km": round(float(row["iri_m_km"]), 1),
        "score_rugosidad": round(float(row["score_rugosidad"]), 1),
        "distancia_colegio_m": round(float(row["dist_colegio_m"])),
        "distancia_hospital_m": round(float(row["dist_hospital_m"])),
        "siniestros_250m_3anios": int(row["siniestros_250m"]),
        "lluvia_30d_mm": round(float(row["lluvia_30d_mm"])),
        "prioridad": {"nivel": row["nivel_prioridad"], "score": round(float(row["score_prioridad"]), 1)},
        "acciones_permitidas": sorted(ALLOWED_ACTIONS[row["estado"]]),
        "urgencia_sugerida": URGENCY[row["nivel_prioridad"]],
    }
    return f


def template(row) -> dict:
    f = facts(row)
    dom = dominant_damage(row)
    dets = [f"{n} {name}" for name, n in f["detecciones"].items() if n > 0]
    partes = [
        f"Segmento de {f['longitud_m']} m en estado {f['estado']}: índice de condición {f['indice_condicion']:.0f}/100 "
        f"(intervalo 90 %: {f['intervalo_90'][0]}–{f['intervalo_90'][1]}).",
        ("Detecciones: " + ", ".join(dets) + ".") if dets else "Sin daños visibles detectados.",
        f"Rugosidad IRI {f['iri_m_km']:.1f} m/km.",
    ]
    ctx = []
    if f["distancia_colegio_m"] <= 500:
        ctx.append(f"colegio a {f['distancia_colegio_m']} m")
    if f["distancia_hospital_m"] <= 1000:
        ctx.append(f"hospital a {f['distancia_hospital_m']} m")
    if f["siniestros_250m_3anios"] > 0:
        ctx.append(f"{f['siniestros_250m_3anios']} siniestros a 250 m")
    if ctx:
        partes.append("Contexto: " + ", ".join(ctx) + ".")
    partes.append(f"Prioridad {f['prioridad']['nivel']} ({f['prioridad']['score']:.0f}/100).")
    accion = rule_action(row["estado"], dom)
    return {
        "descripcion": " ".join(partes),
        "accion_recomendada": accion,
        "urgencia": f["urgencia_sugerida"],
        "justificacion": f"Daño dominante: {f['dano_dominante']}; acción según estado {f['estado']}.",
        "fuente_descripcion": "plantilla",
    }


NUM_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?")


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, (int, float)):
        yield float(obj)


def faithfulness(text: str, f: dict, allowed_extra=(90, 100, 30, 6, 12, 250, 3)) -> tuple[bool, list]:
    """Cada número del texto debe estar en los hechos (tolerancia de redondeo)."""
    vals = list(_numbers(f)) + list(allowed_extra)
    bad = []
    for tok in NUM_RE.findall(text):
        x = float(tok.replace(",", "."))
        if not any(abs(x - v) <= max(0.6, 0.005 * abs(v)) for v in vals):
            bad.append(tok)
    return (len(bad) == 0), bad


def validate(out: dict, row, max_words=80) -> list:
    f = facts(row)
    problems = []
    ok, bad = faithfulness(out["descripcion"] + " " + out.get("justificacion", ""), f)
    if not ok:
        problems.append(f"números no respaldados: {bad}")
    if f["estado"] not in out["descripcion"].lower():
        problems.append("estado no mencionado o distinto")
    if out["accion_recomendada"] not in ALLOWED_ACTIONS[row["estado"]]:
        problems.append("acción fuera de la tabla de reglas")
    if len(out["descripcion"].split()) > max_words * 1.15:
        problems.append("descripción demasiado larga")
    return problems


def _requests(rows, cfg):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    system = [{"type": "text", "text": SYSTEM_PROMPT.format(max_words=cfg["max_words"]), "cache_control": {"type": "ephemeral"}}]
    reqs = []
    for _, row in rows.iterrows():
        reqs.append(
            Request(
                custom_id=str(row["segmento_id"]),
                params=MessageCreateParamsNonStreaming(
                    model=cfg["model"],
                    max_tokens=cfg["max_tokens"],
                    system=system,
                    output_config={"effort": cfg["effort"], "format": {"type": "json_schema", "schema": SCHEMA}},
                    messages=[{"role": "user", "content": "HECHOS:\n" + json.dumps(facts(row), ensure_ascii=False, sort_keys=True)}],
                ),
            )
        )
    return reqs


def describe_with_claude(rows, cfg, client=None):
    """Envía un lote a Claude y devuelve {segmento_id: dict} solo para salidas válidas."""
    import anthropic

    client = client or anthropic.Anthropic()
    batch = client.messages.batches.create(requests=_requests(rows, cfg))
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        time.sleep(cfg["batch_poll_seconds"])
    by_id = rows.set_index("segmento_id")
    results, stats = {}, {"enviados": len(rows), "exitosos": 0, "json_valido": 0, "fieles": 0, "rechazos": 0, "errores": 0}
    for r in client.messages.batches.results(batch.id):
        if r.result.type != "succeeded":
            stats["errores"] += 1
            continue
        msg = r.result.message
        if msg.stop_reason == "refusal":
            stats["rechazos"] += 1
            continue
        stats["exitosos"] += 1
        text = next((blk.text for blk in msg.content if blk.type == "text"), "")
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            continue
        stats["json_valido"] += 1
        if not validate(out, by_id.loc[r.custom_id].to_dict() | {"segmento_id": r.custom_id}, cfg["max_words"]):
            stats["fieles"] += 1
            results[r.custom_id] = {**out, "fuente_descripcion": cfg["model"]}
    return results, stats


def describe_all(rows, cfg=None, use_llm=None, client=None):
    """Devuelve (DataFrame de textos, estadísticas). Plantilla para todo lo que Claude no cubra."""
    import pandas as pd

    cfg = cfg or load_yaml("configs/llm.yaml")
    use_llm = cfg["enabled"] if use_llm is None else use_llm
    llm_out, stats = {}, {"modo": "plantilla"}
    if use_llm and (os.environ.get("ANTHROPIC_API_KEY") or client is not None):
        llm_out, stats = describe_with_claude(rows, cfg, client)
        stats["modo"] = "claude"
    elif use_llm:
        stats["aviso"] = "sin ANTHROPIC_API_KEY: se usa la plantilla"
    texts = []
    for _, row in rows.iterrows():
        t = llm_out.get(str(row["segmento_id"])) or template(row)
        texts.append({"segmento_id": row["segmento_id"], **t})
    df = pd.DataFrame(texts)
    df["accion_recomendada_txt"] = df["accion_recomendada"].map(ACCION_TXT)
    df["urgencia_txt"] = df["urgencia"].map(URG_TXT)
    return df, stats
