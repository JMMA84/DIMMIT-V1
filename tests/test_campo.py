"""Pruebas rápidas y sin red de la etapa de campo."""
import numpy as np
import pandas as pd

from dimmit.campo.gpx import interpolate, read_gpx, smooth_track
from dimmit.campo.scores import etiqueta
from dimmit.campo.tramos import cut_tramos, frames_sin_gps, prueba_id, sitio_id, via_id, via_id_sin_gps, wa_key
from dimmit.campo.filtros import apply as apply_filters, in_road
from dimmit.campo.ultrasonico import align_normalized, astm_severity, depth_features, read_depth

GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.0" xmlns="http://www.topografix.com/GPX/1/0"><trk><trkseg>
<trkpt lat="4.60000" lon="-74.07000"><ele>2600</ele><time>2026-10-05T14:00:00Z</time><speed>1.0</speed><sat>20</sat></trkpt>
<trkpt lat="4.60009" lon="-74.07000"><ele>2600</ele><time>2026-10-05T14:00:10Z</time><speed>1.0</speed><sat>20</sat></trkpt>
<trkpt lat="4.60018" lon="-74.07000"><ele>2600</ele><time>2026-10-05T14:00:20Z</time><speed>1.0</speed><sat>8</sat></trkpt>
</trkseg></trk></gpx>"""

CFG_ET = {"intervencion": {"estados": ["malo", "muy_malo"], "profundidad_cm": 8.0, "bache_conf_min": 0.5},
          "mantenimiento": {"estados": ["regular"], "profundidad_cm": 3.0}}


def test_gpx_parse_smooth_and_interpolate(tmp_path):
    p = tmp_path / "t.gpx"
    p.write_text(GPX)
    tr = read_gpx(p)
    assert len(tr) == 3 and tr["timestamp"].dt.tz is not None and tr["sat"].tolist() == [20, 20, 8]
    sm = smooth_track(tr, 1)
    assert abs(sm["dist_acum_m"].iloc[-1] - 20) < 1.0  # 0.00018° de latitud ~ 20 m
    q = interpolate(sm, pd.to_datetime(["2026-10-05T14:00:05Z", "2026-10-05T14:00:30Z"]))
    assert abs(q["lat"].iloc[0] - 4.600045) < 1e-6 and abs(q["dist_acum_m"].iloc[0] - 5) < 0.6
    assert q["dentro_de_traza"].tolist() == [True, False] and abs(q["gps_dt_s"].iloc[1] - 10) < 1e-6


def test_cut_tramos_merges_short_tail():
    d = np.linspace(0, 23, 24)  # 24 fotos a 1 m
    tr = cut_tramos(d, 10, 5)
    assert tr.min() == 0 and tr.max() == 1  # el tramo 2 (4 fotos) se funde con el 1
    assert (tr == 0).sum() == 10 and (tr == 1).sum() == 14
    assert cut_tramos(np.array([3.0, 3.0, 3.0]), 10, 5).tolist() == [0, 0, 0]


def test_ids_are_deterministic_and_distinct():
    a = via_id("2026-10-05", "Prueba_2", 0, 4.600259, -74.071689)
    assert a == via_id("2026-10-05", "Prueba_2", 0, 4.6002591, -74.0716894)  # mismo centroide a 1e-5
    assert a != via_id("2026-10-05", "Prueba_2", 1, 4.600259, -74.071689) and a.startswith("VIA-") and len(a) == 16
    assert prueba_id("2026-10-05", "Prueba_2") != prueba_id("2026-10-05", "Prueba_3")


def test_whatsapp_frames_order_and_skip_nested(tmp_path):
    names = ["WhatsApp Image 2026-11-15 at 3.35.47 PM (1).jpeg", "WhatsApp Image 2026-11-15 at 3.35.46 PM.jpeg",
             "WhatsApp Image 2026-11-15 at 3.35.47 PM.jpeg", "WhatsApp Image 2026-11-15 at 12.01.02 AM.jpeg", "nota.txt"]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "prueba8").mkdir()  # duplicado anidado: no se lee
    (tmp_path / "prueba8" / "WhatsApp Image 2026-11-15 at 3.36.31 PM.jpeg").write_bytes(b"x")
    assert wa_key(names[0]) == ("2026-11-15", 15 * 3600 + 35 * 60 + 47, 1) and wa_key(names[3])[1] == 62 and wa_key("captura.jpg") is None
    fr = frames_sin_gps(tmp_path, "America/Bogota")
    assert [p.name for p in fr["path_original"]] == [names[3], names[1], names[2], names[0]]
    assert fr["timestamp"].is_monotonic_increasing and fr["t_norm"].tolist() == [0, 1 / 3, 2 / 3, 1]
    a = via_id_sin_gps("2026-11-15", "Prueba_5", 0, "abc")
    assert a.startswith("VIA-") and len(a) == 16 and a != via_id_sin_gps("2026-11-15", "Prueba_5", 0, "abd")
    assert sitio_id("2026-11-15", 1) != sitio_id("2026-11-15", 2)


def test_ultrasonic_cleaning_baseline_and_alignment(tmp_path):
    # posicionamiento (52 cm) y apagado (0,00) se descartan; profundidad = distancia - 8 cm
    p = tmp_path / "Tiempo_ms,Distancia_cm,ProfundidadB9.txt"
    p.write_text("Tiempo_ms,Distancia_cm,ProfundidadBache_cm\n1000,52.0,12\n1250,8.0,-32\n1500,15.5,-24.5\n1750,7.6,-32.4\n2000,0.00,-40\n")
    d = read_depth(p, 8.0, 35.0)
    assert d["valida"].tolist() == [False, True, True, True, False]
    assert d["motivo_descarte"].tolist() == ["fuera_de_rango", "", "", "", "lectura_nula"]
    prof = d["profundidad_cm"].to_numpy()
    assert np.isnan(prof[0]) and np.isnan(prof[4]) and np.allclose(prof[1:4], [0.0, 7.5, -0.4])
    f = depth_features(d, 3.0)
    assert f["n_lecturas"] == 5 and f["n_validas"] == 3 and f["n_fuera_de_rango"] == 1 and f["n_nulas"] == 1
    assert f["prof_max_cm"] == 7.5 and abs(f["frac_mayor_umbral"] - 1 / 3) < 1e-9 and f["prof_max_cm_base40"] == 12.0
    al = align_normalized(d, [0.0, 0.5, 1.0], half_window=0.05)  # extremos solo tienen lecturas descartadas
    assert np.isnan(al[0]) and al[1] == 7.5 and np.isnan(al[2])
    vacia = read_depth(p, 8.0, 5.0)  # rango tan corto que nada es válido
    assert depth_features(vacia)["n_validas"] == 0 and np.isnan(depth_features(vacia)["prof_max_cm"])


def test_etiqueta_rules():
    assert etiqueta("bueno", np.nan, 0.1, CFG_ET) == "prevencion"
    assert etiqueta("regular", np.nan, 0.1, CFG_ET) == "mantenimiento"
    assert etiqueta("bueno", 4.0, 0.1, CFG_ET) == "mantenimiento"  # la profundidad sube la etiqueta
    assert etiqueta("satisfactorio", 9.0, 0.1, CFG_ET) == "intervencion"
    assert etiqueta("bueno", np.nan, 0.6, CFG_ET) == "intervencion"
    assert etiqueta("muy_malo", np.nan, 0.0, CFG_ET) == "intervencion"


def test_fusion_release_loader_matches_saved_members():
    pytest = __import__("pytest")
    from dimmit.utils.io import REPO_ROOT

    path = REPO_ROOT / "models/release/fusion"
    if not (path / "fusion_meta.json").exists():
        pytest.skip("sin pesos de fusión en el repo")
    from dimmit.models.score.fusion import load_release

    members, meta = load_release(path)
    assert len(members) == len(meta["miembros"]) and set(members[0]["model"].groups) == set(meta["grupos"])
    assert all(len(m["pres"][g].cols) == len(meta["miembros"][0]["pre"][g]["cols"]) for m in members for g in meta["grupos"])


def test_astm_severity():
    assert astm_severity(1.0) == "L" and astm_severity(3.0) == "M" and astm_severity(9.0) == "H"
    assert astm_severity(float("nan")) is None and astm_severity(-0.5) is None  # pavimento sano


def test_filters_band_road_persistence_and_scales():
    frames = pd.DataFrame({"image_id": ["a", "b", "c"], "prueba": "P", "timestamp": pd.to_datetime(["2026-10-05T14:00:00.000Z", "2026-10-05T14:00:00.500Z", "2026-10-05T14:00:01.000Z"])})
    det = pd.DataFrame([
        ("a", 2, 0.40, 0.50, 0.60, 0.20, 0.20),  # D20 real: reaparece en b
        ("b", 2, 0.35, 0.50, 0.62, 0.20, 0.20),
        ("a", 1, 0.30, 0.50, 0.50, 0.70, 0.10),  # D10 banda ancha y baja -> 2_banda
        ("a", 3, 0.30, 0.02, 0.40, 0.10, 0.10),  # fuera de la calzada -> 2_fuera_de_via
        ("c", 3, 0.30, 0.50, 0.70, 0.10, 0.10),  # sin pareja -> 3_sin_persistencia
        ("b", 2, 0.10, 0.50, 0.50, 0.20, 0.20),  # bajo el umbral -> 1_confianza
    ], columns=["image_id", "class_id", "conf", "cx", "cy", "w", "h"])
    cfg = {"tau_campo": {"D00": 0.25, "D10": 0.15, "D20": 0.2, "D40": 0.2}, "banda": {"clase": "D10", "w_min": 0.5, "h_max": 0.15},
           "via": {"cy_min": 0.35, "half_w0": 0.38, "half_w_slope": 0.12}, "persistencia": {"vecinos": 2, "iou_min": 0.3, "dy_tol": 0.15},
           "escalas": {"imgsz": [832], "iou_min": 0.5, "minimo": 1}}
    out, resumen = apply_filters(det, frames, cfg)
    assert out["motivo_descarte"].tolist() == ["", "", "2_banda", "2_fuera_de_via", "3_sin_persistencia", "1_confianza"]
    # superficie: en adoquín la "piel de cocodrilo" (D20) son las juntas del patrón; los baches sí cuentan
    sup = {**cfg, "superficie": {"asfalto": ["D00", "D10", "D20", "D40"], "adoquin": ["D40"]}}
    out_s, _ = apply_filters(det, frames.assign(superficie=["adoquin", "adoquin", "asfalto"]), sup)
    assert out_s["motivo_descarte"].tolist()[:2] == ["2_superficie", "2_superficie"]
    pot = pd.DataFrame([("a", 3, 0.5, 0.5, 0.6, 0.2, 0.2), ("b", 3, 0.5, 0.5, 0.6, 0.2, 0.2)], columns=det.columns)
    assert apply_filters(pot, frames.assign(superficie="adoquin"), sup)[0]["conservada"].all()
    assert in_road(np.array([0.5, 0.05]), np.array([0.6, 0.4])).tolist() == [True, False]
    # consenso: la otra escala solo ve la caja de "a"
    other = pd.DataFrame([("a", 2, 0.30, 0.50, 0.60, 0.20, 0.20)], columns=det.columns)
    out2, _ = apply_filters(det, frames, cfg, {"832": other})
    assert out2["motivo_descarte"].tolist()[:2] == ["", "4_sin_consenso_escala"]
