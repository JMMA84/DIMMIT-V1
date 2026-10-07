"""Pruebas rápidas y sin red de la etapa de campo."""
import numpy as np
import pandas as pd

from dimmit.campo.gpx import interpolate, read_gpx, smooth_track
from dimmit.campo.scores import etiqueta
from dimmit.campo.tramos import cut_tramos, prueba_id, via_id
from dimmit.campo.filtros import apply as apply_filters, in_road
from dimmit.campo.ultrasonico import align_normalized, astm_severity, baseline_mode, depth_features, read_depth

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


def test_ultrasonic_depth_and_alignment(tmp_path):
    p = tmp_path / "Tiempo_ms,Distancia_cm,ProfundidadB9.txt"
    p.write_text("Tiempo_ms,Distancia_cm,ProfundidadBache_cm\n1000,40.0,0\n1250,42.0,2\n1500,50.0,10\n1750,41.0,1\n")
    d = read_depth(p, 40.0)
    assert d["profundidad_cm"].tolist() == [0.0, 2.0, 10.0, 1.0] and d["t_norm"].iloc[-1] == 1.0
    f = depth_features(d, 3.0)
    assert f["prof_max_cm"] == 10.0 and abs(f["frac_mayor_umbral"] - 0.25) < 1e-9 and f["n_lecturas"] == 4
    al = align_normalized(d, [0.0, 2 / 3, 0.5], half_window=0.05)
    assert al[0] == 0.0 and al[1] == 10.0 and np.isnan(al[2])


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


def test_baseline_mode_and_astm_severity():
    # 30 lecturas de pavimento a 52 cm y un hueco de 9 cm: la moda es el pavimento, no el firmware
    d = np.r_[np.full(30, 52.0) + np.linspace(-0.4, 0.4, 30), [58.0, 61.0, 60.5]]
    base, fuente = baseline_mode(d, min_n=20, fallback=40.0)
    assert base == 52.0 and fuente == "moda"
    assert baseline_mode(d[:5], min_n=20, fallback=40.0) == (40.0, "fija")
    assert astm_severity(1.0) == "L" and astm_severity(3.0) == "M" and astm_severity(9.0) == "H" and astm_severity(float("nan")) is None


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
    assert in_road(np.array([0.5, 0.05]), np.array([0.6, 0.4])).tolist() == [True, False]
    # consenso: la otra escala solo ve la caja de "a"
    other = pd.DataFrame([("a", 2, 0.30, 0.50, 0.60, 0.20, 0.20)], columns=det.columns)
    out2, _ = apply_filters(det, frames, cfg, {"832": other})
    assert out2["motivo_descarte"].tolist()[:2] == ["", "4_sin_consenso_escala"]
