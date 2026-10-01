"""Pruebas rápidas y sin red del pipeline v1."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dimmit.data.prepare import voc_to_yolo_lines
from dimmit.models.score.model import RuleBasedScorer

VOC_XML = """<annotation>
  <size><width>600</width><height>600</height><depth>3</depth></size>
  <object><name>D00</name>
    <bndbox><xmin>0</xmin><ymin>300</ymin><xmax>300</xmax><ymax>600</ymax></bndbox>
  </object>
  <object><name>D44</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox>
  </object>
</annotation>"""

CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
SRC = Path(__file__).resolve().parents[1] / "src" / "dimmit"


def test_voc_to_yolo(tmp_path):
    xml = tmp_path / "img.xml"
    xml.write_text(VOC_XML)
    lines = voc_to_yolo_lines(xml, CLASS_MAP)
    assert len(lines) == 1  # D44 no está en el mapa de clases
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"
    assert (float(cx), float(cy), float(w), float(h)) == (0.25, 0.75, 0.5, 0.5)


def test_voc_zero_size_uses_image_size(tmp_path):
    xml = tmp_path / "img.xml"
    xml.write_text(VOC_XML.replace("<width>600</width><height>600</height>", "<width>0</width><height>0</height>"))
    assert voc_to_yolo_lines(xml, CLASS_MAP) == []
    assert len(voc_to_yolo_lines(xml, CLASS_MAP, image_size=(600, 600))) == 1


def test_scorer_v0_range_and_order():
    cfg = {"class_weights": {"D00": 4.0, "D10": 4.0, "D20": 8.0, "D40": 12.0}, "sensor_weight": 25.0, "categories": {"bueno": 75, "regular": 50, "malo": 0}}
    feats = pd.DataFrame(
        {"segment_id": ["sano", "danado"], "sev_D00": [0.0, 3.0], "sev_D10": [0.0, 0.0], "sev_D20": [0.0, 2.0], "sev_D40": [0.0, 4.0], "roughness_norm": [0.0, 1.0]}
    )
    s = RuleBasedScorer(cfg).score(feats).set_index("segment_id")
    assert s.loc["sano", "score"] == 100.0 and s.loc["danado", "score"] < 100


# --- pseudo-PCI ---------------------------------------------------------------------------
from dimmit.labels import pseudo_pci as pp  # noqa: E402


def test_deduct_monotone_and_cdv_rule():
    cfg = pp.load_cfg()
    rho = pd.DataFrame({f"rho_{c}": np.linspace(0, 20, 50) for c in ["D00", "D10", "D20", "D40"]})
    dv = pp.deduct_values(rho, cfg)
    assert (dv.diff().dropna() >= -1e-9).all().all()
    assert (dv.iloc[0] == 0).all()
    assert pp.combine_deducts(np.array([[40.0, 30.0, 0, 0]]), 0.35)[0] == pytest.approx(50.5)
    pci = pp.pci_from_rho(rho, cfg)["pci_vis"]
    assert pci.between(0, 100).all() and pci.iloc[0] == 100


def test_states_and_icv():
    cfg = pp.load_cfg()
    assert list(pp.state_of([95, 80, 60, 45, 10], cfg)) == ["bueno", "satisfactorio", "regular", "malo", "muy_malo"]
    assert pp.ride_index(1.0, cfg) == pytest.approx(100)
    assert pp.icv(100, 100, cfg) == pytest.approx(100)


# --- sensores -----------------------------------------------------------------------------
from dimmit.sensors.iri import iri  # noqa: E402
from dimmit.sensors.profile import gd_for_iri, iso8608_profile  # noqa: E402


def test_golden_car_sanity():
    dx = 0.05
    rng = np.random.default_rng(1)
    assert iri(np.zeros(4000), dx) == pytest.approx(0, abs=1e-9)
    p = iso8608_profile(6000, dx, 16e-6, rng)
    assert iri(2 * p, dx) == pytest.approx(2 * iri(p, dx), rel=1e-6)  # lineal en la amplitud
    a = iri(iso8608_profile(8000, dx, 16e-6, rng), dx)
    c = iri(iso8608_profile(8000, dx, 256e-6, rng), dx)
    assert 3 < c / a < 5  # clase C vs A ~ 4
    assert iri(iso8608_profile(8000, dx, gd_for_iri(4.0), rng), dx) == pytest.approx(4.0, rel=0.15)


def test_imu_features_scale_with_vibration():
    from dimmit.sensors.imu import window_features

    rng = np.random.default_rng(0)
    n = 500
    base = {"gx": 0.0, "gy": 0.0, "gz": 0.0, "lat": 4.65, "lon": -74.08, "speed_kmh": 30.0, "timestamp": pd.date_range("2026-01-01", periods=n, freq="20ms")}
    calm = pd.DataFrame({"ax": rng.normal(0, 0.05, n), "ay": rng.normal(0, 0.05, n), "az": 9.81 + rng.normal(0, 0.05, n), **base})
    rough = calm.assign(az=9.81 + rng.normal(0, 1.0, n))
    fc, fr = window_features(calm), window_features(rough)
    assert fr["imu_rms"] > 5 * fc["imu_rms"]
    assert fr["imu_rms_sqrtv"] > fc["imu_rms_sqrtv"]


def test_simulator_schema_and_null_world():
    import yaml

    from dimmit.sensors.simulator import simulate_segment

    cfg = yaml.safe_load((SRC.parents[1] / "configs/sensors.yaml").read_text())
    road = pd.Series(dict(segment_id="S1", lat_a=4.65, lon_a=-74.08, lat_b=4.6505, lon_b=-74.0795, longitud_m=60.0))
    fr = pd.DataFrame(dict(image_id=[f"i{k}" for k in range(6)], pci=np.linspace(95, 20, 6), country="Japan"))
    rows, meta, truth = simulate_segment(road, fr, {}, cfg, np.random.default_rng(3), "2026-07-01 08:00")
    assert list(rows.columns) == ["segment_id", "obs_id", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed_kmh"]
    assert not rows.isna().any().any() and (rows["az"].abs() < 2 * 9.81 + 1e-6).all()
    assert len(meta) == 6 and "iri_real_fotograma" in truth
    _, _, t0 = simulate_segment(road, fr, {}, cfg, np.random.default_rng(3), "2026-07-01 08:00", coupling=0.0)
    assert (t0["acoplamiento"] == 0).all()


def test_features_never_read_truth():
    """La verdad latente del simulador no puede alimentar features ni modelos."""
    offenders = []
    for path in list((SRC / "sensors").glob("*.py")) + list((SRC / "models" / "score").glob("*.py")):
        if path.name in ("simulator.py",):
            continue
        txt = path.read_text()
        if re.search(r"read_parquet\([^)]*sensors_truth", txt) and path.name != "dataset.py":
            offenders.append(path.name)
    assert offenders == []
    # dataset.py solo toma de la verdad el IRI real como OBJETIVO, nunca como grupo de entrada
    from dimmit.models.score.dataset import feature_groups

    cols = sum(feature_groups(pd.DataFrame(columns=["iri_real", "imu_rms", "n_D00", "emb_pc00", "icv", "pci_vis"])).values(), [])
    assert not {"iri_real", "icv", "pci_vis"} & set(cols)


# --- geo, splits, métricas ----------------------------------------------------------------
from dimmit.evaluation import metrics as M  # noqa: E402
from dimmit.geo import geohash as gh  # noqa: E402


def test_geohash_known_values():
    assert gh.encode(57.64911, 10.40744, 11) == "u4pruydqqvj"
    lat, lon = gh.decode("d2g6282")
    assert abs(lat - 4.6166) < 0.01 and abs(lon + 74.1577) < 0.01
    assert gh.haversine_m(4.6, -74.08, 4.6, -74.08) == 0


def test_metrics_against_sklearn():
    from sklearn.metrics import cohen_kappa_score, mean_absolute_error

    y = np.array([10, 50, 70, 90.0])
    p = np.array([12, 45, 75, 80.0])
    assert M.regression(y, p)["mae"] == pytest.approx(mean_absolute_error(y, p))
    yt = ["muy_malo", "malo", "regular", "bueno", "bueno"]
    yp = ["muy_malo", "regular", "regular", "satisfactorio", "bueno"]
    o, _ = M.ordinal(yt, yp)
    idx = lambda s: [pp.STATE_ORDER.index(x) for x in s]  # noqa: E731
    assert o["kappa_ponderada_cuadratica"] == pytest.approx(cohen_kappa_score(idx(yt), idx(yp), weights="quadratic"))
    assert o["exactitud_mas_menos_1"] == 1.0


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert M.psi(a, rng.normal(0, 1, 5000)) < 0.02
    assert M.psi(a, rng.normal(1.0, 1, 5000)) > 0.25
    assert M.status(0.05, 0.1, 0.25, "menor") == "OK" and M.status(0.3, 0.1, 0.25, "menor") == "FALLA"


def test_greedy_chain_and_splits_have_no_leakage():
    from dimmit.data.splits import greedy_chain
    from dimmit.utils.io import INDEX

    h = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0xFFFF, 0, 0, 0], [3, 0, 0, 0]], dtype=np.uint64)
    order = greedy_chain(h)
    assert sorted(order) == [0, 1, 2, 3] and order[0] == 0
    path = INDEX / "splits.csv"
    if not path.exists():
        pytest.skip("splits.csv no generado")
    s = pd.read_csv(path)
    lab = s[s["split"] != "despliegue"]
    assert (lab.groupby("group_id")["split"].nunique() == 1).all()
    assert (lab.groupby("dup_cluster")["split"].nunique() == 1).all()
    assert (s.groupby("segment_id")["split"].nunique() == 1).all()
    assert s["image_id"].is_unique


# --- red de fusión ------------------------------------------------------------------------
def test_fusion_net_shapes_and_overfit():
    import torch

    from dimmit.models.score.fusion import FusionNet, corn_loss, corn_probs

    torch.manual_seed(0)
    net = FusionNet({"vision": 6, "imu": 4}, p_modal=0.0)
    xs = {"vision": torch.randn(32, 6), "imu": torch.randn(32, 4)}
    out = net(xs)
    assert out["rho"].shape == (32, 4) and out["corn"].shape == (32, 4)
    probs = corn_probs(out["corn"])
    assert torch.allclose(probs.sum(1), torch.ones(32), atol=1e-5)
    y = (xs["vision"][:, 0] > 0).float()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(200):
        loss = ((net(xs)["icv"] - y) ** 2).mean() + 0.1 * corn_loss(net(xs)["corn"], (y * 4).long())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert ((net(xs)["icv"] - y) ** 2).mean() < 0.05


# --- lenguaje -----------------------------------------------------------------------------
def _row():
    return {
        "segmento_id": "TR-JA-0001", "longitud_m": 80.0, "score_condicion": 47.3, "score_condicion_p05": 30.0, "score_condicion_p95": 64.0,
        "estado": "malo", "score_grieta_longitudinal": 90.0, "score_grieta_transversal": 100.0, "score_piel_cocodrilo": 70.5, "score_baches": 60.2,
        "n_grieta_longitudinal": 2, "n_grieta_transversal": 0, "n_piel_cocodrilo": 3, "n_baches": 1, "iri_m_km": 6.4, "score_rugosidad": 37.0,
        "dist_colegio_m": 250.0, "dist_hospital_m": 900.0, "siniestros_250m": 4, "lluvia_30d_mm": 80.0, "nivel_prioridad": "alta", "score_prioridad": 72.0,
    }


def test_template_is_faithful():
    from dimmit.llm.describe import facts, faithfulness, template, validate

    row = _row()
    t = template(row)
    assert t["fuente_descripcion"] == "plantilla"
    assert faithfulness(t["descripcion"], facts(row))[0]
    assert validate(t, row) == []


def test_faithfulness_catches_invented_numbers():
    from dimmit.llm.describe import facts, validate

    row = _row()
    bad = {"descripcion": "Estado malo con 17 baches y IRI 9,9 m/km.", "accion_recomendada": "reconstruccion", "urgencia": "inmediata_30_dias", "justificacion": ""}
    problems = validate(bad, row)
    assert any("números" in p for p in problems) and any("acción" in p for p in problems)
    assert re.search("9,9", bad["descripcion"]) and facts(row)["estado"] == "malo"


def test_claude_batch_with_mock_client():
    """Ruta de Claude con un cliente simulado (sin red): válidas se aceptan, inválidas caen a plantilla."""
    import json
    from types import SimpleNamespace as NS

    from dimmit.llm.describe import describe_all

    rows = pd.DataFrame([_row(), {**_row(), "segmento_id": "TR-JA-0002"}])
    good = {"descripcion": "Segmento en estado malo, índice 47 de 100, con baches y piel de cocodrilo; IRI 6.4 m/km.", "accion_recomendada": "bacheo", "urgencia": "inmediata_30_dias", "justificacion": "Baches con IRI 6.4."}
    invented = {**good, "descripcion": "Estado malo con 99 baches."}

    class Batches:
        def create(self, requests):
            assert len(requests) == 2
            return NS(id="b1")

        def retrieve(self, _id):
            return NS(processing_status="ended")

        def results(self, _id):
            for cid, payload in (("TR-JA-0001", good), ("TR-JA-0002", invented)):
                msg = NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(payload))])
                yield NS(custom_id=cid, result=NS(type="succeeded", message=msg))

    client = NS(messages=NS(batches=Batches()))
    cfg = {"enabled": True, "model": "claude-opus-5-5", "effort": "low", "max_tokens": 2000, "max_words": 80, "batch_poll_seconds": 0}
    df, stats = describe_all(rows, cfg=cfg, use_llm=True, client=client)
    assert stats["modo"] == "claude" and stats["fieles"] == 1
    assert list(df["fuente_descripcion"]) == ["claude-opus-5-5", "plantilla"]
