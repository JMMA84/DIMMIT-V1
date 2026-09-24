"""Smoke tests del pipeline: conversión VOC->YOLO, simulador y scorer."""
import numpy as np
import pandas as pd

from dimmit.data.prepare import voc_to_yolo_lines
from dimmit.data.simulate_sensors import simulate_segment
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

SCORE_CFG = {
    "class_weights": {"D00": 4.0, "D10": 4.0, "D20": 8.0, "D40": 12.0},
    "sensor_weight": 25.0,
    "conf_threshold": 0.25,
    "categories": {"bueno": 75, "regular": 50, "malo": 0},
}


def test_voc_to_yolo(tmp_path):
    xml = tmp_path / "img.xml"
    xml.write_text(VOC_XML)
    lines = voc_to_yolo_lines(xml, CLASS_MAP)
    assert len(lines) == 1  # D44 no está en el mapa de clases
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"
    assert (float(cx), float(cy), float(w), float(h)) == (0.25, 0.75, 0.5, 0.5)


def test_simulate_segment_schema():
    rng = np.random.default_rng(0)
    df = simulate_segment(rng, "SEG-0000", severity_norm=0.5, t0=0)
    assert len(df) == 500  # 10 s a 50 Hz
    assert not df.isna().any().any()
    assert (df["az"] > 0).all()
    # más severidad => más vibración
    calm = simulate_segment(np.random.default_rng(0), "SEG-0001", severity_norm=0.0, t0=0)
    assert df["az"].std() > calm["az"].std()


def test_scorer_range_and_order():
    feats = pd.DataFrame(
        {
            "segment_id": ["sano", "danado"],
            "sev_D00": [0.0, 3.0],
            "sev_D10": [0.0, 0.0],
            "sev_D20": [0.0, 2.0],
            "sev_D40": [0.0, 4.0],
            "roughness_norm": [0.0, 1.0],
        }
    )
    scores = RuleBasedScorer(SCORE_CFG).score(feats)
    assert scores["score"].between(0, 100).all()
    sano = scores.set_index("segment_id")
    assert sano.loc["sano", "score"] == 100.0
    assert sano.loc["sano", "category"] == "bueno"
    assert sano.loc["danado", "score"] < sano.loc["sano", "score"]
