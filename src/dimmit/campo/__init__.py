"""Datos de campo propios: ingesta desde Drive, GPS, sensor ultrasónico, tramos con ID y salidas."""
from dimmit.utils.io import DATA, REPORTS, load_yaml

CAMPO = DATA / "campo"
RAW = CAMPO / "raw"
INDEX_CAMPO = CAMPO / "index"
OUT_CAMPO = REPORTS / "campo"


def load_cfg():
    return load_yaml("configs/campo.yaml")
