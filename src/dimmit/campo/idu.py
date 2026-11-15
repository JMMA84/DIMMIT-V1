"""Referencia oficial: estado superficial de calzada del IDU alrededor de los sitios de la toma.

Fuente: Mapa de Referencia de Bogotá (IDECA / Catastro), servicio público ArcGIS:
  capa 15  Calzada (polígono; tipo de superficie, función, longitud)
  tabla 54 Estado superficial de calzada (Bueno / Regular / Malo, con fecha de evaluación del IDU)
Las fotos de la toma no tienen GPS, así que la comparación es por zona (radio alrededor de un punto de
referencia de configs/campo.yaml) y por tipo de superficie, no tramo a tramo.
Salida: data/campo/idu/calzadas_estado.csv y calzadas.geojson (polígonos, para ubicar los puntos GPS)
"""
import json

import numpy as np
import pandas as pd
import requests

from dimmit.campo import CAMPO, load_cfg
from dimmit.utils.io import ensure_dir

URL = "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/Mapa_Referencia/Mapa_Referencia/MapServer"
SUPERFICIE = {0: "sin dato", 1: "rigido", 2: "flexible", 3: "mixto", 4: "adoquin_concreto", 5: "adoquin_arcilla", 6: "afirmado",
              8: "tierra", 12: "piedra_laja", 13: "placa_huella"}
ESTADO = {1: "bueno", 2: "regular", 3: "malo"}  # dominio publicado D_CalEstado; otros códigos no tienen nombre publicado
GRUPO = {"flexible": "asfalto", "adoquin_concreto": "adoquin", "adoquin_arcilla": "adoquin"}
OUT = CAMPO / "idu"


def _get(path, **params):
    r = requests.get(f"{URL}/{path}", params={**params, "f": params.get("f", "json")}, timeout=90)
    r.raise_for_status()
    return r.json()


def download(lat, lon, radio_m):
    g = _get("15/query", geometry=f"{lon},{lat}", geometryType="esriGeometryPoint", inSR=4326, distance=radio_m,
             units="esriSRUnit_Meter", spatialRel="esriSpatialRelIntersects", outFields="*", outSR=4326, f="geojson")
    rows = []
    for f in g["features"]:
        ring = np.array(f["geometry"]["coordinates"][0] if f["geometry"]["type"] == "Polygon" else f["geometry"]["coordinates"][0][0])
        rows.append({**f["properties"], "lat": float(ring[:, 1].mean()), "lon": float(ring[:, 0].mean())})
    cal = pd.DataFrame(rows)
    est = []
    codes = cal["CALCODIGO"].tolist()
    for i in range(0, len(codes), 50):
        r = _get("54/query", where=f"CALCODIGO IN ({','.join(map(str, codes[i:i + 50]))})", outFields="*")
        est += [x["attributes"] for x in r["features"]]
    est = pd.DataFrame(est)
    est["fecha_idu"] = pd.to_datetime(est["FECHA"], unit="ms").dt.date
    est = est.sort_values("fecha_idu").groupby("CALCODIGO").tail(1)  # evaluación más reciente
    t = cal.merge(est[["CALCODIGO", "ESTADO", "fecha_idu"]], on="CALCODIGO", how="left")
    t["superficie"] = t["CALTSUPERF"].map(SUPERFICIE).fillna("otro")
    t["grupo"] = t["superficie"].map(GRUPO).fillna("otro")
    t["estado_idu"] = t["ESTADO"].map(ESTADO).fillna("sin_clasificar")
    t["funcion"] = t["CALFUNCION"].map({0: "no_aplica", 1: "vehicular", 2: "peatonal"})
    t["dist_m"] = np.hypot((t["lat"] - lat) * 111320, (t["lon"] - lon) * 111320 * np.cos(np.radians(lat)))
    cols = ["CALCODIGO", "CALCIV", "funcion", "superficie", "grupo", "estado_idu", "ESTADO", "fecha_idu", "CALLONGITU", "CALANCHO", "lat", "lon", "dist_m"]
    out = t[cols].rename(columns={"CALCODIGO": "calcodigo", "CALCIV": "civ", "ESTADO": "estado_codigo", "CALLONGITU": "longitud_m", "CALANCHO": "ancho_m"}).sort_values("dist_m")
    return out, g


def main():
    ref = load_cfg()["idu"]
    t, g = download(ref["lat"], ref["lon"], ref["radio_m"])
    ensure_dir(OUT)
    t.to_csv(OUT / "calzadas_estado.csv", index=False)
    (OUT / "calzadas.geojson").write_text(json.dumps(g))
    print(f"[ok] {len(t)} calzadas a ≤ {ref['radio_m']} m de {ref['referencia']} (IDU, {t['fecha_idu'].max()}) -> {OUT}")
    print(pd.crosstab(t["grupo"], t["estado_idu"], margins=True))


if __name__ == "__main__":
    main()
