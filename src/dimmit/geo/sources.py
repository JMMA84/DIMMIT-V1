"""Descarga (con caché en data/external/) de las fuentes abiertas de contexto para Bogotá.

Fuentes (datosabiertos.bogota.gov.co, CC BY 4.0 / CC BY-SA 4.0):
  - UMV Modelo de Priorización de Vías 2020: 93 684 calzadas (polígono WKT, CIV, ancho, área,
    longitud, tipo de superficie y puntajes de priorización social/económico/técnico).
  - IDU Estado Superficial (PCI oficial por CIV, ASTM D6433).
  - Colegios (SED), IPS (SDS) e Histórico de siniestros viales (SDM).
Clima y elevación se consultan aparte (Open-Meteo) en geo/context.py.
"""
import io
import json
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from dimmit.geo.geohash import mercator_to_wgs84
from dimmit.utils.io import EXTERNAL, ensure_dir, load_yaml

HEADERS = {"User-Agent": "Mozilla/5.0 (DIMMIT road-condition research)"}


def fetch(url, dest, retries=4, timeout=180):
    """Descarga url a dest si no existe (reintentos con espera exponencial)."""
    dest = EXTERNAL / dest
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"[down] {dest.name} ({len(data) / 1e6:.1f} MB)")
            return dest
        except Exception as e:  # noqa: BLE001 - red inestable: reintentar
            wait = 2 ** (k + 1)
            print(f"[retry] {url} ({e}); espero {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"no se pudo descargar {url}")


def fetch_json(url, retries=8, timeout=60):
    for k in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - el proxy corta conexiones de forma intermitente
            time.sleep(min(30, 2 ** (k + 1)))
            last = e
    raise RuntimeError(f"falló {url}: {last}")


# columnas del CSV de la UMV (nombres truncados a 10 caracteres por el shapefile de origen);
# el mapeo se verifica con las sumas: social = suma de sus 5 componentes, IP = social +
# económica + técnica, IP relativo = IP / IP máx * 100 (ver load_umv)
UMV_COLUMNS = {
    "PUNTAJE_DP": "p_poblacion",
    "PUNTAJE__1": "p_densidad",
    "PUNTAJE_PE": "p_peticiones",
    "PUNTAJE_SI": "p_sitios_sociales",
    "PUNTAJE_US": "p_uso_social",
    "PUNTAJE_DI": "dim_social",
    "PUNTAJE__2": "p_uso_economico",
    "PUNTAJE__3": "p_sitios_economicos",
    "PUNTAJE__4": "dim_economica",
    "PUNTAJE_MA": "p_malla",
    "PUNTAJE_RU": "p_rutas_transporte",
    "PUNTAJE__5": "p_siniestros",
    "PUNTAJE__6": "dim_tecnica",
    "PUNTAJE_IP": "ip_umv",
    "IP_RELATIV": "ip_relativo_umv",
    "PUNTAJE__7": "ip_max_umv",
    "RANGOS_IP": "rango_ip_umv",
}


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_umv(cfg):
    """Calzadas UMV con geometría simplificada: centroide y eje largo (rectángulo mínimo)."""
    out = EXTERNAL / "umv_calzadas.parquet"
    if out.exists():
        return pd.read_parquet(out)
    import shapely

    zpath = fetch(cfg["urls"]["umv"], "umv_priorizacion.zip")
    with zipfile.ZipFile(zpath) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype=str)
    df = df[df["the_geom"].notna()].reset_index(drop=True)
    geoms = shapely.from_wkt(df["the_geom"].to_numpy(), on_invalid="ignore")
    keep = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    df, geoms = df[keep].reset_index(drop=True), geoms[keep]
    rows = []
    for geom in geoms:
        c = geom.centroid
        rect = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)[:4]
        e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
        if np.hypot(*e1) >= np.hypot(*e2):
            a, b = (rect[0] + rect[3]) / 2, (rect[1] + rect[2]) / 2
        else:
            a, b = (rect[0] + rect[1]) / 2, (rect[2] + rect[3]) / 2
        rows.append((c.y, c.x, a[1], a[0], b[1], b[0]))
    geo = pd.DataFrame(rows, columns=["lat", "lon", "lat_a", "lon_a", "lat_b", "lon_b"])
    res = pd.DataFrame(
        {
            "civ": _num(df["CIV"]).astype("Int64"),
            "pk_calzada": _num(df["PK_ID_CALZ"]).astype("Int64"),
            "ancho_m": _num(df["ANCHOCALZA"]),
            "area_m2": _num(df["AREACALZAD"]),
            "longitud_umv_m": _num(df["LONGITUDHO"]),
            "tipo_superficie": _num(df["TIPOSUPERF"]),
        }
    )
    for src, dst in UMV_COLUMNS.items():
        res[dst] = df[src] if src == "RANGOS_IP" else _num(df[src])
    res = pd.concat([res, geo], axis=1)
    # verificación del mapeo de columnas
    soc = res[["p_poblacion", "p_densidad", "p_peticiones", "p_sitios_sociales", "p_uso_social"]].sum(axis=1)
    ok_soc = np.isclose(soc, res["dim_social"], atol=0.05).mean()
    ok_ip = np.isclose(res[["dim_social", "dim_economica", "dim_tecnica"]].sum(axis=1), res["ip_umv"], atol=0.05).mean()
    print(f"[check] UMV: social=suma componentes {ok_soc:.1%}, IP=suma dimensiones {ok_ip:.1%}")
    res.to_parquet(out, index=False)
    return res


def load_schools(cfg):
    d = json.loads(fetch(cfg["urls"]["colegios"], "colegios.geojson").read_text())
    xs = np.array([f["geometry"]["coordinates"][0] for f in d["features"]])
    ys = np.array([f["geometry"]["coordinates"][1] for f in d["features"]])
    lat, lon = mercator_to_wgs84(xs, ys) if np.abs(xs).max() > 180 else (ys, xs)
    names = [f["properties"].get("NOMBRE_EST") for f in d["features"]]
    return pd.DataFrame({"nombre": names, "lat": lat, "lon": lon})


def load_health(cfg):
    d = json.loads(fetch(cfg["urls"]["ips"], "ips.geojson").read_text())
    rows = []
    for f in d["features"]:
        p = f["properties"]
        x, y = f["geometry"]["coordinates"][:2]
        name = str(p.get("nombre") or p.get("nombre_pre") or "")
        hosp = (str(p.get("ese", "")).upper() == "SI") or any(k in name.upper() for k in ("HOSPITAL", "CLINICA", "CLÍNICA"))
        rows.append({"nombre": name, "lat": y, "lon": x, "hospital": hosp, "publica": str(p.get("naturaleza", "")).lower() == "publica"})
    return pd.DataFrame(rows)


def load_crashes(cfg):
    out = EXTERNAL / "siniestros.parquet"
    if out.exists():
        return pd.read_parquet(out)
    path = fetch(cfg["urls"]["siniestros"], "siniestros.csv", timeout=600)
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["LATITUD", "LONGITUD", "GRAVEDAD", "ANO_OCURRENCIA_ACC", "CIV"])
    df = df.rename(columns={"LATITUD": "lat", "LONGITUD": "lon", "GRAVEDAD": "gravedad", "ANO_OCURRENCIA_ACC": "anio", "CIV": "civ"})
    for c in ("lat", "lon", "anio"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    df = df[df["lat"].between(3.5, 5.0) & df["lon"].between(-75, -73.5)]
    df["peso"] = df["gravedad"].map({"SOLO DANOS": 1, "CON HERIDOS": 3, "CON MUERTOS": 10}).fillna(1)
    df = df.dropna(subset=["lat", "lon"])
    df.to_parquet(out, index=False)
    return df


def load_idu_state(cfg):
    """Estado superficial oficial del IDU por CIV (informativo)."""
    out = EXTERNAL / "idu_estado.parquet"
    if out.exists():
        return pd.read_parquet(out)
    import py7zr

    path = fetch(cfg["urls"]["idu_estado"], "idu_estado.7z")
    tmp = ensure_dir(EXTERNAL / "idu_estado")
    with py7zr.SevenZipFile(path) as z:
        z.extractall(tmp)
    csvs = sorted(tmp.rglob("*.csv"))
    df = None
    for enc in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(csvs[0], encoding=enc, sep=None, engine="python")
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().lower() for c in df.columns]
    df.to_parquet(out, index=False)
    return df


def main():
    cfg = load_yaml("configs/context.yaml")
    umv = load_umv(cfg)
    print(f"[ok] UMV {len(umv)} calzadas")
    print(f"[ok] colegios {len(load_schools(cfg))}, IPS {len(load_health(cfg))}, siniestros {len(load_crashes(cfg))}")
    try:
        idu = load_idu_state(cfg)
        print(f"[ok] IDU estado superficial {len(idu)} filas; columnas: {list(idu.columns)[:12]}")
    except Exception as e:  # noqa: BLE001 - fuente informativa
        print(f"[warn] IDU estado superficial no disponible: {e}")


if __name__ == "__main__":
    main()
