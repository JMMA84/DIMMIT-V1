"""Fotogramas de campo <-> (GPS interpolado) <-> tramos <-> id_via determinista.

Las pruebas se declaran en configs/campo.yaml (`pruebas`): carpeta de fotos, archivo del sensor y sitio.
Dos modos según lo que traiga la carpeta:
  * con GPS (`grabacion_*`: captura_<fecha>_<hora>_<ms>_<n>.jpg + .gpx): posición interpolada y tramos
    de ~10 m por distancia acumulada; id_via ancla en el centroide.
  * sin GPS (fotos de celular reenviadas, p. ej. "WhatsApp Image <fecha> at <h.m.s> PM (n).jpeg", sin EXIF):
    el orden sale del nombre, cada prueba es un tramo y el id_via ancla en el hash de sus fotos. Las tomas
    repetidas del mismo sitio se agregan en una fila de nivel "sitio".
Las fotos se copian con nombre único a data/campo/fotos/<prueba>/ (lo que lee el detector y sube a Kaggle).

Salidas (data/campo/index/):
  fotogramas.parquet  una fila por foto: prueba, id_prueba, sitio, id_sitio, id_via, tramo, timestamp, lat, lon,
                      sat, vel_ms, gps_dt_s, gps_incertidumbre_m, dist_acum_m, t_norm, hash_imagen, archivo_original
  tramos.parquet      una fila por tramo: id_via, prueba, tramo, sitio, centroide, extremos, geohash7,
                      longitud_m, n_fotogramas, hora_ini, hora_fin
  campo.txt           lista de rutas para el detector
"""
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dimmit.campo import INDEX_CAMPO, RAW, load_cfg
from dimmit.campo.gpx import interpolate, read_gpx, smooth_track
from dimmit.geo.geohash import encode, haversine_m
from dimmit.utils.io import ensure_dir

IMG_RE = re.compile(r"captura_(\d{8})_(\d{6})_(\d{3})_(\d+)\.jpg$")
WA_RE = re.compile(r"(\d{4}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})\s*(AM|PM)(?:\s*\((\d+)\))?", re.I)
FOTOS = RAW.parent / "fotos"
GEO_COLS = ["lat", "lon", "sat", "vel_ms", "gps_dt_s", "gps_incertidumbre_m", "dist_acum_m", "dentro_de_traza"]


def via_id(fecha, prueba, tramo, lat_c, lon_c):
    """ID determinista: mismo dato -> mismo ID (fecha, prueba, tramo y centroide a 1e-5 grados ~ 1 m)."""
    key = f"{fecha}|{prueba}|{int(tramo):02d}|{lat_c:.5f}|{lon_c:.5f}"
    return "VIA-" + hashlib.sha1(key.encode()).hexdigest()[:12]


def via_id_sin_gps(fecha, prueba, tramo, hash_imagenes):
    """ID determinista sin posición: ancla en el contenido (hash de las fotos del tramo)."""
    key = f"{fecha}|{prueba}|{int(tramo):02d}|sin_gps|{hash_imagenes}"
    return "VIA-" + hashlib.sha1(key.encode()).hexdigest()[:12]


def prueba_id(fecha, prueba):
    return "PRU-" + hashlib.sha1(f"{fecha}|{prueba}".encode()).hexdigest()[:8]


def sitio_id(fecha, sitio):
    return "SIT-" + hashlib.sha1(f"{fecha}|sitio{int(sitio)}".encode()).hexdigest()[:8]


def wa_key(name):
    """(fecha, segundos del día, sufijo) de un nombre de WhatsApp; None si no lo es."""
    m = WA_RE.search(name)
    if not m:
        return None
    fecha, h, mi, se, ap, n = m.groups()
    h = int(h) % 12 + (12 if ap.upper() == "PM" else 0)
    return fecha, h * 3600 + int(mi) * 60 + int(se), int(n or 0)


def frames_sin_gps(folder: Path, tz):
    """Fotos sin EXIF: orden por (hora de envío, sufijo). La hora es de envío, no de captura: el
    timestamp solo conserva el orden y t_norm reparte las fotos uniformemente en la prueba."""
    rows = []
    for p in folder.iterdir():  # solo el primer nivel: subcarpetas (p. ej. duplicados) no cuentan
        if p.suffix.lower() not in (".jpg", ".jpeg") or not p.is_file():
            continue
        k = wa_key(p.name)
        if k is None:
            continue
        rows.append({"path_original": p, "fecha": k[0], "seg": k[1], "suf": k[2]})
    if not rows:
        return pd.DataFrame()
    fr = pd.DataFrame(rows).sort_values(["seg", "suf"]).reset_index(drop=True)
    fr["orden"] = np.arange(len(fr))
    base = pd.Timestamp(fr["fecha"].iloc[0], tz=tz)
    fr["timestamp"] = [(base + pd.Timedelta(seconds=int(sg)) + pd.Timedelta(milliseconds=k)).tz_convert("UTC") for k, sg in enumerate(fr["seg"])]
    fr["t_norm"] = fr["orden"] / max(len(fr) - 1, 1)
    return fr.drop(columns=["seg", "suf"])


def stage(fr, prueba, keep_names):
    """Copia las fotos a data/campo/fotos/<prueba>/ con nombre único y extensión .jpg."""
    dest = ensure_dir(FOTOS / prueba)
    paths, ids = [], []
    for k, src in zip(fr["orden"], fr["path_original"]):
        src = Path(src)
        name = f"{src.stem}.jpg" if keep_names else f"{prueba}_{int(k) + 1:03d}.jpg"
        tgt = dest / name
        if not tgt.exists() or tgt.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, tgt)
        paths.append(str(tgt))
        ids.append(tgt.stem)
    fr = fr.copy()
    fr["path"], fr["image_id"] = paths, ids
    fr["archivo_original"] = [str(Path(p).relative_to(RAW)) for p in fr["path_original"]]
    fr["hash_imagen"] = [hashlib.sha1(Path(p).read_bytes()).hexdigest() for p in fr["path"]]
    return fr.drop(columns=["path_original"])


def frames_of(folder: Path, tz):
    rows = []
    for p in sorted(folder.glob("*.jpg")):
        m = IMG_RE.search(p.name)
        if not m:
            continue
        d, t, ms, n = m.groups()
        local = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}.{ms}", tz=tz)
        rows.append({"path_original": p, "orden": int(n), "timestamp": local.tz_convert("UTC")})
    return pd.DataFrame(rows).sort_values("orden").reset_index(drop=True)


def cut_tramos(dist_m: np.ndarray, tramo_m: float, min_fotos: int) -> np.ndarray:
    """Índice de tramo por foto a partir de la distancia acumulada; el último se funde si es corto."""
    d = np.asarray(dist_m, dtype=float) - np.nanmin(dist_m)
    tr = np.floor(d / tramo_m).astype(int)
    # renumerar consecutivo (puede haber saltos si un tramo quedó sin fotos)
    _, tr = np.unique(tr, return_inverse=True)
    if tr.max() > 0 and (tr == tr.max()).sum() < min_fotos:
        tr[tr == tr.max()] = tr.max() - 1
    return tr


def _tramo_row(g, vid, prueba, k, fecha, tz, id_prueba, sitio, id_sitio):
    lat_c, lon_c = float(g["lat"].mean()), float(g["lon"].mean())
    geo = np.isfinite(lat_c)
    return {
        "id_via": vid, "id_prueba": id_prueba, "prueba": prueba, "tramo": int(k), "fecha": fecha, "sitio": sitio, "id_sitio": id_sitio,
        "superficie": g["superficie"].iloc[0], "conjunto": g["conjunto"].iloc[0], "ubicacion_fuente": g["ubicacion_fuente"].iloc[0],
        "calcodigo_idu": g["calcodigo_idu"].iloc[0],
        "lat": lat_c, "lon": lon_c, "lat_ini": float(g["lat"].iloc[0]), "lon_ini": float(g["lon"].iloc[0]),
        "lat_fin": float(g["lat"].iloc[-1]), "lon_fin": float(g["lon"].iloc[-1]), "geohash7": encode(lat_c, lon_c, 7) if geo else None,
        "longitud_m": float(g["dist_acum_m"].max() - g["dist_acum_m"].min()) if g["dist_acum_m"].notna().any() else np.nan,
        "n_fotogramas": int(len(g)), "hora_ini": g["timestamp"].min().tz_convert(tz).strftime("%H:%M:%S"),
        "hora_fin": g["timestamp"].max().tz_convert(tz).strftime("%H:%M:%S"),
        "gps_incertidumbre_m": float(g["gps_incertidumbre_m"].max()) if geo else np.nan,
        "hash_imagenes": hashlib.sha1("".join(sorted(g["hash_imagen"])).encode()).hexdigest()[:12],
    }


def build():
    cfg = load_cfg()
    tz = cfg["zona_horaria"]
    frames, tramos = [], []
    shutil.rmtree(FOTOS, ignore_errors=True)
    cal = cfg.get("calibracion") or {}
    todas = [(p, s, "evaluacion") for p, s in cfg["pruebas"].items()]
    if cal.get("usar"):
        todas += [(p, s, "calibracion") for p, s in (cal.get("pruebas") or {}).items()]
    sitios = cfg.get("sitios") or {}
    for prueba, spec, conjunto in todas:
        folder = RAW / spec["fotos"]
        gpx = sorted(folder.glob("*.gpx"))
        sitio = spec.get("sitio")
        con_gps = bool(gpx)
        fr = frames_of(folder, tz) if con_gps else frames_sin_gps(folder, tz)
        if fr.empty:
            print(f"[aviso] {folder.name}: sin fotos reconocibles, se omite")
            continue
        fr = stage(fr, prueba, keep_names=con_gps)
        fecha = fr["timestamp"].iloc[0].tz_convert(tz).strftime("%Y-%m-%d")
        if con_gps:
            track = smooth_track(read_gpx(gpx[0]), cfg["gps_suavizado_s"])
            pos = interpolate(track, fr["timestamp"])
            fr = pd.concat([fr.drop(columns=[c for c in ("fecha",) if c in fr]), pos.drop(columns=["timestamp"])], axis=1)
            fr["vel_ms"] = fr.pop("speed")
            fr["gps_incertidumbre_m"] = np.where(fr["sat"] >= cfg["gps_sat_min"], cfg["gps_incertidumbre_m"], 2 * cfg["gps_incertidumbre_m"])
            t0, t1 = fr["timestamp"].min(), fr["timestamp"].max()
            fr["t_norm"] = (fr["timestamp"] - t0) / max((t1 - t0), pd.Timedelta(seconds=1))
            fr["tramo"] = cut_tramos(fr["dist_acum_m"].to_numpy(), cfg["tramo_m"], cfg["tramo_min_fotos"])
        else:
            for c in GEO_COLS:
                fr[c] = np.nan
            fr["tramo"] = 0  # sin posición no hay distancia para cortar: la prueba completa es el tramo
            loc = sitios.get(sitio) or {}
            if "lat" in loc:  # ubicación declarada por el equipo y anclada a la calzada IDU (no es una medición GPS)
                fr["lat"], fr["lon"], fr["gps_incertidumbre_m"] = loc["lat"], loc["lon"], float(loc.get("incertidumbre_m", np.nan))
        fr["conjunto"] = conjunto
        fr["ubicacion_fuente"] = "gps" if con_gps else ("declarada" if fr["lat"].notna().all() else "sin_ubicacion")
        fr["calcodigo_idu"] = (sitios.get(sitio) or {}).get("calcodigo_idu") if not con_gps else None
        fr["prueba"], fr["fecha"], fr["id_prueba"] = prueba, fecha, prueba_id(fecha, prueba)
        fr["con_gps"] = con_gps
        fr["superficie"] = spec.get("superficie", "asfalto")
        fr["sitio"] = sitio if sitio is not None else np.nan
        fr["id_sitio"] = sitio_id(fecha, sitio) if sitio is not None else None
        for k, g in fr.groupby("tramo"):
            if con_gps:
                vid = via_id(fecha, prueba, k, float(g["lat"].mean()), float(g["lon"].mean()))
            else:
                vid = via_id_sin_gps(fecha, prueba, k, hashlib.sha1("".join(sorted(g["hash_imagen"])).encode()).hexdigest()[:12])
            fr.loc[g.index, "id_via"] = vid
            tramos.append(_tramo_row(g, vid, prueba, k, fecha, tz, fr["id_prueba"].iloc[0], fr["sitio"].iloc[0], fr["id_sitio"].iloc[0]))
        modo = "GPS real" if con_gps else ("sin GPS, sitio declarado" if fr["ubicacion_fuente"].iloc[0] == "declarada" else "sin GPS")
        print(f"[tramos] {prueba} ({conjunto}): {len(fr)} fotos, {fr['tramo'].nunique()} tramo(s), {modo}, {fr['superficie'].iloc[0]}", flush=True)
        frames.append(fr)
    frames = pd.concat(frames, ignore_index=True)
    tramos = pd.DataFrame(tramos)
    ensure_dir(INDEX_CAMPO)
    frames.to_parquet(INDEX_CAMPO / "fotogramas.parquet", index=False)
    tramos.to_parquet(INDEX_CAMPO / "tramos.parquet", index=False)
    (INDEX_CAMPO / "campo.txt").write_text("\n".join(sorted(frames["path"])) + "\n")  # mismo orden (y lotes) que el kernel de Kaggle
    return frames, tramos


def main():
    frames, tramos = build()
    print(f"[ok] {len(frames)} fotos en {len(tramos)} tramos de {frames['prueba'].nunique()} pruebas -> {INDEX_CAMPO}")
    print(tramos[["id_via", "conjunto", "prueba", "tramo", "sitio", "ubicacion_fuente", "longitud_m", "n_fotogramas", "lat", "lon"]].to_string(index=False))


if __name__ == "__main__":
    main()
