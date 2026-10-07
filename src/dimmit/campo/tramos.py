"""Fotogramas de campo <-> GPS interpolado <-> tramos de ~10 m <-> id_via determinista.

Salidas (data/campo/index/):
  fotogramas.parquet  una fila por foto: prueba, id_prueba, id_via, tramo, timestamp, lat, lon,
                      sat, vel_ms, gps_dt_s, gps_incertidumbre_m, dist_acum_m, t_norm, hash_imagen
  tramos.parquet      una fila por tramo: id_via, prueba, tramo, centroide, extremos, geohash7,
                      longitud_m, n_fotogramas, hora_ini, hora_fin
  campo.txt           lista de rutas para el detector
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dimmit.campo import INDEX_CAMPO, RAW, load_cfg
from dimmit.campo.gpx import interpolate, read_gpx, smooth_track
from dimmit.geo.geohash import encode, haversine_m
from dimmit.utils.io import ensure_dir

IMG_RE = re.compile(r"captura_(\d{8})_(\d{6})_(\d{3})_(\d+)\.jpg$")
PRUEBA_RE = re.compile(r"Prueba[_ ]?(\d+)", re.I)


def via_id(fecha, prueba, tramo, lat_c, lon_c):
    """ID determinista: mismo dato -> mismo ID (fecha, prueba, tramo y centroide a 1e-5 grados ~ 1 m)."""
    key = f"{fecha}|{prueba}|{int(tramo):02d}|{lat_c:.5f}|{lon_c:.5f}"
    return "VIA-" + hashlib.sha1(key.encode()).hexdigest()[:12]


def prueba_id(fecha, prueba):
    return "PRU-" + hashlib.sha1(f"{fecha}|{prueba}".encode()).hexdigest()[:8]


def frames_of(folder: Path, tz):
    rows = []
    for p in sorted(folder.glob("*.jpg")):
        m = IMG_RE.search(p.name)
        if not m:
            continue
        d, t, ms, n = m.groups()
        local = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}.{ms}", tz=tz)
        rows.append({"image_id": p.stem, "path": str(p), "orden": int(n), "timestamp": local.tz_convert("UTC"),
                     "hash_imagen": hashlib.sha1(p.read_bytes()).hexdigest()})
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


def build():
    cfg = load_cfg()
    tz = cfg["zona_horaria"]
    frames, tramos = [], []
    for folder in sorted(RAW.glob("grabacion_*")):
        m = PRUEBA_RE.search(folder.name)
        prueba = f"Prueba_{m.group(1)}" if m else folder.name
        gpx = sorted(folder.glob("*.gpx"))
        fr = frames_of(folder, tz)
        if fr.empty or not gpx:
            print(f"[aviso] {folder.name}: sin fotos o sin GPX, se omite")
            continue
        fecha = fr["timestamp"].iloc[0].tz_convert(tz).strftime("%Y-%m-%d")
        track = smooth_track(read_gpx(gpx[0]), cfg["gps_suavizado_s"])
        pos = interpolate(track, fr["timestamp"])
        fr = pd.concat([fr, pos.drop(columns=["timestamp"])], axis=1)
        fr["vel_ms"] = fr.pop("speed")
        fr["gps_incertidumbre_m"] = np.where(fr["sat"] >= cfg["gps_sat_min"], cfg["gps_incertidumbre_m"], 2 * cfg["gps_incertidumbre_m"])
        fr["prueba"], fr["fecha"], fr["id_prueba"] = prueba, fecha, prueba_id(fecha, prueba)
        t0, t1 = fr["timestamp"].min(), fr["timestamp"].max()
        fr["t_norm"] = (fr["timestamp"] - t0) / max((t1 - t0), pd.Timedelta(seconds=1))
        fr["tramo"] = cut_tramos(fr["dist_acum_m"].to_numpy(), cfg["tramo_m"], cfg["tramo_min_fotos"])
        for k, g in fr.groupby("tramo"):
            lat_c, lon_c = float(g["lat"].mean()), float(g["lon"].mean())
            vid = via_id(fecha, prueba, k, lat_c, lon_c)
            fr.loc[g.index, "id_via"] = vid
            tramos.append({
                "id_via": vid, "id_prueba": fr["id_prueba"].iloc[0], "prueba": prueba, "tramo": int(k), "fecha": fecha,
                "lat": lat_c, "lon": lon_c, "lat_ini": float(g["lat"].iloc[0]), "lon_ini": float(g["lon"].iloc[0]),
                "lat_fin": float(g["lat"].iloc[-1]), "lon_fin": float(g["lon"].iloc[-1]), "geohash7": encode(lat_c, lon_c, 7),
                "longitud_m": float(g["dist_acum_m"].max() - g["dist_acum_m"].min()),
                "n_fotogramas": int(len(g)), "hora_ini": g["timestamp"].min().tz_convert(tz).strftime("%H:%M:%S"),
                "hora_fin": g["timestamp"].max().tz_convert(tz).strftime("%H:%M:%S"),
                "gps_incertidumbre_m": float(g["gps_incertidumbre_m"].max()),
                "hash_imagenes": hashlib.sha1("".join(sorted(g["hash_imagen"])).encode()).hexdigest()[:12],
            })
        frames.append(fr)
    frames = pd.concat(frames, ignore_index=True)
    tramos = pd.DataFrame(tramos)
    ensure_dir(INDEX_CAMPO)
    frames.to_parquet(INDEX_CAMPO / "fotogramas.parquet", index=False)
    tramos.to_parquet(INDEX_CAMPO / "tramos.parquet", index=False)
    (INDEX_CAMPO / "campo.txt").write_text("\n".join(frames["path"]) + "\n")
    return frames, tramos


def main():
    frames, tramos = build()
    print(f"[ok] {len(frames)} fotos en {len(tramos)} tramos de {frames['prueba'].nunique()} pruebas -> {INDEX_CAMPO}")
    print(tramos[["id_via", "prueba", "tramo", "longitud_m", "n_fotogramas", "lat", "lon"]].to_string(index=False))


if __name__ == "__main__":
    main()
