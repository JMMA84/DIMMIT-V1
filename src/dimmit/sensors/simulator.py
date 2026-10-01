"""Simulador físico de IMU (MPU-6050 vía Arduino) + GPS para un segmento de vía.

Cadena: rugosidad latente por fotograma -> perfil ISO 8608 en dos rodadas (coherencia 0.8) con
eventos (baches de las cajas D40, parches de piel de cocodrilo D20, resaltos y tapas de
alcantarilla como confusores) -> IRI real (Golden Car) -> vehículo de 4 esquinas (cuarto de
vehículo por esquina, ejes separados por la distancia entre ejes) -> montaje del sensor
(resonancia, inclinación, ruido, sesgo, cuantización, saturación) -> 50 Hz.
La verdad (IRI latente/real, impactos, parámetros) se devuelve aparte y no va al parquet de
sensores.
"""
import numpy as np
import pandas as pd
from scipy.signal import bilinear, lfilter, resample_poly

from dimmit.geo.geohash import EARTH_R
from dimmit.labels import pseudo_pci as pp
from dimmit.sensors.iri import body_response, golden_car_slope
from dimmit.sensors.profile import cosine_envelope, gd_for_iri, iso8608_profile

G = 9.81


def box_distance_m(cy):
    """Distancia aproximada desde la cámara a una caja según su altura en la imagen."""
    cy = np.asarray(cy, dtype=float)
    return np.where(cy > 0.4, 3 + 22 * ((1 - cy) / 0.6) ** 2, 25.0)


def _dip(x, x0, length, depth):
    m = (x >= x0) & (x <= x0 + length)
    out = np.zeros_like(x)
    out[m] = -depth * 0.5 * (1 - np.cos(2 * np.pi * (x[m] - x0) / length))
    return out


def _mount_filter(fs, f0, zeta):
    w = 2 * np.pi * f0
    b, a = bilinear([2 * zeta * w, w**2], [1, 2 * zeta * w, w**2], fs)
    return b, a


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def simulate_segment(road, frames, frame_boxes, cfg, rng, start_time, coupling=1.0):
    """Simula un segmento.

    road: fila con lat_a, lon_a, lat_b, lon_b, longitud_m.
    frames: DataFrame ordenado (image_id, pci, country) -> pci de referencia de cada fotograma.
    frame_boxes: dict image_id -> DataFrame de cajas (cls, cx, cy, w, h, sev) que generan eventos.
    coupling: 1 = mundo normal; 0 = "mundo nulo" (sin relación entre daño visual y rugosidad).
    """
    lat_cfg, ev, veh, imu = cfg["latent"], cfg["events"], cfg["vehicle"], cfg["imu"]
    dx, cam, lead = cfg["dx_profile_m"], cfg["camera_lookahead_m"], cfg["lead_m"]
    n = len(frames)
    L = float(road["longitud_m"])
    step = L / n
    s = np.arange(n) * step

    # rugosidad latente por fotograma: acoplamiento parcial con el PCI visual + ruido
    country = frames["country"].iloc[0]
    u = rng.normal(0, lat_cfg["segment_sigma"])
    ar = np.zeros(n)
    for k in range(n):
        ar[k] = (lat_cfg["frame_ar_phi"] * ar[k - 1] if k else 0) + rng.normal(0, lat_cfg["frame_ar_sigma"])
    pci = frames["pci"].to_numpy(dtype=float)
    ln_iri = lat_cfg["ln_iri_base"] + coupling * lat_cfg["pci_coupling"] * (1 - pci / 100) + lat_cfg["country_offset"].get(country, 0) + u + ar
    iri_latent = np.exp(ln_iri)

    x = np.arange(-(lead + cam + 2), L + lead + 25, dx)
    env = cosine_envelope(x, s + step / 2, iri_latent, taper=2.0)
    gd1 = float(gd_for_iri(1.0, dx))
    rho = veh["track_coherence"]
    base_l = iso8608_profile(len(x), dx, gd1, rng)
    base_r = rho * base_l + np.sqrt(1 - rho**2) * iso8608_profile(len(x), dx, gd1, rng)
    zl, zr = env * base_l, env * base_r

    # eventos ligados al daño visual (solo si hay acoplamiento)
    hits = 0
    for k, img in enumerate(frames["image_id"]):
        b = frame_boxes.get(img)
        if b is None or len(b) == 0 or coupling == 0:
            continue
        pos = s[k] - cam + box_distance_m(b["cy"].to_numpy())
        for (_, bx), p in zip(b.iterrows(), pos):
            if bx["cls"] == "D20" and rng.random() < ev["alligator_prob"]:
                length = rng.uniform(*ev["alligator_length_m"])
                gain = rng.uniform(*ev["alligator_gain"])
                m = (x >= p - length / 2) & (x <= p + length / 2)
                zl[m] *= gain
                zr[m] *= gain
            elif bx["cls"] == "D40":
                prob = ev["pothole_hit_prob_center"] if abs(bx["cx"] - 0.5) <= 0.3 else ev["pothole_hit_prob_side"]
                if rng.random() < prob:
                    dip = _dip(x, p, rng.uniform(*ev["pothole_length_m"]), ev["pothole_depth_m"][bx["sev"]])
                    if bx["cx"] < 0.5:
                        zl += dip
                    else:
                        zr += dip
                    hits += 1

    # IRI real (Golden Car) por rodada, promedio de ambas
    xs, sl = golden_car_slope(zl, dx)
    _, sr = golden_car_slope(zr, dx)
    slope = (sl + sr) / 2
    xs = xs + x[0]
    iri_frame = np.array([1000 * slope[(xs >= a) & (xs < a + step)].mean() for a in s])
    iri_seg = 1000 * slope[(xs >= 0) & (xs < L)].mean()

    # confusores fuera del IRI: resalto (policía acostado) y tapas de alcantarilla
    ml, mr = zl.copy(), zr.copy()
    bump_x = None
    if rng.random() < ev["speed_bump_prob"]:
        bump_x = rng.uniform(0.2 * L, 0.8 * L)
        hb, lb = ev["speed_bump"]["height_m"], ev["speed_bump"]["length_m"]
        bump = -_dip(x, bump_x, lb, hb)
        ml += bump
        mr += bump
    for _ in range(rng.poisson(L / 100 * ev["manhole_per_100m"])):
        if rng.random() < ev["manhole_hit_prob"]:
            d = _dip(x, rng.uniform(0, L), 0.7, ev["manhole_amp_m"] * rng.choice([-1, 1]))
            (ml if rng.random() < 0.5 else mr)[:] += d

    # envolvente del neumático
    k_patch = max(1, int(veh["tire_patch_m"] / dx))
    kern = np.ones(k_patch) / k_patch
    ml, mr = np.convolve(ml, kern, mode="same"), np.convolve(mr, kern, mode="same")

    # cinemática a 500 Hz: velocidad OU alrededor de la crucero, frenado en el resalto
    fs = cfg["hz_internal"]
    dt = 1 / fs
    v0 = rng.uniform(*cfg["speed_kmh"]) / 3.6
    th, sig = cfg["speed_ou"]["theta"], cfg["speed_ou"]["sigma_kmh"] / 3.6
    x_start, x_end = -(lead + cam), L + lead
    pos, vel, ou, v = [x_start], [], 0.0, v0
    while pos[-1] < x_end:
        ou += -th * ou * dt + sig * np.sqrt(2 * th * dt) * rng.normal()
        target = v0 + ou
        if bump_x is not None and abs(pos[-1] - bump_x) < 12:
            target = min(target, ev["speed_bump"]["speed_kmh"] / 3.6 + abs(pos[-1] - bump_x) * 0.4)
        v += (max(1.0, target) - v) * dt / 1.0  # inercia del vehículo (constante de 1 s)
        vel.append(v)
        pos.append(pos[-1] + v * dt)
    xv = np.array(pos[:-1])
    v = np.array(vel)
    t = np.arange(len(xv)) * dt

    par = {
        "ms": rng.uniform(*veh["ms_kg"]),
        "mu": rng.uniform(*veh["mu_kg"]),
        "ks": rng.uniform(*veh["ks_n_m"]),
        "cs": rng.uniform(*veh["cs_ns_m"]),
        "kt": rng.uniform(*veh["kt_n_m"]),
    }
    wb, tw = veh["wheelbase_m"], veh["track_m"]
    corners = {}
    for name, prof, off in (("fl", ml, 0.0), ("fr", mr, 0.0), ("rl", ml, -wb), ("rr", mr, -wb)):
        road_t = np.interp(xv + off, x, prof)
        corners[name] = body_response(road_t, par, dt)
    heave = np.mean([c[0] for c in corners.values()], axis=0)
    vel_c = {k: c[1] for k, c in corners.items()}
    roll_rate = ((vel_c["fl"] + vel_c["rl"]) - (vel_c["fr"] + vel_c["rr"])) / (2 * tw)
    pitch_rate = ((vel_c["fl"] + vel_c["fr"]) - (vel_c["rl"] + vel_c["rr"])) / (2 * wb)
    a_long = np.gradient(v, dt)
    sway_a, sway_T = rng.uniform(0.05, 0.25), rng.uniform(4, 8)
    a_lat = sway_a * np.sin(2 * np.pi * t / sway_T + rng.uniform(0, 2 * np.pi))
    yaw_rate = a_lat / np.maximum(v, 1.0)

    # montaje: resonancia, orientación arbitraria, ruido, sesgo, cuantización, saturación
    b, a = _mount_filter(fs, rng.uniform(*imu["mount_freq_hz"]), rng.uniform(*imu["mount_damping"]))
    heave_m = lfilter(b, a, heave)
    tilt = np.radians(imu["tilt_deg"])
    R = _rot(rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(0, 2 * np.pi))
    acc = R @ np.vstack([a_long, a_lat, G + heave_m])
    gyr = R @ np.vstack([roll_rate, pitch_rate, yaw_rate])
    acc += rng.normal(0, rng.uniform(*imu["accel_noise_ms2"]), acc.shape) + rng.normal(0, imu["accel_bias_sigma"], (3, 1))
    gyr += rng.normal(0, imu["gyro_noise_rads"], gyr.shape)
    lsb = G / imu["lsb_per_g"]
    acc = np.clip(np.round(acc / lsb) * lsb, -imu["accel_range_g"] * G, imu["accel_range_g"] * G)
    dec = fs // cfg["hz_output"]
    acc_o = resample_poly(acc, 1, dec, axis=1, padtype="line")
    gyr_o = resample_poly(gyr, 1, dec, axis=1, padtype="line")
    acc_o = np.clip(acc_o, -imu["accel_range_g"] * G, imu["accel_range_g"] * G)  # saturación en la salida del sensor
    t_o = t[::dec][: acc_o.shape[1]]
    xv_o = xv[::dec][: acc_o.shape[1]]
    v_o = v[::dec][: acc_o.shape[1]]

    # GPS a 1 Hz con error AR(1), retenido hasta la siguiente lectura
    lat_a, lon_a, lat_b, lon_b = road["lat_a"], road["lon_a"], road["lat_b"], road["lon_b"]
    east = np.radians(lon_b - lon_a) * EARTH_R * np.cos(np.radians(lat_a))
    north = np.radians(lat_b - lat_a) * EARTH_R
    norm = np.hypot(east, north) or 1.0
    ue, un = east / norm, north / norm
    gps_idx = (t_o // 1.0).astype(int)
    n_fix = gps_idx.max() + 1
    err = np.zeros((n_fix, 2))
    for k in range(1, n_fix):
        err[k] = imu["gps_ar_phi"] * err[k - 1] + rng.normal(0, imu["gps_sigma_m"] * np.sqrt(1 - imu["gps_ar_phi"] ** 2), 2)
    fix_t = np.arange(n_fix) * 1.0
    fix_x = np.interp(fix_t, t_o, xv_o)
    fe, fn = fix_x * ue + err[:, 0], fix_x * un + err[:, 1]
    fix_lat = lat_a + np.degrees(fn / EARTH_R)
    fix_lon = lon_a + np.degrees(fe / (EARTH_R * np.cos(np.radians(lat_a))))
    fix_speed = np.interp(fix_t, t_o, v_o) * 3.6 + rng.normal(0, 0.5, n_fix)

    ts = pd.Timestamp(start_time) + pd.to_timedelta(t_o, unit="s")
    t_cap = np.interp(s - cam, xv_o, t_o)
    win = cfg["imu_window_s"]
    obs = np.full(len(t_o), -1)
    for k in range(n):
        obs[(t_o >= t_cap[k] + win[0]) & (t_o < t_cap[k] + win[1]) & (obs < 0)] = k
    ids = frames["image_id"].to_numpy()
    rows = pd.DataFrame(
        {
            "segment_id": road["segment_id"],
            "obs_id": np.where(obs >= 0, ids[np.clip(obs, 0, n - 1)], ""),
            "timestamp": ts,
            "ax": acc_o[0],
            "ay": acc_o[1],
            "az": acc_o[2],
            "gx": gyr_o[0],
            "gy": gyr_o[1],
            "gz": gyr_o[2],
            "lat": fix_lat[gps_idx],
            "lon": fix_lon[gps_idx],
            "speed_kmh": fix_speed[gps_idx],
        }
    )
    cap_fix = np.minimum((t_cap // 1.0).astype(int), n_fix - 1)
    frames_meta = pd.DataFrame(
        {
            "image_id": ids,
            "segment_id": road["segment_id"],
            "timestamp_captura": pd.Timestamp(start_time) + pd.to_timedelta(t_cap, unit="s"),
            "lat": fix_lat[cap_fix],
            "lon": fix_lon[cap_fix],
        }
    )
    truth = pd.DataFrame(
        {
            "image_id": ids,
            "segment_id": road["segment_id"],
            "iri_latente": iri_latent,
            "iri_real_fotograma": iri_frame,
            "iri_real_segmento": iri_seg,
            "pci_referencia": pci,
            "baches_impactados_segmento": hits,
            "resalto": bump_x is not None,
            "v0_kmh": v0 * 3.6,
            "acoplamiento": coupling,
            **{f"veh_{k}": val for k, val in par.items()},
        }
    )
    return rows, frames_meta, truth


def severity_labels(boxes, cfg_lab, cuts):
    """L/M/H por caja con los mismos cortes del pseudo-PCI."""
    s = pp.size_proxy(boxes, cfg_lab)
    sev = np.full(len(boxes), "M", dtype=object)
    cls = boxes["cls"].to_numpy()
    for c, (lo, hi) in cuts.items():
        m = cls == c
        sev[m & (s < lo)] = "L"
        sev[m & (s >= hi)] = "H"
    return sev
