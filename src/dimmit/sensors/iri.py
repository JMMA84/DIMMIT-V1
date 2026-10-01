"""IRI estándar (Golden Car, Sayers 1995) y modelo de cuarto de vehículo en tiempo.

Ambos se discretizan exactamente (retención de orden cero) y se filtran con lfilter.
"""
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter, ss2tf

# parámetros Golden Car normalizados por la masa suspendida
K1, K2, C, MU = 653.0, 63.3, 6.0, 0.15
V_IRI = 80 / 3.6
DX_IRI = 0.25


def _zoh(A, B, dt):
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n], M[:n, n] = A * dt, B * dt
    E = expm(M)
    return E[:n, :n], E[:n, n]


@lru_cache(maxsize=1)
def _golden_tf():
    A = np.array([[0, 1, 0, 0], [-K2, -C, K2, C], [0, 0, 0, 1], [K2 / MU, C / MU, -(K1 + K2) / MU, -C / MU]])
    B = np.array([0, 0, 0, K1 / MU])
    Ad, Bd = _zoh(A, B, DX_IRI / V_IRI)
    num, den = ss2tf(Ad, Bd[:, None], np.array([[0, 1, 0, -1]]), np.zeros((1, 1)))
    return num[0], den


def golden_car_slope(profile, dx):
    """Pendiente rectificada |zs'-zu'|/V a lo largo del perfil (m/m), a paso DX_IRI.

    El perfil se suaviza con media móvil de 250 mm y se remuestrea a 0.25 m.
    """
    profile = np.asarray(profile, dtype=float)
    k = max(1, int(round(0.25 / dx)))
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    x = np.arange(len(profile)) * dx
    xs = np.arange(0, x[-1], DX_IRI)
    y = np.interp(xs, x, smooth)
    y = y - y[0]
    num, den = _golden_tf()
    v_rel = lfilter(num, den, y)
    return xs, np.abs(v_rel) / V_IRI


def iri(profile, dx, start_m=11.0):
    """IRI (m/km) de un perfil, descartando los primeros start_m de inicialización."""
    xs, s = golden_car_slope(profile, dx)
    m = xs >= start_m
    return 1000.0 * float(s[m].mean()) if m.any() else float("nan")


def quarter_car_tf(ms, mu, ks, cs, kt, dt):
    """Funciones de transferencia discretas perfil -> (aceleración, velocidad) de la carrocería."""
    A = np.array([[0, 1, 0, 0], [-ks / ms, -cs / ms, ks / ms, cs / ms], [0, 0, 0, 1], [ks / mu, cs / mu, -(ks + kt) / mu, -cs / mu]])
    B = np.array([0, 0, 0, kt / mu])
    Ad, Bd = _zoh(A, B, dt)
    # aceleración = fila 2 de A aplicada al estado (sin término directo); velocidad = estado 2
    num_a, den = ss2tf(Ad, Bd[:, None], A[1:2, :], np.zeros((1, 1)))
    num_v, _ = ss2tf(Ad, Bd[:, None], np.array([[0, 1, 0, 0]]), np.zeros((1, 1)))
    return num_a[0], num_v[0], den


def body_response(road_t, params, dt):
    """(aceleración, velocidad) vertical de la carrocería ante el camino en el tiempo."""
    num_a, num_v, den = quarter_car_tf(params["ms"], params["mu"], params["ks"], params["cs"], params["kt"], dt)
    r = np.asarray(road_t) - road_t[0]
    return lfilter(num_a, den, r), lfilter(num_v, den, r)
