"""Perfiles longitudinales de vía según ISO 8608 (síntesis espectral)."""
from functools import lru_cache

import numpy as np

from dimmit.sensors.iri import iri

N0 = 0.1  # ciclos/m


def iso8608_profile(n_points, dx, gd_n0, rng, w=2.0, nmin=0.01, nmax=10.0):
    """Perfil aleatorio con PSD Gd(n) = Gd(n0) (n/n0)^-w entre nmin y nmax."""
    n_fft = int(2 ** np.ceil(np.log2(max(n_points, 16))))
    freqs = np.fft.rfftfreq(n_fft, d=dx)
    dn = freqs[1] - freqs[0]
    band = (freqs >= nmin) & (freqs <= nmax)
    amp = np.zeros_like(freqs)
    amp[band] = np.sqrt(2 * gd_n0 * (freqs[band] / N0) ** (-w) * dn)
    phase = rng.uniform(0, 2 * np.pi, len(freqs))
    spec = amp * np.exp(1j * phase) * n_fft / 2
    z = np.fft.irfft(spec, n=n_fft)[:n_points]
    return z


@lru_cache(maxsize=1)
def iri_per_sqrt_gd(dx=0.05):
    """Constante de calibración: IRI = c * sqrt(Gd(n0)) (IRI es lineal en la amplitud)."""
    rng = np.random.default_rng(0)
    gd = 16e-6
    vals = [iri(iso8608_profile(int(400 / dx), dx, gd, rng), dx) for _ in range(12)]
    return float(np.mean(vals) / np.sqrt(gd))


def gd_for_iri(target_iri, dx=0.05):
    return (np.asarray(target_iri) / iri_per_sqrt_gd(dx)) ** 2


def cosine_envelope(x, knots_x, knots_val, taper=2.0):
    """Envolvente por tramos con transiciones coseno de `taper` m entre tramos."""
    env = np.interp(x, knots_x, knots_val)
    k = max(1, int(taper / max(x[1] - x[0], 1e-6)))
    win = np.hanning(2 * k + 1)
    return np.convolve(np.pad(env, k, mode="edge"), win / win.sum(), mode="valid")
