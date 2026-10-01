"""Geohash (codificador propio) y distancias geodésicas."""
import numpy as np

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
EARTH_R = 6_371_000.0


def encode(lat: float, lon: float, precision: int = 7) -> str:
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    out, bit, ch, even = [], 0, 0, True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                ch, lon_lo = ch | (1 << (4 - bit)), mid
            else:
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch, lat_lo = ch | (1 << (4 - bit)), mid
            else:
                lat_hi = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_BASE32[ch])
            bit, ch = 0, 0
    return "".join(out)


def decode(gh: str):
    """Centro (lat, lon) de una celda geohash."""
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for c in gh:
        v = _BASE32.index(c)
        for b in range(4, -1, -1):
            bitv = (v >> b) & 1
            if even:
                mid = (lon_lo + lon_hi) / 2
                lon_lo, lon_hi = (mid, lon_hi) if bitv else (lon_lo, mid)
            else:
                mid = (lat_lo + lat_hi) / 2
                lat_lo, lat_hi = (mid, lat_hi) if bitv else (lat_lo, mid)
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


def encode_many(lat, lon, precision=7):
    return [encode(a, b, precision) for a, b in zip(lat, lon)]


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def mercator_to_wgs84(x, y):
    """EPSG:3857 -> (lat, lon)."""
    lon = np.degrees(np.asarray(x) / EARTH_R)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y) / EARTH_R)) - np.pi / 2)
    return lat, lon
