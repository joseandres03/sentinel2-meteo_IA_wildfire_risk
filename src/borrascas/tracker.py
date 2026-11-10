"""
Seguimiento (tracking) del centro de una borrasca a partir de campos de presión
a nivel del mar (MSLP).

Algoritmo (sencillo y robusto, del estilo de los trackers operativos):
  1. Se suaviza ligeramente el campo para quitar ruido de rejilla.
  2. En t=0 se busca el mínimo local más profundo cerca de la posición semilla.
  3. En cada paso siguiente se extrapola la posición con la velocidad reciente
     y se busca el mínimo local dentro de un radio de búsqueda alrededor de la
     posición prevista. Se elige el que minimiza  presión + penalización por
     distancia, para no "saltar" a otra borrasca más profunda.
  4. Si no hay mínimo local cerrado dentro del radio, la borrasca se da por
     disipada / absorbida y la trayectoria termina.

Trabaja sólo con numpy/scipy, así que es independiente del modelo de origen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class TrackerParams:
    seed_radius_km: float = 450.0     # radio para encontrar la borrasca en t=0
    max_speed_kmh: float = 70.0       # velocidad máxima plausible del centro
    min_search_km: float = 250.0      # radio mínimo de búsqueda por paso
    distance_penalty_hpa_per_100km: float = 1.0
    min_depth_hpa: float = 1.0        # profundidad mínima respecto al entorno (~400 km)
    smooth_sigma_km: float = 50.0     # suavizado gaussiano
    local_min_window: int = 5         # ventana (puntos) para detectar mínimos locales
    max_pressure_hpa: float = 1020.0  # por encima de esto no lo consideramos borrasca
    min_laplacian: float = 0.8        # plan B: laplaciano mínimo (hPa / (100 km)^2) para aceptar una baja abierta


@dataclass
class TrackPoint:
    step: int          # horas desde la inicialización
    lat: float
    lon: float
    p: float           # presión mínima (hPa)

    def as_dict(self):
        return {"h": self.step, "lat": round(self.lat, 2),
                "lon": round(self.lon, 2), "p": round(self.p, 1)}


@dataclass
class Grid:
    lats: np.ndarray   # 1D, cualquier orden
    lons: np.ndarray   # 1D, en -180..180
    _ll: tuple = field(default=None, repr=False)

    @property
    def mesh(self):
        if self._ll is None:
            self._ll = np.meshgrid(self.lats, self.lons, indexing="ij")
        return self._ll

    @property
    def dlat_km(self):
        return abs(float(self.lats[1] - self.lats[0])) * 111.2


def _prepare(field_hpa: np.ndarray, grid: Grid, prm: TrackerParams):
    sigma_pts = max(prm.smooth_sigma_km / grid.dlat_km, 0.0)
    sm = gaussian_filter(field_hpa, sigma=sigma_pts, mode="nearest") if sigma_pts > 0.3 else field_hpa
    is_min = sm == minimum_filter(sm, size=prm.local_min_window, mode="nearest")
    # descartar el borde del dominio (mínimos falsos por recorte)
    b = prm.local_min_window // 2 + 1
    is_min[:b, :] = is_min[-b:, :] = False
    is_min[:, :b] = is_min[:, -b:] = False
    return sm, is_min


def _depth(sm, grid: Grid, i, j, radius_km=400.0):
    """Profundidad del mínimo: media de un anillo a ~radius_km menos el centro."""
    la, lo = grid.mesh
    d = haversine_km(la[i, j], lo[i, j], la, lo)
    ring = (d > radius_km * 0.8) & (d < radius_km * 1.2)
    if not ring.any():
        return 0.0
    return float(sm[ring].mean() - sm[i, j])


def _refine(sm, grid: Grid, i, j):
    """Refinamiento sub-rejilla con una parábola en cada eje."""
    lat, lon = float(grid.lats[i]), float(grid.lons[j])
    try:
        if 0 < i < sm.shape[0] - 1:
            a, b, c = sm[i - 1, j], sm[i, j], sm[i + 1, j]
            den = a - 2 * b + c
            if den > 0:
                lat += 0.5 * (a - c) / den * float(grid.lats[1] - grid.lats[0])
        if 0 < j < sm.shape[1] - 1:
            a, b, c = sm[i, j - 1], sm[i, j], sm[i, j + 1]
            den = a - 2 * b + c
            if den > 0:
                lon += 0.5 * (a - c) / den * float(grid.lons[1] - grid.lons[0])
    except Exception:
        pass
    return lat, lon


def find_center(field_hpa, grid: Grid, lat0, lon0, radius_km, prm: TrackerParams):
    res = _find_center(field_hpa, field_hpa, grid, lat0, lon0, radius_km, prm)
    if res is None:
        # Plan B: la baja no tiene isobara cerrada porque está metida en el
        # gradiente de otra borrasca mayor (típico en DANAs y bajas secundarias).
        # Se busca el máximo del laplaciano de la presión (~ vorticidad
        # geostrófica ciclónica), que no se desplaza por el gradiente de fondo.
        res = _find_vort_center(field_hpa, grid, lat0, lon0, radius_km, prm)
    return res


def _find_vort_center(field_hpa, grid: Grid, lat0, lon0, radius_km, prm: TrackerParams):
    sigma_pts = 100.0 / grid.dlat_km
    sm = gaussian_filter(field_hpa, sigma=sigma_pts, mode="nearest")
    dy = grid.dlat_km
    dx = dy * np.cos(np.radians(grid.mesh[0]))
    lap = ((np.roll(sm, 1, 0) + np.roll(sm, -1, 0) - 2 * sm) / dy ** 2
           + (np.roll(sm, 1, 1) + np.roll(sm, -1, 1) - 2 * sm) / dx ** 2) * 1e4  # hPa/(100 km)^2
    neg = -lap
    is_min = neg == minimum_filter(neg, size=prm.local_min_window, mode="nearest")
    b = prm.local_min_window // 2 + 1
    is_min[:b, :] = is_min[-b:, :] = False
    is_min[:, :b] = is_min[:, -b:] = False
    la, lo = grid.mesh
    d = haversine_km(lat0, lon0, la, lo)
    cand = np.argwhere(is_min & (d <= radius_km) & (lap >= prm.min_laplacian) & (sm <= prm.max_pressure_hpa))
    if not len(cand):
        return None
    i, j = min(cand, key=lambda ij: d[ij[0], ij[1]] / 100.0 - lap[ij[0], ij[1]])
    lat, lon = _refine(neg, grid, i, j)
    return lat, lon, float(field_hpa[i, j])


def _find_center(field_hpa, raw_hpa, grid: Grid, lat0, lon0, radius_km, prm: TrackerParams):
    sm, is_min = _prepare(field_hpa, grid, prm)
    is_min &= sm <= prm.max_pressure_hpa
    la, lo = grid.mesh
    d = haversine_km(lat0, lon0, la, lo)
    cand = np.argwhere(is_min & (d <= radius_km))
    best, best_score = None, math.inf
    for i, j in cand:
        if _depth(sm, grid, i, j) < prm.min_depth_hpa:
            continue
        score = sm[i, j] + prm.distance_penalty_hpa_per_100km * d[i, j] / 100.0
        if score < best_score:
            best, best_score = (i, j), score
    if best is None:
        return None
    i, j = best
    lat, lon = _refine(sm, grid, i, j)
    # presión real (sin suavizar) en un entorno pequeño
    near = haversine_km(lat, lon, la, lo) < 60
    p = float(raw_hpa[near].min()) if near.any() else float(raw_hpa[i, j])
    return lat, lon, p


def track(fields: dict[int, np.ndarray], grid: Grid, seed_lat: float, seed_lon: float,
          prm: TrackerParams | None = None) -> list[TrackPoint]:
    """
    fields: {paso_en_horas: campo MSLP en hPa (2D, lats x lons)}
    Devuelve la lista de puntos de la trayectoria (puede ser vacía).
    """
    prm = prm or TrackerParams()
    steps = sorted(fields)
    out: list[TrackPoint] = []
    for k, step in enumerate(steps):
        if not out:
            if k > 0:      # no se encontró en t0 -> no hay trayectoria
                break
            res = find_center(fields[step], grid, seed_lat, seed_lon, prm.seed_radius_km, prm)
        else:
            last = out[-1]
            dt = step - last.step
            if len(out) >= 2:
                prev = out[-2]
                f = dt / max(last.step - prev.step, 1)
                plat = last.lat + 0.8 * f * (last.lat - prev.lat)
                plon = last.lon + 0.8 * f * (last.lon - prev.lon)
            else:
                plat, plon = last.lat, last.lon
            radius = max(prm.min_search_km, prm.max_speed_kmh * dt * 0.6)
            res = find_center(fields[step], grid, plat, plon, radius, prm)
            # salvaguarda: el desplazamiento real no puede superar la velocidad máxima
            if res and haversine_km(last.lat, last.lon, res[0], res[1]) > prm.max_speed_kmh * dt:
                res = None
        if res is None:
            break
        out.append(TrackPoint(step, *res))
    return out


def listar_bajas(field_hpa, grid: Grid, prm: TrackerParams | None = None, max_p=1016.0):
    """Todas las bajas cerradas del campo: [(lat, lon, p, profundidad)], de más a menos profunda.
    Sirve para encontrar la semilla de una borrasca nueva."""
    prm = prm or TrackerParams()
    sm, is_min = _prepare(field_hpa, grid, prm)
    out = []
    for i, j in np.argwhere(is_min & (sm <= max_p)):
        dep = _depth(sm, grid, i, j)
        if dep >= prm.min_depth_hpa:
            out.append((round(float(grid.lats[i]), 1), round(float(grid.lons[j]), 1), round(float(field_hpa[i, j]), 1), round(dep, 1)))
    return sorted(out, key=lambda x: x[2])
