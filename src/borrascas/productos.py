"""
Productos derivados de las trayectorias: media del ensemble, dispersión y
estadísticas de impacto sobre Canarias.
"""
from __future__ import annotations

import numpy as np

from .tracker import haversine_km

# Puntos de referencia del archipiélago (aprox. centro de cada isla)
CANARIAS = {
    "El Hierro": (27.75, -18.00),
    "La Palma": (28.68, -17.86),
    "La Gomera": (28.10, -17.23),
    "Tenerife": (28.29, -16.62),
    "Gran Canaria": (27.96, -15.59),
    "Fuerteventura": (28.36, -14.05),
    "Lanzarote": (29.04, -13.63),
}


def dist_canarias(lat, lon):
    """Distancia (km) del punto a la isla más cercana y nombre de la isla."""
    best = min(((float(haversine_km(lat, lon, a, b)), n) for n, (a, b) in CANARIAS.items()))
    return best


def closest_approach(track: list[dict]):
    """Paso de menor distancia a Canarias para una trayectoria."""
    if not track:
        return None
    best = None
    for pt in track:
        d, isla = dist_canarias(pt["lat"], pt["lon"])
        if best is None or d < best["km"]:
            best = {"km": round(d), "isla": isla, "h": pt["h"], "p": pt["p"]}
    return best


def ensemble_mean(members: list[list[dict]], min_fraction=0.5):
    """Trayectoria media (sólo en pasos donde sigue vivo >= min_fraction de miembros)."""
    n = len(members)
    if n == 0:
        return [], []
    by_step: dict[int, list[dict]] = {}
    for tr in members:
        for pt in tr:
            by_step.setdefault(pt["h"], []).append(pt)
    mean, spread = [], []
    for h in sorted(by_step):
        pts = by_step[h]
        if len(pts) < max(2, min_fraction * n):
            break
        lat = float(np.mean([p["lat"] for p in pts]))
        # media de longitudes segura (no cruzamos el antimeridiano aquí, pero por si acaso)
        lons = np.radians([p["lon"] for p in pts])
        lon = float(np.degrees(np.arctan2(np.sin(lons).mean(), np.cos(lons).mean())))
        p = float(np.mean([q["p"] for q in pts]))
        mean.append({"h": h, "lat": round(lat, 2), "lon": round(lon, 2), "p": round(p, 1),
                     "n": len(pts)})
        d = [float(haversine_km(lat, lon, q["lat"], q["lon"])) for q in pts]
        spread.append({"h": h, "km": round(float(np.mean(d)))})
    return mean, spread


def canarias_probabilities(members: list[list[dict]], radii=(100, 200, 300)):
    """% de miembros cuyo centro pasa a menos de R km de alguna isla."""
    n = len(members)
    if n == 0:
        return {}
    out = {}
    cas = [closest_approach(tr) for tr in members]
    for r in radii:
        hits = sum(1 for c in cas if c and c["km"] <= r)
        out[str(r)] = round(100 * hits / n)
    dists = [c["km"] for c in cas if c]
    if dists:
        out["dist_mediana_km"] = int(np.median(dists))
    return out


def summarize_model(model_out: dict) -> dict:
    """Añade media, dispersión y estadísticas a la salida de un modelo."""
    members = [m["track"] for m in model_out.get("members", []) if m["track"]]
    if model_out.get("kind") == "ensemble" and members:
        mean, spread = ensemble_mean(members)
        model_out["mean"] = mean
        model_out["spread"] = spread
        model_out["canarias"] = canarias_probabilities(members)
        pmins = [min(p["p"] for p in tr) for tr in members]
        model_out["pmin"] = {"min": round(min(pmins), 1), "mediana": round(float(np.median(pmins)), 1),
                             "max": round(max(pmins), 1)}
    for m in model_out.get("members", []):
        m["closest"] = closest_approach(m["track"])
    return model_out
