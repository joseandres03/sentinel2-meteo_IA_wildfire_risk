"""
Datos de DEMOSTRACIÓN: pasa campos de presión sintéticos por exactamente el
mismo código que se usa con datos reales (trayectoria real, pasadas
incrementales, tracker y productos), sustituyendo solo las descargas por una
fuente falsa. Sirve para probar el visor sin descargar nada y como test de
extremo a extremo. NO escribe en docs/ por defecto, para no publicar datos falsos:

  python src/trayectorias.py --demo /tmp/demo_borrascas
"""
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from .generar import actualizar_borrasca, load_json, update_catalog, write_json
from .tracker import Grid

T0 = datetime(2026, 11, 8, 0, tzinfo=timezone.utc)
AHORA = T0 + timedelta(hours=48, minutes=30)
DOMAIN = (15, 55, -50, 5)
LATS = np.arange(DOMAIN[0], DOMAIN[1] + 0.01, 0.5)
LONS = np.arange(DOMAIN[2], DOMAIN[3] + 0.01, 0.5)
GRID = Grid(LATS, LONS)
LA, LO = np.meshgrid(LATS, LONS, indexing="ij")


def truth(t_h):
    """'Realidad' sintética: la borrasca baja hacia el sur más de lo previsto.
    Se forma 18 h antes de T0 (antes no existe)."""
    lat = 38 - 11 * (1 - np.exp(-t_h / 80))
    lon = -25 + 9 * (1 - np.exp(-t_h / 100))
    p = 1000 - 8 * np.sin(np.clip(t_h, 0, 150) / 150 * np.pi)
    if t_h < -18:
        p = 1030   # aún no formada
    return lat, lon, p


# sesgo norte/oeste por modelo (º a 120 h en la primera pasada), decrece con las pasadas
BIAS = {"ECMWF": (2.2, -1.5), "AIFS": (1.4, -0.8), "GFS": (3.2, -2.5)}


def field(centers):
    f = 1016 + 0.12 * (LA - 35) + 0.02 * (LO + 20)
    for lat, lon, p, r_km in centers:
        d2 = ((LA - lat) * 111) ** 2 + ((LO - lon) * 111 * np.cos(np.radians(lat))) ** 2
        f = f - (1016 - p) * np.exp(-d2 / (2 * r_km ** 2))
    return f


def _otra(h_abs):
    # otra borrasca al noroeste que NO debe confundir al tracker
    return (50 + 0.02 * h_abs, -38 + 0.15 * h_abs, 985, 600)


def make_fetch(model, ensemble, n_members, seed):
    def fetch(run, steps, domain, deadline=None):
        rng = np.random.default_rng(seed + int(run.timestamp() // 3600))
        t_r = (run - T0).total_seconds() / 3600
        decay = max(0.0, 1 - t_r / 84)
        blat, blon = BIAS[model]
        drift = rng.normal(0, 1, size=(n_members, 2))
        curve = rng.normal(0, 1, size=(n_members, 2))
        out = {}
        for m in range(n_members):
            out[m] = {}
            for h in steps:
                lat, lon, p = truth(t_r + h)
                lat += blat * decay * h / 120
                lon += blon * decay * h / 120
                if ensemble and m > 0:
                    x = h / 120
                    lat += 1.3 * (drift[m, 0] * x + 0.5 * curve[m, 0] * x * x) + rng.normal(0, 0.05)
                    lon += 1.8 * (drift[m, 1] * x + 0.5 * curve[m, 1] * x * x) + rng.normal(0, 0.05)
                    p += 3 * drift[m, 0] * x
                out[m][h] = field([(lat, lon, p, 380), _otra(t_r + h)])
        return out, GRID
    return fetch


class _SinTiempo(Exception):
    pass


class _NoPublicado(Exception):
    pass


def _analisis(tiempos, domain, deadline=None):
    out = {}
    for t in tiempos:
        h = (t - T0).total_seconds() / 3600
        lat, lon, p = truth(h)
        out[t] = field([(lat, lon, p, 380), _otra(h)])
    return out, GRID


def _ultimo(desde, hasta, deadline=None):
    t = min(hasta, AHORA)
    t = t.replace(hour=t.hour - t.hour % 6, minute=0, second=0, microsecond=0)
    return t if t >= desde else None


FUENTES = SimpleNamespace(SinTiempo=_SinTiempo, NoPublicado=_NoPublicado,
                          fetch_analisis_gfs=_analisis, ultimo_analisis_gfs=_ultimo)


def _pub(r, last, dl):
    return r <= AHORA - timedelta(hours=4)


def _cat():
    def m(label, kind, color, step, fetch, parent=None):
        return dict(label=label, kind=kind, color=color, step=step, fetch=fetch, publicado=_pub,
                    max_lead={0: 240, 6: 120, 12: 240, 18: 120}, parent=parent)
    return {
        "ECMWF": m("ECMWF IFS", "deterministic", "#ff4d4d", 6, make_fetch("ECMWF", False, 1, 1)),
        "ECMWF-ENS": m("ECMWF ENS (51)", "ensemble", "#ff9a9a", 12, make_fetch("ECMWF", True, 51, 2), "ECMWF"),
        "AIFS": m("ECMWF AIFS (IA)", "deterministic", "#c77dff", 6, make_fetch("AIFS", False, 1, 3)),
        "GFS": m("NOAA GFS", "deterministic", "#3da5ff", 6, make_fetch("GFS", False, 1, 4)),
        "GEFS": m("NOAA GEFS (31)", "ensemble", "#8fd0ff", 6, make_fetch("GFS", True, 31, 5), "GFS"),
    }


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sdir = os.path.join(out_dir, "demo")
    if os.path.isdir(sdir):
        for f in os.listdir(sdir):
            os.remove(os.path.join(sdir, f))
    lat0, lon0, _ = truth(24)
    cfg = {"name": "Charles (DEMO)", "domain": list(DOMAIN), "max_lead_h": 120, "keep_runs": 16,
           "seed": {"time": "2026-11-09T00:00Z", "lat": float(round(lat0 + 0.6, 1)), "lon": float(round(lon0 - 0.8, 1))},
           "models": ["ECMWF", "ECMWF-ENS", "AIFS", "GFS", "GEFS"]}
    actualizar_borrasca("demo", cfg, out_dir, _cat(), FUENTES, time.monotonic() + 600, ahora=AHORA)
    idx_path = os.path.join(sdir, "index.json")
    idx = load_json(idx_path, {})
    idx["demo"] = True
    write_json(idx_path, idx)
    update_catalog(out_dir)
