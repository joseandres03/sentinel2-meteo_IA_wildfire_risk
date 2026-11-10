#!/usr/bin/env python3
"""
Genera datos de DEMOSTRACIÓN pasando campos de presión sintéticos por el mismo
tracker y la misma cadena de productos que se usan con datos reales.
Sirve para probar el visor sin descargar nada y como test de extremo a extremo.
NO escribe en docs/ por defecto, para no publicar datos falsos:

  python src/trayectorias.py --demo /tmp/demo_borrascas
  cp -r docs/trayectorias.* /tmp/demo_borrascas/..   (o sirve una copia local de docs/)
"""
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from .generar import load_json, process_run, update_catalog, write_json
from .tracker import Grid

T0 = datetime(2026, 11, 8, 0, tzinfo=timezone.utc)
DOMAIN = (15, 55, -50, 5)
LATS = np.arange(DOMAIN[0], DOMAIN[1] + 0.01, 0.5)
LONS = np.arange(DOMAIN[2], DOMAIN[3] + 0.01, 0.5)
GRID = Grid(LATS, LONS)
LA, LO = np.meshgrid(LATS, LONS, indexing="ij")


def truth(t_h):
    """'Realidad' sintética: la borrasca baja hacia el sur más de lo previsto."""
    lat = 38 - 11 * (1 - np.exp(-t_h / 80))
    lon = -25 + 9 * (1 - np.exp(-t_h / 100))
    p = 1000 - 8 * np.sin(np.clip(t_h, 0, 150) / 150 * np.pi)
    return lat, lon, p


# sesgo norte/oeste por modelo (º a 120 h en la primera pasada), decrece con las pasadas
BIAS = {"ECMWF": (2.2, -1.5), "AIFS": (1.4, -0.8), "GFS": (3.2, -2.5)}


def field(centers):
    f = 1016 + 0.12 * (LA - 35) + 0.02 * (LO + 20)
    for lat, lon, p, r_km in centers:
        d2 = ((LA - lat) * 111) ** 2 + ((LO - lon) * 111 * np.cos(np.radians(lat))) ** 2
        f = f - (1016 - p) * np.exp(-d2 / (2 * r_km ** 2))
    return f


def make_fetch(model, ensemble, n_members, seed):
    def fetch(run, steps, domain, **kw):
        rng = np.random.default_rng(seed + int(run.timestamp() // 3600))
        t_r = (run - T0).total_seconds() / 3600
        decay = max(0.0, 1 - t_r / 84)
        blat, blon = BIAS[model]
        out = {}
        # perturbación suave por miembro: deriva lineal + curvatura, que crece con el plazo
        drift = rng.normal(0, 1, size=(n_members, 2))
        curve = rng.normal(0, 1, size=(n_members, 2))
        for m in range(n_members):
            out[m] = {}
            for k, h in enumerate(steps):
                lat, lon, p = truth(t_r + h)
                lat += blat * decay * h / 120
                lon += blon * decay * h / 120
                if ensemble and m > 0:
                    x = h / 120
                    lat += 1.3 * (drift[m, 0] * x + 0.5 * curve[m, 0] * x * x) + rng.normal(0, 0.05)
                    lon += 1.8 * (drift[m, 1] * x + 0.5 * curve[m, 1] * x * x) + rng.normal(0, 0.05)
                    p += 3 * drift[m, 0] * x
                # otra borrasca al noroeste que NO debe confundir al tracker
                other = (50 + 0.02 * h, -38 + 0.15 * h, 985, 600)
                out[m][h] = field([(lat, lon, p, 380), other])
        return out, GRID
    return fetch


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    catalog = {
        "ECMWF": dict(label="ECMWF IFS", kind="deterministic", color="#ff4d4d", step=6,
                      fetch=make_fetch("ECMWF", False, 1, 1)),
        "ECMWF-ENS": dict(label="ECMWF ENS (51)", kind="ensemble", color="#ff9a9a", step=12, parent="ECMWF",
                          fetch=make_fetch("ECMWF", True, 51, 2)),
        "AIFS": dict(label="ECMWF AIFS (IA)", kind="deterministic", color="#c77dff", step=6,
                     fetch=make_fetch("AIFS", False, 1, 3)),
        "GFS": dict(label="NOAA GFS", kind="deterministic", color="#3da5ff", step=6,
                    fetch=make_fetch("GFS", False, 1, 4)),
        "GEFS": dict(label="NOAA GEFS (31)", kind="ensemble", color="#8fd0ff", step=6, parent="GFS",
                     fetch=make_fetch("GFS", True, 31, 5)),
    }
    lat0, lon0, _ = truth(0)
    cfg = {"name": "Charles (DEMO)", "seed": {"lat": float(lat0), "lon": float(lon0)},
           "domain": list(DOMAIN), "max_lead_h": 120, "models": list(catalog), "keep_runs": 20}
    sdir = os.path.join(out_dir, "demo")
    if os.path.isdir(sdir):
        for f in os.listdir(sdir):
            os.remove(os.path.join(sdir, f))
    for i in range(5):
        process_run("demo", cfg, T0 + timedelta(hours=12 * i), out_dir, model_fetchers=catalog)
    idx_path = os.path.join(sdir, "index.json")
    idx = load_json(idx_path, {})
    idx["demo"] = True
    write_json(idx_path, idx)
    update_catalog(out_dir)
