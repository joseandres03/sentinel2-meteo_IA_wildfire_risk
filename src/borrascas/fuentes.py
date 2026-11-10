"""
Descarga y lectura de MSLP de cada modelo.

Todas las fuentes son abiertas y gratuitas:
  * ECMWF Open Data  (IFS determinista, ENS 51 miembros, AIFS, AIFS-ENS)
      paquete `ecmwf-opendata`, servido desde el bucket de AWS/Azure/GCP o ECMWF.
  * NOAA GFS y GEFS (31 miembros) desde el bucket público de AWS (NODD),
      descargando SÓLO el campo PRMSL de cada fichero mediante rangos de bytes
      (.idx), así que cada descarga pesa unos cientos de KB.

Cada fetcher devuelve:  { miembro(int): { paso_h(int): campo_hPa(2D) } }, Grid
ya recortado al dominio de la borrasca.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import requests

from .tracker import Grid

log = logging.getLogger("sources")

# --------------------------------------------------------------------------
# Lectura GRIB genérica (eccodes)
# --------------------------------------------------------------------------

def read_grib_messages(path: str, domain):
    """Lee todos los mensajes MSLP de un GRIB, recortados a domain=(latS, latN, lonW, lonE).
    Devuelve lista de (miembro, paso, grid, campo_hPa)."""
    import eccodes  # import diferido: sólo hace falta en producción

    latS, latN, lonW, lonE = domain
    out = []
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                step = int(eccodes.codes_get(h, "endStep"))
                try:
                    number = int(eccodes.codes_get(h, "number"))
                except Exception:
                    number = 0
                ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                lat1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                lat2 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
                lon1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
                if eccodes.codes_get(h, "iScansNegatively"):
                    dlon = -dlon
                vals = eccodes.codes_get_values(h).reshape(nj, ni)
            finally:
                eccodes.codes_release(h)

            lats = np.linspace(lat1, lat2, nj)
            lons = ((lon1 + np.arange(ni) * dlon + 180.0) % 360.0) - 180.0
            order = np.argsort(lons)
            lons, vals = lons[order], vals[:, order]
            mi = (lats >= latS) & (lats <= latN)
            mj = (lons >= lonW) & (lons <= lonE)
            sub = vals[np.ix_(mi, mj)]
            if np.nanmax(sub) > 2000:          # Pa -> hPa
                sub = sub / 100.0
            out.append((number, step, Grid(lats[mi], lons[mj]), sub))
    return out


def _collect(msgs):
    fields, grid = {}, None
    for number, step, g, data in msgs:
        fields.setdefault(number, {})[step] = data
        grid = grid or g
    return fields, grid


# --------------------------------------------------------------------------
# ECMWF Open Data
# --------------------------------------------------------------------------

def fetch_ecmwf(run: datetime, steps, domain, *, model="ifs", ensemble=False,
                source=None, workdir=None):
    from ecmwf.opendata import Client

    source = source or os.environ.get("ECMWF_SOURCE", "aws")
    client = Client(source=source, model=model, resol="0p25")
    if ensemble:
        req = dict(stream="enfo", type=["cf", "pf"])
    else:
        req = dict(stream="oper" if run.hour in (0, 12) else "scda", type="fc")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        target = os.path.join(tmp, "msl.grib2")
        client.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, param="msl",
                        step=list(steps), target=target, **req)
        return _collect(read_grib_messages(target, domain))


# --------------------------------------------------------------------------
# NOAA GFS / GEFS (AWS NODD + rangos de bytes del .idx)
# --------------------------------------------------------------------------

GFS_URL = ("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{hh}/atmos/"
           "gfs.t{hh}z.pgrb2.0p25.f{fff}")
GEFS_URL = ("https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d}/{hh}/atmos/pgrb2ap5/"
            "{mem}.t{hh}z.pgrb2a.0p50.f{fff}")

_session = requests.Session()


def _get_field_by_idx(url: str, var=":PRMSL:mean sea level:", retries=3) -> bytes:
    for attempt in range(retries):
        try:
            idx = _session.get(url + ".idx", timeout=30)
            idx.raise_for_status()
            lines = idx.text.strip().splitlines()
            for k, line in enumerate(lines):
                if var in line:
                    start = int(line.split(":")[1])
                    end = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else ""
                    r = _session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                    r.raise_for_status()
                    return r.content
            raise KeyError(f"{var} no encontrado en {url}.idx")
        except (requests.RequestException, KeyError) as e:
            if attempt == retries - 1:
                raise
            log.warning("reintento %s (%s)", url, e)
            time.sleep(3 * (attempt + 1))


def _fetch_noaa(urls: dict, domain, workdir=None, workers=8):
    """urls: {(miembro, paso): url}"""
    fields, grid = {}, None
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        def job(key, url):
            data = _get_field_by_idx(url)
            p = os.path.join(tmp, f"{key[0]}_{key[1]}.grib2")
            with open(p, "wb") as fh:
                fh.write(data)
            return key, p

        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(job, k, u) for k, u in urls.items()]
            for fut in as_completed(futs):
                try:
                    (mem, step), p = fut.result()
                except Exception as e:  # un fichero que falte no tumba todo
                    log.warning("NOAA: %s", e)
                    continue
                for _, _, g, data in read_grib_messages(p, domain):
                    fields.setdefault(mem, {})[step] = data
                    grid = grid or g
    return fields, grid


def fetch_gfs(run: datetime, steps, domain, workdir=None):
    d, hh = run.strftime("%Y%m%d"), f"{run.hour:02d}"
    urls = {(0, s): GFS_URL.format(d=d, hh=hh, fff=f"{s:03d}") for s in steps}
    return _fetch_noaa(urls, domain, workdir)


def fetch_gefs(run: datetime, steps, domain, workdir=None, n_members=30):
    d, hh = run.strftime("%Y%m%d"), f"{run.hour:02d}"
    urls = {}
    for m in range(n_members + 1):
        mem = "gec00" if m == 0 else f"gep{m:02d}"
        for s in steps:
            urls[(m, s)] = GEFS_URL.format(d=d, hh=hh, mem=mem, fff=f"{s:03d}")
    return _fetch_noaa(urls, domain, workdir)


# --------------------------------------------------------------------------
# Catálogo de modelos
# --------------------------------------------------------------------------
# step: intervalo en horas entre salidas usadas para el seguimiento.
# Para los ENS de ECMWF se usa 12 h por defecto para reducir la descarga
# (51 miembros x campos globales). Bájalo a 6 h si tu servidor lo aguanta.

MODELS = {
    "ECMWF": dict(label="ECMWF IFS", kind="deterministic", color="#ff4d4d", step=6,
                  fetch=lambda run, steps, dom, **kw: fetch_ecmwf(run, steps, dom, model="ifs", **kw)),
    "ECMWF-ENS": dict(label="ECMWF ENS (51)", kind="ensemble", color="#ff9a9a", step=12,
                      parent="ECMWF",
                      fetch=lambda run, steps, dom, **kw: fetch_ecmwf(run, steps, dom, model="ifs",
                                                                      ensemble=True, **kw)),
    "AIFS": dict(label="ECMWF AIFS (IA)", kind="deterministic", color="#c77dff", step=6,
                 fetch=lambda run, steps, dom, **kw: fetch_ecmwf(run, steps, dom, model="aifs-single", **kw)),
    "AIFS-ENS": dict(label="ECMWF AIFS-ENS (51)", kind="ensemble", color="#dcb6ff", step=12,
                     parent="AIFS",
                     fetch=lambda run, steps, dom, **kw: fetch_ecmwf(run, steps, dom, model="aifs-ens",
                                                                     ensemble=True, **kw)),
    "GFS": dict(label="NOAA GFS", kind="deterministic", color="#3da5ff", step=6,
                fetch=lambda run, steps, dom, **kw: fetch_gfs(run, steps, dom, **kw)),
    "GEFS": dict(label="NOAA GEFS (31)", kind="ensemble", color="#8fd0ff", step=6,
                 parent="GFS",
                 fetch=lambda run, steps, dom, **kw: fetch_gefs(run, steps, dom, **kw)),
}
