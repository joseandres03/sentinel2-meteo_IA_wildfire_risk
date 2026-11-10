"""
Descarga de la presión a nivel del mar (MSLP) de cada modelo.

Todas las fuentes son abiertas y gratuitas y se leen igual: cada fichero GRIB2
publicado tiene al lado un índice (.index en ECMWF, .idx en NOAA) con la
posición de cada campo, así que se pide SOLO el trozo de la presión con una
petición HTTP de rango de bytes. Nada de librerías con reintentos eternos:

  * Peticiones en paralelo, tiempo de espera corto y como mucho 3 reintentos.
  * ECMWF: si un espejo falla (AWS, Azure, ECMWF, Google), se pasa al siguiente.
  * Si el fichero aún no existe (404) se lanza NoPublicado, para reintentarlo
    en la siguiente ejecución sin contarlo como error.
  * Cada descarga respeta una hora límite (deadline) para que el trabajo nunca
    se alargue: lo que no dé tiempo se queda pendiente para la siguiente vez.

Cada fetch devuelve ({miembro: {paso_h: campo_hPa}}, Grid) recortado al dominio.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import requests

from .tracker import Grid

log = logging.getLogger("borrascas.fuentes")

TIMEOUT = (10, 60)          # (conexión, lectura) en segundos
REINTENTOS = 3
HILOS = int(os.environ.get("BORRASCAS_HILOS", "12"))


class NoPublicado(Exception):
    """El fichero todavía no está publicado (o ya no existe)."""


class SinTiempo(Exception):
    """Se ha alcanzado la hora límite de esta ejecución."""


_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.headers["User-Agent"] = "canarywildfirerisk.es (trayectorias de borrascas)"
    return s


def _check(deadline):
    if deadline and time.monotonic() > deadline:
        raise SinTiempo()


def http_get(url, *, rango=None, deadline=None, head=False):
    """GET (o HEAD) con reintentos cortos. 404/403 -> NoPublicado."""
    headers = {"Range": f"bytes={rango[0]}-{rango[1]}"} if rango else {}
    ultimo = None
    for intento in range(REINTENTOS):
        _check(deadline)
        try:
            fn = _session().head if head else _session().get
            r = fn(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code in (403, 404):
                raise NoPublicado(url)
            r.raise_for_status()
            return r
        except NoPublicado:
            raise
        except requests.RequestException as e:
            ultimo = e
            espera = (2, 5, 10)[intento]
            log.debug("reintento %d %s (%s)", intento + 1, url, e)
            time.sleep(espera)
    raise ultimo


# --------------------------------------------------------------------------
# Lectura GRIB (eccodes)
# --------------------------------------------------------------------------

_GRIB_LOCK = threading.Lock()


def read_grib_messages(path: str, domain):
    """Lee los mensajes de un GRIB recortados a domain=(latS, latN, lonW, lonE).
    Devuelve lista de (miembro, paso, grid, campo_hPa)."""
    with _GRIB_LOCK:     # eccodes no garantiza ser seguro entre hilos
        return _read_grib(path, domain)


def _read_grib(path, domain):
    import eccodes  # import diferido: solo hace falta en producción

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


def _descargar_trozos(trozos, domain, deadline, clave_override=None):
    """trozos: lista de (clave, url, (ini, fin)). Descarga en paralelo y decodifica.
    clave = (miembro, paso) o None (se usan los de dentro del GRIB).
    Devuelve ({miembro: {paso: campo}}, grid, trozos_fallidos)."""
    fields, grid, fallos = {}, None, []
    with tempfile.TemporaryDirectory() as tmp:
        def job(i, clave, url, rango):
            data = http_get(url, rango=rango, deadline=deadline).content
            p = os.path.join(tmp, f"{i}.grib2")
            with open(p, "wb") as fh:
                fh.write(data)
            return clave, read_grib_messages(p, domain)

        with ThreadPoolExecutor(HILOS) as ex:
            futs = [ex.submit(job, i, c, u, r) for i, (c, u, r) in enumerate(trozos)]
            for trozo, fut in zip(trozos, futs):
                try:
                    clave, msgs = fut.result()
                except SinTiempo:
                    for f in futs:
                        f.cancel()
                    raise
                except Exception as e:
                    fallos.append(trozo)
                    log.debug("descarga fallida: %s", e)
                    continue
                for number, step, g, data in msgs:
                    mem, st = clave if clave else (number, step)
                    fields.setdefault(mem, {})[st] = data
                    grid = grid or g
    return fields, grid, fallos


# --------------------------------------------------------------------------
# ECMWF Open Data (IFS, ENS, AIFS, AIFS-ENS)
# --------------------------------------------------------------------------

ECMWF_ESPEJOS = {
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "ecmwf": "https://data.ecmwf.int/forecasts",
    "google": "https://storage.googleapis.com/ecmwf-open-data",
}


def _orden_espejos():
    pref = os.environ.get("ECMWF_SOURCE", "aws")
    return [pref] + [k for k in ECMWF_ESPEJOS if k != pref]


def _ecmwf_url(base, run: datetime, model, ens, step):
    d, hh = run.strftime("%Y%m%d"), f"{run.hour:02d}"
    if ens:
        stream, typ = "enfo", "ef"
    elif model == "ifs" and run.hour in (6, 18):
        stream, typ = "scda", "fc"
    else:
        stream, typ = "oper", "fc"
    return f"{base}/{d}/{hh}z/{model}/0p25/{stream}/{d}{hh}0000-{step}h-{stream}-{typ}.grib2"


def _ecmwf_trozos_msl(url, deadline):
    """Lee el .index (JSON por línea) y devuelve los rangos de bytes de 'msl'."""
    txt = http_get(url[:-6] + ".index", deadline=deadline).text
    out = []
    for line in txt.splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("param") == "msl":
            off, ln = int(e["_offset"]), int(e["_length"])
            out.append((int(e.get("number", 0) or 0), (off, off + ln - 1)))
    if not out:
        raise NoPublicado(f"msl no está en {url}.index")
    return out


def fetch_ecmwf(run, steps, domain, *, model="ifs", ensemble=False, deadline=None):
    ultimo_error = None
    for espejo in _orden_espejos():
        base = ECMWF_ESPEJOS[espejo]
        try:
            # índices de todos los pasos (en paralelo); de cada uno, los trozos msl
            def idx(step):
                url = _ecmwf_url(base, run, model, ensemble, step)
                return [((num, step), url, r) for num, r in _ecmwf_trozos_msl(url, deadline)]
            with ThreadPoolExecutor(min(HILOS, len(steps))) as ex:
                trozos = [t for lst in ex.map(idx, steps) for t in lst]
            fields, grid, fallos = _descargar_trozos(trozos, domain, deadline)
            n_fallos = len(fallos)
            # los trozos que fallen (503 sueltos) se reintentan en los otros espejos
            actual = base
            for otro in [e for e in _orden_espejos() if e != espejo]:
                if not fallos:
                    break
                rehacer = [(c, u.replace(actual, ECMWF_ESPEJOS[otro], 1), r) for c, u, r in fallos]
                actual = ECMWF_ESPEJOS[otro]
                f2, g2, fallos = _descargar_trozos(rehacer, domain, deadline)
                for mem, d in f2.items():
                    fields.setdefault(mem, {}).update(d)
                grid = grid or g2
            if grid is None:
                raise RuntimeError("sin campos válidos")
            log.info("    %s %s: %d campos desde %s (%d reintentados en otro espejo, %d perdidos)", model,
                     "ENS" if ensemble else "det", len(trozos) - len(fallos), espejo, n_fallos, len(fallos))
            return fields, grid
        except (SinTiempo, NoPublicado):
            raise
        except Exception as e:
            ultimo_error = e
            log.warning("    espejo %s falló (%s); probando el siguiente", espejo, e)
    raise RuntimeError(f"todos los espejos de ECMWF fallaron: {ultimo_error}")


def publicado_ecmwf(run, last_step, *, model="ifs", ensemble=False, deadline=None):
    for espejo in _orden_espejos():
        url = _ecmwf_url(ECMWF_ESPEJOS[espejo], run, model, ensemble, last_step)
        try:
            http_get(url[:-6] + ".index", head=True, deadline=deadline)
            return True
        except NoPublicado:
            return False          # 404 en un espejo sano: aún no está
        except (requests.RequestException, OSError):
            continue              # espejo caído: probar otro
    return False


# --------------------------------------------------------------------------
# NOAA GFS / GEFS (buckets públicos de NOAA en AWS)
# --------------------------------------------------------------------------

GFS_URL = ("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{hh}/atmos/"
           "gfs.t{hh}z.pgrb2.0p25.f{fff}")
GEFS_URL = ("https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d}/{hh}/atmos/pgrb2ap5/"
            "{mem}.t{hh}z.pgrb2a.0p50.f{fff}")
_PRMSL = ":PRMSL:mean sea level:"


def _noaa_rango(url, deadline):
    lines = http_get(url + ".idx", deadline=deadline).text.strip().splitlines()
    for k, line in enumerate(lines):
        if _PRMSL in line:
            ini = int(line.split(":")[1])
            fin = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else ""
            return ini, fin
    raise NoPublicado(f"PRMSL no está en {url}.idx")


def _fetch_noaa(urls: dict, domain, deadline):
    """urls: {(miembro, paso): url}. Primero los .idx (en paralelo), luego los rangos."""
    def idx(item):
        clave, url = item
        try:
            return clave, url, _noaa_rango(url, deadline)
        except NoPublicado:
            return None
    with ThreadPoolExecutor(HILOS) as ex:
        trozos = [t for t in ex.map(idx, urls.items()) if t]
    if not trozos:
        raise NoPublicado("ningún fichero NOAA publicado")
    fields, grid, fallos = _descargar_trozos(trozos, domain, deadline)
    if fallos:   # segundo intento de los que hayan fallado
        f2, g2, fallos = _descargar_trozos(fallos, domain, deadline)
        for mem, d in f2.items():
            fields.setdefault(mem, {}).update(d)
        grid = grid or g2
    if grid is None:
        raise RuntimeError("sin campos válidos")
    log.info("    NOAA: %d campos (%d no publicados, %d perdidos)", len(trozos) - len(fallos),
             len(urls) - len(trozos), len(fallos))
    return fields, grid


def _gfs(run, s):
    return GFS_URL.format(d=run.strftime("%Y%m%d"), hh=f"{run.hour:02d}", fff=f"{s:03d}")


def _gefs(run, m, s):
    mem = "gec00" if m == 0 else f"gep{m:02d}"
    return GEFS_URL.format(d=run.strftime("%Y%m%d"), hh=f"{run.hour:02d}", mem=mem, fff=f"{s:03d}")


def fetch_gfs(run, steps, domain, deadline=None):
    return _fetch_noaa({(0, s): _gfs(run, s) for s in steps}, domain, deadline)


def fetch_gefs(run, steps, domain, deadline=None, n_members=30):
    return _fetch_noaa({(m, s): _gefs(run, m, s) for m in range(n_members + 1) for s in steps},
                       domain, deadline)


def publicado_url(url, deadline=None):
    try:
        http_get(url, head=True, deadline=deadline)
        return True
    except (NoPublicado, requests.RequestException):
        return False


def fetch_analisis_gfs(tiempos: list[datetime], domain, deadline=None):
    """Análisis del GFS (f000) en varios instantes: {instante: campo}, grid.
    Es lo que se usa para la trayectoria real de la borrasca."""
    urls = {(t, 0): _gfs(t, 0) for t in tiempos}
    fields, grid = _fetch_noaa(urls, domain, deadline)
    return {t: d[0] for t, d in fields.items() if 0 in d}, grid


def ultimo_analisis_gfs(desde: datetime, hasta: datetime, deadline=None):
    """Último instante (cada 6 h, <= hasta) con análisis GFS publicado."""
    t = hasta.replace(hour=hasta.hour - hasta.hour % 6, minute=0, second=0, microsecond=0)
    while t >= desde:
        if publicado_url(_gfs(t, 0) + ".idx", deadline):
            return t
        t -= timedelta(hours=6)
    return None


# --------------------------------------------------------------------------
# Catálogo de modelos
# --------------------------------------------------------------------------
# step: intervalo (h) entre los campos usados para el seguimiento.
# max_lead: plazo máximo que publica el modelo en cada pasada {hora: h}.
# Los ENS de ECMWF van cada 12 h para no multiplicar la descarga (50 campos
# globales por paso; ECMWF no permite recortar por zona). En Open Data el control
# del ENS no incluye la presión, así que salen 50 miembros.

def _m(label, kind, color, step, fetch, publicado, max_lead, parent=None, defecto=True):
    return dict(label=label, kind=kind, color=color, step=step, fetch=fetch, publicado=publicado,
                max_lead=max_lead, parent=parent, defecto=defecto)


_4 = {0: 360, 6: 360, 12: 360, 18: 360}
MODELS = {
    "ECMWF": _m("ECMWF IFS", "deterministic", "#ff4d4d", 6,
                lambda r, s, d, dl: fetch_ecmwf(r, s, d, model="ifs", deadline=dl),
                lambda r, last, dl: publicado_ecmwf(r, last, model="ifs", deadline=dl),
                {0: 240, 6: 90, 12: 240, 18: 90}),
    "ECMWF-ENS": _m("ECMWF ENS", "ensemble", "#ff9a9a", 12,
                    lambda r, s, d, dl: fetch_ecmwf(r, s, d, model="ifs", ensemble=True, deadline=dl),
                    lambda r, last, dl: publicado_ecmwf(r, last, model="ifs", ensemble=True, deadline=dl),
                    {0: 360, 6: 144, 12: 360, 18: 144}, parent="ECMWF"),
    "AIFS": _m("ECMWF AIFS (IA)", "deterministic", "#c77dff", 6,
               lambda r, s, d, dl: fetch_ecmwf(r, s, d, model="aifs-single", deadline=dl),
               lambda r, last, dl: publicado_ecmwf(r, last, model="aifs-single", deadline=dl),
               _4),
    "AIFS-ENS": _m("ECMWF AIFS-ENS", "ensemble", "#dcb6ff", 12,
                   lambda r, s, d, dl: fetch_ecmwf(r, s, d, model="aifs-ens", ensemble=True, deadline=dl),
                   lambda r, last, dl: publicado_ecmwf(r, last, model="aifs-ens", ensemble=True, deadline=dl),
                   _4, parent="AIFS", defecto=False),
    "GFS": _m("NOAA GFS", "deterministic", "#3da5ff", 6,
              lambda r, s, d, dl: fetch_gfs(r, s, d, deadline=dl),
              lambda r, last, dl: publicado_url(_gfs(r, last) + ".idx", dl),
              {0: 384, 6: 384, 12: 384, 18: 384}),
    "GEFS": _m("NOAA GEFS", "ensemble", "#8fd0ff", 6,
               lambda r, s, d, dl: fetch_gefs(r, s, d, deadline=dl),
               lambda r, last, dl: publicado_url(_gefs(r, 30, last) + ".idx", dl),
               {0: 384, 6: 384, 12: 384, 18: 384}, parent="GFS"),
}
