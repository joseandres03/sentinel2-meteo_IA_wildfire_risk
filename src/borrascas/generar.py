"""
Genera los JSON de trayectorias de borrascas para el visor (docs/trayectorias.html).

Funcionamiento INCREMENTAL (pensado para ejecutarse cada hora):
  1. Trayectoria real: sigue la borrasca en los análisis del GFS (cada 6 h) hacia
     atrás y hacia delante desde la posición de borrascas.json. Solo se descargan
     los análisis nuevos.
  2. Pasadas: para cada pasada (00/06/12/18 UTC) de las últimas `keep_runs`, mira
     qué modelos ya están publicados y procesa SOLO los que faltan, de la más
     reciente a la más antigua. Cada pasada arranca en la posición real de la
     borrasca a esa hora.
  3. Todo respeta una hora límite: lo que no dé tiempo queda pendiente para la
     siguiente ejecución, y lo hecho ya está guardado.

Uso (desde la raíz del repositorio):
  python src/trayectorias.py                   # borrascas activas
  python src/trayectorias.py --storm charles --models GFS,GEFS -v
  python src/trayectorias.py --minutos 20      # límite de tiempo total
  python src/trayectorias.py --rehacer         # rehace la trayectoria real y todas las pasadas

Salida en docs/datos_borrascas/<borrasca>/:
  index.json           pasadas, modelos de cada una y trayectoria real
  run_YYYYMMDDHH.json  trayectorias de todos los modelos para esa pasada
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from .productos import summarize_model
from .tracker import TrackerParams, listar_bajas, track

log = logging.getLogger("borrascas")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
OUT_DEFAULT = os.path.join(BASE_DIR, "docs", "datos_borrascas")

ABANDONAR_TRAS_H = 30      # si un modelo no aparece en 30 h, se da por perdido
ANALISIS_ATRAS_H = 120     # cuánto se busca la borrasca hacia atrás desde la semilla


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------

def parse_time(s: str) -> datetime:
    s = s.replace("Z", "")
    for fmt in ("%Y%m%d%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"fecha no reconocida: {s}")


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%MZ")


def run_id(run: datetime) -> str:
    return run.strftime("%Y%m%d%H")


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)   # escritura atómica


def pos_en(observed, t: datetime):
    """Posición real interpolada en el instante t (o None si está fuera)."""
    pts = [(parse_time(o["t"]), o) for o in observed]
    for (ta, a), (tb, b) in zip(pts, pts[1:]):
        if ta <= t <= tb:
            f = (t - ta) / (tb - ta) if tb > ta else 0
            return a["lat"] + f * (b["lat"] - a["lat"]), a["lon"] + f * (b["lon"] - a["lon"])
    if pts and pts[0][0] == t:
        return pts[0][1]["lat"], pts[0][1]["lon"]
    return None


# --------------------------------------------------------------------------
# 1. trayectoria real (análisis GFS)
# --------------------------------------------------------------------------

def _seguir(campos_por_t: dict, inicio: datetime, lat, lon, sentido: int, prm):
    """Sigue la borrasca en una serie de análisis empezando en `inicio`.
    sentido=+1 hacia delante, -1 hacia atrás. Devuelve [(t, lat, lon, p)]."""
    ts = sorted((t for t in campos_por_t if (t - inicio).total_seconds() * sentido >= 0),
                key=lambda t: abs((t - inicio).total_seconds()))
    fields = {int(abs((t - inicio).total_seconds()) // 3600): campos_por_t[t] for t in ts}
    return fields, ts


def actualizar_real(index, cfg, domain, deadline, fuentes, prm, rehacer=False):
    """Actualiza index['observed'] con los análisis del GFS. Devuelve True si cambió."""
    seed = cfg["seed"]
    seed_t = parse_time(seed["time"])
    seed_t = seed_t.replace(hour=seed_t.hour - seed_t.hour % 6, minute=0)
    ahora = datetime.now(timezone.utc)
    observed = index.get("observed") or []
    reconstruir = rehacer or not observed or index.get("seed") != seed

    if reconstruir:
        ultimo = fuentes.ultimo_analisis_gfs(seed_t, ahora, deadline) or seed_t
        atras = [seed_t - timedelta(hours=h) for h in range(0, ANALISIS_ATRAS_H + 1, 6)]
        delante = [seed_t + timedelta(hours=h) for h in range(6, int((ultimo - seed_t).total_seconds() // 3600) + 1, 6)]
        log.info("  trayectoria real: %d análisis GFS (%s → %s)", len(atras) + len(delante), iso(atras[-1]), iso(ultimo))
        campos, grid = fuentes.fetch_analisis_gfs(atras + delante, domain, deadline)
        pts = []
        for sentido in (-1, +1):
            fields, ts = _seguir(campos, seed_t, seed["lat"], seed["lon"], sentido, prm)
            tr = track(fields, grid, seed["lat"], seed["lon"], prm)
            for p in tr:
                t = seed_t + timedelta(hours=p.step * sentido)
                pts.append({"t": iso(t), "lat": round(p.lat, 2), "lon": round(p.lon, 2), "p": round(p.p, 1), "src": "GFS"})
        if not pts:
            bajas = listar_bajas(campos[seed_t], grid, prm) if seed_t in campos else []
            lista = "\n".join(f"    {la:6.1f}  {lo:7.1f}   {p:6.1f} hPa  (profundidad {d} hPa)" for la, lo, p, d in bajas[:12])
            raise RuntimeError(f"no se encuentra la borrasca cerca de {seed['lat']}, {seed['lon']} el {iso(seed_t)}. "
                               f"Bajas presentes en el análisis GFS de esa hora (lat, lon, presión):\n{lista or '    ninguna'}\n"
                               "Corrige la semilla en src/borrascas/borrascas.json")
        uniq = {p["t"]: p for p in pts}
        index["observed"] = [uniq[k] for k in sorted(uniq)]
        index["seed"] = seed
        index.pop("disipada", None)
        return True

    if index.get("disipada"):
        return False
    # extender hacia delante con los análisis nuevos
    last = observed[-1]
    last_t = parse_time(last["t"])
    ultimo = fuentes.ultimo_analisis_gfs(last_t + timedelta(hours=6), ahora, deadline)
    if not ultimo:
        return False
    tiempos = [last_t + timedelta(hours=h) for h in range(0, int((ultimo - last_t).total_seconds() // 3600) + 1, 6)]
    campos, grid = fuentes.fetch_analisis_gfs(tiempos, domain, deadline)
    fields, _ = _seguir(campos, last_t, last["lat"], last["lon"], +1, prm)
    tr = track(fields, grid, last["lat"], last["lon"], prm)
    nuevos = [{"t": iso(last_t + timedelta(hours=p.step)), "lat": round(p.lat, 2), "lon": round(p.lon, 2),
               "p": round(p.p, 1), "src": "GFS"} for p in tr if p.step > 0]
    index["observed"] = observed + nuevos
    if len(tr) < len(fields):
        index["disipada"] = iso(last_t + timedelta(hours=(tr[-1].step if tr else 0)))
        log.info("  la borrasca ya no aparece en los análisis tras %s", index["disipada"])
    if nuevos:
        log.info("  trayectoria real: +%d análisis (hasta %s)", len(nuevos), nuevos[-1]["t"])
    return bool(nuevos) or bool(index.get("disipada"))


# --------------------------------------------------------------------------
# 2. pasadas de los modelos
# --------------------------------------------------------------------------

def tracks_from_fields(spec, fields, grid, seed, prm):
    members = []
    for mem in sorted(fields):
        pts = track(fields[mem], grid, seed[0], seed[1], prm)
        members.append({"id": mem, "track": [p.as_dict() for p in pts]})
    out = {"label": spec["label"], "kind": spec["kind"], "color": spec["color"],
           "parent": spec.get("parent"), "members": members,
           "found": sum(1 for m in members if m["track"]), "total": len(members)}
    return summarize_model(out)


def pasos_modelo(spec, run, cfg):
    lead = min(cfg.get("max_lead_h", 120), spec["max_lead"].get(run.hour, 0))
    return list(range(0, lead + 1, spec["step"]))


def procesar_modelo(key, spec, run, seed, cfg, prm, deadline):
    steps = pasos_modelo(spec, run, cfg)
    if not steps:
        return None
    t0 = time.monotonic()
    fields, grid = spec["fetch"](run, steps, tuple(cfg["domain"]), deadline)
    res = tracks_from_fields(spec, fields, grid, seed, prm)
    log.info("  %s %-9s %2d/%-2d trayectorias  (%.0f s)", run_id(run), key, res["found"], res["total"],
             time.monotonic() - t0)
    return res


def candidatas(index, cfg, ahora):
    """Pasadas a considerar: cada 6 h dentro de la trayectoria real, las `keep_runs` más recientes."""
    obs = index.get("observed") or []
    if not obs:
        return []
    ini, fin = parse_time(obs[0]["t"]), parse_time(obs[-1]["t"])
    fin = min(fin, ahora)
    t = fin.replace(hour=fin.hour - fin.hour % 6, minute=0)
    out = []
    while t >= ini and len(out) < cfg.get("keep_runs", 16):
        out.append(t)
        t -= timedelta(hours=6)
    return out     # de la más reciente a la más antigua


def actualizar_borrasca(key, cfg, out_dir, catalog, fuentes, deadline, models=None, rehacer=False,
                        prm=None, ahora=None):
    prm = prm or TrackerParams()
    ahora = ahora or datetime.now(timezone.utc)
    sdir = os.path.join(out_dir, key)
    os.makedirs(sdir, exist_ok=True)
    index_path = os.path.join(sdir, "index.json")
    index = load_json(index_path, {"runs": [], "observed": []})
    index["storm"] = cfg["name"]
    if rehacer:
        index["runs"] = []

    log.info("[%s] %s", key, cfg["name"])
    if actualizar_real(index, cfg, tuple(cfg["domain"]), deadline, fuentes, prm, rehacer):
        index["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(index_path, index)

    wanted = models or cfg.get("models") or [k for k, s in catalog.items() if s.get("defecto", True)]
    runs = {r["id"]: r for r in index.get("runs", [])}
    cambios = 0
    for run in candidatas(index, cfg, ahora):
        rid = run_id(run)
        entry = runs.get(rid) or {"id": rid, "t": iso(run), "file": f"run_{rid}.json", "models": [], "perdidos": []}
        pend = [m for m in wanted if m in catalog and m not in entry["models"] and m not in entry.get("perdidos", [])]
        if not pend:
            continue
        seed = pos_en(index["observed"], run)
        if seed is None:
            continue
        # ¿cuáles están ya publicados? (comprobación ligera, en paralelo)
        def pub(m):
            steps = pasos_modelo(catalog[m], run, cfg)
            return m, bool(steps) and catalog[m]["publicado"](run, steps[-1], deadline)
        with ThreadPoolExecutor(len(pend)) as ex:
            estado = dict(ex.map(pub, pend))
        listos = [m for m in pend if estado[m]]
        edad_h = (ahora - run).total_seconds() / 3600
        for m in pend:
            if not estado[m] and edad_h > ABANDONAR_TRAS_H:
                entry.setdefault("perdidos", []).append(m)
        runs[rid] = entry
        if not listos:
            continue

        log.info("  pasada %s: procesando %s", rid, ", ".join(listos))
        run_path = os.path.join(sdir, entry["file"])
        data = load_json(run_path, None) or {"storm": cfg["name"], "run": iso(run), "models": {}}
        data["seed"] = {"lat": round(seed[0], 2), "lon": round(seed[1], 2)}
        # deterministas y ensembles en paralelo (cada uno descarga en paralelo por dentro)
        with ThreadPoolExecutor(3) as ex:
            futs = {m: ex.submit(procesar_modelo, m, catalog[m], run, seed, cfg, prm, deadline) for m in listos}
            for m, fut in futs.items():
                try:
                    res = fut.result()
                except fuentes.SinTiempo:
                    log.warning("  %s %s: sin tiempo, queda pendiente", rid, m)
                    continue
                except fuentes.NoPublicado as e:
                    log.info("  %s %s: aún no publicado (%s)", rid, m, e)
                    continue
                except Exception as e:
                    log.error("  %s %s: ERROR %s", rid, m, e)
                    continue
                if res:
                    data["models"][m] = res
                    entry["models"].append(m)
        if data["models"]:
            data["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # orden estable de modelos (el del catálogo)
            data["models"] = {k: data["models"][k] for k in catalog if k in data["models"]}
            write_json(run_path, data)
            entry["models"] = [k for k in catalog if k in data["models"]]
            runs[rid] = entry
            cambios += 1
            _guardar_index(index, runs, sdir, cfg, index_path)
        if time.monotonic() > deadline:
            log.warning("  límite de tiempo alcanzado; el resto queda para la próxima ejecución")
            break

    _guardar_index(index, runs, sdir, cfg, index_path)
    return cambios


def _guardar_index(index, runs, sdir, cfg, index_path):
    lst = sorted((r for r in runs.values() if r["models"]), key=lambda r: r["id"])
    keep = cfg.get("keep_runs", 16)
    for old in lst[:-keep]:
        try:
            os.remove(os.path.join(sdir, old["file"]))
        except FileNotFoundError:
            pass
    index["runs"] = lst[-keep:]
    index["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(index_path, index)
    update_catalog(os.path.dirname(sdir))


def update_catalog(out_dir):
    """datos_borrascas/storms.json: lista de borrascas disponibles (para el selector de la web)."""
    items = []
    for d in sorted(os.listdir(out_dir)):
        if not os.path.isdir(os.path.join(out_dir, d)):
            continue
        idx = load_json(os.path.join(out_dir, d, "index.json"), None)
        if idx and idx.get("runs"):
            items.append({"key": d, "name": idx.get("storm", d), "last_run": idx["runs"][-1]["t"],
                          "updated": idx.get("updated"), "demo": bool(idx.get("demo"))})
    items.sort(key=lambda x: x["last_run"], reverse=True)
    write_json(os.path.join(out_dir, "storms.json"), {"storms": items})


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storm", help="clave de la borrasca en borrascas.json (por defecto, todas las activas)")
    ap.add_argument("--models", help="lista separada por comas (por defecto, la de borrascas.json)")
    ap.add_argument("--minutos", type=float, default=float(os.environ.get("BORRASCAS_MINUTOS", 30)),
                    help="límite de tiempo total (por defecto 30)")
    ap.add_argument("--rehacer", action="store_true", help="rehace la trayectoria real y todas las pasadas")
    ap.add_argument("--config", default=os.path.join(HERE, "borrascas.json"))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    from . import fuentes
    deadline = time.monotonic() + a.minutos * 60
    cfg = {k: v for k, v in load_json(a.config, {}).items() if not k.startswith("_")}
    storms = [a.storm] if a.storm else [k for k, v in cfg.items() if v.get("active")]
    if not storms:
        log.info("No hay borrascas activas en %s; nada que hacer.", a.config)
        return
    os.makedirs(a.out, exist_ok=True)
    models = a.models.split(",") if a.models else None
    errores = 0
    for s in storms:
        try:
            actualizar_borrasca(s, cfg[s], a.out, fuentes.MODELS, fuentes, deadline, models, a.rehacer)
        except fuentes.SinTiempo:
            log.warning("[%s] sin tiempo; se continuará en la próxima ejecución", s)
        except Exception as e:
            errores += 1
            log.exception("[%s] ERROR: %s", s, e)
    sys.exit(1 if errores else 0)
