#!/usr/bin/env python3
"""
Genera los JSON de trayectorias de borrascas para el visor (docs/trayectorias.html).

Uso (desde la raíz del repositorio):
  python src/trayectorias.py                       # borrascas activas, última pasada disponible
  python src/trayectorias.py --storm charles --run 2026111000
  python src/trayectorias.py --storm charles --run 2026111000 --backfill 4
  python src/trayectorias.py --models GFS,GEFS     # sólo algunos modelos

Salida (por defecto en docs/datos_borrascas/<borrasca>/):
  index.json           lista de pasadas + trayectoria "observada" (análisis t+0 de cada pasada)
  run_YYYYMMDDHH.json  todas las trayectorias de todos los modelos para esa pasada
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from .productos import summarize_model
from .tracker import TrackerParams, track

log = logging.getLogger("borrascas")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
OUT_DEFAULT = os.path.join(BASE_DIR, "docs", "datos_borrascas")


def parse_time(s: str) -> datetime:
    s = s.replace("Z", "")
    for fmt in ("%Y%m%d%H", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"fecha no reconocida: {s}")


def latest_run(now=None, delay_h=9) -> datetime:
    """Última pasada 00/12 UTC que debería estar completa (ENS incluido)."""
    now = now or datetime.now(timezone.utc)
    t = now - timedelta(hours=delay_h)
    return t.replace(hour=0 if t.hour < 12 else 12, minute=0, second=0, microsecond=0)


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
    os.replace(tmp, path)   # escritura atómica: la web nunca lee un JSON a medias


def interp_track(track_pts, h):
    """Posición interpolada a la hora h (relativa a la init de esa pasada)."""
    for a, b in zip(track_pts, track_pts[1:]):
        if a["h"] <= h <= b["h"]:
            f = (h - a["h"]) / max(b["h"] - a["h"], 1)
            return a["lat"] + f * (b["lat"] - a["lat"]), a["lon"] + f * (b["lon"] - a["lon"])
    return None


def choose_seed(storm_cfg, index, run: datetime):
    """Semilla para esta pasada: la trayectoria determinista de la pasada anterior
    más reciente, interpolada a la hora de esta pasada. Si no hay, la de config."""
    for prev in sorted(index.get("runs", []), key=lambda r: r["id"], reverse=True):
        prev_t = parse_time(prev["id"])
        if prev_t >= run:
            continue
        seed_track = prev.get("seed_track")
        if seed_track:
            pos = interp_track(seed_track, (run - prev_t).total_seconds() / 3600)
            if pos:
                return pos
    s = storm_cfg["seed"]
    return s["lat"], s["lon"]


def build_model(key, spec, run, seed, storm_cfg, prm, fetch_kwargs):
    max_lead = storm_cfg.get("max_lead_h", 120)
    if spec["kind"] == "deterministic" and key == "ECMWF" and run.hour in (6, 18):
        max_lead = min(max_lead, 90)
    steps = list(range(0, max_lead + 1, spec["step"]))
    log.info("%s: descargando %d pasos", key, len(steps))
    fields, grid = spec["fetch"](run, steps, tuple(storm_cfg["domain"]), **fetch_kwargs)
    if grid is None:
        raise RuntimeError("sin datos")
    return tracks_from_fields(key, spec, fields, grid, seed, prm)


def tracks_from_fields(key, spec, fields, grid, seed, prm):
    members = []
    for mem in sorted(fields):
        pts = track(fields[mem], grid, seed[0], seed[1], prm)
        members.append({"id": mem, "track": [p.as_dict() for p in pts]})
    out = {"label": spec["label"], "kind": spec["kind"], "color": spec["color"],
           "parent": spec.get("parent"), "members": members,
           "found": sum(1 for m in members if m["track"]), "total": len(members)}
    return summarize_model(out)


def process_run(storm_key, storm_cfg, run, out_dir, models=None, fetch_kwargs=None,
                prm=None, model_fetchers=None):
    if model_fetchers is None:
        from .fuentes import MODELS as model_fetchers
    catalog = model_fetchers
    prm = prm or TrackerParams()
    sdir = os.path.join(out_dir, storm_key)
    os.makedirs(sdir, exist_ok=True)
    index_path = os.path.join(sdir, "index.json")
    index = load_json(index_path, {"storm": storm_cfg["name"], "runs": [], "observed": []})

    seed = choose_seed(storm_cfg, index, run)
    log.info("[%s] pasada %s  semilla %.1f, %.1f", storm_key, run_id(run), *seed)

    result = {"storm": storm_cfg["name"], "run": run.isoformat().replace("+00:00", "Z"),
              "seed": {"lat": round(seed[0], 2), "lon": round(seed[1], 2)},
              "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "models": {}}
    for key in (models or storm_cfg["models"]):
        spec = catalog.get(key)
        if not spec:
            log.warning("modelo desconocido %s", key)
            continue
        try:
            result["models"][key] = build_model(key, spec, run, seed, storm_cfg, prm, fetch_kwargs or {})
            m = result["models"][key]
            log.info("  %-10s %d/%d trayectorias", key, m["found"], m["total"])
        except Exception as e:
            log.error("  %-10s ERROR: %s", key, e)

    if not result["models"]:
        log.error("ningún modelo disponible para %s", run_id(run))
        return None

    rid = run_id(run)
    write_json(os.path.join(sdir, f"run_{rid}.json"), result)

    # Trayectoria determinista de referencia (para la semilla de la siguiente pasada)
    seed_track = []
    for k in ("ECMWF", "GFS", "AIFS"):
        m = result["models"].get(k)
        if m and m["members"] and m["members"][0]["track"]:
            seed_track = m["members"][0]["track"]
            break

    # Posición "observada" = análisis (t+0) de los deterministas de esta pasada
    obs = None
    for k in ("ECMWF", "GFS", "AIFS"):
        m = result["models"].get(k)
        if m and m["members"] and m["members"][0]["track"] and m["members"][0]["track"][0]["h"] == 0:
            p0 = m["members"][0]["track"][0]
            obs = {"t": result["run"], "lat": p0["lat"], "lon": p0["lon"], "p": p0["p"], "src": k}
            break

    runs = [r for r in index["runs"] if r["id"] != rid]
    runs.append({"id": rid, "t": result["run"], "file": f"run_{rid}.json",
                 "models": list(result["models"]), "seed_track": seed_track})
    runs.sort(key=lambda r: r["id"])
    keep = storm_cfg.get("keep_runs", 20)
    for old in runs[:-keep]:
        try:
            os.remove(os.path.join(sdir, old["file"]))
        except FileNotFoundError:
            pass
    index["runs"] = runs[-keep:]
    if obs:
        index["observed"] = sorted([o for o in index.get("observed", []) if o["t"] != obs["t"]] + [obs],
                                   key=lambda o: o["t"])
    index["storm"] = storm_cfg["name"]
    index["updated"] = result["generated"]
    write_json(index_path, index)
    update_catalog(out_dir)
    return result


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storm", help="clave de la borrasca en storms.json (por defecto, todas las activas)")
    ap.add_argument("--run", help="pasada YYYYMMDDHH (por defecto, la última 00/12 disponible)")
    ap.add_argument("--backfill", type=int, default=0, help="procesar también N pasadas anteriores")
    ap.add_argument("--models", help="lista separada por comas (por defecto, la de storms.json)")
    ap.add_argument("--config", default=os.path.join(HERE, "borrascas.json"))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = {k: v for k, v in load_json(a.config, {}).items() if not k.startswith("_")}
    storms = [a.storm] if a.storm else [k for k, v in cfg.items() if v.get("active")]
    last = parse_time(a.run) if a.run else latest_run()
    runs = [last - timedelta(hours=12 * i) for i in range(a.backfill, -1, -1)]
    models = a.models.split(",") if a.models else None

    if not storms:
        log.info("No hay borrascas activas en %s; nada que hacer.", a.config)
        return
    os.makedirs(a.out, exist_ok=True)
    ok = True
    for s in storms:
        for r in runs:   # de la más antigua a la más reciente (la semilla se encadena)
            ok &= process_run(s, cfg[s], r, a.out, models) is not None
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
