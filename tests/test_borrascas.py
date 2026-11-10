"""Tests del seguimiento de borrascas (``src/borrascas``). No requieren conexión."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from borrascas.tracker import Grid, track  # noqa: E402
from borrascas.productos import canarias_probabilities, ensemble_mean  # noqa: E402

LATS = np.arange(15, 55.01, 0.5)
LONS = np.arange(-50, 5.01, 0.5)
LA, LO = np.meshgrid(LATS, LONS, indexing="ij")
G = Grid(LATS, LONS)


def low(lat, lon, p, r=380):
    d2 = ((LA - lat) * 111) ** 2 + ((LO - lon) * 111 * np.cos(np.radians(lat))) ** 2
    return -(1016 - p) * np.exp(-d2 / (2 * r ** 2))


def test_sigue_la_borrasca_y_no_salta_a_otra_mas_profunda():
    fields = {}
    for h in range(0, 121, 6):
        lat, lon = 34 - h * 0.04, -22 + h * 0.03
        fields[h] = 1016 + low(lat, lon, 995) + low(50 - h * 0.02, -35 + h * 0.02, 970, 700)
    tr = track(fields, G, 34, -22)
    assert len(tr) == 21
    assert abs(tr[-1].lat - (34 - 4.8)) < 0.5 and abs(tr[-1].lon - (-22 + 3.6)) < 0.5
    assert all(p.p > 990 for p in tr)          # nunca cogió la de 970 hPa


def test_termina_cuando_la_borrasca_se_rellena():
    fields = {h: 1016 + low(30, -20, 995 + h * 0.5) for h in range(0, 121, 6)}
    tr = track(fields, G, 30, -20)
    assert 0 < len(tr) < 21


def test_baja_abierta_en_el_gradiente_de_otra():
    # sin isobara cerrada: debe encontrarla igualmente por el laplaciano (plan B)
    f = 1016 + low(34, -22, 995) + low(44, -30, 970, 700)
    tr = track({0: f}, G, 34, -22)
    assert tr and abs(tr[0].lat - 34) < 0.6 and abs(tr[0].lon + 22) < 0.6


def test_sin_borrasca_cerca_de_la_semilla():
    fields = {0: 1016 + low(45, -40, 990)}
    assert track(fields, G, 28, -16) == []


def test_media_y_probabilidades_canarias():
    cerca = [{"h": 0, "lat": 33.0, "lon": -22.0, "p": 1000}, {"h": 24, "lat": 28.3, "lon": -16.7, "p": 998}]
    lejos = [{"h": 0, "lat": 33.0, "lon": -22.0, "p": 1000}, {"h": 24, "lat": 34.0, "lon": -25.0, "p": 998}]
    prob = canarias_probabilities([cerca, lejos, lejos, lejos])
    assert prob["100"] == 25 and prob["300"] == 25
    media, dispersion = ensemble_mean([cerca, lejos])
    assert len(media) == 2 and media[0]["lat"] == 33.0 and dispersion[1]["km"] > 0


def test_extremo_a_extremo_con_datos_sinteticos(tmp_path):
    import json
    from borrascas import demo
    demo.main(str(tmp_path))
    idx = json.loads((tmp_path / "demo" / "index.json").read_text())
    assert len(idx["runs"]) == 5 and len(idx["observed"]) == 5
    run = json.loads((tmp_path / "demo" / idx["runs"][-1]["file"]).read_text())
    assert run["models"]["ECMWF-ENS"]["found"] == 51
    assert run["models"]["GEFS"]["canarias"]["300"] > 0
