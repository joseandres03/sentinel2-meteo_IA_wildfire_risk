"""Funciones de humedad (fórmula de Magnus)."""
from __future__ import annotations

import numpy as np


def humedad_relativa(t_c, td_c) -> np.ndarray:
    """Humedad relativa (%) a partir de temperatura y punto de rocío en ºC (fórmula de Magnus)."""
    t_c = np.asarray(t_c, dtype=float)
    td_c = np.asarray(td_c, dtype=float)
    a, b = 17.625, 243.04
    hr = 100.0 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))
    return np.clip(hr, 0.0, 100.0)
