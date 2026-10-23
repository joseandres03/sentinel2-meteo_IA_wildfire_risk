"""Conversión de variables meteorológicas de ERA5-Land a las unidades del proyecto."""
from __future__ import annotations

import numpy as np
import pandas as pd

KELVIN = 273.15


def humedad_relativa(t_c, td_c) -> np.ndarray:
    """Humedad relativa (%) a partir de temperatura y punto de rocío en ºC (fórmula de Magnus)."""
    t_c = np.asarray(t_c, dtype=float)
    td_c = np.asarray(td_c, dtype=float)
    a, b = 17.625, 243.04
    hr = 100.0 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))
    return np.clip(hr, 0.0, 100.0)


def derivar_meteo_era5(df: pd.DataFrame) -> pd.DataFrame:
    """Añade ``tmax_c``, ``hr_media``, ``viento_ms`` y ``precip_30d_mm`` a partir de las bandas de ERA5-Land.

    Columnas de entrada (las que exporta ``wildfire.gee``):
    ``temperature_2m_max`` y ``temperature_2m`` (K), ``dewpoint_temperature_2m`` (K),
    ``u_component_of_wind_10m`` y ``v_component_of_wind_10m`` (m/s), ``precip_30d`` (m).
    """
    out = df.copy()
    out["tmax_c"] = out["temperature_2m_max"] - KELVIN
    out["hr_media"] = humedad_relativa(out["temperature_2m"] - KELVIN, out["dewpoint_temperature_2m"] - KELVIN)
    out["viento_ms"] = np.hypot(out["u_component_of_wind_10m"], out["v_component_of_wind_10m"])
    if "precip_30d" in out:
        out["precip_30d_mm"] = out["precip_30d"] * 1000.0
    return out
