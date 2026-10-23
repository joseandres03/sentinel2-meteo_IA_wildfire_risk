"""Índices espectrales sobre parches Sentinel-2 y el índice heurístico de la versión 1 (TFM).

Los parches tienen forma (alto, ancho, 6) con las bandas en el orden de ``BANDAS_S2``:
B2 (azul), B3 (verde), B4 (rojo), B8 (NIR), B11 (SWIR1), B12 (SWIR2).
"""
from __future__ import annotations

import numpy as np

from . import BANDAS_S2

_IDX = {b: i for i, b in enumerate(BANDAS_S2)}
EPS = 1e-8


def diferencia_normalizada(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), sin divisiones por cero."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a - b) / (a + b + EPS)


def _banda(parche: np.ndarray, nombre: str) -> np.ndarray:
    return np.asarray(parche, dtype=np.float32)[..., _IDX[nombre]]


def ndvi(parche: np.ndarray) -> np.ndarray:
    """Índice de vegetación: (NIR - Rojo) / (NIR + Rojo)."""
    return diferencia_normalizada(_banda(parche, "B8"), _banda(parche, "B4"))


def ndwi(parche: np.ndarray) -> np.ndarray:
    """Índice de agua de McFeeters: (Verde - NIR) / (Verde + NIR)."""
    return diferencia_normalizada(_banda(parche, "B3"), _banda(parche, "B8"))


def ndmi(parche: np.ndarray) -> np.ndarray:
    """Índice de humedad de la vegetación: (NIR - SWIR1) / (NIR + SWIR1)."""
    return diferencia_normalizada(_banda(parche, "B8"), _banda(parche, "B11"))


def nbr(parche: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2)."""
    return diferencia_normalizada(_banda(parche, "B8"), _banda(parche, "B12"))


def mascara_vegetacion(parche: np.ndarray, ndwi_max: float = 0.3, ndvi_min: float = 0.1) -> np.ndarray:
    """Píxeles de tierra (NDWI <= ndwi_max) con vegetación (NDVI > ndvi_min). Mismos umbrales que v1."""
    return (ndwi(parche) <= ndwi_max) & (ndvi(parche) > ndvi_min)


def riesgo_combustible_v1(parche: np.ndarray) -> float:
    """Riesgo de combustible del TFM: media de (1 - NDMI) / 2 sobre la vegetación válida (0 si no hay)."""
    mascara = mascara_vegetacion(parche)
    if not mascara.any():
        return 0.0
    riesgo = np.clip((1.0 - ndmi(parche)) / 2.0, 0.0, 1.0)
    return float(riesgo[mascara].mean())


def riesgo_meteorologico_v1(tmax, hr_media, viento) -> np.ndarray:
    """Riesgo meteorológico del TFM a partir de umbrales de temperatura (ºC), humedad (%) y viento."""
    r_temp = np.clip((np.asarray(tmax, dtype=float) - 15.0) / 25.0, 0.0, 1.0)
    r_hum = np.clip((80.0 - np.asarray(hr_media, dtype=float)) / 60.0, 0.0, 1.0)
    r_viento = np.clip(np.asarray(viento, dtype=float) / 50.0, 0.0, 1.0)
    return (r_temp + r_hum + r_viento) / 3.0


def indice_v1(riesgo_meteo, riesgo_combustible) -> np.ndarray:
    """Índice total del TFM (variable objetivo de la v1): 0,35 meteo + 0,65 combustible.

    En la v2 se usa como **modelo de referencia** (baseline) frente a los modelos entrenados
    con incendios reales.
    """
    return 0.35 * np.asarray(riesgo_meteo, dtype=float) + 0.65 * np.asarray(riesgo_combustible, dtype=float)


def resumen_espectral(parche: np.ndarray) -> dict:
    """Estadísticos espectrales de la vegetación válida de un parche, útiles como variables tabulares."""
    mascara = mascara_vegetacion(parche)
    resumen = {"frac_vegetacion": float(mascara.mean())}
    for nombre, funcion in (("ndvi", ndvi), ("ndmi", ndmi), ("nbr", nbr)):
        valores = funcion(parche)[mascara]
        if valores.size == 0:
            resumen[f"{nombre}_media"] = np.nan
            resumen[f"{nombre}_p10"] = np.nan
        else:
            resumen[f"{nombre}_media"] = float(valores.mean())
            resumen[f"{nombre}_p10"] = float(np.percentile(valores, 10))
    return resumen
