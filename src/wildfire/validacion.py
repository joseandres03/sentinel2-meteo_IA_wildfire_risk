"""Validación sin fugas espaciales ni temporales, métricas de clasificación y calibración.

Debilidades de la evaluación de la versión 1 que cubre este módulo:
    1. Separar espacial y temporalmente entrenamiento y prueba  -> ``bloques_espaciales``,
       ``isla_de``, ``particion_temporal`` y ``prediccion_fuera_de_bloque``.
    2. Calibración de probabilidades                            -> ``tabla_fiabilidad``,
       ``corregir_prevalencia``.
    3. Sensibilidad y precisión en zonas de alto riesgo         -> ``metricas_clasificacion``,
       ``captura_en_top``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import BBOX_ISLAS

KM_POR_GRADO_LAT = 111.32


def isla_de(lat, lon) -> np.ndarray:
    """Nombre de la isla a la que pertenece cada punto (``"Fuera"`` si no cae en ninguna caja)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    resultado = np.full(lat.shape, "Fuera", dtype=object)
    for nombre, (lon_min, lat_min, lon_max, lat_max) in BBOX_ISLAS.items():
        dentro = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        resultado[dentro & (resultado == "Fuera")] = nombre
    return resultado


def bloques_espaciales(lat, lon, tam_km: float = 10.0) -> np.ndarray:
    """Identificador de la celda de una malla de ``tam_km`` x ``tam_km`` para cada punto.

    Los puntos de una misma celda van siempre juntos a entrenamiento o a prueba, de modo que
    el modelo no se evalúa sobre zonas vecinas casi idénticas a las que ya ha visto.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    km_por_grado_lon = KM_POR_GRADO_LAT * np.cos(np.radians(28.3))  # latitud media de Canarias
    fila = np.floor(lat * KM_POR_GRADO_LAT / tam_km).astype(int)
    col = np.floor(lon * km_por_grado_lon / tam_km).astype(int)
    return np.array([f"{f}_{c}" for f, c in zip(fila, col)], dtype=object)


def particion_temporal(fechas, fecha_corte: str) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras booleanas (entrenamiento, prueba): antes y a partir de ``fecha_corte``."""
    fechas = pd.to_datetime(pd.Series(fechas)).to_numpy()
    corte = np.datetime64(pd.Timestamp(fecha_corte))
    entrenamiento = fechas < corte
    return entrenamiento, ~entrenamiento


def prediccion_fuera_de_bloque(modelo, X, y, grupos, n_splits: int = 5) -> np.ndarray:
    """Predicciones *out-of-fold* con validación cruzada agrupada (GroupKFold).

    Cada observación se predice con un modelo que no ha visto ninguna observación de su grupo
    (bloque espacial o isla). Devuelve la probabilidad de la clase positiva.
    """
    X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    n_grupos = len(np.unique(grupos))
    if n_grupos < 2:
        raise ValueError("se necesitan al menos 2 grupos distintos")
    cv = GroupKFold(n_splits=min(n_splits, n_grupos))
    oof = np.full(len(y), np.nan)
    for idx_ent, idx_val in cv.split(X, y, grupos):
        m = clone(modelo)
        X_ent = X.iloc[idx_ent] if isinstance(X, pd.DataFrame) else X[idx_ent]
        X_val = X.iloc[idx_val] if isinstance(X, pd.DataFrame) else X[idx_val]
        m.fit(X_ent, y[idx_ent])
        oof[idx_val] = m.predict_proba(X_val)[:, 1]
    return oof


def captura_en_top(y, p, fraccion: float = 0.1) -> float:
    """Proporción de incendios reales que caen en la ``fraccion`` de casos con mayor riesgo predicho.

    Responde a la pregunta operativa: «si vigilo solo el 10 % de zonas más peligrosas,
    ¿qué parte de los incendios cubro?». Un modelo aleatorio obtiene ≈ ``fraccion``.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(fraccion * len(p))))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def metricas_clasificacion(y, p, umbral: float = 0.5) -> dict:
    """Métricas de discriminación, de la zona de alto riesgo y de calibración."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= umbral
    vp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {
        "n": int(len(y)),
        "prevalencia": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "sensibilidad": vp / (vp + fn) if (vp + fn) else float("nan"),
        "precision": vp / (vp + fp) if (vp + fp) else float("nan"),
        "falsos_positivos": fp,
        "incendios_omitidos": fn,
        "captura_top10": captura_en_top(y, p, 0.10),
        "captura_top20": captura_en_top(y, p, 0.20),
    }


def tabla_fiabilidad(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Diagrama de fiabilidad en forma de tabla: probabilidad media predicha frente a frecuencia observada.

    Un modelo bien calibrado tiene ``prob_media`` ≈ ``frecuencia_observada`` en cada intervalo.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    bordes = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bordes[1:-1]), 0, n_bins - 1)
    filas = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            filas.append({
                "intervalo": f"{bordes[b]:.1f}-{bordes[b + 1]:.1f}",
                "n": int(m.sum()),
                "prob_media": float(p[m].mean()),
                "frecuencia_observada": float(y[m].mean()),
            })
    return pd.DataFrame(filas)


def corregir_prevalencia(p, prevalencia_muestra: float, prevalencia_real: float) -> np.ndarray:
    """Ajusta probabilidades entrenadas con negativos submuestreados a la prevalencia real.

    Si el dataset tiene, por ejemplo, 1 incendio por cada 10 negativos, las probabilidades
    salen infladas respecto a la realidad (donde los incendios son mucho más raros). Esta
    corrección de cambio de prior (Elkan, 2001) reescala los *odds*:

        odds_real = odds_modelo * [π_real / (1 - π_real)] / [π_muestra / (1 - π_muestra)]
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    factor = (prevalencia_real / (1 - prevalencia_real)) / (prevalencia_muestra / (1 - prevalencia_muestra))
    odds = p / (1 - p) * factor
    return odds / (1 + odds)
