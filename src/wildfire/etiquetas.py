"""Construcción de la variable objetivo a partir de incendios observados.

Flujo:
    1. ``filtrar_exclusiones``: elimina focos que no son incendios forestales (p. ej. la
       erupción de La Palma de 2021, que el satélite detecta como "fuego").
    2. ``agrupar_eventos``: agrupa los focos térmicos (FIRMS) cercanos en espacio y tiempo
       en un único evento. La fecha del evento es la de su primera detección.
    3. ``muestrear_negativos``: genera puntos y fechas sin incendio, lejos en espacio y
       tiempo de cualquier evento, a partir de un conjunto de puntos con vegetación.
    4. ``construir_dataset``: une positivos (1) y negativos (0) en una única tabla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RADIO_TIERRA_KM = 6371.0088

# Periodos y zonas en los que los focos térmicos NO corresponden a incendios forestales.
# bbox = (lon_min, lat_min, lon_max, lat_max)
EXCLUSIONES = [
    {
        "motivo": "Erupción volcánica de Cumbre Vieja (La Palma)",
        "inicio": "2021-09-19",
        "fin": "2022-01-31",
        "bbox": (-17.98, 28.55, -17.82, 28.68),
    },
]


def distancia_haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia ortodrómica en km (admite arrays y difusión de NumPy)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def filtrar_exclusiones(focos: pd.DataFrame, exclusiones=EXCLUSIONES) -> pd.DataFrame:
    """Descarta los focos que caen dentro de alguna zona y periodo de exclusión.

    ``focos`` debe tener las columnas ``lat``, ``lon`` y ``fecha``.
    """
    fechas = pd.to_datetime(focos["fecha"])
    descartar = np.zeros(len(focos), dtype=bool)
    for exc in exclusiones:
        lon_min, lat_min, lon_max, lat_max = exc["bbox"]
        en_zona = focos["lon"].between(lon_min, lon_max) & focos["lat"].between(lat_min, lat_max)
        en_fecha = fechas.between(pd.Timestamp(exc["inicio"]), pd.Timestamp(exc["fin"]))
        descartar |= (en_zona & en_fecha).to_numpy()
    return focos.loc[~descartar].reset_index(drop=True)


def agrupar_eventos(focos: pd.DataFrame, radio_km: float = 2.0, ventana_dias: int = 3) -> pd.DataFrame:
    """Agrupa focos en eventos de incendio.

    Dos focos pertenecen al mismo evento si están a menos de ``radio_km`` y a menos de
    ``ventana_dias`` días de algún foco del evento (agrupamiento por enlace simple).

    Devuelve una fila por evento con: ``id_evento``, ``fecha`` (primera detección),
    ``fecha_fin``, ``lat``/``lon`` (foco de la primera detección), ``n_focos``.
    """
    if focos.empty:
        return pd.DataFrame(columns=["id_evento", "fecha", "fecha_fin", "lat", "lon", "n_focos"])

    df = focos.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    df = df.sort_values("fecha").reset_index(drop=True)
    n = len(df)
    lat = df["lat"].to_numpy(float)
    lon = df["lon"].to_numpy(float)
    dias = (df["fecha"] - df["fecha"].min()).dt.days.to_numpy()

    # Unión-búsqueda (union-find) sobre los pares de focos cercanos en espacio y tiempo
    padre = np.arange(n)

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    inicio = 0
    for i in range(n):
        while dias[i] - dias[inicio] > ventana_dias:
            inicio += 1
        if inicio < i:
            cand = np.arange(inicio, i)
            cerca = distancia_haversine_km(lat[i], lon[i], lat[cand], lon[cand]) <= radio_km
            for j in cand[cerca]:
                ri, rj = raiz(i), raiz(j)
                if ri != rj:
                    padre[ri] = rj

    df["grupo"] = [raiz(i) for i in range(n)]
    eventos = []
    for _, g in df.groupby("grupo", sort=False):
        primero = g.iloc[0]  # g está ordenado por fecha
        eventos.append({
            "fecha": primero["fecha"],
            "fecha_fin": g["fecha"].max(),
            "lat": float(primero["lat"]),
            "lon": float(primero["lon"]),
            "n_focos": int(len(g)),
        })
    eventos = pd.DataFrame(eventos).sort_values("fecha").reset_index(drop=True)
    eventos.insert(0, "id_evento", np.arange(len(eventos)))
    return eventos


def muestrear_negativos(
    puntos_vegetacion: pd.DataFrame,
    eventos: pd.DataFrame,
    n: int,
    fecha_inicio: str,
    fecha_fin: str,
    exclusion_km: float = 5.0,
    exclusion_dias: int = 30,
    meses: list | None = None,
    semilla: int = 42,
    max_iter: int = 50,
) -> pd.DataFrame:
    """Genera ``n`` ejemplos sin incendio (punto con vegetación + fecha).

    Se descartan los candidatos a menos de ``exclusion_km`` y ``exclusion_dias`` de cualquier
    evento, para no etiquetar como "sin incendio" algo que sí ardió.
    ``meses`` permite restringir las fechas (p. ej. a la campaña de incendios) para que la
    distribución estacional de negativos sea comparable a la de positivos.
    """
    rng = np.random.default_rng(semilla)
    fechas_posibles = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    if meses:
        fechas_posibles = fechas_posibles[fechas_posibles.month.isin(meses)]
    if len(fechas_posibles) == 0 or puntos_vegetacion.empty:
        raise ValueError("no hay fechas o puntos candidatos para muestrear negativos")

    ev_lat = eventos["lat"].to_numpy(float) if len(eventos) else np.empty(0)
    ev_lon = eventos["lon"].to_numpy(float) if len(eventos) else np.empty(0)
    ev_fecha = pd.to_datetime(eventos["fecha"]).to_numpy() if len(eventos) else np.empty(0, dtype="datetime64[ns]")

    aceptados = []
    total = 0
    for _ in range(max_iter):
        faltan = n - total
        if faltan <= 0:
            break
        lote = int(faltan * 1.5) + 10
        idx = rng.integers(0, len(puntos_vegetacion), lote)
        cand = puntos_vegetacion.iloc[idx][["lat", "lon"]].reset_index(drop=True)
        cand["fecha"] = fechas_posibles[rng.integers(0, len(fechas_posibles), lote)]
        if len(ev_lat):
            d_km = distancia_haversine_km(cand["lat"].to_numpy()[:, None], cand["lon"].to_numpy()[:, None],
                                          ev_lat[None, :], ev_lon[None, :])
            d_dias = np.abs((cand["fecha"].to_numpy()[:, None] - ev_fecha[None, :]) / np.timedelta64(1, "D"))
            conflicto = ((d_km <= exclusion_km) & (d_dias <= exclusion_dias)).any(axis=1)
            cand = cand.loc[~conflicto]
        cand = cand.iloc[:faltan]
        aceptados.append(cand)
        total += len(cand)

    negativos = pd.concat(aceptados, ignore_index=True) if aceptados else pd.DataFrame(columns=["lat", "lon", "fecha"])
    if len(negativos) < n:
        raise RuntimeError(f"solo se generaron {len(negativos)} de {n} negativos; relaja las exclusiones")
    return negativos.reset_index(drop=True)


def construir_dataset(eventos: pd.DataFrame, negativos: pd.DataFrame) -> pd.DataFrame:
    """Tabla final ``lat, lon, fecha, incendio`` (1 = evento observado, 0 = sin incendio)."""
    pos = eventos[["lat", "lon", "fecha"]].assign(incendio=1)
    neg = negativos[["lat", "lon", "fecha"]].assign(incendio=0)
    df = pd.concat([pos, neg], ignore_index=True)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    return df.sort_values("fecha").reset_index(drop=True)
