"""Estimación de la meteorología en cualquier punto de una isla teniendo en cuenta el relieve.

En Canarias la temperatura y la humedad cambian mucho con la altitud, y la inversión térmica
de los alisios (normalmente entre 800 y 1.500 m) hace que la temperatura deje de bajar o
incluso suba con la altura. Este módulo estima la meteorología de un punto a partir de las
estaciones de AEMET de su misma isla con varios métodos, para compararlos:

    A. ``cercana``          -> valor de la estación más cercana, sin corregir (referencia).
    C. ``gradiente_fijo``   -> estación más cercana corregida con el gradiente estándar
                               (-6,5 ºC/km); la humedad se recalcula manteniendo el punto de rocío.
    D. ``perfil_dia``       -> perfil vertical de ESE día en ESA isla (recta o recta con un codo
                               en la altura de la inversión), evaluado en la altitud del punto.
    E. ``perfil_residuos``  -> perfil del día + interpolación, por distancia inversa, de lo que
                               cada estación se separa del perfil (sus residuos).

``validacion_dejando_una_fuera`` esconde cada estación y la estima con las demás de su isla,
para medir el error de cada método con datos reales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .etiquetas import distancia_haversine_km
from .meteo import humedad_relativa

GRADIENTE_ESTANDAR = -0.0065            # ºC por metro (atmósfera estándar)
ALTURAS_CODO = np.arange(400, 2001, 100)  # alturas candidatas para la inversión (m)
METODOS = ("cercana", "gradiente_fijo", "perfil_dia", "perfil_residuos")


# ---------------------------------------------------------------- humedad

def punto_rocio(t_c, hr) -> np.ndarray:
    """Punto de rocío (ºC) a partir de temperatura (ºC) y humedad relativa (%) (Magnus)."""
    a, b = 17.625, 243.04
    t_c = np.asarray(t_c, dtype=float)
    hr = np.clip(np.asarray(hr, dtype=float), 1e-3, 100.0)
    gamma = np.log(hr / 100.0) + a * t_c / (b + t_c)
    return b * gamma / (a - gamma)


# ---------------------------------------------------------------- perfil vertical

def _bic(sse: float, n: int, k: int) -> float:
    return n * np.log(max(sse, 1e-9) / n) + k * np.log(n)


def ajustar_perfil(z, v, alturas_codo=ALTURAS_CODO, min_por_tramo: int = 2) -> dict:
    """Ajusta el perfil vertical de una variable con las estaciones de un día.

    Prueba una recta ``v = a + b·z`` y una recta con codo
    ``v = a + b·z + c·max(z − z_codo, 0)`` para cada altura candidata, y se queda con el
    modelo de menor BIC (penaliza el parámetro extra para no inventar inversiones).
    Con menos de 3 estaciones o sin desnivel, devuelve un perfil constante (la media).

    Devuelve un diccionario con ``tipo`` ("constante" | "recta" | "codo"), ``coef`` y ``z_codo``.
    """
    z = np.asarray(z, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(z)
    if n < 3 or np.ptp(z) < 100:
        return {"tipo": "constante", "coef": np.array([v.mean() if n else np.nan]), "z_codo": None}

    X = np.column_stack([np.ones(n), z])
    coef, *_ = np.linalg.lstsq(X, v, rcond=None)
    sse = float(((X @ coef - v) ** 2).sum())
    mejor = {"tipo": "recta", "coef": coef, "z_codo": None, "bic": _bic(sse, n, 2)}

    for zc in alturas_codo:
        if (z < zc).sum() < min_por_tramo or (z >= zc).sum() < min_por_tramo:
            continue
        Xc = np.column_stack([np.ones(n), z, np.maximum(z - zc, 0.0)])
        cc, *_ = np.linalg.lstsq(Xc, v, rcond=None)
        sse_c = float(((Xc @ cc - v) ** 2).sum())
        bic_c = _bic(sse_c, n, 4)  # 3 coeficientes + la altura del codo
        if bic_c < mejor["bic"]:
            mejor = {"tipo": "codo", "coef": cc, "z_codo": float(zc), "bic": bic_c}
    return mejor


def evaluar_perfil(perfil: dict, z) -> np.ndarray:
    """Valor del perfil en la(s) altitud(es) ``z``."""
    z = np.asarray(z, dtype=float)
    c = perfil["coef"]
    if perfil["tipo"] == "constante":
        return np.full(z.shape, c[0])
    if perfil["tipo"] == "recta":
        return c[0] + c[1] * z
    return c[0] + c[1] * z + c[2] * np.maximum(z - perfil["z_codo"], 0.0)


# ---------------------------------------------------------------- métodos

def estimar(metodo: str, z_obj: float, dist_km, z_est, v_est, potencia: float = 2.0,
            gradiente: float = GRADIENTE_ESTANDAR, perfil: dict | None = None) -> float:
    """Estima una variable en un punto (altitud ``z_obj``) a partir de las estaciones dadas.

    ``dist_km``, ``z_est`` y ``v_est`` son las distancias al punto, altitudes y valores de las
    estaciones disponibles (sin la estación escondida). ``gradiente`` solo se usa en
    ``gradiente_fijo`` (ºC/m; 0 para variables que no se corrigen con la altura).
    ``perfil`` permite reutilizar un perfil ya ajustado con esas mismas estaciones.
    """
    dist_km = np.asarray(dist_km, dtype=float)
    z_est = np.asarray(z_est, dtype=float)
    v_est = np.asarray(v_est, dtype=float)
    i = int(np.argmin(dist_km))

    if metodo == "cercana":
        return float(v_est[i])
    if metodo == "gradiente_fijo":
        return float(v_est[i] + gradiente * (z_obj - z_est[i]))

    if perfil is None:
        perfil = ajustar_perfil(z_est, v_est)
    base = float(evaluar_perfil(perfil, z_obj))
    if metodo == "perfil_dia":
        return base
    if metodo == "perfil_residuos":
        residuos = v_est - evaluar_perfil(perfil, z_est)
        pesos = 1.0 / np.maximum(dist_km, 0.1) ** potencia
        return base + float((pesos * residuos).sum() / pesos.sum())
    raise ValueError(f"método desconocido: {metodo}")


def estimar_humedad(metodo: str, z_obj: float, dist_km, z_est, tmed_est, td_est, hr_est,
                    perfiles: dict | None = None) -> float:
    """Humedad relativa media (%) estimada.

    ``cercana`` copia la humedad de la estación más cercana. El resto estima la temperatura
    media y el punto de rocío por separado y recalcula la humedad con ambos: así la humedad
    responde a la altitud (y a la inversión) de forma físicamente coherente. En
    ``gradiente_fijo`` el punto de rocío se mantiene (aire con el mismo vapor de agua).
    """
    if metodo == "cercana":
        return float(np.asarray(hr_est, dtype=float)[int(np.argmin(dist_km))])
    perfiles = perfiles or {}
    t = estimar(metodo, z_obj, dist_km, z_est, tmed_est, perfil=perfiles.get("tmed"))
    td = estimar(metodo, z_obj, dist_km, z_est, td_est, gradiente=0.0, perfil=perfiles.get("td"))
    return float(humedad_relativa(t, min(td, t)))


# ---------------------------------------------------------------- validación

def validacion_dejando_una_fuera(df: pd.DataFrame, min_estaciones: int = 3,
                                 paso_dias: int = 1) -> pd.DataFrame:
    """Esconde cada estación, cada día, y la estima con las demás estaciones de su isla.

    ``df`` necesita: ``indicativo``, ``fecha``, ``isla``, ``altitud``, ``lat``, ``lon``,
    ``tmax``, ``tmed``, ``hrMedia``, ``velmedia``. ``paso_dias`` > 1 usa solo uno de cada
    ``paso_dias`` días para acelerar.

    Devuelve una fila por (estación, día, variable, método) con el valor real y el estimado.
    """
    datos = df.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    if paso_dias > 1:
        dias = np.sort(datos["fecha"].unique())[::paso_dias]
        datos = datos[datos["fecha"].isin(dias)]
    datos["td"] = punto_rocio(datos["tmed"], datos["hrMedia"])

    filas = []
    for (isla, fecha), g in datos.groupby(["isla", "fecha"], sort=False):
        if len(g) < min_estaciones + 1:
            continue
        lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
        z = g["altitud"].to_numpy(float)
        d = distancia_haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        cols = {c: g[c].to_numpy(float) for c in ("tmax", "tmed", "hrMedia", "velmedia", "td")}
        ind = g["indicativo"].to_numpy()

        for k in range(len(g)):
            otras = np.arange(len(g)) != k
            base = {"indicativo": ind[k], "fecha": fecha, "isla": isla, "altitud": z[k]}

            for var, grad in (("tmax", GRADIENTE_ESTANDAR), ("velmedia", 0.0)):
                ok = otras & ~np.isnan(cols[var])
                if np.isnan(cols[var][k]) or ok.sum() < min_estaciones:
                    continue
                perfil = ajustar_perfil(z[ok], cols[var][ok])  # una sola vez para los dos métodos de perfil
                for m in METODOS:
                    if var == "velmedia" and m == "gradiente_fijo":
                        continue  # el viento no tiene un gradiente vertical estándar
                    est = estimar(m, z[k], d[k, ok], z[ok], cols[var][ok], gradiente=grad, perfil=perfil)
                    filas.append({**base, "variable": var, "metodo": m,
                                  "real": cols[var][k], "estimado": est})

            ok = otras & ~np.isnan(cols["tmed"]) & ~np.isnan(cols["td"])
            if not np.isnan(cols["hrMedia"][k]) and ok.sum() >= min_estaciones:
                perfiles = {"tmed": ajustar_perfil(z[ok], cols["tmed"][ok]),
                            "td": ajustar_perfil(z[ok], cols["td"][ok])}
                for m in METODOS:
                    est = estimar_humedad(m, z[k], d[k, ok], z[ok], cols["tmed"][ok],
                                          cols["td"][ok], cols["hrMedia"][ok], perfiles=perfiles)
                    filas.append({**base, "variable": "hrMedia", "metodo": m,
                                  "real": cols["hrMedia"][k], "estimado": est})
    res = pd.DataFrame(filas)
    if not res.empty:
        res["error"] = res["estimado"] - res["real"]
    return res


def resumen_errores(res: pd.DataFrame, umbral_altitud: float = 1000.0) -> pd.DataFrame:
    """Error absoluto medio (MAE) y sesgo por variable, método y franja de altitud."""
    r = res.copy()
    r["franja"] = np.where(r["altitud"] >= umbral_altitud, f">= {umbral_altitud:.0f} m", f"< {umbral_altitud:.0f} m")
    todas = r.assign(franja="todas")
    r = pd.concat([r, todas], ignore_index=True)
    return (r.groupby(["variable", "franja", "metodo"])["error"]
            .agg(MAE=lambda e: e.abs().mean(), sesgo="mean", n="size")
            .round(3).reset_index())


def perfiles_diarios(df: pd.DataFrame, variable: str = "tmax", min_estaciones: int = 4) -> pd.DataFrame:
    """Perfil vertical ajustado cada día en cada isla (con todas sus estaciones).

    Sirve para estudiar la inversión: fracción de días con codo y su altura típica.
    """
    filas = []
    for (isla, fecha), g in df.dropna(subset=[variable]).groupby(["isla", "fecha"], sort=False):
        if len(g) < min_estaciones:
            continue
        p = ajustar_perfil(g["altitud"].to_numpy(float), g[variable].to_numpy(float))
        pendiente = p["coef"][1] * 1000 if p["tipo"] != "constante" else np.nan
        filas.append({"isla": isla, "fecha": fecha, "tipo": p["tipo"], "z_codo": p["z_codo"],
                      "pendiente_por_km": pendiente})
    return pd.DataFrame(filas, columns=["isla", "fecha", "tipo", "z_codo", "pendiente_por_km"])
