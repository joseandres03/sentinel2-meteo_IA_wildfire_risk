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

    F. ``perfil_sondeo`` y ``perfil_sondeo_residuos`` -> como D y E, pero con el codo fijado en
                               la altura de la inversión MEDIDA ese día por el radiosondeo de
                               Güímar (Tenerife; datos de alta resolución en PANGAEA). Si la
                               inversión está por encima de las estaciones de la isla, el perfil
                               es una recta (toda la isla queda bajo la inversión ese día).

    G. ``forma_sondeo`` y ``forma_sondeo_residuos`` -> recta de las estaciones por debajo de la
                               inversión y, por encima, la FORMA del perfil medido por el globo
                               (temperatura y punto de rocío). Corrige la parte alta del perfil,
                               donde hay muy pocas estaciones.

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
METODOS_BASE = ("cercana", "gradiente_fijo", "perfil_dia", "perfil_residuos")
METODOS_SONDEO = ("perfil_sondeo", "perfil_sondeo_residuos")
METODOS_FORMA = ("forma_sondeo", "forma_sondeo_residuos")
METODOS = METODOS_BASE + METODOS_SONDEO + METODOS_FORMA
Z_ANCLA_SIN_INVERSION = 1000.0  # m: ancla del perfil del sondeo los días sin inversión


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


def ajustar_perfil_sondeo(z, v, z_inversion, min_por_tramo: int = 2) -> dict:
    """Perfil con el codo fijado en la altura de la inversión medida por el radiosondeo.

    - Sin dato de inversión ese día -> mismo perfil que ``ajustar_perfil`` (codo buscado con BIC).
    - Inversión por encima de casi todas las estaciones (menos de ``min_por_tramo`` por encima)
      -> recta: la isla, o casi toda, queda bajo la inversión ese día.
    - En otro caso -> recta con codo exactamente en ``z_inversion``.
    """
    z = np.asarray(z, dtype=float)
    v = np.asarray(v, dtype=float)
    if z_inversion is None or not np.isfinite(z_inversion):
        return ajustar_perfil(z, v)
    n = len(z)
    if n < 3 or np.ptp(z) < 100:
        return {"tipo": "constante", "coef": np.array([v.mean() if n else np.nan]), "z_codo": None}
    arriba, abajo = (z >= z_inversion).sum(), (z < z_inversion).sum()
    if arriba < min_por_tramo or abajo < min_por_tramo:
        X = np.column_stack([np.ones(n), z])
        coef, *_ = np.linalg.lstsq(X, v, rcond=None)
        return {"tipo": "recta", "coef": coef, "z_codo": None}
    Xc = np.column_stack([np.ones(n), z, np.maximum(z - z_inversion, 0.0)])
    cc, *_ = np.linalg.lstsq(Xc, v, rcond=None)
    return {"tipo": "codo", "coef": cc, "z_codo": float(z_inversion)}


# ---------------------------------------------------------------- radiosondeo

def inversion_de_sondeo(z, t, z_min: float = 200.0, z_max: float = 3500.0,
                        intensidad_min: float = 0.5, ventana: float = 600.0,
                        espesor_max: float = 1500.0) -> dict:
    """Base, cima e intensidad de la inversión térmica más baja de un sondeo.

    Se busca el nivel más bajo (entre ``z_min`` y ``z_max``) por encima del cual, en los
    ``ventana`` metros siguientes, la temperatura llega a ser al menos ``intensidad_min`` ºC
    mayor. La **base** es el punto más frío entre ese nivel y ese máximo (donde empieza a
    calentarse) y la **cima**, el nivel más cálido hasta ``espesor_max`` metros por encima. Comparar con el máximo de la ventana, y no nivel a nivel, hace el método robusto
    al ruido de los sondeos de alta resolución y a pequeñas bajadas dentro de la inversión.
    Sin inversión -> valores NaN.
    """
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(z) & np.isfinite(t)
    z, t = z[ok], t[ok]
    orden = np.argsort(z)
    z, t = z[orden], t[orden]
    _, unicos = np.unique(z, return_index=True)
    z, t = z[unicos], t[unicos]
    for i in np.where((z >= z_min) & (z <= z_max))[0]:
        arriba = (z > z[i]) & (z <= z[i] + ventana)
        if arriba.any() and t[arriba].max() - t[i] >= intensidad_min:
            # la base es el punto más frío entre este nivel y el máximo de la ventana
            k_max = np.where(arriba)[0][np.argmax(t[arriba])]
            b = i + int(np.argmin(t[i:k_max + 1]))
            capa = (z > z[b]) & (z <= z[b] + espesor_max)
            k = np.where(capa)[0][np.argmax(t[capa])]
            return {"z_base": float(z[b]), "z_cima": float(z[k]), "intensidad": float(t[k] - t[b])}
    return {"z_base": np.nan, "z_cima": np.nan, "intensidad": np.nan}


def leer_radiosondeo_pangaea(texto: str) -> pd.DataFrame:
    """Lee un fichero de PANGAEA «Radiosonde measurements from station Izaña (AAAA-MM)».

    Formato: bloque de metadatos entre ``/*`` y ``*/`` y después una tabla separada por
    tabuladores con ``Date/Time``, ``Altitude [m]``, ``PPPP [hPa]``, ``TTT [°C]``,
    ``TdTdTd [°C]``, ``dd [deg]`` y ``ff [m/s]``. Cada sondeo se identifica por su fecha y hora.

    Devuelve ``fecha``, ``hora``, ``altura_m``, ``temp_c``, ``td_c``.
    """
    import io
    if "*/" in texto:
        texto = texto.split("*/", 1)[1]
    df = pd.read_csv(io.StringIO(texto.strip("\n")), sep="\t")
    col = lambda prefijo: next(c for c in df.columns if c.startswith(prefijo))
    instante = pd.to_datetime(df[col("Date/Time")])
    return pd.DataFrame({
        "fecha": instante.dt.normalize(), "hora": instante.dt.hour,
        "altura_m": pd.to_numeric(df[col("Altitude")], errors="coerce"),
        "temp_c": pd.to_numeric(df[col("TTT")], errors="coerce"),
        "td_c": pd.to_numeric(df[col("TdTdTd")], errors="coerce"),
    }).dropna(subset=["altura_m", "temp_c"])


def remuestrear_sondeo(z, t, td=None, paso: float = 25.0, z_max: float = 4000.0):
    """Promedia un sondeo de alta resolución en capas de ``paso`` metros hasta ``z_max``.

    Suaviza el ruido del sensor y deja todos los sondeos en la misma malla de alturas.
    Devuelve ``(z_malla, t_malla, td_malla)`` (``td_malla`` es None si no se da ``td``).
    """
    z = np.asarray(z, dtype=float)
    malla = np.arange(paso / 2, z_max, paso)
    capa = np.floor(z / paso).astype(int)
    def promediar(v):
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(v) & (z < z_max)
        suma = np.bincount(capa[ok], weights=v[ok], minlength=len(malla))[:len(malla)]
        n = np.bincount(capa[ok], minlength=len(malla))[:len(malla)]
        con_dato = n > 0
        if con_dato.sum() < 2:
            return np.full(len(malla), np.nan)
        return np.interp(malla, malla[con_dato], suma[con_dato] / n[con_dato],
                         left=np.nan, right=np.nan)
    return malla, promediar(t), (promediar(td) if td is not None else None)


def perfiles_de_sondeo(niveles: pd.DataFrame, horas=(11, 12, 13), paso: float = 25.0,
                       z_max: float = 4000.0, **kwargs_inversion) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Para cada día, el sondeo de mediodía remuestreado y su inversión.

    ``niveles`` necesita ``fecha``, ``hora``, ``altura_m``, ``temp_c`` y, si se quiere estimar la
    humedad en altura, ``td_c``.

    Devuelve ``(perfiles, inversiones)``: ``perfiles`` con ``fecha``, ``altura_m``, ``temp_c``,
    ``td_c`` en la malla común, e ``inversiones`` con ``fecha``, ``z_base``, ``z_cima``, ``intensidad``.
    """
    sel = niveles[niveles["hora"].isin(horas)]
    perfiles, inversiones = [], []
    for fecha, g in sel.groupby("fecha"):
        hora = min(g["hora"].unique(), key=lambda h: abs(h - 12))
        s = g[g["hora"] == hora]
        td = s["td_c"] if "td_c" in s else None
        zm, tm, tdm = remuestrear_sondeo(s["altura_m"], s["temp_c"], td, paso, z_max)
        ok = np.isfinite(tm)
        if ok.sum() < 10:
            continue
        perfiles.append(pd.DataFrame({"fecha": fecha, "altura_m": zm[ok], "temp_c": tm[ok],
                                      "td_c": tdm[ok] if tdm is not None else np.nan}))
        inversiones.append({"fecha": fecha, **inversion_de_sondeo(zm[ok], tm[ok], **kwargs_inversion)})
    perfiles = (pd.concat(perfiles, ignore_index=True) if perfiles
                else pd.DataFrame(columns=["fecha", "altura_m", "temp_c", "td_c"]))
    return perfiles, pd.DataFrame(inversiones, columns=["fecha", "z_base", "z_cima", "intensidad"])


def estimar_con_forma_del_sondeo(z_obj: float, z_est, v_est, z_sondeo, v_sondeo, z_ancla: float,
                                 dist_km=None, residuos: bool = False, potencia: float = 2.0) -> float:
    """Recta de las estaciones por debajo de ``z_ancla`` y, por encima, la forma del sondeo.

    - Por debajo del ancla (la base de la inversión): recta ajustada con las estaciones que
      están por debajo, que es donde hay muchas estaciones y el método funciona bien.
    - Por encima: el valor de la recta en el ancla más lo que cambia el sondeo desde el ancla
      hasta la altura del punto. Así la parte alta del perfil (inversión, aire seco por encima)
      la da una medida real en lugar de una o dos estaciones.

    Con ``residuos=True`` se suma la interpolación, por distancia inversa, de lo que cada
    estación se separa de este perfil.
    """
    z_est = np.asarray(z_est, dtype=float)
    v_est = np.asarray(v_est, dtype=float)
    z_sondeo = np.asarray(z_sondeo, dtype=float)
    v_sondeo = np.asarray(v_sondeo, dtype=float)
    ok_s = np.isfinite(z_sondeo) & np.isfinite(v_sondeo)
    z_sondeo, v_sondeo = z_sondeo[ok_s], v_sondeo[ok_s]

    abajo = z_est < z_ancla
    if abajo.sum() >= 2 and np.ptp(z_est[abajo]) >= 100:
        b, a = np.polyfit(z_est[abajo], v_est[abajo], 1)
    elif abajo.sum() >= 1:
        b, a = GRADIENTE_ESTANDAR, float(np.mean(v_est[abajo] - GRADIENTE_ESTANDAR * z_est[abajo]))
    else:  # ninguna estación por debajo del ancla: se ancla en la estación más baja
        k = int(np.argmin(z_est))
        z_ancla = float(z_est[k])
        b, a = 0.0, float(v_est[k])

    def perfil(zz):
        zz = np.asarray(zz, dtype=float)
        recta = a + b * np.minimum(zz, z_ancla)
        if len(z_sondeo) < 2:
            return recta + b * np.maximum(zz - z_ancla, 0.0)
        cambio = np.interp(zz, z_sondeo, v_sondeo) - np.interp(z_ancla, z_sondeo, v_sondeo)
        return np.where(zz > z_ancla, recta + cambio, recta)

    base = float(perfil(z_obj))
    if residuos and dist_km is not None:
        dist_km = np.asarray(dist_km, dtype=float)
        res = v_est - perfil(z_est)
        pesos = 1.0 / np.maximum(dist_km, 0.1) ** potencia
        base += float((pesos * res).sum() / pesos.sum())
    return base


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

    if metodo in METODOS_SONDEO:
        if perfil is None:
            raise ValueError("los métodos con radiosondeo necesitan el perfil ya ajustado")
        metodo = "perfil_dia" if metodo == "perfil_sondeo" else "perfil_residuos"
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
                                 paso_dias: int = 1, inversiones: pd.Series | None = None,
                                 perfiles_sondeo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Esconde cada estación, cada día, y la estima con las demás estaciones de su isla.

    ``df`` necesita: ``indicativo``, ``fecha``, ``isla``, ``altitud``, ``lat``, ``lon``,
    ``tmax``, ``tmed``, ``hrMedia``, ``velmedia``. ``paso_dias`` > 1 usa solo uno de cada
    ``paso_dias`` días para acelerar.

    ``inversiones``: serie fecha -> altura de la base de la inversión (m) del radiosondeo. Si se
    da, se evalúan también los métodos ``perfil_sondeo`` y ``perfil_sondeo_residuos``.

    ``perfiles_sondeo``: perfiles diarios del radiosondeo (``fecha``, ``altura_m``, ``temp_c``,
    ``td_c``, salida de ``perfiles_de_sondeo``). Si se da, se evalúan también ``forma_sondeo``
    y ``forma_sondeo_residuos`` (temperatura y humedad; el viento del globo es de atmósfera libre
    y no representa el viento en superficie, así que no se usa).

    Devuelve una fila por (estación, día, variable, método) con el valor real y el estimado.
    """
    metodos = METODOS_BASE + (METODOS_SONDEO if inversiones is not None else ()) \
        + (METODOS_FORMA if perfiles_sondeo is not None else ())
    sondeos = {}
    if perfiles_sondeo is not None:
        for f, g in perfiles_sondeo.groupby(pd.to_datetime(perfiles_sondeo["fecha"])):
            g = g.sort_values("altura_m")
            sondeos[f] = (g["altura_m"].to_numpy(float), g["temp_c"].to_numpy(float),
                          g["td_c"].to_numpy(float))
    if inversiones is not None:
        inversiones = inversiones.copy()
        inversiones.index = pd.to_datetime(inversiones.index)
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
        z_inv = float(inversiones.get(fecha, np.nan)) if inversiones is not None else np.nan
        sondeo = sondeos.get(fecha)
        z_ancla = z_inv if np.isfinite(z_inv) else Z_ANCLA_SIN_INVERSION

        for k in range(len(g)):
            otras = np.arange(len(g)) != k
            base = {"indicativo": ind[k], "fecha": fecha, "isla": isla, "altitud": z[k]}

            for var, grad in (("tmax", GRADIENTE_ESTANDAR), ("velmedia", 0.0)):
                ok = otras & ~np.isnan(cols[var])
                if np.isnan(cols[var][k]) or ok.sum() < min_estaciones:
                    continue
                perfil = ajustar_perfil(z[ok], cols[var][ok])  # una sola vez para los dos métodos de perfil
                perfil_s = (ajustar_perfil_sondeo(z[ok], cols[var][ok], z_inv)
                            if inversiones is not None else None)
                for m in metodos:
                    if var == "velmedia" and m in ("gradiente_fijo",) + METODOS_FORMA:
                        continue  # el viento no tiene gradiente estándar ni se toma del globo
                    if m in METODOS_FORMA:
                        if sondeo is None:
                            continue
                        est = estimar_con_forma_del_sondeo(
                            z[k], z[ok], cols[var][ok], sondeo[0], sondeo[1], z_ancla,
                            dist_km=d[k, ok], residuos=(m == "forma_sondeo_residuos"))
                    else:
                        est = estimar(m, z[k], d[k, ok], z[ok], cols[var][ok], gradiente=grad,
                                      perfil=perfil_s if m in METODOS_SONDEO else perfil)
                    filas.append({**base, "variable": var, "metodo": m,
                                  "real": cols[var][k], "estimado": est})

            ok = otras & ~np.isnan(cols["tmed"]) & ~np.isnan(cols["td"])
            if not np.isnan(cols["hrMedia"][k]) and ok.sum() >= min_estaciones:
                perfiles = {"tmed": ajustar_perfil(z[ok], cols["tmed"][ok]),
                            "td": ajustar_perfil(z[ok], cols["td"][ok])}
                if inversiones is not None:
                    perfiles_s = {"tmed": ajustar_perfil_sondeo(z[ok], cols["tmed"][ok], z_inv),
                                  "td": ajustar_perfil_sondeo(z[ok], cols["td"][ok], z_inv)}
                for m in metodos:
                    if m in METODOS_FORMA:
                        if sondeo is None or not np.isfinite(sondeo[2]).any():
                            continue
                        con_res = m == "forma_sondeo_residuos"
                        t = estimar_con_forma_del_sondeo(z[k], z[ok], cols["tmed"][ok], sondeo[0],
                                                         sondeo[1], z_ancla, d[k, ok], con_res)
                        td = estimar_con_forma_del_sondeo(z[k], z[ok], cols["td"][ok], sondeo[0],
                                                          sondeo[2], z_ancla, d[k, ok], con_res)
                        est = float(humedad_relativa(t, min(td, t)))
                    else:
                        est = estimar_humedad(m, z[k], d[k, ok], z[ok], cols["tmed"][ok],
                                              cols["td"][ok], cols["hrMedia"][ok],
                                              perfiles=perfiles_s if m in METODOS_SONDEO else perfiles)
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
