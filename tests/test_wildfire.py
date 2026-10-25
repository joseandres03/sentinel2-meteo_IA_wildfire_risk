"""Tests del paquete ``wildfire`` (no requieren Earth Engine ni conexión)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wildfire import indices, etiquetas, validacion, meteo  # noqa: E402


# ---------- índices espectrales ----------

def _parche(b2=500, b3=600, b4=400, b8=3000, b11=1500, b12=800, tam=4):
    valores = np.array([b2, b3, b4, b8, b11, b12], dtype=np.float32)
    return np.broadcast_to(valores, (tam, tam, 6)).copy()


def test_ndvi_vegetacion_densa():
    p = _parche(b4=400, b8=3000)
    assert np.allclose(indices.ndvi(p), (3000 - 400) / 3400, atol=1e-6)


def test_mascara_excluye_agua_y_suelo():
    agua = _parche(b3=1500, b4=300, b8=200)      # NDWI alto
    suelo = _parche(b4=2000, b8=2100)           # NDVI ≈ 0.02
    veg = _parche()
    assert not indices.mascara_vegetacion(agua).any()
    assert not indices.mascara_vegetacion(suelo).any()
    assert indices.mascara_vegetacion(veg).all()


def test_riesgo_combustible_v1_sin_vegetacion_es_cero():
    assert indices.riesgo_combustible_v1(_parche(b4=2000, b8=2100)) == 0.0


def test_riesgo_combustible_v1_mas_seco_mas_riesgo():
    humedo = _parche(b8=3000, b11=1000)
    seco = _parche(b8=3000, b11=2800)
    assert indices.riesgo_combustible_v1(seco) > indices.riesgo_combustible_v1(humedo)


def test_indice_v1_reproduce_formula():
    rm = indices.riesgo_meteorologico_v1(40, 20, 50)   # todo al máximo
    assert np.isclose(rm, 1.0)
    assert np.isclose(indices.indice_v1(1.0, 1.0), 1.0)
    assert np.isclose(indices.indice_v1(0.0, 1.0), 0.65)


def test_resumen_espectral_campos():
    r = indices.resumen_espectral(_parche())
    assert r["frac_vegetacion"] == 1.0
    assert {"ndvi_media", "ndmi_media", "nbr_media", "ndmi_p10"} <= set(r)


# ---------- etiquetas ----------

def test_haversine_un_grado_latitud():
    assert abs(etiquetas.distancia_haversine_km(28, -16, 29, -16) - 111.2) < 0.5


def test_agrupar_eventos_une_focos_cercanos_y_separa_lejanos():
    focos = pd.DataFrame({
        "lat": [28.10, 28.105, 28.11, 28.50],
        "lon": [-17.20, -17.205, -17.21, -16.50],
        "fecha": ["2023-08-01", "2023-08-02", "2023-08-03", "2023-08-01"],
    })
    ev = etiquetas.agrupar_eventos(focos, radio_km=2, ventana_dias=3)
    assert len(ev) == 2
    grande = ev.loc[ev["n_focos"].idxmax()]
    assert grande["n_focos"] == 3
    assert grande["fecha"] == pd.Timestamp("2023-08-01")
    assert grande["fecha_fin"] == pd.Timestamp("2023-08-03")


def test_agrupar_eventos_separa_mismo_sitio_en_fechas_lejanas():
    focos = pd.DataFrame({"lat": [28.1, 28.1], "lon": [-17.2, -17.2], "fecha": ["2020-07-01", "2020-09-01"]})
    assert len(etiquetas.agrupar_eventos(focos)) == 2


def test_filtrar_exclusiones_descarta_erupcion_la_palma():
    focos = pd.DataFrame({
        "lat": [28.61, 28.61, 28.10],
        "lon": [-17.88, -17.88, -17.20],
        "fecha": ["2021-10-15", "2023-08-01", "2021-10-15"],
    })
    filtrado = etiquetas.filtrar_exclusiones(focos)
    assert len(filtrado) == 2
    assert not ((filtrado["fecha"] == "2021-10-15") & (filtrado["lat"] == 28.61)).any()


def test_muestrear_negativos_respeta_exclusion():
    rng = np.random.default_rng(0)
    puntos = pd.DataFrame({"lat": rng.uniform(28.0, 28.2, 500), "lon": rng.uniform(-17.3, -17.1, 500)})
    eventos = pd.DataFrame({"lat": [28.1], "lon": [-17.2], "fecha": [pd.Timestamp("2023-08-01")]})
    neg = etiquetas.muestrear_negativos(puntos, eventos, n=300, fecha_inicio="2023-07-01",
                                        fecha_fin="2023-08-31", exclusion_km=5, exclusion_dias=10)
    assert len(neg) == 300
    d_km = etiquetas.distancia_haversine_km(neg["lat"], neg["lon"], 28.1, -17.2)
    d_dias = (neg["fecha"] - pd.Timestamp("2023-08-01")).abs().dt.days
    assert not ((d_km <= 5) & (d_dias <= 10)).any()


def test_muestrear_negativos_restringe_meses():
    puntos = pd.DataFrame({"lat": [28.1, 28.2], "lon": [-17.2, -17.1]})
    vacio = pd.DataFrame(columns=["lat", "lon", "fecha"])
    neg = etiquetas.muestrear_negativos(puntos, vacio, n=50, fecha_inicio="2022-01-01",
                                        fecha_fin="2022-12-31", meses=[7, 8, 9])
    assert neg["fecha"].dt.month.isin([7, 8, 9]).all()


def test_construir_dataset_etiquetas():
    ev = pd.DataFrame({"lat": [28.1], "lon": [-17.2], "fecha": ["2023-08-01"]})
    neg = pd.DataFrame({"lat": [28.2, 28.3], "lon": [-17.1, -17.0], "fecha": ["2023-07-01", "2023-09-01"]})
    df = etiquetas.construir_dataset(ev, neg)
    assert df["incendio"].tolist() == [0, 1, 0]


# ---------- validación ----------

def test_isla_de():
    islas = validacion.isla_de([28.12, 28.30, 30.0], [-17.24, -16.50, -15.0])
    assert list(islas) == ["La Gomera", "Tenerife", "Fuera"]


def test_bloques_espaciales_agrupa_vecinos():
    b = validacion.bloques_espaciales([28.100, 28.101, 28.60], [-17.200, -17.201, -16.40], tam_km=10)
    assert b[0] == b[1] and b[0] != b[2]


def test_particion_temporal():
    ent, prue = validacion.particion_temporal(["2022-05-01", "2024-01-01", "2025-07-01"], "2024-01-01")
    assert ent.tolist() == [True, False, False] and prue.tolist() == [False, True, True]


def test_captura_en_top():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    assert validacion.captura_en_top(y, p, 0.1) == 0.5


def test_metricas_modelo_perfecto():
    y = np.array([0, 0, 1, 1])
    m = validacion.metricas_clasificacion(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["sensibilidad"] == 1.0 and m["falsos_positivos"] == 0


def test_tabla_fiabilidad_bien_calibrada():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    t = validacion.tabla_fiabilidad(y, p, n_bins=5)
    assert np.allclose(t["prob_media"], t["frecuencia_observada"], atol=0.03)


def test_corregir_prevalencia():
    # con la misma prevalencia no cambia nada
    assert np.isclose(validacion.corregir_prevalencia(0.3, 0.1, 0.1), 0.3)
    # si los incendios reales son más raros que en la muestra, la probabilidad baja
    assert validacion.corregir_prevalencia(0.5, 0.1, 0.01) < 0.5


def test_prediccion_fuera_de_bloque_no_ve_su_grupo():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    grupos = np.repeat(np.arange(10), 20)
    oof = validacion.prediccion_fuera_de_bloque(LogisticRegression(), X, y, grupos, n_splits=5)
    assert not np.isnan(oof).any()
    assert validacion.metricas_clasificacion(y, oof)["roc_auc"] > 0.8


# ---------- meteorología ----------

def test_humedad_relativa_saturacion():
    assert np.isclose(meteo.humedad_relativa(20, 20), 100.0)
    assert 27 < meteo.humedad_relativa(30, 10) < 31  # ≈ 28,9 %


# ---------- meteorología y relieve ----------

from wildfire import meteo_relieve as mr  # noqa: E402


def _perfil_con_inversion(z, z_inv=1200.0):
    # baja 6,5 ºC/km hasta la inversión y sube 3 ºC/km por encima
    z = np.asarray(z, dtype=float)
    return 25.0 - 0.0065 * z + 0.0095 * np.maximum(z - z_inv, 0.0)


def test_punto_rocio_inverso_de_humedad():
    td = mr.punto_rocio(25.0, 60.0)
    assert np.isclose(meteo.humedad_relativa(25.0, td), 60.0, atol=1e-6)


def test_ajustar_perfil_detecta_inversion():
    z = np.array([10, 200, 400, 600, 900, 1100, 1500, 1700, 2100, 2300])
    p = mr.ajustar_perfil(z, _perfil_con_inversion(z))
    assert p["tipo"] == "codo" and abs(p["z_codo"] - 1200) <= 100


def test_ajustar_perfil_sin_inversion_es_recta():
    rng = np.random.default_rng(0)
    z = np.linspace(0, 2000, 12)
    p = mr.ajustar_perfil(z, 25 - 0.0065 * z + rng.normal(0, 0.2, 12))
    assert p["tipo"] == "recta" and abs(p["coef"][1] + 0.0065) < 0.001


def test_ajustar_perfil_pocas_estaciones_es_constante():
    assert mr.ajustar_perfil([10, 50], [20.0, 21.0])["tipo"] == "constante"


def test_perfil_estima_mejor_que_cercana_en_altura():
    z = np.array([10, 300, 600, 850, 2150, 2370])
    v = _perfil_con_inversion(z)
    dist = np.array([5, 4, 3, 2, 8, 9])      # la más cercana está a 850 m
    real = float(_perfil_con_inversion(1800))
    err_cercana = abs(mr.estimar("cercana", 1800, dist, z, v) - real)
    err_perfil = abs(mr.estimar("perfil_dia", 1800, dist, z, v) - real)
    assert err_perfil < err_cercana


def test_gradiente_fijo():
    est = mr.estimar("gradiente_fijo", 1000, [1.0, 2.0], [0.0, 500.0], [20.0, 18.0])
    assert np.isclose(est, 20.0 - 6.5)


def test_validacion_dejando_una_fuera():
    rng = np.random.default_rng(3)
    filas = []
    for fecha in pd.date_range("2023-07-01", periods=5):
        for k, z in enumerate([10, 300, 600, 900, 1500, 2100]):
            t = _perfil_con_inversion(z) + rng.normal(0, 0.3)
            filas.append({"indicativo": f"E{k}", "fecha": fecha, "isla": "Tenerife", "altitud": z,
                          "lat": 28.3 + 0.01 * k, "lon": -16.5, "tmax": t + 5, "tmed": t,
                          "hrMedia": 60.0, "velmedia": 3.0})
    res = mr.validacion_dejando_una_fuera(pd.DataFrame(filas))
    assert set(res["metodo"]) == set(mr.METODOS_BASE)
    tabla = mr.resumen_errores(res)
    mae = tabla[(tabla.variable == "tmax") & (tabla.franja == "todas")].set_index("metodo")["MAE"]
    assert mae["perfil_residuos"] < mae["cercana"]


# ---------- radiosondeo e inversión ----------

def _cabecera_igra2(anio, mes, dia, hora, n):
    # columnas: '#' 1, ID 2-12, año 14-17, mes 19-20, día 22-23, hora 25-26, ..., NUMLEV 33-36
    return f"#SPM00060018 {anio:4d} {mes:02d} {dia:02d} {hora:02d} 1100 {n:4d} ncdc6310 ncdc6310  283000 -163830"


def _nivel_igra2(gph, temp_decimas, hr_decimas):
    # LVLTYP 1-2, ETIME 4-8, PRESS 10-15, PFLAG 16, GPH 17-21, ZFLAG 22, TEMP 23-27, TFLAG 28, RH 29-33
    return f"21 {-9999:5d} {-9999:6d} {gph:5d}B{temp_decimas:5d}B{hr_decimas:5d} {-9999:5d} {-9999:5d} {-9999:5d}"


def _sondeo_igra2(anio, mes, dia, hora, niveles):
    return [_cabecera_igra2(anio, mes, dia, hora, len(niveles))] + [_nivel_igra2(*n) for n in niveles]


def test_formato_lineas_igra2_columnas():
    linea = _nivel_igra2(1234, -56, 789)
    assert linea[16:21] == " 1234" and linea[22:27] == "  -56" and linea[28:33] == "  789"
    cab = _cabecera_igra2(2023, 7, 15, 12, 5)
    assert cab[13:17] == "2023" and cab[18:20] == "07" and cab[21:23] == "15" and cab[24:26] == "12"


def test_leer_igra2_y_detectar_inversion():
    # sondeo de mediodía: baja hasta 1.100 m, sube 4 ºC hasta 1.400 m y vuelve a bajar
    niveles = [(105, 240, 700), (500, 215, 750), (1100, 176, 850), (1250, 196, 300),
               (1400, 216, 150), (2000, 180, 100), (2500, -9999, 100), (3000, 110, 90)]
    lineas = (_sondeo_igra2(2023, 7, 15, 0, [(105, 200, 800), (1000, 150, 800)])
              + _sondeo_igra2(2023, 7, 15, 12, niveles)
              + _sondeo_igra2(2016, 7, 15, 12, niveles))
    df = mr.leer_igra2(lineas, anio_min=2017)
    assert set(df["fecha"].dt.year) == {2023}
    assert len(df[df["hora"] == 12]) == 7          # el nivel con temperatura ausente se descarta
    assert np.isclose(df["temp_c"].iloc[0], 20.0)
    inv = mr.inversiones_diarias(df)
    assert len(inv) == 1
    fila = inv.iloc[0]
    assert fila["z_base"] == 1100 and fila["z_cima"] == 1400 and np.isclose(fila["intensidad"], 4.0)


def test_sondeo_sin_inversion():
    inv = mr.inversion_de_sondeo([100, 500, 1000, 2000], [25, 22, 18, 12])
    assert np.isnan(inv["z_base"])


def test_inversion_debil_no_cuenta():
    inv = mr.inversion_de_sondeo([300, 600, 900, 1500], [20, 18, 18.2, 14])
    assert np.isnan(inv["z_base"])


def test_perfil_sondeo_isla_bajo_la_inversion_es_recta():
    z = np.array([10, 300, 600, 900, 1400])          # p. ej. La Gomera, inversión a 1.600 m
    p = mr.ajustar_perfil_sondeo(z, 25 - 0.0065 * z, 1600)
    assert p["tipo"] == "recta" and np.isclose(p["coef"][1], -0.0065)


def test_perfil_sondeo_situa_el_codo_en_el_hueco_de_tenerife():
    # estaciones como Tenerife: nada entre 870 y 2.150 m; inversión real a 1.200 m
    z = np.array([20, 150, 300, 450, 630, 870, 2150, 2370])
    v = _perfil_con_inversion(z, 1200)
    p = mr.ajustar_perfil_sondeo(z, v, 1200)
    assert p["tipo"] == "codo" and p["z_codo"] == 1200
    assert abs(mr.evaluar_perfil(p, 1600) - _perfil_con_inversion(1600, 1200)) < 1e-6


def test_validacion_con_radiosondeo():
    rng = np.random.default_rng(5)
    alturas = [20, 150, 300, 450, 630, 870, 2150, 2370]
    filas, inv = [], {}
    for j, fecha in enumerate(pd.date_range("2023-07-01", periods=6)):
        z_inv = 1000 + 100 * j
        inv[fecha] = z_inv
        for k, z in enumerate(alturas):
            t = _perfil_con_inversion(z, z_inv) + rng.normal(0, 0.2)
            filas.append({"indicativo": f"E{k}", "fecha": fecha, "isla": "Tenerife", "altitud": z,
                          "lat": 28.3 + 0.01 * k, "lon": -16.5, "tmax": t + 5, "tmed": t,
                          "hrMedia": 60.0, "velmedia": 3.0})
    res = mr.validacion_dejando_una_fuera(pd.DataFrame(filas), inversiones=pd.Series(inv))
    assert set(res["metodo"]) == set(mr.METODOS)
    tabla = mr.resumen_errores(res)
    mae = tabla[(tabla.variable == "tmax") & (tabla.franja == "todas")].set_index("metodo")["MAE"]
    assert mae["perfil_sondeo"] < mae["cercana"]
