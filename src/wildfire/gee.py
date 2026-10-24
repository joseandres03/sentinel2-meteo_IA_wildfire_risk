"""Extracción de datos en Google Earth Engine para el dataset v2.

Requiere ``earthengine-api`` y una sesión inicializada (``inicializar``). Pensado para
ejecutarse desde Google Colab (``notebooks/Cuaderno4_dataset_incendios_reales.ipynb``).

Fuentes:
    - Focos térmicos: FIRMS (MODIS, 1 km, desde 2000)          -> ``FIRMS``
    - Sentinel-2 L2A armonizado con máscara de nubes SCL         -> ``COPERNICUS/S2_SR_HARMONIZED``
    - Meteorología diaria ERA5-Land (coherente en todo el periodo) -> ``ECMWF/ERA5_LAND/DAILY_AGGR``
    - Relieve SRTM 30 m                                           -> ``USGS/SRTMGL1_003``
    - Cobertura del suelo ESA WorldCover 10 m                     -> ``ESA/WorldCover/v200``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import BANDAS_S2, BBOX_ISLAS

try:
    import ee
except ImportError:  # el resto del paquete funciona sin Earth Engine
    ee = None

S2 = "COPERNICUS/S2_SR_HARMONIZED"
FIRMS = "FIRMS"
ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
SRTM = "USGS/SRTMGL1_003"
WORLDCOVER = "ESA/WorldCover/v200"

# Clases SCL que se descartan: sin datos, defectuoso, sombra de nube, nube media/alta, cirro
SCL_DESCARTAR = (0, 1, 3, 8, 9, 10)
# Clases WorldCover consideradas combustible: árboles, matorral, pastizal
WORLDCOVER_VEGETACION = (10, 20, 30)
BANDAS_ERA5 = (
    "temperature_2m_max", "temperature_2m", "dewpoint_temperature_2m",
    "u_component_of_wind_10m", "v_component_of_wind_10m",
)


def _requiere_ee():
    if ee is None:
        raise ImportError("instala earthengine-api para usar wildfire.gee")


def inicializar(proyecto: str) -> None:
    """Autentica (si hace falta) e inicializa Earth Engine con el proyecto de Google Cloud."""
    _requiere_ee()
    try:
        ee.Initialize(project=proyecto)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=proyecto)


def region_canarias():
    """MultiPolígono con las cajas envolventes de las siete islas."""
    _requiere_ee()
    poligonos = [
        [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
        for (x0, y0, x1, y1) in BBOX_ISLAS.values()
    ]
    return ee.Geometry.MultiPolygon(poligonos, None, False)


def enmascarar_nubes(imagen):
    """Aplica la máscara de nubes de la capa SCL de Sentinel-2."""
    scl = imagen.select("SCL")
    despejado = ee.Image.constant(1)
    for clase in SCL_DESCARTAR:
        despejado = despejado.And(scl.neq(clase))
    return imagen.updateMask(despejado)


def composicion_previa(geometria, fecha, dias: int = 30):
    """Mediana sin nubes de Sentinel-2 en los ``dias`` anteriores a ``fecha`` (excluida).

    Se usa solo información **anterior** al día del incendio para que la imagen no muestre
    ya la zona quemada (eso sería una fuga de información).
    """
    fin = ee.Date(fecha)
    inicio = fin.advance(-dias, "day")
    coleccion = (ee.ImageCollection(S2).filterBounds(geometria).filterDate(inicio, fin)
                 .map(enmascarar_nubes).select(list(BANDAS_S2)))
    # Si no hay ninguna escena, se devuelve una imagen enmascarada con las mismas bandas
    # (sus variables saldrán vacías) en lugar de una imagen sin bandas que rompería el cálculo.
    vacia = ee.Image.constant([0] * len(BANDAS_S2)).rename(list(BANDAS_S2)).updateMask(0)
    return ee.Image(ee.Algorithms.If(coleccion.size().gt(0), coleccion.median(), vacia))


def mascara_vegetacion():
    """Imagen binaria (1 = árbol, matorral o pastizal según ESA WorldCover)."""
    wc = ee.ImageCollection(WORLDCOVER).first().select("Map")
    veg = ee.Image.constant(0)
    for clase in WORLDCOVER_VEGETACION:
        veg = veg.Or(wc.eq(clase))
    return veg.rename("vegetacion")


def _con_coordenadas(f):
    coords = f.geometry().coordinates()
    return f.set({"lon": coords.get(0), "lat": coords.get(1)})


def focos_firms(inicio: str, fin: str, confianza_min: int = 50, region=None):
    """Focos térmicos FIRMS (centroides de píxeles de 1 km) con su fecha y coordenadas."""
    _requiere_ee()
    region = region or region_canarias()
    coleccion = ee.ImageCollection(FIRMS).filterDate(inicio, fin).filterBounds(region)

    def a_puntos(img):
        fecha = img.date().format("YYYY-MM-dd")
        confianza = img.select("confidence")
        mascara = confianza.gte(confianza_min).selfMask().toInt().rename("foco")
        vectores = mascara.reduceToVectors(
            geometry=region, scale=1000, geometryType="centroid",
            eightConnected=True, maxPixels=1e10,
        )
        return vectores.map(lambda f: _con_coordenadas(f).set("fecha", fecha))

    return coleccion.map(a_puntos).flatten()


def mascara_tierra():
    """Imagen binaria (1 = tierra firme, cualquier cobertura salvo agua según ESA WorldCover)."""
    wc = ee.ImageCollection(WORLDCOVER).first().select("Map")
    return wc.neq(80).rename("tierra")


def puntos_tierra(n: int, semilla: int = 42, escala: int = 100, region=None):
    """Muestra aleatoria de ``n`` puntos sobre tierra firme, con cualquier cobertura.

    Se usa para generar los negativos con el **mismo criterio espacial que los incendios**:
    los focos FIRMS pueden caer en cualquier cobertura (también cerca de pueblos o cultivos),
    así que los negativos también. Después se aplica a ambas clases el mismo filtro de
    vegetación sobre su entorno. Si los negativos se tomaran solo de píxeles con vegetación,
    el modelo aprendería a distinguir "zona con o sin vegetación/urbana" en vez de riesgo.
    """
    _requiere_ee()
    region = region or region_canarias()
    muestra = mascara_tierra().selfMask().sample(
        region=region, scale=escala, numPixels=n, seed=semilla, geometries=True,
    )
    return muestra.map(_con_coordenadas)


def puntos_vegetacion(n: int, semilla: int = 42, escala: int = 100, region=None):
    """Muestra aleatoria de ``n`` puntos sobre vegetación.

    No usar para los negativos del dataset: introduce un sesgo frente a los focos FIRMS
    (ver ``puntos_tierra``).
    """
    _requiere_ee()
    region = region or region_canarias()
    muestra = mascara_vegetacion().selfMask().sample(
        region=region, scale=escala, numPixels=n, seed=semilla, geometries=True,
    )
    return muestra.map(_con_coordenadas)


def fc_a_dataframe(fc, columnas) -> pd.DataFrame:
    """Descarga una FeatureCollection pequeña (< 5000 elementos) a un DataFrame."""
    datos = fc.reduceColumns(ee.Reducer.toList(len(columnas)), list(columnas)).get("list").getInfo()
    return pd.DataFrame(datos, columns=list(columnas))


def _caracteristicas_punto(f, dias_s2: int = 30, radio_m: int = 320):
    """Variables tabulares de un punto/fecha: Sentinel-2, relieve, cobertura y ERA5-Land."""
    punto = f.geometry()
    entorno = punto.buffer(radio_m)
    fecha = ee.Date(f.get("fecha"))

    # Sentinel-2 previo al día objetivo
    comp = composicion_previa(entorno, fecha, dias_s2)
    indices = ee.Image.cat([
        comp.normalizedDifference(["B8", "B4"]).rename("ndvi"),
        comp.normalizedDifference(["B8", "B11"]).rename("ndmi"),
        comp.normalizedDifference(["B8", "B12"]).rename("nbr"),
        comp.normalizedDifference(["B3", "B8"]).rename("ndwi"),
    ])
    s2 = comp.addBands(indices).reduceRegion(ee.Reducer.mean(), entorno, 20, maxPixels=1e8)

    # Relieve
    dem = ee.Image(SRTM).select("elevation")
    terreno = ee.Terrain.products(dem).select(["elevation", "slope", "aspect"])
    orientacion_norte = terreno.select("aspect").multiply(np.pi / 180).cos().rename("orientacion_norte")
    relieve = terreno.select(["elevation", "slope"]).addBands(orientacion_norte).reduceRegion(
        ee.Reducer.mean(), entorno, 30, maxPixels=1e8)

    # Cobertura del suelo (fracción de cada clase en el entorno)
    wc = ee.ImageCollection(WORLDCOVER).first().select("Map")
    cobertura = ee.Image.cat([
        wc.eq(10).rename("frac_arbolado"), wc.eq(20).rename("frac_matorral"),
        wc.eq(30).rename("frac_pastizal"), wc.eq(40).rename("frac_cultivo"),
        wc.eq(50).rename("frac_urbano"),
    ]).reduceRegion(ee.Reducer.mean(), entorno, 10, maxPixels=1e8)

    # Meteorología ERA5-Land del día objetivo y precipitación de los 30 días previos.
    # Se promedia en 15 km porque las celdas de ERA5-Land (~11 km) pueden no cubrir la costa.
    era = ee.ImageCollection(ERA5)
    entorno_meteo = punto.buffer(15000)
    dia_col = era.filterDate(fecha, fecha.advance(1, "day")).select(list(BANDAS_ERA5))
    dia_vacio = ee.Image.constant([0] * len(BANDAS_ERA5)).rename(list(BANDAS_ERA5)).updateMask(0)
    dia = ee.Image(ee.Algorithms.If(dia_col.size().gt(0), dia_col.mean(), dia_vacio))
    meteo = dia.reduceRegion(ee.Reducer.mean(), entorno_meteo, 11132, maxPixels=1e8)
    lluvia_col = era.filterDate(fecha.advance(-30, "day"), fecha).select("total_precipitation_sum")
    lluvia = ee.Image(ee.Algorithms.If(lluvia_col.size().gt(0), lluvia_col.sum(),
                                       ee.Image.constant(0).updateMask(0)))
    precip = ee.Image(lluvia).rename("precip_30d").reduceRegion(
        ee.Reducer.mean(), entorno_meteo, 11132, maxPixels=1e8)

    n_escenas = (ee.ImageCollection(S2).filterBounds(entorno)
                 .filterDate(fecha.advance(-dias_s2, "day"), fecha).size())

    return (f.set(s2).set(relieve).set(cobertura).set(meteo).set(precip)
            .set("n_escenas_s2", n_escenas))


def dataframe_a_fc(df: pd.DataFrame):
    """Convierte un DataFrame ``id, lat, lon, fecha, incendio`` en FeatureCollection de puntos."""
    _requiere_ee()
    features = [
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                   {"id": int(r.id), "fecha": pd.Timestamp(r.fecha).strftime("%Y-%m-%d"),
                    "incendio": int(r.incendio)})
        for r in df.itertuples(index=False)
    ]
    return ee.FeatureCollection(features)


def exportar_caracteristicas(df: pd.DataFrame, carpeta_drive: str, prefijo: str = "caracteristicas",
                             tam_lote: int = 2000, dias_s2: int = 30, radio_m: int = 500) -> list:
    """Lanza tareas de exportación a Google Drive (CSV) con las variables tabulares por lotes.

    ``radio_m`` es el radio del entorno de cada punto. Por defecto 500 m, del orden de la
    incertidumbre de posición de un foco MODIS (píxel de 1 km).

    El cálculo se hace en los servidores de Earth Engine; se sigue en
    https://code.earthengine.google.com/tasks o con ``[t.status() for t in tareas]``.
    """
    _requiere_ee()
    tareas = []
    for i, inicio in enumerate(range(0, len(df), tam_lote)):
        lote = df.iloc[inicio:inicio + tam_lote]
        fc = dataframe_a_fc(lote).map(lambda f: _caracteristicas_punto(f, dias_s2, radio_m))
        tarea = ee.batch.Export.table.toDrive(
            collection=fc, description=f"{prefijo}_{i:03d}", folder=carpeta_drive,
            fileNamePrefix=f"{prefijo}_{i:03d}", fileFormat="CSV",
        )
        tarea.start()
        tareas.append(tarea)
    return tareas


def extraer_parche(lat: float, lon: float, fecha: str, dias: int = 30, tam: int = 64,
                   metros_pixel: int = 10) -> np.ndarray | None:
    """Parche (tam, tam, 6) de la mediana sin nubes de Sentinel-2 previa a ``fecha`` (para la CNN)."""
    _requiere_ee()
    punto = ee.Geometry.Point([lon, lat])
    caja = punto.buffer(tam * metros_pixel / 2).bounds()
    comp = composicion_previa(caja, fecha, dias).unmask(0)
    proyeccion = ee.Projection("EPSG:32628").atScale(metros_pixel)
    comp = comp.reproject(proyeccion)
    try:
        props = comp.sampleRectangle(region=caja, defaultValue=0).getInfo().get("properties", {})
    except Exception:
        return None
    parche = np.zeros((tam, tam, len(BANDAS_S2)), dtype=np.float32)
    for k, banda in enumerate(BANDAS_S2):
        m = np.array(props.get(banda, []), dtype=np.float32)
        if m.ndim != 2 or m.size == 0:
            return None
        y0 = max(0, (m.shape[0] - tam) // 2)
        x0 = max(0, (m.shape[1] - tam) // 2)
        recorte = m[y0:y0 + tam, x0:x0 + tam]
        parche[:recorte.shape[0], :recorte.shape[1], k] = recorte
    return parche
