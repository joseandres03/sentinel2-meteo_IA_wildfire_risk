import os
import gc
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from PIL import Image
from scipy.interpolate import griddata

import ee
import geemap
import pyproj
import osmnx as ox
import geopandas as gpd
import rasterio
import rasterio.enums
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds

from tensorflow import keras
import joblib

# CONSTANTES GLOBALES Y RUTAS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, 'models', 'modelo_late_fusion_definitivo.keras')
RUTA_ESCALADOR = os.path.join(BASE_DIR, 'models', 'robust_scaler_meteo.pkl')

BBOX_CANARIAS = {
    "La Gomera": [-17.37, 28.01, -17.09, 28.23],
    "Tenerife": [-16.94, 27.97, -16.11, 28.59],
    "Gran Canaria": [-15.83, 27.70, -15.36, 28.18],
    "La Palma": [-18.00, 28.43, -17.72, 28.85],
    "El Hierro": [-18.17, 27.62, -17.88, 27.86],
    "Lanzarote": [-13.91, 28.83, -13.33, 29.26],
    "Fuerteventura": [-14.52, 28.01, -13.82, 28.76]
}

def seleccionar_isla() -> str:
    print("Modelo probabilístico de riesgo de incendio con Deep Learning")
    print("Islas disponibles:", ", ".join(BBOX_CANARIAS.keys()))
    isla = input("\nIntroduce la isla a analizar: ").strip()
    while isla not in BBOX_CANARIAS:
        isla = input("Isla no válida. Escribe el nombre exacto de la lista: ").strip()
    return isla

def descargar_satelite(isla: str, proyecto_gcp: str = "tfm-bbdd-499813") -> tuple:
    print(f"\nBuscando la última imagen de la constelación Sentinel-2 para {isla}...")
    ee.Initialize(project=proyecto_gcp)
    
    region = ee.Geometry.Rectangle(BBOX_CANARIAS[isla])
    hoy = datetime.now()
    hace_un_mes = hoy - timedelta(days=30)
    
    coleccion = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(hace_un_mes.strftime('%Y-%m-%d'), hoy.strftime('%Y-%m-%d'))
    )
    
    ultima_imagen = coleccion.sort('system:time_start', False).first()
    fecha_captura = ee.Date(ultima_imagen.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    print(f"-> última imagen obtenida el: {fecha_captura}")
    
    imagen_mosaico = coleccion.sort('system:time_start', True).mosaic()
    imagen_export = imagen_mosaico.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    
    ruta_salida = os.path.join(BASE_DIR, 'data', 'raw', f'satelite_{isla.replace(" ", "_")}.tif')
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    print("-> descargando GeoTIFF...")
    geemap.download_ee_image(
        image=imagen_export,
        filename=ruta_salida,
        region=region,
        scale=10,
        crs='EPSG:32628'
    )
    return ruta_salida, fecha_captura

def descargar_meteo_malla(isla: str, coords_utm: list) -> np.ndarray:
    print(f"\ndescargando datos meteorológicos del HARMONIE-AROME e interpolando geografía...")
    bbox = BBOX_CANARIAS[isla]
    
    lats = np.linspace(bbox[1], bbox[3], 5)
    lons = np.linspace(bbox[0], bbox[2], 5)
    malla_lons, malla_lats = np.meshgrid(lons, lats)
    
    url = "https://api.open-meteo.com/v1/forecast"
    parametros = {
        "latitude": ",".join(map(str, np.round(malla_lats.flatten(), 4))),
        "longitude": ",".join(map(str, np.round(malla_lons.flatten(), 4))),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "models": "best_match",
        "timezone": "Atlantic/Canary",
        "forecast_days": 1
    }
    
    max_reintentos = 3
    for intento in range(max_reintentos):
        try:
            respuesta_raw = requests.get(url, params=parametros, timeout=15)
            respuesta_raw.raise_for_status() 
            respuesta = respuesta_raw.json()
            if isinstance(respuesta, dict) and respuesta.get("error"):
                raise RuntimeError(f"la API de Open-Meteo rechazó la conexión: {respuesta.get('reason')}")
            break
        except requests.exceptions.RequestException as e:
            if intento < max_reintentos - 1:
                print(f"⚠️ aviso: micro-corte en Open-Meteo. reintentando en 5 segundos... (intento {intento+1}/{max_reintentos})")
                time.sleep(5)
            else:
                raise RuntimeError(f"fallo definitivo de Open-Meteo tras {max_reintentos} intentos: {e}")
    
    t_malla = [loc['hourly']['temperature_2m'][12] for loc in respuesta]
    hr_malla = [loc['hourly']['relative_humidity_2m'][12] for loc in respuesta]
    v_malla = [loc['hourly']['wind_speed_10m'][12] for loc in respuesta]
    puntos_origen = np.column_stack((malla_lons.flatten(), malla_lats.flatten()))
    
    transformador = pyproj.Transformer.from_crs("EPSG:32628", "EPSG:4326", always_xy=True)
    lons_parches, lats_parches = transformador.transform(
        [c[0] for c in coords_utm],
        [c[1] for c in coords_utm]
    )
    puntos_destino = np.column_stack((lons_parches, lats_parches))
    
    t_interp = griddata(puntos_origen, t_malla, puntos_destino, method='linear')
    hr_interp = griddata(puntos_origen, hr_malla, puntos_destino, method='linear')
    v_interp = griddata(puntos_origen, v_malla, puntos_destino, method='linear')
    
    if np.isnan(t_interp).any():
        mascara_nan = np.isnan(t_interp)
        t_interp[mascara_nan] = griddata(puntos_origen, t_malla, puntos_destino[mascara_nan], method='nearest')
        hr_interp[mascara_nan] = griddata(puntos_origen, hr_malla, puntos_destino[mascara_nan], method='nearest')
        v_interp[mascara_nan] = griddata(puntos_origen, v_malla, puntos_destino[mascara_nan], method='nearest')
        
    return np.column_stack((t_interp, hr_interp, v_interp))

def calcular_mascaras_fisicas(ruta: str) -> tuple:
    print("\nprocesando reflectancia y delimitando el litoral...")
    with rasterio.open(ruta) as src:
        raster_data = src.read()
        perfil_geo = src.profile
        
    imagen_bruta = np.transpose(raster_data, (1, 2, 0)).astype(np.float32)
    del raster_data
    gc.collect()
        
    b3_green = imagen_bruta[:, :, 1]
    b4_red   = imagen_bruta[:, :, 2]
    b8_nir   = imagen_bruta[:, :, 3]
    
    ndwi = (b3_green - b8_nir) / (b3_green + b8_nir + 1e-8)
    mascara_tierra = ndwi <= 0.3
    
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-8)
    mascara_vegetacion = ndvi > 0.1
    
    del b3_green, b4_red, b8_nir, ndwi, ndvi
    gc.collect()
    
    return imagen_bruta, mascara_tierra, mascara_vegetacion, perfil_geo

def extraer_parches_solapados(imagen_bruta: np.ndarray, mascara_vegetacion: np.ndarray, 
                              perfil: dict, tamano: int = 64, solape: int = 4) -> tuple:
    print(f"\ngenerando parches de inferencia ({tamano}x{tamano} px, {solape}px solape)...")
    filas_totales, cols_totales, _ = imagen_bruta.shape
    paso = tamano - solape
    parches, coordenadas, coords_utm = [], [], []
    transform = perfil['transform']
    
    for f in range(0, filas_totales - tamano + 1, paso):
        for c in range(0, cols_totales - tamano + 1, paso):
            if np.any(mascara_vegetacion[f:f+tamano, c:c+tamano]):
                parches.append(imagen_bruta[f:f+tamano, c:c+tamano, :])
                coordenadas.append((f, c))
                
                centro_f = f + (tamano / 2.0)
                centro_c = c + (tamano / 2.0)
                utm_x, utm_y = rasterio.transform.xy(transform, centro_f, centro_c)
                coords_utm.append((utm_x, utm_y))
                
    del imagen_bruta, mascara_vegetacion
    gc.collect()
    return np.array(parches), coordenadas, coords_utm, (filas_totales, cols_totales)

def predecir_riesgo(tensores_satelite: np.ndarray, meteo_matriz: np.ndarray) -> np.ndarray:
    print("\ncalculando probabilidad de riesgo mediante el modelo neuronal...")
    original_from_config = keras.layers.Dense.from_config
    @classmethod
    def patched_from_config(cls, config):
        config.pop('quantization_config', None)
        return original_from_config(config)
    keras.layers.Dense.from_config = patched_from_config

    modelo = keras.models.load_model(RUTA_MODELO)
    escalador = joblib.load(RUTA_ESCALADOR)

    meteo_escalada = escalador.transform(meteo_matriz)
    return modelo.predict([tensores_satelite, meteo_escalada], batch_size=32)

def reconstruir_mapa_calor(predicciones: np.ndarray, coordenadas: list, dimensiones_base: tuple, 
                           m_tierra: np.ndarray, m_vegetacion: np.ndarray, perfil: dict, 
                           ruta_salida: str, tamano: int = 64) -> None:
    filas, columnas = dimensiones_base
    mapa_riesgo = np.zeros((filas, columnas), dtype=np.float32)
    mapa_conteo = np.zeros((filas, columnas), dtype=np.float32)
    
    for pred, (f, c) in zip(predicciones, coordenadas):
        valor_limpio = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
        mapa_riesgo[f:f+tamano, c:c+tamano] += valor_limpio
        mapa_conteo[f:f+tamano, c:c+tamano] += 1
        
    with np.errstate(invalid='ignore', divide='ignore'):
        mapa_final = np.divide(mapa_riesgo, mapa_conteo)
        mapa_final = np.nan_to_num(mapa_final, nan=0.0)
        
    mapa_final[~m_vegetacion] = 0.0
    mapa_final[~m_tierra] = np.nan
    
    perfil.update(count=1, dtype=rasterio.float32, nodata=np.nan, compress='lzw')
    with rasterio.open(ruta_salida, 'w', **perfil) as dest:
        dest.write(mapa_final, 1)

def obtener_cmap_personalizado() -> LinearSegmentedColormap:
    """Mapeo de colores estandarizado AEMET en bloques sólidos (escalonado)."""
    nodos = [
        (0.00, '#3182bd'), (0.0999, '#3182bd'), # 0 a 10% (Muy bajo - Azul oscuro)
        (0.10, '#9ecae1'), (0.1999, '#9ecae1'), # 10 a 20% (Bajo - Azul claro)
        (0.20, '#a1d99b'), (0.3999, '#a1d99b'), # 20 a 40% (Moderado - Verde)
        (0.40, '#ffeda0'), (0.4999, '#ffeda0'), # 40 a 50% (Alto - Amarillo)
        (0.50, '#feb24c'), (0.5999, '#feb24c'), # 50 a 60% (Muy Alto - Naranja)
        (0.60, '#f03b20'), (1.00, '#f03b20')    # > 60% (Extremo - Rojo)
    ]
    cmap = LinearSegmentedColormap.from_list("RiesgoAEMET_Escalonado", nodos)
    cmap.set_under('black', alpha=0.0) 
    cmap.set_bad('black', alpha=0.0)
    return cmap

def exportar_visor_interactivo(ruta_tif_riesgo: str, ruta_tif_temp: str, ruta_tif_hr: str, 
                               ruta_tif_v: str, ruta_raw: str, isla: str, dir_salida: str) -> list:
    print(f"\nextrayendo capas puras y variables JavaScript para el visor web ({isla})...")
    
    ruta_base_png = os.path.join(dir_salida, f"base_rgb_{isla.replace(' ', '_')}.png")
    ruta_riesgo_png = os.path.join(dir_salida, f"capa_riesgo_{isla.replace(' ', '_')}.png")
    ruta_datos_js = os.path.join(dir_salida, f"datos_{isla.replace(' ', '_')}.js")
    
    cmap_riesgo = obtener_cmap_personalizado()
    
    # Reproyección táctica del riesgo a EPSG:4326
    with rasterio.open(ruta_tif_riesgo) as src_riesgo:
        transform_4326, width_4326, height_4326 = calculate_default_transform(
            src_riesgo.crs, 'EPSG:4326', src_riesgo.width, src_riesgo.height, *src_riesgo.bounds
        )
        mapa_4326 = np.zeros((height_4326, width_4326), dtype=np.float32)
        reproject(
            source=rasterio.band(src_riesgo, 1),
            destination=mapa_4326,
            src_transform=src_riesgo.transform,
            src_crs=src_riesgo.crs,
            dst_transform=transform_4326,
            dst_crs='EPSG:4326',
            resampling=Resampling.nearest
        )
        lon_min, lat_min, lon_max, lat_max = array_bounds(height_4326, width_4326, transform_4326)
        
        with rasterio.open(ruta_raw) as src_raw:
            banda_azul = np.zeros((height_4326, width_4326), dtype=np.float32)
            reproject(
                source=rasterio.band(src_raw, 1),
                destination=banda_azul,
                src_transform=src_raw.transform,
                src_crs=src_raw.crs,
                dst_transform=transform_4326,
                dst_crs='EPSG:4326',
                resampling=Resampling.nearest
            )
            mascara_nubes = banda_azul > 2200 
        
        valid_mask = (mapa_4326 >= 0.01) & (~np.isnan(mapa_4326)) & (mapa_4326 != 0.0)
        rgba_img = cmap_riesgo(mapa_4326)
        rgba_img[~valid_mask, 3] = 0.0 
        rgba_img[mascara_nubes] = [1.0, 1.0, 1.0, 0.65] 
        
        Image.fromarray((rgba_img * 255).astype(np.uint8)).save(ruta_riesgo_png)

    # Función auxiliar para reproyectar capas meteo de forma limpia
    def reproyectar_meteo(ruta_tif):
        with rasterio.open(ruta_tif) as src_meteo:
            matriz_4326 = np.zeros((height_4326, width_4326), dtype=np.float32)
            reproject(
                source=rasterio.band(src_meteo, 1),
                destination=matriz_4326,
                src_transform=src_meteo.transform,
                src_crs=src_meteo.crs,
                dst_transform=transform_4326,
                dst_crs='EPSG:4326',
                resampling=Resampling.nearest
            )
            return matriz_4326

    temp_4326 = reproyectar_meteo(ruta_tif_temp)
    hr_4326 = reproyectar_meteo(ruta_tif_hr)
    v_4326 = reproyectar_meteo(ruta_tif_v)
        
    factor_json = max(1, max(height_4326, width_4326) // 250) 
    
    with open(ruta_datos_js, 'w', encoding='utf-8') as f:
        f.write(f"window.datos_{isla.replace(' ', '_')} = {{\n")
        f.write(f"  riesgo: {json.dumps(np.nan_to_num(mapa_4326[::factor_json, ::factor_json], nan=-1.0).round(2).tolist())},\n")
        f.write(f"  temp: {json.dumps(np.nan_to_num(temp_4326[::factor_json, ::factor_json], nan=-99.0).round(1).tolist())},\n")
        f.write(f"  hr: {json.dumps(np.nan_to_num(hr_4326[::factor_json, ::factor_json], nan=-99.0).round(1).tolist())},\n")
        f.write(f"  viento: {json.dumps(np.nan_to_num(v_4326[::factor_json, ::factor_json], nan=-99.0).round(1).tolist())},\n")
        f.write(f"  gridH: {int(height_4326 // factor_json)},\n")
        f.write(f"  gridW: {int(width_4326 // factor_json)}\n")
        f.write("};\n")

    return [[lat_min, lon_min], [lat_max, lon_max]]


# BLOQUE PRINCIPAL DE EJECUCIÓN

if __name__ == "__main__":
    print("🚀 iniciando automatización masiva predictiva...")
    
    dir_docs = os.path.join(BASE_DIR, 'docs')
    os.makedirs(dir_docs, exist_ok=True)
    
    config_docs = {
        "islas": {}
    }
    
    for isla in BBOX_CANARIAS.keys():
        try:
            print(f"\n{'='*50}\n🛰️ PROCESANDO: {isla}\n{'='*50}")
            ruta_raw, fecha_satelite = descargar_satelite(isla)
            
            img_bruta, m_tierra, m_vegetacion, perfil = calcular_mascaras_fisicas(ruta_raw)
            tensores, coords, coords_utm, dim_base = extraer_parches_solapados(img_bruta, m_vegetacion, perfil, tamano=64, solape=8)
            
            if len(tensores) > 0:
                meteo_matriz = descargar_meteo_malla(isla, coords_utm)
                riesgos = predecir_riesgo(tensores, meteo_matriz)
                
                # Desglosamos todas las variables para exportarlas
                temperaturas = meteo_matriz[:, 0]
                humedades = meteo_matriz[:, 1]
                vientos = meteo_matriz[:, 2]
                
                ruta_export_tif = os.path.join(dir_docs, f'riesgo_{isla.replace(" ", "_")}.tif')
                ruta_export_temp = os.path.join(dir_docs, f'temp_{isla.replace(" ", "_")}.tif')
                ruta_export_hr = os.path.join(dir_docs, f'hr_{isla.replace(" ", "_")}.tif')
                ruta_export_v = os.path.join(dir_docs, f'viento_{isla.replace(" ", "_")}.tif')
                
                print("consolidando matrices cartográficas...")
                reconstruir_mapa_calor(riesgos, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_tif)
                reconstruir_mapa_calor(temperaturas, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_temp)
                reconstruir_mapa_calor(humedades, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_hr)
                reconstruir_mapa_calor(vientos, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_v)
                
                bounds_isla = exportar_visor_interactivo(ruta_export_tif, ruta_export_temp, ruta_export_hr, ruta_export_v, ruta_raw, isla, dir_docs)
                
                # Guardamos los metadatos específicos de la isla
                config_docs["islas"][isla] = {
                    "bounds": bounds_isla,
                    "fecha_sat": fecha_satelite
                }
                
                print(f"\n✅ {isla} completada con éxito. riesgo máximo detectado: {np.max(riesgos)*100:.1f}%")
            else:
                print(f"\n⚠️ operación omitida: no se detectó biomasa forestal procesable en {isla}.")
                
        except Exception as e:
            print(f"\n[ERROR CRÍTICO] caída del pipeline durante el procesado de {isla}: {e}")
            
        finally:
            if 'img_bruta' in locals(): del img_bruta
            if 'tensores' in locals(): del tensores
            if 'riesgos' in locals(): del riesgos
            if 'meteo_matriz' in locals(): del meteo_matriz
            if 'm_tierra' in locals(): del m_tierra
            if 'm_vegetacion' in locals(): del m_vegetacion
            gc.collect()
            keras.backend.clear_session()
            
    config_docs["fecha_actualizacion"] = datetime.now(ZoneInfo("Atlantic/Canary")).strftime("%Y-%m-%d %H:%M")
            
    ruta_config = os.path.join(dir_docs, 'config.js')
    with open(ruta_config, 'w', encoding='utf-8') as f:
        f.write(f"const configWeb = {json.dumps(config_docs, indent=4)};\n")