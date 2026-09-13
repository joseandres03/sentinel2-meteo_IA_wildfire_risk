import os
import gc
import json
import time
from datetime import datetime, timedelta

import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from PIL import Image
from scipy.interpolate import griddata

import ee
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


# MÓDULO DE INGESTA SATELITAL Y METEOROLÓGICA

def seleccionar_isla() -> str:
    """
    Solicita interactivamente por consola la isla a analizar.

    Returns
    -------
    str
        Nombre exacto de la isla validado contra la constante BBOX_CANARIAS.
    """
    print("Modelo probabilístico de riesgo de incendio con Deep Learning")
    print("Islas disponibles:", ", ".join(BBOX_CANARIAS.keys()))
    
    isla = input("\nIntroduce la isla a analizar: ").strip()
    while isla not in BBOX_CANARIAS:
        isla = input("Isla no válida. Escribe el nombre exacto de la lista: ").strip()
        
    return isla

def descargar_satelite(isla: str, proyecto_gcp: str = "tfm-bbdd-499813") -> tuple:
    """
    Descargo la última imagen Sentinel-2 mediante petición HTTP directa en streaming 
    para evitar bloqueos por interbloqueo de hilos (deadlocks) en GitHub Actions.

    Parameters
    ----------
    isla : str
        Nombre de la isla objetivo.
    proyecto_gcp : str, opcional
        ID del proyecto en Google Cloud para autenticación de Earth Engine.

    Returns
    -------
    tuple
        (ruta_salida, fecha_captura) con la ubicación local del archivo GeoTIFF y su fecha.
    """
    print(f"\nbuscando la última imagen de la constelación Sentinel-2 para {isla}...")
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
    
    print("-> descargando GeoTIFF mediante enlace directo seguro...")
    url_descarga = imagen_export.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:32628',
        'region': region.getInfo()['coordinates'],
        'format': 'GEO_TIFF'
    })
    
    respuesta = requests.get(url_descarga, stream=True)
    respuesta.raise_for_status()
    
    with open(ruta_salida, 'wb') as f:
        for chunk in respuesta.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                
    return ruta_salida, fecha_captura

def descargar_meteo_malla(isla: str, coords_utm: list) -> np.ndarray:
    """
    Genero una cuadrícula perimetral sobre la isla, descargo la predicción 
    del modelo AROME y aplico interpolación espacial para asignar a cada 
    parche de 64x64 su microclima exacto.

    Parameters
    ----------
    isla : str
        Nombre de la isla a analizar.
    coords_utm : list
        Lista de tuplas (X, Y) con los centroides UTM de cada parche de imagen.

    Returns
    -------
    np.ndarray
        Matriz de dimensiones (N_parches, 3) estructurada como [Temp, HR, Viento].
    """
    print(f"\ndescargando datos meteorológicos del HARMONIE-AROME e interpolando geografía...")
    bbox = BBOX_CANARIAS[isla]
    
    # construyo una malla de 5x5 puntos sobre el Bounding Box de la isla
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
    
    # escudo de seguridad: sistema de reintentos contra micro-cortes de la API
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
    
    # aíslo los registros de las 12:00h (índice 12) como referencia del mediodía
    t_malla = [loc['hourly']['temperature_2m'][12] for loc in respuesta]
    hr_malla = [loc['hourly']['relative_humidity_2m'][12] for loc in respuesta]
    v_malla = [loc['hourly']['wind_speed_10m'][12] for loc in respuesta]
    puntos_origen = np.column_stack((malla_lons.flatten(), malla_lats.flatten()))
    
    # convierto los centroides UTM a Lat/Lon para alinear el sistema de coordenadas
    transformador = pyproj.Transformer.from_crs("EPSG:32628", "EPSG:4326", always_xy=True)
    lons_parches, lats_parches = transformador.transform(
        [c[0] for c in coords_utm],
        [c[1] for c in coords_utm]
    )
    puntos_destino = np.column_stack((lons_parches, lats_parches))
    
    # ejecuto el cruce espacial por interpolación bilineal
    t_interp = griddata(puntos_origen, t_malla, puntos_destino, method='linear')
    hr_interp = griddata(puntos_origen, hr_malla, puntos_destino, method='linear')
    v_interp = griddata(puntos_origen, v_malla, puntos_destino, method='linear')
    
    # soluciono los parches periféricos (NaN) usando el nodo de la malla más cercano
    if np.isnan(t_interp).any():
        mascara_nan = np.isnan(t_interp)
        t_interp[mascara_nan] = griddata(puntos_origen, t_malla, puntos_destino[mascara_nan], method='nearest')
        hr_interp[mascara_nan] = griddata(puntos_origen, hr_malla, puntos_destino[mascara_nan], method='nearest')
        v_interp[mascara_nan] = griddata(puntos_origen, v_malla, puntos_destino[mascara_nan], method='nearest')
        
    return np.column_stack((t_interp, hr_interp, v_interp))


# MÓDULO DE INFERENCIA Y CARTOGRAFÍA

def calcular_mascaras_fisicas(ruta: str) -> tuple:
    """
    Leo el archivo TIFF en crudo y aplico álgebra de mapas para aislar 
    el océano de la tierra y la zona urbana de la masa forestal.

    Parameters
    ----------
    ruta : str
        Ruta local del GeoTIFF satelital.

    Returns
    -------
    tuple
        (Matriz bruta 3D, Máscara binaria litoral, Máscara binaria forestal, Perfil de Rasterio)
    """
    print("\nprocesando reflectancia y delimitando el litoral...")
    with rasterio.open(ruta) as src:
        raster_data = src.read()
        perfil_geo = src.profile
        
    imagen_bruta = np.transpose(raster_data, (1, 2, 0)).astype(np.float32)
    
    # libero el búfer nativo para no asfixiar la RAM en islas grandes (ej. Tenerife)
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
    """
    Paso una ventana deslizante sobre la matriz satelital recortando tensores
    allí donde existe vegetación, guardando su anclaje UTM exacto.

    Parameters
    ----------
    imagen_bruta : np.ndarray
        Array 3D con la reflectancia.
    mascara_vegetacion : np.ndarray
        Array 2D binario que valida el bioma.
    perfil : dict
        Metadatos espaciales de Rasterio para la transformación afín.
    tamano : int, opcional
        Resolución de entrada de la CNN (por defecto 64x64).
    solape : int, opcional
        Número de píxeles superpuestos para suavizar la cartografía.

    Returns
    -------
    tuple
        (Matriz de tensores, Coordenadas internas (fil, col), Coordenadas UTM, Dimensiones originales)
    """
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
                
    # destrucción forzosa de la imagen gigante en RAM antes de retornar los fragmentos
    del imagen_bruta, mascara_vegetacion
    gc.collect()
                
    return np.array(parches), coordenadas, coords_utm, (filas_totales, cols_totales)

def predecir_riesgo(tensores_satelite: np.ndarray, meteo_matriz: np.ndarray) -> np.ndarray:
    """Ejecuta la inferencia alimentando el modelo Late Fusion con las dos ramas de datos."""
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
    """
    Fusiona las predicciones de los miles de parches sueltos de vuelta en un lienzo único,
    promediando el riesgo en las zonas solapadas y aplicando la máscara insular.
    """
    print("consolidando matriz cartográfica...")
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
    """Mapeo de colores estandarizado según la escala táctica de CECOPIN."""
    nodos = [
        (0.00, '#228B22'), (0.30, '#ADFF2F'), (0.45, '#FFA500'), 
        (0.60, '#FF0000'), (0.85, '#800080'), (1.00, '#F8E6FF')
    ]
    cmap = LinearSegmentedColormap.from_list("RiesgoCanarias", nodos)
    cmap.set_under('black', alpha=0.0) 
    cmap.set_bad('black', alpha=0.0)
    return cmap

def exportar_dashboard_png(ruta_tif: str, isla: str, ruta_png: str) -> None:
    """Dibuja un dashboard de impacto visual directo usando la librería OSMnx."""
    print(f"\nrenderizando cartografía estática (PNG)...")
    frontera = ox.geocode_to_gdf(f"{isla}, Canarias, España")
    cmap_riesgo = obtener_cmap_personalizado()
    
    with rasterio.open(ruta_tif) as src:
        mapa_riesgo = src.read(1)
        limites = src.bounds
        extension_utm = [limites.left, limites.right, limites.bottom, limites.top]
        frontera_utm = frontera.to_crs(src.crs)
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.set_facecolor('white') 
        
        im = ax.imshow(mapa_riesgo, cmap=cmap_riesgo, vmin=0.01, vmax=1.0, extent=extension_utm)
        frontera_utm.boundary.plot(ax=ax, color='black', linewidth=1.0)
        
        ax.set_xlim(limites.left, limites.right)
        ax.set_ylim(limites.bottom, limites.top)
    
        plt.colorbar(im, ax=ax, label="Probabilidad de riesgo de incendio (0.0 - 1.0)", shrink=0.7)
        ax.set_title(f"Mapa de riesgo forestal - {isla}", fontsize=15, fontweight='bold')
        ax.axis('off')
        
        plt.savefig(ruta_png, bbox_inches='tight', facecolor='white')
        plt.close()

def exportar_visor_interactivo(ruta_tif_riesgo: str, ruta_tif_temp: str, ruta_raw: str, 
                               isla: str, dir_salida: str, fecha_sat: str) -> list:
    """
    Fuerza la reproyección del modelo a coordenadas esféricas (EPSG:4326) para 
    compatibilidad web, extrayendo las capas base en PNG y el array de predicciones en JSON.
    """
    print(f"\nextrayendo capas puras y variables JavaScript para el visor web ({isla})...")
    
    ruta_base_png = os.path.join(dir_salida, f"base_rgb_{isla.replace(' ', '_')}.png")
    ruta_riesgo_png = os.path.join(dir_salida, f"capa_riesgo_{isla.replace(' ', '_')}.png")
    ruta_datos_js = os.path.join(dir_salida, f"datos_{isla.replace(' ', '_')}.js")
    ruta_leyenda = os.path.join(dir_salida, "leyenda.png")
    
    cmap_riesgo = obtener_cmap_personalizado()
    
    # 1. reproyección táctica a EPSG:4326
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
        
        # intercepto nubosidad directamente desde la banda azul del crudo
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

    # 2. compresión de la cartografía y meteo para la capa interactiva del cursor web
    with rasterio.open(ruta_tif_temp) as src_t:
        temp_4326 = np.zeros((height_4326, width_4326), dtype=np.float32)
        reproject(
            source=rasterio.band(src_t, 1),
            destination=temp_4326,
            src_transform=src_t.transform,
            src_crs=src_t.crs,
            dst_transform=transform_4326,
            dst_crs='EPSG:4326',
            resampling=Resampling.nearest
        )
        
        factor_json = max(1, max(height_4326, width_4326) // 250) 
        arr_r_json = mapa_4326[::factor_json, ::factor_json]
        arr_t_json = temp_4326[::factor_json, ::factor_json]
        
        with open(ruta_datos_js, 'w', encoding='utf-8') as f:
            f.write(f"window.datos_{isla.replace(' ', '_')} = {{\n")
            f.write(f"  riesgo: {json.dumps(np.nan_to_num(arr_r_json, nan=-1.0).round(2).tolist())},\n")
            f.write(f"  temp: {json.dumps(np.nan_to_num(arr_t_json, nan=-99.0).round(1).tolist())},\n")
            f.write(f"  gridH: {int(height_4326 // factor_json)},\n")
            f.write(f"  gridW: {int(width_4326 // factor_json)}\n")
            f.write("};\n")

    # 3. renderizado gráfico de la leyenda explicativa
    fig_leg, ax_leg = plt.subplots(figsize=(8, 1.5), dpi=150)
    fig_leg.subplots_adjust(bottom=0.4, top=0.7)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap_riesgo),
                      cax=ax_leg, orientation='horizontal')
    cb.set_label('Probabilidad de Riesgo (0.0 a 1.0)', fontsize=12, fontweight='bold')
    
    nube_patch = mpatches.Patch(color='#FFFFFF', ec='#888888', label='Nubes (Área sin datos)')
    fig_leg.legend(handles=[nube_patch], loc='upper center', bbox_to_anchor=(0.5, 1.4), frameon=False, fontsize=11)
    
    plt.savefig(ruta_leyenda, bbox_inches='tight', transparent=True)
    plt.close()

    # 4. extracción visual del satélite en color real (RGB)
    with rasterio.open(ruta_raw) as src_raw:
        factor = max(1, max(src_raw.height, src_raw.width) // 2000)
        h_new, w_new = src_raw.height // factor, src_raw.width // factor
        b_blue = src_raw.read(1, out_shape=(h_new, w_new), resampling=rasterio.enums.Resampling.bilinear)
        b_green = src_raw.read(2, out_shape=(h_new, w_new), resampling=rasterio.enums.Resampling.bilinear)
        b_red = src_raw.read(3, out_shape=(h_new, w_new), resampling=rasterio.enums.Resampling.bilinear)
        rgb = np.clip(np.dstack((b_red, b_green, b_blue)) / 3000.0, 0, 1) 
        Image.fromarray((rgb * 255).astype(np.uint8)).save(ruta_base_png)

    print(f"-> Archivos web (JS y PNGs) exportados y optimizados para {isla}.")
    return [[lat_min, lon_min], [lat_max, lon_max]]


# BLOQUE PRINCIPAL DE EJECUCIÓN (ENTRY POINT)

if __name__ == "__main__":
    print("🚀 iniciando automatización masiva predictiva...")
    
    dir_docs = os.path.join(BASE_DIR, 'docs')
    os.makedirs(dir_docs, exist_ok=True)
    
    config_docs = {
        "bounds": {},
        "fecha_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    
    # escaneo secuencial y destructivo (en memoria) isla por isla
    for isla in BBOX_CANARIAS.keys():
        try:
            print(f"\n{'='*50}\n🛰️ PROCESANDO: {isla}\n{'='*50}")
            ruta_raw, fecha_satelite = descargar_satelite(isla)
            
            img_bruta, m_tierra, m_vegetacion, perfil = calcular_mascaras_fisicas(ruta_raw)
            tensores, coords, coords_utm, dim_base = extraer_parches_solapados(img_bruta, m_vegetacion, perfil, tamano=64, solape=8)
            
            if len(tensores) > 0:
                meteo_matriz = descargar_meteo_malla(isla, coords_utm)
                riesgos = predecir_riesgo(tensores, meteo_matriz)
                temperaturas = meteo_matriz[:, 0]
                
                ruta_export_tif = os.path.join(dir_docs, f'riesgo_{isla.replace(" ", "_")}.tif')
                ruta_export_temp = os.path.join(dir_docs, f'temp_{isla.replace(" ", "_")}.tif')
                
                reconstruir_mapa_calor(riesgos, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_tif)
                reconstruir_mapa_calor(temperaturas, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_temp)
                
                bounds_isla = exportar_visor_interactivo(ruta_export_tif, ruta_export_temp, ruta_raw, isla, dir_docs, fecha_satelite)
                config_docs["bounds"][isla] = bounds_isla
                
                print(f"\n✅ {isla} completada con éxito. riesgo máximo detectado: {np.max(riesgos)*100:.1f}%")
            else:
                print(f"\n⚠️ operación omitida: no se detectó biomasa forestal procesable en {isla}.")
                
        except Exception as e:
            print(f"\n[ERROR CRÍTICO] caída del pipeline durante el procesado de {isla}: {e}")
            
        finally:
            # recolección de basura estricta para garantizar la supervivencia del servidor CI/CD
            print(f" liberando buffers y limpiando gráficos en RAM para {isla}...")
            if 'img_bruta' in locals(): del img_bruta
            if 'tensores' in locals(): del tensores
            if 'riesgos' in locals(): del riesgos
            if 'meteo_matriz' in locals(): del meteo_matriz
            if 'm_tierra' in locals(): del m_tierra
            if 'm_vegetacion' in locals(): del m_vegetacion
            gc.collect()
            keras.backend.clear_session()
            
    # inyección de los linderos perimetrales al motor web
    ruta_config = os.path.join(dir_docs, 'config.js')
    with open(ruta_config, 'w', encoding='utf-8') as f:
        f.write(f"const configWeb = {json.dumps(config_docs, indent=4)};\n")