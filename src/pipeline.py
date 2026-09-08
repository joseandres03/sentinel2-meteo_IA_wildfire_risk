import os
import ee
import numpy as np
import joblib
import geemap
import rasterio
import requests
import pyproj
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, 'models', 'modelo_late_fusion_definitivo.keras')
RUTA_ESCALADOR = os.path.join(BASE_DIR, 'models', 'robust_scaler_meteo.pkl')

# CONFIGURACIÓN ESPACIAL

BBOX_CANARIAS = {
    "La Gomera": [-17.37, 28.01, -17.09, 28.23],
    "Tenerife": [-16.94, 27.97, -16.11, 28.59],
    "Gran Canaria": [-15.83, 27.70, -15.36, 28.18],
    "La Palma": [-18.00, 28.43, -17.72, 28.85],
    "El Hierro": [-18.17, 27.62, -17.88, 27.86],
    "Lanzarote": [-13.91, 28.83, -13.33, 29.26],
    "Fuerteventura": [-14.52, 28.01, -13.82, 28.76]
}

def seleccionar_isla():
    """
    Pregunta interactivamente al usuario qué isla desea analizar.
    
    Returns:
        str: Nombre exacto de la isla validado contra el diccionario de BBOX.
    """
    print("=== SISTEMA PREDICTIVO DE INCENDIOS CECOPIN ===")
    print("Islas disponibles:", ", ".join(BBOX_CANARIAS.keys()))
    
    isla = input("\nIntroduce la isla a analizar: ").strip()
    while isla not in BBOX_CANARIAS:
        isla = input("Isla no válida. Escribe el nombre exacto de la lista: ").strip()
        
    return isla

# INGESTA SATELITAL Y METEOROLÓGICA

def descargar_satelite(isla, proyecto_gcp="tfm-bbdd-499813"):
    """
    Descarga la última imagen Sentinel-2 forzando la proyección UTM de Canarias.
    
    Args:
        isla (str): Nombre de la isla a procesar.
        proyecto_gcp (str): ID del proyecto en Google Cloud para autenticar Earth Engine.
        
    Returns:
        str: Ruta local donde se ha guardado el GeoTIFF crudo.
    """
    print(f"\n[PASO 1] Buscando última imagen despejada de {isla} en Earth Engine...")
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
    print(f"-> Última pasada detectada: {fecha_captura}. Ensamblando mosaico insular...")
    
    imagen_mosaico = coleccion.sort('system:time_start', True).mosaic()
    imagen_export = imagen_mosaico.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    
    ruta_salida = os.path.join(BASE_DIR, 'data', 'raw', f'satelite_{isla.replace(" ", "_")}.tif')
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    print("-> Descargando GeoTIFF rectificado (6 bandas a 10m de resolución)...")
    geemap.download_ee_image(
        image=imagen_export,
        filename=ruta_salida,
        region=region,
        scale=10,
        crs='EPSG:32628'
    )
    
    return ruta_salida

def descargar_meteo_malla(isla, coords_utm):
    """
    Genera una cuadrícula sobre la isla, descarga el clima (AROME) 
    y calcula la interpolación espacial para asignar a cada parche su microclima exacto.
    
    Args:
        isla (str): Nombre de la isla.
        coords_utm (list): Lista de tuplas (X, Y) con los centroides UTM de cada parche.
        
    Returns:
        np.array: Matriz de dimensiones (N_parches, 3) con [Temp, HR, Viento] para cada cuadrante.
    """
    print(f"\n[PASO 4] Descargando malla AROME e interpolando microclimas...")
    bbox = BBOX_CANARIAS[isla]
    
    # Generamos una malla de 5x5 puntos sobre el Bounding Box de la isla
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
    
    # Petición masiva a la API
    respuesta = requests.get(url, params=parametros).json()
    
    # Barrera de seguridad para cazar errores de la API en lugar de romper el código
    if isinstance(respuesta, dict) and respuesta.get("error"):
        raise RuntimeError(f"La API de Open-Meteo rechazó la conexión: {respuesta.get('reason')}")
    
    # El índice 13 corresponde a las 13:00h del día
    t_malla = [loc['hourly']['temperature_2m'][13] for loc in respuesta]
    hr_malla = [loc['hourly']['relative_humidity_2m'][13] for loc in respuesta]
    v_malla = [loc['hourly']['wind_speed_10m'][13] for loc in respuesta]
    puntos_origen = np.column_stack((malla_lons.flatten(), malla_lats.flatten()))
    
    # Convertimos los centroides UTM de los parches a Lat/Lon para la interpolación
    transformador = pyproj.Transformer.from_crs("EPSG:32628", "EPSG:4326", always_xy=True)
    lons_parches, lats_parches = transformador.transform(
        [c[0] for c in coords_utm],
        [c[1] for c in coords_utm]
    )
    puntos_destino = np.column_stack((lons_parches, lats_parches))
    
    # Interpolamos los datos meteorológicos (cruce espacial)
    t_interp = griddata(puntos_origen, t_malla, puntos_destino, method='linear')
    hr_interp = griddata(puntos_origen, hr_malla, puntos_destino, method='linear')
    v_interp = griddata(puntos_origen, v_malla, puntos_destino, method='linear')
    
    # Si algún parche cae en el borde fuera de la malla (NaN), usamos el punto más cercano
    if np.isnan(t_interp).any():
        mascara_nan = np.isnan(t_interp)
        t_interp[mascara_nan] = griddata(puntos_origen, t_malla, puntos_destino[mascara_nan], method='nearest')
        hr_interp[mascara_nan] = griddata(puntos_origen, hr_malla, puntos_destino[mascara_nan], method='nearest')
        v_interp[mascara_nan] = griddata(puntos_origen, v_malla, puntos_destino[mascara_nan], method='nearest')
        
    return np.column_stack((t_interp, hr_interp, v_interp))


# MOTOR DE INFERENCIA Y CARTOGRAFÍA

def calcular_mascaras_fisicas(ruta):
    """
    Calcula índices espectrales para separar la silueta insular de las zonas forestales.
    
    Args:
        ruta (str): Ruta local del GeoTIFF satelital.
        
    Returns:
        tuple: (Matriz bruta 3D, Máscara binaria de tierra, Máscara binaria de vegetación, Perfil geoespacial)
    """
    print(f"\n[PASO 2] Procesando reflectancia y delimitando litoral...")
    with rasterio.open(ruta) as src:
        imagen_bruta = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)
        perfil_geo = src.profile
        
    b3_green = imagen_bruta[:, :, 1]
    b4_red   = imagen_bruta[:, :, 2]
    b8_nir   = imagen_bruta[:, :, 3]
    
    ndwi = (b3_green - b8_nir) / (b3_green + b8_nir + 1e-8)
    mascara_tierra = ndwi <= 0.3
    
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-8)
    mascara_vegetacion = ndvi > 0.1
    
    return imagen_bruta, mascara_tierra, mascara_vegetacion, perfil_geo

def extraer_parches_solapados(imagen_bruta, mascara_vegetacion, perfil, tamano=64, solape=8):
    """
    Ejecuta una ventana deslizante con superposición sobre la matriz satelital, 
    calculando la coordenada espacial UTM exacta del centro de cada tensor.
    
    Args:
        imagen_bruta (np.array): Matriz 3D con la imagen satelital completa.
        mascara_vegetacion (np.array): Matriz 2D binaria (True = hay vegetación).
        perfil (dict): Metadatos espaciales devueltos por Rasterio.
        tamano (int): Tamaño del parche (64x64 por defecto).
        solape (int): Píxeles de superposición entre un parche y el siguiente.
        
    Returns:
        tuple: (Array de tensores, Lista de coordenadas (fila, col), Lista UTM (x, y), Dimensiones base)
    """
    print(f"\n[PASO 3] Escaneando cuadrícula ({tamano}x{tamano} con solape de {solape}px)...")
    filas_totales, cols_totales, _ = imagen_bruta.shape
    paso = tamano - solape
    parches, coordenadas, coords_utm = [], [], []
    transform = perfil['transform']
    
    for f in range(0, filas_totales - tamano + 1, paso):
        for c in range(0, cols_totales - tamano + 1, paso):
            if np.any(mascara_vegetacion[f:f+tamano, c:c+tamano]):
                parches.append(imagen_bruta[f:f+tamano, c:c+tamano, :])
                coordenadas.append((f, c))
                
                # Calculamos el centro del parche georreferenciado
                centro_f = f + (tamano / 2.0)
                centro_c = c + (tamano / 2.0)
                utm_x, utm_y = rasterio.transform.xy(transform, centro_f, centro_c)
                coords_utm.append((utm_x, utm_y))
                
    return np.array(parches), coordenadas, coords_utm, (filas_totales, cols_totales)

def predecir_riesgo(tensores_satelite, meteo_matriz):
    """
    Inyecta los tensores y la meteorología local escalada al modelo para ejecutar inferencia.
    
    Args:
        tensores_satelite (np.array): Array 4D con los parches satelitales procesados.
        meteo_matriz (np.array): Matriz 2D (N_parches, 3) con la meteorología interpolada.
        
    Returns:
        np.array: Predicciones de probabilidad de riesgo forestal para cada parche.
    """
    print("\n[PASO 5] Calculando probabilidad de riesgo mediante Late Fusion...")
    modelo = keras.models.load_model(RUTA_MODELO)
    escalador = joblib.load(RUTA_ESCALADOR)
    
    meteo_escalada = escalador.transform(meteo_matriz)
    
    return modelo.predict([tensores_satelite, meteo_escalada], batch_size=32)

def reconstruir_mapa_calor(predicciones, coordenadas, dimensiones_base, m_tierra, m_vegetacion, perfil, ruta_salida, tamano=64):
    """
    Promedia el riesgo por píxel basándose en solapes espaciales y delimita la cartografía final.
    
    Args:
        predicciones (np.array): Salida de la red neuronal.
        coordenadas (list): Índices de fila/columna donde se originó cada parche.
        dimensiones_base (tuple): Altura y anchura de la imagen insular completa.
        m_tierra (np.array): Máscara binaria de costa.
        m_vegetacion (np.array): Máscara binaria de superficie forestal.
        perfil (dict): Diccionario de proyección y georreferencia original de Rasterio.
        ruta_salida (str): Ruta local donde se guardará el GeoTIFF predictivo.
        tamano (int): Tamaño utilizado durante el escaneo.
    """
    print("\n[PASO 6] Consolidando cartografía matricial promediada...")
    filas, columnas = dimensiones_base
    mapa_riesgo = np.zeros((filas, columnas), dtype=np.float32)
    mapa_conteo = np.zeros((filas, columnas), dtype=np.float32)
    
    # Acumulamos el riesgo sumando capas superpuestas
    for pred, (f, c) in zip(predicciones, coordenadas):
        mapa_riesgo[f:f+tamano, c:c+tamano] += pred[0]
        mapa_conteo[f:f+tamano, c:c+tamano] += 1
        
    # Calculamos el promedio matemático exacto
    with np.errstate(invalid='ignore', divide='ignore'):
        mapa_final = np.divide(mapa_riesgo, mapa_conteo)
        mapa_final = np.nan_to_num(mapa_final, nan=0.0)
        
    # Limpieza visual y enmascarado operativo
    mapa_final[~m_vegetacion] = 0.0
    mapa_final[~m_tierra] = np.nan
    
    perfil.update(count=1, dtype=rasterio.float32, nodata=np.nan, compress='lzw')
    with rasterio.open(ruta_salida, 'w', **perfil) as dest:
        dest.write(mapa_final, 1)

def exportar_dashboard_png(ruta_tif, isla, ruta_png):
    """
    Descarga la frontera vectorial oficial de OSM y la superpone al mapa predictivo térmico.
    
    Args:
        ruta_tif (str): Ruta del GeoTIFF procesado en el paso anterior.
        isla (str): Nombre de la isla.
        ruta_png (str): Ruta de destino para la exportación visual.
    """
    print(f"\n[PASO 7] Descargando frontera vectorial (OSM) y renderizando dashboard...")
    frontera = ox.geocode_to_gdf(f"{isla}, Canarias, España")
    
    with rasterio.open(ruta_tif) as src:
        mapa_riesgo = src.read(1)
        limites = src.bounds
        extension_utm = [limites.left, limites.right, limites.bottom, limites.top]
        frontera_utm = frontera.to_crs(src.crs)
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.set_facecolor('white') 
        
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_under('darkgray')
        cmap.set_bad('white', alpha=0)
        
        # Renderizado térmico referenciado
        im = ax.imshow(mapa_riesgo, cmap=cmap, vmin=0.01, vmax=1.0, extent=extension_utm)
        
        # Trazo del límite costero oficial
        frontera_utm.boundary.plot(ax=ax, color='black', linewidth=1.0)
    
        plt.colorbar(im, ax=ax, label="Probabilidad de Riesgo Forestal (0.0 - 1.0)", shrink=0.7)
        ax.set_title(f"Mapa Operativo de Riesgo - {isla} (CECOPIN)", fontsize=15, fontweight='bold')
        ax.axis('off')
        
        plt.savefig(ruta_png, bbox_inches='tight', facecolor='white')
        plt.close()
        
    print(f"-> Mapa visual guardado en: {ruta_png}")

# EJECUCIÓN DEL PIPELINE


if __name__ == "__main__":
    try:
        isla = seleccionar_isla()
        
        ruta_raw = descargar_satelite(isla)
        img_bruta, m_tierra, m_vegetacion, perfil = calcular_mascaras_fisicas(ruta_raw)
        
        # OBTENCIÓN DE TENSORES: Ahora extraemos también los centroides UTM espaciales
        tensores, coords, coords_utm, dim_base = extraer_parches_solapados(img_bruta, m_vegetacion, perfil, tamano=64, solape=8)
        
        if len(tensores) > 0:
            # INGESTA CLIMÁTICA: Se crea la matriz local (T, HR, Viento) para cada uno de los tensores
            meteo_matriz = descargar_meteo_malla(isla, coords_utm)
            
            # INFERENCIA: La red ingiere tanto la imagen espacial como su microclima asociado
            riesgos = predecir_riesgo(tensores, meteo_matriz)
            
            ruta_export_tif = os.path.join(BASE_DIR, 'data', 'processed', f'riesgo_{isla.replace(" ", "_")}.tif')
            ruta_export_png = os.path.join(BASE_DIR, 'data', 'processed', f'mapa_{isla.replace(" ", "_")}.png')
            os.makedirs(os.path.dirname(ruta_export_tif), exist_ok=True)
            
            reconstruir_mapa_calor(riesgos, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_tif)
            exportar_dashboard_png(ruta_export_tif, isla, ruta_export_png)
            
            print(f"\n[¡ÉXITO!] Sistema automatizado completado. Riesgo máximo: {np.max(riesgos)*100:.1f}%")
        else:
            print("\n[OPERACIÓN ABORTADA] No se detectó cobertura vegetal en el cuadrante de descarga.")
            
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {e}")