import os
import ee
import numpy as np
import joblib
import geemap
import rasterio
import matplotlib.pyplot as plt
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
    """Pregunta interactivamente al usuario qué isla desea analizar."""
    print("Islas disponibles:", ", ".join(BBOX_CANARIAS.keys()))
    
    isla = input("Introduce la isla a analizar: ").strip()
    while isla not in BBOX_CANARIAS:
        isla = input("Isla no válida. Escribe el nombre exacto de la lista: ").strip()
        
    return isla

def descargar_satelite(isla, proyecto_gcp="tfm-bbdd-499813"):
    """Descarga la última imagen Sentinel-2 forzando la proyección UTM de Canarias."""
    print(f"\nBuscando última imagen despejada de {isla} en Earth Engine...")
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
    
    bandas = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    imagen_export = imagen_mosaico.select(bandas)
    
    # Rutas locales relativas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_salida = os.path.join(base_dir, 'data', 'raw', f'satelite_{isla.replace(" ", "_")}.tif')
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

# MOTOR DE INFERENCIA Y CARTOGRAFÍA

def calcular_mascaras_fisicas(ruta):
    """Calcula índices espectrales para separar la silueta insular de las zonas forestales."""
    print(f"\nProcesando reflectancia y delimitando litoral...")
    with rasterio.open(ruta) as src:
        imagen_bruta = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)
        perfil_geo = src.profile
        
    b3_green = imagen_bruta[:, :, 1]
    b4_red   = imagen_bruta[:, :, 2]
    b8_nir   = imagen_bruta[:, :, 3]
    
    # Filtra cuerpos de agua (mar, embalses, presas, etc.)
    ndwi = (b3_green - b8_nir) / (b3_green + b8_nir + 1e-8)
    mascara_tierra = ndwi <= 0.3
    
    # Filtra todo lo que no sea cubierta vegetal (urbano, roca, carretera, etc.)
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-8)
    mascara_vegetacion = ndvi > 0.1
    
    return imagen_bruta, mascara_tierra, mascara_vegetacion, perfil_geo

def extraer_parches_solapados(imagen_bruta, mascara_tierra, mascara_vegetacion, tamano=64, solape=8):
    """Ejecuta una ventana deslizante con superposición sobre la matriz satelital."""
    print(f"\nEscaneando cuadrícula ({tamano}x{tamano} con solape de {solape}px)...")
    filas_totales, cols_totales, _ = imagen_bruta.shape
    paso = tamano - solape
    parches, coordenadas = [], []
    
    for f in range(0, filas_totales - tamano + 1, paso):
        for c in range(0, cols_totales - tamano + 1, paso):
            # Solo guardamos el parche si contiene al menos un píxel de vegetación válida
            if np.any(mascara_vegetacion[f:f+tamano, c:c+tamano]):
                parches.append(imagen_bruta[f:f+tamano, c:c+tamano, :])
                coordenadas.append((f, c))
                
    return np.array(parches), coordenadas, (filas_totales, cols_totales)

def predecir_riesgo(tensores_satelite, meteo_harmonie):
    """Inyecta los datos meteorológicos escalados y ejecuta la propagación hacia adelante."""

    modelo = keras.models.load_model(RUTA_MODELO)
    escalador = joblib.load(RUTA_ESCALADOR)
    
    meteo_escalada = escalador.transform([meteo_harmonie])
    meteo_masiva = np.repeat(meteo_escalada, tensores_satelite.shape[0], axis=0)
    
    return modelo.predict([tensores_satelite, meteo_masiva], batch_size=32)

def reconstruir_mapa_calor(predicciones, coordenadas, dimensiones_base, m_tierra, m_vegetacion, perfil, ruta_salida, tamano=64):
    """Promedia el riesgo por píxel basándose en los solapes y delimita la cartografía final."""

    filas, columnas = dimensiones_base
    mapa_riesgo = np.zeros((filas, columnas), dtype=np.float32)
    mapa_conteo = np.zeros((filas, columnas), dtype=np.float32)
    
    # Acumulamos el riesgo y contamos cuántos parches han pasado por cada píxel
    for pred, (f, c) in zip(predicciones, coordenadas):
        mapa_riesgo[f:f+tamano, c:c+tamano] += pred[0]
        mapa_conteo[f:f+tamano, c:c+tamano] += 1
        
    # Calculamos el promedio aritmético exacto para cada píxel individual
    with np.errstate(invalid='ignore', divide='ignore'):
        mapa_final = np.divide(mapa_riesgo, mapa_conteo)
        mapa_final = np.nan_to_num(mapa_final, nan=0.0)
        
    # Limpieza visual
    # Todo lo que no sea vegetación (urbano, roca) se queda con riesgo nulo (0.0).
    mapa_final[~m_vegetacion] = 0.0
    # Todo lo que no sea tierra (océano) se vuelve transparente (NaN).
    mapa_final[~m_tierra] = np.nan
    
    perfil.update(count=1, dtype=rasterio.float32, nodata=np.nan, compress='lzw')
    with rasterio.open(ruta_salida, 'w', **perfil) as dest:
        dest.write(mapa_final, 1)
    return mapa_final

def exportar_dashboard_png(mapa_riesgo, mascara_tierra, isla, ruta_png):
    """
    Genera un mapa visual de operaciones tipo meteorológico con la silueta 
    costera en negro y el gradiente térmico, exportándolo como PNG.
    """
    
    plt.figure(figsize=(10, 10), dpi=200)
    
    # Configuración de colores
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under('darkgray')       # Valores = 0.0 (Ciudad/Roca) en gris
    cmap.set_bad('white', alpha=0)   # Valores NaN (Océano) transparentes
    
    # Dibujamos el mapa térmico
    im = plt.imshow(mapa_riesgo, cmap=cmap, vmin=0.01, vmax=1.0)
     
    plt.contour(mascara_tierra, levels=[0.5], colors='black', linewidths=1.2)    
    plt.colorbar(im, label="Probabilidad de Riesgo Forestal (0.0 - 1.0)", shrink=0.7)
    plt.title(f"Mapa Operativo de Riesgo - {isla} (CECOPIN)", fontsize=15, fontweight='bold')
    plt.axis('off')    
    plt.savefig(ruta_png, bbox_inches='tight', transparent=True)
    plt.close()
    
# EJECUCIÓN DEL PIPELINE
if __name__ == "__main__":
    try:
        isla = seleccionar_isla()
        ruta_raw = descargar_satelite(isla)
        
        img_bruta, m_tierra, m_vegetacion, perfil = calcular_mascaras_fisicas(ruta_raw)
        tensores, coords, dim_base = extraer_parches_solapados(img_bruta, m_tierra, m_vegetacion, tamano=64, solape=8)
        
        if len(tensores) > 0:
            # Ejemplo de vector térmico: 35.5ºC, 15% HR, 25 km/h viento
            meteo_operativa = [15, 90.0, 5.0] 
            riesgos = predecir_riesgo(tensores, meteo_operativa)
            
            ruta_export_tif = os.path.join(BASE_DIR, 'data', 'processed', f'riesgo_{isla.replace(" ", "_")}.tif')
            ruta_export_png = os.path.join(BASE_DIR, 'data', 'processed', f'mapa_{isla.replace(" ", "_")}.png')
            os.makedirs(os.path.dirname(ruta_export_tif), exist_ok=True)
            
            mapa_final = reconstruir_mapa_calor(riesgos, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_tif)
            
            exportar_dashboard_png(mapa_final, m_tierra, isla, ruta_export_png)
            
            print(f"\n[SISTEMA COMPLETADO] El índice máximo detectado es: {np.max(riesgos)*100:.2f}%")
        else:
            print("\n[OPERACIÓN ABORTADA] No se detectó cobertura vegetal en el cuadrante de descarga.")
            
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {e}")