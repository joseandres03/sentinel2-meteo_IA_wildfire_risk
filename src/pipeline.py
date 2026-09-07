import os
import numpy as np
import joblib
import rasterio
import ee
import geemap
from tensorflow import keras
from datetime import datetime, timedelta

# CONFIGURACIÓN DE RUTAS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, 'models', 'modelo_late_fusion_definitivo.keras')
RUTA_ESCALADOR = os.path.join(BASE_DIR, 'models', 'robust_scaler_meteo.pkl')

# FASE 0: INGESTA DESDE COPERNICUS
BBOX_CANARIAS = {
    "Canarias": [-18.16, 27.63, -13.33, 29.42]
}

def obtener_ultima_imagen_gee(isla, ruta_salida, proyecto_gcp="tfm-bbdd-499813"):
    """
    Se conecta a Google Earth Engine, localiza la órbita más reciente de la isla de interés 
    y la descarga como GeoTIFF conservando la resolución de 10m y la proyección nativa.
    """
    
    ee.Initialize(project=proyecto_gcp)
        
    bbox = BBOX_CANARIAS.get(isla)
    if not bbox:
        raise ValueError(f"Isla '{isla}' no encontrada en el diccionario espacial.")
        
    region = ee.Geometry.Rectangle(bbox)
    
    hoy = datetime.now()
    hace_un_mes = hoy - timedelta(days=30)
    
    coleccion = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(hace_un_mes.strftime('%Y-%m-%d'), hoy.strftime('%Y-%m-%d'))
        .sort('system:time_start', False)
    )
    
    # Extraemos la imagen más reciente y configuramos las bandas
    imagen_mas_reciente = coleccion.first()
    fecha_captura = ee.Date(imagen_mas_reciente.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss').getInfo()
    print(f"-> Satélite localizado. Fecha de captura: {fecha_captura}")
    
    bandas = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    proyeccion_nativa = imagen_mas_reciente.select('B2').projection()
    imagen_alineada = imagen_mas_reciente.select(bandas).setDefaultProjection(proyeccion_nativa)
        
    geemap.download_ee_image(
        image=imagen_alineada,
        filename=ruta_salida,
        region=region,
        scale=10,
        crs=proyeccion_nativa.crs().getInfo()
    )
    
    return ruta_salida

# FASE 1: INICIALIZACIÓN DEL MOTOR MULTIMODAL

def cargar_motor_inferencia():
    """
    Carga en memoria los artefactos principales del modelo de inteligencia artificial.

    Returns:
        tuple: 
            - modelo (keras.Model): Red neuronal multimodal compilada y lista para predicción.
            - escalador (sklearn.preprocessing.RobustScaler): Objeto con las métricas 
              matemáticas para estandarizar las variables meteorológicas.
    """
    
    modelo = keras.models.load_model(RUTA_MODELO)    
    escalador = joblib.load(RUTA_ESCALADOR)
    
    return modelo, escalador

# FASE 2: INGESTA Y ENMASCARAMIENTO SATELITAL

def procesar_satelital(ruta):
    """
    Ingesta una imagen GeoTIFF de Sentinel-2 y genera una máscara de exclusión física.
    
    Filtra el agua mediante NDWI y las zonas urbanas o de suelo desnudo mediante NDVI,
    garantizando que el modelo solo evalúe el riesgo en celdas con cobertura vegetal real.

    Args:
        ruta (str): Ruta absoluta al archivo .tif de Sentinel-2.

    Returns:
        tuple:
            - imagen_bruta (np.ndarray): Tensor de dimensiones (filas, columnas, bandas) 
              listo para la extracción de parches.
            - mascara_valida (np.ndarray): Matriz booleana bidimensional donde True 
              indica zonas terrestres con vegetación aptas para inferencia.
            - perfil_geo (dict): Metadatos espaciales de rasterio (CRS, transform, etc.) 
              para georreferenciar el mapa de calor resultante.
    """

    print(f"Procesando imagen satelital base: {ruta}")
    
    with rasterio.open(ruta) as src:
        # Transponemos de (bandas, filas, columnas) a (filas, columnas, bandas)
        imagen_bruta = np.transpose(src.read(), (1, 2, 0))
        perfil_geo = src.profile
        
    # Extracción de bandas (B2, B3, B4, B8, B11, B12 según tu preprocesado)
    b3_green = imagen_bruta[:, :, 1]
    b4_red   = imagen_bruta[:, :, 2]
    b8_nir   = imagen_bruta[:, :, 3]
    
    # Filtro de agua (NDWI <= 0.3 es tierra)
    ndwi = (b3_green - b8_nir) / (b3_green + b8_nir + 1e-8)
    mascara_tierra = ndwi <= 0.3
    
    # Filtro urbano/roca (NDVI > 0.1 es vegetación)
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-8)
    mascara_vegetacion = ndvi > 0.1
    
    # Máscara: Solo predecimos donde hay tierra Y vegetación
    mascara_valida = mascara_tierra & mascara_vegetacion
    
    return imagen_bruta, mascara_valida, perfil_geo

# FASE 3: VENTANA DESLIZANTE (SLIDING WINDOW)

def generar_parches_espaciales(imagen_bruta, mascara_valida, tamano=64, solape=32):
    """
    Recorre la imagen satelital utilizando una ventana deslizante para extraer teselas.
    
    Aplica un filtrado espacial dinámico evaluando la matriz booleana de validez.
    Solo se extraen los parches si la cobertura de vegetación supera un umbral 
    mínimo (10%), optimizando el coste computacional al ignorar cuadrantes que 
    representan masa oceánica o suelo urbano desnudo.

    Args:
        imagen_bruta (np.ndarray): Tensor tridimensional de la imagen completa 
            (filas, columnas, bandas).
        mascara_valida (np.ndarray): Matriz booleana bidimensional donde True 
            indica píxeles aptos para la inferencia de riesgo forestal.
        tamano (int, opcional): Lado del cuadrante en píxeles. Por defecto 64.
        solape (int, opcional): Píxeles de superposición entre ventanas adyacentes 
            para garantizar continuidad geográfica espacial. Por defecto 32.

    Returns:
        tuple: 
            - np.ndarray: Matriz de tensores satelitales válidos extraídos.
            - list: Lista de tuplas (fila, columna) indicando la coordenada de origen 
              de cada parche para la posterior reconstrucción cartográfica.
            - tuple: Dimensiones originales de la imagen (filas_totales, cols_totales).
    """
    print("\n[FASE 3] Iniciando escáner de ventana deslizante...")
    filas_totales, cols_totales, _ = imagen_bruta.shape
    paso = tamano - solape
    
    parches = []
    coordenadas = []
    
    for f in range(0, filas_totales - tamano + 1, paso):
        for c in range(0, cols_totales - tamano + 1, paso):
            mascara_parche = mascara_valida[f:f+tamano, c:c+tamano]
            
            if np.mean(mascara_parche) > 0.1:
                parches.append(imagen_bruta[f:f+tamano, c:c+tamano, :])
                coordenadas.append((f, c))
                
    print(f"-> Se han recortado {len(parches)} cuadrantes válidos de {tamano}x{tamano}.")
    return np.array(parches), coordenadas, (filas_totales, cols_totales)

# FASE 4: INYECCIÓN METEO E INFERENCIA RED NEURONAL

def ejecutar_inferencia(modelo, escalador, tensores_satelite, meteo_harmonie):
    """
    Fusiona los datos satelitales con las variables meteorológicas para ejecutar 
    la predicción del modelo de Deep Learning (Late Fusion).
    
    Escala las variables meteorológicas continuas utilizando el Robust-Scaler
    pre-entrenado y las inyecta como un vector replicado para emparejarse 
    con cada parche espacial durante la propagación hacia adelante de la red neuronal.

    Args:
        modelo (keras.Model): Arquitectura neuronal multimodal cargada en memoria.
        escalador (sklearn.preprocessing.RobustScaler): Métrica matemática para 
            estandarizar temperatura, humedad y viento.
        tensores_satelite (np.ndarray): Lote (batch) de parches satelitales.
        meteo_harmonie (list o np.ndarray): Vector 1D con las condiciones 
            termodinámicas [Temperatura_2m, Humedad_Relativa, Velocidad_Viento].

    Returns:
        np.ndarray: Vector columna con las probabilidades continuas de riesgo 
            de incendio (escala 0.0 - 1.0) para cada cuadrante procesado.
    """
    print("\n[FASE 4] Cruzando datos con AEMET e iniciando inferencia...")
    
    meteo_escalada = escalador.transform([meteo_harmonie])
    
    num_parches = tensores_satelite.shape[0]
    meteo_masiva = np.repeat(meteo_escalada, num_parches, axis=0)
    
    predicciones = modelo.predict([tensores_satelite, meteo_masiva], batch_size=32)
    return predicciones

# FASE 5: RECONSTRUCCIÓN Y EXPORTACIÓN CARTOGRÁFICA

def exportar_mapa_calor(predicciones, coordenadas, dimensiones_base, perfil_geo, ruta_salida, tamano=64):
    """
    Construye el mapa de calor promediando los valores de riesgo en las zonas 
    de solape y exporta el resultado como un archivo GeoTIFF georreferenciado.

    Args:
        predicciones (np.ndarray): Vector con los valores de riesgo predichos por la IA.
        coordenadas (list): Lista de tuplas (fila, columna) con el origen de cada parche.
        dimensiones_base (tuple): Tamaño original de la imagen satelital (filas, columnas).
        perfil_geo (dict): Metadatos espaciales heredados de la imagen Sentinel-2 original.
        ruta_salida (str): Ruta absoluta donde se guardará el mapa de calor resultante.
        tamano (int, opcional): Lado del cuadrante en píxeles. Por defecto 64.

    Returns:
        np.ndarray: Matriz bidimensional final con el riesgo forestal (0 a 1) por píxel.
    """
    print("\n[FASE 5] Promediando solapes y generando GeoTIFF final...")
    filas, columnas = dimensiones_base
    
    mapa_riesgo = np.zeros((filas, columnas), dtype=np.float32)
    mapa_conteo = np.zeros((filas, columnas), dtype=np.float32)
    
    for pred, (f, c) in zip(predicciones, coordenadas):
        valor_riesgo = pred[0]
        mapa_riesgo[f:f+tamano, c:c+tamano] += valor_riesgo
        mapa_conteo[f:f+tamano, c:c+tamano] += 1
        
    with np.errstate(invalid='ignore', divide='ignore'):
        mapa_final = np.divide(mapa_riesgo, mapa_conteo)
        
    # Aplicamos la máscara válida: todo lo que no sea tierra/vegetación real se vuelve NaN (transparente)
    mapa_final[~mascara_valida] = np.nan
    mapa_final = np.nan_to_num(mapa_final, nan=0.0)
        
    perfil_geo.update(
        count=1, 
        dtype=rasterio.float32, 
        nodata=0,
        compress='lzw'
    )
    
    with rasterio.open(ruta_salida, 'w', **perfil_geo) as dest:
        dest.write(mapa_final, 1)
        
    print(f"-> Mapa de calor georreferenciado guardado en: {ruta_salida}")
    return mapa_final

# INTERFAZ DE EJECUCION
if __name__ == "__main__":

    ISLA_OBJETIVO = "Canarias"
    PROYECTO_GCP = "tfm-bbdd-499813"
    
    ruta_raw = os.path.join(BASE_DIR, 'data', 'raw', f'satelite_{ISLA_OBJETIVO.replace(" ", "_")}.tif')
    ruta_export = os.path.join(BASE_DIR, 'data', 'processed', f'riesgo_{ISLA_OBJETIVO.replace(" ", "_")}.tif')
    os.makedirs(os.path.dirname(ruta_export), exist_ok=True)
    os.makedirs(os.path.dirname(ruta_raw), exist_ok=True)
    
    try:
        archivo_descargado = obtener_ultima_imagen_gee(ISLA_OBJETIVO, ruta_raw, PROYECTO_GCP)
        
        if archivo_descargado and os.path.exists(archivo_descargado):
            modelo_ia, scaler_meteo = cargar_motor_inferencia()
            img, mascara, perfil = procesar_satelital(archivo_descargado)
            tensores, coords, dimensiones = generar_parches_espaciales(img, mascara)
            
            if len(tensores) > 0:
                meteo_hoy = [36.0, 12.0, 30.0] 
                riesgos = ejecutar_inferencia(modelo_ia, scaler_meteo, tensores, meteo_hoy)
                exportar_mapa_calor(riesgos, coords, dimensiones, mascara, perfil, ruta_export)
                print(f"\n[¡PIPELINE COMPLETADO!] Riesgo máximo detectado: {np.max(riesgos)*100:.2f}%")
            else:
                print("\n[!] El filtro descartó toda la imagen (sin vegetación válida).")
                
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] El sistema se detuvo: {e}")