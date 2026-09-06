import os
import numpy as np
import joblib
import rasterio
from tensorflow import keras

# CONFIGURACIÓN DE RUTAS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, 'models', 'modelo_late_fusion_definitivo.keras')
RUTA_ESCALADOR = os.path.join(BASE_DIR, 'models', 'robust_scaler_meteo.pkl')


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
    
    # 1. Filtro de Agua (NDWI <= 0.3 es Tierra)
    ndwi = (b3_green - b8_nir) / (b3_green + b8_nir + 1e-8)
    mascara_tierra = ndwi <= 0.3
    
    # 2. Filtro Urbano/Roca (NDVI > 0.1 es Vegetación)
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-8)
    mascara_vegetacion = ndvi > 0.1
    
    # Máscara maestra: Solo predecimos donde hay tierra Y vegetación
    mascara_valida = mascara_tierra & mascara_vegetacion
    
    return imagen_bruta, mascara_valida, perfil_geo

# Bloque de prueba del Pipeline
if __name__ == "__main__":
    print("--- INICIANDO TEST DEL PIPELINE ---")
    
    # Test Fase 1
    try:
        modelo_ia, scaler_meteo = cargar_motor_inferencia()
        print("ÉXITO: Fase 1 completada. Motor cargado.\n")
    except Exception as e:
        print(f"ERROR EN FASE 1: {e}\n")
        
    # Test Fase 2
    # Define la ruta a una imagen satelital de prueba (debe ser un .tif real)
    ruta_imagen_prueba = os.path.join(BASE_DIR, 'data', 'raw', 'satelite_prueba.tif')
    
    if os.path.exists(ruta_imagen_prueba):
        try:
            img, mascara, perfil = procesar_satelital(ruta_imagen_prueba)
            print("ÉXITO: Fase 2 completada.")
            print(f"Dimensiones del tensor extraído: {img.shape}")
            print(f"Total de píxeles en la imagen: {img.shape[0] * img.shape[1]}")
            print(f"Píxeles de vegetación válidos: {np.sum(mascara)}")
        except Exception as e:
            print(f"ERROR EN FASE 2: {e}")
    else:
        print(f"AVISO: Para probar la Fase 2, guarda una imagen GeoTIFF en:\n{ruta_imagen_prueba}")