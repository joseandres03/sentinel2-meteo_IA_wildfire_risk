import os
import ee
import geemap
import rasterio
from datetime import datetime, timedelta

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

def verificar_descarga(ruta):
    """Comprueba físicamente el archivo GeoTIFF generado para garantizar su integridad."""
    print("\nLeyendo el archivo GeoTIFF descargado...")
    with rasterio.open(ruta) as src:
        print(f"-> Dimensiones: {src.width} columnas x {src.height} filas")
        print(f"-> Cantidad de bandas: {src.count}")
        print(f"-> Proyección (CRS): {src.crs}")

if __name__ == "__main__":
    PROYECTO_GCP = "tfm-bbdd-499813"
    
    try:
        isla_elegida = seleccionar_isla()
        ruta_tif = descargar_satelite(isla_elegida, PROYECTO_GCP)
        verificar_descarga(ruta_tif)
                
    except Exception as e:
        print(f"\n[ERROR] Algo ha fallado: {e}")