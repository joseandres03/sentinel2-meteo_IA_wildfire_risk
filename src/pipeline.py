import os
import json
import base64
import ee
import numpy as np
import joblib
import geemap
import rasterio
import rasterio.enums
import requests
import pyproj
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    print("Modelo probabilistico de riesgo de incendio con Deep Learning")
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
    print(f"-> Última imagen obtenida el: {fecha_captura}")
    
    imagen_mosaico = coleccion.sort('system:time_start', True).mosaic()
    imagen_export = imagen_mosaico.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    
    ruta_salida = os.path.join(BASE_DIR, 'data', 'raw', f'satelite_{isla.replace(" ", "_")}.tif')
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    print("-> Descargando GeoTIFF...")
    geemap.download_ee_image(
        image=imagen_export,
        filename=ruta_salida,
        region=region,
        scale=10,
        crs='EPSG:32628'
    )
    
    return ruta_salida, fecha_captura

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
    print(f"\n Descargando datos meteorológicos del HARMONIE-AROME e interpolando por la geografía...")
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
    print(f"\nProcesando reflectancia y delimitando el litoral...")
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

def extraer_parches_solapados(imagen_bruta, mascara_vegetacion, perfil, tamano=64, solape=4):
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
    print(f"\nGenerando los parches a partir de la imagen ({tamano}x{tamano} con solape de {solape}px)...")
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
    print("\nCalculando la probabilidad de riesgo mediante el modelo...")
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
    print("Generando cartografía...")
    filas, columnas = dimensiones_base
    mapa_riesgo = np.zeros((filas, columnas), dtype=np.float32)
    mapa_conteo = np.zeros((filas, columnas), dtype=np.float32)
    
    # Acumulamos el riesgo sumando capas superpuestas
    for pred, (f, c) in zip(predicciones, coordenadas):
        # DETECCIÓN DINÁMICA: Si es un array de Keras saca el índice 0, si es temperatura usa el número tal cual
        valor_limpio = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
        
        mapa_riesgo[f:f+tamano, c:c+tamano] += valor_limpio
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

def obtener_cmap_personalizado():
    """Genera la escala térmica a medida según los umbrales operativos."""
    nodos = [
        (0.00, '#228B22'),  # Verde bosque oscuro (Riesgo nulo)
        (0.30, '#ADFF2F'),  # Verde amarillento (Transición)
        (0.45, '#FFA500'),  # Naranja (Riesgo moderado)
        (0.60, '#FF0000'),  # Rojo (Riesgo alto)
        (0.85, '#800080'),  # Morado (Riesgo extremo)
        (1.00, '#F8E6FF')   # Violeta pálido (Peligro máximo)
    ]
    cmap = LinearSegmentedColormap.from_list("RiesgoCanarias", nodos)
    cmap.set_under('black', alpha=0.0) 
    cmap.set_bad('black', alpha=0.0)
    return cmap

def exportar_dashboard_png(ruta_tif, isla, ruta_png):
    print(f"\nRenderizando cartografía...")
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
        
        # Bloqueo espacial estricto para evitar desfases de la línea de costa
        ax.set_xlim(limites.left, limites.right)
        ax.set_ylim(limites.bottom, limites.top)
    
        plt.colorbar(im, ax=ax, label="Probabilidad de riesgo de incendio (0.0 - 1.0)", shrink=0.7)
        ax.set_title(f"Mapa de riesgo - {isla}", fontsize=15, fontweight='bold')
        ax.axis('off')
        
        plt.savefig(ruta_png, bbox_inches='tight', facecolor='white')
        plt.close()

def exportar_visor_interactivo(ruta_tif_riesgo, ruta_tif_temp, ruta_raw, isla, dir_salida, fecha_sat):
    print(f"\nConstruyendo visor web interactivo para {isla}...")
    
    ruta_base_png = os.path.join(dir_salida, f"base_rgb_{isla.replace(' ', '_')}.png")
    ruta_riesgo_png = os.path.join(dir_salida, f"capa_riesgo_{isla.replace(' ', '_')}.png")
    ruta_leyenda = os.path.join(dir_salida, f"leyenda_{isla.replace(' ', '_')}.png")
    ruta_html = os.path.join(dir_salida, f"visor_interactivo_{isla.replace(' ', '_')}.html")
    
    frontera = ox.geocode_to_gdf(f"{isla}, Canarias, España")
    cmap_riesgo = obtener_cmap_personalizado()
    
    with rasterio.open(ruta_raw) as src_raw:
        h_orig, w_orig = src_raw.height, src_raw.width
        factor = max(1, max(h_orig, w_orig) // 2000)
        h_new, w_new = h_orig // factor, w_orig // factor
        
        b_blue = src_raw.read(1, out_shape=(h_new, w_new), resampling=rasterio.enums.Resampling.bilinear)
        b_green = src_raw.read(2, out_shape=(h_new, w_new), resampling=rasterio.enums.Resampling.bilinear)
        b_red = src_raw.read(3, out_shape=(h_new, w_new), resampling=rasterio.enums.Resampling.bilinear)
        
        rgb = np.dstack((b_red, b_green, b_blue))
        rgb = np.clip(rgb / 3000.0, 0, 1) 
        
        limites = src_raw.bounds
        extension_utm = [limites.left, limites.right, limites.bottom, limites.top]
        frontera_utm = frontera.to_crs(src_raw.crs)
        
        # Base de satélite
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.set_position([0, 0, 1, 1])
        ax.set_facecolor('white')
        ax.imshow(rgb, extent=extension_utm)
        frontera_utm.boundary.plot(ax=ax, color='black', linewidth=1.5)
        ax.set_xlim(limites.left, limites.right)
        ax.set_ylim(limites.bottom, limites.top)
        ax.axis('off')
        plt.savefig(ruta_base_png, dpi=200, facecolor='white')
        plt.close()
        
    with rasterio.open(ruta_tif_riesgo) as src_riesgo:
        mapa_riesgo = src_riesgo.read(1, out_shape=(h_new, w_new), resampling=rasterio.enums.Resampling.nearest)
        
        # Capa de riesgo
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.set_position([0, 0, 1, 1])
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.imshow(mapa_riesgo, cmap=cmap_riesgo, vmin=0.01, vmax=1.0, extent=extension_utm)
        ax.set_xlim(limites.left, limites.right)
        ax.set_ylim(limites.bottom, limites.top)
        ax.axis('off')
        plt.savefig(ruta_riesgo_png, dpi=200, transparent=True)
        plt.close()

    # leyenda
    fig_leg, ax_leg = plt.subplots(figsize=(8, 1), dpi=150)
    fig_leg.subplots_adjust(bottom=0.5)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap_riesgo),
                      cax=ax_leg, orientation='horizontal')
    cb.set_label('Riesgo de incendio (0.0 a 1.0)', fontsize=12, fontweight='bold')
    plt.savefig(ruta_leyenda, bbox_inches='tight', transparent=True)
    plt.close()

    # Datos (JSON) para las ventanas flotantes interactivas
    with rasterio.open(ruta_tif_riesgo) as src_r, rasterio.open(ruta_tif_temp) as src_t:
        factor_json = max(1, max(h_orig, w_orig) // 250) 
        h_j, w_j = h_orig // factor_json, w_orig // factor_json
        
        arr_r = src_r.read(1, out_shape=(h_j, w_j), resampling=rasterio.enums.Resampling.nearest)
        arr_t = src_t.read(1, out_shape=(h_j, w_j), resampling=rasterio.enums.Resampling.nearest)
        
        # Formateamos valores inválidos para que JS los detecte como espacios vacíos
        json_r = json.dumps(np.nan_to_num(arr_r, nan=-1.0).round(2).tolist())
        json_t = json.dumps(np.nan_to_num(arr_t, nan=-99.0).round(1).tolist())


    # Base64
    with open(ruta_base_png, "rb") as f: base64_base = base64.b64encode(f.read()).decode('utf-8')
    with open(ruta_riesgo_png, "rb") as f: base64_riesgo = base64.b64encode(f.read()).decode('utf-8')
    with open(ruta_leyenda, "rb") as f: base64_ley = base64.b64encode(f.read()).decode('utf-8')
        
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Visor riesgo de incendio para {isla}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; text-align: center; padding: 20px; }}
            .container {{ display: inline-block; position: relative; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); background: white; border-radius: 6px; overflow: hidden; max-width: 900px; cursor: crosshair; }}
            .map-layer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
            .base-layer {{ position: relative; display: block; width: 100%; height: auto; }}
            .controls {{ margin: 20px auto; padding: 15px; background: white; display: inline-block; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            input[type=range] {{ width: 300px; vertical-align: middle; margin: 0 15px; }}
            .legend {{ margin-top: 15px; max-width: 400px; height: auto; }}
            #tooltip {{ position: absolute; background: rgba(0,0,0,0.85); color: #fff; padding: 8px 12px; border-radius: 5px; font-size: 14px; display: none; z-index: 100; pointer-events: none; text-align: left; box-shadow: 0px 4px 6px rgba(0,0,0,0.3); }}
        </style>
    </head>
    <body>
        <div id="tooltip"></div>
        <h2> Riesgo y cobertura terrestre para {isla}</h2>
        <p style="color: #555; font-size: 15px; margin-top: -10px; margin-bottom: 20px;">
            <strong>🛰️ Fecha Satélite:</strong> {fecha_sat} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>⏱️ Día de la previsión:</strong> {fecha_calc}
        </p>
        
        <div class="controls">
            <label><strong>Transparencia del mapa de riesgo:</strong></label>
            Oculto <input type="range" id="opacitySlider" min="0" max="100" value="75"> Visible
            <br>
            <img src="data:image/png;base64,{base64_ley}" class="legend" alt="Leyenda de Riesgo">
        </div>
        <br>
        
        <div class="container" id="mapContainer">
            <img src="data:image/png;base64,{base64_base}" class="base-layer">
            <img src="data:image/png;base64,{base64_riesgo}" class="map-layer" id="riskLayer" style="opacity: 0.75;">
        </div>
        
        <script>
            const slider = document.getElementById('opacitySlider');
            const riskLayer = document.getElementById('riskLayer');
            const container = document.getElementById('mapContainer');
            const tooltip = document.getElementById('tooltip');
            
            const riskData = {json_r};
            const tempData = {json_t};
            const gridH = {h_j};
            const gridW = {w_j};

            slider.addEventListener('input', function() {{
                riskLayer.style.opacity = this.value / 100;
            }});
            
            container.addEventListener('mousemove', function(e) {{
                const rect = container.getBoundingClientRect();
                let relX = (e.clientX - rect.left) / rect.width;
                let relY = (e.clientY - rect.top) / rect.height;
                
                let gridY = Math.floor(relY * gridH);
                let gridX = Math.floor(relX * gridW);
                
                if (gridY >= 0 && gridY < gridH && gridX >= 0 && gridX < gridW) {{
                    let r = riskData[gridY][gridX];
                    let t = tempData[gridY][gridX];
                    
                    if (r >= 0) {{
                        tooltip.style.display = 'block';
                        tooltip.style.left = (e.pageX + 15) + 'px';
                        tooltip.style.top = (e.pageY + 15) + 'px';
                        tooltip.innerHTML = `<strong>Riesgo:</strong> ${{(r * 100).toFixed(1)}}%<br><strong>Temperatura:</strong> ${{t.toFixed(1)}} °C`;
                    }} else {{
                        tooltip.style.display = 'none';
                    }}
                }}
            }});
            
            container.addEventListener('mouseleave', function() {{
                tooltip.style.display = 'none';
            }});
        </script>
    </body>
    </html>
    """
    with open(ruta_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"-> Visor web interactivo generado con éxito en: {ruta_html}")

if __name__ == "__main__":
    try:
        isla = seleccionar_isla()
        
        # Extraemos también la fecha del satélite
        ruta_raw, fecha_satelite = descargar_satelite(isla)
        print("\nConsolidando cartografía matricial promediada...")
        img_bruta, m_tierra, m_vegetacion, perfil = calcular_mascaras_fisicas(ruta_raw)
        
        tensores, coords, coords_utm, dim_base = extraer_parches_solapados(img_bruta, m_vegetacion, perfil, tamano=64, solape=8)
        
        if len(tensores) > 0:
            meteo_matriz = descargar_meteo_malla(isla, coords_utm)
            riesgos = predecir_riesgo(tensores, meteo_matriz)
            
            temperaturas = meteo_matriz[:, 0]
            
            dir_procesados = os.path.join(BASE_DIR, 'data', 'processed')
            os.makedirs(dir_procesados, exist_ok=True)
            
            ruta_export_tif = os.path.join(dir_procesados, f'riesgo_{isla.replace(" ", "_")}.tif')
            ruta_export_temp = os.path.join(dir_procesados, f'temp_{isla.replace(" ", "_")}.tif')
            ruta_export_png = os.path.join(dir_procesados, f'mapa_{isla.replace(" ", "_")}.png')
            
            reconstruir_mapa_calor(riesgos, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_tif)
            
            reconstruir_mapa_calor(temperaturas, coords, dim_base, m_tierra, m_vegetacion, perfil, ruta_export_temp)
            
            exportar_dashboard_png(ruta_export_tif, isla, ruta_export_png)
            
            exportar_visor_interactivo(ruta_export_tif, ruta_export_temp, ruta_raw, isla, dir_procesados, fecha_satelite)
            
            print(f"\n Finalizado con éxito: {np.max(riesgos)*100:.1f}%")
        else:
            print("\n Algo ha fallado")
            
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {e}")