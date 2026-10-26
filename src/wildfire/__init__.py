"""Utilidades del proyecto de susceptibilidad a incendios forestales en Canarias.

Módulos:
    indices     -> índices espectrales, máscaras de vegetación e índice heurístico de la versión 1.
    etiquetas   -> agrupación de focos de incendio en eventos y muestreo de negativos.
    validacion  -> particiones espaciales/temporales, métricas de clasificación y calibración.
    meteo       -> humedad relativa y punto de rocío.
    meteo_relieve -> meteorología en cualquier punto según el relieve (perfil vertical e inversión).
"""

BANDAS_S2 = ("B2", "B3", "B4", "B8", "B11", "B12")

# Cajas envolventes [lon_min, lat_min, lon_max, lat_max] de cada isla (mismas que el pipeline).
BBOX_ISLAS = {
    "La Gomera": (-17.37, 28.01, -17.09, 28.23),
    "Tenerife": (-16.94, 27.97, -16.11, 28.59),
    "Gran Canaria": (-15.83, 27.70, -15.36, 28.18),
    "La Palma": (-18.00, 28.43, -17.72, 28.85),
    "El Hierro": (-18.17, 27.62, -17.88, 27.86),
    "Lanzarote": (-13.91, 28.83, -13.33, 29.26),
    "Fuerteventura": (-14.52, 28.01, -13.82, 28.76),
}
