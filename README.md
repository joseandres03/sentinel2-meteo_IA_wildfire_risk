\# 🔥 IA \& MLOps: Prevención de incendios forestales en Canarias



\[!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

\[!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)

\[!\[EarthEngine](https://img.shields.io/badge/Google%20Earth%20Engine-API-34A853.svg)](https://earthengine.google.com/)

\[!\[MLOps](https://img.shields.io/badge/CI%2FCD-GitHub\_Actions-2088FF.svg)](https://github.com/features/actions)

\[!\[Leaflet](https://img.shields.io/badge/Frontend-Leaflet-199900.svg?logo=leaflet)](https://leafletjs.com/)



> \*\*Sistema predictivo \*End-to-End\* de Riesgo de Incendio Forestal en Canarias mediante Deep Learning Multimodal (Late Fusion) y Arquitectura MLOps.\*\*



\## 🌍 Visor Táctico en Producción

El modelo se ejecuta de forma autónoma diariamente, extrayendo telemetría satelital y meteorológica en tiempo real. Puedes consultar el mapa de riesgo interactivo actualizado aquí:  

👉 \*\*https://canarywildfirerisk.es/\*\*



\---



\## 📖 Descripción del Proyecto

Este repositorio aloja la infraestructura completa de un Trabajo de Fin de Máster (TFM) orientado a la coordinación táctica de emergencias. El proyecto supera los enfoques de Machine Learning tradicionales mediante la ingesta dinámica de datos espaciales y meteorológicos, procesados a través de una red neuronal de doble rama construida desde cero.



\### 🧠 Arquitectura del modelo (Late Fusion)

La red neuronal aprende de dos naturalezas de datos diametralmente opuestas:

1\. \*\*Rama de Visión Espacial (CNN):\*\* Extrae la orografía y el estrés hídrico de la biomasa forestal procesando tensores de 64x64x6 (6 bandas multiespectrales de Sentinel-2), previamente filtrados mediante álgebra de mapas (NDWI para enmascarar océano y embalses, y NDVI para descartar núcleos urbanos/roca).

2\. \*\*Rama meteorológica (MLP):\*\* Procesa el microclima exacto interpolado para ese punto geográfico (Temperatura, Humedad Relativa y Viento), alimentado por el modelo HARMONIE-AROME (AEMET) y Open-Meteo.



Ambas ramas convergen en un bloque denso de salida lineal, emitiendo una probabilidad continua de ignición validada empíricamente con un Error Absoluto Medio (MAE) del 2.28%.



\---



\## ⚙️ Arquitectura MLOps (CI/CD Pipeline)

El ecosistema no requiere intervención manual. Está diseñado para operar en producción mediante \*\*GitHub Actions\*\* (`.github/workflows/main.yml`), ejecutando diariamente el siguiente ciclo:

1\. Instancia un servidor Ubuntu.

2\. Sincroniza la telemetría satelital reciente (Google Earth Engine) y la predicción meteorológica a 24h vista (Open-Meteo).

3\. Carga el preprocesador matemático (`robust\_scaler\_meteo.pkl`) y el modelo predictivo (`modelo\_late\_fusion\_definitivo.keras`).

4\. Ejecuta la inferencia mediante un algoritmo de ventana deslizante (con solape de 8px para suavizar transiciones topológicas) sobre toda la geografía insular con cobertura forestal.

5\. Reconstruye la cartografía promediando los solapes, exporta las proyecciones en GeoTIFF y actualiza dinámicamente el \*Frontend\* estático desarrollado con HTML, CSS y Leaflet.js (`config.js`).



\---



\## 📂 Estructura del Repositorio



```text

├── data/               # Directorio para los archivos zip con los datasets

├── docs/               # Frontend operativo (HTML, CSS, JS) y cartografía desplegada

├── models/             # Modelo predictivo (.keras) y preprocesador climático (.pkl)

├── notebooks/          # Flujo de investigación, entrenamiento y Data Engineering

│   ├── Cuaderno1.ipynb # Ingesta histórica AEMET y Earth Engine (Creación del dataset bruto)

│   ├── Cuaderno2.ipynb # Imputación KNN, filtro espectral y escalado robusto (dataset final)

│   └── Cuaderno3.ipynb # Construcción, entrenamiento y validación del modelo Late Fusion

├── src/                # Lógica del orquestador backend de producción

│   └── pipeline.py     # Script principal ejecutado por GitHub Actions

├── .github/workflows/  # Archivo YAML de configuración de CI/CD (automatización)

├── requirements.txt    # Dependencias de Python

└── README.md           # Documentación principal

