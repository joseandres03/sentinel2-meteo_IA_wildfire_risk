# 🔥 Susceptibilidad a incendios forestales en Canarias con Sentinel-2 y Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-34A853.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)
![Leaflet](https://img.shields.io/badge/Frontend-Leaflet-199900.svg?logo=leaflet)

> **Prototipo *end-to-end* de susceptibilidad a incendios forestales en Canarias**: ingesta de imágenes Sentinel-2 y datos meteorológicos, red neuronal multimodal (*late fusion*) y visor cartográfico actualizado a diario con GitHub Actions.

Trabajo de Fin de Máster — Máster en Big Data, Data Science e Inteligencia Artificial (Universidad Complutense de Madrid).

## 🌍 Visor

El pipeline se ejecuta automáticamente cada día y publica el mapa en:

👉 **https://canarywildfirerisk.es/**

> ⚠️ El mapa es un **índice de susceptibilidad experimental**, no una alerta oficial. Consulta la sección [Validación y limitaciones](#-validación-y-limitaciones).

---

## 📖 Descripción

El proyecto combina dos fuentes de información para estimar, en cada zona con vegetación del archipiélago, un índice de riesgo entre 0 y 1:

1. **Estado del combustible (Sentinel-2):** parches de 64×64 píxeles (640×640 m) con 6 bandas (B2, B3, B4, B8, B11, B12). Se enmascaran el agua (NDWI) y las zonas sin vegetación (NDVI) y se estima el estrés hídrico de la vegetación (NDMI).
2. **Condiciones meteorológicas:** temperatura máxima, humedad relativa media y velocidad del viento. En el entrenamiento proceden de 58 estaciones de AEMET (2017–2026); en producción, del pronóstico de Open-Meteo, interpolado sobre el territorio.

### 🧠 Modelo (*late fusion*)

- **Rama CNN:** procesa el parche multiespectral de Sentinel-2.
- **Rama MLP:** procesa las variables meteorológicas.
- Ambas ramas se concatenan en un bloque denso con salida lineal (índice de 0 a 1).

**Variable objetivo.** El índice que aprende la red se construye en el `Cuaderno2` como `0,35 × riesgo meteorológico + 0,65 × riesgo de combustible`. El riesgo meteorológico se calcula con umbrales de temperatura, humedad y viento, y el de combustible con el NDMI medio de la vegetación válida del parche.

---

## ⚙️ Pipeline diario (GitHub Actions)

El workflow [`.github/workflows/actualizar.yml`](.github/workflows/actualizar.yml) se ejecuta cada día y:

1. Prepara un servidor Ubuntu con las dependencias.
2. Descarga el mosaico Sentinel-2 más reciente (Google Earth Engine) y el pronóstico meteorológico a 24 h (Open-Meteo).
3. Carga el escalador (`models/robust_scaler_meteo.pkl`) y el modelo (`models/modelo_late_fusion_definitivo.keras`).
4. Ejecuta la inferencia con ventana deslizante (solape de 8 px) sobre las zonas con vegetación.
5. Reproyecta con Rasterio, exporta las capas y actualiza el visor estático (HTML, CSS y Leaflet.js) en `docs/`.

---

## 🔬 Validación y limitaciones

El trabajo se presentó como TFM y el tribunal señaló puntos de mejora que se recogen aquí con transparencia. **El estado actual corresponde a un prototipo de susceptibilidad, no a un sistema de alerta validado.**

| Limitación | Detalle | Mejora prevista |
|---|---|---|
| La variable objetivo no procede de incendios observados | La red aprende a aproximar un índice calculado con fórmulas a partir de sus propias entradas (meteorología y bandas B8/B11). El MAE de 0,0228 en test mide lo bien que reproduce ese índice, **no su capacidad de anticipar incendios reales**. | Entrenar con incendios históricos observados (áreas quemadas y focos activos) como etiqueta. |
| Posible fuga espacial y temporal | La partición aleatoria 80/20 pone las mismas estaciones (mismo parche de 640 m) en entrenamiento y test. | Validación por bloques espaciales (islas o zonas) y separación temporal (entrenar con años anteriores y evaluar con años posteriores). |
| Sin contraste externo | No se ha comparado el mapa con incendios ocurridos después del entrenamiento. | Evaluar con los incendios de las campañas más recientes. |
| Métricas y calibración | Faltan sensibilidad, precisión, falsos positivos en zonas de riesgo alto y calibración de probabilidades. | Curvas ROC/PR, diagrama de fiabilidad y métricas por umbral de riesgo. |
| Cambio de dominio | Se entrena con parches centrados en estaciones de AEMET y datos observados, y se predice sobre todo el territorio forestal con pronósticos. | Muestrear puntos en todo el territorio con vegetación. |
| Nubes | La extracción de entrenamiento no aplica máscara de nubes; en producción solo se enmascaran para la visualización. | Máscara con la capa SCL de Sentinel-2 y composiciones sin nubes. |
| Robustez del pipeline | No hay pruebas automáticas de tiempos de ejecución ni del comportamiento cuando falla una fuente de datos. | Tests, registro de tiempos y alternativas cuando falta una fuente. |

Referencia de comparación recomendada por el tribunal: [mesogeos](https://github.com/paulinebecker2002/mesogeos).

---

## 📂 Estructura del repositorio

```text
├── docs/               # Visor web (HTML, CSS, JS) y capas publicadas
├── models/             # Modelo (.keras) y escalador meteorológico (.pkl)
├── notebooks/
│   ├── Cuaderno1.ipynb # Ingesta histórica de AEMET y Sentinel-2 (dataset bruto)
│   ├── Cuaderno2.ipynb # Limpieza, imputación KNN, índices espectrales y variable objetivo
│   └── Cuaderno3.ipynb # Entrenamiento y evaluación del modelo late fusion
├── pdf/                # Memoria del TFM
├── src/
│   └── pipeline.py     # Pipeline de producción ejecutado por GitHub Actions
├── .github/workflows/  # Workflow de actualización diaria
└── requirements.txt    # Dependencias
```

## 🔑 Credenciales

Ninguna credencial debe escribirse en el código:

- **AEMET OpenData** (`Cuaderno1`): guarda la clave como secreto de Colab con el nombre `AEMET_API_KEY` (icono de la llave en la barra lateral) o como variable de entorno.
- **Google Earth Engine** (GitHub Actions): secreto del repositorio `GEE_CREDENTIALS`.

## 👤 Autor

**José Andrés Ramos Mendoza** — [LinkedIn](https://www.linkedin.com/in/jose-andres-ramos-mendoza-646542153/) · [GitHub](https://github.com/joseandres03)
