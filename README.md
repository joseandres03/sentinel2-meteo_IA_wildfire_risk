# 🔥 Susceptibilidad a incendios forestales en Canarias con Sentinel-2 y Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-34A853.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)
![Leaflet](https://img.shields.io/badge/Frontend-Leaflet-199900.svg?logo=leaflet)

> **Sistema *end-to-end* de susceptibilidad a incendios forestales en Canarias**: ingesta de imágenes Sentinel-2 y datos meteorológicos de AEMET, red neuronal multimodal (*late fusion*) y visor cartográfico que se actualiza cada día con GitHub Actions.

Proyecto personal en desarrollo activo. La versión 1 está en producción, y la versión 2 aprende de incendios observados y se valida con datos que el modelo no ha visto (ver [Hoja de ruta](#-hoja-de-ruta)).

## 🌍 Visor

El pipeline se ejecuta automáticamente cada día y publica el mapa en:

👉 **https://canarywildfirerisk.es/**

> ⚠️ El mapa muestra un **índice de susceptibilidad experimental**, no una alerta oficial. Ver [Limitaciones de la versión 1](#-limitaciones-de-la-versión-1).

---

## 📖 Versión 1 (en producción)

El modelo combina dos fuentes de información para estimar, en cada zona con vegetación del archipiélago, un índice de riesgo entre 0 y 1:

1. **Estado del combustible (Sentinel-2):** parches de 64×64 píxeles (640×640 m) con 6 bandas (B2, B3, B4, B8, B11, B12). Se enmascaran el agua (NDWI) y las zonas sin vegetación (NDVI), y se estima el estrés hídrico de la vegetación (NDMI).
2. **Condiciones meteorológicas:** temperatura máxima, humedad relativa media y velocidad del viento. En el entrenamiento proceden de estaciones de AEMET (2017–2026); en producción, del pronóstico HARMONIE-AROME (vía Open-Meteo), interpolado sobre el territorio.

### 🧠 Modelo (*late fusion*)

- **Rama CNN:** procesa el parche multiespectral de Sentinel-2.
- **Rama MLP:** procesa las variables meteorológicas.
- Ambas ramas se concatenan en un bloque denso con salida lineal (índice de 0 a 1).

**Variable objetivo.** El índice que aprende la red se construye en el `Cuaderno2` como `0,35 × riesgo meteorológico + 0,65 × riesgo de combustible`. El riesgo meteorológico se calcula con umbrales de temperatura, humedad y viento, y el de combustible con el NDMI medio de la vegetación válida del parche.

### ⚙️ Pipeline diario (GitHub Actions)

El workflow [`.github/workflows/actualizar.yml`](.github/workflows/actualizar.yml) se ejecuta cada día y:

1. Prepara un servidor Ubuntu con las dependencias.
2. Compone la imagen Sentinel-2 más reciente sin nubes (máscara SCL, Google Earth Engine) y descarga el pronóstico meteorológico a 24 h.
3. Carga el escalador (`models/robust_scaler_meteo.pkl`) y el modelo (`models/modelo_late_fusion_definitivo.keras`).
4. Ejecuta la inferencia con ventana deslizante (solape de 8 px) sobre las zonas con vegetación.
5. Reproyecta con Rasterio, exporta las capas y actualiza el visor estático (HTML, CSS y Leaflet.js) en `docs/`.
6. Registra el estado de cada isla, los tiempos y los errores en `docs/estado_pipeline.json`. Si una isla falla, se reintenta y, si no se recupera, se conserva su último mapa marcado como desactualizado.

### 🔬 Limitaciones de la versión 1

**La versión 1 es un prototipo de susceptibilidad, no un sistema de alerta validado.** Estas limitaciones motivan la versión 2:

| Limitación | Detalle |
|---|---|
| La variable objetivo no procede de incendios observados | La red aprende a aproximar un índice calculado con fórmulas a partir de sus propias entradas. El MAE de 0,0228 en test mide lo bien que reproduce ese índice, **no su capacidad de anticipar incendios reales**. |
| Posible fuga espacial y temporal | La partición aleatoria 80/20 pone las mismas estaciones (mismo parche de 640 m) en entrenamiento y en test. |
| Sin contraste con incendios posteriores | No se ha comparado el mapa con incendios ocurridos después del entrenamiento. |
| Métricas y calibración | Faltan sensibilidad, precisión, falsos positivos en zonas de riesgo alto y calibración de probabilidades. |
| Cobertura de estaciones | El catálogo original perdía 20 estaciones (AEMET escribe la provincia de Santa Cruz de Tenerife de dos formas): Tenerife quedaba con 2 estaciones y ninguna por encima de 900 m. *Corregido en el catálogo completo.* |
| Cambio de dominio | Se entrena con parches centrados en estaciones y datos observados, y se predice sobre todo el territorio con pronósticos. |

---

## 🧭 Hoja de ruta

**Hecho**
- [x] Composición Sentinel-2 sin nubes (máscara SCL) en producción.
- [x] Corrección de unidades del viento: el modelo se entrenó con m/s (AEMET) y el pronóstico llegaba en km/h.
- [x] Pipeline tolerante a fallos y con registro de ejecución (`estado_pipeline.json`).
- [x] Catálogo completo de estaciones de AEMET en Canarias (77; antes 58). Incorpora Izaña (2.369 m), Las Cañadas del Teide (2.150 m) y el Roque de los Muchachos (2.223 m).
- [x] Descarga de la meteorología diaria completa de todas las estaciones (`Cuaderno1`, apartado 5b).

**En curso**
- [ ] **Meteorología en cualquier punto teniendo en cuenta el relieve** (`Cuaderno4_meteorologia_relieve`). En Canarias la altitud y la inversión térmica de los alisios cambian mucho la temperatura y la humedad en pocos kilómetros. Se comparan cuatro métodos midiendo su error al estimar cada estación con las demás de su isla:
  - copiar la estación más cercana,
  - corregirla con el gradiente estándar (−6,5 °C/km),
  - ajustar cada día el perfil vertical de la isla, con un codo en la altura de la inversión cuando los datos lo justifican,
  - ese mismo perfil corregido con las estaciones vecinas.
  - Siguiente mejora prevista: fijar la altura de la inversión con el radiosondeo diario de Güímar.
- [ ] **Incendios observados como variable objetivo:** focos térmicos de FIRMS (MODIS 1 km y VIIRS 375 m) agrupados en eventos.

**Pendiente**
- [ ] Reentrenar la red *late fusion* como clasificador (incendio sí/no), con validación por bloques espaciales y prueba con años posteriores.
- [ ] Comparar con modelos de referencia (índice de la v1, Random Forest, XGBoost) y calibrar las probabilidades.
- [ ] Detección y seguimiento de incendios activos en el visor.

Las funciones reutilizables están en el paquete [`src/wildfire`](src/wildfire), cubierto por tests (`pytest tests`) que se ejecutan en GitHub Actions.

Proyecto de referencia para comparar enfoques: [mesogeos](https://github.com/paulinebecker2002/mesogeos), predicción del peligro de incendio en el Mediterráneo con datos geoespaciales y modelos temporales.

---

## 📂 Estructura del repositorio

```text
├── docs/               # Visor web (HTML, CSS, JS), capas publicadas y estado del pipeline
├── models/             # Modelo (.keras) y escalador meteorológico (.pkl) de la versión 1
├── notebooks/
│   ├── Cuaderno1.ipynb # Catálogo de estaciones AEMET, meteorología diaria y parches Sentinel-2
│   ├── Cuaderno2.ipynb # v1 · Limpieza, imputación KNN, índices espectrales y variable objetivo
│   ├── Cuaderno3.ipynb # v1 · Entrenamiento y evaluación del modelo late fusion
│   └── Cuaderno4_meteorologia_relieve.ipynb  # Meteorología en cualquier punto: perfil vertical, inversión y validación
├── src/
│   ├── pipeline.py     # Pipeline de producción ejecutado por GitHub Actions
│   └── wildfire/       # Paquete: índices, etiquetas de incendios, validación y meteorología según el relieve
├── tests/              # Tests del paquete wildfire (pytest)
├── .github/workflows/  # Actualización diaria y tests
└── requirements.txt    # Dependencias
```

## 🔑 Credenciales

Ninguna credencial debe escribirse en el código:

- **AEMET OpenData** (`Cuaderno1`): guarda la clave como secreto de Colab con el nombre `AEMET_API_KEY` (icono de la llave en la barra lateral) o como variable de entorno.
- **Google Earth Engine** (GitHub Actions): secreto del repositorio `GEE_CREDENTIALS`.

## 👤 Autor

**José Andrés Ramos Mendoza** — [LinkedIn](https://www.linkedin.com/in/jose-andres-ramos-mendoza-646542153/) · [GitHub](https://github.com/joseandres03)
