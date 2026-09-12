// Inicializacion del mapa centrado en Canarias
const map = L.map('map').setView([28.3, -15.8], 8);

// Cargar el mapa base de satélite (el de ESRI)
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
    maxZoom: 15
}).addTo(map);

// Diccionario con los límites geográficos exactos de cada isla
const boundsCanarias = {
    "La Gomera": [[28.01, -17.37], [28.23, -17.09]],
    "Tenerife": [[27.97, -16.94], [28.59, -16.11]],
    "Gran Canaria": [[27.70, -15.83], [28.18, -15.36]],
    "La Palma": [[28.43, -18.00], [28.85, -17.72]],
    "El Hierro": [[27.62, -18.17], [27.86, -17.88]],
    "Lanzarote": [[28.83, -13.91], [29.26, -13.33]],
    "Fuerteventura": [[28.01, -14.52], [28.76, -13.82]]
};

// Variable global para almacenar la capa térmica
let riesgoLayer = null;

// Función para actualizar la isla en la pantalla
function cargarIsla(nombreIsla) {
    const limites = boundsCanarias[nombreIsla];
    
    // Anima el vuelo del mapa hacia la nueva isla
    map.flyToBounds(limites, { duration: 1.5 });

    // Si ya había una capa de riesgo cargada, la borramos para que no se superpongan
    if (riesgoLayer !== null) {
        map.removeLayer(riesgoLayer);
    }

    // Ruta de la imagen transparente que generó el Pipeline
    const nombreFormateado = nombreIsla.replace(' ', '_');
    const rutaImagen = `capa_riesgo_${nombreFormateado}.png`;
    const opacidadActual = document.getElementById('opacity-slider').value;

    // Crear la nueva capa y añadirla al mapa
    riesgoLayer = L.imageOverlay(rutaImagen, limites, {
        opacity: opacidadActual,
        interactive: false // Para no bloquear el ratón al arrastrar el mapa
    }).addTo(map);

    // Actualizar la leyenda de la interfaz
    document.getElementById('map-legend').src = `leyenda_${nombreFormateado}.png`;
}

// Escuchar cambios en el selector de isla
document.getElementById('island-select').addEventListener('change', function(e) {
    cargarIsla(e.target.value);
});

// Escuchar cambios en la barra de opacidad
document.getElementById('opacity-slider').addEventListener('input', function(e) {
    if (riesgoLayer !== null) {
        riesgoLayer.setOpacity(e.target.value);
    }
});

// Cargar Tenerife por defecto al abrir la web
cargarIsla("Tenerife");