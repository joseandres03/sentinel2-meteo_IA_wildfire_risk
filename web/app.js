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
    "Tenerife": [[27.96079606127456, -16.951279043971304], [28.599334616990024, -16.10359428244968]],
    "Gran Canaria": [[27.70, -15.83], [28.18, -15.36]],
    "La Palma": [[28.43, -18.00], [28.85, -17.72]],
    "El Hierro": [[27.613674980769087, -18.177054395808682], [27.86643158932651, -17.873688515992534]],
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

// 7. Panel de Datos flotante con el cursor
const tooltip = document.getElementById('tooltip');

map.on('mousemove', function(e) {
    if (!riesgoLayer) return;
    
    // Obtenemos los límites y datos de la isla actual
    const nombreIsla = document.getElementById('island-select').value;
    const nombreFormateado = nombreIsla.replace(' ', '_');
    const limites = boundsCanarias[nombreIsla];
    const data = window['datos_' + nombreFormateado];
    
    if (!data) return;

    // Calcular el porcentaje de posición GPS sobre el lienzo
    const latMin = limites[0][0];
    const lonMin = limites[0][1];
    const latMax = limites[1][0];
    const lonMax = limites[1][1];
    
    const pctX = (e.latlng.lng - lonMin) / (lonMax - lonMin);
    const pctY = (latMax - e.latlng.lat) / (latMax - latMin); // Y se invierte de latitud a píxel
    
    let gridX = Math.floor(pctX * data.gridW);
    let gridY = Math.floor(pctY * data.gridH);
    
    if (gridY >= 0 && gridY < data.gridH && gridX >= 0 && gridX < data.gridW) {
        let r = data.riesgo[gridY][gridX];
        let t = data.temp[gridY][gridX];
        
        if (r >= 0) {
            tooltip.style.display = 'block';
            tooltip.style.left = (e.originalEvent.pageX + 15) + 'px';
            tooltip.style.top = (e.originalEvent.pageY + 15) + 'px';
            tooltip.innerHTML = `<strong>Riesgo:</strong> ${(r * 100).toFixed(1)}%<br><strong>Temp:</strong> ${t.toFixed(1)} °C`;
        } else {
            tooltip.style.display = 'none';
        }
    } else {
        tooltip.style.display = 'none';
    }
});

map.on('mouseout', function() {
    tooltip.style.display = 'none';
});

// Cargar Tenerife por defecto al abrir la web
cargarIsla("Tenerife");