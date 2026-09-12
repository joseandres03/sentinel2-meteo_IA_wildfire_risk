// Inicializacion del mapa centrado en Canarias
const map = L.map('map').setView([28.3, -15.8], 8);

// Cargar el mapa base de satélite (el de ESRI)
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
    maxZoom: 15
}).addTo(map);

const boundsCanarias = {
    "La Gomera": [[28.00549084316792, -17.374943281524075], [28.23459328764559, -17.085678399275903]],
    "Tenerife": [[27.96079606127456, -16.951279043971304], [28.599334616990024, -16.10359428244968]],
    "Gran Canaria": [[27.697980795215834, -15.833736277659783], [28.18212529384355, -15.358368388662361]],
    "La Palma": [[28.424068359344787, -18.012054818273228], [28.855970739565613, -17.709141528583167]],
    "El Hierro": [[27.613674980769087, -18.177054395808682], [27.86643158932651, -17.873688515992534]],
    "Lanzarote": [[28.823957180304102, -13.914537070112212], [29.26600701888792, -13.323056438785589]],
    "Fuerteventura": [[28.005751004514416, -14.523466097896096], [28.764327499586976, -13.81161075925612]]
};

let riesgoLayer = null;

function cargarIsla(nombreIsla) {
    const limites = boundsCanarias[nombreIsla];
    map.flyToBounds(limites, { duration: 1.5 });

    if (riesgoLayer !== null) {
        map.removeLayer(riesgoLayer);
    }

    const nombreFormateado = nombreIsla.replace(' ', '_');
    const rutaImagen = `capa_riesgo_${nombreFormateado}.png`;
    const opacidadActual = document.getElementById('opacity-slider').value;

    riesgoLayer = L.imageOverlay(rutaImagen, limites, {
        opacity: opacidadActual,
        interactive: false 
    }).addTo(map);

    // SISTEMA DE CARGA DINÁMICA (Lazy Loading) DE LOS DATOS MATRICIALES
    const scriptId = 'script-datos-' + nombreFormateado;
    if (!document.getElementById(scriptId)) {
        const script = document.createElement('script');
        script.id = scriptId;
        script.src = `datos_${nombreFormateado}.js`;
        document.body.appendChild(script);
    }
}

document.getElementById('island-select').addEventListener('change', function(e) {
    cargarIsla(e.target.value);
});

document.getElementById('opacity-slider').addEventListener('input', function(e) {
    if (riesgoLayer !== null) {
        riesgoLayer.setOpacity(e.target.value);
    }
});

// Panel de Datos flotante con el cursor
const tooltip = document.getElementById('tooltip');

map.on('mousemove', function(e) {
    if (!riesgoLayer) return;
    
    const nombreIsla = document.getElementById('island-select').value;
    const nombreFormateado = nombreIsla.replace(' ', '_');
    const limites = boundsCanarias[nombreIsla];
    
    // Ahora recuperamos los datos del objeto global inyectado por Python
    const data = window['datos_' + nombreFormateado];
    
    if (!data) return;

    const latMin = limites[0][0];
    const lonMin = limites[0][1];
    const latMax = limites[1][0];
    const lonMax = limites[1][1];
    
    const pctX = (e.latlng.lng - lonMin) / (lonMax - lonMin);
    const pctY = (latMax - e.latlng.lat) / (latMax - latMin); 
    
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

cargarIsla("Tenerife");

// Sincronizar fecha de última ejecución
document.getElementById('update-date').innerText = configWeb.fecha_actualizacion;