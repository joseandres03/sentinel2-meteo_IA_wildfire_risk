// Inicializacion del mapa centrado en Canarias
const map = L.map('map').setView([28.3, -15.8], 8);

// Cargar el mapa base de satélite
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
    maxZoom: 15
}).addTo(map);

const riesgoLayers = [];
let islaActualSelect = "Tenerife"; // Isla por defecto al abrir

// INYECCIÓN DINÁMICA DE LA LEYENDA AEMET EN HTML
const legendContainer = document.getElementById('legend-container');
legendContainer.innerHTML = `
    <div class="subtitle" style="margin-bottom: 10px; margin-top:0;"><strong>Niveles de peligro de incendio</strong></div>
    <div class="leyenda-aemet">
        <div class="leyenda-fila"><div class="color-box c-muy-bajo"></div> 0 - 10% (Muy bajo)</div>
        <div class="leyenda-fila"><div class="color-box c-bajo"></div> 10 - 20% (Bajo)</div>
        <div class="leyenda-fila"><div class="color-box c-moderado"></div> 20 - 40% (Moderado)</div>
        <div class="leyenda-fila"><div class="color-box c-alto"></div> 40 - 50% (Alto)</div>
        <div class="leyenda-fila"><div class="color-box c-muy-alto"></div> 50 - 60% (Muy alto)</div>
        <div class="leyenda-fila"><div class="color-box c-extremo"></div> > 60% (Extremo)</div>
        <div class="leyenda-fila" style="margin-top: 10px;"><div class="color-box c-nubes"></div> Nubes (Área sin datos)</div>
    </div>
`;

// Carga simultánea de todas las islas al abrir la web
function inicializarTodasLasIslas() {
    for (const [nombreIsla, meta] of Object.entries(configWeb.islas)) {
        const nombreFormateado = nombreIsla.replace(' ', '_');
        const rutaImagen = `capa_riesgo_${nombreFormateado}.png`;
        const opacidadActual = document.getElementById('opacity-slider').value;

        // Añadimos la imagen superpuesta
        const layer = L.imageOverlay(rutaImagen, meta.bounds, {
            opacity: opacidadActual,
            interactive: false 
        }).addTo(map);
        riesgoLayers.push(layer);

        // Inyectamos el script de telemetría de fondo
        const scriptId = 'script-datos-' + nombreFormateado;
        if (!document.getElementById(scriptId)) {
            const script = document.createElement('script');
            script.id = scriptId;
            script.src = `datos_${nombreFormateado}.js`;
            document.body.appendChild(script);
        }
    }
}

// Al cambiar el desplegable, hacemos un vuelo visual a la isla y actualizamos fechas
document.getElementById('island-select').addEventListener('change', function(e) {
    islaActualSelect = e.target.value;
    const meta = configWeb.islas[islaActualSelect];
    map.flyToBounds(meta.bounds, { duration: 1.5 });
    
    // Actualizar la etiqueta de la fecha del satélite
    const infoFechaHtml = `
        <strong>Última ejecución:</strong> ${configWeb.fecha_actualizacion}<br>
        <strong style="color:#b30000">Imagen Satélite (${islaActualSelect}):</strong> ${meta.fecha_sat}
    `;
    document.getElementById('update-date').innerHTML = infoFechaHtml;
});

// El slider modifica la opacidad de TODAS las capas a la vez
document.getElementById('opacity-slider').addEventListener('input', function(e) {
    const val = e.target.value;
    riesgoLayers.forEach(layer => layer.setOpacity(val));
});

// Panel de Datos flotante (Tooltip) multivariable y multi-isla
const tooltip = document.getElementById('tooltip');

map.on('mousemove', function(e) {
    let islaEncontrada = null;
    let limitesIsla = null;

    // Detectar si el ratón sobrevuela el Bounding Box de alguna isla
    for (const [nombreIsla, meta] of Object.entries(configWeb.islas)) {
        const latMin = meta.bounds[0][0];
        const lonMin = meta.bounds[0][1];
        const latMax = meta.bounds[1][0];
        const lonMax = meta.bounds[1][1];

        if (e.latlng.lat >= latMin && e.latlng.lat <= latMax &&
            e.latlng.lng >= lonMin && e.latlng.lng <= lonMax) {
            islaEncontrada = nombreIsla;
            limitesIsla = meta.bounds;
            break;
        }
    }

    if (!islaEncontrada) {
        tooltip.style.display = 'none';
        return;
    }

    const nombreFormateado = islaEncontrada.replace(' ', '_');
    const data = window['datos_' + nombreFormateado];
    
    if (!data) return; 

    const latMin = limitesIsla[0][0];
    const lonMin = limitesIsla[0][1];
    const latMax = limitesIsla[1][0];
    const lonMax = limitesIsla[1][1];
    
    const pctX = (e.latlng.lng - lonMin) / (lonMax - lonMin);
    const pctY = (latMax - e.latlng.lat) / (latMax - latMin); 
    
    let gridX = Math.floor(pctX * data.gridW);
    let gridY = Math.floor(pctY * data.gridH);
    
    if (gridY >= 0 && gridY < data.gridH && gridX >= 0 && gridX < data.gridW) {
        let r = data.riesgo[gridY][gridX];
        let t = data.temp[gridY][gridX];
        let hr = data.hr[gridY][gridX];
        let v = data.viento[gridY][gridX];
        
        if (r >= 0) {
            tooltip.style.display = 'block';
            tooltip.style.left = (e.originalEvent.pageX + 15) + 'px';
            tooltip.style.top = (e.originalEvent.pageY + 15) + 'px';
            
            // Lógica para traducir el valor numérico al nivel de AEMET
            let nivelTexto = "";
            let colorNivel = "#b30000";
            if (r <= 0.10) { nivelTexto = "Muy bajo"; colorNivel = "#3182bd"; }
            else if (r <= 0.20) { nivelTexto = "Bajo"; colorNivel = "#9ecae1"; }
            else if (r <= 0.40) { nivelTexto = "Moderado"; colorNivel = "#228B22"; }
            else if (r <= 0.50) { nivelTexto = "Alto"; colorNivel = "#d48806"; } 
            else if (r <= 0.60) { nivelTexto = "Muy alto"; colorNivel = "#e65c00"; }
            else { nivelTexto = "Extremo"; colorNivel = "#f03b20"; }

            // Tooltip avanzado con niveles en palabras
            tooltip.innerHTML = `
                <div style="font-weight:bold; border-bottom:1px solid #ccc; margin-bottom:4px; padding-bottom:2px;">
                    ${islaEncontrada}
                </div>
                <strong>Riesgo:</strong> <span style="color:${colorNivel}; font-weight:bold;">${nivelTexto}</span> 
                <span style="font-size:0.85em; color:#555;">(${(r * 100).toFixed(1)}%)</span><br>
                <strong>Temperatura:</strong> ${t.toFixed(1)} °C<br>
                <strong>Humedad:</strong> ${hr.toFixed(1)} %<br>
                <strong>Viento:</strong> ${v.toFixed(1)} km/h
            `;
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

// Arrancar sistema
inicializarTodasLasIslas();
map.flyToBounds(configWeb.islas[islaActualSelect].bounds, { duration: 0 });

// Forzar la actualización inicial del texto de las fechas
const metaInit = configWeb.islas[islaActualSelect];
document.getElementById('update-date').innerHTML = `
    <strong>Última ejecución:</strong> ${configWeb.fecha_actualizacion}<br>
    <strong style="color:#b30000">Imagen satélite (${islaActualSelect}):</strong> ${metaInit.fecha_sat}
`;