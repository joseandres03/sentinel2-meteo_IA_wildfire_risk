/*!
 * Visor de satélite (últimas 3 horas) · canarywildfirerisk.es
 * Imágenes de Meteosat Tercera Generación (MTG-I, cada 10 min) y MSG (cada 15 min)
 * servidas directamente por EUMETSAT (EUMETView, WMS público). No hay servidor propio:
 * el navegador pide cada fotograma a EUMETSAT con el parámetro TIME.
 * Requiere Leaflet 1.9 y trayectorias.css (comparte estilos .ct-).
 */
(function () {
  'use strict';

  const WMS = 'https://view.eumetsat.int/geoserver/wms';
  const CAPS = (ws) => `https://view.eumetsat.int/geoserver/${ws}/ows?service=WMS&version=1.3.0&request=GetCapabilities`;
  const HORAS = 3;

  // Canales. min = cada cuántos minutos hay imagen; lag = retraso típico de publicación.
  const CANALES = [
    { id: 'mtg_fd:rgb_geocolour', nombre: 'GeoColour', sat: 'MTG', min: 10, lag: 20,
      txt: 'Color "real" de día; de noche, nubes en infrarrojo y luces de ciudades. Útil para ver columnas de humo.' },
    { id: 'mtg_fd:vis06_hrfi', nombre: 'Visible 0,6 µm', sat: 'MTG', min: 10, lag: 20,
      txt: 'Máxima resolución (500 m). Solo de día: nubes, nieblas y humo.' },
    { id: 'mtg_fd:ir105_hrfi', nombre: 'Infrarrojo 10,5 µm', sat: 'MTG', min: 10, lag: 20,
      txt: 'Temperatura de la superficie y de los topes nubosos: blanco = frío (nubes altas). Día y noche.' },
    { id: 'msg_fes:wv062', nombre: 'Vapor de agua 6,2 µm', sat: 'MSG', min: 15, lag: 25,
      txt: 'Humedad en niveles medios y altos. Oscuro = aire seco. Deja ver borrascas, DANAs y la circulación aunque no haya nubes.' },
    { id: 'msg_fes:rgb_airmass', nombre: 'Masas de aire', sat: 'MSG', min: 15, lag: 25,
      txt: 'Distingue masas de aire frío (azul), cálido y seco (verde/oliva) y aire estratosférico seco (rojo), típico de DANAs.' },
    { id: 'mtg_fd:rgb_dust', nombre: 'Polvo (calima)', sat: 'MTG', min: 10, lag: 20,
      txt: 'El polvo en suspensión aparece en rosa o magenta. Útil en episodios de calima y aire sahariano.' },
    { id: 'mtg_fd:rgb_firetemperature', nombre: 'Temperatura de fuego', sat: 'MTG', min: 10, lag: 20,
      txt: 'Incendios activos en rojo, amarillo o blanco según su intensidad. Ojo: de día el suelo muy caliente también puede verse rojizo.' },
  ];
  const FRP = { id: 'mtg_fd:frp', nombre: 'Potencia radiativa del fuego (FRP)' };

  const pad = (n) => String(n).padStart(2, '0');
  const fmtUTC = (d) => `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
  const fmtLocal = (d) => d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', timeZone: 'Atlantic/Canary' });
  const fmtDia = (d) => d.toLocaleDateString('es-ES', { weekday: 'short', day: '2-digit', month: '2-digit', timeZone: 'Atlantic/Canary' });

  function floorTo(d, min) {
    const ms = min * 60e3;
    return new Date(Math.floor(d.getTime() / ms) * ms);
  }

  // Último instante publicado según GetCapabilities (si EUMETSAT permite leerlo desde el navegador)
  const cacheCaps = {};
  async function ultimoInstante(capa) {
    const [ws, nombre] = capa.split(':');
    try {
      if (!cacheCaps[ws]) {
        cacheCaps[ws] = fetch(CAPS(ws), { cache: 'no-cache' }).then((r) => (r.ok ? r.text() : Promise.reject(r.status)));
      }
      const xml = new DOMParser().parseFromString(await cacheCaps[ws], 'text/xml');
      for (const layer of xml.getElementsByTagName('Layer')) {
        const n = layer.getElementsByTagName('Name')[0];
        if (!n || n.textContent !== nombre) continue;
        for (const dim of layer.getElementsByTagName('Dimension')) {
          if (dim.getAttribute('name') !== 'time') continue;
          const ext = dim.textContent.trim().split(',').pop();
          const fin = ext.split('/')[1] || ext.split('/')[0];
          const t = new Date(fin);
          if (!isNaN(t)) return t;
        }
      }
    } catch (e) { /* sin acceso (CORS) o sin conexión: se estima la hora */ }
    return null;
  }

  function mount(target, opts = {}) {
    const root = typeof target === 'string' ? document.querySelector(target) : target;
    root.classList.add('ct-root');
    root.innerHTML = `
      <div class="ct-head">
        <div>
          <h3 class="ct-title">Satélite · <span data-k="canal">…</span></h3>
          <div class="ct-sub" data-k="sub">Últimas ${HORAS} horas · EUMETSAT</div>
        </div>
        <div class="ct-spacer"></div>
        <nav class="ct-nav">
          <a href="index.html">🔥 Riesgo de incendio</a><a href="trayectorias.html">🌀 Borrascas</a><a class="ct-active" href="satelite.html">🛰️ Satélite</a>
        </nav>
      </div>
      <div class="ct-body">
        <aside class="ct-side">
          <div class="ct-sec"><h4>Canal</h4><div data-k="canales"></div></div>
          <div class="ct-sec">
            <h4>Capas</h4>
            <label class="ct-opt"><input type="checkbox" data-k="frp"> Puntos calientes (FRP)</label>
            <label class="ct-opt"><input type="checkbox" data-k="costa" checked> Costa y nombres</label>
          </div>
          <div class="ct-sec">
            <h4>Velocidad</h4>
            <select class="ct-select" data-k="vel" style="width:100%">
              <option value="900">Lenta</option><option value="500" selected>Normal</option><option value="250">Rápida</option>
            </select>
          </div>
          <div class="ct-sec ct-note" data-k="info"></div>
          <div class="ct-note">Imágenes: © EUMETSAT ${new Date().getUTCFullYear()}, servicio <a href="https://view.eumetsat.int" target="_blank" rel="noopener">EUMETView</a>. MTG-I publica cada 10 min y MSG cada 15 min, con unos 15–25 min de retraso.</div>
        </aside>
        <div class="ct-main">
          <div class="ct-map" data-k="map"></div>
          <div class="ct-time">
            <button class="ct-btn" data-k="play" title="Animar">❚❚</button>
            <input type="range" data-k="frame" min="0" max="0" step="1" value="0">
            <div class="ct-time-label" data-k="tlabel"></div>
          </div>
        </div>
      </div>`;
    const $ = (k) => root.querySelector(`[data-k="${k}"]`);

    const map = L.map($('map'), { minZoom: 3, maxZoom: 9, zoomSnap: 0.5 })
      .fitBounds(opts.bounds || [[25.3, -19.5], [31.2, -12.0]]);
    map.createPane('frames').style.zIndex = 300;
    map.createPane('frp').style.zIndex = 420;
    map.createPane('ref').style.zIndex = 450;
    map.getPane('ref').style.pointerEvents = 'none';
    // Referencia: costa de EUMETSAT y nombres/límites de Esri, encima de la imagen
    const costa = L.layerGroup([
      L.tileLayer.wms(WMS, { layers: 'backgrounds:ne_10m_coastline', format: 'image/png', transparent: true,
        version: '1.3.0', pane: 'ref', tileSize: 512 }),
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { pane: 'ref', opacity: 0.8, attribution: 'Referencia &copy; Esri' }),
    ]).addTo(map);
    map.attributionControl.addAttribution('Imágenes &copy; EUMETSAT (EUMETView)');

    const S = { canal: CANALES[0], frames: [], times: [], i: 0, timer: null, frp: null, cargados: 0 };

    // lista de canales
    const box = $('canales');
    CANALES.forEach((c, k) => {
      const lab = document.createElement('label');
      lab.className = 'ct-opt';
      lab.innerHTML = `<input type="radio" name="ct-canal" ${k === 0 ? 'checked' : ''}> ${c.nombre} <small style="color:var(--ct-muted)">${c.sat}</small>`;
      lab.querySelector('input').onchange = () => setCanal(c);
      box.appendChild(lab);
    });

    function etiqueta() {
      const t = S.times[S.i];
      if (!t) return;
      const edad = Math.round((Date.now() - t.getTime()) / 60e3);
      $('tlabel').innerHTML = `<b>${fmtLocal(t)}</b> hora canaria · ${fmtUTC(t)} · ${fmtDia(t)}<br><small style="color:var(--ct-muted)">hace ${edad} min · fotograma ${S.i + 1}/${S.times.length}${S.cargados < S.times.length ? ` · cargando ${S.cargados}/${S.times.length}` : ''}</small>`;
    }

    function mostrar(i) {
      S.i = (i + S.frames.length) % S.frames.length;
      S.frames.forEach((l, k) => l.setOpacity(k === S.i ? 1 : 0));
      if (S.frp) S.frp.setParams({ time: S.times[S.i].toISOString() });
      $('frame').value = S.i;
      etiqueta();
    }

    async function setCanal(c) {
      S.canal = c;
      $('canal').textContent = c.nombre;
      $('info').textContent = c.txt;
      S.frames.forEach((l) => map.removeLayer(l));
      S.frames = []; S.cargados = 0;
      const fin = (await ultimoInstante(c.id)) || floorTo(new Date(Date.now() - c.lag * 60e3), c.min);
      const n = Math.round((HORAS * 60) / c.min);
      S.times = Array.from({ length: n }, (_, k) => new Date(fin.getTime() - (n - 1 - k) * c.min * 60e3));
      $('sub').textContent = `Últimas ${HORAS} horas · ${c.sat === 'MTG' ? 'Meteosat Tercera Generación (MTG-I)' : 'Meteosat Segunda Generación (MSG)'} · EUMETSAT`;
      S.frames = S.times.map((t) => {
        const l = L.tileLayer.wms(WMS, {
          layers: c.id, format: 'image/png', transparent: true, version: '1.3.0', pane: 'frames',
          time: t.toISOString(), tileSize: 512, opacity: 0, className: 'ct-frame',
        });
        l.once('load', () => { S.cargados++; etiqueta(); });
        return l.addTo(map);
      });
      $('frame').max = n - 1;
      mostrar(n - 1);
      if (S.timer) arrancar();
    }

    function arrancar() {
      clearInterval(S.timer);
      S.timer = setInterval(() => {
        // pausa más larga en el último fotograma para que se vea la imagen más reciente
        if (S.i === S.frames.length - 1 && !S.espera) { S.espera = true; return; }
        S.espera = false;
        mostrar(S.i + 1);
      }, +$('vel').value);
      $('play').textContent = '❚❚';
    }
    function parar() { clearInterval(S.timer); S.timer = null; $('play').textContent = '▶'; }

    $('play').onclick = () => (S.timer ? parar() : arrancar());
    $('vel').onchange = () => { if (S.timer) arrancar(); };
    $('frame').oninput = (e) => { parar(); mostrar(+e.target.value); };
    $('costa').onchange = (e) => (e.target.checked ? costa.addTo(map) : map.removeLayer(costa));
    $('frp').onchange = (e) => {
      if (e.target.checked) {
        S.frp = L.tileLayer.wms(WMS, { layers: FRP.id, format: 'image/png', transparent: true, version: '1.3.0',
          pane: 'frp', time: S.times[S.i].toISOString(), tileSize: 512 }).addTo(map);
      } else if (S.frp) { map.removeLayer(S.frp); S.frp = null; }
    };

    // refresco: cada 10 min se recargan los fotogramas del canal actual
    setInterval(() => setCanal(S.canal), (opts.refreshMinutes || 10) * 60e3);

    setCanal(CANALES[0]).then(() => { S.timer = 1; arrancar(); });
    return { map, state: S };
  }

  window.CanarySat = { mount, CANALES };
})();
