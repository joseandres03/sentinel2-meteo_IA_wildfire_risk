/*!
 * Visor de satélite (últimas 3 horas) · canarywildfirerisk.es
 * Imágenes de Meteosat Tercera Generación (MTG-I, cada 10 min) y MSG (cada 15 min)
 * servidas directamente por EUMETSAT (EUMETView, WMS público). No hay servidor propio:
 * el navegador pide cada fotograma a EUMETSAT con el parámetro TIME.
 * EUMETView es lento (10–60 s por imagen), así que cada fotograma es UNA imagen de la
 * vista actual (no cientos de mosaicos), se piden pocas a la vez empezando por la más
 * reciente, y la animación usa solo los fotogramas que ya han llegado.
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
  const SIMULTANEAS = 3;      // EUMETView es lento: pocas peticiones a la vez
  const ESPERA_MAX = 90e3;    // tiempo máximo por fotograma

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
    map.createPane('ref').style.zIndex = 450;
    map.getPane('ref').style.pointerEvents = 'none';
    // Referencia: nombres y límites de Esri (rápido) + costa de EUMETSAT (una imagen por vista)
    const costa = L.layerGroup([
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        { pane: 'ref', opacity: 0.85, attribution: 'Referencia &copy; Esri' }),
    ]).addTo(map);
    let costaImg = null;
    map.attributionControl.addAttribution('Imágenes &copy; EUMETSAT (EUMETView)');

    const S = { canal: CANALES[0], frames: [], times: [], estado: [], i: 0, timer: null, gen: 0 };

    // lista de canales
    const box = $('canales');
    CANALES.forEach((c, k) => {
      const lab = document.createElement('label');
      lab.className = 'ct-opt';
      lab.innerHTML = `<input type="radio" name="ct-canal" ${k === 0 ? 'checked' : ''}> ${c.nombre} <small style="color:var(--ct-muted)">${c.sat}</small>`;
      lab.querySelector('input').onchange = () => setCanal(c);
      box.appendChild(lab);
    });

    const listos = () => S.estado.map((e, k) => (e === 'ok' ? k : -1)).filter((k) => k >= 0);

    function etiqueta() {
      const t = S.times[S.i];
      const ok = listos().length, n = S.times.length;
      const err = S.estado.filter((e) => e === 'error').length;
      if (!t || S.estado[S.i] !== 'ok') {
        $('tlabel').innerHTML = S.estado.length && S.estado[n - 1] === 'error'
          ? '<b>EUMETSAT no responde ahora mismo</b><br><small style="color:var(--ct-muted)">Se reintentará en unos minutos</small>'
          : `<b>Cargando imágenes de EUMETSAT…</b><br><small style="color:var(--ct-muted)">${ok}/${n} fotogramas · su servidor tarda a veces hasta un minuto</small>`;
        return;
      }
      const edad = Math.round((Date.now() - t.getTime()) / 60e3);
      $('tlabel').innerHTML = `<b>${fmtLocal(t)}</b> hora canaria · ${fmtUTC(t)} · ${fmtDia(t)}<br><small style="color:var(--ct-muted)">hace ${edad} min · ${ok < n ? `cargados ${ok}/${n}${err ? ` (${err} sin imagen)` : ''}` : `${n} fotogramas`}</small>`;
    }

    function mostrar(i) {
      if (S.estado[i] !== 'ok') return;
      S.i = i;
      S.frames.forEach((l, k) => l && l.setOpacity(k === i ? 1 : 0));
      $('frame').value = i;
      etiqueta();
    }

    function siguiente() {
      const ok = listos();
      if (!ok.length) return;
      const pos = ok.indexOf(S.i);
      mostrar(ok[(pos + 1) % ok.length]);
    }

    function urlWMS(capa, t, b, w, h, fmt) {
      const p = map.options.crs.project;
      const sw = p(b.getSouthWest()), ne = p(b.getNorthEast());
      const q = new URLSearchParams({ service: 'WMS', version: '1.3.0', request: 'GetMap', layers: capa, styles: '',
        format: fmt, transparent: 'true', crs: 'EPSG:3857', bbox: [sw.x, sw.y, ne.x, ne.y].join(','),
        width: w, height: h });
      if (t) q.set('time', t.toISOString());
      return `${WMS}?${q}`;
    }

    function cargarImagen(url) {
      return new Promise((ok, mal) => {
        const img = new Image();
        const to = setTimeout(() => { img.src = ''; mal(new Error('tiempo')); }, ESPERA_MAX);
        img.onload = () => { clearTimeout(to); ok(); };
        img.onerror = () => { clearTimeout(to); mal(new Error('error')); };
        img.src = url;
      });
    }

    // Pide todos los fotogramas para la vista actual (el más reciente primero)
    function cargarVista() {
      const gen = ++S.gen;
      const c = S.canal;
      const b = map.getBounds();
      const sz = map.getSize();
      const k = Math.min(1, 1400 / sz.x);   // como mucho 1400 px de ancho
      const w = Math.round(sz.x * k), h = Math.round(sz.y * k);
      S.frames.forEach((l) => l && map.removeLayer(l));
      S.frames = S.times.map(() => null);
      S.estado = S.times.map(() => 'pendiente');
      etiqueta();
      const cola = S.times.map((_, i) => S.times.length - 1 - i);   // del más reciente al más antiguo
      const fmt = c.id.includes('rgb_') ? 'image/jpeg' : 'image/png';
      const uno = async () => {
        while (cola.length && gen === S.gen) {
          const i = cola.shift();
          const url = urlWMS(c.id, S.times[i], b, w, h, fmt);
          try {
            await cargarImagen(url);
            if (gen !== S.gen) return;
            S.frames[i] = L.imageOverlay(url, b, { pane: 'frames', opacity: 0, className: 'ct-frame' }).addTo(map);
            S.estado[i] = 'ok';
            if (S.estado[S.i] !== 'ok' || i === S.times.length - 1) mostrar(i);
          } catch (e) {
            if (gen === S.gen) S.estado[i] = 'error';
          }
          if (gen === S.gen) etiqueta();
        }
      };
      for (let n = 0; n < SIMULTANEAS; n++) uno();
      // costa de EUMETSAT para esta vista
      if ($('costa').checked) {
        const url = urlWMS('backgrounds:ne_10m_coastline', null, b, w, h, 'image/png');
        cargarImagen(url).then(() => {
          if (gen !== S.gen) return;
          if (costaImg) map.removeLayer(costaImg);
          costaImg = L.imageOverlay(url, b, { pane: 'ref', interactive: false }).addTo(map);
        }).catch(() => {});
      }
    }

    async function setCanal(c) {
      S.canal = c;
      $('canal').textContent = c.nombre;
      $('info').textContent = c.txt;
      const fin = (await ultimoInstante(c.id)) || floorTo(new Date(Date.now() - c.lag * 60e3), c.min);
      const n = Math.round((HORAS * 60) / c.min);
      S.times = Array.from({ length: n }, (_, k) => new Date(fin.getTime() - (n - 1 - k) * c.min * 60e3));
      S.i = n - 1;
      $('sub').textContent = `Últimas ${HORAS} horas · ${c.sat === 'MTG' ? 'Meteosat Tercera Generación (MTG-I)' : 'Meteosat Segunda Generación (MSG)'} · EUMETSAT`;
      $('frame').max = n - 1;
      cargarVista();
    }

    function arrancar() {
      clearInterval(S.timer);
      S.timer = setInterval(() => {
        // pausa más larga en el último fotograma para que se vea la imagen más reciente
        const ok = listos();
        if (S.i === ok[ok.length - 1] && !S.espera) { S.espera = true; return; }
        S.espera = false;
        siguiente();
      }, +$('vel').value);
      $('play').textContent = '❚❚';
    }
    function parar() { clearInterval(S.timer); S.timer = null; $('play').textContent = '▶'; }

    $('play').onclick = () => (S.timer ? parar() : arrancar());
    $('vel').onchange = () => { if (S.timer) arrancar(); };
    $('frame').oninput = (e) => {
      parar();
      // ir al fotograma cargado más cercano
      const v = +e.target.value, ok = listos();
      if (ok.length) mostrar(ok.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a)));
    };
    $('costa').onchange = (e) => {
      if (e.target.checked) { costa.addTo(map); if (costaImg) costaImg.addTo(map); }
      else { map.removeLayer(costa); if (costaImg) map.removeLayer(costaImg); }
    };
    // al mover o hacer zoom se piden de nuevo las imágenes para la vista nueva
    let tMove = null;
    map.on('moveend', () => { clearTimeout(tMove); tMove = setTimeout(cargarVista, 700); });

    // refresco: cada 10 min se recargan los fotogramas del canal actual
    setInterval(() => setCanal(S.canal), (opts.refreshMinutes || 10) * 60e3);

    setCanal(CANALES[0]).then(() => { S.timer = 1; arrancar(); });
    return { map, state: S };
  }

  window.CanarySat = { mount, CANALES };
})();
