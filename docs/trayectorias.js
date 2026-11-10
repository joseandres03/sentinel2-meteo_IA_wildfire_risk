/*!
 * Trayectorias de borrascas · canarywildfirerisk.es
 * Requiere Leaflet 1.9 (L global) y trayectorias.css. Datos: docs/datos_borrascas/
 * (los genera src/trayectorias.py desde .github/workflows/trayectorias.yml).
 *
 *   <div id="trayectorias"></div>
 *   <script>CanaryTracks.mount('#trayectorias', { base: 'datos_borrascas/' });</script>
 */
(function () {
  'use strict';

  const CANARIAS = [
    ['El Hierro', 27.75, -18.00], ['La Palma', 28.68, -17.86], ['La Gomera', 28.10, -17.23],
    ['Tenerife', 28.29, -16.62], ['Gran Canaria', 27.96, -15.59],
    ['Fuerteventura', 28.36, -14.05], ['Lanzarote', 29.04, -13.63],
  ];
  const DIAS = ['dom', 'lun', 'mar', 'mié', 'jue', 'vie', 'sáb'];
  const pad = (n) => String(n).padStart(2, '0');
  const fmtT = (d) => `${DIAS[d.getUTCDay()]} ${pad(d.getUTCDate())}/${pad(d.getUTCMonth() + 1)} ${pad(d.getUTCHours())} UTC`;
  const fmtShort = (d) => `${pad(d.getUTCDate())}/${pad(d.getUTCMonth() + 1)} ${pad(d.getUTCHours())}Z`;
  const addH = (iso, h) => new Date(new Date(iso).getTime() + h * 3600e3);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function el(tag, attrs = {}, html = '') {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v; else e.setAttribute(k, v);
    }
    if (html) e.innerHTML = html;
    return e;
  }

  // posición interpolada de una trayectoria a la hora h
  function posAt(track, h) {
    if (!track || !track.length) return null;
    for (let i = 0; i < track.length; i++) {
      const a = track[i];
      if (a.h === h) return a;
      const b = track[i + 1];
      if (b && a.h < h && h < b.h) {
        const f = (h - a.h) / (b.h - a.h);
        return { h, lat: a.lat + f * (b.lat - a.lat), lon: a.lon + f * (b.lon - a.lon), p: a.p + f * (b.p - a.p) };
      }
    }
    return null;
  }

  async function getJSON(url) {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json();
  }

  function mount(target, opts = {}) {
    const root = typeof target === 'string' ? document.querySelector(target) : target;
    const base = (opts.base || 'datos_borrascas/').replace(/\/?$/, '/');
    const S = {
      storms: [], storm: opts.storm || null, index: null, runIdx: -1, cache: {},
      visible: {}, hour: 0, validTime: null, playing: null,
      o: { det: true, members: true, mean: true, obs: true, ring: true, prev: false, dots: true },
    };

    root.classList.add('ct-root');
    root.innerHTML = `
      <div class="ct-head">
        <div>
          <h3 class="ct-title">Trayectoria de borrascas · <span data-k="name">…</span> <span class="ct-demo" data-k="demo" hidden>DATOS DE DEMOSTRACIÓN</span></h3>
          <div class="ct-sub" data-k="sub">Cargando…</div>
        </div>
        <div class="ct-spacer"></div>
        <select class="ct-select" data-k="storm" title="Borrasca"></select>
        <nav class="ct-nav"><a href="index.html">🔥 Riesgo de incendio</a><a class="ct-active" href="trayectorias.html">🌀 Borrascas</a><a href="satelite.html">🛰️ Satélite</a></nav>
      </div>
      <div class="ct-body">
        <aside class="ct-side">
          <div class="ct-sec">
            <h4>Pasada del modelo</h4>
            <div class="ct-run">
              <button class="ct-btn" data-k="prevRun" title="Pasada anterior">◀</button>
              <select class="ct-select" data-k="run"></select>
              <button class="ct-btn" data-k="nextRun" title="Pasada siguiente">▶</button>
            </div>
            <div style="margin-top:8px;display:flex;gap:6px">
              <button class="ct-btn ct-primary" data-k="play" style="flex:1">▶ Evolución de pasadas</button>
            </div>
            <div class="ct-note">La animación recorre las pasadas (cada 6 h) con la fecha de validez fija: así se ve cómo han ido corrigiendo los modelos. La línea amarilla es lo que pasó de verdad después de cada pasada.</div>
          </div>
          <div class="ct-sec"><h4>Modelos</h4><div data-k="models"></div></div>
          <div class="ct-sec">
            <h4>Mostrar</h4>
            <label class="ct-opt"><input type="checkbox" data-o="det"> Deterministas (línea gruesa)</label>
            <label class="ct-opt"><input type="checkbox" data-o="members"> Miembros del ensemble</label>
            <label class="ct-opt"><input type="checkbox" data-o="mean"> Media del ensemble</label>
            <label class="ct-opt"><input type="checkbox" data-o="dots"> Posiciones en la hora elegida</label>
            <label class="ct-opt"><input type="checkbox" data-o="obs"> Trayectoria real (antes y después de la pasada)</label>
            <label class="ct-opt"><input type="checkbox" data-o="prev"> Pasada anterior (en gris)</label>
            <label class="ct-opt"><input type="checkbox" data-o="ring"> Radio de 200 km alrededor de Canarias</label>
          </div>
          <div class="ct-sec ct-legend">
            <h4>Leyenda</h4>
            <div><span class="ct-l"></span>Determinista (escenario único)</div>
            <div><span class="ct-l ct-thin"></span>Miembro del ensemble</div>
            <div><span class="ct-l ct-dash"></span>Media del ensemble</div>
            <div><span class="ct-l" style="border-top-style:dotted"></span>Trayectoria real hasta la pasada</div>
            <div><span class="ct-l" style="border-color:#ffe14d"></span>Trayectoria real posterior</div>
          </div>
        </aside>
        <div class="ct-main">
          <div class="ct-map" data-k="map"></div>
          <div class="ct-time">
            <button class="ct-btn" data-k="playH" title="Animar en el tiempo">▶</button>
            <input type="range" data-k="hour" min="0" max="120" step="6" value="0">
            <div class="ct-time-label" data-k="tlabel"></div>
          </div>
          <div class="ct-stats" data-k="stats"></div>
        </div>
      </div>`;
    const $ = (k) => root.querySelector(`[data-k="${k}"]`);

    // ---------------- mapa ----------------
    const map = L.map($('map'), { zoomControl: true, worldCopyJump: true, preferCanvas: true })
      .setView([30, -18], 5);
    // mismo mapa base que el visor de riesgo
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 10,
      attribution: 'Tiles &copy; Esri · Modelos: ECMWF Open Data (CC-BY-4.0), NOAA GFS/GEFS',
    }).addTo(map);
    const G = {
      ring: L.layerGroup().addTo(map), prev: L.layerGroup().addTo(map),
      tracks: L.layerGroup().addTo(map), obs: L.layerGroup().addTo(map), dots: L.layerGroup().addTo(map),
    };
    let fitted = false;

    // ---------------- datos ----------------
    async function loadStorms() {
      try {
        S.storms = (await getJSON(base + 'storms.json')).storms || [];
      } catch (e) { S.storms = []; }
      const sel = $('storm');
      sel.innerHTML = S.storms.map((s) => `<option value="${esc(s.key)}">${esc(s.name)}</option>`).join('');
      sel.hidden = S.storms.length < 2;
      if (!S.storm) S.storm = (S.storms.find((s) => !s.demo) || S.storms[0] || {}).key;
      if (!S.storm) {
        $('name').textContent = 'sin borrascas activas';
        root.querySelectorAll('.ct-run button, .ct-run select, [data-k=play], [data-k=playH], [data-k=hour]').forEach((b) => { b.disabled = true; });
        $('sub').innerHTML = '<div class="ct-empty">Ahora mismo no se está siguiendo ninguna borrasca. Cuando haya una que pueda afectar a Canarias aparecerá aquí, con las trayectorias de cada modelo actualizadas tras cada pasada (00 y 12 UTC).</div>';
        return;
      }
      sel.value = S.storm;
      await loadStorm(S.storm);
    }

    async function loadStorm(key) {
      S.storm = key; S.cache = {}; fitted = false;
      S.index = await getJSON(`${base}${key}/index.json`);
      $('name').textContent = S.index.storm;
      $('demo').hidden = !S.index.demo;
      const runSel = $('run');
      runSel.innerHTML = S.index.runs.map((r, i) => `<option value="${i}">${fmtT(new Date(r.t))}</option>`).join('');
      await setRun(S.index.runs.length - 1);
    }

    async function runData(i) {
      const r = S.index.runs[i];
      if (!r) return null;
      if (!S.cache[r.id]) S.cache[r.id] = await getJSON(`${base}${S.storm}/${r.file}`);
      return S.cache[r.id];
    }

    async function setRun(i) {
      i = Math.max(0, Math.min(S.index.runs.length - 1, i));
      S.runIdx = i;
      $('run').value = String(i);
      $('prevRun').disabled = i === 0;
      $('nextRun').disabled = i === S.index.runs.length - 1;
      const d = await runData(i);
      // modelos visibles: por defecto todos
      for (const k of Object.keys(d.models)) if (!(k in S.visible)) S.visible[k] = true;
      // hora: mantener la fecha de validez si existe en esta pasada
      const maxH = Math.max(...Object.values(d.models).flatMap((m) => m.members.flatMap((x) => x.track.map((p) => p.h))), 0);
      const slider = $('hour');
      slider.max = maxH;
      if (S.validTime) {
        const h = Math.round((S.validTime - new Date(d.run)) / 3600e3 / 6) * 6;
        S.hour = Math.max(0, Math.min(maxH, h));
      }
      slider.value = S.hour;
      $('sub').innerHTML = `Pasada <b>${fmtT(new Date(d.run))}</b> · generado ${esc(new Date(d.generated).toLocaleString('es-ES', { timeZone: 'Atlantic/Canary' }))} (hora canaria)`
        + (S.index.disipada ? `<br><span style="color:#b30000">La baja no se localiza en los análisis del GFS desde ${esc(fmtT(new Date(S.index.disipada)))}: puede haberse disipado o ser demasiado débil. Las pasadas posteriores parten de su última posición conocida.</span>` : '');
      renderModels(d);
      if (S.o.prev && i > 0) await runData(i - 1);
      draw();
    }

    // ---------------- panel de modelos ----------------
    function renderModels(d) {
      const box = $('models');
      box.innerHTML = '';
      for (const [k, m] of Object.entries(d.models)) {
        const row = el('div', { class: `ct-model${S.visible[k] ? '' : ' ct-off'}${m.parent && d.models[m.parent] ? ' ct-child' : ''}` },
          `<span class="ct-sw${m.kind === 'ensemble' ? ' ct-ens' : ''}" style="background-color:${m.color}"></span>
           <span>${esc(m.label)}</span>
           <small>${m.kind === 'ensemble' ? `${m.found}/${m.total}` : (m.found ? '' : 'sin centro')}</small>`);
        row.onclick = () => { S.visible[k] = !S.visible[k]; row.classList.toggle('ct-off', !S.visible[k]); draw(); };
        box.appendChild(row);
      }
    }

    // ---------------- dibujo ----------------
    function line(track, style, tip) {
      const ll = track.map((p) => [p.lat, p.lon]);
      const pl = L.polyline(ll, style);
      if (tip) pl.bindTooltip(tip, { sticky: true, className: 'ct-tip' });
      return pl;
    }

    function trackTip(label, track, run) {
      const pmin = Math.min(...track.map((p) => p.p));
      const end = track[track.length - 1];
      return `<b>${esc(label)}</b><br>P. mínima: ${pmin.toFixed(0)} hPa<br>Hasta: ${fmtT(addH(run, end.h))} (+${end.h} h)`;
    }

    function draw() {
      const d = S.cache[S.index.runs[S.runIdx].id];
      if (!d) return;
      Object.values(G).forEach((g) => g.clearLayers());
      const bounds = L.latLngBounds(CANARIAS.map((c) => [c[1], c[2]]));

      // Canarias
      if (S.o.ring) {
        for (const [n, la, lo] of CANARIAS) {
          L.circle([la, lo], { radius: 200e3, color: '#ffffff', weight: 1, opacity: 0.35, dashArray: '4 6', fillOpacity: 0.03, interactive: false }).addTo(G.ring);
        }
      }

      // pasada anterior en gris
      if (S.o.prev && S.runIdx > 0) {
        const p = S.cache[S.index.runs[S.runIdx - 1].id];
        if (p) for (const [k, m] of Object.entries(p.models)) {
          if (!S.visible[k]) continue;
          const t = m.kind === 'ensemble' ? m.mean : m.members[0] && m.members[0].track;
          if (t && t.length > 1) line(t, { color: '#9aa7b8', weight: 2, opacity: 0.55, dashArray: '2 5' },
            `<b>${esc(m.label)}</b> · pasada anterior (${fmtT(new Date(p.run))})`).addTo(G.prev);
        }
      }

      // ensembles primero (debajo), deterministas encima
      const entries = Object.entries(d.models).filter(([k]) => S.visible[k]);
      entries.sort((a, b) => (a[1].kind === 'ensemble' ? 0 : 1) - (b[1].kind === 'ensemble' ? 0 : 1));
      for (const [k, m] of entries) {
        if (m.kind === 'ensemble') {
          if (S.o.members) for (const mem of m.members) {
            if (mem.track.length < 2) continue;
            line(mem.track, { color: m.color, weight: 1, opacity: 0.42 },
              trackTip(`${m.label} · ${mem.id === 0 ? 'control' : 'miembro ' + mem.id}`, mem.track, d.run)).addTo(G.tracks);
            mem.track.forEach((p) => bounds.extend([p.lat, p.lon]));
          }
          if (S.o.mean && m.mean && m.mean.length > 1) {
            line(m.mean, { color: '#000', weight: 6, opacity: 0.35, interactive: false }).addTo(G.tracks);
            line(m.mean, { color: m.color, weight: 3, dashArray: '8 6' }, trackTip(`${m.label} · media`, m.mean, d.run)).addTo(G.tracks);
            m.mean.forEach((p) => bounds.extend([p.lat, p.lon]));
          }
        } else if (S.o.det) {
          const t = m.members[0] && m.members[0].track;
          if (!t || t.length < 2) continue;
          line(t, { color: '#000', weight: 7, opacity: 0.45, interactive: false }).addTo(G.tracks);
          line(t, { color: m.color, weight: 3.5 }, trackTip(m.label, t, d.run)).addTo(G.tracks);
          t.filter((p) => p.h % 24 === 0).forEach((p) => {
            L.circleMarker([p.lat, p.lon], { radius: 3.5, color: '#000', weight: 1, fillColor: m.color, fillOpacity: 1 })
              .bindTooltip(`<b>${esc(m.label)}</b><br>${fmtT(addH(d.run, p.h))} (+${p.h} h)<br>${p.p.toFixed(0)} hPa`, { className: 'ct-tip' })
              .addTo(G.tracks);
          });
          t.forEach((p) => bounds.extend([p.lat, p.lon]));
        }
      }

      // trayectoria real (análisis): hasta la hora de la pasada, y la posterior en otro estilo
      // para comparar lo que preveía esta pasada con lo que ocurrió después
      if (S.o.obs && S.index.observed && S.index.observed.length) {
        const upto = new Date(d.run).getTime();
        const antes = S.index.observed.filter((o) => new Date(o.t).getTime() <= upto);
        const despues = S.index.observed.filter((o) => new Date(o.t).getTime() >= upto);
        if (antes.length > 1) L.polyline(antes.map((o) => [o.lat, o.lon]), { color: '#fff', weight: 2.5, dashArray: '1 6', lineCap: 'round' }).addTo(G.obs);
        if (despues.length > 1) {
          L.polyline(despues.map((o) => [o.lat, o.lon]), { color: '#000', weight: 6, opacity: 0.35, interactive: false }).addTo(G.obs);
          L.polyline(despues.map((o) => [o.lat, o.lon]), { color: '#ffe14d', weight: 3 })
            .bindTooltip('<b>Trayectoria real posterior</b><br>Lo que ocurrió después de esta pasada', { sticky: true, className: 'ct-tip' }).addTo(G.obs);
        }
        S.index.observed.forEach((o, i) => {
          const post = new Date(o.t).getTime() > upto;
          L.circleMarker([o.lat, o.lon], { radius: post ? 3.5 : 5, color: post ? '#ffe14d' : '#fff', weight: 2, fillColor: '#222', fillOpacity: 1 })
            .bindTooltip(`<b>${post ? 'Real (posterior)' : 'Real'} · ${fmtT(new Date(o.t))}</b><br>${o.p.toFixed(0)} hPa · análisis ${esc(o.src)}`, { className: 'ct-tip' })
            .addTo(G.obs);
          if (new Date(o.t).getUTCHours() === 0) {
            L.tooltip({ permanent: true, direction: 'left', offset: [-6, 0], className: 'ct-obs-label' })
              .setLatLng([o.lat, o.lon]).setContent(fmtShort(new Date(o.t))).addTo(G.obs);
          }
          bounds.extend([o.lat, o.lon]);
        });
      }

      drawDots(d, entries);
      renderStats(d);
      if (!fitted) { map.fitBounds(bounds.pad(0.08)); fitted = true; }
    }

    function drawDots(d, entries) {
      G.dots.clearLayers();
      const h = S.hour;
      S.validTime = addH(d.run, h);
      $('tlabel').innerHTML = `<b>${fmtT(S.validTime)}</b> · +${h} h`;
      if (!S.o.dots) return;
      for (const [k, m] of entries) {
        if (m.kind === 'ensemble') {
          if (S.o.members) for (const mem of m.members) {
            const p = posAt(mem.track, h);
            if (p) L.circleMarker([p.lat, p.lon], { radius: 3, color: m.color, weight: 1, fillColor: m.color, fillOpacity: 0.85 })
              .bindTooltip(`${esc(m.label)} · ${mem.id === 0 ? 'control' : 'm' + mem.id}<br>${p.p.toFixed(0)} hPa`, { className: 'ct-tip' }).addTo(G.dots);
          }
          const pm = S.o.mean && posAt(m.mean, h);
          if (pm) L.circleMarker([pm.lat, pm.lon], { radius: 6, color: '#000', weight: 2, fillColor: m.color, fillOpacity: 1 })
            .bindTooltip(`<b>${esc(m.label)} · media</b><br>${pm.p.toFixed(0)} hPa`, { className: 'ct-tip' }).addTo(G.dots);
        } else if (S.o.det) {
          const p = posAt(m.members[0] && m.members[0].track, h);
          if (p) L.circleMarker([p.lat, p.lon], { radius: 8, color: '#fff', weight: 2.5, fillColor: m.color, fillOpacity: 1 })
            .bindTooltip(`<b>${esc(m.label)}</b><br>${fmtT(S.validTime)}<br>${p.p.toFixed(0)} hPa`, { className: 'ct-tip' }).addTo(G.dots);
        }
      }
    }

    function pct(v) {
      if (v == null) return '—';
      return `<span class="ct-bar" style="width:${Math.max(2, v * 0.5)}px"></span>${v}%`;
    }

    function renderStats(d) {
      const rows = Object.entries(d.models).filter(([k]) => S.visible[k]).map(([k, m]) => {
        if (m.kind === 'ensemble') {
          const c = m.canarias || {};
          return `<tr><td><span class="ct-sw ct-ens" style="display:inline-block;vertical-align:-2px;margin-right:6px;background-color:${m.color}"></span>${esc(m.label)}</td>
            <td>${pct(c['100'])}</td><td>${pct(c['200'])}</td><td>${pct(c['300'])}</td>
            <td>${c.dist_mediana_km != null ? c.dist_mediana_km + ' km' : '—'}</td>
            <td>${m.pmin ? `${m.pmin.mediana.toFixed(0)} hPa <small style="color:var(--ct-muted)">(${m.pmin.min.toFixed(0)}–${m.pmin.max.toFixed(0)})</small>` : '—'}</td></tr>`;
        }
        const mem = m.members[0] || {};
        const ca = mem.closest;
        const pmin = mem.track && mem.track.length ? Math.min(...mem.track.map((p) => p.p)).toFixed(0) + ' hPa' : '—';
        const yes = (r) => (ca ? (ca.km <= r ? '<b style="color:var(--ct-warn)">Sí</b>' : 'No') : '—');
        return `<tr><td><span class="ct-sw" style="display:inline-block;vertical-align:-2px;margin-right:6px;background:${m.color}"></span>${esc(m.label)}</td>
          <td>${yes(100)}</td><td>${yes(200)}</td><td>${yes(300)}</td>
          <td>${ca ? `${ca.km} km <small style="color:var(--ct-muted)">${esc(ca.isla)}, ${fmtShort(addH(d.run, ca.h))}</small>` : '—'}</td>
          <td>${pmin}</td></tr>`;
      }).join('');
      $('stats').innerHTML = `
        <table>
          <thead><tr><th>Modelo</th><th>Centro a &lt;100 km</th><th>&lt;200 km</th><th>&lt;300 km</th><th>Distancia mínima a Canarias</th><th>Presión mínima</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div class="ct-note">En los ensembles, el porcentaje es la fracción de miembros cuyo <b>centro</b> pasa a menos de esa distancia de alguna isla (en los deterministas, sí/no). La distancia es la mediana de los miembros. Mide la incertidumbre de la trayectoria, no la probabilidad de viento o lluvia: los efectos de una borrasca llegan mucho más lejos que su centro.</div>`;
    }

    // ---------------- controles ----------------
    root.querySelectorAll('[data-o]').forEach((inp) => {
      const k = inp.dataset.o;
      inp.checked = S.o[k];
      inp.onchange = async () => {
        S.o[k] = inp.checked;
        if (k === 'prev' && S.runIdx > 0) await runData(S.runIdx - 1);
        draw();
      };
    });
    $('storm').onchange = (e) => loadStorm(e.target.value);
    $('run').onchange = (e) => setRun(+e.target.value);
    $('prevRun').onclick = () => setRun(S.runIdx - 1);
    $('nextRun').onclick = () => setRun(S.runIdx + 1);
    $('hour').oninput = (e) => {
      S.hour = +e.target.value;
      const d = S.cache[S.index.runs[S.runIdx].id];
      drawDots(d, Object.entries(d.models).filter(([k]) => S.visible[k]));
    };

    function stopAll() {
      clearInterval(S.playing); S.playing = null;
      $('play').textContent = '▶ Evolución de pasadas'; $('playH').textContent = '▶';
    }
    $('play').onclick = async () => {
      if (S.playing) return stopAll();
      $('play').textContent = '❚❚ Pausar';
      if (S.runIdx === S.index.runs.length - 1) await setRun(0);
      S.playing = setInterval(async () => {
        if (S.runIdx >= S.index.runs.length - 1) return stopAll();
        await setRun(S.runIdx + 1);
      }, opts.interval || 1400);
    };
    $('playH').onclick = () => {
      if (S.playing) return stopAll();
      $('playH').textContent = '❚❚';
      const sl = $('hour');
      S.playing = setInterval(() => {
        let h = +sl.value + 6;
        if (h > +sl.max) h = 0;
        sl.value = h; sl.oninput({ target: sl });
      }, 500);
    };

    // refresco automático: si aparece una pasada nueva, se añade
    setInterval(async () => {
      try {
        const idx = await getJSON(`${base}${S.storm}/index.json`);
        if (idx.runs.length && (!S.index || idx.runs[idx.runs.length - 1].id !== S.index.runs[S.index.runs.length - 1].id)) {
          const atLast = S.runIdx === S.index.runs.length - 1;
          S.index = idx;
          $('run').innerHTML = idx.runs.map((r, i) => `<option value="${i}">${fmtT(new Date(r.t))}</option>`).join('');
          if (atLast && !S.playing) setRun(idx.runs.length - 1); else $('run').value = String(S.runIdx);
        }
      } catch (e) { /* sin conexión: ignorar */ }
    }, (opts.refreshMinutes || 15) * 60e3);

    loadStorms().catch((e) => { $('sub').textContent = 'Error cargando datos: ' + e.message; });
    return { map, state: S };
  }

  window.CanaryTracks = { mount };
})();
