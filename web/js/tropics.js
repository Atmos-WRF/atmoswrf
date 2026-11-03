'use strict';
// Hurricane mode (hurricane.py): the Tropics tab, the SST map preview, the surface-low tool, and the cyclones on
// the map -- each one's track colored by its Saffir-Simpson category and an L at its centre (its lowest pressure)
// that opens a card with its category, maximum 1-min sustained wind and lowest pressure.

const CAT_COLORS = { TD: '#5ab4ff', TS: '#3ccf7a', C1: '#f5e13a', C2: '#f5a63a', C3: '#ec5a35', C4: '#e83a8a', C5: '#c86cff' };
const CAT_NAMES = { TD: 'Tropical depression', TS: 'Tropical storm', C1: 'Category 1 hurricane', C2: 'Category 2 hurricane',
  C3: 'Category 3 hurricane', C4: 'Category 4 hurricane', C5: 'Category 5 hurricane' };

const Trop = (() => {
  const num = (id, lo, hi, d) => { const v = parseFloat($(id).value); return isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; };
  const catOf = kt => kt >= 137 ? 'C5' : kt >= 113 ? 'C4' : kt >= 96 ? 'C3' : kt >= 83 ? 'C2' : kt >= 64 ? 'C1' : kt >= 34 ? 'TS' : 'TD';
  let preview = null;            // the SST map shown before a run: { key, raster }
  let cardId = null;             // the cyclone whose card is open

  function apply(s) {
    const h = { ...((S.defaults && S.defaults.settings.hurricane) || {}), ...(s.hurricane || {}) };
    $('hurOn').checked = !!h.on;
    $('hurSeed').value = h.sst_seed ?? 1;
    $('hurOffset').value = h.sst_offset ?? 0;
    $('hurMld').value = h.mld ?? 50;
    $('hurDomain').value = h.domain_km ?? 900;
    $('hurDomainY').value = h.domain_km_y ?? h.domain_km ?? 900;
    $('hurWake').checked = h.wake !== false;
    $('hurAmbient').value = h.ambient ?? 0.3;
    $('hurPush').checked = h.core_pushes !== false;
    $('hurPushEvery').value = h.push_every_min ?? 10;
    $('hurPushUntil').value = h.push_until_kt ?? 64;
    s.cyclones = s.cyclones || [];
    const tr = (s.sounding || {}).tropical;
    if (tr) {
      for (const [id, k] of [['tsSst', 'sst'], ['tsRh', 'rh_mid'], ['tsSteer', 'steer_kt'], ['tsSteerDir', 'steer_dir'],
                             ['tsShear', 'shear_kt'], ['tsShearFrom', 'shear_from']]) if (tr[k] !== undefined) $(id).value = tr[k];
    }
    domainInfo();
    render();
  }

  function read() {
    return { on: $('hurOn').checked, sst_seed: Math.round(num('hurSeed', 1, 2 ** 31 - 1, 1)), sst_offset: num('hurOffset', -4, 3, 0),
      mld: num('hurMld', 10, 150, 50), domain_km: num('hurDomain', 200, 2400, 900),
      domain_km_y: num('hurDomainY', 200, 2400, 900), wake: $('hurWake').checked, ambient: num('hurAmbient', 0, 1, 0.3),
      core_pushes: $('hurPush').checked, push_every_min: num('hurPushEvery', 2, 60, 10), push_until_kt: num('hurPushUntil', 20, 140, 64) };
  }

  const tropicalSpec = () => ({ tropical: { sst: num('tsSst', 24, 32, 29), rh_mid: num('tsRh', 20, 95, 70),
    steer_kt: num('tsSteer', 0, 40, 10), steer_dir: num('tsSteerDir', 0, 359, 280), shear_kt: num('tsShear', 0, 50, 5),
    shear_from: num('tsShearFrom', 0, 359, 270) } });

  async function useSounding() {
    clearSoundingMods();
    await loadSounding(tropicalSpec());
  }

  // everything a hurricane run wants, in one click (the sounding last: it reads the form)
  async function preset() {
    $('hurOn').checked = true;
    $('optCor').checked = true;
    $('physSuite').value = ''; showSuite();
    $('grid').value = '3km';
    $('gridMotion').value = 'mean'; showGridMotion();
    $('frameEvery').value = '300';
    domainInfo();
    readSetupFromForm();
    showDomain(); renderGridNow();
    if ((S.setup.cyclones || []).length) await sstUnderFirstLow(true);   // the air in balance with the sea under the low
    await useSounding();
    $('product').value = 'ir'; S.product = 'ir'; fillPaletteSelect(); showFrame();
    render();
  }

  // the domain: its size on the grid chosen, GPU memory and a rough speed (3 km, 700 km: ~400x real time while a
  // storm organises, ~150x once it is a hurricane, measured on an RTX 5090), and the shape menu kept in step
  function domainInfo() {
    const g = S.defaults && S.defaults.grids[$('grid').value];
    if (!g) return;
    if (!$('hurGrid').options.length)              // the server's grids (they arrive after init)
      for (const [k, gr] of Object.entries(S.defaults.grids))
        if (gr.dx >= 750) $('hurGrid').add(new Option(`${gr.label} (${gr.nz} levels)`, k));
    if ($('hurGrid').value !== $('grid').value) $('hurGrid').value = $('grid').value;
    const W = num('hurDomain', 200, 2400, 900), H = num('hurDomainY', 200, 2400, 900);
    const blocks = km => Math.max(32, Math.ceil(km * 1000 / g.dx / 16) * 16);
    const nx = blocks(W), ny = blocks(H), cells = nx * ny * g.nz;
    const gb = cells * (S.defaults.cell_bytes || 600) / 2 ** 30;
    const rate = (240 * 240 * 45 / cells) * (g.dt / 18);
    const fmt = v => v >= 10 ? Math.round(v) : v.toFixed(1);
    $('hurDomainInfo').textContent = `${nx} × ${ny} cells × ${g.nz} levels on the ${g.label} grid: about ${gb.toFixed(1)} GB of ` +
      `GPU memory, roughly ${fmt(400 * rate)}× real time while it organises and ${fmt(150 * rate)}× as a hurricane` +
      (g.dx < 1000 ? ' (finer than 1 km: very slow for a whole hurricane)' : '');
    const key = `${W}x${H}`;
    $('hurShape').value = [...$('hurShape').options].some(o => o.value === key) ? key : '';
  }

  // the SST map from the server (hurricane.ProceduralSST over the sea around a point), packed like a map product
  async function fetchSst(lat0, lon0, half_km, step_km) {
    const h = read();
    const r = await api('/api/hurricane/sst', { seed: h.sst_seed, offset: h.sst_offset, lat0, lon0, half_km, step_km });
    return r.arrayBuffer();
  }

  async function sstUnderFirstLow(quiet) {
    const lows = S.setup.cyclones || [];
    if (!lows.length) { if (quiet !== true) alert('Place a surface low first (Surface low, below).'); return; }
    const [lat, lon] = map.inv(lows[0].x / 1000, lows[0].y / 1000);
    const { header, data } = unpack(await fetchSst(lat, lon, 200, 40));
    const p = header.patches[0], c = data[p.offset + (p.ny >> 1) * p.nx + (p.nx >> 1)];
    if (!c) { if (quiet !== true) alert('That surface low is over land.'); return; }
    const [scale, offset] = header.codec;
    $('tsSst').value = ((c - offset) / scale).toFixed(1);
  }

  // Show SST map: the preview replaces the map product until switched off (before a run only)
  function previewOn() { return $('hurShowSst').classList.contains('on'); }
  async function togglePreview() {
    $('hurShowSst').classList.toggle('on');
    if (!previewOn()) { preview = null; map.setRaster(null); S.rasterInfo = null; showFrame(); return; }
    await drawPreview();
  }
  async function drawPreview() {
    const h = read(), m = S.setup.map;
    const key = [h.sst_seed, h.sst_offset, m.lat0, m.lon0].join(',');
    try {
      if (!preview || preview.key !== key) {
        const buf = await fetchSst(m.lat0, m.lon0, 2600, 12);
        preview = { key, raster: packedToRaster(buf, Colors.DEFAULT_PALETTES.sst) };
      }
      S.rasterInfo = preview.raster;
      map.setRaster({ patches: preview.raster.patches, smooth: true });
      drawLegend(Colors.DEFAULT_PALETTES.sst, 'sst');
    } catch (e) { alert('SST map: ' + e.message); }
    map.redraw();
  }
  // drawFrame's hook: true while the preview owns the map
  function showingPreview() {
    if (!previewOn()) return false;
    if (runActive()) { $('hurShowSst').classList.remove('on'); preview = null; return false; }
    drawPreview();
    return true;
  }

  async function newSeed() {
    $('hurSeed').value = 1 + Math.floor(Math.random() * 999999);
    if (previewOn()) await drawPreview();
  }

  // the surface-low tool
  async function place(x, y) {
    const low = { x: x * 1000, y: y * 1000, vmax_kt: num('lowVmax', 10, 120, 30), rmw_km: num('lowRmw', 15, 250, 90),
      r0_km: num('lowR0', 100, 1400, 500) };
    if (runActive()) {
      if (!(S.run.settings.hurricane || {}).on) { alert('This run is not in hurricane mode: surface lows go in a run launched with it on.'); return; }
      await api('/api/run/cyclone', low);
    } else {
      if (!$('hurOn').checked) $('hurOn').checked = true;
      S.setup.cyclones = (S.setup.cyclones || []).concat([low]);
    }
    render();
    map.redraw();
  }

  function cyclones() {
    return runActive() || (S.run && S.run.cyclones && S.run.cyclones.length && !S.detached) ? (S.run.cyclones || []) : null;
  }

  function render() {
    const live = cyclones();
    if (live) {
      $('lowList').innerHTML = live.map(c => `<div class="item"><b style="color:${CAT_COLORS[c.cat]}">${c.cat}</b>
        cyclone ${c.id}: ${Math.round(c.vmax_kt)} kt, ${c.pmin ? c.pmin.toFixed(0) + ' hPa' : '–'}${c.active ? '' : ' (left the domain)'}</div>`).join('');
      return;
    }
    const lows = (S.setup && S.setup.cyclones) || [];
    $('lowList').innerHTML = lows.map((c, i) => `<div class="item">L ${i + 1}: ${c.vmax_kt} kt at ${c.rmw_km} km, circulation ${c.r0_km} km
      <button data-lowremove="${i}" title="Remove this surface low">✕</button></div>`).join('');
  }

  // the cyclones on the map: tracks colored by category, dots every 6 h, the L at each centre
  function draw(g, m) {
    const live = cyclones();
    g.save();
    if (live) {
      const fr = S.frames.length ? S.frames[S.frameIdx < 0 ? S.frames.length - 1 : Math.min(S.frameIdx, S.frames.length - 1)] : null;
      const tNow = fr && !S.live ? fr.t : Infinity;
      for (const c of live) {
        const pts = c.track.filter(p => p[0] <= tNow + 1);
        g.lineCap = 'round'; g.lineJoin = 'round';
        if (pts.length > 1) {
          g.strokeStyle = 'rgba(0,0,0,.55)'; g.lineWidth = 5;
          g.beginPath();
          pts.forEach((p, i) => { const [x, y] = m.toScreen(p[1] / 1000, p[2] / 1000); i ? g.lineTo(x, y) : g.moveTo(x, y); });
          g.stroke();
          g.lineWidth = 2.5;
          for (let i = 0; i < pts.length - 1; i++) {
            const [a, b] = m.toScreen(pts[i][1] / 1000, pts[i][2] / 1000), [cx, cy] = m.toScreen(pts[i + 1][1] / 1000, pts[i + 1][2] / 1000);
            g.strokeStyle = CAT_COLORS[catOf(pts[i + 1][4])];
            g.beginPath(); g.moveTo(a, b); g.lineTo(cx, cy); g.stroke();
          }
          for (const p of pts) {
            if (Math.round(p[0]) % 21600 >= 60) continue;
            const [x, y] = m.toScreen(p[1] / 1000, p[2] / 1000);
            g.fillStyle = CAT_COLORS[catOf(p[4])]; g.strokeStyle = '#000'; g.lineWidth = 1;
            g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fill(); g.stroke();
          }
        }
        const last = pts.length ? pts[pts.length - 1] : null;
        const at = tNow === Infinity ? [c.x, c.y, c.pmin, c.vmax_kt] : last ? [last[1], last[2], last[3], last[4]] : null;
        if (at) lowMark(g, m, at[0], at[1], at[2], at[3], c.active);
      }
    } else {
      // the domain the run starts with, centred on the first surface low
      const first = ((S.setup && S.setup.cyclones) || [])[0];
      if (first && $('hurOn').checked) {
        const W = num('hurDomain', 200, 2400, 900) / 2, H = num('hurDomainY', 200, 2400, 900) / 2;
        const [a, b] = m.toScreen(first.x / 1000 - W, first.y / 1000 + H), [cx, cy] = m.toScreen(first.x / 1000 + W, first.y / 1000 - H);
        g.strokeStyle = 'rgba(120,180,255,.8)'; g.setLineDash([8, 6]); g.lineWidth = 1.5;
        g.strokeRect(a, b, cx - a, cy - b); g.setLineDash([]);
        g.font = '11px Segoe UI'; g.fillStyle = 'rgba(170,205,255,.95)';
        g.fillText(`hurricane domain ${2 * W} × ${2 * H} km`, a + 4, b + 13);
      }
      for (const c of (S.setup && S.setup.cyclones) || []) {
        const [x, y] = m.toScreen(c.x / 1000, c.y / 1000);
        g.strokeStyle = 'rgba(230,60,60,.55)'; g.setLineDash([4, 4]); g.lineWidth = 1;
        for (const r of [c.rmw_km, c.r0_km]) { g.beginPath(); g.arc(x, y, r * m.scale, 0, 7); g.stroke(); }
        g.setLineDash([]);
        lowMark(g, m, c.x, c.y, null, c.vmax_kt, true);
      }
    }
    g.restore();
  }

  function lowMark(g, m, X, Y, p, kt, active) {
    const [x, y] = m.toScreen(X / 1000, Y / 1000), cat = catOf(kt || 0);
    g.font = '800 22px Inter, Segoe UI, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.lineWidth = 4; g.strokeStyle = 'rgba(0,0,0,.75)'; g.strokeText('L', x, y);
    g.fillStyle = active ? '#ff3b30' : '#a0a0a0'; g.fillText('L', x, y);
    const label = p ? `${p.toFixed(0)} hPa · ${cat} ${Math.round(kt)} kt` : `${cat} ${Math.round(kt)} kt`;
    g.font = '600 11px Inter, Segoe UI, sans-serif'; g.textAlign = 'left';
    const w = g.measureText(label).width;
    g.fillStyle = 'rgba(10,14,20,.8)'; g.fillRect(x + 12, y - 9, w + 10, 17);
    g.fillStyle = CAT_COLORS[cat]; g.fillText(label, x + 17, y);
    g.textBaseline = 'alphabetic';
  }

  // the cyclone (or planned low) under a click, if any: { c, live }
  function hit(x, y) {
    const live = cyclones();
    const list = live ? live.map(c => ({ c, X: c.x, Y: c.y })) : ((S.setup && S.setup.cyclones) || []).map((c, i) => ({ c: { ...c, i }, X: c.x, Y: c.y }));
    let best = null, bd = 1e9;
    for (const q of list) {
      const [a, b] = map.toScreen(q.X / 1000, q.Y / 1000), [px, py] = map.toScreen(x, y);
      const d = Math.hypot(a - px, b - py);
      if (d < 18 && d < bd) { bd = d; best = { c: q.c, live: !!live }; }
    }
    return best;
  }

  function card(h) {
    const el = $('cycCard');
    if (!h) { el.classList.add('hidden'); cardId = null; return; }
    cardId = h.live ? h.c.id : null;
    fillCard(h);
    const [x, y] = map.toScreen(h.c.x / 1000, h.c.y / 1000);
    el.style.left = Math.min(x + 22, el.parentElement.clientWidth - 300) + 'px';
    el.style.top = Math.max(y - 30, 60) + 'px';
    el.classList.remove('hidden');
  }
  function fillCard(h) {
    const c = h.c, el = $('cycCard');
    const kt = h.live ? c.vmax_kt : c.vmax_kt, cat = catOf(kt);
    const row = (a, b) => `<div class="cyc-row"><span>${a}</span><span>${b}</span></div>`;
    const lines = h.live ? [
      row('Max 1-min sustained wind', `${Math.round(kt)} kt · ${Math.round(kt * 1.15078)} mph · ${(kt * 0.514444).toFixed(0)} m/s`),
      row('Lowest pressure', c.pmin ? `${c.pmin.toFixed(1)} hPa` : '–'),
      row('Radius of max wind', `${c.rmw_km.toFixed(0)} km`),
      row('Moving', `${(Math.hypot(...c.motion) / 0.514444).toFixed(0)} kt toward ${Math.round((Math.atan2(c.motion[0], c.motion[1]) * 180 / Math.PI + 360) % 360)}°`),
      row('Core updraft pushes', c.pushes),
    ] : [
      row('Starting max wind', `${kt} kt at ${c.rmw_km} km`), row('Circulation', `${c.r0_km} km`),
      '<div class="cyc-row"><span>Planned for the next run</span><button data-lowremove="' + c.i + '">Remove</button></div>',
    ];
    el.innerHTML = `<div class="cyc-head"><span class="cyc-badge" style="background:${CAT_COLORS[cat]}">${cat}</span>
      ${h.live ? 'Cyclone ' + c.id : 'Surface low'}<button class="cyc-x" title="Close">✕</button></div>
      <div class="cyc-big">${CAT_NAMES[cat]}</div>${lines.join('')}`;
  }
  // a live card follows its cyclone from poll to poll
  function onPoll() {
    render();
    if (cardId === null) return;
    const c = (cyclones() || []).find(q => q.id === cardId);
    if (c) card({ c, live: true }); else card(null);
  }

  function init() {
    $('hurPreset').onclick = preset;
    $('hurGrid').onchange = () => {
      $('grid').value = $('hurGrid').value;
      $('grid').dispatchEvent(new Event('change'));
      domainInfo();
    };
    $('grid').addEventListener('change', domainInfo);
    $('hurShape').onchange = () => {
      const v = $('hurShape').value;
      if (!v) return;
      const [w, h] = v.split('x');
      $('hurDomain').value = w; $('hurDomainY').value = h;
      domainInfo(); map.redraw();
    };
    for (const id of ['hurDomain', 'hurDomainY']) $(id).oninput = () => { domainInfo(); map.redraw(); };
    $('hurOn').addEventListener('change', () => map.redraw());
    $('tsMake').onclick = useSounding;
    $('tsFromLow').onclick = sstUnderFirstLow;
    $('hurShowSst').onclick = togglePreview;
    $('hurNewSeed').onclick = newSeed;
    for (const id of ['hurSeed', 'hurOffset']) $(id).onchange = () => { if (previewOn()) drawPreview(); };
    $('toolLow').onclick = () => setTool('tclow');
    // seed sizes: (RMW, circulation) km; Large needs room, so the domain grows to hold it
    const SIZES = { midget: [40, 250], average: [90, 500], large: [140, 800] };
    $('lowSize').onchange = () => {
      const v = SIZES[$('lowSize').value];
      if (!v) return;
      $('lowRmw').value = v[0]; $('lowR0').value = v[1];
      const need = Math.ceil(2 * v[1] / 1.1 / 100) * 100;      // the run cuts a circulation past 1.1 x the half-domain
      for (const id of ['hurDomain', 'hurDomainY']) if (num(id, 200, 2400, 900) < need) $(id).value = Math.min(need, 2400);
      domainInfo(); map.redraw();
    };
    for (const id of ['lowRmw', 'lowR0']) $(id).addEventListener('input', () => { $('lowSize').value = ''; });
    const remove = e => {
      const b = e.target.closest('button[data-lowremove]');
      if (!b) return false;
      S.setup.cyclones.splice(+b.dataset.lowremove, 1);
      card(null); render(); map.redraw();
      return true;
    };
    $('lowList').addEventListener('click', remove);
    $('cycCard').addEventListener('click', e => {
      if (remove(e)) return;
      if (e.target.closest('.cyc-x')) card(null);
    });
  }

  return { init, apply, read, draw, hit, card, place, onPoll, showingPreview, render };
})();
window.Trop = Trop;
