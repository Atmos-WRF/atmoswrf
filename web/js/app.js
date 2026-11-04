'use strict';
// Atmos-WRF front end: setup, run control, map products, radars, replay.

const $ = id => document.getElementById(id);
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = r.status + '';
    try { msg = (await r.json()).error || msg; } catch (e) { /* binary */ }
    throw new Error(msg);
  }
  return r;
};
const getJSON = async (path, body) => (await api(path, body)).json();

const GRID_PRODUCTS = [
  ['cref', 'Composite reflectivity'], ['ref1km', '1 km AGL reflectivity'], ['reflow', 'Lowest-level reflectivity'],
  ['thp', 'Surface θ′ (cold pools)'], ['wmax', 'Max updraft'], ['uh25', '2–5 km updraft helicity'],
  ['wspd', 'Lowest-level wind'], ['vort', 'Lowest-level vorticity'], ['rain', 'Accumulated rain'],
  ['uhtrack', 'Storm tracks (run-max UH)'], ['windtrack', 'Wind swaths (run-max)'],
  ['ir', 'Satellite IR (simulated)'], ['wind10', '10 m wind (1-min sustained)'], ['slp', 'Sea-level pressure'],
  ['sst', 'Sea-surface temperature'],
];
const RADAR_PRODUCTS = [
  ['REF', 'Radar: base reflectivity 0.5°'], ['VEL', 'Radar: base velocity'], ['SRV', 'Radar: storm-relative velocity'],
  ['ZDR', 'Radar: differential reflectivity'], ['RHO', 'Radar: correlation coefficient'], ['SW', 'Radar: spectrum width'],
];
const LEGEND_RANGE = {
  dbz: [-10, 80, 10], vel: [-60, 60, 20], zdr: [-2, 6, 1], cc: [0.2, 1.05, 0.1], sw: [0, 20, 4], thp: [-12, 5, 3],
  w: [0, 80, 10], uh: [0, 2000, 250], wspd: [0, 55, 10], vort: [-0.05, 0.25, 0.05], rain: [0, 8, 1],
  ir: [-90, 40, 10], sst: [20, 32, 2], slp: [900, 1020, 20], hwind: [0, 75, 15],
};

const S = {
  defaults: null, setup: null, snd: null, run: null, frames: [], frameIdx: -1, live: true,
  product: 'cref', radar: null, sites: [], cities: [], palettes: [], palCache: {},
  tool: 'pan', soundingFiles: [], lastFrameCount: -1, playing: false, mapKey: '',
  hover: null, rasterInfo: null, ppiInfo: null, fetching: false, frameAgain: false,
};
let map;

// ------------------------------------------------------------------ boot
async function boot() {
  map = new MapView($('map'));
  map.overlay = drawOverlay;
  map.onClick = onMapClick;
  map.onHover = onHover;
  map.onDragRect = onZoneDrawn;
  map.onDragPath = onBrushPainted;
  map.onDblClick = () => { if (S.tool === 'front') Fronts.finish(); };
  map.onPress = (x, y, e) => S.tool === 'pan' ? Fronts.press(x, y, e) : S.tool === 'move' ? moveGrab(x, y) : null;  // drag the L
  map.grabbable = (x, y) => (S.tool === 'pan' && !!Fronts.lowAt(x, y)) || (S.tool === 'move' && !!moveHit(x, y));
  bindUI();
  const d = await getJSON('/api/defaults');
  S.defaults = d;
  try {                               // synoptic mode's scenarios (synoptic.py)
    const ss = await getJSON('/api/synoptic/status');
    S.synScenarios = ss.scenarios;
    Syn.init(ss.scenarios);
  } catch (e) { Syn.init(null); }
  if (d.backend) {                    // what runs ArWen: the NVIDIA card or the processor (backend.py)
    const sub = document.querySelector('.brand .sub');
    sub.textContent = d.cpu ? 'ArWen WRF core · CPU' : 'ArWen WRF core · GPU';
    sub.classList.toggle('cpu', !!d.cpu);
    sub.title = d.backend + (d.cpu ? ': no NVIDIA card, so the model runs on the processor (much slower: use 2-3 km grids)' : '');
  }
  showPollLinks(d.poll);
  S.setup = JSON.parse(JSON.stringify(d.settings));
  S.setup.persistence = 0;               // pure physics by default: the staying-power lift is opt-in
  S.setup.boundaries = S.setup.boundaries || [];
  S.setup.nudgers = S.setup.nudgers || [];
  Fronts.init(d.boundary_kinds);
  for (const [k, g] of Object.entries(d.grids)) $('grid').add(new Option(`${g.label}  (dt ${g.dt} s, ${g.nz} levels)`, k));
  fillPhysics(d);
  for (const [k, l] of GRID_PRODUCTS.concat(RADAR_PRODUCTS)) $('product').add(new Option(l, k));
  S.soundingFiles = await getJSON('/api/soundings');
  fillSoundingList('');
  S.palettes = await getJSON('/api/palettes');
  const st = await getJSON('/api/state');
  if (st.status && st.status !== 'none') adoptRun(st);
  applySetupToForm();
  await loadSounding(S.setup.sounding);
  await reloadMap();
  setInterval(poll, 1000);
  setInterval(libraryPoll, 1500);
  libraryPoll(true);
  poll();
}

function adoptRun(st) {
  S.run = st;
  S.setup = JSON.parse(JSON.stringify(st.settings));
  if (S.setup.persistence === undefined) S.setup.persistence = 0;
  S.setup.boundaries = S.setup.boundaries || [];
  S.setup.nudgers = S.setup.nudgers || [];
}

// a saved setup (or one someone sent) becomes the setup for the next run (presets.js); no run is going
async function useSetup(settings) {
  S.setup = { ...JSON.parse(JSON.stringify(S.defaults.settings)), ...JSON.parse(JSON.stringify(settings)) };
  if (S.setup.persistence === undefined) S.setup.persistence = 0;
  for (const k of ['storms', 'zones', 'radars', 'boundaries', 'nudgers', 'cyclones']) S.setup[k] = S.setup[k] || [];
  applySetupToForm();
  await loadSounding(S.setup.sounding);
  await recenterSetup();
  fillRadarSelect();
  map.redraw();
}

// ------------------------------------------------------------------ setup form
function applySetupToForm() {
  const s = S.setup;
  $('lat0').value = s.map.lat0; $('lon0').value = s.map.lon0;
  $('startTime').value = s.start_time.replace('Z', '').slice(0, 16);
  $('grid').value = s.grid;
  const dom = s.domain || {};
  $('domainMode').value = dom.mode === 'fixed' ? 'fixed' : 'unlimited';
  $('domainSize').value = dom.size_km || 160;
  $('domainMax').value = dom.max_km > 0 ? dom.max_km : '';
  const rf = s.refine || {};
  $('refineRatio').value = String(rf.ratio || 0);
  $('refineKm').value = rf.size_km || 40;
  $('refineMax').value = rf.max || 2;
  $('refineInner').value = String(rf.inner_ratio || 0);
  $('refineInnerKm').value = rf.inner_km || 16;
  $('refineRow').classList.toggle('hidden', !(+$('refineRatio').value));
  const tr = s.terrain || {};
  $('terrainKind').value = tr.on ? (tr.kind || 'real') : '';
  $('terrainScale').value = tr.scale ?? 1;
  $('terrainH').value = tr.height_m ?? 800;
  $('terrainA').value = tr.half_width_km ?? 10;
  $('terrainX').value = tr.x_km ?? 0;
  $('terrainY').value = tr.y_km ?? 0;
  $('terrainAngle').value = tr.angle_deg ?? 0;
  $('terrainLen').value = tr.length_km ?? 0;
  showTerrain();
  showDomain();
  $('physSuite').value = s.physics.suite || '';
  const mpWant = String(s.physics.mp_physics || 28);
  if (![...$('physMp').options].some(o => o.value === mpWant))    // a setup with a scheme no longer listed
    $('physMp').add(new Option(`Scheme ${mpWant} (no longer listed)`, mpWant));
  $('physMp').value = mpWant;
  $('soilMoist').value = s.physics.soil_moisture ?? 0.25;
  $('vegFrac').value = s.physics.vegetation ?? 50;
  showSuite();
  const gm = s.grid_motion || 'rm';
  $('gridMotion').value = Array.isArray(gm) ? 'custom' : gm;
  if (Array.isArray(gm)) { $('gridU').value = gm[0]; $('gridV').value = gm[1]; }
  $('gridFollowMin').value = String(s.grid_follow_min || 10);
  showGridMotion();
  $('optCor').checked = s.physics.coriolis !== false;
  $('optDrag').checked = s.physics.surface_drag !== false;
  $('optLes').checked = !!s.physics.les;
  $('optSteady').checked = !!(s.engine && s.engine.steady_environment);
  $('persist').value = s.persistence; $('persistVal').textContent = (+s.persistence).toFixed(2);
  $('speed').value = String(s.speed_limit || 0);
  $('frameEvery').value = String(s.frame_interval || 60);
  // the sounding's own changes, so reloading it (loadSounding reads them from the form) keeps them
  $('lsW').value = (s.large_scale || {}).w_cms || 0;
  $('lsTop').value = (s.large_scale || {}).top_km || 10;
  const heat = (s.sounding || {}).heating || {};
  $('heatHours').value = heat.hours || 0;
  $('heatFlux').value = heat.flux || 300;
  const mix = (s.sounding || {}).mix;
  $('mixOn').checked = !!(mix && mix.depth);
  if (mix && mix.depth) { $('mixDepth').value = mix.depth; $('mixHeat').value = mix.heat || 0; }
  const moist = (s.sounding || {}).moist || {};
  $('moistDtd').value = moist.dtd || 0;
  $('moistDepth').value = moist.depth || 100;
  const mid = (s.sounding || {}).mid_moist || {};
  $('midRh').value = mid.rh || 0;
  $('midBottom').value = mid.bottom || 750;
  $('midTop').value = mid.top || 450;
  const cap = (s.sounding || {}).cap || {};
  $('capCin').value = cap.cin || 0;
  $('capBase').value = cap.base || 100;
  $('windScale').value = (s.sounding || {}).wind_scale || 1;
  $('iceCap').checked = (s.sounding || {}).ice_cap !== false;
  const jet = (s.sounding || {}).llj || {};
  $('lljSpeed').value = jet.speed || 0;
  $('lljHeight').value = jet.height || 750;
  const aloft = (s.sounding || {}).aloft || {};
  $('aloftSpeed').value = aloft.speed || 0;
  $('aloftBase').value = (aloft.base || 5000) / 1000;
  applyAutoToForm();
  Syn.apply(s.synoptic);
  Trop.apply(s);
  renderLists();
}

function readSetupFromForm() {
  const s = S.setup;
  // a new outbreak every run (the server draws a seed, kept in settings.json), unless asked to repeat the last
  if (!$('keepSeed').checked) delete s.seed;
  s.map = { lat0: +$('lat0').value, lon0: +$('lon0').value };
  s.start_time = ($('startTime').value || '2011-04-27T20:00') + ':00Z';
  s.grid = $('grid').value;
  s.domain = { mode: $('domainMode').value, size_km: domainKm(), max_km: domainMaxKm() };
  s.grid_motion = gridMotionFromForm();
  s.grid_follow_min = +$('gridFollowMin').value;
  s.large_scale = { w_cms: Math.max(0, +$('lsW').value || 0), top_km: Math.min(16, Math.max(4, +$('lsTop').value || 10)) };
  s.refine = { ratio: +$('refineRatio').value, size_km: Math.min(400, Math.max(4, +$('refineKm').value || 40)),
    max: Math.min(12, Math.max(1, Math.round(+$('refineMax').value || 2))),
    inner_ratio: +$('refineInner').value, inner_km: Math.min(200, Math.max(2, +$('refineInnerKm').value || 16)) };
  const tk = $('terrainKind').value;
  s.terrain = tk ? { on: true, kind: tk, scale: Math.min(5, Math.max(0, +$('terrainScale').value || 0)),
    height_m: Math.min(4000, Math.max(0, +$('terrainH').value || 0)),
    half_width_km: Math.min(100, Math.max(1, +$('terrainA').value || 10)),
    x_km: +$('terrainX').value || 0, y_km: +$('terrainY').value || 0,
    angle_deg: +$('terrainAngle').value || 0, length_km: Math.max(0, +$('terrainLen').value || 0) } : { on: false };
  s.physics = { coriolis: $('optCor').checked, surface_drag: $('optDrag').checked, z0: 0.1,
    mp_physics: +$('physMp').value, suite: $('physSuite').value || null, les: $('optLes').checked,
    soil_moisture: +$('soilMoist').value, vegetation: +$('vegFrac').value };
  s.engine = { steady_environment: $('optSteady').checked };
  s.persistence = +$('persist').value;
  s.speed_limit = +$('speed').value;
  s.frame_interval = +$('frameEvery').value;
  s.auto_storms = readAutoForm();
  s.synoptic = Syn.read();
  s.hurricane = Trop.read();
  s.cyclones = s.cyclones || [];
  return s;
}

// a fixed-size domain: one square that follows its storms but never grows (engine.Engine.fixed);
// an unlimited one whose patches grow up to a largest size (engine.Engine.cap), 0 = no limit
const domainKm = () => { const v = parseFloat($('domainSize').value); return isFinite(v) ? Math.min(1000, Math.max(48, v)) : 160; };
const domainMaxKm = () => { const v = parseFloat($('domainMax').value); return isFinite(v) && v > 0 ? Math.min(1000, Math.max(48, v)) : 0; };

function domainCells(km, g) {
  // the engine rounds the side up to whole 16-cell blocks; cell_bytes is its GPU memory per cell
  const n = Math.max(32, Math.ceil(km * 1000 / g.dx / 16) * 16);
  const gb = n * n * g.nz * (S.defaults.cell_bytes || 600) / 2 ** 30;
  return `${n}×${n} cells at ${g.label}: about ${gb < 1 ? gb.toFixed(2) : gb.toFixed(1)} GB of GPU memory`;
}

// the grid's motion (CM1's umove/vmove): a named motion, "follow", or [umove, vmove] in m/s
function gridMotionFromForm() {
  const v = $('gridMotion').value;
  if (v !== 'custom') return v;
  const n = id => { const x = parseFloat($(id).value); return Number.isFinite(x) ? Math.max(-60, Math.min(60, x)) : 0; };
  return [n('gridU'), n('gridV')];
}

function showGridMotion() {
  const v = $('gridMotion').value;
  $('gridFollowRow').classList.toggle('hidden', v !== 'follow');
  $('gridCustomRow').classList.toggle('hidden', v !== 'custom');
  if (v === 'custom' && $('gridU').value === '' && $('gridV').value === '') {
    // start from the motion the grid has now, or the sounding's Bunkers right mover
    const m = (runActive() && S.run.grid_motion) || (S.snd && S.snd.diag && S.snd.diag.rm) || [0, 0];
    $('gridU').value = (+m[0]).toFixed(1);
    $('gridV').value = (+m[1]).toFixed(1);
  }
  renderGridNow();
}

// what the running run's grid does now, and the button that hands it the motion chosen above
function renderGridNow() {
  const st = runActive() && !S.detached ? S.run : null;
  $('gridNowRow').classList.toggle('hidden', !st || !st.grid_motion);
  if (!st || !st.grid_motion) return;
  const [u, v] = st.grid_motion;
  const sp = Math.hypot(u, v), dir = Math.round((Math.atan2(u, v) * 180 / Math.PI + 360) % 360);
  let txt = `This run's grid: ${sp.toFixed(1)} m/s toward ${dir}° (umove ${u.toFixed(1)}, vmove ${v.toFixed(1)})`;
  const f = st.follow;
  if (st.ground_fixed) {
    const dm = st.domain_motion;
    txt += ', fixed to the ground by its physics suite; ' + (dm
      ? `the domain moves instead at ${Math.hypot(dm[0], dm[1]).toFixed(1)} m/s toward ${Math.round((Math.atan2(dm[0], dm[1]) * 180 / Math.PI + 360) % 360)}°.`
      : 'the domain follows its storms.');
  }
  else if (f) {
    const next = Math.max(0, f.next - st.t);
    txt += f.measured ? `, following the storms (measured ${Math.hypot(...f.measured).toFixed(1)} m/s toward ` +
      `${Math.round((Math.atan2(f.measured[0], f.measured[1]) * 180 / Math.PI + 360) % 360)}°)` :
      ', following the storms (measuring: needs 15 min of a deep updraft)';
    txt += next > 0 ? `; next check in ${Math.ceil(next / 60)} min.` : '; checking at the next re-fit.';
  } else txt += '; a domain at its size limit moves with it.';
  setHTML('gridNow', esc(txt));
  $('gridApply').disabled = false;          // a suite's run: sets how its domain moves
}

// ------------------------------------------------------------------ panel helpers
// Each section of a tab (an h3 and what follows it, up to the next h3) folds away with a click on its heading;
// which ones are folded is remembered in this browser.
function makeCollapsible() {
  let folded = new Set();
  try { folded = new Set(JSON.parse(localStorage.getItem('arht.folded') || '[]')); } catch (e) { /* none */ }
  const save = () => { try { localStorage.setItem('arht.folded', JSON.stringify([...folded])); } catch (e) { /* private window */ } };
  for (const h of document.querySelectorAll('.panel-main .tab h3')) {
    const body = document.createElement('div');
    body.className = 'sec-body';
    let n = h.nextSibling;
    while (n && !(n.nodeType === 1 && n.tagName === 'H3')) { const next = n.nextSibling; body.appendChild(n); n = next; }
    h.after(body);
    const key = h.closest('.tab').id + '/' + h.textContent.trim();
    h.classList.add('fold');
    h.tabIndex = 0;
    h.title = 'Click to fold or unfold this section';
    const set = on => { h.classList.toggle('folded', on); body.classList.toggle('hidden', on); };
    set(folded.has(key));
    const toggle = () => {
      const on = !h.classList.contains('folded');
      set(on);
      if (on) folded.add(key); else folded.delete(key);
      save();
    };
    h.addEventListener('click', toggle);
    h.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  }
}

// One line over Launch run: what the next run will be, each part a link to its tab (warnings in amber).
function setupSummary() {
  const el = $('setupSummary');
  if (!el || !S.setup) return;
  const s = S.setup, parts = [];
  const add = (tab, text, warn) => parts.push(`<a data-go="${tab}"${warn ? ' class="warn"' : ''}>${text}</a>`);
  const env = $('envSource').value;
  if ($('hurOn').checked) add('tropics', 'Hurricane mode');
  else if (env === 'sounding') add('atmos', ($('sndTitle').textContent || 'Sounding').replace(/\s+/g, ' ').trim().slice(0, 30));
  else add(env === 'sim' ? 'weather' : 'atmos', { hrrr: 'HRRR soundings', era5: 'ERA5 reanalysis', sim: 'Simulated weather' }[env] || env);
  const tk = $('terrainKind').value;
  add('domain', $('grid').value.replace(/^([\d.]+)(k?m)$/, '$1 $2') + (tk ? ` · ${tk === 'real' ? 'real terrain' : tk}` : ''));
  const nStorms = (s.storms || []).length, nZones = (s.zones || []).length;
  const storms = [];
  if ($('olOn').checked) storms.push('auto outbreak');
  if (nStorms) storms.push(`${nStorms} cell${nStorms > 1 ? 's' : ''}`);
  if (nZones) storms.push(`${nZones} zone${nZones > 1 ? 's' : ''}`);
  const fed = env !== 'sounding' || $('hurOn').checked;          // the weather / sea starts its own storms
  add('storms', storms.length ? storms.join(' + ') : fed ? 'storms from the weather' : 'no storms yet', !storms.length && !fed);
  const nb = (s.boundaries || []).length;
  if (nb) add('fronts', `${nb} front${nb > 1 ? 's' : ''}`);
  const nr = (s.radars || []).length;
  add('radars', nr ? `${nr} radar${nr > 1 ? 's' : ''}` : 'no radar', !nr);
  add('physics', ($('physMp').selectedOptions[0]?.text || '').split(/[(,]/)[0].trim());
  const html = parts.join('<span class="sep">·</span>');
  if (el.innerHTML !== html) el.innerHTML = html;
}

async function resetSetup() {
  if (!confirm('Reset every setting to the defaults for the next run?\n\nThe sounding, place, grid, physics, storms, fronts and radars all go back to how the viewer starts. A run that is going is not touched. (Save the setup first if you want to keep it.)')) return;
  await useSetup(S.defaults.settings);
  setupSummary();
}

function showTerrain() {
  const k = $('terrainKind').value;
  $('terrainRealRow').classList.toggle('hidden', k !== 'real');
  $('terrainIdealRow').classList.toggle('hidden', k !== 'ridge' && k !== 'hill');
  $('terrainAngleLbl').classList.toggle('hidden', k !== 'ridge');
  $('terrainLenLbl').classList.toggle('hidden', k !== 'ridge');
}

function showDomain() {
  const fixed = $('domainMode').value === 'fixed';
  $('domainSizeRow').classList.toggle('hidden', !fixed);
  $('domainMaxRow').classList.toggle('hidden', fixed);
  const g = S.defaults && S.defaults.grids[$('grid').value];
  const cap = domainMaxKm();
  if (g) {
    $('domainMem').textContent = domainCells(domainKm(), g);
    $('domainMaxMem').textContent = cap ? 'Each patch at most ' + domainCells(cap, g) : '';
  }
  $('domainHint').textContent = fixed
    ? 'One fixed square that follows its storms. Storms start only inside it.'
    : cap ? `Unlimited, with storm areas up to ${cap} km wide.`
    : 'Unlimited: storms can go anywhere on the map.';
}

// ------------------------------------------------------------------ library
// Soundings and color tables on this computer (library.py): its folders, the computer scan and
// imported files.  The server looks through the folders in the background; the lists are
// reloaded when it is done.
let libBusy = true;                   // the server starts by looking through its folders
let libCount = -1;                    // soundings + color tables last shown
const esc = v => String(v).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function refreshLibraryLists() {
  S.soundingFiles = await getJSON('/api/soundings');
  fillSoundingList($('sndFilter').value);
  S.palettes = await getJSON('/api/palettes');
  fillPaletteSelect();
}

async function libraryPoll(force) {
  if (!force && !libBusy && !$('tab-files').classList.contains('on')) return;
  let d;
  try { d = await getJSON('/api/library'); } catch (e) { return; }
  const busy = d.status.state !== 'idle';
  // a look-through just finished, or (first one on a new PC) more folders are done
  if ((libBusy && !busy) || (busy && d.soundings + d.palettes !== libCount)) await refreshLibraryLists();
  libCount = d.soundings + d.palettes;
  libBusy = busy;
  renderLibrary(d);
}

function renderLibrary(d) {
  const st = d.status;
  $('libStatus').textContent = st.state === 'scanning'
    ? `Scanning this computer… ${st.dirs.toLocaleString()} folders, ${st.found.toLocaleString()} files found`
    : st.state === 'indexing' ? 'Looking through the library folders…'
    : `${d.soundings.toLocaleString()} soundings and ${d.palettes.toLocaleString()} color tables` +
      (/^(Scanned|Stopped)/.test(st.message) ? ` · ${st.message}` : '');
  $('libScan').textContent = st.state === 'scanning' ? 'Stop the scan' : 'Scan this computer';
  const tag = { app: 'built-in', yours: 'added', usual: 'usual place', found: 'found by the scan' };
  const shown = d.roots.filter(r => r.origin === 'yours' || (r.exists && (r.origin === 'app' || r.soundings || r.palettes)));
  setHTML('libFolders', shown.map(r =>
    `<div class="item"><div><b>${esc(r.label)}</b> <span class="meta">${tag[r.origin]} · ${r.soundings} soundings · ` +
    `${r.palettes} color tables${r.exists ? '' : ' · folder missing'}<br>${esc(r.path)}</span></div>` +
    (r.origin === 'yours' || r.origin === 'found' ? `<button data-libremove="${esc(r.path)}">Remove</button>` : '') +
    '</div>').join(''));
}

async function libraryCall(path, body) {
  try { renderLibrary(await getJSON(path, body)); libBusy = true; }
  catch (e) { alert('Library: ' + e.message); }
}

// copy opened files into the library (data/imported) so runs and later sessions find them
async function importFiles(kind, files) {
  const done = [], failed = [];
  for (const f of files) {
    try {
      if (f.size > 3e6) throw new Error('larger than 3 MB');
      done.push(await getJSON('/api/library/import', { kind, name: f.name, text: await f.text() }));
    } catch (e) { failed.push(`${f.name}: ${e.message}`); }
  }
  if (failed.length) alert('Not imported:\n' + failed.join('\n'));
  await refreshLibraryLists();
  libraryPoll(true);
  return done;
}

// ------------------------------------------------------------------ OutbreakLab mode
// storms form on their own from the setup (sim._warm_sector_step, boundaries, _start_storm)
const runActive = () => !!(S.run && !['stopped', 'error'].includes(S.run.status));

function autoCfg() {                  // the run's while it runs, else the setup's
  const a = runActive() ? S.run.settings && S.run.settings.auto_storms : S.setup && S.setup.auto_storms;
  return { ...((S.defaults && S.defaults.settings.auto_storms) || {}), ...(a || {}) };
}

function applyAutoToForm() {
  const a = autoCfg();
  $('olOn').checked = !!a.on; $('olTowers').checked = a.towers !== false; $('olMergers').checked = a.mergers !== false;
  $('olRate').value = a.rate; $('olArea').value = a.area_km;
  $('olStart').value = a.start_min; $('olEnd').value = a.end_min;
  $('olPush').value = a.push; $('olRadius').value = a.radius_km;
  $('olTexture').value = a.texture === undefined ? 1 : a.texture;
}

function readAutoForm() {
  const num = (id, lo, hi, d) => { const v = parseFloat($(id).value); return isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; };
  return { on: $('olOn').checked, towers: $('olTowers').checked, mergers: $('olMergers').checked, rate: num('olRate', 0, 50, 1.2),
           area_km: num('olArea', 50, 3000, 400), start_min: num('olStart', 0, 1e5, 15), end_min: num('olEnd', 0, 1e5, 270),
           push: num('olPush', 3, 20, 10), radius_km: num('olRadius', 4, 20, 12), texture: num('olTexture', 0, 3, 1) };
}

async function onAutoChange() {
  const a = readAutoForm();
  try {
    if (runActive()) await api('/api/run/auto', a);
    else S.setup.auto_storms = a;
  } catch (e) { alert('OutbreakLab mode: ' + e.message); }
  map.redraw();
}

// ------------------------------------------------------------------ physics menu
function fillPhysics(d) {
  const mp = $('physMp');
  for (const m of d.microphysics || []) mp.add(new Option(m.label, m.id));
  const sel = $('physSuite');
  sel.add(new Option('Built-in core', ''));
  const groups = {};
  for (const s of d.suites || []) (groups[s.maturity || 'other'] = groups[s.maturity || 'other'] || []).push(s);
  const order = ['wrf-matched-run', 'wrf-matched-run-candidate', 'supported', 'implemented-unverified', 'experimental-runtime'];
  const names = { 'wrf-matched-run': 'ArWen suites: matched to WRF', 'wrf-matched-run-candidate': 'ArWen suites: WRF-match candidates',
    'supported': 'ArWen suites: supported', 'implemented-unverified': 'ArWen suites: implemented, unverified',
    'experimental-runtime': 'ArWen suites: experimental' };
  for (const key of order.concat(Object.keys(groups).filter(k => !order.includes(k)))) {
    if (!groups[key]) continue;
    const og = document.createElement('optgroup'); og.label = names[key] || ('ArWen suites: ' + key);
    for (const s of groups[key]) og.appendChild(new Option(s.label, s.id));
    sel.appendChild(og);
  }
  S.suites = d.suites || [];
  sel.onchange = showSuite;
}

// the sounding's Surface heating is skipped for a land-surface suite without the steady environment (sim._double_heating)
function showHeatHint() {
  const s = (S.suites || []).find(x => x.id === $('physSuite').value);
  $('heatHint').classList.toggle('hidden', !(s && s.surface && !$('optSteady').checked && +$('heatHours').value > 0));
}

function showSuite() {
  showHeatHint();
  const id = $('physSuite').value;
  const s = (S.suites || []).find(x => x.id === id);
  $('physMp').disabled = !!s;
  if (s) $('physMp').value = String(s.mp_physics);
  $('landRow').classList.toggle('hidden', !(s && s.surface));
  $('lesRow').classList.toggle('hidden', !(s && s.surface));   // the built-in core has no PBL scheme to switch off
  // a suite with surface physics runs on a ground-fixed grid (sim.py): say so where the grid motion is set
  const locked = !!(s && s.surface);
  $('gridMotion').disabled = locked;
  $('gridLockHint').classList.toggle('hidden', !locked);
  const info = $('suiteInfo');
  info.classList.toggle('hidden', !s);
  if (s) {
    const c = s.components || {};
    info.textContent = `PBL ${c.pbl || '—'} · surface layer ${c.surface_layer || '—'} · land ${c.land_surface || '—'} · ` +
      `radiation ${c.radiation || '—'}` + (c.cumulus && c.cumulus !== 'off' ? ` · cumulus ${c.cumulus} (off below 3 km)` : '') +
      (s.surface ? '. Runs on a ground-fixed grid (its surface layer needs ground-relative wind).' : '');
  }
}

function fillSoundingList(filter) {
  const sel = $('sndList');
  sel.innerHTML = '';
  const f = filter.toLowerCase();
  let group = null, og = null, n = 0;
  for (const it of S.soundingFiles) {
    if (f && !(it.name.toLowerCase().includes(f) || it.group.toLowerCase().includes(f))) continue;
    if (it.group !== group) { group = it.group; og = document.createElement('optgroup'); og.label = group; sel.appendChild(og); }
    og.appendChild(new Option(it.name, it.path));
    if (++n > 1500) break;
  }
}

// The Atmosphere tab's changes to a sounding (mix, scale winds; under Advanced: moisten, cap, ice
// saturation, jet, winds aloft), applied on top of its file, text or analytic profile every time it loads
const SOUNDING_MODS = ['heating', 'mix', 'moist', 'mid_moist', 'cap', 'wind_scale', 'ice_cap', 'llj', 'aloft'];
const baseSpec = spec => { const b = { ...(spec || {}) }; for (const k of SOUNDING_MODS) delete b[k]; return b; };

function soundingMods() {
  const m = {};
  const hh = +$('heatHours').value;
  if (hh > 0) m.heating = { hours: hh, flux: +$('heatFlux').value || 300, bowen: 1.0 };
  if ($('mixOn').checked) m.mix = { depth: +$('mixDepth').value, heat: +$('mixHeat').value };
  const dtd = +$('moistDtd').value;
  if (dtd > 0) m.moist = { dtd, depth: +$('moistDepth').value || 100 };
  const rh = +$('midRh').value;
  if (rh > 0) m.mid_moist = { rh, bottom: +$('midBottom').value || 750, top: +$('midTop').value || 450 };
  const cin = Math.abs(+$('capCin').value);
  if (cin > 0) m.cap = { cin, base: +$('capBase').value || 100 };
  const scale = +$('windScale').value;
  if (scale > 0 && scale !== 1) m.wind_scale = scale;
  if (!$('iceCap').checked) m.ice_cap = false;
  const jet = +$('lljSpeed').value;
  if (jet > 0) m.llj = { speed: jet, height: +$('lljHeight').value || 750 };
  const up = +$('aloftSpeed').value;
  if (up > 0) {
    const base = (+$('aloftBase').value || 5) * 1000;
    m.aloft = { speed: up, base, full: base + 4000 };
  }
  return m;
}

// every change off but the ice-saturation limit (the sounding editor baked them into its profile)
function clearSoundingMods() {
  $('mixOn').checked = false; $('moistDtd').value = 0; $('midRh').value = 0; $('capCin').value = 0;
  $('windScale').value = 1; $('lljSpeed').value = 0; $('aloftSpeed').value = 0;
}

async function loadSounding(spec, onError) {
  try {
    spec = baseSpec(spec);
    if (spec.file && !/^[A-Za-z]:/.test(spec.file)) {
      const hit = S.soundingFiles.find(f => f.path.replace(/\\/g, '/').endsWith(spec.file.replace(/^data\/soundings\//, '')));
      if (hit) spec.file = hit.path;
    }
    Object.assign(spec, soundingMods());
    const d = await getJSON('/api/sounding/load', { spec });
    S.snd = d;
    S.setup.sounding = spec;
    showSounding();
    return true;
  } catch (e) { if (onError) onError(e); else alert('Sounding: ' + e.message); return false; }
}

// the parameters under the skew-T (and in the sounding editor, with the change from ``ref``)
const DIAG = [
  ['MLCAPE', g => g.mlcape, 0], ['MUCAPE', g => g.mucape, 0], ['MLCIN', g => g.mlcin, 0],
  ['LCL m', g => g.lcl, 0], ['0–1 SRH', g => g.srh1, 0], ['0–3 SRH', g => g.srh3, 0],
  ['0–6 shear kt', g => g.shear6 / 0.5144, 0], ['STP', g => g.stp, 1], ['SCP', g => g.scp, 1],
  ['RM kt', g => Math.hypot(...g.rm) / 0.5144, 0], ['PW mm', g => g.pw_mm, 0], ['T/Td °C', g => g.t2m, 0, g => g.td2m],
  ['0–0.5 SRH', g => g.srh05, 0], ['DCAPE', g => g.dcape, 0], ['SR 9–11 km m/s', g => g.sr_anvil, 0],
];

function diagHTML(g, ref) {
  const num = v => typeof v === 'number' && isFinite(v);
  const f = (v, n) => num(v) ? v.toFixed(n) : '—';
  return DIAG.map(([k, val, n, second]) => {
    const a = val(g);
    let s = f(a, n) + (second ? '/' + f(second(g), n) : '');
    const b = ref && !second ? val(ref) : null;
    if (num(a) && num(b) && Math.abs(a - b) >= 0.5 * 10 ** -n) {
      s += ` <i class="${a > b ? 'up' : 'dn'}">${a > b ? '+' : '−'}${Math.abs(a - b).toFixed(n)}</i>`;
    }
    return `<div><b>${k}</b>${s}</div>`;
  }).join('');
}

function showSounding() {
  const d = S.snd;
  if (!d) return;
  $('sndTitle').textContent = d.sounding.title;
  SkewT.drawSkewT($('skewt'), d.sounding);
  SkewT.drawHodo($('hodo'), d.sounding, d.diag);
  $('diag').innerHTML = diagHTML(d.diag);
}

// ------------------------------------------------------------------ map data
// the map shows the run's projection, until a new center is picked after the run has ended: then
// the next run's (S.detached), and the ended run's frames, drawn around the old center, go away
async function reloadMap() {
  const run = S.run && !S.detached ? S.run : null;
  const lat0 = run ? run.proj.lat0 : S.setup.map.lat0, lon0 = run ? run.proj.lon0 : S.setup.map.lon0;
  const key = `${lat0},${lon0}`;
  if (key === S.mapKey) return;
  S.mapKey = key;
  const proj = run ? run.proj : await projFor(lat0, lon0);
  map.setProjection(proj);
  const [lines, sites, cities] = await Promise.all([
    getJSON(`/api/map?lat0=${lat0}&lon0=${lon0}`),
    getJSON(`/api/sites?lat0=${lat0}&lon0=${lon0}`),
    getJSON(`/api/cities?lat0=${lat0}&lon0=${lon0}&minpop=5000`),
  ]);
  S.sites = sites; S.cities = cities;
  map.setLines(lines);
  if (!S.centered) { map.center(0, 0, 1.6); S.centered = true; }
}

// Lambert parameters computed the same way as the server (geo.Lambert)
async function projFor(lat0, lon0) {
  const d2r = Math.PI / 180, R = 6370000;
  const t1 = lat0 >= 0 ? lat0 - 5 : lat0 + 5, t2 = lat0 >= 0 ? lat0 + 5 : lat0 - 5;
  const p1 = t1 * d2r, p2 = t2 * d2r;
  const n = Math.log(Math.cos(p1) / Math.cos(p2)) / Math.log(Math.tan(Math.PI / 4 + p2 / 2) / Math.tan(Math.PI / 4 + p1 / 2));
  const F = Math.cos(p1) * Math.pow(Math.tan(Math.PI / 4 + p1 / 2), n) / n;
  const rho0 = R * F / Math.pow(Math.tan(Math.PI / 4 + lat0 * d2r / 2), n);
  return { lat0, lon0, n, F, rho0, R };
}

// GR2Analyst's polling URLs: this PC's (the page's own address when it is opened through another
// name, a tunnel say) and, when the viewer shares the feed (--lan), other computers'
function showPollLinks(p) {
  const loopback = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
  $('pollUrl').textContent = loopback && p ? p.local : location.origin + '/poll/';
  $('placeUh').textContent = $('pollUrl').textContent + 'uhtrack.txt';
  $('placeWind').textContent = 'windtrack.txt';
  $('placeWarn').textContent = $('pollUrl').textContent + 'warnings.txt';
  $('placeMetar').textContent = $('pollUrl').textContent + 'metars.txt';
  const el = $('pollLan');
  el.textContent = '';
  const lan = (p && p.lan) || [];
  if (!lan.length) {
    el.textContent = 'Other computers: this PC has no network address right now.';
    return;
  }
  el.append('Other computers: ');
  lan.forEach((u, i) => {
    if (i) el.append(' or ');
    const c = document.createElement('code');
    c.textContent = u;
    el.append(c);
  });
  el.append(' (only the feed is shared).');
}

// ------------------------------------------------------------------ polling
async function poll() {
  setupSummary();                      // storms, radars and fronts are added on the map too, not only in forms
  let st;
  try { st = await getJSON('/api/state'); } catch (e) { setStatus('error', 'Server offline'); return; }
  if (!st.status || st.status === 'none') {
    setStatus(st.notice ? 'error' : 'idle', st.notice ? 'Restarted after a GPU error' : 'No run');
    $('status').title = st.notice || '';
    return;
  }
  const firstAdopt = !S.run || S.run.run_id !== st.run_id;
  S.run = st;
  Syn.onPoll(st);
  Survey.onPoll(st);
  if (firstAdopt) {
    S.detached = false;
    S.mapKey = ''; await reloadMap(); S.lastFrameCount = -1; S.frames = [];
    if (st.sounding) { S.snd = { sounding: st.sounding, diag: st.diag }; showSounding(); }
  }
  setStatus(st.status === 'waiting' ? 'idle' : st.status,
    st.status === 'running' ? 'Running' : st.status === 'waiting' ? 'Waiting for storms'
      : st.status[0].toUpperCase() + st.status.slice(1));
  if (st.status === 'error') {
    console.error(st.error);
    $('status').title = st.error || '';
    $('status').textContent = 'Error: ' + String(st.error || '').split(/\r?\n/)[0].slice(0, 90);
  }
  const t = st.model_ms ? new Date(st.model_ms) : null;
  $('clock').textContent = t ? t.toISOString().replace('T', ' ').slice(0, 19) + 'Z' : '';
  if (st.t !== undefined) {
    const cells = (st.cells / 1e6).toFixed(1);
    if (!st.patches.length) {
      // no patches = nothing to compute: the speed is meaningless, say what is missing instead
      const nothing = !(st.starts || []).length && !(st.zones || []).length && !(st.boundaries || []).length &&
        !(st.settings.auto_storms && st.settings.auto_storms.on) && !st.synoptic;
      $('perf').textContent = `+${fmtDur(st.t)}  ·  ` + (st.synoptic && !st.memory_full
        ? 'the simulated weather is evolving: storms form where it lets them (Weather tab: Where storms can form)'
        : nothing
        ? 'no storms yet: turn on Auto outbreak, place a cell, drag out an initiation zone or draw a front'
        : st.memory_full ? 'GPU memory is full: new storms cannot start (see Log)' : 'waiting for storms to start');
    } else {
      const layout = st.fixed_cells ? `fixed ${(st.fixed_cells * st.dx / 1000).toFixed(0)}` +
        (st.fixed_cells_y && st.fixed_cells_y !== st.fixed_cells ? ` × ${(st.fixed_cells_y * st.dx / 1000).toFixed(0)}` : '') + ' km domain'
        : `${st.patches.length} patch${st.patches.length === 1 ? '' : 'es'}` +
          (st.max_side ? ` (up to ${(st.max_side * st.dx / 1000).toFixed(0)} km)` : '');
      // the run's real speed over the last ~20 s (radar, frames and every other job between the steps included);
  // the model steps' own in the tooltip
  $('perf').title = st.step_speed ? `The model steps alone: ${st.step_speed.toFixed(0)}× real time. The figure shown ` +
    'is the real one: the radar sweeps, frames, warnings and patch re-fits between the steps included.' : '';
  $('perf').textContent = `+${fmtDur(st.t)}  ·  ${st.speed ? (st.speed >= 100 ? st.speed.toFixed(0) : st.speed.toFixed(1)) : '–'}× real time  ·  dt ${(+st.dt).toFixed(1)} s  ·  ` +
        `${layout}, ${cells} M cells` + (st.gpu_bytes ? `, GPU ${(st.gpu_bytes / 2 ** 30).toFixed(1)} GB` : '') +
        (st.uh_peak ? `  ·  run-max UH ${Math.round(st.uh_peak.value)} m²/s² (+${fmtDur(st.uh_peak.t).slice(0, -3)})` : '') +
        (st.memory_full ? '  ·  GPU memory full: new storms wait' : '');
    }
  }
  $('btnPause').textContent = st.status === 'paused' ? 'Resume' : 'Pause';
  Presets.update();
  if ($('l2dir')) $('l2dir').textContent = 'output\\runs\\' + st.run_id;
  renderLog(st.events || []);
  renderLists();
  Trop.onPoll();
  Metar.onPoll();
  renderGridNow();
  fillRadarSelect();
  if (!S.detached && st.frames !== S.lastFrameCount) {
    S.lastFrameCount = st.frames;
    S.frames = await getJSON('/api/frames');
    $('rpSlider').max = Math.max(0, S.frames.length - 1);
    if (S.live) { S.frameIdx = S.frames.length - 1; $('rpSlider').value = S.frameIdx; showFrame(); }
  }
  map.redraw();
}

function setStatus(cls, text) {
  const el = $('status');
  el.className = 'status ' + cls;
  el.textContent = text;
}

const fmtDur = s => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };

// ------------------------------------------------------------------ products
function isRadarProduct(p) { return RADAR_PRODUCTS.some(r => r[0] === p); }

async function paletteFor(product) {
  const fam = Colors.PRODUCT_PALETTE[product];
  const sel = $('palette').value;
  if (sel && sel !== 'default' && (Colors.PAL_FAMILY[fam] || []).length) {
    if (!S.palCache[sel]) {
      const pal = await getJSON('/api/palette?path=' + encodeURIComponent(sel));
      S.palCache[sel] = Colors.adaptGR(pal, fam);
    }
    return S.palCache[sel];
  }
  return Colors.DEFAULT_PALETTES[fam];
}

function fillPaletteSelect() {
  const fam = Colors.PRODUCT_PALETTE[S.product];
  const keys = Colors.PAL_FAMILY[fam] || [];
  const sel = $('palette'), cur = sel.value;
  sel.innerHTML = '';
  sel.add(new Option('Default colors', 'default'));
  for (const p of S.palettes) {
    if (!keys.length) break;
    if (p.product && !keys.includes(p.product)) continue;
    sel.add(new Option(p.name, p.path));
  }
  sel.value = [...sel.options].some(o => o.value === cur) ? cur : 'default';
  sel.disabled = !keys.length;
}

// One frame request at a time.  A change made while one loads (another product, radar, color table
// or frame) is drawn as soon as it is done; it used to be dropped, and the frame that was loading
// could come out with the new product's legend or the old one's colors.
async function showFrame() {
  if (Trop.showingPreview()) return;            // the Tropics tab's SST map owns the map until switched off
  if (!S.frames.length) return;
  if (S.fetching) { S.frameAgain = true; return; }
  S.fetching = true;
  try {
    do {
      S.frameAgain = false;
      await drawFrame();
    } while (S.frameAgain && S.frames.length);
  } finally { S.fetching = false; }
}

async function drawFrame() {
  const i = S.frameIdx < 0 ? S.frames.length - 1 : Math.min(S.frameIdx, S.frames.length - 1);
  const fr = S.frames[i], product = S.product, radar = isRadarProduct(product);
  $('rpTime').textContent = `${new Date(fr.ms).toISOString().slice(11, 19)}Z  (+${fmtDur(fr.t)})  ${i + 1}/${S.frames.length}`;
  if (window.Vol) Vol.followTime(S.live ? null : fr.t);      // the 3-D view's replay goes with the map's
  try {
    const pal = await paletteFor(product);
    if (radar) {
      map.setRaster(null);
      const icao = S.radar || (fr.radars && fr.radars[0]);
      if (!icao) { map.setPPI(null); S.ppiInfo = null; return; }
      const buf = await (await api(`/api/ppi?i=${i}&icao=${icao}&moment=${product}`)).arrayBuffer();
      const { header, data } = unpack(buf);
      const lut = Colors.buildLUT(pal, header.scale, header.offset);
      S.ppiInfo = { meta: header, codes: data };
      map.setPPI({ meta: header, codes: data, lut, smooth: $('smooth').checked });
      drawLegend(pal, Colors.PRODUCT_PALETTE[product]);
    } else {
      map.setPPI(null); S.ppiInfo = null;
      const buf = await (await api(`/api/frame?i=${i}&product=${product}`)).arrayBuffer();
      S.rasterInfo = packedToRaster(buf, pal);
      map.setRaster({ patches: S.rasterInfo.patches, smooth: $('smooth').checked });
      drawLegend(pal, Colors.PRODUCT_PALETTE[product]);
    }
  } catch (e) {
    if (radar) { map.setPPI(null); S.ppiInfo = null; }
  }
}

// a packed map product (products.pack_frame_product) as one canvas per patch in the palette's colors
function packedToRaster(buf, pal) {
  const { header, data } = unpack(buf);
  const [scale, offset] = header.codec;
  const lut = Colors.buildLUT(pal, scale, offset);
  const patches = header.patches.map(p => {
    const codes = data.subarray(p.offset, p.offset + p.nx * p.ny);
    const cv = document.createElement('canvas');
    cv.width = p.nx; cv.height = p.ny;
    const g = cv.getContext('2d'), img = g.createImageData(p.nx, p.ny);
    // patch row 0 (south) goes to canvas row 0: map.js draws with a y-flipped transform,
    // which puts canvas row 0 at the patch's south edge
    for (let j = 0; j < p.ny; j++) {
      const src = j * p.nx, dst = j * p.nx;
      for (let k = 0; k < p.nx; k++) {
        const c = codes[src + k] * 4, o = (dst + k) * 4;
        img.data[o] = lut[c]; img.data[o + 1] = lut[c + 1]; img.data[o + 2] = lut[c + 2]; img.data[o + 3] = lut[c + 3];
      }
    }
    g.putImageData(img, 0, 0);
    return { x0: p.x0 / 1000, y0: p.y0 / 1000, x1: p.x1 / 1000, y1: p.y1 / 1000, nx: p.nx, ny: p.ny, canvas: cv, codes };
  });
  return { header, patches, scale, offset };
}

// the "PC explosion" details' warning before one is placed or switched to (size: the box, km); true to go on
function pcConfirm(det, size) {
  if (det !== 'pc10' && det !== 'pc1') return true;
  return confirm(det === 'pc1'
    ? 'PC EXPLOSION (1 m): a debris-scale grid ' + size.toFixed(2) + ' km a side, about ' + Math.round(size * 1000) + ' x ' + Math.round(size * 1000) +
      ' cells on ~1 m levels. Expect minutes of real time per model SECOND, and many GB of GPU memory (the box is made smaller to fit). Go?'
    : 'PC explosion (10 m): a suction-vortex-scale grid ' + size.toFixed(2) + ' km a side, about ' + Math.round(size * 100) + ' x ' + Math.round(size * 100) +
      ' cells on ~8 m levels. Far slower than real time, several GB of GPU memory. Go?');
}

// the running tornado zoom switched to another detail where it is (no redrawing): the levels both share are kept
async function changeZoomDetail() {
  const tz = S.run && S.run.tornado, det = $('tornadoDetail').value;
  if (!runActive() || !tz || tz.detail === det) return;
  const usual = { lite: 20, medium: 14, high: 10, ultra: 6, extreme: 4, pc10: 2, pc1: 0.3 };
  if (!pcConfirm(det, tz.size_km * usual[det] / usual[tz.detail])) { $('tornadoDetail').value = tz.detail; return; }
  try { await api('/api/run/tornado/detail', { detail: det }); }
  catch (e) { alert('Tornado zoom: ' + e.message); $('tornadoDetail').value = tz.detail; }
}

// a grid spacing in m for a label: whole metres from 10 m, two figures under (the pc grids: 6.2, 0.83)
function fmtM(m) { return m >= 10 ? String(Math.round(m)) : String(+m.toPrecision(2)); }

function unpack(buf) {
  const dv = new DataView(buf);
  const hl = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  return { header, data: new Uint8Array(buf, 4 + hl) };
}

function drawLegend(pal, fam) {
  const [lo, hi, step] = LEGEND_RANGE[fam] || [0, 1, 0.25];
  const cv = $('legendBar'), g = cv.getContext('2d');
  const cols = Colors.legendGradient(pal, lo, hi, cv.width);
  for (let i = 0; i < cv.width; i++) {
    const c = cols[i];
    g.fillStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${(c[3] === undefined ? 255 : c[3]) / 255})`;
    g.fillRect(i, 0, 1, cv.height);
  }
  const ticks = [];
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(`<span style="left:${((v - lo) / (hi - lo) * 100).toFixed(1)}%">${+v.toFixed(3)}</span>`);
  $('legendTicks').innerHTML = ticks.join('');
}

// ------------------------------------------------------------------ overlays
function drawOverlay(g, m) {
  const run = S.run;
  if (!S.setup) return;
  // the simulated weather (synoptic mode), then fronts and air masses, under everything else
  g.save(); Syn.draw(g, m); g.restore();
  g.save(); Fronts.draw(g, m); g.restore();
  // the automatic warnings in force (warnings_auto.py), NWS colours in a black casing
  if ($('showWarnings').checked && run && run.warnings) drawWarnings(g, m, run.warnings);
  drawBrushes(g, m);
  // the storm firing area: orange dashes
  const fa = (S.run && S.run.settings && S.run.settings.fire_area) || S.setup.fire_area;
  if (fa) {
    const [ax, ay] = m.toScreen(fa[0] / 1000, fa[1] / 1000), [bx, by] = m.toScreen(fa[2] / 1000, fa[3] / 1000);
    g.save(); g.strokeStyle = 'rgba(249,115,22,.9)'; g.lineWidth = 2; g.setLineDash([10, 6]);
    g.strokeRect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay));
    g.fillStyle = 'rgba(249,115,22,.95)'; g.font = '11px Segoe UI';
    g.fillText('storm firing area', Math.min(ax, bx) + 4, Math.min(ay, by) + 14); g.restore();
  }
  // patches
  if ($('showPatches').checked && run && run.patches && !S.detached) {
    g.strokeStyle = 'rgba(120,180,255,.55)'; g.setLineDash([6, 5]); g.lineWidth = 1;
    g.font = '11px Segoe UI'; g.fillStyle = 'rgba(160,200,255,.8)';
    for (const p of run.patches) {
      const [a, b] = m.toScreen(p.x0 / 1000, p.y1 / 1000), [c, d] = m.toScreen(p.x1 / 1000, p.y0 / 1000);
      g.strokeRect(a, b, c - a, d - b);
      const held = p.held < 0 ? '  · locked' : p.held > 0 ? `  · held ${Math.ceil(p.held / 60)} min` : '';
      g.fillText(`${run.fixed_cells ? 'domain' : 'patch ' + p.id}  ${p.nx}×${p.ny}${held}`, a + 4, b + 13);
    }
    // refinement nests: solid, with their spacing
    // (white: the warnings' yellow, red and moccasin sit on the same storms)
    g.strokeStyle = 'rgba(255,255,255,.85)'; g.setLineDash([]); g.lineWidth = 1.5; g.fillStyle = 'rgba(255,255,255,.9)';
    for (const p of (run.nests || [])) {
      const [a, b] = m.toScreen(p.x0 / 1000, p.y1 / 1000), [c, d] = m.toScreen(p.x1 / 1000, p.y0 / 1000);
      g.strokeRect(a, b, c - a, d - b);
      g.fillText(`nest ${p.id}  ${fmtM(p.dx)} m${p.hand ? ' (yours)' : ''}`, a + 4, b + 13);
    }
    // a patch or nest being dragged with the Move tool
    if (S.moveGhost) {
      const q = S.moveGhost, [a, b] = m.toScreen(q.x0, q.y1), [c, d] = m.toScreen(q.x1, q.y0);
      g.save(); g.strokeStyle = '#f2b84b'; g.lineWidth = 2; g.setLineDash([8, 4]);
      g.fillStyle = 'rgba(242,184,75,.08)'; g.fillRect(a, b, c - a, d - b); g.strokeRect(a, b, c - a, d - b);
      g.fillStyle = '#f2b84b'; g.font = '12px Segoe UI'; g.fillText(q.label, a + 4, b - 5); g.restore();
    }
    g.strokeStyle = 'rgba(120,180,255,.55)'; g.setLineDash([6, 5]); g.lineWidth = 1; g.fillStyle = 'rgba(160,200,255,.8)';
    // a fixed-size domain: where new storms can start
    const sa = runActive() && run.start_area;
    if (sa && sa[2] > sa[0] && sa[3] > sa[1]) {
      const [a, b] = m.toScreen(sa[0] / 1000, sa[3] / 1000), [c, d] = m.toScreen(sa[2] / 1000, sa[1] / 1000);
      g.strokeStyle = 'rgba(120,180,255,.3)'; g.setLineDash([2, 4]);
      g.strokeRect(a, b, c - a, d - b);
    }
    g.setLineDash([]);
  }
  // OutbreakLab mode: the square where warm-sector storms can form
  const ao = autoCfg();
  if (ao.on) {
    const h = ao.area_km / 2, [a, b] = m.toScreen(-h, h), [c, d] = m.toScreen(h, -h);
    g.strokeStyle = 'rgba(255,190,90,.6)'; g.setLineDash([10, 6]); g.lineWidth = 1.2;
    g.strokeRect(a, b, c - a, d - b); g.setLineDash([]);
    g.font = '11px Segoe UI'; g.fillStyle = 'rgba(255,200,120,.85)';
    g.fillText('OutbreakLab mode: storms form in the warm sector here', a + 4, d - 6);
  }
  // cities
  if (m.scale > 1.5) {
    g.font = '11px Segoe UI'; g.fillStyle = 'rgba(220,226,235,.75)';
    const minpop = m.scale > 6 ? 5000 : m.scale > 3 ? 25000 : 100000;
    for (const c of S.cities) {
      if (c.pop < minpop) continue;
      const [x, y] = m.toScreen(c.x, c.y);
      if (x < -50 || y < -20 || x > m.W + 50 || y > m.H + 20) continue;
      g.fillRect(x - 1.5, y - 1.5, 3, 3); g.fillText(c.name, x + 4, y - 3);
    }
  }
  // radar sites while choosing
  if (S.tool === 'radar') {
    const want = +$('radarKind').value;
    g.fillStyle = want === 3 ? '#f2913a' : want === 4 ? '#9b6cf2' : '#4aa3ff'; g.font = '10px Segoe UI';
    for (const s of S.sites) {
      if (!siteOfKind(s, want)) continue;
      const [x, y] = m.toScreen(s.x, s.y);
      g.beginPath(); g.arc(x, y, 3, 0, 7); g.fill(); g.fillText(s.icao, x + 5, y + 3);
    }
  }
  if ($('showTorTracks').checked && run && run.tornado_tracks && !S.detached) drawTornadoTracks(g, m, run.tornado_tracks);
  // radars, storm starts and zones: the run's while it runs, else the setup's (what the next run gets)
  const live = runActive() ? run : null;
  const radars = live ? live.radars : S.setup.radars.map(r => {
    if (r.kind === 'dow') return r.x !== undefined ? r : null;             // parked where it was clicked
    const s = S.sites.find(q => q.icao === r.icao); return s ? { ...r, x: s.x * 1000, y: s.y * 1000 } : null;
  }).filter(Boolean);
  for (const r of radars) {
    const [x, y] = m.toScreen(r.x / 1000, r.y / 1000);
    const dow = r.kind === 'dow';
    g.strokeStyle = r.icao === S.radar ? '#ffffff' : dow ? '#f2b84b' : '#9fc3ff'; g.lineWidth = 1.5;
    if (dow) { g.strokeRect(x - 5, y - 4, 10, 8); } else { g.beginPath(); g.arc(x, y, 5, 0, 7); g.stroke(); }
    g.fillStyle = '#fff'; g.font = 'bold 11px Segoe UI'; g.fillText(r.icao, x + 7, y - 6);
    g.strokeStyle = dow ? 'rgba(242,184,75,.35)' : 'rgba(160,190,230,.25)'; g.lineWidth = 1;
    for (const rk of dow ? [30, 60] : [100, 230]) { g.beginPath(); g.arc(x, y, rk * m.scale, 0, 7); g.stroke(); }
  }
  // storm starts (not the latch pushes, nor the seed vortices: the cyclones draw their own L)
  const starts = live && live.starts ? live.starts.filter(s => s.why !== 'latch' && s.why !== 'thermal' && s.kind !== 'vortex') : S.setup.storms;
  for (const s of $('showStarts').checked ? starts : []) {
    const [x, y] = m.toScreen(s.x / 1000, s.y / 1000);
    g.strokeStyle = '#f2b84b'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x - 6, y); g.lineTo(x + 6, y); g.moveTo(x, y - 6); g.lineTo(x, y + 6); g.stroke();
    if (s.vx !== undefined && (s.vx || s.vy)) {
      const L = 15 * 60 / 1000 * m.scale;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + s.vx * L, y - s.vy * L); g.stroke();
    }
  }
  // zones
  const zones = live && live.zones ? live.zones : S.setup.zones;
  g.strokeStyle = 'rgba(242,184,75,.7)'; g.setLineDash([4, 4]);
  for (const z of zones) {
    const [a, b] = m.toScreen(z.x0 / 1000, z.y1 / 1000), [c, d] = m.toScreen(z.x1 / 1000, z.y0 / 1000);
    g.strokeRect(a, b, c - a, d - b);
  }
  g.setLineDash([]);
  // nudger cells: each one's planned path, from its start to where it meets the storm; while a storm
  // is being picked, the storms' deep updrafts (interactions.Tracks)
  if (live) {
    for (const n of $('showStarts').checked ? live.nudgers || [] : []) {
      const [a, b] = m.toScreen(n.start[0] / 1000, n.start[1] / 1000);
      const [c, d] = m.toScreen(n.meet_point[0] / 1000, n.meet_point[1] / 1000);
      const over = ['merged', 'missed', 'no updraft', 'storm gone'].includes(n.status);
      g.strokeStyle = over ? 'rgba(200,160,255,.35)' : 'rgba(200,160,255,.9)'; g.lineWidth = 1.5;
      g.setLineDash([5, 4]); g.beginPath(); g.moveTo(a, b); g.lineTo(c, d); g.stroke(); g.setLineDash([]);
      g.beginPath(); g.arc(c, d, 4, 0, 7); g.stroke();
      g.fillStyle = 'rgba(220,195,255,.9)'; g.font = '11px Segoe UI';
      g.fillText(`nudger #${n.id}: ${n.status}`, a + 6, b + 4);
    }
    if (S.tool === 'nudger') {
      g.font = '10px Segoe UI';
      for (const t of live.tracks || []) {
        const [x, y] = m.toScreen(t.x / 1000, t.y / 1000);
        g.strokeStyle = '#ff9ad5'; g.lineWidth = 1.5;
        g.beginPath(); g.arc(x, y, 3 + Math.min(t.w, 80) / 10, 0, 7); g.stroke();
        g.fillStyle = '#ffd0ea'; g.fillText(`storm ${t.id} · ${t.w.toFixed(0)} m/s`, x + 9, y - 6);
      }
    }
  }
  if (!S.detached) Metar.draw(g, m);
  drawTornadoZoom(g, m, run);
  Vol.drawBox(g, m);
  Trop.draw(g, m);
}

// the tornado zoom (tornado.py): its finest nest's box, the vortex it follows, and the surface pressure minimum's
// track (orange, like the "surface pressure minima" line of a research rendering)
function drawTornadoZoom(g, m, run) {
  const tz = run && !S.detached ? run.tornado : null;
  $('tornadoView').classList.toggle('hidden', !tz);
  $('tornadoRemove').classList.toggle('hidden', !tz);
  if (!tz) return;
  g.save();
  const tr = tz.track || [];
  if (tr.length > 1) {
    g.strokeStyle = 'rgba(0,0,0,.6)'; g.lineWidth = 4; g.beginPath();
    tr.forEach((p, i) => { const [a, b] = m.toScreen(p[1] / 1000, p[2] / 1000); i ? g.lineTo(a, b) : g.moveTo(a, b); });
    g.stroke();
    g.strokeStyle = '#ff8a1e'; g.lineWidth = 2; g.stroke();
  }
  if (tz.box) {
    const [a, b] = m.toScreen(tz.box[0] / 1000, tz.box[3] / 1000), [c, d] = m.toScreen(tz.box[2] / 1000, tz.box[1] / 1000);
    g.strokeStyle = '#ffb02e'; g.lineWidth = 1.5; g.strokeRect(a, b, c - a, d - b);
    g.font = '11px Segoe UI'; g.fillStyle = '#ffd38a';
    const f = tz.found;
    g.fillText(`tornado zoom ${fmtM(tz.dx)} m` + (tz.dz0 ? ` (${fmtM(tz.dz0)} m lowest layer)` : '') +
      (f ? ` · ${(f.dp).toFixed(0)} hPa · ${Math.round(f.vmax)} m/s` : ''), a + 4, b - 5);
  } else {
    const [a, b] = m.toScreen(tz.x / 1000, tz.y / 1000);
    g.fillStyle = '#ffd38a'; g.font = '11px Segoe UI';
    g.fillText(tz.waiting ? `tornado zoom: ${tz.waiting}` : 'tornado zoom: placing…', a + 8, b);
  }
  const [x, y] = m.toScreen(tz.x / 1000, tz.y / 1000);
  g.strokeStyle = '#ffb02e'; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(x - 6, y); g.lineTo(x + 6, y); g.moveTo(x, y - 6); g.lineTo(x, y + 6); g.stroke();
  g.restore();
}

// tornado tracks (Simulation.tornado_tracks): each path colored by its estimated EF rating along it, up to
// the frame on screen when replaying; smoothed again on screen (Chaikin corner cutting) so it reads as a path
const EF_COLORS = ['#7fdcff', '#3ccf7a', '#f5d33a', '#f2913a', '#e8413c', '#c13ae8'];
function chaikin(p, n) {
  for (let k = 0; k < n && p.length > 2; k++) {
    const q = [p[0]];
    for (let i = 0; i < p.length - 1; i++) {
      const [x0, y0, e0] = p[i], [x1, y1, e1] = p[i + 1];
      q.push([0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1, e0], [0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1, e1]);
    }
    q.push(p[p.length - 1]);
    p = q;
  }
  return p;
}
function drawTornadoTracks(g, m, tracks) {
  const fr = S.frames.length ? S.frames[S.frameIdx < 0 ? S.frames.length - 1 : Math.min(S.frameIdx, S.frames.length - 1)] : null;
  const tNow = fr && !S.live ? fr.t : Infinity;
  g.save(); g.lineCap = 'round'; g.lineJoin = 'round';
  for (const tr of tracks) {
    const pts = tr.pts.filter(p => p[0] <= tNow + 1);
    if (pts.length < 2) continue;
    const scr = chaikin(pts.map(p => { const [x, y] = m.toScreen(p[1] / 1000, p[2] / 1000); return [x, y, Math.max(p[3], 0)]; }), 3);
    // a dark casing, then each piece in its EF color and width
    g.strokeStyle = 'rgba(0,0,0,.55)';
    g.lineWidth = 4 + 1.2 * Math.max(...scr.map(p => p[2]));
    g.beginPath(); g.moveTo(scr[0][0], scr[0][1]); for (const p of scr) g.lineTo(p[0], p[1]); g.stroke();
    for (let i = 0; i < scr.length - 1; i++) {
      const ef = Math.round(scr[i][2]);
      g.strokeStyle = EF_COLORS[ef]; g.lineWidth = 2 + 1.2 * ef;
      g.beginPath(); g.moveTo(scr[i][0], scr[i][1]); g.lineTo(scr[i + 1][0], scr[i + 1][1]); g.stroke();
    }
    const efMax = Math.max(...pts.map(p => p[3]));
    const [sx, sy] = scr[0], [ex, ey] = scr[scr.length - 1];
    g.fillStyle = '#fff'; g.beginPath(); g.arc(sx, sy, 3, 0, 7); g.fill();
    if (tr.active && tNow === Infinity) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(ex, ey, 6, 0, 7); g.stroke(); }
    const label = `T${tr.id} EF${efMax} (est.)`;
    g.font = '600 11px Inter, Segoe UI, sans-serif';
    const w = g.measureText(label).width;
    g.fillStyle = 'rgba(10,14,20,.8)'; g.fillRect(ex + 8, ey - 16, w + 10, 17);
    g.fillStyle = EF_COLORS[Math.max(efMax, 0)]; g.fillText(label, ex + 13, ey - 4);
  }
  g.restore();
}

// a nudger cell from the Storms tab (sim._send_nudger / _plan_nudger)
function nudgerForm() {
  const num = (id, lo, hi, d) => { const v = parseFloat($(id).value); return isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; };
  return { motion: $('ndMotion').value, lead_min: num('ndLead', 10, 60, 20), aim_km: num('ndAim', -40, 40, 0),
           strength: num('ndStrength', 3, 20, 8), radius_km: num('ndRadius', 3, 15, 6) };
}

// the Move tool: what is under (x, y) km -- a nest (the smallest first, they sit inside patches), else a patch
function moveHit(x, y) {
  const run = runActive() ? S.run : null;
  if (!run) return null;
  const inside = p => x * 1000 >= p.x0 && x * 1000 <= p.x1 && y * 1000 >= p.y0 && y * 1000 <= p.y1;
  const nests = (run.nests || []).filter(n => inside(n) && !n.pinned)          // the tornado zoom's follow its vortex
    .sort((a, b) => (a.x1 - a.x0) - (b.x1 - b.x0));
  if (nests.length) return { kind: 'nest', p: nests[0] };
  const p = (run.patches || []).find(inside);
  return p ? { kind: 'patch', p } : null;
}

// dragging one: its outline follows the mouse (S.moveGhost, drawn by drawOverlay), dropped where it is let go;
// let go without moving: a click (a nest added on the storm there)
function moveGrab(x, y) {
  const hit = moveHit(x, y);
  if (!hit) return null;
  const p = hit.p, w = (p.x1 - p.x0) / 1000, h = (p.y1 - p.y0) / 1000;
  const cx = (p.x0 + p.x1) / 2000, cy = (p.y0 + p.y1) / 2000;
  let moved = false, shift = false;
  const at = (a, b) => ({ x: cx + a - x, y: cy + b - y });
  const onKey = e => { shift = e.shiftKey; };
  window.addEventListener('keydown', onKey); window.addEventListener('keyup', onKey);
  return {
    move(a, b) {
      if (Math.hypot(a - x, b - y) * map.scale > 4) moved = true;
      const c = at(a, b);
      S.moveGhost = { x0: c.x - w / 2, y0: c.y - h / 2, x1: c.x + w / 2, y1: c.y + h / 2,
        label: `${hit.kind} ${p.id}` + (hit.kind === 'patch' ? (shift ? ': lock here' : ': hold here 15 min') : '') };
      map.redraw();
    },
    async end(a, b) {
      window.removeEventListener('keydown', onKey); window.removeEventListener('keyup', onKey);
      S.moveGhost = null; map.redraw();
      if (!moved) { await addNestAt(x, y); return; }
      const c = at(a, b);
      try {
        if (hit.kind === 'nest') await api('/api/run/nest/move', { id: p.id, x: c.x, y: c.y });
        else await api('/api/run/patch/move', { id: p.id, x: c.x, y: c.y, lock: shift });
      } catch (e) { alert('Move: ' + e.message); }
    },
  };
}

async function addNestAt(x, y) {
  const run = runActive() ? S.run : null;
  if (!run) { alert('Moving patches and nests works on a running model.'); return; }
  try { await api('/api/run/nest/add', { x, y }); } catch (e) { alert('Nest: ' + e.message); }
}

// the Radar kind menu: 1 WSR-88D, 3 TDWR, 4 the world's other radars (types 4 S band and 5 C band)
const siteOfKind = (s, want) => want === 4 ? (s.type === 4 || s.type === 5) : s.type === want;

// ------------------------------------------------------------------ map tools
function setTool(t) {
  S.tool = S.tool === t ? 'pan' : t;
  map.tool = S.tool;
  if (S.tool !== 'front') Fronts.cancelDraft();
  for (const [id, name] of [['toolStorm', 'storm'], ['toolZone', 'zone'], ['toolRadar', 'radar'], ['pickCenter', 'center'],
                            ['toolFront', 'front'], ['toolFire', 'fire'], ['toolBrush', 'brush'], ['ndSend', 'nudger'],
                            ['synSndTool', 'synsnd'], ['synDomainTool', 'fire'], ['toolLow', 'tclow'], ['toolVolume', 'volume'], ['toolTornado', 'tornado'], ['toolMove', 'move'], ['toolDow', 'dow']]) {
    $(id).classList.toggle('on', S.tool === name);
  }
  $('map').classList.toggle('crosshair', S.tool !== 'pan');
  const hints = { storm: 'Click to start a storm', zone: 'Drag a rectangle for random storm starts',
    nudger: 'Click a storm (its updraft is circled) to send a nudger at it',
    radar: 'Click near a radar site to add it', dow: 'Click where the DOW parks',
    center: 'Click the new map center (before launching a run)',
    front: 'Click points along the boundary — double-click or Enter to finish, Esc to cancel',
    fire: 'Drag a rectangle: new storms only fire inside it',
    volume: 'Drag a box around a storm: its 3-D view',
    tornado: 'Drag a small box (2-10 km) around a tornado or the low-level mesocyclone: a ~100 m grid follows it',
    move: 'Drag a patch or a nest to move it (Shift when dropping a patch: lock it there) · click a storm: add a nest on it',
    synsnd: 'Click anywhere: the simulated sounding there (Weather tab)',
    tclow: 'Click over warm water to place a surface low (Tropics tab settings)',
    brush: 'Hold the mouse and paint: the low-level winds under the stroke change (Atmosphere tab settings)' };
  $('toolhint').textContent = hints[S.tool] || '';
  $('toolhint').classList.toggle('hidden', S.tool === 'pan');
  map.redraw();
}

async function onMapClick(x, y) {
  if (S.tool === 'front') { Fronts.addPoint(x, y); return; }
  if (S.tool === 'tclow') { try { await Trop.place(x, y); } catch (e) { alert('Surface low: ' + e.message); } return; }
  if (S.tool === 'pan') {
    const st = Metar.hit(x, y);
    if (st) { Metar.card(st.icao); return; }
    Trop.card(Trop.hit(x, y));
    return;
  }
  if (S.tool === 'synsnd') { Syn.soundingAt(x, y); return; }
  if (S.tool === 'move') { await addNestAt(x, y); return; }
  if (S.tool === 'storm') {
    const st = {
      x: x * 1000, y: y * 1000, kind: $('stormKind').value, motion: $('stormMotion').value,
      strength: +$('stormStrength').value, radius: +$('stormRadius').value * 1000,
    };
    const a = runActive() && S.run.start_area;          // a fixed-size domain takes storms only inside it
    if (a && (st.x < a[0] || st.x > a[2] || st.y < a[1] || st.y > a[3]) &&
        !confirm('This run has a fixed-size domain: storms start only inside the dotted box. It shifts to take ' +
                 'one outside it only if its storms still fit (the Log says if not). Try anyway?')) return;
    if (runActive()) await api('/api/run/storm', st);
    else { S.setup.storms.push(st); renderLists(); }
  } else if (S.tool === 'nudger') {
    if (!runActive()) { alert('Nudgers go at the storms of a running run.'); return; }
    await api('/api/run/nudger', { ...nudgerForm(), target: { x: x * 1000, y: y * 1000 } });
    setTool('pan');
  } else if (S.tool === 'radar') {
    let best = null, bd = 1e9;
    const want = +$('radarKind').value;
    for (const s of S.sites) { const d = Math.hypot(s.x - x, s.y - y); if (d < bd && siteOfKind(s, want)) { bd = d; best = s; } }
    if (!best || bd > 150) { alert(`No ${want === 3 ? 'TDWR' : want === 4 ? 'international radar' : 'WSR-88D'} within 150 km of that point.`); return; }
    const r = { icao: best.icao, kind: want === 3 ? 'tdwr' : best.type === 5 ? 'cband' : 'wsr88d', vcp: +$('radarVcp').value, sails: +$('radarSails').value,
      dealias: $('radarDealias').checked, volume_min: +$('radarVolume').value, floor: +$('radarFloor').value,
      noise: $('radarNoise').checked, clear_air: Math.max(0, Math.min(3, +$('radarClearAir').value || 0)), export: true };
    if (runActive()) await api('/api/run/radar', r);
    // a radar added during a run stays for the next run too
    S.setup.radars = S.setup.radars.filter(q => q.icao !== r.icao).concat([r]);
    renderLists(); fillRadarSelect();
    S.radar = r.icao;
  } else if (S.tool === 'dow') {
    setTool('pan');
    await deployDow(x, y);
  } else if (S.tool === 'center') {
    if (runActive()) { alert('Stop the run before moving the map.'); return; }
    const [lat, lon] = map.inv(x, y);
    $('lat0').value = lat.toFixed(2); $('lon0').value = lon.toFixed(2);
    readSetupFromForm(); await recenterSetup(); setTool('pan');
  }
  map.redraw();
}

// a new map center for the next run (none is running): the map moves there, and an ended run's
// frames, drawn around its own center, leave the map (reloadMap)
async function recenterSetup() {
  const r = S.run;
  if (r && (r.proj.lat0 !== S.setup.map.lat0 || r.proj.lon0 !== S.setup.map.lon0)) {
    S.detached = true; S.frames = []; S.lastFrameCount = -1;
    map.setRaster(null); map.setPPI(null); S.rasterInfo = null; S.ppiInfo = null;
  }
  S.centered = false; S.mapKey = ''; await reloadMap();
}

async function onZoneDrawn(x0, y0, x1, y1) {
  if (S.tool === 'fire') { await setFireArea([x0 * 1000, y0 * 1000, x1 * 1000, y1 * 1000]); setTool('pan'); return; }
  if (S.tool === 'volume') { setTool('pan'); Vol.open([x0, y0, x1, y1]); return; }
  if (S.tool === 'tornado') {
    setTool('pan');
    if (!runActive()) { alert('The tornado zoom works on a running model: start a run first.'); return; }
    // any size: the server keeps it to 0.05-60 km and what the card holds (it says so in the Log when a level won't fit)
    const size = Math.min(60, Math.max(0.05, Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0))));
    const det = $('tornadoDetail').value;
    if (!pcConfirm(det, size)) return;
    try { await api('/api/run/tornado', { x: (x0 + x1) / 2, y: (y0 + y1) / 2, size_km: size, detail: det }); }
    catch (e) { alert('Tornado zoom: ' + e.message); }
    return;
  }
  const z = { x0: x0 * 1000, y0: y0 * 1000, x1: x1 * 1000, y1: y1 * 1000, rate: +$('zoneRate').value,
    spacing: +$('zoneSpacing').value * 1000, t_start: +$('zoneStart').value * 60, t_end: +$('zoneEnd').value * 60 };
  if (runActive()) {
    z.t_start += S.run.t; z.t_end += S.run.t;
    await api('/api/run/zone', z);
  } else { S.setup.zones.push(z); renderLists(); }
  setTool('pan');
}

const WARN_STYLE = { SPS: ['#ffe4b5', 2, 1], SVR: ['#ffff00', 2, 1], SVRC: ['#ffff00', 3, 2], SVRD: ['#ffff00', 3, 3],
  TOR: ['#ff0000', 2, 1], TORR: ['#ff00ff', 3, 2], TORP: ['#ff00ff', 3, 3], TORE: ['#ff00ff', 4, 3] };
function drawWarnings(g, m, list) {
  const order = { SPS: 0, SVR: 1, TOR: 2 };
  for (const w of list.filter(w => w.active).sort((a, b) => order[a.kind] - order[b.kind])) {
    const [col, wd, cs] = WARN_STYLE[w.code] || WARN_STYLE.SVR;
    const pts = w.poly.map(([x, y]) => m.toScreen(x, y));
    g.save(); g.lineJoin = 'round';
    for (const [c, lw] of [['#000', wd + 2 * cs], [col, wd]]) {
      g.strokeStyle = c; g.lineWidth = lw; g.beginPath();
      pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.closePath(); g.stroke();
    }
    g.fillStyle = col; g.font = 'bold 11px Segoe UI';
    g.fillText(w.code, pts[1][0] + 4, pts[1][1]); g.restore();
  }
}

// hodograph brush strokes (boundaries.WindBrush): world-metre paths with their wind change
function brushList() { return (S.setup.wind_brushes = S.setup.wind_brushes || []); }

async function onBrushPainted(path) {
  brushList().push({ points: path.map(([x, y]) => [x * 1000, y * 1000]), radius: +$('brSize').value * 1000,
    back: +$('brBack').value, add: +$('brAdd').value, depth: +$('brDepth').value });
  await pushBrushes();
}

async function pushBrushes() {
  if (runActive()) {
    try { await api('/api/run/brushes', { brushes: brushList() }); }
    catch (e) { alert('Could not change the winds: ' + e.message); }
  }
  renderBrushes(); map.redraw();
}

function renderBrushes() {
  const el = $('brushList');
  if (!el) return;
  el.innerHTML = brushList().map((b, i) => `<div class="item"><div>Stroke ${i + 1} <span class="meta">${b.back > 0 ? 'backed ' + b.back + '°' : b.back < 0 ? 'veered ' + -b.back + '°' : 'same direction'}` +
    `, ${b.add >= 0 ? '+' : ''}${b.add} m/s below ${b.depth} m, ${b.radius / 1000} km wide</span></div>` +
    `<button data-brush="${i}" title="Remove this stroke">✕</button></div>`).join('');
}

function drawBrushes(g, m) {
  const list = (S.run && S.run.settings && S.run.settings.wind_brushes) || S.setup.wind_brushes || [];
  for (const b of list) {
    const pts = b.points.map(([x, y]) => m.toScreen(x / 1000, y / 1000));
    const wpx = Math.max(4, 2 * b.radius / 1000 * m.scale);
    g.save(); g.lineCap = 'round'; g.lineJoin = 'round';
    for (const [f, a] of [[1, 0.10], [0.6, 0.14], [0.3, 0.18]]) {            // feathered toward the edge
      g.strokeStyle = `rgba(167,139,250,${a})`; g.lineWidth = wpx * f;
      g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
    }
    g.restore();
  }
}

// the storm firing area (sim._start_box): for the next run, and for the running one at once
async function setFireArea(box) {
  S.setup.fire_area = box;
  if (runActive()) {
    try { await api('/api/run/fire_area', { box }); } catch (e) { alert('Could not change the firing area: ' + e.message); }
  }
  showFireNote(); map.redraw();
}

function showFireNote() {
  const fa = S.setup.fire_area;
  $('fireNote').textContent = fa ? `New storms fire only inside the orange area (${Math.round(Math.abs(fa[2] - fa[0]) / 1000)} x ` +
    `${Math.round(Math.abs(fa[3] - fa[1]) / 1000)} km).` : 'No firing area: storms can fire anywhere.';
  $('synDomainNote').textContent = fa ? `Storm domain: ${Math.round(Math.abs(fa[2] - fa[0]) / 1000)} x ` +
    `${Math.round(Math.abs(fa[3] - fa[1]) / 1000)} km (orange box). New storms form only inside it.` : 'No storm domain: storms form anywhere the weather allows.';
}

function onHover(x, y) {
  if (S.tool === 'front') Fronts.hover(x, y);
  let txt = `x ${x.toFixed(1)}  y ${y.toFixed(1)} km`;
  if (map.proj) { const [lat, lon] = map.inv(x, y); txt += `\n${lat.toFixed(3)}°, ${lon.toFixed(3)}°`; }
  const v = valueAt(x, y);
  if (v) txt += `\n${v}`;
  $('readout').textContent = txt;
}

function valueAt(x, y) {
  if (S.ppiInfo) {
    const m = S.ppiInfo.meta, dx = x - m.x / 1000, dy = y - m.y / 1000;
    const r = Math.hypot(dx, dy) / Math.cos(m.elev * Math.PI / 180);
    const gi = Math.floor((r - m.first / 1000) / (m.gate / 1000) + 0.5);
    let az = Math.atan2(dx, dy) * 57.29578 + (m.conv || 0); if (az < 0) az += 360;
    const ri = Math.floor(az / (360 / m.nrad)) % m.nrad;
    if (gi < 0 || gi >= m.ngate) return null;
    const c = S.ppiInfo.codes[ri * m.ngate + gi];
    return `${m.icao} ${S.product}: ${c ? ((c - m.offset) / m.scale).toFixed(1) : 'no echo'}   az ${az.toFixed(1)}° rng ${r.toFixed(1)} km`;
  }
  if (S.rasterInfo) {
    // the nests come after their parents (finer cells, drawn on top): the last one under the cursor is the finest
    for (const p of [...S.rasterInfo.patches].reverse()) {
      if (x < p.x0 || x >= p.x1 || y < p.y0 || y >= p.y1) continue;
      const i = Math.floor((x - p.x0) / (p.x1 - p.x0) * p.nx), j = Math.floor((y - p.y0) / (p.y1 - p.y0) * p.ny);
      const c = p.codes[j * p.nx + i];
      const units = S.rasterInfo.header.codec[2];
      return c ? `${((c - S.rasterInfo.offset) / S.rasterInfo.scale).toFixed(S.product === 'vort' ? 4 : 1)} ${units}` : 'no data';
    }
  }
  return null;
}

// a Doppler on Wheels parked at (x, y) km: scanning now in a run (else the next run's), its spot into GR2Analyst's list
async function deployDow(x, y) {
  const [lat, lon] = map.inv(x, y);
  const icao = $('dowId').value;
  const r = { icao, kind: 'dow', lat: +lat.toFixed(5), lon: +lon.toFixed(5), gate: +$('dowGate').value,
    dealias: $('radarDealias').checked, floor: +$('radarFloor').value, noise: $('radarNoise').checked,
    clear_air: Math.max(0, Math.min(3, +$('radarClearAir').value || 0)), export: true, x: x * 1000, y: y * 1000 };
  if (runActive()) await api('/api/run/radar', r);
  S.setup.radars = S.setup.radars.filter(q => q.icao !== icao).concat([r]);
  renderLists(); fillRadarSelect();
  S.radar = icao;
  map.redraw();
  if (!$('dowGr2').checked) return;
  const hint = $('toolhint');
  hint.textContent = `${icao}: waiting for the Windows permission prompt (GR2Analyst's site list)...`;
  hint.classList.remove('hidden');
  try {
    const d = await getJSON('/api/gr2/dow', { icao, lat: r.lat, lon: r.lon });
    hint.textContent = d.written
      ? `${icao} is in GR2Analyst's list at ${r.lat.toFixed(3)}, ${r.lon.toFixed(3)}` + (d.gr2_running ? ': restart GR2Analyst to see it there' : '')
      : `${icao}: GR2Analyst's site list was not changed (permission declined?)`;
  } catch (e) { hint.textContent = `${icao}: GR2Analyst's site list: ` + e.message; }
  setTimeout(() => { if (S.tool === 'pan') hint.classList.add('hidden'); }, 12000);
}

// ------------------------------------------------------------------ lists / log
function renderLists() {
  const run = runActive() ? S.run : null;
  const radars = run ? run.radars : S.setup.radars;
  const every = r => { const m = r.volume_min ?? 2.5; return m > 0 ? ` · every ${m < 1 ? Math.round(m * 60) + ' s' : m + ' min'}` : ''; };
  setHTML('radarList', radars.map(r => `<div class="item"><div><b>${r.icao}</b> <span class="meta">` +
    (r.kind === 'dow' ? `DOW, X band, ${r.gate || 75} m gates at ${(+r.lat).toFixed(3)}, ${(+r.lon).toFixed(3)}`
      : `VCP ${r.vcp}${r.sails ? ' SAILS×' + r.sails : ''}${every(r)}`) +
    `${r.dealias === false ? ' · aliased' : ''}` +
    `${r.noise === false ? ' · clean' : ''}` +
    `${r.volumes !== undefined ? ' · ' + r.volumes + ' volumes' : ''}</span></div>` +
    `<button data-radar="${r.icao}">${r.kind === 'dow' ? 'Pick up' : 'Remove'}</button></div>`).join('') || '<div class="hint">No radars yet.</div>');
  const starts = run ? (run.starts || []).filter(s => s.why !== 'tower' && s.why !== 'thermal' && s.why !== 'latch') : S.setup.storms;   // not the satellites
  const zones = run ? (run.zones || []) : S.setup.zones;
  setHTML('stormList',
    starts.map((s, i) => `<div class="item"><div>Storm ${s.id || i + 1} <span class="meta">${s.why || s.kind} at ${(s.x / 1000).toFixed(0)}, ` +
      `${(s.y / 1000).toFixed(0)} km${s.t0 !== undefined ? ', +' + fmtDur(s.t0) : ''}` +
      `${s.waiting ? ' · waiting for GPU memory' : ''}</span></div>` +
      (run ? '' : `<button data-storm="${i}">Remove</button>`) + '</div>').join('') +
    zones.map((z, i) => `<div class="item"><div>Zone ${z.id || i + 1} <span class="meta">${z.rate}/h, ` +
      `${((z.x1 - z.x0) / 1000).toFixed(0)}×${((z.y1 - z.y0) / 1000).toFixed(0)} km</span></div>` +
      `<button data-zone="${run ? z.id : i}">Remove</button></div>`).join('') +
    // a run's patches (not a fixed domain's): one in the wrong place can be deleted with its storms
    (run && !run.fixed_cells ? (run.patches || []).map(p => `<div class="item"><div>Patch ${p.id} <span class="meta">` +
      `${p.nx}×${p.ny} cells, ${((p.x1 - p.x0) / 1000).toFixed(0)}×${((p.y1 - p.y0) / 1000).toFixed(0)} km` +
      `${p.held < 0 ? ' · locked' : p.held > 0 ? ` · held ${Math.ceil(p.held / 60)} min` : ''}</span></div>` +
      (p.held ? `<button data-release="${p.id}" title="Follow its storms again now">Follow</button>` : '') +
      `<button data-patch="${p.id}">Delete</button></div>`).join('') : ''));
  // nudger cells: the scheduled ones (the setup's, or the run's), then those this run has sent
  const plans = run ? (run.nudger_plans || []) : (S.setup.nudgers || []);
  const side = a => !+a ? 'into the updraft' : `${Math.abs(a)} km ${a > 0 ? 'ahead of' : 'behind'} it`;
  const sent = S.run && !S.detached ? (S.run.nudgers || []) : [];
  setHTML('nudgerList',
    plans.map((n, i) => `<div class="item"><div>Nudger at +${n.at_min} min <span class="meta">meets the strongest ` +
      `storm ${n.lead_min ?? 20} min later, ${side(n.aim_km)} (${n.motion || 'mean'})${n.sent ? ' · sent' : ''}</span></div>` +
      (n.sent ? '' : `<button data-ndplan="${run ? n.id : i}">Remove</button>`) + '</div>').join('') +
    sent.map(n => `<div class="item"><div>Nudger #${n.id} <span class="meta">at storm ${n.target}, sent +${fmtDur(n.sent)}, ` +
      `meets +${fmtDur(n.meet)} · ${n.status}` + (n.closest != null ? ` · closest ${(n.closest / 1000).toFixed(1)} km at ` +
      `+${fmtDur(n.closest_t)}` : '') + '</span></div></div>').join(''));
  Fronts.render();
  showFireNote();
  renderBrushes();
}

function setHTML(id, html) {
  const el = $(id);
  if (el._last !== html) { el.innerHTML = html; el._last = html; }
}

function fillRadarSelect() {
  const sel = $('radarSel');
  const radars = (S.run && !S.detached ? S.run.radars : S.setup.radars).map(r => r.icao);
  // the last radar picked, if this run has it: a radar of an earlier run would select nothing
  const cur = radars.includes(S.radar) ? S.radar : radars[0];
  if ([...sel.options].map(o => o.value).join() !== radars.join()) {
    sel.innerHTML = '';
    for (const r of radars) sel.add(new Option(r, r));
  }
  if (cur) sel.value = cur;
  S.radar = sel.value || null;
}

function renderLog(events) {
  const el = $('log');
  const html = events.slice().reverse().map(e => `<div class="ev-${esc(e.kind)}">+${fmtDur(e.t)}  ${esc(e.kind)}: ${esc(e.text)}</div>`).join('');
  if (el._last !== html) { el.innerHTML = html; el._last = html; }
}

// ------------------------------------------------------------------ bindings
// light (default) or dark: the header's theme button, remembered in this browser
function applyTheme(dark) {
  if (dark) document.documentElement.setAttribute('data-theme', 'dark');
  else document.documentElement.removeAttribute('data-theme');
  const b = $('themeToggle');
  if (b) { b.textContent = dark ? '\u2600' : '\u263e'; b.title = dark ? 'Light mode' : 'Dark mode'; }
  try { localStorage.setItem('arht-theme', dark ? 'dark' : 'light'); } catch (e) { /* private window */ }
  if (typeof map !== 'undefined' && map && map.redraw) map.redraw();
}

function bindUI() {
  applyTheme(document.documentElement.getAttribute('data-theme') === 'dark');
  $('themeToggle').onclick = () => applyTheme(document.documentElement.getAttribute('data-theme') !== 'dark');
  $('libScan').onclick = () => libraryCall('/api/library/scan', $('libScan').textContent.startsWith('Stop') ? { stop: true } : {});
  $('libRefresh').onclick = () => libraryCall('/api/library/refresh', {});
  $('libAdd').onclick = async () => {
    const path = $('libPath').value.trim();
    if (!path) return;
    await libraryCall('/api/library/folder', { path });
    $('libPath').value = '';
  };
  $('libBrowse').onclick = async () => {
    const d = await getJSON('/api/library/browse', {});         // the system's folder dialog
    if (d.path) { $('libPath').value = d.path; $('libAdd').onclick(); }
  };
  $('libFolders').addEventListener('click', e => {
    const b = e.target.closest('button[data-libremove]');
    if (b) libraryCall('/api/library/folder', { path: b.dataset.libremove, remove: true });
  });
  $('libOpenSnd').onchange = async e => { await importFiles('sounding', [...e.target.files]); e.target.value = ''; };
  $('libOpenPal').onchange = async e => { await importFiles('palette', [...e.target.files]); e.target.value = ''; };
  $('sndOpen').onchange = async e => {
    const got = await importFiles('sounding', [...e.target.files]);
    e.target.value = '';
    if (got.length) { $('sndFilter').value = ''; fillSoundingList(''); $('sndList').value = got[0].path; await loadSounding({ file: got[0].path }); }
  };
  $('palOpen').onchange = async e => {
    const got = await importFiles('palette', [...e.target.files]);
    e.target.value = '';
    const fit = got.find(g => [...$('palette').options].some(o => o.value === g.path));
    if (fit) { $('palette').value = fit.path; showFrame(); }
  };
  for (const id of ['olOn', 'olTowers', 'olMergers', 'olRate', 'olArea', 'olStart', 'olEnd', 'olPush', 'olRadius', 'olTexture']) $(id).onchange = onAutoChange;
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.id === 'tab-' + b.dataset.tab));
  });
  $('sndSource').onchange = () => {
    const v = $('sndSource').value;
    $('sndLibrary').classList.toggle('hidden', v !== 'library');
    $('sndPaste').classList.toggle('hidden', v !== 'paste');
    $('sndWK').classList.toggle('hidden', v !== 'wk');
  };
  SndEdit.init();
  Presets.init();
  Trop.init();
  Metar.init();
  Vol.init();
  $('saveSetup').onclick = () => Presets.save(false);
  $('resetSetup').onclick = resetSetup;
  makeCollapsible();
  $('setupSummary').addEventListener('click', e => {
    const a = e.target.closest('[data-go]');
    if (a) document.querySelector(`.tabs button[data-tab="${a.dataset.go}"]`).click();
  });
  $('showPresets').onclick = () => document.querySelector('.tabs button[data-tab="files"]').click();
  $('sndEdit').onclick = () => SndEdit.open();
  SkewTLabView.init();
  $('sndSkewtLab').onclick = () => SkewTLabView.open();
  $('skewt').ondblclick = () => SndEdit.open();
  $('sndFilter').oninput = () => fillSoundingList($('sndFilter').value);
  $('sndList').onchange = () => loadSounding({ file: $('sndList').value });
  $('sndParse').onclick = () => loadSounding({ text: $('sndText').value });
  $('wkApply').onclick = () => loadSounding({ analytic: { theta_sfc: +$('wkTh').value, qv0: +$('wkQv').value,
    shear: +$('wkShear').value, hodograph: $('wkHodo').value } });
  $('persist').oninput = () => { $('persistVal').textContent = (+$('persist').value).toFixed(2); };
  $('optSteady').addEventListener('change', showHeatHint);
  $('heatHours').addEventListener('input', showHeatHint);
  $('heatHours').onchange = $('heatFlux').onchange =
  $('mixOn').onchange = $('mixDepth').onchange = $('mixHeat').onchange = $('windScale').onchange =
    $('iceCap').onchange = $('lljSpeed').onchange = $('lljHeight').onchange =
    $('moistDtd').onchange = $('moistDepth').onchange = $('capCin').onchange = $('capBase').onchange =
    $('midRh').onchange = $('midBottom').onchange = $('midTop').onchange =
    $('aloftSpeed').onchange = $('aloftBase').onchange = () => loadSounding(S.setup.sounding);
  // the Atmosphere tab's own sub-tabs: Sounding / Advanced
  document.querySelectorAll('.subtabs button').forEach(b => b.onclick = () => {
    b.parentElement.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    document.querySelectorAll('#tab-atmos .sub').forEach(s => s.classList.toggle('on', s.id === 'sub-' + b.dataset.sub));
  });
  $('pickCenter').onclick = () => setTool('center');
  $('toolStorm').onclick = () => setTool('storm');
  $('toolZone').onclick = () => setTool('zone');
  $('toolFire').onclick = () => setTool('fire');
  $('toolVolume').onclick = () => setTool('volume');
  $('toolTornado').onclick = () => setTool('tornado');
  $('toolMove').onclick = () => setTool('move');
  $('tornadoView').onclick = () => Vol.openTornado();
  $('tornadoRemove').onclick = async () => {
    Vol.closeTornado();
    if (runActive()) { try { await api('/api/run/tornado/remove', {}); } catch (e) { alert(e.message); } }
  };
  $('toolBrush').onclick = () => { map.brushKm = +$('brSize').value; setTool('brush'); };
  $('brSize').onchange = () => { map.brushKm = +$('brSize').value; };
  $('brClear').onclick = () => { S.setup.wind_brushes = []; pushBrushes(); };
  $('brushList').addEventListener('click', e => {
    const b = e.target.closest('button[data-brush]');
    if (b) { brushList().splice(+b.dataset.brush, 1); pushBrushes(); }
  });
  $('fireClear').onclick = () => setFireArea(null);
  $('synDomainTool').onclick = () => setTool('fire');
  $('synDomainClear').onclick = () => setFireArea(null);
  $('ndSend').onclick = async () => {
    if (!runActive()) { alert('Nudgers go at the storms of a running run: start one, or schedule a nudger in the setup.'); return; }
    if ($('ndTarget').value === 'click') { setTool('nudger'); return; }
    await api('/api/run/nudger', { ...nudgerForm(), target: 'strongest' });
  };
  $('ndPlan').onclick = async () => {
    const n = { ...nudgerForm(), at_min: Math.max(0, +$('ndAt').value || 0) };
    if (runActive()) await api('/api/run/nudger/plan', n);
    else { S.setup.nudgers.push(n); renderLists(); }
  };
  $('toolRadar').onclick = () => setTool('radar');
  $('toolDow').onclick = () => setTool('dow');
  // the phone remote (remote.py): its switch, PINs and addresses
  const rmShow = d => { $('rmOn').checked = !!d.on; $('rmWatch').value = d.watch_pin; $('rmControl').value = d.control_pin;
    $('rmUrl').textContent = (d.urls || []).join('  or  '); };
  const rmSave = async extra => {
    try { rmShow(await getJSON('/api/remote/config', Object.assign({ on: $('rmOn').checked, watch_pin: $('rmWatch').value,
      control_pin: $('rmControl').value }, extra || {}))); } catch (e) { alert('Phone remote: ' + e.message); }
  };
  fetch('/api/remote/config').then(r => r.json()).then(rmShow).catch(() => {});
  $('rmOn').onchange = () => rmSave();
  $('rmWatch').onchange = $('rmControl').onchange = () => rmSave();
  $('rmNew').onclick = () => rmSave({ new_pins: true });
  $('toolLoop').onclick = () => LoopMaker.open(S.detached ? null : (S.run && S.run.run_id));
  $('toolSurvey').onclick = () => Survey.open(S.detached ? null : (S.run && S.run.run_id));
  $('lat0').onchange = $('lon0').onchange = async () => { readSetupFromForm(); if (!runActive()) await recenterSetup(); };
  $('domainMode').onchange = $('domainSize').oninput = $('domainMax').oninput = $('grid').onchange = showDomain;
  $('refineRatio').onchange = () => $('refineRow').classList.toggle('hidden', !(+$('refineRatio').value));
  $('tornadoDetail').onchange = changeZoomDetail;       // a running zoom switches in place
  $('terrainKind').onchange = showTerrain;
  $('newRun').onclick = async () => {
    readSetupFromForm();
    const s = S.setup;
    if (!(s.storms || []).length && !(s.zones || []).length && !(s.boundaries || []).length && !s.auto_storms.on &&
        !(s.synoptic && s.synoptic.on)) {
      // the viewer only computes the air around storms: an empty run shows nothing
      if (confirm('This run has no cells, initiation zones, fronts or auto outbreak yet, so the map would stay empty.\n\n' +
                  'Start one storm at the map center? (Cancel starts the run empty; you can place storms later.)')) {
        s.storms = [{ x: 0, y: 0, kind: $('stormKind').value, motion: $('stormMotion').value,
                      strength: +$('stormStrength').value, radius: +$('stormRadius').value * 1000 }];
        renderLists();
      }
    }
    if (s.domain.mode === 'fixed' && (s.storms || []).length > 1) {
      // the first storm places the domain; one too far from it for both to fit does not start
      const xs = s.storms.map(q => q.x / 1000), ys = s.storms.map(q => q.y / 1000);
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      if (span > s.domain.size_km - 60 && !confirm(`These storms are up to ${span.toFixed(0)} km apart, too far for ` +
          `the fixed ${s.domain.size_km} km domain to hold them all: the ones that do not fit will not start.\n\nStart anyway?`)) return;
    }
    if (runActive() && !confirm('Stop the current run and start a new one?')) return;
    $('newRun').disabled = true;
    try {
      S.run = null; S.frames = []; S.lastFrameCount = -1; S.live = true; S.mapKey = ''; S.detached = false;
      map.setRaster(null); map.setPPI(null);
      await api('/api/run/new', { settings: S.setup, start: true });
      setStatus('building', 'Building… (the first run compiles GPU kernels, ~1 min)');
    } catch (e) { alert('Could not start: ' + e.message); }
    $('newRun').disabled = false;
  };
  $('btnPause').onclick = async () => {
    if (!S.run) return;
    await api(S.run.status === 'paused' ? '/api/run/resume' : '/api/run/pause', {});
  };
  $('btnStop').onclick = async () => { if (S.run && confirm('Stop this run? (frames and Level II files are kept)')) await api('/api/run/stop', {}); };
  $('speed').onchange = async () => { if (S.run) await api('/api/run/speed', { limit: +$('speed').value }); };
  $('gridMotion').onchange = showGridMotion;
  $('gridApply').onclick = async () => {
    if (!runActive()) return;
    try { await api('/api/run/grid', { motion: gridMotionFromForm(), follow_min: +$('gridFollowMin').value }); }
    catch (e) { alert('Could not change the grid motion: ' + e.message); }
  };
  $('product').onchange = () => { S.product = $('product').value; fillPaletteSelect(); showFrame(); };
  $('radarSel').onchange = () => { S.radar = $('radarSel').value; showFrame(); };
  $('palette').onchange = () => showFrame();
  $('smooth').onchange = () => {
    if (map.raster) map.raster.smooth = $('smooth').checked;
    if (map.ppi) { map.ppi.smooth = $('smooth').checked; map._ppiCache = null; }
    map.redraw();
  };
  $('showPatches').onchange = () => map.redraw();
  $('showCounties').onchange = () => { map.showCounties = $('showCounties').checked; map.redraw(); };
  {
    let b = 'plain';
    try { b = localStorage.getItem('arht.basemap') || 'plain'; } catch (e) { /* private window */ }
    if (![...$('basemap').options].some(o => o.value === b)) b = 'plain';
    $('basemap').value = b; map.setBase(b);
    $('basemap').onchange = () => {
      map.setBase($('basemap').value);
      try { localStorage.setItem('arht.basemap', $('basemap').value); } catch (e) { /* private window */ }
      if (window.volBaseChanged) window.volBaseChanged($('basemap').value);
    };
  }
  $('showTorTracks').onchange = () => map.redraw();
  $('showWarnings').onchange = () => map.redraw();
  $('showStarts').onchange = () => map.redraw();
  // a TDWR's velocities come folded (its Nyquist is ~20 m/s): dealiasing off by default for one
  // the world radars into GR2Analyst's own site list (gr2sites.py): one Windows permission prompt
  const gr2Note = d => { $('gr2SitesNote').textContent = !d.found ? 'GR2Analyst was not found on this PC.'
    : d.missing ? `${d.missing} of ${d.world} world radars are not in GR2Analyst's site list yet.`
    : `All ${d.world} world radars are in GR2Analyst's site list (restart GR2Analyst if it was open).`; };
  getJSON('/api/gr2/sites', {}).then(gr2Note).catch(() => {});
  $('gr2Sites').onclick = async () => {
    $('gr2SitesNote').textContent = 'Waiting for the Windows permission prompt...';
    try { gr2Note(await getJSON('/api/gr2/sites', { install: true })); }
    catch (e) { $('gr2SitesNote').textContent = 'Could not add them: ' + e.message; }
  };
  $('radarKind').onchange = () => { $('radarDealias').checked = $('radarKind').value !== '3'; map.redraw(); };
  $('rpSlider').oninput = () => { S.live = false; $('rpLive').classList.remove('on'); S.frameIdx = +$('rpSlider').value; showFrame(); };
  $('rpPrev').onclick = () => step(-1);
  $('rpNext').onclick = () => step(1);
  $('rpLive').onclick = () => { S.live = true; $('rpLive').classList.add('on'); S.frameIdx = S.frames.length - 1; $('rpSlider').value = S.frameIdx; showFrame(); };
  $('rpPlay').onclick = togglePlay;
  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.code === 'ArrowLeft') step(-1);
    if (e.code === 'ArrowRight') step(1);
    if (e.code === 'Escape') setTool('pan');
    if ((e.code === 'Enter' || e.code === 'NumpadEnter') && S.tool === 'front') Fronts.finish();
  });
  document.body.addEventListener('click', async e => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.radar) {
      const icao = b.dataset.radar;
      if (runActive()) await api('/api/run/radar/remove', { icao });
      S.setup.radars = S.setup.radars.filter(r => r.icao !== icao);       // the next run's too
      renderLists(); fillRadarSelect();
    } else if (b.dataset.storm !== undefined) {
      S.setup.storms.splice(+b.dataset.storm, 1); renderLists(); map.redraw();
    } else if (b.dataset.ndplan !== undefined) {
      if (runActive()) await api('/api/run/nudger/remove', { id: +b.dataset.ndplan });
      else { S.setup.nudgers.splice(+b.dataset.ndplan, 1); renderLists(); }
    } else if (b.dataset.patch !== undefined) {
      if (runActive() && confirm(`Delete patch ${b.dataset.patch} and the storms in it?`))
        await api('/api/run/patch/remove', { id: +b.dataset.patch });
    } else if (b.dataset.release !== undefined) {
      if (runActive()) await api('/api/run/patch/release', { id: +b.dataset.release });
    } else if (b.dataset.zone !== undefined) {
      if (runActive()) await api('/api/run/zone/remove', { id: +b.dataset.zone });
      else { S.setup.zones.splice(+b.dataset.zone, 1); renderLists(); map.redraw(); }
    }
  });
  fillPaletteSelect();
}

function step(d) {
  if (!S.frames.length) return;
  S.live = false; $('rpLive').classList.remove('on');
  const i = S.frameIdx < 0 ? S.frames.length - 1 : S.frameIdx;
  S.frameIdx = Math.max(0, Math.min(S.frames.length - 1, i + d));
  $('rpSlider').value = S.frameIdx;
  showFrame();
}

function togglePlay() {
  S.playing = !S.playing;
  $('rpPlay').textContent = S.playing ? '❚❚' : '▶';
  if (S.playing) {
    S.live = false; $('rpLive').classList.remove('on');
    const tick = async () => {
      if (!S.playing) return;
      if (S.frameIdx >= S.frames.length - 1) S.frameIdx = 0; else S.frameIdx++;
      $('rpSlider').value = S.frameIdx;
      await showFrame();
      setTimeout(tick, 250);
    };
    tick();
  }
}

boot();
