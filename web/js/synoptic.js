'use strict';
// Synoptic mode (synoptic.py): the Weather tab, the simulated weather's maps (simulated radar and satellite,
// shaded fields, isobars, 500 hPa heights, fronts and drylines, highs and lows, wind barbs) and soundings
// anywhere on the map.  A preview (any forecast hour, looped) and a running run's weather are kept apart:
// the map shows the one picked with "Showing", and a preview stays until another is made.

const Syn = (() => {
  const st = {
    preview: null, runData: null, show: 'preview', layer: 'sfcradar', canvas: null, canvasKey: '',
    building: false, fetchingRun: false, lastRun: 0, runId: null, frames: {}, framesKey: '', ready: [],
    playing: false, playTimer: null, point: null,
  };

  // colour ramps: [value, r, g, b, a]
  const RAMPS = {
    t2m: [[-10, 80, 60, 170, 150], [0, 70, 130, 220, 150], [10, 60, 190, 170, 150], [20, 220, 210, 80, 150],
          [30, 240, 120, 40, 160], [40, 190, 30, 60, 170]],
    td2m: [[-15, 140, 90, 50, 140], [0, 190, 160, 100, 120], [10, 120, 190, 110, 130], [16, 40, 170, 80, 160],
           [20, 20, 130, 160, 170], [25, 90, 60, 190, 180]],
    mlcape: [[0, 0, 0, 0, 0], [250, 240, 230, 120, 90], [1000, 250, 190, 60, 150], [2000, 245, 120, 40, 170],
             [3000, 220, 40, 50, 180], [5000, 190, 40, 200, 190]],
    mlcin: [[-300, 40, 70, 200, 170], [-150, 70, 120, 230, 140], [-50, 120, 170, 240, 100], [-10, 170, 200, 250, 40],
            [0, 0, 0, 0, 0]],
    stp: [[0, 0, 0, 0, 0], [0.3, 250, 230, 140, 90], [1, 250, 200, 60, 150], [3, 240, 90, 40, 175],
          [6, 210, 30, 90, 190], [10, 170, 40, 210, 200]],
    srh1: [[-100, 0, 0, 0, 0], [50, 0, 0, 0, 0], [100, 150, 210, 240, 110], [200, 90, 150, 240, 150],
           [300, 200, 90, 220, 170], [600, 240, 60, 120, 190]],
    shear6: [[0, 0, 0, 0, 0], [10, 170, 220, 200, 80], [20, 90, 190, 160, 130], [30, 40, 140, 200, 160],
             [40, 120, 70, 200, 180]],
    ci: [[0, 0, 0, 0, 0], [0.05, 250, 240, 160, 80], [0.5, 250, 180, 60, 150], [1.5, 230, 70, 50, 180],
         [3, 190, 30, 140, 200]],
    wspd500: [[0, 0, 0, 0, 0], [20, 160, 200, 240, 60], [30, 110, 160, 240, 120], [45, 160, 100, 240, 160],
              [60, 230, 90, 200, 190]],
    // the NWS reflectivity scale
    dbz: [[5, 4, 233, 231, 150], [10, 1, 159, 244, 170], [15, 3, 0, 244, 190], [20, 2, 253, 2, 210],
          [25, 1, 197, 1, 215], [30, 0, 142, 0, 220], [35, 253, 248, 2, 225], [40, 229, 188, 0, 230],
          [45, 253, 149, 0, 235], [50, 253, 0, 0, 240], [55, 212, 0, 0, 245], [60, 188, 0, 0, 245],
          [65, 248, 0, 253, 250], [70, 152, 84, 198, 250], [75, 255, 255, 255, 255]],
    // infrared: the warm ground dark, cold tops white, the coldest (deep convection) enhanced
    ctt: [[40, 25, 25, 28, 235], [20, 55, 55, 60, 235], [0, 110, 110, 115, 235], [-20, 175, 175, 180, 240],
          [-35, 230, 230, 235, 245], [-45, 90, 180, 255, 250], [-55, 30, 90, 230, 250], [-62, 250, 220, 40, 255],
          [-70, 240, 60, 40, 255], [-80, 255, 255, 255, 255]],
  };
  // what each layer shades, and whether it draws the surface analysis over it
  const LAYERS = {
    sfcradar: { field: 'dbz', unit: 'dBZ', label: 'Surface analysis + simulated radar', analysis: true },
    radar: { field: 'dbz', unit: 'dBZ', label: 'Simulated radar (composite)' },
    sat: { field: 'ctt', unit: '°C', label: 'Simulated infrared satellite', analysis: 'fronts', alpha: 1 },
    lines: { field: null, label: 'Fronts, isobars, highs and lows', analysis: true },
    t2m: { field: 't2m', unit: '°C', label: '2 m temperature', analysis: true },
    td2m: { field: 'td2m', unit: '°C', label: '2 m dewpoint', analysis: true },
    mlcape: { field: 'mlcape', unit: 'J/kg', label: 'Mixed-layer CAPE', analysis: 'fronts' },
    mlcin: { field: 'mlcin', unit: 'J/kg', label: 'Mixed-layer CIN (the cap)', analysis: 'fronts' },
    stp: { field: 'stp', unit: '', label: 'Significant tornado parameter', analysis: 'fronts' },
    srh1: { field: 'srh1', unit: 'm²/s²', label: '0–1 km storm-relative helicity', analysis: 'fronts' },
    shear6: { field: 'shear6', unit: 'm/s', label: '0–6 km shear', analysis: 'fronts' },
    ci: { field: 'ci', unit: '/h /10⁵ km²', label: 'Where storms can form', analysis: 'fronts' },
    z500: { field: 'wspd500', unit: 'm/s', label: '500 hPa heights, wind and jet', z500: true },
    none: { field: null, label: '' },
  };

  function rampColor(ramp, v) {
    if (v <= ramp[0][0]) return ramp[0].slice(1);
    for (let i = 1; i < ramp.length; i++) {
      if (v <= ramp[i][0]) {
        const a = ramp[i - 1], b = ramp[i], t = (v - a[0]) / (b[0] - a[0]);
        return [1, 2, 3, 4].map(k => a[k] + (b[k] - a[k]) * t);
      }
    }
    return ramp[ramp.length - 1].slice(1);
  }
  // ramps listed warm-to-cold (ctt) are walked from the other end
  function colorFor(field, v) {
    const r = RAMPS[field];
    if (r[0][0] > r[r.length - 1][0]) return rampColor(r.slice().reverse(), v);
    return rampColor(r, v);
  }

  const on = () => $('synOn').checked;
  const hrrrOn = () => $('synSource').value === 'hrrr';
  // ERA5 rides the HRRR's path (its panel, maps, storm domain and soundings): the same settings, model 'era5'
  const era5On = () => hrrrOn() && $('hrrrModel').value === 'era5';
  const pad2 = n => String(n).padStart(2, '0');
  // the HRRR run picked in the form: {cycle, start_fh, hours}, or null when the date is not set
  function hrrrForm() {
    const date = $('hrrrDate').value;
    if (!date) return null;
    const num = (id, lo, hi, d) => { const v = parseInt($(id).value, 10); return isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; };
    const cyc = +$('hrrrCycle').value, maxFh = cyc % 6 ? 18 : 48;
    if (era5On()) return { cycle: `${date}T${pad2(cyc)}:00:00Z`, start_fh: 0, hours: num('hrrrHours', 1, 72, 12), model: 'era5' };
    const start = num('hrrrStart', 0, maxFh - 1, 0);
    return { cycle: `${date}T${pad2(cyc)}:00:00Z`, start_fh: start, hours: num('hrrrHours', 1, maxFh - start, 12) };
  }
  // the time the run starts at (cycle + start hour) as the Domain tab's start time
  function hrrrStartTime(h) {
    const t = new Date(Date.parse(h.cycle) + h.start_fh * 3600e3);
    return t.toISOString().slice(0, 16);
  }
  const runSyn = () => runActive() && S.run && S.run.synoptic;
  const showingRun = () => st.show === 'run' && runSyn();
  const current = () => (showingRun() ? st.runData : st.preview);

  // ------------------------------------------------------------ settings
  function read() {
    const num = (id, lo, hi, d) => { const v = parseFloat($(id).value); return isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; };
    const cur = (S.setup && S.setup.synoptic) || {};
    return { on: on(), scenario: $('synScenario').value, peak_h: num('synPeak', 0, 24, 6), area_km: num('synArea', 0, 5000, 0),
             rate: num('synRate', 0, 10, 1.5), max_storms: Math.round(num('synMax', 1, 60, 10)),
             spacing_km: num('synSpacing', 10, 200, 45), area_v: 2, seed: cur.seed || null,
             source: hrrrOn() ? 'hrrr' : 'sim', hrrr: hrrrForm(), hrrr_mode: $('hrrrMode').value };
  }

  function apply(cfg) {
    const old = cfg && Object.keys(cfg).length && cfg.area_v === undefined;   // saved before "anywhere" existed
    cfg = { ...((S.defaults && S.defaults.settings.synoptic) || {}), ...(cfg || {}) };
    if (old) cfg.area_km = 0;
    $('synOn').checked = !!cfg.on;
    $('synScenario').value = cfg.scenario || 'outbreak';
    $('synPeak').value = cfg.peak_h ?? 6;
    $('synArea').value = cfg.area_km ?? 0;
    $('synRate').value = cfg.rate ?? 1.5;
    $('synMax').value = cfg.max_storms ?? 10;
    $('synSpacing').value = cfg.spacing_km ?? 45;
    $('synSource').value = cfg.source === 'hrrr' ? 'hrrr' : 'sim';
    $('hrrrModel').value = cfg.hrrr && cfg.hrrr.model === 'era5' ? 'era5' : 'hrrr';
    $('hrrrMode').value = cfg.hrrr_mode === 'run' ? 'run' : 'time';
    const h = cfg.hrrr;
    if (h && h.cycle) {
      $('hrrrDate').value = h.cycle.slice(0, 10);
      $('hrrrCycle').value = String(+h.cycle.slice(11, 13));
      $('hrrrStart').value = h.start_fh ?? 0;
      $('hrrrHours').value = h.hours ?? 12;
      $('hrrrValid').value = hrrrStartTime(h);
    }
    showMode();
  }

  function store() {
    if (!S.setup) return;
    S.setup.synoptic = read();
    // an HRRR run starts at its cycle plus the start hour: the start time follows it
    const h = S.setup.synoptic.hrrr;
    if (S.setup.synoptic.source === 'hrrr' && h) $('startTime').value = hrrrStartTime(h);
  }

  function showMode() {
    const o = on();
    document.body.classList.toggle('synoptic', o);
    const hr = o && hrrrOn();
    // the HRRR is the storm environment: its panel (run, maps, domain, soundings) sits in the Atmosphere tab
    const e5 = hr && era5On();
    $('envSource').value = !o ? 'sounding' : hr ? (e5 ? 'era5' : 'hrrr') : 'sim';
    document.querySelectorAll('#synHrrrBox .hrrrOnly').forEach(e => e.classList.toggle('hidden', e5));
    document.querySelectorAll('#synHrrrBox .era5Only').forEach(e => e.classList.toggle('hidden', !e5));
    $('hrrrCycleLab').textContent = e5 ? 'Start hour' : 'Cycle';
    $('hrrrDate').min = e5 ? '1940-01-01' : '2014-07-30';
    $('hrrrHours').max = e5 ? 72 : 48;
    [...$('hrrrCycle').options].forEach((op, hh) => { op.text = `${pad2(hh)}Z${e5 || hh % 6 ? '' : ' (48 h)'}`; });
    $('synHrrrElsewhere').innerHTML = e5
      ? '<b>ERA5 is the storm environment</b> (Atmosphere tab, Storm environment): its day, maps, storm domain and soundings are there.'
      : '<b>NOAA HRRR is the storm environment</b> (Atmosphere tab, Storm environment): its run, maps, storm domain and soundings are there.';
    const panel = $('synPanel');
    if (hr && panel.parentElement !== $('envHrrrSlot')) $('envHrrrSlot').appendChild(panel);
    if (!hr && panel.nextElementSibling !== $('synWeatherSlot')) $('synWeatherSlot').before(panel);
    $('synHrrrElsewhere').classList.toggle('hidden', !hr);
    $('synHrrrBox').classList.toggle('hidden', !hr);
    $('hrrrTimeRow').classList.toggle('hidden', e5 || $('hrrrMode').value !== 'time');
    $('synSimHint').classList.toggle('hidden', hr);
    $('synAtmosNote').innerHTML = e5
      ? '<b>ERA5 is the storm environment</b>: every storm patch takes ERA5’s soundings for that day where it is, hour by hour, and the run’s base sounding is the ERA5 column at the map centre at the start. The sounding below is not used.'
      : hr
      ? '<b>NOAA HRRR is the storm environment</b>: every storm patch takes the HRRR’s soundings where it is, hour by hour, and the run’s base sounding is the HRRR column at the map centre at the start. The sounding below is not used.'
      : '<b>Synoptic mode is on</b> (Weather tab): the run’s air comes from the simulated weather, and its base sounding is the simulated column at the map centre at the start. The sounding below is not used.';
    document.querySelectorAll('.synSimOnly').forEach(e => e.classList.toggle('hidden', hr));
    $('synAtmosNote').classList.toggle('hidden', !o);
    $('synSeed').textContent = hr ? '' : (S.setup && S.setup.synoptic && S.setup.synoptic.seed)
      ? `Weather #${S.setup.synoptic.seed}: the run gets exactly this weather (New weather draws another).` : 'A new weather each preview until you keep one.';
    const r = !!runSyn();
    $('synShowRow').classList.toggle('hidden', !r);
    $('synShow').textContent = showingRun() ? 'Showing: the run’s weather now (switch to the preview)' : 'Showing: the preview (switch to the run’s weather)';
    $('synHourRow').classList.toggle('hidden', showingRun());
  }

  // ------------------------------------------------------------ data
  const previewKey = () => {
    const c = S.setup.synoptic || {};
    if (c.source === 'hrrr') return JSON.stringify([S.setup.map, 'hrrr', c.hrrr]);
    return JSON.stringify([S.setup.map, S.setup.start_time, c.scenario, c.peak_h, c.seed]);
  };
  const lastHour = () => (st.preview && st.preview.max_hour) ?? 24;

  // a preview: built on the server (the first time ~20 s), then every hour there in the background
  async function fetchPreview() {
    if (st.building) return;
    store();
    readSetupFromForm();
    if (S.setup.synoptic.source === 'hrrr' && !S.setup.synoptic.hrrr) {
      $('synStatus').textContent = era5On() ? 'Pick the day first (Date and Start hour).' : 'Pick the HRRR run’s date first (or Latest run).';
      return;
    }
    st.building = true; stopPlay();
    $('synPreview').disabled = true;
    const poller = setInterval(async () => {
      try {
        const s = await getJSON('/api/synoptic/status');
        if (s.preview) $('synStatus').textContent = `${hrrrOn() ? 'Loading' : 'Building'} the weather: ${s.preview[1]} (${Math.round(100 * s.preview[0])} %)…`;
      } catch (e) { /* the server is busy */ }
    }, 1000);
    $('synStatus').textContent = era5On() ? 'Downloading ERA5 (the first time ~30 s per hour)…'
      : hrrrOn() ? 'Downloading the HRRR (the first time ~20 s per hour)…' : 'Building the weather…';
    try {
      const d = await getJSON('/api/synoptic', { settings: S.setup, hour: +$('synHour').value });
      S.setup.synoptic.seed = d.seed;
      st.frames = {}; st.framesKey = previewKey();
      $('synHour').max = d.max_hour ?? 24;
      st.frames[d.hour] = d;
      st.show = 'preview';
      setPreview(d);
      watchReady();
    } catch (e) {
      $('synStatus').textContent = 'Could not build the weather: ' + e.message;
    } finally {
      clearInterval(poller);
      st.building = false;
      $('synPreview').disabled = false;
      showMode();
    }
  }

  // another hour of the same preview: instant once the server has worked it out
  async function showHour(h) {
    if (!st.preview || st.framesKey !== previewKey()) return fetchPreview();
    if (st.frames[h]) { setPreview(st.frames[h]); return; }
    try {
      const d = await getJSON('/api/synoptic', { settings: S.setup, hour: h });
      st.frames[d.hour] = d;
      if (+$('synHour').value === d.hour) setPreview(d);
    } catch (e) { $('synStatus').textContent = 'Could not get that hour: ' + e.message; }
  }

  // which hours the server has ready (it works through 0..24 after a new preview)
  function watchReady() {
    clearInterval(st.readyTimer);
    st.readyTimer = setInterval(async () => {
      try {
        const s = await getJSON('/api/synoptic/status');
        st.ready = s.ready || [];
        const n = lastHour() + 1;
        $('synReady').textContent = st.ready.length >= n ? `All ${n - 1} hours ready.` : `Forecast hours ready: ${st.ready.length} of ${n}…`;
        if (st.ready.length >= n) clearInterval(st.readyTimer);
      } catch (e) { /* busy */ }
    }, 1500);
  }

  async function fetchRun() {
    if (st.fetchingRun) return;
    st.fetchingRun = true;
    try {
      st.runData = await getJSON('/api/synoptic', { run: true });
      if (showingRun()) { st.canvasKey = ''; describe(st.runData); drawLegend(); map.redraw(); }
    } catch (e) { /* the run ended */ }
    finally { st.fetchingRun = false; st.lastRun = Date.now(); }
  }

  function setPreview(d) {
    st.preview = d;
    st.canvasKey = '';
    $('synHour').value = d.hour; $('synHourVal').textContent = `+${d.hour} h`;
    if (!showingRun()) describe(d);
    drawLegend();
    map.redraw();
  }

  function describe(d) {
    const c = d.centre || {};
    $('synStatus').innerHTML = `<b>${d.source === 'run' ? 'The run’s weather' : 'Preview'}, valid ${d.valid.replace('T', ' ')}</b><br>` +
      `${d.scenario || ''}<br>At the map centre: ${c.t2m}/${c.td2m} °C, MLCAPE ${Math.round(c.mlcape)} J/kg, MLCIN ${Math.round(c.mlcin)}, ` +
      `0–1 km SRH ${Math.round(c.srh1)} m²/s², 0–6 km shear ${Math.round(c.shear6)} m/s, STP ${c.stp}` +
      ((d.notes || []).length ? '<br>' + d.notes.join('<br>') : '');
  }

  function onPoll(run) {
    const p = run && run.synoptic_progress;
    if (p) setStatus('building', `Building the weather: ${p.message} (${Math.round(100 * p.fraction)} %)`);
    // a new synoptic run: show its weather (a preview made later switches back to the preview)
    if (run && run.synoptic && run.run_id !== st.runId) { st.runId = run.run_id; st.show = 'run'; st.lastRun = 0; showMode(); }
    if (showingRun() && st.layer !== 'none' && Date.now() - st.lastRun > 20000) fetchRun();
  }

  // ------------------------------------------------------------ loop
  function stopPlay() {
    st.playing = false; clearTimeout(st.playTimer);
    $('synPlay').textContent = '▶ Loop';
  }
  function togglePlay() {
    if (st.playing) { stopPlay(); return; }
    if (!st.preview) { fetchPreview(); return; }
    st.playing = true; $('synPlay').textContent = '❚❚ Stop';
    const tick = async () => {
      if (!st.playing) return;
      const hours = (st.ready.length ? st.ready : [0]).filter(h => h <= lastHour());
      const cur = +$('synHour').value;
      const next = hours.find(h => h > cur) ?? hours[0];
      await showHour(next);
      st.playTimer = setTimeout(tick, next === hours[hours.length - 1] ? 1600 : 650);
    };
    tick();
  }

  // ------------------------------------------------------------ point soundings
  async function soundingAt(x, y) {
    try {
      const body = showingRun() ? { run: true, x, y } : { settings: (store(), readSetupFromForm(), S.setup), hour: +$('synHour').value, x, y };
      if (!showingRun() && !on()) { alert('Turn on "Simulate the atmosphere" first (Weather tab).'); return; }
      $('synSndTitle').textContent = 'Building the sounding…';
      const d = await getJSON('/api/synoptic/sounding', body);
      st.point = [x, y];
      $('synSndTitle').textContent = d.sounding.title;
      SkewT.drawSkewT($('synSkewt'), d.sounding);
      SkewT.drawHodo($('synHodo'), d.sounding, d.diag);
      $('synDiag').innerHTML = diagHTML(d.diag);
      map.redraw();
    } catch (e) { $('synSndTitle').textContent = 'Could not build the sounding: ' + e.message; }
  }

  // ------------------------------------------------------------ drawing
  function decode(f, n) {
    const raw = atob(f.data), out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const c = raw.charCodeAt(i);
      out[i] = c ? f.lo + (c - 1) / 254 * (f.hi - f.lo) : NaN;
    }
    return out;
  }

  // the shaded field, drawn 3x finer than its 50 km grid with bilinear values (smooth, no blocks)
  function rasterCanvas(d) {
    const L = LAYERS[st.layer];
    if (!d || !L || !L.field || !d.fields[L.field]) return null;
    const key = d.valid + st.layer + d.source;
    if (st.canvasKey === key) return st.canvas;
    const { nx, ny } = d.grid, up = 3, W = nx * up, H = ny * up;
    const v = decode(d.fields[L.field], nx * ny);
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const g = cv.getContext('2d'), img = g.createImageData(W, H);
    const at = (i, j) => v[Math.min(ny - 1, Math.max(0, j)) * nx + Math.min(nx - 1, Math.max(0, i))];
    for (let J = 0; J < H; J++) {
      const fy = (J + 0.5) / up - 0.5, j0 = Math.floor(fy), ty = fy - j0;
      for (let I = 0; I < W; I++) {
        const fx = (I + 0.5) / up - 0.5, i0 = Math.floor(fx), tx = fx - i0;
        const a = at(i0, j0), b = at(i0 + 1, j0), c = at(i0, j0 + 1), e = at(i0 + 1, j0 + 1);
        let val;
        if (isFinite(a) && isFinite(b) && isFinite(c) && isFinite(e)) val = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + e * tx) * ty;
        else {                                          // an edge of the echo: the nearest value, no smearing
          val = at(Math.round(fx), Math.round(fy));
          if (!isFinite(val)) continue;
        }
        const col = colorFor(L.field, val), o = ((H - 1 - J) * W + I) * 4;   // row 0 is the south edge
        img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = col[3];
      }
    }
    g.putImageData(img, 0, 0);
    st.canvas = cv; st.canvasKey = key;
    return cv;
  }

  function drawSegments(g, m, segs, style, width, labels, dash) {
    g.strokeStyle = style; g.lineWidth = width; g.setLineDash(dash || []);
    for (const lev of segs) {
      const s = lev.s;
      g.beginPath();
      for (let k = 0; k < s.length; k += 4) {
        const [a, b] = m.toScreen(s[k], s[k + 1]), [c, e] = m.toScreen(s[k + 2], s[k + 3]);
        g.moveTo(a, b); g.lineTo(c, e);
      }
      g.stroke();
    }
    g.setLineDash([]);
    if (!labels) return;
    // labels along each line, ~260 px apart, in a dark halo
    g.font = 'bold 10px Segoe UI'; g.textAlign = 'center'; g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.7)'; g.fillStyle = style;
    for (const lev of segs) {
      const s = lev.s, placed = [];
      for (let k = 0; k < s.length; k += 4 * 9) {
        const [a, b] = m.toScreen(s[k], s[k + 1]);
        if (a < 20 || b < 20 || a > m.W - 20 || b > m.H - 20) continue;
        if (placed.some(([x, y]) => Math.hypot(x - a, y - b) < 260)) continue;
        placed.push([a, b]);
        g.strokeText(labels(lev.v), a, b + 3); g.fillText(labels(lev.v), a, b + 3);
      }
    }
    g.textAlign = 'left';
  }

  const FRONT_COLORS = { cold: '#2f6bff', warm: '#ff3b3b', stationary: '#ff3b3b', dry: '#d9913f' };

  function drawFront(g, m, fr) {
    const pts = fr.points.map(([x, y]) => m.toScreen(x, y));
    if (pts.length < 2) return;
    const col = FRONT_COLORS[fr.kind] || '#fff';
    g.lineJoin = 'round'; g.lineCap = 'round';
    g.strokeStyle = 'rgba(0,0,0,.6)'; g.lineWidth = 6;
    g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
    g.strokeStyle = fr.kind === 'stationary' ? '#2f6bff' : col; g.lineWidth = 3;
    g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
    // symbols every ~34 px along the line, on the side it moves toward (dryline: the moist side)
    const spacing = 34, size = 7.5;
    let acc = spacing / 2, n = 0;
    for (let i = 1; i < pts.length; i++) {
      const [ax, ay] = pts[i - 1], [bx, by] = pts[i];
      const L = Math.hypot(bx - ax, by - ay);
      if (L < 1e-6) continue;
      const tx = (bx - ax) / L, ty = (by - ay) / L;
      // the warm side: +1 = left of the line's direction in world (y up); on screen (y down) that is the right
      const wx = -ty * -fr.side, wy = tx * -fr.side;
      let s = acc;
      while (s < L) {
        const px = ax + tx * s, py = ay + ty * s;
        let kind = fr.kind;
        if (kind === 'stationary') kind = n % 2 ? 'warm' : 'cold';
        const toward = kind === 'warm' ? -1 : 1;            // a warm front's bumps point into the cold air
        const nx = wx * toward, ny = wy * toward;
        g.fillStyle = fr.kind === 'stationary' ? (kind === 'cold' ? '#2f6bff' : '#ff3b3b') : col;
        g.strokeStyle = g.fillStyle;
        g.beginPath();
        if (kind === 'cold') {
          g.moveTo(px - tx * size, py - ty * size); g.lineTo(px + nx * size * 1.25, py + ny * size * 1.25);
          g.lineTo(px + tx * size, py + ty * size); g.closePath(); g.fill();
        } else if (kind === 'warm') {
          const a0 = Math.atan2(ty, tx);
          g.arc(px, py, size * 0.85, a0, a0 + Math.PI, ny * tx - nx * ty > 0 ? false : true); g.closePath(); g.fill();
        } else {                                            // dryline: open scallops toward the moist air
          const a0 = Math.atan2(ty, tx);
          g.lineWidth = 2.2; g.arc(px, py, size * 0.75, a0, a0 + Math.PI, ny * tx - nx * ty > 0 ? false : true); g.stroke();
        }
        s += spacing; n++;
      }
      acc = s - L;
    }
  }

  // a wind barb at (x, y) for (u, v) m/s: staff into the wind, flags of 50 kt, barbs of 10 and 5 kt
  function drawBarb(g, x, y, u, v) {
    let kt = Math.round(Math.hypot(u, v) * 1.94384 / 5) * 5;
    if (kt < 5) { g.beginPath(); g.arc(x, y, 2.5, 0, 7); g.stroke(); return; }
    const sp = Math.hypot(u, v), dx = -u / sp, dy = v / sp;     // toward where it blows from (screen y down)
    const L = 24, ex = x + dx * L, ey = y + dy * L;
    const px = -dy, py = dx;                                   // barbs on the left of the staff (NH convention)
    g.beginPath(); g.moveTo(x, y); g.lineTo(ex, ey);
    let s = L;
    while (kt >= 50) {
      const bx = x + dx * s, by = y + dy * s, cx = x + dx * (s - 6), cy = y + dy * (s - 6);
      g.moveTo(bx, by); g.lineTo(bx + px * 11 - dx * 3, by + py * 11 - dy * 3); g.lineTo(cx, cy);
      kt -= 50; s -= 8;
    }
    while (kt >= 10) {
      const bx = x + dx * s, by = y + dy * s;
      g.moveTo(bx, by); g.lineTo(bx + px * 11 + dx * 4, by + py * 11 + dy * 4);
      kt -= 10; s -= 4.5;
    }
    if (kt >= 5) {
      if (s === L) s -= 4.5;
      const bx = x + dx * s, by = y + dy * s;
      g.moveTo(bx, by); g.lineTo(bx + px * 6 + dx * 2, by + py * 6 + dy * 2);
    }
    g.stroke();
  }

  function drawBarbs(g, m, b, ku, kv, style) {
    if (!b) return;
    g.strokeStyle = style; g.lineWidth = 1.3; g.lineCap = 'round';
    const step = m.scale < 0.25 ? 3 : m.scale < 0.6 ? 2 : 1;
    for (let k = 0; k < b.x.length; k++) {
      const [x, y] = m.toScreen(b.x[k], b.y[k]);
      if (x < 0 || y < 0 || x > m.W || y > m.H) continue;
      const gi = Math.round(b.x[k] / 200), gj = Math.round(b.y[k] / 200);      // thin the grid when zoomed out
      if (((gi % step) + step) % step || ((gj % step) + step) % step) continue;
      drawBarb(g, x, y, b[ku][k], b[kv][k]);
    }
  }

  function draw(g, m) {
    if (st.layer === 'none') return;
    if (!(on() || runSyn())) return;
    const d = current(), L = LAYERS[st.layer] || {};
    g.save();
    const cv = rasterCanvas(d);
    if (cv) {
      const { x0, y0, dx, nx, ny } = d.grid;
      const [ax, ay] = m.toScreen(x0 - dx / 2, y0 + (ny - 0.5) * dx), [bx, by] = m.toScreen(x0 + (nx - 0.5) * dx, y0 - dx / 2);
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.globalAlpha = L.alpha || (showingRun() ? 0.6 : 0.92);
      g.drawImage(cv, ax, ay, bx - ax, by - ay);
      g.globalAlpha = 1;
    }
    if (d) {
      const over = $('synZ500Over').checked && !L.z500;     // 500 hPa over whatever is shaded
      if (L.z500) {
        drawSegments(g, m, d.contours.z500, 'rgba(255,255,255,.9)', 1.5, v => (v / 10).toFixed(0));
        drawBarbs(g, m, d.barbs, 'u500', 'v500', 'rgba(255,255,255,.85)');
      } else if (L.analysis) {
        if (L.analysis === true) {
          drawSegments(g, m, d.contours.mslp, 'rgba(255,255,255,.7)', 1.1, v => v);
          if (!over) drawBarbs(g, m, d.barbs, 'usfc', 'vsfc', 'rgba(235,242,255,.75)');
        }
      }
      if (over) {
        drawSegments(g, m, d.contours.z500, 'rgba(20,20,28,.85)', 3.2);
        drawSegments(g, m, d.contours.z500, 'rgba(255,255,255,.95)', 1.6, v => (v / 10).toFixed(0));
        drawBarbs(g, m, d.barbs, 'u500', 'v500', 'rgba(255,255,255,.9)');
      }
      if (L.analysis || L.z500 === undefined) {
        // fronts and centres: the run's latest (every poll) when showing the run, else the preview's
        const fr = showingRun() ? S.run.synoptic.fronts : d.fronts, cs = showingRun() ? S.run.synoptic.centres : d.centres;
        if (L.analysis) for (const f of fr || []) drawFront(g, m, f);
        if (L.analysis === true) {
          g.textAlign = 'center';
          for (const c of cs || []) {
            const [x, y] = m.toScreen(c.x, c.y);
            if (x < -40 || y < -40 || x > m.W + 40 || y > m.H + 40) continue;
            g.font = 'bold 28px Segoe UI'; g.lineWidth = 5; g.strokeStyle = 'rgba(0,0,0,.65)';
            g.fillStyle = c.kind === 'L' ? '#ff4040' : '#4a8cff';
            g.strokeText(c.kind, x, y + 10); g.fillText(c.kind, x, y + 10);
            g.font = 'bold 12px Segoe UI'; g.lineWidth = 3; g.strokeText(c.value, x, y + 26); g.fillText(c.value, x, y + 26);
          }
          g.textAlign = 'left';
        }
      }
      drawTitle(g, m, d, L);
    }
    // a storm area, when one is set (0: anywhere), and the last point sounding
    const cfg = showingRun() ? S.run.settings.synoptic : read();
    if (cfg && cfg.area_km > 0 && cfg.area_v !== undefined) {
      const h = cfg.area_km / 2, [a, b] = m.toScreen(-h, h), [c, e] = m.toScreen(h, -h);
      g.strokeStyle = 'rgba(120,230,255,.7)'; g.setLineDash([8, 6]); g.lineWidth = 1.3;
      g.strokeRect(a, b, c - a, e - b); g.setLineDash([]);
      g.font = '11px Segoe UI'; g.fillStyle = 'rgba(160,235,255,.9)';
      g.fillText('storms form where the weather lets them, inside here', a + 4, e - 6);
    }
    if (st.point) {
      const [x, y] = m.toScreen(st.point[0], st.point[1]);
      g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 6, 0, 7); g.stroke();
    }
    g.restore();
  }

  // the map's title, like a model graphic: what it shows, valid time, forecast hour, and its colour scale
  function drawTitle(g, m, d, L) {
    const when = new Date(d.valid_ms);
    const day = when.toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });
    const h = d.kind === 'hrrr' && (current() === st.runData ? (S.run.settings.synoptic || {}).hrrr : (S.setup.synoptic || {}).hrrr);
    const fh = d.source === 'run' ? 'the run now' : `F${String((d.hour ?? 0) + (h ? h.start_fh : 0)).padStart(3, '0')}`;
    const lab = (L.label || '') + ($('synZ500Over').checked && !L.z500 ? ' + 500 hPa' : '');
    const title = h ? `${h.model === 'era5' ? 'ERA5' : 'HRRR'} ${h.cycle.slice(0, 10)} ${h.cycle.slice(11, 13)}Z → Atmos-WRF · ${lab}` : `Atmos-WRF synoptic · ${lab}`;
    const sub = `Valid ${day} ${d.valid.slice(0, 10)} ${d.valid.slice(11, 16)}Z · ${fh}`;
    g.font = 'bold 13px Segoe UI';
    const w = Math.max(g.measureText(title).width, 200) + 20;
    const x = 12, y = m.H - (L.field ? 78 : 56) - 46;          // above the replay bar
    g.fillStyle = 'rgba(8,12,18,.78)'; g.fillRect(x, y, Math.max(w, 260), L.field ? 66 : 44);
    g.fillStyle = '#e8eef7'; g.fillText(title, x + 10, y + 18);
    g.font = '12px Segoe UI'; g.fillStyle = '#aab7c8'; g.fillText(sub, x + 10, y + 36);
    if (L.field && d.fields[L.field]) {
      const f = d.fields[L.field], bw = Math.max(w, 260) - 20;
      for (let i = 0; i < bw; i++) {
        const c = colorFor(L.field, f.lo + (f.hi - f.lo) * i / (bw - 1));
        g.fillStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${Math.max(c[3], 60) / 255})`;
        g.fillRect(x + 10 + i, y + 44, 1, 8);
      }
      g.font = '10px Segoe UI'; g.fillStyle = '#aab7c8';
      g.fillText(`${f.lo}`, x + 10, y + 63);
      g.textAlign = 'right'; g.fillText(`${f.hi} ${L.unit || ''}`, x + 10 + bw, y + 63); g.textAlign = 'left';
    }
  }

  function drawLegend() {
    const d = current(), L = LAYERS[st.layer], cv = $('synLegendBar');
    $('synLegend').classList.toggle('hidden', !(L && L.field && d));
    if (!(L && L.field && d && d.fields[L.field])) return;
    const f = d.fields[L.field];
    const g = cv.getContext('2d');
    g.clearRect(0, 0, cv.width, cv.height);
    for (let i = 0; i < cv.width; i++) {
      const c = colorFor(L.field, f.lo + (f.hi - f.lo) * i / (cv.width - 1));
      g.fillStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${Math.max(c[3], 40) / 255})`;
      g.fillRect(i, 0, 1, cv.height);
    }
    $('synLegendText').textContent = `${L.label}: ${f.lo} … ${f.hi} ${L.unit || ''}`;
  }

  // ------------------------------------------------------------ wiring
  function init(scenarios) {
    if (scenarios) {
      $('synScenario').innerHTML = '';
      for (const [k, v] of Object.entries(scenarios)) $('synScenario').add(new Option(v.split(':')[0], k));
    }
    for (const id of ['synOn', 'synScenario', 'synPeak', 'synArea', 'synRate', 'synMax', 'synSpacing']) {
      $(id).addEventListener('change', () => {
        if (['synScenario', 'synPeak'].includes(id) && S.setup && S.setup.synoptic) S.setup.synoptic.seed = null;
        if (id === 'synOn') $('synSource').value = 'sim';          // the Weather tab's switch: the simulated weather
        store(); showMode(); map.redraw();
      });
    }
    $('synScenario').addEventListener('change', () => {
      const d = (S.synScenarios || {})[$('synScenario').value];
      $('synScenarioHint').textContent = d || '';
    });
    $('synPreview').onclick = fetchPreview;
    $('synNew').onclick = () => { store(); S.setup.synoptic.seed = null; showMode(); fetchPreview(); };
    $('synHour').oninput = () => { $('synHourVal').textContent = `+${$('synHour').value} h`; stopPlay(); showHour(+$('synHour').value); };
    $('synPlay').onclick = togglePlay;
    $('synShow').onclick = () => {
      st.show = showingRun() ? 'preview' : 'run';
      if (st.show === 'run') { st.lastRun = 0; fetchRun(); } else if (st.preview) describe(st.preview);
      st.canvasKey = ''; showMode(); drawLegend(); map.redraw();
    };
    $('synLayer').onchange = () => { st.layer = $('synLayer').value; st.canvasKey = ''; drawLegend(); map.redraw(); };
    $('synZ500Over').onchange = () => map.redraw();
    // the HRRR run
    for (let hh = 0; hh < 24; hh++) $('hrrrCycle').add(new Option(`${pad2(hh)}Z${hh % 6 ? '' : ' (48 h)'}`, String(hh)));
    $('hrrrCycle').value = '12';
    // the Atmosphere tab's Storm environment: one sounding, the HRRR's soundings, or the simulated weather
    $('envSource').onchange = async () => {
      const v = $('envSource').value;
      $('synOn').checked = v !== 'sounding';
      $('synSource').value = v === 'hrrr' || v === 'era5' ? 'hrrr' : 'sim';
      const was = $('hrrrModel').value;
      $('hrrrModel').value = v === 'era5' ? 'era5' : 'hrrr';
      // switching between the two: the other's date means nothing here
      if ((v === 'era5' || v === 'hrrr') && was !== $('hrrrModel').value) { $('hrrrDate').value = ''; $('hrrrNote').textContent = ''; }
      if (v === 'era5' && !$('hrrrDate').value) {                 // a classic outbreak to start from
        $('hrrrDate').value = '2011-04-27'; $('hrrrCycle').value = '12'; $('hrrrHours').value = 18;
      }
      store(); showMode(); map.redraw();
      if (v === 'hrrr' && !$('hrrrDate').value) await findHrrr(true);
    };
    $('hrrrMode').onchange = () => { store(); showMode(); };
    $('hrrrFind').onclick = () => findHrrr(false);
    $('hrrrValid').addEventListener('change', () => findHrrr(false));
    for (const id of ['hrrrDate', 'hrrrCycle', 'hrrrStart', 'hrrrHours']) {
      $(id).addEventListener('change', () => { store(); $('hrrrNote').textContent = ''; });
    }
    $('hrrrLatest').onclick = latestHrrr;
    $('hrrrCheck').onclick = checkHrrr;
    $('synSndTool').onclick = () => setTool('synsnd');
    st.layer = $('synLayer').value;
  }

  // the run for a start time (past, now or up to ~2 days ahead: the newest run reaching it); auto (switching
  // to the HRRR with nothing picked yet): now, unless something was picked meanwhile
  async function findHrrr(auto) {
    if (!$('hrrrValid').value) {
      const now = new Date(); now.setUTCMinutes(0, 0, 0);
      $('hrrrValid').value = now.toISOString().slice(0, 16);
    }
    const v = $('hrrrValid').value.slice(0, 13) + ':00:00Z';
    $('hrrrNote').textContent = 'Finding the HRRR run for that time…';
    try {
      const d = await getJSON('/api/hrrr/for?valid=' + encodeURIComponent(v));
      if (auto === true && $('hrrrDate').value) { $('hrrrNote').textContent = ''; return; }
      if (d.error) { $('hrrrNote').textContent = d.error; return; }
      $('hrrrDate').value = d.cycle.slice(0, 10);
      $('hrrrCycle').value = String(+d.cycle.slice(11, 13));
      $('hrrrStart').value = d.start_fh;
      $('hrrrHours').value = Math.max(1, Math.min(+$('hrrrHours').value || 12, d.max_fh - d.start_fh));
      store();
      $('hrrrNote').textContent = `Using the ${d.cycle.slice(0, 10)} ${d.cycle.slice(11, 13)}Z HRRR run from forecast hour ${d.start_fh}` +
        (d.start_fh ? ` (a forecast: ${v.slice(0, 10)} ${v.slice(11, 13)}Z is ${d.start_fh} h after it ran)` : ' (its analysis)') +
        `; ${d.max_fh - d.start_fh} more hours available.`;
    } catch (e) { $('hrrrNote').textContent = 'Could not reach NOAA’s archive: ' + e.message; }
  }

  // auto (switching to the HRRR with no date yet): fills the newest run in unless a date was picked meanwhile
  async function latestHrrr(auto) {
    $('hrrrNote').textContent = 'Looking for the newest HRRR run…';
    try {
      const d = await getJSON('/api/hrrr/latest');
      if (auto === true && $('hrrrDate').value) { $('hrrrNote').textContent = ''; return; }
      $('hrrrDate').value = d.cycle.slice(0, 10);
      $('hrrrCycle').value = String(+d.cycle.slice(11, 13));
      store();
      $('hrrrNote').textContent = `Newest: ${d.cycle.slice(0, 10)} ${d.cycle.slice(11, 13)}Z (out to ${d.max_fh} h).`;
    } catch (e) { $('hrrrNote').textContent = 'Could not reach NOAA’s archive: ' + e.message; }
  }

  async function checkHrrr() {
    const h = hrrrForm();
    if (!h) { $('hrrrNote').textContent = 'Pick a date first.'; return; }
    $('hrrrNote').textContent = 'Checking…';
    try {
      const d = await getJSON('/api/hrrr/check?cycle=' + encodeURIComponent(h.cycle));
      $('hrrrNote').textContent = (d.available ? `${h.cycle.slice(0, 10)} ${h.cycle.slice(11, 13)}Z is there (out to ${d.max_fh} h).`
        : `${h.cycle.slice(0, 10)} ${h.cycle.slice(11, 13)}Z is not in the archive (too new, or before mid-2014?).`) +
        (d.decoder ? '' : ' The HRRR decoder is not installed yet: run Setup.cmd once more.');
    } catch (e) { $('hrrrNote').textContent = 'Could not reach NOAA’s archive: ' + e.message; }
  }

  return { init, read, apply, store, draw, onPoll, soundingAt, showMode };
})();

window.Syn = Syn;
