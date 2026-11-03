'use strict';
// The sounding editor: the profile level by level in a table, and by dragging its points on a large
// skew-T and hodograph, with the parameters recomputed as it changes.  It edits the sounding itself
// (its file, pasted text or analytic profile); the Setup panel's changes (mix, moisten, cap, winds,
// jet) still apply on top of it, as they do to any sounding, unless they are baked in.  Heights follow
// the temperatures hydrostatically.  "Use this sounding" makes it the setup's sounding (kept as text in
// the run's settings); "Save to library" also keeps it as a file.

const SndEdit = (() => {
  const KT = 0.514444, RD = 287.0, G = 9.81, EPS = 287.0 / 461.6;
  // how far a Shift-drag reaches on the skew-T / the hodograph: levels 50 hPa (500 m) away move half as much
  const BRUSH_HPA = 100, BRUSH_M = 1000;
  const HODO_TOP = 12000;                       // m AGL of winds on the hodograph
  const STEP = { p: 1, z: 10, T: 0.5, Td: 0.5, rh: 5, dir: 5, kt: 1 };   // Shift+Up/Down in the table
  let L = [], orig = [];                        // levels, surface first: {p hPa, z m MSL, T C, Td C, u, v m/s}
  let title = '', origTitle = '';
  let undo = [], redo = [];
  let sel = -1, hover = null, drag = null;
  let preview = null, refDiag = null, bake = false, dirty = false, isOpen = false;
  let seq = 0, timer = null;
  let skewT = null, hodoT = null;               // the plots' transforms, for picking points

  const el = id => document.getElementById(id);
  const clone = a => a.map(l => ({ ...l }));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const es = t => 611.2 * Math.exp(17.67 * t / (t + 243.5));          // Pa over water (Bolton), t in C
  const qv = (p, td) => { const e = Math.min(es(td), 50 * p); return EPS * e / (100 * p - e); };
  const tv = l => { const q = qv(l.p, l.Td); return (l.T + 273.15) * (1 + q / EPS) / (1 + q); };
  const thick = (a, b) => RD / G * 0.5 * (tv(a) + tv(b)) * Math.log(a.p / b.p);
  const dirOf = l => (Math.atan2(-l.u, -l.v) * 180 / Math.PI + 360) % 360;
  const ktOf = l => Math.hypot(l.u, l.v) / KT;
  const rhOf = l => 100 * es(l.Td) / es(l.T);
  const fmt = (v, n) => (isFinite(v) ? (+v).toFixed(n) : '');
  const agl = l => l.z - L[0].z;

  // ---------------------------------------------------------------- open / close
  async function open() {
    if (!S.snd) { alert('Load a sounding first.'); return; }
    let base, first;
    try {
      // the profile itself, without the Setup panel's changes: they apply on top of it
      base = await getJSON('/api/sounding/load', { spec: { ...baseSpec(S.setup.sounding), ice_cap: false } });
      L = base.sounding.p.map((p, i) => ({ p, z: base.sounding.z[i], T: base.sounding.T[i], Td: base.sounding.Td[i],
                                           u: base.sounding.u[i], v: base.sounding.v[i] }));
      title = origTitle = base.sounding.title;
      bake = false;
      // the parameters as they start: changes are shown against these (the editor's own text of the
      // profile, so rounding does not show up as a change)
      first = await getJSON('/api/sounding/load', { spec: previewSpec() });
    } catch (e) { alert('Sounding: ' + e.message); return; }
    orig = clone(L);
    preview = first; refDiag = first.diag;
    undo = []; redo = []; sel = -1; hover = null; drag = null; dirty = false;
    el('seTitle').value = title;
    el('seError').textContent = '';
    el('sndEditor').classList.remove('hidden');
    isOpen = true;
    renderTable(); draw(); showDiag(); status();
    el('seClose').focus();
  }

  function close(force) {
    if (!isOpen) return;
    if (!force && dirty && !confirm('Close the sounding editor and lose your changes?')) return;
    isOpen = false; drag = null;
    clearTimeout(timer);
    el('sndEditor').classList.add('hidden');
  }

  // ---------------------------------------------------------------- changes
  // Heights follow the temperatures: every layer whose ends changed gets the hypsometric change of its
  // thickness, the rest keep theirs exactly (so an untouched profile keeps its own heights).
  function restack(O) {
    for (let j = 1; j < L.length; j++) {
      L[j].z = L[j - 1].z + (O[j].z - O[j - 1].z) + thick(L[j - 1], L[j]) - thick(O[j - 1], O[j]);
    }
  }

  const snapshot = () => ({ L: clone(L), title, bake });

  function remember(snap) {
    undo.push(snap);
    if (undo.length > 300) undo.shift();
    redo = [];
    dirty = true;
  }

  // one value of level k; false (and nothing changed) if it is not allowed
  function setField(k, f, v) {
    if (!isFinite(v)) return false;
    const l = L[k];
    if (f === 'p') {
      const above = k + 1 < L.length ? L[k + 1].p : 0, below = k > 0 ? L[k - 1].p : 1100;
      if (v <= above + 0.05 || v >= below - 0.05 || v < 10) return false;      // between its neighbours
      l.p = v;
    } else if (f === 'z') {
      if (k !== 0 || v < -500 || v > 6000) return false;    // the ground's height: the profile moves with it
      l.z = v;
    } else if (f === 'T') {
      l.T = clamp(v, -120, 60);
      if (l.Td > l.T) l.Td = l.T;
    } else if (f === 'Td') {
      l.Td = clamp(v, -150, l.T);
    } else if (f === 'rh') {
      const e = clamp(v, 1, 100) / 100 * es(l.T), x = Math.log(e / 611.2);
      l.Td = Math.min(l.T, 243.5 * x / (17.67 - x));
    } else if (f === 'dir' || f === 'kt') {
      const d = (f === 'dir' ? ((v % 360) + 360) % 360 : dirOf(l)) * Math.PI / 180;
      const s = (f === 'kt' ? clamp(v, 0, 300) : ktOf(l)) * KT;
      l.u = -s * Math.sin(d); l.v = -s * Math.cos(d);
    } else return false;
    return true;
  }

  // a value change: undo point, heights, table from row k0 up, plots, parameters
  function change(k0, fn) {
    const O = clone(L), snap = { L: O, title, bake };
    if (!fn()) return false;
    remember(snap);
    restack(O);
    refresh(k0); draw(); status(); schedule();
    return true;
  }

  // levels added or removed
  function structural(fn) {
    const snap = snapshot();
    if (!fn()) return false;
    remember(snap);
    renderTable(); draw(); status(); schedule();
    return true;
  }

  function insertAt(k, p) {          // a level at pressure p between levels k and k + 1 (log-p interpolation)
    const a = L[k], b = L[k + 1];
    const f = Math.log(a.p / p) / Math.log(a.p / b.p);
    const n = { p, T: a.T + f * (b.T - a.T), Td: a.Td + f * (b.Td - a.Td), u: a.u + f * (b.u - a.u), v: a.v + f * (b.v - a.v) };
    n.z = a.z + thick(a, n);
    L.splice(k + 1, 0, n);
    sel = k + 1;
    return true;
  }

  function addAt(p) {
    if (!isFinite(p)) return;
    const k = L.findIndex((l, i) => i + 1 < L.length && l.p > p && L[i + 1].p < p);
    if (k < 0) { say(p >= L[0].p ? 'That is below the ground: change the surface pressure instead.' : 'That is above the top level.'); return; }
    if (Math.min(L[k].p - p, p - L[k + 1].p) < 0.1) { say('There is a level at that pressure already.'); return; }
    structural(() => insertAt(k, p));
    selectRow(sel, true);
  }

  function removeAt(k) {
    if (k <= 0) return;
    if (L.length <= 5) { say('A sounding needs at least 5 levels.'); return; }
    structural(() => { L.splice(k, 1); sel = Math.min(k, L.length - 1); return true; });
  }

  function thin() {
    const kept = L.filter((l, i) => i === 0 || i === L.length - 1 || i % 2 === 0);
    if (kept.length < 5 || kept.length === L.length) { say('Too few levels to thin out.'); return; }
    structural(() => { L = kept; sel = -1; return true; });
  }

  function bakeIn() {
    if (!S.snd || !Object.keys(setupChanges()).length) { say('The Atmosphere tab has no adjustments to bake in.'); return; }
    structural(() => {
      const s = S.snd.sounding;
      L = s.p.map((p, i) => ({ p, z: s.z[i], T: s.T[i], Td: s.Td[i], u: s.u[i], v: s.v[i] }));
      title = s.title; el('seTitle').value = title;
      bake = true; sel = -1;
      return true;
    });
  }

  function restore(s) { L = s.L; title = s.title; bake = s.bake; el('seTitle').value = title; renderTable(); draw(); status(); schedule(); }
  function undoIt() { if (!undo.length) return; redo.push(snapshot()); restore(undo.pop()); dirty = true; }
  function redoIt() { if (!redo.length) return; undo.push(snapshot()); restore(redo.pop()); dirty = true; }
  function reset() {
    if (!dirty) return;
    remember(snapshot());
    restore({ L: clone(orig), title: origTitle, bake: false });
    dirty = false;
  }

  // ---------------------------------------------------------------- the Setup panel and the server
  // the Atmosphere tab's changes that apply on top of the edited profile (none of them once baked in, but
  // the ice-saturation limit, which does nothing to a profile that has it already)
  const setupChanges = () => { const m = soundingMods(); delete m.ice_cap; return m; };
  const previewSpec = () => ({ text: raw(), title: name(),
                               ...(bake ? (el('iceCap').checked ? {} : { ice_cap: false }) : soundingMods()) });

  const name = () => title.trim() || 'Edited sounding';

  const raw = () => rawText(L, name());

  // a profile (levels {p, z, T, Td, u, v}, surface first) or a /api/sounding/load sounding ({p: [], z: [], ...})
  // as SHARPpy text, the way the editor hands it to the server (Skew-T Lab gets it the same way)
  function sharppyText(snd, ttl) {
    const lv = Array.isArray(snd) ? snd : snd.p.map((p, i) => ({ p, z: snd.z[i], T: snd.T[i], Td: snd.Td[i], u: snd.u[i], v: snd.v[i] }));
    return rawText(lv, (ttl || (Array.isArray(snd) ? '' : snd.title) || '').trim() || 'Sounding');
  }

  function rawText(L, name) {        // SHARPpy %RAW%: pressure, height, temperature, dewpoint, wind from, knots
    const rows = L.map(l => [l.p.toFixed(2), l.z.toFixed(1), l.T.toFixed(2), l.Td.toFixed(2), dirOf(l).toFixed(2),
                             ktOf(l).toFixed(2)].join(', '));
    return `%TITLE%\n ${name}\n\n   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD\n` +
      `-------------------------------------------------------------------\n%RAW%\n${rows.join('\n')}\n%END%\n`;
  }

  function schedule() { clearTimeout(timer); timer = setTimeout(previewNow, 250); }

  async function previewNow() {
    const my = ++seq;
    try {
      const d = await getJSON('/api/sounding/load', { spec: previewSpec() });
      if (my !== seq || !isOpen) return;
      preview = d;
      el('seError').textContent = '';
      showDiag(); draw();
    } catch (e) { if (my === seq) el('seError').textContent = e.message; }
  }

  // an edited sounding that kept its old title says so (in Setup, the run's settings and its log)
  function markEdited() {
    if (dirty && title === origTitle && !/\(edited\)$/.test(title)) { title = `${title} (edited)`; el('seTitle').value = title; }
  }

  async function use() {
    markEdited();
    if (bake) clearSoundingMods();
    if (await loadSounding({ text: raw(), title: name() })) { dirty = false; close(true); }
  }

  async function save() {
    markEdited();
    const def = name().replace(/[<>:"/\\|?*]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80) || 'sounding';
    const typed = prompt('Save the edited sounding in the library (data\\imported\\soundings) as:', def);
    if (!typed || !typed.trim()) return;
    const file = typed.trim().replace(/\.txt$/i, '') + '.txt';
    const there = S.soundingFiles.some(f => /[\\/]imported[\\/]soundings[\\/]/i.test(f.path) &&
                                            f.path.split(/[\\/]/).pop().toLowerCase() === file.toLowerCase());
    if (there && !confirm(`${file} is in the library already. Replace it?`)) return;
    try {
      const got = await getJSON('/api/library/import', { kind: 'sounding', name: file, text: raw() });
      await refreshLibraryLists();
      if (bake) clearSoundingMods();
      if (await loadSounding({ file: got.path })) {
        $('sndList').value = got.path;
        dirty = false; close(true);
      }
    } catch (e) { alert('Save: ' + e.message); }
  }

  // ---------------------------------------------------------------- table
  const input = (f, v) => `<td><input data-f="${f}" value="${v}" inputmode="decimal" autocomplete="off" spellcheck="false"></td>`;
  const shown = (l, f) => f === 'p' ? fmt(l.p, 1) : f === 'z' ? fmt(l.z, 0) : f === 'T' ? fmt(l.T, 1) : f === 'Td' ? fmt(l.Td, 1)
    : f === 'rh' ? fmt(rhOf(l), 0) : f === 'dir' ? fmt(dirOf(l), 0) : fmt(ktOf(l), 0);
  const value = (l, f) => f === 'rh' ? rhOf(l) : f === 'dir' ? dirOf(l) : f === 'kt' ? ktOf(l) : l[f];

  function rowHTML(l, k) {
    const top = k === L.length - 1;
    return `<tr data-k="${k}"${k === sel ? ' class="sel"' : ''}><td class="lv">${k === 0 ? 'Sfc' : k}</td>` +
      input('p', shown(l, 'p')) +
      (k === 0 ? input('z', shown(l, 'z')) : `<td class="ro" data-c="z">${fmt(l.z, 0)}</td>`) +
      `<td class="ro" data-c="agl">${fmt(agl(l), 0)}</td>` +
      input('T', shown(l, 'T')) + input('Td', shown(l, 'Td')) + input('rh', shown(l, 'rh')) +
      input('dir', shown(l, 'dir')) + input('kt', shown(l, 'kt')) +
      `<td class="acts">${top ? '' : '<button data-act="ins" title="Add a level between this one and the next">+</button>'}` +
      `${k ? '<button data-act="del" title="Remove this level">×</button>' : ''}</td></tr>`;
  }

  function renderTable() { el('seRows').innerHTML = L.map(rowHTML).join(''); }

  // the values of rows k0 and up (heights change above an edit), leaving the cell being typed in alone
  function refresh(k0 = 0) {
    const rows = el('seRows').rows, active = document.activeElement;
    for (let k = Math.max(0, k0); k < L.length; k++) {
      const tr = rows[k];
      if (!tr) continue;
      const l = L[k];
      for (const inp of tr.getElementsByTagName('input')) if (inp !== active) inp.value = shown(l, inp.dataset.f);
      const z = tr.querySelector('[data-c="z"]');
      if (z) z.textContent = fmt(l.z, 0);
      tr.querySelector('[data-c="agl"]').textContent = fmt(agl(l), 0);
    }
  }

  function selectRow(k, scroll) {
    sel = k;
    for (const tr of el('seRows').rows) tr.classList.toggle('sel', +tr.dataset.k === k);
    const tr = el('seRows').rows[k];
    if (scroll && tr) tr.scrollIntoView({ block: 'nearest' });
    draw();
  }

  function bad(inp, why) {
    inp.classList.add('bad');
    setTimeout(() => inp.classList.remove('bad'), 700);
    const k = +inp.closest('tr').dataset.k;
    if (L[k]) inp.value = shown(L[k], inp.dataset.f);
    if (why) say(why);
  }

  function onCell(e) {
    const inp = e.target.closest('input[data-f]');
    if (!inp) return;
    const k = +inp.closest('tr').dataset.k, f = inp.dataset.f;
    const v = parseFloat(inp.value.replace(',', '.'));
    if (!isFinite(v)) { bad(inp, 'Not a number.'); return; }
    if (!change(k, () => setField(k, f, v))) {
      bad(inp, f === 'p' ? 'A level\'s pressure has to stay between the levels below and above it.' : 'Not allowed there.');
      return;
    }
    inp.value = shown(L[k], f);
  }

  function onKey(e) {
    const inp = e.target.closest('input[data-f]');
    if (!inp) return;
    const k = +inp.closest('tr').dataset.k, f = inp.dataset.f;
    if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && e.shiftKey) {        // nudge the value
      e.preventDefault();
      const v = value(L[k], f) + STEP[f] * (e.key === 'ArrowUp' ? 1 : -1);
      if (change(k, () => setField(k, f, v))) inp.value = shown(L[k], f);
      return;
    }
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'Enter') {   // like a spreadsheet
      e.preventDefault();
      const tr = el('seRows').rows[k + (e.key === 'ArrowUp' ? -1 : 1)];
      if (!tr) { inp.blur(); inp.focus(); return; }
      const next = tr.querySelector(`input[data-f="${f}"]`) || tr.querySelector('input');
      next.focus(); next.select();
    }
  }

  function onRowButton(e) {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    const k = +b.closest('tr').dataset.k;
    if (b.dataset.act === 'ins') { structural(() => insertAt(k, Math.sqrt(L[k].p * L[k + 1].p))); selectRow(sel, true); }
    else removeAt(k);
  }

  // ---------------------------------------------------------------- plots
  function draw() { if (!isOpen) return; drawSkew(); drawHodo(); }

  function canvas(id) {
    const cv = el(id), dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return null;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#0a0d12'; g.fillRect(0, 0, W, H);
    return { g, W, H };
  }

  function drawSkew() {
    const c = canvas('seSkew');
    if (!c || !L.length) return;
    const { g, W, H } = c;
    const ML = 38, MR = 10, MT = 10, MB = 22, pw = W - ML - MR, ph = H - MT - MB;
    const pTop = 100, pBot = Math.max(1050, Math.ceil((L[0].p + 15) / 10) * 10), t0 = -40, t1 = 50;
    const skew = ph / pw;                         // isotherms at 45 degrees
    const yOf = p => MT + ph * Math.log(p / pTop) / Math.log(pBot / pTop);
    const xOf = (T, p) => ML + pw * ((T - t0) / (t1 - t0) + skew * (MT + ph - yOf(p)) / ph);
    const tOf = (x, y) => t0 + ((x - ML) / pw - skew * (MT + ph - y) / ph) * (t1 - t0);
    skewT = { xOf, yOf, tOf, pTop };
    g.save();
    g.beginPath(); g.rect(ML, MT, pw, ph); g.clip();
    g.lineWidth = 1;
    g.strokeStyle = '#1b2a22';                    // dry adiabats
    for (let th = 250; th <= 470; th += 10) {
      g.beginPath();
      for (let p = pBot; p >= pTop - 1; p -= 10) {
        const x = xOf(th * Math.pow(p / 1000, 0.2857) - 273.15, p), y = yOf(p);
        p === pBot ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      g.stroke();
    }
    g.strokeStyle = '#26303d';                    // isotherms
    for (let T = -130; T <= 50; T += 10) { g.beginPath(); g.moveTo(xOf(T, pBot), yOf(pBot)); g.lineTo(xOf(T, pTop), yOf(pTop)); g.stroke(); }
    g.strokeStyle = '#4a5a70'; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(xOf(0, pBot), yOf(pBot)); g.lineTo(xOf(0, pTop), yOf(pTop)); g.stroke();
    g.setLineDash([]);
    g.strokeStyle = '#26303d';                    // isobars
    for (const p of [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150]) { g.beginPath(); g.moveTo(ML, yOf(p)); g.lineTo(ML + pw, yOf(p)); g.stroke(); }
    const path = (ps, ts, color, width, dash) => {
      g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []);
      g.beginPath();
      let on = false;
      for (let i = 0; i < ps.length; i++) {
        if (ps[i] < pTop || !isFinite(ts[i])) continue;
        const x = xOf(ts[i], ps[i]), y = yOf(ps[i]);
        on ? g.lineTo(x, y) : g.moveTo(x, y); on = true;
      }
      g.stroke(); g.setLineDash([]);
    };
    if (dirty && orig.length) {                   // as it was when the editor opened
      path(orig.map(l => l.p), orig.map(l => l.Td), 'rgba(60,207,122,.28)', 1.2);
      path(orig.map(l => l.p), orig.map(l => l.T), 'rgba(239,91,91,.28)', 1.2);
    }
    const mods = !bake && Object.keys(setupChanges()).length && preview;
    if (mods) {                                   // what the run gets: with the Setup changes on top
      path(preview.sounding.p, preview.sounding.Td, 'rgba(120,230,160,.75)', 1.2, [5, 4]);
      path(preview.sounding.p, preview.sounding.T, 'rgba(255,140,140,.75)', 1.2, [5, 4]);
    }
    if (preview && preview.parcel) path(preview.parcel.p, preview.parcel.T, 'rgba(230,230,240,.8)', 1.3, [2, 3]);
    path(L.map(l => l.p), L.map(l => l.Td), '#3ccf7a', 2);
    path(L.map(l => l.p), L.map(l => l.T), '#ef5b5b', 2);
    for (let k = 0; k < L.length; k++) {          // the points to drag
      const l = L[k];
      if (l.p < pTop) continue;
      for (const [f, col] of [['Td', '#3ccf7a'], ['T', '#ef5b5b']]) {
        const x = xOf(l[f], l.p), y = yOf(l.p), on = hover && hover.k === k && hover.f === f;
        g.fillStyle = col; g.beginPath(); g.arc(x, y, on ? 5 : 2.6, 0, 7); g.fill();
        if (k === sel) { g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 5.5, 0, 7); g.stroke(); }
      }
    }
    g.restore();
    g.fillStyle = '#8a96a8'; g.font = '10px Segoe UI'; g.textAlign = 'right';
    for (const p of [1000, 850, 700, 500, 300, 200]) if (p < pBot) g.fillText(p, ML - 4, yOf(p) + 3);
    g.textAlign = 'center';
    for (let T = -40; T <= 50; T += 10) g.fillText(T, xOf(T, pBot), H - 7);
    g.textAlign = 'left';
    g.fillText('hPa · °C', 4, H - 7);
    if (mods || (preview && preview.parcel)) {
      g.fillStyle = 'rgba(230,230,240,.85)'; g.textAlign = 'right';
      g.fillText((preview && preview.parcel ? 'dotted: mixed-layer parcel' : '') + (mods ? '   dashed: with the Setup changes' : ''), W - MR - 4, MT + 12);
      g.textAlign = 'left';
    }
  }

  function drawHodo() {
    const c = canvas('seHodo');
    if (!c || !L.length) return;
    const { g, W, H } = c;
    let vmax = 20;
    for (const l of L) if (agl(l) <= HODO_TOP) vmax = Math.max(vmax, Math.hypot(l.u, l.v));
    vmax = drag && drag.vmax ? drag.vmax : Math.ceil(vmax / 10) * 10;
    const cx = W / 2, cy = H / 2, s = (Math.min(W, H) / 2 - 12) / vmax;
    hodoT = { cx, cy, s, vmax };
    g.strokeStyle = '#26303d'; g.lineWidth = 1;
    for (let r = 10; r <= vmax; r += 10) { g.beginPath(); g.arc(cx, cy, r * s, 0, 7); g.stroke(); }
    g.beginPath(); g.moveTo(0, cy); g.lineTo(W, cy); g.moveTo(cx, 0); g.lineTo(cx, H); g.stroke();
    g.fillStyle = '#5d6878'; g.font = '9px Segoe UI';
    for (let r = 10; r <= vmax; r += 10) g.fillText(r, cx + r * s + 2, cy - 2);
    const band = h => h <= 1000 ? '#c060ff' : h <= 3000 ? '#ef5b5b' : h <= 6000 ? '#3ccf7a' : h <= 9000 ? '#f2d24b' : '#4ad8e8';
    const trace = (us, vs, zs, color, dash) => {
      g.strokeStyle = color; g.lineWidth = 1.3; g.setLineDash(dash || []);
      g.beginPath();
      for (let i = 0; i < us.length && zs[i] - zs[0] <= HODO_TOP; i++) {
        const x = cx + us[i] * s, y = cy - vs[i] * s;
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke(); g.setLineDash([]);
    };
    if (dirty && orig.length) trace(orig.map(l => l.u), orig.map(l => l.v), orig.map(l => l.z), 'rgba(200,200,215,.3)');
    if (!bake && Object.keys(setupChanges()).length && preview) {     // the winds the run gets
      const w = preview.sounding;
      trace(w.u, w.v, w.z, 'rgba(230,230,240,.8)', [5, 4]);
    }
    g.lineWidth = 2;
    for (let k = 1; k < L.length && agl(L[k]) <= HODO_TOP; k++) {
      g.strokeStyle = band(agl(L[k]));
      g.beginPath(); g.moveTo(cx + L[k - 1].u * s, cy - L[k - 1].v * s); g.lineTo(cx + L[k].u * s, cy - L[k].v * s); g.stroke();
    }
    for (let k = 0; k < L.length && agl(L[k]) <= HODO_TOP; k++) {
      const x = cx + L[k].u * s, y = cy - L[k].v * s, on = hover && hover.k === k && hover.f === 'wind';
      g.fillStyle = band(agl(L[k])); g.beginPath(); g.arc(x, y, on ? 5 : 2.4, 0, 7); g.fill();
      if (k === sel) { g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 5.5, 0, 7); g.stroke(); g.lineWidth = 2; }
    }
    const d = preview && preview.diag;
    if (d && d.rm) {
      for (const [m, col, lab] of [[d.rm, '#ff7070', 'RM'], [d.lm, '#70a0ff', 'LM']]) {
        g.fillStyle = col; g.beginPath(); g.arc(cx + m[0] * s, cy - m[1] * s, 3.5, 0, 7); g.fill();
        g.fillText(lab, cx + m[0] * s + 5, cy - m[1] * s + 3);
      }
    }
    g.fillStyle = '#8a96a8'; g.font = '10px Segoe UI'; g.fillText('m/s · 0–12 km', 5, 13);
    if (!bake && Object.keys(setupChanges()).length) g.fillText('dashed: with the Setup changes', 5, H - 6);
  }

  function showDiag() { if (preview) el('seDiag').innerHTML = diagHTML(preview.diag, refDiag); }

  // ---------------------------------------------------------------- dragging points
  const at = (e, cv) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

  function pickSkew(x, y) {
    if (!skewT) return null;
    let best = null, bd = 9;
    L.forEach((l, k) => {
      if (l.p < skewT.pTop) return;
      for (const f of ['T', 'Td']) {
        const d = Math.hypot(skewT.xOf(l[f], l.p) - x, skewT.yOf(l.p) - y);
        if (d < bd) { bd = d; best = { k, f }; }
      }
    });
    return best;
  }

  function pickHodo(x, y) {
    if (!hodoT) return null;
    let best = null, bd = 9;
    for (let k = 0; k < L.length && agl(L[k]) <= HODO_TOP; k++) {
      const d = Math.hypot(hodoT.cx + L[k].u * hodoT.s - x, hodoT.cy - L[k].v * hodoT.s - y);
      if (d < bd) { bd = d; best = { k, f: 'wind' }; }
    }
    return best;
  }

  // the levels a drag moves and how much: the one grabbed, or with Shift its neighbours too, tapering off
  function weights(k, f, brush) {
    if (!brush) return [[k, 1]];
    const out = [];
    L.forEach((l, j) => {
      const d = f === 'wind' ? Math.abs(agl(l) - agl(L[k])) / BRUSH_M : Math.abs(l.p - L[k].p) / BRUSH_HPA;
      if (d < 1) out.push([j, 0.5 * (1 + Math.cos(Math.PI * d))]);
    });
    return out;
  }

  function startDrag(e, plot) {
    const cv = el(plot === 'skew' ? 'seSkew' : 'seHodo');
    const [x, y] = at(e, cv);
    const hit = plot === 'skew' ? pickSkew(x, y) : pickHodo(x, y);
    if (!hit || e.button !== 0) return;
    e.preventDefault();
    const l = L[hit.k];                     // where the point is: it keeps its offset from the cursor
    const [px, py] = plot === 'skew' ? [skewT.xOf(l[hit.f], l.p), skewT.yOf(l.p)]
      : [hodoT.cx + l.u * hodoT.s, hodoT.cy - l.v * hodoT.s];
    drag = { plot, ...hit, O: clone(L), snap: snapshot(), w: weights(hit.k, hit.f, e.shiftKey), moved: false,
             vmax: plot === 'hodo' ? hodoT.vmax : 0, ox: px - x, oy: py - y };
    selectRow(hit.k, true);
  }

  function moveDrag(e) {
    if (!drag) return;
    const cv = el(drag.plot === 'skew' ? 'seSkew' : 'seHodo');
    let [x, y] = at(e, cv);
    x += drag.ox; y += drag.oy;
    const { k, f, O } = drag;
    L = clone(O);
    if (drag.plot === 'skew') {
      const d = skewT.tOf(x, skewT.yOf(O[k].p)) - O[k][f];
      for (const [j, w] of drag.w) {
        const l = L[j];
        if (f === 'T') { l.T = clamp(O[j].T + w * d, -120, 60); if (l.Td > l.T) l.Td = l.T; }
        else l.Td = clamp(O[j].Td + w * d, -150, l.T);
      }
      restack(O);
    } else {
      const du = (x - hodoT.cx) / hodoT.s - O[k].u, dv = (hodoT.cy - y) / hodoT.s - O[k].v;
      for (const [j, w] of drag.w) { L[j].u = O[j].u + w * du; L[j].v = O[j].v + w * dv; }
    }
    drag.moved = true;
    refresh(Math.min(...drag.w.map(([j]) => j)));
    draw(); status(); schedule();
  }

  function endDrag() {
    if (!drag) return;
    if (drag.moved) remember(drag.snap);
    drag = null;
    draw(); status();
  }

  function hoverAt(e, plot) {
    if (drag) return;
    const cv = el(plot === 'skew' ? 'seSkew' : 'seHodo');
    const [x, y] = at(e, cv);
    const hit = plot === 'skew' ? pickSkew(x, y) : pickHodo(x, y);
    const was = hover;
    hover = hit;
    cv.style.cursor = hit ? 'grab' : 'crosshair';
    if ((was && was.k) !== (hit && hit.k) || (was && was.f) !== (hit && hit.f)) { draw(); status(); }
  }

  // ---------------------------------------------------------------- status line
  let note = '', noteTimer = null;
  function say(text) { note = text; clearTimeout(noteTimer); noteTimer = setTimeout(() => { note = ''; status(); }, 4000); status(); }

  function status() {
    const parts = [];
    const h = hover || (sel >= 0 && L[sel] ? { k: sel } : null);
    if (h && L[h.k]) {
      const l = L[h.k];
      parts.push(`${h.k === 0 ? 'surface' : 'level ' + h.k}: ${fmt(l.p, 1)} hPa, ${fmt(agl(l), 0)} m AGL, T ${fmt(l.T, 1)} °C, ` +
                 `Td ${fmt(l.Td, 1)} °C, RH ${fmt(rhOf(l), 0)} %, ${fmt(dirOf(l), 0)}° ${fmt(ktOf(l), 0)} kt`);
    }
    parts.push(`${L.length} levels${dirty ? ', edited' : ''}`);
    if (bake) parts.push('the Setup changes are in this profile: using it turns them off');
    else if (Object.keys(setupChanges()).length) parts.push('parameters with the Setup changes on top, as the run gets it');
    if (note) parts.unshift(note);
    el('seStatus').textContent = parts.join('  ·  ');
    el('seUndo').disabled = !undo.length;
    el('seRedo').disabled = !redo.length;
    el('seBake').disabled = bake || !Object.keys(setupChanges()).length;
  }

  // ---------------------------------------------------------------- wiring
  function init() {
    el('seClose').onclick = () => close(false);
    el('seUndo').onclick = undoIt;
    el('seRedo').onclick = redoIt;
    el('seReset').onclick = reset;
    el('seUse').onclick = use;
    el('seSave').onclick = save;
    el('seThin').onclick = thin;
    el('seBake').onclick = bakeIn;
    el('seAdd').onclick = () => addAt(parseFloat(el('seAddP').value));
    el('seAddP').onkeydown = e => { if (e.key === 'Enter') el('seAdd').onclick(); };
    el('seTitle').oninput = () => { title = el('seTitle').value; dirty = true; status(); };
    const rows = el('seRows');
    rows.addEventListener('change', onCell);
    rows.addEventListener('keydown', onKey);
    rows.addEventListener('click', onRowButton);
    rows.addEventListener('focusin', e => { const tr = e.target.closest('tr'); if (tr && +tr.dataset.k !== sel) { selectRow(+tr.dataset.k); status(); } });
    for (const [id, plot] of [['seSkew', 'skew'], ['seHodo', 'hodo']]) {
      el(id).addEventListener('mousedown', e => startDrag(e, plot));
      el(id).addEventListener('mousemove', e => hoverAt(e, plot));
      el(id).addEventListener('mouseleave', () => { if (!drag && hover) { hover = null; draw(); status(); } });
    }
    window.addEventListener('mousemove', moveDrag);
    window.addEventListener('mouseup', endDrag);
    // while the editor is open the map's keys (Space, arrows) are off; Esc closes it, Ctrl+Z / Ctrl+Y undo / redo
    document.addEventListener('keydown', e => {
      if (!isOpen) return;
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(false); return; }
      if ((e.ctrlKey || e.metaKey) && !e.altKey && e.target.id !== 'seTitle' && /^[zy]$/i.test(e.key)) {
        e.preventDefault(); e.stopPropagation();
        (e.key.toLowerCase() === 'y' || e.shiftKey) ? redoIt() : undoIt();
        return;
      }
      if (!typing) e.stopPropagation();
    }, true);
    new ResizeObserver(() => draw()).observe(el('seSkew'));
  }

  return { init, open, close, isOpen: () => isOpen, sharppyText };
})();

window.SndEdit = SndEdit;
