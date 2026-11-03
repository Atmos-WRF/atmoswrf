'use strict';
// Fronts, drylines and outflow boundaries: the drawing tool, the list / editor,
// weather-map symbols on the map, and air-mass shading (from /api/airmass).

const Fronts = (() => {
  const COLOR = { cold: '#3b82f6', warm: '#ef4444', dry: '#d6a24a', stationary: '#3b82f6', outflow: '#5eead4', occluded: '#a855f7' };
  let kinds = {};
  let draft = null;                 // [[x, y] km, ...] while drawing
  let hoverPt = null;
  let editing = null;               // {key, run, kind}
  let shade = null, shadeKey = '', shadeBusy = false, shadeTimer = null;

  const running = () => !!(S.run && !['stopped', 'error'].includes(S.run.status));
  const list = () => running() ? (S.run.boundaries || []) : (S.setup.boundaries || (S.setup.boundaries = []));
  const label = k => (kinds[k] && kinds[k].label) || k;

  const JOIN_KM = 8;                  // vertices of different boundaries this close meet at a surface low
  const FRONT_KINDS = ['cold', 'warm', 'stationary'];
  const motionOf = b => { const h = b.heading * Math.PI / 180; return [b.speed * Math.sin(h), b.speed * Math.cos(h)]; };
  // a boundary left to move on its own (empty speed): as boundaries.auto_speed -- a cold front or outflow
  // runs as a density current, 0.7 sqrt(g |dT| / 300 K x depth); a warm front at a third of that; an
  // occluded front at half; a dryline drifts 3 m/s; a stationary front stays put
  function autoSpeed(kind, dT, depth) {
    if (kind === 'stationary') return 0;
    if (kind === 'dry') return 3;
    const c = 0.7 * Math.sqrt(9.81 * Math.abs(dT) / 300 * depth);
    return Math.round((kind === 'warm' ? c / 3 : kind === 'occluded' ? c / 2 : c) * 10) / 10;
  }
  // per-point velocities (m/s); during a run the server's: where fronts meet, the low's motion
  const velsOf = b => (b.vel && b.vel.length === b.points.length) ? b.vel : b.points.map(() => motionOf(b));
  function pointsAt(b, t) {           // metres at model time t
    const V = velsOf(b), dt = t - (b.at || 0);
    return b.points.map(([x, y], i) => [x + V[i][0] * dt, y + V[i][1] * dt]);
  }

  // ---------------------------------------------------------------- form
  function init(defs) {
    kinds = defs || {};
    const sel = $('bKind');
    for (const [k, d] of Object.entries(kinds)) sel.add(new Option(d.label, k));
    sel.value = 'cold';
    fillForm(defaultsFor('cold'));
    sel.onchange = () => { if (!editing) fillForm(defaultsFor(sel.value)); };
    $('toolFront').onclick = () => setTool('front');
    $('bApply').onclick = applyEdit;
    $('bCancel').onclick = endEdit;
    $('showAirmass').onchange = () => map.redraw();
    $('frontList').addEventListener('click', onListClick);
    $('lowAuto').onchange = () => setLow({ auto: $('lowAuto').checked });
    $('lowSpeed').onchange = () => setLow({ speed: Math.max(0, parseFloat($('lowSpeed').value) || 0) });
    $('lowHeading').onchange = () => setLow({ heading: (((parseFloat($('lowHeading').value) || 0) % 360) + 360) % 360 });
    $('occlBtn').onclick = displaceLow;
  }

  function defaultsFor(kind) { return { kind, ...kinds[kind], init_start: 0, init_end: 4 * 3600 }; }

  function fillForm(b) {
    $('bKind').value = b.kind;
    $('bHeading').value = Math.round(b.heading); $('bSpeed').value = b.auto_speed ? '' : b.speed;
    $('bDT').value = b.dT; $('bDepth').value = b.depth;
    $('bDq').value = Math.round(b.dq * 100); $('bHw').value = +(b.hw / 1000).toFixed(1);
    $('bWdir').value = b.wind_dir === null || b.wind_dir === undefined ? '' : b.wind_dir;
    $('bWspd').value = b.wind_speed;
    $('bRate').value = b.init_rate; $('bVar').value = b.variation;
    $('bMode').value = b.mode || 'auto';
    const ws = kinds[b.kind] || {};
    $('bBack').value = b.back ?? ws.back ?? 0; $('bJet').value = b.jet ?? ws.jet ?? 0; $('bConv').value = b.conv ?? ws.conv ?? 0;
    $('bInitStart').value = Math.round((b.init_start || 0) / 60);
    $('bInitEnd').value = Math.round((b.init_end === undefined ? 14400 : b.init_end) / 60);
    $('bKeep').checked = !!b.keep;
  }

  function readForm() {
    const kind = $('bKind').value, d = kinds[kind];
    const num = (id, dflt) => { const v = parseFloat($(id).value); return isFinite(v) ? v : dflt; };
    const wd = $('bWdir').value.trim();
    const dT = num('bDT', d.dT), depth = Math.max(200, num('bDepth', d.depth));
    const auto = $('bSpeed').value.trim() === '';
    return {
      kind, heading: num('bHeading', d.heading), dT, depth,
      speed: auto ? autoSpeed(kind, dT, depth) : Math.max(0, num('bSpeed', d.speed)), auto_speed: auto,
      dq: Math.min(0.9, Math.max(0, num('bDq', d.dq * 100) / 100)),
      hw: Math.max(1, num('bHw', d.hw / 1000)) * 1000,
      wind_dir: wd === '' ? null : +wd, wind_speed: Math.max(0, num('bWspd', d.wind_speed)),
      init_rate: Math.max(0, num('bRate', d.init_rate)), variation: Math.min(1, Math.max(0, num('bVar', d.variation))),
      init_start: Math.max(0, num('bInitStart', 0)) * 60, init_end: Math.max(0, num('bInitEnd', 240)) * 60,
      keep: $('bKeep').checked,
      mode: $('bMode').value || 'auto',
      back: Math.min(90, Math.max(0, num('bBack', d.back || 0))), jet: Math.min(25, Math.max(0, num('bJet', d.jet || 0))),
      conv: Math.min(15, Math.max(0, num('bConv', d.conv || 0))),
    };
  }

  // Air mass behind a cold front / dryline / outflow, ahead of a warm front
  // (same rule as boundaries.Boundary.default_side); +1 = left of the drawn direction.
  function defaultSide(b) {
    const p = b.points, n = p.length;
    if (n < 2) return 1;
    const lx = -(p[n - 1][1] - p[0][1]), ly = p[n - 1][0] - p[0][0];
    const h = b.heading * Math.PI / 180, vx = b.speed * Math.sin(h), vy = b.speed * Math.cos(h);
    if (Math.abs(vx) + Math.abs(vy) < 1e-6) return ly >= 0 ? 1 : -1;
    return (Math.sign(lx * vx + ly * vy) || 1) * (b.kind === 'warm' ? 1 : -1);
  }

  // ---------------------------------------------------------------- surface lows
  // Where boundaries meet, their meeting points move as one: a surface low (boundaries.join_at_lows).
  // The run's lows come from the server; the setup's are found the same way here.
  const lowCfg = () => (running() ? S.run.settings && S.run.settings.low : S.setup.low) || { auto: true, speed: 10, heading: 45 };

  function lowsNow() {
    if (running()) return S.run.lows || [];
    const bs = list(), V = [], lows = [], c = lowCfg();
    bs.forEach((b, bi) => b.points.forEach(([x, y], i) => V.push({ bi, i, x, y })));
    const used = V.map(() => false);
    for (let a = 0; a < V.length; a++) {
      if (used[a]) continue;
      const g = [a];
      for (let k = a + 1; k < V.length; k++) {
        if (used[k] || g.some(q => V[q].bi === V[k].bi)) continue;
        if (Math.hypot(V[k].x - V[a].x, V[k].y - V[a].y) < JOIN_KM * 1000) g.push(k);
      }
      if (g.length < 2) continue;
      g.forEach(q => { used[q] = true; });
      const mem = g.map(q => V[q]), n = mem.length;
      let v;
      if (c.auto === false) { const h = (c.heading || 0) * Math.PI / 180; v = [c.speed * Math.sin(h), c.speed * Math.cos(h)]; }
      else { const ms = mem.map(q => motionOf(bs[q.bi])); v = [ms.reduce((s, m) => s + m[0], 0) / n, ms.reduce((s, m) => s + m[1], 0) / n]; }
      const front = mem.some(q => FRONT_KINDS.includes(bs[q.bi].kind)), occl = mem.some(q => bs[q.bi].kind === 'occluded');
      lows.push({ x: mem.reduce((s, q) => s + q.x, 0) / n, y: mem.reduce((s, q) => s + q.y, 0) / n, vx: v[0], vy: v[1],
                  members: mem.map(q => [q.bi, q.i]), front: front && !occl, triple: front && occl });
    }
    // an occluded front joined at one end: its other end is the low's centre (drawn as the L)
    const joined = new Set(lows.flatMap(L => L.members.map(([bi, i]) => bi + ':' + i)));
    bs.forEach((b, bi) => {
      const n = b.points.length;
      if (b.kind !== 'occluded' || n < 2) return;
      for (const [i, j] of [[0, n - 1], [n - 1, 0]]) {
        if (joined.has(bi + ':' + i) || !joined.has(bi + ':' + j)) continue;
        let v = motionOf(b);
        if (c.auto === false) { const h = (c.heading || 0) * Math.PI / 180; v = [c.speed * Math.sin(h), c.speed * Math.cos(h)]; }
        lows.push({ x: b.points[i][0], y: b.points[i][1], vx: v[0], vy: v[1], members: [[bi, i]], front: true, centre: true });
      }
    });
    return lows;
  }

  // the setup's meeting points snap to their low (the run does the same when it starts)
  function joinSetup() {
    if (running()) return;
    const bs = list();
    for (const L of lowsNow()) for (const [bi, i] of L.members) bs[bi].points[i] = [L.x, L.y];
  }

  async function setLow(f) {
    try {
      if (running()) await api('/api/run/low', { ...lowCfg(), ...f });
      else S.setup.low = { ...lowCfg(), ...f };
    } catch (e) { alert('Surface low: ' + e.message); }
    render(); map.redraw();
  }

  function renderLow() {
    const all = lowsNow(), lows = all.filter(L => L.front);
    $('lowBox').classList.toggle('hidden', !lows.length);
    if (!lows.length) return;
    // a low whose fronts still meet at it can be occluded: its centre moved away from the triple point
    $('occlRow').classList.toggle('hidden', !lows.some(L => !L.centre));
    $('occlNote').textContent = all.some(L => L.centre) ? 'Occluded: the L is the low\'s centre, the fronts meet at the triple point. Drag the L to move the centre alone.' : '';
    const L = lows.find(q => q.centre) || lows[0], c = lowCfg(), t = running() ? displayTime() : 0;
    const ll = map.proj ? map.inv((L.x + L.vx * t) / 1000, (L.y + L.vy * t) / 1000) : null;
    $('lowWhere').textContent = (ll ? `${ll[0].toFixed(2)}°, ${ll[1].toFixed(2)}°` : '') + (lows.length > 1 ? ` (+${lows.length - 1} more)` : '');
    const auto = c.auto !== false;
    $('lowAuto').checked = auto;
    $('lowSet').classList.toggle('hidden', auto);
    $('lowMotion').textContent = `${Math.hypot(L.vx, L.vy).toFixed(1)} m/s toward ${Math.round((Math.atan2(L.vx, L.vy) * 180 / Math.PI + 360) % 360)}°`;
    if (document.activeElement !== $('lowSpeed')) $('lowSpeed').value = c.speed ?? 10;
    if (document.activeElement !== $('lowHeading')) $('lowHeading').value = c.heading ?? 45;
  }

  // before the run: drag the L to move the low with its fronts; Shift-drag moves only the meeting point
  function lowAt(x, y) {
    if (running() || draft) return null;
    const tol = 14 / map.scale;
    return lowsNow().find(L => (L.front || L.triple) && Math.hypot(L.x / 1000 - x, L.y / 1000 - y) < tol) || null;
  }

  // the L of a plain low or the triple point drags its fronts along (Shift: the meeting point only); an occluded
  // low's centre moves on its own, stretching or shortening the occluded front
  function press(x, y, e) {
    const L = lowAt(x, y);
    if (!L) return null;
    const bs = list(), orig = bs.map(b => b.points.map(p => p.slice()));
    const whole = e.shiftKey || L.centre ? null : [...new Set(L.members.map(([bi]) => bi))];
    return {
      move(wx, wy) {
        const dx = (wx - x) * 1000, dy = (wy - y) * 1000;
        if (whole) for (const bi of whole) bs[bi].points = orig[bi].map(([px, py]) => [px + dx, py + dy]);
        else for (const [bi, i] of L.members) bs[bi].points[i] = [orig[bi][i][0] + dx, orig[bi][i][1] + dy];
        map.redraw();
      },
      end() { joinSetup(); render(); map.redraw(); },
    };
  }

  // the first plain low becomes an occluded one: its centre moves occlKm toward occlDir, an occluded front joins
  // it to the old meeting point (now the triple point), moving across itself the way its fronts go
  async function displaceLow() {
    const t = running() ? displayTime() : 0;
    const L = lowsNow().find(q => q.front && !q.centre);
    if (!L) return;
    const km = Math.max(20, parseFloat($('occlKm').value) || 150), dir = (parseFloat($('occlDir').value) || 0) * Math.PI / 180;
    const bs = list();
    const tx = L.x + L.vx * t, ty = L.y + L.vy * t;
    const cx = tx + km * 1000 * Math.sin(dir), cy = ty + km * 1000 * Math.cos(dir);
    const b = { ...defaultsFor('occluded'), points: [[tx, ty], [cx, cy]], auto_speed: true };
    b.speed = autoSpeed('occluded', b.dT, b.depth);
    const ms = L.members.map(([bi]) => motionOf(bs[bi]));
    const mv = [ms.reduce((s, m) => s + m[0], 0) / ms.length, ms.reduce((s, m) => s + m[1], 0) / ms.length];
    const hd = (vx, vy) => (Math.atan2(vx, vy) * 180 / Math.PI + 360) % 360;
    const dx = cx - tx, dy = cy - ty, c1 = hd(-dy, dx), c2 = hd(dy, -dx), want = hd(mv[0], mv[1]);
    const dist = (a, c) => Math.abs(((a - c + 540) % 360) - 180);
    b.heading = Math.round(dist(c1, want) <= dist(c2, want) ? c1 : c2);
    b.side = defaultSide(b);
    try {
      if (running()) {
        b.at = t; b.init_start += S.run.t; b.init_end += S.run.t;
        await api('/api/run/boundary', b);
      } else {
        b.at = 0;
        bs.push(b);
        joinSetup();
      }
    } catch (e) { alert('Occluded front: ' + e.message); }
    render(); map.redraw();
  }

  // ---------------------------------------------------------------- drawing tool
  // a click near another boundary's point lands on it: the boundaries meet there (a surface low)
  function snap(x, y) {
    const t = running() ? displayTime() : 0;
    let best = null, bd = 12 / map.scale;
    for (const b of list()) {
      for (const [px, py] of pointsAt(b, t)) {
        const d = Math.hypot(px / 1000 - x, py / 1000 - y);
        if (d < bd) { bd = d; best = [px / 1000, py / 1000]; }
      }
    }
    return best;
  }
  function addPoint(x, y) { (draft = draft || []).push(snap(x, y) || [x, y]); map.redraw(); }
  function hover(x, y) {
    const s = snap(x, y), was = hoverPt && hoverPt.snapped;
    hoverPt = s || [x, y]; hoverPt.snapped = !!s;
    if (draft || s || was) map.redraw();
  }
  function cancelDraft() { draft = null; hoverPt = null; map.redraw(); }

  async function finish() {
    if (!draft) return;
    const tol = 4 / map.scale;            // double-click adds the last point twice
    const pts = draft.filter((p, i) => i === 0 || Math.hypot(p[0] - draft[i - 1][0], p[1] - draft[i - 1][1]) > tol);
    draft = null;
    if (pts.length < 2) { setTool('pan'); return; }
    const b = readForm();
    b.points = pts.map(([x, y]) => [x * 1000, y * 1000]);
    // fronts move across themselves: the perpendicular nearest the chosen direction
    if (b.speed > 0) {
      const p = b.points, dx = p[p.length - 1][0] - p[0][0], dy = p[p.length - 1][1] - p[0][1];
      const hd = (vx, vy) => (Math.atan2(vx, vy) * 180 / Math.PI + 360) % 360;
      const c1 = hd(-dy, dx), c2 = hd(dy, -dx);
      const dist = (a, c) => Math.abs(((a - c + 540) % 360) - 180);
      b.heading = Math.round(dist(c1, b.heading) <= dist(c2, b.heading) ? c1 : c2);
      $('bHeading').value = b.heading;
    }
    b.side = defaultSide(b);
    setTool('pan');
    try {
      if (running()) {
        b.at = displayTime();             // where it was clicked at the time the map shows
        b.init_start += S.run.t; b.init_end += S.run.t;
        await api('/api/run/boundary', b);
      } else {
        b.at = 0;
        list().push(b);
        joinSetup();
      }
    } catch (e) { alert('Boundary: ' + e.message); }
    render(); map.redraw();
  }

  // ---------------------------------------------------------------- list / editor
  const lengthKm = pts => pts.reduce((s, p, i) => i ? s + Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) : 0, 0) / 1000;

  function render() {
    const bs = list(), run = running();
    const html = bs.map((b, i) => {
      const key = run ? b.id : i;
      const motion = b.speed ? `${b.speed} m/s → ${Math.round(b.heading)}°` : 'stationary';
      return `<div class="item"><div><b style="color:${COLOR[b.kind] || '#ccc'}">${label(b.kind)}${run ? ' #' + b.id : ''}</b>` +
        ` <span class="meta">${lengthKm(b.points).toFixed(0)} km · ${motion} · ΔT ${b.dT} K` +
        `${b.dq ? ' · ' + Math.round(b.dq * 100) + '% drier' : ''} · ${b.init_rate}/h/100 km${b.mode === 'line' ? ' · squall line' : b.mode === 'cells' ? ' · cells' : ''}</span></div>` +
        `<span class="btns"><button data-bedit="${key}">Edit</button><button data-bflip="${key}">Flip</button>` +
        `<button data-bdel="${key}">Remove</button></span></div>`;
    }).join('') || '<div class="hint">No boundaries yet.</div>';
    const el = $('frontList');
    if (el._last !== html) { el.innerHTML = html; el._last = html; }
    renderLow();
  }

  function find(key) { return running() ? list().find(q => q.id === key) : list()[key]; }

  async function onListClick(e) {
    const btn = e.target.closest('button');
    if (!btn) return;
    const run = running();
    const key = +(btn.dataset.bedit ?? btn.dataset.bflip ?? btn.dataset.bdel);
    const b = find(key);
    if (!b) return;
    try {
      if (btn.dataset.bedit !== undefined) {
        editing = { key, run, kind: b.kind };
        fillForm(b);
        $('bApply').textContent = `Apply to ${label(b.kind).toLowerCase()}${run ? ' #' + b.id : ''}`;
        $('bApply').classList.remove('hidden'); $('bCancel').classList.remove('hidden');
      } else if (btn.dataset.bflip !== undefined) {
        if (run) await api('/api/run/boundary/update', { id: b.id, side: -(b.side || 1) });
        else b.side = -(b.side || defaultSide(b));
      } else {
        if (run) await api('/api/run/boundary/remove', { id: b.id });
        else list().splice(key, 1);
        if (editing && editing.key === key) endEdit();
      }
    } catch (err) { alert('Boundary: ' + err.message); }
    render(); map.redraw();
  }

  async function applyEdit() {
    if (!editing) return;
    const f = readForm();
    try {
      if (editing.run) {
        await api('/api/run/boundary/update', { id: editing.key, ...f });
      } else {
        const b = list()[editing.key];
        if (b) { Object.assign(b, f); if (b.kind !== editing.kind) b.side = defaultSide(b); }
      }
    } catch (e) { alert('Boundary: ' + e.message); }
    endEdit(); render(); map.redraw();
  }

  function endEdit() {
    editing = null;
    $('bApply').classList.add('hidden'); $('bCancel').classList.add('hidden');
  }

  // ---------------------------------------------------------------- map
  function displayTime() {
    if (S.frames && S.frames.length) {
      const i = S.frameIdx < 0 ? S.frames.length - 1 : Math.min(S.frameIdx, S.frames.length - 1);
      return S.frames[i].t;
    }
    return S.run && S.run.t !== undefined ? S.run.t : 0;
  }

  function draw(g, m) {
    const bs = list();
    // a run's boundaries move with the time shown; the setup's belong to the next run, which
    // starts at t = 0 (the clock of a stopped run pushed them ahead by speed x its time)
    const t = running() ? displayTime() : 0;
    if ($('showAirmass').checked && bs.length) {
      requestShade(m, bs, t);
      if (shade) {
        const [a, b] = m.toScreen(shade.x0, shade.y1), [c, d] = m.toScreen(shade.x1, shade.y0);
        g.imageSmoothingEnabled = true;
        g.drawImage(shade.canvas, a, b, c - a, d - b);
      }
    }
    for (const b of bs) drawBoundary(g, m, b, t);
    for (const L of lowsNow()) if (L.front) drawLow(g, m, (L.x + L.vx * t) / 1000, (L.y + L.vy * t) / 1000);
    if (draft) drawDraft(g, m);
    else if (hoverPt && hoverPt.snapped && S.tool === 'front') drawSnap(g, m);
  }

  function drawLow(g, m, x, y) {
    const [px, py] = m.toScreen(x, y);
    g.font = 'bold 26px Segoe UI'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.lineJoin = 'round';
    g.lineWidth = 5; g.strokeStyle = 'rgba(0,0,0,0.8)'; g.strokeText('L', px, py);
    g.fillStyle = '#ff3b30'; g.fillText('L', px, py);
    g.textAlign = 'start'; g.textBaseline = 'alphabetic'; g.lineJoin = 'miter';
  }

  function drawSnap(g, m) {
    const [x, y] = m.toScreen(...hoverPt);
    g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 7, 0, 7); g.stroke();
  }

  function requestShade(m, bs, t) {
    const [x0, y1] = m.toWorld(0, 0), [x1, y0] = m.toWorld(m.W, m.H);
    const key = JSON.stringify([Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1), Math.round(t / 30), lowCfg(),
      bs.map(b => [b.kind, b.points, b.vel, b.side, b.speed, b.heading, b.dT, b.dq, b.depth, b.hw, b.variation, b.at || 0])]);
    if (key === shadeKey) return;
    clearTimeout(shadeTimer);
    shadeTimer = setTimeout(async () => {
      if (shadeBusy) return;
      shadeBusy = true;
      try {
        const nx = Math.min(240, Math.max(16, Math.round(m.W / 6))), ny = Math.min(240, Math.max(16, Math.round(m.H / 6)));
        const d = await getJSON('/api/airmass', { boundaries: bs, low: lowCfg(), t, x0: x0 * 1000, x1: x1 * 1000, y0: y0 * 1000, y1: y1 * 1000, nx, ny });
        shade = d.dT && d.dT.length ? shadeImage(d) : null;
        shadeKey = key;
        map.redraw();
      } catch (e) { /* server busy: next redraw retries */ }
      finally { shadeBusy = false; }
    }, 200);
  }

  function shadeImage(d) {
    const cv = Object.assign(document.createElement('canvas'), { width: d.nx, height: d.ny });
    const g = cv.getContext('2d'), img = g.createImageData(d.nx, d.ny);
    for (let j = 0; j < d.ny; j++) {
      for (let i = 0; i < d.nx; i++) {
        const k = j * d.nx + i, dT = d.dT[k], dq = d.dq[k];
        let r = 0, gg = 0, b = 0, a = 0;
        if (dT < -0.3) { r = 80; gg = 150; b = 255; a = Math.min(0.36, -dT / 8 * 0.36); }
        else if (dq > 0.05) { r = 205; gg = 150; b = 80; a = Math.min(0.28, dq * 0.42); }
        else if (dT > 0.3) { r = 255; gg = 150; b = 60; a = Math.min(0.25, dT / 6 * 0.25); }
        const o = ((d.ny - 1 - j) * d.nx + i) * 4;          // row 0 = south: draw at the bottom
        img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = Math.round(a * 255);
      }
    }
    g.putImageData(img, 0, 0);
    return { canvas: cv, x0: d.x0 / 1000, x1: d.x1 / 1000, y0: d.y0 / 1000, y1: d.y1 / 1000 };
  }

  function screenPoints(m, b, t) {
    return pointsAt(b, t).map(([x, y]) => m.toScreen(x / 1000, y / 1000));
  }

  function drawBoundary(g, m, b, t) {
    const P = screenPoints(m, b, t);
    if (P.length < 2) return;
    const side = b.side || 1;
    g.lineJoin = 'round'; g.lineCap = 'round';
    if (b.kind === 'outflow') {
      g.strokeStyle = COLOR.outflow; g.lineWidth = 2.2; g.setLineDash([9, 6]);
      g.beginPath(); P.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
      g.setLineDash([]);
      const mid = P[Math.floor((P.length - 1) / 2)];
      g.fillStyle = COLOR.outflow; g.font = 'bold 11px Segoe UI'; g.fillText('OFB', mid[0] + 6, mid[1] - 6);
      return;
    }
    if (b.kind !== 'stationary') {
      g.strokeStyle = COLOR[b.kind] || '#ccc'; g.lineWidth = 2.6;
      g.beginPath(); P.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
    }
    // symbols every `gap` px; n is the screen normal pointing to the world-left of the line
    const gap = 36, size = 7;
    if (b.kind === 'stationary') {                    // alternating blue / red, one colour per symbol
      g.lineWidth = 2.6;
      for (const [col, off] of [[COLOR.cold, 0], [COLOR.warm, gap]]) {
        g.strokeStyle = col; g.setLineDash([gap, gap]); g.lineDashOffset = off;
        g.beginPath(); P.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
      }
      g.setLineDash([]); g.lineDashOffset = 0;
    }
    let acc = gap / 2, idx = 0;
    for (let s = 0; s < P.length - 1; s++) {
      const [ax, ay] = P[s], [bx, by] = P[s + 1];
      const L = Math.hypot(bx - ax, by - ay);
      if (L < 1e-6) continue;
      const tx = (bx - ax) / L, ty = (by - ay) / L;
      const nlx = ty, nly = -tx;                              // world-left in screen space
      while (acc <= L) {
        const px = ax + tx * acc, py = ay + ty * acc;
        let kind = b.kind, dir = -side;                       // toward the warm / moist side
        if (kind === 'warm') dir = side;                      // toward the cold air it moves into
        if (kind === 'stationary') { kind = idx % 2 ? 'warm' : 'cold'; dir = kind === 'warm' ? side : -side; }
        let col = null;
        if (kind === 'occluded') { kind = idx % 2 ? 'warm' : 'cold'; col = COLOR.occluded; }   // both toward its motion
        symbol(g, kind, px, py, tx, ty, nlx * dir, nly * dir, size, col);
        acc += gap; idx++;
      }
      acc -= L;
    }
  }

  function symbol(g, kind, px, py, tx, ty, nx, ny, r, col) {
    if (kind === 'cold') {
      g.fillStyle = col || COLOR.cold;
      g.beginPath(); g.moveTo(px - tx * r, py - ty * r); g.lineTo(px + tx * r, py + ty * r);
      g.lineTo(px + nx * r * 1.3, py + ny * r * 1.3); g.closePath(); g.fill();
      return;
    }
    // semicircle bulging toward n
    const th = Math.atan2(ty, tx);
    const ccw = (tx * ny - ty * nx) < 0;
    g.beginPath(); g.arc(px, py, r, th, th + Math.PI, ccw);
    if (kind === 'dry') { g.strokeStyle = COLOR.dry; g.lineWidth = 2; g.stroke(); }
    else { g.fillStyle = col || COLOR.warm; g.closePath(); g.fill(); }
  }

  function drawDraft(g, m) {
    const kind = $('bKind').value;
    const pts = draft.map(([x, y]) => m.toScreen(x, y));
    if (hoverPt) pts.push(m.toScreen(...hoverPt));
    g.strokeStyle = COLOR[kind] || '#fff'; g.lineWidth = 2; g.setLineDash([6, 4]);
    g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y)); g.stroke();
    g.setLineDash([]); g.fillStyle = '#fff';
    for (const [x, y] of pts.slice(0, draft.length)) { g.beginPath(); g.arc(x, y, 3, 0, 7); g.fill(); }
    if (hoverPt && hoverPt.snapped) drawSnap(g, m);
  }

  return { init, render, draw, addPoint, hover, finish, cancelDraft, press, lowAt, drawing: () => !!draft };
})();

window.Fronts = Fronts;
