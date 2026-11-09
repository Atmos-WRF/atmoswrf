'use strict';
// Chase mode (web/chase.html): drive a car on the real roads under the live run's storms.
//  - the storm: the run's cloud volume (100 km around the car, re-fetched every few seconds while the run goes,
//    re-centred when the car has driven far), drawn by the 3-D view's cloud renderer with the sun where it is at
//    the model's clock;
//  - the ground: WorldCover's land cover, OpenStreetMap's roads as drivable geometry (roads.js);
//  - the car: carphys.js, on what is under each wheel (asphalt, gravel, the verge, fields, water); rain from the
//    model wets the roads;
//  - the HUD: speed, gear, revs, the surface, the nearest tornado's distance and bearing, a minimap with the roads,
//    the warnings and the tornadoes.
const $ = id => document.getElementById(id);
const api = async (path, body) => {
  const h = window.ChaseAuth ? window.ChaseAuth() : {};
  const r = await fetch(path, body === undefined ? { headers: h } : {
    method: 'POST', headers: { 'Content-Type': 'application/json', ...h }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = r.status + '';
    try { msg = (await r.json()).error || msg; } catch (e) { /* binary */ }
    throw new Error(msg);
  }
  return r;
};
const getJSON = async p => (await api(p)).json();
function unpack(buf) {
  const dv = new DataView(buf), hl = dv.getUint32(0, true);
  return { header: JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl))), data: new Uint8Array(buf, 4 + hl) };
}

const Chase = (() => {
  // the storm is fetched about once a second (VOL_MIN) and blended into the picture over the time between fetches;
  // a fetch that takes long (a slow model step, the CPU backend, a slow link) spaces the next ones out (VOL_LOAD x
  // its time, up to VOL_MAX), so the page never holds the run's lock for more than a small share of its time
  const BOX_KM = 100, RECENTRE_KM = 18, VOL_MIN = 1000, VOL_MAX = 8000, VOL_LOAD = 2.5;
  const BLEND_MIN = 300, BLEND_MAX = 90000;          // ms: the blend into each new volume (the time they take to come)
  const TARGETS_EVERY = 1000;                         // ms: the tornadoes' positions (cheap)
  const COLORS = { suv: [196, 30, 36], sedan: [235, 235, 232], truck: [70, 74, 66] };
  // what is under a wheel: grip (mu), rolling drag, the slip at the grip's peak (driving, cornering), the curve's
  // shape (lower: a flatter peak, looser), roughness m
  const SURF = {
    asphalt: { name: 'Asphalt', mu: 1.05, crr: 0.012, kpk: 0.12, apk: 0.14, C: 1.4, rough: 0 },
    wet: { name: 'Wet asphalt', mu: 0.88, crr: 0.014, kpk: 0.11, apk: 0.13, C: 1.4, rough: 0 },
    gravel: { name: 'Gravel', mu: 0.85, crr: 0.03, kpk: 0.16, apk: 0.18, C: 1.3, rough: 0.015 },
    // off the road: nearly a dirt road's grip (it holds its line), but rolling drag that grows with speed
    verge: { name: 'Verge', mu: 0.9, crr: 0.05, crrV: 18, kpk: 0.15, apk: 0.17, C: 1.35, rough: 0.025 },
    grass: { name: 'Grass', mu: 0.88, crr: 0.07, crrV: 14, kpk: 0.15, apk: 0.17, C: 1.35, rough: 0.05 },
    crops: { name: 'Field', mu: 0.85, crr: 0.1, crrV: 11, kpk: 0.16, apk: 0.18, C: 1.35, rough: 0.07 },
    bare: { name: 'Dirt', mu: 0.88, crr: 0.05, crrV: 16, kpk: 0.15, apk: 0.17, C: 1.35, rough: 0.04 },
    town: { name: 'Lawns and lots', mu: 0.92, crr: 0.03, crrV: 20, kpk: 0.14, apk: 0.16, C: 1.35, rough: 0.02 },
    wood: { name: 'Woods', mu: 0.85, crr: 0.15, crrV: 8, kpk: 0.16, apk: 0.18, C: 1.35, rough: 0.1 },
    marsh: { name: 'Marsh', mu: 0.8, crr: 0.25, crrV: 6, kpk: 0.17, apk: 0.19, C: 1.3, rough: 0.06 },
    water: { name: 'Water', mu: 0.75, crr: 0.5, crrV: 4, kpk: 0.18, apk: 0.2, C: 1.3, rough: 0.0 },
    mud: { name: 'Mud', mu: 0.8, crr: 0.18, crrV: 7, kpk: 0.17, apk: 0.19, C: 1.3, rough: 0.06 },
  };
  const LC = { 10: 'wood', 20: 'grass', 30: 'grass', 40: 'crops', 50: 'town', 60: 'bare', 70: 'bare', 80: 'water', 90: 'marsh', 95: 'marsh', 100: 'grass' };
  const CAMS = ['chase', 'hood', 'driver'];
  // the look (the 3-D view's cloud-mode settings, plus quality): the menu's Settings tab, kept in this browser
  const VIS0 = { density: 0.5, rain: 1, detail: 0.7, relief: 0.6, haze: 180, sunMode: 'model', sunAz: 235, sunH: 30,
                 base: 'plain', scale: 0.75, tricubic: 1, blend: 0.3, trees: 'full',
                 contrast: 1.3, bounce: 1, mmBase: 'osmdark', mmProduct: 'cref', mmSite: 'nearest', mmOpacity: 0.8, mmDim: 0.15, mmBig: 'small',
                 vol: 0.7, precip: 'on', map: 'detailed', far: 'on', nestStep: '2', markers: 'off' };
  const QUICK = ['cref', 'REF', 'VEL', 'SRV', 'RHO', 'none'];          // K steps through these
  const TREE_BUDGET = { off: 0, light: 25000, full: 70000 };

  const st = {
    run: null, targets: null, car: null, box: null, volBusy: false, volAt: 0, volErr: '', started: false,
    cam: { mode: 'chase', yaw: 0, look: 0, pitch: 0, lookHeld: false, lookT: 0, dist: 1, eye: null },
    keys: new Set(), last: 0, fps: 0, units: 'mph', wet: 0, rain: 0, vis: null,
    name: 'Chaser', kind: 'suv', color: null, spawned: false, msg: '', msgT: 0, help: false,
  };

  // ---------------------------------------------------------------- the ground under a point (world m)
  function vnoise(x, y) {
    const h = (i, j) => { let n = Math.imul(i | 0, 374761393) + Math.imul(j | 0, 668265263); n = Math.imul(n ^ (n >>> 13), 1274126177); return ((n ^ (n >>> 16)) >>> 0) / 4294967296; };
    const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    return (h(i, j) * (1 - sx) + h(i + 1, j) * sx) * (1 - sy) + (h(i, j + 1) * (1 - sx) + h(i + 1, j + 1) * sx) * sy;
  }
  // the detailed map's surfaces (worldlink.js: its road surfaces, its land classes) as chase mode's
  const W_ROAD = { asphalt: 'asphalt', gravel: 'gravel', dirt: 'bare', grass: 'grass', mud: 'mud', water: 'water' };
  const W_LAND = { grass: 'grass', 'crop field': 'crops', 'bare soil': 'bare', forest: 'wood', mud: 'mud',
                   'gravel / rock': 'gravel', urban: 'town', water: 'water' };
  function worldSurf(x, y) {
    if (!detailed() || !WorldLink.ready()) return null;
    const g = WorldLink.ground(x, y, st.car ? st.car.p[2] : undefined);
    if (!g) return null;
    let s;
    if (g.road) s = SURF[W_ROAD[g.surface] || 'asphalt'];
    else s = SURF[W_LAND[g.land] || 'grass'];
    if (s === SURF.asphalt && st.wet > 0.3) s = SURF.wet;
    if (!g.road && st.wet > 0.5 && [SURF.grass, SURF.crops, SURF.bare].includes(s)) s = SURF.mud;
    if (g.water > 0.25) s = SURF.water;
    let h = g.h;
    if (s.rough) h += s.rough * (vnoise(x / 1.6, y / 1.6) + 0.5 * vnoise(x / 0.55 + 31, y / 0.55)) - s.rough * 0.6;
    return { h, s };
  }
  const detailed = () => st.vis && st.vis.map !== 'basic';
  function surfAt(x, y) {
    const ap = Stations.apron(x, y);                   // a gas station's concrete (stations.js)
    if (ap !== null) return { h: ap, s: st.wet > 0.3 ? SURF.wet : SURF.asphalt };
    const w = worldSurf(x, y);
    if (w) return w;
    const r = Roads.at(x, y);
    let s;
    if (r.seg >= 0) s = r.paved ? (st.wet > 0.3 ? SURF.wet : SURF.asphalt) : SURF.gravel;
    else if (r.raise > 0.02) s = st.wet > 0.5 ? SURF.mud : SURF.verge;
    else {
      s = SURF[LC[Land.coverAt(x / 1000, y / 1000)] || 'grass'];
      if (st.wet > 0.5 && ['grass', 'crops', 'bare', 'verge'].includes(LC[Land.coverAt(x / 1000, y / 1000)])) s = SURF.mud;
    }
    let h = Render.terrain(x, y) + r.raise;
    if (s.rough) h += s.rough * (vnoise(x / 1.6, y / 1.6) + 0.5 * vnoise(x / 0.55 + 31, y / 0.55)) - s.rough * 0.6;
    if (s === SURF.water) h -= 0.35;
    return { h, s };
  }
  function ground(x, y) {
    const a = surfAt(x, y), d = 0.3;
    const hx = surfAt(x + d, y).h, hy = surfAt(x, y + d).h;
    const nx = -(hx - a.h) / d, ny = -(hy - a.h) / d, l = Math.hypot(nx, ny, 1);
    return { h: a.h, n: [nx / l, ny / l, 1 / l], ...a.s };
  }

  // ---------------------------------------------------------------- the sun at the model's clock
  function sunAt(ms, lat, lon) {
    const rad = Math.PI / 180, d = ms / 86400000 + 2440587.5 - 2451545.0;
    const g = (357.529 + 0.98560028 * d) * rad, q = 280.459 + 0.98564736 * d;
    const Ls = (q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * rad, e = (23.439 - 0.00000036 * d) * rad;
    const ra = Math.atan2(Math.cos(e) * Math.sin(Ls), Math.cos(Ls)), dec = Math.asin(Math.sin(e) * Math.sin(Ls));
    const gmst = (18.697374558 + 24.06570982441908 * d) % 24;
    const H = ((gmst * 15 + lon) * rad - ra);
    const la = lat * rad;
    const el = Math.asin(Math.sin(la) * Math.sin(dec) + Math.cos(la) * Math.cos(dec) * Math.cos(H));
    const az = Math.atan2(-Math.sin(H), Math.tan(dec) * Math.cos(la) - Math.sin(la) * Math.cos(H));
    return { az, el, Ls };
  }
  // the moon (low precision: its main terms, ~1 deg): where it is and how much of it is lit
  function moonAt(ms, lat, lon) {
    const rad = Math.PI / 180, d = ms / 86400000 + 2440587.5 - 2451545.0;
    const L = (218.316 + 13.176396 * d) * rad, M = (134.963 + 13.064993 * d) * rad, F = (93.272 + 13.229350 * d) * rad;
    const lam = L + 6.289 * rad * Math.sin(M), bet = 5.128 * rad * Math.sin(F), e = 23.4397 * rad;
    const ra = Math.atan2(Math.sin(lam) * Math.cos(e) - Math.tan(bet) * Math.sin(e), Math.cos(lam));
    const dec = Math.asin(Math.sin(bet) * Math.cos(e) + Math.cos(bet) * Math.sin(e) * Math.sin(lam));
    const gmst = (18.697374558 + 24.06570982441908 * d) % 24, H = (gmst * 15 + lon) * rad - ra, la = lat * rad;
    const el = Math.asin(Math.sin(la) * Math.sin(dec) + Math.cos(la) * Math.cos(dec) * Math.cos(H));
    const az = Math.atan2(-Math.sin(H), Math.tan(dec) * Math.cos(la) - Math.sin(la) * Math.cos(H));
    const s = sunAt(ms, lat, lon);
    return { az, el, lit: (1 - Math.cos(lam - s.Ls)) / 2 };
  }
  // the night for render.js: daylight (1) to night (0) through civil twilight, the eye's adaptation, the moon,
  // the lamps on
  function nightView() {
    const r = st.run, vs = st.vis;
    let el, mo = null;
    if (vs.sunMode === 'manual' || !(r && r.model_ms && r.settings && r.settings.map)) el = vs.sunH * Math.PI / 180;
    else {
      el = sunAt(r.model_ms, r.settings.map.lat0, r.settings.map.lon0).el;
      mo = moonAt(r.model_ms, r.settings.map.lat0, r.settings.map.lon0);
    }
    const e = el * 180 / Math.PI, sm = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
    const dayK = sm(-9, 5, e);
    const moonDir = mo ? [Math.sin(mo.az) * Math.cos(mo.el), Math.cos(mo.az) * Math.cos(mo.el), Math.sin(mo.el)] : [0, 0, -1];
    const moonK = mo ? mo.lit * sm(-1, 8, mo.el * 180 / Math.PI) : 0;
    return { dayK, expo: 1 + 7 * (1 - dayK) * (1 - dayK), moonDir, moonK, lampK: sm(0.75, 0.3, dayK), el: e };
  }
  // the headlights: yours (L: auto, on, off; auto: on at dusk or in heavy rain) and the nearest others'
  function headlights(night) {
    const auto = night.dayK < 0.6 || wx.rainI > 0.5;
    const mine = st.lights === 'on' || (st.lights !== 'off' && auto);
    const out = [];
    const lamp = (p, M, len, on) => {
      const f = M[0], h = len / 2 + 0.1;
      out.push({ p: [p[0] + f[0] * h, p[1] + f[1] * h, p[2] + f[2] * h + 0.15], d: norm3([f[0], f[1], f[2] - 0.07]), on: on ? 1 : 0 });
    };
    const c = st.car;
    lamp(c.p, c.matrix, c.spec.dims[0], mine);
    const me = st.foot || c;
    const others = (st.lastOthers || []).map(o => ({ o, d: Math.hypot(o.p[0] - me.p[0], o.p[1] - me.p[1]) })).sort((a, b) => a.d - b.d).slice(0, 3);
    for (const { o } of others) if (o.q) lamp(o.p, CarPhys.qmat(o.q), (CarPhys.SPECS[o.kind] || CarPhys.SPECS.suv).dims[0], auto);
    return out;
  }
  const norm3 = v => { const l = Math.hypot(...v) || 1; return v.map(x => x / l); };
  function sunDir() {
    let az = 240 * Math.PI / 180, el = 35 * Math.PI / 180;
    const r = st.run;
    const vs = st.vis;
    if (vs.sunMode === 'manual' || !(r && r.model_ms && r.settings && r.settings.map)) { az = vs.sunAz * Math.PI / 180; el = vs.sunH * Math.PI / 180; }
    else {
      const s = sunAt(r.model_ms, r.settings.map.lat0, r.settings.map.lon0);
      az = s.az; el = Math.max(s.el, 0.035);
    }
    return [Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el)];
  }

  // ---------------------------------------------------------------- the run, the storm, the volume
  async function poll() {
    try {
      st.run = await getJSON('/api/state');
    } catch (e) { st.run = null; }
    await pollTargets();
  }
  // the tornadoes' and storms' positions (pmw.targets: small, and quick to make), more often than the whole state
  async function pollTargets() {
    if (st.targetsBusy) return;
    st.targetsBusy = true;
    try { st.targets = await getJSON('/api/pmw/targets'); } catch (e) { st.targets = null; }
    st.targetsBusy = false;
  }
  // a tornado's position on the screen glides from where it was drawn to each new position over about the time
  // between them, as the cloud blends between volumes, instead of jumping every update
  const torGlide = new Map();                         // id -> { fx, fy, tx, ty, t0, dur, seen }
  function glide(id, x, y) {
    const now = performance.now();
    let g = torGlide.get(id);
    if (!g) { g = { fx: x, fy: y, tx: x, ty: y, t0: now, dur: 0, at: now }; torGlide.set(id, g); }
    if (x !== g.tx || y !== g.ty) {
      const k = g.dur > 0 ? Math.min(1, (now - g.t0) / g.dur) : 1;
      g.fx += (g.tx - g.fx) * k; g.fy += (g.ty - g.fy) * k;          // from where it is drawn now
      // a jump of kilometres is a different vortex (or the first fix): no glide
      const far = Math.hypot(x - g.fx, y - g.fy) > 4000;
      g.dur = far ? 0 : Math.max(BLEND_MIN, Math.min(BLEND_MAX, now - g.at));
      if (far) { g.fx = x; g.fy = y; }
      g.tx = x; g.ty = y; g.t0 = now; g.at = now;
    }
    g.seen = now;
    const k = g.dur > 0 ? Math.min(1, (now - g.t0) / g.dur) : 1;
    return [g.fx + (g.tx - g.fx) * k, g.fy + (g.ty - g.fy) * k];
  }
  // the storm to chase: the strongest tornado, else the tornado zoom, else the strongest storm cell (world m),
  // with its motion
  function storm() {
    const tg = st.targets, r = st.run;
    const cells = (r && r.tracks || []).filter(c => c.ended === null || c.ended === undefined);
    const motionNear = (x, y) => {
      let b = null, bd = 3e4;
      for (const c of cells) { const d = Math.hypot(c.x - x, c.y - y); if (d < bd) { bd = d; b = c; } }
      return b ? [b.vx, b.vy] : [10, 5];
    };
    if (tg && tg.tornadoes && tg.tornadoes.length) {
      const t = tg.tornadoes.reduce((a, b) => ((b.wind || 0) > (a.wind || 0) ? b : a));
      return { x: t.x, y: t.y, v: motionNear(t.x, t.y), what: `tornado ${t.id}` };
    }
    if (tg && tg.zoom) return { x: tg.zoom.x, y: tg.zoom.y, v: motionNear(tg.zoom.x, tg.zoom.y), what: 'the tornado zoom' };
    if (cells.length) {
      const c = cells.reduce((a, b) => (b.w > a.w ? b : a));
      return { x: c.x, y: c.y, v: [c.vx, c.vy], what: `storm ${c.id}` };
    }
    if (tg && tg.patches && tg.patches.length) return { x: tg.patches[0].x, y: tg.patches[0].y, v: [10, 5], what: 'the storm patch' };
    return null;
  }
  // the tornadoes on the ground now: { id, x, y, wind, ef } (world m)
  function tornadoes() {
    const out = [];
    const r = st.run;
    for (const tr of (r && r.tornado_tracks) || []) {
      if (!tr.active || !tr.pts.length) continue;
      const p = tr.pts[tr.pts.length - 1];
      out.push({ id: tr.id, x: p[1], y: p[2], ef: p[3], wind: p[4], pts: tr.pts });
    }
    if (st.targets) for (const t of st.targets.tornadoes || []) {
      const o = out.find(o => o.id === t.id) || out.find(o => Math.hypot(o.x - t.x, o.y - t.y) < 3000);
      if (o) { o.x = t.x; o.y = t.y; if (t.wind) o.wind = t.wind; } else out.push({ id: t.id, x: t.x, y: t.y, wind: t.wind, ef: efOf(t.wind), pts: [] });
    }
    for (const o of out) [o.x, o.y] = glide(o.id, o.x, o.y);
    const now = performance.now();
    for (const [id, g] of torGlide) if (now - g.seen > 30000) torGlide.delete(id);
    return out;
  }
  // The model's air at the car (chase.py air(): its 10 m wind, gust, temperature, dew point), twice a second; between
  // answers the numbers glide.  st.fake overrides any of { wind [u, v] m/s, gust, t (deg C), rain (0..1 intensity),
  // tornado (0..1 roar) } for testing, from the console: Chase.st.fake = { wind: [30, 5], gust: 45, t: -3, rain: 0.8 }.
  const AIR_EVERY = 250, AIR_CALM = 1000;   // ms between asks: in a storm's patch, outside them
  const air = { busy: false, now: null, err: '', grid: null, at: 0 };
  async function pollAir() {
    const c = st.foot || st.car;                // the air where you are: on foot, round you
    if (!c || air.busy) return;
    if (!(air.now && air.now.inside) && performance.now() - air.at < AIR_CALM) return;
    air.busy = true; air.at = performance.now();
    try {
      const d = await getJSON(`/api/c/air?grid=1&p=${c.p[0].toFixed(1)},${c.p[1].toFixed(1)}`);
      air.now = d.air && d.air[0] ? d.air[0] : air.now; air.err = '';
      air.grid = air.now && air.now.grid ? air.now.grid : null;
      if (d.air && d.air[0]) Mesonet.record(d.air[0], d.t, c.p[0], c.p[1], wx.rainI);     // the mesonet probe's log
    } catch (e) { air.err = e.message; }
    air.busy = false;
  }

  // what the car is in now: { wind [u, v], W m/s, gust, t, td, rainI 0..1, snow 0..1, roar 0..1 } (smoothed per frame)
  const wx = { u: 0, v: 0, gust: 0, t: null, td: null, rainI: 0, snow: 0, roar: 0, camV: [0, 0, 0], eye: null };
  function weather(dt, view) {
    const f = st.fake || {}, a = air.now, k = Math.min(1, dt * 2.5);
    const u = f.wind ? f.wind[0] : a ? a.u : 0, v = f.wind ? f.wind[1] : a ? a.v : 0;
    const W = Math.hypot(u, v), gust = f.gust !== undefined ? f.gust : a ? a.gust : W;
    wx.u += (u - wx.u) * k; wx.v += (v - wx.v) * k; wx.gust += (Math.max(gust, W) - wx.gust) * k;
    const T = f.t !== undefined ? f.t : a ? a.t : null;
    wx.t = T === null ? null : wx.t === null ? T : wx.t + (T - wx.t) * k;
    wx.td = f.td !== undefined ? f.td : a ? a.td : null;
    // the model's rain at the car (g/kg a few hundred metres up) as an intensity: 0.03 g/kg nothing, 1 heavy, 3 extreme
    const ri = f.rain !== undefined ? f.rain : st.rain > 0.03 ? Math.min(1, (Math.log10(st.rain) + 1.5) / 2) : 0;
    wx.rainI += (ri - wx.rainI) * Math.min(1, dt * 0.8);
    // snow below freezing, sleet and wet flakes just above, rain from 2 deg C (the wet bulb would be better)
    const sn = wx.t === null ? 0 : Math.max(0, Math.min(1, (1.8 - wx.t) / 2.3));
    wx.snow += (sn - wx.snow) * Math.min(1, dt * 0.5);
    // a tornado's roar: closer than ~5 km, louder the stronger and nearer it is
    let roar = 0;
    const c = st.car;
    for (const t of tornadoes()) {
      const d = Math.hypot(t.x - c.p[0], t.y - c.p[1]), V = t.wind || [32, 43, 55, 67, 80, 95][Math.max(0, t.ef || 0)];
      roar = Math.max(roar, Math.max(0, Math.min(1, (5000 - d) / 4600)) ** 1.5 * Math.min(1, Math.max(0.3, V / 70)));
    }
    if (f.tornado !== undefined) roar = f.tornado;
    wx.roar += (roar - wx.roar) * k;
    // the camera's own motion, for the streaks
    if (wx.eye && dt > 0) {
      const cv = view.eye.map((x, i) => (x - wx.eye[i]) / dt);
      wx.camV = wx.camV.map((x, i) => x + (Math.max(-80, Math.min(80, cv[i])) - x) * Math.min(1, dt * 8));
    }
    wx.eye = view.eye.slice();
    return wx;
  }
  // ---------------------------------------------------------------- the wind on the car (and in the trees)
  // The wind at a point (world m), the model's alone: its 10 m wind on the finest grid holding the car (chase.py
  // air(grid): the tornado zoom's 50 m nest where it is), read bilinearly every physics step from the grid round the car
  // (refreshed four times a second in a storm), else its 10 m wind at the car (the environment's, outside the storms).
  // No gusts made up on top: the nest's own turbulence is in its winds.  st.fake.wind overrides it for testing.
  function windAt(x, y) {
    const g = air.grid;
    if (g && !(st.fake && st.fake.wind)) {
      const fx = (x - g.x0) / g.dx, fy = (y - g.y0) / g.dx;
      if (fx >= 0 && fy >= 0 && fx <= g.nx - 1 && fy <= g.ny - 1) {
        const i0 = Math.min(g.nx - 2, Math.floor(fx)), j0 = Math.min(g.ny - 2, Math.floor(fy)), a = fx - i0, b = fy - j0;
        const at = (A, i, j) => A[j * g.nx + i];
        const bl = A => (at(A, i0, j0) * (1 - a) + at(A, i0 + 1, j0) * a) * (1 - b) + (at(A, i0, j0 + 1) * (1 - a) + at(A, i0 + 1, j0 + 1) * a) * b;
        if (g.nx >= 2 && g.ny >= 2) return [bl(g.u), bl(g.v), 0];
      }
    }
    return [wx.u, wx.v, 0];
  }
  // the trees' wind (render.js, world.js): the model's air at the camera
  function treeWind() {
    const W = Math.hypot(wx.u, wx.v) || 1;
    return { u: wx.u, v: wx.v, gust: Math.max(1, wx.gust / W) };
  }

  // the rain and snow for Render.frame, and the sound
  function precipView(view) {
    const I = wx.rainI;
    if (I < 0.01) return null;
    const sun = sunDir(), inside = !st.foot && st.cam.mode !== 'chase';
    const nk = st.night ? st.night.dayK : 1, fl = Lightning.light(performance.now(), view.eye).amb[2];
    const L = (0.32 + 0.5 * Math.min(1, Math.max(0, sun[2]) * 2.2)) * (1 - 0.35 * I) * Math.max(0.06, nk) + 0.3 * Math.min(1, fl);
    return { n: I * (9000 * (1 - wx.snow) + 15000 * wx.snow), snow: wx.snow, cube: wx.snow > 0.5 ? 20 : 24,
             minD: inside ? 1.4 : 0.4, wind: [wx.u, wx.v], camV: wx.camV,
             col: [0.25 + 0.62 * L, 0.27 + 0.65 * L, 0.30 + 0.68 * L] };     // drops catch the light: brighter than the murk
  }
  function sound(dt) {
    const c = st.car, cyl = { suv: 6, sedan: 4, truck: 8 }[c.kind] || 6;
    const slip = Math.max(0, ...c.wheels.map(w => (w.contact ? w.slip || 0 : 0)));
    const rough = Math.max(0, ...c.wheels.map(w => (w.surf && w.surf.rough ? w.surf.rough * 12 : 0)));
    Sound.update({ rpm: c.rpm, idle: c.spec.idle, redline: c.spec.redline, thr: (c.in && c.in.thr) || 0, cyl, speed: c.speed, slip,
                   rough: Math.min(1, rough), wind: Math.hypot(wx.u, wx.v), gust: wx.gust, tornado: wx.roar, rain: wx.rainI,
                   snow: wx.snow, inside: !st.foot && st.cam.mode !== 'chase', vol: st.vis.vol === undefined ? 0.7 : st.vis.vol, dt });
  }

  const efOf = w => (w === null || w === undefined ? null : w < 38 ? 0 : w < 50 ? 1 : w < 61 ? 2 : w < 74 ? 3 : w < 89 ? 4 : 5);

  async function loadVolume(force) {
    if (st.volBusy || !st.car) return;
    const x = st.car.p[0] / 1000, y = st.car.p[1] / 1000;
    let b = st.box;
    const far = !b || Math.hypot(x - (b[0] + b[2]) / 2, y - (b[1] + b[3]) / 2) > RECENTRE_KM;
    if (!far && !force && performance.now() - st.volAt < (st.volGap || VOL_MIN)) return;
    if (far) { const h = BOX_KM / 2, cx = Math.round(x), cy = Math.round(y); b = [cx - h, cy - h, cx + h, cy + h]; }
    const t0 = performance.now();
    st.volBusy = true; st.volAt = t0;
    // wind=1: the model's wind and vortices too, so the blend carries the cloud along them (render.js MIX_FS)
    const q = `/api/volume?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&field=cloud&wind=1`;
    // the same box as the one shown: blend into the new volume over the time volumes take to come in (smoothed, and
    // a little longer), so the picture plays about one volume behind the newest and never stops between them
    // (render.js carries it on along the wind if the next is late)
    const same = !far && !!st.box && Render.head();
    const blendMs = () => {
      if (!same || !st.volNewAt) return 0;
      const gap = performance.now() - st.volNewAt;
      st.volIv = st.volIv ? st.volIv + (gap - st.volIv) * 0.35 : gap;
      return Math.max(BLEND_MIN, Math.min(BLEND_MAX, 1.15 * st.volIv));
    };
    try {
      let r = null;
      if (st.run && st.run.status === 'running') {
        try {
          // since=: the server answers 204 while the model has not moved on from the volume shown
          const since = same && st.volModelT !== null && st.volModelT !== undefined ? `&since=${st.volModelT}` : '';
          r = await api(q + since);
          st.volT = null;
        } catch (e) { r = null; }
        if (r && r.status === 204) {
          st.volGap = Math.max(VOL_MIN, Math.min(VOL_MAX, VOL_LOAD * (performance.now() - t0)));
          st.volBusy = false;
          return;
        }
        if (r) {
          const mt = parseFloat(r.headers.get('X-Model-Time'));
          st.volModelT = Number.isFinite(mt) ? mt : null;
        }
      }
      if (!r) {
        st.volModelT = null;
        // a stopped run (or none of its fields in the box yet): the newest 3-D snapshot it saved
        const tt = await getJSON('/api/volume/times');
        if (!tt.times || !tt.times.length) throw new Error('no 3-D snapshots saved in this run');
        const t = tt.times[tt.times.length - 1];
        if (!far && !force && st.volT === t && st.box) { st.volBusy = false; return; }
        r = await api(`${q}&t=${t}`);
        st.volT = t;
      }
      const buf = await r.arrayBuffer();
      Render.setVolume(b, buf, blendMs());
      st.volNewAt = performance.now();
      st.box = b; st.volErr = ''; st.noStorm = false;
    } catch (e) {
      st.volErr = st.noStorm && !far ? '' : e.message;
      // no storm in the box (none in the run yet, or far from them): an empty sky over it, so the ground and the car
      // are drawn anyway; the storm comes in when a fetch first finds it
      if (!Render.head() || far) { Render.setVolume(b, emptyVolume(b), 0); st.box = b; st.volErr = ''; st.noStorm = true; }
    }
    st.volGap = Math.max(VOL_MIN, Math.min(VOL_MAX, VOL_LOAD * (performance.now() - t0)));
    st.volBusy = false;
  }

  // ---------------------------------------------------------------- the far cloud (render.js setFar)
  // Past the box: two coarse volumes round the car out to where the haze has it all (3x the visibility, FAR_MAX_KM
  // at most) -- the nearer out to FAR_NEAR_KM on a ~1.25 km grid, the wider beyond on a coarser one -- fetched one at
  // a time every FAR_EVERY ms each, re-centred when the car has gone an eighth of one's width, each blended into
  // over the time to the next
  const FAR_EVERY = [15000, 30000], FAR_NEAR_KM = 160, FAR_MAX_KM = 600;
  const FAR_BLEND = 3000;                      // ms: a far storm barely moves in 15 s; a short blend, then no more mixing
  const farSt = [{ at: 0, box: null }, { at: 0, box: null }];
  let farBusy = false;
  function farTiers() {
    const R = Math.min(FAR_MAX_KM, 3 * (st.vis.haze || 180));
    // grids sized for a laptop's GPU too (the light pass and the blend run over every point): ~1.7 km and ~4-7 km cells
    return [R > BOX_KM / 2 + 5 ? { half: Math.min(R, FAR_NEAR_KM), side: 192, nz: 36 } : null,
            R > FAR_NEAR_KM + 10 ? { half: R, side: 160, nz: 24 } : null];
  }
  async function loadFar() {
    if (farBusy || !st.car || !st.run) return;
    const on = st.vis.far !== 'off', tiers = farTiers(), x = st.car.p[0] / 1000, y = st.car.p[1] / 1000;
    for (let i = 0; i < 2; i++) {
      const T = tiers[i], s = farSt[i];
      if (!on || !T) {
        if (s.box) { Render.setFar(i, null); s.box = null; }
        continue;
      }
      const step = T.half / 8, cx = Math.round(x / step) * step, cy = Math.round(y / step) * step;
      const b = [cx - T.half, cy - T.half, cx + T.half, cy + T.half];
      const moved = !s.box || b.some((v, k) => Math.abs(v - s.box[k]) > 1e-6);
      if (!moved && performance.now() - s.at < FAR_EVERY[i]) continue;
      farBusy = true;
      try {
        let q = `/api/volume?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&field=cloud&far=1&side=${T.side}&nz=${T.nz}`;
        if (st.run.status !== 'running') {
          // a stopped run: its newest 3-D snapshot, as the box's
          if (st.volT === null || st.volT === undefined) throw new Error('no snapshot');
          q += `&t=${st.volT}`;
        }
        const r = await api(q);
        Render.setFar(i, await r.arrayBuffer(), s.box ? FAR_BLEND : 0);
      } catch (e) {
        Render.setFar(i, null);                 // no storm patch out there (or none yet)
      }
      s.box = b; s.at = performance.now();
      farBusy = false;
      return;                                   // one a call
    }
  }

  // a cloud volume with nothing in it over box b (km), in the server's format (volume.py extract)
  function emptyVolume(b) {
    const n = 4, head = new TextEncoder().encode(JSON.stringify({
      field: 'cloud', units: 'log10 g/kg', nx: n, ny: n, nz: n, box: b.map(v => v * 1000), ztop: 18000, scale: 40, offset: 121,
      channels: 2, ground: false, insets: [] }));
    const buf = new Uint8Array(4 + head.length + n * n * n * 2);
    new DataView(buf.buffer).setUint32(0, head.length, true);
    buf.set(head, 4);
    return buf.buffer;
  }

  // ---------------------------------------------------------------- the car
  function place(x, y, yawHint) {
    const n = Roads.nearest(x, y, 6, 3000) || Roads.nearest(x, y, 7, 15000);
    let px = x, py = y, yaw = yawHint || 0;
    if (n) {
      px = n.x; py = n.y; yaw = n.yaw;
      const d = Math.atan2(Math.sin(yawHint - yaw), Math.cos(yawHint - yaw));
      if (Math.abs(d) > Math.PI / 2) yaw += Math.PI;
      // keep right: half a lane off the middle on two-way roads
      if (n.cls >= 2) { const o = Roads.HALF[n.cls] * 0.45; px += Math.sin(yaw) * o; py -= Math.cos(yaw) * o; }
    }
    st.car.reset(px, py, yaw);
    st.cam.yaw = yaw; st.cam.eye = null;
    return !!n;
  }

  async function start() {
    st.name = ($('name').value || 'Chaser').slice(0, 20);
    st.kind = $('vehicle').value;
    st.color = hexRgb($('color').value);
    try { localStorage.setItem('chase.name', st.name); localStorage.setItem('chase.kind', st.kind); localStorage.setItem('chase.color', $('color').value); } catch (e) { /* private */ }
    $('go').disabled = true; $('startMsg').textContent = Net.local ? '' : 'Joining…';
    try { Sound.start(); } catch (e) { console.warn('chase sound:', e); }       // on the click: browsers want a gesture
    try { await Net.join(st.name, st.kind, st.color, $('code').value.trim()); }
    catch (e) {
      // this PC on a viewer older than chase multiplayer: drive alone
      if (Net.local && /^404|^unknown/.test(e.message)) Net.st.err = 'restart the viewer for multiplayer and buildings';
      else { $('go').disabled = false; $('startMsg').textContent = `Could not join: ${e.message}`; return; }
    }
    if (Net.local) Net.hostStatus();
    $('start').classList.add('hidden');
    st.started = true;
    await poll();
    if (!st.run || st.run.status === 'none' || !st.run.run_id) { say('No run: start one in the viewer first.', 1e9); return; }
    // no storm yet: a drive anyway, from the middle of the run's map (or a spot picked on it)
    const s = storm() || { x: 0, y: 0, v: [10, 5], what: 'the open road', none: true };
    // where to start: ahead of the storm on its inflow side (12 km, or 4 km: close), or a spot picked on the map
    const mode = $('spawn').value;
    try { localStorage.setItem('chase.spawn', mode); } catch (e) { /* private */ }
    let [sx, sy] = s.none ? [0, 0] : startNear(s, mode === 'close' ? 4000 : 12000);
    if (mode === 'pick') {
      const p = await openPicker('Click where you want to start · wheel: zoom · drag: pan');
      if (Array.isArray(p)) [sx, sy] = p;
    }
    st.car = new CarPhys.Car(st.kind, ground);
    st.car.solid = (x, y, z) => Stations.solid(x, y, z) || Scenery.solid(x, y, z);
    st.car.windAt = windAt;     // the model's wind at the car, every physics step
    st.extraCars = remoteCars; st.extraMap = remoteMap;
    st.car.reset(sx, sy, 0);
    st.spawnHint = { x: sx, y: sy, yaw: Math.atan2(s.y - sy, s.x - sx), what: s.what };
    say(s.none ? 'No storm in the run yet: a drive anyway (the storm comes in when one forms)…' : `Heading for ${s.what}: loading the storm and the roads…`, 4000);
    await loadVolume(true);
    if (!Render.head()) { say(`No cloud volume: ${st.volErr || 'no storm fields yet'}`, 1e9); }
    st.last = performance.now();
    requestAnimationFrame(tick);
    setInterval(poll, 4000);
    setInterval(pollTargets, TARGETS_EVERY);
    setInterval(pollAir, AIR_EVERY);
    // the streaming (network) on its own clock, not the frames'
    const stream = () => {
      const c = st.car;
      Roads.update(c.p[0], c.p[1], Render.terrain, () => Render.setRoads(Roads.state.mesh, Roads.state.origin));
      loadVolume(false);
      Scenery.updateBuildings(c.p[0], c.p[1], Render.terrain, () => Render.setBuildings(Scenery.state.bMesh, Scenery.state.bOrigin));
      Stations.update(c.p[0], c.p[1]);
      Lightning.poll(c.p[0], c.p[1]);
      Damage.poll(c.p[0], c.p[1]);
      const wl = detailed() && WorldLink.ready();
      if (Stations.near(c.p[0], c.p[1], stationGround, wl ? 'world' : 'basic')) Scenery.setSkip(Stations.covers);
      const me = st.foot || c;
      if (Scenery.damageUpdate(me.p[0], me.p[1])) Render.setBuildings(Scenery.state.bMesh, Scenery.state.bOrigin);
      if (wl) WorldLink.damage(me.p[0], me.p[1]);
      if (Lights.update(me.p[0], me.p[1], (x, y) => surfAt(x, y).h)) Render.setLamps(Lights.state.inst, Lights.state.origin);
      Stations.follow(me.p[0], me.p[1]);
      if (Stations.route.arrived && !st.arriveSaid) {
        st.arriveSaid = true;
        const d = Stations.route.dest;
        say(`Arrived at ${d ? d.name : 'the gas station'}${st.foot ? '' : ' · F to get out and walk into the store'}`, 6000);
        Stations.clear();
      }
    };
    stream();
    setInterval(stream, 500);
    loadFar();
    setInterval(loadFar, 1000);
    // the car to the others on its own clock: it keeps going while this tab is in the background
    setInterval(() => {
      if (!st.car) return;
      const c = st.car, hg = c.p[2] - surfAt(c.p[0], c.p[1]).h, f = st.foot;
      const r2 = v => Math.round(v * 100) / 100;
      const fs = f ? [f.p[0], f.p[1], f.p[2] - surfAt(f.p[0], f.p[1]).h, f.yaw, f.phase % 1000, f.gait, f.down, f.lean, f.leanDir || 0, f.air ? 1 : 0, f.speed].map(r2) : null;
      Net.tick(c, { name: st.name, kind: st.kind, color: st.color, hg, f: fs });
    }, 40);
  }

  // the ground a gas station is built on (stations.js): the detailed map's, or the model's terrain (NaN: not loaded)
  function stationGround(x, y) {
    if (detailed() && WorldLink.ready()) { const g = WorldLink.ground(x, y); return g ? g.h : NaN; }
    return Render.terrain(x, y);
  }
  // the dust the model's wind raises round you: past ~8 m/s off bare fields, crops and gravel (less off grass and
  // lawns, little off woods, none off wet ground or water); and in a tornado's or a downburst's winds off anything
  const DUSTY = { bare: 1, crops: 0.85, gravel: 0.8, verge: 0.4, grass: 0.35, town: 0.45, asphalt: 0.25, wet: 0.05, wood: 0.12, marsh: 0.05, water: 0, mud: 0.05 };
  function dustView(view, night) {
    const me = st.foot || st.car, sf = surfAt(me.p[0], me.p[1]);
    const name = Object.keys(SURF).find(k => SURF[k] === sf.s) || 'grass';
    const W = Math.hypot(wx.u, wx.v), dry = 1 - 0.85 * Math.min(1, st.wet * 1.2 + wx.rainI);
    const lift = Math.max(0, Math.min(1, (W - 8) / 16)) * (DUSTY[name] === undefined ? 0.3 : DUSTY[name]) * dry;
    const tor = Math.max(0, Math.min(1, (wx.gust - 38) / 30));
    const k = Math.max(lift, tor);
    if (k < 0.01) return null;
    const L = 0.18 + 0.72 * night.dayK + Math.min(0.5, (Lightning.light(performance.now(), view.eye).amb[0]) * 0.3);
    return { k, gz: sf.h, wind: [wx.u, wx.v], col: [0.52 * L, 0.45 * L, 0.35 * L] };
  }
  // O: the way to the nearest gas station, on and off
  function toggleRoute() {
    const on = Stations.toggle();
    st.arriveSaid = false;
    if (on) { const c = st.foot || st.car; Stations.follow(c.p[0], c.p[1]); }
    say(on ? (Stations.route.pts ? `Route to ${Stations.route.dest.name}: ${fmtDist(Stations.route.len)}` : 'Finding the nearest gas station…') : 'Route off', 2500);
  }
  const fmtDist = m => (st.units === 'mph' ? (m < 400 ? `${Math.round(m * 3.281 / 10) * 10} ft` : `${(m / 1609.34).toFixed(1)} mi`) : (m < 1000 ? `${Math.round(m / 10) * 10} m` : `${(m / 1000).toFixed(1)} km`));

  // ---------------------------------------------------------------- on foot (walker.js)
  // F: out of the car by the driver's door (stopped or nearly), and back in when close to it
  function toggleFoot() {
    const c = st.car;
    if (!c || !st.spawned) return;
    if (!st.foot) {
      if (c.speed > 2.5) { say('Stop the car to get out', 1500); return; }
      const S = c.spec, a = S.cgFront, xF = a + (S.dims[0] - S.wheelbase) / 2;
      const door = CarPhys.toWorld(c.matrix, [xF - 2.3, S.dims[1] / 2 + 0.6, 0]).map((v, i) => v + c.p[i]);
      st.foot = new Walker.Person(door[0], door[1], ground(door[0], door[1]).h, c.yaw + Math.PI / 2);
      st.footCtl = { dir: [0, 0], run: false, jump: false };
      Object.assign(st.cam, { yaw: c.yaw, look: 0, pitch: 0, eye: null });
      say('On foot: W A S D walk · Shift run · Space jump · drag or ← → to look · F back in the car', 6000);
    } else {
      const f = st.foot, d = Math.hypot(f.p[0] - c.p[0], f.p[1] - c.p[1]);
      if (d > c.spec.dims[0] / 2 + 2) { say('Walk back to the car to get in', 1500); return; }
      st.foot = null;
      Object.assign(st.cam, { yaw: c.yaw, look: 0, pitch: 0, eye: null });
      say('Back in the car', 1200);
    }
  }
  // the keys (and a pad's left stick) on foot: where to go, relative to the camera
  function footInput(dt) {
    const k = st.keys, cam = st.cam;
    let fwd = ((k.has('KeyW') || k.has('ArrowUp')) ? 1 : 0) - ((k.has('KeyS') || k.has('ArrowDown')) ? 1 : 0);
    let side = (k.has('KeyA') ? 1 : 0) - (k.has('KeyD') ? 1 : 0);
    cam.yaw += (((k.has('ArrowLeft')) ? 1 : 0) - (k.has('ArrowRight') ? 1 : 0)) * 1.8 * dt;
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const g of pads) {
      if (!g || !g.connected) continue;
      const ax = g.axes[0] || 0, ay = g.axes[1] || 0;
      if (Math.abs(ax) > 0.15) side = -ax;
      if (Math.abs(ay) > 0.15) fwd = -ay;
      const lx = g.axes[2] || 0;
      if (Math.abs(lx) > 0.15) cam.yaw -= lx * 2 * dt;
      break;
    }
    const yaw = cam.yaw + cam.look, l = Math.hypot(fwd, side) || 1, m = Math.min(1, Math.hypot(fwd, side));
    const dir = [(Math.cos(yaw) * fwd - Math.sin(yaw) * side) / l * m, (Math.sin(yaw) * fwd + Math.cos(yaw) * side) / l * m];
    const jump = k.has('Space') && !st.jumpHeld;
    st.jumpHeld = k.has('Space');
    st.footCtl = { dir, run: k.has('ShiftLeft') || k.has('ShiftRight'), jump };
    st.car.control(0, 0, 0, true);                    // the car left in park: the handbrake on
  }
  // the wind on you: the model's at your spot, gusting between its mean and its gusts
  function footWind(now) {
    const W = Math.hypot(wx.u, wx.v) || 1, g = Math.max(1, wx.gust / W);
    const t = now / 1000, k = 1 + (g - 1) * Math.max(0, Math.sin(t * 0.9) * Math.sin(t * 2.3 + 1) + 0.2 * Math.sin(t * 5.1));
    return [wx.u * k, wx.v * k];
  }
  // a person against walls and trunks: the detailed map's, or chase mode's own buildings
  function footPush(x, y, rad) {
    const z = st.foot ? st.foot.p[2] + 0.6 : 0;
    const sp = Stations.push(x, y, z, rad);
    if (sp) return sp;
    if (Stations.covers(x, y)) return null;
    if (detailed() && WorldLink.ready()) return WorldLink.pushOut(x, y, rad);
    let s = null;
    for (const [ox, oy] of [[rad, 0], [-rad, 0], [0, rad], [0, -rad], [0, 0]]) {
      s = Scenery.solid(x + ox, y + oy, z);
      if (s) return [s.n[0] * (s.d + 0.02), s.n[1] * (s.d + 0.02)];
    }
    return null;
  }
  function footCamera(dt) {
    const f = st.foot, cam = st.cam;
    // a drag turns the camera for good once let go (on foot the view doesn't swing back behind anything)
    if (!cam.lookHeld && cam.look) { cam.yaw += cam.look; cam.look = 0; }
    const yaw = cam.yaw + cam.look, pitch = cam.pitch;
    if (cam.mode !== 'chase') {
      const eye = [f.p[0], f.p[1], f.p[2] + (f.down > 0 && !f.air ? 0.35 : 1.65)];
      const dir = [Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch)];
      cam.eye = eye;
      return { eye, target: eye.map((v, i) => v + dir[i] * 10), up: [0, 0, 1], fov: 1.2 };
    }
    const dist = 3.6 * cam.dist, hgt = 1.2 + 0.5 * cam.dist;
    const tgt = [f.p[0], f.p[1], f.p[2] + 1.45];
    let eye = [f.p[0] - Math.cos(yaw) * dist * Math.cos(pitch), f.p[1] - Math.sin(yaw) * dist * Math.cos(pitch), f.p[2] + hgt + dist * Math.sin(pitch)];
    if (cam.eye) { const k = 1 - Math.exp(-dt * 14); eye = cam.eye.map((v, i) => v + (eye[i] - v) * k); }
    const gz = surfAt(eye[0], eye[1]).h + 0.3;
    if (eye[2] < gz) eye[2] = gz;
    cam.eye = eye;
    return { eye, target: tgt, up: [0, 0, 1], fov: 1.1 };
  }

  // ---------------------------------------------------------------- input
  function input(dt) {
    if (st.foot) { footInput(dt); return; }
    const k = st.keys;
    let acc = (k.has('KeyW') || k.has('ArrowUp')) ? 1 : 0, brk = (k.has('KeyS') || k.has('ArrowDown')) ? 1 : 0;
    let steer = ((k.has('KeyA') || k.has('ArrowLeft')) ? 1 : 0) - ((k.has('KeyD') || k.has('ArrowRight')) ? 1 : 0);
    let hand = k.has('Space');
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const g of pads) {
      if (!g || !g.connected) continue;
      const ax = g.axes[0] || 0, dz = Math.abs(ax) < 0.08 ? 0 : (Math.abs(ax) - 0.08) / 0.92;
      if (dz) steer = -Math.sign(ax) * dz * dz;
      if (g.buttons[7] && g.buttons[7].value > 0.02) acc = Math.max(acc, g.buttons[7].value);
      if (g.buttons[6] && g.buttons[6].value > 0.02) brk = Math.max(brk, g.buttons[6].value);
      if (g.buttons[0] && g.buttons[0].pressed) hand = true;
      const camB = g.buttons[3] && g.buttons[3].pressed, rst = g.buttons[2] && g.buttons[2].pressed;
      if (camB && !st.padCam) cycleCam();
      if (rst && !st.padRst) resetCar();
      st.padCam = camB; st.padRst = rst;
      const lx = g.axes[2] || 0, ly = g.axes[3] || 0;
      if (Math.abs(lx) > 0.15 || Math.abs(ly) > 0.15) { st.cam.look = -lx * 2.2; st.cam.pitch = -ly * 0.6; st.cam.lookT = performance.now(); }
      break;
    }
    st.car.control(acc, brk, steer, hand);
  }
  function cycleCam() { st.cam.mode = CAMS[(CAMS.indexOf(st.cam.mode) + 1) % CAMS.length]; st.cam.look = 0; st.cam.pitch = 0; st.cam.eye = null; }
  function resetCar() {
    const c = st.car;
    if (!c) return;
    if (place(c.p[0], c.p[1], c.yaw)) { st.offroadStart = null; say('Back on the road', 1500); }
    else say(Roads.state.pending ? 'No road loaded near here yet: set upright, the roads are still coming in'
                                 : 'No road or track within 15 km: set upright where you are', 3500);
  }

  // ---------------------------------------------------------------- the camera
  const angd = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
  function camera(dt) {
    if (st.foot) return footCamera(dt);
    const c = st.car, S = c.spec, M = c.matrix, cam = st.cam;
    if (!cam.lookHeld && performance.now() - cam.lookT > 1200) {
      const f = 1 - Math.exp(-dt * 3);
      cam.look -= cam.look * f; cam.pitch -= cam.pitch * f;
    }
    if (cam.mode === 'chase') {
      const vf = c.forwardSpeed;
      const head = c.speed > 4 && vf > 0 ? Math.atan2(c.v[1], c.v[0]) : c.yaw;
      cam.yaw += angd(head, cam.yaw) * (1 - Math.exp(-dt * 3.5));
      const dist = (S.dims[0] * 1.35 + 2.2) * cam.dist, hgt = S.dims[2] * 0.85 + 1.1 * cam.dist;
      const yaw = cam.yaw + cam.look, pitch = cam.pitch;
      const tgt = [c.p[0] + Math.cos(c.yaw) * 1.5, c.p[1] + Math.sin(c.yaw) * 1.5, c.p[2] + S.dims[2] * 0.55];
      let eye = [c.p[0] - Math.cos(yaw) * dist * Math.cos(pitch), c.p[1] - Math.sin(yaw) * dist * Math.cos(pitch), c.p[2] + hgt + dist * Math.sin(pitch)];
      if (cam.eye) { const f = 1 - Math.exp(-dt * 12); eye = cam.eye.map((v, i) => v + (eye[i] - v) * f); }
      const gz = surfAt(eye[0], eye[1]).h + 0.5;
      if (eye[2] < gz) eye[2] = gz;
      cam.eye = eye;
      return { eye, target: tgt, up: [0, 0, 1], fov: 1.05 };
    }
    // hood and driver: fixed in the car, looking along it (turned by the mouse / right stick), rolling with it
    const a = S.cgFront, b = S.wheelbase - a, xF = a + (S.dims[0] - S.wheelbase) / 2;
    const loc = cam.mode === 'hood' ? [xF - 0.35, 0, S.ground + (c.kind === 'sedan' ? 0.72 : c.kind === 'truck' ? 1.2 : 0.98) - S.cgH]
      : [xF - (c.kind === 'truck' ? 2.0 : c.kind === 'sedan' ? 2.25 : 2.05), S.dims[1] * 0.2, S.ground + (c.kind === 'sedan' ? 0.95 : c.kind === 'truck' ? 1.55 : 1.28) - S.cgH];
    const eye = CarPhys.toWorld(M, loc).map((v, i) => v + c.p[i]);
    const yaw = cam.look, pitch = cam.pitch - (cam.mode === 'driver' ? 0.05 : 0.02);
    const dirB = [Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch)];
    const dir = CarPhys.toWorld(M, dirB);
    return { eye, target: eye.map((v, i) => v + dir[i] * 10), up: M[2], fov: cam.mode === 'driver' ? 1.2 : 1.1 };
  }

  // ---------------------------------------------------------------- the frame
  function tick(now) {
    const dt = Math.min(0.1, (now - st.last) / 1000);
    st.last = now;
    st.fps += ((dt > 0 ? 1 / dt : 60) - st.fps) * 0.05;
    const c = st.car;
    // (the roads fetched round where the car was dropped, not the ones from before a jump)
    const h0 = st.spawnHint, rc = Roads.state.centre;
    if (!st.spawned && Roads.state.segs && Roads.state.segs.n && rc && h0 && Math.hypot(rc[0] - h0.x, rc[1] - h0.y) < 3000) {
      const h = st.spawnHint;
      if (place(h.x, h.y, h.yaw)) { st.spawned = true; say(`On the road, ${(Math.hypot(c.p[0] - h.x, c.p[1] - h.y) / 1000).toFixed(1)} km from where you were dropped. Chase ${h.what}!`, 5000); }
    }
    if (!st.spawned && !st.dropT) st.dropT = now;
    if (!st.spawned && now - st.dropT > 45000) {
      st.spawned = true; st.offroadStart = [c.p[0], c.p[1]];
      say('The roads are still loading: off-road for now (you go onto a road as soon as one arrives, or press R).', 6000);
    }
    // started off-road: onto the first road that arrives, unless the car has already driven off somewhere
    if (st.offroadStart && Roads.state.segs && Roads.state.segs.n) {
      if (Math.hypot(c.p[0] - st.offroadStart[0], c.p[1] - st.offroadStart[1]) < 150 && place(c.p[0], c.p[1], c.yaw)) say('On the road now', 2500);
      st.offroadStart = null;
    }
    input(dt);
    // the model's rain at the car: a wet road (the lowest levels' precipitation)
    const pr = Render.sample ? Render.sample(c.p[0], c.p[1]) : 0;
    st.rain += (pr - st.rain) * Math.min(1, dt * 0.5);
    st.wet = Math.max(st.wet - dt / 600, Math.min(1, st.rain / 0.6));
    if (st.spawned) c.update(dt);
    // rolled over (on its side or roof, stopped): R puts it back on its wheels on the road
    if (st.spawned && !st.foot && c.upright < 0.35 && c.speed < 3) {
      st.rolledT = (st.rolledT || 0) + dt;
      if (st.rolledT > 2 && !st.rolledSaid) { st.rolledSaid = true; say('Rolled over by the wind! Press R to get back on your wheels', 6000); }
    } else if (c.upright > 0.8) { st.rolledT = 0; st.rolledSaid = false; }
    if (!isFinite(c.p[0]) || c.p[2] < -50) resetCar();
    // on foot: you, on the same ground, in the model's wind, against walls, trunks and the car
    if (st.foot) {
      st.foot.update(dt, st.footCtl, { ground, push: footPush, wind: footWind(now), car: c });
      if (!isFinite(st.foot.p[0]) || st.foot.p[2] < -50) { st.foot = null; say('Back in the car', 1500); }
    }
    // the detailed map: loaded the first time it is wanted, streamed in round the car (round you, on foot), what the
    // car runs into
    if (detailed() && st.run && st.run.proj) {
      WorldLink.start(Render.gl(), st.run.proj, c.p[0], c.p[1]);
      WorldLink.update(st.foot || c, Render.terrain);
      if (st.spawned) WorldLink.collide(c);
    }
    if (Scenery.updateTrees(c.p[0], c.p[1], Render.terrain, TREE_BUDGET[st.vis.trees] || 0)) {
      const ss = Scenery.state;
      Render.setTrees(ss.trees, ss.treeN, ss.treeNear, ss.treeOrigin);
    }
    if (st.box) Land.want(st.box, st.vis.base, { x: c.p[0] / 1000, y: c.p[1] / 1000 }, (x, y) => Render.terrain(x * 1000, y * 1000) / 1000);
    const view = camera(dt);
    const q = st.vis;
    weather(dt, view);
    if (st.fake && st.fake.rain !== undefined) st.wet = Math.max(st.wet, Math.min(1, st.fake.rain * 1.5));
    sound(dt);
    const cars = [{ kind: c.kind, color: st.color, p: c.p, M: c.matrix, wheels: c.wheelPoses(), hidden: false }];
    if (st.foot && !(st.cam.mode !== 'chase')) cars.push({ parts: st.foot.figure(st.color) });   // (first person: not yourself)
    if (st.extraCars) cars.push(...st.extraCars());
    {
      const sp = Stations.parts().concat(Lights.parts()), rb = Stations.ribbon((x, y) => surfAt(x, y).h);
      if (rb) sp.push(rb);
      if (sp.length) cars.push({ parts: sp });
    }
    const sun = sunDir();
    const night = nightView();
    // the debris flying in the model's wind and coming to rest (damage.js)
    Damage.step(dt, windAt, (x, y) => surfAt(x, y).h, [Math.round(c.p[0]), Math.round(c.p[1])]);
    Render.setDebris(Damage.state.inst, Damage.state.n, Damage.state.origin);
    Lightning.step(now, c.p, (x, y) => surfAt(x, y).h, (x, y) => Render.cloudBase(x, y), f => Sound.thunder(f.d, f.cg));
    st.night = night;
    const tw = treeWind();
    if (detailed() && WorldLink.ready() && WorldLink.wind) WorldLink.wind(tw);
    const world = detailed() && WorldLink.ready() ? WorldLink.drawer(view, sun) : null;
    const fr = Render.frame({ ...view, sun, world, treeWind: tw, scale: q.scale, trees: q.trees !== 'off', density: 2 * q.density, rain: q.rain, detail: q.detail, relief: q.relief, haze: q.haze, base: q.base, wet: st.wet, contrast: q.contrast, bounce: q.bounce, far: q.far !== 'off', nestStep: Number.isFinite(+q.nestStep) ? +q.nestStep : 2, quality: q.tricubic, blend: q.blend, seed: (now / 16.7) % 64, precip: q.precip === 'off' ? null : precipView(view),
                              night, flash: Lightning.light(now, view.eye), heads: headlights(night), bolts: Lightning.bolts(now, view.eye, wx.rainI),
                              dust: dustView(view, night) }, cars);
    hud(fr, view);
    requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------- the HUD
  function say(m, ms) { st.msg = m; st.msgT = performance.now() + ms; }
  const COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  const bearing = (dx, dy) => COMPASS[Math.round(((Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360) / 22.5) % 16];
  const fmtT = s => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };

  function hud(fr, view) {
    const cv = $('hud'), dpr = Math.min(window.devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const c = st.car, now = performance.now();
    if (st.cam.mode === 'driver' && !st.foot) dash(g, W, H);
    // the tornadoes on the screen (and on the minimap, and the nearest's distance in the HUD): only with Settings ->
    // Tornado markers on; off, you find them as a chaser does, by the radar, the warnings and your own eyes
    const marks = st.vis.markers === 'on', tors = marks ? tornadoes() : [];
    if (fr) for (const t of tors) {
      const s = Render.project(fr.VP, fr.G, [t.x, t.y, Render.terrain(t.x, t.y) + 250]);
      const d = Math.hypot(t.x - c.p[0], t.y - c.p[1]);
      if (!s || s[0] < 0 || s[0] > W || s[1] < 0 || s[1] > H) continue;
      g.strokeStyle = 'rgba(255,70,70,.9)'; g.lineWidth = 2;
      g.beginPath(); g.moveTo(s[0] - 8, s[1] - 14); g.lineTo(s[0], s[1] - 4); g.lineTo(s[0] + 8, s[1] - 14); g.stroke();
      g.font = '600 12px Segoe UI'; g.fillStyle = '#ffd0d0'; g.textAlign = 'center';
      g.fillText(`T${t.id}${t.ef !== null && t.ef !== undefined ? ' EF' + t.ef : ''} · ${(d / 1000).toFixed(1)} km`, s[0], s[1] - 20);
    }
    // the other chasers' names over their cars
    if (fr) for (const o of st.lastOthers || []) {
      const at = o.fp ? [o.fp[0], o.fp[1], o.fp[2] - 0.4] : o.p;
      const s = Render.project(fr.VP, fr.G, [at[0], at[1], at[2] + 2.4]);
      if (!s || s[0] < 0 || s[0] > W || s[1] < 0 || s[1] > H) continue;
      const me = st.foot || c, d = Math.hypot(at[0] - me.p[0], at[1] - me.p[1]);
      g.font = '600 13px Segoe UI'; g.textAlign = 'center';
      const label = `${o.name}${o.fp ? ' (on foot)' : ''} · ${d < 1000 ? Math.round(d) + ' m' : (d / 1000).toFixed(1) + ' km'}`;
      const w = g.measureText(label).width + 12;
      g.fillStyle = 'rgba(8,12,18,.55)'; roundRect(g, s[0] - w / 2, s[1] - 18, w, 20, 6); g.fill();
      g.fillStyle = `rgb(${o.color.map(v => Math.min(255, v + 90)).join(',')})`; g.fillText(label, s[0], s[1] - 4);
    }
    // the speedo
    const v = c.forwardSpeed, sp = st.units === 'mph' ? Math.abs(v) * 2.23694 : Math.abs(v) * 3.6;
    const x0 = 18, y0 = H - 18;
    g.fillStyle = 'rgba(8,12,18,.55)'; roundRect(g, x0, y0 - 104, 250, 104, 10); g.fill();
    g.fillStyle = '#fff'; g.textAlign = 'left'; g.font = '700 46px Segoe UI';
    g.fillText(String(Math.round(sp)), x0 + 14, y0 - 48);
    g.font = '600 13px Segoe UI'; g.fillStyle = '#b8c4d0';
    g.fillText(st.units, x0 + 16 + g.measureText(String(Math.round(sp))).width * 46 / 13 + 6, y0 - 50);
    const gear = c.gear === -1 ? 'R' : c.gear === 0 ? 'N' : String(c.gear);
    g.font = '700 30px Segoe UI'; g.fillStyle = c.gear === -1 ? '#ffb347' : '#e8eef4'; g.textAlign = 'right';
    g.fillText(gear, x0 + 236, y0 - 50);
    const rf = Math.min(1, c.rpm / c.spec.redline);
    g.fillStyle = 'rgba(255,255,255,.12)'; g.fillRect(x0 + 14, y0 - 40, 222, 7);
    g.fillStyle = rf > 0.92 ? '#ff5b4a' : rf > 0.8 ? '#ffb347' : '#6fd3ff'; g.fillRect(x0 + 14, y0 - 40, 222 * rf, 7);
    g.textAlign = 'left'; g.font = '12px Segoe UI'; g.fillStyle = '#c9d4de';
    const ws = c.wheels.map(w => w.surf && w.surf.name).filter(Boolean);
    const surf = ws.length ? ws.sort((a, b) => ws.filter(x => x === b).length - ws.filter(x => x === a).length)[0] : '';
    const slide = c.wheels.some(w => w.contact && w.slip > 1.3);
    g.fillText(`${surf}${st.wet > 0.3 ? ' · wet' : ''}${slide ? ' · sliding' : ''}`, x0 + 14, y0 - 16);
    g.textAlign = 'right'; g.fillStyle = '#8fa0b0';
    g.fillText(`${c.tc ? 'TC' : 'tc off'} ${c.abs ? 'ABS' : 'abs off'}`, x0 + 236, y0 - 16);
    // the run and the nearest tornado
    const r = st.run;
    g.textAlign = 'left'; g.font = '600 13px Segoe UI'; g.fillStyle = '#e6edf3';
    const lines = [];
    if (r && r.t !== undefined) lines.push(`Model ${r.model_ms ? new Date(r.model_ms).toISOString().slice(11, 16) + 'Z ' : ''}+${fmtT(r.t)} · ${r.speed ? r.speed.toFixed(1) + '× real time' : r.status}`);
    let near = null;
    for (const t of tors) { const d = Math.hypot(t.x - c.p[0], t.y - c.p[1]); if (!near || d < near.d) near = { ...t, d }; }
    if (near) lines.push(`Tornado ${near.id}${near.wind ? ` · ${Math.round(near.wind * 2.23694)} mph` : ''}${near.ef !== null && near.ef !== undefined ? ` (EF${near.ef})` : ''} · ${(near.d / 1000).toFixed(1)} km ${bearing(near.x - c.p[0], near.y - c.p[1])}`);
    else if (marks) { const s = storm(); if (s) lines.push(`No tornado on the ground · ${s.what} ${(Math.hypot(s.x - c.p[0], s.y - c.p[1]) / 1000).toFixed(1)} km ${bearing(s.x - c.p[0], s.y - c.p[1])}`); }
    {
      const W = Math.hypot(wx.u, wx.v), mph = st.units === 'mph', ws = x => Math.round(x * (mph ? 2.23694 : 3.6));
      const from = W > 0.5 ? COMPASS[Math.round(((Math.atan2(-wx.u, -wx.v) * 180 / Math.PI + 360) % 360) / 22.5) % 16] + ' ' : '';
      const T = wx.t === null ? '' : ` · ${mph ? Math.round(wx.t * 9 / 5 + 32) + ' °F' : wx.t.toFixed(1) + ' °C'}`;
      const pr = wx.rainI < 0.05 ? '' : ` · ${wx.snow > 0.7 ? 'snow' : wx.snow > 0.3 ? 'sleet' : 'rain'} ${wx.rainI < 0.35 ? 'light' : wx.rainI < 0.7 ? 'moderate' : 'heavy'}`;
      if (air.now || st.fake) lines.push(`Wind ${from}${ws(W)}${wx.gust > W + 2 ? ` gusting ${ws(wx.gust)}` : ''} ${mph ? 'mph' : 'km/h'}${T}${pr}${st.fake ? ' (test values)' : ''}`);
      else if (air.err && !/no run/.test(air.err)) lines.push(`Wind at the car: ${/^404$|^unknown/.test(air.err) ? 'restart the viewer (new in this version)' : air.err}`);
    }
    {
      const Ls = Lightning.state;
      if (Ls.total > 0) {
        let nd = Infinity, nm = null;
        for (const m of Ls.marks) { const d = Math.hypot(m.x - c.p[0], m.y - c.p[1]); if (m.cg && now - m.t < 60000 && d < nd) { nd = d; nm = m; } }
        lines.push(`Lightning ${Math.round(Ls.total * 60)} flashes/min in the area${nm ? ` · nearest strike ${fmtDist(nd)} ${bearing(nm.x - c.p[0], nm.y - c.p[1])}` : ''}`);
      } else if (Ls.err) lines.push(`Lightning: ${Ls.err}`);
      const dg = Damage.gustAt(c.p[0], c.p[1]);
      if (dg >= 29) { const e = Damage.ef(dg); lines.push(`Damage here: peak gust ${Math.round(dg * (st.units === 'mph' ? 2.23694 : 3.6))} ${st.units === 'mph' ? 'mph' : 'km/h'}${e >= 0 ? ` (EF${e})` : ''}`); }
      else if (Damage.state.err) lines.push(`Damage: ${Damage.state.err}`);
      if (st.night && st.night.dayK < 0.6) lines.push(`Headlights ${st.lights === 'on' ? 'on' : st.lights === 'off' ? 'off' : 'auto'} (L)`);
    }
    const wr = activeWarnings(c.p[0] / 1000, c.p[1] / 1000);
    if (wr.length) lines.push(`You are in a ${wr.map(w => w.kind === 'TOR' ? 'TORNADO WARNING' : w.kind === 'SVR' ? 'severe thunderstorm warning' : w.kind).join(' and a ')}`);
    const ns = Net.st;
    if (Net.local && ns.hosting) lines.push(`Hosting · join code ${ns.code} · ${ns.others.size} friend${ns.others.size === 1 ? '' : 's'} here`);
    else if (!Net.local) lines.push(`Joined the host · ${ns.others.size + 1} chasers · ${Math.round(ns.rtt)} ms`);
    else if (ns.others.size) lines.push(`${ns.others.size} other chaser${ns.others.size > 1 ? 's' : ''} on this PC`);
    if (ns.err) lines.push(`Multiplayer: ${ns.err}`);
    const osmBusy = (Roads.state.pending && Roads.state.osmBusy) || (Scenery.state.bPending && Scenery.state.bOsmBusy);
    if (osmBusy) lines.push(`OpenStreetMap (Overpass) is busy: ${Roads.state.pending + Scenery.state.bPending} area${Roads.state.pending + Scenery.state.bPending > 1 ? 's' : ''} of roads / buildings fill in when it answers`);
    else if (Scenery.state.bPending) lines.push(`Fetching buildings from FEMA USA Structures (${Scenery.state.bPending} area${Scenery.state.bPending > 1 ? 's' : ''} to go)…`);
    if (Scenery.state.bErr) lines.push(`Buildings: ${Scenery.state.bErr}`);
    if (Roads.state.pending && !osmBusy) lines.push(`Fetching roads from OpenStreetMap (${Roads.state.pending} area${Roads.state.pending > 1 ? 's' : ''} to go)…`);
    if (st.volErr) lines.push(`Storm volume: ${st.volErr}`);
    g.fillStyle = 'rgba(8,12,18,.5)'; roundRect(g, 14, 14, Math.max(...lines.map(l => g.measureText(l).width), 10) + 24, 12 + lines.length * 20, 8); g.fill();
    lines.forEach((l, i) => { g.fillStyle = l.includes('TORNADO WARNING') ? '#ff6b6b' : '#e6edf3'; g.fillText(l, 26, 34 + i * 20); });
    g.textAlign = 'right'; g.font = '11px Segoe UI'; g.fillStyle = 'rgba(220,230,240,.7)';
    // with the GPU the browser draws with (a laptop's browser is often on its integrated one) and at what size
    if (st.gpu === undefined) {
      const gl = Render.gl(), ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
      st.gpu = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)).replace(/^ANGLE \(|\)$/g, '').replace(/ Direct3D.*$/, '').replace(/^[^,]*, /, '').replace(/ \(0x[0-9a-f]+\)/i, '') : '';
    }
    const glc = Render.gl() && Render.gl().canvas;
    g.fillText(`${Math.round(st.fps)} fps · ${st.gpu ? st.gpu + ' · ' : ''}${glc ? `${glc.width}×${glc.height} · ` : ''}${st.cam.mode} camera (C) · H: help`, W - 16, 22);
    routeHud(g, W);
    Mesonet.draw(g, 14, H - 134 - Math.min(300, H - 330), Math.min(380, W * 0.42), Math.min(300, H - 330), st.units);
    if (st.msg && now < st.msgT) {
      g.font = '600 16px Segoe UI'; g.textAlign = 'center';
      const w = g.measureText(st.msg).width + 30;
      g.fillStyle = 'rgba(8,12,18,.65)'; roundRect(g, W / 2 - w / 2, 60, w, 34, 8); g.fill();
      g.fillStyle = '#fff'; g.fillText(st.msg, W / 2, 82);
    }
    minimap(g, W, H, tors);
  }

  // the route's next turn, top middle: an arrow, how far, onto what; the station and how far to it
  function routeHud(g, W) {
    const R = Stations.route;
    if (!R.on) return;
    const n = Stations.next(), y = 104;
    g.save();
    if (!n) {
      g.font = '600 14px Segoe UI'; g.textAlign = 'center';
      const t = `Gas station: ${R.msg || 'finding the way…'}`, w = g.measureText(t).width + 28;
      g.fillStyle = 'rgba(8,12,18,.65)'; roundRect(g, W / 2 - w / 2, y, w, 30, 8); g.fill();
      g.fillStyle = '#9fe8ff'; g.fillText(t, W / 2, y + 20);
      g.restore(); return;
    }
    const what = n.dir === 'arrive' ? `${n.dest.name} on the ${n.side}` : `${n.sharp ? 'Sharp ' : n.slight ? 'Bear ' : 'Turn '}${n.dir} onto ${Roads.NAMES[n.cls].toLowerCase()}`;
    const big = fmtDist(n.dist), sub = n.dir === 'arrive' ? 'Gas station' : `${n.dest.name} · ${fmtDist(n.total)}`;
    g.font = '700 22px Segoe UI'; const w1 = g.measureText(big).width;
    g.font = '600 14px Segoe UI'; const w2 = Math.max(g.measureText(what).width, g.measureText(sub).width);
    const w = 62 + Math.max(w1, w2) + 20, x = W / 2 - w / 2;
    g.fillStyle = 'rgba(8,12,18,.72)'; roundRect(g, x, y, w, 64, 10); g.fill();
    g.strokeStyle = 'rgba(79,195,255,.6)'; g.lineWidth = 1.5; g.stroke();
    // the arrow
    g.save();
    g.translate(x + 32, y + 32);
    g.strokeStyle = '#4fd8ff'; g.fillStyle = '#4fd8ff'; g.lineWidth = 4; g.lineCap = 'round'; g.lineJoin = 'round';
    if (n.dir === 'arrive') {
      g.beginPath(); g.arc(0, -4, 9, Math.PI, 0); g.lineTo(0, 16); g.closePath(); g.fill();
      g.fillStyle = '#04121c'; g.beginPath(); g.arc(0, -4, 3.5, 0, 7); g.fill();
    } else {
      const a = (n.dir === 'left' ? -1 : 1) * (n.sharp ? 2.4 : n.slight ? 0.6 : Math.PI / 2);
      const ex = Math.sin(a) * 13, ey = -Math.cos(a) * 13;
      g.beginPath(); g.moveTo(0, 16); g.lineTo(0, 0); g.lineTo(ex, ey); g.stroke();
      g.translate(ex, ey); g.rotate(a);
      g.beginPath(); g.moveTo(0, -7); g.lineTo(-7, 2); g.lineTo(7, 2); g.closePath(); g.fill();
    }
    g.restore();
    g.textAlign = 'left';
    g.font = '700 22px Segoe UI'; g.fillStyle = '#fff'; g.fillText(big, x + 62, y + 26);
    g.font = '600 14px Segoe UI'; g.fillStyle = '#cfe9f5'; g.fillText(what, x + 62, y + 44);
    g.font = '12px Segoe UI'; g.fillStyle = '#8fb3c4'; g.fillText(sub, x + 62, y + 59);
    g.restore();
  }

  function activeWarnings(xk, yk) {
    const out = [];
    for (const w of (st.run && st.run.warnings) || []) {
      if (!w.active || !w.poly) continue;
      let inside = false;
      for (let i = 0, j = w.poly.length - 1; i < w.poly.length; j = i++) {
        const [xi, yi] = w.poly[i], [xj, yj] = w.poly[j];
        if ((yi > yk) !== (yj > yk) && xk < (xj - xi) * (yk - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (inside) out.push(w);
    }
    return out.sort((a, b) => (a.kind === 'TOR' ? -1 : 1));
  }

  // ---------------------------------------------------------------- the minimap (a map you can zoom and drag)
  // st.mapKm: the side shown (any, by the wheel; M steps through MAP_STEPS); st.mmPan: the centre (world m) when
  // dragged away from the car, null while it follows the car.  The roads (the drivable ones, roads.js) and, close
  // in, the buildings are drawn over the map picture like a road map: cased lines, sized by class and zoom.
  const MAP_STEPS = [6, 20, 60, 150], MAP_MIN = 0.6, MAP_MAX = 300;
  // road classes (roads.js k: 0 motorway, 1 trunk, 2 primary, 3 secondary, 4 tertiary, 5 residential, 6 service,
  // 7 track): real width m, fill (dark map, light map), casing, the zoom (side km) above which it is left out
  const ROAD_STYLE = [
    { w: 24, dark: '#f0a52e', light: '#f6b74a', case: '#5a3a08', lcase: '#c48a22', max: 1e9 },
    { w: 20, dark: '#f3c04c', light: '#fbd06a', case: '#5c4410', lcase: '#c9a23c', max: 1e9 },
    { w: 15, dark: '#f6dc7a', light: '#fde89a', case: '#5f5326', lcase: '#c9b25a', max: 160 },
    { w: 12, dark: '#ffffff', light: '#ffffff', case: '#3d434b', lcase: '#a8adb4', max: 70 },
    { w: 10, dark: '#e8eaed', light: '#ffffff', case: '#3a4048', lcase: '#b4b9bf', max: 80 },
    { w: 8, dark: '#d3d7dc', light: '#ffffff', case: '#353a42', lcase: '#c2c6cb', max: 30 },
    // "minor": every residential, unclassified and service road (OpenFreeMap), most of the local network
    { w: 7, dark: '#c9ced4', light: '#ffffff', case: '#30353c', lcase: '#c2c6cb', max: 30 },
    { w: 4, dark: '#b39b74', light: '#d8c7a5', case: '#3a3226', lcase: '#a8977a', max: 10 },
  ];
  const mmCache = { roads: null, rPaths: null, rOrg: null, bGen: -1, bPath: null, bOrg: null };
  function roadPaths() {
    const polys = Roads.state.polys;
    if (polys !== mmCache.roads) {
      mmCache.roads = polys;
      const org = polys && polys.length && polys[0].pts.length ? [Math.round(polys[0].pts[0][0]), Math.round(polys[0].pts[0][1])] : [0, 0];
      const paths = ROAD_STYLE.map(() => ({ paved: new Path2D(), dirt: new Path2D(), n: 0 }));
      for (const { k, pv, pts } of polys || []) {
        const p = paths[Math.max(0, Math.min(7, k | 0))], path = pv ? p.paved : p.dirt;
        pts.forEach((q, i) => (i ? path.lineTo(q[0] - org[0], q[1] - org[1]) : path.moveTo(q[0] - org[0], q[1] - org[1])));
        p.n++;
      }
      mmCache.rPaths = paths; mmCache.rOrg = org;
    }
    return mmCache;
  }
  function buildingPath() {
    const s = Scenery.state;
    if (s.bGen !== mmCache.bGen) {
      mmCache.bGen = s.bGen; mmCache.bOrg = s.bOrigin.slice();
      const p = new Path2D();
      for (const b of s.polys || []) {
        const P = b.P;
        p.moveTo(P[0], P[1]);
        for (let i = 2; i < P.length; i += 2) p.lineTo(P[i], P[i + 1]);
        p.closePath();
      }
      mmCache.bPath = p;
    }
    return mmCache;
  }
  function mapCentre() { const c = st.car; return st.mmPan || [c.p[0], c.p[1]]; }
  function mapButtons() {
    const r = st.mmRect;
    if (!r) return [];
    const b = 26, x = r.x0 + r.S - b - 8;
    return [{ id: 'in', x, y: r.y0 + r.S - 3 * b - 18, b, label: '+' }, { id: 'out', x, y: r.y0 + r.S - 2 * b - 14, b, label: '−' },
            { id: 'car', x, y: r.y0 + r.S - b - 8, b, label: '⌖' }];
  }
  function zoomMap(f, mx, my) {
    const r = st.mmRect, km = Math.max(MAP_MIN, Math.min(MAP_MAX, st.mapKm * f));
    if (r && st.mmPan && mx !== undefined) {
      // the point under the pointer stays where it is
      const [cx, cy] = mapCentre(), sc0 = r.S / (st.mapKm * 1000), sc1 = r.S / (km * 1000);
      const ox = mx - (r.x0 + r.S / 2), oy = my - (r.y0 + r.S / 2);
      st.mmPan = [cx + ox / sc0 - ox / sc1, cy - oy / sc0 + oy / sc1];
    }
    st.mapKm = km;
  }
  const inMap = (x, y) => { const r = st.mmRect; return r && x >= r.x0 && x <= r.x0 + r.S && y >= r.y0 && y <= r.y0 + r.S; };
  const mapBtnAt = (x, y) => mapButtons().find(b => x >= b.x && x <= b.x + b.b && y >= b.y && y <= b.y + b.b);
  function mapClick(id) {
    if (id === 'in') zoomMap(1 / 1.6);
    else if (id === 'out') zoomMap(1.6);
    else if (id === 'car') st.mmPan = null;
  }
  const fmtKm = k => (k < 1 ? `${Math.round(k * 1000)} m` : k < 10 ? `${k.toFixed(1).replace(/\.0$/, '')} km` : `${Math.round(k)} km`);

  function minimap(g, W, H, tors) {
    const v = st.vis, big = v.mmBig === 'big';
    const S = big ? Math.min(H - 140, W * 0.5) : Math.min(260, W * 0.3), x0 = W - S - 16, y0 = H - S - 16, c = st.car;
    st.mmRect = { x0, y0, S };
    const [cx, cy] = mapCentre();
    const R = st.mapKm * 1000 / 2, sc = S / (2 * R);
    const P = (x, y) => [x0 + S / 2 + (x - cx) * sc, y0 + S / 2 - (y - cy) * sc];
    g.save();
    g.fillStyle = 'rgba(14,20,16,.85)'; roundRect(g, x0, y0, S, S, 10); g.fill(); g.clip();
    // the real map and the radar under everything (minimap.js)
    MiniMap.draw(g, x0, y0, S, cx / 1000, cy / 1000, st.mapKm / 2,
                 { base: v.mmBase, product: v.mmProduct, site: v.mmSite, opacity: v.mmOpacity, dim: v.mmDim });
    const dark = v.mmBase === 'osmdark' || v.mmBase === 'none' || v.mmBase.startsWith('sat');
    // the buildings, close in
    if (st.mapKm <= 3 && Scenery.state.polys && Scenery.state.polys.length) {
      const m = buildingPath();
      g.save();
      g.translate(...P(m.bOrg[0], m.bOrg[1])); g.scale(sc, -sc);
      g.globalAlpha = Math.min(1, (3 - st.mapKm) / 1.2) * 0.9;
      g.fillStyle = dark ? '#5d636b' : '#d6d1c6'; g.fill(m.bPath);
      g.lineWidth = 1 / sc; g.strokeStyle = dark ? '#7a8088' : '#bdb6a8'; g.stroke(m.bPath);
      g.restore();
    }
    // the roads: casings first, then the fills, the small roads under the big
    if (Roads.state.polys && Roads.state.polys.length) {
      const m = roadPaths();
      g.save();
      g.translate(...P(m.rOrg[0], m.rOrg[1])); g.scale(sc, -sc);
      g.lineCap = 'round'; g.lineJoin = 'round';
      const width = k => Math.max(k <= 1 ? 2.6 : k <= 3 ? 2 : 1.3, Math.min(22, ROAD_STYLE[k].w * sc)) / sc;
      for (const pass of ['case', 'fill']) {
        for (let k = 7; k >= 0; k--) {
          const s = ROAD_STYLE[k], p = m.rPaths[k];
          if (!p.n || st.mapKm > s.max) continue;
          const w = width(k);
          if (pass === 'case') {
            g.strokeStyle = dark ? s.case : s.lcase; g.lineWidth = w + 1.6 / sc;
            g.setLineDash([]); g.stroke(p.paved); g.stroke(p.dirt);
          } else {
            g.strokeStyle = dark ? s.dark : s.light; g.lineWidth = w;
            g.setLineDash([]); g.stroke(p.paved);
            g.strokeStyle = dark ? '#b39b74' : '#d8c7a5'; g.setLineDash([4 / sc, 3 / sc]); g.stroke(p.dirt); g.setLineDash([]);
          }
        }
      }
      g.restore();
    }
    // the warnings
    for (const w of (st.run && st.run.warnings) || []) {
      if (!w.active || !w.poly) continue;
      g.strokeStyle = w.kind === 'TOR' ? '#ff3030' : w.kind === 'SVR' ? '#ffd23a' : '#7fd0ff'; g.lineWidth = 2;
      g.beginPath(); w.poly.forEach((p, i) => { const s = P(p[0] * 1000, p[1] * 1000); if (i) g.lineTo(s[0], s[1]); else g.moveTo(s[0], s[1]); }); g.closePath(); g.stroke();
    }
    // the tornadoes and their tracks
    for (const t of tors) {
      if (t.pts && t.pts.length > 1) {
        g.strokeStyle = 'rgba(255,120,120,.8)'; g.lineWidth = 2; g.beginPath();
        t.pts.forEach((p, i) => { const s = P(p[1], p[2]); if (i) g.lineTo(s[0], s[1]); else g.moveTo(s[0], s[1]); }); g.stroke();
      }
      let s = P(t.x, t.y);
      const off = s[0] < x0 || s[0] > x0 + S || s[1] < y0 || s[1] > y0 + S;
      s = [Math.max(x0 + 8, Math.min(x0 + S - 8, s[0])), Math.max(y0 + 8, Math.min(y0 + S - 8, s[1]))];
      g.fillStyle = off ? 'rgba(255,60,60,.6)' : '#ff3b3b'; g.beginPath(); g.arc(s[0], s[1], 6, 0, 7); g.fill();
      g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.stroke();
    }
    Lightning.map(g, P, performance.now());
    // the gas stations (close in) and the route to one
    {
      const R = Stations.route;
      if (R.pts) {
        g.strokeStyle = 'rgba(4,18,28,.8)'; g.lineWidth = 6; g.lineCap = 'round'; g.lineJoin = 'round';
        const line = () => { g.beginPath(); R.pts.forEach((q, i) => { const s = P(q[0], q[1]); if (i) g.lineTo(s[0], s[1]); else g.moveTo(s[0], s[1]); }); };
        line(); g.stroke();
        g.strokeStyle = '#3fd4ff'; g.lineWidth = 3.5; line(); g.stroke();
      }
      if (st.mapKm <= 25) for (const sn of Stations.state.list) {
        const s = P(sn.x, sn.y);
        if (s[0] < x0 - 8 || s[0] > x0 + S + 8 || s[1] < y0 - 8 || s[1] > y0 + S + 8) continue;
        const dest = R.dest === sn;
        g.fillStyle = dest ? '#3fd4ff' : '#f0a020'; g.strokeStyle = '#fff'; g.lineWidth = 1.2;
        roundRect(g, s[0] - 5, s[1] - 5, 10, 10, 2); g.fill(); g.stroke();
        g.fillStyle = '#1a1206'; g.fillRect(s[0] - 2.2, s[1] - 2.8, 3.4, 5.6);
        if (st.mapKm <= 3 || dest) { g.font = '600 10px Segoe UI'; g.textAlign = 'left'; g.fillStyle = '#fff'; g.fillText(sn.name, s[0] + 8, s[1] + 3); }
      }
    }
    if (st.extraMap) st.extraMap(g, P);
    // the car (at the edge, pointing, when the map is dragged away from it)
    let s = P(c.p[0], c.p[1]);
    const away = s[0] < x0 || s[0] > x0 + S || s[1] < y0 || s[1] > y0 + S;
    s = [Math.max(x0 + 10, Math.min(x0 + S - 10, s[0])), Math.max(y0 + 10, Math.min(y0 + S - 10, s[1]))];
    g.save();
    g.translate(s[0], s[1]); g.rotate(-c.yaw);
    g.globalAlpha = away ? 0.6 : 1;
    g.fillStyle = '#4fc3ff'; g.strokeStyle = '#fff'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(9, 0); g.lineTo(-6, 6); g.lineTo(-3, 0); g.lineTo(-6, -6); g.closePath(); g.fill(); g.stroke();
    g.restore();
    // the zoom and back-to-the-car buttons
    for (const b of mapButtons()) {
      const hot = b.id === 'car' && st.mmPan;
      g.fillStyle = hot ? 'rgba(79,195,255,.9)' : 'rgba(8,12,18,.72)'; roundRect(g, b.x, b.y, b.b, b.b, 6); g.fill();
      g.strokeStyle = 'rgba(255,255,255,.25)'; g.lineWidth = 1; g.stroke();
      g.fillStyle = hot ? '#04121c' : '#e8eef4'; g.font = '600 17px Segoe UI'; g.textAlign = 'center';
      g.fillText(b.label, b.x + b.b / 2, b.y + b.b / 2 + 6);
    }
    // a scale bar
    const nice = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
    const len = nice.reduce((a, m) => (m * sc <= S * 0.3 ? m : a), nice[0]);
    g.strokeStyle = 'rgba(240,244,248,.9)'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x0 + 10, y0 + S - 12); g.lineTo(x0 + 10 + len * sc, y0 + S - 12); g.stroke();
    g.font = '10px Segoe UI'; g.textAlign = 'left'; g.fillStyle = 'rgba(240,244,248,.95)';
    g.fillText(len >= 1000 ? `${len / 1000} km` : `${len} m`, x0 + 12, y0 + S - 17);
    g.restore();
    g.font = '11px Segoe UI'; g.fillStyle = 'rgba(220,230,240,.8)'; g.textAlign = 'left';
    const lab = `N ↑  ${fmtKm(st.mapKm)} (M)  ${MiniMap.label(v.mmProduct)}${v.mmProduct !== 'none' ? ' (K)' : ''}${st.mmPan ? '  · dragged: ⌖ or double-click to follow' : ''}`;
    g.fillStyle = 'rgba(8,12,18,.6)'; roundRect(g, x0 + 4, y0 + 4, Math.min(S - 8, g.measureText(lab).width + 10), 17, 5); g.fill();
    g.fillStyle = 'rgba(230,236,242,.95)';
    g.save(); g.beginPath(); g.rect(x0 + 4, y0, S - 8, 30); g.clip();
    g.fillText(lab, x0 + 9, y0 + 16);
    g.restore();
  }

  // ---------------------------------------------------------------- the big map: where to start, where to go
  // A full-screen map (the minimap's picture and radar, minimap.js slot 'pick') with the warnings, the storm cells,
  // the tornadoes, the other chasers and the suggested start; a click (not a drag) picks the spot (world m), and the
  // car goes onto the nearest road there.  Resolves [x, y], 'auto' (ahead of the storm) or null (cancelled).
  const pick = { open: false, cx: 0, cy: 0, km: 90, drag: null, hover: null, done: null, raf: 0 };
  // a spot `dist` m ahead of the storm and to the right of its motion: the inflow side, where chasers come from
  function startNear(s, dist) {
    const sp = Math.hypot(...s.v) || 1, ux = s.v[0] / sp, uy = s.v[1] / sp, a = -1.0;
    const dx = ux * Math.cos(a) - uy * Math.sin(a), dy = ux * Math.sin(a) + uy * Math.cos(a);
    return [s.x + dx * dist, s.y + dy * dist];
  }
  function openPicker(msg) {
    const c = st.car, s = storm(), driving = c && st.spawned;
    const at = driving ? [c.p[0], c.p[1]] : s ? [s.x, s.y] : [0, 0];
    Object.assign(pick, { open: true, cx: at[0], cy: at[1], km: driving ? 40 : 90, drag: null, hover: null });
    $('pickMsg').textContent = msg;
    $('picker').classList.remove('hidden');
    return new Promise(res => { pick.done = res; drawPicker(); });
  }
  function closePicker(v) {
    if (!pick.open) return;
    pick.open = false; $('picker').classList.add('hidden');
    cancelAnimationFrame(pick.raf);
    const d = pick.done; pick.done = null;
    if (d) d(v);
  }
  function pickGeom() {
    const cv = $('pickCv'), W = cv.clientWidth, H = cv.clientHeight, S = Math.max(W, H);
    return { W, H, S, x0: (W - S) / 2, y0: (H - S) / 2, sc: S / (pick.km * 1000) };
  }
  const pickToWorld = (mx, my) => { const G = pickGeom(); return [pick.cx + (mx - G.W / 2) / G.sc, pick.cy - (my - G.H / 2) / G.sc]; };
  function drawPicker() {
    if (!pick.open) return;
    pick.raf = requestAnimationFrame(drawPicker);
    const cv = $('pickCv'), dpr = Math.min(window.devicePixelRatio || 1, 2), G = pickGeom();
    if (cv.width !== Math.round(G.W * dpr) || cv.height !== Math.round(G.H * dpr)) { cv.width = Math.round(G.W * dpr); cv.height = Math.round(G.H * dpr); }
    const g = cv.getContext('2d'), v = st.vis;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#0e1410'; g.fillRect(0, 0, G.W, G.H);
    MiniMap.draw(g, G.x0, G.y0, G.S, pick.cx / 1000, pick.cy / 1000, pick.km / 2,
                 { base: v.mmBase === 'none' ? 'osmdark' : v.mmBase, product: v.mmProduct, site: v.mmSite, opacity: v.mmOpacity, dim: v.mmDim, slot: 'pick' });
    const P = (x, y) => [G.W / 2 + (x - pick.cx) * G.sc, G.H / 2 - (y - pick.cy) * G.sc];
    const label = (t, x, y, col) => {
      g.font = '600 12px Segoe UI'; g.textAlign = 'left';
      const w = g.measureText(t).width + 10;
      g.fillStyle = 'rgba(8,12,18,.7)'; roundRect(g, x, y - 13, w, 18, 5); g.fill();
      g.fillStyle = col; g.fillText(t, x + 5, y);
    };
    for (const w of (st.run && st.run.warnings) || []) {
      if (!w.active || !w.poly) continue;
      g.strokeStyle = w.kind === 'TOR' ? '#ff3030' : w.kind === 'SVR' ? '#ffd23a' : '#7fd0ff'; g.lineWidth = 2;
      g.beginPath(); w.poly.forEach((p, i) => { const s = P(p[0] * 1000, p[1] * 1000); if (i) g.lineTo(s[0], s[1]); else g.moveTo(s[0], s[1]); }); g.closePath(); g.stroke();
    }
    // the storm cells: a ring, their motion, their name
    for (const cl of ((st.run && st.run.tracks) || []).filter(c => c.ended === null || c.ended === undefined)) {
      const s = P(cl.x, cl.y), sp = Math.hypot(cl.vx || 0, cl.vy || 0);
      g.strokeStyle = 'rgba(255,255,255,.9)'; g.lineWidth = 2; g.beginPath(); g.arc(s[0], s[1], 8, 0, 7); g.stroke();
      if (sp > 0.5) {
        const ex = s[0] + cl.vx / sp * 26, ey = s[1] - cl.vy / sp * 26;
        g.beginPath(); g.moveTo(s[0], s[1]); g.lineTo(ex, ey); g.stroke();
      }
      label(`storm ${cl.id}`, s[0] + 11, s[1] - 6, '#e6edf3');
    }
    for (const t of tornadoes()) {
      const s = P(t.x, t.y);
      g.fillStyle = '#ff3b3b'; g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(s[0], s[1], 7, 0, 7); g.fill(); g.stroke();
      label(`tornado ${t.id}${t.ef !== null && t.ef !== undefined ? ' EF' + t.ef : ''}`, s[0] + 10, s[1] + 4, '#ffd0d0');
    }
    // the suggested starts
    const s = storm();
    if (s) for (const [d, nm] of [[12000, 'ahead of the storm'], [4000, 'close']]) {
      const q = P(...startNear(s, d));
      g.fillStyle = '#5ee08a'; g.strokeStyle = '#06210f'; g.lineWidth = 1.5; g.beginPath(); g.arc(q[0], q[1], 5, 0, 7); g.fill(); g.stroke();
      label(nm, q[0] + 8, q[1] + 4, '#b8f5cc');
    }
    remoteMap(g, P);
    if (st.car && st.spawned) {
      const q = P(st.car.p[0], st.car.p[1]);
      g.save(); g.translate(q[0], q[1]); g.rotate(-st.car.yaw);
      g.fillStyle = '#4fc3ff'; g.strokeStyle = '#fff'; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(10, 0); g.lineTo(-7, 7); g.lineTo(-3, 0); g.lineTo(-7, -7); g.closePath(); g.fill(); g.stroke();
      g.restore();
      label('you', q[0] + 12, q[1] + 4, '#bfe9ff');
    }
    // under the pointer: where that is from the storm
    if (pick.hover && !pick.drag) {
      const [mx, my] = pick.hover, [wx, wy] = pickToWorld(mx, my);
      g.strokeStyle = 'rgba(255,255,255,.8)'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(mx - 12, my); g.lineTo(mx + 12, my); g.moveTo(mx, my - 12); g.lineTo(mx, my + 12); g.stroke();
      if (s) {
        const d = Math.hypot(wx - s.x, wy - s.y);
        label(`${(d / 1000).toFixed(1)} km ${bearing(wx - s.x, wy - s.y)} of ${s.what} · click to go here`, mx + 14, my + 22, '#fff');
      }
    }
    const nice = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
    const len = nice.reduce((a, m) => (m * G.sc <= 160 ? m : a), nice[0]);
    g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.moveTo(20, G.H - 24); g.lineTo(20 + len * G.sc, G.H - 24); g.stroke();
    label(len >= 1000 ? `${len / 1000} km` : `${len} m`, 20, G.H - 32, '#fff');
  }
  function bindPicker() {
    const cv = $('pickCv');
    const at = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener('mousedown', e => { pick.drag = { x: e.clientX, y: e.clientY, cx: pick.cx, cy: pick.cy, moved: false }; });
    window.addEventListener('mousemove', e => {
      if (!pick.open) return;
      if (pick.drag) {
        const dx = e.clientX - pick.drag.x, dy = e.clientY - pick.drag.y, sc = pickGeom().sc;
        if (Math.abs(dx) + Math.abs(dy) > 4) pick.drag.moved = true;
        if (pick.drag.moved) { pick.cx = pick.drag.cx - dx / sc; pick.cy = pick.drag.cy + dy / sc; }
      }
      pick.hover = e.target === cv ? at(e) : null;
    });
    window.addEventListener('mouseup', e => {
      if (!pick.open || !pick.drag) return;
      const d = pick.drag; pick.drag = null;
      if (!d.moved && e.target === cv) closePicker(pickToWorld(...at(e)));
    });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      const [mx, my] = at(e), [wx, wy] = pickToWorld(mx, my);
      pick.km = Math.max(2, Math.min(400, pick.km * Math.pow(1.0015, e.deltaY)));
      const G = pickGeom();                             // the point under the pointer stays put
      pick.cx = wx - (mx - G.W / 2) / G.sc; pick.cy = wy + (my - G.H / 2) / G.sc;
    }, { passive: false });
    $('pickAuto').onclick = () => closePicker('auto');
    $('pickCancel').onclick = () => closePicker(null);
  }
  // the car to (x, y) world m, facing the storm: it waits there until the roads around have come in, then goes
  // onto the nearest one (like the start)
  function teleport(x, y) {
    const c = st.car;
    if (!c) return;
    const s = storm(), yaw = s ? Math.atan2(s.y - y, s.x - x) : c.yaw;
    c.reset(x, y, yaw);
    WorldLink.moved();
    st.cam.yaw = yaw; st.cam.eye = null;
    Object.assign(st, { spawned: false, dropT: 0, offroadStart: null, mmPan: null,
                        spawnHint: { x, y, yaw, what: s ? s.what : 'the storm' } });
    say('Going there: loading the roads…', 4000);
  }
  async function goSomewhere() {
    if (!st.car || pick.open) return;
    const p = await openPicker('Click where you want to go · wheel: zoom · drag: pan · Esc: cancel');
    const s = storm();
    if (p === 'auto' && s) teleport(...startNear(s, 12000));
    else if (Array.isArray(p)) teleport(p[0], p[1]);
  }

  function dash(g, W, H) {
    // the dashboard and the A pillars, seen from the driver's seat
    g.fillStyle = 'rgba(10,10,12,.96)';
    g.beginPath(); g.moveTo(0, H); g.lineTo(0, H * 0.8); g.quadraticCurveTo(W * 0.5, H * 0.72, W, H * 0.8); g.lineTo(W, H); g.fill();
    g.beginPath(); g.moveTo(0, 0); g.lineTo(W * 0.07, 0); g.lineTo(0, H * 0.75); g.fill();
    g.beginPath(); g.moveTo(W, 0); g.lineTo(W * 0.9, 0); g.lineTo(W, H * 0.72); g.fill();
    g.fillStyle = 'rgba(30,30,34,.98)'; g.beginPath(); g.arc(W * 0.33, H * 1.02, H * 0.2, Math.PI, 2 * Math.PI); g.fill();
    g.strokeStyle = '#111'; g.lineWidth = H * 0.025; g.beginPath(); g.arc(W * 0.33, H * 1.02, H * 0.2, Math.PI * 1.05, Math.PI * 1.95); g.stroke();
  }

  // ---------------------------------------------------------------- the other chasers
  const mounts = new Map();                         // kind -> the wheels' mounts (from a car of that kind)
  function wheelMounts(kind) {
    if (!mounts.has(kind)) mounts.set(kind, new CarPhys.Car(kind, () => ({ h: 0, n: [0, 0, 1], ...SURF.asphalt })).wheels.map(w => w.mount));
    return mounts.get(kind);
  }
  function remoteCars() {
    const list = Net.others();
    st.lastOthers = list;
    return list.map(o => {
      const kind = CarPhys.SPECS[o.kind] ? o.kind : 'suv', M = CarPhys.qmat(o.q), up = M[2], m = wheelMounts(kind);
      // set on our own ground at the height they are above theirs (older pages send no hg: as sent)
      const z = Number.isFinite(o.hg) ? surfAt(o.p[0], o.p[1]).h + o.hg : o.p[2];
      const p = [o.p[0], o.p[1], z];
      o.p = p;                                          // the name tag over it (st.lastOthers) goes by this too
      const wheels = m.map((mt, i) => {
        const w = CarPhys.toWorld(M, mt);
        return { c: [p[0] + w[0] - up[0] * o.L[i], p[1] + w[1] - up[1] * o.L[i], p[2] + w[2] - up[2] * o.L[i]],
                 steer: i < 2 ? o.st[i] : 0, spin: o.sp[i], left: i % 2 === 0 };
      });
      return { kind, color: o.color, p, M, wheels, hidden: false };
    }).concat(list.filter(o => o.f).map(o => {
      let w = walkers.get(o.id);
      if (!w) walkers.set(o.id, w = new Walker.Person(0, 0, 0, 0));
      const f = o.f;
      w.p = [f[0], f[1], surfAt(f[0], f[1]).h + f[2]];
      Object.assign(w, { yaw: f[3], phase: f[4], gait: f[5], down: f[6], lean: f[7], leanDir: f[8], air: !!f[9], v: [f[10], 0, 0] });
      o.fp = w.p;                                        // the name tag and the minimap go by this
      return { parts: w.figure(o.color) };
    }));
  }
  const walkers = new Map();                         // the others on foot: id -> a Walker.Person posed from theirs
  function remoteMap(g, P) {
    for (const o of st.lastOthers || []) {
      const s = o.fp ? P(o.fp[0], o.fp[1]) : P(o.p[0], o.p[1]);
      g.fillStyle = `rgb(${o.color.join(',')})`; g.strokeStyle = '#fff'; g.lineWidth = 1.5;
      g.beginPath(); g.arc(s[0], s[1], 5, 0, 7); g.fill(); g.stroke();
      g.font = '600 10px Segoe UI'; g.fillStyle = '#fff'; g.textAlign = 'left'; g.fillText(o.name, s[0] + 7, s[1] + 3);
    }
  }
  // the internet link's part of the Multiplayer tab
  function tunnelHtml(ns) {
    const t = ns.tunnel || { state: 'off' }, on = t.state === 'on', busy = t.state === 'starting';
    const link = on ? `${t.url}/chase` : '';
    return `<h3>Internet link (friends need only a browser)</h3>` +
      (on ? `<p>Send friends this link and the code: <b>${esc(link)}</b></p>` +
            `<button data-copy="${esc(link)}">Copy link</button> <button data-copy="${esc(link + '?code=' + ns.code)}">Copy link with the code in it</button>` +
            `<p>Anyone with the link still needs the code (unless you send the one with the code in it). A new link each time you start it.</p>`
          : busy ? '<p>Starting a free Cloudflare tunnel… (the first time it downloads Cloudflare’s cloudflared, ~60 MB, once)</p>'
          : `<p>A free Cloudflare quick tunnel gives you an https link to chase mode only: the rest of the viewer stays private.${t.state === 'error' ? ` <span style="color:#ffb347">Last try: ${esc(t.error)}</span>` : ''}</p>`) +
      `<button id="tunBtn">${on || busy ? 'Stop the internet link' : 'Start the internet link'}</button>`;
  }
  const esc = t => String(t).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  // the Multiplayer tab
  async function fillMulti() {
    const ns = Net.st, el = $('multi');
    if (!el) return;
    if (Net.local) await Net.hostStatus();
    const players = [...ns.others.values()];
    const rows = players.map(o => `<li><span style="color:rgb(${o.color.map(v => Math.min(255, v + 80)).join(',')})">●</span> ${esc(o.name)} <button data-go="${o.id}">Drive to</button></li>`).join('');
    const where = ns.urls.length ? ns.urls.map(u => `<b>${esc(u)}</b>`).join(' or ')
      : '<b>http://&lt;this PC address&gt;:8830/chase</b> (the viewer needs --lan, which Atmos-WRF.cmd gives it)';
    el.innerHTML = Net.local
      ? `<label class="chk"><input id="hostOn" type="checkbox" ${ns.hosting ? 'checked' : ''}> Let friends join (LAN or Tailscale)</label>` +
        (ns.hosting ? `<div class="code">${ns.code}</div><p>On your network (or a VPN such as Tailscale) friends open ${where} and type this code.</p>` +
                      '<button id="newCode">New code (kicks everyone)</button>' + tunnelHtml(ns)
                    : '<p>Off: nobody else can reach chase mode or the run.</p>')
      : `<p>Joined the host as <b>${esc(st.name)}</b>.</p>`;
    el.innerHTML += `<h3>Chasers</h3><ul class="plist"><li>● ${esc(st.name)} (you)</li>${rows}</ul>`;
    const on = $('hostOn');
    if (on) on.onchange = async () => { try { await Net.host(on.checked); } catch (e) { say(e.message, 3000); } fillMulti(); };
    const tb = $('tunBtn');
    if (tb) tb.onclick = async () => {
      tb.disabled = true;
      try { await Net.tunnel(!(ns.tunnel && ['on', 'starting'].includes(ns.tunnel.state))); } catch (e) { say(e.message, 3000); }
      fillMulti();
    };
    for (const b of el.querySelectorAll('[data-copy]')) b.onclick = async () => {
      try { await navigator.clipboard.writeText(b.dataset.copy); say('Copied', 1000); } catch (e) { prompt('Copy this:', b.dataset.copy); }
    };
    const nc = $('newCode');
    if (nc) nc.onclick = async () => { await Net.host(true, true); fillMulti(); };
    for (const b of el.querySelectorAll('[data-go]')) b.onclick = () => {
      const o = (st.lastOthers || []).find(o => o.id === +b.dataset.go);
      if (!o || !st.car) return;
      const M = CarPhys.qmat(o.q), yaw = Math.atan2(M[0][1], M[0][0]);
      place(o.p[0] - Math.cos(yaw) * 25, o.p[1] - Math.sin(yaw) * 25, yaw);
      say(`Behind ${o.name}`, 1500);
    };
  }

  // a photo: the view at full size and quality (Shift+P: with the HUD), saved as a PNG
  let showVis = () => {};
  function photo(withHud) {
    const q = st.vis, keep = { scale: q.scale, blend: q.blend };
    Object.assign(q, { scale: 1, blend: 0 });
    const view = camera(0), c = st.car;
    const cars = [{ kind: c.kind, color: st.color, p: c.p, M: c.matrix, wheels: c.wheelPoses(), hidden: false }];
    if (st.extraCars) cars.push(...st.extraCars());
    Render.frame({ ...view, sun: sunDir(), scale: 1, trees: q.trees !== 'off', density: 2 * q.density, rain: q.rain, detail: q.detail, relief: q.relief,
                   haze: q.haze, base: q.base, wet: st.wet, contrast: q.contrast, bounce: q.bounce, quality: 1, blend: 0, seed: 0 }, cars);
    Object.assign(q, keep);
    const v = $('view'), out = document.createElement('canvas');
    out.width = v.width; out.height = v.height;
    const g = out.getContext('2d');
    g.drawImage(v, 0, 0);
    if (withHud) g.drawImage($('hud'), 0, 0, out.width, out.height);
    const r = st.run, t = r && r.model_ms ? new Date(r.model_ms).toISOString().slice(0, 16).replace(/[-:T]/g, '') : 'chase';
    out.toBlob(b => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b); a.download = `atmoswrf-chase-${t}Z.png`; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, 'image/png');
    say('Photo saved', 1200);
  }

  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }
  const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));

  // ---------------------------------------------------------------- the page
  function init() {
    st.mapKm = 6;
    CarMesh.load();                  // the cars' models (web/models/cars): until they come in, the cars made from code
    Walker.setSay(m => say(m, 2500));
    try {
      $('name').value = localStorage.getItem('chase.name') || 'Chaser';
      $('vehicle').value = localStorage.getItem('chase.kind') || 'suv';
      const col = localStorage.getItem('chase.color');
      if (col) $('color').value = col;
    } catch (e) { /* private */ }
    $('vehicle').onchange = () => { $('color').value = '#' + COLORS[$('vehicle').value].map(v => v.toString(16).padStart(2, '0')).join(''); };
    try { if (!Render.init($('view'))) { $('startMsg').textContent = 'This browser has no WebGL2.'; return; } }
    catch (e) { $('startMsg').textContent = 'WebGL: ' + e.message; return; }
    $('go').onclick = start;
    window.addEventListener('keydown', e => {
      // driving keys always reach the car, unless a text box is being typed in: a menu control that still has
      // the keyboard (a dropdown, a slider, a button) lets go of it instead of eating W A S D and the arrows
      const t = e.target, tag = t && t.tagName;
      if (!st.started || (tag === 'INPUT' && ['text', 'number', 'search', 'password'].includes(t.type)) || tag === 'TEXTAREA') return;
      if (pick.open) { if (e.code === 'Escape') closePicker(null); return; }       // the big map has the keys
      if (e.code === 'KeyG' && !e.repeat) { goSomewhere(); return; }
      if (e.code === 'KeyF' && !e.repeat) { toggleFoot(); return; }
      if (e.code === 'KeyO' && !e.repeat) { toggleRoute(); return; }
      if (e.code === 'KeyJ' && !e.repeat) {
        if (e.shiftKey) say(Mesonet.save() ? 'Mesonet log saved (CSV)' : 'Nothing logged yet', 2000);
        else say(Mesonet.toggle() ? 'Mesonet probe: the model’s air where you are, logged every second' : 'Mesonet graphs hidden (still logging)', 2000);
        return;
      }
      if (e.code === 'KeyL' && !e.repeat) {
        st.lights = st.lights === 'on' ? 'off' : st.lights === 'off' ? 'auto' : 'on';
        say(`Headlights ${st.lights}`, 1200); return;
      }
      if (tag === 'SELECT' || tag === 'INPUT' || tag === 'BUTTON') { t.blur(); e.preventDefault(); }
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
      if (e.repeat) return;
      st.keys.add(e.code);
      if (e.code === 'KeyC') cycleCam();
      if (e.code === 'KeyR') resetCar();
      if (e.code === 'KeyH' || e.code === 'F1' || e.code === 'Escape') { e.preventDefault(); $('help').classList.toggle('hidden'); }
      if (e.code === 'KeyU') st.units = st.units === 'mph' ? 'km/h' : 'mph';
      if (e.code === 'KeyT' && st.car) { st.car.tc = !st.car.tc; say(`Traction control ${st.car.tc ? 'on' : 'off'}`, 1200); }
      if (e.code === 'KeyB' && st.car) { st.car.abs = !st.car.abs; say(`ABS ${st.car.abs ? 'on' : 'off'}`, 1200); }
      if (e.code === 'KeyP') photo(e.shiftKey);
      if (e.code === 'KeyN') { let on = true; if (!Sound.ctx()) Sound.start(); else on = Sound.toggle(); say(`Sound ${on ? 'on' : 'off'}`, 1200); }
      if (e.code === 'KeyM' && e.shiftKey) { st.vis.mmBig = st.vis.mmBig === 'big' ? 'small' : 'big'; showVis(); }
      else if (e.code === 'KeyM') st.mapKm = MAP_STEPS.find(k => k > st.mapKm + 0.01) || MAP_STEPS[0];
      if (e.code === 'KeyK') {
        const i = QUICK.indexOf(st.vis.mmProduct);
        st.vis.mmProduct = QUICK[(i + 1) % QUICK.length]; showVis();
        say(`Minimap: ${st.vis.mmProduct === 'none' ? 'no radar' : (MiniMap.GRID.concat(MiniMap.RADAR).find(p => p[0] === st.vis.mmProduct) || [0, ''])[1]}`, 1200);
      }
    });
    window.addEventListener('keyup', e => st.keys.delete(e.code));
    // a dropdown picked: give the keyboard back to the car
    document.addEventListener('change', e => { if (e.target && e.target.tagName === 'SELECT') e.target.blur(); });
    // clicking the view gives the keyboard back too
    $('hud').addEventListener('mousedown', () => { if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); });
    window.addEventListener('blur', () => st.keys.clear());
    const hudEl = $('hud');
    let drag = null, mapDrag = null;
    const local = e => { const r = hudEl.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    hudEl.addEventListener('mousedown', e => {
      const [x, y] = local(e);
      if (st.car && inMap(x, y)) {                    // the minimap: its buttons, or a drag of the map
        const b = mapBtnAt(x, y);
        if (b) mapClick(b.id);
        else mapDrag = { x: e.clientX, y: e.clientY, c: mapCentre().slice(), moved: false };
        e.preventDefault();
        return;
      }
      drag = { x: e.clientX, y: e.clientY, look: st.cam.look, pitch: st.cam.pitch }; st.cam.lookHeld = true;
    });
    hudEl.addEventListener('dblclick', e => { const [x, y] = local(e); if (inMap(x, y) && !mapBtnAt(x, y)) st.mmPan = null; });
    // right-click on the minimap: go there
    hudEl.addEventListener('contextmenu', e => {
      const [x, y] = local(e);
      if (!st.car || !inMap(x, y)) return;
      e.preventDefault();
      const r = st.mmRect, sc = r.S / (st.mapKm * 1000), [cx, cy] = mapCentre();
      teleport(cx + (x - (r.x0 + r.S / 2)) / sc, cy - (y - (r.y0 + r.S / 2)) / sc);
    });
    bindPicker();
    try { const sp = localStorage.getItem('chase.spawn'); if (sp) $('spawn').value = sp; } catch (e) { /* private */ }
    window.addEventListener('mousemove', e => {
      if (mapDrag) {
        const dx = e.clientX - mapDrag.x, dy = e.clientY - mapDrag.y, sc = st.mmRect.S / (st.mapKm * 1000);
        if (Math.abs(dx) + Math.abs(dy) > 3) mapDrag.moved = true;
        if (mapDrag.moved) st.mmPan = [mapDrag.c[0] - dx / sc, mapDrag.c[1] + dy / sc];
        return;
      }
      const [x, y] = local(e);
      hudEl.style.cursor = st.car && inMap(x, y) ? (mapBtnAt(x, y) ? 'pointer' : 'grab') : drag ? 'grabbing' : 'grab';
      if (!drag) return;
      st.cam.look = drag.look - (e.clientX - drag.x) * 0.006;
      st.cam.pitch = Math.max(-0.6, Math.min(0.9, drag.pitch + (e.clientY - drag.y) * 0.004));
    });
    window.addEventListener('mouseup', () => {
      mapDrag = null;
      if (drag) { drag = null; st.cam.lookHeld = false; st.cam.lookT = performance.now(); }
    });
    hudEl.addEventListener('wheel', e => {
      e.preventDefault();
      const [x, y] = local(e);
      if (st.car && inMap(x, y)) { zoomMap(Math.pow(1.0015, e.deltaY), x, y); return; }
      st.cam.dist = Math.max(0.6, Math.min(3, st.cam.dist * (e.deltaY > 0 ? 1.1 : 0.9)));
    }, { passive: false });
    // the menu: Controls / Settings tabs, the settings bound to st.vis by their data-k
    st.vis = { ...VIS0 };
    try { Object.assign(st.vis, JSON.parse(localStorage.getItem('chase.vis') || '{}')); } catch (e) { /* private */ }
    const prodSel = document.querySelector('[data-k=mmProduct]');
    prodSel.innerHTML = '<option value="none">None</option>' +
      `<optgroup label="Map products">${MiniMap.GRID.map(([k, l]) => `<option value="${k}">${l}</option>`).join('')}</optgroup>` +
      `<optgroup label="Radar (nearest or picked site)">${MiniMap.RADAR.map(([k, l]) => `<option value="${k}">${l}</option>`).join('')}</optgroup>`;
    const els = [...document.querySelectorAll('#tabSettings [data-k]')];
    const show = showVis = () => {
      const site = document.querySelector('[data-k=mmSite]'), list = MiniMap.radars();
      const want = ['nearest', ...list].join(',');
      if (site.dataset.list !== want) {
        site.dataset.list = want;
        site.innerHTML = '<option value="nearest">Nearest to me</option>' + list.map(r => `<option value="${r}">${r}</option>`).join('');
      }
      for (const el of els) {
        const k = el.dataset.k;
        el.value = String(st.vis[k]);
        const o = el.nextElementSibling;
        if (o && o.tagName === 'OUTPUT') o.textContent = k === 'bounce' ? (st.vis[k] ? `×${st.vis[k]}` : 'off') : k === 'mmOpacity' || k === 'mmDim' || k === 'vol' ? `${Math.round(st.vis[k] * 100)} %` : k === 'sunAz' ? `${st.vis[k]}°` : k === 'sunH' ? `${st.vis[k]}°` : k === 'haze' ? `${st.vis[k]} km` : String(st.vis[k]);
        if (k === 'sunAz' || k === 'sunH') el.disabled = st.vis.sunMode !== 'manual';
      }
    };
    const save = () => { try { localStorage.setItem('chase.vis', JSON.stringify(st.vis)); } catch (e) { /* private */ } };
    for (const el of els) el.oninput = el.onchange = () => {
      const k = el.dataset.k, num = typeof VIS0[k] === 'number';
      st.vis[k] = num ? Number(el.value) : el.value;
      if (k === 'base') Land.reset();
      show(); save();
    };
    $('visReset').onclick = () => { st.vis = { ...VIS0 }; Land.reset(); show(); save(); };
    show();
    for (const b of document.querySelectorAll('.tabs button')) b.onclick = () => {
      for (const o of document.querySelectorAll('.tabs button')) { o.classList.toggle('on', o === b); $(o.dataset.tab).classList.toggle('hidden', o !== b); }
      if (b.dataset.tab === 'tabMulti') fillMulti();
      if (b.dataset.tab === 'tabSettings') showVis();
    };
    setInterval(() => {
      if (st.started && !$('tabMulti').classList.contains('hidden') && !$('help').classList.contains('hidden')) fillMulti();
    }, 3000);
    if (!Net.local) {
      $('codeRow').classList.remove('hidden');
      const c = new URLSearchParams(location.search).get('code');
      if (c) $('code').value = c;
      $('go').textContent = 'Join and drive';
    }
    $('menuBtn').onclick = () => $('help').classList.toggle('hidden');
    $('menuX').onclick = () => $('help').classList.add('hidden');
  }

  // goTo(x, y): the car to world x, y m (as the map's click does), from the console
  return { init, st, surfAt, tornadoes, say, goTo: (x, y) => teleport(x, y) };
})();
window.addEventListener('DOMContentLoaded', Chase.init);
