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
  const BOX_KM = 100, RECENTRE_KM = 18, VOL_EVERY = 8000;
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
                 contrast: 1.3, mmBase: 'osmdark', mmProduct: 'cref', mmSite: 'nearest', mmOpacity: 0.8, mmDim: 0.15, mmBig: 'small' };
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
  function surfAt(x, y) {
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
    return { az, el };
  }
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
    try { st.targets = await getJSON('/api/pmw/targets'); } catch (e) { st.targets = null; }
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
    return out;
  }
  const efOf = w => (w === null || w === undefined ? null : w < 38 ? 0 : w < 50 ? 1 : w < 61 ? 2 : w < 74 ? 3 : w < 89 ? 4 : 5);

  async function loadVolume(force) {
    if (st.volBusy || !st.car) return;
    const x = st.car.p[0] / 1000, y = st.car.p[1] / 1000;
    let b = st.box;
    const far = !b || Math.hypot(x - (b[0] + b[2]) / 2, y - (b[1] + b[3]) / 2) > RECENTRE_KM;
    if (!far && !force && performance.now() - st.volAt < VOL_EVERY) return;
    if (far) { const h = BOX_KM / 2, cx = Math.round(x), cy = Math.round(y); b = [cx - h, cy - h, cx + h, cy + h]; }
    st.volBusy = true; st.volAt = performance.now();
    const q = `/api/volume?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&field=cloud`;
    try {
      let r = null;
      if (st.run && st.run.status === 'running') { try { r = await api(q); st.volT = null; } catch (e) { r = null; } }
      if (!r) {
        // a stopped run (or none of its fields in the box yet): the newest 3-D snapshot it saved
        const tt = await getJSON('/api/volume/times');
        if (!tt.times || !tt.times.length) throw new Error('no 3-D snapshots saved in this run');
        const t = tt.times[tt.times.length - 1];
        if (!far && !force && st.volT === t && st.box) { st.volBusy = false; return; }
        r = await api(`${q}&t=${t}`);
        st.volT = t;
      }
      Render.setVolume(b, await r.arrayBuffer());
      st.box = b; st.volErr = '';
    } catch (e) { st.volErr = e.message; }
    st.volBusy = false;
  }

  // ---------------------------------------------------------------- the car
  function place(x, y, yawHint) {
    const n = Roads.nearest(x, y, 6, 6000);
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
    const s = storm();
    if (!s) { say('No storm in the run yet.', 1e9); return; }
    // ahead and to the right of the storm's motion: the inflow side, where chasers come from
    const sp = Math.hypot(...s.v) || 1, ux = s.v[0] / sp, uy = s.v[1] / sp, a = -1.0;
    const dx = ux * Math.cos(a) - uy * Math.sin(a), dy = ux * Math.sin(a) + uy * Math.cos(a);
    const sx = s.x + dx * 12000, sy = s.y + dy * 12000;
    st.car = new CarPhys.Car(st.kind, ground);
    st.car.solid = Scenery.solid;
    st.extraCars = remoteCars; st.extraMap = remoteMap;
    st.car.reset(sx, sy, 0);
    st.spawnHint = { x: sx, y: sy, yaw: Math.atan2(s.y - sy, s.x - sx), what: s.what };
    say(`Heading for ${s.what}: loading the storm and the roads…`, 4000);
    await loadVolume(true);
    if (!Render.head()) { say(`No cloud volume: ${st.volErr || 'no storm fields yet'}`, 1e9); }
    st.last = performance.now();
    requestAnimationFrame(tick);
    setInterval(poll, 4000);
    // the streaming (network) on its own clock, not the frames'
    const stream = () => {
      const c = st.car;
      Roads.update(c.p[0], c.p[1], Render.terrain, () => Render.setRoads(Roads.state.mesh, Roads.state.origin));
      loadVolume(false);
      Scenery.updateBuildings(c.p[0], c.p[1], Render.terrain, () => Render.setBuildings(Scenery.state.bMesh, Scenery.state.bOrigin));
    };
    stream();
    setInterval(stream, 500);
    // the car to the others on its own clock: it keeps going while this tab is in the background
    setInterval(() => { if (st.car) Net.tick(st.car, { name: st.name, kind: st.kind, color: st.color }); }, 40);
  }

  // ---------------------------------------------------------------- input
  function input(dt) {
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
    place(c.p[0], c.p[1], c.yaw);
    say('Back on the road', 1500);
  }

  // ---------------------------------------------------------------- the camera
  const angd = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
  function camera(dt) {
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
    if (!st.spawned && Roads.state.segs && Roads.state.segs.n) {
      const h = st.spawnHint;
      if (place(h.x, h.y, h.yaw)) { st.spawned = true; say(`On the road, ${(Math.hypot(c.p[0] - h.x, c.p[1] - h.y) / 1000).toFixed(1)} km from where you were dropped. Chase ${h.what}!`, 5000); }
    }
    if (!st.spawned && !st.dropT) st.dropT = now;
    if (!st.spawned && now - st.dropT > 90000) { st.spawned = true; say('No roads came in from OpenStreetMap: off-road it is (R puts you on a road once they arrive).', 6000); }
    input(dt);
    // the model's rain at the car: a wet road (the lowest levels' precipitation)
    const pr = Render.sample ? Render.sample(c.p[0], c.p[1]) : 0;
    st.rain += (pr - st.rain) * Math.min(1, dt * 0.5);
    st.wet = Math.max(st.wet - dt / 600, Math.min(1, st.rain / 0.6));
    if (st.spawned) c.update(dt);
    if (!isFinite(c.p[0]) || c.p[2] < -50) resetCar();
    if (Scenery.updateTrees(c.p[0], c.p[1], Render.terrain, TREE_BUDGET[st.vis.trees] || 0)) {
      const ss = Scenery.state;
      Render.setTrees(ss.trees, ss.treeN, ss.treeNear, ss.treeOrigin);
    }
    if (st.box) Land.want(st.box, st.vis.base, { x: c.p[0] / 1000, y: c.p[1] / 1000 }, (x, y) => Render.terrain(x * 1000, y * 1000) / 1000);
    const view = camera(dt);
    const q = st.vis;
    const cars = [{ kind: c.kind, color: st.color, p: c.p, M: c.matrix, wheels: c.wheelPoses(), hidden: false }];
    if (st.extraCars) cars.push(...st.extraCars());
    const fr = Render.frame({ ...view, sun: sunDir(), scale: q.scale, trees: q.trees !== 'off', density: 2 * q.density, rain: q.rain, detail: q.detail, relief: q.relief, haze: q.haze, base: q.base, wet: st.wet, contrast: q.contrast, quality: q.tricubic, blend: q.blend, seed: (now / 16.7) % 64 }, cars);
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
    if (st.cam.mode === 'driver') dash(g, W, H);
    // the tornadoes on the screen
    const tors = tornadoes();
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
      const s = Render.project(fr.VP, fr.G, [o.p[0], o.p[1], o.p[2] + 2.4]);
      if (!s || s[0] < 0 || s[0] > W || s[1] < 0 || s[1] > H) continue;
      const d = Math.hypot(o.p[0] - c.p[0], o.p[1] - c.p[1]);
      g.font = '600 13px Segoe UI'; g.textAlign = 'center';
      const label = `${o.name} · ${d < 1000 ? Math.round(d) + ' m' : (d / 1000).toFixed(1) + ' km'}`;
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
    else { const s = storm(); if (s) lines.push(`No tornado on the ground · ${s.what} ${(Math.hypot(s.x - c.p[0], s.y - c.p[1]) / 1000).toFixed(1)} km ${bearing(s.x - c.p[0], s.y - c.p[1])}`); }
    const wr = activeWarnings(c.p[0] / 1000, c.p[1] / 1000);
    if (wr.length) lines.push(`You are in a ${wr.map(w => w.kind === 'TOR' ? 'TORNADO WARNING' : w.kind === 'SVR' ? 'severe thunderstorm warning' : w.kind).join(' and a ')}`);
    const ns = Net.st;
    if (Net.local && ns.hosting) lines.push(`Hosting · join code ${ns.code} · ${ns.others.size} friend${ns.others.size === 1 ? '' : 's'} here`);
    else if (!Net.local) lines.push(`Joined the host · ${ns.others.size + 1} chasers · ${Math.round(ns.rtt)} ms`);
    else if (ns.others.size) lines.push(`${ns.others.size} other chaser${ns.others.size > 1 ? 's' : ''} on this PC`);
    if (ns.err) lines.push(`Multiplayer: ${ns.err}`);
    if (Scenery.state.bPending) lines.push(`Fetching buildings from OpenStreetMap (${Scenery.state.bPending} area${Scenery.state.bPending > 1 ? 's' : ''} to go)…`);
    if (Scenery.state.bErr) lines.push(`Buildings: ${Scenery.state.bErr}`);
    if (Roads.state.pending) lines.push(`Fetching roads from OpenStreetMap (${Roads.state.pending} area${Roads.state.pending > 1 ? 's' : ''} to go)…`);
    if (st.volErr) lines.push(`Storm volume: ${st.volErr}`);
    g.fillStyle = 'rgba(8,12,18,.5)'; roundRect(g, 14, 14, Math.max(...lines.map(l => g.measureText(l).width), 10) + 24, 12 + lines.length * 20, 8); g.fill();
    lines.forEach((l, i) => { g.fillStyle = l.includes('TORNADO WARNING') ? '#ff6b6b' : '#e6edf3'; g.fillText(l, 26, 34 + i * 20); });
    g.textAlign = 'right'; g.font = '11px Segoe UI'; g.fillStyle = 'rgba(220,230,240,.7)';
    g.fillText(`${Math.round(st.fps)} fps · ${st.cam.mode} camera (C) · H: help`, W - 16, 22);
    if (st.msg && now < st.msgT) {
      g.font = '600 16px Segoe UI'; g.textAlign = 'center';
      const w = g.measureText(st.msg).width + 30;
      g.fillStyle = 'rgba(8,12,18,.65)'; roundRect(g, W / 2 - w / 2, 60, w, 34, 8); g.fill();
      g.fillStyle = '#fff'; g.fillText(st.msg, W / 2, 82);
    }
    minimap(g, W, H, tors);
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

  function minimap(g, W, H, tors) {
    const v = st.vis, big = v.mmBig === 'big';
    const S = big ? Math.min(H - 140, W * 0.5) : Math.min(260, W * 0.3), x0 = W - S - 16, y0 = H - S - 16, c = st.car;
    const R = st.mapKm * 1000 / 2, sc = S / (2 * R), cx = c.p[0], cy = c.p[1];
    const P = (x, y) => [x0 + S / 2 + (x - cx) * sc, y0 + S / 2 - (y - cy) * sc];
    g.save();
    g.fillStyle = 'rgba(14,20,16,.85)'; roundRect(g, x0, y0, S, S, 10); g.fill(); g.clip();
    // the real map and the radar under everything (minimap.js)
    MiniMap.draw(g, x0, y0, S, cx / 1000, cy / 1000, st.mapKm / 2,
                 { base: v.mmBase, product: v.mmProduct, site: v.mmSite, opacity: v.mmOpacity, dim: v.mmDim });
    // the roads (drawn ourselves when there is no map under them; else only the ones the car can be on, faintly)
    g.lineCap = 'round';
    for (const { k, pv, pts } of v.mmBase === 'none' ? Roads.state.polys || [] : []) {
      if (st.mapKm > 12 && k > 4) continue;
      g.strokeStyle = !pv ? 'rgba(170,150,120,.7)' : k <= 1 ? 'rgba(255,210,120,.95)' : k <= 3 ? 'rgba(240,240,240,.9)' : 'rgba(170,175,180,.75)';
      g.lineWidth = k <= 1 ? 2.4 : k <= 3 ? 1.8 : 1;
      g.beginPath();
      pts.forEach((p, i) => { const s = P(p[0], p[1]); if (i) g.lineTo(s[0], s[1]); else g.moveTo(s[0], s[1]); });
      g.stroke();
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
    if (st.extraMap) st.extraMap(g, P);
    // the car
    const s = P(cx, cy), yaw = c.yaw;
    g.translate(s[0], s[1]); g.rotate(-yaw);
    g.fillStyle = '#4fc3ff'; g.strokeStyle = '#fff'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(9, 0); g.lineTo(-6, 6); g.lineTo(-3, 0); g.lineTo(-6, -6); g.closePath(); g.fill(); g.stroke();
    g.restore();
    g.font = '11px Segoe UI'; g.fillStyle = 'rgba(220,230,240,.8)'; g.textAlign = 'left';
    const lab = `N ↑  ${st.mapKm} km (M)  ${MiniMap.label(v.mmProduct)}${v.mmProduct !== 'none' ? ' (K)' : ''}`;
    g.fillStyle = 'rgba(8,12,18,.6)'; roundRect(g, x0 + 4, y0 + 4, g.measureText(lab).width + 10, 17, 5); g.fill();
    g.fillStyle = 'rgba(230,236,242,.95)';
    g.fillText(lab, x0 + 9, y0 + 16);
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
      const wheels = m.map((mt, i) => {
        const w = CarPhys.toWorld(M, mt);
        return { c: [o.p[0] + w[0] - up[0] * o.L[i], o.p[1] + w[1] - up[1] * o.L[i], o.p[2] + w[2] - up[2] * o.L[i]],
                 steer: i < 2 ? o.st[i] : 0, spin: o.sp[i], left: i % 2 === 0 };
      });
      return { kind, color: o.color, p: o.p, M, wheels, hidden: false };
    });
  }
  function remoteMap(g, P) {
    for (const o of st.lastOthers || []) {
      const s = P(o.p[0], o.p[1]);
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
                   haze: q.haze, base: q.base, wet: st.wet, contrast: q.contrast, quality: 1, blend: 0, seed: 0 }, cars);
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
      if (!st.started || (e.target.tagName === 'INPUT' && e.target.type !== 'range') || e.target.tagName === 'SELECT') return;
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
      if (e.code === 'KeyM' && e.shiftKey) { st.vis.mmBig = st.vis.mmBig === 'big' ? 'small' : 'big'; showVis(); }
      else if (e.code === 'KeyM') st.mapKm = st.mapKm === 6 ? 20 : st.mapKm === 20 ? 60 : st.mapKm === 60 ? 150 : 6;
      if (e.code === 'KeyK') {
        const i = QUICK.indexOf(st.vis.mmProduct);
        st.vis.mmProduct = QUICK[(i + 1) % QUICK.length]; showVis();
        say(`Minimap: ${st.vis.mmProduct === 'none' ? 'no radar' : (MiniMap.GRID.concat(MiniMap.RADAR).find(p => p[0] === st.vis.mmProduct) || [0, ''])[1]}`, 1200);
      }
    });
    window.addEventListener('keyup', e => st.keys.delete(e.code));
    window.addEventListener('blur', () => st.keys.clear());
    const hudEl = $('hud');
    let drag = null;
    hudEl.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY, look: st.cam.look, pitch: st.cam.pitch }; st.cam.lookHeld = true; });
    window.addEventListener('mousemove', e => {
      if (!drag) return;
      st.cam.look = drag.look - (e.clientX - drag.x) * 0.006;
      st.cam.pitch = Math.max(-0.6, Math.min(0.9, drag.pitch + (e.clientY - drag.y) * 0.004));
    });
    window.addEventListener('mouseup', () => { if (drag) { drag = null; st.cam.lookHeld = false; st.cam.lookT = performance.now(); } });
    hudEl.addEventListener('wheel', e => { e.preventDefault(); st.cam.dist = Math.max(0.6, Math.min(3, st.cam.dist * (e.deltaY > 0 ? 1.1 : 0.9))); }, { passive: false });
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
        if (o && o.tagName === 'OUTPUT') o.textContent = k === 'mmOpacity' || k === 'mmDim' ? `${Math.round(st.vis[k] * 100)} %` : k === 'sunAz' ? `${st.vis[k]}°` : k === 'sunH' ? `${st.vis[k]}°` : k === 'haze' ? `${st.vis[k]} km` : String(st.vis[k]);
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

  return { init, st, surfAt, tornadoes, say };
})();
window.addEventListener('DOMContentLoaded', Chase.init);
