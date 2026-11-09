'use strict';
// Chase mode's damage, from the model's own winds: the run's peak 3-s gusts at 10 m where they could do damage
// (arhtmodelviewer/chase.py damage(): each grid's lowest level taken down to 10 m plus its turbulence, 25 m cells
// from the tornado nests), polled as they rise.  Each building's peak gust over its footprint gives its degree of
// damage on the EF scale's tables for its kind (WSEC 2006 expected winds: FR12 one- and two-family residences,
// SBO small barns and outbuildings, SRB / MBS small retail and metal buildings, ASR / IB apartments and institutions);
// scenery.js builds it so (roof cover stripped, decking gone, roof off, walls down, swept off its slab) and stops
// the car accordingly.  As a building near the car loses more, its pieces fly off: debris (shingles, boards,
// panels) carried by the model's wind where they are (its 10 m wind grid round the car), falling and coming to rest
// on the ground, at most MAX of them.  World metres; gusts m/s.
const Damage = (() => {
  const POLL_MS = 1500, BOX_KM = 8, MAX = 3000, REST_MS = 600000;
  const st = { ver: -1, fine: new Map(), coarse: new Map(), cell: [25, 250], busy: false, last: 0, gen: 0, err: '',
               deg: new Map(), debris: [], inst: null, n: 0 };
  // degrees of damage by kind (0 house, 1 shop / office, 2 farm / industrial, 3 tall, 4 shed / garage, 5 civic;
  // barns as 4): the expected 3-s gust (mph) at which each of DOD 1..10 is reached (the last ones of the shorter
  // tables run on to "destroyed")
  const DOD = {
    0: [65, 79, 81, 97, 104, 121, 132, 152, 170, 200],      // FR12
    1: [65, 83, 92, 101, 110, 121, 133, 150, 170, 196],     // SRB / SMB
    2: [67, 83, 95, 99, 110, 124, 137, 150, 165, 185],      // MBS
    3: [70, 88, 98, 110, 125, 140, 150, 165, 185, 210],     // ASR / MFR (mid-rise)
    4: [62, 72, 81, 91, 98, 110, 125, 140, 155, 170],       // SBO
    5: [68, 85, 97, 108, 120, 132, 145, 160, 178, 205],     // ES / IB
  };
  const MPH = 2.23694;
  const key = (i, j) => i * 2000003 + j;

  async function poll(x, y) {
    const now = performance.now();
    if (st.busy || now - st.last < POLL_MS) return;
    st.busy = true; st.last = now;
    try {
      const h = BOX_KM / 2, b = [x / 1000 - h, y / 1000 - h, x / 1000 + h, y / 1000 + h].map(v => v.toFixed(2));
      const box = b.join(',');
      const since = box === st.box ? st.ver : -1;          // a new box: all of it again
      const r = await fetch(`/api/c/damage?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&since=${since}`, { headers: window.ChaseAuth ? window.ChaseAuth() : {} });
      if (r.status === 404) { const t = await r.text(); st.err = /no run/.test(t) ? '' : 'damage needs the viewer restarted (new in this version)'; }
      else if (!r.ok) throw new Error(await r.text());
      else {
        const d = await r.json();
        if (d.fine) {
          if (d.reset) { st.fine.clear(); st.coarse.clear(); }
          st.cell = d.cell;
          for (const [i, j, g] of d.fine) st.fine.set(key(i, j), g / 10);
          for (const [i, j, g] of d.coarse) st.coarse.set(key(i, j), g / 10);
          if (d.reset || d.fine.length || d.coarse.length) st.gen++;
          st.box = box;
        }
        st.ver = d.ver; st.err = '';
      }
    } catch (e) { st.err = e.message; }
    st.busy = false;
  }

  // the peak gust at x, y, or the most over a box (world m)
  function gustAt(x, y) {
    const [f, c] = st.cell;
    return Math.max(st.fine.get(key(Math.floor(x / f), Math.floor(y / f))) || 0, st.coarse.get(key(Math.floor(x / c), Math.floor(y / c))) || 0);
  }
  function gustOver(x0, y0, x1, y1) {
    if (!st.fine.size && !st.coarse.size) return 0;
    const [f, c] = st.cell;
    let g = 0;
    for (let i = Math.floor(x0 / f); i <= Math.floor(x1 / f); i++) for (let j = Math.floor(y0 / f); j <= Math.floor(y1 / f); j++) {
      const v = st.fine.get(key(i, j));
      if (v > g) g = v;
    }
    for (let i = Math.floor(x0 / c); i <= Math.floor(x1 / c); i++) for (let j = Math.floor(y0 / c); j <= Math.floor(y1 / c); j++) {
      const v = st.coarse.get(key(i, j));
      if (v > g) g = v;
    }
    return g;
  }
  // the degree of damage (0..10) a gust does to a building of this kind
  function degree(kind, gust) {
    const T = DOD[kind] || DOD[0], mph = gust * MPH;
    let d = 0;
    while (d < 10 && mph >= T[d]) d++;
    return d;
  }
  // the EF rating of a gust (mph thresholds of the EF scale)
  const ef = g => { const m = g * MPH; return m < 65 ? -1 : m < 86 ? 0 : m < 111 ? 1 : m < 136 ? 2 : m < 166 ? 3 : m < 201 ? 4 : 5; };

  // ---------------------------------------------------------------- debris
  // a building near the car went from degree d0 to d1: its pieces off it (b: scenery's footprint record, world
  // origin o, its kind, roof and wall colours)
  const PIECES = [
    { s: [0.42, 0.32, 0.02], k: 0.9 },   // a shingle (light, carried far)
    { s: [2.4, 0.14, 0.04], k: 0.35 },   // a board
    { s: [1.2, 0.6, 0.02], k: 0.6 },     // a panel, sheathing
    { s: [0.5, 0.25, 0.12], k: 0.15 },   // a lump (brick, block)
  ];
  function shed(b, o, d0, d1, kind, roof, wall, near) {
    if (d1 <= d0 || d1 < 2) return;
    const area = Math.abs((b.x1 - b.x0) * (b.y1 - b.y0));
    let n = Math.round(Math.min(90, area / 25) * (d1 - d0) / 4 * (near ? 1 : 0.3));
    n = Math.min(n, MAX - st.debris.length + 200);
    const now = performance.now();
    for (let k = 0; k < n; k++) {
      const P = PIECES[d1 < 4 ? 0 : d1 < 6 ? (Math.random() < 0.6 ? 0 : 2) : Math.floor(Math.random() * 4)];
      const x = o[0] + b.x0 + Math.random() * (b.x1 - b.x0), y = o[1] + b.y0 + Math.random() * (b.y1 - b.y0);
      const z = b.zb + 0.3 + (b.top - b.zb) * (0.5 + 0.5 * Math.random());
      const sc = 0.7 + 0.6 * Math.random();
      st.debris.push({ p: [x, y, z], v: [0, 0, 2 + 6 * Math.random()], a: [Math.random() * 6.3, Math.random() * 6.3, Math.random() * 6.3],
                       w: [(Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12],
                       s: P.s.map(v => v * sc), k: P.k, col: P === PIECES[0] ? roof : P === PIECES[3] ? [0.45, 0.4, 0.36] : Math.random() < 0.5 ? wall : [0.62, 0.5, 0.36],
                       rest: false, t: now });
    }
    while (st.debris.length > MAX) st.debris.shift();
  }
  // the pieces moved on by dt s in wind(x, y) -> [u, v] m/s (10 m), on ground(x, y) m; render.js's instances
  // (st.inst, st.n) from origin o (world m)
  function step(dt, wind, ground, o) {
    st.origin = o;
    const now = performance.now();
    if (!st.debris.length) { st.n = 0; return; }
    st.debris = st.debris.filter(q => !(q.rest && now - q.t > REST_MS));
    const L = st.debris.length;
    if (!st.inst || st.inst.length < L * 12) st.inst = new Float32Array(Math.max(L, 256) * 12);
    const I = st.inst;
    let n = 0;
    for (const q of st.debris) {
      if (!q.rest) {
        const gz = ground(q.p[0], q.p[1]), hz = Math.max(0.5, q.p[2] - gz);
        const w = wind(q.p[0], q.p[1]), f = Math.min(1.6, Math.pow(hz / 10, 0.14));   // the wind with height (log-ish)
        const rx = w[0] * f - q.v[0], ry = w[1] * f - q.v[1], rz = -q.v[2], rs = Math.hypot(rx, ry, rz);
        const K = 0.012 * q.k * rs;                         // drag over mass, per m: light flat pieces most
        q.v[0] += rx * K * dt; q.v[1] += ry * K * dt; q.v[2] += (rz * K - 9.81) * dt;
        // a flat piece in the wind lifts a little (its lift ~ its drag at an angle): the shingles fly
        q.v[2] += Math.min(9.0, 0.004 * q.k * (rx * rx + ry * ry)) * dt;
        for (let i = 0; i < 3; i++) { q.p[i] += q.v[i] * dt; q.a[i] += q.w[i] * dt; }
        const g2 = ground(q.p[0], q.p[1]);
        if (q.p[2] <= g2 + q.s[2] / 2) {
          q.p[2] = g2 + q.s[2] / 2;
          if (Math.hypot(q.v[0], q.v[1]) < 3 && Math.hypot(w[0], w[1]) < 25) { q.rest = true; q.t = now; q.a[1] = (Math.random() - 0.5) * 0.2; q.a[2] = (Math.random() - 0.5) * 0.2; }
          else { q.v[2] = Math.abs(q.v[2]) * 0.2; q.v[0] *= 0.6; q.v[1] *= 0.6; }
        }
      }
      I.set([q.p[0] - o[0], q.p[1] - o[1], q.p[2], q.a[0], q.a[1], q.a[2], q.s[0], q.s[1], q.s[2], q.col[0], q.col[1], q.col[2]], n * 12);
      n++;
    }
    st.n = n;
  }
  return { poll, gustAt, gustOver, degree, ef, shed, step, state: st, DOD };
})();
window.Damage = Damage;
