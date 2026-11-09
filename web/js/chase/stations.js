'use strict';
// Chase mode's gas stations and the way to the nearest one.
//  - Where: OpenStreetMap's fuel stations (amenity=fuel, through /api/volume/fuel: arhtmodelviewer/chase.py), round
//    the car in a FETCH_KM box, fetched again as the car goes on.
//  - What: at each one close to the car, a station laid out off its nearest road: a concrete apron from the road's
//    edge, a canopy on pillars over two pump islands, a price sign on a pole, and behind them a store you can walk
//    into through its door: glass front, a counter with the register, aisles of shelves, coolers along the back
//    wall.  The car drives on the apron (asphalt) and hits the islands, pillars and the store; on foot the walls,
//    shelves and counter stop you.  (OpenStreetMap's own building there, if it has one, is left out of chase
//    mode's buildings: scenery.js setSkip.)
//  - The way: a road graph from roads.js's segments (vertices shared by ways joined, a way's end on another way's
//    side joined to it), Dijkstra from the car to whichever station is nearest by road; the route as a ribbon on the
//    road ahead, a line on the minimap and the next turn in the HUD.
// World metres (x east, y north, z up).  A station's own frame: u along its road, w away from it, from the road's edge.
const Stations = (() => {
  const FETCH_KM = 12, REFETCH_KM = 3.5, MESH_M = 1500, DROP_M = 2200;
  const ROAD_EDGE = 2, MERGE = 55;                          // the apron starts this far past the carriageway's edge, m
  const st = { list: [], centre: null, busy: false, pending: 0, lastTry: 0, gen: 0, err: '', built: new Map(),
               baseH: null, hKey: '', skipGen: 0 };

  // ---------------------------------------------------------------- the list
  async function update(x, y) {
    if (st.busy) return;
    const far = !st.centre || Math.hypot(x - st.centre[0], y - st.centre[1]) > REFETCH_KM * 1000;
    const now = performance.now();
    if (!far && !(st.pending && now - st.lastTry > 6000)) return;
    st.busy = true; st.lastTry = now;
    try {
      const h = FETCH_KM / 2, cx = x / 1000, cy = y / 1000;
      const b = [cx - h, cy - h, cx + h, cy + h].map(v => v.toFixed(3));
      const r = await fetch(`/api/volume/fuel?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}`, { headers: window.ChaseAuth ? window.ChaseAuth() : {} });
      if (!r.ok) throw new Error(r.status === 404 ? 'gas stations need the viewer restarted (new in this version)' : await r.text());
      const d = await r.json();
      st.pending = d.pending;
      const old = new Map(st.list.map(s => [s.id, s]));
      // one station where OpenStreetMap has two points (the forecourt and the shop) within MERGE m: the named one
      const pts = d.fuel.map(([xk, yk, name]) => ({ x: xk * 1000, y: yk * 1000, name: name || '' }))
        .sort((a, b) => (b.name ? 1 : 0) - (a.name ? 1 : 0));
      const keep = [];
      for (const q of pts) if (!keep.some(k => Math.hypot(k.x - q.x, k.y - q.y) < MERGE)) keep.push(q);
      st.list = keep.map(q => {
        const id = `${Math.round(q.x)},${Math.round(q.y)}`;
        return old.get(id) || { id, x: q.x, y: q.y, name: q.name || 'Gas station' };
      });
      st.gen++; st.centre = [x, y]; st.err = '';
    } catch (e) { st.err = e.message; st.pending = st.pending || 0; st.centre = st.centre || [x, y]; }
    st.busy = false;
  }

  // ---------------------------------------------------------------- a station's layout off its road
  const BRANDS = [
    [/shell/i, [0.96, 0.8, 0.1], [0.85, 0.12, 0.1]], [/7.?eleven/i, [0.05, 0.5, 0.3], [0.95, 0.45, 0.1]],
    [/phillips|conoco|76/i, [0.8, 0.1, 0.12], [0.95, 0.95, 0.95]], [/quik ?trip|qt\b/i, [0.75, 0.08, 0.1], [0.95, 0.95, 0.95]],
    [/casey/i, [0.82, 0.12, 0.12], [0.98, 0.85, 0.2]], [/love'?s/i, [0.9, 0.82, 0.1], [0.82, 0.12, 0.12]],
    [/exxon|mobil/i, [0.12, 0.2, 0.6], [0.85, 0.1, 0.12]], [/bp\b|amoco/i, [0.1, 0.55, 0.25], [0.95, 0.85, 0.1]],
    [/valero/i, [0.1, 0.35, 0.6], [0.95, 0.7, 0.1]], [/chevron|texaco/i, [0.1, 0.3, 0.65], [0.85, 0.12, 0.12]],
    [/sinclair/i, [0.1, 0.45, 0.25], [0.95, 0.95, 0.95]], [/marathon|speedway/i, [0.12, 0.25, 0.55], [0.9, 0.15, 0.12]],
    [/kum ?& ?go|kwik/i, [0.85, 0.15, 0.12], [0.95, 0.85, 0.2]], [/oncue|on cue/i, [0.1, 0.3, 0.6], [0.95, 0.6, 0.1]],
  ];
  const PALETTE = [[[0.15, 0.3, 0.6], [0.9, 0.9, 0.9]], [[0.75, 0.12, 0.1], [0.95, 0.95, 0.95]], [[0.12, 0.45, 0.25], [0.95, 0.8, 0.15]],
                   [[0.9, 0.55, 0.1], [0.2, 0.2, 0.25]], [[0.3, 0.3, 0.34], [0.9, 0.2, 0.15]]];
  function brand(name) {
    for (const [re, a, b] of BRANDS) if (re.test(name)) return [a, b];
    let h = 0;
    for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return PALETTE[h % PALETTE.length];
  }
  let rng = 1;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 4294967296; };

  // the station's frame: its road's nearest point, along it (X) and away (Y)
  function frame(s) {
    if (s.L && s.Lgen === Roads.state.gen) return s.L;
    const ns = Roads.state.segs ? nearSeg(s.x, s.y, 160) : null, S = Roads.state.segs;
    const r = ns && { x: ns.x, y: ns.y, cls: S.cls[ns.s], yaw: Math.atan2(S.by[ns.s] - S.ay[ns.s], S.bx[ns.s] - S.ax[ns.s]) };
    let X = [1, 0], Y = [0, 1], O = [s.x - 20, s.y - 20];
    if (r) {
      const dx = s.x - r.x, dy = s.y - r.y, d = Math.hypot(dx, dy);
      const fx = Math.cos(r.yaw), fy = Math.sin(r.yaw);
      Y = d > 1 ? [dx / d, dy / d] : [-fy, fx];
      // straight off the road: Y square to it, on the station's side
      const side = Y[0] * -fy + Y[1] * fx >= 0 ? 1 : -1;
      Y = [-fy * side, fx * side]; X = [Y[1], -Y[0]];
      const e = Roads.HALF[r.cls] + ROAD_EDGE;
      O = [r.x + Y[0] * e, r.y + Y[1] * e];
    }
    s.L = { X, Y, O }; s.Lgen = Roads.state.gen;
    return s.L;
  }
  const toW = (L, u, w) => [L.O[0] + L.X[0] * u + L.Y[0] * w, L.O[1] + L.X[1] * u + L.Y[1] * w];
  const toL = (L, x, y) => { const dx = x - L.O[0], dy = y - L.O[1]; return [dx * L.X[0] + dy * L.X[1], dx * L.Y[0] + dy * L.Y[1]]; };

  // the parts (u0, u1, w0, w1, z0 and z1 above the apron's base, colour; car: stops the car; walk: stops a person)
  const APRON = [-17, 17, 0, 40], STORE = [-10, 10, 25, 37], DOOR = [-1.1, 1.1];
  function build(s, baseH) {
    const L = frame(s), B = CarMesh.box, V = [];
    rng = (Math.round(s.x) * 7919 + Math.round(s.y) * 104729) >>> 0 || 1;
    const [c1, c2] = brand(s.name);
    const hAt = (u, w) => { const p = toW(L, u, w); return baseH(p[0], p[1]); };
    // the store's floor level: the highest ground under it (no store sunk into a slope)
    let z0 = -Infinity;
    for (let u = STORE[0]; u <= STORE[1]; u += 5) for (let w = STORE[2]; w <= STORE[3]; w += 4) z0 = Math.max(z0, hAt(u, w));
    const zC = hAt(0, 12);
    const solids = [];
    const part = (u0, u1, w0, w1, za, zb, col, car, walk) => {
      B(V, u0, u1, w0, w1, za, zb, col);
      if (car || walk) solids.push({ u0, u1, w0, w1, bot: za, top: zb, car, walk });
    };
    // the apron, draped over the ground (4 m cells), concrete with a darker strip of tyre marks
    const A = APRON, N = 4, slab = [0.56, 0.56, 0.54];
    for (let u = A[0]; u < A[1]; u += N) for (let w = A[2]; w < A[3]; w += N) {
      const q = [[u, w], [u + N, w], [u + N, w + N], [u, w + N]].map(([a, b]) => [a, b, (a >= STORE[0] && a <= STORE[1] && b >= STORE[2] && b <= STORE[3] ? z0 : hAt(a, b)) + 0.06]);
      const k = (w >= 6 && w < 18) ? 0.9 : 1, col = slab.map(v => v * k * (0.97 + 0.06 * rand()));
      for (const t of [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]) for (const p of t) V.push(p[0], p[1], p[2], 0, 0, 1, col[0], col[1], col[2]);
    }
    // the canopy over the pumps: pillars, the deck, its band in the brand's colour, its lit underside
    const zt = zC + 5.0;
    const CAN = [-12, 12, 5, 19];
    part(CAN[0], CAN[1], CAN[2], CAN[3], zt, zt + 0.25, [0.92, 0.92, 0.9]);
    part(CAN[0] - 0.1, CAN[1] + 0.1, CAN[2] - 0.1, CAN[3] + 0.1, zt + 0.25, zt + 1.05, c1);
    part(CAN[0] - 0.12, CAN[1] + 0.12, CAN[2] - 0.12, CAN[3] + 0.12, zt + 0.42, zt + 0.62, c2);
    for (let u = -9; u <= 9; u += 6) for (let w = 7; w <= 17; w += 10) {
      part(u - 0.4, u + 0.4, w - 0.4, w + 0.4, zt - 0.06, zt, [11, 11, 10.95]);    // the lights in its underside (lit)
    }
    // two islands of pumps, cars on both sides of each
    for (const w of [9.5, 14.5]) for (const uc of [-5.5, 5.5]) {
      part(uc - 3, uc + 3, w - 0.65, w + 0.65, zC - 0.5, zC + 0.2, [0.75, 0.74, 0.7], true, true);
      for (const du of [-1.6, 1.6]) {
        const u = uc + du;
        part(u - 0.45, u + 0.45, w - 0.3, w + 0.3, zC + 0.2, zC + 1.75, [0.88, 0.88, 0.86], true, true);
        part(u - 0.46, u + 0.46, w - 0.31, w + 0.31, zC + 1.35, zC + 1.75, c1);
        part(u - 0.25, u + 0.25, w - 0.32, w + 0.32, zC + 0.9, zC + 1.2, [0.08, 0.1, 0.12]);   // the screens
      }
      part(uc - 0.22, uc + 0.22, w - 0.22, w + 0.22, zC + 0.2, zt, [0.85, 0.85, 0.83], true, true);   // a pillar
      part(uc - 3.05, uc - 2.75, w - 0.15, w + 0.15, zC + 0.2, zC + 1.0, [0.95, 0.75, 0.1], true, true);   // bollards
      part(uc + 2.75, uc + 3.05, w - 0.15, w + 0.15, zC + 0.2, zC + 1.0, [0.95, 0.75, 0.1], true, true);
    }
    // the price sign by the road
    {
      const zs = hAt(15, 2), wall = [0.95, 0.95, 0.93];
      part(14.8, 15.2, 1.8, 2.2, zs - 0.5, zs + 6.0, [0.4, 0.4, 0.42], true, true);
      part(13.2, 16.8, 1.7, 2.3, zs + 6.0, zs + 8.2, wall);
      part(13.1, 16.9, 1.65, 2.35, zs + 7.4, zs + 8.3, c1);
      for (let i = 0; i < 3; i++) part(13.5, 16.5, 1.62, 2.38, zs + 6.25 + i * 0.37, zs + 6.52 + i * 0.37, [10.95, 10.25, 10.15]);
    }
    // the store
    const S = STORE, T = 0.25, zr = z0 + 4.6, wallC = [0.86, 0.83, 0.77], glassLit = [10.8, 10.82, 10.78], trim = c1;
    const zf = z0 + 0.15;                        // the floor's top
    part(S[0], S[1], S[2], S[3], z0 - 1.2, zf, [0.8, 0.79, 0.76]);        // the floor slab
    part(S[0], S[1], S[3] - T, S[3], z0 - 1.2, zr, wallC, false, true);       // back
    part(S[0], S[0] + T, S[2], S[3], z0 - 1.2, zr, wallC, false, true);       // the ends
    part(S[1] - T, S[1], S[2], S[3], z0 - 1.2, zr, wallC, false, true);
    // the front: brick piers either side, glass between, the door's opening in the middle, a band over it
    const fw = [S[2], S[2] + T];
    part(S[0], S[0] + 1.5, fw[0], fw[1], z0 - 1.2, zr, wallC, false, true);
    part(S[1] - 1.5, S[1], fw[0], fw[1], z0 - 1.2, zr, wallC, false, true);
    for (const [a, b] of [[S[0] + 1.5, DOOR[0]], [DOOR[1], S[1] - 1.5]]) {
      part(a, b, fw[0], fw[1], z0 - 1.2, zf + 0.6, wallC, false, true);
      part(a, b, fw[0] + 0.05, fw[1] - 0.05, zf + 0.6, zf + 2.9, glassLit, false, true);
      for (let u = a + 2.4; u < b - 0.5; u += 2.4) part(u - 0.06, u + 0.06, fw[0] - 0.02, fw[1] + 0.02, zf + 0.6, zf + 2.9, [0.25, 0.25, 0.27]);
    }
    part(S[0] + 1.5, S[1] - 1.5, fw[0], fw[1], zf + 2.9, zr, wallC, false, true);
    part(DOOR[0] - 0.08, DOOR[0], fw[0] - 0.02, fw[1] + 0.02, zf, zf + 2.9, [0.25, 0.25, 0.27]);
    part(DOOR[1], DOOR[1] + 0.08, fw[0] - 0.02, fw[1] + 0.02, zf, zf + 2.9, [0.25, 0.25, 0.27]);
    part(S[0] - 0.1, S[1] + 0.1, S[2] - 0.35, S[2] + 0.05, zf + 3.2, zr + 0.2, trim);      // the sign band
    part(S[0] - 0.3, S[1] + 0.3, S[2] - 0.3, S[3] + 0.3, zr, zr + 0.35, [0.45, 0.45, 0.47]);  // the roof
    part(S[0], S[1], S[2] + T, S[3] - T, zr - 0.9, zr - 0.8, [0.93, 0.93, 0.9]);          // the ceiling
    for (let u = S[0] + 3; u < S[1] - 1; u += 4) for (let w = S[2] + 3; w < S[3] - 1; w += 4)
      part(u - 0.6, u + 0.6, w - 0.25, w + 0.25, zr - 0.93, zr - 0.9, [11, 11, 10.97]);
    // the counter and register, left of the door, a walkway behind it
    part(S[0] + 1.6, S[0] + 6.5, S[2] + 1.6, S[2] + 2.5, zf, zf + 1.0, [0.5, 0.36, 0.24], false, true);
    part(S[0] + 1.55, S[0] + 6.55, S[2] + 1.55, S[2] + 2.55, zf + 1.0, zf + 1.05, [0.2, 0.2, 0.2]);
    part(S[0] + 3.5, S[0] + 4.0, S[2] + 1.9, S[2] + 2.3, zf + 1.05, zf + 1.4, [0.1, 0.1, 0.12]);
    part(S[0] + 1.6, S[0] + 6.5, S[2] + 4.2, S[2] + 4.5, zf, zf + 2.4, [0.3, 0.3, 0.32], false, true);   // cigarettes
    // the aisles: gondolas of shelves, goods in bright packs on both faces
    const goods = [[0.85, 0.15, 0.12], [0.95, 0.75, 0.1], [0.15, 0.4, 0.8], [0.2, 0.6, 0.25], [0.9, 0.45, 0.1], [0.55, 0.2, 0.6], [0.95, 0.95, 0.9]];
    for (const w of [S[2] + 5.5, S[2] + 7.8, S[2] + 10.1]) {
      const u0 = S[0] + 8.2, u1 = S[1] - 2.2;
      part(u0, u1, w - 0.45, w + 0.45, zf, zf + 1.65, [0.82, 0.82, 0.84], false, true);
      for (let lv = 0; lv < 3; lv++) {
        const zl = zf + 0.15 + lv * 0.5;
        for (let u = u0 + 0.1; u < u1 - 0.3; u += 0.42 + 0.3 * rand()) {
          const ue = Math.min(u1 - 0.1, u + 0.22 + 0.16 * rand()), col = goods[Math.floor(rand() * goods.length)], ht = 0.16 + 0.18 * rand();
          B(V, u, ue, w - 0.5, w - 0.45, zl, zl + ht, col);
          B(V, u, ue, w + 0.45, w + 0.5, zl, zl + ht, goods[Math.floor(rand() * goods.length)]);
        }
      }
    }
    // the coolers along the back: glass doors lit from inside
    {
      const w0 = S[3] - T - 0.9, w1 = S[3] - T;
      part(S[0] + 1, S[1] - 1, w0, w1, zf, zf + 2.2, [0.85, 0.85, 0.87], false, true);
      for (let u = S[0] + 1.1; u < S[1] - 1.6; u += 1.0) {
        part(u, u + 0.9, w0 - 0.04, w0, zf + 0.15, zf + 2.05, [10.75, 10.88, 10.95]);
        for (let lv = 0; lv < 4; lv++) B(V, u + 0.1, u + 0.8, w0 - 0.03, w0 - 0.01, zf + 0.35 + lv * 0.45, zf + 0.5 + lv * 0.45, goods[Math.floor(rand() * goods.length)]);
      }
    }
    // the car can't drive into the store at all
    solids.push({ u0: S[0], u1: S[1], w0: S[2], w1: S[3], bot: z0 - 1.2, top: zr + 0.35, car: true, walk: false });
    return { L, data: new Float32Array(V), solids, z0, zf, zC, hAt, s };
  }

  // the stations close to the car built, far ones dropped (baseH(x, y): the ground's height there; hKey: changes
  // when it does, e.g. the detailed map on or off)
  function near(x, y, baseH, hKey) {
    if (hKey !== st.hKey) { for (const b of st.built.values()) if (window.Render && Render.dropPart) Render.dropPart(b.data); st.built.clear(); st.hKey = hKey; }
    st.baseH = baseH;
    let changed = false;
    for (const s of st.list) {
      const d = Math.hypot(s.x - x, s.y - y);
      if (d < MESH_M && !st.built.has(s.id) && Roads.state.segs) {
        if (!Number.isFinite(baseH(s.x, s.y))) continue;
        st.built.set(s.id, build(s, baseH)); changed = true;
      }
    }
    for (const [id, b] of st.built) {
      if (Math.hypot(b.s.x - x, b.s.y - y) > DROP_M || !st.list.includes(b.s)) {
        if (window.Render && Render.dropPart) Render.dropPart(b.data);
        st.built.delete(id); changed = true;
      }
    }
    if (changed) st.skipGen++;
    return changed;
  }
  // render.js parts for the built ones
  function parts() {
    const out = [];
    for (const b of st.built.values()) {
      const L = b.L;
      out.push({ data: b.data, M: [[L.X[0], L.X[1], 0], [L.Y[0], L.Y[1], 0], [0, 0, 1]], p: [L.O[0], L.O[1], 0] });
    }
    return out;
  }
  function inBox(L, x, y, r) { const [u, w] = toL(L, x, y); return u > r[0] && u < r[1] && w > r[2] && w < r[3]; }
  // the apron's or store floor's height at x, y (or null): driven and walked on, asphalt to the tyres
  function apron(x, y) {
    for (const b of st.built.values()) {
      const [u, w] = toL(b.L, x, y);
      if (u < APRON[0] || u > APRON[1] || w < APRON[2] || w > APRON[3]) continue;
      if (u > STORE[0] && u < STORE[1] && w > STORE[2] && w < STORE[3]) return b.zf;
      return b.hAt(u, w) + 0.06;
    }
    return null;
  }
  // is x, y on a station's ground (OpenStreetMap's own building there is left out)
  function covers(x, y) {
    for (const b of st.built.values()) if (inBox(b.L, x, y, [APRON[0] + 2, APRON[1] - 2, 4, APRON[3] - 2])) return true;
    return false;
  }
  // a part's push on a circle (radius rad) at x, y: its local [du, dw] out, or null
  function pushBox(sb, u, w, rad) {
    const cu = Math.max(sb.u0, Math.min(sb.u1, u)), cw = Math.max(sb.w0, Math.min(sb.w1, w));
    const du = u - cu, dw = w - cw, d = Math.hypot(du, dw);
    if (d > 1e-6) return d < rad ? [du / d * (rad - d), dw / d * (rad - d)] : null;
    const pen = [u - sb.u0 + rad, sb.u1 - u + rad, w - sb.w0 + rad, sb.w1 - w + rad], i = pen.indexOf(Math.min(...pen));
    return [[-pen[0], 0], [pen[1], 0], [0, -pen[2]], [0, pen[3]]][i];
  }
  // a person (circle) against walls, shelves, islands: the world push [dx, dy] or null
  function push(x, y, z, rad) {
    for (const b of st.built.values()) {
      if (!inBox(b.L, x, y, [APRON[0] - 3, APRON[1] + 3, APRON[2] - 3, APRON[3] + 3])) continue;
      const [u, w] = toL(b.L, x, y);
      let pu = 0, pw = 0, hit = false;
      for (const sb of b.solids) {
        if (!sb.walk || z > sb.top || z + 1.2 < sb.bot) continue;    // (z: the person's middle)
        const p = pushBox(sb, u + pu, w + pw, rad);
        if (p) { pu += p[0]; pw += p[1]; hit = true; }
      }
      if (hit) return [b.L.X[0] * pu + b.L.Y[0] * pw, b.L.X[1] * pu + b.L.Y[1] * pw];
    }
    return null;
  }
  // the car's body points (carphys solid(x, y, z)): { n (world, out), d (m in) } or null
  function solid(x, y, z) {
    for (const b of st.built.values()) {
      if (!inBox(b.L, x, y, [APRON[0] - 1, APRON[1] + 1, APRON[2] - 1, APRON[3] + 1])) continue;
      const [u, w] = toL(b.L, x, y);
      for (const sb of b.solids) {
        if (!sb.car || z > sb.top || u < sb.u0 || u > sb.u1 || w < sb.w0 || w > sb.w1) continue;
        const pen = [u - sb.u0, sb.u1 - u, w - sb.w0, sb.w1 - w], i = pen.indexOf(Math.min(...pen));
        const nl = [[-1, 0], [1, 0], [0, -1], [0, 1]][i];
        return { n: [b.L.X[0] * nl[0] + b.L.Y[0] * nl[1], b.L.X[1] * nl[0] + b.L.Y[1] * nl[1], 0], d: Math.min(pen[i], 1.5) };
      }
    }
    return null;
  }
  // where the route ends: the apron in front of the pumps
  const target = s => { const b = st.built.get(s.id); const L = b ? b.L : frame(s); return toW(L, 0, 4); };

  // ---------------------------------------------------------------- the road graph
  const CLASS_COST = [0.8, 0.82, 0.88, 0.92, 0.96, 1.05, 1.1, 1.6];   // a little for the bigger roads
  let G = null;
  function graph() {
    const R = Roads.state, S = R.segs;
    if (!S) return null;
    if (G && G.gen === R.gen) return G;
    const cellN = new Map(), xs = [], ys = [], adj = [];
    const node = (x, y) => {
      const i = Math.round(x / 2), j = Math.round(y / 2);
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
        const l = cellN.get(`${i + a},${j + b}`);
        if (l) for (const n of l) if (Math.abs(xs[n] - x) < 2 && Math.abs(ys[n] - y) < 2) return n;
      }
      const n = xs.length;
      xs.push(x); ys.push(y); adj.push([]);
      const k = `${i},${j}`;
      let l = cellN.get(k);
      if (!l) cellN.set(k, l = []);
      l.push(n);
      return n;
    };
    const segA = new Int32Array(S.n), segB = new Int32Array(S.n);
    const link = (a, b, len, cls) => { if (a === b) return; adj[a].push(b, len * CLASS_COST[cls], cls); adj[b].push(a, len * CLASS_COST[cls], cls); };
    for (let s = 0; s < S.n; s++) {
      const a = node(S.ax[s], S.ay[s]), b = node(S.bx[s], S.by[s]);
      segA[s] = a; segB[s] = b;
      link(a, b, Math.hypot(S.bx[s] - S.ax[s], S.by[s] - S.ay[s]), S.cls[s]);
    }
    // a way that ends on another's side (no shared vertex): joined to that one's nearest vertex
    const ends = [];
    for (let n = 0; n < xs.length; n++) if (adj[n].length === 3) ends.push(n);
    for (const n of ends) {
      const x = xs[n], y = ys[n];
      let best = null, bd = Infinity;
      for (const s of segsNear(x, y, 12)) {
        if (segA[s] === n || segB[s] === n) continue;
        const ax = S.ax[s], ay = S.ay[s], dx = S.bx[s] - ax, dy = S.by[s] - ay, l2 = dx * dx + dy * dy;
        const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
        const d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
        if (d < Roads.HALF[S.cls[s]] + 4 && d < bd) { bd = d; best = t < 0.5 ? segA[s] : segB[s]; }
      }
      if (best !== null && best !== n) link(n, best, Math.max(1, Math.hypot(xs[best] - x, ys[best] - y)), 6);
    }
    G = { gen: R.gen, xs, ys, adj, segA, segB };
    return G;
  }
  // segments with a cell within r m of x, y (roads.js's spatial hash)
  function segsNear(x, y, r) {
    const R = Roads.state, C = 32, out = new Set();
    for (let i = Math.floor((x - r) / C); i <= Math.floor((x + r) / C); i++)
      for (let j = Math.floor((y - r) / C); j <= Math.floor((y + r) / C); j++) {
        const l = R.grid.get(i * 73856093 ^ j * 19349663);
        if (l) for (const s of l) out.add(s);
      }
    return out;
  }
  // the nearest segment to x, y within r m: { s, t, d, x, y }
  function nearSeg(x, y, r) {
    const S = Roads.state.segs;
    let best = null;
    for (const s of segsNear(x, y, r)) {
      const ax = S.ax[s], ay = S.ay[s], dx = S.bx[s] - ax, dy = S.by[s] - ay, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
      const px = ax + t * dx, py = ay + t * dy, d = Math.hypot(x - px, y - py);
      if (d <= r && (!best || d < best.d)) best = { s, t, d, x: px, y: py, len: Math.sqrt(l2) };
    }
    return best;
  }

  // a binary heap of [cost, node]
  class Heap {
    constructor() { this.c = []; this.n = []; }
    get size() { return this.c.length; }
    push(c, n) {
      const C = this.c, N = this.n;
      let i = C.length; C.push(c); N.push(n);
      while (i > 0) { const p = (i - 1) >> 1; if (C[p] <= c) break; C[i] = C[p]; N[i] = N[p]; i = p; }
      C[i] = c; N[i] = n;
    }
    pop() {
      const C = this.c, N = this.n, top = N[0], tc = C[0], lc = C.pop(), ln = N.pop();
      if (C.length) {
        let i = 0;
        for (;;) {
          let k = 2 * i + 1;
          if (k >= C.length) break;
          if (k + 1 < C.length && C[k + 1] < C[k]) k++;
          if (C[k] >= lc) break;
          C[i] = C[k]; N[i] = N[k]; i = k;
        }
        C[i] = lc; N[i] = ln;
      }
      return [tc, top];
    }
  }

  // ---------------------------------------------------------------- the route
  const route = { on: false, pts: null, cum: null, turns: null, dest: null, at: 0, gen: -1, lastCalc: 0, msg: '', mesh: null, meshAt: -1, arrived: false };

  // the way from x, y to the station nearest by road
  function plan(x, y) {
    const g = graph(), S = Roads.state.segs;
    if (!g) return { err: 'no roads yet' };
    const from = nearSeg(x, y, 300);
    if (!from) return { err: 'no road near you' };
    // the stations' ends: their road's nearest segment, both its ends
    const goals = new Map();
    for (const s of st.list) {
      const t = target(s), sg = nearSeg(t[0], t[1], 200);
      if (!sg) continue;
      const tail = Math.hypot(t[0] - sg.x, t[1] - sg.y);
      for (const [n, along] of [[g.segA[sg.s], sg.t * sg.len], [g.segB[sg.s], (1 - sg.t) * sg.len]]) {
        const c = along + tail;
        const e = goals.get(n);
        if (!e || c < e.c) goals.set(n, { c, s, sg, t });
      }
    }
    if (!goals.size) return { err: st.list.length ? 'no station on the roads loaded so far' : 'no gas station in the area yet' };
    const N = g.xs.length, dist = new Float64Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1), done = new Uint8Array(N);
    const H = new Heap();
    const a = g.segA[from.s], b = g.segB[from.s];
    dist[a] = from.t * from.len; dist[b] = (1 - from.t) * from.len;
    H.push(dist[a], a); H.push(dist[b], b);
    let best = null, bestC = Infinity;
    while (H.size) {
      const [c, n] = H.pop();
      if (done[n] || c > dist[n]) continue;
      if (c >= bestC) break;
      done[n] = 1;
      const gl = goals.get(n);
      if (gl && c + gl.c < bestC) { bestC = c + gl.c; best = { n, gl }; }
      const A = g.adj[n];
      for (let k = 0; k < A.length; k += 3) {
        const m = A[k], nc = c + A[k + 1];
        if (nc < dist[m]) { dist[m] = nc; prev[m] = n; H.push(nc, m); }
      }
    }
    if (!best) return { err: 'no way by road to a station' };
    const nodes = [];
    for (let n = best.n; n >= 0; n = prev[n]) nodes.push(n);
    nodes.reverse();
    const pts = [[from.x, from.y]];
    for (const n of nodes) pts.push([g.xs[n], g.ys[n]]);
    pts.push([best.gl.sg.x, best.gl.sg.y], best.gl.t);
    // the turns at junctions (a node with 3 or more ways), from the bearings 15 m either side
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const at = (d) => {
      d = Math.max(0, Math.min(cum[cum.length - 1], d));
      let i = 1;
      while (i < cum.length - 1 && cum[i] < d) i++;
      const f = (d - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1]);
      return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f];
    };
    const turns = [];
    for (let j = 0; j < nodes.length; j++) {
      const n = nodes[j], pi = j + 1, i = j + 1;          // (pts[0] is where we join the road)
      if (g.adj[n].length < 9) continue;                  // fewer than three ways there
      const p = pts[pi], p0 = at(cum[pi] - 15), p1 = at(cum[pi] + 15);
      const h0 = Math.atan2(p[1] - p0[1], p[0] - p0[0]), h1 = Math.atan2(p1[1] - p[1], p1[0] - p[0]);
      const da = Math.atan2(Math.sin(h1 - h0), Math.cos(h1 - h0));
      if (Math.abs(da) < 0.5) continue;
      // the class of the way it turns onto
      let cls = 6;
      const nx = nodes[i], A = g.adj[n];
      for (let k = 0; k < A.length; k += 3) if (A[k] === nx) cls = A[k + 2];
      const last = turns[turns.length - 1];
      if (last && cum[pi] - last.d < 20) continue;
      turns.push({ d: cum[pi], dir: da > 0 ? 'left' : 'right', sharp: Math.abs(da) > 2.1, slight: Math.abs(da) < 0.8, cls, p });
    }
    // which side the station is on, coming along the last stretch
    const end = pts[pts.length - 1], rp = pts[pts.length - 2], q = pts[pts.length - 3] || pts[0];
    const side = (rp[0] - q[0]) * (end[1] - rp[1]) - (rp[1] - q[1]) * (end[0] - rp[0]) > 0 ? 'left' : 'right';
    return { pts, cum, turns, dest: best.gl.s, side, len: cum[cum.length - 1] };
  }

  // each half second: plan again when off the route, the roads changed or the stations did; how far along it
  function follow(x, y) {
    if (!route.on) return;
    const now = performance.now();
    let off = Infinity;
    if (route.pts) {
      // nearest point on the route, from a little behind where we were along it
      const P = route.pts;
      let bd = Infinity, bi = route.at;
      for (let i = Math.max(1, route.atI - 5 || 1); i < P.length; i++) {
        const ax = P[i - 1][0], ay = P[i - 1][1], dx = P[i][0] - ax, dy = P[i][1] - ay, l2 = dx * dx + dy * dy;
        const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
        const d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
        if (d < bd) { bd = d; bi = route.cum[i - 1] + t * Math.sqrt(l2); route.atI = i; }
        if (route.cum[i - 1] > route.at + 600) break;
      }
      off = bd; route.at = bi;
    }
    const stale = route.gen !== Roads.state.gen || route.sgen !== st.gen;
    if ((off > 40 || stale || !route.pts) && now - route.lastCalc > (route.pts ? 2500 : 1500)) {
      route.lastCalc = now;
      const t0 = performance.now(), r = plan(x, y);
      route.ms = performance.now() - t0;
      route.gen = Roads.state.gen; route.sgen = st.gen;
      if (r.err) { route.msg = r.err; if (off > 40) route.pts = null; }
      else { Object.assign(route, r, { at: 0, atI: 1, msg: '', meshAt: -1e9 }); }
    }
    if (route.pts && route.len - route.at < 25) {
      route.arrived = true;
    }
  }
  function toggle() {
    route.on = !route.on;
    if (!route.on) { clear(); } else { route.lastCalc = 0; route.arrived = false; route.pts = null; route.msg = 'finding the nearest gas station…'; }
    return route.on;
  }
  function clear() {
    if (route.mesh && Render.dropPart) Render.dropPart(route.mesh.data);
    Object.assign(route, { on: false, pts: null, mesh: null, turns: null, dest: null, msg: '', arrived: false });
  }
  // the next thing to do: { dist m, dir 'left'|'right'|'arrive', sharp, slight, cls, total m, dest }
  function next() {
    if (!route.pts) return null;
    const left = route.len - route.at;
    for (const t of route.turns) if (t.d > route.at + 5) return { dist: t.d - route.at, dir: t.dir, sharp: t.sharp, slight: t.slight, cls: t.cls, total: left, dest: route.dest, side: route.side };
    return { dist: left, dir: 'arrive', total: left, dest: route.dest, side: route.side };
  }
  // the ribbon on the road ahead (render.js part): 1.3 km of it, rebuilt every 40 m of progress; h(x, y) the road's top
  function ribbon(h) {
    if (!route.pts) return null;
    if (route.mesh && Math.abs(route.at - route.meshAt) < 40 && route.meshKey === route.gen + '/' + route.lastCalc) return route.mesh;
    if (route.mesh && Render.dropPart) Render.dropPart(route.mesh.data);
    const P = route.pts, C = route.cum, d0 = Math.max(0, route.at - 30), d1 = Math.min(route.len, route.at + 1300);
    const o = [Math.round(P[0][0]), Math.round(P[0][1])], V = [], hw = 0.5, STEP = 4;
    const at = (d) => {
      let i = 1;
      while (i < C.length - 1 && C[i] < d) i++;
      const f = (d - C[i - 1]) / Math.max(1e-6, C[i] - C[i - 1]);
      return [P[i - 1][0] + (P[i][0] - P[i - 1][0]) * f, P[i - 1][1] + (P[i][1] - P[i - 1][1]) * f];
    };
    let a = at(d0), k = 0;
    for (let d = d0 + STEP; d <= d1 + 1e-6; d += STEP, k++) {
      const b = at(Math.min(d, d1)), dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy);
      if (l > 0.2) {
        const lx = -dy / l * hw, ly = dx / l * hw;
        const za = h(a[0], a[1]) + 0.12, zb = h(b[0], b[1]) + 0.12;
        const col = (Math.floor(d / 8) % 2) ? [0.15, 0.85, 1.0] : [0.55, 0.95, 1.0];
        const q = [[a[0] - lx, a[1] - ly, za], [b[0] - lx, b[1] - ly, zb], [b[0] + lx, b[1] + ly, zb], [a[0] + lx, a[1] + ly, za]];
        for (const t of [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]) for (const p of t) V.push(p[0] - o[0], p[1] - o[1], p[2], 0, 0, 1, col[0], col[1], col[2]);
      }
      a = b;
    }
    route.mesh = { data: new Float32Array(V), M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], p: [o[0], o[1], 0] };
    route.meshAt = route.at; route.meshKey = route.gen + '/' + route.lastCalc;
    return route.mesh;
  }

  return { update, near, parts, apron, covers, push, solid, toggle, clear, follow, next, ribbon, plan, route, state: st, target,
         };
})();
window.Stations = Stations;
