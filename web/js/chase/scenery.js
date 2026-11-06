'use strict';
// Chase mode's scenery, made on this computer from real data:
//  - trees: wherever ESA WorldCover has tree cover (thinner in shrubland, wetland, yards in town, the odd one in
//    grassland) around the car, a low-poly trunk and lumpy crown each, drawn instanced, every tree out to
//    NEAR m then fewer and bigger with distance; none on the roads or in buildings;
//  - buildings: OpenStreetMap's footprints (chase.py through /api/volume/buildings): small ones as houses and sheds
//    (their footprint's best rectangle, walls and a gable roof), the rest extruded to their height with a flat roof;
//  - both are solid: the car hits trunks and walls (solid(x, y, z) for carphys.js).
// Positions are world metres; meshes are stored relative to an origin near the car.
const Scenery = (() => {
  const NEAR_MESH = 650, NEAR = 450, TREE_KM = 3.4, CELL = 32, B_FETCH_KM = 7, B_REFETCH_KM = 2.2;
  const PER_M2 = { 10: 1 / 38, 20: 1 / 220, 30: 1 / 2500, 50: 1 / 450, 60: 1 / 6000, 90: 1 / 160, 95: 1 / 60, 100: 1 / 4000 };
  const st = {
    trees: null, treeN: 0, treeOrigin: [0, 0], treeKey: '', treeGen: 0, treeAt: 0, trunks: new Map(),
    bOrigin: [0, 0], bMesh: null, bGen: 0, bCentre: null, bBusy: false, bPending: 0, bLast: 0, bN: -1, polys: [], bGrid: new Map(),
    budget: 70000,
  };
  const key = (i, j) => i * 73856093 ^ j * 19349663;
  const hash = (a, b, c = 0) => {
    let h = (Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263) + Math.imul(c | 0, 2147483647)) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  };

  // ---------------------------------------------------------------- buildings
  function pip(x, y, P) {
    let inside = false;
    for (let i = 0, j = P.length - 2; i < P.length; j = i, i += 2) {
      const xi = P[i], yi = P[i + 1], xj = P[j], yj = P[j + 1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  // the smallest rectangle around a small footprint: its centre, axes and half sizes
  function rect(P) {
    let best = null;
    for (let i = 0; i < P.length; i += 2) {
      const j = (i + 2) % P.length, ex = P[j] - P[i], ey = P[j + 1] - P[i + 1], l = Math.hypot(ex, ey);
      if (l < 0.5) continue;
      const ux = ex / l, uy = ey / l;
      let a0 = 1e9, a1 = -1e9, b0 = 1e9, b1 = -1e9;
      for (let k = 0; k < P.length; k += 2) {
        const a = P[k] * ux + P[k + 1] * uy, b = -P[k] * uy + P[k + 1] * ux;
        a0 = Math.min(a0, a); a1 = Math.max(a1, a); b0 = Math.min(b0, b); b1 = Math.max(b1, b);
      }
      const area = (a1 - a0) * (b1 - b0);
      if (!best || area < best.area) {
        const ca = (a0 + a1) / 2, cb = (b0 + b1) / 2;
        best = { area, cx: ca * ux - cb * uy, cy: ca * uy + cb * ux, ux, uy, ha: (a1 - a0) / 2, hb: (b1 - b0) / 2 };
      }
    }
    return best;
  }
  // ear clipping of a simple counter-clockwise polygon: triangle index triples
  function earcut(P) {
    const n = P.length / 2, idx = [...Array(n).keys()], out = [];
    const cross = (a, b, c) => (P[2 * b] - P[2 * a]) * (P[2 * c + 1] - P[2 * a + 1]) - (P[2 * b + 1] - P[2 * a + 1]) * (P[2 * c] - P[2 * a]);
    let guard = 0;
    while (idx.length > 3 && guard++ < 4 * n * n) {
      let cut = false;
      for (let i = 0; i < idx.length; i++) {
        const a = idx[(i + idx.length - 1) % idx.length], b = idx[i], c = idx[(i + 1) % idx.length];
        if (cross(a, b, c) <= 1e-9) continue;
        let ear = true;
        for (const p of idx) {
          if (p === a || p === b || p === c) continue;
          if (cross(a, b, p) >= 0 && cross(b, c, p) >= 0 && cross(c, a, p) >= 0) { ear = false; break; }
        }
        if (!ear) continue;
        out.push(a, b, c); idx.splice(i, 1); cut = true; break;
      }
      if (!cut) break;
    }
    if (idx.length === 3) out.push(...idx);
    return out;
  }
  const WALLS = [[232, 226, 210], [214, 205, 186], [190, 176, 150], [236, 236, 232], [176, 184, 190], [200, 186, 168], [160, 120, 96], [222, 214, 196]];
  const ROOFS = [[62, 62, 66], [80, 72, 66], [96, 64, 52], [48, 50, 54], [110, 104, 96], [70, 80, 92]];
  const srgb = c => c.map(v => v / 255);

  function buildBuildings(list, origin, terrain) {
    const V = [], polys = [], grid = new Map();
    const tri = (a, b, c, n, col) => { for (const p of [a, b, c]) V.push(p[0], p[1], p[2], n[0], n[1], n[2], col[0], col[1], col[2]); };
    const quad = (a, b, c, d, n, col) => { tri(a, b, c, n, col); tri(a, c, d, n, col); };
    const norm = (a, b, c) => {
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]], l = Math.hypot(...n) || 1;
      return n.map(x => x / l);
    };
    let id = 0;
    for (const [h, kind, xy] of list) {
      const P = [];
      for (let i = 0; i < xy.length; i += 2) P.push(xy[i] * 1000 - origin[0], xy[i + 1] * 1000 - origin[1]);
      if (P.length >= 4 && Math.hypot(P[0] - P[P.length - 2], P[1] - P[P.length - 1]) < 0.05) P.splice(P.length - 2, 2);
      if (P.length < 6) continue;
      let A = 0;
      for (let i = 0, j = P.length - 2; i < P.length; j = i, i += 2) A += P[j] * P[i + 1] - P[i] * P[j + 1];
      A /= 2;
      if (Math.abs(A) < 6) continue;
      if (A < 0) { const Q = []; for (let i = P.length - 2; i >= 0; i -= 2) Q.push(P[i], P[i + 1]); P.splice(0, P.length, ...Q); }
      const s = hash(Math.round(P[0] * 10), Math.round(P[1] * 10), 3);
      let zb = 1e9;
      for (let i = 0; i < P.length; i += 2) zb = Math.min(zb, terrain(P[i] + origin[0], P[i + 1] + origin[1]));
      zb -= 0.3;
      const barn = kind === 2 && Math.abs(A) < 1500 && s < 0.45;
      const wall = srgb(barn ? [150, 42, 34] : kind === 2 ? [178, 182, 186] : kind === 3 ? [184, 176, 166] : kind === 1 ? WALLS[Math.floor(s * 3) + 4] || WALLS[5] : WALLS[Math.floor(s * WALLS.length)]);
      const roof = srgb(kind === 2 && !barn ? [150, 156, 162] : ROOFS[Math.floor(hash(id, 9) * ROOFS.length)]);
      let top = zb + 0.3 + h;
      const R = (kind === 0 || kind === 4 || barn) && Math.abs(A) < (barn ? 1500 : 450) ? rect(P) : null;
      if (R && R.ha > 0.8 && R.hb > 0.8) {
        // a house / shed / barn: walls up to the eaves, a gable roof along the long side
        const along = R.ha >= R.hb, la = along ? R.ha : R.hb, lb = along ? R.hb : R.ha;
        const ux = along ? R.ux : -R.uy, uy = along ? R.uy : R.ux, vx = -uy, vy = ux;
        const rise = Math.min(lb * (kind === 4 ? 0.35 : 0.6), kind === 4 ? 1.0 : 3.2), eave = Math.max(2.3, h - rise);
        const ze = zb + 0.3 + eave, zr = ze + rise, ov = kind === 4 ? 0.15 : 0.45;
        const C = (a, b, z) => [R.cx + ux * a + vx * b, R.cy + uy * a + vy * b, z];
        const c0 = C(-la, -lb, zb), c1 = C(la, -lb, zb), c2 = C(la, lb, zb), c3 = C(-la, lb, zb);
        const e0 = C(-la, -lb, ze), e1 = C(la, -lb, ze), e2 = C(la, lb, ze), e3 = C(-la, lb, ze);
        quad(c0, c1, e1, e0, [vx * -1, vy * -1, 0], wall);
        quad(c2, c3, e3, e2, [vx, vy, 0], wall);
        quad(c1, c2, e2, e1, [ux, uy, 0], wall);
        quad(c3, c0, e0, e3, [-ux, -uy, 0], wall);
        const g0 = C(-la, 0, zr), g1 = C(la, 0, zr);
        tri(e1, e2, g1, [ux, uy, 0], wall); tri(e3, e0, g0, [-ux, -uy, 0], wall);
        const o0 = C(-la - ov, -lb - ov, ze - ov * rise / lb), o1 = C(la + ov, -lb - ov, ze - ov * rise / lb);
        const o2 = C(la + ov, lb + ov, ze - ov * rise / lb), o3 = C(-la - ov, lb + ov, ze - ov * rise / lb);
        const r0 = C(-la - ov, 0, zr), r1 = C(la + ov, 0, zr);
        const upn = n => (n[2] < 0 ? n.map(v => -v) : n);
        quad(o0, o1, r1, r0, upn(norm(o0, o1, r1)), roof);
        quad(o2, o3, r0, r1, upn(norm(o2, o3, r0)), roof);
        top = zr;
      } else {
        // walls round the footprint, a flat roof (with a parapet colour band on big ones)
        const zt = zb + 0.3 + h;
        for (let i = 0; i < P.length; i += 2) {
          const j = (i + 2) % P.length, ex = P[j] - P[i], ey = P[j + 1] - P[i + 1], l = Math.hypot(ex, ey) || 1;
          const n = [ey / l, -ex / l, 0];
          quad([P[i], P[i + 1], zb], [P[j], P[j + 1], zb], [P[j], P[j + 1], zt], [P[i], P[i + 1], zt], n, wall);
        }
        const T = earcut(P);
        for (let k = 0; k < T.length; k += 3) {
          const a = [P[2 * T[k]], P[2 * T[k] + 1], zt], b = [P[2 * T[k + 1]], P[2 * T[k + 1] + 1], zt], c = [P[2 * T[k + 2]], P[2 * T[k + 2] + 1], zt];
          tri(a, b, c, [0, 0, 1], kind === 0 ? roof : srgb([128, 128, 124]));
        }
        top = zt;
      }
      // for the car: the footprint, hashed
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      for (let i = 0; i < P.length; i += 2) { x0 = Math.min(x0, P[i]); x1 = Math.max(x1, P[i]); y0 = Math.min(y0, P[i + 1]); y1 = Math.max(y1, P[i + 1]); }
      const b = { P, top, zb, x0, x1, y0, y1 };
      polys.push(b);
      for (let i = Math.floor(x0 / CELL); i <= Math.floor(x1 / CELL); i++) for (let j = Math.floor(y0 / CELL); j <= Math.floor(y1 / CELL); j++) {
        const k = key(i, j);
        let l = grid.get(k);
        if (!l) grid.set(k, l = []);
        l.push(b);
      }
      id++;
    }
    return { mesh: new Float32Array(V), polys, grid };
  }

  async function updateBuildings(x, y, terrain, done) {
    if (st.bBusy) return;
    const far = !st.bCentre || Math.hypot(x - st.bCentre[0], y - st.bCentre[1]) > B_REFETCH_KM * 1000;
    const now = performance.now();
    if (!far && !(st.bPending && now - st.bLast > 7000)) return;
    st.bBusy = true; st.bLast = now;
    try {
      const h = B_FETCH_KM / 2, cx = x / 1000, cy = y / 1000;
      const b = [cx - h, cy - h, cx + h, cy + h].map(v => v.toFixed(3));
      const r = await fetch(`/api/volume/buildings?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}`, { headers: window.ChaseAuth ? window.ChaseAuth() : {} });
      if (!r.ok) throw new Error(r.status === 404 ? 'buildings need the viewer restarted (new in this version)' : await r.text());
      const d = await r.json();
      st.bPending = d.pending;
      if (far || d.b.length !== st.bN) {
        const origin = [Math.round(x), Math.round(y)];
        const out = buildBuildings(d.b, origin, terrain);
        Object.assign(st, { bOrigin: origin, bMesh: out.mesh, polys: out.polys, bGrid: out.grid, bN: d.b.length });
        st.bGen++;
        st.treeKey = '';                       // the trees again, out of the new buildings
        if (done) done();
      }
      st.bCentre = [x, y]; st.bErr = '';
    } catch (e) { st.bErr = e.message; st.bPending = st.bPending || 0; st.bCentre = st.bCentre || [x, y]; }
    st.bBusy = false;
  }
  function building(x, y) {
    const lx = x - st.bOrigin[0], ly = y - st.bOrigin[1];
    const l = st.bGrid.get(key(Math.floor(lx / CELL), Math.floor(ly / CELL)));
    if (!l) return null;
    for (const b of l) if (lx >= b.x0 && lx <= b.x1 && ly >= b.y0 && ly <= b.y1 && pip(lx, ly, b.P)) return { b, lx, ly };
    return null;
  }

  // ---------------------------------------------------------------- trees
  // around the car (world m), out of the close-up land cover (Land.cover); terrain(x, y) m
  function updateTrees(x, y, terrain, budget) {
    const lc = Land.cover();
    if (!lc) return false;
    const now = performance.now();
    const k = `${lc.box.join(',')}/${st.bGen}/${Roads.state.gen}/${budget}`;
    const moved = Math.hypot(x - st.treeOrigin[0], y - st.treeOrigin[1]) > 900;
    if ((k === st.treeKey && !moved) || now - st.treeAt < 2500) return false;
    st.treeKey = k; st.treeAt = now;
    const origin = [Math.round(x), Math.round(y)];
    const [bx0, by0, bx1, by1] = lc.box.map(v => v * 1000), dx = (bx1 - bx0) / lc.W, dy = (by1 - by0) / lc.H;
    const out = new Float32Array(budget * 6), trunks = new Map();
    let n = 0;
    const R = TREE_KM * 1000;
    const i0 = Math.max(0, Math.floor((x - R - bx0) / dx)), i1 = Math.min(lc.W - 1, Math.ceil((x + R - bx0) / dx));
    const j0 = Math.max(0, Math.floor((by1 - (y + R)) / dy)), j1 = Math.min(lc.H - 1, Math.ceil((by1 - (y - R)) / dy));
    const fill = budget / 70000;
    for (let j = j0; j <= j1 && n < budget; j++) {
      for (let i = i0; i <= i1 && n < budget; i++) {
        const c = lc.a[j * lc.W + i], per = PER_M2[c];
        if (!per) continue;
        const px = bx0 + (i + 0.5) * dx, py = by1 - (j + 0.5) * dy;
        const r = Math.hypot(px - x, py - y);
        if (r > R) continue;
        const lod = r < NEAR ? 1 : (NEAR / r) ** 2;
        const want = per * dx * dy * lod * Math.min(1, fill * 1.4);
        const gi = Math.round(px / dx), gj = Math.round(py / dy);
        let m = Math.floor(want) + (hash(gi, gj, 1) < want - Math.floor(want) ? 1 : 0);
        for (let t = 0; t < m && n < budget; t++) {
          const tx = px + (hash(gi, gj, 10 + t) - 0.5) * dx, ty = py + (hash(gi, gj, 20 + t) - 0.5) * dy;
          const rd = Roads.at(tx, ty);
          if (rd.raise > 0.005 || rd.seg >= 0) continue;
          if (building(tx, ty)) continue;
          const s = hash(gi, gj, 30 + t);
          const big = Math.min(2.2, 1 / Math.sqrt(lod));
          const hgt = (c === 10 || c === 95 ? 9 + 9 * s : c === 20 ? 3.5 + 3 * s : 6 + 7 * s) * Math.min(1.3, Math.sqrt(big));
          const crown = hgt * (0.3 + 0.14 * hash(gi, gj, 40 + t)) * Math.min(1.6, big);
          out[6 * n] = tx - origin[0]; out[6 * n + 1] = ty - origin[1]; out[6 * n + 2] = terrain(tx, ty);
          out[6 * n + 3] = hgt; out[6 * n + 4] = crown; out[6 * n + 5] = hash(gi, gj, 50 + t);
          if (hgt > 6 && Math.hypot(tx - x, ty - y) < 1200) {
            const kk = key(Math.floor(tx / 8), Math.floor(ty / 8));
            let l = trunks.get(kk);
            if (!l) trunks.set(kk, l = []);
            l.push([tx, ty, 0.12 + hgt * 0.018, out[6 * n + 2] + hgt]);
          }
          n++;
        }
      }
    }
    // the close ones first (they get the detailed tree)
    const nearI = [], farI = [];
    for (let i = 0; i < n; i++) (Math.hypot(out[6 * i], out[6 * i + 1]) < NEAR_MESH ? nearI : farI).push(i);
    const sorted = new Float32Array(6 * n);
    [...nearI, ...farI].forEach((i, k) => sorted.set(out.subarray(6 * i, 6 * i + 6), 6 * k));
    st.trees = sorted; st.treeN = n; st.treeNear = nearI.length; st.treeOrigin = origin; st.trunks = trunks; st.treeGen++;
    return true;
  }

  // ---------------------------------------------------------------- solid things for the car
  // a point (world m) inside a wall or a trunk: { n: outward [x, y, 0], d: how deep m } or null
  function solid(x, y, z) {
    const hit = building(x, y);
    if (hit && z < hit.b.top && z > hit.b.zb - 1) {
      const P = hit.b.P;
      let best = 1e9, n = null;
      for (let i = 0; i < P.length; i += 2) {
        const j = (i + 2) % P.length, ax = P[i], ay = P[i + 1], ex = P[j] - ax, ey = P[j + 1] - ay, l2 = ex * ex + ey * ey;
        if (l2 < 1e-6) continue;
        const t = Math.max(0, Math.min(1, ((hit.lx - ax) * ex + (hit.ly - ay) * ey) / l2));
        const d = Math.hypot(hit.lx - ax - t * ex, hit.ly - ay - t * ey);
        if (d < best) { best = d; const l = Math.sqrt(l2); n = [ey / l, -ex / l, 0]; }
      }
      if (n) return { n, d: Math.min(best, 1.5) };
    }
    const l = st.trunks.get(key(Math.floor(x / 8), Math.floor(y / 8)));
    if (l) for (const [tx, ty, r, top] of l) {
      const d = Math.hypot(x - tx, y - ty);
      if (d < r && z < top) { const k = d > 1e-4 ? 1 / d : 0; return { n: [(x - tx) * k || 1, (y - ty) * k, 0], d: r - d }; }
    }
    // trunks across a cell edge
    for (const [ox, oy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const l2 = st.trunks.get(key(Math.floor(x / 8) + ox, Math.floor(y / 8) + oy));
      if (l2) for (const [tx, ty, r, top] of l2) {
        const d = Math.hypot(x - tx, y - ty);
        if (d < r && z < top) { const k = d > 1e-4 ? 1 / d : 0; return { n: [(x - tx) * k || 1, (y - ty) * k, 0], d: r - d }; }
      }
    }
    return null;
  }

  return { updateTrees, updateBuildings, solid, building, state: st };
})();
window.Scenery = Scenery;
