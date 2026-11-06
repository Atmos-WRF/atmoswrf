'use strict';
// Chase mode's roads, built from OpenStreetMap's (arhtmodelviewer/landscape.py roads, through /api/volume/roads):
//  - drivable geometry: each way a carriageway as wide as its class, raised a little on its bed with grassy
//    banks down to the fields (more for the big roads), round pads where ways end so junctions close up, paved
//    roads asphalt with US markings (double or dashed yellow centre lines, white edges, lane lines on the
//    motorways), unpaved ones gravel with wheel ruts;
//  - the same shape for the car: a spatial hash of the segments answers "how high is the road here, and what is
//    under this wheel" (asphalt, gravel, the bank) so the car drives on what is drawn and drops off the edge.
// Positions are world metres (the run's map plane); the meshes are stored relative to the fetch's centre.
const Roads = (() => {
  // landscape.ROAD_CLASSES: motorway, trunk, primary, secondary, tertiary, unclassified, residential, track
  const NAMES = ['Motorway', 'Trunk road', 'Primary road', 'Secondary road', 'Tertiary road', 'Rural road', 'Street', 'Track'];
  const HALF = [5.6, 5.0, 4.4, 3.9, 3.5, 3.0, 3.1, 1.8];          // half the carriageway, m
  const RAISE = [0.55, 0.45, 0.32, 0.28, 0.22, 0.16, 0.10, 0.04];   // the road bed above the fields, m
  const BANK = [3.2, 2.8, 2.2, 1.9, 1.6, 1.3, 0.8, 0.5];            // the banks' width, m
  const CELL = 32;                                                   // the hash's cells, m
  const FETCH_KM = 12, REFETCH_KM = 3.5, STEP = 10;                  // fetched box side, re-fetch distance, mesh step m

  const st = { origin: null, centre: null, segs: null, grid: new Map(), mesh: null, busy: false, pending: 0, n: 0,
               polys: [], gen: 0, lastTry: 0, terrain: null };

  // ---------------------------------------------------------------- the car's queries
  function cellKey(i, j) { return i * 73856093 ^ j * 19349663; }
  // the road bed under x, y (world m): { raise m, seg index or -1 on the carriageway, bank 0..1 }
  function at(x, y) {
    const S = st.segs;
    let raise = 0, seg = -1, best = -1;
    if (!S) return { raise, seg, cls: -1 };
    const list = st.grid.get(cellKey(Math.floor(x / CELL), Math.floor(y / CELL)));
    if (!list) return { raise, seg, cls: -1 };
    for (const s of list) {
      const ax = S.ax[s], ay = S.ay[s], dx = S.bx[s] - ax, dy = S.by[s] - ay;
      const l2 = dx * dx + dy * dy;
      let t = l2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const d = Math.hypot(x - ax - t * dx, y - ay - t * dy), k = S.cls[s];
      const hw = HALF[k], r = RAISE[k];
      if (d <= hw) {
        if (r > raise) raise = r;
        if (best < 0 || k < S.cls[best]) best = s;
      } else if (d < hw + BANK[k]) {
        const z = r * (1 - (d - hw) / BANK[k]);
        if (z > raise) raise = z;
      }
    }
    seg = best;
    return { raise, seg, cls: seg >= 0 ? S.cls[seg] : -1, paved: seg >= 0 ? S.paved[seg] : 0 };
  }
  // the nearest point on a road (any class up to maxCls) to x, y: { x, y, yaw, cls } or null
  function nearest(x, y, maxCls = 6, within = 4000) {
    const S = st.segs;
    if (!S) return null;
    let bd = within, out = null;
    for (let s = 0; s < S.n; s++) {
      if (S.cls[s] > maxCls) continue;
      const ax = S.ax[s], ay = S.ay[s], dx = S.bx[s] - ax, dy = S.by[s] - ay, l2 = dx * dx + dy * dy;
      if (l2 < 1) continue;
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2));
      const d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
      if (d < bd) { bd = d; out = { x: ax + t * dx, y: ay + t * dy, yaw: Math.atan2(dy, dx), cls: S.cls[s], d }; }
    }
    return out;
  }

  // ---------------------------------------------------------------- building
  function build(roads, origin, terrain) {
    // the segments, densified so the beds follow the terrain
    const ax = [], ay = [], bx = [], by = [], cls = [], paved = [];
    const polys = [];
    for (const [k, pv, xy] of roads) {
      const pts = [];
      for (let i = 0; i < xy.length; i += 2) {
        const x = xy[i] * 1000, y = xy[i + 1] * 1000;
        if (pts.length) {
          const [px, py] = pts[pts.length - 1], d = Math.hypot(x - px, y - py);
          if (d < 0.5) continue;
          const n = Math.ceil(d / STEP);
          for (let j = 1; j < n; j++) pts.push([px + (x - px) * j / n, py + (y - py) * j / n]);
        }
        pts.push([x, y]);
      }
      if (pts.length < 2) continue;
      polys.push({ k, pv, pts });
      for (let i = 1; i < pts.length; i++) {
        ax.push(pts[i - 1][0]); ay.push(pts[i - 1][1]); bx.push(pts[i][0]); by.push(pts[i][1]); cls.push(k); paved.push(pv ? 1 : 0);
      }
    }
    const n = ax.length;
    const segs = { n, ax: Float64Array.from(ax), ay: Float64Array.from(ay), bx: Float64Array.from(bx), by: Float64Array.from(by),
                   cls: Uint8Array.from(cls), paved: Uint8Array.from(paved) };
    const grid = new Map();
    for (let s = 0; s < n; s++) {
      const r = HALF[cls[s]] + BANK[cls[s]] + 0.5;
      const i0 = Math.floor((Math.min(ax[s], bx[s]) - r) / CELL), i1 = Math.floor((Math.max(ax[s], bx[s]) + r) / CELL);
      const j0 = Math.floor((Math.min(ay[s], by[s]) - r) / CELL), j1 = Math.floor((Math.max(ay[s], by[s]) + r) / CELL);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const key = cellKey(i, j);
        let l = grid.get(key);
        if (!l) grid.set(key, l = []);
        l.push(s);
      }
    }
    // the mesh: per vertex x, y, z (m from the origin), u (across: -1..1 the carriageway, beyond on the banks),
    // v (m along), class, paved
    const V = [];
    const push = (x, y, z, u, v, k, pv) => V.push(x - origin[0], y - origin[1], z, u, v, k, pv);
    const zAt = (x, y, k, onBed) => terrain(x, y) + (onBed ? RAISE[k] + 0.02 * (6 - Math.min(k, 6)) * 0.1 : 0);
    // a quad strip across (a list of [offset m, u, on the bed]) along the polyline
    for (const { k, pv, pts } of polys) {
      const hw = HALF[k], bk = BANK[k];
      const across = [[-(hw + bk), -1 - bk / hw, false], [-hw, -1, true], [hw, 1, true], [hw + bk, 1 + bk / hw, false]];
      const rows = [];
      let vdist = 0;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i], a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
        let tx = b[0] - a[0], ty = b[1] - a[1];
        const tl = Math.hypot(tx, ty) || 1; tx /= tl; ty /= tl;
        // the miter: the offset's length grows where the road bends (capped)
        let m = 1;
        if (i > 0 && i < pts.length - 1) {
          const d1 = [p[0] - a[0], p[1] - a[1]], l1 = Math.hypot(...d1) || 1;
          const nx1 = -d1[1] / l1, ny1 = d1[0] / l1;
          m = 1 / Math.max(0.5, nx1 * -ty + ny1 * tx);
        }
        if (i > 0) vdist += Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]);
        const nx = -ty * m, ny = tx * m;
        rows.push(across.map(([o, u, bed]) => {
          const x = p[0] + nx * o, y = p[1] + ny * o;
          return [x, y, zAt(x, y, k, bed), u, vdist];
        }));
      }
      for (let i = 1; i < rows.length; i++) {
        for (let c = 0; c < 3; c++) {
          const A = rows[i - 1][c], B = rows[i - 1][c + 1], C = rows[i][c + 1], D = rows[i][c];
          for (const q of [A, B, C, A, C, D]) push(q[0], q[1], q[2], q[3], q[4], k, pv ? 1 : 0);
        }
      }
      // round pads at both ends: the junctions close up (the hash's distance to an end is round too)
      for (const e of [pts[0], pts[pts.length - 1]]) {
        const z = zAt(e[0], e[1], k, true) - 0.004, N = 10;
        for (let j = 0; j < N; j++) {
          const a0 = 2 * Math.PI * j / N, a1 = 2 * Math.PI * (j + 1) / N;
          push(e[0], e[1], z, 0, -1, k, pv ? 1 : 0);
          push(e[0] + Math.cos(a0) * hw, e[1] + Math.sin(a0) * hw, z, 0.9, -1, k, pv ? 1 : 0);
          push(e[0] + Math.cos(a1) * hw, e[1] + Math.sin(a1) * hw, z, 0.9, -1, k, pv ? 1 : 0);
        }
      }
    }
    return { segs, grid, verts: new Float32Array(V), polys };
  }

  // the roads around the car (world m): fetched when it has gone REFETCH_KM from the last fetch's centre, and
  // again while OpenStreetMap's cells are still coming in; terrain(x, y) m; done() after a rebuild
  async function update(x, y, terrain, done) {
    if (st.busy) return;
    const far = !st.centre || Math.hypot(x - st.centre[0], y - st.centre[1]) > REFETCH_KM * 1000;
    const now = performance.now();
    if (!far && !(st.pending && now - st.lastTry > 6000) && !(terrain !== st.terrain && st.segs)) return;
    st.busy = true; st.lastTry = now;
    try {
      const h = FETCH_KM / 2, cx = x / 1000, cy = y / 1000;
      const b = [cx - h, cy - h, cx + h, cy + h].map(v => v.toFixed(3));
      const r = await fetch(`/api/volume/roads?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&major=0`, { headers: authHeaders() });
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      st.pending = d.pending;
      if (far || d.roads.length !== st.n || terrain !== st.terrain) {
        const origin = [Math.round(x), Math.round(y)];
        const out = build(d.roads, origin, terrain);
        Object.assign(st, { origin, segs: out.segs, grid: out.grid, polys: out.polys, n: d.roads.length, terrain });
        st.mesh = out.verts; st.gen++;
        if (done) done();
      }
      st.centre = [x, y];
    } catch (e) { console.warn('chase roads:', e); st.pending = st.pending || 1; }
    st.busy = false;
  }

  const authHeaders = () => (window.ChaseAuth ? window.ChaseAuth() : {});

  return { at, nearest, update, state: st, NAMES, HALF, RAISE };
})();
window.Roads = Roads;
