'use strict';
// Chase mode's lights at night: street lamps along the roads in towns (where OpenStreetMap's buildings stand close),
// each a pole with an arm and a head that glows at night, and a pool of light on the road under it (render.js
// setLamps: drawn into the geometry buffer's light target); the gas stations' canopies light the pumps.  Built
// round the car out to NEAR m, again when it has gone REBUILD m.  World metres.
const Lights = (() => {
  const NEAR = 900, REBUILD = 300, SPACING = 38;
  const st = { centre: null, roadsGen: -1, bGen: -1, sGen: -1, parts: [], inst: null, origin: [0, 0], n: 0, dirty: false };
  const hash = (a, b) => { let h = Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };

  // buildings within r m of x, y (scenery.js's spatial hash of footprints)
  function townness(x, y) {
    const S = Scenery.state;
    if (!S.bGrid || !S.bOrigin) return 0;
    const lx = x - S.bOrigin[0], ly = y - S.bOrigin[1], C = 32;
    let n = 0;
    const seen = new Set();
    for (let i = Math.floor((lx - 70) / C); i <= Math.floor((lx + 70) / C); i++)
      for (let j = Math.floor((ly - 70) / C); j <= Math.floor((ly + 70) / C); j++) {
        const l = S.bGrid.get(i * 73856093 ^ j * 19349663);
        if (l) for (const b of l) if (!seen.has(b)) { seen.add(b); n++; }
      }
    return n;
  }

  // the lamps round x, y: [{ x, y, z, hx, hy, yaw, col }] -- along the paved roads of classes primary to street,
  // every SPACING m, on alternate sides, where there are a few buildings round about (towns, not open country)
  function place(x, y, h) {
    const R = Roads.state, out = [];
    if (!R.polys) return out;
    for (const { k, pv, pts } of R.polys) {
      if (!pv || k < 1 || k > 6) continue;
      let acc = hash(Math.round(pts[0][0]), Math.round(pts[0][1])) * SPACING, side = 1;
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i], dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy);
        if (l < 1e-3) continue;
        let t = acc;
        while (t < l) {
          const px = a[0] + dx * t / l, py = a[1] + dy * t / l;
          t += SPACING; side = -side;
          if (Math.abs(px - x) > NEAR || Math.abs(py - y) > NEAR) continue;
          if (townness(px, py) < 6) continue;
          const nx = -dy / l * side, ny = dx / l * side, off = Roads.HALF[k] + 1.6;
          const fx = px + nx * off, fy = py + ny * off;
          if (Scenery.building(fx, fy) || (Stations.covers && Stations.covers(fx, fy))) continue;
          if (window.Damage && Damage.gustAt(fx, fy) > 35) continue;           // the power's out (or the pole down)
          const reach = Math.min(2.6, Roads.HALF[k] * 0.6);
          out.push({ x: fx, y: fy, z: h(fx, fy), hx: fx - nx * reach, hy: fy - ny * reach, yaw: Math.atan2(-ny, -nx),
                     led: hash(Math.round(fx / 400), Math.round(fy / 400)) < 0.7 });
        }
        acc = t - l;
      }
    }
    return out;
  }

  // the poles (render.js parts, one mesh) and the pools
  function build(x, y, h) {
    const L = place(x, y, h), B = CarMesh.box, V = [], o = [Math.round(x), Math.round(y)];
    const inst = [];
    for (const lp of L) {
      const c = Math.cos(lp.yaw), s = Math.sin(lp.yaw);
      // a box in the lamp's frame (u along the arm, v across), turned and placed
      const local = [];
      B(local, -0.09, 0.09, -0.09, 0.09, -0.4, 8.2, [0.42, 0.43, 0.45]);              // the pole
      const reach = Math.hypot(lp.hx - lp.x, lp.hy - lp.y);
      B(local, 0, reach + 0.3, -0.05, 0.05, 7.9, 8.05, [0.42, 0.43, 0.45]);             // the arm
      const col = lp.led ? [10.92, 10.95, 11.0] : [11.0, 10.72, 10.42];
      B(local, reach - 0.35, reach + 0.35, -0.16, 0.16, 7.75, 7.92, col);                 // the head (lit)
      for (let i = 0; i < local.length; i += 9) {
        const u = local[i], v = local[i + 1], nu = local[i + 3], nv = local[i + 4];
        V.push(lp.x + c * u - s * v - o[0], lp.y + s * u + c * v - o[1], lp.z + local[i + 2],
               c * nu - s * nv, s * nu + c * nv, local[i + 5], local[i + 6], local[i + 7], local[i + 8]);
      }
      const lc = lp.led ? [0.9, 0.95, 1.0] : [1.0, 0.68, 0.32];
      inst.push(lp.hx - o[0], lp.hy - o[1], h(lp.hx, lp.hy) + 0.35, 13, lc[0], lc[1], lc[2], 0.32);
    }
    // the gas stations: their canopies' lights on the pumps, their stores' windows
    for (const b of (Stations.state.built || new Map()).values()) {
      for (const [u, w, r, I] of [[-6, 12, 13, 0.55], [6, 12, 13, 0.55], [0, 30, 11, 0.4], [0, 21, 9, 0.35]]) {
        const p = [b.L.O[0] + b.L.X[0] * u + b.L.Y[0] * w, b.L.O[1] + b.L.X[1] * u + b.L.Y[1] * w];
        if (Math.hypot(p[0] - x, p[1] - y) > NEAR * 1.5) continue;
        inst.push(p[0] - o[0], p[1] - o[1], (w > 20 ? b.zf : b.hAt(u, w)) + 0.4, r, 0.95, 0.97, 1.0, I);
      }
    }
    return { data: new Float32Array(V), inst: new Float32Array(inst), origin: o, n: L.length };
  }

  // each half second: rebuilt when the car has gone on or the roads, buildings or stations changed; true when the
  // lamps changed (render.js setLamps them)
  function update(x, y, h) {
    const far = !st.centre || Math.hypot(x - st.centre[0], y - st.centre[1]) > REBUILD;
    const S = Scenery.state;
    if (!far && st.roadsGen === Roads.state.gen && st.bGen === S.bGen && st.sGen === Stations.state.skipGen) return false;
    if (st.mesh && window.Render && Render.dropPart) Render.dropPart(st.mesh.data);
    const b = build(x, y, h);
    st.mesh = { data: b.data, M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], p: [b.origin[0], b.origin[1], 0] };
    st.inst = b.inst; st.origin = b.origin; st.n = b.n;
    st.centre = [x, y]; st.roadsGen = Roads.state.gen; st.bGen = S.bGen; st.sGen = Stations.state.skipGen;
    return true;
  }
  const parts = () => (st.mesh && st.mesh.data.length ? [st.mesh] : []);
  return { update, parts, state: st };
})();
window.Lights = Lights;
