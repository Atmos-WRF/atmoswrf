'use strict';
// Chase mode's detailed map (web/js/chase/world/, from Real Roads Drive): real terrain, aerial-photo ground, roads with
// their lanes and markings, bridges, water, buildings, trees, grass and crops, roadside props, farms and landmarks.
// It is a Three.js world of its own (world.js), in its own coordinates (x east, z south, y up, m: Web Mercator about
// where the car first was), drawn in chase mode's WebGL context into a target the storm's renderer composites and
// lights (render.js: mergeWorld).  This links the two: the run's map plane (Lambert conformal, x east, y north, z up
// above the model's zero) to and from the map's coordinates, the ground under a wheel, what the car runs into, and
// the map drawn from chase mode's eye.
const WorldLink = (() => {
  const st = { W: null, loading: false, err: '', proj: null, zOff: null, ready: false };
  const D2R = Math.PI / 180;
  // the run's map plane (as map.js): metres <-> degrees
  function inv(x, y) {
    const p = st.proj, dy = p.rho0 - y, sg = p.n > 0 ? 1 : -1;
    const rho = sg * Math.hypot(x, dy), th = Math.atan2(sg * x, sg * dy);
    return [(2 * Math.atan(Math.pow(p.R * p.F / rho, 1 / p.n)) - Math.PI / 2) / D2R, p.lon0 + th / p.n / D2R];
  }
  function fwd(lat, lon) {
    const p = st.proj, dlon = ((lon - p.lon0 + 540) % 360) - 180;
    const rho = p.R * p.F / Math.pow(Math.tan(Math.PI / 4 + lat * D2R / 2), p.n), th = p.n * dlon * D2R;
    return [rho * Math.sin(th), p.rho0 - rho * Math.cos(th)];
  }
  // chase (x, y m) -> the map's [x, z]; and back
  const toW = (x, y) => { const [la, lo] = inv(x, y); return st.W.fromLatLon(la, lo); };
  const fromW = (wx, wz) => { const [la, lo] = st.W.toLatLon(wx, wz); return fwd(la, lo); };
  // a chase direction at a point -> the map's (x, y up, z)
  function dirToW(p, d) {
    const a = toW(p[0], p[1]), b = toW(p[0] + d[0] * 10, p[1] + d[1] * 10);
    return [(b[0] - a[0]) / 10, d[2], (b[1] - a[1]) / 10];
  }

  // loads the map the first time it is wanted, about the car (x, y m on the run's map plane)
  async function start(gl, proj, x, y) {
    if (st.W || st.loading || !proj) return;
    st.loading = true; st.proj = proj;
    try {
      const [lat, lon] = inv(x, y);
      const lm = await fetch('/static/js/chase/world/landmarks.json').then(r => (r.ok ? r.json() : [])).catch(() => []);
      globalThis.CHASE_WORLD = { gl, lat, lon, workerUrl: '/static/js/chase/world/tileworker.js', landmarks: lm };
      await import('/static/js/chase/world/world.js');
      st.W = globalThis.ChaseWorld;
    } catch (e) { st.err = e.message || String(e); console.warn('chase: detailed map failed', e); }
    st.loading = false;
  }
  // chase metres per metre of the map, at a point (the two projections' scales differ by a fraction of a percent)
  function scaleAt(x, y) {
    const a = toW(x, y), b = toW(x + 500, y);
    return 500 / Math.hypot(b[0] - a[0], b[1] - a[1]);
  }

  // once a frame: the map streams in round the car; the heights' offset (the map's ground is above sea level, the
  // run's above the model's zero) is set the first time the ground under the car has come in, from the model's
  // terrain there, so the car and the storm keep their heights above the ground
  function update(car, modelTerrain) {
    if (!st.W || !car) return;
    const [wx, wz] = toW(car.p[0], car.p[1]);
    const f = [Math.cos(car.yaw || 0), Math.sin(car.yaw || 0), 0], fw = dirToW(car.p, f);
    st.W.focus({ x: wx, z: wz, y: car.p[2] + (st.zOff || 0), heading: Math.atan2(fw[0], fw[2]), vx: 0, vz: 0 });
    st.W.update();
    st.ready = st.W.readyAt(wx, wz);
    if (st.ready && st.zOff === null) {
      const g = st.W.ground(wx, wz);
      if (g) st.zOff = g.h - modelTerrain(car.p[0], car.p[1]);
    }
  }
  // a new place (a jump): the heights' offset is taken again there
  function moved() { st.zOff = null; st.ready = false; }

  // what is under a point (chase x, y m): { h (m above the model's zero), surface, land, road, water } or null
  function ground(x, y, zRef) {
    if (!st.W || st.zOff === null) return null;
    const [wx, wz] = toW(x, y), g = st.W.ground(wx, wz, zRef === undefined ? Infinity : zRef + st.zOff);
    if (!g) return null;
    return { ...g, h: g.h - st.zOff };
  }

  // the car pushed out of what it ran into (trees, buildings, landmarks, posts, fences): its position and velocity
  // changed in place (chase m, m/s)
  function collide(car) {
    if (!st.W || st.zOff === null || !car) return false;
    const p = car.p, [wx, wz] = toW(p[0], p[1]);
    const f = [Math.cos(car.yaw || 0), Math.sin(car.yaw || 0), 0], fw = dirToW(p, f), vw = dirToW(p, [car.v[0], car.v[1], 0]);
    const sc = scaleAt(p[0], p[1]), S = car.spec || {}, dims = S.dims || [4.8, 1.95];
    const yr = car.w ? car.w[2] : 0;                  // counter-clockwise from above: the map's +heading turns left too
    const V = st.W.collide({ x: wx, z: wz, y: p[2] + st.zOff, heading: Math.atan2(fw[0], fw[2]), vx: vw[0], vz: vw[2],
                             yawRate: yr, len: dims[0] / sc, wid: dims[1] / sc, kg: S.mass || 2150,
                             Iz: car.I ? car.I[2] / (sc * sc) : 4800, wheelR: S.wheelR || 0.38 });
    const moved = Math.hypot(V.x - wx, V.z - wz) > 1e-4 || Math.hypot(V.vx - vw[0], V.vz - vw[2]) > 1e-4;
    if (car.w && Math.abs(V.yawRate - yr) > 1e-6) car.w[2] = V.yawRate;   // a corner clipped: the car turned
    if (!moved) return false;
    const np = fromW(V.x, V.z), nv = fromW(V.x + V.vx, V.z + V.vz);
    car.p[0] = np[0]; car.p[1] = np[1];
    car.v[0] = nv[0] - np[0]; car.v[1] = nv[1] - np[1];
    return true;
  }

  // something round (a person on foot: rad m) at chase x, y pushed out of the map's buildings, landmarks and trunks:
  // [dx, dy] m or null
  function pushOut(x, y, rad) {
    if (!st.W || st.zOff === null || !st.W.pushOut) return null;
    const [wx, wz] = toW(x, y), d = st.W.pushOut(wx, wz, rad);
    if (!d) return null;
    const a = fromW(wx + d[0], wz + d[1]);
    return [a[0] - x, a[1] - y];
  }

  // what render.js needs to draw the map from the camera: (W, H) -> { color, depth, near, far, mToBox (map metres
  // to chase metres) }; sun: chase's direction toward the sun
  function drawer(view, sun) {
    if (!st.W || st.zOff === null) return null;
    const z = st.zOff, e = view.eye, t = view.target;
    const [ex, ez] = toW(e[0], e[1]), [tx, tz] = toW(t[0], t[1]);
    const up = dirToW(e, view.up), sw = dirToW(e, sun);
    const el = sun[2], hi = Math.min(1, Math.max(0, (el - 0.02) / 0.43)), day = Math.min(1, Math.max(0, (el + 0.02) / 0.32));
    const sunCol = [0.70 + 0.30 * hi, 0.39 + 0.56 * hi, 0.20 + 0.66 * hi].map(c => c * (1.7 + 0.6 * hi) * day);
    const sky = 1.1 * (0.45 + 0.55 * day), s = scaleAt(e[0], e[1]);
    return (W, H) => {
      const r = st.W.draw({ w: W, h: H, fov: view.fov, eye: [ex, e[2] + z, ez], look: [tx, t[2] + z, tz], up,
                            light: { dir: sw, sun: sunCol, sky: [0.87 * sky, 0.94 * sky, 1.0 * sky], ground: [0.29 * sky, 0.35 * sky, 0.19 * sky] } });
      return { ...r, scale: s };
    };
  }

  // the wind in the map's trees and grass: chase's { u, v (east, north m/s: the model's at the camera), gust }
  function wind(tw) {
    if (!st.W || !st.W.wind) return;
    st.W.wind({ u: tw.u, z: -tw.v, gust: tw.gust });
  }

  return { st, start, update, moved, ground, collide, pushOut, drawer, wind, toW, fromW, ready: () => !!(st.W && st.ready && st.zOff !== null) };
})();
window.WorldLink = WorldLink;
