// ===================================================================== chase mode's API
// Chase mode keeps its own car and camera and draws the storm; this map streams in around the point it is given
// (focus), is drawn from chase mode's eye into a render target (draw: colour and depth, which chase mode's storm
// renderer composites and lights), and answers what is under a point (ground) and what the car runs into (collide).
// Coordinates here are this map's own: x east, z south, y up, metres (Web Mercator about START_LAT, START_LON);
// toLatLon / fromLatLon convert.
// the point the map streams in and is built round: the friend's page keeps it as FOC (its car or its walker, set in
// the parts build_world.py leaves out); here it is chase mode's focus, V (focus() below)
const FOC = V;
// and the car its collisions reckon with (the friend's page keeps it with its own car): chase mode's car's size (map
// metres), mass (kg), yaw inertia (kg m^2) and wheel radius (m), from collide()
const CAR = { len: 4.8, wid: 1.95, kg: 2150, Iz: 4800, wheelR: 0.38 };
const fromLatLon = (lat, lon) => [(lonToU(lon) - U0) * SCALE, (latToV(lat) - V0) * SCALE];
let rt = null, lastNow = performance.now(), focusSet = false;
function target(w, h) {
  if (rt && rt.width === w && rt.height === h) return rt;
  if (rt) rt.dispose();
  const depth = new THREE.DepthTexture(w, h);
  depth.type = THREE.UnsignedIntType;
  rt = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, depthTexture: depth, depthBuffer: true });
  return rt;
}
// ---- the wind in the trees, grass and crops (the model's at the camera, through ChaseWorld.wind): each one leans
// downwind from its foot (bend ~ W^2: 1 % of its height at 10 m/s, 10 % at 30, 40 % at 60), sways and shivers about
// it, more in the gusts.  A few instructions a vertex on the GPU, added to the materials' own hooks (their gentle
// idle sway stays).
const windU = { value: new THREE.Vector3(0, 0, 1) };              // x, z m/s (this map's axes), gust factor
const windTU = { value: 0 };
const SWAY_GLSL = `
uniform vec3 uWind; uniform float uWT;
vec2 windAtW(vec2 xz) {
  vec2 w = uWind.xy;
  return w;
}
vec2 swayW(vec2 xz, float h, float f, float sd) {
  vec2 w = windAtW(xz); float W = length(w);
  if (W < 0.3) return vec2(0.0);
  vec2 dir = w / W;
  float lean = min(0.00011 * W * W, 0.42), fr = 0.35 + 0.012 * W;
  float osc = sin(uWT * 6.2832 * fr + sd * 40.0) * (0.25 + 0.35 * clamp((W - 15.0) / 40.0, 0.0, 1.0));
  float gust = (uWind.z - 1.0) * max(0.0, sin(uWT * 0.9 + xz.x * 0.01) * sin(uWT * 2.3 + xz.y * 0.013 + 1.0));
  vec2 perp = vec2(-dir.y, dir.x) * sin(uWT * 6.2832 * 2.7 * fr + sd * 17.0) * 0.12 * lean;
  return (dir * lean * (1.0 + 0.6 * gust) * (1.0 + osc) + perp) * h * f * f;
}
`;
// three's project_vertex with the sway added after the instance's own transform (f: the vertex's height share)
const projectSway = f => `
vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_INSTANCING
  mvPosition = instanceMatrix * mvPosition;
  {
    float hh = length(instanceMatrix[1].xyz); vec3 ip0 = instanceMatrix[3].xyz;
    vec2 dsw = swayW(ip0.xz, hh, clamp(${f}, 0.0, 1.2), fract(ip0.x * 0.137 + ip0.z * 0.071));
    mvPosition.xz += dsw; mvPosition.y -= 0.5 * dot(dsw, dsw) / max(hh, 0.5);
  }
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;
`;
for (const mat of [treeMat, farMat, plantMat]) {
  const own = mat.onBeforeCompile, key = mat.customProgramCacheKey ? mat.customProgramCacheKey() : 'plant';
  mat.onBeforeCompile = (sh, r) => {
    if (own) own.call(mat, sh, r);
    Object.assign(sh.uniforms, { uWind: windU, uWT: windTU });
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\n' + SWAY_GLSL)
      .replace('#include <project_vertex>', projectSway('position.y'));
    // the trees broken where the damage is (ChaseWorld.damage): snapped to stubs, their branches stripped
    if (mat !== plantMat) {
      Object.assign(sh.uniforms, dmgU);
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\n' + DMG_HEAD)
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + TREE_DMG);
    }
  };
  mat.customProgramCacheKey = () => key + '-wind2-dmg';
  mat.needsUpdate = true;
}
// ---- the damage (chase mode's damage.js, through ChaseWorld.damage): each building's fragments above what is left
// standing at its degree of damage (EF scale, from the model's peak gusts) cut away, its roof stripped to the deck
// before that -- a texture over the ground round the car: r the degree / 10, g the ground's height there
const dmgU = { uDmg: { value: null }, uDmgBox: { value: new THREE.Vector4(0, 0, 1, 0) } };
const DMG_HEAD = `
uniform sampler2D uDmg; uniform vec4 uDmgBox;
`;
const DMG_MAIN = `
  if (uDmgBox.w > 0.5) {
    vec2 duv = (vWorldP.xz - uDmgBox.xy) / uDmgBox.z;
    if (all(greaterThan(duv, vec2(0.0))) && all(lessThan(duv, vec2(1.0)))) {
      vec4 dm = texture2D(uDmg, duv);
      float dg = dm.r * 10.0, hg = vWorldP.y - dm.g;
      if (dg >= 5.5) {
        float keep = dg >= 9.5 ? 0.15 : dg >= 8.5 ? 0.6 : dg >= 7.5 ? 1.2 : dg >= 6.5 ? 2.0 : 3.0;
        if (hg > keep + 0.35 * sin(vWorldP.x * 1.7) + 0.3 * sin(vWorldP.z * 2.3)) discard;
      }
      vec3 fn = normalize(cross(dFdx(vWorldP), dFdy(vWorldP)));
      if (dg >= 1.5 && abs(fn.y) > 0.6) diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.42, 0.33, 0.24), clamp((dg - 1.0) / 3.0, 0.0, 1.0));
      if (dg >= 3.0) diffuseColor.rgb *= 0.85;
    }
  }
`;
// a tree (instanced, its height the instance's y scale) at its degree of damage: from 5 (trunks snapped, ~100 mph)
// a stub of 2.6 m (1.2 m from 9), from 3 its big branches gone
const TREE_DMG = `
#ifdef USE_INSTANCING
  {
    vec3 ipd = (modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz;
    float hd = max(length(instanceMatrix[1].xyz), 0.5);
    vec2 duv = (ipd.xz - uDmgBox.xy) / uDmgBox.z;
    if (uDmgBox.w > 0.5 && all(greaterThan(duv, vec2(0.0))) && all(lessThan(duv, vec2(1.0)))) {
      float dg = texture2D(uDmg, duv).r * 10.0;
      if (dg >= 4.5) { float cut = (dg >= 8.5 ? 1.2 : 2.6) / hd; if (transformed.y > cut) { transformed.xz *= 0.1; transformed.y = cut; } }
      else if (dg >= 2.5 && transformed.y > 3.0 / hd) transformed.xz *= 0.6;
    }
  }
#endif
`;
{
  const own = buildingMaterial;
  // (a function declaration: the tiles built from now on take this one)
  buildingMaterial = o => {
    const m = own(o), ob = m.onBeforeCompile;
    m.side = THREE.DoubleSide;                     // a building cut open shows its inside walls
    m.onBeforeCompile = (sh, r) => {
      if (ob) ob.call(m, sh, r);
      Object.assign(sh.uniforms, dmgU);
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + DMG_HEAD)
        .replace('#include <alphatest_fragment>', DMG_MAIN + '#include <alphatest_fragment>');
    };
    m.customProgramCacheKey = () => 'building-v2-dmg';
    return m;
  };
}
// ---- what the car runs into: chase mode's own collisions (as before the OpenDrive rebuild, 7445792), not the friend's
// page's impulse ones (carHitCircle / carImpulse: they threw the car, and were called here without their dt).  Two
// circles along the car (front and back, 1 m), each pushed out of a trunk, a building's footprint, a post or a fence
// and its speed into it turned back (a little bounce); a knockable prop hit at speed goes flying.  The map's newer
// data as it is: tunnels (V.inTunnel: nothing up on the surface), footbridge decks, deck-level railings.
function chaseHitTrees() {
  if (V.inTunnel) return;
  const fx = Math.sin(V.heading), fz = Math.cos(V.heading);
  for (const along of [1.3, -1.3]) {
    const px = V.x + fx * along, pz = V.z + fz * along;
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
      const list = trunkHash.get(thKey(px + dx * TH_CELL, pz + dz * TH_CELL));
      if (!list) continue;
      for (let i = 0; i < list.length; i += 3) {
        const ex = px - list[i], ez = pz - list[i + 1], d = Math.hypot(ex, ez), min = 0.95 + list[i + 2];
        if (d >= min || d < 1e-4) continue;
        const nx = ex / d, nz = ez / d, push = min - d;
        V.x += nx * push; V.z += nz * push;
        const vn = V.vx * nx + V.vz * nz;
        if (vn < 0) { V.vx -= nx * vn * 1.35; V.vz -= nz * vn * 1.35; V.hit = Math.max(V.hit || 0, -vn); }
      }
    }
  }
}
function chaseHitBuildings() {
  if (V.inTunnel) return;
  const fx = Math.sin(V.heading), fz = Math.cos(V.heading);
  for (const along of [1.4, -1.4]) {
    const p = pushOutOfBuildings(V.x + fx * along, V.z + fz * along, 1.0, V.segOff >= 0 || V.onDeck);
    if (!p) continue;
    const l = Math.hypot(p[0], p[1]);
    if (l < 1e-6) continue;
    const nx = p[0] / l, nz = p[1] / l;
    V.x += p[0]; V.z += p[1];
    const vn = V.vx * nx + V.vz * nz;
    if (vn < 0) { V.vx -= nx * vn * 1.3; V.vz -= nz * vn * 1.3; V.hit = Math.max(V.hit || 0, -vn); }
  }
}
function chaseHitProps() {
  if (!colCircles.length && !colSegs.length) return;
  if (V.onDeck) return;
  const fx = Math.sin(V.heading), fz = Math.cos(V.heading), speed = Math.hypot(V.vx, V.vz);
  for (const along of [1.4, -1.4]) {
    const px = V.x + fx * along, pz = V.z + fz * along, R = 1.0;
    for (let o = 0; o < colCircles.length && !V.inTunnel; o += 4) {
      const ex = px - colCircles[o], ez = pz - colCircles[o + 1];
      if (Math.abs(ex) > 3 || Math.abs(ez) > 3) continue;
      const d = Math.hypot(ex, ez), min = R + colCircles[o + 2];
      if (d >= min || d < 1e-4) continue;
      const kid = colCircles[o + 3], nx = ex / d, nz = ez / d;
      if (kid > 0 && speed > 2.5) { knock(kid, nx, nz, speed); colCircles[o + 3] = -2; colCircles[o] = 1e9; V.vx *= 0.96; V.vz *= 0.96; continue; }
      V.x += nx * (min - d); V.z += nz * (min - d);
      const vn = V.vx * nx + V.vz * nz;
      if (vn < 0) { V.vx -= nx * vn * 1.35; V.vz -= nz * vn * 1.35; }
    }
    for (let o = 0; o < colSegs.length; o += 5) {
      const ax = colSegs[o], az = colSegs[o + 1], bx = colSegs[o + 2], bz = colSegs[o + 3], sy = colSegs[o + 4];
      if (!Number.isNaN(sy) && (V.y < sy - 1.5 || V.y > sy + 2.5)) continue;   // bridge railings only at deck level
      if (Number.isNaN(sy) && V.inTunnel) continue;
      if (Math.min(ax, bx) - 3 > px || Math.max(ax, bx) + 3 < px || Math.min(az, bz) - 3 > pz || Math.max(az, bz) + 3 < pz) continue;
      const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz || 1e-9, u = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / l2));
      const qx = ax + dx * u, qz = az + dz * u, ex = px - qx, ez = pz - qz, d = Math.hypot(ex, ez);
      if (d >= R + 0.1 || d < 1e-4) continue;
      const nx = ex / d, nz = ez / d;
      V.x += nx * (R + 0.1 - d); V.z += nz * (R + 0.1 - d);
      const vn = V.vx * nx + V.vz * nz;
      if (vn < 0) { V.vx -= nx * vn * 1.25; V.vz -= nz * vn * 1.25; }
    }
  }
}
const ChaseWorld = {
  START_LAT, START_LON, TILE_SIZE, SURF, SURF_NAMES, LAND, LAND_SURFACE, CLASS_LIST, SEG,
  toLatLon, fromLatLon, scene, camera, renderer, V, tiles,
  // the point the map streams in around and the props, grass and trees are built round: { x, y, z, heading, vx, vz }
  focus(p) {
    Object.assign(V, p);
    focusSet = true;
  },
  // once a frame: tiles, grass and crops, trees, props, landmarks, birds, culling
  update() {
    if (!focusSet) return;
    const now = performance.now(), dt = Math.max(0, Math.min(0.1, (now - lastNow) / 1000));
    lastNow = now;
    updateTiles(V.x, V.z);
    updateDetail(dt);
    updateProps(dt);
    updateLandmarks(dt);
    updateCulling(dt);
  },
  // the detailed tile under a point has come in (its roads and heights)
  readyAt(x, z) { const t = tileFor(x, z); return !!(t && t.lod === 1 && t.H); },
  // draws the map from eye toward look (this map's coordinates), vertical fov in radians, into a w x h target:
  // { color, depth } WebGL textures (linear colour, half float; depth 24-bit) and the projection's near and far.
  // light: { dir [x, y, z] toward the sun, sun [r, g, b] x intensity, sky, ground [r, g, b] x intensity }
  draw(o) {
    const r = target(o.w, o.h);
    camera.fov = o.fov * 180 / Math.PI; camera.aspect = o.w / o.h;
    camera.near = o.near || 0.3; camera.far = o.far || TILE_SIZE * (FAR_RADIUS + 1.5) * 1.42;
    camera.updateProjectionMatrix();
    camera.position.set(o.eye[0], o.eye[1], o.eye[2]);
    camera.up.set(o.up[0], o.up[1], o.up[2]);
    camera.lookAt(o.look[0], o.look[1], o.look[2]);
    camera.updateMatrixWorld();
    const L = o.light;
    if (L) {
      SUN_DIR.set(L.dir[0], L.dir[1], L.dir[2]).normalize();
      sun.color.setRGB(L.sun[0], L.sun[1], L.sun[2]); sun.intensity = 1;
      hemi.color.setRGB(L.sky[0], L.sky[1], L.sky[2]); hemi.groundColor.setRGB(L.ground[0], L.ground[1], L.ground[2]); hemi.intensity = 1;
      waterMat.uniforms.uSun.value.copy(SUN_DIR);
    }
    sun.position.set(V.x + SUN_DIR.x * 150, V.y + SUN_DIR.y * 150, V.z + SUN_DIR.z * 150);
    sun.target.position.set(V.x, V.y, V.z);
    renderer.resetState();
    renderer.setRenderTarget(r);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.resetState();
    const P = renderer.properties;
    return { color: P.get(r.texture).__webglTexture, depth: P.get(r.depthTexture).__webglTexture, near: camera.near, far: camera.far };
  },
  // what is under a point: { h (m), road (its class name or null), surface (SURF_NAMES), bridge, land (LAND name or
  // null), water (depth m) }, or null before its tile has come in
  ground(x, z, yRef = Infinity) {
    const g = groundAt(x, z, yRef);
    if (!g) return null;
    const out = { h: g.h, road: null, surface: null, bridge: false, land: null, water: 0 };
    if (g.seg >= 0) {
      const s = g.segs, o = g.seg;
      out.road = CLASS_LIST[s[o + 7]]; out.surface = SURF_NAMES[s[o + 5]]; out.bridge = !!s[o + 8];
    }
    if (g.deck) out.surface = 'asphalt';
    const l = landAt(x, z);
    if (l) { out.land = LAND[l.mat]; if (!out.surface) out.surface = SURF_NAMES[LAND_SURFACE[l.mat]]; }
    const wl = out.bridge ? null : waterLevelAt(x, z);
    if (wl !== null && wl > g.h) out.water = wl - g.h;
    return out;
  },
  // the car (here at V: x, z, heading, vx, vz, y, yawRate; len, wid, kg, Iz, wheelR its size, mass, yaw inertia and
  // wheels) pushed out of trees, buildings, landmarks and props and turned by what it hit: V afterwards
  collide(p) {
    for (const k of ['len', 'wid', 'kg', 'Iz', 'wheelR']) if (p[k] > 0) CAR[k] = p[k];
    Object.assign(V, p);
    chaseHitTrees();
    chaseHitBuildings();
    chaseHitProps();
    return V;
  },
  // something round (a person on foot: rad m) at x, z pushed out of buildings, landmarks and tree trunks: [dx, dz]
  // or null
  pushOut(x, z, rad) {
    let mx = 0, mz = 0, hit = false;
    const p = pushOutOfBuildings(x, z, rad, false);
    if (p) { mx += p[0]; mz += p[1]; hit = true; }
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
      const list = trunkHash.get(thKey(x + dx * TH_CELL, z + dz * TH_CELL));
      if (!list) continue;
      for (let i = 0; i < list.length; i += 3) {
        const ex = x + mx - list[i], ez = z + mz - list[i + 1], d = Math.hypot(ex, ez), min = rad + list[i + 2];
        if (d >= min || d < 1e-4) continue;
        mx += ex / d * (min - d); mz += ez / d * (min - d); hit = true;
      }
    }
    return hit ? [mx, mz] : null;
  },
  nearestRoad,
  // the wind for the trees, grass and crops: { u, z m/s (this map's axes), gust (factor) } -- the model's at the camera
  wind(o) {
    windU.value.set(o.u || 0, o.z || 0, o.gust || 1);
    windTU.value = performance.now() / 1000 % 3600;
  },
  // the damage round the car: o { x0, z0 (this map's corner), size (its side), n (texels a side), deg: Float32Array
  // n * n (row by row along x, then z: the degree of damage / 10) }; the ground's height is put in here
  damage(o) {
    const N = o.n, a = new Float32Array(N * N * 4), c = o.size / N;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const k = j * N + i, d = o.deg[k];
      if (!(d > 0)) continue;
      const g = groundAt(o.x0 + (i + 0.5) * c, o.z0 + (j + 0.5) * c, Infinity);
      if (!g) continue;
      a[4 * k] = d; a[4 * k + 1] = g.h;
    }
    const t = new THREE.DataTexture(a, N, N, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter; t.needsUpdate = true;
    if (dmgU.uDmg.value) dmgU.uDmg.value.dispose();
    dmgU.uDmg.value = t;
    dmgU.uDmgBox.value.set(o.x0, o.z0, o.size, 1);
  },
  // the landmark close by (within 400 m), its name
  landmarkNear: () => (lmNear ? lmNear.name : null),
};
globalThis.ChaseWorld = ChaseWorld;
