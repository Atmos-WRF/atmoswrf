// ===================================================================== chase mode's API
// Chase mode keeps its own car and camera and draws the storm; this map streams in around the point it is given
// (focus), is drawn from chase mode's eye into a render target (draw: colour and depth, which chase mode's storm
// renderer composites and lights), and answers what is under a point (ground) and what the car runs into (collide).
// Coordinates here are this map's own: x east, z south, y up, metres (Web Mercator about START_LAT, START_LON);
// toLatLon / fromLatLon convert.
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
  };
  mat.customProgramCacheKey = () => key + '-wind2';
  mat.needsUpdate = true;
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
  // the car (here at V: x, z, heading, vx, vz, y) pushed out of trees, buildings, landmarks and props: V afterwards
  collide(p) {
    Object.assign(V, p);
    collideTrees();
    collideBuildings();
    collideProps();
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
  // the landmark close by (within 400 m), its name
  landmarkNear: () => (lmNear ? lmNear.name : null),
};
globalThis.ChaseWorld = ChaseWorld;
