'use strict';
// The 3-D view (arhtmodelviewer/volume.py), like GR2Analyst's volume explorer: drag a box on the map, and the run's
// latest reflectivity (or vertical velocity, or vertical vorticity, or the cloud as the eye sees it) in it is
// ray-marched on the GPU (WebGL2) as an isosurface, a translucent volume or a sunlit cloud.  Orbit (drag to turn,
// wheel to zoom, right-drag to move), a free camera or one on the ground (W A S D, drag to look); Live keeps it on
// the newest model step, the time slider goes back through the volumes the run saved with its radar frames.
const Vol = (() => {
  const FIELDS = {
    dbz: { name: 'Reflectivity', fam: 'dbz', min: 0, max: 75, step: 1, iso: 40, vol: 15, fmt: v => `${v} dBZ`, signed: false },
    w: { name: 'Vertical velocity', fam: 'w', min: 1, max: 60, step: 1, iso: 15, vol: 5, fmt: v => `${v} m/s`, signed: true },
    zeta: { name: 'Vertical vorticity', fam: 'vort', min: 0.002, max: 0.05, step: 0.001, iso: 0.01, vol: 0.004,
            fmt: v => `${v.toFixed(3)} /s`, signed: true },
    // the tornado view only: the pressure perturbation, drawn where it is at least the threshold BELOW its
    // surroundings (the vortex's low-pressure tube)
    pp: { name: 'Pressure drop', fam: 'pp', min: 1, max: 60, step: 0.5, iso: 6, vol: 2, fmt: v => `−${v} hPa`,
          signed: false, low: true },
  };
  const P_STOPS = [
    { v: -80, c: [120, 40, 200, 255] }, { v: -40, c: [40, 70, 230, 255] }, { v: -15, c: [40, 160, 255, 255] },
    { v: -5, c: [110, 225, 235, 255] }, { v: -1, c: [200, 245, 250, 0] }, { v: 30, c: [255, 255, 255, 0] },
  ];
  // vertical velocity: rising air warm, sinking air blue (the map's w table starts blue and has no downdrafts)
  const W_STOPS = [
    { v: -64, c: [20, 30, 160, 255] }, { v: -25, c: [40, 80, 230, 255] }, { v: -8, c: [100, 170, 255, 255] },
    { v: -2, c: [170, 210, 255, 0] }, { v: 2, c: [255, 240, 150, 0] }, { v: 6, c: [255, 225, 80, 255] },
    { v: 20, c: [255, 140, 30, 255] }, { v: 40, c: [225, 30, 30, 255] }, { v: 64, c: [255, 100, 240, 255] },
  ];
  let gl = null, prog = null, cloudProg = null, volTex = null, lutTex = null, gndTex = null, vao = null;
  let mapTex = null, mapKey = '', mapOn = false;            // the map picture on the ground (tiles.py ground())
  let head = null, box = null, busy = false, again = false, timer = null, lastKey = '', loadSeq = 0;
  let tor = false, paths = null;               // the tornado view (tornado.py): its box, followed; its parcels' paths
  // the camera: "orbit" (turn around the box), "free" (fly: WASD, Q/E, drag to look) or "ground" (a chaser's eye,
  // EYE_KM above the ground, walking or driving with WASD); fp: the free and ground cameras' place (km from the
  // box's middle, km up) and where they look
  const CAM0 = { mode: 'orbit', yaw: -0.75, pitch: 0.38, dist: 3.6, px: 0, py: 0 };
  const cam = { ...CAM0, fp: null, speed: 1 };
  const EYE_KM = 0.002;
  let noiseTex = null, gMap = null;
  const insTex = [];                               // the nests' volumes (volume.py insets)
  let lightProg = null, lightTex = null, lightFbo = null, lightKey = '';   // the cloud view's light volume                 // the cloud's detail noise; the ground under the box (km)
  // the cloud view's frames, jittered and averaged while the view is still (half size, no smoothing while it moves)
  const ACC_MAX = 16;
  let acc = 0, raf = 0, lastMove = 0, drawQ = 1, lastTick = 0;
  const keys = new Set();
  // the replay (volume.py history): the saved volumes' model times, the one on screen (null: the newest step)
  let times = [], ms0 = null, replayT = null, followMap = false, playTok = 0;
  const cache = new Map(); let cacheBytes = 0;
  const CACHE_MAX = 400e6;

  // ---------------------------------------------------------------- matrices (column-major, like GL)
  const mul = (a, b) => {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  };
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f; o[10] = (far + near) / (near - far); o[11] = -1; o[14] = 2 * far * near / (near - far);
    return o;
  }
  function lookAt(e, t, up) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const nrm = a => { const l = Math.hypot(...a) || 1; return a.map(x => x / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const f = nrm(sub(t, e)), s = nrm(cross(f, up)), u = cross(s, f);
    return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
      -dot(s, e), -dot(u, e), dot(f, e), 1]);
  }
  function invert(m) {
    const inv = new Float32Array(16);
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    for (let i = 0; i < 16; i++) inv[i] /= det || 1;
    return inv;
  }

  // ---------------------------------------------------------------- shaders
  const VS = `#version 300 es
  out vec2 ndc;
  void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    ndc = p; gl_Position = vec4(p, 0.0, 1.0);
  }`;
  const FS = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 ndc; out vec4 frag;
  uniform sampler3D vol; uniform sampler2D lut;
  uniform mat4 invVP; uniform vec3 eye; uniform vec3 size;        // box: x in [-size.x/2, size.x/2], z in [0, size.z]
  uniform float thrHi, thrLo, opacity, gridStep; uniform int mode, signedField; uniform vec3 texel;
  uniform sampler2D gnd; uniform int hasG;
  uniform sampler2D mapTex; uniform int hasMap, mapDark;
  vec3 mapAt(vec2 xy, vec3 fallback) {      // the map picture (OSM / satellite) on the ground under xy
    if (hasMap == 0) return fallback;
    vec2 uv = vec2(xy.x / size.x + 0.5, xy.y / size.y + 0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return fallback;
    vec3 c = texture(mapTex, uv).rgb;
    if (mapDark == 1) { float L = dot(c, vec3(0.333)); c = clamp(1.0 - 2.0 * L + c, 0.0, 1.0) * 0.8; }
    return c;
  }
  float gz(vec2 xy) {                       // the ground's height (box units) under xy; 0 outside the box / flat runs
    if (hasG == 0) return 0.0;
    vec2 uv = vec2(xy.x / size.x + 0.5, xy.y / size.y + 0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    return texture(gnd, uv).r * size.z;
  }
  vec3 gnormal(vec2 xy) {
    vec2 h = 0.004 * size.xy;
    return normalize(vec3(gz(xy - vec2(h.x, 0)) - gz(xy + vec2(h.x, 0)), gz(xy - vec2(0, h.y)) - gz(xy + vec2(0, h.y)),
                          2.0 * h.x));
  }
  // marching from ta (above the ground) to tb (below it): the crossing
  float gcross(vec3 ro, vec3 rd, float ta, float tb) {
    for (int k = 0; k < 8; k++) { float tm = 0.5 * (ta + tb); vec3 p = ro + rd * tm; if (p.z < gz(p.xy)) tb = tm; else ta = tm; }
    return tb;
  }
  // the nests (volume.py insets): up to four finer volumes, finest first, each over part of the box from the ground
  // to its own top; a point inside one reads it, blended into the box's volume over its outer 8 %
  uniform sampler3D ins0, ins1, ins2, ins3; uniform int insN; uniform vec4 insBox[4]; uniform float insTop[4];
  float insW(vec3 p, int i, out vec3 u) {
    vec4 b = insBox[i];
    u = vec3((p.x - b.x) / (b.z - b.x), (p.y - b.y) / (b.w - b.y), p.z / insTop[i]);
    if (any(lessThan(u, vec3(0.0))) || any(greaterThan(u, vec3(1.0)))) return 0.0;
    vec2 e = min(u.xy, 1.0 - u.xy);
    return clamp(min(min(e.x, e.y), 1.0 - u.z) / 0.08, 0.0, 1.0);
  }
  vec4 nested(vec3 p, vec4 base) {
    if (insN == 0) return base;
    vec3 u; float w = insW(p, 0, u);
    if (w > 0.0) return mix(base, textureLod(ins0, u, 0.0), w);
    if (insN > 1) { w = insW(p, 1, u); if (w > 0.0) return mix(base, textureLod(ins1, u, 0.0), w); }
    if (insN > 2) { w = insW(p, 2, u); if (w > 0.0) return mix(base, textureLod(ins2, u, 0.0), w); }
    if (insN > 3) { w = insW(p, 3, u); if (w > 0.0) return mix(base, textureLod(ins3, u, 0.0), w); }
    return base;
  }
  float val(vec3 p) { return nested(p, texture(vol, vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z))).r * 255.0; }
  vec4 col(float code) { return texture(lut, vec2((code + 0.5) / 256.0, 0.5)); }
  bool inside(float c) { return c >= thrHi || (signedField == 1 && c > 0.5 && c <= thrLo); }
  vec3 normalAt(vec3 p) {
    vec3 h = texel * size;
    vec3 g = vec3(val(p + vec3(h.x, 0, 0)) - val(p - vec3(h.x, 0, 0)), val(p + vec3(0, h.y, 0)) - val(p - vec3(0, h.y, 0)),
                  val(p + vec3(0, 0, h.z)) - val(p - vec3(0, 0, h.z))) / (2.0 * h);
    return length(g) > 1e-6 ? -normalize(g) : vec3(0, 0, 1);
  }
  vec3 shade(vec3 c, vec3 n, vec3 rd) {
    vec3 L = normalize(vec3(-0.4, -0.5, 0.75));
    float d = max(dot(n, L), 0.0), a = 0.35 + 0.15 * n.z;
    float s = pow(max(dot(reflect(-L, n), -rd), 0.0), 24.0) * 0.25;
    return c * (a + 0.75 * d) + s;
  }
  float edge(vec3 p) {   // box edges: close to two faces at once
    vec3 q = vec3(abs(p.x) - size.x * 0.5, abs(p.y) - size.y * 0.5, min(p.z, size.z - p.z));
    vec3 d = vec3(abs(q.x), abs(q.y), abs(q.z));
    float w = 0.004 * length(eye);
    int n = int(d.x < w) + int(d.y < w) + int(d.z < w);
    return n >= 2 ? 1.0 : 0.0;
  }
  void main() {
    vec4 a = invVP * vec4(ndc, -1.0, 1.0), b = invVP * vec4(ndc, 1.0, 1.0);
    vec3 ro = eye, rd = normalize(b.xyz / b.w - a.xyz / a.w);
    vec3 lo = vec3(-size.x * 0.5, -size.y * 0.5, 0.0), hi = vec3(size.x * 0.5, size.y * 0.5, size.z);
    vec3 t0 = (lo - ro) / rd, t1 = (hi - ro) / rd;
    vec3 tmn = min(t0, t1), tmx = max(t0, t1);
    float tn = max(max(tmn.x, tmn.y), max(tmn.z, 0.0)), tf = min(min(tmx.x, tmx.y), tmx.z);
    vec3 bg = vec3(0.043, 0.055, 0.075) + vec3(0.03, 0.04, 0.06) * (1.0 - ndc.y) * 0.5;
    if (tf <= tn) { frag = vec4(bg, 1.0); return; }
    float diag = length(size), dt = diag / 600.0;
    vec4 acc = vec4(0.0);
    float t = tn + dt * 0.5;
    bool hit = false;
    float tgr = -1.0;                                      // where the ray meets raised ground (terrain), if it does
    for (int i = 0; i < 1200; i++) {
      if (t > tf || acc.a > 0.97) break;
      vec3 p = ro + rd * t;
      if (hasG == 1 && p.z < gz(p.xy)) { tgr = gcross(ro, rd, t - dt, t); break; }
      float c = val(p);
      if (inside(c)) {
        if (mode == 0) {                                   // isosurface: refine the crossing, light it
          float ta = t - dt, tb = t;
          for (int k = 0; k < 6; k++) { float tm = 0.5 * (ta + tb); if (inside(val(ro + rd * tm))) tb = tm; else ta = tm; }
          vec3 q = ro + rd * tb;
          bool up = val(q + rd * dt) >= thrHi || signedField == 0;
          vec3 base = col(up ? thrHi : thrLo).rgb;
          vec3 n = normalAt(q);
          if (!up) n = -n;
          acc = vec4(shade(base, n, rd), 1.0);
          hit = true;
          break;
        } else {                                           // translucent volume, lit by the local gradient
          vec4 s = col(c);
          float al = clamp(s.a * opacity * dt * 22.0, 0.0, 1.0);
          vec3 sc = shade(s.rgb, normalAt(p), rd);
          acc.rgb += (1.0 - acc.a) * al * sc;
          acc.a += (1.0 - acc.a) * al;
        }
      }
      t += dt;
    }
    vec3 rgb = acc.rgb;
    float rest = 1.0 - acc.a;
    if (!hit && rest > 0.0 && tgr > 0.0) {               // the terrain: shaded relief with the grid
      vec3 g = ro + rd * tgr;
      vec3 n = gnormal(g.xy);
      float d = max(dot(n, normalize(vec3(-0.4, -0.5, 0.75))), 0.0);
      vec2 f = abs(fract(g.xy / gridStep + 0.5) - 0.5) * gridStep;
      float w = 0.0025 * length(eye);
      float line = (f.x < w || f.y < w) ? 1.0 : 0.0;
      vec3 c = mix(vec3(0.15, 0.19, 0.16), vec3(0.30, 0.27, 0.22), clamp(g.z / (0.25 * size.z), 0.0, 1.0));
      c = mapAt(g.xy, c);
      c = c * (0.45 + 0.75 * d);
      rgb += rest * mix(c, vec3(0.30, 0.35, 0.40), line * (hasMap == 1 ? 0.25 : 0.6)); rest = 0.0;
    }
    if (!hit && rest > 0.0) {
      // the ground (z = 0) under the box, with a grid
      vec3 ground = vec3(0.10, 0.12, 0.14);
      if (rd.z < 0.0) {
        float tg = -ro.z / rd.z;
        vec3 g = ro + rd * tg;
        if (abs(g.x) <= size.x * 0.5 && abs(g.y) <= size.y * 0.5) {
          vec2 f = abs(fract(g.xy / gridStep + 0.5) - 0.5) * gridStep;
          float w = 0.0025 * length(eye);
          float line = (f.x < w || f.y < w) ? 1.0 : 0.0;
          ground = hasMap == 1 ? mix(mapAt(g.xy, vec3(0.13)) * 0.85, vec3(0.45, 0.50, 0.56), line * 0.3)
                               : mix(vec3(0.13, 0.15, 0.18), vec3(0.27, 0.31, 0.36), line);
          rgb += rest * ground; rest = 0.0;
        }
      }
      rgb += rest * bg;
    }
    float e = max(edge(ro + rd * tn), edge(ro + rd * tf));
    rgb = mix(rgb, vec3(0.55, 0.62, 0.70), e * 0.6);
    frag = vec4(rgb, 1.0);
  }`;

  // Cloud mode: what the storm looks like to the eye (after the Godot tornado sim's volume renderer).  The cloud
  // (cloud water, ice, snow) and the precipitation (rain, graupel, hail) in g/kg (two channels, log-coded) absorb
  // and scatter sunlight: extinction per km ~25 x the cloud's g/kg and ~1.2 x the precipitation's.  Read smooth
  // (tricubic), the cloud's thin edges eroded into wisps and billows by a 3-D noise and the rain drawn as curtains,
  // each sample lit by the sun through a shadow ray with a few multiple-scattering octaves, a powder term (dark
  // rims, glowing insides), a two-lobe phase (silver edges toward the sun) and the sky's light, which the cloud
  // above blocks (dark bases, the gloom under the anvil).  Haze grows with distance; the frames are jittered and
  // averaged while the view is still.
  const FS_CLOUD = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 ndc; out vec4 frag;
  uniform sampler3D vol; uniform sampler3D noiseTex;
  uniform mat4 invVP; uniform vec3 eye; uniform vec3 size; uniform vec3 toKm; uniform vec3 sunDir; uniform float density, rain;
  uniform sampler2D gnd; uniform int hasG; uniform float relief; uniform vec3 texel;
  uniform sampler2D mapTex; uniform int hasMap, mapDark;
  uniform float detail, detailKm, frameSeed; uniform vec3 kmOrigin; uniform int quality;
  uniform float mainDx; uniform float insDx[4];               // km: the model grids' spacing (box, nests)
  uniform sampler3D lightTex; uniform int useLight;           // the light volume (FS_LIGHT): sun and sky depth
  uniform float hazeKm;                                       // the visibility: the air's haze grows over it
  uniform sampler2D mapTex2; uniform int hasMap2; uniform vec4 map2Box;   // the sharp ground around the camera
  uniform sampler2D geoA, geoT; uniform int hasGeo;           // the trees (landscape.js): colour, distance
  vec3 mapAt(vec2 xy, vec3 fallback) {      // the ground's picture under xy: the box's map, the sharp one close up
    vec3 c = fallback;
    vec2 uv = vec2(xy.x / size.x + 0.5, xy.y / size.y + 0.5);
    if (hasMap == 1 && all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)))) {
      c = texture(mapTex, uv).rgb;
      if (mapDark == 1) { float L = dot(c, vec3(0.333)); c = clamp(1.0 - 2.0 * L + c, 0.0, 1.0) * 0.8; }
    }
    if (hasMap2 == 1) {
      vec2 u2 = (xy - map2Box.xy) / (map2Box.zw - map2Box.xy);
      if (all(greaterThanEqual(u2, vec2(0.0))) && all(lessThanEqual(u2, vec2(1.0)))) {
        vec2 e = min(u2, 1.0 - u2) / 0.06;                    // faded into the box's map at its edges
        vec3 d = texture(mapTex2, u2).rgb;
        if (mapDark == 1) { float L = dot(d, vec3(0.333)); d = clamp(1.0 - 2.0 * L + d, 0.0, 1.0) * 0.8; }
        c = mix(c, d, clamp(min(e.x, e.y), 0.0, 1.0));
      }
    }
    return c;
  }
  float gz(vec2 xy) {                       // the ground's height (box units) under xy; 0 outside the box / flat runs
    if (hasG == 0) return 0.0;
    vec2 uv = vec2(xy.x / size.x + 0.5, xy.y / size.y + 0.5);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    return texture(gnd, uv).r * size.z;
  }
  vec3 gnormal(vec2 xy) {
    vec2 h = 0.004 * size.xy;
    return normalize(vec3(gz(xy - vec2(h.x, 0)) - gz(xy + vec2(h.x, 0)), gz(xy - vec2(0, h.y)) - gz(xy + vec2(0, h.y)),
                          2.0 * h.x));
  }
  float gcross(vec3 ro, vec3 rd, float ta, float tb) {
    for (int k = 0; k < 8; k++) { float tm = 0.5 * (ta + tb); vec3 p = ro + rd * tm; if (p.z < gz(p.xy)) tb = tm; else ta = tm; }
    return tb;
  }
  // the nests (volume.py insets): up to four finer volumes, finest first, each over part of the box from the ground
  // to its own top; a point inside one reads it, blended into the box's volume over its outer 8 %
  uniform sampler3D ins0, ins1, ins2, ins3; uniform int insN; uniform vec4 insBox[4]; uniform float insTop[4];
  float insW(vec3 p, int i, out vec3 u) {
    vec4 b = insBox[i];
    u = vec3((p.x - b.x) / (b.z - b.x), (p.y - b.y) / (b.w - b.y), p.z / insTop[i]);
    if (any(lessThan(u, vec3(0.0))) || any(greaterThan(u, vec3(1.0)))) return 0.0;
    vec2 e = min(u.xy, 1.0 - u.xy);
    return clamp(min(min(e.x, e.y), 1.0 - u.z) / 0.08, 0.0, 1.0);
  }
  vec4 nested(vec3 p, vec4 base) {
    if (insN == 0) return base;
    vec3 u; float w = insW(p, 0, u);
    if (w > 0.0) return mix(base, textureLod(ins0, u, 0.0), w);
    if (insN > 1) { w = insW(p, 1, u); if (w > 0.0) return mix(base, textureLod(ins1, u, 0.0), w); }
    if (insN > 2) { w = insW(p, 2, u); if (w > 0.0) return mix(base, textureLod(ins2, u, 0.0), w); }
    if (insN > 3) { w = insW(p, 3, u); if (w > 0.0) return mix(base, textureLod(ins3, u, 0.0), w); }
    return base;
  }
  vec3 SUN;                                                   // the sunlight: warmer and dimmer as the sun gets low
  // smooth (C2) tricubic B-spline read from 8 trilinear ones (Sigg & Hadwiger 2005): no grid facets in the light
  vec2 cubic(vec3 uvw) {
    vec3 sz = vec3(textureSize(vol, 0));
    vec3 c = uvw * sz - 0.5, i = floor(c), f = c - i, f2 = f * f, f3 = f2 * f;
    vec3 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0, w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec3 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0, w3 = f3 / 6.0;
    vec3 g0 = w0 + w1, g1 = w2 + w3;
    vec3 t0 = (i - 0.5 + w1 / g0) / sz, t1 = (i + 1.5 + w3 / g1) / sz;
    vec2 r = vec2(0.0);
    for (int k = 0; k < 8; k++) {
      vec3 hi = vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
      vec3 g = mix(g0, g1, hi);
      r += g.x * g.y * g.z * textureLod(vol, mix(t0, t1, hi), 0.0).rg;
    }
    return r;
  }
  vec2 gkg(vec2 c) {                                          // codes (0..255) -> g/kg
    c *= 255.0;
    return vec2(c.x < 0.5 ? 0.0 : pow(10.0, (c.x - 1.0) / 40.0 - 3.0), c.y < 0.5 ? 0.0 : pow(10.0, (c.y - 1.0) / 40.0 - 3.0));
  }
  // extinction per km (cloud, precipitation) at p.  mode 0: plain trilinear (shadow and sky rays); 1: with the
  // sub-grid detail; 2: that from the smooth (tricubic) read (the view's own march)
  vec2 media(vec3 p, int mode) {
    vec3 uvw = vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z);
    if (any(lessThan(uvw, vec3(-0.01))) || any(greaterThan(uvw, vec3(1.01)))) return vec2(0.0);
    vec3 k = p * toKm + kmOrigin;                              // km on the ground: the detail stays put as the box moves
    float det = detail;
    if (mode > 0 && detail > 0.0) {
      // the detail is the structure a coarse grid has not got: none where the model's own grid resolves it (a
      // tornado zoom's nests, ~37-110 m), whose cloud it shredded -- the funnel's thin lower part eroded away
      float dxl = mainDx > 0.0 ? mainDx : 1.0e3;
      vec3 bp = vec3((uvw.x - 0.5) * size.x, (uvw.y - 0.5) * size.y, uvw.z * size.z), iu;
      for (int i = 0; i < 4; i++) {
        if (i < insN && insDx[i] > 0.0 && insW(bp, i, iu) > 0.0) dxl = min(dxl, insDx[i]);
      }
      det = detail * smoothstep(0.12, 0.35, dxl);
    }
    if (mode > 0 && det > 0.0) {
      // domain warp: the surface pushed in and out by about a grid cell -- towers, billows, cauliflower
      vec3 wq = k / (detailKm * 5.0);
      vec3 wv = vec3(texture(noiseTex, wq).rg, texture(noiseTex, wq.zxy + 0.41).r) - 0.5;
      uvw += wv * texel * 2.2 * det;
    }
    vec2 raw = mode == 2 && quality > 0 ? cubic(uvw) : texture(vol, uvw).rg;
    if (insN > 0) raw = nested(vec3((uvw.x - 0.5) * size.x, (uvw.y - 0.5) * size.y, uvw.z * size.z), vec4(raw, 0.0, 0.0)).rg;
    vec2 m = gkg(raw);
    vec2 s = vec2(25.0 * m.x, 1.2 * rain * m.y) * density;   // rain: the Rain slider (x the model's)
    vec2 e = min(uvw.xy, 1.0 - uvw.xy) / 0.03;                 // the box's sides fade: no walls of cloud
    s *= clamp(min(e.x, e.y), 0.0, 1.0);
    if (mode > 0 && det > 0.0 && s.x + s.y > 1e-3) {
      float n = texture(noiseTex, k / (detailKm * 20.0)).r * 0.55 + texture(noiseTex, k / (detailKm * 7.0) + 0.37).r * 0.45;
      // erosion: thin cloud breaks into wisps and billows, the solid core keeps its shape
      float cov = clamp(s.x / 30.0, 0.0, 1.0);
      float thr = 0.8 * det * smoothstep(0.3, 0.7, n) * (1.0 - cov);
      float kept = clamp((cov - thr) / max(1.0 - thr, 0.05), 0.0, 1.0);
      s.x *= cov > 1e-4 ? kept / cov : 0.0;
      s.x *= max(mix(1.0, 2.6 * n - 0.3, 0.55 * det), 0.0);   // lumps inside, the mean kept
      // rain curtains: streaks that hang down, thicker and thinner side by side
      float r = texture(noiseTex, vec3(k.xy / (detailKm * 9.0), k.z / (detailKm * 90.0)) + 0.61).g;
      s.y *= max(mix(1.0, 2.4 * r - 0.2, 0.8 * det), 0.0);
    }
    return s;
  }
  float sig(vec2 s) { return s.x + s.y; }
  float hg(float c, float g) { return (1.0 - g * g) / (12.566 * pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5)); }
  float sunTau(vec3 p, vec3 wd, float kmPer) {
    float tau = 0.0, t = 0.0, ds = 0.01 * length(size);
    for (int j = 0; j < 7; j++) { t += ds; tau += sig(media(p + wd * t, 0)) * ds * kmPer; ds *= 1.6; }
    return tau;
  }
  float upTau(vec3 p) {                                       // the cloud straight above: how much sky gets through
    float tau = 0.0, z = 0.0;
    for (int j = 0; j < 4; j++) {
      float dk = 0.6 * pow(2.0, float(j));                     // km: 0.6, 1.2, 2.4, 4.8
      z += dk / toKm.z;
      tau += sig(media(p + vec3(0.0, 0.0, z), 0)) * dk;
    }
    return tau;
  }
  vec2 lightAt(vec3 p, vec3 wd, float kmPer) {             // optical depth to the sun, and to the sky above
    vec3 uvw = vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z);
    if (useLight == 1 && all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0))))
      return texture(lightTex, uvw).rg;
    return vec2(sunTau(p, wd, kmPer), upTau(p));
  }
  vec3 sky(vec3 rd) {
    float h = max(rd.z, 0.0), s = max(dot(rd, sunDir), 0.0);
    vec3 c = mix(vec3(0.66, 0.76, 0.88), vec3(0.11, 0.28, 0.64), pow(h, 0.5));
    float low = 1.0 - smoothstep(0.02, 0.4, sunDir.z);
    c = mix(c, vec3(1.0, 0.60, 0.34), low * pow(s, 5.0) * (1.0 - h) * 0.85);
    c *= mix(0.45, 1.0, smoothstep(-0.02, 0.3, sunDir.z));
    return c + SUN * (smoothstep(0.99955, 0.9998, s) * 8.0 + pow(s, 80.0) * 0.10);
  }
  vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
  void main() {
    float low = smoothstep(0.02, 0.45, sunDir.z);
    SUN = mix(vec3(1.0, 0.55, 0.28) * 2.4, vec3(1.0, 0.95, 0.88) * 3.4, low);
    vec4 a = invVP * vec4(ndc, -1.0, 1.0), b = invVP * vec4(ndc, 1.0, 1.0);
    vec3 ro = eye, rd = normalize(b.xyz / b.w - a.xyz / a.w);
    vec3 wd = normalize(sunDir / toKm);                      // the sun's direction in the (stretched) box
    float kmPerW = length(wd * toKm);
    vec3 amb = vec3(0.40, 0.50, 0.68);
    vec3 haze = mix(vec3(0.70, 0.62, 0.55), vec3(0.64, 0.71, 0.80), low);
    float jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy + 5.588238 * frameSeed, vec2(0.06711056, 0.00583715))));
    vec3 bg = sky(rd);
    float tg = rd.z < 0.0 ? -ro.z / rd.z : 1e9;
    vec3 lo = vec3(-size.x * 0.5, -size.y * 0.5, 0.0), hi = vec3(size.x * 0.5, size.y * 0.5, size.z);
    vec3 t0 = (lo - ro) / rd, t1 = (hi - ro) / rd;
    vec3 tmn = min(t0, t1), tmx = max(t0, t1);
    float tn = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
    float tx = min(min(tmx.x, tmx.y), tmx.z);
    if (hasG == 1 && tx > tn) {                               // raised ground in the box: where the ray first meets it
      float te = min(tx, tg), ds = length(size) / 500.0;
      for (int i = 0; i < 1000; i++) {
        float t = tn + (float(i) + jit) * ds;
        if (t >= te) break;
        vec3 p = ro + rd * t;
        if (p.z < gz(p.xy)) { tg = gcross(ro, rd, max(t - ds, 0.0), t); break; }
      }
    }
    bool tree = false; vec4 ga = vec4(0.0);
    if (hasGeo == 1) {                                        // a tree in front of the ground
      vec2 su = gl_FragCoord.xy / vec2(textureSize(geoT, 0));
      float tt = texture(geoT, su).r;
      if (tt > 0.0 && tt < tg) { tg = tt; tree = true; ga = texture(geoA, su); }
    }
    if (tg < 1e8) {
      vec3 g = ro + rd * tg;
      vec2 lt = lightAt(g + vec3(0, 0, 1e-4), wd, kmPerW);
      float lit = exp(-lt.x), skyv = exp(-0.5 * lt.y);
      vec3 n = hasG == 1 ? gnormal(g.xy) : vec3(0, 0, 1);
      float sl = max(dot(n, sunDir), 0.0);
      vec3 k = g * toKm + kmOrigin;
      // fields and woods, down to a few metres (the noise's mipmaps keep the distance from shimmering)
      float v = texture(noiseTex, k / 6.0).g * 0.4 + texture(noiseTex, k / 1.3).r * 0.25 + texture(noiseTex, k / 0.25).g * 0.15
              + texture(noiseTex, k / 0.03).r * 0.12 + texture(noiseTex, k / 0.004).g * 0.08;
      vec3 alb = mix(vec3(0.15, 0.21, 0.08), vec3(0.28, 0.27, 0.16), smoothstep(0.15, 0.85, v));
      alb *= 0.9 + 0.2 * texture(noiseTex, k / 0.6 + 0.29).r;
      if (hasMap == 1 || hasMap2 == 1) { alb = pow(mapAt(g.xy, pow(alb, vec3(1.0 / 2.2))), vec3(2.2)) * 0.9; }
      if (tree) { alb = pow(ga.rgb, vec3(2.2)); sl = ga.a * smoothstep(0.0, 0.15, sunDir.z); n = vec3(0, 0, 1); }
      vec3 grd = alb * (SUN * sl * lit * 0.35 + amb * (0.10 + 0.30 * skyv) * (0.6 + 0.4 * n.z));
      float dist = length((g - ro) * toKm);
      bg = mix(grd, haze * mix(0.5, 1.0, skyv), 1.0 - exp(-dist / hazeKm));
    }
    float tf = min(tx, tg);
    vec3 col = vec3(0.0); float T = 1.0;
    if (tf > tn) {
      float dt = length(size) / (quality > 0 ? 420.0 : 260.0), kmPer = length(rd * toKm);
      float cth = dot(rd, sunDir);
      float t = tn + dt * jit;                                // jittered start: no banding, averaged away
      for (int i = 0; i < 900; i++) {
        if (t > tf || T < 0.01) break;
        vec3 p = ro + rd * t;
        vec2 s = media(p, 2);
        float sg = sig(s);
        if (sg <= 1e-3) { t += dt * 1.8; continue; }            // empty air: longer steps
        {
          float ds = dt * kmPer;
          vec2 lt = lightAt(p, wd, kmPerW);
          float ts = lt.x;
          // multiple scattering octaves (Wrenninge 2015): each one dimmer, less shadowed, less forward
          float L = 0.0, aa = 1.0, bb = 1.0, cc = 1.0;
          for (int o = 0; o < 3; o++) {
            L += aa * exp(-bb * ts) * mix(hg(cth, 0.7 * cc), hg(cth, -0.25 * cc), 0.3);
            aa *= 0.5; bb *= 0.4; cc *= 0.5;
          }
          float pw = 1.0 - exp(-2.0 * sg * 0.35);              // powder: thin edges send less back
          L *= mix(1.0, pw, 0.55 * (0.6 - 0.4 * cth));
          float skyv = exp(-0.45 * lt.y);
          float hgt = clamp(p.z / size.z * 2.0, 0.0, 1.0);
          float albedo = mix(0.98, 0.78, s.y / (sg + 1e-6));     // rain shafts grey
          vec3 S = albedo * (SUN * L * 4.5 + amb * (0.18 + 0.42 * skyv) * (0.55 + 0.45 * hgt) * (0.45 + 0.55 * exp(-0.12 * ts)));
          if (relief > 0.0) {
            // relief: the cloud's own lumps lit as a surface (its outward normal from the density's gradient)
            vec3 h = texel * size;
            float c0 = sig(media(p, 1));
            vec3 gr = vec3(sig(media(p + vec3(h.x, 0, 0), 1)) - c0, sig(media(p + vec3(0, h.y, 0), 1)) - c0,
                           sig(media(p + vec3(0, 0, h.z), 1)) - c0) / h;
            if (length(gr) > 1e-4) {
              vec3 n = -normalize(gr);
              float lam = max(dot(n, wd), 0.0);
              S *= mix(1.0, 0.35 + 1.1 * lam + 0.15 * max(n.z, 0.0), 0.6 * relief);
            }
          }
          float fog = 1.0 - exp(-t * kmPer / hazeKm);           // the air between: hazier with distance
          S = mix(S, haze * mix(0.5, 1.0, skyv), fog);
          float al = 1.0 - exp(-sg * ds);
          col += T * al * S;
          T *= 1.0 - al;
        }
        t += dt;
      }
    }
    vec3 c = col + T * bg;
    c = aces(c * 0.85);
    frag = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
  }`;

  // The cloud view's light volume: at every point of the box, the optical depth toward the sun (a long march) and
  // straight up to the sky, rendered layer by layer into a half-float 3-D texture whenever the data, the sun, the
  // density or the heights change; the view's march then reads it instead of marching its own shadow rays.
  const FS_LIGHT = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 ndc; out vec4 frag;
  uniform sampler3D vol; uniform vec3 size, toKm, sunDir, lres; uniform float density, rain, layer;
  float sig0(vec3 p) {
    vec3 uvw = vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z);
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return 0.0;
    vec2 c = texture(vol, uvw).rg * 255.0;
    vec2 m = vec2(c.x < 0.5 ? 0.0 : pow(10.0, (c.x - 1.0) / 40.0 - 3.0), c.y < 0.5 ? 0.0 : pow(10.0, (c.y - 1.0) / 40.0 - 3.0));
    vec2 e = min(uvw.xy, 1.0 - uvw.xy) / 0.03;
    return (25.0 * m.x + 1.2 * rain * m.y) * density * clamp(min(e.x, e.y), 0.0, 1.0);
  }
  void main() {
    vec3 uvw = vec3(gl_FragCoord.xy / lres.xy, (layer + 0.5) / lres.z);
    vec3 p = vec3((uvw.x - 0.5) * size.x, (uvw.y - 0.5) * size.y, uvw.z * size.z);
    vec3 wd = normalize(sunDir / toKm);
    float kmPer = length(wd * toKm);
    // from a voxel out toward the sun: the coarse grid's cloud edge would otherwise shadow its own lit skin
    float t = length(wd * size / lres), ts = 0.0, ds = 0.003 * length(size);
    for (int j = 0; j < 22; j++) { t += ds; ts += sig0(p + wd * (t - 0.5 * ds)) * ds * kmPer; ds *= 1.22; }
    float tu = 0.0, z = size.z / lres.z, dk = 0.25;
    for (int j = 0; j < 9; j++) { z += dk / toKm.z; tu += sig0(p + vec3(0.0, 0.0, z - 0.5 * dk / toKm.z)) * dk; dk *= 1.45; }
    frag = vec4(ts, tu, 0.0, 1.0);
  }`;

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function link(fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  function initGL() {
    if (gl) return true;
    gl = $('volCanvas').getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (!gl) return false;
    prog = link(FS);
    cloudProg = link(FS_CLOUD);
    if (gl.getExtension('EXT_color_buffer_float')) { lightProg = link(FS_LIGHT); lightFbo = gl.createFramebuffer(); lightTex = gl.createTexture(); }
    vao = gl.createVertexArray();
    volTex = gl.createTexture(); lutTex = gl.createTexture(); gndTex = gl.createTexture(); mapTex = gl.createTexture();
    noiseTex = makeNoise();
    for (let i = 0; i < 4; i++) insTex.push(gl.createTexture());
    try { Land.init(gl, () => { if (head) draw(); }); } catch (e) { console.warn('3-D landscape:', e); }
    return true;
  }

  // a tileable 3-D noise, 64^3, two channels: r billowy (inverted Worley fBm over value noise), g value-noise fBm
  function makeNoise(g) {
    const gl = g || glNow();
    const N = 64, out = new Uint8Array(N * N * N * 2);
    let seed = 1234567;
    const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    const lattice = C => { const a = new Float32Array(C * C * C * 3); for (let i = 0; i < a.length; i++) a[i] = rnd(); return a; };
    const W = [4, 8, 16].map(C => ({ C, p: lattice(C) })), V = [4, 8, 16, 32].map(C => ({ C, p: lattice(C) }));
    const worley = (x, y, z, { C, p }) => {
      const fx = x * C, fy = y * C, fz = z * C, ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
      let best = 9;
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const cx = ix + dx, cy = iy + dy, cz = iz + dz;
        const k = 3 * ((((cz % C) + C) % C) * C * C + (((cy % C) + C) % C) * C + (((cx % C) + C) % C));
        const ddx = cx + p[k] - fx, ddy = cy + p[k + 1] - fy, ddz = cz + p[k + 2] - fz;
        best = Math.min(best, ddx * ddx + ddy * ddy + ddz * ddz);
      }
      return Math.min(1, Math.sqrt(best));
    };
    const value = (x, y, z, { C, p }) => {
      const fx = x * C, fy = y * C, fz = z * C, ix = Math.floor(fx), iy = Math.floor(fy), iz = Math.floor(fz);
      const s = t => t * t * (3 - 2 * t), u = s(fx - ix), v = s(fy - iy), w = s(fz - iz);
      const at = (a, b, c) => p[3 * (((c % C) * C + (b % C)) * C + (a % C))];
      const l = (a, b, t) => a + (b - a) * t;
      return l(l(l(at(ix, iy, iz), at(ix + 1, iy, iz), u), l(at(ix, iy + 1, iz), at(ix + 1, iy + 1, iz), u), v),
               l(l(at(ix, iy, iz + 1), at(ix + 1, iy, iz + 1), u), l(at(ix, iy + 1, iz + 1), at(ix + 1, iy + 1, iz + 1), u), v), w);
    };
    for (let k = 0, o = 0; k < N; k++) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++, o += 2) {
      const x = (i + 0.5) / N, y = (j + 0.5) / N, z = (k + 0.5) / N;
      const vn = 0.5 * value(x, y, z, V[0]) + 0.27 * value(x, y, z, V[1]) + 0.15 * value(x, y, z, V[2]) + 0.08 * value(x, y, z, V[3]);
      const wn = 1 - (0.6 * worley(x, y, z, W[0]) + 0.27 * worley(x, y, z, W[1]) + 0.13 * worley(x, y, z, W[2]));
      out[o] = Math.round(255 * Math.min(1, Math.max(0, 0.65 * wn + 0.35 * vn)));
      out[o + 1] = Math.round(255 * Math.min(1, Math.max(0, (vn - 0.2) / 0.6)));
    }
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, t);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, N, N, N, 0, gl.RG, gl.UNSIGNED_BYTE, out);
    gl.generateMipmap(gl.TEXTURE_3D);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                          [gl.TEXTURE_WRAP_S, gl.REPEAT], [gl.TEXTURE_WRAP_T, gl.REPEAT], [gl.TEXTURE_WRAP_R, gl.REPEAT]])
      gl.texParameteri(gl.TEXTURE_3D, k, v);
    return t;
  }

  const glNow = () => gl;

  // ---------------------------------------------------------------- data
  const field = () => $('volField').value;
  const isoMode = () => $('volMode').value === 'iso';
  const cloudMode = () => $('volMode').value === 'cloud';
  const want = () => (cloudMode() ? 'cloud' : field());       // what the server is asked for

  function setSlider(keepValue) {
    const F = FIELDS[field()], s = $('volThr');
    s.min = F.min; s.max = F.max; s.step = F.step;
    if (!keepValue) s.value = isoMode() ? F.iso : F.vol;
    $('volThrV').textContent = F.fmt(+s.value);
    const c = cloudMode();
    $('volBoth').parentElement.classList.toggle('hidden', !F.signed || c);
    for (const id of ['volField', 'volThrLbl']) $(id).classList.toggle('hidden', c);
    for (const id of ['volSunLbl', 'volSunHLbl', 'volReliefLbl', 'volDetailLbl', 'volRainLbl', 'volTreesLbl']) $(id).classList.toggle('hidden', !c);
    showTime();
    $('volOpaName').textContent = c ? 'Density' : 'Opacity';
    // the tornado view reads the model's state (no radar reflectivity there), and has its pressure drop
    document.querySelectorAll('.volTor').forEach(e => e.classList.toggle('hidden', !tor));
    document.querySelectorAll('.volTorOpt').forEach(e => { e.hidden = !tor; });
    const dbz = $('volField').querySelector('option[value="dbz"]');
    if (dbz) dbz.hidden = tor;
    $('volTopV').textContent = `${$('volTop').value} km`;
  }

  async function uploadLUT() {
    const F = FIELDS[field()];
    let pal = F.fam === 'w' ? { stops: W_STOPS } : F.fam === 'pp' ? { stops: P_STOPS } : Colors.DEFAULT_PALETTES[F.fam];
    if (F.fam === 'dbz' && Colors.PRODUCT_PALETTE[S.product] === 'dbz') {
      try { pal = await paletteFor(S.product); } catch (e) { /* the default table */ }
    }
    const lut = Colors.buildLUT(pal, head.scale, head.offset);
    for (let c = 1; c < 256; c++) {
      const v = (c - head.offset) / head.scale;
      if (lut[c * 4 + 3] === 0 && field() !== 'dbz') {
        // a table that fades its weak end out: the isosurface still needs a colour there
        const cc = Colors.colorAt(pal.stops, v);
        if (cc) { lut[c * 4] = cc[0]; lut[c * 4 + 1] = cc[1]; lut[c * 4 + 2] = cc[2]; }
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, lutTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(lut.buffer));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  async function load(force) {
    if (tor) {                                   // the box goes where the tornado zoom's finest nest is now
      const tz = S.run && S.run.tornado;
      if (tz && tz.box) box = tz.box.map(v => +(v / 1000).toFixed(2));
    }
    if (!box) return;
    if (busy) { again = true; return; }           // a change during a load: loaded again right after it
    const past = replayT !== null && !tor;
    busy = true;
    if (follow.on && !tor) {                      // Follow tornado: the box onto it at this volume's time first
      try { await recentre(past ? replayT : null); } catch (e) { console.warn('follow:', e); }
      if (typeof map !== 'undefined') map.redraw();
    }
    const key = `${box.join(',')}/${want()}/${past ? replayT : 'now'}`;
    if (!cache.has(key)) $('volStatus').textContent = 'Loading…';
    try {
      let q = `x0=${box[0]}&y0=${box[1]}&x1=${box[2]}&y1=${box[3]}&field=${want()}`;
      if (tor) q += `&live=1&ztop=${$('volTop').value}`;
      if (past) q += `&t=${replayT}`;
      let buf = past ? cache.get(key) : null;
      if (!buf) {
        buf = await (await api('/api/volume?' + q)).arrayBuffer();
        if (past) {                                 // the replay's volumes kept for looping (the oldest out first)
          cache.set(key, buf); cacheBytes += buf.byteLength;
          for (const [k, v] of cache) { if (cacheBytes <= CACHE_MAX) break; cache.delete(k); cacheBytes -= v.byteLength; }
        }
      }
      if (tor && ($('volPaths').checked || $('volTrack').checked)) {
        try { paths = await (await api('/api/tornado/paths')).json(); } catch (e) { paths = null; }
      }
      const { header, data } = unpack(buf);
      head = header;
      gMap = null; loadSeq++;
      const two = header.channels === 2;
      gl.bindTexture(gl.TEXTURE_3D, volTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, two ? gl.RG8 : gl.R8, header.nx, header.ny, header.nz, 0, two ? gl.RG : gl.RED,
                    gl.UNSIGNED_BYTE, data);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
                            [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
      if (header.ground) {          // over terrain: the ground under each column (uint16 m), as a fraction of the top
        const n = header.nx * header.ny, off = data.byteOffset + n * header.nz * header.channels;
        const g16 = new Uint16Array(data.buffer.slice(off, off + 2 * n)), gf = new Float32Array(n);
        for (let i = 0; i < n; i++) gf[i] = g16[i] / header.ztop;
        gMap = { nx: header.nx, ny: header.ny, g16 };
        gl.bindTexture(gl.TEXTURE_2D, gndTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, header.nx, header.ny, 0, gl.RED, gl.FLOAT, gf);
        for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                              [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
          gl.texParameteri(gl.TEXTURE_2D, k, v);
      }
      // the nests at their own spacing, after the box's points and its ground
      let off = data.byteOffset + header.nx * header.ny * header.nz * header.channels + (header.ground ? 2 * header.nx * header.ny : 0);
      header.insets = (header.insets || []).slice(0, 4);
      header.insets.forEach((ins, i) => {
        const n = ins.nx * ins.ny * ins.nz * header.channels;
        gl.bindTexture(gl.TEXTURE_3D, insTex[i]);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage3D(gl.TEXTURE_3D, 0, two ? gl.RG8 : gl.R8, ins.nx, ins.ny, ins.nz, 0, two ? gl.RG : gl.RED,
                      gl.UNSIGNED_BYTE, new Uint8Array(data.buffer, off, n));
        for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                              [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],
                              [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
        off += n;
      });
      if (!two) await uploadLUT();
      await loadMap();
      const run = past ? `  ·  replay ${clock(header.t)}` : S.run && S.run.t !== undefined ? `  ·  model +${fmtDur(S.run.t)}` : '';
      $('volStatus').textContent = (tor ? `Tornado zoom (${Math.round(header.dx)} m grid) · ` : '') +
        `${(box[2] - box[0]).toFixed(tor ? 1 : 0)} × ${(box[3] - box[1]).toFixed(tor ? 1 : 0)} km, ` +
        `${header.nx}×${header.ny}×${header.nz} points` +
        (header.insets.length ? ` + ${header.insets.length} nest${header.insets.length > 1 ? 's' : ''} down to ` +
          `${Math.round(Math.min(...header.insets.map(s => s.dx)))} m` : '') + run;
      if (key !== lastKey) lastKey = key;
      draw();
    } catch (e) {
      $('volStatus').textContent = (e && e.message) || 'No volume';
      if (force) { head = null; draw(); }
    }
    busy = false;
    if (again) { again = false; load(true); }
  }

  // ---------------------------------------------------------------- drawing
  function geometry() {
    const sx = box[2] - box[0], sy = box[3] - box[1];
    const k = 2 / Math.max(sx, sy);                      // world units: the longer side is 2
    const ex = +$('volEx').value;
    return { sx: sx * k, sy: sy * k, sz: (head ? head.ztop / 1000 : 18) * k * ex, k, ex };
  }

  function orbitEye(G) {
    const target = [cam.px, cam.py, G.sz * 0.3];
    const cp = Math.cos(cam.pitch);
    return { target, eye: [target[0] + cam.dist * cp * Math.cos(cam.yaw), target[1] + cam.dist * cp * Math.sin(cam.yaw),
                           target[2] + cam.dist * Math.sin(cam.pitch)] };
  }

  // the ground's height (km above the model's zero) at x, y km from the box's middle: 0 on flat runs / outside
  function groundKm(x, y) {
    if (!gMap || !box) return 0;
    const i = Math.floor((x / (box[2] - box[0]) + 0.5) * gMap.nx), j = Math.floor((y / (box[3] - box[1]) + 0.5) * gMap.ny);
    if (i < 0 || j < 0 || i >= gMap.nx || j >= gMap.ny) return 0;
    return gMap.g16[j * gMap.nx + i] / 1000;
  }

  function matrices(W, H) {
    const G = geometry();
    let eye, target, fov = 0.7;
    if (cam.mode === 'orbit' || !cam.fp) ({ eye, target } = orbitEye(G));
    else {
      const f = cam.fp;
      if (cam.mode === 'ground') f.z = groundKm(f.x, f.y) + EYE_KM;
      else f.z = Math.max(f.z, groundKm(f.x, f.y) + EYE_KM);
      eye = [f.x * G.k, f.y * G.k, f.z * G.k * G.ex];
      const cp = Math.cos(f.pitch);
      target = [eye[0] + cp * Math.cos(f.yaw), eye[1] + cp * Math.sin(f.yaw), eye[2] + Math.sin(f.pitch)];
      fov = cam.mode === 'ground' ? 1.15 : 1.0;
    }
    const V = lookAt(eye, target, [0, 0, 1]);
    const VP = mul(perspective(fov, W / H, 0.001, 50), V);
    return { G, eye, VP, V, fov };
  }

  // the free and ground cameras: from the orbit's eye (free), or out at the box's side looking in (ground)
  function setCamMode(m) {
    if (m === cam.mode) return;
    const side = Math.max(box[2] - box[0], box[3] - box[1]);
    if (m !== 'orbit') {
      const was = cam.mode;
      if (m === 'ground' && +$('volEx').value !== 1) { $('volEx').value = 1; }
      const G1 = geometry();
      if (was === 'orbit' || !cam.fp) {
        const { eye, target } = orbitEye(G1);
        const km = v => [v[0] / G1.k, v[1] / G1.k, v[2] / (G1.k * G1.ex)];
        const e = km(eye), t = km(target);
        if (m === 'free') {
          cam.fp = { x: e[0], y: e[1], z: e[2], yaw: Math.atan2(t[1] - e[1], t[0] - e[0]),
                     pitch: Math.atan2(t[2] - e[2], Math.hypot(t[0] - e[0], t[1] - e[1])) };
        } else {
          const d = Math.hypot(e[0] - t[0], e[1] - t[1]) || 1, ux = (e[0] - t[0]) / d, uy = (e[1] - t[1]) / d;
          const x = t[0] + ux * 0.42 * side, y = t[1] + uy * 0.42 * side;
          cam.fp = { x, y, z: 0, yaw: Math.atan2(t[1] - y, t[0] - x), pitch: 0.12 };
        }
      } else if (m === 'ground') cam.fp.pitch = Math.min(cam.fp.pitch, 0.6);
      cam.speed = m === 'ground' ? 0.4 : Math.max(0.5, 0.08 * side);
    }
    cam.mode = m; $('volCam').value = m; cam.placed = false;
    $('volOverlay').title = m === 'orbit' ? 'Drag: turn · right-drag or Shift-drag: move · wheel: zoom'
      : 'Drag: look around · W A S D: move · ' + (m === 'free' ? 'Q / E: down / up · ' : '') + 'Shift: faster · wheel: speed';
    if (m !== 'orbit') $('volOverlay').focus();
    draw();
  }

  function moveKeys(dt) {
    if (cam.mode === 'orbit' || !cam.fp || !keys.size) return false;
    const f = cam.fp, v = cam.speed * dt * (keys.has('ShiftLeft') || keys.has('ShiftRight') ? 5 : 1);
    const fw = (keys.has('KeyW') ? 1 : 0) - (keys.has('KeyS') ? 1 : 0), rt = (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0);
    const up = cam.mode === 'free' ? (keys.has('KeyE') ? 1 : 0) - (keys.has('KeyQ') ? 1 : 0) : 0;
    if (!fw && !rt && !up) return false;
    const cy = Math.cos(f.yaw), sy = Math.sin(f.yaw), cp = cam.mode === 'free' ? Math.cos(f.pitch) : 1;
    f.x += v * (fw * cy * cp + rt * sy);
    f.y += v * (fw * sy * cp - rt * cy);
    if (cam.mode === 'free') f.z = Math.max(0, f.z + v * (fw * Math.sin(f.pitch) + up));
    return true;
  }

  const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];

  // the nests' volumes for either shader, their boxes in the view's units
  function bindInsets(u, G) {
    const ins = (head && head.insets) || [], cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    const B = new Float32Array(16), T = new Float32Array(4).fill(1);
    ins.forEach((s, i) => {
      B.set([(s.box[0] / 1000 - cx) * G.k, (s.box[1] / 1000 - cy) * G.k, (s.box[2] / 1000 - cx) * G.k, (s.box[3] / 1000 - cy) * G.k], 4 * i);
      T[i] = s.ztop / 1000 * G.k * G.ex;
    });
    for (let i = 0; i < 4; i++) {
      gl.activeTexture(gl.TEXTURE9 + i); gl.bindTexture(gl.TEXTURE_3D, i < ins.length ? insTex[i] : null);
      gl.uniform1i(u('ins' + i), 9 + i);
    }
    gl.uniform1i(u('insN'), ins.length);
    gl.uniform4fv(u('insBox'), B); gl.uniform1fv(u('insTop'), T);
    // the model grids' spacing (km) for the cloud's detail (none where a grid resolves its own)
    const D = new Float32Array(4);
    ins.forEach((s, i) => { D[i] = (s.dx || 0) / 1000; });
    gl.uniform1fv(u('insDx'), D);
    const runDx = (head && head.dx) || (S.run && S.run.dx) || 0;
    gl.uniform1f(u('mainDx'), runDx ? Math.max(runDx / 1000, (box[2] - box[0]) / head.nx) : 0);
  }

  function bindGround(u) {
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, gndTex); gl.uniform1i(u('gnd'), 2);
    gl.uniform1i(u('hasG'), head && head.ground ? 1 : 0);
    const land = cloudMode() && baseName() === 'plain' ? Land.coarse() : null;
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, land || mapTex); gl.uniform1i(u('mapTex'), 3);
    gl.uniform1i(u('hasMap'), mapOn || land ? 1 : 0);
    gl.uniform1i(u('mapDark'), baseName() === 'osmdark' ? 1 : 0);
    const d = cloudMode() && cam.mode !== 'orbit' ? Land.detail() : null;
    gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, d ? d.tex : null); gl.uniform1i(u('mapTex2'), 8);
    gl.uniform1i(u('hasMap2'), d ? 1 : 0);
    if (d) {
      const k = 2 / Math.max(box[2] - box[0], box[3] - box[1]), cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
      gl.uniform4f(u('map2Box'), (d.box[0] - cx) * k, (d.box[1] - cy) * k, (d.box[2] - cx) * k, (d.box[3] - cy) * k);
    }
  }

  // the map picked on the map bar (app.js), drawn on the ground: one picture of the box from the server
  const baseName = () => { const e = $('basemap'); return e ? e.value : 'plain'; };
  async function loadMap() {
    const b = baseName();
    if (!box || b === 'plain') { mapOn = false; mapKey = ''; return; }
    const src = b.startsWith('sat') ? 'sat' : 'osm', labels = b === 'satlabels' ? 1 : 0;
    const key = `${box.join(',')}/${src}/${labels}`;
    if (key === mapKey) return;
    try {
      const r = await api(`/api/volume/ground?x0=${box[0]}&y0=${box[1]}&x1=${box[2]}&y1=${box[3]}&src=${src}&labels=${labels}&px=1536`);
      const bmp = await createImageBitmap(await r.blob());
      gl.bindTexture(gl.TEXTURE_2D, mapTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);         // row 0 (north) at the top: v = 1
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.generateMipmap(gl.TEXTURE_2D);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
        gl.texParameteri(gl.TEXTURE_2D, k, v);
      mapKey = key; mapOn = true;
    } catch (e) { mapOn = false; }
  }
  window.volBaseChanged = () => { if (gl && box) { Land.reset(); loadMap().then(() => { if (head) draw(); }); } };

  // the light volume for the current data, sun, density and heights (FS_LIGHT), rebuilt only when one changes
  function buildLight(G, sun, dens, rain) {
    if (!lightProg) return false;
    const key = [head.nx, head.ny, head.nz, loadSeq, ...sun.map(v => v.toFixed(4)), dens, rain, G.sx, G.sy, G.sz].join('/');
    if (key === lightKey) return true;
    const nx = head.nx, ny = head.ny, nz = head.nz;
    gl.bindTexture(gl.TEXTURE_3D, lightTex);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG16F, nx, ny, nz, 0, gl.RG, gl.HALF_FLOAT, null);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                          [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_3D, k, v);
    gl.useProgram(lightProg); gl.bindVertexArray(vao);
    const u = n => gl.getUniformLocation(lightProg, n);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, volTex); gl.uniform1i(u('vol'), 0);
    gl.uniform3f(u('size'), G.sx, G.sy, G.sz);
    gl.uniform3f(u('toKm'), 1 / G.k, 1 / G.k, 1 / (G.k * G.ex));
    gl.uniform3f(u('sunDir'), ...sun);
    gl.uniform3f(u('lres'), nx, ny, nz);
    gl.uniform1f(u('density'), dens);
    gl.uniform1f(u('rain'), rain);
    gl.bindFramebuffer(gl.FRAMEBUFFER, lightFbo);
    gl.viewport(0, 0, nx, ny);
    let ok = true;
    for (let k = 0; k < nz; k++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, lightTex, 0, k);
      if (k === 0 && gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { ok = false; break; }
      gl.uniform1f(u('layer'), k);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const cv = $('volCanvas'); gl.viewport(0, 0, cv.width, cv.height);
    lightKey = ok ? key : '';
    if (!ok) lightProg = null;                       // no float targets after all: the march's own shadow rays
    return ok;
  }

  function drawCloud(G, eye, VP, V, fov) {
    const az = +$('volSun').value * Math.PI / 180, el = +$('volSunH').value * Math.PI / 180;
    const sun = [Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el)], dens = 2 * +$('volOpa').value;
    const rain = +$('volRain').value;
    const blend = gl.isEnabled(gl.BLEND);
    if (blend) gl.disable(gl.BLEND);
    const lit = buildLight(G, sun, dens, rain);
    // close up: the real landscape around the camera (landscape.js) and its trees into the geometry buffer
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2, close = cam.mode !== 'orbit' && cam.fp && !tor;
    Land.want(tor ? null : box, baseName(), close ? { x: cam.fp.x + cx, y: cam.fp.y + cy } : null,
              (x, y) => groundKm(x - cx, y - cy));
    if (close && cam.mode === 'ground' && !cam.placed) {       // off the road or out of the woods: onto a road
      const s = Land.standNear(cam.fp.x + cx, cam.fp.y + cy);
      if (s) { cam.snap = [s[0] - cx - cam.fp.x, s[1] - cy - cam.fp.y]; cam.fp.x = s[0] - cx; cam.fp.y = s[1] - cy; cam.placed = true; acc = 0; }
    }
    const cvs = $('volCanvas');
    const geo = close && $('volTrees').checked && Land.drawTrees(cvs.width, cvs.height, V, fov, eye, G, sun, [cx, cy]);
    gl.viewport(0, 0, cvs.width, cvs.height);
    if (blend) gl.enable(gl.BLEND);
    gl.useProgram(cloudProg); gl.bindVertexArray(vao);
    const u = n => gl.getUniformLocation(cloudProg, n);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, volTex); gl.uniform1i(u('vol'), 0);
    bindGround(u);
    gl.uniformMatrix4fv(u('invVP'), false, invert(VP));
    gl.uniform3f(u('eye'), ...eye);
    gl.uniform3f(u('size'), G.sx, G.sy, G.sz);
    gl.uniform3f(u('toKm'), 1 / G.k, 1 / G.k, 1 / (G.k * G.ex));
    gl.uniform3f(u('sunDir'), Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el));
    gl.uniform1f(u('density'), dens);
    gl.uniform1f(u('rain'), rain);
    gl.uniform1f(u('relief'), +$('volRelief').value);
    gl.uniform3f(u('texel'), 1 / head.nx, 1 / head.ny, 1 / head.nz);
    bindInsets(u, G);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_3D, noiseTex); gl.uniform1i(u('noiseTex'), 4);
    gl.uniform1f(u('detail'), +$('volDetail').value);
    gl.uniform1f(u('detailKm'), Math.max(0.05, (box[2] - box[0]) / head.nx));
    gl.uniform1f(u('frameSeed'), acc);
    gl.uniform3f(u('kmOrigin'), (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0);
    gl.uniform1i(u('quality'), drawQ < 1 ? 0 : 1);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_3D, lit ? lightTex : null); gl.uniform1i(u('lightTex'), 5);
    gl.uniform1i(u('useLight'), lit ? 1 : 0);
    gl.uniform1f(u('hazeKm'), cam.mode === 'orbit' ? 900 : 180);
    const tr = Land.trees();
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, geo ? tr.colTex : null); gl.uniform1i(u('geoA'), 6);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, geo ? tr.tTex : null); gl.uniform1i(u('geoT'), 7);
    gl.uniform1i(u('hasGeo'), geo ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    $('volLegend').textContent = `Cloud view · sun in the ${COMPASS[Math.round(+$('volSun').value / 45) % 8]}, ` +
      `${$('volSunH').value}° up · height ×${G.ex}` + camNote();
  }

  function camNote() {
    if (cam.mode === 'orbit' || !cam.fp) return '';
    const f = cam.fp, h = cam.mode === 'ground' ? '' : ` · ${(f.z - groundKm(f.x, f.y)).toFixed(2)} km up`;
    const sp = cam.speed >= 1 ? `${cam.speed.toFixed(1)} km/s` : `${Math.round(cam.speed * 1000)} m/s`;
    return `  ·  ${cam.mode === 'ground' ? 'ground camera' : 'free camera'}${h} · ${sp} (wheel) · ` +
      `facing ${COMPASS[Math.round(((90 - f.yaw * 180 / Math.PI) % 360 + 360) % 360 / 45) % 8]}`;
  }

  // a point of the view (ground x, y m; z m up) on the overlay, or null behind the camera
  function projector(G, VP, W, H) {
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    return (x, y, z) => {
      const p = [(x / 1000 - cx) * G.k, (y / 1000 - cy) * G.k, z / 1000 * G.k * G.ex, 1], o = [0, 0, 0, 0];
      for (let r = 0; r < 4; r++) o[r] = VP[r] * p[0] + VP[4 + r] * p[1] + VP[8 + r] * p[2] + VP[12 + r] * p[3];
      if (o[3] <= 0) return null;
      return [(o[0] / o[3] * 0.5 + 0.5) * W, (1 - (o[1] / o[3] * 0.5 + 0.5)) * H];
    };
  }

  // the tornado view's overlays: the surface pressure minimum's track (orange, on the ground) and the parcels'
  // paths, colored by height (blue low, red high), the newest end brightest
  function drawPaths(g, G, VP, W, H) {
    if (!tor || !paths) return;
    const P = projector(G, VP, W, H), inBox = (x, y) => x >= box[0] * 1000 && x <= box[2] * 1000 &&
      y >= box[1] * 1000 && y <= box[3] * 1000;
    if ($('volPaths').checked) {
      g.lineWidth = 1.2;
      const ztop = head ? head.ztop : 5000;
      for (const a of paths.paths) {
        const n = a.length / 3;
        for (let i = 1; i < n; i++) {
          const x0 = a[3 * i - 3], y0 = a[3 * i - 2], z0 = a[3 * i - 1], x1 = a[3 * i], y1 = a[3 * i + 1], z1 = a[3 * i + 2];
          if (!inBox(x1, y1) || z1 > ztop) continue;
          const s0 = P(x0, y0, z0), s1 = P(x1, y1, z1);
          if (!s0 || !s1) continue;
          const h = Math.min(1, z1 / 3000), al = 0.25 + 0.65 * i / n;
          g.strokeStyle = `rgba(${Math.round(60 + 195 * h)},${Math.round(150 - 60 * h)},${Math.round(255 - 200 * h)},${al.toFixed(2)})`;
          g.beginPath(); g.moveTo(s0[0], s0[1]); g.lineTo(s1[0], s1[1]); g.stroke();
        }
      }
    }
    if ($('volTrack').checked && paths.track && paths.track.length > 1) {
      const pts = paths.track.filter(p => inBox(p[1], p[2])).map(p => P(p[1], p[2], 5)).filter(Boolean);
      if (pts.length > 1) {
        g.lineWidth = 3; g.strokeStyle = '#ff8a1e'; g.beginPath();
        pts.forEach((s, i) => i ? g.lineTo(s[0], s[1]) : g.moveTo(s[0], s[1]));
        g.stroke();
      }
    }
  }

  // a change to the view: drawn on the next animation frame (the cloud view then keeps refining while still)
  function draw() {
    acc = 0; lastMove = performance.now();
    if (!raf) raf = requestAnimationFrame(frame);
  }
  function frame(now) {
    raf = 0;
    if (!gl || !box) return;
    const dt = lastTick ? Math.min(0.1, (now - lastTick) / 1000) : 0;
    lastTick = now;
    if (moveKeys(dt)) { acc = 0; lastMove = now; }
    const cloud = cloudMode() && head;
    const q = cloud && now - lastMove < 160 ? 0.5 : 1;       // half size while the view moves
    if (q !== drawQ) { drawQ = q; acc = 0; }
    if (acc < (cloud ? ACC_MAX : 1)) render();
    if (keys.size || (cloud && (acc < ACC_MAX || drawQ < 1))) raf = requestAnimationFrame(frame);
    else lastTick = 0;
  }

  function render() {
    // a loop export renders at its own size (forced), full quality
    const cv = $('volCanvas'), dpr = forced ? 1 : Math.min(window.devicePixelRatio || 1, 2);
    const W = forced ? forced.W : cv.clientWidth, H = forced ? forced.H : cv.clientHeight;
    if (!W || !H) { acc = ACC_MAX; drawQ = 1; return; }
    const q = cloudMode() && !forced ? drawQ : 1;
    if (cv.width !== Math.round(W * dpr * q) || cv.height !== Math.round(H * dpr * q)) {
      cv.width = Math.round(W * dpr * q); cv.height = Math.round(H * dpr * q); acc = 0;
    }
    gl.viewport(0, 0, cv.width, cv.height);
    gl.disable(gl.BLEND);
    if (acc === 0) { gl.clearColor(0.043, 0.055, 0.075, 1); gl.clear(gl.COLOR_BUFFER_BIT); }
    const ov = $('volOverlay'), g = ov.getContext('2d');
    if (ov.width !== Math.round(W * dpr) || ov.height !== Math.round(H * dpr)) { ov.width = Math.round(W * dpr); ov.height = Math.round(H * dpr); }
    if (acc === 0) { g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H); }
    if (!head || !box || head.field !== want()) { acc = ACC_MAX; return; }
    const { G, eye, VP, V, fov } = matrices(W, H);
    if (cloudMode()) {
      if (acc > 0) {                                  // the running mean of the jittered frames
        gl.enable(gl.BLEND); gl.blendColor(0, 0, 0, 1 / (acc + 1));
        gl.blendFunc(gl.CONSTANT_ALPHA, gl.ONE_MINUS_CONSTANT_ALPHA);
      }
      drawCloud(G, eye, VP, V, fov);
      gl.disable(gl.BLEND);
      if (acc === 0) { drawPaths(g, G, VP, W, H); drawFollowTrack(g, G, VP, W, H); }
      acc++;
      return;
    }
    acc = ACC_MAX;
    const F = FIELDS[field()], thr = +$('volThr').value;
    gl.useProgram(prog); gl.bindVertexArray(vao);
    const u = n => gl.getUniformLocation(prog, n);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, volTex); gl.uniform1i(u('vol'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, lutTex); gl.uniform1i(u('lut'), 1);
    bindGround(u);
    gl.uniformMatrix4fv(u('invVP'), false, invert(VP));
    gl.uniform3f(u('eye'), ...eye);
    gl.uniform3f(u('size'), G.sx, G.sy, G.sz);
    gl.uniform3f(u('texel'), 1 / head.nx, 1 / head.ny, 1 / head.nz);
    bindInsets(u, G);
    // a low field (the pressure drop): only "at or below minus the threshold", through the signed path
    gl.uniform1f(u('thrHi'), F.low ? 999 : thr * head.scale + head.offset);
    gl.uniform1f(u('thrLo'), -thr * head.scale + head.offset);
    gl.uniform1f(u('opacity'), +$('volOpa').value);
    gl.uniform1f(u('gridStep'), (tor ? 1 : 10) * G.k);   // 1 km in the tornado view, else 10 km
    gl.uniform1i(u('mode'), isoMode() ? 0 : 1);
    gl.uniform1i(u('signedField'), F.low || (F.signed && $('volBoth').checked) ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // heights up the nearest-to-camera corner and km along the ground edges
    const proj = (x, y, z) => {
      const p = [x, y, z, 1], o = [0, 0, 0, 0];
      for (let r = 0; r < 4; r++) o[r] = VP[r] * p[0] + VP[4 + r] * p[1] + VP[8 + r] * p[2] + VP[12 + r] * p[3];
      if (o[3] <= 0) return null;
      return [(o[0] / o[3] * 0.5 + 0.5) * W, (1 - (o[1] / o[3] * 0.5 + 0.5)) * H];
    };
    g.font = '11px Segoe UI'; g.fillStyle = 'rgba(200,210,220,.85)'; g.textBaseline = 'middle';
    let best = null;
    for (const cx of [-1, 1]) for (const cy of [-1, 1]) {
      const d = Math.hypot(eye[0] - cx * G.sx / 2, eye[1] - cy * G.sy / 2);
      if (!best || d < best.d) best = { d, cx, cy };
    }
    for (let km = 0; km <= (head.ztop / 1000); km += (tor ? 1 : 3)) {
      const s = proj(best.cx * G.sx / 2, best.cy * G.sy / 2, km * G.k * G.ex);
      if (s) { g.textAlign = 'right'; g.fillText(`${km} km`, s[0] - 6, s[1]); }
    }
    const w = $('volLegend');
    w.textContent = `${F.name} ${isoMode() ? 'isosurface at' : 'from'} ${F.fmt(thr)}` +
      (F.signed && $('volBoth').checked ? ` (and ${F.fmt(-thr)})` : '') + `  ·  vertical ×${G.ex}` + camNote();
    drawPaths(g, G, VP, W, H);
    drawFollowTrack(g, G, VP, W, H);
  }

  // ---------------------------------------------------------------- the window
  function open(b) {
    if (tor) {                                   // a dragged box: the ordinary radar view again
      tor = false; paths = null;
      if (want() === 'pp') $('volField').value = 'dbz';
    }
    show(b);
  }

  // the tornado zoom in 3-D: the cloud from close by at its true height, following the zoom's box
  function openTornado() {
    const tz = S.run && S.run.tornado;
    if (!tz || !tz.box) { alert('The tornado zoom is still being placed: try again in a few seconds.'); return; }
    tor = true; paths = null;
    if ($('volField').value === 'dbz') $('volField').value = 'pp';
    $('volMode').value = 'cloud'; $('volMode').dispatchEvent(new Event('change'));
    $('volEx').value = 1;
    Object.assign(cam, { yaw: -1.2, pitch: 0.12, dist: 2.2 });
    show(tz.box.map(v => v / 1000));
  }
  function closeTornado() { if (tor) close(); }

  function show(b) {
    box = b.map(v => +v.toFixed(tor ? 2 : 1));
    $('volWin').classList.remove('hidden');
    try {
      if (!initGL()) { $('volStatus').textContent = 'This browser has no WebGL2: the 3-D view needs it.'; return; }
    } catch (e) { $('volStatus').textContent = 'WebGL error: ' + e.message; return; }
    cam.px = 0; cam.py = 0;
    if (cam.mode !== 'orbit') { cam.mode = 'orbit'; $('volCam').value = 'orbit'; cam.fp = null; }
    head = null; setSlider(false); draw();
    follow.at = null;
    if (follow.on && !tor) fillFollow();
    if (!tor && replayT === null && !S.live && S.frames.length && S.frameIdx >= 0 && S.frames[S.frameIdx])
      replayT = S.frames[S.frameIdx].t;            // opened on a replayed radar frame: that time
    if (replayT !== null) $('volLive').checked = false;
    refreshTimes().then(() => load(true));
    map.redraw();
    clearInterval(timer);
    timer = setInterval(() => {
      if (document.hidden) return;
      refreshTimes();
      if ($('volLive').checked && S.run && S.live) load(false);
    }, 8000);
  }
  function close() {
    $('volWin').classList.add('hidden');
    clearInterval(timer); timer = null; box = null; tor = false; paths = null;
    Land.reset();
    stopPlay(); keys.clear(); replayT = null; followMap = false; $('volLive').checked = true;
    if (exp) exp.cancel = true;
    $('volExp').classList.add('hidden');
    cache.clear(); cacheBytes = 0;
    map.redraw();
  }

  // ---------------------------------------------------------------- the replay
  const clock = t => {
    if (t === null || t === undefined) return '';
    const z = ms0 !== null ? `${new Date(ms0 + 1000 * t).toISOString().slice(11, 16)}Z ` : '';
    return `${z}(+${fmtDur(t)})`;
  };
  async function refreshTimes() {
    try {
      const d = await (await api('/api/volume/times')).json();
      times = d.times || []; ms0 = d.ms0;
    } catch (e) { times = []; }
    const s = $('volT');
    s.max = Math.max(0, times.length - 1);
    if (replayT === null) s.value = s.max; else s.value = nearest(replayT);
    showTime();
  }
  const nearest = t => {
    let b = 0;
    for (let i = 1; i < times.length; i++) if (Math.abs(times[i] - t) < Math.abs(times[b] - t)) b = i;
    return b;
  };
  function showTime() {
    $('volTimeRow').classList.toggle('hidden', tor);
    $('volTV').textContent = !times.length ? 'no saved volumes yet' : replayT === null ? `newest · ${times.length} saved`
      : `${clock(times[nearest(replayT)])} · ${nearest(replayT) + 1}/${times.length}`;
  }
  function goTime(i, fromMap) {
    if (!times.length) return;
    i = Math.max(0, Math.min(times.length - 1, i));
    const t = times[i];
    followMap = !!fromMap;
    $('volLive').checked = false;
    $('volT').value = i;
    if (t === replayT) { showTime(); return; }
    replayT = t; showTime();
    return load(true);
  }
  function goLive() {
    stopPlay(); replayT = null; followMap = false; $('volLive').checked = true;
    $('volT').value = $('volT').max; showTime(); load(true);
  }
  function stopPlay() { playTok++; $('volPlay').textContent = '▶'; }
  async function play() {
    if ($('volPlay').textContent !== '▶') { stopPlay(); return; }
    await refreshTimes();
    if (times.length < 2) return;
    const tok = ++playTok;
    $('volPlay').textContent = '❚❚';
    let i = replayT === null ? 0 : nearest(replayT);
    while (tok === playTok && box) {
      i = i >= times.length - 1 ? 0 : i + 1;
      const t0 = performance.now();
      await goTime(i, false);
      while (busy && tok === playTok) await new Promise(r => setTimeout(r, 30));
      const wait = (i === times.length - 1 ? 1200 : 350) - (performance.now() - t0);
      if (wait > 0) await new Promise(r => setTimeout(r, wait));
    }
  }
  // the map's replay bar: the 3-D view follows it to the saved volume nearest its frame (t: null = live again)
  function followTime(t) {
    if (!box || tor || exp || $('volPlay').textContent !== '▶') return;
    if (t === null) { if (followMap) goLive(); return; }
    if (!times.length) { refreshTimes().then(() => { if (times.length) goTime(nearest(t), true); }); return; }
    if (t > times[times.length - 1] + 30) refreshTimes().then(() => goTime(nearest(t), true));
    else goTime(nearest(t), true);
  }

  // ---------------------------------------------------------------- following the tornado
  // Follow tornado: before each volume loads, the box (its size kept) is centred on the tornado picked in
  // volFollowId at that volume's time: on its track (Simulation.tornado_tracks) while it is on the ground; before
  // and after, on the strongest low-level rotation (the snapshot's vertical vorticity, lowest 1.5 km) near where the
  // track's mean motion puts it.  "rot" needs no track: the strongest low-level rotation near the last centre.
  // The cameras are placed relative to the box's middle, so they ride along.
  const follow = { on: false, id: 'auto', at: null, label: '', tr: null };
  const rotCache = new Map();
  const runTracks = () => (S.run && S.run.tornado_tracks) || [];
  function trackFor(id) {
    if (id === 'rot') return null;
    const ts = runTracks();
    if (id !== 'auto') return ts.find(t => String(t.id) === String(id)) || null;
    let best = null;                               // the run's strongest: highest EF, then peak wind
    for (const t of ts) if (!best || t.ef > best.ef || (t.ef === best.ef && t.wind > best.wind)) best = t;
    return best;
  }
  function trackAt(tr, t) {                        // km; null when it is not on the ground at t
    const p = tr.pts;
    if (t < p[0][0] - 1 || t > p[p.length - 1][0] + 1) return null;
    for (let i = 1; i < p.length; i++) if (p[i][0] >= t) {
      const a = p[i - 1], b = p[i], f = b[0] > a[0] ? Math.max(0, (t - a[0]) / (b[0] - a[0])) : 0;
      return { x: (a[1] + f * (b[1] - a[1])) / 1000, y: (a[2] + f * (b[2] - a[2])) / 1000, ef: Math.max(f < 0.5 ? a[3] : b[3], 0) };
    }
    const q = p[p.length - 1];
    return { x: q[1] / 1000, y: q[2] / 1000, ef: Math.max(q[3], 0) };
  }
  function trackGuess(tr, t) {                     // off its ends: carried along the track's mean motion
    const p = tr.pts, a = p[0], b = p[p.length - 1], dt = Math.max(1, b[0] - a[0]);
    const u = (b[1] - a[1]) / dt, v = (b[2] - a[2]) / dt, e = t < a[0] ? a : b;
    return { x: (e[1] + u * (t - e[0])) / 1000, y: (e[2] + v * (t - e[0])) / 1000 };
  }
  // the strongest cyclonic low-level column within r km of (cx, cy) km at model time t (null: the newest step)
  async function rotationNear(cx, cy, r, t) {
    const key = `${t}/${cx.toFixed(1)}/${cy.toFixed(1)}/${r}`;
    if (rotCache.has(key)) return rotCache.get(key);
    let q = `x0=${(cx - r).toFixed(2)}&y0=${(cy - r).toFixed(2)}&x1=${(cx + r).toFixed(2)}&y1=${(cy + r).toFixed(2)}&field=zeta`;
    if (t !== null) q += `&t=${t}`;
    let out = null;
    try {
      const { header: h, data } = unpack(await (await api('/api/volume?' + q)).arrayBuffer());
      const nx = h.nx, ny = h.ny, n = nx * ny, ch = h.channels || 1;
      const kz = Math.min(h.nz, Math.max(1, Math.round(1500 / (h.ztop / h.nz))));
      const col = new Float32Array(n);
      for (let k = 0; k < kz; k++) for (let i = 0; i < n; i++) {
        const c = data[(k * n + i) * ch];
        if (c) col[i] += (c - h.offset) / h.scale / kz;
      }
      const bx = h.box.map(v => v / 1000);
      let bv = 0.003, bi = -1, bj = -1;          // 0.003 /s: weaker is no rotation worth following
      for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
        let s = 0;
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) s += col[(j + dj) * nx + i + di];
        const x = bx[0] + (i + 0.5) * (bx[2] - bx[0]) / nx, y = bx[1] + (j + 0.5) * (bx[3] - bx[1]) / ny;
        const d2 = ((x - cx) ** 2 + (y - cy) ** 2) / (r * r);
        if (d2 > 1) continue;
        s = s / 9 * (1 - 0.25 * d2);               // nearer the guess wins between equals
        if (s > bv) { bv = s; bi = i; bj = j; }
      }
      if (bi >= 0) out = { x: bx[0] + (bi + 0.5) * (bx[2] - bx[0]) / nx, y: bx[1] + (bj + 0.5) * (bx[3] - bx[1]) / ny, zeta: bv };
    } catch (e) { out = null; }
    rotCache.set(key, out);
    return out;
  }
  async function recentre(t) {
    const hw = (box[2] - box[0]) / 2, hh = (box[3] - box[1]) / 2;
    const tt = t !== null ? t : (S.run && S.run.t) || 0;
    const tr = trackFor(follow.id);
    let c = null, label = '';
    follow.tr = tr;
    if (tr) {
      const p = trackAt(tr, tt);
      if (p) { c = p; label = `T${tr.id} EF${p.ef} (est.)`; }
      else {
        const g = trackGuess(tr, tt), r = await rotationNear(g.x, g.y, 10, t);
        c = r || g;
        label = `${tt < tr.pts[0][0] ? 'before' : 'after'} T${tr.id}` + (r ? ' · low-level rotation' : '');
      }
    } else {
      const s = follow.at || { x: (box[0] + box[2]) / 2, y: (box[1] + box[3]) / 2 };
      const r = await rotationNear(s.x, s.y, Math.min(25, Math.max(8, Math.min(hw, hh))), t);
      c = r || s;
      label = r ? `strongest low-level rotation ${(r.zeta * 1000).toFixed(0)}×10⁻³ /s` : 'no low-level rotation';
    }
    follow.at = { x: c.x, y: c.y }; follow.label = label;
    if (cam.mode === 'ground' && cam.fp) {         // riding along: onto a road / open ground again, from where it meant to be
      if (cam.snap) { cam.fp.x -= cam.snap[0]; cam.fp.y -= cam.snap[1]; cam.snap = null; }
      cam.placed = false;
    }
    const r2 = v => Math.round(v * 20) / 20;
    box = [r2(c.x - hw), r2(c.y - hh), r2(c.x + hw), r2(c.y + hh)];
  }
  function fillFollow() {
    const sel = $('volFollowId'), keep = sel.value || follow.id;
    const ts = runTracks().slice().sort((a, b) => b.ef - a.ef || b.wind - a.wind);
    sel.innerHTML = `<option value="auto">Strongest tornado</option>` +
      ts.map(t => `<option value="${t.id}">T${t.id} · EF${t.ef} (est.) · +${fmtDur(t.t0).slice(0, -3)}–${fmtDur(t.t1).slice(0, -3)}</option>`).join('') +
      `<option value="rot">Strongest rotation (no track)</option>`;
    sel.value = [...sel.options].some(o => o.value === keep) ? keep : 'auto';
    follow.id = sel.value;
  }
  function setFollow(on) {
    follow.on = on; follow.at = null; rotCache.clear();
    $('volFollow').checked = on;
    $('volFollowId').classList.toggle('hidden', !on);
    $('volFTrackLbl').classList.toggle('hidden', !on);
    if (on) { fillFollow(); cam.px = 0; cam.py = 0; }
    if (box) load(true);
  }
  // the followed tornado's track on the ground, in its EF colors, up to the volume's time
  function drawFollowTrack(g, G, VP, W, H) {
    if (!follow.on || tor || !follow.tr || !$('volFTrack').checked || !box) return;
    const tNow = replayT !== null ? replayT : Infinity, P = projector(G, VP, W, H);
    const pts = follow.tr.pts.filter(p => p[0] <= tNow + 1);
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    const s = pts.map(p => [P(p[1], p[2], 1000 * groundKm(p[1] / 1000 - cx, p[2] / 1000 - cy) + 20), Math.max(p[3], 0)]);
    const cols = typeof EF_COLORS !== 'undefined' ? EF_COLORS : ['#7fdcff', '#3ccf7a', '#f5d33a', '#f2913a', '#e8413c', '#c13ae8'];
    g.save(); g.lineCap = 'round';
    for (let i = 1; i < s.length; i++) {
      if (!s[i - 1][0] || !s[i][0]) continue;
      g.strokeStyle = cols[Math.min(5, s[i][1])]; g.lineWidth = 2 + 0.8 * s[i][1];
      g.beginPath(); g.moveTo(...s[i - 1][0]); g.lineTo(...s[i][0]); g.stroke();
    }
    g.restore();
  }

  // ---------------------------------------------------------------- the loop export
  // Steps through a range of the saved volumes as the view is set up (camera, mode, sliders, Follow tornado),
  // renders each fully (the cloud view's frames all averaged) at the chosen size, then encodes: WebM / MP4 through
  // MediaRecorder (the frames played back at the frame time), a GIF (one palette for the whole loop, median cut),
  // or the PNG frames in a .zip.
  let forced = null, exp = null, expUrl = null;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const expMsg = s => { $('volExMsg').textContent = s; };
  function fillExport() {
    const opts = times.map((t, i) => `<option value="${i}">${clock(t)}</option>`).join('');
    $('volExFrom').innerHTML = opts; $('volExTo').innerHTML = opts;
    let a = 0, b = Math.max(0, times.length - 1);
    const tr = follow.on ? trackFor(follow.id) : null;
    if (tr && times.length) { a = nearest(tr.t0 - 600); b = nearest(tr.t1 + 300); }
    $('volExFrom').value = a; $('volExTo').value = b;
    const mp4 = ['video/mp4;codecs=avc1', 'video/mp4'].some(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    $('volExFmt').querySelector('option[value="mp4"]').disabled = !mp4;
    if (!mp4 && $('volExFmt').value === 'mp4') $('volExFmt').value = 'webm';
    expMsg(times.length ? `${times.length} saved volumes` : 'no saved volumes yet: they are saved with the radar frames');
  }
  function caption(g, W, H, t) {
    const s = Math.max(12, Math.round(H / 40)), pad = Math.round(s * 0.8);
    const z = ms0 !== null ? new Date(ms0 + 1000 * t).toISOString().slice(11, 19) + 'Z' : '';
    const line = [z, `+${fmtDur(t)}`, follow.on ? follow.label : ''].filter(Boolean).join('   ·   ');
    g.save(); g.textBaseline = 'alphabetic'; g.shadowColor = 'rgba(0,0,0,.85)'; g.shadowBlur = s / 3;
    g.font = `600 ${s}px Inter, Segoe UI, sans-serif`; g.fillStyle = '#f2f6fa'; g.textAlign = 'left';
    g.fillText(line, pad, H - pad);
    g.font = `500 ${Math.round(s * 0.75)}px Inter, Segoe UI, sans-serif`; g.fillStyle = 'rgba(230,238,245,.8)'; g.textAlign = 'right';
    g.fillText(`Atmos-WRF${S.run && S.run.run_id ? '  ·  ' + S.run.run_id : ''}`, W - pad, pad + s * 0.75);
    g.restore();
  }
  const toBlob = (c, type, q) => new Promise((res, rej) => c.toBlob(b => (b ? res(b) : rej(new Error('could not encode a frame'))), type, q));

  async function exportLoop() {
    if (exp) { exp.cancel = true; return; }
    await refreshTimes();
    if (!times.length || !box) { expMsg('No saved volumes yet: they are saved with the radar frames'); return; }
    let a = +$('volExFrom').value, b = +$('volExTo').value;
    if (a > b) [a, b] = [b, a];
    const fmt = $('volExFmt').value, ms = +$('volExMs').value, stamp = $('volExStamp').checked, sz = $('volExSize').value;
    const cv = $('volCanvas');
    const [W, H] = sz === 'win' ? [cv.clientWidth & ~1, cv.clientHeight & ~1] : sz.split('x').map(Number);
    stopPlay();
    const wasLive = $('volLive').checked, wasT = replayT;
    exp = { cancel: false };
    $('volExGo').textContent = 'Cancel'; $('volExDl').classList.add('hidden');
    const out = document.createElement('canvas'); out.width = W; out.height = H;
    const g = out.getContext('2d');
    const shots = [];
    forced = { W, H };
    try {
      for (let i = a; i <= b && !exp.cancel; i++) {
        expMsg(`Rendering ${i - a + 1} / ${b - a + 1}…`);
        await goTime(i, false);
        while (busy) await sleep(30);
        if (!head || head.field !== want()) continue;           // nothing in the box then
        if (cloudMode()) {
          // the ground's textures for this box and camera (requested by the render), up to 20 s
          drawQ = 1; acc = 0; render();
          const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
          const landWant = () => [baseName() === 'plain' ? box : null,
                              cam.mode !== 'orbit' && cam.fp ? { x: cam.fp.x + cx, y: cam.fp.y + cy } : null];
          for (let w = 0; w < 200 && !exp.cancel && !Land.ready(...landWant()); w++) { await sleep(100); render(); }
          await sleep(50);
        }
        drawQ = 1; acc = 0;
        const n = cloudMode() ? ACC_MAX : 1;
        for (let k = 0; k < n; k++) render();
        g.fillStyle = '#0b0e13'; g.fillRect(0, 0, W, H);
        g.drawImage(cv, 0, 0, W, H); g.drawImage($('volOverlay'), 0, 0, W, H);
        if (stamp) caption(g, W, H, times[i]);
        shots.push(await toBlob(out, fmt === 'zip' ? 'image/png' : 'image/webp', 0.95));
      }
      forced = null; draw();
      if (exp.cancel) { expMsg('Cancelled'); return; }
      if (!shots.length) { expMsg('No volume in the box over that range'); return; }
      const tag = follow.on && follow.tr ? `_T${follow.tr.id}` : follow.on ? '_rotation' : '';
      const name = `atmos-wrf_${(S.run && S.run.run_id) || 'run'}${tag}_3d_${times[a]}-${times[b]}s`;
      let blob, ext = fmt;
      if (fmt === 'zip') {
        expMsg('Packing the frames…');
        const files = [];
        for (let i = 0; i < shots.length; i++)
          files.push({ name: `${name}_${String(i + 1).padStart(3, '0')}.png`, data: new Uint8Array(await shots[i].arrayBuffer()) });
        blob = makeZip(files);
      } else if (fmt === 'gif') blob = await makeGif(shots, W, H, ms);
      else ({ blob, ext } = await recordVideo(shots, W, H, ms, fmt));
      if (exp.cancel) { expMsg('Cancelled'); return; }
      saveBlob(blob, `${name}.${ext}`);
      expMsg(`Saved ${name}.${ext} · ${shots.length} frames · ${(blob.size / 1e6).toFixed(1)} MB`);
    } catch (e) {
      console.error(e);
      expMsg('Export failed: ' + ((e && e.message) || e));
    } finally {
      forced = null; exp = null;
      $('volExGo').textContent = 'Make the loop';
      if (wasLive && wasT === null) goLive();
      else if (wasT !== null) goTime(nearest(wasT), false);
      draw();
    }
  }
  function saveBlob(blob, name) {
    if (expUrl) URL.revokeObjectURL(expUrl);
    expUrl = URL.createObjectURL(blob);
    Vol.lastExport = { blob, name };
    const a = $('volExDl');
    a.href = expUrl; a.download = name; a.classList.remove('hidden');
    a.click();
  }

  async function recordVideo(shots, W, H, ms, fmt) {
    if (!window.MediaRecorder) throw new Error('this browser cannot record video: pick GIF or PNG frames');
    const types = fmt === 'mp4' ? ['video/mp4;codecs=avc1.640033', 'video/mp4;codecs=avc1', 'video/mp4']
      : ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mime = types.find(t => MediaRecorder.isTypeSupported(t));
    if (!mime) throw new Error(`this browser cannot record ${fmt.toUpperCase()}: pick another format`);
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d');
    const stream = c.captureStream(0), track = stream.getVideoTracks()[0];
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: Math.round(Math.min(25e6, Math.max(4e6, W * H * 8))) });
    const parts = [];
    rec.ondataavailable = e => { if (e.data && e.data.size) parts.push(e.data); };
    const stopped = new Promise(r => { rec.onstop = r; });
    let next = await createImageBitmap(shots[0]);
    g.drawImage(next, 0, 0, W, H);
    rec.start(1000);
    const t0 = performance.now();
    let at = 0;
    for (let i = 0; i < shots.length && !exp.cancel; i++) {
      expMsg(`Encoding ${fmt.toUpperCase()} ${i + 1} / ${shots.length}… (keep this window in front)`);
      g.drawImage(next, 0, 0, W, H); next.close();
      track.requestFrame();
      at += i === shots.length - 1 ? Math.max(1000, 4 * ms) : ms;   // a pause on the last frame before it loops
      next = i + 1 < shots.length ? await createImageBitmap(shots[i + 1]) : null;
      const wait = t0 + at - performance.now();
      if (wait > 0) await sleep(wait);
    }
    if (next) next.close();
    g.drawImage(c, 0, 0);                          // the canvas changed again, so this frame counts: it closes the
    track.requestFrame();                          // last one's time (the pause before the loop)
    await sleep(100);
    rec.stop(); await stopped;
    track.stop();
    const type = mime.split(';')[0];
    return { blob: new Blob(parts, { type }), ext: type === 'video/mp4' ? 'mp4' : 'webm' };
  }

  // GIF89a, looping, one 256-colour palette for every frame (median cut over pixels sampled from all of them),
  // a light ordered dither against banding in the sky, LZW as the spec has it
  async function makeGif(shots, W, H, ms) {
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d', { willReadFrequently: true });
    const pixels = async s => { const b = await createImageBitmap(s); g.drawImage(b, 0, 0, W, H); b.close(); return g.getImageData(0, 0, W, H).data; };
    const hist = new Uint32Array(32768);
    for (let f = 0; f < shots.length; f++) {
      expMsg(`GIF palette ${f + 1} / ${shots.length}…`);
      const d = await pixels(shots[f]);
      for (let i = (f * 4) % 28; i < d.length; i += 28) hist[((d[i] >> 3) << 10) | ((d[i + 1] >> 3) << 5) | (d[i + 2] >> 3)]++;
    }
    const pal = medianCut(hist, 256), np = pal.length / 3;
    const lut = new Int16Array(32768).fill(-1);
    const nearestPal = k => {
      const r = ((k >> 10) << 3) + 4, gg = (((k >> 5) & 31) << 3) + 4, b = ((k & 31) << 3) + 4;
      let bi = 0, bd = Infinity;
      for (let p = 0; p < np; p++) {
        const dr = r - pal[3 * p], dg = gg - pal[3 * p + 1], db = b - pal[3 * p + 2], dd = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
        if (dd < bd) { bd = dd; bi = p; }
      }
      return bi;
    };
    const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => Math.round((v - 7.5) * 1.2));
    const parts = [];
    const u16 = v => [v & 255, v >> 8];
    const ptab = new Uint8Array(768); ptab.set(pal);
    parts.push(new Uint8Array([71, 73, 70, 56, 57, 97, ...u16(W), ...u16(H), 0xF7, 0, 0]), ptab,
      new Uint8Array([0x21, 0xFF, 11, ...[...'NETSCAPE2.0'].map(ch => ch.charCodeAt(0)), 3, 1, 0, 0, 0]));
    const idx = new Uint8Array(W * H);
    for (let f = 0; f < shots.length; f++) {
      if (exp && exp.cancel) break;
      expMsg(`GIF frame ${f + 1} / ${shots.length}…`);
      const d = await pixels(shots[f]);
      for (let y = 0, p = 0; y < H; y++) for (let x = 0; x < W; x++, p++) {
        const o = BAYER[((y & 3) << 2) | (x & 3)], i = 4 * p;
        const r = Math.min(255, Math.max(0, d[i] + o)), gg = Math.min(255, Math.max(0, d[i + 1] + o)), b = Math.min(255, Math.max(0, d[i + 2] + o));
        const k = ((r >> 3) << 10) | ((gg >> 3) << 5) | (b >> 3);
        let v = lut[k];
        if (v < 0) v = lut[k] = nearestPal(k);
        idx[p] = v;
      }
      const cs = Math.max(2, Math.round((f === shots.length - 1 ? Math.max(1000, 4 * ms) : ms) / 10));
      parts.push(new Uint8Array([0x21, 0xF9, 4, 0, ...u16(cs), 0, 0, 0x2C, 0, 0, 0, 0, ...u16(W), ...u16(H), 0, 8]), lzw(idx), new Uint8Array([0]));
      await sleep(0);
    }
    parts.push(new Uint8Array([0x3B]));
    return new Blob(parts, { type: 'image/gif' });
  }
  function medianCut(hist, n) {
    const all = [];
    for (let k = 0; k < 32768; k++) if (hist[k]) all.push(k);
    const comp = (k, a) => (a === 0 ? k >> 10 : a === 1 ? (k >> 5) & 31 : k & 31);
    const stat = bins => {
      const lo = [31, 31, 31], hi = [0, 0, 0];
      let cnt = 0;
      for (const k of bins) { cnt += hist[k]; for (let a = 0; a < 3; a++) { const v = comp(k, a); if (v < lo[a]) lo[a] = v; if (v > hi[a]) hi[a] = v; } }
      const rg = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]], ax = rg.indexOf(Math.max(...rg));
      return { bins, cnt, ax, span: rg[ax] };
    };
    const boxes = [stat(all)];
    while (boxes.length < n) {
      let bi = -1, bs = 0;
      boxes.forEach((bx, i) => { const s = bx.span * Math.sqrt(bx.cnt); if (bx.bins.length > 1 && bx.span > 0 && s > bs) { bs = s; bi = i; } });
      if (bi < 0) break;
      const bx = boxes[bi];
      bx.bins.sort((p, q) => comp(p, bx.ax) - comp(q, bx.ax));
      let acc2 = 0, cut = 1;
      for (let i = 0; i < bx.bins.length - 1; i++) { acc2 += hist[bx.bins[i]]; if (acc2 >= bx.cnt / 2) { cut = i + 1; break; } }
      boxes.splice(bi, 1, stat(bx.bins.slice(0, cut)), stat(bx.bins.slice(cut)));
    }
    const pal = [];
    for (const bx of boxes) {
      let r = 0, gg = 0, b = 0;
      for (const k of bx.bins) { const w = hist[k]; r += w * ((k >> 10) * 8 + 4); gg += w * (((k >> 5) & 31) * 8 + 4); b += w * ((k & 31) * 8 + 4); }
      pal.push(Math.round(r / bx.cnt), Math.round(gg / bx.cnt), Math.round(b / bx.cnt));
    }
    while (pal.length < 3 * n) pal.push(0, 0, 0);
    return pal;
  }
  function lzw(idx) {
    const MIN = 8, CLEAR = 256, EOI = 257;
    let size = MIN + 1, next = EOI + 1, cur = 0, nb = 0;
    const table = new Map();
    let buf = new Uint8Array(65536), len = 0;
    const put = byte => { if (len === buf.length) { const nbuf = new Uint8Array(buf.length * 2); nbuf.set(buf); buf = nbuf; } buf[len++] = byte; };
    const emit = code => { cur |= code << nb; nb += size; while (nb >= 8) { put(cur & 255); cur >>>= 8; nb -= 8; } };
    emit(CLEAR);
    let ib = idx[0];
    for (let i = 1; i < idx.length; i++) {
      const k = idx[i], key = (ib << 8) | k, hit = table.get(key);
      if (hit !== undefined) { ib = hit; continue; }
      emit(ib);
      if (next === 4096) { emit(CLEAR); next = EOI + 1; size = MIN + 1; table.clear(); }
      else { if (next >= (1 << size)) size++; table.set(key, next++); }
      ib = k;
    }
    emit(ib); emit(EOI);
    if (nb > 0) put(cur & 255);
    const out = new Uint8Array(len + Math.ceil(len / 255));     // 255-byte sub-blocks, each with its length first
    let o = 0;
    for (let i = 0; i < len; i += 255) { const m = Math.min(255, len - i); out[o++] = m; out.set(buf.subarray(i, i + m), o); o += m; }
    return out.subarray(0, o);
  }
  // a .zip of stored (uncompressed: PNGs are compressed already) files
  let crcTab = null;
  function crc32(d) {
    if (!crcTab) { crcTab = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; crcTab[n] = c >>> 0; } }
    let c = 0xFFFFFFFF;
    for (let i = 0; i < d.length; i++) c = crcTab[(c ^ d[i]) & 255] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }
  function makeZip(files) {
    const parts = [], cen = [];
    let off = 0;
    for (const f of files) {
      const nm = new TextEncoder().encode(f.name), crc = crc32(f.data), n = f.data.length;
      const h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(12, 0x21, true);
      h.setUint32(14, crc, true); h.setUint32(18, n, true); h.setUint32(22, n, true); h.setUint16(26, nm.length, true);
      parts.push(h.buffer, nm, f.data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(14, 0x21, true);
      c.setUint32(16, crc, true); c.setUint32(20, n, true); c.setUint32(24, n, true); c.setUint16(28, nm.length, true);
      c.setUint32(42, off, true);
      cen.push(c.buffer, nm);
      off += 30 + nm.length + n;
    }
    const csize = cen.reduce((s, p) => s + p.byteLength, 0);
    const e = new DataView(new ArrayBuffer(22));
    e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
    e.setUint32(12, csize, true); e.setUint32(16, off, true);
    return new Blob([...parts, ...cen, e.buffer], { type: 'application/zip' });
  }

  // the box on the map while the window is open
  function drawBox(g, m) {
    if (!box) return;
    const [a, b] = m.toScreen(box[0], box[1]), [c, d] = m.toScreen(box[2], box[3]);
    g.save(); g.strokeStyle = '#7cc8ff'; g.lineWidth = 1.5; g.setLineDash([6, 4]);
    g.strokeRect(Math.min(a, c), Math.min(b, d), Math.abs(c - a), Math.abs(d - b));
    g.restore();
  }

  function init() {
    $('volX').onclick = close;
    $('volField').onchange = () => { setSlider(false); load(true); };
    let lastMode = $('volMode').value;
    $('volMode').onchange = () => {
      const c = cloudMode(), was = lastMode === 'cloud';
      lastMode = $('volMode').value;
      if (c !== was) {
        // the eye's view: true heights from low down, like a photo from the ground; back to stretched for radar
        $('volEx').value = c ? 1 : 3;
        $('volOpa').value = 0.5;
        Object.assign(cam, c ? { pitch: 0.07, dist: 1.25 } : { pitch: 0.38, dist: 3.6 });
      }
      setSlider(false);
      if (c !== was) load(true); else draw();
    };
    $('volThr').oninput = () => { setSlider(true); draw(); };
    for (const id of ['volOpa', 'volEx', 'volBoth', 'volSun', 'volSunH', 'volRelief', 'volTrack', 'volPaths', 'volDetail', 'volRain', 'volTrees']) $(id).oninput = draw;
    $('volCam').onchange = () => setCamMode($('volCam').value);
    $('volT').oninput = () => { stopPlay(); goTime(+$('volT').value, false); };
    $('volPlay').onclick = play;
    $('volFollow').onchange = () => setFollow($('volFollow').checked);
    $('volFollowId').onchange = () => { follow.id = $('volFollowId').value; follow.at = null; load(true); };
    $('volFTrack').onchange = draw;
    $('volExport').onclick = async () => {
      const p = $('volExp'), opening = p.classList.contains('hidden');
      p.classList.toggle('hidden', !opening);
      if (opening) { if (follow.on) fillFollow(); await refreshTimes(); fillExport(); }
    };
    $('volExGo').onclick = exportLoop;
    $('volLive').onchange = () => { if ($('volLive').checked) goLive(); };
    $('volTop').oninput = () => { $('volTopV').textContent = `${$('volTop').value} km`; };
    $('volTop').onchange = () => load(true);
    $('volRefresh').onclick = () => load(true);
    $('volReset').onclick = () => {
      const first = tor ? 'pp' : 'dbz', reload = want() !== first;
      $('volField').value = first; $('volMode').value = 'iso'; lastMode = 'iso';
      $('volRelief').value = 0.6;
      $('volOpa').value = 0.5; $('volEx').value = 3; $('volSun').value = 235; $('volSunH').value = 30;
      $('volBoth').checked = false; $('volDetail').value = 0.7; $('volRain').value = 1;
      Object.assign(cam, CAM0); cam.fp = null; $('volCam').value = 'orbit';
      if (replayT !== null) { stopPlay(); replayT = null; followMap = false; $('volLive').checked = true; refreshTimes(); return load(true); }
      $('volLive').checked = true;
      setSlider(false);
      if (reload) load(true); else draw();
    };
    const cv = $('volOverlay');
    let drag = null;
    cv.addEventListener('contextmenu', e => e.preventDefault());
    cv.tabIndex = 0;
    cv.addEventListener('mousedown', e => {
      drag = { x: e.clientX, y: e.clientY, b: e.button, yaw: cam.yaw, pitch: cam.pitch, px: cam.px, py: cam.py,
               fy: cam.fp ? cam.fp.yaw : 0, fp: cam.fp ? cam.fp.pitch : 0 };
      cv.focus(); e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (cam.mode !== 'orbit' && cam.fp) {            // free / ground: look around
        cam.fp.yaw = drag.fy - dx * 0.004;
        cam.fp.pitch = Math.min(1.45, Math.max(-1.45, drag.fp - dy * 0.004));
      } else if (drag.b === 2 || e.shiftKey) {                    // move the view across the ground
        const s = 0.0025 * cam.dist, c = Math.cos(cam.yaw), sn = Math.sin(cam.yaw);
        cam.px = drag.px + (dx * sn - dy * c) * s;
        cam.py = drag.py + (-dx * c - dy * sn) * s;
      } else {
        cam.yaw = drag.yaw - dx * 0.008;
        cam.pitch = Math.min(1.5, Math.max(0.03, drag.pitch + dy * 0.006));
      }
      draw();
    });
    window.addEventListener('mouseup', () => { drag = null; });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      if (cam.mode !== 'orbit') cam.speed = Math.min(50, Math.max(0.01, cam.speed * Math.exp(-e.deltaY * 0.0015)));
      else cam.dist = Math.min(12, Math.max(0.4, cam.dist * Math.exp(e.deltaY * 0.0012)));
      draw();
    }, { passive: false });
    // the free and ground cameras' keys (W A S D, Q / E, Shift) while the 3-D window is open and no field has focus
    const MOVE = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight']);
    window.addEventListener('keydown', e => {
      if (!box || cam.mode === 'orbit' || !MOVE.has(e.code) || e.ctrlKey || e.altKey || e.metaKey) return;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) && e.target.type !== 'range') return;
      e.preventDefault();
      if (!keys.has(e.code)) { keys.add(e.code); if (!raf) raf = requestAnimationFrame(frame); }
    });
    window.addEventListener('keyup', e => keys.delete(e.code));
    window.addEventListener('blur', () => keys.clear());
    new ResizeObserver(() => { if (box) draw(); }).observe($('volCanvas'));
  }

  const state = () => ({ box, replayT, follow: { on: follow.on, id: follow.id, label: follow.label, track: follow.tr && follow.tr.id },
    cam: { mode: cam.mode, dist: cam.dist, speed: cam.speed, fp: cam.fp && { ...cam.fp } } });
  // for other pages that draw the same cloud (web/js/chase): the shaders and the detail noise, made in their own context
  const lib = { VS, FS_CLOUD, FS_LIGHT, makeNoise, perspective, lookAt, invert, mul };
  return { init, open, close, drawBox, openTornado, closeTornado, followTime, state, lib };
})();
window.Vol = Vol;
