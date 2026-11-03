'use strict';
// The 3-D view (arhtmodelviewer/volume.py), like GR2Analyst's volume explorer: drag a box on the map, and the run's
// latest reflectivity (or vertical velocity, or vertical vorticity) in it is ray-marched on the GPU (WebGL2) as an
// isosurface or a translucent volume.  Drag to turn it, wheel to zoom, right-drag to move it; Live keeps it on the
// newest model step.
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
  let head = null, box = null, busy = false, again = false, timer = null, lastKey = '';
  let tor = false, paths = null;               // the tornado view (tornado.py): its box, followed; its parcels' paths
  const CAM0 = { yaw: -0.75, pitch: 0.38, dist: 3.6, px: 0, py: 0 };
  const cam = { ...CAM0 };

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
  float val(vec3 p) { return texture(vol, vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z)).r * 255.0; }
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

  // Cloud mode: what the storm looks like to the eye.  The cloud (cloud water, ice, snow) and the precipitation
  // (rain, graupel, hail) in g/kg (two channels, log-coded) absorb and scatter sunlight: extinction per km
  // ~25 x the cloud's g/kg and ~1.2 x the precipitation's, each sample lit by the sun through a shadow ray (the dark
  // base, the lit anvil top, the storm's shadow on the ground), a forward-scattering phase (bright silver edges
  // toward the sun) and the sky's light from above.
  const FS_CLOUD = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 ndc; out vec4 frag;
  uniform sampler3D vol;
  uniform mat4 invVP; uniform vec3 eye; uniform vec3 size; uniform vec3 toKm; uniform vec3 sunDir; uniform float density;
  uniform sampler2D gnd; uniform int hasG; uniform float relief; uniform vec3 texel;
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
  vec3 SUN;                                                   // the sunlight: warmer and dimmer as the sun gets low
  vec2 q(vec3 p) {
    vec3 uvw = vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z);
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return vec2(0.0);
    vec2 c = texture(vol, uvw).rg * 255.0;
    return vec2(c.x < 0.5 ? 0.0 : pow(10.0, (c.x - 1.0) / 40.0 - 3.0), c.y < 0.5 ? 0.0 : pow(10.0, (c.y - 1.0) / 40.0 - 3.0));
  }
  float sigma(vec2 m) { return (25.0 * m.x + 1.2 * m.y) * density; }          // per km
  float hg(float c, float g) { return (1.0 - g * g) / (12.566 * pow(1.0 + g * g - 2.0 * g * c, 1.5)); }
  float sunTau(vec3 p, vec3 wd, float kmPer) {
    float tau = 0.0, t = 0.0, ds = 0.012 * length(size);
    for (int j = 0; j < 7; j++) {
      t += ds;
      tau += sigma(q(p + wd * t)) * ds * kmPer;
      ds *= 1.55;
    }
    return tau;
  }
  vec3 sky(vec3 rd) {
    float h = clamp(rd.z, 0.0, 1.0);
    vec3 c = mix(vec3(0.70, 0.79, 0.90), vec3(0.16, 0.34, 0.72), pow(h, 0.45));
    float s = max(dot(rd, sunDir), 0.0);
    return c + SUN * (pow(s, 900.0) * 3.0 + pow(s, 12.0) * 0.06);
  }
  void main() {
    float low = smoothstep(0.02, 0.45, sunDir.z);
    SUN = mix(vec3(1.0, 0.55, 0.28) * 2.2, vec3(1.0, 0.95, 0.86) * 3.2, low);
    vec4 a = invVP * vec4(ndc, -1.0, 1.0), b = invVP * vec4(ndc, 1.0, 1.0);
    vec3 ro = eye, rd = normalize(b.xyz / b.w - a.xyz / a.w);
    vec3 wd = normalize(sunDir / toKm);                      // the sun's direction in the (stretched) box
    float kmPerW = length(wd * toKm);
    vec3 amb = vec3(0.42, 0.52, 0.70);
    // what is behind the storm: the ground (lit, shadowed by the storm, hazy with distance) or the sky
    vec3 bg = sky(rd);
    float tg = rd.z < 0.0 ? -ro.z / rd.z : 1e9;
    vec3 lo = vec3(-size.x * 0.5, -size.y * 0.5, 0.0), hi = vec3(size.x * 0.5, size.y * 0.5, size.z);
    vec3 t0 = (lo - ro) / rd, t1 = (hi - ro) / rd;
    vec3 tmn = min(t0, t1), tmx = max(t0, t1);
    float tn = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
    if (hasG == 1) {                                         // raised ground in the box: where the ray first meets it
      float te = min(min(min(tmx.x, tmx.y), tmx.z), tg), ds = length(size) / 500.0;
      for (int i = 0; i < 1000; i++) {
        float t = tn + float(i) * ds;
        if (t >= te) break;
        vec3 p = ro + rd * t;
        if (p.z < gz(p.xy)) { tg = gcross(ro, rd, max(t - ds, tn), t); break; }
      }
    }
    if (tg < 1e8) {
      vec3 g = ro + rd * tg;
      float lit = exp(-sunTau(g + vec3(0, 0, 1e-4), wd, kmPerW));
      vec3 n = hasG == 1 ? gnormal(g.xy) : vec3(0, 0, 1);
      float sl = max(dot(n, sunDir), 0.0);
      vec3 alb = mix(vec3(0.17, 0.24, 0.10), vec3(0.26, 0.24, 0.17), clamp(g.z / (0.12 * size.z), 0.0, 1.0));
      if (hasMap == 1) { alb = pow(mapAt(g.xy, pow(alb, vec3(1.0 / 2.2))), vec3(2.2)) * 0.9; }
      vec3 grd = alb *
                 (SUN * sl * lit * 0.35 + amb * (0.25 + 0.15 * n.z));
      float dist = length((g - ro) * toKm);
      bg = mix(grd, vec3(0.62, 0.70, 0.80), 1.0 - exp(-dist / 260.0));
    }
    float tf = min(min(min(tmx.x, tmx.y), tmx.z), tg);
    vec3 col = vec3(0.0); float T = 1.0;
    if (tf > tn) {
      float dt = length(size) / 340.0, kmPer = length(rd * toKm);
      float cth = dot(rd, sunDir);
      float ph = mix(hg(cth, 0.65), hg(cth, -0.25), 0.3);
      float t = tn + dt * fract(sin(dot(ndc, vec2(12.9898, 78.233))) * 43758.5453);     // jittered start: no banding
      for (int i = 0; i < 700; i++) {
        if (t > tf || T < 0.01) break;
        vec3 p = ro + rd * t;
        vec2 m = q(p);
        float sg = sigma(m);
        if (sg > 1e-3) {
          float ds = dt * kmPer;
          float ts = sunTau(p, wd, kmPerW);
          float sunT = exp(-ts) + 0.25 * exp(-0.25 * ts);       // multiple scattering lets some light deeper in
          float hgt = clamp(p.z / size.z * 2.0, 0.0, 1.0);
          float albedo = mix(0.98, 0.80, m.y / (m.x + m.y + 1e-6));   // rain shafts grey
          vec3 S = albedo * (SUN * sunT * ph * 4.0 + amb * (0.30 + 0.55 * hgt) * (0.4 + 0.6 * exp(-0.1 * ts)));
          if (relief > 0.0) {
            // relief: the cloud's own lumps lit as a surface (its outward normal from the density's gradient)
            vec3 h = texel * size;
            vec3 gr = vec3(sigma(q(p + vec3(h.x, 0, 0))) - sigma(q(p - vec3(h.x, 0, 0))),
                           sigma(q(p + vec3(0, h.y, 0))) - sigma(q(p - vec3(0, h.y, 0))),
                           sigma(q(p + vec3(0, 0, h.z))) - sigma(q(p - vec3(0, 0, h.z)))) / h;
            if (length(gr) > 1e-4) {
              vec3 n = -normalize(gr);
              float lam = max(dot(n, wd), 0.0);
              S *= mix(1.0, 0.22 + 1.25 * lam + 0.2 * max(n.z, 0.0), relief);
            }
          }
          float al = 1.0 - exp(-sg * ds);
          col += T * al * S;
          T *= 1.0 - al;
        }
        t += dt;
      }
    }
    vec3 c = col + T * bg;
    c = vec3(1.0) - exp(-c * 0.95);                             // exposure
    frag = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
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
    vao = gl.createVertexArray();
    volTex = gl.createTexture(); lutTex = gl.createTexture(); gndTex = gl.createTexture(); mapTex = gl.createTexture();
    return true;
  }

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
    for (const id of ['volSunLbl', 'volSunHLbl', 'volReliefLbl']) $(id).classList.toggle('hidden', !c);
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
    const key = `${box.join(',')}/${want()}`;
    busy = true;
    $('volStatus').textContent = 'Loading…';
    try {
      let q = `x0=${box[0]}&y0=${box[1]}&x1=${box[2]}&y1=${box[3]}&field=${want()}`;
      if (tor) q += `&live=1&ztop=${$('volTop').value}`;
      const r = await api('/api/volume?' + q);
      if (tor && ($('volPaths').checked || $('volTrack').checked)) {
        try { paths = await (await api('/api/tornado/paths')).json(); } catch (e) { paths = null; }
      }
      const { header, data } = unpack(await r.arrayBuffer());
      head = header;
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
        gl.bindTexture(gl.TEXTURE_2D, gndTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, header.nx, header.ny, 0, gl.RED, gl.FLOAT, gf);
        for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                              [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
          gl.texParameteri(gl.TEXTURE_2D, k, v);
      }
      if (!two) await uploadLUT();
      await loadMap();
      const run = S.run && S.run.t !== undefined ? `  ·  model +${fmtDur(S.run.t)}` : '';
      $('volStatus').textContent = (tor ? `Tornado zoom (${Math.round(header.dx)} m grid) · ` : '') +
        `${(box[2] - box[0]).toFixed(tor ? 1 : 0)} × ${(box[3] - box[1]).toFixed(tor ? 1 : 0)} km, ` +
        `${header.nx}×${header.ny}×${header.nz} points${run}`;
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

  function matrices(W, H) {
    const G = geometry();
    const target = [cam.px, cam.py, G.sz * 0.3];
    const cp = Math.cos(cam.pitch);
    const eye = [target[0] + cam.dist * cp * Math.cos(cam.yaw), target[1] + cam.dist * cp * Math.sin(cam.yaw),
                 target[2] + cam.dist * Math.sin(cam.pitch)];
    const VP = mul(perspective(0.7, W / H, 0.01, 50), lookAt(eye, target, [0, 0, 1]));
    return { G, eye, VP };
  }

  const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];

  function bindGround(u) {
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, gndTex); gl.uniform1i(u('gnd'), 2);
    gl.uniform1i(u('hasG'), head && head.ground ? 1 : 0);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, mapTex); gl.uniform1i(u('mapTex'), 3);
    gl.uniform1i(u('hasMap'), mapOn ? 1 : 0);
    gl.uniform1i(u('mapDark'), baseName() === 'osmdark' ? 1 : 0);
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
  window.volBaseChanged = () => { if (gl && box) loadMap().then(() => { if (head) draw(); }); };

  function drawCloud(G, eye, VP) {
    const az = +$('volSun').value * Math.PI / 180, el = +$('volSunH').value * Math.PI / 180;
    gl.useProgram(cloudProg); gl.bindVertexArray(vao);
    const u = n => gl.getUniformLocation(cloudProg, n);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, volTex); gl.uniform1i(u('vol'), 0);
    bindGround(u);
    gl.uniformMatrix4fv(u('invVP'), false, invert(VP));
    gl.uniform3f(u('eye'), ...eye);
    gl.uniform3f(u('size'), G.sx, G.sy, G.sz);
    gl.uniform3f(u('toKm'), 1 / G.k, 1 / G.k, 1 / (G.k * G.ex));
    gl.uniform3f(u('sunDir'), Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el));
    gl.uniform1f(u('density'), 2 * +$('volOpa').value);
    gl.uniform1f(u('relief'), +$('volRelief').value);
    gl.uniform3f(u('texel'), 1 / head.nx, 1 / head.ny, 1 / head.nz);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    $('volLegend').textContent = `Cloud view · sun in the ${COMPASS[Math.round(+$('volSun').value / 45) % 8]}, ` +
      `${$('volSunH').value}° up · height ×${G.ex}`;
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

  function draw() {
    const cv = $('volCanvas'), dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    }
    gl.viewport(0, 0, cv.width, cv.height);
    gl.clearColor(0.043, 0.055, 0.075, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const ov = $('volOverlay'), g = ov.getContext('2d');
    if (ov.width !== cv.width || ov.height !== cv.height) { ov.width = cv.width; ov.height = cv.height; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    if (!head || !box || head.field !== want()) return;
    const { G, eye, VP } = matrices(W, H);
    if (cloudMode()) { drawCloud(G, eye, VP); drawPaths(g, G, VP, W, H); return; }
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
      (F.signed && $('volBoth').checked ? ` (and ${F.fmt(-thr)})` : '') + `  ·  vertical ×${G.ex}`;
    drawPaths(g, G, VP, W, H);
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
    head = null; setSlider(false); draw();
    load(true);
    map.redraw();
    clearInterval(timer);
    timer = setInterval(() => { if ($('volLive').checked && S.run && S.live && !document.hidden) load(false); }, 8000);
  }
  function close() {
    $('volWin').classList.add('hidden');
    clearInterval(timer); timer = null; box = null; tor = false; paths = null;
    map.redraw();
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
    for (const id of ['volOpa', 'volEx', 'volBoth', 'volSun', 'volSunH', 'volRelief', 'volTrack', 'volPaths']) $(id).oninput = draw;
    $('volTop').oninput = () => { $('volTopV').textContent = `${$('volTop').value} km`; };
    $('volTop').onchange = () => load(true);
    $('volRefresh').onclick = () => load(true);
    $('volReset').onclick = () => {
      const first = tor ? 'pp' : 'dbz', reload = want() !== first;
      $('volField').value = first; $('volMode').value = 'iso'; lastMode = 'iso';
      $('volRelief').value = 0.6;
      $('volOpa').value = 0.5; $('volEx').value = 3; $('volSun').value = 235; $('volSunH').value = 30;
      $('volBoth').checked = false; $('volLive').checked = true;
      Object.assign(cam, CAM0);
      setSlider(false);
      if (reload) load(true); else draw();
    };
    const cv = $('volOverlay');
    let drag = null;
    cv.addEventListener('contextmenu', e => e.preventDefault());
    cv.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY, b: e.button, yaw: cam.yaw, pitch: cam.pitch, px: cam.px, py: cam.py }; e.preventDefault(); });
    window.addEventListener('mousemove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (drag.b === 2 || e.shiftKey) {                    // move the view across the ground
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
      cam.dist = Math.min(12, Math.max(0.4, cam.dist * Math.exp(e.deltaY * 0.0012)));
      draw();
    }, { passive: false });
    new ResizeObserver(() => { if (box) draw(); }).observe($('volCanvas'));
  }

  return { init, open, close, drawBox, openTornado, closeTornado };
})();
window.Vol = Vol;
