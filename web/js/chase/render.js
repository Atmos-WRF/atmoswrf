'use strict';
// Chase mode's picture: the 3-D view's cloud renderer (volume.js's FS_CLOUD: the sunlit cloud and rain from the
// run's volume, the real land cover on the ground, its light volume) seen from a car, with the roads and the cars
// drawn first into a small geometry buffer (colour + distance) that the cloud's ray march stops at and lights --
// so they sit under the storm's shadow, fade into the rain and haze like the ground does.
// World positions are metres on the run's map plane; the volume's box is world km; its units: the longer side 2.
const Render = (() => {
  const L = () => Vol.lib;
  let gl = null, cv = null, cloudProg = null, lightProg = null, roadProg = null, meshProg = null, vao = null;
  let volTex = null, gndTex = null, noiseTex = null, lightTex = null, lightFbo = null, lightKey = '', loadSeq = 0;
  const insTex = [];
  // Blending between volumes: volA / insA / lightA hold the picture shown when the newest volume came in (a blend
  // under way included), volB / insB / lightB the newest; volTex / insTex / lightTex, which the cloud pass reads,
  // are mixed from them (mixProg) as the blend runs, so the storm moves smoothly instead of jumping each fetch.
  let volA = null, volB = null, lightA = null, lightB = null, mixProg = null, mixFbo = null;
  const insA = [], insB = [];
  const mixSt = { t0: 0, dur: 0, f: 1, at: 0, dirty: false, fMain: 1, atMain: 0, fIns: 1, atIns: 0, insBlend: [false, false, false, false], lightB: -1,
                  tA: null, tB: null, warp: false, insWarp: [false, false, false, false], insBoxA: [null, null, null, null] };
  // a blend may run on past its end, carried along the wind, by this much (x its length) while the next volume is late
  const OVERRUN = 1.75;
  // the model's wind with each volume (header.wind): the box's and each nest's, for the start of the blend (A) and
  // the newest (B); hasWind: the shown volume came with it
  let windA = null, windB = null, hasWind = false, flowEpoch = 0;
  const wInsA = [], wInsB = [];
  const WARP_MAX = 240;                          // s of model time: a longer step between volumes just fades
  const MIX_EVERY = 33;                          // ms between mixing passes (~30 a second at most)
  const MIX_MAIN = 125, MIX_INS = 50;            // ms: the storm's box, its nests (advance)
  let head = null, box = null, gMap = null, floatLinear = false, volData = null;
  // the far cloud (setFar, FAR_GLSL): per tier the newest as it came (raw: the codes) and lit (B: the codes and their
  // light, farLightSource), the one shown when it came (A, lit), and the one read (tex), mixed from A and B; head:
  // the newest's header (box m), from: the older's box while blending into a moved one, key: what B was lit for
  const farV = [0, 1].map(() => ({ tex: null, A: null, B: null, raw: null, head: null, from: null, key: '', litAt: 0, t0: 0, dur: 0, f: 1, at: 0, dirty: false }));
  const FAR_MIX_EVERY = 250;                     // ms: they change slowly
  const FAR_RELIGHT = 20000;                     // ms: at most this often for the sun's own motion
  const gb ={ fbo: null, col: null, dist: null, light: null, depth: null, w: 0, h: 0 };
  const roads = { buf: null, vao: null, n: 0, origin: [0, 0] };
  const meshes = new Map();                      // car kind/colour -> { body, wheel }
  const partBufs = new WeakMap();                // a figure's part (its vertex data) -> its buffer
  let treeProg = null, precipProg = null, mergeProg = null, farLightProg = null;
  const trees = { mesh: null, near: 0, far: 0, inst: null, count: 0, nearCount: 0, origin: [0, 0], vao: null };
  const blds = { buf: null, vao: null, n: 0, origin: [0, 0] };
  const lamps = { buf: null, vao: null, n: 0, origin: [0, 0] };
  let lampProg = null, boltProg = null;
  const bolts = { vao: null, buf: null };

  // ---------------------------------------------------------------- shaders
  const ROAD_VS = `#version 300 es
  layout(location = 0) in vec3 p; layout(location = 1) in vec2 uv; layout(location = 2) in vec2 kp;
  uniform mat4 VP; uniform vec3 org; uniform float s;
  out vec3 wpos; out vec2 vuv; flat out vec2 vk; out vec2 wm;
  void main() {
    wpos = org + p * s; vuv = uv; vk = kp; wm = p.xy;
    gl_Position = VP * vec4(wpos, 1.0);
  }`;
  const ROAD_FS = `#version 300 es
  precision highp float;
  in vec3 wpos; in vec2 vuv; flat in vec2 vk; in vec2 wm;
  uniform vec3 eye, sunDir; uniform float wet;
  layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1; layout(location = 2) out vec4 o2;
  const float HALF[8] = float[8](5.6, 5.0, 4.4, 3.9, 3.5, 3.0, 3.1, 1.8);
  float h21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
  float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
  float line(float x, float x0, float w, float aa) { return 1.0 - smoothstep(w * 0.5 - aa, w * 0.5 + aa, abs(x - x0)); }
  void main() {
    int k = int(vk.x + 0.5); bool paved = vk.y > 0.5;
    float u = vuv.x, v = vuv.y, au = abs(u);
    float hw = HALF[k], x = u * hw;
    float n1 = vn(wm * 0.7), n2 = vn(wm * 3.1), n3 = h21(floor(wm * 9.0));
    vec3 c; float lam = max(sunDir.z, 0.0);
    if (au > 1.0) {                                   // the banks: mown verge, a little darker in the ditch
      float t = clamp((au - 1.0) / 1.5, 0.0, 1.0);
      c = mix(vec3(0.33, 0.36, 0.20), vec3(0.27, 0.33, 0.17), n1) * (0.85 + 0.25 * n2) * mix(1.0, 0.85, t);
      lam *= 0.95;
    } else if (paved) {
      float aa = max(fwidth(x), 0.02), av = fwidth(v);
      float fade = 1.0 - smoothstep(0.08, 0.4, aa);      // far away: the paint fades instead of shimmering
      c = vec3(0.235, 0.235, 0.24) * (0.82 + 0.18 * n1 + 0.12 * (n3 - 0.5)) * mix(1.0, 0.6, wet);
      c *= 1.0 - 0.08 * smoothstep(0.35, 0.0, abs(abs(x) - hw * 0.45)) ;   // the wheel tracks, worn darker
      vec3 W = vec3(0.86, 0.86, 0.84), Y = vec3(0.86, 0.70, 0.18);
      float pw = 0.0, py = 0.0;
      if (v >= 0.0) {
        if (k <= 4) pw = max(pw, line(abs(x), hw - 0.3, 0.13, aa));                           // white edges
        if (k <= 1) {                                                                           // a carriageway
          float dash = step(fract(v / 12.0), 0.25);
          pw = max(pw, dash * max(line(x, -hw / 3.0, 0.13, aa), line(x, hw / 3.0, 0.13, aa)));
        } else if (k <= 3) py = max(line(x, -0.13, 0.1, aa), line(x, 0.13, 0.1, aa));          // double yellow
        else if (k == 4) py = step(fract(v / 12.0), 0.25) * line(x, 0.0, 0.11, aa);           // dashed yellow
      }
      float wear = 0.75 + 0.25 * n2;
      c = mix(c, W * wear, pw * fade); c = mix(c, Y * wear, py * fade);
    } else {                                            // gravel: two wheel ruts, the crown between them
      c = mix(vec3(0.50, 0.44, 0.34), vec3(0.42, 0.36, 0.27), n1) * (0.85 + 0.3 * (n3 - 0.5));
      float rut = max(line(x, -0.85, 0.5, 0.15), line(x, 0.85, 0.5, 0.15));
      c *= 1.0 - 0.15 * rut;
      if (k == 7) c = mix(c, vec3(0.34, 0.37, 0.21), (1.0 - rut) * smoothstep(0.5, 0.2, abs(x)) * 0.7);
      c *= mix(1.0, 0.7, wet);
    }
    o0 = vec4(c, lam);
    o1 = vec4(length(wpos - eye), 0.0, 0.0, 1.0);
    o2 = vec4(0.0);
  }`;
  // a street lamp's pool of light: a disc at its foot (radius r m) a little over the ground, its colour x intensity
  // added into the geometry buffer's light target where it lies over what was drawn (the cloud pass lights the
  // surfaces there with it)
  const LAMP_VS = `#version 300 es
  layout(location = 0) in vec4 pr; layout(location = 1) in vec4 ci;
  uniform mat4 VP; uniform vec3 org; uniform float s;
  out vec2 q; out vec3 lc;
  void main() {
    vec2 c = vec2(gl_VertexID == 1 || gl_VertexID == 2 || gl_VertexID == 4 ? 1.0 : -1.0, gl_VertexID >= 2 && gl_VertexID != 3 ? 1.0 : -1.0);
    q = c; lc = ci.rgb * ci.a;
    vec3 w = org + vec3(pr.xy + c * pr.w, pr.z) * s;
    gl_Position = VP * vec4(w, 1.0);
  }`;
  const LAMP_FS = `#version 300 es
  precision highp float;
  in vec2 q; in vec3 lc;
  layout(location = 2) out vec4 o2;
  void main() {
    float r2 = dot(q, q);
    if (r2 > 1.0) discard;
    float f = (1.0 - r2) * (1.0 - r2);
    o2 = vec4(lc * f, 0.0);
  }`;
  // lightning's channels (lightning.js bolts): ribbons facing the eye, a white core in a blue-white glow, added over
  // the finished picture where nothing drawn is nearer (the geometry pass's distance, as the rain)
  const BOLT_VS = `#version 300 es
  layout(location = 0) in vec4 pv; layout(location = 1) in float a;
  uniform mat4 VP;
  out float vv; out float va; out vec3 wpos;
  void main() { wpos = pv.xyz; vv = pv.w; va = a; gl_Position = VP * vec4(pv.xyz, 1.0); }`;
  const BOLT_FS = `#version 300 es
  precision highp float;
  in float vv; in float va; in vec3 wpos;
  uniform sampler2D geoT; uniform vec2 res; uniform vec3 eye;
  out vec4 frag;
  void main() {
    float g = abs(texture(geoT, gl_FragCoord.xy / res).r);
    if (g > 0.0 && g < length(wpos - eye)) discard;
    float f = 1.0 - vv * vv;
    frag = vec4(vec3(0.82, 0.88, 1.0) * va * f * f + vec3(1.0) * va * smoothstep(0.55, 0.95, f), 1.0);
  }`;
  const MESH_VS = `#version 300 es
  layout(location = 0) in vec3 p; layout(location = 1) in vec3 n; layout(location = 2) in vec3 c;
  uniform mat4 VP; uniform mat3 R; uniform vec3 org; uniform float s;
  out vec3 wpos; out vec3 nw; out vec3 col;
  void main() { wpos = org + R * p * s; nw = R * n; col = c; gl_Position = VP * vec4(wpos, 1.0); }`;
  const MESH_FS = `#version 300 es
  precision highp float;
  in vec3 wpos; in vec3 nw; in vec3 col;
  uniform vec3 eye, sunDir;
  layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1; layout(location = 2) out vec4 o2;
  void main() {
    vec3 n = normalize(nw);
    float lam = max(dot(n, sunDir), 0.0);
    // a light (colour + 10: a lit window, a lamp, a canopy's lights): dark glass by day, its colour glowing at night
    bool em = col.r > 9.5;
    vec3 cc = em ? vec3(0.13, 0.15, 0.18) : col;
    // the sky's light on faces turned up, less on the sides and under (the cloud pass adds one flat ambient)
    vec3 c = cc * (0.72 + 0.28 * n.z);
    o0 = vec4(c, lam);
    o1 = vec4(length(wpos - eye), 0.0, 0.0, 1.0);
    o2 = em ? vec4(col - 10.0, 1.0) : vec4(0.0);
  }`;

  // The rain and snow falling round the camera (setPrecip's amounts): drops and flakes fixed in the world, falling
  // and blown with the model's wind at the car, in a cube of side `cube` m that wraps round the eye.  Each is a
  // quad drawn from the vertex id: a rain drop is a streak along its motion past the camera during `expo` s (what
  // the eye smears it to), a flake a soft dot.  Behind the roads, buildings, trees and cars (the geometry pass's
  // distance), faded at the cube's edge and right at the eye.
  const PRECIP_VS = `#version 300 es
  uniform mat4 VP; uniform vec3 eye; uniform float s, t, cube, snow, minD, expo, focal;
  uniform vec3 camV, wind; uniform vec2 res;
  out float va; out vec2 vq; out float vd; flat out float vsn;
  float h(float n) { return fract(sin(n) * 43758.5453); }
  void main() {
    float id = float(gl_InstanceID);
    int c = gl_VertexID;
    vec3 r = vec3(h(id * 1.13 + 0.1), h(id * 2.71 + 0.3), h(id * 3.97 + 0.7));
    bool sn = h(id * 7.31 + 0.9) < snow;
    float fall = sn ? 0.9 + 0.7 * r.x : 6.5 + 3.0 * r.y;
    vec3 vel = vec3(wind.xy * (sn ? 1.0 : 0.9), -fall);
    vec3 p0 = r * cube + vel * t;
    if (sn) p0.xy += 0.35 * vec2(sin(t * (1.1 + r.z) + id), cos(t * (0.8 + r.x) + id * 1.7));
    vec3 eyeM = eye / s;
    vec3 rel = mod(p0 - eyeM, cube) - 0.5 * cube;
    float d = length(rel);
    // the streak's two ends (a flake: one point, made a dot)
    vec3 dv = (vel - camV) * (sn ? 0.0 : expo);
    vec4 c0 = VP * vec4(eye + rel * s, 1.0), c1 = VP * vec4(eye + (rel - dv) * s, 1.0);
    if (c0.w < 0.3 * s || c1.w < 0.3 * s || d < minD) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); va = 0.0; return; }
    vec2 a = c0.xy / c0.w * res * 0.5, b = c1.xy / c1.w * res * 0.5;    // pixels from the middle
    vec2 dir = b - a; float L = length(dir);
    dir = L > 0.01 ? dir / L : vec2(0.0, 1.0);
    vec2 nrm = vec2(-dir.y, dir.x);
    // size on the screen: drops ~1 px across, flakes their ~7 mm, never under 1.2 px (fainter instead)
    float px = (sn ? 0.007 : 0.0016) * s * focal / max(c0.w, 1e-6);     // focal: pixels per unit at distance 1
    float wpx = max(px, sn ? 1.4 : 0.9);
    float cover = clamp(px / wpx, sn ? 0.45 : 0.7, 1.0);
    int corner[6] = int[6](0, 1, 2, 2, 1, 3);
    int k = corner[c];
    float e = float(k / 2), side = float(k % 2) * 2.0 - 1.0;
    vec2 P = mix(a, b, e) + nrm * side * wpx + dir * (e * 2.0 - 1.0) * (sn ? wpx : 0.5);
    float w = mix(c0.w, c1.w, e);
    gl_Position = vec4(P / (res * 0.5) * w, mix(c0.z, c1.z, e), w);
    vq = vec2(e * 2.0 - 1.0, side);
    vd = d * s; vsn = sn ? 1.0 : 0.0;
    float edge = smoothstep(0.5 * cube, 0.32 * cube, max(abs(rel.x), max(abs(rel.y), abs(rel.z))));
    va = edge * smoothstep(minD, minD + 0.6, d) * cover * (sn ? 0.9 : 0.5 * clamp(2.5 / max(L, 1.0) + 0.55, 0.45, 1.0));
  }`;
  const PRECIP_FS = `#version 300 es
  precision highp float;
  in float va; in vec2 vq; in float vd; flat in float vsn;
  uniform sampler2D geoT; uniform vec2 res; uniform vec3 col;
  out vec4 frag;
  void main() {
    float g = abs(texture(geoT, gl_FragCoord.xy / res).r);
    if (g > 0.0 && g < vd) discard;
    float a = vsn > 0.5 ? va * smoothstep(1.0, 0.25, length(vq)) : va * (1.0 - abs(vq.y)) * smoothstep(1.0, 0.6, abs(vq.x));
    if (a < 0.004) discard;
    frag = vec4(col * (vsn > 0.5 ? 1.15 : 1.0), a);
  }`;

  // the trees (scenery.js): a hexagonal trunk and a lumpy low-poly crown, one instance per tree
  const TREE_VS = `#version 300 es
  layout(location = 0) in vec3 p; layout(location = 1) in vec3 n; layout(location = 2) in float part;
  layout(location = 3) in vec4 a; layout(location = 4) in vec2 b;
  uniform mat4 VP; uniform vec3 org; uniform float s;
  // the wind in the trees: uWind (the model's u, v m/s at the camera, its gust factor), uTime s.  A tree leans
  // downwind (bend ~ W^2: 1 % of its height at 10 m/s, 10 % at 30, 40 % at 60), sways and shivers about it, more in
  // the gusts.
  uniform vec3 uWind; uniform float uTime;
  out vec3 wpos; out vec3 nw; out vec3 col;
  vec2 swayAt(vec2 xy, float h, float f, float sd) {
    vec2 w = uWind.xy; float W = length(w);
    if (W < 0.3) return vec2(0.0);
    vec2 dir = w / W;
    float lean = min(0.00011 * W * W, 0.42), fr = 0.35 + 0.012 * W;
    float osc = sin(uTime * 6.2832 * fr + sd * 40.0) * (0.25 + 0.35 * clamp((W - 15.0) / 40.0, 0.0, 1.0));
    float gust = (uWind.z - 1.0) * max(0.0, sin(uTime * 0.9 + xy.x * 0.01) * sin(uTime * 2.3 + xy.y * 0.013 + 1.0));
    vec2 perp = vec2(-dir.y, dir.x) * sin(uTime * 6.2832 * 2.7 * fr + sd * 17.0) * 0.12 * lean;
    return (dir * lean * (1.0 + 0.6 * gust) * (1.0 + osc) + perp) * h * f * f;
  }
  void main() {
    float h = a.w, cr = b.x, sd = b.y;
    vec3 local, nn;
    if (part < 0.5) {
      float tr = 0.12 + h * 0.018;
      local = vec3(p.xy * tr, p.z * (h - cr * 1.1));
      nn = n; col = vec3(0.42, 0.35, 0.29);
    } else {
      float lump = 1.0 + 0.2 * sin(p.x * 3.1 + sd * 40.0) * sin(p.y * 2.7 + sd * 17.0) + 0.14 * sin(p.z * 4.3 + sd * 9.0);
      vec3 sc = vec3(cr, cr * (0.85 + 0.3 * fract(sd * 3.7)), cr * (0.8 + 0.45 * fract(sd * 7.3)));
      local = vec3(0.0, 0.0, h - sc.z) + p * sc * lump;
      nn = normalize(n / sc);
      col = mix(vec3(0.34, 0.43, 0.23), vec3(0.50, 0.55, 0.32), fract(sd * 13.7)) * (0.82 + 0.22 * p.z);
    }
    float c = cos(sd * 31.0), si = sin(sd * 31.0);
    mat2 rot = mat2(c, si, -si, c);
    local.xy = rot * local.xy; nn.xy = rot * nn.xy;
    nw = nn;
    vec2 sw = swayAt(a.xy, h, clamp(local.z / max(h, 0.5), 0.0, 1.2), sd);
    local.xy += sw; local.z -= 0.5 * dot(sw, sw) / max(h, 0.5);
    wpos = org + (a.xyz + local) * s;
    gl_Position = VP * vec4(wpos, 1.0);
  }`;
  function treeMesh() {
    const V = [];
    const P = (x, y, z, nx, ny, nz, part) => V.push(x, y, z, nx, ny, nz, part);
    for (let i = 0; i < 6; i++) {                     // the trunk: a hexagonal prism, radius 1, 0..1 high
      const a0 = i * Math.PI / 3, a1 = (i + 1) * Math.PI / 3, am = (a0 + a1) / 2;
      const q = [[Math.cos(a0), Math.sin(a0), 0], [Math.cos(a1), Math.sin(a1), 0], [Math.cos(a1), Math.sin(a1), 1], [Math.cos(a0), Math.sin(a0), 1]];
      for (const k of [0, 1, 2, 0, 2, 3]) P(...q[k], Math.cos(am), Math.sin(am), 0, 0);
    }
    // the crown: an icosahedron, each face split in four, pushed out to the sphere
    const t = (1 + Math.sqrt(5)) / 2;
    let vs = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
    const fs = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    const nrm = v => { const l = Math.hypot(...v); return v.map(x => x / l); };
    vs = vs.map(nrm);
    for (const [a, b, c] of fs) {
      const A = vs[a], B = vs[b], C = vs[c];
      const ab = nrm(A.map((x, i) => x + B[i])), bc = nrm(B.map((x, i) => x + C[i])), ca = nrm(C.map((x, i) => x + A[i]));
      for (const tri of [[A, ab, ca], [ab, B, bc], [ca, bc, C], [ab, bc, ca]]) for (const v of tri) P(v[0], v[1], v[2], v[0], v[1], v[2], 1);
    }
    const near = V.length / 7;
    // far away: the plain icosahedron, no trunk
    for (const f of fs) for (const k of f) P(...vs[k], ...vs[k], 1);
    return { data: new Float32Array(V), near, far: V.length / 7 - near };
  }

  // The model's own cloud, alive: its small detail (volume.js's erosion and lumps) carried with the storm (drift m/s
  // over the model's time flowT s).  The cloud's shape is the model's (the volume, carried along its own 3-D wind
  // between snapshots by the blend: MIX_FS).
  const FLOW_GLSL = `
  uniform float flowT; uniform vec2 drift;
  // k: the detail's coordinates (km), carried with the storm
  vec3 flowK(vec3 p) { return p * toKm + kmOrigin - vec3(drift * flowT / 1000.0, 0.0); }
  // the detail's scale where media() is reading (km): the cell of the finest grid there -- a nest's own, not the
  // box's, or the box's coarse cell would shift and erode a 37 m nest's funnel (100-200 m wide) into nothing
  // and how much of it (the Detail slider's): it fades out on grids under ~150 m, which resolve the cloud's real
  // structure -- made-up erosion and lumps there would break the model's own funnel and wall cloud into nothing
  float gDkm = 1.0, gDet = 0.0;
  // a nest's share of the detail's scale at p: 1 in its middle, down to 0 over its outer 30 % (and its top 15 %) --
  // far wider than its data's own blend (insW, 8 %): the look changes with the scale, and over a narrow band that
  // change showed as a hard line at the nest's edge
  float insWd(vec3 p, int i) {
    vec4 b = insBox[i];
    vec3 u = vec3((p.x - b.x) / (b.z - b.x), (p.y - b.y) / (b.w - b.y), p.z / insTop[i]);
    if (any(lessThan(u, vec3(0.0))) || any(greaterThan(u, vec3(1.0)))) return 0.0;
    vec2 e = min(u.xy, 1.0 - u.xy);
    return smoothstep(0.0, 1.0, clamp(min(min(e.x, e.y) / 0.3, (1.0 - u.z) / 0.15), 0.0, 1.0));
  }
  // the grid's cell at box point p (box units): the box's, each nest's blended in over it (log), coarsest to finest
  // (the insets come finest first), so nests inside nests go over smoothly too
  float cellAt(vec3 p) {
    float lc = log(size.x * texel.x);
    if (insN > 3) lc = mix(lc, log((insBox[3].z - insBox[3].x) / float(textureSize(ins3, 0).x)), insWd(p, 3));
    if (insN > 2) lc = mix(lc, log((insBox[2].z - insBox[2].x) / float(textureSize(ins2, 0).x)), insWd(p, 2));
    if (insN > 1) lc = mix(lc, log((insBox[1].z - insBox[1].x) / float(textureSize(ins1, 0).x)), insWd(p, 1));
    if (insN > 0) lc = mix(lc, log((insBox[0].z - insBox[0].x) / float(textureSize(ins0, 0).x)), insWd(p, 0));
    return exp(lc);
  }
  // the view's march step at p: in a nest at most k of its cell (the box's own step is ~1/420 of the box, ~340 m
  // over chase's 100 km, which steps over a 200 m funnel), else dt.  Only the tornado's own nests (cells under
  // 100-200 m, faded in between), and coarser with the distance from the eye (a 200 m funnel 6 km off is a few
  // pixels): the step is k cells x nestStep x the km past 1.5 km over 1.5
  uniform float nestStep;                                      // x the cell (0: off): the Settings' quality
  float nestDt(vec3 p, float dt, float k) {
    if (insN == 0 || nestStep <= 0.0) return dt;
    float cell = cellAt(p), fade = smoothstep(0.1, 0.2, cell * toKm.x);
    if (fade >= 1.0) return dt;
    float dk = length((p - eye) * toKm);
    return mix(min(dt, cell * k * nestStep * max(1.0, dk / 1.5)), dt, fade);
  }
  vec2 shapeDetail(vec3 k, vec2 s) {
    // the noise on the box's scale everywhere (detailKm): on a nest's (gDkm, down to 37 m) its ~20 m features
    // aliased into streaks and a fine comb seen from km off; in the nests only its strength fades (gDet)
    float n = texture(noiseTex, k / (detailKm * 20.0)).r * 0.55 + texture(noiseTex, k / (detailKm * 7.0) + 0.37).r * 0.45;
    // erosion: thin cloud breaks into wisps and billows, the solid core keeps its shape
    float cov = clamp(s.x / 30.0, 0.0, 1.0);
    float thr = 0.8 * gDet * smoothstep(0.3, 0.7, n) * (1.0 - cov);
    float kept = clamp((cov - thr) / max(1.0 - thr, 0.05), 0.0, 1.0);
    s.x *= cov > 1e-4 ? kept / cov : 0.0;
    s.x *= max(mix(1.0, 2.6 * n - 0.3, 0.55 * gDet), 0.0);     // lumps inside, the mean kept
    // rain curtains: streaks that hang down, thicker and thinner side by side -- on the box's scale wherever they
    // are (detailKm, ~140 m to a few km across): rain shafts are that size, and on a nest's (gDkm) the 10x vertical
    // stretch made a fine comb of streaks tens of metres wide across the rain round a tornado
    float r = texture(noiseTex, vec3(k.xy / (detailKm * 9.0), k.z / (detailKm * 90.0)) + 0.61).g;
    s.y *= max(mix(1.0, 2.4 * r - 0.2, 0.8 * gDet), 0.0);
    return s;
  }
  // the light the ground sends up (bounce): the ground under and round about lit by the sun and the sky (the light
  // volume's lowest layer), its share coming up through the cloud below (the light volume's third channel)
  uniform float bounce;
  vec3 bounceAt(vec3 p) {
    if (bounce <= 0.0 || useLight == 0) return vec3(0.0);
    vec3 uvw = vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z);
    float td = texture(lightTex, uvw).b;
    float z0 = 0.5 / float(textureSize(lightTex, 0).z), hk = p.z * toKm.z;
    vec2 off = vec2(max(2.0, 2.0 * hk) / toKm.x) / size.xy;
    vec2 g = 0.4 * exp(-vec2(1.0, 0.5) * texture(lightTex, vec3(uvw.xy, z0)).rg);
    for (int j = 0; j < 4; j++) {
      vec2 o = j == 0 ? vec2(off.x, 0.0) : j == 1 ? vec2(-off.x, 0.0) : j == 2 ? vec2(0.0, off.y) : vec2(0.0, -off.y);
      g += 0.15 * exp(-vec2(1.0, 0.5) * texture(lightTex, vec3(uvw.xy + o, z0)).rg);
    }
    vec3 E = SUN * max(sunDir.z, 0.0) * g.x * 0.35 + vec3(0.40, 0.50, 0.68) * (0.10 + 0.30 * g.y);
    return bounce * 0.6 * vec3(0.17, 0.19, 0.13) * E * exp(-0.35 * td);
  }`;

  // Night (game.js sky()): dayK 1 by day, 0 at night (the real sun's height, civil twilight between); expo the eye's
  // adaptation; the moon's direction and how much of it is lit (moonK, 0 under the horizon).  Lightning
  // (lightning.js): a flash at flashP (box coords; w: its brightness) lighting the cloud round it, and flashC the
  // light it throws on everything near.  Headlights: up to 4 (headP: box coords, w: on; headD: where they point).
  // lampK: the lamps and lit windows on (lights.js), the geometry buffer's light target (geoE) the light they throw.
  const NIGHT_UNI = `
  uniform float dayK, expo, moonK, lampK; uniform vec3 moonDir;
  uniform vec4 flashP; uniform vec3 flashC;
  uniform vec4 headP[4]; uniform vec3 headD[4]; uniform int headN;
  uniform sampler2D geoE;
`;
  const NIGHT_GLSL = `
  vec3 nightAmb() { return vec3(0.006, 0.008, 0.016) + vec3(0.010, 0.011, 0.013) * moonK; }
  vec3 nightSky(vec3 rd) {
    float h = max(rd.z, 0.0);
    vec3 c = mix(vec3(0.011, 0.012, 0.018), vec3(0.002, 0.003, 0.008), pow(h, 0.35));
    c += vec3(0.006, 0.008, 0.014) * moonK * (1.0 - h);
    // the stars: one in a few hundred cells of the sky, a soft dot, dimmed near the horizon and by moonlight
    vec3 q = rd * 320.0, cell = floor(q), f = fract(q) - 0.5;
    float hs = fract(sin(dot(cell, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
    float star = step(0.986, hs) * smoothstep(0.32, 0.05, length(f)) * pow((hs - 0.986) / 0.014, 2.0);
    c += vec3(0.85, 0.9, 1.0) * star * 0.09 * smoothstep(0.02, 0.2, rd.z) * (1.0 - 0.6 * moonK);
    // the moon: its disc (half a degree) and a glow round it
    float md = dot(rd, moonDir), up = step(-0.01, moonDir.z);
    c += vec3(0.95, 0.93, 0.86) * up * (smoothstep(0.999955, 0.99997, md) * (0.25 + 1.2 * moonK) + pow(max(md, 0.0), 300.0) * 0.01 * moonK);
    return c;
  }
  vec3 flashAt(vec3 p) {
    if (flashP.w <= 0.0) return flashC;
    vec3 d = (p - flashP.xyz) * toKm;
    return flashC + flashP.w * vec3(0.82, 0.86, 1.0) / (1.0 + dot(d, d) / 1.6);
  }
  vec3 headLight(vec3 g, vec3 n) {
    vec3 L = vec3(0.0);
    for (int h = 0; h < 4; h++) {
      if (h >= headN) break;
      vec3 dv = (g - headP[h].xyz) * toKm * 1000.0;
      float dd = length(dv);
      if (dd > 160.0 || headP[h].w <= 0.0) continue;
      vec3 l = dv / max(dd, 1e-3);
      float cs = dot(l, headD[h]);
      float beam = smoothstep(0.9, 0.985, cs) + 0.08 * smoothstep(0.6, 0.9, cs);
      L += vec3(1.0, 0.94, 0.84) * headP[h].w * beam * 34.0 / (dd * dd + 9.0) * max(dot(n, -l), 0.3);
    }
    return L;
  }
`;

  // The rain's extinction per km from its g/kg (x the Rain slider): as the model's (1.2 a g/kg) for heavy rain, from
  // RAIN_HEAVY g/kg up, and much less below it -- light rain is a thin veil, not a grey wall (0.3 g/kg: a quarter of
  // the linear, 0.1 g/kg: a ninth)
  const RAIN_GLSL = `
  float rainExt(float q) { return 1.2 * q * min(1.0, pow(q / 1.5, 0.9)); }
  // light that has diffused through the cloud: deep in a thick cloud the sun and the sky are scattered many times
  // over and fall off about as 1 / optical depth (two-stream), not exponentially, which keeps a storm's base and the
  // ground under it grey, not black -- the part of it the scattering octaves (exponential) leave out
  float fdiff(float tau) { return 1.0 / (1.0 + 0.12 * tau) - exp(-0.12 * tau); }
`;

  // volume.js's media(): the cloud and rain at box point p (extinction per km), its detail (mode > 0) carried along
  // with the storm (flowK)
  const MEDIA_GLSL = `
  vec2 media(vec3 p, int mode) {
    vec3 uvw = vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, p.z / size.z);
    if (any(lessThan(uvw, vec3(-0.01))) || any(greaterThan(uvw, vec3(1.01)))) return vec2(0.0);
    vec3 ka = vec3(0.0);
    gDkm = detailKm; gDet = detail;
    if (mode > 0 && detail > 0.0) {
      ka = flowK(p);
      // the grid's cell here, as a share of the box's (1 outside the nests)
      float rc = insN > 0 ? cellAt(p) / (size.x * texel.x) : 1.0;
      gDkm = detailKm * rc;
      gDet = detail * mix(0.1, 1.0, smoothstep(0.05, 0.2, gDkm));
      // domain warp: the surface pushed in and out by about a box cell -- towers, billows, cauliflower; on the
      // box's scale (see shapeDetail), weaker in the nests (gDet: a tenth in a tornado's, which it shredded)
      vec3 wq = (p * toKm + kmOrigin - vec3(drift * flowT / 1000.0, 0.0)) / (detailKm * 5.0);
      vec3 wv = vec3(texture(noiseTex, wq).rg, texture(noiseTex, wq.zxy + 0.41).r) - 0.5;
      uvw += wv * texel * 2.2 * gDet;
    }
    vec2 raw = mode == 2 && quality > 0 ? cubic(uvw) : texture(vol, uvw).rg;
    if (insN > 0) raw = nested(vec3((uvw.x - 0.5) * size.x, (uvw.y - 0.5) * size.y, uvw.z * size.z), vec4(raw, 0.0, 0.0)).rg;
    vec2 m = gkg(raw);
    vec2 s = vec2(25.0 * m.x, rain * rainExt(m.y)) * density;  // rain: the Rain slider (x the model's)
    vec2 e = min(uvw.xy, 1.0 - uvw.xy) / 0.03;                 // the box's sides fade: no walls of cloud
    s *= clamp(min(e.x, e.y), 0.0, 1.0);
    if (mode > 0 && detail > 0.0 && s.x + s.y > 1e-3) {
      s = shapeDetail(ka, s);
    }
    return s;
  }`;

  // The far cloud (setFar; game.js loadFar): two coarse volumes round the box out to the visibility, 0 the nearer and
  // finer, 1 the wider and coarser, marched after the box's own march where the ray leaves the box's inner part
  // (the box's cloud fades out over its sides as theirs fades in).  Each tier carries its own light (farLightSource:
  // the codes, and the optical depth to the sun and to the sky, worked out when it comes).  The level of detail drops
  // with the distance: the steps grow with it (1 % of it), the sub-grid warp stops past 250 km, and the wider,
  // coarser grid takes over past the nearer's edge.
  const FAR_GLSL = `
  uniform sampler3D farT0, farT1; uniform ivec2 farHas; uniform vec4 farBox[2]; uniform vec2 farTop, farCell;
  vec3 farU(vec3 p, int i) {
    vec4 b = farBox[i];
    return vec3((p.x - b.x) / (b.z - b.x), (p.y - b.y) / (b.w - b.y), p.z / farTop[i]);
  }
  float farEdge(vec3 u, float w) {
    if (any(lessThan(u, vec3(0.0))) || any(greaterThan(u, vec3(1.0)))) return 0.0;
    vec2 e = min(u.xy, 1.0 - u.xy) / w;
    return clamp(min(e.x, e.y), 0.0, 1.0);
  }
  vec3 farWarp(vec3 k, float cell) {
    vec3 q = k / (cell * 3.0);
    return vec3(texture(noiseTex, q).rg, texture(noiseTex, q.zxy + 0.41).r) - 0.5;
  }
  // extinction per km (cloud, precipitation) of the far cloud at p, less the box's own share there (media() fades
  // the box's sides over 3 %), and in lt its light (optical depth to the sun, to the sky: each tier's own light
  // pass); warpK: the sub-grid warp, in cells
  vec2 farMedia(vec3 p, float warpK, out vec2 lt) {
    lt = vec2(0.0);
    vec3 ub = vec3(p.x / size.x + 0.5, p.y / size.y + 0.5, 0.5);
    float wb = p.z > size.z ? 0.0 : farEdge(ub, 0.03);
    if (wb >= 0.999) return vec2(0.0);
    vec3 k = p * toKm + kmOrigin;
    vec4 c = vec4(0.0);
    float w0 = 0.0;
    if (farHas.x == 1) {
      vec3 u = farU(p, 0);
      w0 = farEdge(u, 0.08);
      if (w0 > 0.0) {
        if (warpK > 0.0) u += farWarp(k, farCell.x) * warpK / vec3(textureSize(farT0, 0));
        c = textureLod(farT0, u, 0.0);
      }
    }
    vec2 m = gkg(c.rg) * w0;
    lt = c.ba;
    if (farHas.y == 1 && w0 < 1.0) {
      vec3 u = farU(p, 1);
      float w1 = farEdge(u, 0.04);
      if (w1 > 0.0) {
        if (warpK > 0.0) u += farWarp(k, farCell.y) * warpK / vec3(textureSize(farT1, 0));
        vec4 c1 = textureLod(farT1, u, 0.0);
        m += gkg(c1.rg) * w1 * (1.0 - w0);
        lt = w0 > 0.0 ? mix(c1.ba, c.ba, w0) : c1.ba;
      }
    }
    return vec2(25.0 * m.x, rain * rainExt(m.y)) * density * (1.0 - wb);
  }
  vec3 farMarch(vec3 ro, vec3 rd, float tg, float jit, vec3 haze, vec3 amb, vec3 wd, float kmPerW, out float Tf) {
    Tf = 1.0;
    vec3 col = vec3(0.0);
    if (farHas.x + farHas.y == 0) return col;
    int o = farHas.y == 1 ? 1 : 0;
    vec4 b = farBox[o];
    vec3 t0 = (vec3(b.xy, 0.0) - ro) / rd, t1 = (vec3(b.zw, farTop[o]) - ro) / rd;
    vec3 tmn = min(t0, t1), tmx = max(t0, t1);
    float ta = max(max(tmn.x, tmn.y), max(tmn.z, 0.0)), tb = min(min(tmx.x, tmx.y), tmx.z);
    // the box's inner part is its own cloud's alone: from where the ray leaves it
    vec3 il = vec3(-size.x * 0.47, -size.y * 0.47, -1e3), ih = vec3(size.x * 0.47, size.y * 0.47, size.z);
    if (all(greaterThan(ro, il)) && all(lessThan(ro, ih))) {
      vec3 e = max((il - ro) / rd, (ih - ro) / rd);
      ta = max(ta, min(min(e.x, e.y), e.z));
    }
    float kmPer = length(rd * toKm);
    tb = min(min(tb, tg), 4.5 * hazeKm / kmPer);                // past this the haze has it all
    if (tb <= ta) return col;
    float cth = dot(rd, sunDir), cell = farHas.x == 1 ? farCell.x : farCell.y;
    float t = ta + jit * clamp(ta * kmPer * 0.01, 0.3, 8.0) / kmPer;
    for (int i = 0; i < 400; i++) {
      if (t > tb || Tf < 0.01) break;
      float dk = t * kmPer;
      float sk = clamp(dk * 0.01, max(0.3, 0.35 * cell), 8.0);  // km: the step grows with the distance
      vec3 p = ro + rd * t;
      vec2 lt;
      vec2 s = farMedia(p, dk < 250.0 ? 1.5 * detail : 0.0, lt);
      float sg = sig(s);
      if (sg <= 1e-3) { t += 2.0 * sk / kmPer; continue; }
      float ts = lt.x, tu = lt.y;
      float L = 0.0, aa = 1.0, bb = 1.0, cc = 1.0;
      for (int q = 0; q < 3; q++) {
        L += aa * exp(-bb * ts) * mix(hg(cth, 0.7 * cc), hg(cth, -0.25 * cc), 0.3);
        aa *= 0.5; bb *= 0.4; cc *= 0.5;
      }
      float skyv = exp(-0.45 * tu);
      float hgt = clamp(p.z / size.z * 2.0, 0.0, 1.0);
      float albedo = mix(0.98, 0.78, s.y / (sg + 1e-6));
      vec3 S = albedo * (flashAt(p) + SUN * max(sunDir.z, 0.0) * 1.1 * fdiff(ts) + amb * 0.9 * fdiff(tu) + SUN * L * 4.5 +
                         amb * (0.18 + 0.42 * skyv) * (0.55 + 0.45 * hgt) * (0.45 + 0.55 * exp(-0.12 * ts)));
      S = mix(S, haze * mix(0.5, 1.0, skyv), 1.0 - exp(-dk / hazeKm));
      float al = 1.0 - exp(-sg * sk);
      col += Tf * al * S;
      Tf *= 1.0 - al;
      t += sk / kmPer;
    }
    return col;
  }
`;

  // the 3-D view's cloud shader with a contrast control on the cloud alone (the ground and sky as they are):
  // dark bases darker, sunlit towers brighter, around a mid grey in log light
  function cloudSource() {
    let src = L().FS_CLOUD;
    const head = 'in vec2 ndc; out vec4 frag;', tail = 'vec3 c = col + T * bg;';
    if (!src.includes(head) || !src.includes(tail)) return src;
    // geoWins: with the detailed map the geometry (the map, the cars on it) is never hidden by the model's coarser ground
    src = src.replace(head, head + '\n  uniform float cloudContrast; uniform int geoWins;' + NIGHT_UNI + RAIN_GLSL);
    // the cloud's detail carried with the storm (FLOW_GLSL): media() as volume.js's, its detail from flowK's coordinates
    const media = 'vec2 media(vec3 p, int mode) {', ret = '    return s;\n  }\n  float sig(vec2 s)';
    const m0 = src.indexOf(media), m1 = src.indexOf('\n  float sig(vec2 s)');
    if (m0 >= 0 && m1 > m0 && src.slice(m0, m1).includes('vec3 k = p * toKm + kmOrigin;')) {
      src = src.slice(0, m0) + MEDIA_GLSL.trim() + src.slice(m1);
    } else console.warn('chase: the cloud\'s media() not as expected; its detail stays put');
    // the detailed map's pixels (distance stored negative): already lit as under a clear sky, taken wherever they
    // are (the model's coarser ground would otherwise cut through them), then darkened by the cloud's shadow and
    // the sky the cloud hides
    const geoHooks = [
      ['bool tree = false; vec4 ga = vec4(0.0);', 'bool tree = false; vec4 ga = vec4(0.0); bool prelit = false;'],
      ['float tt = texture(geoT, su).r;', 'float tr = texture(geoT, su).r, tt = abs(tr);'],
      ['if (tt > 0.0 && tt < tg) { tg = tt; tree = true; ga = texture(geoA, su); }',
       'if (tt > 0.0 && (tt < tg || tr < 0.0 || geoWins == 1)) { tg = tt; tree = true; ga = texture(geoA, su); prelit = tr < 0.0; }'],
      ['vec3 grd = alb * (SUN * sl * lit * 0.35 + amb * (0.10 + 0.30 * skyv) * (0.6 + 0.4 * n.z));',
       'vec3 grd = alb * (SUN * sl * lit * 0.35 + amb * (0.10 + 0.30 * skyv) * (0.6 + 0.4 * n.z));\n' +
       '      grd += alb * (SUN * max(sunDir.z, 0.0) * 0.35 * fdiff(lt.x) + amb * 0.3 * fdiff(lt.y));\n' +
       '      if (prelit) grd = pow(ga.rgb, vec3(2.2)) * (0.68 * lit + 0.32 * (0.25 + 0.75 * skyv) + 0.7 * fdiff(lt.x));'],
    ];
    if (geoHooks.every(([a]) => src.includes(a))) for (const [a, b] of geoHooks) src = src.replace(a, b);
    else console.warn('chase: geometry hooks not found; the detailed map is drawn as plain geometry');
    // night: the sun's light and the sky's faded out under a night sky, the eye adapted (expo); lightning lighting
    // the cloud from inside and everything round; headlights, lamps and lit windows on what is drawn
    const nightHooks = [
      ['  vec3 sky(vec3 rd) {', NIGHT_GLSL + '  vec3 sky(vec3 rd) {'],
      ['    c *= mix(0.45, 1.0, smoothstep(-0.02, 0.3, sunDir.z));\n', '    c *= mix(0.45, 1.0, smoothstep(-0.02, 0.3, sunDir.z));\n    c = mix(nightSky(rd), c, dayK);\n'],
      ['    SUN = mix(vec3(1.0, 0.55, 0.28) * 2.4, vec3(1.0, 0.95, 0.88) * 3.4, low);\n',
       '    SUN = mix(vec3(1.0, 0.55, 0.28) * 2.4, vec3(1.0, 0.95, 0.88) * 3.4, low) * dayK;\n'],
      ['    vec3 amb = vec3(0.40, 0.50, 0.68);\n', '    vec3 amb = mix(nightAmb(), vec3(0.40, 0.50, 0.68), dayK) + flashC;\n'],
      ['    vec3 haze = mix(vec3(0.70, 0.62, 0.55), vec3(0.64, 0.71, 0.80), low);\n',
       '    vec3 haze = mix(vec3(0.70, 0.62, 0.55), vec3(0.64, 0.71, 0.80), low) * mix(0.025, 1.0, dayK) + flashC * 0.5;\n'],
      ['      if (prelit) grd = pow(ga.rgb, vec3(2.2)) * (0.68 * lit + 0.32 * (0.25 + 0.75 * skyv) + 0.7 * fdiff(lt.x));',
       '      if (prelit) grd = pow(ga.rgb, vec3(2.2)) * (0.68 * lit + 0.32 * (0.25 + 0.75 * skyv) + 0.7 * fdiff(lt.x)) * mix(0.03 + 0.03 * moonK, 1.0, dayK);\n' +
       '      if (prelit) grd += alb * flashC * 0.4;\n' +
       '      grd += alb * headLight(g, n);\n' +
       '      if (tree) {\n' +
       '        vec4 E = texture(geoE, gl_FragCoord.xy / vec2(textureSize(geoT, 0)));\n' +
       '        grd += E.a > 0.5 ? E.rgb * E.rgb * 1.4 * lampK : alb * E.rgb * 3.0;\n' +
       '      }'],
      ['    c = aces(c * 0.85);', '    c = aces(c * 0.85 * expo);'],
    ];
    if (nightHooks.every(([a]) => src.split(a).length === 2)) for (const [a, b] of nightHooks) src = src.replace(a, b);
    else console.warn('chase: night hooks not found:', nightHooks.filter(([a]) => src.split(a).length !== 2).map(([a]) => a.trim().slice(0, 40)));
    // the light coming up off the ground
    const sHook = 'vec3 S = albedo * (SUN * L * 4.5 + ';
    // and the light diffused down through the cloud above (fdiff): the storm's base grey, not black
    if (src.includes(sHook) && src.includes(media) && src.includes(ret)) {
      src = src.replace(sHook, 'vec3 S = albedo * (bounceAt(p) + flashAt(p) + SUN * max(sunDir.z, 0.0) * 1.1 * fdiff(ts) + amb * 0.9 * fdiff(lt.y) + SUN * L * 4.5 + ');
    } else console.warn('chase: ground bounce hook not found');
    // the march through the nests at their own resolution (nestDt): the same step for the extinction and the advance;
    // fine only in cloud (the funnel, the wall cloud), not through rain alone, which is smooth and would cost the most
    // (every step lit)
    const marchHooks = [
      ['for (int i = 0; i < 900; i++) {', 'for (int i = 0; i < 2400; i++) {'],
      ['if (sg <= 1e-3) { t += dt * 1.8; continue; }', 'if (sg <= 1e-3) { t += nestDt(p, dt * 1.8, 3.0); continue; }'],
      ['float ds = dt * kmPer;', 'float ds = nestDt(p, dt, s.x > 0.2 * sg ? 0.6 : 4.0) * kmPer;'],
      ['        t += dt;\n', '        t += nestDt(p, dt, s.x > 0.2 * sg ? 0.6 : 4.0);\n'],
    ];
    if (src.includes(media) && src.includes(ret)) {
      src = src.replace(media, FLOW_GLSL + '\n  ' + media);
      if (marchHooks.every(([a]) => src.includes(a))) for (const [a, b] of marchHooks) src = src.replace(a, b);
      else console.warn('chase: march hooks not found; the nests are marched at the box\'s step');
    }
    // relief (four more reads of the cloud a step, a box cell apart) only on the coarser grids and where the step
    // is mostly cloud: inside a fine nest its box-sized offsets mean nothing and it costs the most.  Faded in and
    // out (rlf), not switched: a switch showed as a hard line at the nest's edge
    const rHead = 'if (relief > 0.0) {', rMix = '0.6 * relief);';
    if (src.includes(rHead) && src.includes(rMix)) {
      src = src.replace(rHead, 'float rlf = relief * smoothstep(0.1, 0.2, gDkm) * smoothstep(0.15, 0.4, s.x / (sg + 1e-6));\n' +
                        '          if (rlf > 0.0) {').replace(rMix, '0.6 * rlf);');
    } else console.warn('chase: relief hook not found');
    // the far cloud (FAR_GLSL) behind the box's, in front of the ground and the sky
    const mainHead = '  void main() {';
    if (src.split(mainHead).length === 2) {
      src = src.replace(mainHead, FAR_GLSL + mainHead);
      src = src.replace(tail, `if (T > 0.01) {
      float Tf; vec3 fc = farMarch(ro, rd, tg, jit, haze, amb, wd, kmPerW, Tf);
      col += T * fc; T *= Tf;
    }
    ${tail}`);
    } else console.warn('chase: no place for the far cloud in the cloud shader');
    src = src.replace(tail, `if (cloudContrast != 1.0 && T < 0.995) {
      vec3 Lc = col / (1.0 - T);
      float l = max(dot(Lc, vec3(0.2126, 0.7152, 0.0722)), 1e-4), piv = 0.55;
      col *= pow(l / piv, cloudContrast - 1.0);
    }
    ${tail}`);
    return src;
  }

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function program(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    p.u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); const nm = a.name.replace(/\[0\]$/, ''); p.u[nm] = gl.getUniformLocation(p, a.name); }
    return p;
  }
  const texParams = (target, filter, wrap) => {
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, wrap],
                          [gl.TEXTURE_WRAP_T, wrap]].concat(target === gl.TEXTURE_3D ? [[gl.TEXTURE_WRAP_R, wrap]] : []))
      gl.texParameteri(target, k, v);
  };

  // ---------------------------------------------------------------- blending volumes
  // With the model's wind (warp 1) the blend carries the picture along it instead of fading: the start (a, model
  // time sa s ago) read from where the air here was then, the newest (b, sb s from now) from where it will be, each
  // by its own wind (wa, wb: 3 bytes a point, 0.5 m/s steps about 128).  org, span: the grid's box (its corner and
  // size, m east, north, up).
  // A grid that moved between the two (a nest following its tornado: moved 1) has its start (a) in its old box (orgA,
  // spanA): read there by world position, the newest alone where the old box did not reach.  sa, sb come from the
  // blend's own fraction, which runs past 1 while the next volume is late (the newest carried on along its wind);
  // f, the share of the newest, stops at 1.
  const MIX_FS = `#version 300 es
  precision highp float; precision highp sampler3D;
  out vec4 frag;
  uniform sampler3D a, b, wa, wb; uniform float f, sa, sb; uniform int layer, warp, moved;
  uniform vec3 org, span, orgA, spanA;
  vec3 wind(sampler3D t, vec3 uvw) { return (texture(t, uvw).rgb * 255.0 - 128.0) * 0.5; }
  // how much the start (a) has to say at its point ua: all of it well inside its box, fading to nothing over its
  // outer part (its sides and top; the ground is its floor): 15 % for a nest that moved, 2 % for the box -- a cut
  // there showed as a hard straight line across the cloud wherever the storm had changed between the two volumes
  float inA(vec3 ua) {
    if (any(lessThan(ua, vec3(0.0))) || any(greaterThan(ua, vec3(1.0)))) return 0.0;
    vec2 e = min(ua.xy, 1.0 - ua.xy);
    return smoothstep(0.0, 1.0, clamp(min(min(e.x, e.y), 1.0 - ua.z) / (moved == 1 ? 0.15 : 0.02), 0.0, 1.0));
  }
  void main() {
    ivec3 c = ivec3(ivec2(gl_FragCoord.xy), layer);
    if (warp == 0 && moved == 0) { frag = mix(texelFetch(a, c, 0), texelFetch(b, c, 0), f); return; }
    vec3 uvw = (vec3(c) + 0.5) / vec3(textureSize(b, 0));
    vec3 pm = org + uvw * span;
    vec3 pa = pm, pb = pm;
    if (warp == 0) {
      vec3 ua0 = (pa - orgA) / spanA;
      frag = mix(texture(a, ua0), texture(b, uvw), mix(1.0, f, inA(ua0)));
      return;
    }
    pa = pm - wind(wa, (pm - orgA) / spanA) * sa; pb = pm + wind(wb, uvw) * sb;
    vec3 ua = (pa - orgA) / spanA;
    frag = mix(texture(a, ua), texture(b, (pb - org) / span), mix(1.0, f, inA(ua)));
  }`;
  // The detailed map (worldlink.js), drawn by Three.js into its own target, into the geometry buffer: its colour as
  // it is lit under a clear sky (the cloud pass then darkens it under the storm's shadow and its cloud above), its
  // distance along each ray in box units, stored negative to say so, and the depth the cars are tested against.
  const MERGE_FS = `#version 300 es
  precision highp float;
  in vec2 ndc;
  uniform sampler2D wCol, wDep; uniform mat4 invVP, VP; uniform vec3 eye;
  uniform float n, f, tanY, aspect, mToBox;
  layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1; layout(location = 2) out vec4 o2;
  void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(wDep, 0));
    float d = texture(wDep, uv).r;
    if (d >= 1.0) discard;
    float vz = 2.0 * n * f / ((f + n) - (2.0 * d - 1.0) * (f - n));     // m along the view axis
    vec2 q = uv * 2.0 - 1.0;
    float dist = vz * length(vec3(q.x * tanY * aspect, q.y * tanY, 1.0)) * mToBox;
    vec4 a = invVP * vec4(q, -1.0, 1.0), b = invVP * vec4(q, 1.0, 1.0);
    vec3 p = eye + normalize(b.xyz / b.w - a.xyz / a.w) * dist;
    vec4 c = VP * vec4(p, 1.0);
    gl_FragDepth = clamp((c.z / c.w) * 0.5 + 0.5, 0.0, 1.0);
    o0 = vec4(pow(max(texture(wCol, uv).rgb, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
    o1 = vec4(-dist, 0.0, 0.0, 1.0);
    o2 = vec4(0.0);
  }`;
  // the down channel of chase mode's light volume: the cloud between a point and the ground (for the light the
  // ground sends back up, bounceAt)
  // (and the rain's extinction as the cloud pass's, rainExt)
  function lightSource() {
    const end = 'frag = vec4(ts, tu, 0.0, 1.0);', sig = 'float sig0(vec3 p) {', lin = '(25.0 * m.x + 1.2 * rain * m.y)';
    let src = L().FS_LIGHT;
    if (src.includes(sig) && src.includes(lin)) {
      src = src.replace(sig, RAIN_GLSL + '\n  ' + sig).replace(lin, '(25.0 * m.x + rain * rainExt(m.y))');
    } else console.warn('chase: light volume rain hook not found');
    if (!src.includes(end)) { console.warn('chase: light volume hook not found; no ground bounce'); return src; }
    return src.replace(end, `float td = 0.0, dzb = p.z / 8.0;
    for (int j = 0; j < 8; j++) td += sig0(p - vec3(0.0, 0.0, (float(j) + 0.5) * dzb)) * dzb * toKm.z;
    frag = vec4(ts, tu, td, 1.0);`);
  }
  // the far cloud's tiers: the cloud and rain (their codes) and their light (to the sun, to the sky) in one texture
  function farLightSource() {
    const end = 'frag = vec4(ts, tu, td, 1.0);', src = lightSource();
    if (!src.includes(end)) throw new Error('chase: far cloud light hook not found');
    return src.replace(end, 'frag = vec4(texture(vol, uvw).rg, ts, tu);');
  }
  // a 3-D texture of this size and format (re-allocated only when they change), optionally filled with data
  function alloc3D(tex, ifmt, nx, ny, nz, fmt, type, data) {
    const k = `${ifmt}/${nx}/${ny}/${nz}`;
    gl.bindTexture(gl.TEXTURE_3D, tex);
    if (data || tex._k !== k) {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, ifmt, nx, ny, nz, 0, fmt, type, data || null);
      texParams(gl.TEXTURE_3D, gl.LINEAR, gl.CLAMP_TO_EDGE);
      tex._k = k;
    }
  }
  // dst = mix(a, b, f), layer by layer (all three the same size); w: { org, span (this grid's box, m), orgA, spanA
  // (the start's, when the grid moved), warp: carried along the wind (wa, wb), fw: the blend's own fraction (past 1
  // while the next volume is late) } -- the times from mixSt
  function mixInto(dst, a, b, f, nx, ny, nz, w) {
    gl.disable(gl.BLEND);
    gl.useProgram(mixProg); gl.bindVertexArray(vao);
    const u = mixProg.u;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, a); gl.uniform1i(u.a, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, b); gl.uniform1i(u.b, 1);
    gl.uniform1f(u.f, f);
    gl.uniform1i(u.warp, w && w.warp ? 1 : 0);
    gl.uniform1i(u.moved, w && w.moved ? 1 : 0);
    if (w) {
      gl.uniform3f(u.org, ...w.org); gl.uniform3f(u.span, ...w.span);
      gl.uniform3f(u.orgA, ...(w.orgA || w.org)); gl.uniform3f(u.spanA, ...(w.spanA || w.span));
    }
    if (w && w.warp) {
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_3D, w.wa); gl.uniform1i(u.wa, 2);
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_3D, w.wb); gl.uniform1i(u.wb, 3);
      const dtm = mixSt.tB - mixSt.tA, fw = w.fw === undefined ? f : w.fw;
      gl.uniform1f(u.sa, fw * dtm); gl.uniform1f(u.sb, (1 - fw) * dtm);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, mixFbo);
    gl.viewport(0, 0, nx, ny);
    for (let k = 0; k < nz; k++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, dst, 0, k);
      gl.uniform1i(u.layer, k);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  const sameBox = (p, q) => p && q && p.length === q.length && p.every((v, i) => Math.abs(v - q[i]) < 1e-6);
  const sameGrid = (p, q) => p && q && p.nx === q.nx && p.ny === q.ny && p.nz === q.nz;

  function init(canvas) {
    cv = canvas;
    gl = cv.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (!gl) return false;
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('this browser cannot render to float textures (EXT_color_buffer_float)');
    floatLinear = !!gl.getExtension('OES_texture_float_linear');
    cloudProg = program(L().VS, cloudSource());
    lightProg = program(L().VS, lightSource());
    try { farLightProg = program(L().VS, farLightSource()); } catch (e) { console.warn('chase far cloud:', e.message); }
    mixProg = program(L().VS, MIX_FS);
    mergeProg = program(L().VS, MERGE_FS);
    roadProg = program(ROAD_VS, ROAD_FS);
    meshProg = program(MESH_VS, MESH_FS);
    treeProg = program(TREE_VS, MESH_FS);
    try { precipProg = program(PRECIP_VS, PRECIP_FS); } catch (e) { console.warn('chase rain and snow:', e.message); }
    trees.vao = gl.createVertexArray(); trees.mesh = gl.createBuffer(); trees.inst = gl.createBuffer();
    const tm = treeMesh();
    gl.bindVertexArray(trees.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, trees.mesh); gl.bufferData(gl.ARRAY_BUFFER, tm.data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 28, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 28, 24);
    gl.bindBuffer(gl.ARRAY_BUFFER, trees.inst);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 24, 0); gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4); gl.vertexAttribPointer(4, 2, gl.FLOAT, false, 24, 16); gl.vertexAttribDivisor(4, 1);
    gl.bindVertexArray(null);
    trees.near = tm.near; trees.far = tm.far;
    blds.vao = gl.createVertexArray(); blds.buf = gl.createBuffer();
    vao = gl.createVertexArray();
    volTex = gl.createTexture(); gndTex = gl.createTexture(); lightTex = gl.createTexture(); lightFbo = gl.createFramebuffer();
    volA = gl.createTexture(); volB = gl.createTexture(); lightA = gl.createTexture(); lightB = gl.createTexture();
    mixFbo = gl.createFramebuffer();
    for (let i = 0; i < 4; i++) { insTex.push(gl.createTexture()); insA.push(gl.createTexture()); insB.push(gl.createTexture()); }
    windA = gl.createTexture(); windB = gl.createTexture();
    for (let i = 0; i < 4; i++) { wInsA.push(gl.createTexture()); wInsB.push(gl.createTexture()); }
    noiseTex = L().makeNoise(gl);
    gb.fbo = gl.createFramebuffer(); gb.col = gl.createTexture(); gb.dist = gl.createTexture(); gb.depth = gl.createRenderbuffer();
    gb.light = gl.createTexture();
    lampProg = program(LAMP_VS, LAMP_FS);
    boltProg = program(BOLT_VS, BOLT_FS);
    bolts.vao = gl.createVertexArray(); bolts.buf = gl.createBuffer();
    gl.bindVertexArray(bolts.vao); gl.bindBuffer(gl.ARRAY_BUFFER, bolts.buf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 20, 16);
    gl.bindVertexArray(null);
    lamps.vao = gl.createVertexArray(); lamps.buf = gl.createBuffer();
    gl.bindVertexArray(lamps.vao); gl.bindBuffer(gl.ARRAY_BUFFER, lamps.buf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 16); gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);
    roads.buf = gl.createBuffer(); roads.vao = gl.createVertexArray();
    Land.init(gl, () => {});
    Land.setTrees(false);                          // no trees in chase mode (yet)
    return true;
  }

  // ---------------------------------------------------------------- the volume
  // blendMs > 0: blend from what is shown now to this volume over that many ms (when it is the same box on the same
  // grid; otherwise, and with 0, it replaces it at once)
  function setVolume(b, buf, blendMs = 0) {
    const { header, data } = unpack(buf);
    header.insets = (header.insets || []).slice(0, 4);
    const two = header.channels === 2, ifmt = two ? gl.RG8 : gl.R8, fmt = two ? gl.RG : gl.RED;
    const blend = blendMs > 0 && !!head && sameBox(box, b) && sameGrid(head, header) && head.channels === header.channels;
    const oldIns = head ? head.insets || [] : [];
    // the model time shown now (a blend under way: part way between its two) is the new blend's start
    const tNow = shownT(), tNew = Number.isFinite(header.t) ? header.t : null, hadWind = hasWind;
    if (blend) {
      // the picture on the screen now becomes the blend's start, carried by the wind the newest volume had
      [volA, volTex] = [volTex, volA];
      [lightA, lightTex] = [lightTex, lightA];
      for (let i = 0; i < 4; i++) [insA[i], insTex[i]] = [insTex[i], insA[i]];
      [windA, windB] = [windB, windA];
      for (let i = 0; i < 4; i++) [wInsA[i], wInsB[i]] = [wInsB[i], wInsA[i]];
    }
    head = header; box = b; loadSeq++; volData = data;
    alloc3D(volB, ifmt, header.nx, header.ny, header.nz, fmt, gl.UNSIGNED_BYTE, data);
    alloc3D(volA, ifmt, header.nx, header.ny, header.nz, fmt, gl.UNSIGNED_BYTE, null);
    alloc3D(volTex, ifmt, header.nx, header.ny, header.nz, fmt, gl.UNSIGNED_BYTE, null);
    for (const t of [lightA, lightB, lightTex]) alloc3D(t, gl.RGBA16F, header.nx, header.ny, header.nz, gl.RGBA, gl.HALF_FLOAT, null);
    hasWind = !!(header.wind && header.wind.dims);
    if (tNew !== null && (mixSt.tB === null || !blend)) flowEpoch = flowEpoch || tNew;
    const dtm = tNew !== null && tNow !== null ? tNew - tNow : 0;
    const warp = blend && hasWind && hadWind && dtm > 0 && dtm < WARP_MAX;
    Object.assign(mixSt, { t0: performance.now(), dur: blend ? blendMs : 0, f: blend ? 0 : 1, dirty: true, lightB: -1,
                           tA: blend && tNow !== null ? tNow : tNew, tB: tNew, warp });
    gMap = null;
    if (header.ground) {
      const n = header.nx * header.ny, off = data.byteOffset + n * header.nz * header.channels;
      const g16 = new Uint16Array(data.buffer.slice(off, off + 2 * n)), gf = new Float32Array(n);
      for (let i = 0; i < n; i++) gf[i] = g16[i] / header.ztop;
      gMap = { nx: header.nx, ny: header.ny, g16 };
      gl.bindTexture(gl.TEXTURE_2D, gndTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, header.nx, header.ny, 0, gl.RED, gl.FLOAT, gf);
      texParams(gl.TEXTURE_2D, floatLinear ? gl.LINEAR : gl.NEAREST, gl.CLAMP_TO_EDGE);
    }
    let off = data.byteOffset + header.nx * header.ny * header.nz * header.channels + (header.ground ? 2 * header.nx * header.ny : 0);
    header.insets.forEach((ins, i) => {
      const n = ins.nx * ins.ny * ins.nz * header.channels;
      // a nest blends from where it was (a tornado's nest follows it: its box moves every volume), on the same grid
      // and levels, unless it jumped more than half its width
      const o = oldIns[i];
      mixSt.insBlend[i] = blend && sameGrid(o, ins) && o.ztop === ins.ztop &&
        Math.hypot(o.box[0] - ins.box[0], o.box[1] - ins.box[1]) < 0.5 * (ins.box[2] - ins.box[0]);
      mixSt.insBoxA[i] = mixSt.insBlend[i] && !sameBox(o.box, ins.box) ? { box: o.box, ztop: o.ztop } : null;
      alloc3D(insB[i], ifmt, ins.nx, ins.ny, ins.nz, fmt, gl.UNSIGNED_BYTE, new Uint8Array(data.buffer, off, n));
      alloc3D(insA[i], ifmt, ins.nx, ins.ny, ins.nz, fmt, gl.UNSIGNED_BYTE, null);
      alloc3D(insTex[i], ifmt, ins.nx, ins.ny, ins.nz, fmt, gl.UNSIGNED_BYTE, null);
      off += n;
    });
    // then the wind: the box's, each nest's (RGB bytes at every second point)
    if (hasWind) {
      const put = (tex, d) => { const n = d[0] * d[1] * d[2] * 3; alloc3D(tex, gl.RGB8, d[0], d[1], d[2], gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array(data.buffer, off, n)); off += n; };
      put(windB, header.wind.dims);
      header.insets.forEach((ins, i) => {
        const prev = oldIns[i];
        if (ins.wind) put(wInsB[i], ins.wind);
        mixSt.insWarp[i] = warp && mixSt.insBlend[i] && !!ins.wind && !!(prev && prev.wind) && prev.wind.join() === ins.wind.join();
      });
      mixSt.warp = warp && windA._k === windB._k;     // the same wind grid at both ends
    } else mixSt.insWarp.fill(false);
    return header;
  }

  // the far cloud's tier i (0 the nearer, 1 the wider): a volume from volume.py extract (far=1), blended into over
  // blendMs from the one shown (on the same grid; it may have moved); buf null: none
  function setFar(i, buf, blendMs = 0) {
    const s = farV[i];
    if (!gl || !s || !farLightProg) return;
    if (!buf) { s.head = null; return; }
    const { header, data } = unpack(buf);
    if (header.channels !== 2) return;
    if (!s.tex) for (const k of ['tex', 'A', 'B', 'raw']) s[k] = gl.createTexture();
    const blend = blendMs > 0 && !!s.head && sameGrid(s.head, header) && s.head.ztop === header.ztop;
    if (blend) [s.A, s.tex] = [s.tex, s.A];
    s.from = blend && !sameBox(s.head.box, header.box) ? { box: s.head.box, ztop: s.head.ztop } : null;
    const { nx, ny, nz } = header;
    alloc3D(s.raw, gl.RG8, nx, ny, nz, gl.RG, gl.UNSIGNED_BYTE, data.subarray(0, nx * ny * nz * 2));
    for (const t of [s.A, s.B, s.tex]) alloc3D(t, gl.RGBA16F, nx, ny, nz, gl.RGBA, gl.HALF_FLOAT, null);
    Object.assign(s, { head: header, key: '', t0: performance.now(), dur: blend ? blendMs : 0, f: blend ? 0 : 1, dirty: true });
  }
  // a tier's own units (as geometry()'s for the box: its longer side 2) for its light pass
  function farGeom(h) {
    const sx = (h.box[2] - h.box[0]) / 1000, sy = (h.box[3] - h.box[1]) / 1000, k = 2 / Math.max(sx, sy);
    return { sx: sx * k, sy: sy * k, sz: h.ztop / 1000 * k, k };
  }
  function advanceFar(sun, dens, rain) {
    const now = performance.now();
    let lit = false;
    for (const s of farV) {
      if (!s.head) continue;
      // the newest lit when it comes and when the sun (to about half a degree; at most every FAR_RELIGHT ms as it
      // moves with the model's clock) or the sliders change -- one tier a frame, so two never land in one frame;
      // the start of a blend keeps the light it had
      const key = [...sun.map(v => v.toFixed(2)), dens, rain].join('/'), slidKey = `${dens}/${rain}`;
      const due = s.key === '' || !s.key.endsWith(slidKey) || now - s.litAt > FAR_RELIGHT;
      if (key !== s.key && due && !lit) {
        lightInto(s.B, s.raw, farGeom(s.head), sun, dens, rain, s.head, farLightProg);
        s.key = key; s.litAt = now; s.dirty = true; lit = true;
      }
      if (s.key === '') continue;                // its newest not lit yet (the other tier's turn): next frame
      const f = s.dur > 0 ? Math.min(1, (now - s.t0) / s.dur) : 1;
      if (!s.dirty && (f === s.f || now - s.at < FAR_MIX_EVERY)) continue;
      const h = s.head, w = s.from ? warpOf(null, null, h.box, h.ztop, s.from, false, f) : null;
      mixInto(s.tex, s.A, s.B, f, h.nx, h.ny, h.nz, w);
      Object.assign(s, { f, at: now, dirty: false });
    }
  }

  // the model time of the picture shown now (part way through a blend: between its two volumes), or null
  function shownT() {
    if (mixSt.tB === null) return null;
    if (mixSt.tA === null) return mixSt.tB;
    return mixSt.tA + blendFrac(performance.now()) * (mixSt.tB - mixSt.tA);
  }
  // the blend's own fraction now: 0 .. 1, on past 1 (to OVERRUN) when carried along the wind and the next is late
  function blendFrac(now) {
    if (!(mixSt.dur > 0)) return 1;
    return Math.min(mixSt.warp ? OVERRUN : 1, (now - mixSt.t0) / mixSt.dur);
  }
  // what the blend of one grid needs to carry it along the wind (MIX_FS): its wind at both ends and its box (m)
  function warpOf(wa, wb, bx, ztop, from, warp, fw) {
    const w = { wa, wb, warp, fw, org: [bx[0], bx[1], 0], span: [bx[2] - bx[0], bx[3] - bx[1], ztop] };
    if (from) { w.moved = true; w.orgA = [from.box[0], from.box[1], 0]; w.spanA = [from.box[2] - from.box[0], from.box[3] - from.box[1], from.ztop]; }
    return w;
  }

  // the blend now: the volume (and its nests) and their light mixed for this moment, at most every MIX_EVERY ms
  function advance(G, sun, dens, rain) {
    const now = performance.now();
    // fw: the blend's own fraction (on past 1, carried along the wind, while the next volume is late); f: the newest's share
    const fw = blendFrac(now), f = Math.min(1, fw);
    const lit = buildLight(G, sun, dens, rain, f < 1);
    // A blend pass over everything costs ~12 ms even on a fast GPU with a tornado's nests in the box (some ten
    // million points, carried along the wind), far too much 30 times a second -- a laptop fell to 20 fps.  So the
    // box (500 m cells: its cloud moves a few metres between passes) at most every MIX_MAIN ms, the nests (the
    // tornado: faster, finer) every MIX_INS ms, never both in one frame; everything at once when a volume comes in
    // or the light changed.
    const all = mixSt.dirty || lit;
    const mainDue = all || (fw !== mixSt.fMain && now - mixSt.atMain >= MIX_MAIN);
    const insDue = all || (!mainDue && fw !== mixSt.fIns && now - mixSt.atIns >= MIX_INS);
    if (!mainDue && !insDue) return;
    if (mainDue) {
      const { nx, ny, nz } = head;
      // carried along the wind while blending, and on along the newest's wind past the end until the next comes in
      const w = mixSt.warp ? warpOf(windA, windB, head.box, head.ztop, null, true, fw) : null;
      mixInto(volTex, volA, volB, f, nx, ny, nz, w);
      mixInto(lightTex, lightA, lightB, f, nx, ny, nz, w);
      mixSt.fMain = fw; mixSt.atMain = now;
    }
    if (insDue) {
      (head.insets || []).forEach((ins, i) => {
        const fi = mixSt.insBlend[i] ? f : 1, from = mixSt.insBlend[i] ? mixSt.insBoxA[i] : null;
        const wi = mixSt.insWarp[i] ? warpOf(wInsA[i], wInsB[i], ins.box, ins.ztop, from, true, mixSt.insBlend[i] ? fw : 1)
          : from ? warpOf(null, null, ins.box, ins.ztop, from, false, fi) : null;
        mixInto(insTex[i], insA[i], insB[i], fi, ins.nx, ins.ny, ins.nz, wi);
      });
      mixSt.fIns = fw; mixSt.atIns = now;
    }
    mixSt.f = fw; mixSt.at = now; mixSt.dirty = false;
  }

  // the ground's height (m above the model's zero) at world x, y m: bilinear, the same as the cloud pass reads it
  function terrain(x, y) {
    if (!gMap || !box) return 0;
    const fx = ((x / 1000 - box[0]) / (box[2] - box[0])) * gMap.nx - 0.5, fy = ((y / 1000 - box[1]) / (box[3] - box[1])) * gMap.ny - 0.5;
    const nx = gMap.nx, ny = gMap.ny;
    if (!floatLinear) {
      const i = Math.round(fx), j = Math.round(fy);
      return i < 0 || j < 0 || i >= nx || j >= ny ? 0 : gMap.g16[j * nx + i];
    }
    const i0 = Math.max(0, Math.min(nx - 1, Math.floor(fx))), j0 = Math.max(0, Math.min(ny - 1, Math.floor(fy)));
    const i1 = Math.min(nx - 1, i0 + 1), j1 = Math.min(ny - 1, j0 + 1);
    const tx = Math.max(0, Math.min(1, fx - i0)), ty = Math.max(0, Math.min(1, fy - j0)), g = gMap.g16;
    return (g[j0 * nx + i0] * (1 - tx) + g[j0 * nx + i1] * tx) * (1 - ty) + (g[j1 * nx + i0] * (1 - tx) + g[j1 * nx + i1] * tx) * ty;
  }

  // the precipitation (g/kg) a few hundred metres up over world x, y m: the rain reaching the ground there
  function sample(x, y) {
    if (!volData || !head || head.channels !== 2) return 0;
    const i = Math.floor((x / 1000 - box[0]) / (box[2] - box[0]) * head.nx), j = Math.floor((y / 1000 - box[1]) / (box[3] - box[1]) * head.ny);
    if (i < 0 || j < 0 || i >= head.nx || j >= head.ny) return 0;
    let m = 0;
    for (const k of [1, 2]) {
      const c = volData[((k * head.ny + j) * head.nx + i) * 2 + 1];
      if (c) m = Math.max(m, Math.pow(10, (c - 1) / 40 - 3));
    }
    return m;
  }

  function cloudBase(x, y) {
    if (!volData || !head || head.channels !== 2) return null;
    const ci = Math.floor((x / 1000 - box[0]) / (box[2] - box[0]) * head.nx), cj = Math.floor((y / 1000 - box[1]) / (box[3] - box[1]) * head.ny);
    const dz = head.ztop / head.nz, c0 = Math.round((Math.log10(0.08) + 3) * 40 + 1);   // 0.08 g/kg
    let best = null;
    for (let j = cj - 2; j <= cj + 2; j++) for (let i = ci - 2; i <= ci + 2; i++) {
      if (i < 0 || j < 0 || i >= head.nx || j >= head.ny) continue;
      for (let k = 0; k < head.nz; k++) {
        if (volData[((k * head.ny + j) * head.nx + i) * 2] >= c0) { const z = (k + 0.5) * dz; if (best === null || z < best) best = z; break; }
      }
    }
    return best === null ? null : best - terrain(x, y);
  }

  function geometry() {
    const sx = box[2] - box[0], sy = box[3] - box[1], k = 2 / Math.max(sx, sy);
    return { sx: sx * k, sy: sy * k, sz: (head ? head.ztop / 1000 : 18) * k, k, ex: 1 };
  }
  // world m -> the box's units
  function toBox(G, p) {
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    return [(p[0] / 1000 - cx) * G.k, (p[1] / 1000 - cy) * G.k, p[2] / 1000 * G.k];
  }

  // the light volumes: the newest volume's (lightB) when it comes in, and both ends of a blend under way when the sun
  // or the sliders change.  True when one was built (the blend's mix then runs again).
  function buildLight(G, sun, dens, rain, blending) {
    const key = [head.nx, head.ny, head.nz, ...sun.map(v => v.toFixed(3)), dens, rain, G.sx, G.sy, G.sz].join('/');
    if (key !== lightKey) {
      lightInto(lightB, volB, G, sun, dens, rain);
      if (blending) lightInto(lightA, volA, G, sun, dens, rain);
      lightKey = key; mixSt.lightB = loadSeq;
      return true;
    }
    if (mixSt.lightB !== loadSeq) {
      lightInto(lightB, volB, G, sun, dens, rain);
      mixSt.lightB = loadSeq;
      return true;
    }
    return false;
  }
  // dims: the grid (the box's); prog: lightProg, or farLightProg for a far tier
  function lightInto(dst, src, G, sun, dens, rain, dims = head, prog = lightProg) {
    const { nx, ny, nz } = dims;
    alloc3D(dst, gl.RGBA16F, nx, ny, nz, gl.RGBA, gl.HALF_FLOAT, null);
    gl.disable(gl.BLEND);
    gl.useProgram(prog); gl.bindVertexArray(vao);
    const u = prog.u;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, src); gl.uniform1i(u.vol, 0);
    gl.uniform3f(u.size, G.sx, G.sy, G.sz);
    gl.uniform3f(u.toKm, 1 / G.k, 1 / G.k, 1 / G.k);
    gl.uniform3f(u.sunDir, ...sun);
    gl.uniform3f(u.lres, nx, ny, nz);
    gl.uniform1f(u.density, dens); gl.uniform1f(u.rain, rain);
    gl.bindFramebuffer(gl.FRAMEBUFFER, lightFbo);
    gl.viewport(0, 0, nx, ny);
    for (let k = 0; k < nz; k++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, dst, 0, k);
      gl.uniform1f(u.layer, k);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ---------------------------------------------------------------- roads and cars
  function setRoads(verts, origin) {
    gl.bindVertexArray(roads.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, roads.buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 28, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 28, 20);
    gl.bindVertexArray(null);
    roads.n = verts.length / 7; roads.origin = origin;
  }

  // the trees' instances (6 floats each), the first nearCount of them close enough for the detailed tree
  function setTrees(inst, n, nearCount, origin) {
    gl.bindBuffer(gl.ARRAY_BUFFER, trees.inst);
    gl.bufferData(gl.ARRAY_BUFFER, inst, gl.STATIC_DRAW);
    trees.count = n; trees.nearCount = nearCount; trees.origin = origin;
  }
  // the lamps' pools (lights.js): per lamp x, y, z (m from origin), radius m, r, g, b, intensity
  function setLamps(inst, origin) {
    gl.bindBuffer(gl.ARRAY_BUFFER, lamps.buf);
    gl.bufferData(gl.ARRAY_BUFFER, inst, gl.DYNAMIC_DRAW);
    lamps.n = inst.length / 8; lamps.origin = origin;
  }
  function setBuildings(mesh, origin) {
    gl.bindVertexArray(blds.vao); gl.bindBuffer(gl.ARRAY_BUFFER, blds.buf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24);
    gl.bindVertexArray(null);
    blds.n = mesh.length / 9; blds.origin = origin;
  }

  // a part's buffers let go (stations.js: a station or the route's ribbon rebuilt or out of range)
  function dropPart(data) {
    const b = partBufs.get(data);
    if (!b) return;
    gl.deleteVertexArray(b.vao); if (b.buf) gl.deleteBuffer(b.buf);
    partBufs.delete(data);
  }
  function meshBuffer(arr) {
    const v = gl.createVertexArray(), b = gl.createBuffer();
    gl.bindVertexArray(v); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24);
    gl.bindVertexArray(null);
    return { vao: v, buf: b, n: arr.length / 9 };
  }
  function carMeshes(kind, color) {
    // (built again once the car's model has come in: carmesh.js)
    const key = `${kind}/${color.join(',')}/${CarMesh.ready && CarMesh.ready(kind) ? 'model' : 'code'}`;
    if (!meshes.has(key)) {
      const m = CarMesh.build(CarPhys.SPECS[kind], kind, color);
      meshes.set(key, { body: meshBuffer(m.body), wheel: meshBuffer(m.wheel), wheelVis: m.wheelVis || null });
    }
    return meshes.get(key);
  }

  function targets(w, h) {
    if (gb.w === w && gb.h === h) return;
    gb.w = w; gb.h = h;
    for (const [t, fmt, f, ty] of [[gb.col, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE], [gb.dist, gl.R32F, gl.RED, gl.FLOAT], [gb.light, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE]]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, f, ty, null);
      texParams(gl.TEXTURE_2D, gl.NEAREST, gl.CLAMP_TO_EDGE);
    }
    gl.bindRenderbuffer(gl.RENDERBUFFER, gb.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, gb.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gb.col, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, gb.dist, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, gb.light, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, gb.depth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // rotation (column-major mat3) from the body's axes M (rows: body x, y, z in world), times a local one
  const mat3 = (M) => new Float32Array([M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2]]);
  function wheelMat(M, steer, spin) {
    // body: x fwd, y left, z up; the wheel turns about z by steer, then spins about its axle (y)
    const cs = Math.cos(steer), ss = Math.sin(steer), cp = Math.cos(spin), sp = Math.sin(spin);
    const f = [M[0][0] * cs + M[1][0] * ss, M[0][1] * cs + M[1][1] * ss, M[0][2] * cs + M[1][2] * ss];
    const l = [-M[0][0] * ss + M[1][0] * cs, -M[0][1] * ss + M[1][1] * cs, -M[0][2] * ss + M[1][2] * cs];
    const u = M[2];
    const x = [f[0] * cp - u[0] * sp, f[1] * cp - u[1] * sp, f[2] * cp - u[2] * sp];
    const z = [f[0] * sp + u[0] * cp, f[1] * sp + u[1] * cp, f[2] * sp + u[2] * cp];
    return new Float32Array([...x, ...l, ...z]);
  }

  // where a wheel is drawn: the physics' wheel centre, or with a model car (vis: its wheels' places along and across
  // the body, m from the centre of gravity) there, at the physics' suspension height
  function wheelAt(c, w, vis) {
    if (!vis) return w.c;
    const M = c.M, d = [w.c[0] - c.p[0], w.c[1] - c.p[1], w.c[2] - c.p[2]];
    const bx = d[0] * M[0][0] + d[1] * M[0][1] + d[2] * M[0][2], by = d[0] * M[1][0] + d[1] * M[1][1] + d[2] * M[1][2];
    const bz = d[0] * M[2][0] + d[1] * M[2][1] + d[2] * M[2][2];
    const x = bx > 0 ? vis.xf : vis.xr, y = by > 0 ? vis.half : -vis.half;
    return [0, 1, 2].map(i => c.p[i] + M[0][i] * x + M[1][i] * y + M[2][i] * bz);
  }

  // the detailed map's picture (wimg: worldlink.js drawer's) into the geometry buffer, under the cars
  function mergeWorld(wimg, G, VP, eye, fov) {
    gl.useProgram(mergeProg); gl.bindVertexArray(vao);
    const u = mergeProg.u;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, wimg.color); gl.uniform1i(u.wCol, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, wimg.depth); gl.uniform1i(u.wDep, 1);
    gl.uniformMatrix4fv(u.invVP, false, L().invert(VP)); gl.uniformMatrix4fv(u.VP, false, VP);
    gl.uniform3f(u.eye, ...eye);
    gl.uniform1f(u.n, wimg.near); gl.uniform1f(u.f, wimg.far);
    gl.uniform1f(u.tanY, Math.tan(fov / 2)); gl.uniform1f(u.aspect, gb.w / gb.h);
    gl.uniform1f(u.mToBox, (wimg.scale || 1) * G.k / 1000);
    gl.depthFunc(gl.ALWAYS);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.depthFunc(gl.LESS);
  }

  function drawGeometry(G, VP, eye, sun, cars, wet, showTrees, wimg, fov, tw, lampK) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, gb.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    gl.viewport(0, 0, gb.w, gb.h);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]); gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]); gl.clearBufferfv(gl.COLOR, 2, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.DEPTH, 0, [1]);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.depthMask(true);
    const s = G.k / 1000;
    // the detailed map in place of chase mode's own roads, trees and buildings
    if (wimg && mergeProg) { mergeWorld(wimg, G, VP, eye, fov); showTrees = false; }
    if (roads.n && !wimg) {
      gl.useProgram(roadProg);
      const u = roadProg.u;
      gl.uniformMatrix4fv(u.VP, false, VP);
      gl.uniform3f(u.org, ...toBox(G, [roads.origin[0], roads.origin[1], 0]));
      gl.uniform1f(u.s, s);
      gl.uniform3f(u.eye, ...eye); gl.uniform3f(u.sunDir, ...sun); gl.uniform1f(u.wet, wet);
      gl.bindVertexArray(roads.vao);
      gl.drawArrays(gl.TRIANGLES, 0, roads.n);
    }
    if (trees.count && showTrees) {
      gl.useProgram(treeProg);
      const t = treeProg.u;
      gl.uniformMatrix4fv(t.VP, false, VP);
      gl.uniform3f(t.org, ...toBox(G, [trees.origin[0], trees.origin[1], 0]));
      gl.uniform1f(t.s, s); gl.uniform3f(t.eye, ...eye); gl.uniform3f(t.sunDir, ...sun);
      gl.uniform3f(t.uWind, tw ? tw.u : 0, tw ? tw.v : 0, tw ? tw.gust : 1);
      gl.uniform1f(t.uTime, performance.now() / 1000 % 3600);
      gl.bindVertexArray(trees.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, trees.inst);
      gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 24, 0); gl.vertexAttribPointer(4, 2, gl.FLOAT, false, 24, 16);
      if (trees.nearCount) gl.drawArraysInstanced(gl.TRIANGLES, 0, trees.near, trees.nearCount);
      const nf = trees.count - trees.nearCount;
      if (nf > 0) {
        const o = trees.nearCount * 24;
        gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 24, o); gl.vertexAttribPointer(4, 2, gl.FLOAT, false, 24, o + 16);
        gl.drawArraysInstanced(gl.TRIANGLES, trees.near, trees.far, nf);
      }
    }
    gl.useProgram(meshProg);
    const u = meshProg.u;
    gl.uniformMatrix4fv(u.VP, false, VP);
    gl.uniform3f(u.eye, ...eye); gl.uniform3f(u.sunDir, ...sun); gl.uniform1f(u.s, s);
    if (blds.n && !wimg) {
      gl.uniformMatrix3fv(u.R, false, new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
      gl.uniform3f(u.org, ...toBox(G, [blds.origin[0], blds.origin[1], 0]));
      gl.bindVertexArray(blds.vao); gl.drawArrays(gl.TRIANGLES, 0, blds.n);
    }
    gl.enable(gl.CULL_FACE);
    for (const c of cars) {
      if (c.hidden) continue;
      if (c.parts) {
        // a figure (walker.js): each part its own mesh, turned and placed
        for (const pt of c.parts) {
          let b = partBufs.get(pt.data);
          if (!b) { b = meshBuffer(pt.data); partBufs.set(pt.data, b); }
          gl.uniformMatrix3fv(u.R, false, mat3(pt.M));
          gl.uniform3f(u.org, ...toBox(G, pt.p));
          gl.bindVertexArray(b.vao); gl.drawArrays(gl.TRIANGLES, 0, b.n);
        }
        continue;
      }
      const m = carMeshes(c.kind, c.color);
      gl.uniformMatrix3fv(u.R, false, mat3(c.M));
      gl.uniform3f(u.org, ...toBox(G, c.p));
      gl.bindVertexArray(m.body.vao); gl.drawArrays(gl.TRIANGLES, 0, m.body.n);
      gl.bindVertexArray(m.wheel.vao);
      for (const w of c.wheels) {
        gl.uniformMatrix3fv(u.R, false, wheelMat(c.M, w.steer, -w.spin));
        gl.uniform3f(u.org, ...toBox(G, wheelAt(c, w, m.wheelVis)));
        gl.drawArrays(gl.TRIANGLES, 0, m.wheel.n);
      }
    }
    gl.disable(gl.CULL_FACE);
    // the lamps' pools of light, at night, added into the light target over what is drawn
    if (lamps.n && lampK > 0.01) {
      gl.drawBuffers([gl.NONE, gl.NONE, gl.COLOR_ATTACHMENT2]);
      gl.useProgram(lampProg);
      gl.uniformMatrix4fv(lampProg.u.VP, false, VP);
      gl.uniform3f(lampProg.u.org, ...toBox(G, [lamps.origin[0], lamps.origin[1], 0]));
      gl.uniform1f(lampProg.u.s, s);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.depthMask(false); gl.depthFunc(gl.LEQUAL);
      gl.bindVertexArray(lamps.vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, lamps.n);
      gl.disable(gl.BLEND); gl.depthMask(true); gl.depthFunc(gl.LESS);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    }
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // the box's map picture for a non-plain ground (fetched once per box and map)
  const bmap = { key: '', tex: null, ok: false, busy: false };
  function boxMap(base) {
    const src = base.startsWith('sat') ? 'sat' : 'osm', labels = base === 'satlabels' ? 1 : 0;
    const key = `${box.join(',')}/${src}/${labels}`;
    if (key !== bmap.key && !bmap.busy) {
      bmap.busy = true; bmap.key = key; bmap.ok = false;
      fetch(`/api/volume/ground?x0=${box[0]}&y0=${box[1]}&x1=${box[2]}&y1=${box[3]}&src=${src}&labels=${labels}&px=1536`,
            { headers: window.ChaseAuth ? window.ChaseAuth() : {} })
        .then(r => (r.ok ? r.blob() : Promise.reject(new Error(r.status)))).then(createImageBitmap).then(bmp => {
          if (!bmap.tex) bmap.tex = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, bmap.tex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.generateMipmap(gl.TEXTURE_2D);
          texParams(gl.TEXTURE_2D, gl.LINEAR, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
          bmap.ok = true;
        }).catch(e => console.warn('chase ground map:', e)).finally(() => { bmap.busy = false; });
    }
    return bmap.ok && bmap.key === key ? bmap.tex : Land.coarse();
  }

  // ---------------------------------------------------------------- a frame
  // view: { eye, target, up (world m / unit), fov, sun [x, y, z], scale, density, rain, wet, quality, blend }
  function frame(view, cars) {
    if (!gl) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2), W = Math.max(1, Math.round(cv.clientWidth * dpr * view.scale));
    const H = Math.max(1, Math.round(cv.clientHeight * dpr * view.scale));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    if (!head || !box) { gl.viewport(0, 0, W, H); gl.clearColor(0.4, 0.45, 0.52, 1); gl.clear(gl.COLOR_BUFFER_BIT); return; }
    const G = geometry();
    const eye = toBox(G, view.eye), tgt = toBox(G, view.target);
    const V = L().lookAt(eye, tgt, view.up);
    const near = 0.4 * G.k / 1000, far = 40 * G.k;
    const VPr = L().mul(L().perspective(view.fov, W / H, near, far), V);
    const VPc = L().mul(L().perspective(view.fov, W / H, 0.001, 50), V);
    // the detailed map (worldlink.js), drawn first by Three.js in this same context (it leaves the state reset)
    let wimg = null;
    if (view.world) {
      try { wimg = view.world(W, H); } catch (e) { console.warn('chase: detailed map', e); wimg = null; }
    }
    targets(W, H);
    const nt = view.night || { dayK: 1, expo: 1, moonDir: [0, 0, -1], moonK: 0, lampK: 0 };
    drawGeometry(G, VPr, eye, view.sun, cars, view.wet, view.trees !== false, wimg, view.fov, view.treeWind, nt.lampK);
    advance(G, view.sun, view.density, view.rain);
    if (view.far) advanceFar(view.sun, view.density, view.rain);
    gl.viewport(0, 0, W, H);
    if (view.blend > 0) {
      gl.enable(gl.BLEND); gl.blendColor(0, 0, 0, 1 - view.blend);
      gl.blendFunc(gl.CONSTANT_ALPHA, gl.ONE_MINUS_CONSTANT_ALPHA);
    } else gl.disable(gl.BLEND);
    const p = cloudProg, u = p.u;
    gl.useProgram(p); gl.bindVertexArray(vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, volTex); gl.uniform1i(u.vol, 0);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, gndTex); gl.uniform1i(u.gnd, 2);
    gl.uniform1i(u.hasG, head.ground ? 1 : 0);
    // the whole box's ground: the land cover painted, or the map picked (tiles.py ground(), like the 3-D view)
    const coarse = view.base === 'plain' ? Land.coarse() : boxMap(view.base);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, coarse); gl.uniform1i(u.mapTex, 3);
    gl.uniform1i(u.hasMap, coarse ? 1 : 0); gl.uniform1i(u.mapDark, view.base === 'osmdark' ? 1 : 0);
    const d = Land.detail();
    gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, d ? d.tex : null); gl.uniform1i(u.mapTex2, 8);
    gl.uniform1i(u.hasMap2, d ? 1 : 0);
    if (d) {
      const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
      gl.uniform4f(u.map2Box, (d.box[0] - cx) * G.k, (d.box[1] - cy) * G.k, (d.box[2] - cx) * G.k, (d.box[3] - cy) * G.k);
    }
    gl.uniformMatrix4fv(u.invVP, false, L().invert(VPc));
    gl.uniform3f(u.eye, ...eye);
    gl.uniform3f(u.size, G.sx, G.sy, G.sz);
    gl.uniform3f(u.toKm, 1 / G.k, 1 / G.k, 1 / G.k);
    gl.uniform3f(u.sunDir, ...view.sun);
    gl.uniform1f(u.density, view.density); gl.uniform1f(u.rain, view.rain);
    gl.uniform1f(u.relief, view.relief);
    gl.uniform1f(u.cloudContrast, view.contrast || 1);
    gl.uniform1i(u.geoWins, wimg ? 1 : 0);
    // the storm's drift over the model's time (the detail carried with it)
    const ft = shownT();
    gl.uniform1f(u.flowT, ft === null ? 0 : ft - flowEpoch);
    const dr = (head && head.drift) || [0, 0];
    gl.uniform2f(u.drift, dr[0], dr[1]);
    gl.uniform1f(u.bounce, view.bounce === undefined ? 1 : view.bounce);
    if (u.nestStep) gl.uniform1f(u.nestStep, view.nestStep === undefined ? 1 : view.nestStep);
    gl.uniform3f(u.texel, 1 / head.nx, 1 / head.ny, 1 / head.nz);
    // the nests
    const ins = head.insets || [], cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2;
    const B = new Float32Array(16), T = new Float32Array(4).fill(1);
    ins.forEach((s, i) => {
      B.set([(s.box[0] / 1000 - cx) * G.k, (s.box[1] / 1000 - cy) * G.k, (s.box[2] / 1000 - cx) * G.k, (s.box[3] / 1000 - cy) * G.k], 4 * i);
      T[i] = s.ztop / 1000 * G.k;
    });
    for (let i = 0; i < 4; i++) { gl.activeTexture(gl.TEXTURE9 + i); gl.bindTexture(gl.TEXTURE_3D, i < ins.length ? insTex[i] : null); gl.uniform1i(u['ins' + i], 9 + i); }
    gl.uniform1i(u.insN, ins.length); gl.uniform4fv(u.insBox, B); gl.uniform1fv(u.insTop, T);
    // the far cloud's tiers (units 1 and 15: the cloud pass has nothing else there)
    if (u.farHas) {
      const FB = new Float32Array(8), FT = [1, 1], FC = [1, 1], has = [0, 0];
      farV.forEach((s, i) => {
        const on = !!(view.far && s.head);
        gl.activeTexture(i ? gl.TEXTURE15 : gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, on ? s.tex : null);
        gl.uniform1i(u['farT' + i], i ? 15 : 1);
        if (!on) return;
        const fb = s.head.box;
        FB.set([(fb[0] / 1000 - cx) * G.k, (fb[1] / 1000 - cy) * G.k, (fb[2] / 1000 - cx) * G.k, (fb[3] / 1000 - cy) * G.k], 4 * i);
        FT[i] = s.head.ztop / 1000 * G.k; FC[i] = (fb[2] - fb[0]) / 1000 / s.head.nx; has[i] = 1;
      });
      gl.uniform2i(u.farHas, has[0], has[1]); gl.uniform4fv(u.farBox, FB);
      gl.uniform2f(u.farTop, FT[0], FT[1]); gl.uniform2f(u.farCell, FC[0], FC[1]);
    }
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_3D, noiseTex); gl.uniform1i(u.noiseTex, 4);
    gl.uniform1f(u.detail, view.detail);
    gl.uniform1f(u.detailKm, Math.max(0.05, (box[2] - box[0]) / head.nx));
    gl.uniform1f(u.frameSeed, view.seed || 0);
    gl.uniform3f(u.kmOrigin, cx, cy, 0);
    gl.uniform1i(u.quality, view.quality);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_3D, lightTex); gl.uniform1i(u.lightTex, 5);
    gl.uniform1i(u.useLight, 1);
    gl.uniform1f(u.hazeKm, view.haze);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, gb.col); gl.uniform1i(u.geoA, 6);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, gb.dist); gl.uniform1i(u.geoT, 7);
    gl.uniform1i(u.hasGeo, 1);
    // night, lightning, headlights, the lamps (unit 13: free in the cloud pass)
    if (u.dayK) {
      gl.uniform1f(u.dayK, nt.dayK); gl.uniform1f(u.expo, nt.expo); gl.uniform1f(u.moonK, nt.moonK); gl.uniform1f(u.lampK, nt.lampK);
      gl.uniform3f(u.moonDir, ...nt.moonDir);
      const fl = view.flash;
      if (fl && fl.I > 0) { const fp = toBox(G, fl.p); gl.uniform4f(u.flashP, fp[0], fp[1], fp[2], fl.I); gl.uniform3f(u.flashC, ...fl.amb); }
      else { gl.uniform4f(u.flashP, 0, 0, 0, 0); gl.uniform3f(u.flashC, ...(fl ? fl.amb : [0, 0, 0])); }
      const hs = (view.heads || []).slice(0, 4), HP = new Float32Array(16), HD = new Float32Array(12);
      hs.forEach((h, i) => { HP.set([...toBox(G, h.p), h.on], 4 * i); HD.set(h.d, 3 * i); });
      gl.uniform4fv(u.headP, HP); gl.uniform3fv(u.headD, HD); gl.uniform1i(u.headN, hs.length);
      gl.activeTexture(gl.TEXTURE13); gl.bindTexture(gl.TEXTURE_2D, gb.light); gl.uniform1i(u.geoE, 13);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    drawPrecip(view, G, VPr, eye, W, H);
    drawBolts(view, G, VPr, eye, W, H);
    gl.disable(gl.BLEND);
    return { G, VP: VPc, W, H };
  }

  // lightning's channels over the finished picture: view.bolts [{ pts: [[x, y, z] world m], I 0..1, w m (the core's
  // half width) }]
  function drawBolts(view, G, VP, eye, W, H) {
    const list = view.bolts;
    if (!boltProg || !list || !list.length) return;
    const V = [];
    for (const b of list) {
      const P = b.pts.map(q => toBox(G, q)), wk = b.w * G.k / 1000;
      for (const [hw, a] of [[wk * 9, 0.22 * b.I], [wk, 1.0 * b.I]]) {
        for (let i = 1; i < P.length; i++) {
          const A = P[i - 1], B = P[i], d = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
          const m = [(A[0] + B[0]) / 2 - eye[0], (A[1] + B[1]) / 2 - eye[1], (A[2] + B[2]) / 2 - eye[2]];
          let sx = d[1] * m[2] - d[2] * m[1], sy = d[2] * m[0] - d[0] * m[2], sz = d[0] * m[1] - d[1] * m[0];
          const l = Math.hypot(sx, sy, sz) || 1;
          sx *= hw / l; sy *= hw / l; sz *= hw / l;
          const q = [[A[0] - sx, A[1] - sy, A[2] - sz, -1], [A[0] + sx, A[1] + sy, A[2] + sz, 1], [B[0] + sx, B[1] + sy, B[2] + sz, 1], [B[0] - sx, B[1] - sy, B[2] - sz, -1]];
          for (const k of [0, 1, 2, 0, 2, 3]) V.push(...q[k], a);
        }
      }
    }
    if (!V.length) return;
    gl.useProgram(boltProg); gl.bindVertexArray(bolts.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, bolts.buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(V), gl.STREAM_DRAW);
    gl.uniformMatrix4fv(boltProg.u.VP, false, VP); gl.uniform3f(boltProg.u.eye, ...eye); gl.uniform2f(boltProg.u.res, W, H);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, gb.dist); gl.uniform1i(boltProg.u.geoT, 7);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.TRIANGLES, 0, V.length / 5);
    gl.disable(gl.BLEND);
  }

  // the rain and snow round the eye, over the finished picture.  view.precip: { n drops / flakes, snow (their
  // share that are flakes), cube m, minD m (none closer: inside the car), wind [u, v] m/s, camV m/s, col }
  function drawPrecip(view, G, VP, eye, W, H) {
    const pr = view.precip;
    if (!precipProg || !pr || pr.n < 1) return;
    const p = precipProg, u = p.u;
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(p); gl.bindVertexArray(vao);
    gl.uniformMatrix4fv(u.VP, false, VP);
    gl.uniform3f(u.eye, ...eye); gl.uniform1f(u.s, G.k / 1000);
    gl.uniform1f(u.t, performance.now() / 1000 % 600);
    gl.uniform1f(u.cube, pr.cube); gl.uniform1f(u.snow, pr.snow); gl.uniform1f(u.minD, pr.minD);
    gl.uniform1f(u.expo, pr.expo || 0.03); gl.uniform1f(u.focal, H * 0.5 / Math.tan(view.fov / 2));
    gl.uniform3f(u.camV, ...pr.camV); gl.uniform3f(u.wind, pr.wind[0], pr.wind[1], 0);
    gl.uniform2f(u.res, W, H); gl.uniform3f(u.col, ...pr.col);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, gb.dist); gl.uniform1i(u.geoT, 7);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, Math.min(40000, Math.round(pr.n)));
  }

  // a world point (m) on the screen (CSS px of the canvas), or null behind the camera
  function project(VP, G, p) {
    const b = toBox(G, p), o = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) o[r] = VP[r] * b[0] + VP[4 + r] * b[1] + VP[8 + r] * b[2] + VP[12 + r];
    if (o[3] <= 0) return null;
    return [(o[0] / o[3] * 0.5 + 0.5) * cv.clientWidth, (1 - (o[1] / o[3] * 0.5 + 0.5)) * cv.clientHeight];
  }

  // how the picture is moving now (the HUD, testing): carried along the wind or fading, the model times
  const motion = () => ({ warp: mixSt.warp, wind: hasWind, f: mixSt.f, tA: mixSt.tA, tB: mixSt.tB, shown: shownT(),
                          insWarp: mixSt.insWarp.slice() });
  // (testing) one level of the shown volume ('vol') or nest i ('ins'), its first channel, after the blend's next step
  function _peek(which, i, k) {
    const t = which === 'vol' ? volTex : insTex[i], h = which === 'vol' ? head : head.insets[i];
    gl.bindFramebuffer(gl.FRAMEBUFFER, mixFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, 0, k);
    const px = new Uint8Array(h.nx * h.ny * 4);
    gl.readPixels(0, 0, h.nx, h.ny, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Uint8Array(h.nx * h.ny);
    for (let j = 0; j < out.length; j++) out[j] = px[4 * j];
    return { nx: h.nx, ny: h.ny, data: out };
  }
  return { init, setVolume, setFar, farInfo: () => farV.map(s => s.head && { box: s.head.box, nx: s.head.nx, ny: s.head.ny, nz: s.head.nz }), blending: () => mixSt.f < 1, motion, _gb: () => gb, _peek,
           _step: () => { mixSt.dirty = true; advance(geometry(), [0.5, 0.3, 0.81], 1, 1); }, setRoads, setTrees, setBuildings, setLamps, dropPart, frame, cloudBase, terrain, project, sample, head: () => head, box: () => box, gl: () => gl };
})();
window.Render = Render;
