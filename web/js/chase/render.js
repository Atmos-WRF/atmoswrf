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
  let head = null, box = null, gMap = null, floatLinear = false, volData = null;
  const gb = { fbo: null, col: null, dist: null, depth: null, w: 0, h: 0 };
  const roads = { buf: null, vao: null, n: 0, origin: [0, 0] };
  const meshes = new Map();                      // car kind/colour -> { body, wheel }
  let treeProg = null;
  const trees = { mesh: null, near: 0, far: 0, inst: null, count: 0, nearCount: 0, origin: [0, 0], vao: null };
  const blds = { buf: null, vao: null, n: 0, origin: [0, 0] };

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
  layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1;
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
  layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1;
  void main() {
    vec3 n = normalize(nw);
    float lam = max(dot(n, sunDir), 0.0);
    // the sky's light on faces turned up, less on the sides and under (the cloud pass adds one flat ambient)
    vec3 c = col * (0.72 + 0.28 * n.z);
    o0 = vec4(c, lam);
    o1 = vec4(length(wpos - eye), 0.0, 0.0, 1.0);
  }`;

  // the trees (scenery.js): a hexagonal trunk and a lumpy low-poly crown, one instance per tree
  const TREE_VS = `#version 300 es
  layout(location = 0) in vec3 p; layout(location = 1) in vec3 n; layout(location = 2) in float part;
  layout(location = 3) in vec4 a; layout(location = 4) in vec2 b;
  uniform mat4 VP; uniform vec3 org; uniform float s;
  out vec3 wpos; out vec3 nw; out vec3 col;
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

  // A tornado's condensation funnel, which the model's grid cannot condense by itself: the vortex's pressure drop
  // (Rankine: rho V^2 in the core, half that at the core's edge, falling as 1/r^2 outside) cools the air drawn
  // into it, and the cloud comes down to where that drop equals the pressure difference to the cloud base
  // (rho g dz, ~0.12 hPa a metre).  So: a wide lowering under the base, narrowing to a funnel V^2/g x funnelK
  // below it (down to the ground for strong ones), its edge torn by a noise that turns with the vortex and rises;
  // and near the ground, for winds over ~35 m/s, a debris cloud.  torA: x, y (box units), core radius m, wind m/s;
  // torB: cloud base m above the ground, the ground m, -, -.
  const FUNNEL_GLSL = `
  float gDebris = 0.0;                                         // the debris' share of the extinction here
  vec2 funnels(vec3 p) {
    gDebris = 0.0;
    vec2 out2 = vec2(0.0);
    for (int i = 0; i < 4; i++) {
      if (i >= torN) break;
      vec4 A = torA[i], B = torB[i];
      vec2 dxy = (p.xy - A.xy) * toKm.xy * 1000.0;
      float d = length(dxy), z = p.z * toKm.z * 1000.0 - B.y, R = A.z, V = A.w, zb = B.x;
      if (d > 6.0 * R || z > zb + 60.0 || z < -5.0) continue;
      float dp0 = 1.2 * V * V * funnelK;                       // Pa: the drop in the core
      float q = 11.8 * max(zb - z, 0.0);                        // Pa: what it takes to condense at this height
      float rf = q < 0.5 * dp0 ? R * sqrt(dp0 / (2.0 * max(q, 1.0))) : R * sqrt(max(2.0 * (1.0 - q / dp0), 0.0));
      rf = min(rf, 5.0 * R);
      if (q < dp0) rf = max(rf, 0.3 * R);                     // the tip: a rope down to the depth, not a fade
      // turning with the vortex, rising: the funnel's striations
      float om = 0.6 * V / max(R, 20.0), a = -uTime * om;
      vec2 rx = mat2(cos(a), sin(a), -sin(a), cos(a)) * dxy;
      float n = texture(noiseTex, vec3(rx / 160.0, z / 260.0 - uTime * 0.03)).r;
      float n2 = texture(noiseTex, vec3(rx / 45.0, z / 70.0 - uTime * 0.08) + 0.37).g;
      float edge = d * (1.0 + 0.35 * (n - 0.5) + 0.2 * (n2 - 0.5));
      float f = smoothstep(rf * 1.12, rf * 0.6, edge) * smoothstep(-5.0, 15.0, zb + 60.0 - z);
      out2.x += 34.0 * f * smoothstep(0.0, 0.3 * R, rf);         // per km: a thick, milky funnel
      // the debris cloud: dust and debris lifted where the winds at the ground pass ~35 m/s
      if (V > 35.0) {
        float hd = 25.0 + 6.0 * (V - 35.0), rd = R * (1.0 + 0.9 * clamp(z / hd, 0.0, 1.0));
        float dd = smoothstep(rd, rd * 0.45, d * (1.0 + 0.5 * (n2 - 0.5))) * (1.0 - smoothstep(0.2 * hd, hd, z)) * smoothstep(0.3, 0.7, n + 0.3);
        float de = 9.0 * dd * clamp((V - 35.0) / 25.0, 0.2, 1.0);
        out2.y += de; gDebris += de;
      }
    }
    return out2;
  }`;

  // the 3-D view's cloud shader with a contrast control on the cloud alone (the ground and sky as they are):
  // dark bases darker, sunlit towers brighter, around a mid grey in log light
  function cloudSource() {
    let src = L().FS_CLOUD;
    const head = 'in vec2 ndc; out vec4 frag;', tail = 'vec3 c = col + T * bg;';
    if (!src.includes(head) || !src.includes(tail)) return src;
    src = src.replace(head, head + '\n  uniform float cloudContrast;\n  uniform int torN; uniform vec4 torA[4]; uniform vec4 torB[4]; uniform float uTime, funnelK;');
    // the condensation funnels and debris clouds (funnelGLSL), added to the cloud wherever it is read
    const media = 'vec2 media(vec3 p, int mode) {', ret = '    return s;\n  }\n  float sig(vec2 s)';
    if (!src.includes('float albedo = mix(0.98, 0.78, s.y / (sg + 1e-6));')) console.warn('chase: debris colour hook not found');
    if (src.includes(media) && src.includes(ret)) {
      src = src.replace(media, FUNNEL_GLSL + '\n  ' + media);
      src = src.replace(ret, '    if (torN > 0 && funnelK > 0.0) s += funnels(p) * density;\n' + ret);
    }
    src = src.replace('float albedo = mix(0.98, 0.78, s.y / (sg + 1e-6));',
                      'float albedo = mix(mix(0.98, 0.78, s.y / (sg + 1e-6)), 0.3, clamp(gDebris * density / (sg + 1e-6), 0.0, 1.0));');
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

  function init(canvas) {
    cv = canvas;
    gl = cv.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (!gl) return false;
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('this browser cannot render to float textures (EXT_color_buffer_float)');
    floatLinear = !!gl.getExtension('OES_texture_float_linear');
    cloudProg = program(L().VS, cloudSource());
    lightProg = program(L().VS, L().FS_LIGHT);
    roadProg = program(ROAD_VS, ROAD_FS);
    meshProg = program(MESH_VS, MESH_FS);
    treeProg = program(TREE_VS, MESH_FS);
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
    for (let i = 0; i < 4; i++) insTex.push(gl.createTexture());
    noiseTex = L().makeNoise(gl);
    gb.fbo = gl.createFramebuffer(); gb.col = gl.createTexture(); gb.dist = gl.createTexture(); gb.depth = gl.createRenderbuffer();
    roads.buf = gl.createBuffer(); roads.vao = gl.createVertexArray();
    Land.init(gl, () => {});
    Land.setTrees(false);                          // no trees in chase mode (yet)
    return true;
  }

  // ---------------------------------------------------------------- the volume
  function setVolume(b, buf) {
    const { header, data } = unpack(buf);
    head = header; box = b; loadSeq++; volData = data;
    const two = header.channels === 2;
    gl.bindTexture(gl.TEXTURE_3D, volTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, two ? gl.RG8 : gl.R8, header.nx, header.ny, header.nz, 0, two ? gl.RG : gl.RED, gl.UNSIGNED_BYTE, data);
    texParams(gl.TEXTURE_3D, gl.LINEAR, gl.CLAMP_TO_EDGE);
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
    header.insets = (header.insets || []).slice(0, 4);
    header.insets.forEach((ins, i) => {
      const n = ins.nx * ins.ny * ins.nz * header.channels;
      gl.bindTexture(gl.TEXTURE_3D, insTex[i]);
      gl.texImage3D(gl.TEXTURE_3D, 0, two ? gl.RG8 : gl.R8, ins.nx, ins.ny, ins.nz, 0, two ? gl.RG : gl.RED, gl.UNSIGNED_BYTE,
                    new Uint8Array(data.buffer, off, n));
      texParams(gl.TEXTURE_3D, gl.LINEAR, gl.CLAMP_TO_EDGE);
      off += n;
    });
    return header;
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

  function buildLight(G, sun, dens, rain) {
    const key = [head.nx, head.ny, head.nz, loadSeq, ...sun.map(v => v.toFixed(3)), dens, rain, G.sx, G.sy, G.sz].join('/');
    if (key === lightKey) return true;
    const nx = head.nx, ny = head.ny, nz = head.nz;
    gl.bindTexture(gl.TEXTURE_3D, lightTex);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG16F, nx, ny, nz, 0, gl.RG, gl.HALF_FLOAT, null);
    texParams(gl.TEXTURE_3D, gl.LINEAR, gl.CLAMP_TO_EDGE);
    gl.useProgram(lightProg); gl.bindVertexArray(vao);
    const u = lightProg.u;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, volTex); gl.uniform1i(u.vol, 0);
    gl.uniform3f(u.size, G.sx, G.sy, G.sz);
    gl.uniform3f(u.toKm, 1 / G.k, 1 / G.k, 1 / G.k);
    gl.uniform3f(u.sunDir, ...sun);
    gl.uniform3f(u.lres, nx, ny, nz);
    gl.uniform1f(u.density, dens); gl.uniform1f(u.rain, rain);
    gl.bindFramebuffer(gl.FRAMEBUFFER, lightFbo);
    gl.viewport(0, 0, nx, ny);
    for (let k = 0; k < nz; k++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, lightTex, 0, k);
      gl.uniform1f(u.layer, k);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    lightKey = key;
    return true;
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
  function setBuildings(mesh, origin) {
    gl.bindVertexArray(blds.vao); gl.bindBuffer(gl.ARRAY_BUFFER, blds.buf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24);
    gl.bindVertexArray(null);
    blds.n = mesh.length / 9; blds.origin = origin;
  }

  function meshBuffer(arr) {
    const v = gl.createVertexArray(), b = gl.createBuffer();
    gl.bindVertexArray(v); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24);
    gl.bindVertexArray(null);
    return { vao: v, n: arr.length / 9 };
  }
  function carMeshes(kind, color) {
    const key = `${kind}/${color.join(',')}`;
    if (!meshes.has(key)) {
      const m = CarMesh.build(CarPhys.SPECS[kind], kind, color);
      meshes.set(key, { body: meshBuffer(m.body), wheel: meshBuffer(m.wheel) });
    }
    return meshes.get(key);
  }

  function targets(w, h) {
    if (gb.w === w && gb.h === h) return;
    gb.w = w; gb.h = h;
    for (const [t, fmt, f, ty] of [[gb.col, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE], [gb.dist, gl.R32F, gl.RED, gl.FLOAT]]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, f, ty, null);
      texParams(gl.TEXTURE_2D, gl.NEAREST, gl.CLAMP_TO_EDGE);
    }
    gl.bindRenderbuffer(gl.RENDERBUFFER, gb.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, gb.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gb.col, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, gb.dist, 0);
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

  function drawGeometry(G, VP, eye, sun, cars, wet, showTrees) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, gb.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, gb.w, gb.h);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]); gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]); gl.clearBufferfv(gl.DEPTH, 0, [1]);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    const s = G.k / 1000;
    if (roads.n) {
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
    if (blds.n) {
      gl.uniformMatrix3fv(u.R, false, new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
      gl.uniform3f(u.org, ...toBox(G, [blds.origin[0], blds.origin[1], 0]));
      gl.bindVertexArray(blds.vao); gl.drawArrays(gl.TRIANGLES, 0, blds.n);
    }
    gl.enable(gl.CULL_FACE);
    for (const c of cars) {
      if (c.hidden) continue;
      const m = carMeshes(c.kind, c.color);
      gl.uniformMatrix3fv(u.R, false, mat3(c.M));
      gl.uniform3f(u.org, ...toBox(G, c.p));
      gl.bindVertexArray(m.body.vao); gl.drawArrays(gl.TRIANGLES, 0, m.body.n);
      gl.bindVertexArray(m.wheel.vao);
      for (const w of c.wheels) {
        gl.uniformMatrix3fv(u.R, false, wheelMat(c.M, w.steer, -w.spin));
        gl.uniform3f(u.org, ...toBox(G, w.c));
        gl.drawArrays(gl.TRIANGLES, 0, m.wheel.n);
      }
    }
    gl.disable(gl.CULL_FACE); gl.disable(gl.DEPTH_TEST);
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
    targets(W, H);
    drawGeometry(G, VPr, eye, view.sun, cars, view.wet, view.trees !== false);
    buildLight(G, view.sun, view.density, view.rain);
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
    // the tornadoes' funnels: [{x, y (world m), R m, V m/s, base m above the ground, ground m}]
    const tors = (view.tornadoes || []).slice(0, 4), TA = new Float32Array(16), TB = new Float32Array(16);
    tors.forEach((t, i) => {
      const b = toBox(G, [t.x, t.y, 0]);
      TA.set([b[0], b[1], t.R, t.V], 4 * i); TB.set([t.base, t.ground, 0, 0], 4 * i);
    });
    gl.uniform1i(u.torN, tors.length); gl.uniform4fv(u.torA, TA); gl.uniform4fv(u.torB, TB);
    gl.uniform1f(u.uTime, performance.now() / 1000 % 10000); gl.uniform1f(u.funnelK, view.funnel === undefined ? 2 : view.funnel);
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
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.BLEND);
    return { G, VP: VPc, W, H };
  }

  // a world point (m) on the screen (CSS px of the canvas), or null behind the camera
  function project(VP, G, p) {
    const b = toBox(G, p), o = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) o[r] = VP[r] * b[0] + VP[4 + r] * b[1] + VP[8 + r] * b[2] + VP[12 + r];
    if (o[3] <= 0) return null;
    return [(o[0] / o[3] * 0.5 + 0.5) * cv.clientWidth, (1 - (o[1] / o[3] * 0.5 + 0.5)) * cv.clientHeight];
  }

  return { init, setVolume, setRoads, setTrees, setBuildings, frame, cloudBase, terrain, project, sample, head: () => head, box: () => box, gl: () => gl };
})();
window.Render = Render;
