'use strict';
// The 3-D view's landscape from real data (arhtmodelviewer/landscape.py): ESA WorldCover's land cover (trees,
// grass, crops, towns, water ...) and OpenStreetMap's roads, for the cloud view.
//  - the whole box: with "Plain map" picked, the ground is painted from the land cover (fields as a patchwork,
//    towns grey, forests dark) with the main roads on it;
//  - close up (free and ground cameras): a sharp picture DETAIL_KM across around the camera -- the map tiles
//    picked on the map bar, or the land cover with every road down to tracks -- re-centred as the camera moves;
//  - trees: one for each bit of WorldCover's tree cover (and a few in towns and grassland), as sprites standing on
//    the ground, fewer and larger with distance, drawn into a small geometry buffer (colour + distance) that the
//    cloud's ray march stops at and lights (volume.js).
const Land = (() => {
  const DETAIL_KM = 8, DETAIL_PX = 2048, COARSE_PX = 2048;
  const TREE_R1 = 0.8;                        // km: every tree out to here, then fewer (1/r^2) and larger
  const MAX_TREES = 220000;
  // WorldCover class -> sRGB colour, and trees per m^2 standing on it
  const COLORS = {
    10: [58, 74, 38], 20: [112, 104, 64], 30: [124, 136, 72], 40: [150, 140, 90], 50: [150, 146, 140],
    60: [176, 162, 132], 70: [236, 240, 244], 80: [38, 58, 72], 90: [72, 96, 70], 95: [40, 70, 42], 100: [130, 128, 100],
  };
  const TREES_PER_M2 = { 10: 1 / 42, 20: 1 / 400, 30: 1 / 5000, 40: 1 / 40000, 50: 1 / 700, 90: 1 / 200, 95: 1 / 40 };
  const FIELDS = [[96, 120, 52], [168, 150, 100], [122, 98, 72], [140, 148, 70], [110, 128, 60], [185, 170, 115]];
  // road class (landscape.ROAD_CLASSES order) -> width m
  const ROAD_W = [24, 18, 12, 9, 7, 6, 6, 3.5];

  let gl = null, onReady = null;
  const coarse = { key: '', tex: null, ok: false, busy: false };
  const detail = { key: '', box: null, tex: null, ok: false, busy: false, lc: null, roads: null, base: null, poll: 0 };
  const trees = { buf: null, vao: null, n: 0, prog: null, fbo: null, colTex: null, tTex: null, depth: null, w: 0, h: 0 };

  // a stable hash of integers -> [0, 1)
  const hash = (a, b, c = 0) => {
    let h = (Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263) + Math.imul(c | 0, 2147483647)) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  };

  function tex2D(t, src) {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);          // row 0 (north) at the top: v = 1
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.generateMipmap(gl.TEXTURE_2D);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
                          [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_2D, k, v);
  }

  async function landcover(b, px) {
    const r = await api(`/api/volume/landcover?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&px=${px}`);
    const { header, data } = unpack(await r.arrayBuffer());
    return { W: header.W, H: header.H, a: data, box: b };
  }

  // the land cover painted: each class its colour, crop fields a patchwork of 400 m fields, a little grain
  function paint(lc) {
    const cv = document.createElement('canvas'); cv.width = lc.W; cv.height = lc.H;
    const g = cv.getContext('2d'), im = g.createImageData(lc.W, lc.H), d = im.data;
    const [x0, , x1, y1] = lc.box, dx = (x1 - x0) / lc.W, dy = (lc.box[3] - lc.box[1]) / lc.H;
    for (let j = 0, o = 0; j < lc.H; j++) {
      const y = y1 - (j + 0.5) * dy;
      for (let i = 0; i < lc.W; i++, o += 4) {
        const c = lc.a[j * lc.W + i], x = x0 + (i + 0.5) * dx;
        let col = COLORS[c] || COLORS[30];
        if (c === 40) col = FIELDS[Math.floor(hash(Math.floor(x / 0.4), Math.floor(y / 0.4), 7) * FIELDS.length)];
        const n = 0.9 + 0.2 * hash(Math.floor(x * 50), Math.floor(y * 50), c);
        d[o] = col[0] * n; d[o + 1] = col[1] * n; d[o + 2] = col[2] * n; d[o + 3] = 255;
      }
    }
    g.putImageData(im, 0, 0);
    return cv;
  }

  function drawRoads(cv, b, roads, minClass) {
    if (!roads) return;
    const g = cv.getContext('2d'), sx = cv.width / (b[2] - b[0]), sy = cv.height / (b[3] - b[1]);
    const mpp = 1000 / sx;
    g.lineCap = 'round'; g.lineJoin = 'round';
    const order = roads.filter(r => r[0] <= minClass).sort((a, c) => c[0] - a[0]);   // small roads first
    for (const [k, paved, xy] of order) {
      const w = ROAD_W[k] / mpp;
      g.lineWidth = Math.max(w, 0.8);
      g.strokeStyle = !paved ? `rgba(160,140,108,${Math.min(1, w + 0.25)})`
        : k <= 1 ? `rgba(128,128,124,${Math.min(1, w + 0.35)})` : `rgba(64,64,66,${Math.min(1, w + 0.35)})`;
      g.beginPath();
      for (let i = 0; i < xy.length; i += 2) {
        const px = (xy[i] - b[0]) * sx, py = (b[3] - xy[i + 1]) * sy;
        if (i) g.lineTo(px, py); else g.moveTo(px, py);
      }
      g.stroke();
    }
  }

  async function roadsIn(b, major) {
    try { return await (await api(`/api/volume/roads?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&major=${major ? 1 : 0}`)).json(); }
    catch (e) { return null; }
  }

  // ---------------------------------------------------------------- the whole box ("Plain map")
  async function loadCoarse(box) {
    const key = box.join(',');
    if (coarse.key === key || coarse.busy) return;
    coarse.busy = true;
    let done = false;
    try {
      const lc = await landcover(box, COARSE_PX);
      const cv = paint(lc);
      let rd = await roadsIn(box, true);
      drawRoads(cv, box, rd && rd.roads, 3);
      if (!coarse.tex) coarse.tex = gl.createTexture();
      tex2D(coarse.tex, cv);
      coarse.key = key; coarse.ok = true; done = true;
      coarse.busy = false;            // a moved box (Follow tornado) can load while the roads below still come in
      if (onReady) onReady();
      // the main roads still coming from OpenStreetMap: painted again once they are all in
      for (let tries = 0; rd && rd.pending && tries < 40 && coarse.key === key; tries++) {
        await new Promise(r => setTimeout(r, 8000));
        rd = await roadsIn(box, true);
        if (!rd || rd.pending || coarse.key !== key) continue;
        const c2 = paint(lc); drawRoads(c2, box, rd.roads, 3); tex2D(coarse.tex, c2);
        if (onReady) onReady();
      }
    } catch (e) {
      if (!done) { coarse.ok = false; coarse.key = key; }
      console.warn('3-D land cover:', e);
    }
    if (!done) coarse.busy = false;
  }

  // ---------------------------------------------------------------- close up
  async function loadDetail(cx, cy, base, groundKm) {
    const h = DETAIL_KM / 2, b = [+(cx - h).toFixed(3), +(cy - h).toFixed(3), +(cx + h).toFixed(3), +(cy + h).toFixed(3)];
    const key = `${b.join(',')}/${base}`;
    if (detail.busy) return;
    detail.busy = true;
    let done = false;
    try {
      const lc = await landcover(b, 1024);
      let cv = paint(lc);
      let img = null;
      if (base !== 'plain') {
        const src = base.startsWith('sat') ? 'sat' : 'osm', labels = base === 'satlabels' ? 1 : 0;
        try {
          const r = await api(`/api/volume/ground?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}&src=${src}&labels=${labels}&px=${DETAIL_PX}`);
          img = await createImageBitmap(await r.blob());
        } catch (e) { img = null; }
      }
      const compose = rd => {
        const out = document.createElement('canvas'); out.width = DETAIL_PX; out.height = DETAIL_PX;
        const g = out.getContext('2d');
        if (img) g.drawImage(img, 0, 0, DETAIL_PX, DETAIL_PX);
        else {
          g.imageSmoothingEnabled = false; g.drawImage(cv, 0, 0, DETAIL_PX, DETAIL_PX);
          drawRoads(out, b, rd && rd.roads, 9);
        }
        return out;
      };
      let rd = await roadsIn(b, false);
      detail.roads = rd && rd.roads;
      if (!detail.tex) detail.tex = gl.createTexture();
      tex2D(detail.tex, compose(rd));
      Object.assign(detail, { key, box: b, ok: true, lc });
      buildTrees(lc, cx, cy, groundKm);
      detail.busy = false; done = true;   // the camera can move on (re-centre) while the roads below still come in
      if (onReady) onReady();
      const poll = ++detail.poll;
      for (let tries = 0; rd && rd.pending && tries < 40 && poll === detail.poll; tries++) {
        await new Promise(r => setTimeout(r, 6000));
        const r2 = await roadsIn(b, false);
        if (!r2 || poll !== detail.poll) break;
        detail.roads = r2.roads;
        if (r2.roads.length !== rd.roads.length || !r2.pending) { tex2D(detail.tex, compose(r2)); if (onReady) onReady(); }
        rd = r2;
      }
    } catch (e) { if (!done) detail.key = key; console.warn('3-D close-up ground:', e); }
    if (!done) detail.busy = false;
  }

  // the trees on a 4 m grid fixed to the ground (so they stay put as the picture is re-centred): each cell's land
  // cover gives its chance of a tree, fewer (and bigger) away from the camera
  function buildTrees(lc, cx, cy, groundKm) {
    const CELL = 0.004, [x0, y0, x1, y1] = lc.box, dx = (x1 - x0) / lc.W, dy = (y1 - y0) / lc.H;
    const out = new Float32Array(MAX_TREES * 6);
    let n = 0;
    const i0 = Math.ceil(x0 / CELL), i1 = Math.floor(x1 / CELL), j0 = Math.ceil(y0 / CELL), j1 = Math.floor(y1 / CELL);
    for (let j = j0; j <= j1 && n < MAX_TREES; j++) {
      const y = j * CELL, row = Math.min(lc.H - 1, Math.max(0, Math.floor((y1 - y) / dy)));
      for (let i = i0; i <= i1; i++) {
        const x = i * CELL, c = lc.a[row * lc.W + Math.min(lc.W - 1, Math.max(0, Math.floor((x - x0) / dx)))];
        const per = TREES_PER_M2[c];
        if (!per) continue;
        const r = Math.hypot(x - cx, y - cy), lod = r < TREE_R1 ? 1 : (TREE_R1 / r) ** 2;
        if (r > DETAIL_KM / 2) continue;
        if (hash(i, j, 1) >= per * 16 * lod) continue;
        const u = hash(i, j, 2), v = hash(i, j, 3), s = hash(i, j, 4);
        const tx = x + (u - 0.5) * CELL, ty = y + (v - 0.5) * CELL;
        const big = Math.min(1.8, 1 / Math.sqrt(lod));
        const hgt = (c === 10 || c === 95 ? 7 + 10 * s : 5 + 7 * s) * Math.min(1.25, Math.sqrt(big));
        out[6 * n] = tx; out[6 * n + 1] = ty; out[6 * n + 2] = groundKm(tx, ty); out[6 * n + 3] = hgt;
        out[6 * n + 4] = hgt * (0.32 + 0.16 * hash(i, j, 5)) * big; out[6 * n + 5] = hash(i, j, 6);
        if (++n >= MAX_TREES) break;
      }
    }
    if (!trees.buf) trees.buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, trees.buf);
    gl.bufferData(gl.ARRAY_BUFFER, out.subarray(0, 6 * n), gl.STATIC_DRAW);
    trees.n = n;
  }

  // ---------------------------------------------------------------- the trees' pass
  const TREE_VS = `#version 300 es
  layout(location = 0) in vec4 a;            // x, y km (world), ground km, height m
  layout(location = 1) in vec2 b;            // crown radius m, seed
  uniform mat4 VP; uniform vec3 eye; uniform float k, ex; uniform vec2 c0;     // c0: the box's middle, km
  out vec2 uv; out float seed; out vec3 wpos; out vec3 rgt; out vec3 toE;
  const vec2 Q[6] = vec2[6](vec2(-1, 0), vec2(1, 0), vec2(1, 1), vec2(-1, 0), vec2(1, 1), vec2(-1, 1));
  void main() {
    vec2 c = Q[gl_VertexID];
    vec3 base = vec3((a.x - c0.x) * k, (a.y - c0.y) * k, a.z * k * ex);
    vec2 e = eye.xy - base.xy; e = length(e) > 1e-9 ? normalize(e) : vec2(0, 1);
    rgt = vec3(-e.y, e.x, 0.0); toE = vec3(e, 0.0);
    float R = b.x * 0.001 * k, H = a.w * 0.001 * k * ex;
    wpos = base + rgt * c.x * R + vec3(0.0, 0.0, c.y * H);
    uv = c; seed = b.y;
    gl_Position = VP * vec4(wpos, 1.0);
  }`;
  const TREE_FS = `#version 300 es
  precision highp float;
  in vec2 uv; in float seed; in vec3 wpos; in vec3 rgt; in vec3 toE;
  uniform vec3 eye, sunDir;
  layout(location = 0) out vec4 col;          // albedo (sRGB), how much the crown faces the sun
  layout(location = 1) out vec4 dist;         // distance from the eye along the ray (world units)
  void main() {
    float cy = (uv.y - 0.6) / 0.4, cx = uv.x;
    float ang = atan(cy, cx), rr = length(vec2(cx, cy));
    float edge = 0.78 + 0.07 * sin(ang * 5.0 + seed * 31.0) + 0.05 * sin(ang * 11.0 + seed * 17.0);
    // two side lobes and a top one, placed by the seed: an uneven, leafy outline
    vec2 l1 = vec2(-0.45 + 0.2 * fract(seed * 3.1), -0.25 + 0.3 * fract(seed * 5.3));
    vec2 l2 = vec2(0.45 - 0.2 * fract(seed * 7.7), -0.2 + 0.3 * fract(seed * 2.9));
    vec2 l3 = vec2(0.3 * fract(seed * 4.3) - 0.15, 0.45);
    float lobes = min(min(length(vec2(cx, cy) - l1) / 0.5, length(vec2(cx, cy) - l2) / 0.5), length(vec2(cx, cy) - l3) / 0.45);
    bool crown = rr < edge || lobes < 1.0, trunk = abs(uv.x) < 0.05 && uv.y < 0.32;
    if (!crown && !trunk) discard;
    vec3 c;
    float lam;
    if (crown) {
      float w = sqrt(max(0.0, 1.0 - min(min(rr / edge, lobes), 1.0)));
      vec3 n = normalize(rgt * cx + vec3(0, 0, 1) * cy + toE * w * 1.2);
      lam = 0.25 + 0.75 * max(dot(n, sunDir), 0.0);
      float v = fract(seed * 7.13);
      c = mix(vec3(0.16, 0.24, 0.09), vec3(0.27, 0.32, 0.13), v) * (0.8 + 0.35 * fract(sin(dot(uv * 9.0 + seed, vec2(12.9, 78.2))) * 43758.5));
    } else {
      c = vec3(0.13, 0.10, 0.075); lam = 0.35;
    }
    col = vec4(pow(c, vec3(1.0 / 2.2)), lam);
    dist = vec4(length(wpos - eye), 0, 0, 1);
  }`;

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  function init(g, ready) {
    gl = g; onReady = ready;
    if (!gl.getExtension('EXT_color_buffer_float')) return;
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, TREE_VS));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, TREE_FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    trees.prog = p;
    trees.vao = gl.createVertexArray();
    trees.fbo = gl.createFramebuffer();
    trees.colTex = gl.createTexture(); trees.tTex = gl.createTexture(); trees.depth = gl.createRenderbuffer();
  }

  function targets(w, h) {
    if (trees.w === w && trees.h === h) return;
    trees.w = w; trees.h = h;
    for (const [t, fmt, f, ty] of [[trees.colTex, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE], [trees.tTex, gl.R32F, gl.RED, gl.FLOAT]]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, f, ty, null);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                            [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
        gl.texParameteri(gl.TEXTURE_2D, k, v);
    }
    gl.bindRenderbuffer(gl.RENDERBUFFER, trees.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, trees.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, trees.colTex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, trees.tTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, trees.depth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // the trees into the geometry buffer (w x h, the view's size): true when there is one to read
  function drawTrees(w, h, V, fov, eye, G, sun, c0) {
    if (!trees.prog || !trees.n) return false;
    targets(w, h);
    const near = 0.0005 * G.k, far = 40 * G.k;
    const P = perspectiveM(fov, w / h, near, far);
    gl.bindFramebuffer(gl.FRAMEBUFFER, trees.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, w, h);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.DEPTH, 0, [1]);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    gl.useProgram(trees.prog);
    gl.bindVertexArray(trees.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, trees.buf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 24, 0); gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 24, 16); gl.vertexAttribDivisor(1, 1);
    const u = n => gl.getUniformLocation(trees.prog, n);
    gl.uniformMatrix4fv(u('VP'), false, mulM(P, V));
    gl.uniform3f(u('eye'), ...eye);
    gl.uniform1f(u('k'), G.k); gl.uniform1f(u('ex'), G.ex);
    gl.uniform3f(u('sunDir'), ...sun);
    gl.uniform2f(u('c0'), c0[0], c0[1]);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, trees.n);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return true;
  }

  // matrices (column-major), the same as volume.js's
  function perspectiveM(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f; o[10] = (far + near) / (near - far); o[11] = -1; o[14] = 2 * far * near / (near - far);
    return o;
  }
  function mulM(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  }

  // what the view needs now: the box's painted ground ("Plain map"), and close up the sharp ground and the trees
  // around the camera (cam: x, y km world, or null for the orbit view)
  function want(box, base, cam, groundKm) {
    if (!gl || !box) return;
    if (base === 'plain') loadCoarse(box);
    if (!cam) return;
    const key = `/${base}`;
    const far = !detail.box || Math.hypot(cam.x - (detail.box[0] + detail.box[2]) / 2, cam.y - (detail.box[1] + detail.box[3]) / 2) > DETAIL_KM * 0.22;
    if (far || !detail.key.endsWith(key)) loadDetail(cam.x, cam.y, base, groundKm);
  }
  // where a chaser would stand near x, y (world km): on the nearest road within 2 km, else the nearest open land
  // (grass, crops, bare ground) -- not inside a wood or a lake; null until the close-up land is in
  function standNear(x, y) {
    if (!detail.ok || !detail.lc) return null;
    let best = null, bd = 2.0;
    for (const [k, , xy] of detail.roads || []) {
      if (k > 6) continue;
      for (let i = 0; i < xy.length; i += 2) {
        const d = Math.hypot(xy[i] - x, xy[i + 1] - y);
        if (d < bd) { bd = d; best = [xy[i], xy[i + 1]]; }
      }
    }
    if (best) return best;
    const lc = detail.lc, [x0, , x1, y1] = lc.box, dx = (x1 - x0) / lc.W, dy = (lc.box[3] - lc.box[1]) / lc.H;
    const i0 = Math.floor((x - x0) / dx), j0 = Math.floor((y1 - y) / dy);
    for (let r = 0; r < 200; r++) for (let j = j0 - r; j <= j0 + r; j++) for (let i = i0 - r; i <= i0 + r; i++) {
      if (Math.max(Math.abs(i - i0), Math.abs(j - j0)) !== r || i < 0 || j < 0 || i >= lc.W || j >= lc.H) continue;
      if ([30, 40, 60].includes(lc.a[j * lc.W + i])) return [x0 + (i + 0.5) * dx, y1 - (j + 0.5) * dy];
    }
    return null;
  }

  function reset() {
    coarse.key = ''; coarse.ok = false; detail.key = ''; detail.box = null; detail.ok = false; detail.poll++;
    trees.n = 0;
  }

  return {
    init, want, reset, drawTrees, standNear,
    coarse: () => (coarse.ok ? coarse.tex : null),
    detail: () => (detail.ok ? { tex: detail.tex, box: detail.box } : null),
    trees: () => trees,
    // the loop export waits for this: the box's land cover (box: null when the map is not Plain) and the close-up
    // ground around the camera (cam: null for the orbit camera) are in
    ready: (box, cam) => (!box || (coarse.ok && coarse.key === box.join(','))) &&
      (!cam || (detail.ok && !!detail.box && Math.hypot(cam.x - (detail.box[0] + detail.box[2]) / 2,
                                                        cam.y - (detail.box[1] + detail.box[3]) / 2) <= DETAIL_KM * 0.22)),
  };
})();
window.Land = Land;
