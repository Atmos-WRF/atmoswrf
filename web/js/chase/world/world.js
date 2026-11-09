// Chase mode's detailed map, from Real Roads Drive: real terrain (AWS Terrain Tiles), aerial photos and land
// classes (NAIP / Sentinel-2, USDA CDL, ESA WorldCover), roads with their lanes and markings (OpenFreeMap),
// bridges, water, buildings (OpenStreetMap, FEMA USA Structures), trees, grass and crops, roadside props, farms,
// landmarks and birds -- its own car, camera, minimap, navigation, search, collection and HUD left out.
// Built by build_world.py; chase mode drives it through the API at the end (window.ChaseWorld).
import * as THREE from 'three';

// ===================================================================== setup

const CW = globalThis.CHASE_WORLD;
const START_LAT = CW.lat, START_LON = CW.lon;
const DEFAULT_START = false, START_FACE = null;

const Z = 14, NT = 2 ** Z, M = 16, G = 257 + 2 * M, SEG = 13;

// ------------------------------------------------ region: driving side, sign style, units
// warn: warning-sign shape · limit: speed-limit sign shape · yellow: yellow centre lines
// mailbox: roadside mailboxes · wood: wooden utility poles · delin: delineator post style
function regionFor(lat, lon) {
  const box = (a, b, c, d) => lat >= a && lat <= b && lon >= c && lon <= d;
  const R = (id, o) => Object.assign({ id, left: false, mph: false, warn: 'triangle', limit: 'circle', yellow: false, mailbox: false, wood: false, delin: 'eu' }, o);
  if (box(24, 50, -125, -66) || box(51, 72, -170, -129) || box(18, 23, -161, -154))
    return R('us', { mph: true, warn: 'diamond', limit: 'rect', yellow: true, mailbox: true, wood: true, delin: 'us' });
  if (box(41.5, 84, -141, -52)) return R('ca', { warn: 'diamond', limit: 'rect', yellow: true, mailbox: true, wood: true, delin: 'us' });
  if (box(-56, 33, -118, -34)) return R('latam', { warn: 'diamond', yellow: true });
  if (box(49.8, 61, -11, 2)) {
    const ireland = lon < -6 && !(lat > 54 && lon > -8.2);
    return R(ireland ? 'ie' : 'uk', { left: true, mph: !ireland, warn: ireland ? 'diamond' : 'triangle', wood: true, delin: 'uk' });
  }
  if (box(-48, -9, 112, 179)) return R('au', { left: true, warn: 'diamond', mailbox: true, wood: true, delin: 'au' });
  if (box(24, 46, 122, 146)) return R('jp', { left: true, warn: 'diamond' });
  if (box(6, 36, 68, 97) || box(-11, 21, 95, 141) || box(-35, -15, 11, 41) || box(-5, 5, 33, 42)) return R('left', { left: true });
  return R('eu', {});
}
const REGION = regionFor(START_LAT, START_LON);
const FULL_RADIUS = 1;   // tiles around the car with full detail + roads (3x3)
const FAR_RADIUS = 8;    // low-detail terrain rings out to the horizon (17x17, ~13-20 km), coarser with distance
const CLASS_LIST = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'minor', 'busway', 'raceway', 'service', 'track', 'rail', 'path'];
const SURF = { asphalt: 0, gravel: 1, dirt: 2, grass: 3, mud: 4, water: 5 };
const SURF_NAMES = ['asphalt', 'gravel', 'dirt', 'grass', 'mud', 'water'];

// ------------------------------------------------ local metric coordinates
// x = east, z = south, y = up, meters; origin at the start location (Web Mercator).
const lonToU = lon => (lon + 180) / 360;
const latToV = lat => { const r = lat * Math.PI / 180; return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2; };
const U0 = lonToU(START_LON), V0 = latToV(START_LAT);
const SCALE = 40075016.686 * Math.cos(START_LAT * Math.PI / 180);
const TILE_SIZE = SCALE / NT, CELL = TILE_SIZE / 256;

const tileAt = (x, z) => [Math.floor((U0 + x / SCALE) * NT), Math.floor((V0 + z / SCALE) * NT)];
const tileOrigin = (tx, ty) => [(tx / NT - U0) * SCALE, (ty / NT - V0) * SCALE];
const toLatLon = (x, z) => {
  const u = U0 + x / SCALE, v = V0 + z / SCALE;
  return [Math.atan(Math.sinh(Math.PI * (1 - 2 * v))) * 180 / Math.PI, u * 360 - 180];
};

// ------------------------------------------------ renderer / scene
const renderer = new THREE.WebGLRenderer({ canvas: CW.gl.canvas, context: CW.gl, antialias: false });
renderer.autoClear = true;
renderer.setClearColor(0x000000, 0);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const SKY = new THREE.Color(0xa9c8e6);
const scene = new THREE.Scene();
scene.background = null;
// haze starts past the detailed tiles and closes in just before the last terrain ring
scene.fog = null;

const camera = new THREE.PerspectiveCamera(65, 1, 0.3, TILE_SIZE * (FAR_RADIUS + 1.5) * 1.42);

const hemi = new THREE.HemisphereLight(0xdfefff, 0x4a5a30, 0.85);
scene.add(hemi);   // sky fill; kept below the sun so shadows read
const sun = new THREE.DirectionalLight(0xfff0d8, 3.1);
const SUN_DIR = new THREE.Vector3(-0.62, 0.55, 0.56).normalize();   // afternoon sun, ~33 degrees up
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: 1, far: 300 });
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.05;
scene.add(sun, sun.target);

// ---- graphics quality. Real ray tracing isn't available to web pages (WebGL/WebGPU have no access to RTX
// hardware), so "High" is the screen-space approximation of what it buys: ambient occlusion (GTAO - soft contact
// shadows in corners, under cars and trees), bloom round the sun and bright highlights, multisampled edges, and a
// sharper sun shadow map over twice the area. The effects are loaded only when High is switched on.
const gfx = { high: false, composer: null, loading: false };
async function setGraphics(high) {
  gfx.high = high;
  try { localStorage.setItem('rrd.gfx', high ? 'high' : 'normal'); } catch (e) {}
  const btn = document.getElementById('gfxBtn');
  if (btn) btn.textContent = high ? 'High' : 'Normal';
  // sun shadows: 4096 px over 120 m on High, 2048 px over 60 m on Normal
  const half = high ? 60 : 30;
  Object.assign(sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.mapSize.set(high ? 4096 : 2048, high ? 4096 : 2048);
  if (sun.shadow.map) { sun.shadow.map.dispose(); sun.shadow.map = null; }
  if (!high) { if (gfx.composer) { gfx.composer.dispose(); gfx.composer = null; } return; }
  if (gfx.composer || gfx.loading) return;
  gfx.loading = true;
  try {
    const [{ EffectComposer }, { RenderPass }, { GTAOPass }, { UnrealBloomPass }, { OutputPass }] = await Promise.all([
      import('three/addons/postprocessing/EffectComposer.js'), import('three/addons/postprocessing/RenderPass.js'),
      Promise.resolve({ GTAOPass: null }), import('three/addons/postprocessing/UnrealBloomPass.js'),
      import('three/addons/postprocessing/OutputPass.js')]);
    if (!gfx.high) return;
    const w = innerWidth, h = innerHeight, pr = renderer.getPixelRatio();
    const rt = new THREE.WebGLRenderTarget(w * pr, h * pr, { type: THREE.HalfFloatType, samples: 4 });
    const comp = new EffectComposer(renderer, rt);
    comp.addPass(new RenderPass(scene, camera));
    // (ambient occlusion was tried here and dropped: it read the riverbed under see-through water as a deep hole
    // and painted every river black)
    comp.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), 0.32, 0.5, 0.92));
    comp.addPass(new OutputPass());
    gfx.composer = comp;
  } catch (e) { console.warn('high graphics unavailable', e); gfx.high = false; if (btn) btn.textContent = 'Normal'; }
  finally { gfx.loading = false; }
}
addEventListener('resize', () => { if (gfx.composer) gfx.composer.setSize(innerWidth, innerHeight); });
{ let saved = null; try { saved = localStorage.getItem('rrd.gfx'); } catch (e) {} if (saved === 'high') setTimeout(() => setGraphics(true), 0); }

// ---- the sky: a dome that always sits on the camera. Haze at the horizon (the fog colour, so distant land melts
// into it) deepening to blue overhead, the sun as a bright disc with a glow, and a layer of drifting clouds ~1.8 km up
// lit from the sun's side. Drawn first and never written to depth, so everything else goes in front of it.
const skyU = { uSun: { value: SUN_DIR }, uHaze: { value: SKY }, uTime: { value: 0 }, uCover: { value: 0.5 } };
const skyDome = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.ShaderMaterial({
  uniforms: skyU, side: THREE.BackSide, depthWrite: false, depthTest: false, fog: false,
  vertexShader: /* glsl */`
    varying vec3 vDir;
    void main() {
      vDir = position;
      // centred on the camera, just inside the far plane (the model matrix is ignored)
      vec4 p = projectionMatrix * viewMatrix * vec4(cameraPosition + position * 1000.0, 1.0);
      gl_Position = vec4(p.xy, p.w * 0.9999, p.w);
    }`,
  fragmentShader: /* glsl */`
    uniform vec3 uSun, uHaze; uniform float uTime, uCover;
    varying vec3 vDir;
    float h21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
    float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(mix(h21(i), h21(i + vec2(1, 0)), u.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), u.x), u.y); }
    float fbm(vec2 p) { float s = 0.0, a = 0.5; mat2 r = mat2(1.6, 1.2, -1.2, 1.6); for (int i = 0; i < 6; i++) { s += a * vnoise(p); p = r * p; a *= 0.5; } return s; }
    void main() {
      vec3 d = normalize(vDir);
      float up = max(d.y, 0.0);
      // sky gradient: haze at the horizon, clearer blue overhead
      vec3 zenith = vec3(0.10, 0.27, 0.62), mid = vec3(0.30, 0.52, 0.82);
      vec3 col = mix(uHaze, mid, smoothstep(0.0, 0.25, up));
      col = mix(col, zenith, smoothstep(0.25, 1.0, up));
      // the sun: a glow round it and a hot disc
      float cs = max(dot(d, uSun), 0.0);
      col += vec3(1.0, 0.85, 0.6) * (pow(cs, 8.0) * 0.18 + pow(cs, 64.0) * 0.35);
      // clouds on a layer 1.8 km up, drifting with the wind
      if (d.y > 0.01) {
        float H = max(2400.0 - cameraPosition.y, 700.0);          // cloud base ~2.4 km, always well above you
        vec2 p = (cameraPosition.xz + d.xz * (H / d.y)) / 2600.0 + vec2(uTime * 0.0035, uTime * 0.0012);
        float n = fbm(p);
        float c = smoothstep(1.0 - uCover - 0.08, 1.0 - uCover + 0.22, n);
        // light: lit edges towards the sun, darker thick middles; silver lining near the sun
        float thick = smoothstep(0.4, 0.95, n);
        float lit = fbm(p + uSun.xz * 0.06) ;
        vec3 cc = mix(vec3(0.97, 0.97, 0.98), vec3(0.62, 0.66, 0.72), thick * 0.8);
        cc *= 0.92 + 0.25 * clamp((n - lit) * 6.0, -1.0, 1.0);
        cc += vec3(1.0, 0.9, 0.75) * pow(cs, 12.0) * 0.5 * (1.0 - thick);
        // far clouds thin out into the haze at the horizon
        float fade = smoothstep(0.02, 0.22, d.y);
        col = mix(col, mix(uHaze, cc, fade), c * fade * 0.95);
      }
      float disc = smoothstep(0.99985, 0.99995, cs);
      col = mix(col, vec3(1.0, 0.98, 0.92) * 1.6, disc);
      gl_FragColor = vec4(col, 1.0);
      #include <colorspace_fragment>
    }`,
}));
skyDome.frustumCulled = false;
skyDome.renderOrder = -1000;
skyDome.onBeforeRender = () => { skyU.uTime.value = performance.now() / 1000; };
scene.add(skyDome);

function detailTexture() {
  const s = 256, cv = document.createElement('canvas');
  cv.width = cv.height = s;
  const g = cv.getContext('2d');
  g.fillStyle = '#d9d9d9';
  g.fillRect(0, 0, s, s);
  for (let i = 0; i < 9000; i++) {
    g.fillStyle = `hsl(0, 0%, ${62 + Math.random() * 38}%)`;
    g.fillRect(Math.random() * s, Math.random() * s, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
const terrainMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, map: detailTexture(), roughness: 1 });

// ===================================================================== ground textures + shader
// Land classes (same ids as the worker). Layers 0..6 of the detail texture array; 7 = water (photo only).
const LAND = ['grass', 'crop field', 'bare soil', 'forest', 'mud', 'gravel / rock', 'urban', 'water', 'sand', 'snow'];
const L_GRASS = 0, L_CROP = 1, L_SOIL = 2, L_FOREST = 3, L_MUD = 4, L_GRAVEL = 5, L_URBAN = 6, L_WATER = 7, L_SAND = 8, L_SNOW = 9;
// CC0 textures from Poly Haven, with their real-world size in meters
const DETAIL_SETS = [
  ['leafy_grass', 2.0], ['sparse_grass', 2.0], ['farm_soil', 2.0], ['forest_floor', 2.14],
  ['brown_mud', 1.3], ['gravel_ground_01', 3.0], ['concrete_pavement', 1.8], ['rock_face', 4.0],
  ['sand_01', 1.5], ['snow_02', 2.0],   // layers 8, 9: sand (deserts, dunes, beaches), snow and ice
];
const DETAIL_RES = 1024;
const shared = {
  uDetail: { value: new THREE.DataArrayTexture(new Uint8Array(4 * DETAIL_SETS.length).fill(128), 1, 1, DETAIL_SETS.length) },
  uMean: { value: DETAIL_SETS.map(() => new THREE.Vector3(0.2, 0.2, 0.2)) },
  uScale: { value: DETAIL_SETS.map(d => d[1]) },
  uDetailOn: { value: 0 },
  uDetailN: { value: new THREE.DataArrayTexture(new Uint8Array(4 * DETAIL_SETS.length).fill(255).map((v, i) => i % 4 < 2 ? 128 : v), 1, 1, DETAIL_SETS.length) },
  uDetailNOn: { value: 0 },
  uFade: { value: new THREE.Vector2(140, 650) },  // detail sharp until 140 m, pure photo by 650 m
  uTileSize: { value: TILE_SIZE },
};
shared.uDetail.value.needsUpdate = true;
shared.uDetailN.value.needsUpdate = true;

// Load a set of Poly Haven CC0 colour maps into one texture array; returns per-layer mean colour (linear)
async function loadTextureArray(sets, kind = 'diff') {
  const S = DETAIL_RES, layer = S * S * 4;
  const data = new Uint8Array(layer * sets.length);
  const ctx = new OffscreenCanvas(S, S).getContext('2d', { willReadFrequently: true });
  const toLin = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const bitmaps = await Promise.all(sets.map(async ([name]) => {
    const res = await fetch(`https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/${name}/${name}_${kind}_1k.jpg`);
    return createImageBitmap(await res.blob());
  }));
  const means = bitmaps.map((bmp, i) => {
    ctx.drawImage(bmp, 0, 0, S, S);
    const px = ctx.getImageData(0, 0, S, S).data;
    data.set(px, i * layer);
    let r = 0, g = 0, b = 0;
    for (let k = 0; k < px.length; k += 4 * 7) { r += toLin(px[k]); g += toLin(px[k + 1]); b += toLin(px[k + 2]); }
    const n = px.length / (4 * 7);
    return new THREE.Vector3(r / n, g / n, b / n);
  });
  const tex = new THREE.DataArrayTexture(data, S, S, sets.length);
  tex.format = THREE.RGBAFormat;
  tex.colorSpace = kind === 'diff' ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  tex.needsUpdate = true;
  return { tex, means };
}
loadTextureArray(DETAIL_SETS).then(({ tex, means }) => {
  means.forEach((m, i) => shared.uMean.value[i].copy(m));
  shared.uDetail.value = tex;
  shared.uDetailOn.value = 1;
}).catch(e => console.warn('detail textures failed', e));
loadTextureArray(DETAIL_SETS, 'nor_gl').then(({ tex }) => {
  shared.uDetailN.value = tex;
  shared.uDetailNOn.value = 1;
}).catch(e => console.warn('detail normals failed', e));

function dataTexture(data, size, srgb, mips) {
  const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.magFilter = THREE.LinearFilter;
  if (mips) { t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter; t.anisotropy = renderer.capabilities.getMaxAnisotropy(); }
  else { t.generateMipmaps = false; t.minFilter = THREE.LinearFilter; }
  t.needsUpdate = true;
  return t;
}
const blankWeights = dataTexture(new Uint8Array(4), 1, false, false);

const GROUND_VERT_HEAD = /* glsl */`
  uniform float uTileSize;
  varying vec3 vWorldP;
  varying vec2 vTileUv;
`;
const GROUND_FRAG_HEAD = /* glsl */`
  uniform sampler2D uPhoto, uWA, uWB, uWC;
  uniform mediump sampler2DArray uDetail, uDetailN;
  uniform vec3 uMean[10];
  uniform float uScale[10];
  uniform float uDetailOn, uDetailNOn, uHasClasses;
  uniform vec2 uFade;
  uniform float uTileSize;
  varying vec3 vWorldP;
  varying vec2 vTileUv;
  const mat2 ROT = mat2(0.8, -0.6, 0.6, 0.8);
  vec3 gTri;   // triplanar weights from the surface normal (x: east-facing, y: up, z: south-facing)

  // one projection: two rotated scales to hide tiling
  vec3 planar(int i, vec2 p) {
    float s = uScale[i];
    vec3 a = texture(uDetail, vec3(p / s, float(i))).rgb;
    vec3 b = texture(uDetail, vec3(ROT * p / (s * 3.3) + 0.37, float(i))).rgb;
    return mix(a, b, 0.35);
  }
  // triplanar on slopes so hillsides get the same sharp detail instead of stretched smears
  vec3 detailTex(int i) {
    if (gTri.y > 0.97) return planar(i, vWorldP.xz);
    vec3 c = vec3(0.0);
    if (gTri.y > 0.01) c += gTri.y * planar(i, vWorldP.xz);
    if (gTri.x > 0.01) c += gTri.x * planar(i, vWorldP.zy);
    if (gTri.z > 0.01) c += gTri.z * planar(i, vWorldP.xy);
    return c;
  }
  // tangent-space normal (x = east, y = south) from the top-down projection
  vec3 detailNor(int i, vec2 p) {
    float s = uScale[i];
    vec3 a = texture(uDetailN, vec3(p / s, float(i))).xyz * 2.0 - 1.0;
    vec3 b = texture(uDetailN, vec3(ROT * p / (s * 3.3) + 0.37, float(i))).xyz * 2.0 - 1.0;
    b.xy = transpose(ROT) * b.xy;
    return normalize(vec3(a.xy + b.xy * 0.5, a.z));
  }
  // crop rows: procedural soil ridges (30-inch rows) with plant rows whose width follows canopy cover (NDVI)
  bool gAlongX;
  float gFurrow;   // ridge slope across the rows, for the lighting normal
  vec3 cropRatio(float ndviByte) {
    float ndvi = ndviByte * 2.0 - 1.0;
    float amt = clamp((ndvi - 0.15) / 0.5, 0.0, 1.0);
    vec2 p = gAlongX ? vWorldP.xz : vWorldP.zx;
    float rc = p.y / 0.76, ph = fract(rc);
    float fw = fwidth(rc);
    float fadeR = clamp(fw * 2.5 - 0.25, 0.0, 1.0);                 // ridges smaller than a pixel fade out (no moire)
    float ridge = mix(0.5 + 0.5 * cos((ph - 0.5) * 6.2831853), 0.5, fadeR);
    gFurrow = -sin((ph - 0.5) * 6.2831853) * (1.0 - fadeR);
    float hw = mix(0.05, 0.5, amt), on = step(0.02, amt);
    float aa = fw + 0.02;
    float mask = (1.0 - smoothstep(hw - aa, hw + aa, abs(ph - 0.5))) * on;
    vec3 soil = planar(2, p) * (0.78 + 0.4 * ridge);
    vec3 plant = planar(1, p * 1.3);
    vec3 c = mix(soil, plant, mask);
    vec3 cm = mix(uMean[2] * 0.98, uMean[1], min(1.0, 2.0 * hw) * on);
    return c / max(cm, vec3(0.01));
  }
`;
const GROUND_FRAG_MAIN = /* glsl */`
  float camDist = distance(vWorldP, cameraPosition);
  // near the camera the tint comes from a blurred photo, so its ~2 m pixels never show; the CC0 texture
  // provides all the fine detail there. Further away the sharp photo takes over.
  float nearK = 1.0 - smoothstep(40.0, 260.0, camDist);
  vec4 photo = texture(uPhoto, vTileUv, 2.2 * nearK * uDetailOn * uHasClasses);
  vec3 groundCol = photo.rgb;
  float fade = (1.0 - smoothstep(uFade.x, uFade.y, camDist)) * uDetailOn * uHasClasses;
  vec3 gN = vec3(0.0, 0.0, 1.0);
  float gNs = 0.0;
  if (fade > 0.002) {
    vec3 Nw0 = normalize((vec4(vNormal, 0.0) * viewMatrix).xyz);
    gTri = pow(abs(Nw0), vec3(4.0));
    gTri /= gTri.x + gTri.y + gTri.z;
    vec2 cell = floor(vWorldP.xz / 400.0);
    gAlongX = mod(cell.x + cell.y * 3.0, 2.0) > 0.5;      // fields mostly follow the N-S / E-W survey grid
    // warp the class map a little with texture noise so boundaries look organic, not like grid cells
    vec2 wn = texture(uDetail, vec3(vWorldP.xz / 11.0, 2.0)).rg / max(uMean[2].rg, vec2(0.01)) - 1.0;
    vec2 wuv = vTileUv + wn * (3.5 / uTileSize);
    vec4 wa = texture2D(uWA, wuv), wb = texture2D(uWB, wuv), wc = texture2D(uWC, wuv);
    gFurrow = 0.0;
    // steep ground (cliffs, road cuts, scree) turns to bare rock: weight slot 8, texture layer 7
    float rockW = smoothstep(0.17, 0.32, 1.0 - Nw0.y) * (1.0 - wb.a);   // ~34° starts, ~47° fully rock
    // slots: 0-7 land classes, 8 rock (steep), 9 sand, 10 snow; texture layers: rock 7, sand 8, snow 9
    float ws[11] = float[11](wa.r, wa.g, wa.b, wa.a, wb.r, wb.g, wb.b, wb.a, rockW, wc.r, wc.g);
    for (int i = 0; i < 11; i++) if (i != 8) ws[i] *= 1.0 - rockW;
    vec3 r[11];
    float sc[11];
    float best = -1.0;
    for (int i = 0; i < 11; i++) {
      r[i] = vec3(1.0);
      sc[i] = -1.0;
      if (i == 7 || ws[i] < 0.01) continue;
      int lay = i == 8 ? 7 : i > 8 ? i - 1 : i;
      r[i] = (i == 1 && gTri.y > 0.8) ? cropRatio(photo.a) : detailTex(lay) / max(uMean[lay], vec3(0.01));
      // gently height-aware blend: raised texels of a material win a little at boundaries
      sc[i] = ws[i] + 0.25 * (dot(r[i], vec3(0.3, 0.59, 0.11)) - 1.0);
      best = max(best, sc[i]);
    }
    sc[7] = ws[7];
    best = max(best, sc[7]);
    vec3 acc = vec3(0.0), nacc = vec3(0.0);
    float wsum = 0.0, snowW = 0.0;
    for (int i = 0; i < 11; i++) {
      float w = max(sc[i] - best + 0.35, 0.0);
      if (w <= 0.0) continue;
      wsum += w;
      if (i == 10) snowW = w;
      if (i != 7) {
        acc += w * r[i];
        if (i == 1 && gTri.y > 0.8) nacc += w * normalize(gAlongX ? vec3(0.0, -0.45 * gFurrow, 1.0) : vec3(-0.45 * gFurrow, 0.0, 1.0));
        else if (uDetailNOn > 0.5) nacc += w * detailNor(i == 8 ? 7 : i > 8 ? i - 1 : i, vWorldP.xz);
      } else {
        acc += w * vec3(1.0);
        nacc += w * vec3(0.0, 0.0, 1.0);
      }
    }
    vec3 ratio = acc / max(wsum, 1e-4);
    groundCol = mix(groundCol, groundCol * clamp(ratio, 0.45, 1.9), fade);
    // snow: summer photos show grey, dirty ice - lift it towards clean blue-white, keeping the texture's relief
    snowW /= max(wsum, 1e-4);
    if (snowW > 0.01) groundCol = mix(groundCol, vec3(0.9, 0.93, 0.98) * clamp(ratio, 0.75, 1.15), snowW * 0.6 * fade);
    if (wsum > 0.0 && uDetailNOn > 0.5) { gN = normalize(nacc / wsum); gNs = fade * gTri.y; }
  }
  diffuseColor.rgb *= groundCol * 0.72;   // photos already contain sunlight; keep them from washing out
`;
// perturb the lighting normal with the blended detail normal (world-aligned tangent frame)
const GROUND_NORMAL = /* glsl */`
  #include <normal_fragment_maps>
  if (gNs > 0.0) {
    vec3 Nw = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
    vec3 Tw = normalize(vec3(1.0, 0.0, 0.0) - Nw * Nw.x);
    vec3 Bw = cross(Tw, Nw);
    vec3 dn = normalize(vec3(gN.xy * 1.1, gN.z));
    vec3 nW = normalize(mix(Nw, Tw * dn.x + Bw * dn.y + Nw * dn.z, gNs));
    normal = normalize((viewMatrix * vec4(nW, 0.0)).xyz);
  }
`;

/** Per-tile ground material: aerial photo far away, classified CC0 textures up close. */
function groundMaterial(photoTex, waTex, wbTex, wcTex) {
  const m = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0 });
  const own = { uPhoto: { value: photoTex }, uWA: { value: waTex || blankWeights }, uWB: { value: wbTex || blankWeights }, uWC: { value: wcTex || blankWeights }, uHasClasses: { value: waTex ? 1 : 0 } };
  m.customProgramCacheKey = () => 'ground-v8';
  m.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, shared, own);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n' + GROUND_VERT_HEAD)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n vWorldP = (modelMatrix * vec4(position, 1.0)).xyz;\n vTileUv = position.xz / uTileSize;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + GROUND_FRAG_HEAD)
      .replace('#include <map_fragment>', GROUND_FRAG_MAIN)
      .replace('#include <normal_fragment_maps>', GROUND_NORMAL);
  };
  m.userData.textures = [photoTex, waTex, wbTex, wcTex].filter(Boolean);
  return m;
}

// ===================================================================== road shaders
// Road surfaces: CC0 asphalt / gravel road / dirt at real-world scale, matched to the aerial photo,
// with lane markings drawn per pixel from each road's lane layout.
const ROAD_SETS = [['asphalt_02', 3.0, 0], ['gravel_road', 2.0, 1], ['dirt', 2.0, 1]]; // name, meters, photo hue tint
const IN_US = START_LAT > 24.4 && START_LAT < 49.5 && START_LON > -125 && START_LON < -66.8;
const roadShared = {
  uRoadTex: { value: new THREE.DataArrayTexture(new Uint8Array(4 * 3).fill(60), 1, 1, 3) },
  uRoadMean: { value: ROAD_SETS.map(() => new THREE.Vector3(0.05, 0.05, 0.05)) },
  uRoadScale: { value: ROAD_SETS.map(r => r[1]) },
  uRoadHue: { value: ROAD_SETS.map(r => r[2]) },
  uRoadOn: { value: 0 },
  uRoadN: { value: new THREE.DataArrayTexture(new Uint8Array(4 * 3).fill(255).map((v, i) => i % 4 < 2 ? 128 : v), 1, 1, 3) },
  uRoadNOn: { value: 0 },
  uUS: { value: REGION.yellow ? 1 : 0 },   // yellow centre lines (the Americas)
};
roadShared.uRoadTex.value.needsUpdate = true;
roadShared.uRoadN.value.needsUpdate = true;
loadTextureArray(ROAD_SETS).then(({ tex, means }) => {
  means.forEach((m, i) => roadShared.uRoadMean.value[i].copy(m));
  roadShared.uRoadTex.value = tex;
  roadShared.uRoadOn.value = 1;
}).catch(e => console.warn('road textures failed', e));
loadTextureArray(ROAD_SETS, 'nor_gl').then(({ tex }) => {
  roadShared.uRoadN.value = tex;
  roadShared.uRoadNOn.value = 1;
}).catch(e => console.warn('road normals failed', e));

const ROAD_VERT = /* glsl */`
  uniform float uTileSize;
  attribute vec4 rd;   // across (m from left edge), along (m), width (m), marking fade
  attribute vec4 rs;   // surface, lanes, shoulder (m), style bits
  varying vec4 vRd;
  varying vec4 vRs;
  varying vec3 vWorldP;
  varying vec2 vTileUv;
`;
const ROAD_FRAG_HEAD = /* glsl */`
  uniform sampler2D uPhoto;
  uniform float uHasPhoto;
  uniform mediump sampler2DArray uRoadTex, uRoadN;
  uniform float uRoadNOn;
  uniform vec3 uRoadMean[3];
  uniform float uRoadScale[3];
  uniform float uRoadHue[3];
  uniform float uRoadOn, uUS;
  varying vec4 vRd;
  varying vec4 vRs;
  varying vec3 vWorldP;
  varying vec2 vTileUv;

  float lineAA(float d, float hw) { float aa = fwidth(d) * 0.8 + 0.003; return 1.0 - smoothstep(hw - aa, hw + aa, abs(d)); }
  float dashed(float along) { return step(fract(along / 12.0), 0.25); }   // 3 m paint, 9 m gap
`;
const ROAD_FRAG_MAIN = /* glsl */`
  int s = int(vRs.x + 0.5);
  vec3 base;
  vec3 rN = vec3(0.0, 0.0, 1.0);
  float rNs = 0.0;
  if (uRoadOn > 0.5) {
    vec2 p = vWorldP.xz;
    float sc = uRoadScale[s];
    vec3 a = texture(uRoadTex, vec3(p / sc, float(s))).rgb;
    vec3 b = texture(uRoadTex, vec3(mat2(0.8, -0.6, 0.6, 0.8) * p / (sc * 3.1) + 0.21, float(s))).rgb;
    base = mix(a, b, 0.35);
    if (uRoadNOn > 0.5) {
      vec3 na = texture(uRoadN, vec3(p / sc, float(s))).xyz * 2.0 - 1.0;
      rN = normalize(vec3(na.xy * 1.3, na.z));
      rNs = 1.0 - smoothstep(60.0, 250.0, distance(vWorldP, cameraPosition));
    }
  } else {
    base = s == 0 ? vec3(0.035) : s == 1 ? vec3(0.28, 0.24, 0.18) : vec3(0.2, 0.14, 0.08);
  }
  // match the real road's brightness (and, for unpaved roads, colour) from the aerial photo
  if (uHasPhoto > 0.5 && uRoadOn > 0.5) {
    vec3 ph = texture2D(uPhoto, vTileUv).rgb * 0.72;
    vec3 m = uRoadMean[s];
    vec3 rc = ph / max(m, vec3(0.004));
    float rl = dot(ph, vec3(0.2126, 0.7152, 0.0722)) / max(dot(m, vec3(0.2126, 0.7152, 0.0722)), 0.004);
    base *= clamp(mix(vec3(1.0), mix(vec3(rl), rc, uRoadHue[s]), 0.55), 0.6, 1.6);
  }
  // lane markings
  float across = vRd.x, W = vRd.z, fade = vRd.w, along = vRd.y;
  float lanes = max(vRs.y, 1.0), sh = vRs.z, style = vRs.w;
  float centre = mod(style, 4.0);
  bool edges = mod(floor(style / 4.0), 2.0) > 0.5;
  bool oneway = style >= 8.0;
  float white = 0.0, yellow = 0.0;
  if (s == 0 && fade > 0.01) {
    float lw = (W - 2.0 * sh) / lanes;
    if (edges) {
      white = max(white, lineAA(across - (sh + 0.1), 0.075));
      white = max(white, lineAA(across - (W - sh - 0.1), 0.075));
    }
    if (oneway) {
      for (int i = 1; i < 4; i++) {
        if (float(i) >= lanes) break;
        white = max(white, lineAA(across - (sh + float(i) * lw), 0.06) * dashed(along));
      }
    } else {
      float c = W * 0.5, cm = 0.0;
      if (centre > 1.5) cm = max(lineAA(across - c - 0.13, 0.055), lineAA(across - c + 0.13, 0.055));
      else if (centre > 0.5) cm = lineAA(across - c, 0.06) * dashed(along);
      if (uUS > 0.5) yellow = cm; else white = max(white, cm);
      float perSide = floor(lanes * 0.5);
      for (int i = 1; i < 3; i++) {
        if (float(i) >= perSide) break;
        white = max(white, lineAA(across - (c - float(i) * lw), 0.06) * dashed(along));
        white = max(white, lineAA(across - (c + float(i) * lw), 0.06) * dashed(along));
      }
    }
    float wear = 0.8 + 0.2 * smoothstep(0.2, 0.6, texture(uRoadTex, vec3(vWorldP.xz / 0.7, 0.0)).g * 4.0);
    base = mix(base, vec3(0.72), white * fade * wear);
    base = mix(base, vec3(0.72, 0.45, 0.04), yellow * fade * wear);
  }
  diffuseColor.rgb *= base;
`;

const VERGE_FRAG_HEAD = /* glsl */`
  uniform sampler2D uPhoto;
  uniform float uHasPhoto;
  uniform mediump sampler2DArray uDetail;
  uniform vec3 uMean[8];
  uniform float uScale[8];
  uniform float uDetailOn;
  varying float vVt;
  varying vec3 vWorldP;
  varying vec2 vTileUv;
`;
const VERGE_FRAG_MAIN = /* glsl */`
  vec3 col = uHasPhoto > 0.5 ? texture2D(uPhoto, vTileUv).rgb * 0.72 : vec3(0.2, 0.17, 0.13);
  float grit = 0.5;
  if (uDetailOn > 0.5) {
    vec3 g = texture(uDetail, vec3(vWorldP.xz / uScale[5], 5.0)).rgb;
    col *= clamp(g / max(uMean[5], vec3(0.01)), 0.3, 2.2) * 0.9;
    grit = texture(uDetail, vec3(vWorldP.xz / 0.9, 5.0)).g / max(uMean[5].g, 0.01) * 0.5;
  }
  // gravel edge that thins out into the ground, with a ragged outline
  float a = pow(1.0 - clamp(vVt, 0.0, 1.0), 1.3) * clamp(0.55 + 0.6 * grit, 0.0, 1.3);
  diffuseColor.rgb *= col;
  diffuseColor.a = clamp(a, 0.0, 1.0);
`;

/** Per-tile road + verge materials; own.uPhoto / own.uHasPhoto follow the tile's aerial photo. */
function roadMaterials(own) {
  const common = sh => sh.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
    vWorldP = (modelMatrix * vec4(position, 1.0)).xyz;
    vTileUv = position.xz / uTileSize;`);
  const road = new THREE.MeshStandardMaterial({ roughness: 0.88, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
  road.customProgramCacheKey = () => 'road-v2';
  road.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, roadShared, own, { uTileSize: shared.uTileSize });
    sh.vertexShader = common(sh).replace('#include <common>', '#include <common>\n' + ROAD_VERT)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n vRd = rd; vRs = rs;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + ROAD_FRAG_HEAD)
      .replace('#include <map_fragment>', ROAD_FRAG_MAIN)
      .replace('#include <normal_fragment_maps>', GROUND_NORMAL.replace(/gNs/g, 'rNs').replace(/gN/g, 'rN'));
  };
  const verge = new THREE.MeshStandardMaterial({ roughness: 1, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
  verge.customProgramCacheKey = () => 'verge-v2';
  verge.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, shared, own);
    sh.vertexShader = common(sh).replace('#include <common>', '#include <common>\n uniform float uTileSize; attribute float vt; varying float vVt; varying vec3 vWorldP; varying vec2 vTileUv;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n vVt = vt;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + VERGE_FRAG_HEAD)
      .replace('#include <map_fragment>', VERGE_FRAG_MAIN);
  };
  return { road, verge };
}

// ===================================================================== tiles

const worker = new Worker(CW.workerUrl);
let tileUrl = null;
const tileJsonP = fetch('https://tiles.openfreemap.org/planet').then(r => r.json()).then(j => j.tiles[0]);

// Register Planetary Computer mosaic searches (photo, NDVI and land-class sources)
const PC_MOSAIC = 'https://planetarycomputer.microsoft.com/api/data/v1/mosaic/register';
const IMG = { ids: {}, us: START_LAT > 24.4 && START_LAT < 49.5 && START_LON > -125 && START_LON < -66.8 };
async function registerSearch(body) {
  const r = await fetch(PC_MOSAIC, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(r.status);
  return (await r.json()).searchid;
}
const newest = [{ field: 'datetime', direction: 'desc' }];
const today = new Date(), s2Year = today.getUTCMonth() >= 9 ? today.getUTCFullYear() : today.getUTCFullYear() - 1;
// growing-season window by latitude: northern summer, southern summer, or (in the cloudy tropics) any time over the
// last two years with a looser cloud limit, least cloudy first
const s2Window = Math.abs(START_LAT) < 23 ? { from: `${today.getUTCFullYear() - 2}-01-01`, to: today.toISOString().slice(0, 10), cloud: 30 }
  : START_LAT < 0 ? { from: `${s2Year - 1}-11-15`, to: `${s2Year}-03-31`, cloud: 10 }
  : { from: `${s2Year}-05-15`, to: `${s2Year}-09-30`, cloud: 10 };
const searches = {
  naip: { collections: ['naip'], sortby: newest },
  cdl: { collections: ['usda-cdl'], sortby: newest, 'filter-lang': 'cql2-json', filter: { op: '=', args: [{ property: 'usda_cdl:type' }, 'cropland'] } },
  s2: { collections: ['sentinel-2-l2a'], sortby: [{ field: 'eo:cloud_cover', direction: 'asc' }], 'filter-lang': 'cql2-json',
        filter: { op: 'and', args: [
          { op: '<=', args: [{ property: 'eo:cloud_cover' }, s2Window.cloud] },
          { op: 'anyinteracts', args: [{ property: 'datetime' }, { interval: [`${s2Window.from}T00:00:00Z`, `${s2Window.to}T23:59:59Z`] }] },
        ] } },
  wc: { collections: ['esa-worldcover'], sortby: newest },
};
if (!IMG.us) { delete searches.naip; delete searches.cdl; }
const imageryP = Promise.allSettled(Object.entries(searches).map(async ([k, body]) => { IMG.ids[k] = await registerSearch(body); }));
Promise.all([tileJsonP, Promise.race([imageryP, new Promise(r => setTimeout(r, 6000))])]).then(([u]) => { tileUrl = u; });

const tiles = new Map(); // "tx,ty" -> tile
let reqId = 0, inFlight = 0;
const MAX_IN_FLIGHT = 5;

function geometry(d, uv) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(d.positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(d.normals, 3));
  g.setAttribute('color', new THREE.BufferAttribute(d.colors, 3));
  if (uv) g.setAttribute('uv', new THREE.BufferAttribute(d.uvs, 2));
  g.setIndex(new THREE.BufferAttribute(d.indices, 1));
  return g;
}

function disposeMeshes(t) {
  for (const m of [t.terrainMesh, t.roadMesh, t.vergeMesh, t.waterMesh, t.buildingMesh, t.structMesh]) if (m) { scene.remove(m); m.geometry.dispose(); }
  t.terrainMesh = t.roadMesh = t.vergeMesh = t.waterMesh = t.buildingMesh = t.structMesh = null;
  t.footHash = null;
}
function disposeTile(t) {
  disposeMeshes(t);
  if (t.driveMesh) { scene.remove(t.driveMesh); t.driveMesh.geometry.dispose(); t.driveMesh = null; }
  if (t.props) { t.props = null; t.propCol = null; propsDirty = true; }
  if (t.walkMesh) { scene.remove(t.walkMesh); t.walkMesh.geometry.dispose(); t.walkMesh = null; }
  disposeGround(t);
  if (t.roadMats) { t.roadMats.road.dispose(); t.roadMats.verge.dispose(); t.roadMats = null; }
  if (t.buildingMat) { t.buildingMat.dispose(); t.buildingMat = null; }
}
function disposeGround(t) {
  if (!t.groundMat) return;
  for (const tex of t.groundMat.userData.textures) tex.dispose();
  t.groundMat.dispose();
  t.groundMat = null;
  t.img = null;
  if (t.trees) { t.trees = null; treesDirty = true; }
}

function applyImagery(t, d) {
  disposeGround(t);
  const photo = dataTexture(d.photo, d.photoSize, true, true);
  const wa = d.wa ? dataTexture(d.wa, 512, false, false) : null;
  const wb = d.wb ? dataTexture(d.wb, 512, false, false) : null;
  const wc = d.wc ? dataTexture(d.wc, 512, false, false) : null;
  t.groundMat = groundMaterial(photo, wa, wb, wc);
  t.img = d.mat ? { photo: d.photo, photoSize: d.photoSize, mat: d.mat, code: d.code, source: d.source, clsKind: d.clsKind } : null;
  if (d.trees) { t.trees = d.trees; t.treeY = null; treesDirty = true; }
  if (t.terrainMesh) t.terrainMesh.material = t.groundMat;
  t.roadU.uPhoto.value = photo;
  t.roadU.uHasPhoto.value = d.source === 'NAIP' ? 1 : 0.75;
  t.imgAt = performance.now();
  if (d.mat) t.propsGen = false;
  detailDirty = true;
}

worker.onmessage = e => {
  const d = e.data;
  const t = tiles.get(`${d.tx},${d.ty}`);
  if (d.type === 'log') { console.warn(d.msg); return; }
  if (d.type === 'buildings') {
    if (t && t.reqId === d.id) applyBuildings(t, d);
    return;
  }
  if (d.type === 'imagery') {
    if (t && t.reqId === d.id) applyImagery(t, d);
    return;
  }
  inFlight--;
  if (!t || t.reqId !== d.id) return; // unloaded or superseded
  t.loading = false;
  if (!d.ok) { console.warn('tile failed', d.tx, d.ty, d.error); t.retryAt = performance.now() + 3000; return; }
  disposeMeshes(t);
  t.lod = d.lod;
  t.H = d.H;
  { let mx = -Infinity; for (let j = M; j <= M + 256; j += 4) for (let i = M; i <= M + 256; i += 4) mx = Math.max(mx, d.H[j * G + i]); t.hMax = mx; }
  t.terrainMesh = new THREE.Mesh(geometry(d.terrain, true), t.groundMat || terrainMaterial);
  t.terrainMesh.receiveShadow = true;
  t.terrainMesh.position.set(t.ox, 0, t.oz);
  scene.add(t.terrainMesh);
  if (d.roads) {
    if (!t.roadMats) t.roadMats = roadMaterials(t.roadU);
    const rg = new THREE.BufferGeometry();
    rg.setAttribute('position', new THREE.BufferAttribute(d.roads.road.positions, 3));
    rg.setAttribute('normal', new THREE.BufferAttribute(d.roads.road.normals, 3));
    rg.setAttribute('rd', new THREE.BufferAttribute(d.roads.road.rd, 4));
    rg.setAttribute('rs', new THREE.BufferAttribute(d.roads.road.rs, 4));
    rg.setIndex(new THREE.BufferAttribute(d.roads.road.indices, 1));
    t.roadMesh = new THREE.Mesh(rg, t.roadMats.road);
    t.roadMesh.receiveShadow = true;
    t.roadMesh.position.set(t.ox, 0, t.oz);
    scene.add(t.roadMesh);
    if (d.roads.verge.indices.length) {
      const vg = new THREE.BufferGeometry();
      vg.setAttribute('position', new THREE.BufferAttribute(d.roads.verge.positions, 3));
      vg.setAttribute('normal', new THREE.BufferAttribute(d.roads.verge.normals, 3));
      vg.setAttribute('vt', new THREE.BufferAttribute(d.roads.verge.vt, 1));
      vg.setIndex(new THREE.BufferAttribute(d.roads.verge.indices, 1));
      t.vergeMesh = new THREE.Mesh(vg, t.roadMats.verge);
      t.vergeMesh.receiveShadow = true;
      t.vergeMesh.position.set(t.ox, 0, t.oz);
      scene.add(t.vergeMesh);
    }
  }
  t.waterLevel = d.water ? d.water.level : null;
  t.waterCurrent = d.water ? d.water.current : null;
  t.junc = d.junc || null; t.stops = d.stops || null; t.rings = d.rings || null; t.propsGen = false;
  t.structCol = null;
  if (d.structures) {
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(d.structures.mesh.positions, 3));
    sg.setAttribute('normal', new THREE.BufferAttribute(d.structures.mesh.normals, 3));
    sg.setAttribute('color', new THREE.BufferAttribute(d.structures.mesh.colors, 3));
    t.structMesh = new THREE.Mesh(sg, structMat);
    t.structMesh.castShadow = t.structMesh.receiveShadow = true;
    t.structMesh.position.set(t.ox, 0, t.oz);
    scene.add(t.structMesh);
    t.structCol = d.structures.col;
  }
  if (d.water && d.water.mesh) {
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', new THREE.BufferAttribute(d.water.mesh.positions, 3));
    wg.setAttribute('depth', new THREE.BufferAttribute(d.water.mesh.depth, 1));
    wg.setAttribute('wflow', new THREE.BufferAttribute(d.water.mesh.wflow, 4));
    wg.setAttribute('wcover', new THREE.BufferAttribute(d.water.mesh.cover || new Float32Array(d.water.mesh.depth.length).fill(1), 1));
    wg.setIndex(new THREE.BufferAttribute(d.water.mesh.indices, 1));
    t.waterMesh = new THREE.Mesh(wg, waterMat);
    t.waterMesh.position.set(t.ox, 0, t.oz);
    scene.add(t.waterMesh);
  }
  setSegs(t, d.segs);
  if (t.occluded) for (const m of tileMeshes(t)) if (m) m.visible = false;   // still behind a ridge
};

const BUCKETS = 32, BCELL = TILE_SIZE / BUCKETS;
function setSegs(t, segs) {
  t.segs = segs;
  t.buckets = Array.from({ length: BUCKETS * BUCKETS }, () => []);
  const cb = v => Math.max(0, Math.min(BUCKETS - 1, Math.floor(v / BCELL)));
  for (let o = 0; o < segs.length; o += SEG) {
    const hw = segs[o + 4];
    const i0 = cb(Math.min(segs[o], segs[o + 2]) - hw), i1 = cb(Math.max(segs[o], segs[o + 2]) + hw);
    const j0 = cb(Math.min(segs[o + 1], segs[o + 3]) - hw), j1 = cb(Math.max(segs[o + 1], segs[o + 3]) + hw);
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) t.buckets[j * BUCKETS + i].push(o);
  }
}

function updateTiles(x, z) {
  const [cx, cy] = tileAt(x, z);
  for (let dy = -FAR_RADIUS; dy <= FAR_RADIUS; dy++) for (let dx = -FAR_RADIUS; dx <= FAR_RADIUS; dx++) {
    const key = `${cx + dx},${cy + dy}`;
    let t = tiles.get(key);
    if (!t) {
      const [ox, oz] = tileOrigin(cx + dx, cy + dy);
      t = { tx: cx + dx, ty: cy + dy, ox, oz, lod: 0, loading: false, retryAt: 0, H: null, segs: null, buckets: null,
            roadU: { uPhoto: { value: blankWeights }, uHasPhoto: { value: 0 } } };
      tiles.set(key, t);
    }
    t.ring = Math.max(Math.abs(dx), Math.abs(dy));
    // mesh step: coarser further out; far tiles load coarse first and only refine once they can be seen
    // (behind a ridge they keep whatever they have - see updateCulling)
    const fine = t.ring <= FULL_RADIUS ? 1 : t.ring <= 3 ? 4 : t.ring <= 4 ? 8 : 16;
    t.want = fine === 1 ? 1 : !t.lod ? 16 : t.occluded ? t.lod : fine;
  }
  for (const [key, t] of tiles) {
    if (Math.max(Math.abs(t.tx - cx), Math.abs(t.ty - cy)) > FAR_RADIUS + 1) { disposeTile(t); tiles.delete(key); detailDirty = true; }
  }
  if (!tileUrl) return;
  const now = performance.now();
  const todo = [...tiles.values()].filter(t => !t.loading && t.lod !== t.want && now >= t.retryAt).sort((a, b) => a.ring - b.ring);
  for (const t of todo) {
    if (inFlight >= MAX_IN_FLIGHT) break;
    t.loading = true;
    t.reqId = ++reqId;
    inFlight++;
    worker.postMessage({ id: t.reqId, tx: t.tx, ty: t.ty, lod: t.want, tileSize: TILE_SIZE, ox: t.ox, oz: t.oz, url: tileUrl, img: IMG, lms: landmarksForTile(t.ox, t.oz) });
  }
}

// ------------------------------------------------ ground queries

function tileFor(x, z) { const [tx, ty] = tileAt(x, z); return tiles.get(`${tx},${ty}`); }

function terrainHeight(x, z) {
  const t = tileFor(x, z);
  if (!t || !t.H) return null;
  const H = t.H;
  const gx = Math.min(G - 1.0001, Math.max(0, (x - t.ox) / CELL + M));
  const gz = Math.min(G - 1.0001, Math.max(0, (z - t.oz) / CELL + M));
  const i = Math.floor(gx), j = Math.floor(gz), fx = gx - i, fz = gz - j, k = j * G + i;
  const ha = H[k], hb = H[k + 1], hc = H[k + G], hd = H[k + G + 1];
  return fz >= fx ? ha + (hd - hc) * fx + (hc - ha) * fz : ha + (hb - ha) * fx + (hd - hb) * fz;
}

function segDist(s, o, px, pz) {
  const dx = s[o + 2] - s[o], dz = s[o + 3] - s[o + 1];
  const l2 = dx * dx + dz * dz || 1e-9;
  const t = Math.max(0, Math.min(1, ((px - s[o]) * dx + (pz - s[o + 1]) * dz) / l2));
  return [Math.hypot(s[o] + dx * t - px, s[o + 1] + dz * t - pz), t];
}

/**
 * What's under a point: { h, seg, segs } — h includes bridge decks below yRef.
 * seg is the offset of the road segment in t.segs (or -1 for off-road).
 */
const groundHit = { h: 0, seg: -1, segs: null }, _gcand = new Float64Array(3 * 96), _pick = { seg: -1, h: 0, bridge: false };
// Which surface carries the car in tile t: roads and decks overhead (above what the wheels can climb from yRef)
// don't count - a car on a lower road ignores the ramp or deck above it. Of the rest, the TOP level wins (so a car on
// a deck never drops through to the road below), and within that level (surfaces less than 1 m apart: one road
// on a slope, roads meeting at a junction) a deck beats a road, a bigger road a smaller one, then the nearest
// segment (wide roads overlap many short segments; picking another makes the height jump).
function pickSurface(t, lx, lz, yRef) {
  _pick.seg = -1;
  if (!t || !t.buckets) return _pick;
  const s = t.segs;
  const bi = Math.max(0, Math.min(BUCKETS - 1, Math.floor(lx / BCELL)));
  const bj = Math.max(0, Math.min(BUCKETS - 1, Math.floor(lz / BCELL)));
  const lim = yRef + 0.75, cand = _gcand;
  let nc = 0, top = -Infinity;
  for (const o of t.buckets[bj * BUCKETS + bi]) {
    const [d, u] = segDist(s, o, lx, lz);
    if (d > s[o + 4] + 0.1) continue;
    const surfH = s[o + 9] + (s[o + 10] - s[o + 9]) * u;   // the road's own smoothed height profile
    if (surfH > lim) continue;                                // overhead: we're under it
    if (s[o + 8] === 2 && surfH < yRef - 2.5) continue;      // a tunnel far below: we're on the hill above it
    cand[nc++] = o; cand[nc++] = surfH; cand[nc++] = d;
    if (surfH > top) top = surfH;
    if (nc >= cand.length) break;
  }
  let bestRank = -1, bestD = Infinity;
  for (let q = 0; q < nc; q += 3) {
    const o = cand[q], surfH = cand[q + 1], d = cand[q + 2];
    if (surfH < top - 1) continue;
    const rank = s[o + 8] === 1 ? 99 : s[o + 6];
    if (rank > bestRank || (rank === bestRank && d < bestD)) { bestRank = rank; bestD = d; _pick.h = surfH; _pick.seg = o; }
  }
  _pick.bridge = bestRank === 99;
  return _pick;
}
const SEAM_BLEND = 8;
function groundAt(x, z, yRef = Infinity) {
  const th = terrainHeight(x, z);
  if (th === null) return null;
  groundHit.h = th; groundHit.seg = -1; groundHit.segs = null; groundHit.tunnel = false;
  const t = tileFor(x, z);
  if (!t.buckets) return groundHit;
  const lx = x - t.ox, lz = z - t.oz, s = t.segs;
  const pk = pickSurface(t, lx, lz, yRef);
  if (pk.seg >= 0) {
    const bestH = pk.h;
    groundHit.seg = pk.seg; groundHit.segs = s;
    groundHit.h = pk.bridge ? Math.max(th, bestH) : bestH;
    groundHit.tunnel = s[pk.seg + 8] === 2 && th > bestH + 1.5;
    // near a tile border, ease into the neighbour tile's height for the same road: any small disagreement
    // between the two tiles becomes a gentle ramp instead of a step (half-and-half exactly on the border)
    const ex = Math.min(lx, TILE_SIZE - lx), ez = Math.min(lz, TILE_SIZE - lz), e = Math.min(ex, ez);
    if (e < SEAM_BLEND) {
      const nt = ex <= ez ? tiles.get(`${t.tx + (lx < TILE_SIZE / 2 ? -1 : 1)},${t.ty}`) : tiles.get(`${t.tx},${t.ty + (lz < TILE_SIZE / 2 ? -1 : 1)}`);
      if (nt && nt.lod === 1) {
        const own = groundHit.h, seg = groundHit.seg;
        const q = pickSurface(nt, x - nt.ox, z - nt.oz, yRef);
        if (q.seg >= 0 && nt.segs[q.seg + 7] === s[seg + 7] && Math.abs(q.h - own) < 4) groundHit.h = own + (q.h - own) * 0.5 * (1 - e / SEAM_BLEND);
      }
    }
  }
  groundHit.deck = false;
  for (const L of footDecks) {
    const fd = L.footDeck, dx = x - L.px, dz = z - L.pz;
    if (Math.abs(dx) > fd.reach || Math.abs(dz) > fd.reach) continue;
    const mx = dx * Math.cos(L.prot) - dz * Math.sin(L.prot) - fd.cx, mz = dx * Math.sin(L.prot) + dz * Math.cos(L.prot);
    if (Math.abs(mx) > fd.half || Math.abs(mz) > fd.w / 2) continue;
    // flat deck, easing into the ground over the last stretch at each end
    let h = L.y + fd.top;
    const e = fd.half - Math.abs(mx);
    if (e < fd.ramp) { const u = e / fd.ramp, w = u * u * (3 - 2 * u); h = th + (h - th) * w; }
    if (h > yRef + 1.5 || h < groundHit.h) continue;      // overhead (we're under it) or below the ground
    groundHit.h = h; groundHit.deck = true;
  }
  return groundHit;
}

function nearestRoad(x, z, minRank = 3) {
  let best = null, bestD = Infinity;
  for (const t of tiles.values()) {
    if (!t.segs) continue;
    const s = t.segs;
    for (let o = 0; o < s.length; o += SEG) {
      if (s[o + 6] < minRank) continue;
      const [d, u] = segDist(s, o, x - t.ox, z - t.oz);
      if (d < bestD) {
        bestD = d;
        // drop the car in the right-hand lane, facing along the road
        const heading = Math.atan2(s[o + 2] - s[o], s[o + 3] - s[o + 1]);
        const lanes = Math.abs(s[o + 11]) || 1, hw = s[o + 4], lw = 2 * hw / lanes;
        const side = REGION.left ? -1 : 1;   // keep to the region's driving side
        const off = lanes < 2 ? 0 : (s[o + 11] < 0 ? hw - lw * 0.55 : lw * 0.45) * side;
        best = { x: t.ox + s[o] + (s[o + 2] - s[o]) * u - Math.cos(heading) * off,
                 z: t.oz + s[o + 1] + (s[o + 3] - s[o + 1]) * u + Math.sin(heading) * off, heading };
      }
    }
  }
  return best;
}

// ===================================================================== land under a point

const CROP_NAMES = {
  1: 'corn', 2: 'cotton', 3: 'rice', 4: 'sorghum', 5: 'soybeans', 6: 'sunflower', 10: 'peanuts', 11: 'tobacco',
  12: 'sweet corn', 13: 'popcorn', 21: 'barley', 22: 'durum wheat', 23: 'spring wheat', 24: 'winter wheat',
  25: 'small grains', 26: 'winter wheat / soybeans', 27: 'rye', 28: 'oats', 29: 'millet', 31: 'canola', 33: 'safflower',
  36: 'alfalfa', 37: 'hay', 41: 'sugarbeets', 42: 'dry beans', 43: 'potatoes', 44: 'other crops', 45: 'sugarcane',
  49: 'onions', 53: 'peas', 54: 'tomatoes', 58: 'clover', 59: 'sod / grass seed', 61: 'fallow', 66: 'cherries',
  67: 'peaches', 68: 'apples', 69: 'grapes', 72: 'citrus', 74: 'pecans', 75: 'almonds', 76: 'walnuts',
  111: 'open water', 121: 'developed, open space', 122: 'developed, low', 123: 'developed, medium', 124: 'developed, high',
  131: 'barren', 141: 'deciduous forest', 142: 'evergreen forest', 143: 'mixed forest', 152: 'shrubland',
  176: 'grassland / pasture', 190: 'woody wetland', 195: 'herbaceous wetland', 205: 'triticale',
  225: 'winter wheat / corn', 236: 'winter wheat / sorghum', 238: 'winter wheat / cotton', 241: 'corn / soybeans',
};
const LAND_SURFACE = [SURF.grass, SURF.dirt, SURF.dirt, SURF.dirt, SURF.mud, SURF.gravel, SURF.asphalt, SURF.water, SURF.dirt, SURF.mud];

const landHit = {};
/** Classified land + aerial photo colour + NDVI at a world point (null until imagery loads). */
function landAt(x, z) {
  const t = tileFor(x, z);
  if (!t || !t.img) return null;
  const u = (x - t.ox) / TILE_SIZE, v = (z - t.oz) / TILE_SIZE;
  const ci = Math.min(511, Math.max(0, Math.floor(v * 512))) * 512 + Math.min(511, Math.max(0, Math.floor(u * 512)));
  const S = t.img.photoSize;
  const pi = (Math.min(S - 1, Math.max(0, Math.floor(v * S))) * S + Math.min(S - 1, Math.max(0, Math.floor(u * S)))) * 4;
  const p = t.img.photo;
  landHit.mat = t.img.mat[ci];
  landHit.code = t.img.code[ci];
  landHit.r = p[pi]; landHit.g = p[pi + 1]; landHit.b = p[pi + 2];
  landHit.ndvi = p[pi + 3] / 255 * 2 - 1;
  landHit.source = t.img.source;
  landHit.clsKind = t.img.clsKind;
  return landHit;
}
function landName(l) {
  const crop = l.clsKind === 'cdl' ? CROP_NAMES[l.code] : null;
  if (l.mat === L_CROP) return crop ? `${crop} field` : 'crop field';
  return crop && crop !== LAND[l.mat] ? `${LAND[l.mat]} · ${crop}` : LAND[l.mat];
}

// ===================================================================== grass + crop detail meshes

const timeU = { value: 0 };
// a clump of tapered blades (both windings so it lights the same from either side)
function bladeClump(blades, width, spread, lean) {
  const pos = [], col = [], idx = [];
  for (let b = 0; b < blades; b++) {
    const a = Math.random() * Math.PI * 2, r = Math.random() * spread;
    const bx = Math.cos(a) * r, bz = Math.sin(a) * r;
    const fa = Math.random() * Math.PI, fx = Math.cos(fa), fz = Math.sin(fa);
    const h = 0.7 + Math.random() * 0.45, w = width * (0.7 + Math.random() * 0.6);
    const la = Math.random() * Math.PI * 2, lx = Math.cos(la) * lean * h, lz = Math.sin(la) * lean * h;
    const v = pos.length / 3;
    pos.push(
      bx - fx * w / 2, 0, bz - fz * w / 2, bx + fx * w / 2, 0, bz + fz * w / 2,
      bx - fx * w * 0.35 + lx * 0.35, h * 0.55, bz - fz * w * 0.35 + lz * 0.35,
      bx + fx * w * 0.35 + lx * 0.35, h * 0.55, bz + fz * w * 0.35 + lz * 0.35,
      bx + lx, h, bz + lz);
    for (const c of [0.45, 0.45, 0.8, 0.8, 1.1]) col.push(c, c, c);
    idx.push(v, v + 1, v + 3, v, v + 3, v + 2, v + 2, v + 3, v + 4,
             v, v + 3, v + 1, v, v + 2, v + 3, v + 2, v + 4, v + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(pos.map((_, i) => i % 3 === 1 ? 1 : 0), 3));
  g.setIndex(idx);
  return g;
}

const plantMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85 });
plantMat.onBeforeCompile = sh => {
  sh.uniforms.uTime = timeU;
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', '#include <common>\nuniform float uTime;')
    .replace('#include <begin_vertex>', `#include <begin_vertex>
      #ifdef USE_INSTANCING
        vec3 ip = instanceMatrix[3].xyz;
        float sway = sin(uTime * 1.8 + ip.x * 0.31 + ip.z * 0.23) + 0.4 * sin(uTime * 3.7 + ip.x * 1.1 - ip.z * 0.7);
        float k = position.y * position.y;
        transformed.x += sway * 0.07 * k;
        transformed.z += sway * 0.04 * k;
      #endif`);
};

const GRASS_MAX = 9000, CROP_MAX = 16000;
function plantMesh(geo, max) {
  const m = new THREE.InstancedMesh(geo, plantMat, max);
  m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(max * 3), 3);
  m.count = 0;
  m.frustumCulled = false;
  m.receiveShadow = true;
  scene.add(m);
  return m;
}
const grassMesh = plantMesh(bladeClump(7, 0.06, 0.28, 0.25), GRASS_MAX);
const cropMesh = plantMesh(bladeClump(6, 0.13, 0.06, 0.45), CROP_MAX);

// typical mature height (m) by USDA crop code
const CROP_HEIGHT = { 1: 2.3, 12: 2.0, 13: 2.0, 4: 1.3, 6: 1.8, 2: 1.0, 3: 0.9, 5: 0.8, 21: 0.85, 22: 0.85, 23: 0.85,
  24: 0.85, 25: 0.8, 26: 0.8, 27: 1.1, 28: 0.9, 29: 0.9, 205: 0.9, 41: 0.5, 43: 0.5, 42: 0.5, 225: 1.5, 241: 1.5 };
const GRASS_R = 55, GRASS_STEP = 1.0, CROP_R = 42, ROW = 0.76, PLANT = 0.45;
let detailDirty = true, detailX = 1e9, detailZ = 1e9;
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3(), _c = new THREE.Color(), _up = new THREE.Vector3(0, 1, 0);
const hash = (a, b) => { let h = (a * 374761393 + b * 668265263) | 0; h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16; return (h >>> 0) / 4294967296; };
const sstep = (e0, e1, x) => THREE.MathUtils.smoothstep(x, e0, e1);

function placePlant(mesh, n, x, z, y, h, w, rot, l, boost) {
  _p.set(x, y, z);
  _q.setFromAxisAngle(_up, rot);
  _s.set(w, h, w);
  mesh.setMatrixAt(n, _m.compose(_p, _q, _s));
  _c.setRGB(l.r / 255, l.g / 255, l.b / 255, THREE.SRGBColorSpace).multiplyScalar(boost);
  mesh.setColorAt(n, _c);
}

function rebuildDetail(cx, cz) {
  // grass: jittered world grid, so clumps stay put as the car moves
  let n = 0;
  const i0 = Math.floor((cx - GRASS_R) / GRASS_STEP), i1 = Math.ceil((cx + GRASS_R) / GRASS_STEP);
  const j0 = Math.floor((cz - GRASS_R) / GRASS_STEP), j1 = Math.ceil((cz + GRASS_R) / GRASS_STEP);
  grass: for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
    const hx = hash(i, j), hz = hash(j + 7919, i - 104729);
    const x = (i + hx) * GRASS_STEP, z = (j + hz) * GRASS_STEP;
    const d = Math.hypot(x - cx, z - cz);
    if (d > GRASS_R) continue;
    const l = landAt(x, z);
    if (!l || l.mat !== L_GRASS || waterLevelAt(x, z) !== null) continue;
    const tw = tileFor(x, z);
    if (tw && tw.walkMask && tw.walkMask[Math.floor((z - tw.oz) / 2) * Math.ceil(TILE_SIZE / 2) + Math.floor((x - tw.ox) / 2)]) continue;
    const lush = THREE.MathUtils.clamp((l.ndvi - 0.05) / 0.5, 0, 1);
    if (hash(i * 3 + 1, j * 5 + 2) > 0.3 + 0.7 * lush) continue;     // sparser where it's dry
    const g = groundAt(x, z);
    if (!g || g.seg >= 0) continue;                                    // never on roads
    const edge = 1 - sstep(GRASS_R * 0.7, GRASS_R, d);
    const h = (0.22 + 0.4 * lush) * (0.7 + 0.6 * hash(i + 11, j + 13)) * edge;
    placePlant(grassMesh, n, x, z, g.h, h, (1 + hz * 0.6) * edge, hx * 6.283, l, 1.2);
    if (++n >= GRASS_MAX) break grass;
  }
  grassMesh.count = n;
  grassMesh.instanceMatrix.needsUpdate = true;
  grassMesh.instanceColor.needsUpdate = true;

  // crops: plants sit on the same rows the ground shader paints
  n = 0;
  crops: for (const alongX of [true, false]) {
    const ca = alongX ? cx : cz, cr = alongX ? cz : cx;
    const r0 = Math.floor((cr - CROP_R) / ROW), r1 = Math.ceil((cr + CROP_R) / ROW);
    const a0 = Math.floor((ca - CROP_R) / PLANT), a1 = Math.ceil((ca + CROP_R) / PLANT);
    for (let r = r0; r <= r1; r++) for (let a = a0; a <= a1; a++) {
      const across = (r + 0.5) * ROW, along = (a + 0.2 + hash(a, r) * 0.6) * PLANT;
      const x = alongX ? along : across, z = alongX ? across : along;
      const d = Math.hypot(x - cx, z - cz);
      if (d > CROP_R) continue;
      const cell = Math.floor(x / 400) + Math.floor(z / 400) * 3;
      if (((cell % 2) + 2) % 2 === 1 !== alongX) continue;            // row direction for this field cell
      const l = landAt(x, z);
      if (!l || l.mat !== L_CROP || waterLevelAt(x, z) !== null) continue;
      const amt = THREE.MathUtils.clamp((l.ndvi - 0.15) / 0.5, 0, 1);
      if (amt < 0.08) continue;
      const g = groundAt(x, z);
      if (!g || g.seg >= 0) continue;
      const edge = 1 - sstep(CROP_R * 0.75, CROP_R, d);
      const h = (CROP_HEIGHT[l.code] ?? 0.7) * (0.35 + 0.65 * amt) * (0.85 + 0.3 * hash(r, a + 3)) * edge;
      placePlant(cropMesh, n, x, z, g.h, h, (0.6 + amt * 0.8) * edge, hash(a + 5, r) * 6.283, l, 1.15);
      if (++n >= CROP_MAX) break crops;
    }
  }
  cropMesh.count = n;
  cropMesh.instanceMatrix.needsUpdate = true;
  cropMesh.instanceColor.needsUpdate = true;
}

function updateDetail(dt) {
  timeU.value += dt;
  if (detailDirty || Math.hypot(FOC.x - detailX, FOC.z - detailZ) > 6) {
    detailX = FOC.x; detailZ = FOC.z; detailDirty = false;
    rebuildDetail(FOC.x, FOC.z);
  }
  updateTrees(dt);
}

// ===================================================================== trees
// Trees come from the worker: crowns detected in the aerial photo, each with a species drawn from a smooth blend of
// regional biomes (gums in Australia, spruce and pine in Scandinavia, emergents and palms in rainforest...).
// Within TREE_NEAR they are low-poly species models (bark keeps its colour, crowns are tinted per tree, two shape
// variants each); out to TREE_MID the same models built lighter and without shadows; out to TREE_FAR crossed-quad
// impostors in one of eight silhouettes; beyond, the photo.
const TREE_NEAR = 320, TREE_MID = 1200, TREE_FAR = 2600, SPECIES_MAX = 16000, FAR_MAX = 260000, NEAR_BUDGET = 11000, MID_BUDGET = 60000, FAR_THIN = 450;
// must match the worker's SPECIES order
const SPECIES_NAMES = ['spruce', 'fir', 'pine', 'ponderosa', 'larch', 'cedar', 'cypress', 'juniper', 'oak', 'maple', 'beech', 'birch',
  'poplar', 'willow', 'olive', 'gum', 'mallee', 'acacia', 'stonepine', 'palm', 'emergent', 'canopy', 'baobab', 'casuarina'];
// impostor silhouette: 0 cone, 1 pine, 2 round, 3 oval/columnar, 4 umbrella, 5 palm, 6 gum, 7 emergent
const SPECIES_FORM = [0, 0, 1, 1, 0, 0, 3, 0, 2, 2, 2, 3, 3, 2, 2, 6, 6, 4, 4, 5, 7, 2, 4, 0];
// trunk radius in model units (x crown radius) for collisions
// crown spread relative to the detected crown radius
const SPECIES_WIDE = [1, 1, 1, 1, 1, 1, 1, 1, 1.05, 1, 1, 0.95, 0.9, 1.05, 1, 1.05, 1, 1.3, 1.25, 1, 1.9, 1.3, 1.2, 1];
const SPECIES_TRUNK = [0.06, 0.06, 0.07, 0.08, 0.05, 0.07, 0.06, 0.06, 0.1, 0.08, 0.07, 0.045, 0.06, 0.08, 0.08, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.28, 0.05];

function mergeGeometries(geos, shade) {
  const pos = [], nor = [], col = [];
  for (const g0 of geos) { const g = g0.index ? g0.toNonIndexed() : g0; pos.push(...g.attributes.position.array); nor.push(...g.attributes.normal.array); }
  for (let i = 0; i < pos.length / 3; i++) { const v = shade ? 0.78 + Math.random() * 0.3 : 1; col.push(v, v, v); }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  out.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return out;
}

// ---- tree model kit (model space: crown radius 1 across, height 1)
const BARK = { brown: [0.33, 0.25, 0.18], dark: [0.24, 0.18, 0.13], pine: [0.55, 0.33, 0.2], pale: [0.82, 0.78, 0.72], white: [0.9, 0.89, 0.85],
               grey: [0.48, 0.46, 0.42], palm: [0.5, 0.42, 0.32], baobab: [0.6, 0.55, 0.5] };
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
let TREE_LITE = false;   // while set, the kit builds lighter versions (mid-distance trees)
const blob = (rx, ry, rz, x, y, z, det = 0) => new THREE.IcosahedronGeometry(1, TREE_LITE ? 0 : det).scale(rx, ry, rz).translate(x, y, z);
const cone = (r, y0, y1, seg = 8, x = 0, z = 0) => new THREE.ConeGeometry(r, y1 - y0, TREE_LITE ? Math.min(seg, 5) : seg, 1).translate(x, (y0 + y1) / 2, z);
function limb(A, B, r0, r1 = r0 * 0.7, seg = 5) {
  const d = new THREE.Vector3(B[0] - A[0], B[1] - A[1], B[2] - A[2]), L = d.length();
  const g = new THREE.CylinderGeometry(r1, r0, L, TREE_LITE ? 3 : seg, 1).translate(0, L / 2, 0);
  g.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize())));
  return g.translate(A[0], A[1], A[2]);
}
const stem = (r0, r1, y1, x = 0, z = 0, y0 = 0) => limb([x, y0, z], [x, y1, z], r0, r1, 6);
function frond(a, R, lift, drop) {
  // a drooping palm leaf: tapered strip, both windings so it shows from below
  const pos = [], N = 5, ux = Math.cos(a), uz = Math.sin(a), sx = -uz, sz = ux;
  for (let k = 0; k <= N; k++) {
    const t = k / N, w = 0.14 * (1 - t) + 0.015, d = R * t, y = 1 + lift * t - drop * t * t;
    pos.push([[ux * d + sx * w, y, uz * d + sz * w], [ux * d - sx * w, y - 0.01, uz * d - sz * w]]);
  }
  const p = [];
  for (let k = 0; k < N; k++) {
    const [a0, b0] = pos[k], [a1, b1] = pos[k + 1];
    p.push(...a0, ...b0, ...a1, ...b0, ...b1, ...a1, ...a0, ...a1, ...b0, ...b0, ...a1, ...b1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
  g.computeVertexNormals();
  return g;
}
// A palm frond in palm model space (x/z in crown radii, y in tree heights; HR = height / radius of a typical palm):
// a midrib that rises a little and droops towards the tip, with leaflets folded down either side of it in a V and
// cut into a saw-tooth edge so it reads as a feathery leaf, not a ribbon.
const HR = 5.5;
function palmFrond(a, reach, rise, droop, width) {
  const ux = Math.cos(a), uz = Math.sin(a), sx = -uz, sz = ux, N = TREE_LITE ? 6 : 14, pos = [];
  const mid = t => [ux * reach * t, (rise * t - droop * t * t) / HR, uz * reach * t];
  for (let k = 0; k < N; k++) {
    const t0 = k / N, t1 = (k + 1) / N, m0 = mid(t0), m1 = mid(t1);
    // leaflet width: grows from the stalk, tapers to the tip; every other one shorter (the saw-tooth)
    const wAt = (t, j) => width * Math.min(1, Math.max(0, (t - 0.08) / 0.15)) * (1 - 0.7 * t) * (j % 2 ? 0.55 : 1);
    const w0 = wAt(t0, k), w1 = wAt(t1, k + 1), f0 = w0 * 0.55 / HR, f1 = w1 * 0.55 / HR;
    for (const s of [1, -1]) {
      const e0 = [m0[0] + sx * w0 * s, m0[1] - f0, m0[2] + sz * w0 * s], e1 = [m1[0] + sx * w1 * s, m1[1] - f1, m1[2] + sz * w1 * s];
      // leaflets sweep forward a little: the outer edge sits ahead of where it meets the midrib
      e0[0] += ux * w0 * 0.35; e0[2] += uz * w0 * 0.35; e1[0] += ux * w1 * 0.35; e1[2] += uz * w1 * 0.35;
      if (s > 0) pos.push(...m0, ...e0, ...e1, ...m0, ...e1, ...m1);
      else pos.push(...m0, ...e1, ...e0, ...m0, ...m1, ...e1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}
function treeGeo(parts, seed) {
  const R = rng(seed), pos = [], nor = [], col = [], tint = [], c = new THREE.Color();
  for (const { geo, bark } of parts) {
    const g = geo.index ? geo.toNonIndexed() : geo;
    if (!g.attributes.normal) g.computeVertexNormals();
    const P = g.attributes.position.array, Nn = g.attributes.normal.array;
    for (let i = 0; i < P.length; i += 3) {
      pos.push(P[i], P[i + 1], P[i + 2]); nor.push(Nn[i], Nn[i + 1], Nn[i + 2]);
      if (bark) { c.setRGB(bark[0], bark[1], bark[2], THREE.SRGBColorSpace); col.push(c.r, c.g, c.b); tint.push(0); }
      else { const v = 0.8 + R() * 0.28; col.push(v, v, v); tint.push(1); }
    }
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  out.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  out.setAttribute('tint', new THREE.Float32BufferAttribute(tint, 1));
  return out;
}
const K = (geo) => ({ geo }), B = (geo, bark) => ({ geo, bark });
// each builder gets a random source so its two variants differ
const SPECIES_BUILD = {
  spruce: R => [B(stem(0.06, 0.03, 0.3), BARK.dark), ...Array.from({ length: 6 }, (_, k) => K(cone((1 - k * 0.16) * (0.92 + R() * 0.16), 0.08 + k * 0.15, 0.4 + k * 0.12, 8)))],
  fir: R => [B(stem(0.06, 0.03, 0.25), BARK.dark), ...Array.from({ length: 7 }, (_, k) => K(cone((1 - k * 0.13) * (0.95 + R() * 0.1), 0.06 + k * 0.13, 0.3 + k * 0.12, 9)))],
  pine: R => [B(stem(0.07, 0.04, 0.82), BARK.pine), B(limb([0, 0.6, 0], [0.35, 0.78, 0.1], 0.03), BARK.pine), B(limb([0, 0.66, 0], [-0.3, 0.8, -0.2], 0.03), BARK.pine),
    K(blob(0.55, 0.18, 0.5, 0.25 + R() * 0.15, 0.8, 0.1)), K(blob(0.5, 0.2, 0.55, -0.25, 0.86, -0.15 - R() * 0.1)), K(blob(0.42, 0.18, 0.42, 0.05, 0.95, 0.15))],
  ponderosa: R => [B(stem(0.08, 0.04, 0.6), BARK.pine), K(blob(0.75, 0.22, 0.75, 0, 0.5, 0, 1)), K(blob(0.62, 0.2, 0.62, R() * 0.1, 0.68, 0)), K(blob(0.45, 0.18, 0.45, 0, 0.84, 0)), K(cone(0.28, 0.85, 1.0, 7))],
  larch: R => [B(stem(0.05, 0.02, 0.4), BARK.brown), ...Array.from({ length: 4 }, (_, k) => K(cone((0.95 - k * 0.22) * (0.9 + R() * 0.2), 0.12 + k * 0.2, 0.42 + k * 0.17, 7)))],
  cedar: R => [B(stem(0.07, 0.04, 0.2), BARK.brown), K(cone(0.75, 0.06, 1.0, 9)), K(cone(0.6 + R() * 0.1, 0.25, 0.85, 8)), K(cone(0.45, 0.45, 0.98, 7))],
  cypress: R => [B(stem(0.05, 0.03, 0.15), BARK.dark), K(blob(0.42, 0.3, 0.42, 0, 0.3, 0)), K(cone(0.42, 0.25, 0.8, 8)), K(cone(0.3, 0.6, 1.0, 7)), K(blob(0.33 + R() * 0.05, 0.2, 0.33, 0, 0.55, 0))],
  juniper: R => [B(stem(0.08, 0.05, 0.25), BARK.grey), K(blob(0.7, 0.45, 0.65, 0, 0.42, 0)), K(blob(0.5, 0.4, 0.5, 0.2 * R(), 0.7, -0.1)), K(blob(0.35, 0.3, 0.35, -0.15, 0.88, 0.1))],
  oak: R => [B(stem(0.1, 0.07, 0.42), BARK.dark), B(limb([0, 0.35, 0], [0.4, 0.62, 0.15], 0.05), BARK.dark), B(limb([0, 0.38, 0], [-0.38, 0.6, -0.2], 0.05), BARK.dark),
    ...Array.from({ length: 5 }, (_, k) => { const a = k * 1.26 + R(), d = 0.35 + R() * 0.25; return K(blob(0.5 + R() * 0.15, 0.3, 0.5, Math.cos(a) * d, 0.62 + R() * 0.15, Math.sin(a) * d, 1)); }), K(blob(0.55, 0.32, 0.55, 0, 0.8, 0, 1))],
  maple: R => [B(stem(0.08, 0.05, 0.4), BARK.brown), K(blob(0.92, 0.45, 0.92, 0, 0.6, 0, 1)), ...Array.from({ length: 3 }, (_, k) => { const a = k * 2.1 + R(); return K(blob(0.42, 0.3, 0.42, Math.cos(a) * 0.55, 0.62 + R() * 0.2, Math.sin(a) * 0.55)); })],
  beech: R => [B(stem(0.07, 0.05, 0.45), BARK.grey), K(blob(0.85, 0.5, 0.85, 0, 0.62, 0, 1)), K(blob(0.5, 0.32, 0.5, 0.4 * R(), 0.85, -0.2)), K(blob(0.45, 0.3, 0.45, -0.45, 0.55, 0.3 * R()))],
  birch: R => [B(stem(0.045, 0.025, 0.8), BARK.white), ...[0.18, 0.32, 0.47, 0.6].map(y => B(new THREE.BoxGeometry(0.05, 0.02, 0.05).translate(0.02, y, 0.01), [0.12, 0.12, 0.12])),
    K(blob(0.55, 0.32, 0.5, 0, 0.62, 0, 1)), K(blob(0.42, 0.28, 0.42, 0.15 * R(), 0.82, 0.1)), K(blob(0.35, 0.22, 0.35, -0.2, 0.5, -0.15 * R()))],
  poplar: R => [B(stem(0.06, 0.04, 0.2), BARK.grey), K(blob(0.42, 0.42, 0.42, 0, 0.55, 0, 1)), K(blob(0.32, 0.18, 0.32, 0, 0.92, 0)), K(blob(0.36, 0.25, 0.36, 0.05 * R(), 0.25, 0))],
  willow: R => [B(stem(0.08, 0.06, 0.45), BARK.dark), K(blob(0.9, 0.35, 0.9, 0, 0.72, 0, 1)), K(new THREE.ConeGeometry(1.0, 0.55, 12, 1, true).translate(0, 0.45, 0)), K(new THREE.ConeGeometry(0.98, 0.55, 12, 1, true).rotateX(Math.PI).rotateX(Math.PI).translate(0, 0.45 + R() * 0.02, 0))],
  olive: R => [B(limb([0, 0, 0], [0.12, 0.4, 0.05], 0.09, 0.06), BARK.grey), B(limb([0.05, 0.2, 0], [-0.25, 0.45, -0.1], 0.05), BARK.grey),
    ...Array.from({ length: 4 }, (_, k) => { const a = k * 1.6 + R(); return K(blob(0.45, 0.25, 0.45, Math.cos(a) * 0.4, 0.55 + R() * 0.25, Math.sin(a) * 0.4)); })],
  gum: R => {
    const p = [B(stem(0.07, 0.035, 0.62), BARK.pale)];
    for (let k = 0; k < 3; k++) { const a = k * 2.1 + R(), e = [Math.cos(a) * 0.5, 0.82 + R() * 0.1, Math.sin(a) * 0.5]; p.push(B(limb([0, 0.55, 0], e, 0.03), BARK.pale), K(blob(0.32, 0.18, 0.3, e[0], e[1] + 0.05, e[2]))); }
    p.push(K(blob(0.3, 0.2, 0.3, 0, 0.95, 0)), K(blob(0.25, 0.15, 0.25, 0.25 * R(), 0.72, -0.3)));
    return p;
  },
  mallee: R => { const p = []; for (let k = 0; k < 4; k++) { const a = k * 1.57 + R(), e = [Math.cos(a) * 0.35, 0.65, Math.sin(a) * 0.35]; p.push(B(limb([0, 0, 0], e, 0.04, 0.025), BARK.pale), K(blob(0.4, 0.25, 0.4, e[0] * 1.2, 0.75 + R() * 0.15, e[2] * 1.2))); } return p; },
  acacia: R => [B(stem(0.06, 0.04, 0.45), BARK.dark), B(limb([0, 0.4, 0], [0.4, 0.75, 0.1], 0.035), BARK.dark), B(limb([0, 0.42, 0], [-0.35, 0.74, -0.2], 0.035), BARK.dark),
    K(blob(0.75, 0.1, 0.7, 0.15, 0.8, 0.05)), K(blob(0.6, 0.09, 0.6, -0.3, 0.78, -0.15)), K(blob(0.45, 0.08, 0.45, 0.1 * R(), 0.86, 0.3))],
  stonepine: R => [B(limb([0, 0, 0], [0.08 * R(), 0.68, 0], 0.07, 0.05), BARK.pine), B(limb([0, 0.55, 0], [0.4, 0.72, 0], 0.035), BARK.pine), B(limb([0, 0.58, 0], [-0.35, 0.72, 0.2], 0.035), BARK.pine),
    K(blob(1.0, 0.22, 0.95, 0, 0.8, 0, 1)), K(blob(0.6, 0.15, 0.6, 0.15, 0.93, -0.1))],
  palm: R => {
    // Model x/z are in crown radii but y in tree heights (a palm is ~5.5 radii tall), so everything sideways-sized
    // that also has vertical extent (frond arcs, leaflet folds, coconuts) is divided by HR on y.
    const p = [], lean = 0.15 + R() * 0.35, dir = R() * 6.28, cx = Math.cos(dir), cz = Math.sin(dir);
    const off = y => lean * Math.pow(y, 1.6), TOP = 0.93;
    // ringed, gently curving, tapering trunk
    const NT = 9;
    for (let k = 0; k < NT; k++) {
      const y0 = k / NT * TOP, y1 = (k + 1) / NT * TOP, r0 = 0.075 - k * 0.0035, r1 = 0.075 - (k + 1) * 0.0035;
      p.push(B(limb([cx * off(y0), y0, cz * off(y0)], [cx * off(y1), y1, cz * off(y1)], r0, r1, 7), k % 2 ? BARK.palm : [0.4, 0.33, 0.25]));
    }
    const tx = cx * off(TOP), tz = cz * off(TOP);
    p.push(B(limb([tx, TOP - 0.05, tz], [tx, TOP + 0.02, tz], 0.065, 0.05, 7), [0.36, 0.4, 0.26]));   // crown shaft
    // coconuts tucked under the crown
    for (let k = 0; k < 4; k++) { const a = k * 1.6 + R(); p.push(B(blob(0.07, 0.07 / HR, 0.07, tx + Math.cos(a) * 0.08, TOP - 0.025, tz + Math.sin(a) * 0.08), [0.42, 0.3, 0.14])); }
    // three tiers of fronds: upper ones arch up and out, lower ones hang
    const tiers = [[6, 0.82, 0.55, 0.5], [8, 1.02, 0.25, 0.85], [5, 0.9, 0.0, 1.25]];
    let a0 = R() * 6.28;
    tiers.forEach(([n, reach, rise, droop], ti) => {
      for (let k = 0; k < n; k++) {
        const a = a0 + k * 6.283 / n + (R() - 0.5) * 0.35;
        p.push(K(palmFrond(a, reach * (0.9 + R() * 0.2), rise, droop * (0.85 + R() * 0.3), 0.15 + R() * 0.03).translate(tx, TOP + 0.01 - ti * 0.004, tz)));
      }
      a0 += 6.283 / n / 2;
    });
    // now and then a dead brown frond hanging against the trunk
    if (R() < 0.6) { const a = R() * 6.28; p.push(B(palmFrond(a, 0.45, -0.2, 2.6, 0.1), [0.45, 0.36, 0.22])); p[p.length - 1].geo.translate(tx, TOP - 0.01, tz); }
    return p;
  },
  emergent: R => [B(stem(0.05, 0.03, 0.9), BARK.grey), ...[0, 2.1, 4.2].map(a => B(new THREE.BoxGeometry(0.035, 0.07, 0.14).translate(0, 0.035, 0.07).rotateY(a + R()), BARK.grey)),
    B(limb([0, 0.86, 0], [0.38, 0.93, 0.1], 0.02), BARK.grey), B(limb([0, 0.87, 0], [-0.35, 0.94, -0.2], 0.02), BARK.grey),
    ...Array.from({ length: 5 }, (_, k) => { const a = k * 1.26 + R(); return K(blob(0.42, 0.045, 0.42, Math.cos(a) * 0.5, 0.94 + R() * 0.03, Math.sin(a) * 0.5)); }), K(blob(0.5, 0.05, 0.5, 0, 0.97, 0))],
  canopy: R => [B(stem(0.07, 0.05, 0.62), BARK.grey), B(limb([0, 0.55, 0], [0.45, 0.75, 0.1], 0.035), BARK.grey), B(limb([0, 0.58, 0], [-0.4, 0.78, -0.25], 0.035), BARK.grey),
    ...Array.from({ length: 6 }, (_, k) => { const a = k * 1.05 + R(), d = 0.35 + R() * 0.3; return K(blob(0.5, 0.16, 0.5, Math.cos(a) * d, 0.74 + R() * 0.16, Math.sin(a) * d, k === 0 ? 1 : 0)); }), K(blob(0.55, 0.15, 0.55, 0, 0.9, 0))],
  baobab: R => [B(stem(0.28, 0.2, 0.62), BARK.baobab), ...Array.from({ length: 5 }, (_, k) => { const a = k * 1.26 + R(), e = [Math.cos(a) * 0.45, 0.8 + R() * 0.1, Math.sin(a) * 0.45]; return B(limb([0, 0.58, 0], e, 0.06, 0.03), BARK.baobab); }),
    ...Array.from({ length: 4 }, (_, k) => { const a = k * 1.57 + R(); return K(blob(0.28, 0.12, 0.28, Math.cos(a) * 0.5, 0.88, Math.sin(a) * 0.5)); })],
  casuarina: R => [B(stem(0.05, 0.025, 0.5), BARK.dark), ...Array.from({ length: 4 }, (_, k) => K(new THREE.ConeGeometry((0.9 - k * 0.2) * (0.9 + R() * 0.2), 0.32, 7, 1, true).translate(0, 0.35 + k * 0.17, 0)))],
};
const SPECIES_GEO = SPECIES_NAMES.map((name, i) => [0, 1].map(v => treeGeo(SPECIES_BUILD[name](rng(i * 977 + v * 131 + 7)), i * 31 + v)));
// the same trees, lighter (fewer sides and facets), for the middle distance
TREE_LITE = true;
const SPECIES_GEO_LITE = SPECIES_NAMES.map((name, i) => [0, 1].map(v => treeGeo(SPECIES_BUILD[name](rng(i * 977 + v * 131 + 7)), i * 31 + v)));
TREE_LITE = false;

// crowns take the per-tree tint, bark keeps its colour; tops sway gently in the wind
const treeMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.9, side: THREE.DoubleSide });
treeMat.onBeforeCompile = sh => {
  sh.uniforms.uTime = timeU;
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', '#include <common>\nattribute float tint;\nattribute float aSnow;\nuniform float uTime;\nvarying float vSnowK;')
    .replace('#include <color_vertex>', `vColor = vec3(1.0);
      vColor *= color;
      #ifdef USE_INSTANCING_COLOR
        vColor.xyz *= mix(vec3(1.0), instanceColor.xyz, tint);
      #endif
      // snow settles on the crown's upward faces (bark, sides and undersides stay green). Trees are stretched to
      // their height per instance, so the face's real slope comes from the normal un-stretched by that scale.
      #ifdef USE_INSTANCING
        vec3 isc = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
        float upW = normalize(normal / max(isc, vec3(1e-3))).y;
      #else
        float upW = normal.y;
      #endif
      // real snow: pure white where it lies, patchy (some branches stay green), and a touch of its own light in the
      // fragment shader so shade doesn't turn it grey
      float snowPatch = fract(sin(dot(floor(position * 7.0), vec3(12.9898, 78.233, 45.164))) * 43758.5453);
      vSnowK = aSnow * tint * smoothstep(0.1, 0.22, upW) * step(0.28 - aSnow * 0.15, snowPatch);
      vColor.xyz = mix(vColor.xyz, vec3(1.0), vSnowK);`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>
      #ifdef USE_INSTANCING
        vec3 ip = instanceMatrix[3].xyz;
        float k = position.y * position.y * tint;
        transformed.x += sin(uTime * 0.9 + ip.x * 0.07 + ip.z * 0.05) * 0.025 * k;
        transformed.z += sin(uTime * 0.7 + ip.z * 0.06) * 0.02 * k;
      #endif`);
  sh.fragmentShader = sh.fragmentShader
    .replace('#include <common>', '#include <common>\nvarying float vSnowK;')
    .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n  totalEmissiveRadiance += vec3(0.3, 0.32, 0.36) * vSnowK;');
};
treeMat.customProgramCacheKey = () => 'tree-species-v5';
const canopyMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.9 });
const treeMeshes = new Map(); // "species:variant" (+100 for the lighter mid-distance set) -> InstancedMesh (made on first use)
function speciesMesh(sp, v, lite = false) {
  const key = sp * 2 + v + (lite ? 100 : 0);
  let m = treeMeshes.get(key);
  if (!m) {
    m = new THREE.InstancedMesh((lite ? SPECIES_GEO_LITE : SPECIES_GEO)[sp][v], treeMat, SPECIES_MAX);
    m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(SPECIES_MAX * 3), 3);
    m.userData.snow = new THREE.InstancedBufferAttribute(new Float32Array(SPECIES_MAX), 1);
    m.geometry.setAttribute('aSnow', m.userData.snow);
    m.count = 0; m.frustumCulled = false; m.castShadow = !lite; m.receiveShadow = true;
    m.userData.n = 0;
    scene.add(m);
    treeMeshes.set(key, m);
  }
  return m;
}

// impostor atlas: eight silhouettes side by side (crowns white so the per-tree colour shows; trunks darker)
function impostorAtlas() {
  const cv = document.createElement('canvas');
  cv.width = 1024; cv.height = 128;
  const g = cv.getContext('2d'), R = rng(99);
  const trunk = (x, y0, y1, w, col = '#6b5642') => { g.fillStyle = col; g.fillRect(x - w / 2, y0, w, y1 - y0); };
  const disc = (x, y, rx, ry) => { g.beginPath(); g.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2); g.fill(); };
  const cell = (k, fn) => { g.save(); g.translate(k * 128, 0); fn(); g.restore(); };
  cell(0, () => { trunk(64, 100, 128, 5); g.fillStyle = '#fff'; for (let k = 0; k < 7; k++) { const y = 14 + k * 13, w = 8 + k * 6.5; g.beginPath(); g.moveTo(64, y - 16); g.lineTo(64 - w, y + 14); g.lineTo(64 + w, y + 14); g.closePath(); g.fill(); } });
  cell(1, () => { trunk(64, 20, 128, 6, '#8a5a38'); g.fillStyle = '#fff'; for (let k = 0; k < 10; k++) disc(64 + (R() - 0.5) * 60, 18 + R() * 22, 14 + R() * 10, 7 + R() * 5); });
  cell(2, () => { trunk(64, 70, 128, 7); g.fillStyle = '#fff'; for (let k = 0; k < 40; k++) { const a = R() * 6.28, r = R() * 36; disc(64 + Math.cos(a) * r * 1.2, 48 + Math.sin(a) * r * 0.75, 10 + R() * 11, 10 + R() * 11); } });
  cell(3, () => { trunk(64, 90, 128, 5); g.fillStyle = '#fff'; disc(64, 52, 22, 46); for (let k = 0; k < 12; k++) disc(64 + (R() - 0.5) * 30, 14 + R() * 80, 9, 9); });
  cell(4, () => { trunk(64, 40, 128, 5); g.fillStyle = '#fff'; disc(64, 34, 58, 12); for (let k = 0; k < 8; k++) disc(64 + (R() - 0.5) * 90, 30 + R() * 8, 14, 7); });
  // palm: slim curved trunk; crown of arching feathery fronds (the cell is ~5.5 crown radii tall and 2.2 wide, so
  // a frond reaching one radius sideways (~55 px) only drops ~20 px)
  cell(5, () => { g.strokeStyle = '#7a6650'; g.lineWidth = 4; g.lineCap = 'round'; g.beginPath(); g.moveTo(58, 128); g.quadraticCurveTo(58, 60, 66, 14); g.stroke();
    const fr = (dirX, reach, rise, droop) => {
      const P = t => [66 + dirX * reach * t, 13 - rise * t + droop * t * t];
      g.strokeStyle = '#fff'; g.lineWidth = 2.5; g.beginPath(); for (let t = 0; t <= 1.001; t += 0.1) { const [x, y] = P(t); t ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke();
      g.lineWidth = 2; for (let t = 0.12; t < 0.95; t += 0.07) { const [x, y] = P(t), w = 7 * (1 - 0.7 * t); g.beginPath(); g.moveTo(x, y); g.lineTo(x + dirX * w * 0.4, y + w); g.moveTo(x, y); g.lineTo(x + dirX * w * 0.5, y - w * 0.35); g.stroke(); }
    };
    for (const [dx, reach, rise, droop] of [[1, 56, 8, 24], [-1, 54, 8, 26], [0.7, 40, 12, 18], [-0.7, 42, 12, 16], [0.35, 30, 14, 14], [-0.3, 28, 14, 15], [1, 46, 2, 34], [-1, 48, 2, 32], [0.05, 10, 10, 6]]) fr(dx, reach, rise, droop);
    g.fillStyle = '#5a3d1e'; disc(66, 17, 4, 3); });
  cell(6, () => { trunk(64, 30, 128, 6, '#d8d0c4'); g.strokeStyle = '#d8d0c4'; g.lineWidth = 3; for (const s of [-1, 1]) { g.beginPath(); g.moveTo(64, 45); g.lineTo(64 + s * 30, 20); g.stroke(); }
    g.fillStyle = '#fff'; for (let k = 0; k < 14; k++) disc(64 + (R() - 0.5) * 86, 10 + R() * 34, 8 + R() * 8, 5 + R() * 5); });
  cell(7, () => { trunk(64, 18, 128, 6, '#9a9488'); g.fillStyle = '#fff'; for (let k = 0; k < 12; k++) disc(64 + (R() - 0.5) * 100, 10 + R() * 12, 14 + R() * 10, 6 + R() * 4); });
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function crossedQuads() {
  const pos = [], uv = [], nor = [], idx = [];
  for (const [ax, az] of [[1, 0], [0, 1]]) {
    const b = pos.length / 3;
    pos.push(-ax, 0, -az, ax, 0, az, ax, 1, az, -ax, 1, -az);
    uv.push(0, 0, 1, 0, 1, 1, 0, 1);
    nor.push(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0);
    idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setIndex(idx);
  return g;
}
const farGeo = crossedQuads(), farForm = new THREE.InstancedBufferAttribute(new Float32Array(FAR_MAX), 1);
farGeo.setAttribute('form', farForm);
const farMat = new THREE.MeshStandardMaterial({ map: impostorAtlas(), alphaTest: 0.5, side: THREE.DoubleSide, roughness: 1 });
farMat.onBeforeCompile = sh => {
  sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute float form;')
    .replace('#include <uv_vertex>', '#include <uv_vertex>\n  vMapUv.x = (vMapUv.x + form) / 8.0;');
};
farMat.customProgramCacheKey = () => 'tree-far-v2';
const farTrees = new THREE.InstancedMesh(farGeo, farMat, FAR_MAX);
farTrees.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(FAR_MAX * 3), 3);
farTrees.count = 0; farTrees.frustumCulled = false; farTrees.receiveShadow = true;
scene.add(farTrees);

let treesDirty = true, treeX = 1e9, treeZ = 1e9, treeCooldown = 0;
const trunkHash = new Map(); // 6 m cells -> [x, z, radius, ...] for collisions
const TH_CELL = 6, thKey = (x, z) => Math.floor(x / TH_CELL) * 73856093 ^ Math.floor(z / TH_CELL) * 19349663;

const _snowC = new THREE.Color(0.9, 0.93, 0.98);
function rebuildTrees(cx, cz) {
  for (const m of treeMeshes.values()) m.userData.n = 0;
  let nf = 0;
  trunkHash.clear();
  // full models out to TREE_NEAR, or less in thick forest so the count stays within budget
  const bins = new Uint32Array(Math.ceil(TREE_NEAR / 10));
  for (const t of tiles.values()) {
    if (!t.trees) continue;
    const dx = Math.max(t.ox - cx, 0, cx - t.ox - TILE_SIZE), dz = Math.max(t.oz - cz, 0, cz - t.oz - TILE_SIZE);
    if (dx * dx + dz * dz > TREE_NEAR * TREE_NEAR) continue;
    const T = t.trees, step = T.length > 80000 * 8 ? 4 : 1;   // trees are shuffled: every 4th is a fair sample
    for (let o = 0; o < T.length; o += 8 * step) {
      const d = Math.hypot(t.ox + T[o] - cx, t.oz + T[o + 1] - cz);
      if (d < TREE_NEAR) bins[Math.floor(d / 10)] += step;
    }
  }
  let nearR = TREE_NEAR, acc = 0;
  for (let k = 0; k < bins.length; k++) { acc += bins[k]; if (acc > NEAR_BUDGET) { nearR = Math.max(90, k * 10); break; } }
  // lighter models further out, or less far in thick forest so the count stays within budget
  const mbins = new Float32Array(Math.ceil(TREE_MID / 25));
  for (const t of tiles.values()) {
    if (!t.trees) continue;
    const dx = Math.max(t.ox - cx, 0, cx - t.ox - TILE_SIZE), dz = Math.max(t.oz - cz, 0, cz - t.oz - TILE_SIZE);
    if (dx * dx + dz * dz > TREE_MID * TREE_MID) continue;
    const T = t.trees, step = T.length > 40000 * 8 ? 8 : 2;
    for (let o = 0; o < T.length; o += 8 * step) {
      const d = Math.hypot(t.ox + T[o] - cx, t.oz + T[o + 1] - cz);
      if (d < TREE_MID && d >= nearR) mbins[Math.floor(d / 25)] += step * (d > FAR_THIN ? FAR_THIN * FAR_THIN / (d * d) : 1);
    }
  }
  let midR = TREE_MID; acc = 0;
  for (let k = 0; k < mbins.length; k++) { acc += mbins[k]; if (acc > MID_BUDGET) { midR = Math.max(nearR, k * 25); break; } }
  const near2 = nearR * nearR, mid2 = midR * midR, far2 = TREE_FAR * TREE_FAR;
  // no trees growing out of landmarks (their footprint, slightly enlarged)
  const lmCut = LANDMARKS.filter(L => L.mesh && L.hx && Math.hypot(L.px - cx, L.pz - cz) < TREE_FAR + 600)
    .map(L => { const [ax, az] = lmPad(L); return { x: L.px, z: L.pz, c: Math.cos(L.prot), s: Math.sin(L.prot), hx: ax + 6, hz: az + 6, r: Math.hypot(ax, az) + 6 }; });
  const underLm = (x, z) => {
    for (const q of lmCut) {
      const dx = x - q.x, dz = z - q.z;
      if (Math.abs(dx) > q.r || Math.abs(dz) > q.r) continue;
      if (Math.abs(dx * q.c - dz * q.s) < q.hx && Math.abs(dx * q.s + dz * q.c) < q.hz) return true;
    }
    return false;
  };
  for (const t of tiles.values()) {
    if (!t.trees) continue;
    const dx = Math.max(t.ox - cx, 0, cx - t.ox - TILE_SIZE), dz = Math.max(t.oz - cz, 0, cz - t.oz - TILE_SIZE);
    const dmin2 = dx * dx + dz * dz;
    if (dmin2 > far2) continue;
    const T = t.trees;
    // trees arrive shuffled, so thinning keeps the first `keep` share of them: a distant tile only reads a prefix
    const keepMax = dmin2 > FAR_THIN * FAR_THIN ? FAR_THIN * FAR_THIN / dmin2 : 1;
    const end = Math.min(T.length, Math.ceil(T.length / 8 * keepMax) * 8), inv = 8 / T.length;
    for (let o = 0; o < end; o += 8) {
      const x = t.ox + T[o], z = t.oz + T[o + 1];
      const d2 = (x - cx) * (x - cx) + (z - cz) * (z - cz);
      if (d2 > far2 || (lmCut.length && underLm(x, z))) continue;
      // far away, keep a thinning share of trees and grow them so the canopy still reads as closed
      let grow = 1;
      if (d2 > FAR_THIN * FAR_THIN) {
        const keep = FAR_THIN * FAR_THIN / d2;
        if (o / 8 * inv > keep) continue;
        grow = Math.min(2.4, 1 / Math.sqrt(keep));
      }
      if (!t.treeY || t.treeY.length !== T.length / 8) t.treeY = new Float32Array(T.length / 8).fill(NaN);
      let y = t.treeY[o / 8];
      if (y !== y) { const th = terrainHeight(x, z); if (th === null) continue; y = t.treeY[o / 8] = th; }
      const sp = Math.min(SPECIES_NAMES.length - 1, Math.max(0, Math.round(T[o + 7]))), hv = hash(Math.round(x * 3), Math.round(z * 3));
      const snow = Math.min(1, Math.max(0, (T[o + 7] - sp) / 0.4));
      const r = T[o + 2], h = T[o + 3], rot = hv * 6.283;
      _c.setRGB(T[o + 4], T[o + 5], T[o + 6], THREE.SRGBColorSpace).multiplyScalar(1.15);
      _p.set(x, y - 0.2, z);
      if (d2 < near2) {
        const m = speciesMesh(sp, hv < 0.5 ? 0 : 1), n = m.userData.n;
        if (n >= SPECIES_MAX) continue;
        // per-tree proportions: crowns a little wider or narrower, a slight lean
        const rw = r * SPECIES_WIDE[sp], wx = rw * (0.88 + hash(Math.round(z * 5), 7) * 0.24), wz = rw * (0.88 + hash(Math.round(x * 5), 11) * 0.24);
        _q.setFromEuler(_eu.set((hash(Math.round(x), 3) - 0.5) * 0.06, rot, (hash(Math.round(z), 5) - 0.5) * 0.06, 'YXZ'));
        _s.set(wx, h, wz);
        m.setMatrixAt(n, _m.compose(_p, _q, _s));
        m.setColorAt(n, _c);
        m.userData.snow.setX(n, snow);
        m.userData.n = n + 1;
        const tr = Math.max(0.12, SPECIES_TRUNK[sp] * rw);
        const k = thKey(x, z);
        (trunkHash.get(k) || trunkHash.set(k, []).get(k)).push(x, z, tr);
      } else if (d2 < mid2) {
        const m = speciesMesh(sp, hv < 0.5 ? 0 : 1, true), n = m.userData.n;
        if (n >= SPECIES_MAX) continue;
        const rw = r * SPECIES_WIDE[sp] * grow;
        _q.setFromAxisAngle(_up, rot);
        _s.set(rw, h * (0.85 + 0.15 * grow), rw);
        m.setMatrixAt(n, _m.compose(_p, _q, _s));
        m.setColorAt(n, _c);
        m.userData.snow.setX(n, snow);
        m.userData.n = n + 1;
      } else {
        if (nf >= FAR_MAX) continue;
        _q.setFromAxisAngle(_up, rot);
        _s.set(r * 1.1 * SPECIES_WIDE[sp] * grow, h * (0.85 + 0.15 * grow), r * 1.1 * SPECIES_WIDE[sp] * grow);
        farTrees.setMatrixAt(nf, _m.compose(_p, _q, _s));
        farTrees.setColorAt(nf, snow > 0 ? _c.lerp(_snowC, snow * 0.4) : _c);
        farForm.setX(nf, SPECIES_FORM[sp]);
        nf++;
      }
    }
  }
  for (const m of treeMeshes.values()) { m.count = m.userData.n; m.instanceMatrix.needsUpdate = true; m.instanceColor.needsUpdate = true; m.userData.snow.needsUpdate = true; }
  farTrees.count = nf;
  farTrees.instanceMatrix.needsUpdate = true;
  farTrees.instanceColor.needsUpdate = true;
  farForm.needsUpdate = true;
}

function updateTrees(dt) {
  treeCooldown -= dt;
  if (treeCooldown > 0) return;
  if (treesDirty || Math.hypot(V.x - treeX, V.z - treeZ) > 40) {
    treeX = V.x; treeZ = V.z; treesDirty = false; treeCooldown = 0.4;
    rebuildTrees(V.x, V.z);
  }
}

/** The car against tree trunks (the car's box, swept over the last step at speed - see carHitCircle). */
function collideTrees(dt) {
  if (V.inTunnel) return;            // the trees are on the hill above
  const reach = Math.hypot(CAR.len / 2 + Math.hypot(V.vx, V.vz) * dt, CAR.wid / 2) + 1.5;
  const n = Math.ceil(reach / TH_CELL), ci = Math.floor(V.x / TH_CELL), cj = Math.floor(V.z / TH_CELL), seen = new Set();
  for (let dj = -n; dj <= n; dj++) for (let di = -n; di <= n; di++) {
    const key = thKey((ci + di + 0.5) * TH_CELL, (cj + dj + 0.5) * TH_CELL);
    if (seen.has(key)) continue;
    seen.add(key);
    const list = trunkHash.get(key);
    if (!list) continue;
    for (let i = 0; i < list.length; i += 3) if (Math.abs(list[i] - V.x) < reach && Math.abs(list[i + 1] - V.z) < reach) carHitCircle(list[i], list[i + 1], list[i + 2], dt);
  }
}

// ===================================================================== landmark models
// Each landmark is a list of primitive parts in metres. Model space: x = east, z = south, y = up, origin at the
// centre of the base on the ground; `heading` turns the whole model clockwise (compass degrees).
//   ["box",  cx, y0, cz, sx, h, sz, col, rotY]                  box standing on y0
//   ["cyl",  cx, y0, cz, rBottom, rTop, h, col, segs]          cylinder / cone / frustum
//   ["pyr",  cx, y0, cz, sx, h, sz, col, rotY, topFrac]         square pyramid (topFrac > 0: frustum)
//   ["sph",  cx, cy, cz, rx, ry, rz, col, dome, segs]           ellipsoid; dome = 1: upper half only
//   ["gable", cx, y0, cz, sx, h, sz, col, rotY]                 triangular roof, ridge along x
//   ["arch", cx, y0, cz, w, h, d, t, col, rotY]                 arched opening w wide in a wall/gate, h tall, d deep, t thick
//   ["beam", x1, y1, z1, x2, y2, z2, r, col, segs]              strut or cable between two points
//   ["lathe", cx, y0, cz, [[r, y], ...], col, segs]             profile turned about the vertical axis
//   ["ext",  y0, h, [[x, z], ...], col]                         footprint polygon extruded upward
//   ["ring", cx, cy, cz, R, tube, col, rotY, flat]              torus (vertical; flat = 1 lies horizontal)
//   ["loft", y0, h, [[x, z], ...], [[x, z], ...], col]          skin between a bottom and a top outline (same point count)
//   ["rad",  n, R, a0, part]                                    n copies of part around the y axis at radius R (part built facing +x)
//   ["rep",  n, dx, dy, dz, part]                               n copies stepping by (dx, dy, dz)
//   ["grp",  x, y, z, rotY, scale, [parts], tiltX, tiltZ]       transformed group; optional tilts (deg) about x then z, applied before rotY
//   ["mirx", [parts]] / ["mirz", [parts]]                       parts plus their mirror image across x = 0 / z = 0
// Colours: "#rrggbb".
const LM_DEG = Math.PI / 180;
function lmColor(c) {
  const col = new THREE.Color(typeof c === 'string' ? c : '#999999');
  return col;
}
function lmPrim(p) {
  const t = p[0];
  let g = null, col = null;
  switch (t) {
    case 'box': {
      const [, cx, y0, cz, sx, h, sz, c, rot = 0] = p;
      g = new THREE.BoxGeometry(sx, h, sz).translate(0, h / 2, 0).rotateY(-rot * LM_DEG).translate(cx, y0, cz); col = c; break;
    }
    case 'cyl': {
      const [, cx, y0, cz, rb, rt, h, c, segs = 12] = p;
      g = new THREE.CylinderGeometry(rt, rb, h, Math.max(3, Math.min(32, segs)), 1).translate(cx, y0 + h / 2, cz); col = c; break;
    }
    case 'pyr': {
      const [, cx, y0, cz, sx, h, sz, c, rot = 0, top = 0] = p;
      g = new THREE.CylinderGeometry(top * Math.SQRT1_2, Math.SQRT1_2, 1, 4, 1).rotateY(Math.PI / 4).scale(sx, h, sz)
        .translate(0, h / 2, 0).rotateY(-rot * LM_DEG).translate(cx, y0, cz); col = c; break;
    }
    case 'sph': {
      const [, cx, cy, cz, rx, ry, rz, c, dome = 0, segs = 12] = p;
      const s = Math.max(4, Math.min(32, segs));
      g = new THREE.SphereGeometry(1, s, Math.max(3, Math.round(s / 2)), 0, Math.PI * 2, 0, dome ? Math.PI / 2 : Math.PI).scale(rx, ry, rz).translate(cx, cy, cz); col = c; break;
    }
    case 'gable': {
      const [, cx, y0, cz, sx, h, sz, c, rot = 0] = p;
      const sh = new THREE.Shape([new THREE.Vector2(-sz / 2, 0), new THREE.Vector2(sz / 2, 0), new THREE.Vector2(0, h)]);
      g = new THREE.ExtrudeGeometry(sh, { depth: sx, bevelEnabled: false }).translate(0, 0, -sx / 2).rotateY(Math.PI / 2)
        .rotateY(-rot * LM_DEG).translate(cx, y0, cz); col = c; break;
    }
    case 'arch': {
      const [, cx, y0, cz, w, h, d, tk, c, rot = 0] = p;
      const W = w / 2 + tk, r = w / 2, spring = Math.max(0.01, h - tk - r);
      const sh = new THREE.Shape([new THREE.Vector2(-W, 0), new THREE.Vector2(W, 0), new THREE.Vector2(W, h), new THREE.Vector2(-W, h)]);
      const hole = new THREE.Path();
      hole.moveTo(-r, 0); hole.lineTo(-r, spring); hole.absarc(0, spring, r, Math.PI, 0, true); hole.lineTo(r, 0); hole.lineTo(-r, 0);
      sh.holes.push(hole);
      g = new THREE.ExtrudeGeometry(sh, { depth: d, bevelEnabled: false, curveSegments: 8 }).translate(0, 0, -d / 2)
        .rotateY(-rot * LM_DEG).translate(cx, y0, cz); col = c; break;
    }
    case 'beam': {
      const [, x1, y1, z1, x2, y2, z2, r, c, segs = 4] = p;
      const a = new THREE.Vector3(x1, y1, z1), b = new THREE.Vector3(x2, y2, z2), d = b.clone().sub(a), L = d.length();
      if (L < 1e-3) return null;
      g = new THREE.CylinderGeometry(r, r, L, Math.max(3, Math.min(12, segs)), 1).translate(0, L / 2, 0);
      g.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize())));
      g.translate(x1, y1, z1); col = c; break;
    }
    case 'lathe': {
      const [, cx, y0, cz, prof, c, segs = 16] = p;
      if (!Array.isArray(prof) || prof.length < 2) return null;
      g = new THREE.LatheGeometry(prof.map(([r, y]) => new THREE.Vector2(Math.max(0, r), y)), Math.max(3, Math.min(32, segs))).translate(cx, y0, cz); col = c; break;
    }
    case 'ext': {
      const [, y0, h, pts, c] = p;
      if (!Array.isArray(pts) || pts.length < 3) return null;
      const sh = new THREE.Shape(pts.map(([x, z]) => new THREE.Vector2(x, -z)));
      g = new THREE.ExtrudeGeometry(sh, { depth: h, bevelEnabled: false }).rotateX(-Math.PI / 2).translate(0, y0, 0); col = c; break;
    }
    case 'loft': {
      // skin between two outlines with the same number of points (top points may repeat to make triangles)
      const [, y0, h, lo, hi, c] = p;
      if (!Array.isArray(lo) || !Array.isArray(hi) || lo.length < 3 || lo.length !== hi.length) return null;
      const n = lo.length, P = [], V3 = (q, y) => [q[0], y, q[1]];
      const tri = (a, b, d) => { if (Math.hypot(a[0] - b[0], a[2] - b[2], a[1] - b[1]) < 1e-6 || Math.hypot(b[0] - d[0], b[2] - d[2], b[1] - d[1]) < 1e-6 || Math.hypot(a[0] - d[0], a[2] - d[2], a[1] - d[1]) < 1e-6) return; P.push(...a, ...b, ...d); };
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n, a = V3(lo[i], y0), b = V3(lo[j], y0), t0 = V3(hi[i], y0 + h), t1 = V3(hi[j], y0 + h);
        tri(a, b, t1); tri(a, t1, t0);
      }
      for (let i = 1; i < n - 1; i++) { tri(V3(lo[0], y0), V3(lo[i + 1], y0), V3(lo[i], y0)); tri(V3(hi[0], y0 + h), V3(hi[i], y0 + h), V3(hi[i + 1], y0 + h)); }
      g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); col = c; break;
    }
    case 'ring': {
      const [, cx, cy, cz, R, tube, c, rot = 0, flat = 0] = p;
      g = new THREE.TorusGeometry(R, tube, 4, Math.max(8, Math.min(48, Math.round(R * 2))));
      if (flat) g.rotateX(Math.PI / 2);
      g.rotateY(-rot * LM_DEG).translate(cx, cy, cz); col = c; break;
    }
    default: return null;
  }
  return { g, col };
}
function lmFlip(g) {                 // mirrored geometry: restore winding
  const a = g.attributes.position.array;
  for (let i = 0; i < a.length; i += 9) for (let k = 0; k < 3; k++) { const t = a[i + 3 + k]; a[i + 3 + k] = a[i + 6 + k]; a[i + 6 + k] = t; }
  return g;
}
// parts -> list of { g (non-indexed), col }
function lmParts(parts, out, depth = 0) {
  if (!Array.isArray(parts) || depth > 6) return out;
  for (const p of parts) {
    if (!Array.isArray(p)) continue;
    try {
      if (p[0] === 'rad') {
        const [, n, R, a0 = 0, part] = p;
        const sub = lmParts([part], [], depth + 1);
        for (let k = 0; k < Math.min(n, 200); k++) {
          const a = (a0 + k * 360 / n) * LM_DEG;
          for (const s of sub) out.push({ g: s.g.clone().translate(R, 0, 0).rotateY(-a), col: s.col });
        }
      } else if (p[0] === 'rep') {
        const [, n, dx, dy, dz, part] = p;
        const sub = lmParts([part], [], depth + 1);
        for (let k = 0; k < Math.min(n, 200); k++) for (const s of sub) out.push({ g: s.g.clone().translate(dx * k, dy * k, dz * k), col: s.col });
      } else if (p[0] === 'grp') {
        const [, x, y, z, rot = 0, sc = 1, list, tx = 0, tz = 0] = p;
        for (const s of lmParts(list, [], depth + 1)) out.push({ g: s.g.scale(sc, sc, sc).rotateX(tx * LM_DEG).rotateZ(tz * LM_DEG).rotateY(-rot * LM_DEG).translate(x, y, z), col: s.col });
      } else if (p[0] === 'mirx' || p[0] === 'mirz') {
        const sub = lmParts(p[1], [], depth + 1);
        for (const s of sub) {
          out.push(s);
          out.push({ g: lmFlip(s.g.clone().scale(p[0] === 'mirx' ? -1 : 1, 1, p[0] === 'mirz' ? -1 : 1)), col: s.col });
        }
      } else {
        const r = lmPrim(p);
        if (r) out.push({ g: r.g.index ? r.g.toNonIndexed() : r.g, col: r.col });
      }
    } catch (e) { /* skip a malformed part */ }
  }
  return out;
}
function buildLandmarkGeometry(parts) {
  const list = lmParts(parts, []);
  let n = 0;
  for (const s of list) n += s.g.attributes.position.count;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let o = 0;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity, y1 = 0;
  for (const s of list) {
    const P = s.g.attributes.position.array, c = lmColor(s.col);
    pos.set(P, o * 3);
    for (let i = 0; i < P.length / 3; i++) {
      col[(o + i) * 3] = c.r; col[(o + i) * 3 + 1] = c.g; col[(o + i) * 3 + 2] = c.b;
      const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
      if (y < 2) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z); }
      y1 = Math.max(y1, y);
    }
    o += P.length / 3;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.userData.base = isFinite(x0) ? { x0, x1, z0, z1 } : { x0: -5, x1: 5, z0: -5, z1: 5 };
  g.userData.height = y1;
  return g;
}

// ---- placing landmarks in the world
// Every landmark has a real position (Wikidata) and a compass heading. Near the car its model is built once; the
// worker then snaps buildings to the matching OSM footprint (centre and long axis) and drops the generic buildings
// under it, bridges snap onto the bridge road's deck, and a coarse ground-level occupancy grid makes it solid.
const LM_VIEW = TILE_SIZE * (FAR_RADIUS + 0.5);
const landmarkMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.85, side: THREE.DoubleSide });
const LANDMARKS = (() => {
  let list = [];
  try { list = (CW.landmarks || []); } catch (e) { console.warn('landmark data', e); }
  return list.map(L => ({
    ...L,
    // some landmarks come in versions; one is picked each time the game loads
    parts: Array.isArray(L.variants) && L.variants.length ? L.variants[Math.floor(Math.random() * L.variants.length)] : L.parts,
    x: (lonToU(L.lon) - U0) * SCALE, z: (latToV(L.lat) - V0) * SCALE,
    rot: -(L.heading || 0) * LM_DEG,   // three.js rotation.y (heading is clockwise)
    geo: null, mesh: null, grid: null, snap: null, placed: false, y: null,
  }));
})();
// polished steel without an environment map: shade each steel-coloured face by which way it looks - sky above,
// the city's greys around the horizon, the ground below (Cloud Gate)
function chromeShade(g) {
  const C = g.attributes.color.array, N = g.attributes.normal.array;
  const sky = new THREE.Color(0xeef5fb), hz = new THREE.Color(0xb7c3cd), city = new THREE.Color(0x7d858d), gnd = new THREE.Color(0x5b5f5f), c = new THREE.Color();
  for (let i = 0; i < C.length; i += 3) {
    const r = C[i], gg = C[i + 1], b = C[i + 2];
    if (!(r > 0.35 && Math.abs(r - gg) < 0.06 && b >= r)) continue;          // only the steel, not the visor or the plaza
    const t = N[i + 1];
    if (t > 0) c.copy(hz).lerp(sky, Math.min(1, t * 1.3));
    else c.copy(hz).lerp(gnd, Math.min(1, -t * 1.6));
    if (Math.abs(t) < 0.22) c.lerp(city, 0.45 * (1 - Math.abs(t) / 0.22));   // the skyline reflected round the middle
    C[i] = c.r; C[i + 1] = c.g; C[i + 2] = c.b;
  }
  g.attributes.color.needsUpdate = true;
}
function lmGeometry(L) {
  if (!L.geo) {
    L.geo = buildLandmarkGeometry(L.parts);
    if (L.chrome) chromeShade(L.geo);
    const b = L.geo.userData.base;
    L.hx = Math.max(3, Math.max(-b.x0, b.x1)); L.hz = Math.max(3, Math.max(-b.z0, b.z1));
  }
  return L.geo;
}
// model space <-> world (rotation.y = rot)
const lmToWorld = (L, mx, mz) => [L.px + mx * Math.cos(L.prot) + mz * Math.sin(L.prot), L.pz - mx * Math.sin(L.prot) + mz * Math.cos(L.prot)];
const lmToModel = (L, wx, wz) => { const dx = wx - L.px, dz = wz - L.pz; return [dx * Math.cos(L.prot) - dz * Math.sin(L.prot), dx * Math.sin(L.prot) + dz * Math.cos(L.prot)]; };

/** Half-size of the level pad under a landmark: its footprint plus an apron; statues, monuments and towers get
 *  at least an 18 m platform so a summit statue stands on a real terrace. */
function lmPad(L) {
  const small = L.kind === 'statue' || L.kind === 'monument' || L.kind === 'tower' ? 18 : 0;
  return [Math.max(small, L.hx + Math.max(4, 0.12 * L.hx)), Math.max(small, L.hz + Math.max(4, 0.12 * L.hz))];
}
/** Landmarks whose base could reach into a tile, in tile-local coordinates, for the worker request. */
function landmarksForTile(ox, oz) {
  const out = [];
  for (const L of LANDMARKS) {
    if (L.kind === 'dam') continue;
    // as far out as the worker's road window reaches (plus the distance roads look for landmarks), so neighbouring
    // tiles see the same landmarks and agree on the roads between them
    if (L.x < ox - 900 || L.z < oz - 900 || L.x > ox + TILE_SIZE + 900 || L.z > oz + TILE_SIZE + 900) continue;
    lmGeometry(L);
    out.push({ qid: L.qid, x: L.x - ox, z: L.z - oz, hx: L.hx, hz: L.hz, h: L.geo.userData.height, rot: L.rot, kind: L.kind, pits: L.pits || null, anchor: L.anchor || null, pax: lmPad(L)[0], paz: lmPad(L)[1], bridge: L.kind === 'bridge',
                own: L.kind !== 'bridge' && !L.fixed && L.x >= ox && L.z >= oz && L.x < ox + TILE_SIZE && L.z < oz + TILE_SIZE });
  }
  return out;
}
/** Worker reply: footprint snaps for landmarks whose point lies in this tile. */
function applyLandmarkSnaps(t, snaps) {
  if (!snaps) return;
  for (const s of snaps) {
    const L = LANDMARKS.find(l => l.qid === s.qid);
    if (!L) continue;
    L.snap = { x: t.ox + s.x, z: t.oz + s.z, rot: s.rot };
    L.placed = false;
  }
}
// bridges: onto the nearest bridge road, deck axis along the road, towers standing on the deck height
function snapBridge(L) {
  let best = null;
  for (const t of tiles.values()) {
    if (!t.segs || t.lod !== 1) continue;
    if (L.x < t.ox - 400 || L.z < t.oz - 400 || L.x > t.ox + TILE_SIZE + 400 || L.z > t.oz + TILE_SIZE + 400) continue;
    const s = t.segs, lx = L.x - t.ox, lz = L.z - t.oz;
    for (let o = 0; o < s.length; o += SEG) {
      if (s[o + 8] !== 1 || CLASS_LIST[s[o + 7]] === 'path') continue;
      const ax = s[o], az = s[o + 1], dx = s[o + 2] - ax, dz = s[o + 3] - az, l2 = dx * dx + dz * dz || 1e-9;
      const u = Math.max(0, Math.min(1, ((lx - ax) * dx + (lz - az) * dz) / l2));
      const qx = ax + dx * u, qz = az + dz * u, d = Math.hypot(lx - qx, lz - qz);
      if (d < 80 && (!best || d < best.d)) best = { d, x: t.ox + qx, z: t.oz + qz, h: s[o + 9] + (s[o + 10] - s[o + 9]) * u, phi: Math.atan2(dz, dx), hw: s[o + 4] };
    }
  }
  if (!best) return null;
  // model x along the road: rotation -phi or -phi + PI, whichever is nearer the authored heading
  let r = -best.phi;
  const dd = Math.atan2(Math.sin(r - L.rot), Math.cos(r - L.rot));
  if (Math.abs(dd) > Math.PI / 2) r += Math.PI;
  return { x: best.x, z: best.z, rot: r, y: best.h - (L.deck || 0), hw: best.hw };
}
// A bridge model must leave the road free: once it's on its road, everything inside the traffic corridor
// (road width plus a margin, deck level to 5.5 m above) is cut out of it. Triangles are split exactly at the
// corridor walls, so towers and portals end up with clean openings instead of missing panels.
function clipCorridor(src, deck, lanes) {
  let P = src.attributes.position.array, C = src.attributes.color.array;
  for (const [lo, hi] of lanes) { const r = clipSlab(P, C, deck, lo, hi); P = r[0]; C = r[1]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(C, 3));
  g.computeVertexNormals(); g.computeBoundingSphere();
  g.userData = { ...src.userData };
  return g;
}
/** Cut the box lo < z < hi, deck-0.3 < y < deck+5.5 out of a triangle soup; returns [positions, colours]. */
function clipSlab(P, C, deck, lo, hi) {
  const outP = [], outC = [];
  const y0 = deck - 0.3, y1 = deck + 5.5;
  const clip = (poly, ax, val, keepAbove) => {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const da = keepAbove ? a[ax] - val : val - a[ax], db = keepAbove ? b[ax] - val : val - b[ax];
      if (da >= 0) out.push(a);
      if ((da >= 0) !== (db >= 0)) { const t = da / (da - db); out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]); }
    }
    return out;
  };
  const emit = (poly, i) => {
    for (let k = 1; k < poly.length - 1; k++) for (const v of [poly[0], poly[k], poly[k + 1]]) { outP.push(v[0], v[1], v[2]); outC.push(C[i], C[i + 1], C[i + 2]); }
  };
  for (let i = 0; i < P.length; i += 9) {
    const tri = [[P[i], P[i + 1], P[i + 2]], [P[i + 3], P[i + 4], P[i + 5]], [P[i + 6], P[i + 7], P[i + 8]]];
    const zmin = Math.min(P[i + 2], P[i + 5], P[i + 8]), zmax = Math.max(P[i + 2], P[i + 5], P[i + 8]);
    const ymin = Math.min(P[i + 1], P[i + 4], P[i + 7]), ymax = Math.max(P[i + 1], P[i + 4], P[i + 7]);
    if (zmin >= hi || zmax <= lo || ymin >= y1 || ymax <= y0) { emit(tri, i); continue; }
    emit(clip(tri, 2, hi, true), i);
    emit(clip(tri, 2, lo, false), i);
    const mid = clip(clip(tri, 2, hi, false), 2, lo, true);
    emit(clip(mid, 1, y1, true), i);
    emit(clip(mid, 1, y0, false), i);
  }
  return [outP, outC];
}
/** The traffic lanes through a placed bridge model: every road or rail line on a bridge at the model's deck
 *  height, as model-space z intervals (its offset from the axis plus its half-width and a margin), merged. */
function bridgeLanes(L, fallbackHw) {
  const cos = Math.cos(L.prot), sin = Math.sin(L.prot), b = L.geo.userData.base, deckY = L.y + (L.deck || 0), lines = new Map();
  for (const t of tiles.values()) {
    if (!t.segs || t.lod !== 1) continue;
    if (L.px < t.ox - 2500 || L.pz < t.oz - 2500 || L.px > t.ox + TILE_SIZE + 2500 || L.pz > t.oz + TILE_SIZE + 2500) continue;
    const s = t.segs;
    for (let o = 0; o < s.length; o += SEG) {
      if (s[o + 8] !== 1 || CLASS_LIST[s[o + 7]] === 'path') continue;
      for (const f of [0, 0.5, 1]) {
        const wx = t.ox + s[o] + (s[o + 2] - s[o]) * f, wz = t.oz + s[o + 1] + (s[o + 3] - s[o + 1]) * f, h = s[o + 9] + (s[o + 10] - s[o + 9]) * f;
        if (Math.abs(h - deckY) > 3.5) continue;
        const dx = wx - L.px, dz = wz - L.pz, mx = dx * cos - dz * sin, mz = dx * sin + dz * cos;
        if (mx < b.x0 - 5 || mx > b.x1 + 5 || Math.abs(mz) > 60) continue;
        const id = s[o + 12], q = lines.get(id) || { lo: Infinity, hi: -Infinity, hw: s[o + 4] };
        q.lo = Math.min(q.lo, mz); q.hi = Math.max(q.hi, mz); lines.set(id, q);
      }
    }
  }
  const iv = [...lines.values()].map(q => [q.lo - q.hw - 0.8, q.hi + q.hw + 0.8]);
  if (!iv.length) iv.push([-Math.max(4.5, fallbackHw + 0.8), Math.max(4.5, fallbackHw + 0.8)]);
  iv.sort((a, c) => a[0] - c[0]);
  const out = [];
  for (const v of iv) { const last = out[out.length - 1]; if (last && v[0] <= last[1]) last[1] = Math.max(last[1], v[1]); else out.push([...v]); }
  return out;
}
function cutBridge(L, hw) {
  const lanes = bridgeLanes(L, hw), key = lanes.map(v => v[0].toFixed(1) + ',' + v[1].toFixed(1)).join(';');
  if (key === L.cutKey) return;
  if (L.mesh.geometry !== L.geo) L.mesh.geometry.dispose();
  L.mesh.geometry = clipCorridor(L.geo, L.deck || 0, lanes);
  L.cutKey = key;
}

// Footbridges (bridge landmarks with no road to snap to): give them a walkway deck along their length at deck
// height, as wide as the structure around the deck, and make it solid ground in groundAt.
const footDecks = [];
const footDeckMat = new THREE.MeshStandardMaterial({ color: 0xa39c8e, roughness: 0.9 });
function makeFootDeck(L) {
  if (L.footDeck) return;
  const P = L.geo.attributes.position.array, deck = L.deck || 0;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (let i = 0; i < P.length; i += 3) {
    x0 = Math.min(x0, P[i]); x1 = Math.max(x1, P[i]);
    if (P[i + 1] > deck - 2 && P[i + 1] < deck + 3) { z0 = Math.min(z0, P[i + 2]); z1 = Math.max(z1, P[i + 2]); }
  }
  const w = isFinite(z0) ? Math.max(3, Math.min(8, Math.min(-z0, z1) * 2 - 0.6)) : 5, len = x1 - x0, cx = (x0 + x1) / 2;
  const slab = new THREE.Mesh(new THREE.BoxGeometry(len, 0.6, w), footDeckMat);
  slab.position.set(cx, deck - 0.25, 0);
  slab.castShadow = slab.receiveShadow = true;
  L.mesh.add(slab);
  L.footDeck = { cx, half: len / 2, w, top: deck + 0.05, ramp: Math.min(25, len * 0.15), reach: len / 2 + Math.abs(cx) + w };
  footDecks.push(L);
}

// solid parts: cells (1.5 m) where the model has geometry between 0.6 and 2.4 m above its base
function lmGrid(L) {
  if (L.grid) return L.grid;
  const P = L.geo.attributes.position.array, CS = Math.max(L.hx, L.hz) < 150 ? 0.75 : 1.5, cells = new Set();
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (let i = 0; i < P.length; i += 9) {
    const ya = P[i + 1], yb = P[i + 4], yc = P[i + 7];
    if (Math.min(ya, yb, yc) > 2.4 || Math.max(ya, yb, yc) < 0.6) continue;
    const e = Math.max(Math.hypot(P[i] - P[i + 3], P[i + 2] - P[i + 5], ya - yb), Math.hypot(P[i] - P[i + 6], P[i + 2] - P[i + 8], ya - yc));
    const n = Math.min(160, Math.max(1, Math.ceil(e / (CS * 0.6))));
    for (let a = 0; a <= n; a++) for (let b = 0; a + b <= n; b++) {
      const u = a / n, v = b / n, w = 1 - u - v;
      const y = P[i + 1] * w + P[i + 4] * u + P[i + 7] * v;
      if (y < 0.6 || y > 2.4) continue;
      const x = P[i] * w + P[i + 3] * u + P[i + 6] * v, z = P[i + 2] * w + P[i + 5] * u + P[i + 8] * v;
      cells.add(Math.floor(x / CS) * 65536 + Math.floor(z / CS));
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
    }
  }
  return (L.grid = { cells, CS, r: isFinite(x0) ? Math.hypot(Math.max(-x0, x1), Math.max(-z0, z1)) + 2 : 0 });
}
/** Push a circle (world x, z, radius) out of the solid parts of nearby landmarks; [dx, dz] or null. */
function pushOutOfLandmarks(px, pz, rad) {
  let mx = 0, mz = 0, hit = false;
  for (const L of lmActive) {
    const G = L.grid;
    if (!G || !G.cells.size || Math.hypot(px - L.px, pz - L.pz) > G.r + rad) continue;
    const [ux, uz] = lmToModel(L, px, pz), CS = G.CS, reach = Math.ceil((rad + CS) / CS);
    const ci = Math.floor(ux / CS), cj = Math.floor(uz / CS);
    let sx = 0, sz = 0, n = 0, dmin = Infinity;
    for (let j = -reach; j <= reach; j++) for (let i = -reach; i <= reach; i++) {
      if (!G.cells.has((ci + i) * 65536 + cj + j)) continue;
      const cx = (ci + i + 0.5) * CS, cz = (cj + j + 0.5) * CS, d = Math.hypot(ux - cx, uz - cz);
      if (d > rad + CS * 0.75) continue;
      sx += cx; sz += cz; n++; dmin = Math.min(dmin, d);
    }
    if (!n) continue;
    let nx = ux - sx / n, nz = uz - sz / n;
    const l = Math.hypot(nx, nz) || 1; nx /= l; nz /= l;
    const push = Math.max(0.05, rad + CS * 0.75 - dmin);
    // back to world space
    const wx = nx * Math.cos(L.prot) + nz * Math.sin(L.prot), wz = -nx * Math.sin(L.prot) + nz * Math.cos(L.prot);
    mx += wx * push; mz += wz * push; hit = true;
  }
  return hit ? [mx, mz] : null;
}
// ---- the skyline: very tall landmarks (Burj Khalifa, CN Tower...) stay visible far beyond the world's fog and
// draw distance, out to their horizon. Beyond SKY_NEAR the real model is swapped for a stand-in drawn with its own
// light haze (not the scene fog). Past the camera's far plane the stand-in is shrunk and pulled in along the line of
// sight by the same factor, which looks identical; it also sinks behind the curve of the earth.
const SKY_TALL = 150, SKY_NEAR = 4000;
let skyList = [];
function skyMat(L) {
  const m = landmarkMat.clone();
  m.fog = false;
  m.userData.haze = { value: new THREE.Vector4(0.66, 0.78, 0.9, 0) };
  m.onBeforeCompile = sh => {
    sh.uniforms.uHaze = m.userData.haze;
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform vec4 uHaze;')
      .replace('#include <dithering_fragment>', '#include <dithering_fragment>\ngl_FragColor.rgb = mix(gl_FragColor.rgb, uHaze.rgb, uHaze.a);');
  };
  m.customProgramCacheKey = () => 'landmark-sky-v1';
  return m;
}
function updateSkyline() {
  const C = camera.position, Dmax = camera.far * 0.8, Re = 6371000 / 0.87;   // refraction lifts the horizon a little
  for (const L of skyList) {
    const gy = L.y ?? FOC.y - 1, h = L.geo.userData.height, dx = L.px - C.x, dz = L.pz - C.z, d = Math.hypot(dx, dz);
    const drop = d * d / (2 * Re), k = Math.min(1, Dmax / Math.max(1, d));
    const m = L.sky;
    m.position.set(C.x + dx * k, C.y + (gy - drop - C.y) * k, C.z + dz * k);
    m.scale.setScalar(k);
    m.rotation.y = L.prot ?? L.rot;
    const hz = m.material.userData.haze.value, fc = scene.fog.color;
    hz.set(fc.r, fc.g, fc.b, Math.min(0.85, Math.max(0, 1 - Math.exp(-(d - 3000) / 38000))));
    m.visible = gy - drop + h > C.y - 1e4 && (d > 20000 || lineOfSight(C.x, C.y, C.z, L.px, gy + h * 0.8, L.pz));
  }
}

let lmActive = [], lmTimer = 0, lmNear = null;
function updateLandmarks(dt) {
  lmTimer -= dt;
  if (lmTimer > 0) return;
  lmTimer = 0.5;
  const act = [], sky = [];
  lmNear = null;
  for (const L of LANDMARKS) {
    const d = Math.hypot(L.x - FOC.x, L.z - FOC.z);
    // tall enough to see over the horizon from here? (horizon distance 3.57 km x sqrt(height), plus some for the eye)
    if (d > SKY_NEAR && d < 130000 && (L.geo || d < 120000)) {
      const g = lmGeometry(L), h = g.userData.height;
      if (h >= SKY_TALL && d < 3570 * (Math.sqrt(h) + Math.sqrt(Math.max(2, FOC.y - (L.y ?? FOC.y) + 2))) * 1.05) {
        if (!L.sky) { L.sky = new THREE.Mesh(g, skyMat(L)); L.sky.frustumCulled = false; scene.add(L.sky); }
        if (L.px === undefined) { L.px = L.x; L.pz = L.z; L.prot = L.rot; }
        sky.push(L);
      }
    }
    if (d > LM_VIEW) { if (L.mesh) L.mesh.visible = false; continue; }
    lmGeometry(L);
    if (!L.placed) {
      // position: footprint snap > bridge snap > authored; height: lowest ground under the base
      let p = L.snap ? { ...L.snap } : { x: L.x, z: L.z, rot: L.rot }, bs = null;
      if (L.kind === 'bridge') { bs = snapBridge(L); if (bs) p = bs; }
      L.px = p.x; L.pz = p.z; L.prot = p.rot;
      let y = p.y ?? null;
      const hs = [];
      if (y === null) {
        for (const [a, b] of [[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const [wx, wz] = lmToWorld(L, a * L.hx * 0.9, b * L.hz * 0.9), h = terrainHeight(wx, wz);
          if (h !== null) hs.push(h);
        }
        // terrain not loaded that far out yet: stand it at the car's ground level for now (refined once it loads)
        // sunken landmarks cut their own pits, so the lowest sample would be a pit floor: take the middle one
        y = hs.length ? (L.pits ? hs.sort((a, b) => a - b)[hs.length >> 1] : Math.min(...hs)) - 0.3 : (terrainHeight(FOC.x, FOC.z) ?? FOC.y) - 1;
        // carvings in a mountain (Mount Rushmore) hang off the ground at their centre instead of the lowest point
        if (L.anchor === 'centre' && hs.length) y = hs[0] + (L.yoff || 0);
        if (!hs.length) L.placed = false;
      }
      L.y = y;
      if (!L.mesh) {
        L.mesh = new THREE.Mesh(L.geo, landmarkMat);
        L.mesh.castShadow = L.mesh.receiveShadow = true;
        scene.add(L.mesh);
      }
      if (Math.abs(L.mesh.position.x - L.px) + Math.abs(L.mesh.position.z - L.pz) > 0.5 || Math.abs(L.mesh.rotation.y - L.prot) > 0.01 || !L.mesh.userData.set) treesDirty = true;
      L.mesh.userData.set = true;
      L.mesh.position.set(L.px, y, L.pz);
      L.mesh.rotation.y = L.prot;
      // bridges: clear every traffic lane crossing the deck out of the model
      if (bs) { L.cutHw = bs.hw; cutBridge(L, bs.hw); }
      // keep it provisional until the detailed tile under it has reported (snaps arrive with its buildings)
      const [ttx, tty] = tileAt(L.px, L.pz), tt = tiles.get(`${ttx},${tty}`);
      const ready = !!(tt && tt.lod === 1 && tt.buildingInfo !== undefined);
      // a bridge with no road under it is a footbridge: it gets its own walkway deck, level with its banks
      if (L.kind === 'bridge' && !bs && ready) {
        const P = L.geo.attributes.position.array; let x0 = Infinity, x1 = -Infinity;
        for (let i = 0; i < P.length; i += 3) { x0 = Math.min(x0, P[i]); x1 = Math.max(x1, P[i]); }
        const ends = [x0 + 3, x1 - 3].map(mx => { const [wx, wz] = lmToWorld(L, mx, 0); return terrainHeight(wx, wz); }).filter(h => h !== null);
        if (ends.length) { L.y = y = Math.max(...ends) - (L.deck || 0) - 0.05; L.mesh.position.y = y; }
        makeFootDeck(L);
      }
      L.placed = (hs.length > 0 || p.y != null) && ready && (L.kind !== 'bridge' || !!bs || !!L.footDeck);
    }
    L.mesh.visible = (d < 600 || !landmarkHidden(L)) && !sky.includes(L);   // far tall ones: the skyline stand-in draws them
    if (L.kind === 'bridge' && L.placed && L.cutHw && d < 3000 && (L.cutCheck = (L.cutCheck || 0) + 1) % 6 === 0) cutBridge(L, L.cutHw);
    if (d < 3000 && !L.footDeck) { lmGrid(L); act.push(L); }     // footbridges: walk on, not into (their piers stand in the river)
    if (d < 400 && (!lmNear || d < lmNear.d)) lmNear = { d, name: L.name };
  }
  lmActive = act;
  for (const L of skyList) if (!sky.includes(L)) L.sky.visible = false;
  skyList = sky;
}

// ===================================================================== bridges / culverts (built in the worker)
const structMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, side: THREE.DoubleSide });

// ===================================================================== buildings

const BUILD_FRAG_HEAD = /* glsl */`
  uniform sampler2D uPhoto;
  uniform float uHasPhoto;
  varying vec4 vBw;      // along wall (m), height above base (m), wall height to the roof line (m), roof + 2 * type
  varying vec3 vWorldP;
  varying vec2 vTileUv;
  float bnd(float v, float a, float b, float w) { return smoothstep(a - w, a + w, v) * (1.0 - smoothstep(b - w, b + w, v)); }
  // a grid of rectangles: cell coords g, rectangle from lo to hi inside each cell
  float cells(vec2 g, vec2 lo, vec2 hi) {
    vec2 f = fract(g), aa = fwidth(g) * 0.7 + 0.005;
    return bnd(f.x, lo.x, hi.x, aa.x) * bnd(f.y, lo.y, hi.y, aa.y);
  }
`;
const BUILD_FRAG_MAIN = /* glsl */`
  #include <color_fragment>
  float wcode = vBw.w + 0.01;
  float isRoof = mod(floor(wcode), 2.0);
  float ty = floor(wcode / 2.0);
  if (isRoof > 0.5) {
    // roofs: flat roof colour (set per building in the worker)
    diffuseColor.rgb = vColor;
  } else if (ty < 10.5) {
    float y = vBw.y, wh = vBw.z, al = vBw.x;
    float dist = distance(vWorldP, cameraPosition), near = 1.0 - smoothstep(250.0, 600.0, dist), fine = 1.0 - smoothstep(40.0, 140.0, dist);
    vec3 glass = vec3(0.035, 0.045, 0.06);
    float win = 0.0;
    vec3 base = diffuseColor.rgb;
    float inWall = step(0.6, y) * step(y, wh - 0.3);
    if (ty < 0.5) {                                   // house: spaced windows, a door on some walls
      float fh = wh / max(1.0, floor(wh / 2.9));
      win = cells(vec2(al / 3.8, y / fh), vec2(0.3, 0.34), vec2(0.7, 0.8)) * inWall;
      float door = bnd(fract(al / 11.0), 0.45, 0.53, 0.004) * step(y, 2.6) * step(fract(al / 33.0), 0.34);
      base = mix(base, base * 0.45, door);
    } else if (ty < 1.5) {                            // apartment: dense windows, balcony slabs
      win = cells(vec2(al / 2.7, y / 3.0), vec2(0.18, 0.3), vec2(0.82, 0.86)) * inWall;
      base *= 1.0 - 0.18 * bnd(fract(y / 3.0), 0.0, 0.07, 0.01) * step(3.0, y);
    } else if (ty < 2.5) {                            // shop: glazed ground floor, offices or flats above
      float front = bnd(fract(al / 3.2), 0.05, 0.95, 0.01) * bnd(y, 0.7, 3.6, 0.03);
      base = mix(base, vec3(0.16, 0.15, 0.13), front * 0.9);
      base = mix(base, base * 0.5, bnd(y, 3.6, 3.8, 0.02));
      win = cells(vec2(al / 2.8, (y - 4.4) / 3.0), vec2(0.22, 0.3), vec2(0.78, 0.85)) * step(5.0, y) * inWall;
    } else if (ty < 3.5) {                            // office: curtain wall
      win = cells(vec2(al / 1.6, y / 3.6), vec2(0.05, 0.14), vec2(0.95, 0.94)) * inWall;
      glass = vec3(0.08, 0.12, 0.17);
    } else if (ty < 4.5) {                            // warehouse: ribbed metal, roller doors, high strip lights
      base *= 1.0 - 0.07 * fine * step(0.5, fract(al / 0.3));
      float door = bnd(fract(al / 17.0), 0.34, 0.6, 0.004) * step(y, 4.8) * step(6.0, wh);
      base = mix(base, vec3(0.4, 0.42, 0.44) * (0.92 + 0.08 * step(0.5, fract(y * 3.0))), door);
      win = bnd(y, wh - 1.6, wh - 1.0, 0.03) * bnd(fract(al / 4.5), 0.1, 0.9, 0.02) * step(6.0, wh);
    } else if (ty < 5.5) {                            // school: brick, wide classroom windows
      base *= 1.0 - 0.07 * fine * bnd(fract(y * 13.3), 0.0, 0.14, 0.03);
      win = cells(vec2(al / 3.4, y / 3.6), vec2(0.07, 0.3), vec2(0.93, 0.85)) * inWall;
    } else if (ty < 6.5) {                            // hospital: ribbon windows
      win = bnd(fract(y / 3.7), 0.35, 0.8, 0.02) * (1.0 - bnd(fract(al / 1.8), 0.0, 0.05, 0.01)) * inWall;
      glass = vec3(0.07, 0.1, 0.14);
    } else if (ty < 7.5) {                            // church: tall narrow windows
      win = cells(vec2(al / 4.0, y / max(wh, 3.0)), vec2(0.42, 0.22), vec2(0.58, 0.8));
      glass = vec3(0.07, 0.05, 0.1);
    } else if (ty < 8.5) {                            // barn: vertical boards, a big door
      base *= 1.0 - 0.1 * fine * step(0.86, fract(al / 0.32));
      float door = bnd(fract(al / 26.0), 0.38, 0.62, 0.004) * step(y, 4.0);
      base = mix(base, base * 0.55, door);
    } else if (ty < 9.5) {                            // shed: ribbed, no windows
      base *= 1.0 - 0.06 * fine * step(0.5, fract(al / 0.3));
    } else {                                          // civic: tall formal windows
      win = cells(vec2(al / 3.0, y / 4.2), vec2(0.3, 0.2), vec2(0.7, 0.84)) * inWall;
    }
    win *= near;
    diffuseColor.rgb = mix(base, glass, win * 0.92);
    diffuseColor.rgb *= mix(0.72, 1.0, smoothstep(0.0, 1.5, y));   // grime near the ground
  }
`;
function buildingMaterial(own) {
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85 });
  m.customProgramCacheKey = () => 'building-v2';
  m.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, own, { uTileSize: shared.uTileSize });
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\n uniform float uTileSize; attribute vec4 bw; varying vec4 vBw; varying vec3 vWorldP; varying vec2 vTileUv;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vBw = bw;
        vWorldP = (modelMatrix * vec4(position, 1.0)).xyz;
        vTileUv = position.xz / uTileSize;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\n' + BUILD_FRAG_HEAD)
      .replace('#include <color_fragment>', BUILD_FRAG_MAIN);
  };
  return m;
}

function applyBuildings(t, d) {
  applyLandmarkSnaps(t, d.snaps);
  if (t.buildingMesh) { scene.remove(t.buildingMesh); t.buildingMesh.geometry.dispose(); t.buildingMesh = null; }
  if (!d.mesh) { t.buildingInfo = { count: 0, osm: 0, usa: 0 }; t.foot = d.foot; t.footOff = d.footOff; t.footHash = new Map(); t.propsGen = false; return; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(d.mesh.positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(d.mesh.normals, 3));
  g.setAttribute('color', new THREE.BufferAttribute(d.mesh.colors, 3));
  g.setAttribute('bw', new THREE.BufferAttribute(d.mesh.bw, 4));
  g.setIndex(new THREE.BufferAttribute(d.mesh.indices, 1));
  if (!t.buildingMat) t.buildingMat = buildingMaterial(t.roadU);
  t.buildingMesh = new THREE.Mesh(g, t.buildingMat);
  t.buildingMesh.castShadow = t.buildingMesh.receiveShadow = true;
  t.buildingMesh.position.set(t.ox, 0, t.oz);
  scene.add(t.buildingMesh);
  t.buildingInfo = { count: d.count, osm: d.osm, usa: d.usa };
  // footprints for collisions, hashed into 16 m cells
  t.foot = d.foot; t.footOff = d.footOff; t.footHash = new Map(); t.propsGen = false;
  for (let b = 0; b < d.footOff.length - 1; b++) {
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (let i = d.footOff[b]; i < d.footOff[b + 1]; i++) { x0 = Math.min(x0, d.foot[2 * i]); x1 = Math.max(x1, d.foot[2 * i]); z0 = Math.min(z0, d.foot[2 * i + 1]); z1 = Math.max(z1, d.foot[2 * i + 1]); }
    for (let gz = Math.floor(z0 / 16); gz <= Math.floor(z1 / 16); gz++) for (let gx = Math.floor(x0 / 16); gx <= Math.floor(x1 / 16); gx++) {
      const k = gx * 100003 + gz;
      (t.footHash.get(k) || t.footHash.set(k, []).get(k)).push(b);
    }
  }
}

/** Push a circle (world x, z, radius) out of building footprints; returns [dx, dz] applied or null. */
function pushOutOfBuildings(px, pz, rad, onRoad = false) {
  let mx = 0, mz = 0, hit = false;
  // a mapped road running through a landmark or building is a real passage (arch, gate, arcade, covered street):
  // a car on the road isn't stopped by it
  if (onRoad) return null;
  const lp = lmActive.length ? pushOutOfLandmarks(px, pz, rad) : null;
  if (lp) { mx += lp[0]; mz += lp[1]; hit = true; }
  const [ttx, tty] = tileAt(px, pz), cand = [];
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const tt = tiles.get(`${ttx + dx},${tty + dy}`); if (tt) cand.push(tt); }
  for (const t of cand) {
    if (!t.footHash || px < t.ox - 80 || pz < t.oz - 80 || px > t.ox + TILE_SIZE + 80 || pz > t.oz + TILE_SIZE + 80) continue;
    const lx = px - t.ox, lz = pz - t.oz;
    const seen = new Set();
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
      const list = t.footHash.get((Math.floor(lx / 16) + dx) * 100003 + Math.floor(lz / 16) + dz);
      if (!list) continue;
      for (const b of list) {
        if (seen.has(b)) continue;
        seen.add(b);
        const F = t.foot, i0 = t.footOff[b], i1 = t.footOff[b + 1];
        let inside = false, best = Infinity, bx = 0, bz = 0;
        for (let i = i0, j = i1 - 1; i < i1; j = i++) {
          const xi = F[2 * i], zi = F[2 * i + 1], xj = F[2 * j], zj = F[2 * j + 1];
          if ((zi > lz) !== (zj > lz) && lx < (xj - xi) * (lz - zi) / (zj - zi) + xi) inside = !inside;
          const ex = xi - xj, ez = zi - zj, l2 = ex * ex + ez * ez || 1e-9;
          const u = Math.max(0, Math.min(1, ((lx - xj) * ex + (lz - zj) * ez) / l2));
          const qx = xj + ex * u, qz = zj + ez * u, d = Math.hypot(lx - qx, lz - qz);
          if (d < best) { best = d; bx = qx; bz = qz; }
        }
        if (!inside && best >= rad) continue;
        let nx = lx - bx, nz = lz - bz;
        const nl = Math.hypot(nx, nz) || 1;
        nx /= nl; nz /= nl;
        if (inside) { nx = -nx; nz = -nz; }
        const push = inside ? best + rad : rad - best;
        mx += nx * push; mz += nz * push; hit = true;
      }
    }
  }
  return hit ? [mx, mz] : null;
}

// walls: three circles the car's width along it (front, back, middle), each touch answered at the point of contact
// (carImpulse), so a wall clipped with a corner turns the car instead of throwing it straight back
function collideBuildings() {
  const fx = Math.sin(V.heading), fz = Math.cos(V.heading), rad = Math.max(0.6, CAR.wid / 2), off = Math.max(0, CAR.len / 2 - rad);
  for (const along of [off, -off, 0]) {
    const cx = V.x + fx * along, cz = V.z + fz * along;
    const p = pushOutOfBuildings(cx, cz, rad, V.segOff >= 0 || V.onDeck);
    if (!p) continue;
    const l = Math.hypot(p[0], p[1]);
    if (l < 1e-6) continue;
    const nx = p[0] / l, nz = p[1] / l;
    carImpulse(cx - nx * rad, cz - nz * rad, nx, nz);
    V.x += p[0]; V.z += p[1];
  }
}

// ===================================================================== world props: roadside life
// Generated per tile (main thread) from data we already have — roads, junctions, land class, buildings,
// terrain — and seeded by world position, so every road looks the same on every visit. Drawn as instanced
// low-poly families around the car. Small things can be knocked flying; poles, rails and fences are solid.

// ---- geometry kit: merge coloured primitives into one vertex-coloured geometry
const _eu = new THREE.Euler(), _mk = new THREE.Matrix4();
function part(geo, color, x = 0, y = 0, z = 0, ry = 0, rx = 0, rz = 0) {
  geo.applyMatrix4(_mk.makeRotationFromEuler(_eu.set(rx, ry, rz)));
  geo.translate(x, y, z);
  return { geo, color };
}
const pBox = (w, h, d, c, x, y, z, ry, rx, rz) => part(new THREE.BoxGeometry(w, h, d), c, x, y, z, ry, rx, rz);
const pCyl = (rt, rb, h, seg, c, x, y, z, ry, rx, rz) => part(new THREE.CylinderGeometry(rt, rb, h, seg), c, x, y, z, ry, rx, rz);
function kit(parts) {
  const pos = [], nor = [], col = [], c = new THREE.Color();
  for (const { geo, color } of parts) {
    const g = geo.index ? geo.toNonIndexed() : geo;
    pos.push(...g.attributes.position.array);
    nor.push(...g.attributes.normal.array);
    c.setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace);
    for (let i = 0; i < g.attributes.position.count; i++) col.push(c.r, c.g, c.b);
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  out.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  out.computeBoundingSphere();
  return out;
}
const C = {
  wood: [0.43, 0.34, 0.25], woodGrey: [0.55, 0.52, 0.47], steel: [0.66, 0.68, 0.7], dark: [0.12, 0.12, 0.13],
  white: [0.93, 0.93, 0.91], concrete: [0.68, 0.67, 0.64], tpost: [0.2, 0.38, 0.24], insul: [0.6, 0.66, 0.6],
  amber: [0.95, 0.72, 0.15], red: [0.8, 0.1, 0.1], black: [0.06, 0.06, 0.06], wire: [0.16, 0.16, 0.17],
};

// ---- sign atlas, drawn at runtime in the region's style
const SIGN_PX = 128, SIGN_GRID = 8;
const signCanvas = document.createElement('canvas');
signCanvas.width = signCanvas.height = SIGN_PX * SIGN_GRID;
const signCtx = signCanvas.getContext('2d');
const signTex = new THREE.CanvasTexture(signCanvas);
signTex.colorSpace = THREE.SRGBColorSpace;
signTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
const signCells = new Map();
function signCell(key, draw) {
  let i = signCells.get(key);
  if (i !== undefined) return i;
  i = signCells.size;
  if (i >= SIGN_GRID * SIGN_GRID) return 0;
  signCells.set(key, i);
  signCtx.save();
  signCtx.translate((i % SIGN_GRID) * SIGN_PX, Math.floor(i / SIGN_GRID) * SIGN_PX);
  signCtx.clearRect(0, 0, SIGN_PX, SIGN_PX);
  draw(signCtx);
  signCtx.restore();
  signTex.needsUpdate = true;
  return i;
}
const cellRect = i => [(i % SIGN_GRID) / SIGN_GRID, 1 - (Math.floor(i / SIGN_GRID) + 1) / SIGN_GRID, 1 / SIGN_GRID, 1 / SIGN_GRID];
const poly = (g, pts, fill) => { g.beginPath(); g.moveTo(pts[0], pts[1]); for (let k = 2; k < pts.length; k += 2) g.lineTo(pts[k], pts[k + 1]); g.closePath(); g.fillStyle = fill; g.fill(); };
function warnPlate(g) {
  if (REGION.warn === 'diamond') { poly(g, [64, 2, 126, 64, 64, 126, 2, 64], '#111'); poly(g, [64, 9, 119, 64, 64, 119, 9, 64], '#f2c00f'); g.translate(64, 64); g.scale(0.85, 0.85); }
  else { poly(g, [64, 6, 125, 116, 3, 116], '#c8102e'); poly(g, [64, 27, 105, 104, 23, 104], '#fff'); g.translate(64, 82); g.scale(0.5, 0.5); }
  g.fillStyle = g.strokeStyle = '#111';
}
const SIGNS = {
  curve: (dir, sharp) => signCell(`curve${dir}${sharp}`, g => {
    warnPlate(g); g.scale(dir, 1); g.lineWidth = 10; g.lineCap = 'round'; g.beginPath();
    if (sharp) { g.moveTo(-8, 38); g.lineTo(-8, -6); g.lineTo(18, -6); g.stroke(); poly(g, [16, -22, 38, -6, 16, 10], '#111'); }
    else { g.moveTo(-6, 40); g.lineTo(-6, 6); g.quadraticCurveTo(-6, -14, 14, -24); g.stroke(); poly(g, [6, -38, 32, -30, 18, -8], '#111'); }
  }),
  chevron: dir => signCell(`chev${dir}`, g => {
    const diamond = REGION.warn === 'diamond';
    g.fillStyle = diamond ? '#f2c00f' : '#c8102e'; g.fillRect(26, 4, 76, 120);
    g.translate(64, 64); g.scale(dir, 1);
    poly(g, [-22, -38, 12, 0, -22, 38, -6, 38, 28, 0, -6, -38], diamond ? '#111' : '#fff');
  }),
  deer: () => signCell('deer', g => {
    warnPlate(g);
    g.beginPath(); g.ellipse(0, 4, 24, 11, -0.1, 0, Math.PI * 2); g.fill();
    g.lineWidth = 6; g.lineCap = 'round'; g.beginPath();
    g.moveTo(-16, 10); g.lineTo(-22, 36); g.moveTo(-8, 12); g.lineTo(-6, 36); g.moveTo(12, 12); g.lineTo(16, 36); g.moveTo(20, 8); g.lineTo(26, 34);
    g.moveTo(20, -2); g.lineTo(28, -20); g.stroke();
    g.beginPath(); g.ellipse(32, -22, 8, 5, -0.4, 0, Math.PI * 2); g.fill();
    g.lineWidth = 3; g.beginPath(); g.moveTo(28, -26); g.lineTo(22, -40); g.moveTo(25, -33); g.lineTo(16, -38); g.moveTo(32, -26); g.lineTo(36, -40); g.stroke();
  }),
  tractor: () => signCell('tractor', g => {
    warnPlate(g);
    g.beginPath(); g.arc(-12, 14, 17, 0, Math.PI * 2); g.fill();
    g.beginPath(); g.arc(24, 22, 9, 0, Math.PI * 2); g.fill();
    g.fillRect(-10, -8, 40, 18); g.fillRect(-26, -30, 20, 28);
    g.fillStyle = REGION.warn === 'diamond' ? '#f2c00f' : '#fff'; g.fillRect(-22, -26, 12, 12);
  }),
  crossroads: () => signCell('cross', g => { warnPlate(g); g.fillRect(-7, -38, 14, 76); g.fillRect(-36, -7, 72, 14); }),
  stop: () => signCell('stop', g => {
    const oct = r => { const p = []; for (let k = 0; k < 8; k++) { const a = Math.PI / 8 + k * Math.PI / 4; p.push(64 + Math.cos(a) * r, 64 + Math.sin(a) * r); } return p; };
    poly(g, oct(63), '#fff'); poly(g, oct(57), '#c8102e');
    g.fillStyle = '#fff'; g.font = 'bold 34px Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('STOP', 64, 66);
  }),
  yield: () => signCell('yield', g => {
    poly(g, [3, 10, 125, 10, 64, 122], '#c8102e'); poly(g, [26, 23, 102, 23, 64, 98], '#fff');
    if (REGION.id === 'us') { g.fillStyle = '#c8102e'; g.font = 'bold 15px Arial, sans-serif'; g.textAlign = 'center'; g.fillText('YIELD', 64, 48); }
  }),
  crossbuck: () => signCell('xbuck', g => {
    g.translate(64, 64);
    for (const a of [0.62, -0.62]) {
      g.save(); g.rotate(a);
      if (REGION.limit === 'rect' || REGION.id === 'au') { g.fillStyle = '#111'; g.fillRect(-63, -11, 126, 22); g.fillStyle = '#fff'; g.fillRect(-61, -9, 122, 18); }
      else { g.fillStyle = '#c8102e'; g.fillRect(-63, -11, 126, 22); g.fillStyle = '#fff'; g.fillRect(-58, -6, 116, 12); }
      g.restore();
    }
    if (REGION.id === 'us') {
      g.fillStyle = '#111'; g.font = 'bold 9px Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.save(); g.rotate(0.62); g.fillText('RAILROAD', 0, 0); g.restore();
      g.save(); g.rotate(-0.62); g.fillText('CROSSING', 0, 0); g.restore();
    }
  }),
  roadwork: () => signCell('roadwork', g => {
    if (REGION.warn === 'diamond') { poly(g, [64, 2, 126, 64, 64, 126, 2, 64], '#111'); poly(g, [64, 9, 119, 64, 64, 119, 9, 64], '#f07d10'); g.translate(64, 64); g.scale(0.85, 0.85); }
    else { poly(g, [64, 6, 125, 116, 3, 116], '#c8102e'); poly(g, [64, 27, 105, 104, 23, 104], '#fff'); g.translate(64, 82); g.scale(0.5, 0.5); }
    g.fillStyle = g.strokeStyle = '#111';
    g.beginPath(); g.arc(-4, -26, 7, 0, Math.PI * 2); g.fill();
    g.lineWidth = 7; g.lineCap = 'round'; g.beginPath(); g.moveTo(-6, -16); g.lineTo(-10, 8); g.lineTo(-20, 32); g.moveTo(-10, 8); g.lineTo(4, 32); g.moveTo(-8, -10); g.lineTo(14, 2); g.stroke();
    g.lineWidth = 4; g.beginPath(); g.moveTo(14, 2); g.lineTo(26, 24); g.stroke(); g.fillRect(16, 26, 22, 8);
  }),
  gantry: () => signCell('gantry', g => {
    const us = REGION.yellow || REGION.id === 'au';
    g.fillStyle = '#fff'; g.fillRect(2, 30, 124, 68);
    g.fillStyle = us ? '#106b3a' : '#14408f'; g.fillRect(4, 32, 120, 64);
    g.fillStyle = '#fff'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.font = 'bold 13px Arial, sans-serif'; g.fillText(us ? 'EXIT 2 MILE' : 'AUSFAHRT 2 km', 64, 46);
    g.font = 'bold 17px Arial, sans-serif'; g.fillText(['CITY CENTER', 'AIRPORT', 'NORTH', 'DOWNTOWN'][Math.floor(Math.random() * 4)], 64, 68);
    poly(g, [64, 76, 74, 88, 68, 88, 68, 94, 60, 94, 60, 88, 54, 88], '#fff');
  }),
  limit: v => signCell(`lim${v}`, g => {
    g.textAlign = 'center'; g.textBaseline = 'middle';
    if (REGION.limit === 'rect') {
      g.fillStyle = '#111'; g.fillRect(20, 2, 88, 124); g.fillStyle = '#fff'; g.fillRect(24, 6, 80, 116);
      g.fillStyle = '#111'; g.font = 'bold 15px Arial, sans-serif';
      if (REGION.id === 'ca') g.fillText('MAXIMUM', 64, 30); else { g.fillText('SPEED', 64, 24); g.fillText('LIMIT', 64, 42); }
      g.font = `bold ${v >= 100 ? 40 : 52}px Arial, sans-serif`; g.fillText(String(v), 64, 88);
    } else {
      g.fillStyle = '#c8102e'; g.beginPath(); g.arc(64, 64, 62, 0, Math.PI * 2); g.fill();
      g.fillStyle = '#fff'; g.beginPath(); g.arc(64, 64, 49, 0, Math.PI * 2); g.fill();
      g.fillStyle = '#111'; g.font = `bold ${v >= 100 ? 40 : 52}px Arial, sans-serif`; g.fillText(String(v), 64, 67);
    }
  }),
};
function speedLimit(rank, urban) {
  const pick = (rural, town) => (urban ? town : rural)[Math.max(0, Math.min(7, rank - 2))];
  //                 track svc  minor tert  sec  prim trunk mway
  if (REGION.id === 'us') return pick([25, 15, 35, 45, 55, 55, 65, 70], [15, 15, 25, 30, 35, 40, 50, 60]);
  if (REGION.mph) return pick([30, 20, 40, 50, 60, 60, 70, 70], [20, 20, 20, 30, 30, 30, 40, 50]);
  if (REGION.id === 'au') return pick([40, 20, 60, 80, 100, 100, 110, 110], [20, 20, 40, 50, 60, 60, 80, 100]);
  return pick([30, 20, 50, 70, 80, 90, 100, 120], [20, 20, 30, 50, 50, 50, 70, 100]);
}

// ---- families: instanced meshes filled around the car
const propMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.82 });
const metalMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.25 });
const walkMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
function signMaterial(back) {
  const m = new THREE.MeshStandardMaterial({ map: signTex, alphaTest: 0.5, roughness: back ? 0.6 : 0.45, metalness: back ? 0.45 : 0, color: back ? 0x8c8f93 : 0xffffff });
  m.customProgramCacheKey = () => back ? 'sign-back' : 'sign-front';
  m.onBeforeCompile = sh => {
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute vec4 uvRect;')
      .replace('#include <uv_vertex>', '#include <uv_vertex>\n  vMapUv = uv * uvRect.zw + uvRect.xy;');
    if (back) sh.fragmentShader = sh.fragmentShader.replace('#include <map_fragment>', 'diffuseColor.a *= texture2D(map, vMapUv).a;');
  };
  return m;
}
const FAM = {};
function family(name, geo, mat, max, dist, opts = {}) {
  const mesh = new THREE.InstancedMesh(geo, mat, max);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(max * 3).fill(1), 3);
  mesh.count = 0; mesh.frustumCulled = false;
  mesh.castShadow = opts.shadow !== false; mesh.receiveShadow = true;
  let uv = null;
  if (opts.uv) { uv = new THREE.InstancedBufferAttribute(new Float32Array(max * 4), 4); geo.setAttribute('uvRect', uv); }
  scene.add(mesh);
  FAM[name] = { mesh, max, dist, n: 0, uv, geo, mat };
}

// utility poles (built 10 m tall; wires hang from ATTACH points, local x = across the road)
const POLE_C = REGION.wood ? C.wood : C.concrete, POLE_R = REGION.wood ? [0.13, 0.17] : [0.11, 0.16];
const poleShaft = () => pCyl(POLE_R[0], POLE_R[1], 10.6, REGION.wood ? 7 : 5, POLE_C, 0, 4.95, 0);
const insulators = (xs, y) => xs.map(x => pCyl(0.045, 0.06, 0.22, 6, C.insul, x, y, 0));
family('poleA', kit([poleShaft(), pBox(2.5, 0.11, 0.11, REGION.wood ? C.woodGrey : C.steel, 0, 9.45, 0), ...insulators([-1.05, 0, 1.05], 9.62)]), propMat, 3000, 650);
family('poleB', kit([poleShaft(), pBox(2.5, 0.11, 0.11, REGION.wood ? C.woodGrey : C.steel, 0, 9.45, 0), pBox(1.8, 0.1, 0.1, REGION.wood ? C.woodGrey : C.steel, 0, 8.7, 0), ...insulators([-1.05, 0, 1.05], 9.62), ...insulators([-0.75, 0.75], 8.86)]), propMat, 3000, 650);
family('poleC', kit([poleShaft(), pBox(1.2, 0.09, 0.09, C.steel, 0, 9.5, 0), ...insulators([0], 10.35), ...insulators([-0.5, 0.5], 9.66)]), propMat, 3000, 650);
const ATTACH = { poleA: [[-1.05, 9.72], [0, 9.72], [1.05, 9.72]], poleB: [[-1.05, 9.72], [0, 9.72], [1.05, 9.72]], poleC: [[-0.5, 9.76], [0, 10.45], [0.5, 9.76]] };
family('xfmr', kit([pCyl(0.27, 0.27, 0.95, 10, [0.55, 0.57, 0.58], 0.42, 8.2, 0), pCyl(0.29, 0.29, 0.06, 10, [0.45, 0.47, 0.48], 0.42, 8.7, 0), pBox(0.1, 0.3, 0.1, C.steel, 0.2, 8.4, 0)]), metalMat, 1500, 450);
family('wire', kit([pCyl(0.016, 0.016, 1, 4, C.wire, 0, 0, 0, 0, Math.PI / 2)]), propMat, 30000, 500, { shadow: false });

// delineator posts, by region
const DELIN = {
  us: [pBox(0.09, 1.25, 0.025, C.white, 0, 0.62, 0), pBox(0.08, 0.16, 0.03, C.amber, 0, 1.05, 0.005)],
  eu: [pBox(0.12, 1.05, 0.1, C.white, 0, 0.52, 0), pBox(0.125, 0.25, 0.105, C.black, 0, 0.82, 0), pBox(0.05, 0.12, 0.11, C.amber, 0, 0.82, 0)],
  uk: [pBox(0.1, 1.0, 0.1, C.white, 0, 0.5, 0), pBox(0.06, 0.1, 0.11, C.red, 0, 0.88, 0)],
  au: [pBox(0.11, 1.1, 0.06, C.white, 0, 0.55, 0), pBox(0.06, 0.1, 0.07, C.red, 0, 0.98, 0)],
};
family('delin', kit(DELIN[REGION.delin]), propMat, 6000, 380);

// fences
family('tpost', kit([pBox(0.045, 1.35, 0.045, C.tpost, 0, 0.6, 0)]), metalMat, 20000, 260);
family('wpost', kit([pCyl(0.065, 0.075, 1.5, 6, C.wood, 0, 0.6, 0)]), propMat, 12000, 300);
family('rpost', kit([pBox(0.12, 1.45, 0.12, [1, 1, 1], 0, 0.62, 0)]), propMat, 12000, 300);
family('fwire', kit([pBox(0.012, 0.012, 1, [0.3, 0.3, 0.31], 0, 0, 0)]), metalMat, 40000, 200, { shadow: false });
family('frail', kit([pBox(0.05, 0.12, 1, [1, 1, 1], 0, 0, 0)]), propMat, 30000, 300);
family('ftape', kit([pBox(0.008, 0.035, 1, [0.95, 0.95, 0.9], 0, 0, 0)]), propMat, 20000, 200, { shadow: false });
family('gate', kit([
  pBox(4, 0.06, 0.06, [1, 1, 1], 2, 1.15, 0), pBox(4, 0.06, 0.06, [1, 1, 1], 2, 0.25, 0),
  pBox(4, 0.05, 0.05, [1, 1, 1], 2, 0.55, 0), pBox(4, 0.05, 0.05, [1, 1, 1], 2, 0.85, 0),
  ...[0, 1, 2, 3, 4].map(x => pBox(0.06, 0.96, 0.06, [1, 1, 1], x, 0.7, 0)), pBox(0.06, 1.25, 0.06, [1, 1, 1], 2, 0.7, 0, 0, 0, 0.29),
]), metalMat, 1500, 300);

// W-beam guardrail
family('grpost', kit([pBox(0.15, 1.05, 0.11, C.steel, 0, 0.3, 0)]), metalMat, 15000, 380);
family('grrail', kit([pBox(0.07, 0.32, 1, C.steel, 0, 0.62, 0), pBox(0.09, 0.08, 1, [0.72, 0.74, 0.76], 0, 0.62, 0)]), metalMat, 15000, 380);

// signs
family('spost', kit([pBox(0.06, 2.6, 0.06, C.steel, 0, 1.3, 0)]), metalMat, 3000, 450);
family('sfront', new THREE.PlaneGeometry(1, 1), signMaterial(false), 3000, 450, { uv: true, shadow: false });
family('sback', new THREE.PlaneGeometry(1, 1), signMaterial(true), 3000, 450, { uv: true, shadow: false });

// mailboxes (body + flag tinted per instance)
family('mpost', kit([pBox(0.1, 1.15, 0.1, [1, 1, 1], 0, 0.55, 0), pBox(0.1, 0.06, 0.5, [1, 1, 1], 0, 1.0, 0.05)]), propMat, 4000, 220);
family('mbox', kit([pBox(0.2, 0.15, 0.48, [1, 1, 1], 0, 1.12, 0.05), pCyl(0.1, 0.1, 0.48, 10, [1, 1, 1], 0, 1.2, 0.05, 0, Math.PI / 2)]), metalMat, 4000, 220);
family('mflag', kit([pBox(0.02, 0.2, 0.07, C.red, 0.11, 1.28, -0.08)]), propMat, 4000, 180, { shadow: false });

// verge plants
family('bush', kit([part(new THREE.IcosahedronGeometry(0.8, 0).scale(1, 0.7, 1), [1, 1, 1], 0, 0.5, 0), part(new THREE.IcosahedronGeometry(0.55, 0).scale(1, 0.75, 1), [0.9, 0.9, 0.9], 0.45, 0.62, 0.2)]), canopyMat, 12000, 300);
family('flower', kit([
  ...Array.from({ length: 9 }, (_, k) => pBox(0.07, 0.07, 0.07, [1, 1, 1], Math.cos(k * 2.4) * 0.3 * Math.sqrt(k / 9), 0.28 + (k % 3) * 0.08, Math.sin(k * 2.4) * 0.3 * Math.sqrt(k / 9))),
  ...Array.from({ length: 5 }, (_, k) => pBox(0.015, 0.32, 0.015, [0.25, 0.45, 0.2], Math.cos(k * 2.1) * 0.22, 0.14, Math.sin(k * 2.1) * 0.22)),
]), propMat, 12000, 140, { shadow: false });

// railways: rails, sleepers, level-crossing gates
family('railbar', kit([pBox(0.07, 0.15, 1, [0.4, 0.34, 0.3], 0, 0, 0), pBox(0.072, 0.03, 1, [0.72, 0.72, 0.74], 0, 0.075, 0)]), metalMat, 12000, 650, { shadow: false });
family('sleeper', kit([pBox(2.6, 0.15, 0.24, [1, 1, 1], 0, 0, 0)]), propMat, 30000, 230);
family('xgate', kit([
  pBox(0.26, 1.5, 0.26, [0.92, 0.92, 0.9], 0, 0.75, 0), pBox(1.0, 0.36, 0.22, C.black, 0, 1.75, 0), pBox(0.22, 0.22, 0.05, C.red, -0.3, 1.75, 0.12), pBox(0.22, 0.22, 0.05, C.red, 0.3, 1.75, 0.12),
  pBox(0.24, 0.24, 0.24, C.black, 0, 2.05, 0),
  ...Array.from({ length: 10 }, (_, k) => pBox(0.1, 0.6, 0.08, k % 2 ? C.red : C.white, 0.35, 1.3 + k * 0.6 + 0.3, 0)),
]), propMat, 400, 400);

// rocks: six noise-shaped variants, flat-shaded
const rockMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.95 });
function rockGeo(seed) {
  const g = new THREE.IcosahedronGeometry(1, 1), pa = g.attributes.position, cols = [];
  const rr = k => { const x = Math.sin(seed * 91.7 + k * 12.9898) * 43758.5453; return x - Math.floor(x); };
  const sq = 0.55 + rr(1) * 0.35, st = 0.75 + rr(2) * 0.5;
  for (let i = 0; i < pa.count; i++) {
    const x = pa.getX(i), y = pa.getY(i), z = pa.getZ(i);
    const n = 0.72 + 0.42 * rr(Math.round(x * 7 + 31) * 131 + Math.round(y * 7 + 31) * 17 + Math.round(z * 7 + 31));
    pa.setXYZ(i, x * n * st, Math.max(y * n * sq, -0.25), z * n);
    const v = 0.78 + rr(i + 9) * 0.3; cols.push(v, v, v);
  }
  g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  g.computeVertexNormals();
  return g;
}
for (let k = 0; k < 6; k++) family('rock' + k, rockGeo(k + 1), rockMat, 6000, 450);

// cacti (built life-size, scaled a little per plant): saguaro (Sonoran Desert only), prickly pear, barrel cactus,
// and columnar cacti (organ pipe / cardon / the Andes' and Brazil's tall cacti)
const CACT = [0.3, 0.44, 0.27], CACT2 = [0.36, 0.5, 0.3], CACT_D = [0.25, 0.36, 0.22];
const pSph = (r, sx, sy, sz, c, x, y, z, ry = 0, rx = 0, rz = 0, w = 10, h = 7) => part(new THREE.SphereGeometry(r, w, h).scale(sx, sy, sz), c, x, y, z, ry, rx, rz);
const dome = (r, c, x, y, z) => part(new THREE.SphereGeometry(r, 12, 5, 0, Math.PI * 2, 0, Math.PI / 2), c, x, y, z);
function saguaroGeo(arms) {
  const P = [pCyl(0.25, 0.28, 6.2, 12, CACT, 0, 3.1, 0), dome(0.25, CACT, 0, 6.2, 0)];
  for (const [side, y0, up, ang] of arms) {
    const ox = Math.cos(ang) * side, oz = Math.sin(ang) * side;
    P.push(pCyl(0.17, 0.17, 0.62, 10, CACT, ox * 0.42, y0, oz * 0.42, -ang, 0, Math.PI / 2));   // elbow out
    P.push(pCyl(0.17, 0.18, up, 10, CACT, ox * 0.74, y0 + up / 2 - 0.05, oz * 0.74));              // and up
    P.push(dome(0.17, CACT, ox * 0.74, y0 + up - 0.05, oz * 0.74));
  }
  return kit(P);
}
family('saguaro0', saguaroGeo([[1, 2.4, 1.9, 0], [1, 3.1, 1.5, Math.PI * 0.95]]), propMat, 5000, 900);
family('saguaro1', saguaroGeo([[1, 2.8, 1.6, 0.4]]), propMat, 5000, 900);
family('saguaro2', saguaroGeo([[1, 2.2, 2.2, 1.2], [1, 2.9, 1.5, 3.4], [1, 3.4, 1.2, 5.2]]), propMat, 5000, 900);
function pearGeo(seed) {
  const R = rng(seed), P = [];
  const pad = (x, y, z, ry, rz, s) => P.push(pSph(0.27 * s, 1, 1.25, 0.26, R() < 0.5 ? CACT2 : CACT, x, y, z, ry, 0, rz));
  for (let k = 0; k < 5; k++) { const a = k * 1.25 + R(), d = 0.18 + R() * 0.15; pad(Math.cos(a) * d, 0.3, Math.sin(a) * d, a + Math.PI / 2, (R() - 0.5) * 0.7, 1); }
  for (let k = 0; k < 4; k++) { const a = k * 1.6 + R(), d = 0.12 + R() * 0.1; pad(Math.cos(a) * d, 0.75 + R() * 0.1, Math.sin(a) * d, a + Math.PI / 2, (R() - 0.5) * 0.9, 0.85); }
  for (let k = 0; k < 4; k++) { const a = R() * 6.28; P.push(pSph(0.045, 1, 1.2, 1, [0.72, 0.16, 0.3], Math.cos(a) * 0.22, 1.05 + R() * 0.15, Math.sin(a) * 0.22)); }   // red fruit
  return kit(P);
}
family('pear0', pearGeo(11), propMat, 6000, 500);
family('pear1', pearGeo(29), propMat, 6000, 500);
family('barrel', kit([pCyl(0.3, 0.34, 0.75, 16, CACT_D, 0, 0.375, 0), dome(0.3, CACT_D, 0, 0.75, 0),
  ...Array.from({ length: 6 }, (_, k) => pSph(0.05, 1, 0.7, 1, [0.95, 0.78, 0.2], Math.cos(k * 1.05) * 0.16, 0.98, Math.sin(k * 1.05) * 0.16))]), propMat, 6000, 450);
function columnGeo(seed) {
  const R = rng(seed), P = [];
  for (let k = 0; k < 6; k++) {
    const a = k * 1.05 + R(), d = k ? 0.22 + R() * 0.12 : 0, h = 2.4 + R() * 2.4, lean = k ? 0.12 + R() * 0.1 : 0;
    const x = Math.cos(a) * d, z = Math.sin(a) * d;
    P.push(pCyl(0.13, 0.15, h, 10, k % 2 ? CACT : CACT2, x + Math.cos(a) * lean * h / 2, h / 2, z + Math.sin(a) * lean * h / 2, 0, Math.sin(a) * lean, -Math.cos(a) * lean));
    P.push(dome(0.13, CACT, x + Math.cos(a) * lean * h, h, z + Math.sin(a) * lean * h));
  }
  return kit(P);
}
family('column0', columnGeo(5), propMat, 5000, 800);
family('column1', columnGeo(17), propMat, 5000, 800);

// farms: bins, silos, machinery, bales, livestock, irrigation (body parts in white are tinted per instance)
const W1 = [1, 1, 1], TYRE = [0.07, 0.07, 0.07], GLASS = [0.12, 0.16, 0.2], STEEL2 = [0.7, 0.72, 0.74];
const wheel = (r, w, x, y, z) => pCyl(r, r, w, 14, TYRE, x, y, z, 0, 0, Math.PI / 2);
function binGeo() {
  const parts = [];
  for (let k = 0; k < 8; k++) parts.push(pCyl(1, 1, 0.125, 28, k % 2 ? [0.78, 0.8, 0.82] : [0.66, 0.68, 0.7], 0, k * 0.125 + 0.0625, 0)); // corrugated rings
  parts.push(pCyl(0.12, 1.03, 0.32, 28, [0.74, 0.76, 0.78], 0, 1.16, 0), pCyl(0.13, 0.13, 0.06, 10, [0.5, 0.52, 0.54], 0, 1.34, 0));
  return kit(parts);
}
family('bin', binGeo(), propMat, 1500, 900);
family('silo', kit([pCyl(1, 1, 1, 18, W1, 0, 0.5, 0), part(new THREE.SphereGeometry(1, 18, 8, 0, Math.PI * 2, 0, Math.PI / 2).scale(1, 0.12, 1), W1, 0, 1, 0)]), propMat, 400, 1100);
family('tractor', kit([
  pBox(0.95, 0.9, 2.3, W1, 0, 1.05, 0.45), pBox(1.35, 0.25, 1.4, W1, 0, 1.4, -0.75), pBox(1.25, 1.25, 1.25, GLASS, 0, 2.15, -0.75), pBox(1.4, 0.1, 1.45, W1, 0, 2.82, -0.75),
  wheel(0.85, 0.55, 0.95, 0.85, -0.8), wheel(0.85, 0.55, -0.95, 0.85, -0.8), wheel(0.5, 0.35, 0.82, 0.5, 1.3), wheel(0.5, 0.35, -0.82, 0.5, 1.3),
  pCyl(0.06, 0.06, 1.2, 6, [0.15, 0.15, 0.15], 0.3, 2.1, 1.1),
]), propMat, 600, 450);
family('combine', kit([
  pBox(3, 2.6, 6.5, W1, 0, 2.3, -0.5), pBox(2.2, 1.8, 2.2, GLASS, 0, 4.3, 2.0), pBox(2.4, 0.15, 2.4, W1, 0, 5.25, 2.0),
  pBox(8.5, 0.9, 1.6, W1, 0, 0.9, 4.2), pCyl(0.25, 0.25, 4.5, 8, W1, 2.6, 4.0, -0.5, 0, 0, Math.PI / 2 - 0.25),
  wheel(1.0, 0.8, 1.7, 1.0, 1.4), wheel(1.0, 0.8, -1.7, 1.0, 1.4), wheel(0.6, 0.45, 1.4, 0.6, -3.0), wheel(0.6, 0.45, -1.4, 0.6, -3.0),
]), propMat, 300, 550);
family('pickup', kit([
  pBox(1.9, 0.75, 5.4, W1, 0, 0.8, 0), pBox(1.8, 0.75, 2.0, W1, 0, 1.55, 0.6), pBox(1.82, 0.6, 1.9, GLASS, 0, 1.55, 0.62),
  wheel(0.4, 0.3, 0.88, 0.4, 1.7), wheel(0.4, 0.3, -0.88, 0.4, 1.7), wheel(0.4, 0.3, 0.88, 0.4, -1.6), wheel(0.4, 0.3, -0.88, 0.4, -1.6),
]), propMat, 1500, 400);
family('graincart', kit([
  part(new THREE.CylinderGeometry(1.4, 0.4, 1.8, 4, 1).rotateY(Math.PI / 4).scale(1, 1, 1.8), W1, 0, 2.2, 0), pBox(0.25, 0.25, 3.5, [0.2, 0.2, 0.2], 0, 1.0, 2.6),
  wheel(0.8, 0.6, 1.25, 0.8, 0), wheel(0.8, 0.6, -1.25, 0.8, 0),
]), propMat, 400, 400);
family('fueltank', kit([pCyl(0.65, 0.65, 2.6, 14, W1, 0, 1.9, 0, 0, 0, Math.PI / 2), ...[[-0.9, 0.4], [0.9, 0.4], [-0.9, -0.4], [0.9, -0.4]].map(([x, z]) => pBox(0.08, 1.3, 0.08, [0.3, 0.3, 0.3], x, 0.65, z))]), propMat, 400, 350);
family('stocktank', kit([pCyl(1.2, 1.2, 0.6, 18, STEEL2, 0, 0.3, 0), pCyl(1.12, 1.12, 0.02, 18, [0.25, 0.32, 0.35], 0, 0.55, 0)]), metalMat, 400, 250);
family('bale', kit([pCyl(0.75, 0.75, 1.2, 14, W1, 0, 0.75, 0, 0, 0, Math.PI / 2)]), propMat, 8000, 500);
family('sqbale', kit([pBox(1.2, 0.9, 2.4, W1, 0, 0.45, 0)]), propMat, 4000, 400);
const cowParts = (body, patch) => [
  pBox(0.62, 0.72, 1.55, body, 0, 1.0, 0), pBox(0.32, 0.38, 0.5, patch || body, 0, 1.18, 0.95), pBox(0.36, 0.06, 0.08, [0.85, 0.85, 0.8], 0, 1.4, 0.8),
  ...[[0.2, 0.55], [-0.2, 0.55], [0.2, -0.55], [-0.2, -0.55]].map(([x, z]) => pBox(0.13, 0.66, 0.13, body, x, 0.33, z)),
  pBox(0.05, 0.6, 0.05, body, 0, 0.85, -0.8, 0, 0.3),
];
family('cow', kit(cowParts(W1)), propMat, 3000, 400);
family('holstein', kit([...cowParts([0.95, 0.95, 0.93], [0.08, 0.08, 0.08]), pBox(0.64, 0.4, 0.5, [0.08, 0.08, 0.08], 0, 1.1, 0.3), pBox(0.64, 0.35, 0.45, [0.08, 0.08, 0.08], 0, 0.95, -0.45)]), propMat, 1500, 400);
family('horse', kit([
  pBox(0.5, 0.65, 1.6, W1, 0, 1.25, 0), pBox(0.3, 0.75, 0.32, W1, 0, 1.75, 0.85, 0, -0.55), pBox(0.26, 0.28, 0.6, W1, 0, 2.05, 1.15),
  ...[[0.17, 0.6], [-0.17, 0.6], [0.17, -0.6], [-0.17, -0.6]].map(([x, z]) => pBox(0.11, 0.95, 0.11, W1, x, 0.48, z)),
  pBox(0.08, 0.7, 0.12, [0.1, 0.08, 0.06], 0, 1.2, -0.85, 0, 0.35),
]), propMat, 600, 400);
// centre-pivot irrigation: one 50 m span (pipe, truss rods, A-frame tower with two wheels at the outer end)
family('pivotspan', kit([
  pCyl(0.11, 0.11, 50, 8, STEEL2, 0, 3.9, 25, 0, Math.PI / 2),
  ...Array.from({ length: 9 }, (_, k) => pBox(0.04, 1.2 * Math.sin((k + 1) / 10 * Math.PI), 0.04, STEEL2, 0, 3.9 - 0.6 * Math.sin((k + 1) / 10 * Math.PI), (k + 1) * 5)),
  pBox(0.06, 0.06, 50, STEEL2, 0, 3.2, 25),
  pBox(0.12, 3.9, 0.12, STEEL2, 0.9, 1.95, 50, 0, 0, 0.22), pBox(0.12, 3.9, 0.12, STEEL2, -0.9, 1.95, 50, 0, 0, -0.22),
  pBox(2.0, 0.15, 0.25, STEEL2, 0, 0.6, 50), wheel(0.6, 0.35, 1.25, 0.6, 50), wheel(0.6, 0.35, -1.25, 0.6, 50),
]), metalMat, 600, 1500);
family('pivotcentre', kit([pBox(1.6, 0.3, 1.6, CONCRETE_C(), 0, 0.15, 0), ...[[0.6, 0.6], [-0.6, 0.6], [0.6, -0.6], [-0.6, -0.6]].map(([x, z]) => pBox(0.12, 4.2, 0.12, STEEL2, x * 0.7, 2.1, z * 0.7, 0, x * 0.25, -z * 0.25)), pBox(0.6, 0.6, 0.6, [0.6, 0.6, 0.6], 0, 4.1, 0)]), metalMat, 100, 1500);
function CONCRETE_C() { return [0.66, 0.65, 0.62]; }

// towns: street lights, traffic signals, hydrants, bins, parked cars, bus stops, crosswalk paint
const LAMP_GREY = [0.55, 0.57, 0.6];
family('lamp', kit(REGION.id === 'us' || REGION.id === 'ca' || REGION.id === 'latam'
  ? [pCyl(0.07, 0.12, 8.6, 8, LAMP_GREY, 0, 4.3, 0), pCyl(0.045, 0.045, 2.4, 6, LAMP_GREY, 0, 8.45, 1.15, 0, Math.PI / 2 - 0.12), pBox(0.36, 0.14, 0.75, LAMP_GREY, 0, 8.55, 2.35), pBox(0.3, 0.03, 0.55, [1, 0.95, 0.8], 0, 8.47, 2.35)]
  : [pCyl(0.06, 0.1, 8, 8, [0.35, 0.37, 0.4], 0, 4, 0), pBox(0.08, 0.08, 1.0, [0.35, 0.37, 0.4], 0, 7.95, 0.5), pBox(0.3, 0.1, 0.7, [0.3, 0.32, 0.35], 0, 7.95, 1.1), pBox(0.26, 0.02, 0.6, [1, 0.97, 0.88], 0, 7.89, 1.1)]), metalMat, 6000, 600);
family('sigpole', kit([pCyl(0.15, 0.2, 6.6, 10, [0.62, 0.63, 0.6], 0, 3.3, 0), pBox(0.4, 0.4, 0.4, [0.55, 0.56, 0.54], 0, 0.2, 0)]), metalMat, 1500, 500);
family('sigarm', kit([pBox(0.14, 0.18, 1, [0.62, 0.63, 0.6], 0, 0, 0.5)]), metalMat, 1500, 500);
family('sighead', kit([
  pBox(0.36, 1.05, 0.26, REGION.yellow ? [0.85, 0.7, 0.1] : [0.12, 0.12, 0.12], 0, 0, 0), pBox(0.5, 1.2, 0.03, [0.08, 0.08, 0.08], 0, 0, -0.14),
  pCyl(0.11, 0.11, 0.04, 12, [0.9, 0.12, 0.08], 0, 0.33, 0.14, 0, Math.PI / 2), pCyl(0.11, 0.11, 0.04, 12, [0.35, 0.25, 0.05], 0, 0, 0.14, 0, Math.PI / 2), pCyl(0.11, 0.11, 0.04, 12, [0.05, 0.25, 0.1], 0, -0.33, 0.14, 0, Math.PI / 2),
]), propMat, 3000, 450);
family('hydrant', kit([pCyl(0.13, 0.15, 0.6, 10, W1, 0, 0.3, 0), part(new THREE.SphereGeometry(0.13, 10, 6, 0, Math.PI * 2, 0, Math.PI / 2), W1, 0, 0.6, 0), pCyl(0.05, 0.05, 0.36, 8, W1, 0, 0.38, 0, 0, 0, Math.PI / 2), pCyl(0.18, 0.18, 0.06, 10, W1, 0, 0.03, 0)]), propMat, 2000, 200);
family('wheelie', kit([pBox(0.58, 0.95, 0.7, W1, 0, 0.5, 0), pBox(0.62, 0.06, 0.76, [0.9, 0.9, 0.9], 0, 1.0, 0), pCyl(0.1, 0.1, 0.06, 8, TYRE, 0.25, 0.1, -0.33, 0, 0, Math.PI / 2), pCyl(0.1, 0.1, 0.06, 8, TYRE, -0.25, 0.1, -0.33, 0, 0, Math.PI / 2)]), propMat, 3000, 180);
const carKit = (len, wid, h, cabLen, cabOff, cabH, wheelR) => kit([
  pBox(wid, h * 0.55, len, W1, 0, wheelR + h * 0.18, 0), pBox(wid * 0.92, cabH, cabLen, W1, 0, wheelR + h * 0.45 + cabH / 2, cabOff), pBox(wid * 0.94, cabH * 0.72, cabLen * 0.96, GLASS, 0, wheelR + h * 0.45 + cabH * 0.42, cabOff),
  pBox(wid * 1.01, 0.12, len * 1.01, [0.08, 0.08, 0.08], 0, wheelR * 0.75, 0),
  wheel(wheelR, 0.24, wid / 2 - 0.08, wheelR, len * 0.32), wheel(wheelR, 0.24, -wid / 2 + 0.08, wheelR, len * 0.32), wheel(wheelR, 0.24, wid / 2 - 0.08, wheelR, -len * 0.32), wheel(wheelR, 0.24, -wid / 2 + 0.08, wheelR, -len * 0.32),
  pBox(0.3, 0.1, 0.03, [1, 0.95, 0.8], wid * 0.32, wheelR + h * 0.3, len / 2), pBox(0.3, 0.1, 0.03, [1, 0.95, 0.8], -wid * 0.32, wheelR + h * 0.3, len / 2),
  pBox(0.3, 0.1, 0.03, [0.7, 0.05, 0.05], wid * 0.33, wheelR + h * 0.33, -len / 2), pBox(0.3, 0.1, 0.03, [0.7, 0.05, 0.05], -wid * 0.33, wheelR + h * 0.33, -len / 2),
]);
family('sedan', carKit(4.6, 1.8, 0.9, 2.3, -0.2, 0.62, 0.32), propMat, 4000, 350);
family('hatch', carKit(4.0, 1.75, 0.9, 2.2, -0.45, 0.66, 0.31), propMat, 4000, 350);
family('suv', carKit(4.8, 1.92, 1.05, 2.9, -0.5, 0.78, 0.38), propMat, 4000, 350);
family('van', carKit(5.2, 1.95, 1.2, 3.6, -0.6, 0.95, 0.36), propMat, 2000, 350);
family('busstop', kit([pCyl(0.04, 0.04, 2.6, 6, C.steel, 0, 1.3, 0), pBox(0.45, 0.45, 0.04, REGION.id === 'us' ? [0.1, 0.3, 0.7] : [0.95, 0.85, 0.2], 0, 2.4, 0.03), pBox(0.36, 0.08, 0.05, [0.95, 0.95, 0.95], 0, 2.4, 0.06)]), propMat, 800, 300);
family('shelter', kit([
  ...[[-1.6, -0.6], [1.6, -0.6], [-1.6, 0.6], [1.6, 0.6]].map(([x, z]) => pBox(0.06, 2.3, 0.06, [0.3, 0.32, 0.34], x, 1.15, z)),
  pBox(3.4, 0.08, 1.5, [0.3, 0.32, 0.34], 0, 2.32, 0), pBox(3.3, 1.8, 0.02, [0.55, 0.65, 0.7], 0, 1.25, -0.6), pBox(0.02, 1.8, 1.2, [0.55, 0.65, 0.7], -1.6, 1.25, 0),
  pBox(2.2, 0.06, 0.4, [0.45, 0.32, 0.2], 0, 0.5, -0.35), pBox(0.06, 0.5, 0.3, [0.3, 0.32, 0.34], -1.0, 0.25, -0.35), pBox(0.06, 0.5, 0.3, [0.3, 0.32, 0.34], 1.0, 0.25, -0.35),
]), propMat, 500, 300);
family('stripe', kit([pBox(0.5, 0.012, 1, [0.92, 0.92, 0.9], 0, 0, 0)]), propMat, 12000, 250, { shadow: false });

// high-voltage lattice pylon (built 30 m tall; scaled uniformly by height)
function pylonGeo() {
  const P = [], st = [0.58, 0.6, 0.62];
  const beam = (a, b, t) => {
    const d = new THREE.Vector3(b[0] - a[0], b[1] - a[1], b[2] - a[2]), L = d.length();
    const g = new THREE.BoxGeometry(t, t, L);
    g.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), d.normalize())));
    g.translate((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
    P.push({ geo: g, color: st });
  };
  const w = y => 3.2 * (1 - y / 30) + 0.5 * (y / 30); // half-width of the legs at height y
  const lv = [0, 6, 12, 18, 24, 30];
  for (const [sx, sz] of [[1, 1], [1, -1], [-1, -1], [-1, 1]]) for (let k = 0; k < lv.length - 1; k++) beam([sx * w(lv[k]), lv[k], sz * w(lv[k])], [sx * w(lv[k + 1]), lv[k + 1], sz * w(lv[k + 1])], 0.25);
  for (let k = 0; k < lv.length - 1; k++) {
    const a = lv[k], b = lv[k + 1];
    for (const [s1, s2] of [[[1, 1], [1, -1]], [[1, -1], [-1, -1]], [[-1, -1], [-1, 1]], [[-1, 1], [1, 1]]]) {
      beam([s1[0] * w(a), a, s1[1] * w(a)], [s2[0] * w(b), b, s2[1] * w(b)], 0.12);
      beam([s2[0] * w(a), a, s2[1] * w(a)], [s1[0] * w(b), b, s1[1] * w(b)], 0.12);
    }
  }
  for (const y of [19, 24.5]) { beam([-9, y, 0], [9, y, 0], 0.35); beam([-9, y, 0], [-w(y), y - 2.5, 0], 0.15); beam([9, y, 0], [w(y), y - 2.5, 0], 0.15); }
  beam([0, 30, 0], [0, 33, 0], 0.2);
  for (const [x, y] of [[-8, 19], [8, 19], [-8, 24.5], [8, 24.5]]) P.push(pCyl(0.12, 0.18, 1.8, 6, [0.3, 0.42, 0.48], x, y - 1.0, 0));
  return kit(P);
}
family('pylon', pylonGeo(), metalMat, 400, 2600);
const PYLON_ATTACH = [[-8, 17.0], [8, 17.0], [-8, 22.5], [8, 22.5], [0, 33]];
family('hvwire', kit([pCyl(0.035, 0.035, 1, 4, C.wire, 0, 0, 0, 0, Math.PI / 2)]), propMat, 30000, 2600, { shadow: false });
// water towers, tanks, masts
family('wtower', kit([
  pCyl(1.2, 1.4, 30, 12, W1, 0, 15, 0), part(new THREE.SphereGeometry(6.5, 18, 10).scale(1, 0.7, 1), W1, 0, 33, 0), pCyl(1.0, 1.2, 1.5, 10, W1, 0, 38, 0),
  pBox(0.3, 0.3, 0.3, C.red, 0, 39, 0),
]), propMat, 60, 3000);
family('tank', kit([pCyl(1, 1, 1, 24, W1, 0, 0.5, 0), part(new THREE.SphereGeometry(1, 24, 6, 0, Math.PI * 2, 0, Math.PI / 2).scale(1, 0.08, 1), W1, 0, 1, 0)]), propMat, 300, 1500);
family('mast', kit([
  ...Array.from({ length: 10 }, (_, k) => pBox(0.6 - k * 0.03, 6, 0.6 - k * 0.03, k % 2 ? C.white : [0.85, 0.2, 0.15], 0, 3 + k * 6, 0)),
  pBox(0.25, 0.25, 0.25, C.red, 0, 60.3, 0),
]), propMat, 60, 3000);
// motorway sign gantry: two legs + a truss beam (scaled to the road width) + a big guide sign
family('gantryleg', kit([pBox(0.5, 7.2, 0.5, [0.62, 0.64, 0.66], 0, 3.6, 0), pBox(1.0, 0.3, 1.0, [0.55, 0.55, 0.55], 0, 0.15, 0)]), metalMat, 300, 1200);
family('gantrybeam', kit([pBox(0.12, 0.12, 1, [0.62, 0.64, 0.66], 0, 0.6, 0.5), pBox(0.12, 0.12, 1, [0.62, 0.64, 0.66], 0, -0.6, 0.5), pBox(0.12, 0.12, 1, [0.62, 0.64, 0.66], 0.6, 0, 0.5), pBox(0.08, 1.2, 1, [0.62, 0.64, 0.66], 0, 0, 0.5)]), metalMat, 300, 1200);
// roadworks and other set pieces
family('cone', kit([pCyl(0.03, 0.17, 0.62, 10, [0.95, 0.42, 0.06], 0, 0.36, 0), pCyl(0.11, 0.135, 0.12, 10, [0.95, 0.95, 0.95], 0, 0.42, 0), pBox(0.4, 0.04, 0.4, [0.1, 0.1, 0.1], 0, 0.02, 0)]), propMat, 3000, 300);
family('barrel', kit([...Array.from({ length: 5 }, (_, k) => pCyl(0.3, 0.3, 0.2, 12, k % 2 ? C.white : [0.95, 0.45, 0.08], 0, 0.1 + k * 0.2, 0)), pCyl(0.32, 0.32, 0.06, 12, [0.1, 0.1, 0.1], 0, 0.02, 0)]), propMat, 1500, 300);
family('person', kit([
  pBox(0.38, 0.6, 0.24, W1, 0, 1.25, 0), part(new THREE.SphereGeometry(0.12, 8, 6), [0.75, 0.58, 0.45], 0, 1.7, 0),
  pBox(0.15, 0.85, 0.16, [0.18, 0.2, 0.3], 0.1, 0.45, 0), pBox(0.15, 0.85, 0.16, [0.18, 0.2, 0.3], -0.1, 0.45, 0),
  pBox(0.1, 0.6, 0.1, W1, 0.25, 1.2, 0), pBox(0.1, 0.6, 0.1, W1, -0.25, 1.2, 0),
]), propMat, 600, 250);
family('trailer', kit([pBox(2.4, 0.15, 6, [0.35, 0.3, 0.25], 0, 0.9, 0), wheel(0.45, 0.3, 1.1, 0.45, -1.5), wheel(0.45, 0.3, -1.1, 0.45, -1.5), wheel(0.45, 0.3, 1.1, 0.45, 1.5), wheel(0.45, 0.3, -1.1, 0.45, 1.5),
  ...Array.from({ length: 8 }, (_, k) => pBox(1.1, 0.9, 2.2, [0.82, 0.7, 0.42], (k % 2 ? 0.58 : -0.58), 1.45 + Math.floor(k / 4) * 0.9, (Math.floor(k / 2) % 2 ? 1.2 : -1.2)))]), propMat, 200, 400);
family('warntri', kit([part(new THREE.CylinderGeometry(0.28, 0.28, 0.03, 3).rotateX(Math.PI / 2).rotateZ(Math.PI / 2), [0.85, 0.1, 0.08], 0, 0.3, 0), pBox(0.3, 0.02, 0.15, [0.1, 0.1, 0.1], 0, 0.01, 0)]), propMat, 300, 150, { shadow: false });

// ---- per-tile generation
const PSTRIDE = 15; // x y z  qx qy qz qw  sx sy sz  r g b  aux  knockId
let nextKid = 1;
const knocked = new Set();
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion(), _va = new THREE.Vector3(), _vb = new THREE.Vector3(), _zAxis = new THREE.Vector3(0, 0, 1);

let lastGenTimes = {};
// synchronous version (debugging); the game runs propGen time-sliced from updateProps
function generateProps(t) { const it = propGen(t); while (!it.next().done); }
function* propGen(t) {
  t.propsGen = true; // claimed; anything that invalidates the tile meanwhile sets it back to false
  const out = {}, col = { c: [], s: [] };
  let _gt = performance.now(); const times = {}; const mark = k => { const n = performance.now(); times[k] = Math.round(n - _gt); _gt = n; };
  const S = TILE_SIZE, inside = (x, z) => x >= 0 && x < S && z >= 0 && z < S;
  const rnd = (x, z, k = 0) => hash(Math.floor((t.ox + x) * 3.1) + k * 7919, Math.floor((t.oz + z) * 2.9) - k * 104729);
  const add = (fam, wx, y, wz, q, sx, sy, sz, rgb = [1, 1, 1], aux = 0, kid = -1) => {
    (out[fam] || (out[fam] = [])).push(wx, y, wz, q.x, q.y, q.z, q.w, sx, sy, sz, rgb[0], rgb[1], rgb[2], aux, kid);
  };
  const yawQ = (yaw, tiltX = 0, tiltZ = 0) => _qa.setFromEuler(_eu.set(tiltX, yaw, tiltZ, 'YXZ'));
  const lin = c => [Math.pow(c[0], 2.2), Math.pow(c[1], 2.2), Math.pow(c[2], 2.2)];

  // junctions (tile-local) for "keep clear" checks
  const J = t.junc || new Float32Array(0);
  // junction lookup grid (32 m cells) so "how close is the nearest junction" is cheap
  const JG = 32, JW = Math.ceil(S / JG) + 4, jcells = new Map();
  for (let k = 0; k < J.length; k += 3) {
    const ci = Math.floor(J[k] / JG) + 2, cj = Math.floor(J[k + 1] / JG) + 2;
    for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) { const key = (cj + dj) * JW + ci + di; (jcells.get(key) || jcells.set(key, []).get(key)).push(k); }
  }
  const nearJ = (x, z) => {
    const l = jcells.get((Math.floor(z / JG) + 2) * JW + Math.floor(x / JG) + 2);
    if (!l) return 40;
    let m = 40;
    for (const k of l) { const d = Math.hypot(J[k] - x, J[k + 1] - z) - J[k + 2]; if (d < m) m = d; }
    return m;
  };
  // building density (count of footprints within ~70 m)
  // urban test from a precomputed density grid (buildings within ~70 m), 16 m cells
  // A spot is urban with 14+ footprints within ~70 m, or when the land cover says built-up around it and there are
  // at least a few buildings (OSM building coverage is patchy outside the US, so footprints alone miss suburbs)
  const UG = 16, UW = Math.ceil(S / UG) + 10, urb = new Uint8Array(UW * UW);
  {
    const cnt = new Uint16Array(UW * UW), built = new Uint16Array(UW * UW);
    if (t.foot) for (let b = 0; b < t.footOff.length - 1; b++) {
      const i = t.footOff[b], ci = Math.floor(t.foot[2 * i] / UG) + 5, cj = Math.floor(t.foot[2 * i + 1] / UG) + 5;
      if (ci >= 0 && cj >= 0 && ci < UW && cj < UW) cnt[cj * UW + ci]++;
    }
    for (let j = 0; j < UW; j++) for (let i = 0; i < UW; i++) {
      const l = landAt(t.ox + (i - 5 + 0.5) * UG, t.oz + (j - 5 + 0.5) * UG);
      built[j * UW + i] = l && l.mat === L_URBAN ? 1 : 0;
    }
    const sums = (a) => {
      const sat = new Uint32Array((UW + 1) * (UW + 1));
      for (let j = 0; j < UW; j++) for (let i = 0; i < UW; i++) sat[(j + 1) * (UW + 1) + i + 1] = a[j * UW + i] + sat[j * (UW + 1) + i + 1] + sat[(j + 1) * (UW + 1) + i] - sat[j * (UW + 1) + i];
      return (i0, j0, i1, j1) => sat[j1 * (UW + 1) + i1] - sat[j0 * (UW + 1) + i1] - sat[j1 * (UW + 1) + i0] + sat[j0 * (UW + 1) + i0];
    };
    const cS = sums(cnt), bS = sums(built);
    for (let j = 0; j < UW; j++) for (let i = 0; i < UW; i++) {
      const i0 = Math.max(0, i - 4), j0 = Math.max(0, j - 4), i1 = Math.min(UW, i + 5), j1 = Math.min(UW, j + 5);
      const n = cS(i0, j0, i1, j1), bf = bS(i0, j0, i1, j1) / ((i1 - i0) * (j1 - j0));
      urb[j * UW + i] = n >= 14 || (bf >= 0.5 && n >= 3) ? 1 : 0;
    }
  }
  const urbanAt = (x, z) => {
    const i = Math.floor(x / UG) + 5, j = Math.floor(z / UG) + 5;
    return i >= 0 && j >= 0 && i < UW && j < UW && urb[j * UW + i] === 1;
  };
  const blocked = (wx, wz, r) => waterLevelAt(wx, wz) !== null || !clear(wx - t.ox, wz - t.oz, r) || !!pushOutOfBuildings(wx, wz, r);
  const landMatAt = (wx, wz) => { const l = landAt(wx, wz); return l ? l.mat : L_GRASS; };
  // lowest ground under a round footprint, so tanks and silos on slopes don't hang over the downhill side
  const footY = (wx, wz, r) => {
    let m = terrainHeight(wx, wz) ?? 0;
    for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4, h = terrainHeight(wx + Math.cos(a) * r, wz + Math.sin(a) * r); if (h !== null && h < m) m = h; }
    return m;
  };
  // ---- town helpers
  const BRANDS = [[0.2, 0.45, 0.18], [0.75, 0.12, 0.1], [0.14, 0.3, 0.62], [0.92, 0.45, 0.08], [0.55, 0.75, 0.2], [0.62, 0.62, 0.64], [0.85, 0.8, 0.2]];
  const CAR = [[0.85, 0.86, 0.87], [0.08, 0.08, 0.09], [0.55, 0.06, 0.06], [0.15, 0.22, 0.45], [0.45, 0.46, 0.47], [0.6, 0.55, 0.45], [0.2, 0.35, 0.25], [0.93, 0.93, 0.92], [0.75, 0.75, 0.77], [0.1, 0.3, 0.55], [0.7, 0.55, 0.15]];
  const osm = t.osm || null;
  const walk = { pos: [], col: [], idx: [] };
  const WM = Math.ceil(S / 2), walkMask = new Uint8Array(WM * WM);
  const zebra = (x, z, h, ttx, ttz, hw) => {
    const q = yawQ(Math.atan2(ttx, ttz));
    for (let u = -hw + 0.6; u <= hw - 0.5; u += 1.0) add('stripe', t.ox + x - ttz * u, h + 0.012, t.oz + z + ttx * u, q, 1, 1, 3.0);
  };
  // traffic signal for traffic approaching a junction: e = entry point, dir = travel direction along the road
  const signal = (e, dir, hw, lanes) => {
    const ax = e.tx * dir, az = e.tz * dir, ks = REGION.left ? -1 : 1;
    const rx = -az * ks, rz = ax * ks;
    let px = 0, pz = 0, found = false;
    for (let back = 0; back <= 14 && !found; back += 2) {
      px = e.x + rx * (hw + 1.0) - ax * back; pz = e.z + rz * (hw + 1.0) - az * back;
      found = inside(px, pz) && clear(px, pz, 0.5) && !pushOutOfBuildings(t.ox + px, t.oz + pz, 0.4);
    }
    if (!found) return;
    const wx = t.ox + px, wz = t.oz + pz;
    const y = terrainHeight(wx, wz) ?? e.h, face = yawQ(Math.atan2(-ax, -az));
    if (REGION.id === 'us' || REGION.id === 'ca' || REGION.id === 'latam') {
      add('sigpole', wx, y, wz, yawQ(0), 1, 1, 1);
      const armLen = Math.min(hw * 1.6, 11);
      add('sigarm', wx, y + 6.2, wz, yawQ(Math.atan2(-rx, -rz)), 1, 1, armLen);
      const nHeads = Math.abs(lanes) <= 2 ? 2 : 3;
      for (let k = 0; k < nHeads; k++) { const d = armLen * (0.35 + 0.6 * k / (nHeads - 1)); add('sighead', wx - rx * d, y + 5.45, wz - rz * d, face, 1, 1, 1); }
    } else {
      add('sigpole', wx, y, wz, yawQ(0), 1, 0.55, 1);
      add('sighead', wx, y + 3.0, wz, face, 1, 1, 1);
    }
    col.c.push(wx, wz, 0.3, -1);
  };


  // rebuild road polylines from the physics segments
  const s = t.segs, roads = [];
  let cur = null;
  for (let o = 0; o < s.length; o += SEG) {
    if (!cur || cur.rid !== s[o + 12] || Math.abs(cur.x[cur.x.length - 1] - s[o]) > 0.01) {
      cur = { rid: s[o + 12], x: [s[o]], z: [s[o + 1]], h: [s[o + 9]], br: [], hw: s[o + 4], surf: s[o + 5], rank: s[o + 6], cls: CLASS_LIST[s[o + 7]], lanes: s[o + 11] };
      roads.push(cur);
    }
    cur.x.push(s[o + 2]); cur.z.push(s[o + 3]); cur.h.push(s[o + 10]); cur.br.push(s[o + 8]);
  }
  // is a circle (tile-local x, z, radius) clear of every ground-level road, railway and roundabout island?
  // `own` skips one road id (for things that deliberately sit on or against their own road)
  const RG = t.rings || [];
  const clear = (x, z, rad, own = -1) => {
    if (t.buckets) {
      const bi = Math.floor(x / BCELL), bj = Math.floor(z / BCELL);
      let lastCell = -1;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const ci = Math.min(BUCKETS - 1, Math.max(0, bi + di)), cj = Math.min(BUCKETS - 1, Math.max(0, bj + dj)), cell = cj * BUCKETS + ci;
        if (cell === lastCell) continue;
        lastCell = cell;
        for (const o of t.buckets[cell]) {
          if (s[o + 8] || s[o + 12] === own) continue;
          const lim = s[o + 4] + rad, ax = s[o], az = s[o + 1], bx = s[o + 2], bz = s[o + 3];
          if (Math.min(ax, bx) - lim > x || Math.max(ax, bx) + lim < x || Math.min(az, bz) - lim > z || Math.max(az, bz) + lim < z) continue;
          const dx = bx - ax, dz = bz - az, u = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz || 1e-9)));
          const ex = ax + dx * u - x, ez = az + dz * u - z;
          if (ex * ex + ez * ez < lim * lim) return false;
        }
      }
    }
    for (let k = 0; k < RG.length; k += 5) if (Math.hypot(RG[k] - x, RG[k + 1] - z) < RG[k + 2] + RG[k + 3] + rad) return false;
    return true;
  };

  // driveways first, so fences can leave gaps for them
  const mouths = [];
  const drive = { pos: [], nor: [], rd: [], rs: [], idx: [] };
  if (t.foot && t.buckets) {
    for (let b = 0; b < t.footOff.length - 1; b++) {
      if ((b & 31) === 31) yield;
      const i0 = t.footOff[b], i1 = t.footOff[b + 1];
      let cx = 0, cz = 0, area = 0;
      for (let i = i0, j = i1 - 1; i < i1; j = i++) { cx += t.foot[2 * i]; cz += t.foot[2 * i + 1]; area += t.foot[2 * j] * t.foot[2 * i + 1] - t.foot[2 * i] * t.foot[2 * j + 1]; }
      cx /= (i1 - i0); cz /= (i1 - i0); area = Math.abs(area / 2);
      if (area < 40 || area > 450 || !inside(cx, cz)) continue;
      // nearest drivable minor road
      let best = null;
      const bi = Math.floor(cx / BCELL), bj = Math.floor(cz / BCELL);
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        if (bi + di < 0 || bj + dj < 0 || bi + di >= BUCKETS || bj + dj >= BUCKETS) continue;
        for (const o of t.buckets[(bj + dj) * BUCKETS + bi + di]) {
          if (s[o + 6] < 3 || s[o + 6] > 6 || s[o + 8]) continue;
          const ax = s[o], az = s[o + 1], dx = s[o + 2] - ax, dz = s[o + 3] - az;
          if (Math.abs(ax - cx) > 80 || Math.abs(az - cz) > 80) continue;
          const u = Math.max(0, Math.min(1, ((cx - ax) * dx + (cz - az) * dz) / (dx * dx + dz * dz || 1e-9)));
          const ex2 = ax + dx * u - cx, ez2 = az + dz * u - cz, d = Math.sqrt(ex2 * ex2 + ez2 * ez2);
          if (!best || d < best.d) best = { o, d, u };
        }
      }
      if (!best || best.d > 65 || best.d < s[best.o + 4] + 5) continue;
      const o = best.o, hw = s[o + 4];
      const qx = s[o] + (s[o + 2] - s[o]) * best.u, qz = s[o + 1] + (s[o + 3] - s[o + 1]) * best.u, qh = s[o + 9] + (s[o + 10] - s[o + 9]) * best.u;
      let nx = cx - qx, nz = cz - qz; const nl = Math.hypot(nx, nz); nx /= nl; nz /= nl;
      const ex = qx + nx * (hw - 0.3), ez = qz + nz * (hw - 0.3);
      // closest point of the footprint to the road end
      let bx = cx, bz = cz, bd = Infinity;
      for (let i = i0, j = i1 - 1; i < i1; j = i++) {
        const ax = t.foot[2 * j], az = t.foot[2 * j + 1], dx = t.foot[2 * i] - ax, dz = t.foot[2 * i + 1] - az, l2 = dx * dx + dz * dz || 1e-9;
        const u = Math.max(0, Math.min(1, ((ex - ax) * dx + (ez - az) * dz) / l2)), px = ax + dx * u, pz = az + dz * u, d = Math.hypot(px - ex, pz - ez);
        if (d < bd) { bd = d; bx = px; bz = pz; }
      }
      if (bd < 2 || bd > 70) continue;
      const urban = urbanAt(ex, ez);
      // ribbon from road edge to the house, draped on the terrain, drawn with the road shader
      const dl = Math.hypot(bx - ex, bz - ez), ux = (bx - ex) / dl, uz = (bz - ez) / dl, px = -uz * 1.6, pz = ux * 1.6;
      let crosses = false;
      for (let f = 2; f < dl && !crosses; f += 2) if (!clear(ex + ux * f, ez + uz * f, 1.6, s[o + 12])) crosses = true;
      if (crosses) continue;
      const steps = Math.max(1, Math.ceil(dl / 2)), base = drive.pos.length / 3;
      for (let k = 0; k <= steps; k++) {
        const f = k / steps, x = ex + (bx - ex) * f, z = ez + (bz - ez) * f;
        for (const sg of [1, -1]) {
          const vx = x + px * sg, vz = z + pz * sg, th = terrainHeight(t.ox + vx, t.oz + vz) ?? qh;
          drive.pos.push(vx, k === 0 ? qh - 0.005 : Math.max(th + 0.06, k === 1 ? qh - 0.1 : -Infinity), vz);
          drive.nor.push(0, 1, 0);
          drive.rd.push(sg > 0 ? 0 : 3.2, f * dl, 3.2, 0);
          drive.rs.push(urban ? 0 : 1, 1, 0, 0);
        }
      }
      for (let k = 0; k < steps; k++) { const a = base + 2 * k; drive.idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
      mouths.push(ex, ez);
      // wheelie bins at the kerb on some houses, a car on many driveways
      if (urban && rnd(cx, cz, 280) < 0.35) for (let k = 0; k < 1 + (rnd(cx, cz, 281) < 0.5 ? 1 : 0); k++) {
        const bx2 = ex + nx * 0.5 - nz * (2.0 + k * 0.75), bz2 = ez + nz * 0.5 + nx * (2.0 + k * 0.75), wx = t.ox + bx2, wz = t.oz + bz2, y = terrainHeight(wx, wz);
        if (y === null || !clear(bx2, bz2, 0.4, s[o + 12])) continue;
        const kid = nextKid++, lid = [[0.15, 0.4, 0.18], [0.15, 0.3, 0.6], [0.1, 0.1, 0.1], [0.55, 0.45, 0.1]][Math.floor(rnd(bx2, bz2, 282) * 4)];
        add('wheelie', wx, Math.max(y, qh - 0.02), wz, yawQ(Math.atan2(-nx, -nz) + (rnd(bx2, bz2, 283) - 0.5) * 0.5), 1, 1, 1, lin(lid), 0, kid);
        col.c.push(wx, wz, 0.35, kid);
      }
      if (false) {   // no parked cars
        const along = Math.min(dl - 3.5, 6 + rnd(cx, cz, 286) * 4), vx = ex + ux * along, vz = ez + uz * along, wx = t.ox + vx, wz = t.oz + vz, y = terrainHeight(wx, wz);
        if (y !== null && !blocked(wx, wz, 1.2)) {
          const rv = rnd(cx, cz, 287), kind = REGION.id === 'us' ? (rv < 0.35 ? 'suv' : rv < 0.6 ? 'pickup' : rv < 0.85 ? 'sedan' : 'van') : (rv < 0.4 ? 'hatch' : rv < 0.7 ? 'sedan' : rv < 0.9 ? 'suv' : 'van');
          const cc = CAR[Math.floor(rnd(cx, cz, 288) * CAR.length)];
          add(kind, wx, y + 0.04, wz, yawQ(Math.atan2(ux, uz) + (rnd(cx, cz, 289) < 0.5 ? Math.PI : 0)), 1, 1, 1, lin(cc));
          col.c.push(wx + ux * 1.2, wz + uz * 1.2, 0.95, -1, wx - ux * 1.2, wz - uz * 1.2, 0.95, -1);
        }
      }
      // mailbox beside the driveway mouth
      if (REGION.mailbox && !urban) {
        const tx_ = -nz, tz_ = nx, side = rnd(cx, cz, 5) < 0.5 ? 1 : -1;
        const mx = ex + nx * 1.1 + tx_ * 2.4 * side, mz = ez + nz * 1.1 + tz_ * 2.4 * side;
        const wx = t.ox + mx, wz = t.oz + mz, y = terrainHeight(wx, wz);
        if (y !== null && clear(mx, mz, 0.3)) {
          const r1 = rnd(mx, mz, 6), r2 = rnd(mx, mz, 7), lean = (rnd(mx, mz, 8) - 0.5) * (r2 < 0.08 ? 0.5 : 0.12);
          const body = [[0.08, 0.08, 0.08], [0.85, 0.86, 0.87], [0.62, 0.64, 0.66], [0.15, 0.32, 0.18], [0.6, 0.1, 0.08], [0.12, 0.2, 0.45], [0.45, 0.33, 0.2]][r1 < 0.4 ? 0 : r1 < 0.55 ? 1 : r1 < 0.75 ? 2 : r1 < 0.83 ? 3 : r1 < 0.88 ? 4 : r1 < 0.93 ? 5 : 6];
          const post = r2 < 0.5 ? C.wood : r2 < 0.8 ? [0.1, 0.1, 0.1] : C.white;
          const kid = nextKid++, q = yawQ(Math.atan2(-nx, -nz), lean, (rnd(mx, mz, 9) - 0.5) * 0.1);
          add('mpost', wx, y, wz, q, 1, 1, 1, lin(post), 0, kid);
          add('mbox', wx, y, wz, q, 1, 1, 1, lin(body), 0, kid);
          if (r1 > 0.3) add('mflag', wx, y, wz, q, 1, 1, 1, [1, 1, 1], 0, kid);
          col.c.push(wx, wz, 0.25, kid);
        }
      }
    }
  }
  mark('setup+driveways');
  const MG = 16, mouthGrid = new Map();
  for (let k = 0; k < mouths.length; k += 2) { const key = Math.floor(mouths[k] / MG) * 100003 + Math.floor(mouths[k + 1] / MG); (mouthGrid.get(key) || mouthGrid.set(key, []).get(key)).push(mouths[k], mouths[k + 1]); }
  const nearMouth = (x, z, r) => {
    const ci = Math.floor(x / MG), cj = Math.floor(z / MG);
    for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      const l = mouthGrid.get((ci + di) * 100003 + cj + dj);
      if (l) for (let k = 0; k < l.length; k += 2) if (Math.hypot(l[k] - x, l[k + 1] - z) < r) return true;
    }
    return false;
  };

  const signPlate = (cell, wx, wz, y0, faceX, faceZ, size, height, kid) => {
    const yaw = Math.atan2(faceX, faceZ), q = yawQ(yaw, 0, 0);
    add('spost', wx, y0, wz, q, 1, height / 2.6, 1, [1, 1, 1], 0, kid);
    const cy = y0 + height - size / 2 - 0.05;
    add('sfront', wx + faceX * 0.04, cy, wz + faceZ * 0.04, q, size, size, 1, [1, 1, 1], cell, kid);
    add('sback', wx + faceX * 0.03, cy, wz + faceZ * 0.03, yawQ(yaw + Math.PI), size, size, 1, [1, 1, 1], cell, kid);
  };
  // place a sign at (x,z) local, facing (fx,fz); several plates can share a post
  const sign = (cells, x, z, fx, fz, size = 0.85) => {
    const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz);
    if (y === null || blocked(wx, wz, 0.4)) return;
    const kid = nextKid++;
    signPlate(cells[0], wx, wz, y, fx, fz, size, 2.55, kid);
    for (let k = 1; k < cells.length; k++) {
      const yaw = Math.atan2(fx, fz), q = yawQ(yaw), cy = y + 2.55 - size / 2 - 0.05 - k * (size + 0.1);
      add('sfront', wx + fx * 0.04, cy, wz + fz * 0.04, q, size * 0.8, size * 0.8, 1, [1, 1, 1], cells[k], kid);
      add('sback', wx + fx * 0.03, cy, wz + fz * 0.03, yawQ(yaw + Math.PI), size * 0.8, size * 0.8, 1, [1, 1, 1], cells[k], kid);
    }
    col.c.push(wx, wz, 0.12, kid);
  };

  for (const r of roads) {
    yield;
    const n = r.x.length;
    if (n < 3 || r.cls === 'busway' || r.cls === 'raceway') continue;
    const len = new Float32Array(n), tx = new Float32Array(n), tz = new Float32Array(n), turn = new Float32Array(n);
    for (let i = 1; i < n; i++) len[i] = len[i - 1] + Math.hypot(r.x[i] - r.x[i - 1], r.z[i] - r.z[i - 1]);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1), l = Math.hypot(r.x[b] - r.x[a], r.z[b] - r.z[a]) || 1;
      tx[i] = (r.x[b] - r.x[a]) / l; tz[i] = (r.z[b] - r.z[a]) / l;
    }
    // signed curvature (1/m), positive = turning left when travelling along +t
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 4), b = Math.min(n - 1, i + 4);
      const ha = Math.atan2(tx[a], tz[a]), hb = Math.atan2(tx[b], tz[b]);
      let dh = hb - ha; dh = Math.atan2(Math.sin(dh), Math.cos(dh));
      turn[i] = len[b] > len[a] ? dh / (len[b] - len[a]) : 0;
    }
    const L = len[n - 1];
    // roundabout rings get no roadside furniture (their islands come from the worker); other closed loops (courts,
    // crescents) are ordinary streets
    if (n > 8 && L < 450 && RG.length && Math.hypot(r.x[0] - r.x[n - 1], r.z[0] - r.z[n - 1]) < 1.5 && r.cls !== 'rail') {
      let cx = 0, cz = 0;
      for (let k = 0; k < n; k++) { cx += r.x[k]; cz += r.z[k]; }
      cx /= n; cz /= n;
      let isRing = false;
      for (let k = 0; k < RG.length; k += 5) if (Math.hypot(RG[k] - cx, RG[k + 1] - cz) < Math.max(4, RG[k + 2] * 0.35)) isRing = true;
      if (isRing) continue;
    }
    let ptr = 0;
    const at = d => {           // point at distance d (monotonic walks reset ptr themselves)
      while (ptr < n - 2 && len[ptr + 1] < d) ptr++;
      while (ptr > 0 && len[ptr] > d) ptr--;
      const f = Math.max(0, Math.min(1, (d - len[ptr]) / ((len[ptr + 1] - len[ptr]) || 1)));
      return { x: r.x[ptr] + (r.x[ptr + 1] - r.x[ptr]) * f, z: r.z[ptr] + (r.z[ptr + 1] - r.z[ptr]) * f, h: r.h[ptr] + (r.h[ptr + 1] - r.h[ptr]) * f,
               tx: tx[ptr], tz: tz[ptr], k: turn[ptr], br: r.br[Math.min(ptr, n - 2)], i: ptr };
    };
    // --- railways: two rails and sleepers on the ballast bed
    if (r.cls === 'rail') {
      const sl = REGION.id === 'us' || REGION.id === 'ca' ? lin([0.36, 0.3, 0.25]) : lin([0.62, 0.61, 0.58]);
      for (let i = 0; i < n - 1; i++) {
        const mx = (r.x[i] + r.x[i + 1]) / 2, mz = (r.z[i] + r.z[i + 1]) / 2;
        if (!inside(mx, mz)) continue;
        _va.set(r.x[i + 1] - r.x[i], r.h[i + 1] - r.h[i], r.z[i + 1] - r.z[i]); const l = _va.length(); _va.normalize();
        const q = _qb.setFromUnitVectors(_zAxis, _va), my = (r.h[i] + r.h[i + 1]) / 2 + 0.17;
        for (const g of [-0.72, 0.72]) add('railbar', t.ox + mx - tz[i] * g, my, t.oz + mz + tx[i] * g, q, 1, 1, l + 0.02);
      }
      for (let d = 0.3; d < L; d += 0.62) {
        const p = at(d);
        if (!inside(p.x, p.z)) continue;
        const shade = 0.85 + rnd(p.x, p.z, 61) * 0.25;
        add('sleeper', t.ox + p.x, p.h + 0.04, t.oz + p.z, yawQ(Math.atan2(p.tx, p.tz) + (rnd(p.x, p.z, 62) - 0.5) * 0.03), 1, 1, 1, [sl[0] * shade, sl[1] * shade, sl[2] * shade]);
      }
      continue;
    }
    const paved = r.surf === 0, major = r.rank >= 5, hw = r.hw, two = r.lanes > 0;
    const keepSide = REGION.left ? -1 : 1; // +1 = right of travel

    // --- utility poles + wires
    if (r.rank >= 3 && r.rank <= 7 && r.cls !== 'service') {
      let prev = null;
      for (let d = 8 + rnd(r.x[0], r.z[0], 1) * 30; d < L; d += 41 + rnd(r.x[0], r.z[0], Math.floor(d)) * 10) {
        const p = at(d);
        const rx = -p.tz, rz = p.tx, side = rx + rz * 0.7 > 0 ? 1 : -1;
        const off = hw + 3.3 + rnd(p.x, p.z, 2) * 1.5, x = p.x + rx * side * off, z = p.z + rz * side * off;
        if (p.br || !inside(x, z) || nearJ(x, z) < 3) { prev = null; continue; }
        const wx = t.ox + x, wz = t.oz + z;
        if (blocked(wx, wz, 0.6)) { prev = null; continue; }
        const y = terrainHeight(wx, wz) - 0.3, rv = rnd(x, z, 3);
        const variant = rv < 0.62 ? 'poleA' : rv < 0.86 ? 'poleB' : 'poleC', sy = 0.9 + rnd(x, z, 4) * 0.22;
        const yaw = Math.atan2(p.tx, p.tz), tiltX = (rnd(x, z, 5) - 0.5) * 0.06, tiltZ = (rnd(x, z, 6) - 0.5) * 0.06;
        const shade = 0.85 + rnd(x, z, 7) * 0.25;
        add(variant, wx, y, wz, yawQ(yaw, tiltX, tiltZ), 1, sy, 1, [shade, shade, shade]);
        if (rnd(x, z, 8) < (urbanAt(x, z) ? 0.3 : 0.12)) add('xfmr', wx, y, wz, yawQ(yaw, tiltX, tiltZ), 1, sy, 1);
        col.c.push(wx, wz, 0.25, -1);
        const ax = Math.cos(yaw), az = -Math.sin(yaw), pts = ATTACH[variant].map(([u, v]) => [wx + ax * u, y + v * sy, wz + az * u]);
        if (prev && Math.hypot(prev[0][0] - wx, prev[0][2] - wz) < 75) {
          for (let k = 0; k < 3; k++) {
            const A = prev[k], B = pts[k], span = Math.hypot(B[0] - A[0], B[2] - A[2]), sag = span * span / 3000;
            let last = A;
            for (let q = 1; q <= 6; q++) {
              const f = q / 6, P = [A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f - 4 * sag * f * (1 - f), A[2] + (B[2] - A[2]) * f];
              _va.set(P[0] - last[0], P[1] - last[1], P[2] - last[2]);
              const l = _va.length();
              add('wire', (P[0] + last[0]) / 2, (P[1] + last[1]) / 2, (P[2] + last[2]) / 2, _qb.setFromUnitVectors(_zAxis, _va.normalize()), 1, 1, l);
              last = P;
            }
          }
        }
        prev = pts;
      }
    }

    // --- delineator posts on rural paved roads (closer on bends), knockable
    if (paved && r.rank >= 4 && r.cls !== 'service') {
      for (let d = 5; d < L; ) {
        const p = at(d), curvy = Math.abs(p.k) > 1 / 250;
        d += curvy ? 25 : 50;
        if (p.br) continue;
        for (const side of [1, -1]) {
          const rx = -p.tz * side, rz = p.tx * side, x = p.x + rx * (hw + 0.65), z = p.z + rz * (hw + 0.65);
          if (!inside(x, z) || nearJ(x, z) < 4 || urbanAt(x, z) || nearMouth(x, z, 5) || !clear(x, z, 0.25)) continue;
          if (rnd(x, z, 11) < 0.08) continue; // missing ones
          const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz);
          if (y === null || waterLevelAt(wx, wz) !== null) continue;
          // reflectors face the traffic that drives on this side
          const face = side * keepSide > 0 ? -1 : 1, kid = nextKid++;
          const dirt = 0.8 + rnd(x, z, 12) * 0.2;
          add('delin', wx, y, wz, yawQ(Math.atan2(p.tx * face, p.tz * face), (rnd(x, z, 13) - 0.5) * 0.14, (rnd(x, z, 14) - 0.5) * 0.14), 1, 0.9 + rnd(x, z, 15) * 0.15, 1, [dirt, dirt, dirt * 0.97], 0, kid);
          col.c.push(wx, wz, 0.12, kid);
        }
      }
    }

    // --- guardrails: embankments, the outside of tight bends with a drop, bridge approaches
    if (paved && r.rank >= 4) {
      for (const side of [1, -1]) {
        let run = [];
        const flush = () => {
          if (run.length >= 9) {
            for (let k = 0; k < run.length; k++) {
              const [x, y, z, yaw] = run[k];
              add('grpost', x, y, z, yawQ(yaw), 1, 1, 1);
              if (k > 0) {
                const [x0, y0, z0] = run[k - 1];
                _va.set(x - x0, y - y0, z - z0); const l = _va.length();
                add('grrail', (x + x0) / 2, (y + y0) / 2, (z + z0) / 2, _qb.setFromUnitVectors(_zAxis, _va.normalize()), 1, 1, l + 0.05);
                col.s.push(x0, z0, x, z);
              }
            }
          }
          run = [];
        };
        for (let d = 0; d < L; d += 2) {
          const p = at(d);
          const rx = -p.tz * side, rz = p.tx * side, x = p.x + rx * (hw + 0.55), z = p.z + rz * (hw + 0.55);
          const farX = t.ox + p.x + rx * (hw + 18), farZ = t.oz + p.z + rz * (hw + 18), th = terrainHeight(farX, farZ);
          const drop = th === null ? 0 : p.h - th;
          const outside = p.k * side > 0 ? Math.abs(p.k) : 0; // this side is the outside of the bend
          let nearBridge = false;
          for (const dd of [-30, -15, 15, 30]) { const qd = d + dd; if (qd > 0 && qd < L && at(qd).br) nearBridge = true; }
          at(d);
          const need = !p.br && (drop > 3.8 || (outside > 1 / 140 && drop > 1.8) || nearBridge) && inside(x, z) && nearJ(x, z) > 2 && !nearMouth(x, z, 4) && clear(x, z, 0.35);
          if (need) run.push([t.ox + x, p.h - 0.05, t.oz + z, Math.atan2(p.tx, p.tz)]);
          else flush();
        }
        flush();
      }
    }

    // --- signs
    if (paved && r.rank >= 3) {
      // speed limits shortly after each junction (and every ~1.5 km), for both directions of travel
      const events = [];
      let wasNear = true;
      for (let d = 0; d < L; d += 5) { const p = at(d), near = nearJ(p.x, p.z) < 6; if (wasNear && !near) events.push([d + 45, 1]); if (!wasNear && near) events.push([d - 45, -1]); wasNear = near; }
      for (let d = 750; d < L; d += 1500) events.push([d, 1], [d + 30, -1]);
      const lastAt = { 1: -1e9, '-1': 1e9 };
      for (const [d, dir] of events.sort((a, b) => a[0] - b[0])) {
        if (d < 2 || d > L - 2 || (!two && dir < 0)) continue;
        const p = at(d);
        if (p.br || (r.rank < 5 && urbanAt(p.x, p.z)) || r.rank < 4) continue;
        if (dir > 0 ? d - lastAt[1] < 300 : false) continue;
        if (dir < 0 && Math.abs(d - lastAt[-1]) < 300) continue;
        lastAt[dir] = d;
        const side = dir * keepSide, x = p.x - p.tz * side * (hw + 1.4), z = p.z + p.tx * side * (hw + 1.4);
        if (!inside(x, z) || nearJ(x, z) < 4) continue;
        sign([SIGNS.limit(speedLimit(r.rank, urbanAt(x, z)))], x, z, -p.tx * dir, -p.tz * dir, 0.9);
      }
      // curve warnings before bends, chevrons around sharp ones (rural roads)
      if (r.rank >= 4) {
        let i = 0;
        while (i < n) {
          if (Math.abs(turn[i]) < 1 / 160) { i++; continue; }
          const sgn = Math.sign(turn[i]); let j = i, total = 0;
          while (j < n - 1 && Math.sign(turn[j]) === sgn && Math.abs(turn[j]) > 1 / 200) { total += turn[j] * (len[j + 1] - len[j]); j++; }
          const radius = Math.abs(total) > 0 ? (len[j] - len[i]) / Math.abs(total) : 1e9;
          if (Math.abs(total) > 0.45 && !urbanAt(r.x[i], r.z[i])) {
            const sharp = radius < 60;
            for (const dir of two ? [1, -1] : [1]) {
              const d = dir > 0 ? len[i] - 90 : len[j] + 90;
              if (d < 1 || d > L - 1) continue;
              const p = at(d), side = dir * keepSide, x = p.x - p.tz * side * (hw + 1.4), z = p.z + p.tx * side * (hw + 1.4);
              if (!inside(x, z) || nearJ(x, z) < 4) continue;
              const turnDir = sgn * dir > 0 ? -1 : 1; // drawn arrow: -1 left, +1 right
              sign([SIGNS.curve(turnDir, sharp)], x, z, -p.tx * dir, -p.tz * dir, 0.85);
            }
            if (sharp) {
              for (let d = len[i] + 6, c = 0; d <= len[j] && c < 9; d += 18, c++) {
                const p = at(d), side = sgn, x = p.x - p.tz * side * (hw + 1.8), z = p.z + p.tx * side * (hw + 1.8);
                if (!inside(x, z) || nearJ(x, z) < 4) continue;
                // chevrons on the outside, pointing the way the road bends, one plate per direction
                for (const dir of two ? [1, -1] : [1]) sign([SIGNS.chevron(sgn * dir > 0 ? -1 : 1)], x - p.tx * dir * 0.05, z - p.tz * dir * 0.05, -p.tx * dir, -p.tz * dir, 0.7);
              }
            }
          }
          i = j + 1;
        }
      }
      // occasional context warnings
      for (let d = 400; d < L; d += 900) {
        const p = at(d), m = landMatAt(t.ox + p.x, t.oz + p.z), dir = rnd(p.x, p.z, 21) < 0.5 || !two ? 1 : -1;
        if (p.br || urbanAt(p.x, p.z) || rnd(p.x, p.z, 22) > 0.45) continue;
        const cell = m === L_FOREST ? SIGNS.deer() : m === L_CROP ? SIGNS.tractor() : nearJ(at(Math.min(L - 1, d + 150)).x, at(Math.min(L - 1, d + 150)).z) < 5 ? SIGNS.crossroads() : -1;
        if (cell < 0) continue;
        const side = dir * keepSide, x = p.x - p.tz * side * (hw + 1.4), z = p.z + p.tx * side * (hw + 1.4);
        if (inside(x, z) && nearJ(x, z) > 6) sign([cell], x, z, -p.tx * dir, -p.tz * dir, 0.85);
      }
    }

    // --- fences along fields and pastures (rural), with gates and gaps for driveways
    if (r.rank >= 2 && r.rank <= 7) {
      for (const side of [1, -1]) {
        let prev = null;
        for (let d = 2; d < L; ) {
          const p = at(d);
          const rx = -p.tz * side, rz = p.tx * side;
          const cellX = Math.floor((t.ox + p.x) / 260), cellZ = Math.floor((t.oz + p.z) / 260);
          const st = hash(cellX * 3 + (side > 0 ? 1 : 0), cellZ * 7 + 11);
          const style = REGION.id === 'us' || REGION.id === 'ca' || REGION.id === 'au'
            ? (st < 0.32 ? 'none' : st < 0.68 ? 'barbed' : st < 0.8 ? 'woven' : st < 0.92 ? 'rail' : 'electric')
            : (st < 0.5 ? 'none' : st < 0.65 ? 'rail' : st < 0.82 ? 'electric' : 'woven');
          const step = style === 'rail' ? 2.8 : style === 'electric' ? 6 : 3.2;
          d += step;
          if (style === 'none') { prev = null; continue; }
          const off = hw + 4.5 + hash(cellX, cellZ * 5) * 2.5, x = p.x + rx * off, z = p.z + rz * off;
          const wx = t.ox + x, wz = t.oz + z;
          const m = landMatAt(wx, wz);
          if (p.br || !inside(x, z) || (m !== L_CROP && m !== L_GRASS && m !== L_SOIL) || urbanAt(x, z) || nearJ(x, z) < 7 || nearMouth(x, z, 6) || blocked(wx, wz, 3)) { prev = null; continue; }
          // field gates every few hundred metres
          if (prev && rnd(x, z, 31) < 0.012) {
            const ux = p.tx, uz = p.tz, y = terrainHeight(wx, wz), gc = [[0.75, 0.77, 0.8], [0.25, 0.42, 0.25], [0.6, 0.15, 0.1], [0.2, 0.3, 0.55]][Math.floor(rnd(x, z, 32) * 4)];
            add('gate', wx, y, wz, yawQ(Math.atan2(ux, uz) - Math.PI / 2), 1, 1, 1, lin(gc));
            col.s.push(wx, wz, wx + ux * 4, wz + uz * 4);
            d += 4; prev = null; continue;
          }
          const y = terrainHeight(wx, wz);
          if (y === null) { prev = null; continue; }
          const lean = (rnd(x, z, 33) - 0.5) * 0.12, yaw = Math.atan2(p.tx, p.tz);
          if (style === 'barbed') add(rnd(x, z, 34) < 0.2 ? 'wpost' : 'tpost', wx, y, wz, yawQ(yaw, lean, lean * 0.5), 1, 1, 1);
          else if (style === 'woven' || style === 'electric') add(style === 'electric' ? 'tpost' : 'wpost', wx, y, wz, yawQ(yaw, lean, lean * 0.5), 1, style === 'electric' ? 0.85 : 1, 1, style === 'electric' ? lin([0.9, 0.85, 0.3]) : [1, 1, 1]);
          else add('rpost', wx, y, wz, yawQ(yaw, lean * 0.5, 0), 1, 1, 1, lin(st < 0.75 ? [0.93, 0.92, 0.9] : [0.45, 0.35, 0.25]));
          const cur = [wx, y, wz];
          if (prev && Math.hypot(prev[0] - wx, prev[2] - wz) < step * 1.8) {
            const heights = style === 'barbed' ? [0.45, 0.75, 1.05] : style === 'woven' ? [0.15, 0.35, 0.55, 0.75, 0.95, 1.1] : style === 'electric' ? [0.6, 0.95] : [0.4, 0.8, 1.15];
            const fam = style === 'rail' ? 'frail' : style === 'electric' ? 'ftape' : 'fwire';
            const railCol = style === 'rail' ? lin(st < 0.75 ? [0.93, 0.92, 0.9] : [0.45, 0.35, 0.25]) : [1, 1, 1];
            _va.set(wx - prev[0], y - prev[1], wz - prev[2]); const l = _va.length(); _va.normalize();
            const q = _qb.setFromUnitVectors(_zAxis, _va);
            for (const hh of heights) add(fam, (wx + prev[0]) / 2, (y + prev[1]) / 2 + hh, (wz + prev[2]) / 2, q, 1, 1, l, railCol);
            col.s.push(prev[0], prev[2], wx, wz);
          }
          prev = cur;
        }
      }
    }

    // --- town life: street lights, kerbs + sidewalks, parked cars, hydrants, crosswalks, traffic signals
    if (r.rank >= 3 && r.cls !== 'service' && r.cls !== 'track' && paved) {
      const realLamps = osm && osm.lamps.length >= 12;
      const lampSides = r.rank >= 6 ? [1, -1] : [hash(Math.floor((t.ox + r.x[0]) / 300), Math.floor((t.oz + r.z[0]) / 300)) < 0.5 ? 1 : -1];
      if (!realLamps) for (let d = 10; d < L; d += 33) {
        for (const side of lampSides) {
          const p = at(Math.min(L - 1, d + (lampSides.length > 1 && side < 0 ? 16 : 0)));
          if (p.br || !urbanAt(p.x, p.z) || nearJ(p.x, p.z) < 2) continue;
          const x = p.x - p.tz * side * (hw + 0.7), z = p.z + p.tx * side * (hw + 0.7);
          if (!inside(x, z) || nearMouth(x, z, 3)) continue;
          const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz);
          if (y === null || blocked(wx, wz, 0.3)) continue;
          add('lamp', wx, Math.max(y, p.h) - 0.05, wz, yawQ(Math.atan2(p.tz * side, -p.tx * side)), 1, 1, 1);
          col.c.push(wx, wz, 0.2, -1);
        }
      }
      // kerbs and sidewalks along urban streets: continuous runs only (no stubs), clear of other roads
      for (const side of [1, -1]) {
        let run = [];
        const flushRun = () => {
          if (run.length >= 5) {
            for (const pp of run) for (let u = hw; u <= hw + 2.2; u += 0.9) {
              const mx = Math.floor((pp.x + pp.ux * u) / 2), mz = Math.floor((pp.z + pp.uz * u) / 2);
              if (mx >= 0 && mz >= 0 && mx < WM && mz < WM) walkMask[mz * WM + mx] = 1;
            }
            for (let k = 1; k < run.length; k++) {
              const b = walk.pos.length / 3;
              for (const v of [...run[k - 1].v, ...run[k].v]) walk.pos.push(v[0], v[1], v[2]);
              const nk = Math.pow(0.6 + rnd(run[k].x, run[k].z, 230) * 0.07, 2.2), kb = Math.pow(0.7, 2.2);
              for (let q = 0; q < 8; q++) { const kk = q % 4 === 1 ? kb : nk; walk.col.push(kk, kk, kk * 0.97); }
              for (const [a0, a1] of [[0, 1], [1, 2], [2, 3]]) walk.idx.push(b + a0, b + a1, b + 4 + a1, b + a0, b + 4 + a1, b + 4 + a0);
            }
          }
          run = [];
        };
        for (let d = 0; d <= L; d += 2) {
          const p = at(Math.min(d, L - 0.01));
          const ux = -p.tz * side, uz = p.tx * side, top = p.h + 0.15;
          // runs right up to the corner: the strip only has to stay clear of the cross street (junction zones can be
          // 20+ m deep at shallow angles and roundabouts, which used to leave only stubs)
          const ok = !p.br && urbanAt(p.x, p.z) && inside(p.x, p.z)
            && clear(p.x + ux * (hw + 0.4), p.z + uz * (hw + 0.4), 0.8, r.rid) && clear(p.x + ux * (hw + 1.8), p.z + uz * (hw + 1.8), 0.8, r.rid);
          if (!ok) { flushRun(); continue; }
          const ox = p.x + ux * (hw + 2.1), oz = p.z + uz * (hw + 2.1), gth = terrainHeight(t.ox + ox, t.oz + oz);
          run.push({ x: p.x, z: p.z, ux, uz, v: [[p.x + ux * hw, p.h - 0.02, p.z + uz * hw], [p.x + ux * hw, top, p.z + uz * hw], [ox, top, oz], [ox, Math.min(top, (gth ?? top) - 0.05), oz]] });
        }
        flushRun();
      }
      // parked cars on residential streets
      if (false) {   // no parked cars
        for (let d = 9; d < L - 9; d += 6.5) {
          const p = at(d);
          for (const side of [1, -1]) {
            if (rnd(p.x, p.z, 240 + side) > 0.22 || p.br || !urbanAt(p.x, p.z) || nearJ(p.x, p.z) < 8) continue;
            const x = p.x - p.tz * side * (hw - 1.05), z = p.z + p.tx * side * (hw - 1.05);
            if (!inside(x, z) || nearMouth(x, z, 4) || !clear(x, z, 1.2, r.rid)) continue;
            const wx = t.ox + x, wz = t.oz + z, face = side * keepSide > 0 ? 1 : -1, rv = rnd(x, z, 242);
            const kind = rv < 0.35 ? 'sedan' : rv < 0.65 ? 'suv' : rv < 0.85 ? 'hatch' : rv < 0.93 ? 'van' : 'pickup';
            const cc = CAR[Math.floor(rnd(x, z, 243) * CAR.length)], sh = 0.85 + rnd(x, z, 244) * 0.25;
            add(kind, wx, p.h - 0.02, wz, yawQ(Math.atan2(p.tx * face, p.tz * face) + (rnd(x, z, 245) - 0.5) * 0.06), 1, 1, 1, lin([cc[0] * sh, cc[1] * sh, cc[2] * sh]));
            col.c.push(wx + p.tx * 1.2, wz + p.tz * 1.2, 0.95, -1, wx - p.tx * 1.2, wz - p.tz * 1.2, 0.95, -1);
          }
        }
      }
      // fire hydrants on the sidewalk
      if (REGION.id !== 'eu' && REGION.id !== 'uk' && !(osm && osm.hydrants.length)) for (let d = 40; d < L; d += 110) {
        const p = at(d), side = rnd(p.x, p.z, 250) < 0.5 ? 1 : -1;
        if (p.br || !urbanAt(p.x, p.z)) continue;
        const x = p.x - p.tz * side * (hw + 0.8), z = p.z + p.tx * side * (hw + 0.8);
        if (!inside(x, z) || nearJ(x, z) < 3 || nearMouth(x, z, 3) || !clear(x, z, 0.3)) continue;
        const wx = t.ox + x, wz = t.oz + z, kid = nextKid++;
        add('hydrant', wx, p.h + 0.12, wz, yawQ(rnd(x, z, 251) * 6.28), 1, 1, 1, lin(rnd(x, z, 252) < 0.75 ? [0.75, 0.1, 0.08] : [0.95, 0.8, 0.1]), 0, kid);
        col.c.push(wx, wz, 0.15, kid);
      }
      // crosswalks + traffic signals where this road enters a junction in town
      if (r.rank >= 4) {
        let was = nearJ(at(0).x, at(0).z) < 0;
        for (let d = 1; d < L; d += 1) {
          const p = at(d), inJ = nearJ(p.x, p.z) < 0;
          if (inJ !== was) {
            const enter = inJ, dir = enter ? 1 : -1;       // travel direction that is heading into the junction
            const e = at(Math.max(0, Math.min(L - 0.01, d - dir * 1.8)));
            let JR = 0; for (let k = 0; k < J.length; k += 3) if (Math.hypot(J[k] - p.x, J[k + 1] - p.z) < J[k + 2] + 3) JR = Math.max(JR, J[k + 2]);
            if (urbanAt(e.x, e.z) && inside(e.x, e.z) && !e.br) {
              const realX = osm && osm.crossings.length;
              if (!realX && JR >= 4.5 && r.rank >= 5 && rnd(e.x, e.z, 260) < 0.35) zebra(e.x, e.z, e.h, e.tx, e.tz, hw);
              const realSig = osm && osm.signals.length;
              const sigHere = realSig ? osm.signals.some((v, k) => k % 2 === 0 && Math.hypot(v - p.x, osm.signals[k + 1] - p.z) < 30) : (r.rank >= 7 && JR >= 6.2);
              if (sigHere && (two || dir > 0)) signal(e, dir, hw, r.lanes);
            }
            was = inJ;
            d += 4;
          }
        }
      }
    }

    // --- motorway sign gantries over a carriageway every ~1.6 km
    if (r.rank === 9 && paved && !two) {
      for (let d = 300 + rnd(r.x[0], r.z[0], 310) * 900; d < L; d += 1600) {
        const p = at(d);
        if (p.br || !inside(p.x, p.z) || nearJ(p.x, p.z) < 10) continue;
        const rx = -p.tz, rz = p.tx, wx = t.ox + p.x, wz = t.oz + p.z, yaw = Math.atan2(p.tx, p.tz);
        const span = 2 * hw + 3;
        for (const side of [1, -1]) {
          const lx = wx + rx * side * (hw + 1.5), lz = wz + rz * side * (hw + 1.5), ly = terrainHeight(lx, lz) ?? p.h;
          add('gantryleg', lx, ly - 0.2, lz, yawQ(yaw), 1, (p.h + 7.2 - ly + 0.2) / 7.2, 1);
          col.c.push(lx, lz, 0.5, -1);
        }
        const bx = wx - rx * (hw + 1.5), bz = wz - rz * (hw + 1.5);
        add('gantrybeam', bx, p.h + 7.0, bz, yawQ(Math.atan2(rx, rz)), 1, 1, span);
        const cell = SIGNS.gantry(), face = Math.atan2(-p.tx, -p.tz);
        add('sfront', wx - p.tx * 0.3, p.h + 6.2, wz - p.tz * 0.3, yawQ(face), 5.2, 5.2, 1, [1, 1, 1], cell);
        add('sback', wx - p.tx * 0.25, p.h + 6.2, wz - p.tz * 0.25, yawQ(face + Math.PI), 5.2, 5.2, 1, [1, 1, 1], cell);
      }
    }
    // --- rare set pieces on rural roads: roadworks, a broken-down car, a hay wagon on the verge
    if (paved && r.rank >= 4 && r.rank <= 7) for (let d = 200; d < L - 200; d += 1100) {
      const p0 = at(d);
      const rv = rnd(p0.x, p0.z, 320);
      if (rv > 0.2 || p0.br || urbanAt(p0.x, p0.z) || !inside(p0.x, p0.z) || nearJ(p0.x, p0.z) < 40) continue;
      const side = keepSide, kind = rnd(p0.x, p0.z, 321);
      if (kind < 0.4) {
        // roadworks: warning sign, then a taper of cones closing the outer lane, barrels, truck and a flagger
        const s0 = at(d - 140), sx = s0.x - s0.tz * side * (hw + 1.4), sz = s0.z + s0.tx * side * (hw + 1.4);
        if (inside(sx, sz)) sign([SIGNS.roadwork()], sx, sz, -s0.tx, -s0.tz, 0.95);
        for (let q = 0; q < 14; q++) {
          const p = at(d + q * 4), u = hw - 0.3 - Math.min(q, 5) / 5 * (hw * 0.5), x = p.x - p.tz * side * u, z = p.z + p.tx * side * u;
          const wx = t.ox + x, wz = t.oz + z, kid = nextKid++;
          add(q % 5 === 0 ? 'barrel' : 'cone', wx, p.h - 0.02, wz, yawQ(rnd(x, z, 330) * 6), 1, 1, 1, [1, 1, 1], 0, kid);
          col.c.push(wx, wz, q % 5 === 0 ? 0.3 : 0.18, kid);
        }
        const pt = at(d + 30), tx2 = pt.x - pt.tz * side * (hw * 0.55), tz2 = pt.z + pt.tx * side * (hw * 0.55);
        const pf = at(d - 12), fx = pf.x - pf.tz * side * (hw + 0.8), fz = pf.z + pf.tx * side * (hw + 0.8);
        add('person', t.ox + fx, (terrainHeight(t.ox + fx, t.oz + fz) ?? pf.h), t.oz + fz, yawQ(Math.atan2(-pf.tx, -pf.tz)), 1, 1, 1, lin([0.95, 0.55, 0.05]));
      } else if (false) {
        // broken-down car half on the shoulder, warning triangle behind it, driver standing by
        const p = at(d), x = p.x - p.tz * side * (hw - 0.4), z = p.z + p.tx * side * (hw - 0.4), wx = t.ox + x, wz = t.oz + z;
        if (!clear(x, z, 1.2, r.rid)) continue;
        const kindCar = ['sedan', 'hatch', 'suv', 'van'][Math.floor(rnd(x, z, 340) * 4)], cc = CAR[Math.floor(rnd(x, z, 341) * CAR.length)];
        add(kindCar, wx, p.h - 0.02, wz, yawQ(Math.atan2(p.tx, p.tz) + 0.06 * side), 1, 1, 1, lin(cc));
        col.c.push(wx + p.tx * 1.2, wz + p.tz * 1.2, 0.95, -1, wx - p.tx * 1.2, wz - p.tz * 1.2, 0.95, -1);
        const pb = at(d - 30), bx = pb.x - pb.tz * side * (hw - 1.0), bz = pb.z + pb.tx * side * (hw - 1.0), kid = nextKid++;
        add('warntri', t.ox + bx, pb.h, t.oz + bz, yawQ(Math.atan2(-pb.tx, -pb.tz)), 1, 1, 1, [1, 1, 1], 0, kid);
        col.c.push(t.ox + bx, t.oz + bz, 0.2, kid);
        const px2 = x - p.tz * side * 1.8, pz2 = z + p.tx * side * 1.8;
        add('person', t.ox + px2, terrainHeight(t.ox + px2, t.oz + pz2) ?? p.h, t.oz + pz2, yawQ(rnd(x, z, 342) * 6.28), 1, 1, 1, lin(CAR[Math.floor(rnd(x, z, 343) * CAR.length)]));
      } else {
        const p = at(d), x = p.x - p.tz * side * (hw + 2.6), z = p.z + p.tx * side * (hw + 2.6), wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz);
        if (y !== null && !blocked(wx, wz, 3)) { add('trailer', wx, y, wz, yawQ(Math.atan2(p.tx, p.tz)), 1, 1, 1); col.c.push(wx + p.tx * 1.5, wz + p.tz * 1.5, 1.3, -1, wx - p.tx * 1.5, wz - p.tz * 1.5, 1.3, -1); }
      }
    }

    // --- road cuts: where the road was cut into a hillside, fallen rocks lie in the ditch
    if (r.rank >= 3) for (let d = 4; d < L; d += 9) {
      const p = at(d);
      if (p.br) continue;
      for (const side of [1, -1]) {
        const ux = -p.tz * side, uz = p.tx * side, up = terrainHeight(t.ox + p.x + ux * (hw + 9), t.oz + p.z + uz * (hw + 9));
        if (up === null || up - p.h < 3 || rnd(p.x, p.z, 70 + side) > 0.35) continue;
        const off = hw + 1.3 + rnd(p.x, p.z, 72) * 1.8, x = p.x + ux * off, z = p.z + uz * off;
        if (!inside(x, z) || !clear(x, z, 0.8)) continue;
        const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz), sz = 0.2 + Math.pow(rnd(x, z, 73), 2) * 0.8;
        add('rock' + Math.floor(rnd(x, z, 74) * 6), wx, y - sz * 0.25, wz, yawQ(rnd(x, z, 75) * 6.28, (rnd(x, z, 76) - 0.5) * 0.6), sz * 1.1, sz * 0.8, sz, lin([0.58, 0.55, 0.5]));
        if (sz > 0.7) col.c.push(wx, wz, sz * 0.8, -1);
      }
    }

    // --- verge life: bushes and wildflowers on grassy verges
    for (let d = 3; d < L; d += 7) {
      const p = at(d);
      for (const side of [1, -1]) {
        const rv = rnd(p.x, p.z, 40 + side);
        if (rv > 0.24 || p.br) continue;
        const off = hw + 2 + rnd(p.x, p.z, 42 + side) * 7, x = p.x - p.tz * side * off, z = p.z + p.tx * side * off;
        if (!inside(x, z) || nearJ(x, z) < 3 || nearMouth(x, z, 4)) continue;
        const wx = t.ox + x, wz = t.oz + z, l = landAt(wx, wz);
        if (!l || (l.mat !== L_GRASS && l.mat !== L_SOIL && l.mat !== L_FOREST) || blocked(wx, wz, 1)) continue;
        const y = terrainHeight(wx, wz);
        if (y === null) continue;
        if (rv < 0.15) {
          const sc = 0.5 + rnd(x, z, 43) * 1.2, g = 0.85 + rnd(x, z, 44) * 0.35;
          add('bush', wx, y - 0.1, wz, yawQ(rnd(x, z, 45) * 6.28), sc, sc * (0.7 + rnd(x, z, 46) * 0.5), sc, [Math.pow(l.r / 255, 2.2) * g * 1.1, Math.pow(l.g / 255, 2.2) * g * 1.2, Math.pow(l.b / 255, 2.2) * g]);
        } else {
          const fc = [[1, 0.85, 0.1], [0.75, 0.4, 0.95], [1, 1, 0.95], [1, 0.45, 0.1], [0.4, 0.55, 1], [0.95, 0.2, 0.3]][Math.floor(rnd(x, z, 47) * 6)];
          add('flower', wx, y, wz, yawQ(rnd(x, z, 48) * 6.28), 1, 0.8 + rnd(x, z, 49) * 0.6, 1, lin(fc));
        }
      }
    }
  }

  mark('roads');
  // ---- farms
  const used = [];
  const free = (x, z, r) => {
    for (let k = 0; k < used.length; k += 3) if (Math.hypot(used[k] - x, used[k + 1] - z) < used[k + 2] + r) return false;
    if (blocked(x, z, r)) return false;
    const g = groundAt(x, z);
    if (!g || g.seg >= 0) return false;
    for (const [dx, dz] of [[r, 0], [-r, 0], [0, r], [0, -r]]) { const q = groundAt(x + dx, z + dz); if (!q || q.seg >= 0) return false; }
    return true;
  };
  const occupy = (x, z, r) => used.push(x, z, r);
  const isCrop = l => l && l.mat === L_CROP, isPasture = l => l && l.mat === L_GRASS;
  if (t.foot) {
    const B = [];
    for (let b = 0; b < t.footOff.length - 1; b++) {
      const i0 = t.footOff[b], i1 = t.footOff[b + 1]; let cx = 0, cz = 0, a = 0, rad = 0;
      for (let i = i0, j = i1 - 1; i < i1; j = i++) { cx += t.foot[2 * i]; cz += t.foot[2 * i + 1]; a += t.foot[2 * j] * t.foot[2 * i + 1] - t.foot[2 * i] * t.foot[2 * j + 1]; }
      cx /= (i1 - i0); cz /= (i1 - i0);
      for (let i = i0; i < i1; i++) rad = Math.max(rad, Math.hypot(t.foot[2 * i] - cx, t.foot[2 * i + 1] - cz));
      B.push({ cx, cz, area: Math.abs(a / 2), rad });
    }
    const order = B.map((_, i) => i).sort((a, b) => B[b].area - B[a].area), done = new Set();
    for (const bi of order) {
      if ((bi & 15) === 15) yield;
      const A = B[bi];
      if (done.has(bi) || !inside(A.cx, A.cz) || urbanAt(A.cx, A.cz)) continue;
      const wx0 = t.ox + A.cx, wz0 = t.oz + A.cz;
      let crop = 0, past = 0;
      for (const [dx, dz] of [[100, 0], [-100, 0], [0, 100], [0, -100], [70, 70], [-70, -70], [70, -70], [-70, 70]]) { const l = landAt(wx0 + dx, wz0 + dz); if (isCrop(l)) crop++; else if (isPasture(l)) past++; }
      if (crop + past < 5) continue;
      const members = B.map((m, i) => i).filter(i => !done.has(i) && Math.hypot(B[i].cx - A.cx, B[i].cz - A.cz) < 90);
      members.forEach(i => done.add(i));
      if (members.length < 2 && A.area < 150) continue;
      let ccx = 0, ccz = 0, crad = 0;
      for (const i of members) { ccx += B[i].cx; ccz += B[i].cz; }
      ccx /= members.length; ccz /= members.length;
      for (const i of members) crad = Math.max(crad, Math.hypot(B[i].cx - ccx, B[i].cz - ccz) + B[i].rad);
      const fx = t.ox + ccx, fz = t.oz + ccz, rr = k => rnd(ccx, ccz, 200 + k);
      const spot = (r, minD, maxD, tries = 24) => {
        for (let k = 0; k < tries; k++) {
          const a = rr(300 + k * 2) * 6.283, d = crad + minD + rr(301 + k * 2) * (maxD - minD), x = fx + Math.cos(a) * d, z = fz + Math.sin(a) * d;
          if (free(x, z, r)) return [x, z, a];
        }
        return null;
      };
      const brand = BRANDS[Math.floor(rr(1) * BRANDS.length)];
      // grain bins in a row (big on North American / Australian crop farms)
      const binny = REGION.id === 'us' || REGION.id === 'ca' || REGION.id === 'au' || REGION.id === 'latam';
      const nBins = crop >= 3 ? (binny ? 2 + Math.floor(rr(2) * 5) : (rr(2) < 0.4 ? 1 + Math.floor(rr(3) * 2) : 0)) : 0;
      if (nBins) {
        const D = 6 + rr(4) * 5, Hh = D * (0.85 + rr(5) * 0.35), st = spot(D, 10, 45);
        if (st) {
          const dir = rr(6) * 6.283, ux = Math.cos(dir), uz = Math.sin(dir);
          for (let k = 0; k < nBins; k++) {
            const x = st[0] + ux * k * (D + 1.6), z = st[1] + uz * k * (D + 1.6), sz = k === 0 ? 1 : 0.85 + rr(10 + k) * 0.3;
            if (!free(x, z, D * sz / 2 + 0.4)) continue;
            const y = footY(x, z, D * sz / 2);
            add('bin', x, y - 0.1, z, yawQ(rr(20 + k) * 6.283), D * sz / 2, Hh * sz, D * sz / 2);
            col.c.push(x, z, D * sz / 2, -1); occupy(x, z, D * sz / 2 + 0.6);
          }
        }
      }
      // tall stave silo on livestock farms
      if (past >= 3 && rr(7) < 0.45) {
        const st = spot(3.5, 5, 30);
        if (st) {
          const r0 = 2.6 + rr(8) * 1.2, y = footY(st[0], st[1], r0), hgt = 15 + rr(9) * 10;
          add('silo', st[0], y - 0.1, st[1], yawQ(0), r0, hgt, r0, lin(rr(10) < 0.35 ? [0.13, 0.2, 0.42] : [0.72, 0.7, 0.66]));
          col.c.push(st[0], st[1], r0, -1); occupy(st[0], st[1], r0 + 1);
        }
      }
      // machinery and vehicles parked around the yard
      const kinds = ['tractor'];
      if (crop >= 3) kinds.push('combine', 'graincart', 'tractor');
      if (past >= 3) kinds.push('tractor', 'sqbale');
      const nEq = 1 + Math.floor(rr(11) * 4);
      for (let k = 0; k < nEq; k++) {
        const kind = kinds[Math.floor(rr(12 + k) * kinds.length)], rad = kind === 'combine' ? 5 : kind === 'pickup' ? 3 : 2.8;
        const st = spot(rad, 4, 35);
        if (!st) continue;
        const y = terrainHeight(st[0], st[1]);
        const colr = kind === 'pickup' ? CAR[Math.floor(rr(30 + k) * CAR.length)] : kind === 'sqbale' ? [0.8, 0.68, 0.42] : brand;
        add(kind, st[0], y, st[1], yawQ(rr(40 + k) * 6.283), 1, 1, 1, lin(colr));
        col.c.push(st[0], st[1], rad * 0.75, -1); occupy(st[0], st[1], rad);
      }
      if (rr(13) < 0.7) {
        const st = spot(2, 4, 25);
        if (st) { const y = terrainHeight(st[0], st[1]); add('fueltank', st[0], y, st[1], yawQ(rr(14) * 6.283), 1, 1, 1, lin([[0.75, 0.1, 0.08], [0.9, 0.9, 0.88], [0.6, 0.62, 0.64]][Math.floor(rr(15) * 3)])); col.c.push(st[0], st[1], 1.2, -1); occupy(st[0], st[1], 2); }
      }
      // corral with a stock tank and a few animals on livestock farms
      if (past >= 3 && rr(16) < 0.6) {
        const st = spot(14, 8, 40);
        if (st) {
          const a = rr(17) * 6.283, ux = Math.cos(a), uz = Math.sin(a), hwc = 12, hdc = 9;
          const corners = [[-hwc, -hdc], [hwc, -hdc], [hwc, hdc], [-hwc, hdc]].map(([u, v]) => [st[0] + ux * u - uz * v, st[1] + uz * u + ux * v]);
          for (let e = 0; e < 4; e++) {
            const [ax, az] = corners[e], [bx, bz] = corners[(e + 1) % 4], len = Math.hypot(bx - ax, bz - az), nP = Math.ceil(len / 2.8);
            let prev = null;
            for (let q = 0; q <= nP; q++) {
              if (e === 0 && q >= Math.floor(nP / 2) - 1 && q <= Math.floor(nP / 2) + 1) { prev = null; continue; } // opening
              const x = ax + (bx - ax) * q / nP, z = az + (bz - az) * q / nP, y = terrainHeight(x, z);
              if (y === null || !clear(x - t.ox, z - t.oz, 0.3) || pushOutOfBuildings(x, z, 0.2)) { prev = null; continue; }
              add('rpost', x, y, z, yawQ(Math.atan2(bx - ax, bz - az)), 1, 1, 1, lin([0.48, 0.38, 0.27]));
              if (prev) {
                _va.set(x - prev[0], y - prev[1], z - prev[2]); const l = _va.length(); _va.normalize();
                const q2 = _qb.setFromUnitVectors(_zAxis, _va);
                for (const hh of [0.45, 0.85, 1.2]) add('frail', (x + prev[0]) / 2, (y + prev[1]) / 2 + hh, (z + prev[2]) / 2, q2, 1, 1, l, lin([0.48, 0.38, 0.27]));
                col.s.push(prev[0], prev[2], x, z);
              }
              prev = [x, y, z];
            }
          }
          occupy(st[0], st[1], 15);
          add('stocktank', st[0] + ux * 6, terrainHeight(st[0] + ux * 6, st[1] + uz * 6), st[1] + uz * 6, yawQ(0), 1, 1, 1);
          const nA = 2 + Math.floor(rr(18) * 5);
          for (let k = 0; k < nA; k++) {
            const u = (rr(50 + k) - 0.5) * 18, v = (rr(60 + k) - 0.5) * 12, x = st[0] + ux * u - uz * v, z = st[1] + uz * u + ux * v;
            const horse = rr(19) < 0.35, y = terrainHeight(x, z);
            if (y === null) continue;
            add(horse ? 'horse' : 'cow', x, y, z, yawQ(rr(70 + k) * 6.283), 1, 1, 1, lin(horse ? [[0.35, 0.22, 0.12], [0.15, 0.1, 0.07], [0.6, 0.45, 0.3], [0.85, 0.82, 0.78]][Math.floor(rr(80 + k) * 4)] : [[0.1, 0.09, 0.08], [0.45, 0.22, 0.12], [0.85, 0.8, 0.7]][Math.floor(rr(80 + k) * 3)]));
          }
        }
      }
      // round-bale stack near the barn
      if (past + crop >= 5 && rr(20) < 0.55) {
        const st = spot(6, 6, 40);
        if (st) {
          const a = rr(21) * 6.283, ux = Math.cos(a), uz = Math.sin(a), wrapped = rr(22) < 0.3;
          const n = 4 + Math.floor(rr(23) * 7);
          for (let k = 0; k < n; k++) {
            const row = k < Math.ceil(n * 0.6) ? 0 : 1, idx = row ? k - Math.ceil(n * 0.6) : k;
            const x = st[0] + ux * (idx * 1.25 + row * 0.6), z = st[1] + uz * (idx * 1.25 + row * 0.6), y = terrainHeight(x, z);
            const shade = 0.88 + rr(90 + k) * 0.2;
            add('bale', x, y + row * 1.2, z, yawQ(a + Math.PI / 2), 1, 1, 1, lin(wrapped ? [0.92 * shade, 0.92 * shade, 0.9 * shade] : [0.78 * shade, 0.65 * shade, 0.38 * shade]));
          }
          col.c.push(st[0] + ux * n * 0.35, st[1] + uz * n * 0.35, n * 0.5, -1); occupy(st[0], st[1], 7);
        }
      }
    }
  }
  // hay fields: rows of round bales; pastures: grazing herds
  for (let gz = 14; gz < S; gz += 28) for (let gx = 14; gx < S; gx += 28) {
    const x = gx + (rnd(gx, gz, 120) - 0.5) * 10, z = gz + (rnd(gx, gz, 121) - 0.5) * 10, rv = rnd(gx, gz, 122);
    if (rv > 0.3) continue;
    const wx = t.ox + x, wz = t.oz + z, l = landAt(wx, wz);
    if (!l || urbanAt(x, z)) continue;
    const hay = l.clsKind === 'cdl' && (l.code === 36 || l.code === 37 || l.code === 58);
    if (!(hay ? rv < 0.3 : l.mat === L_GRASS && l.ndvi < 0.3 && rv < 0.012)) continue;
    if (!free(wx, wz, 1)) continue;
    const y = terrainHeight(wx, wz), cellYaw = hash(Math.floor(wx / 400), Math.floor(wz / 400)) < 0.5 ? 0 : Math.PI / 2, shade = 0.85 + rnd(x, z, 123) * 0.25;
    add('bale', wx, y, wz, yawQ(cellYaw + (rnd(x, z, 124) - 0.5) * 0.3), 1, 1, 1, lin(rnd(x, z, 125) < 0.2 ? [0.9 * shade, 0.9 * shade, 0.88 * shade] : [0.76 * shade, 0.64 * shade, 0.36 * shade]));
    col.c.push(wx, wz, 0.75, -1);
  }
  for (let gz = 60; gz < S; gz += 150) for (let gx = 60; gx < S; gx += 150) {
    const x = gx + (rnd(gx, gz, 130) - 0.5) * 80, z = gz + (rnd(gx, gz, 131) - 0.5) * 80;
    if (rnd(gx, gz, 132) > 0.22) continue;
    const wx = t.ox + x, wz = t.oz + z, l = landAt(wx, wz);
    if (!l || l.mat !== L_GRASS || urbanAt(x, z) || (l.clsKind === 'cdl' && l.code !== 176 && l.code !== 37 && l.code !== 62)) continue;
    const horses = rnd(x, z, 133) < 0.15, n = horses ? 2 + Math.floor(rnd(x, z, 134) * 5) : 4 + Math.floor(rnd(x, z, 134) * 13);
    const breed = rnd(x, z, 135), herdDir = rnd(x, z, 136) * 6.283;
    for (let k = 0; k < n; k++) {
      const a = rnd(x, z, 140 + k) * 6.283, d = Math.sqrt(rnd(x, z, 160 + k)) * 32, px = wx + Math.cos(a) * d, pz = wz + Math.sin(a) * d;
      const lk = landAt(px, pz);
      if (!lk || lk.mat !== L_GRASS || !free(px, pz, 1.2)) continue;
      const y = terrainHeight(px, pz), yaw = herdDir + (rnd(x, z, 180 + k) - 0.5) * 2.2;
      if (horses) add('horse', px, y, pz, yawQ(yaw), 1, 1, 1, lin([[0.35, 0.22, 0.12], [0.15, 0.1, 0.07], [0.6, 0.45, 0.3], [0.85, 0.82, 0.78]][Math.floor(rnd(x, z, 200 + k) * 4)]));
      else if (breed < 0.2 && REGION.id === 'us') add('holstein', px, y, pz, yawQ(yaw), 1, 1, 1);
      else add('cow', px, y, pz, yawQ(yaw), 1, 1, 1, lin(breed < 0.55 ? [0.09, 0.08, 0.08] : breed < 0.8 ? [0.45, 0.22, 0.12] : [0.86, 0.8, 0.7]));
      col.c.push(px, pz, 0.8, -1); occupy(px, pz, 1.5);
    }
  }
  // centre-pivot irrigation on circular fields (crop inside a circle but not in the corners around it)
  const pivots = [];
  for (let gz = 300; gz < S - 300; gz += 100) for (let gx = 300; gx < S - 300; gx += 100) {
    const wx = t.ox + gx, wz = t.oz + gz;
    if (!isCrop(landAt(wx, wz)) || pivots.some(([px, pz]) => Math.hypot(px - wx, pz - wz) < 700)) continue;
    let ring = 0, corners = 0;
    for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4; if (isCrop(landAt(wx + Math.cos(a) * 300, wz + Math.sin(a) * 300))) ring++; }
    for (const [dx, dz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) if (!isCrop(landAt(wx + dx * 330, wz + dz * 330))) corners++;
    if (ring < 8 || corners < 2) continue;
    pivots.push([wx, wz]);
    const y0 = terrainHeight(wx, wz);
    add('pivotcentre', wx, y0, wz, yawQ(0), 1, 1, 1);
    col.c.push(wx, wz, 1.2, -1);
    const ang = rnd(gx, gz, 210) * 6.283, ux = Math.sin(ang), uz = Math.cos(ang);
    let ay = y0;
    for (let k = 0; k < 7; k++) {
      const sx = wx + ux * k * 50, sz = wz + uz * k * 50, ex = sx + ux * 50, ez = sz + uz * 50, ey = terrainHeight(ex, ez);
      if (ey === null || !clear(ex - t.ox, ez - t.oz, 2) || !clear((sx + ex) / 2 - t.ox, (sz + ez) / 2 - t.oz, 1)) break;
      _va.set(ex - sx, ey - ay, ez - sz).normalize();
      add('pivotspan', sx, ay, sz, _qb.setFromUnitVectors(_zAxis, _va), 1, 1, 1);
      col.c.push(ex, ez, 1.4, -1);
      ay = ey;
    }
  }

  mark('farms');
  // boulders and stones across the landscape: scree below steep ground, bare ground, river banks, forests
  for (let gz = 0; gz < S; gz += 8) for (let gx = 0; gx < S; gx += 8) {
    const x = gx + rnd(gx, gz, 80) * 8, z = gz + rnd(gx, gz, 81) * 8, rv = rnd(x, z, 82);
    if (rv > 0.2) continue;
    const wx = t.ox + x, wz = t.oz + z, h = terrainHeight(wx, wz);
    if (h === null) continue;
    const hx = terrainHeight(wx + 3, wz), hz = terrainHeight(wx, wz + 3);
    if (hx === null || hz === null) continue;
    const steep = Math.hypot(hx - h, hz - h) / 3, l = landAt(wx, wz), m = l ? l.mat : L_GRASS;
    let prob = 0, smin = 0.3, smax = 1.2, tint = null;
    if (steep > 0.7) { prob = 0.1; smin = 0.4; smax = 2.4; }
    else if (steep > 0.3) {
      // talus: rocks collect below cliffs
      const ux = -(hx - h), uz = -(hz - h), ul = Math.hypot(ux, uz) || 1, upH = terrainHeight(wx - ux / ul * 14, wz - uz / ul * 14);
      if (upH !== null && upH - h > 9) { prob = 0.18; smin = 0.3; smax = 1.8; }
    }
    if (!prob && m === L_GRAVEL) { prob = 0.06; smin = 0.25; smax = 1.4; }
    if (!prob && m === L_SOIL && !urbanAt(x, z)) { prob = 0.012; }
    if (!prob && m === L_FOREST) { prob = 0.015; smin = 0.5; smax = 2.2; tint = [0.82, 0.95, 0.72]; }
    if (!prob && m === L_GRASS && !urbanAt(x, z)) { prob = 0.004; smin = 0.5; smax = 1.5; }
    if (!prob && (waterLevelAt(wx + 7, wz) !== null || waterLevelAt(wx - 7, wz) !== null) && waterLevelAt(wx, wz) === null) { prob = 0.1; smin = 0.15; smax = 0.6; }
    if (rv > prob) continue;
    const g = groundAt(wx, wz);
    if (!g || g.seg >= 0 || blocked(wx, wz, 1.5)) continue;
    const sz = smin * Math.pow(smax / smin, Math.pow(rnd(x, z, 83), 2.2));
    let c0 = l ? [l.r / 255, l.g / 255, l.b / 255] : [0.5, 0.5, 0.5];
    const grey = (c0[0] + c0[1] + c0[2]) / 3;
    c0 = [grey * 0.6 + c0[0] * 0.4 + 0.08, grey * 0.6 + c0[1] * 0.4 + 0.08, grey * 0.6 + c0[2] * 0.4 + 0.08];
    if (tint) c0 = [c0[0] * tint[0], c0[1] * tint[1], c0[2] * tint[2]];
    const k = Math.floor(rnd(x, z, 84) * 6);
    add('rock' + k, wx, h - sz * 0.28, wz, yawQ(rnd(x, z, 85) * 6.28, (rnd(x, z, 86) - 0.5) * 0.5, (rnd(x, z, 87) - 0.5) * 0.5),
        sz * (0.85 + rnd(x, z, 88) * 0.4), sz * (0.7 + rnd(x, z, 89) * 0.4), sz * (0.85 + rnd(x, z, 90) * 0.4), lin(c0));
    if (sz > 0.7) col.c.push(wx, wz, sz * 0.85, -1);
    // small rocks cluster around big ones
    if (sz > 1.0) for (let q = 0; q < 3; q++) {
      const a = rnd(x, z, 91 + q) * 6.28, dd = sz * (1.2 + rnd(x, z, 95 + q)), px = wx + Math.cos(a) * dd, pz = wz + Math.sin(a) * dd, py = terrainHeight(px, pz);
      const s2 = sz * (0.15 + rnd(x, z, 99 + q) * 0.25);
      if (py === null || !clear(px - t.ox, pz - t.oz, s2)) continue;
      add('rock' + ((k + q + 1) % 6), px, py - s2 * 0.3, pz, yawQ(a * 3), s2, s2 * 0.7, s2, lin(c0));
    }
  }

  mark('rocks');
  // cacti, only where they grow wild - the deserts and dry scrub of the Americas: saguaro in the Sonoran Desert,
  // prickly pear and barrel cactus across the US Southwest and Mexico, columnar cacti in Mexico, the Atacama/Andes,
  // Argentina's Monte and Brazil's caatinga. Bare ground, sand or sparse scrub; not in towns, on roads or cliffs.
  {
    const [tla, tlo] = toLatLon(t.ox + S / 2, t.oz + S / 2);
    const zone = (a, b, c, d) => tla >= a && tla <= b && tlo >= c && tlo <= d;
    const sonoran = zone(26, 34.6, -116, -108.4), southwest = zone(25, 37.6, -118.5, -100), mexico = zone(15, 26, -111, -94);
    const southAm = zone(-32, -15, -72, -64) || zone(-40, -22, -70, -62) || zone(-16, -3, -45, -35);
    if (sonoran || southwest || mexico || southAm) for (let gz = 0; gz < S; gz += 9) for (let gx = 0; gx < S; gx += 9) {
      const x = gx + rnd(gx, gz, 130) * 9, z = gz + rnd(gx, gz, 131) * 9;
      if (rnd(x, z, 132) > 0.045) continue;
      const wx = t.ox + x, wz = t.oz + z, l = landAt(wx, wz);
      if (!l) continue;
      const m = l.mat, dry = m === L_SOIL || m === L_GRAVEL || m === L_SAND || (m === L_GRASS && l.ndvi < 0.22);
      if (!dry || urbanAt(x, z)) continue;
      const h = terrainHeight(wx, wz);
      if (h === null) continue;
      const hx = terrainHeight(wx + 3, wz), hz = terrainHeight(wx, wz + 3);
      if (hx === null || hz === null || Math.hypot(hx - h, hz - h) / 3 > 0.6) continue;
      const g = groundAt(wx, wz);
      if (!g || g.seg >= 0 || blocked(wx, wz, 1.4)) continue;
      const pick = rnd(x, z, 133), v = rnd(x, z, 134);
      let fam, sc, solid = 0;
      if (sonoran && pick < 0.4) { fam = 'saguaro' + Math.floor(v * 3); sc = 0.55 + rnd(x, z, 135) * 0.85; solid = 0.32 * sc; }
      else if ((sonoran || mexico || southAm) && pick < 0.62) { fam = 'column' + Math.floor(v * 2); sc = 0.7 + rnd(x, z, 135) * 0.7; solid = 0.45 * sc; }
      else if (pick < 0.86) { fam = 'pear' + Math.floor(v * 2); sc = 0.7 + rnd(x, z, 135) * 0.9; }
      else { fam = 'barrel'; sc = 0.6 + rnd(x, z, 135) * 0.7; }
      add(fam, wx, h - 0.05, wz, yawQ(rnd(x, z, 136) * 6.28), sc, sc, sc);
      if (solid) col.c.push(wx, wz, solid, -1);
    }
  }
  mark('cacti');
  // stop / give-way signs where a lesser road meets a bigger one
  const ST = t.stops || [];
  for (let k = 0; k < ST.length; k += 8) {
    const [x, z, dx, dz, hw, shw, rank, srank] = ST.slice(k, k + 8);
    const ex = x - dx * (shw + 2.5), ez = z - dz * (shw + 2.5);
    const side = REGION.left ? -1 : 1, sx = ex - dz * side * (hw + 1.2), sz = ez + dx * side * (hw + 1.2);
    if (!inside(sx, sz) || rank < 3) continue;
    if (osm && [...osm.stops, ...osm.gives].some((v, q, a) => q % 2 === 0 && Math.hypot(v - x, a[q + 1] - z) < 25)) continue;
    const stop = srank >= 0 && (srank - rank >= 2 || srank >= 7 || rnd(x, z, 51) < (REGION.id === 'us' ? 0.8 : 0.45));
    sign([stop ? SIGNS.stop() : SIGNS.yield()], sx, sz, -dx, -dz, stop ? 0.78 : 0.85);
  }

  // level crossings: crossbucks on each road approach, flashing-light gates on bigger roads
  if (t.buckets) {
    const railIdx = CLASS_LIST.indexOf('rail'), seen = new Set();
    for (let b = 0; b < t.buckets.length; b++) {
      const list = t.buckets[b];
      for (const o of list) {
        if (s[o + 7] !== railIdx || s[o + 8]) continue;
        for (const q of list) {
          if (s[q + 7] === railIdx || s[q + 8] || s[q + 6] < 3) continue;
          const ax = s[o], az = s[o + 1], bx = s[o + 2], bz = s[o + 3], cx = s[q], cz = s[q + 1], dx = s[q + 2], dz = s[q + 3];
          const den = (bx - ax) * (dz - cz) - (bz - az) * (dx - cx);
          if (Math.abs(den) < 1e-9) continue;
          const u = ((cx - ax) * (dz - cz) - (cz - az) * (dx - cx)) / den, v = ((cx - ax) * (bz - az) - (cz - az) * (bx - ax)) / den;
          if (u < 0 || u > 1 || v < 0 || v > 1) continue;
          const X = ax + (bx - ax) * u, Z = az + (bz - az) * u, key = `${Math.round(X / 8)},${Math.round(Z / 8)}`;
          if (seen.has(key) || !inside(X, Z)) continue;
          seen.add(key);
          const rl = Math.hypot(dx - cx, dz - cz) || 1, rdx = (dx - cx) / rl, rdz = (dz - cz) / rl, hw = s[q + 4];
          const ks = REGION.left ? -1 : 1;
          for (const dir of s[q + 11] > 0 ? [1, -1] : [1]) {
            const fx = rdx * dir, fz = rdz * dir, back = 2.2 + 5;
            const px = X - fx * back - fz * ks * (hw + 1.3), pz = Z - fz * back + fx * ks * (hw + 1.3);
            if (!inside(px, pz)) continue;
            sign([SIGNS.crossbuck()], px, pz, -fx, -fz, 1.25);
            if (s[q + 6] >= 5) {
              const gx = X - fx * (back - 1.5) - fz * ks * (hw + 2.3), gz = Z - fz * (back - 1.5) + fx * ks * (hw + 2.3), wx = t.ox + gx, wz = t.oz + gz, y = terrainHeight(wx, wz);
              if (y !== null && clear(gx, gz, 0.4)) { add('xgate', wx, y, wz, yawQ(Math.atan2(-fx, -fz)), 1, 1, 1); col.c.push(wx, wz, 0.3, -1); }
            }
          }
        }
      }
    }
  }

  // ---- real OpenStreetMap features from Overpass (when they have arrived for this tile)
  if (osm) {
    const ks = REGION.left ? -1 : 1;
    const nearestSeg = (x, z, maxD = 25) => {
      let best = null;
      const bi = Math.floor(x / BCELL), bj = Math.floor(z / BCELL);
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        if (bi + di < 0 || bj + dj < 0 || bi + di >= BUCKETS || bj + dj >= BUCKETS) continue;
        for (const o of t.buckets[(bj + dj) * BUCKETS + bi + di]) {
          if (s[o + 6] < 3 || s[o + 7] === CLASS_LIST.indexOf('rail')) continue;
          const [d, u] = segDist(s, o, x, z);
          if (d < maxD && (!best || d < best.d)) best = { o, d, u };
        }
      }
      if (!best) return null;
      const o = best.o, l = Math.hypot(s[o + 2] - s[o], s[o + 3] - s[o + 1]) || 1;
      return { d: best.d, x: s[o] + (s[o + 2] - s[o]) * best.u, z: s[o + 1] + (s[o + 3] - s[o + 1]) * best.u, h: s[o + 9] + (s[o + 10] - s[o + 9]) * best.u,
               tx: (s[o + 2] - s[o]) / l, tz: (s[o + 3] - s[o + 1]) / l, hw: s[o + 4], two: s[o + 11] > 0 };
    };
    // stop / give-way nodes: sign on the driving side, facing traffic heading to the nearest junction
    for (const [list, isStop] of [[osm.stops, true], [osm.gives, false]]) for (let k = 0; k < list.length; k += 2) {
      const x = list[k], z = list[k + 1];
      if (!inside(x, z)) continue;
      const ns = nearestSeg(x, z, 15);
      if (!ns) continue;
      let jd = Infinity, jx = 0, jz = 0;
      for (let q = 0; q < J.length; q += 3) { const d = Math.hypot(J[q] - x, J[q + 1] - z); if (d < jd) { jd = d; jx = J[q]; jz = J[q + 1]; } }
      let ax = ns.tx, az = ns.tz;
      if ((jx - x) * ax + (jz - z) * az < 0) { ax = -ax; az = -az; }
      const px = ns.x - ax * 1.0 - az * ks * (ns.hw + 1.2), pz = ns.z - az * 1.0 + ax * ks * (ns.hw + 1.2);
      sign([isStop ? SIGNS.stop() : SIGNS.yield()], px, pz, -ax, -az, isStop ? 0.78 : 0.85);
    }
    // mapped street lamps (procedural ones are skipped when a tile has enough of these)
    if (osm.lamps.length >= 12) for (let k = 0; k < osm.lamps.length; k += 2) {
      let x = osm.lamps[k], z = osm.lamps[k + 1];
      if (!inside(x, z)) continue;
      const ns = nearestSeg(x, z, 30);
      let yaw = 0;
      if (ns) {
        let nx = x - ns.x, nz = z - ns.z; const nl = Math.hypot(nx, nz) || 1; nx /= nl; nz /= nl;
        if (nl < ns.hw + 0.6) { x = ns.x + nx * (ns.hw + 0.7); z = ns.z + nz * (ns.hw + 0.7); }
        yaw = Math.atan2(-nx, -nz);
      }
      const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz);
      if (y === null || blocked(wx, wz, 0.3)) continue;
      add('lamp', wx, y - 0.05, wz, yawQ(yaw), 1, 1, 1);
      col.c.push(wx, wz, 0.2, -1);
    }
    // mapped hydrants
    for (let k = 0; k < osm.hydrants.length; k += 2) {
      const x = osm.hydrants[k], z = osm.hydrants[k + 1];
      if (!inside(x, z)) continue;
      const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz);
      if (y === null) continue;
      const kid = nextKid++;
      add('hydrant', wx, y, wz, yawQ(0), 1, 1, 1, lin(REGION.id === 'uk' || REGION.id === 'eu' ? [0.85, 0.75, 0.1] : [0.75, 0.1, 0.08]), 0, kid);
      col.c.push(wx, wz, 0.15, kid);
    }
    // mapped pedestrian crossings: zebra paint across the road at the node
    for (let k = 0; k < osm.crossings.length; k += 2) {
      const x = osm.crossings[k], z = osm.crossings[k + 1];
      if (!inside(x, z)) continue;
      const ns = nearestSeg(x, z, 4);
      if (ns && ns.d < ns.hw) zebra(ns.x, ns.z, ns.h, ns.tx, ns.tz, ns.hw);
    }
    // mapped bus stops: sign at the kerb, shelter behind it at about half of them
    for (let k = 0; k < osm.bus.length; k += 2) {
      const x = osm.bus[k], z = osm.bus[k + 1];
      if (!inside(x, z)) continue;
      const ns = nearestSeg(x, z, 30);
      if (!ns) continue;
      let nx = x - ns.x, nz = z - ns.z; const nl = Math.hypot(nx, nz) || 1; nx /= nl; nz /= nl;
      const sx = ns.x + nx * (ns.hw + 1.0), sz = ns.z + nz * (ns.hw + 1.0), wx = t.ox + sx, wz = t.oz + sz, y = terrainHeight(wx, wz);
      if (y === null || blocked(wx, wz, 0.3)) continue;
      const kid = nextKid++;
      add('busstop', wx, Math.max(y, ns.h), wz, yawQ(Math.atan2(ns.tx, ns.tz)), 1, 1, 1, [1, 1, 1], 0, kid);
      col.c.push(wx, wz, 0.12, kid);
      if (rnd(x, z, 270) < 0.5) {
        const hx = ns.x + nx * (ns.hw + 3.0), hz = ns.z + nz * (ns.hw + 3.0), hwx = t.ox + hx + ns.tx * 2.5, hwz = t.oz + hz + ns.tz * 2.5, hy = terrainHeight(hwx, hwz);
        if (hy !== null && !blocked(hwx, hwz, 1.8)) { add('shelter', hwx, hy, hwz, yawQ(Math.atan2(-nx, -nz)), 1, 1, 1); col.c.push(hwx, hwz, 1.4, -1); }
      }
    }
    // high-voltage lines with lattice pylons (wood poles for minor lines), sagging conductors between them
    for (const ln of osm.lines) {
      const P = ln.pts, n = P.length / 2, big = !ln.minor && (ln.voltage === 0 || ln.voltage >= 60000);
      const H = ln.voltage >= 300000 ? 45 : ln.voltage >= 200000 ? 36 : ln.voltage >= 100000 ? 30 : 26, sc = H / 30;
      const att = [];
      for (let i = 0; i < n; i++) {
        const x = P[2 * i], z = P[2 * i + 1], a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
        const dx = P[2 * b] - P[2 * a], dz = P[2 * b + 1] - P[2 * a + 1], yaw = Math.atan2(dx, dz);
        const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz);
        if (y === null) { att.push(null); continue; }
        const ax = Math.cos(yaw), az = -Math.sin(yaw);
        if (big) {
          if (inside(x, z) && clear(x, z, 3.5 * sc)) { add('pylon', wx, y - 0.3, wz, yawQ(yaw), sc, sc, sc); col.c.push(wx, wz, 3.3 * sc, -1); }
          att.push(PYLON_ATTACH.map(([u, v]) => [wx + ax * u * sc, y + v * sc, wz + az * u * sc]));
        } else {
          if (inside(x, z) && !blocked(wx, wz, 0.4)) { add('poleA', wx, y - 0.3, wz, yawQ(yaw), 1, 1.05, 1); col.c.push(wx, wz, 0.25, -1); }
          att.push(ATTACH.poleA.map(([u, v]) => [wx + ax * u, y - 0.3 + v * 1.05, wz + az * u]));
        }
      }
      for (let i = 0; i < n - 1; i++) {
        if (!inside(P[2 * i], P[2 * i + 1]) || !att[i] || !att[i + 1]) continue;
        const A0 = att[i], B0 = att[i + 1], segN = big ? 10 : 6;
        for (let k = 0; k < A0.length; k++) {
          const A = A0[k], B = B0[k], span = Math.hypot(B[0] - A[0], B[2] - A[2]), sag = span * span / (big ? 14000 : 3000);
          let last = A;
          for (let q = 1; q <= segN; q++) {
            const f = q / segN, Pq = [A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f - 4 * sag * f * (1 - f), A[2] + (B[2] - A[2]) * f];
            _va.set(Pq[0] - last[0], Pq[1] - last[1], Pq[2] - last[2]); const l = _va.length();
            add(big ? 'hvwire' : 'wire', (Pq[0] + last[0]) / 2, (Pq[1] + last[1]) / 2, (Pq[2] + last[2]) / 2, _qb.setFromUnitVectors(_zAxis, _va.normalize()), 1, 1, l);
            last = Pq;
          }
        }
      }
    }
    // water towers, silos, tanks, masts
    const place = (list, fn) => { for (let k = 0; k < list.length; k += 4) { const x = list[k], z = list[k + 1]; if (!inside(x, z)) continue; const wx = t.ox + x, wz = t.oz + z, y = terrainHeight(wx, wz); if (y !== null) fn(wx, y, wz, list[k + 2], list[k + 3]); } };
    place(osm.wtowers, (wx, y, wz, r, h) => { const s2 = h > 10 ? h / 40 : 1; add('wtower', wx, y, wz, yawQ(0), s2, s2, s2, lin([[0.92, 0.93, 0.94], [0.72, 0.84, 0.92], [0.8, 0.82, 0.8]][Math.floor(rnd(wx, wz, 300) * 3)])); col.c.push(wx, wz, 1.6 * s2, -1); });
    place(osm.silos, (wx, y, wz, r, h) => { const rr = r > 1 ? r : 3; y = Math.min(y, footY(wx, wz, rr)); add('silo', wx, y - 0.1, wz, yawQ(0), rr, h > 3 ? h : 18, rr, lin([0.72, 0.7, 0.66])); col.c.push(wx, wz, rr, -1); });
    place(osm.tanks, (wx, y, wz, r, h) => { const rr = r > 1 ? r : 6; add('tank', wx, y - 0.05, wz, yawQ(0), rr, h > 2 ? h : rr * 0.9, rr, lin([[0.9, 0.9, 0.88], [0.7, 0.72, 0.72], [0.85, 0.82, 0.72]][Math.floor(rnd(wx, wz, 301) * 3)])); col.c.push(wx, wz, rr, -1); });
    place(osm.masts, (wx, y, wz, r, h) => { const s2 = h > 10 ? h / 60 : 0.7; add('mast', wx, y, wz, yawQ(0), 1, s2, 1); col.c.push(wx, wz, 0.5, -1); });
  }

  if ((REGION.id === 'us' || REGION.id === 'ca') && t.footOff && t.footOff.length > 300 && !(osm && osm.wtowers.length)) {
    let cx = 0, cz = 0; const nb = t.footOff.length - 1;
    for (let b = 0; b < nb; b += 7) { cx += t.foot[2 * t.footOff[b]]; cz += t.foot[2 * t.footOff[b] + 1]; }
    cx /= Math.ceil(nb / 7); cz /= Math.ceil(nb / 7);
    for (let k = 0; k < 30; k++) {
      const a = rnd(cx, cz, 400 + k) * 6.283, d = 80 + rnd(cx, cz, 430 + k) * 400, x = cx + Math.cos(a) * d, z = cz + Math.sin(a) * d;
      if (!inside(x, z)) continue;
      const wx = t.ox + x, wz = t.oz + z, g = groundAt(wx, wz);
      if (!g || g.seg >= 0 || blocked(wx, wz, 8)) continue;
      add('wtower', wx, g.h, wz, yawQ(0), 1, 1, 1, lin([[0.92, 0.93, 0.94], [0.72, 0.84, 0.92]][Math.floor(rnd(x, z, 460) * 2)]));
      col.c.push(wx, wz, 1.6, -1);
      break;
    }
  }

  // driveway mesh (road shader: gravel in the country, asphalt in town)
  if (drive.idx.length && t.roadMats) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(drive.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(drive.nor, 3));
    g.setAttribute('rd', new THREE.Float32BufferAttribute(drive.rd, 4));
    g.setAttribute('rs', new THREE.Float32BufferAttribute(drive.rs, 4));
    g.setIndex(drive.idx);
    if (t.driveMesh) { scene.remove(t.driveMesh); t.driveMesh.geometry.dispose(); }
    t.driveMesh = new THREE.Mesh(g, t.roadMats.road);
    t.driveMesh.position.set(t.ox, 0, t.oz);
    t.driveMesh.receiveShadow = true;
    scene.add(t.driveMesh);
  }
  if (t.walkMesh) { scene.remove(t.walkMesh); t.walkMesh.geometry.dispose(); t.walkMesh = null; }
  if (walk.idx.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(walk.pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(walk.col, 3));
    g.setIndex(walk.idx);
    g.computeVertexNormals();
    t.walkMesh = new THREE.Mesh(g, walkMat);
    t.walkMesh.receiveShadow = true;
    t.walkMesh.position.set(t.ox, 0, t.oz);
    scene.add(t.walkMesh);
  }
  mark('stops+osm+meshes');
  lastGenTimes = times;
  for (const k in out) out[k] = new Float32Array(out[k]);
  t.walkMask = walkMask;
  t.props = out;
  t.propCol = { c: new Float32Array(col.c), s: new Float32Array(col.s) };
}

// ---- near-car rebuild + collisions + knock-overs
let propsDirty = true, propX = 1e9, propZ = 1e9, propCooldown = 0;
const kidMap = new Map();       // knock id -> [[family, index], ...] in the current fill
let colCircles = [], colSegs = [];
const _mm = new THREE.Matrix4(), _pp = new THREE.Vector3(), _qq = new THREE.Quaternion(), _ss = new THREE.Vector3();

function rebuildProps(cx, cz) {
  for (const f of Object.values(FAM)) f.n = 0;
  kidMap.clear();
  const circ = [], segs = [];
  for (const t of tiles.values()) {
    if (!t.props) continue;
    const dx = Math.max(t.ox - cx, 0, cx - t.ox - TILE_SIZE), dz = Math.max(t.oz - cz, 0, cz - t.oz - TILE_SIZE);
    if (dx * dx + dz * dz > 700 * 700) continue;
    for (const name in t.props) {
      const F = FAM[name], A = t.props[name];
      if (!F) continue;
      const r2 = F.dist * F.dist;
      for (let o = 0; o < A.length; o += PSTRIDE) {
        const ex = A[o] - cx, ez = A[o + 2] - cz;
        if (ex * ex + ez * ez > r2 || F.n >= F.max) continue;
        const kid = A[o + 14];
        if (kid > 0 && knocked.has(kid)) continue;
        _mm.compose(_pp.set(A[o], A[o + 1], A[o + 2]), _qq.set(A[o + 3], A[o + 4], A[o + 5], A[o + 6]), _ss.set(A[o + 7], A[o + 8], A[o + 9]));
        F.mesh.setMatrixAt(F.n, _mm);
        F.mesh.setColorAt(F.n, _c.setRGB(A[o + 10], A[o + 11], A[o + 12]));
        if (F.uv) { const rc = cellRect(A[o + 13]); F.uv.setXYZW(F.n, rc[0], rc[1], rc[2], rc[3]); }
        if (kid > 0) (kidMap.get(kid) || kidMap.set(kid, []).get(kid)).push([name, F.n, o, t]);
        F.n++;
      }
    }
    const P = t.propCol;
    if (P) {
      for (let o = 0; o < P.c.length; o += 4) if (Math.abs(P.c[o] - cx) < 200 && Math.abs(P.c[o + 1] - cz) < 200 && !knocked.has(P.c[o + 3])) circ.push(P.c[o], P.c[o + 1], P.c[o + 2], P.c[o + 3]);
      for (let o = 0; o < P.s.length; o += 4) if (Math.abs(P.s[o] - cx) < 200 && Math.abs(P.s[o + 1] - cz) < 200) segs.push(P.s[o], P.s[o + 1], P.s[o + 2], P.s[o + 3], NaN);
    }
    const B = t.structCol;
    if (B) for (let o = 0; o < B.length; o += 5) {
      const x0 = t.ox + B[o], z0 = t.oz + B[o + 1];
      if (Math.abs(x0 - cx) < 200 && Math.abs(z0 - cz) < 200) segs.push(x0, z0, t.ox + B[o + 2], t.oz + B[o + 3], B[o + 4]);
    }
  }
  for (const f of Object.values(FAM)) {
    f.mesh.count = f.n;
    f.mesh.instanceMatrix.needsUpdate = true;
    f.mesh.instanceColor.needsUpdate = true;
    if (f.uv) f.uv.needsUpdate = true;
  }
  colCircles = circ; colSegs = segs;
}

const debris = [];
function knock(kid, nx, nz, speed) {
  knocked.add(kid);
  const parts = kidMap.get(kid) || [];
  const vx0 = V.vx, vz0 = V.vz;
  for (const [name, idx, o, t] of parts) {
    const F = FAM[name], A = t.props[name];
    F.mesh.setMatrixAt(idx, _mm.makeScale(0, 0, 0));
    F.mesh.instanceMatrix.needsUpdate = true;
    if (name === 'sback') continue;
    const geo = name === 'sfront' ? new THREE.BoxGeometry(1, 1, 0.03) : F.geo;
    const mat = name === 'sfront' ? metalMat : F.mat;
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(A[o], A[o + 1], A[o + 2]);
    mesh.quaternion.set(A[o + 3], A[o + 4], A[o + 5], A[o + 6]);
    mesh.scale.set(A[o + 7], A[o + 8], A[o + 9]);
    mesh.castShadow = true;
    scene.add(mesh);
    const kick = 0.6 + Math.random() * 0.6;
    debris.push({ mesh, vx: vx0 * kick - nx * 2, vy: 2 + speed * 0.18 + Math.random() * 2, vz: vz0 * kick - nz * 2,
                  ax: (Math.random() - 0.5) * 12, ay: (Math.random() - 0.5) * 8, az: (Math.random() - 0.5) * 12, life: 0, ownGeo: name === 'sfront' });
  }
  while (debris.length > 60) { const d = debris.shift(); scene.remove(d.mesh); if (d.ownGeo) d.mesh.geometry.dispose(); }
}

function updateDebris(dt) {
  for (const d of debris) {
    if (d.rest) continue;
    d.life += dt;
    d.vy -= 9.81 * dt;
    d.mesh.position.x += d.vx * dt; d.mesh.position.y += d.vy * dt; d.mesh.position.z += d.vz * dt;
    _qq.setFromEuler(_eu.set(d.ax * dt, d.ay * dt, d.az * dt));
    d.mesh.quaternion.premultiply(_qq);
    const g = terrainHeight(d.mesh.position.x, d.mesh.position.z);
    if (g !== null && d.mesh.position.y < g + 0.08) {
      d.mesh.position.y = g + 0.08;
      d.vy = Math.abs(d.vy) * 0.3; d.vx *= 0.55; d.vz *= 0.55; d.ax *= 0.5; d.ay *= 0.5; d.az *= 0.5;
      if (Math.hypot(d.vx, d.vy, d.vz) < 0.6) d.rest = true;
    }
  }
}

// posts, signs, lamps and hydrants (circles; the light ones get knocked flying at speed) and fences and railings
// (segments: their nearest point to the car), all against the car's box - see carHitCircle
function collideProps(dt) {
  if (!colCircles.length && !colSegs.length) return;
  if (V.onDeck) return;              // on a footbridge deck: the railings along the banks below don't apply
  const speed = Math.hypot(V.vx, V.vz), reach = CAR.len / 2 + speed * dt + 3;
  for (let o = 0; o < colCircles.length && !V.inTunnel; o += 4) {
    const cx = colCircles[o], cz = colCircles[o + 1];
    if (Math.abs(cx - V.x) > reach || Math.abs(cz - V.z) > reach) continue;
    const kid = colCircles[o + 3];
    if (kid > 0 && speed > 2.5) {
      if (!carHitCircle(cx, cz, colCircles[o + 2], dt, true)) continue;
      const d = Math.hypot(V.x - cx, V.z - cz) || 1;
      knock(kid, (V.x - cx) / d, (V.z - cz) / d, speed); colCircles[o + 3] = -2; colCircles[o] = 1e9; V.vx *= 0.96; V.vz *= 0.96;
      continue;
    }
    carHitCircle(cx, cz, colCircles[o + 2], dt);
  }
  for (let o = 0; o < colSegs.length; o += 5) {
    const ax = colSegs[o], az = colSegs[o + 1], bx = colSegs[o + 2], bz = colSegs[o + 3], sy = colSegs[o + 4];
    if (!Number.isNaN(sy) && (V.y < sy - 1.5 || V.y > sy + 2.5)) continue;   // bridge railings only at deck level
    if (Number.isNaN(sy) && V.inTunnel) continue;                             // fences etc. are up on the surface
    if (Math.min(ax, bx) - reach > V.x || Math.max(ax, bx) + reach < V.x || Math.min(az, bz) - reach > V.z || Math.max(az, bz) + reach < V.z) continue;
    const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz || 1e-9, u = Math.max(0, Math.min(1, ((V.x - ax) * dx + (V.z - az) * dz) / l2));
    carHitCircle(ax + dx * u, az + dz * u, 0.1, dt);
  }
}

// ---- Overpass (live OpenStreetMap) layer: one tile at a time in the background, cached in IndexedDB.
// Real traffic signals, stop signs, street lamps, crossings, bus stops, hydrants, power lines and towers.
const OVERPASS = ['https://overpass-api.de/api/interpreter', 'https://maps.mail.ru/osm/tools/overpass/api/interpreter'];
let osmEp = 0;
const osmQueue = [];
let osmBusy = false, osmBackoff = 0;
const osmDB = new Promise(res => {
  try { const r = indexedDB.open('real-roads-drive', 1); r.onupgradeneeded = () => r.result.createObjectStore('osm'); r.onsuccess = () => res(r.result); r.onerror = () => res(null); }
  catch { res(null); }
});
async function osmCacheGet(key) {
  const db = await osmDB;
  if (!db) return null;
  return new Promise(res => { try { const q = db.transaction('osm').objectStore('osm').get(key); q.onsuccess = () => res(q.result || null); q.onerror = () => res(null); } catch { res(null); } });
}
async function osmCachePut(key, val) { const db = await osmDB; if (db) try { db.transaction('osm', 'readwrite').objectStore('osm').put(val, key); } catch { /* full or blocked */ } }
function osmQuery(tx, ty) {
  const lon = x => x / NT * 360 - 180, lat = y => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / NT))) * 180 / Math.PI;
  const b = `${lat(ty + 1).toFixed(6)},${lon(tx).toFixed(6)},${lat(ty).toFixed(6)},${lon(tx + 1).toFixed(6)}`;
  return `[out:json][timeout:60][bbox:${b}];(node["highway"~"^(traffic_signals|stop|give_way|street_lamp|crossing|bus_stop)$"];`
    + `node["emergency"="fire_hydrant"];node["power"~"^(tower|pole)$"];way["power"~"^(line|minor_line)$"];nwr["man_made"~"^(water_tower|silo|storage_tank|mast)$"];);out geom qt;`;
}
function parseOsm(t, data) {
  const out = { signals: [], stops: [], gives: [], lamps: [], crossings: [], bus: [], hydrants: [], towers: [], poles: [], lines: [], wtowers: [], silos: [], tanks: [], masts: [] };
  const loc = (la, lo) => [(lonToU(lo) - U0) * SCALE - t.ox, (latToV(la) - V0) * SCALE - t.oz];
  for (const e of data.elements || []) {
    const g = e.tags || {};
    if (e.type === 'node') {
      const [x, z] = loc(e.lat, e.lon), h = g.highway;
      if (h === 'traffic_signals') out.signals.push(x, z);
      else if (h === 'stop') out.stops.push(x, z);
      else if (h === 'give_way') out.gives.push(x, z);
      else if (h === 'street_lamp') out.lamps.push(x, z);
      else if (h === 'crossing' && g.crossing !== 'unmarked' && g.crossing !== 'no') out.crossings.push(x, z);
      else if (h === 'bus_stop') out.bus.push(x, z);
      if (g.emergency === 'fire_hydrant' && g['fire_hydrant:type'] !== 'underground') out.hydrants.push(x, z);
      if (g.power === 'tower') out.towers.push(x, z, parseFloat(g.height) || 0);
      if (g.power === 'pole') out.poles.push(x, z);
    }
    const mm = g.man_made;
    if (mm === 'water_tower' || mm === 'silo' || mm === 'storage_tank' || mm === 'mast') {
      let x, z, r = 0;
      if (e.type === 'node') [x, z] = loc(e.lat, e.lon);
      else if (e.geometry) {
        const pts = e.geometry.map(p => loc(p.lat, p.lon));
        x = pts.reduce((a, p) => a + p[0], 0) / pts.length; z = pts.reduce((a, p) => a + p[1], 0) / pts.length;
        r = Math.max(...pts.map(p => Math.hypot(p[0] - x, p[1] - z)));
      }
      if (x === undefined) continue;
      (mm === 'water_tower' ? out.wtowers : mm === 'silo' ? out.silos : mm === 'storage_tank' ? out.tanks : out.masts).push(x, z, r, parseFloat(g.height) || 0);
    }
    if (e.type === 'way' && (g.power === 'line' || g.power === 'minor_line') && e.geometry) {
      const pts = [];
      for (const p of e.geometry) pts.push(...loc(p.lat, p.lon));
      out.lines.push({ pts, minor: g.power === 'minor_line', voltage: parseFloat(g.voltage) || 0 });
    }
  }
  return out;
}
async function pumpOsm() {
  if (osmBusy || !osmQueue.length || performance.now() < osmBackoff) return;
  const dist = t => Math.hypot(t.ox + TILE_SIZE / 2 - V.x, t.oz + TILE_SIZE / 2 - V.z);
  osmQueue.sort((a, b) => dist(a) - dist(b));
  const t = osmQueue.shift(), key = `${t.tx},${t.ty}`;
  if (tiles.get(key) !== t) return;
  osmBusy = true;
  try {
    let data = null;
    const cached = await osmCacheGet(key);
    if (cached && Date.now() - cached.time < 30 * 86400e3) data = cached.data;
    else {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 70000);
      const res = await fetch(OVERPASS[osmEp], { method: 'POST', signal: ctl.signal, body: 'data=' + encodeURIComponent(osmQuery(t.tx, t.ty)), headers: { 'Content-Type': 'application/x-www-form-urlencoded' } });
      clearTimeout(timer);
      if (res.status === 429 || res.status >= 500) { osmEp = (osmEp + 1) % OVERPASS.length; osmBackoff = performance.now() + 8000; t.osmState = null; return; }
      if (!res.ok) throw new Error(res.status);
      data = await res.json();
      osmCachePut(key, { time: Date.now(), data });
    }
    if (tiles.get(key) === t) { t.osm = parseOsm(t, data); t.osmState = 'done'; t.propsGen = false; }
  } catch {
    t.osmState = null; osmEp = (osmEp + 1) % OVERPASS.length; osmBackoff = performance.now() + 8000;
  } finally {
    osmBusy = false;
  }
}

// ---- birds: a few flocks circling over open country near the car
const BIRDS_PER = 18, FLOCKS = 4;
const birdGeo = (() => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0.12, -0.5, 0.08, -0.08, 0, 0, -0.1, 0, 0, 0.12, 0, 0, -0.1, 0.5, 0.08, -0.08], 3));
  g.computeVertexNormals();
  return g;
})();
const birdMesh = new THREE.InstancedMesh(birdGeo, new THREE.MeshBasicMaterial({ color: 0x1d1f22, side: THREE.DoubleSide, fog: true }), BIRDS_PER * FLOCKS);
birdMesh.frustumCulled = false;
scene.add(birdMesh);
const flocks = [];
function updateBirds(dt) {
  for (let f = 0; f < FLOCKS; f++) {
    let F = flocks[f];
    if (!F || Math.hypot(F.cx - V.x, F.cz - V.z) > 900) {
      const a = Math.random() * 6.283, d = 150 + Math.random() * 450, cx = V.x + Math.cos(a) * d, cz = V.z + Math.sin(a) * d;
      const g = terrainHeight(cx, cz);
      F = flocks[f] = { cx, cz, y: (g ?? V.y) + 30 + Math.random() * 40, r: 25 + Math.random() * 40, w: (0.15 + Math.random() * 0.2) * (Math.random() < 0.5 ? 1 : -1), ph: Math.random() * 6.283, drift: [(Math.random() - 0.5) * 3, (Math.random() - 0.5) * 3],
                         off: Array.from({ length: BIRDS_PER }, () => [Math.random() * 6.283, (Math.random() - 0.5) * 14, (Math.random() - 0.5) * 6, 0.85 + Math.random() * 0.3, Math.random() * 6.283]) };
    }
    F.ph += F.w * dt; F.cx += F.drift[0] * dt; F.cz += F.drift[1] * dt;
    for (let b = 0; b < BIRDS_PER; b++) {
      const o = F.off[b], a = F.ph * o[3] + o[0] * 0.15, rr = F.r + o[1];
      const x = F.cx + Math.cos(a) * rr, z = F.cz + Math.sin(a) * rr, y = F.y + o[2] + Math.sin(timeU.value * 0.7 + o[0]) * 1.5;
      const head = Math.atan2(-Math.sin(a) * Math.sign(F.w), Math.cos(a) * Math.sign(F.w)) + Math.PI / 2;
      const flap = 0.4 + 0.6 * Math.abs(Math.sin(timeU.value * 9 + o[4]));
      _mm.compose(_pp.set(x, y, z), _qq.setFromEuler(_eu.set(0, head, 0)), _ss.set(1, flap * 1.6 - 0.3, 1));
      birdMesh.setMatrixAt(f * BIRDS_PER + b, _mm);
    }
  }
  birdMesh.instanceMatrix.needsUpdate = true;
}

let genJob = null;
function updateProps(dt) {
  // generate tiles once their roads, imagery and (briefly awaited) buildings are in — at most ~6 ms a frame
  const now = performance.now();
  if (genJob && tiles.get(genJob.key) !== genJob.t) genJob = null;
  if (!genJob) {
    // nearest ready tile first, so the one you are driving in is never stuck behind the outskirts
    let best = null, bd = Infinity;
    for (const t of tiles.values()) {
      if (t.propsGen || t.lod !== 1 || !t.segs || !t.img) continue;
      if (!t.footHash && now - (t.imgAt || 0) < 6000) continue;
      const d = Math.hypot(t.ox + TILE_SIZE / 2 - FOC.x, t.oz + TILE_SIZE / 2 - FOC.z);
      if (d < bd) { bd = d; best = t; }
    }
    if (best) genJob = { t: best, key: `${best.tx},${best.ty}`, it: propGen(best) };
  }
  if (genJob) while (performance.now() - now < 6) {
    if (genJob.it.next().done) { genJob = null; propsDirty = true; break; }
  }
  for (const t of tiles.values()) if (t.propsGen && !t.osmState && t.lod === 1) { t.osmState = 'queued'; osmQueue.push(t); }
  pumpOsm();
  propCooldown -= dt;
  if (propCooldown <= 0 && (propsDirty || Math.hypot(FOC.x - propX, FOC.z - propZ) > 30)) {
    propX = FOC.x; propZ = FOC.z; propsDirty = false; propCooldown = 0.3;
    rebuildProps(FOC.x, FOC.z);
  }
  updateDebris(dt);
  updateBirds(dt);
}

// ===================================================================== occlusion culling
// Terrain hides what's behind it: every quarter second each far tile and landmark is tested for a line of sight
// from the camera over the loaded terrain. Hidden tiles aren't drawn (and don't refine past their coarse mesh);
// hidden landmarks aren't drawn. A tile counts as visible if any of its corners or centre, raised to its highest
// ground plus a margin, can be seen - so ridges and tall buildings on it are never wrongly cut.
function lineOfSight(ex, ey, ez, px, py, pz) {
  const dx = px - ex, dy = py - ey, dz = pz - ez, dist = Math.hypot(dx, dz);
  if (dist < 300) return true;
  const n = Math.max(8, Math.min(48, Math.round(dist / 120)));
  for (let i = 1; i < n; i++) {
    const f = 0.03 + 0.9 * i / n, h = terrainHeight(ex + dx * f, ez + dz * f);
    if (h !== null && h > ey + dy * f + 2) return false;
  }
  return true;
}
let cullTimer = 0;
const tileMeshes = t => [t.terrainMesh, t.roadMesh, t.vergeMesh, t.waterMesh, t.buildingMesh, t.structMesh];
function updateCulling(dt) {
  if ((cullTimer -= dt) > 0) return;
  cullTimer = 0.25;
  const e = camera.position;
  for (const t of tiles.values()) {
    let vis = true;
    if (t.ring > FULL_RADIUS && t.hMax !== undefined) {
      const y = t.hMax + 30, S = TILE_SIZE;
      vis = false;
      for (const [a, b] of [[0.5, 0.5], [0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]) {
        if (lineOfSight(e.x, e.y, e.z, t.ox + a * S, y, t.oz + b * S)) { vis = true; break; }
      }
    }
    // hide only after two hidden checks in a row (no flicker at the edge of a ridge); show at once
    t.hiddenRuns = vis ? 0 : (t.hiddenRuns || 0) + 1;
    const occ = t.hiddenRuns >= 2;
    if (occ !== !!t.occluded) { t.occluded = occ; for (const m of tileMeshes(t)) if (m) m.visible = !occ; }
  }
}
function landmarkHidden(L) {
  if (!L.mesh || L.y === null) return false;
  const e = camera.position, top = L.y + L.geo.userData.height;
  return !lineOfSight(e.x, e.y, e.z, L.px, top, L.pz) && !lineOfSight(e.x, e.y, e.z, L.px, L.y + L.geo.userData.height * 0.5, L.pz);
}

// ===================================================================== water

// Tileable ripple normal map (RG = normal xz, B = foam noise, A = slow noise), made once from integer-frequency
// waves so it wraps seamlessly; the shader scrolls it with the wind on still water and drags it along the
// current on rivers (two-phase flow mapping).
const waterNormalTex = (() => {
  const N = 256, gx = new Float32Array(N * N), gz = new Float32Array(N * N), foam = new Float32Array(N * N), slow = new Float32Array(N * N);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const waves = [];
  for (let k = 0; k < 48; k++) {
    const kx = Math.round((rnd() - 0.5) * 22), kz = Math.round((rnd() - 0.5) * 22), m = Math.hypot(kx, kz);
    if (m < 1.5) continue;
    waves.push([kx, kz, 0.9 / Math.pow(m, 1.6), rnd() * 6.283]);
  }
  const fw = [];
  for (let k = 0; k < 40; k++) { const kx = Math.round((rnd() - 0.5) * 40), kz = Math.round((rnd() - 0.5) * 40); if (Math.hypot(kx, kz) > 4) fw.push([kx, kz, rnd() * 6.283]); }
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const u = i / N * 6.2832, v = j / N * 6.2832, k = j * N + i;
    for (const [kx, kz, a, ph] of waves) { const sn = Math.sin(kx * u + kz * v + ph); gx[k] -= a * kx * sn; gz[k] -= a * kz * sn; }
    let f = 0; for (const [kx, kz, ph] of fw) f += Math.cos(kx * u + kz * v + ph);
    foam[k] = f; slow[k] = Math.cos(2 * u + v + 1) + Math.cos(-u + 3 * v + 2) + Math.cos(3 * u - 2 * v);
  }
  let gm = 0, fm = 0; for (let k = 0; k < N * N; k++) { gm = Math.max(gm, Math.abs(gx[k]), Math.abs(gz[k])); fm = Math.max(fm, Math.abs(foam[k])); }
  const data = new Uint8Array(N * N * 4);
  for (let k = 0; k < N * N; k++) {
    data[k * 4] = 128 + 127 * gx[k] / gm; data[k * 4 + 1] = 128 + 127 * gz[k] / gm;
    data[k * 4 + 2] = 128 + 127 * foam[k] / fm; data[k * 4 + 3] = 128 + 127 * slow[k] / 3;
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.needsUpdate = true;
  return t;
})();

// Ocean swell (the car uses the same function so it bobs in step with what you see): a few long
// deep-water waves; swellAt returns [height, d/dx, d/dz]
const SWELL = [[0.8, 0.6, 46, 0.32, 0.0], [-0.3, 1.0, 29, 0.2, 1.7], [0.95, -0.3, 17, 0.1, 4.1]];
const SWELL_GLSL = SWELL.map(([dx, dz, L, a, ph]) => {
  const l = Math.hypot(dx, dz), k = 2 * Math.PI / L, w = Math.sqrt(9.81 * k);
  return `{ vec2 d = vec2(${(dx / l).toFixed(5)}, ${(dz / l).toFixed(5)}); float th = dot(p, d) * ${k.toFixed(5)} - t * ${w.toFixed(5)} + ${ph.toFixed(2)};
        h += ${a.toFixed(3)} * sin(th); g += d * ${(a * k).toFixed(5)} * cos(th); }`;
}).join('\n      ');
function swellAt(x, z, t) {
  let h = 0, gx = 0, gz = 0;
  for (const [dx, dz, L, a, ph] of SWELL) {
    const l = Math.hypot(dx, dz), ux = dx / l, uz = dz / l, k = 2 * Math.PI / L, th = (x * ux + z * uz) * k - t * Math.sqrt(9.81 * k) + ph;
    h += a * Math.sin(th); gx += ux * a * k * Math.cos(th); gz += uz * a * k * Math.cos(th);
  }
  return [h, gx, gz];
}

const waterMat = new THREE.ShaderMaterial({
  uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, {
    uTime: { value: 0 }, uSun: { value: SUN_DIR.clone() },
    uHorizon: { value: new THREE.Color(0xb4d0ea) }, uZenith: { value: new THREE.Color(0x4a7fbf) },
    uNorm: { value: null }, uCar: { value: new THREE.Vector4(0, 0, 0, 0) }, uCarDir: { value: new THREE.Vector2(0, 1) },
  }]),
  fog: true, transparent: true, depthWrite: false,
  vertexShader: /* glsl */`
    uniform float uTime;
    attribute float depth;
    attribute vec4 wflow;
    attribute float wcover;
    varying float vDepth, vCover;
    varying vec4 vFlow;
    varying vec3 vWorld;
    varying vec2 vSwellG;
    #include <fog_pars_vertex>
    void main() {
      vDepth = depth; vFlow = wflow; vCover = wcover;
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vec2 p = wp.xz; float t = uTime, h = 0.0; vec2 g = vec2(0.0);
      ${SWELL_GLSL}
      float amp = wflow.z * smoothstep(0.3, 5.0, depth);       // sea only, calming towards the beach
      wp.y += h * amp; vSwellG = g * amp;
      vWorld = wp.xyz;
      vec4 mvPosition = viewMatrix * wp;
      gl_Position = projectionMatrix * mvPosition;
      #include <fog_vertex>
    }`,
  fragmentShader: /* glsl */`
    uniform float uTime;
    uniform vec3 uSun, uHorizon, uZenith;
    uniform sampler2D uNorm;
    uniform vec4 uCar;      // x, z, speed through the water, in water (0..1)
    uniform vec2 uCarDir;
    varying float vDepth, vCover;
    varying vec4 vFlow;     // current x, z (m/s), sea (0/1), turbulence (steep reaches)
    varying vec3 vWorld;
    varying vec2 vSwellG;
    #include <fog_pars_fragment>
    vec4 tex(vec2 uv) { vec4 s = texture2D(uNorm, uv); return vec4(s.xy * 2.0 - 1.0, s.zw); }
    void main() {
      vec2 p = vWorld.xz, fl = vFlow.xy;
      float speed = length(fl), sea = vFlow.z, turb = vFlow.w, t = uTime;
      // wind ripples: two layers drifting across each other
      vec4 w1 = tex(p / 11.0 + vec2(0.021, 0.013) * t), w2 = tex(p / 4.3 + vec2(-0.017, 0.029) * t);
      vec2 nW = w1.xy * 0.6 + w2.xy * 0.4;
      // current: the ripple pattern is carried downstream (two phases half a cycle apart, cross-faded)
      const float T = 2.2;
      float ph0 = fract(t / T), ph1 = fract(t / T + 0.5), b0 = 1.0 - abs(2.0 * ph0 - 1.0);
      vec4 fA = mix(tex((p - fl * ph1 * T) / 5.0 + 0.5), tex((p - fl * ph0 * T) / 5.0), b0);
      vec4 fB = mix(tex((p - fl * ph1 * T) / 2.1 + 0.75), tex((p - fl * ph0 * T) / 2.1 + 0.25), b0);
      vec2 nF = fA.xy * 0.65 + fB.xy * 0.45;
      float flowing = smoothstep(0.04, 0.5, speed);
      float rough = mix(0.16, 0.26, flowing) + 0.25 * turb + 0.06 * sea;
      vec2 nxy = mix(nW, nF, flowing) * rough + vSwellG;
      vec3 N = normalize(vec3(-nxy.x, 1.0, -nxy.y));
      vec3 V = normalize(cameraPosition - vWorld);
      float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
      vec3 R = reflect(-V, N);
      R.y = abs(R.y);
      vec3 sky = mix(uHorizon, uZenith, clamp(R.y * 1.6, 0.0, 1.0));
      // body colour: shallow water shows a lighter tint (bed + silt), deep water goes dark
      vec3 shallow = mix(vec3(0.16, 0.22, 0.14), vec3(0.10, 0.36, 0.36), sea);
      vec3 deep = mix(vec3(0.025, 0.06, 0.055), vec3(0.01, 0.05, 0.10), sea);
      vec3 body = mix(shallow, deep, smoothstep(0.0, mix(2.5, 7.0, sea), vDepth));
      // light scattering through wave crests facing the sun
      float sss = pow(max(dot(V, -normalize(vec3(uSun.x, 0.0, uSun.z))), 0.0), 3.0) * max(nxy.x * uSun.x + nxy.y * uSun.z, 0.0);
      body += vec3(0.05, 0.16, 0.13) * sss * 2.0;
      vec3 col = mix(body, sky, fres);
      float sp = max(dot(R, uSun), 0.0);
      col += (pow(sp, 600.0) * 6.0 + pow(sp, 90.0) * 0.6) * vec3(1.0, 0.95, 0.85);
      // foam: breaking wash on beaches, streaks on fast or steep water, churn and wake around the car
      float n = mix(w1.z, fA.z, flowing), n2 = mix(w2.z, fB.z, flowing);
      float shore = 1.0 - smoothstep(0.05, mix(0.45, 1.2, sea), vDepth);
      float wash = sea * (0.5 + 0.5 * sin(vDepth * 9.0 - t * 1.6 + n * 3.0));
      float foam = shore * shore * mix(0.5, 1.0, sea) * smoothstep(0.6, 0.85, n * 0.6 + n2 * 0.4 + shore * 0.15 + wash * 0.3);
      float rush = clamp((speed - 0.6) * 0.5, 0.0, 0.6) + turb;
      foam += smoothstep(0.85 - 0.45 * rush, 1.0 - 0.4 * rush, n * 0.55 + n2 * 0.45) * min(1.0, rush * 1.6);
      vec2 cd = p - uCar.xy;
      float along = dot(cd, uCarDir), lat = abs(dot(cd, vec2(uCarDir.y, -uCarDir.x)));
      float ringF = 1.0 - smoothstep(1.2, 2.2, length(vec2(lat, along * 0.5)));
      float wake = along < -1.0 ? (1.0 - smoothstep(0.0, 0.5 - along * 0.04, abs(lat - 1.0 + along * 0.36))) * exp(along / 18.0) : 0.0;
      float churn = uCar.w * min(1.0, uCar.z / 4.0) * max(ringF, wake);
      foam += churn * smoothstep(0.5, 0.75, n2 * 0.5 + n * 0.5 + churn * 0.2);
      foam = clamp(foam, 0.0, 1.0);
      col = mix(col, vec3(0.92, 0.95, 0.95) * (0.75 + 0.25 * max(uSun.y, 0.0)), foam * 0.9);
      float a = mix(0.35, 0.95, smoothstep(0.0, 1.4, vDepth));
      // the waterline: follows the fine outline of the water (not the coarse grid), with a soft edge
      float waterline = smoothstep(0.3, 0.62, vCover);
      gl_FragColor = vec4(col, clamp(max(max(a, fres), foam), 0.0, 1.0) * waterline);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
      #include <fog_fragment>
    }`,
});
waterMat.uniforms.uTime = timeU;
waterMat.uniforms.uNorm.value = waterNormalTex;

/** Water surface height at a point, or null if there's no water. */
function waterLevelAt(x, z) {
  const t = tileFor(x, z);
  if (!t || !t.waterLevel) return null;
  const gi = Math.round((x - t.ox) / CELL + M), gj = Math.round((z - t.oz) / CELL + M);
  if (gi < 0 || gj < 0 || gi >= G || gj >= G) return null;
  const l = t.waterLevel[gj * G + gi];
  return Number.isNaN(l) ? null : l;
}
/** The worker stores the sea at a level of -0.001 so it can be told apart (it carries a swell). */
const isSeaLevel = l => l !== null && l < 0 && l > -0.002;
/** Surface current at a point (m/s, world x/z), bilinear. Reuses one array. */
const _flow = [0, 0];
function waterFlowAt(x, z) {
  _flow[0] = _flow[1] = 0;
  const t = tileFor(x, z);
  if (!t || !t.waterCurrent) return _flow;
  const fx = (x - t.ox) / CELL + M, fz = (z - t.oz) / CELL + M, i0 = Math.floor(fx), j0 = Math.floor(fz), u = fx - i0, v = fz - j0;
  if (i0 < 0 || j0 < 0 || i0 >= G - 1 || j0 >= G - 1) return _flow;
  const C = t.waterCurrent;
  const acc = (i, j, w) => { const k = (j * G + i) * 2; _flow[0] += C[k] / 32 * w; _flow[1] += C[k + 1] / 32 * w; };
  acc(i0, j0, (1 - u) * (1 - v)); acc(i0 + 1, j0, u * (1 - v)); acc(i0, j0 + 1, (1 - u) * v); acc(i0 + 1, j0 + 1, u * v);
  return _flow;
}

// Arcade vehicle. heading h: forward = (sin h, cos h) in (x, z); left = (cos h, -sin h); +h turns left.
const V = {
  x: 0, y: 0, z: 0, heading: 0,
  vx: 0, vy: 0, vz: 0,
  air: false,
  steer: 0, wheelSpin: 0,
  gPitch: 0, gRoll: 0,    // ground slope under the car
  roll: 0, pitch: 0,      // cosmetic body lean (sprung: rollV/pitchV are their rates)
  rollV: 0, pitchV: 0, heave: 0, heaveV: 0, wheelslip: 0,
  yawRate: 0, wd: 0, gear: 1, rpm: 800, shiftT: 0, spinD: 0, susp: [0, 0, 0, 0],
  surface: SURF.grass, segOff: -1, segs: null, grade: 0,
};

const SURF_PHYS = [
  // grip: lateral velocity decay /s · vmax m/s · rolling decel m/s² · accel scale
  { grip: 9.0, vmax: 62, roll: 0.25, accel: 1.0 },  // asphalt
  { grip: 4.5, vmax: 38, roll: 0.9, accel: 0.8 },   // gravel
  { grip: 4.0, vmax: 32, roll: 1.2, accel: 0.75 },  // dirt
  { grip: 3.2, vmax: 24, roll: 2.0, accel: 0.6 },   // grass
  { grip: 2.2, vmax: 16, roll: 3.5, accel: 0.45 },  // mud
  { grip: 1.8, vmax: 7, roll: 6.0, accel: 0.3 },    // water (wading)
];
let WHEELBASE = 2.7, TRACK = 1.6;
const GRAV = 9.81;
const DRAFT = 0.6;   // water deeper than this lifts the car off the bed
const HIT_MU = 0.45;                 // car body sliding along bark, brick or a fence
function carImpulse(px, pz, nx, nz) {
  const m = CAR.kg, I = CAR.Iz * (CAR.yawI || 1), rx = px - V.x, rz = pz - V.z, w = V.yawRate;
  // velocity of the car's skin at the contact (the yaw turns a point r at w * (rz, -rx))
  const vcx = V.vx + w * rz, vcz = V.vz - w * rx, vn = vcx * nx + vcz * nz;
  if (vn >= 0) return 0;
  const e = 0.06 + 0.2 * Math.exp(vn / 4);                         // a nudge bounces a little, a crash hardly
  const rn = rz * nx - rx * nz, jn = -(1 + e) * vn / (1 / m + rn * rn / I);
  const tx = -nz, tz = nx, vt = vcx * tx + vcz * tz, rt = rz * tx - rx * tz, lim = HIT_MU * jn;
  const jt = Math.max(-lim, Math.min(lim, -vt / (1 / m + rt * rt / I)));
  const Jx = nx * jn + tx * jt, Jz = nz * jn + tz * jt;
  V.vx += Jx / m; V.vz += Jz / m;
  V.yawRate = Math.max(-4, Math.min(4, V.yawRate + (rz * Jx - rx * Jz) / I));
  if (vn < -2) V.wd = (V.vx * Math.sin(V.heading) + V.vz * Math.cos(V.heading)) / CAR.wheelR;
  V.hit = Math.max(V.hit || 0, -vn);
  return -vn;
}
/** The car against a circle (x, z, radius). probe: only say whether they touch. */
function carHitCircle(cx, cz, r, dt, probe = false) {
  const fx = Math.sin(V.heading), fz = Math.cos(V.heading), lx = fz, lz = -fx, hl = CAR.len / 2, hw = CAR.wid / 2;
  const ex = cx - V.x, ez = cz - V.z, a = ex * fx + ez * fz, b = ex * lx + ez * lz;
  const va = V.vx * fx + V.vz * fz, vb = V.vx * lx + V.vz * lz, sw = Math.abs(va) * dt;
  if (Math.abs(b) >= hw + r || a >= hl + r + (va < 0 ? sw : 0) || a <= -hl - r - (va > 0 ? sw : 0)) return false;
  // the box has rounded corners (a car's are): a trunk caught just inside a corner glances off it instead of
  // stopping the car as if it had hit a brick. Inner box shrunk by RC, circle grown by RC.
  const RC = Math.min(0.35, hw * 0.6), il = hl - RC, iw = hw - RC, R2 = r + RC;
  const qa = Math.max(-il, Math.min(il, a)), qb = Math.max(-iw, Math.min(iw, b));
  let na, nb, depth, pa, pb;
  const d = Math.hypot(qa - a, qb - b);
  if (d > 1e-4 && d < R2 && Math.abs(a) <= hl + r) {
    // touching from outside (a side, or a corner): straight out from the circle, touching the car's skin
    na = (qa - a) / d; nb = (qb - b) / d; depth = R2 - d; pa = qa - na * RC; pb = qb - nb * RC;
  } else if (d > 1e-4 && Math.abs(a) <= hl + r) return false;      // near a corner but clear of it
  else {
    // inside the box, or swept past: out through the face it came in by (the one it crossed most recently,
    // going by how the car is moving), not just the nearest - a trunk deep in the bonnet goes out the front
    const faces = [];
    if (va > 0.05) faces.push([(hl + r - a) / va, 0]);
    if (va < -0.05) faces.push([(a + hl + r) / -va, 1]);
    if (vb > 0.05) faces.push([(hw + r - b) / vb, 2]);
    if (vb < -0.05) faces.push([(b + hw + r) / -vb, 3]);
    if (!faces.length) faces.push([hl + r - a, 0], [a + hl + r, 1], [hw + r - b, 2], [b + hw + r, 3]);
    faces.sort((p, q) => p[0] - q[0]);
    const f = faces[0][1];
    if (f === 0) { na = -1; nb = 0; depth = hl + r - a; pa = hl; pb = qb; }
    else if (f === 1) { na = 1; nb = 0; depth = a + hl + r; pa = -hl; pb = qb; }
    else if (f === 2) { na = 0; nb = -1; depth = hw + r - b; pa = qa; pb = hw; }
    else { na = 0; nb = 1; depth = b + hw + r; pa = qa; pb = -hw; }
  }
  if (probe) return true;
  const nx = na * fx + nb * lx, nz = na * fz + nb * lz;
  carImpulse(V.x + pa * fx + pb * lx, V.z + pa * fz + pb * lz, nx, nz);
  V.x += nx * depth; V.z += nz * depth;
  return true;
}

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
  // the car (here at V: x, z, heading, vx, vz, y, yawRate; len, wid, kg, Iz, wheelR its size, mass, yaw inertia and
  // wheels) pushed out of trees, buildings, landmarks and props and turned by what it hit: V afterwards
  collide(p) {
    for (const k of ['len', 'wid', 'kg', 'Iz', 'wheelR']) if (p[k] > 0) CAR[k] = p[k];
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
