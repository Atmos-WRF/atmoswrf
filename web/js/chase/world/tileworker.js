// The detailed map's tile worker (from Real Roads Drive): terrain, roads, water, structures,
// buildings, imagery and trees for one z14 tile at a time.
const Z = 14, NT = 1 << Z, M = 16, G = 257 + 2 * M;
const DEM_URL = (x, y) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${Z}/${x}/${y}.png`;
const SEG = 13; // x0 z0 x1 z1 hw surf rank cls bridge h0 h1 lanes(negative = one-way) roadId
const RESAMPLE = 3, BLEND = 10, TEX_M = 12;
const ROAD_LIFT = 0.03; // road surface above its height profile
const CUT = 0.12;       // terrain under a road sits this far below the surface so it never pokes through
const VERGE = 1.6;      // soft gravel edge that blends the road into the ground (m)

// lane width (m), lanes on a two-way road, paved shoulder each side (m), draw rank,
// centre line (0 none, 1 dashed, 2 double solid), edge lines
const ROAD_CLASSES = {
  motorway:  [3.65, 4, 2.5, 9, 2, 1],
  trunk:     [3.6, 4, 1.5, 8, 2, 1],
  primary:   [3.5, 2, 1.0, 7, 2, 1],
  secondary: [3.4, 2, 0.6, 6, 1, 1],
  tertiary:  [3.2, 2, 0.3, 5, 1, 0],
  minor:     [3.1, 2, 0.2, 4, 0, 0],
  busway:    [3.5, 2, 0.3, 4, 1, 0],
  raceway:   [5.0, 2, 0.5, 4, 0, 1],
  service:   [2.7, 2, 0.0, 3, 0, 0],
  track:     [3.0, 1, 0.0, 2, 0, 0],
  rail:      [3.6, 1, 0.0, 1, 0, 0],   // railway: drawn as a ballast bed; rails + sleepers are props
  path:      [2.6, 1, 0.0, 1, 0, 0],   // footpaths, steps, pedestrian streets: only kept near landmarks
};
const CLASS_LIST = Object.keys(ROAD_CLASSES);
const SURF = { asphalt: 0, gravel: 1, dirt: 2, grass: 3 };

// ------------------------------------------------ minimal vector tile reader
class PB {
  constructor(buf) { this.b = buf; this.p = 0; this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength); }
  varint() { let r = 0, m = 1, b; do { b = this.b[this.p++]; r += (b & 0x7f) * m; m *= 128; } while (b & 0x80); return r; }
  svarint() { const v = this.varint(); return v % 2 === 1 ? (v + 1) / -2 : v / 2; }
  bytes() { const l = this.varint(); const s = this.b.subarray(this.p, this.p + l); this.p += l; return s; }
  string() { return new TextDecoder().decode(this.bytes()); }
  skip(wt) { if (wt === 0) this.varint(); else if (wt === 1) this.p += 8; else if (wt === 2) this.p += this.varint(); else if (wt === 5) this.p += 4; }
  packed() { const end = this.varint() + this.p, out = []; while (this.p < end) out.push(this.varint()); return out; }
}
function readValue(pb) {
  const end = pb.varint() + pb.p; let v = null;
  while (pb.p < end) {
    const tag = pb.varint(), f = tag >> 3, wt = tag & 7;
    if (f === 1) v = pb.string();
    else if (f === 2) { v = pb.dv.getFloat32(pb.p, true); pb.p += 4; }
    else if (f === 3) { v = pb.dv.getFloat64(pb.p, true); pb.p += 8; }
    else if (f === 4 || f === 5) v = pb.varint();
    else if (f === 6) v = pb.svarint();
    else if (f === 7) v = !!pb.varint();
    else pb.skip(wt);
  }
  return v;
}
function readLayer(buf) {
  const pb = new PB(buf);
  const layer = { name: '', extent: 4096, keys: [], values: [], features: [] };
  while (pb.p < buf.length) {
    const tag = pb.varint(), f = tag >> 3, wt = tag & 7;
    if (f === 1) layer.name = pb.string();
    else if (f === 2) layer.features.push(pb.bytes());
    else if (f === 3) layer.keys.push(pb.string());
    else if (f === 4) layer.values.push(readValue(pb));
    else if (f === 5) layer.extent = pb.varint();
    else pb.skip(wt);
  }
  return layer;
}
function readFeature(buf, layer) {
  const pb = new PB(buf);
  let type = 0, tags = [], geom = [];
  while (pb.p < buf.length) {
    const tag = pb.varint(), f = tag >> 3, wt = tag & 7;
    if (f === 2) tags = pb.packed();
    else if (f === 3) type = pb.varint();
    else if (f === 4) geom = pb.packed();
    else pb.skip(wt);
  }
  const props = {};
  for (let i = 0; i < tags.length; i += 2) props[layer.keys[tags[i]]] = layer.values[tags[i + 1]];
  const parts = []; let cur = null, x = 0, y = 0, i = 0;
  while (i < geom.length) {
    const c = geom[i++], cmd = c & 7, n = c >> 3;
    if (cmd === 7) { if (cur && cur.length >= 2) cur.push(cur[0], cur[1]); continue; }
    for (let k = 0; k < n; k++) {
      const dx = geom[i++], dy = geom[i++];
      x += (dx >> 1) ^ -(dx & 1); y += (dy >> 1) ^ -(dy & 1);
      if (cmd === 1) { cur = [x, y]; parts.push(cur); } else cur.push(x, y);
    }
  }
  return { type, props, parts };
}
function readTile(arrayBuffer, wanted) {
  const buf = new Uint8Array(arrayBuffer), pb = new PB(buf), out = {};
  while (pb.p < buf.length) {
    const tag = pb.varint(), f = tag >> 3, wt = tag & 7;
    if (f === 3) { const l = readLayer(pb.bytes()); if (wanted.includes(l.name)) out[l.name] = l; }
    else pb.skip(wt);
  }
  return out;
}

// ------------------------------------------------ fetching with small caches
const sleep = ms => new Promise(r => setTimeout(r, ms));
function lru(map, key, make, max) {
  let p = map.get(key);
  if (p) { map.delete(key); map.set(key, p); return p; }
  p = make();
  map.set(key, p);
  p.then(v => { if (!v) map.delete(key); });
  if (map.size > max) map.delete(map.keys().next().value);
  return p;
}

const demCache = new Map();
function getDem(x, y) {
  x = ((x % NT) + NT) % NT; y = Math.min(NT - 1, Math.max(0, y));
  return lru(demCache, x + '/' + y, async () => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const res = await fetch(DEM_URL(x, y));
        if (!res.ok) throw new Error(res.status);
        const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
        const ctx = new OffscreenCanvas(256, 256).getContext('2d', { willReadFrequently: true });
        ctx.drawImage(bmp, 0, 0); bmp.close();
        const d = ctx.getImageData(0, 0, 256, 256).data, out = new Float32Array(65536);
        for (let i = 0; i < 65536; i++) out[i] = d[i * 4] * 256 + d[i * 4 + 1] + d[i * 4 + 2] / 256 - 32768;
        return out;
      } catch { await sleep(400 * (attempt + 1)); }
    }
    return null;
  }, 100);
}

const vtCache = new Map();
function getVectorLayers(template, x, y) {
  return lru(vtCache, x + '/' + y, async () => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const res = await fetch(template.replace('{z}', Z).replace('{x}', x).replace('{y}', y));
        if (res.status === 404 || res.status === 204) return { empty: true };
        if (!res.ok) throw new Error(res.status);
        const layers = readTile(await res.arrayBuffer(), ['transportation', 'water', 'waterway', 'building', 'landuse', 'poi']);
        return Object.keys(layers).length ? layers : { empty: true };
      } catch { await sleep(400 * (attempt + 1)); }
    }
    return null;
  }, 30);
}

// Heights for the tile plus M pixels on every side (pixel-corner convention).
async function assembleHeights(tx, ty) {
  const jobs = [];
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) jobs.push(getDem(tx + dx, ty + dy));
  const t = await Promise.all(jobs), center = t[4];
  const H = new Float32Array(G * G);
  for (let gj = 0; gj < G; gj++) {
    const py = gj - M, oy = py < 0 ? -1 : py >= 256 ? 1 : 0, ly = py - oy * 256;
    for (let gi = 0; gi < G; gi++) {
      const px = gi - M, ox = px < 0 ? -1 : px >= 256 ? 1 : 0, lx = px - ox * 256;
      const tile = t[(oy + 1) * 3 + ox + 1];
      H[gj * G + gi] = tile ? tile[ly * 256 + lx]
        : center ? center[Math.min(255, Math.max(0, py)) * 256 + Math.min(255, Math.max(0, px))] : 0;
    }
  }
  return H;
}

// ------------------------------------------------ height sampling
function sampleBilinear(H, x, z, c) {
  const gx = Math.min(G - 1.0001, Math.max(0, x / c + M)), gz = Math.min(G - 1.0001, Math.max(0, z / c + M));
  const i = Math.floor(gx), j = Math.floor(gz), fx = gx - i, fz = gz - j, k = j * G + i;
  return (H[k] * (1 - fx) + H[k + 1] * fx) * (1 - fz) + (H[k + G] * (1 - fx) + H[k + G + 1] * fx) * fz;
}
// exact height of the full-res terrain mesh (same triangle split as the mesh)
function sampleTri(H, x, z, c) {
  const gx = Math.min(G - 1.0001, Math.max(0, x / c + M)), gz = Math.min(G - 1.0001, Math.max(0, z / c + M));
  const i = Math.floor(gx), j = Math.floor(gz), fx = gx - i, fz = gz - j, k = j * G + i;
  const ha = H[k], hb = H[k + 1], hc = H[k + G], hd = H[k + G + 1];
  return fz >= fx ? ha + (hd - hc) * fx + (hc - ha) * fz : ha + (hb - ha) * fx + (hd - hb) * fz;
}
function normalAt(H, x, z, c) {
  const dx = (sampleTri(H, x + 1, z, c) - sampleTri(H, x - 1, z, c)) / 2;
  const dz = (sampleTri(H, x, z + 1, c) - sampleTri(H, x, z - 1, c)) / 2;
  const inv = 1 / Math.sqrt(dx * dx + 1 + dz * dz);
  return [-dx * inv, inv, -dz * inv];
}

// ------------------------------------------------ roads
function extractRoads(layer, tileSize, H0, c, lms) {
  // distance to the nearest landmark pad edge-ish (centre distance minus its size), tile-local metres
  const lmNear = (x, z) => { let d = Infinity; for (const L of lms || []) if (!L.bridge) d = Math.min(d, Math.hypot(x - L.x, z - L.z) - Math.max(L.pax || L.hx, L.paz || L.hz)); return d; };
  const k = tileSize / layer.extent, lo = -M * c, hi = (256 + M) * c;
  const roads = [];
  for (const fb of layer.features) {
    const f = readFeature(fb, layer);
    const p = f.props, cls = p.class, info = ROAD_CLASSES[cls];
    if (f.type !== 2 || !info || p.brunnel === 'tunnel' || p.indoor === 1) continue;
    if (cls === 'path') continue;       // footpaths, steps and pedestrian ways are not built

    if (cls === 'rail' && p.subclass && !/^(rail|narrow_gauge|preserved)$/.test(p.subclass)) continue;
    let [laneW, lanes, shoulder, rank, centre, edges] = info;
    const oneway = p.oneway === 1 || p.oneway === -1;
    if (p.ramp === 1) { lanes = 1; laneW = 3.8; shoulder = 1.0; centre = 0; edges = 1; }
    else if (oneway) {
      if (cls === 'motorway' || cls === 'trunk') { lanes = 2; shoulder = Math.min(shoulder, 1.8); }
      else if (cls === 'primary' || cls === 'secondary') lanes = 2;
      else { lanes = 1; laneW = Math.max(laneW, 3.6); }
      centre = 0;
    }
    let surf = SURF.asphalt;
    if (p.surface === 'unpaved') surf = cls === 'track' ? SURF.dirt : SURF.gravel;
    else if (p.surface !== 'paved' && cls === 'track') surf = SURF.dirt;
    if (cls === 'rail') surf = SURF.gravel;
    if (surf !== SURF.asphalt) { centre = 0; edges = 0; shoulder = 0; }
    const hw = lanes * laneW / 2 + shoulder;
    const bridge = p.brunnel === 'bridge';
    const zl = p.layer != null && Number.isFinite(+p.layer) ? +p.layer : bridge ? 1 : 0;   // OSM layer: which way goes on top

    for (const part of f.parts) {
      const orig = []; let inside = false;
      for (let i = 0; i < part.length; i += 2) {
        const x = part[i] * k, z = part[i + 1] * k;
        if (orig.length && Math.abs(x - orig[orig.length - 2]) < 0.05 && Math.abs(z - orig[orig.length - 1]) < 0.05) continue;
        orig.push(x, z);
        if (x > lo && x < hi && z > lo && z < hi) inside = true;
      }
      if (!inside || orig.length < 4) continue;
      const pts = [orig[0], orig[1]], len = [0], origIdx = [0];
      let acc = 0;
      for (let i = 2; i < orig.length; i += 2) {
        const x0 = orig[i - 2], z0 = orig[i - 1], x1 = orig[i], z1 = orig[i + 1];
        const sl = Math.hypot(x1 - x0, z1 - z0), n = Math.max(1, Math.ceil(sl / RESAMPLE));
        for (let s = 1; s <= n; s++) { pts.push(x0 + (x1 - x0) * s / n, z0 + (z1 - z0) * s / n); len.push(acc + sl * s / n); }
        acc += sl;
        origIdx.push(pts.length / 2 - 1);
      }
      const np = pts.length / 2, raw = new Float32Array(np), h = new Float32Array(np);
      for (let i = 0; i < np; i++) raw[i] = sampleBilinear(H0, pts[2 * i], pts[2 * i + 1], c);
      if (bridge) {
        const total = len[np - 1] || 1;
        for (let i = 0; i < np; i++) h[i] = raw[0] + (raw[np - 1] - raw[0]) * len[i] / total;
      } else {
        // smooth the profile (about ±15 m); endpoints stay on true terrain. Within a few hundred metres of a
        // landmark the road follows the ground closely (±3 m) so it meets plazas, pads and summits exactly
        for (let i = 0; i < np; i++) {
          const dl = lms && lms.length ? lmNear(pts[2 * i], pts[2 * i + 1]) : Infinity;
          const rmax = dl < 250 ? 1 : dl < 450 ? 3 : 5;
          const r = Math.min(rmax, i, np - 1 - i);
          let s = 0; for (let j = i - r; j <= i + r; j++) s += raw[j];
          h[i] = s / (2 * r + 1);
        }
      }
      // roundabout: a short closed loop of a public road that is close to a circle (not a parking or service loop)
      let ring = np > 8 && len[np - 1] < 450 && Math.hypot(pts[0] - pts[2 * np - 2], pts[1] - pts[2 * np - 1]) < 1.5 && !/^(rail|service|track)$/.test(cls);
      if (ring) {
        let cx = 0, cz = 0; for (let i = 0; i < np; i++) { cx += pts[2 * i]; cz += pts[2 * i + 1]; } cx /= np; cz /= np;
        let mr = 0; for (let i = 0; i < np; i++) mr += Math.hypot(pts[2 * i] - cx, pts[2 * i + 1] - cz) / np;
        let dev = 0; for (let i = 0; i < np; i++) dev = Math.max(dev, Math.abs(Math.hypot(pts[2 * i] - cx, pts[2 * i + 1] - cz) - mr));
        ring = mr >= 8 && mr <= 60 && dev < mr * 0.25;
      }
      roads.push({ pts, h, len, origIdx, hw, laneW, lanes, shoulder, centre: ring ? 0 : centre, edges, oneway, surf, rank, bridge, ring, layer: zl,
                   cls: CLASS_LIST.indexOf(cls), fade: new Float32Array(np).fill(1) });
    }
  }
  levelBridges(roads, tileSize);
  return roads;
}

// A bridge often arrives as several pieces (one way per span, split at a border, or cut at the tile edge).
// Sampling the DEM where two pieces meet puts that joint on the valley floor and the deck dips into the gorge.
// Instead: join the pieces into a graph, pin the real abutments (free ends on solid ground) to the terrain,
// and solve the joints as length-weighted averages - for a simple chain that is one straight deck end to end.
function levelBridges(roads, tileSize) {
  const br = roads.filter(r => r.bridge);
  if (!br.length) return;
  const nodes = [], nodeAt = (x, z, h) => {
    for (let i = 0; i < nodes.length; i++) if (Math.abs(nodes[i].x - x) < 1.5 && Math.abs(nodes[i].z - z) < 1.5) return i;
    nodes.push({ x, z, raw: h, edges: [], h, fixed: false });
    return nodes.length - 1;
  };
  const pieces = br.map(r => {
    const n = r.pts.length / 2, a = nodeAt(r.pts[0], r.pts[1], r.h[0]), b = nodeAt(r.pts[2 * n - 2], r.pts[2 * n - 1], r.h[n - 1]);
    const L = r.len[n - 1] || 1;
    nodes[a].edges.push([b, L]); nodes[b].edges.push([a, L]);
    return { r, a, b };
  });
  // connected groups
  const grp = new Int32Array(nodes.length).fill(-1);
  let ng = 0;
  for (let i = 0; i < nodes.length; i++) {
    if (grp[i] >= 0) continue;
    const st = [i]; grp[i] = ng;
    while (st.length) { const k = st.pop(); for (const [q] of nodes[k].edges) if (grp[q] < 0) { grp[q] = ng; st.push(q); } }
    ng++;
  }
  const clipped = nd => nd.x < -1 || nd.z < -1 || nd.x > tileSize + 1 || nd.z > tileSize + 1;
  for (let gi = 0; gi < ng; gi++) {
    const ns = nodes.filter((_, i) => grp[i] === gi), ends = ns.filter(nd => nd.edges.length === 1);
    const solid = ends.filter(nd => !clipped(nd));
    let top = -Infinity; for (const nd of solid) top = Math.max(top, nd.raw);
    if (!solid.length) for (const nd of ends) top = Math.max(top, nd.raw);
    // ends inside the tile are abutments; ends cut off at the tile edge only count if they're on high ground
    for (const nd of ends) if (!clipped(nd) || nd.raw >= top - 5 || !solid.length) nd.fixed = true;
    if (!ns.some(nd => nd.fixed)) { for (const nd of ns) { nd.h = top; } continue; }
    for (const nd of ns) if (!nd.fixed) nd.h = top;
    for (let it = 0; it < 300; it++) for (const nd of ns) {
      if (nd.fixed) continue;
      let sw = 0, sh = 0;
      for (const [q, L] of nd.edges) { sw += 1 / L; sh += nodes[q].h / L; }
      nd.h = sh / sw;
    }
  }
  for (const { r, a, b } of pieces) {
    const n = r.pts.length / 2, L = r.len[n - 1] || 1, ha = nodes[a].h, hb = nodes[b].h;
    for (let i = 0; i < n; i++) r.h[i] = ha + (hb - ha) * r.len[i] / L;
  }
}

// Junctions: make roads meet flush (the lesser road bends its profile to meet the bigger one)
// and record junction spots so lane markings stop short of them.
function connectRoads(roads) {
  const GC = 16, key = (gx, gz) => gx * 100003 + gz, grid = new Map();
  roads.forEach((r, ri) => {
    for (let i = 0; i < r.pts.length / 2 - 1; i++) {
      const ax = r.pts[2 * i], az = r.pts[2 * i + 1], bx = r.pts[2 * i + 2], bz = r.pts[2 * i + 3];
      for (let gz = Math.floor((Math.min(az, bz) - 3.5) / GC); gz <= Math.floor((Math.max(az, bz) + 3.5) / GC); gz++)
        for (let gx = Math.floor((Math.min(ax, bx) - 3.5) / GC); gx <= Math.floor((Math.max(ax, bx) + 3.5) / GC); gx++) {
          const k = key(gx, gz);
          (grid.get(k) || grid.set(k, []).get(k)).push(ri, i);
        }
    }
  });
  const vkey = (r, i) => Math.round(r.pts[2 * i] * 2) + ',' + Math.round(r.pts[2 * i + 1] * 2);
  const shared = new Map();
  roads.forEach((r, ri) => { for (const i of r.origIdx) { const k = vkey(r, i); (shared.get(k) || shared.set(k, new Set()).get(k)).add(ri); } });

  const crossAt = new Map(); // road index -> Set of point indices where another road crosses it
  const seenX = new Set();
  for (const list of grid.values()) {
    for (let a = 0; a < list.length; a += 2) for (let b = a + 2; b < list.length; b += 2) {
      const ri = list[a], i = list[a + 1], rj = list[b], j = list[b + 1];
      if (ri === rj) continue;
      const r = roads[ri], q = roads[rj];
      if (r.bridge !== q.bridge) continue;
      const ax = r.pts[2 * i], az = r.pts[2 * i + 1], bx = r.pts[2 * i + 2], bz = r.pts[2 * i + 3];
      const cx = q.pts[2 * j], cz = q.pts[2 * j + 1], dx = q.pts[2 * j + 2], dz = q.pts[2 * j + 3];
      const den = (bx - ax) * (dz - cz) - (bz - az) * (dx - cx);
      if (Math.abs(den) < 1e-9) continue;
      const u = ((cx - ax) * (dz - cz) - (cz - az) * (dx - cx)) / den, v = ((cx - ax) * (bz - az) - (cz - az) * (bx - ax)) / den;
      if (u < 0 || u > 1 || v < 0 || v > 1) continue;
      const k = ri < rj ? `${ri}:${i}:${rj}:${j}` : `${rj}:${j}:${ri}:${i}`;
      if (seenX.has(k)) continue;
      seenX.add(k);
      (crossAt.get(ri) || crossAt.set(ri, new Set()).get(ri)).add(u < 0.5 ? i : i + 1);
      (crossAt.get(rj) || crossAt.set(rj, new Set()).get(rj)).add(v < 0.5 ? j : j + 1);
    }
  }
  const dirAt = (r, i) => { const n = r.pts.length / 2, a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1), dx = r.pts[2 * b] - r.pts[2 * a], dz = r.pts[2 * b + 1] - r.pts[2 * a + 1], l = Math.hypot(dx, dz) || 1; return [dx / l, dz / l]; };

  // bigger roads first; each road may only snap to roads already processed
  const order = roads.map((_, i) => i).sort((a, b) => roads[b].rank - roads[a].rank || a - b);
  const pos = new Int32Array(roads.length);
  order.forEach((ri, k) => { pos[ri] = k; });
  const junctions = [], stops = [];
  for (const ri of order) {
    const r = roads[ri], n = r.pts.length / 2;
    const cand = new Set([0, n - 1]);
    for (const i of r.origIdx) if (shared.get(vkey(r, i)).size > 1) cand.add(i);
    for (const i of crossAt.get(ri) || []) cand.add(i);
    for (const i of cand) {
      const px = r.pts[2 * i], pz = r.pts[2 * i + 1];
      const list = grid.get(key(Math.floor(px / GC), Math.floor(pz / GC)));
      if (!list) continue;
      let best = null;
      for (let q = 0; q < list.length; q += 2) {
        const si = list[q], j = list[q + 1];
        if (si === ri) continue;
        const s = roads[si];
        if (s.bridge !== r.bridge) continue;
        const ax = s.pts[2 * j], az = s.pts[2 * j + 1], dx = s.pts[2 * j + 2] - ax, dz = s.pts[2 * j + 3] - az;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / (dx * dx + dz * dz || 1e-9)));
        const d = Math.hypot(ax + dx * t - px, az + dz * t - pz);
        if (d > 3.2) continue;
        if (!best || pos[si] < best.pos || (si === best.si && d < best.d)) best = { si, s, j, t, d, pos: pos[si] };
      }
      if (!best) continue;
      const [d1x, d1z] = dirAt(r, i), [d2x, d2z] = dirAt(best.s, best.j);
      const sinT = Math.abs(d1x * d2z - d1z * d2x);
      junctions.push(px, pz, Math.min(70, Math.max(Math.max(r.hw, best.s.hw) + 1.5, (r.hw + best.s.hw) / Math.max(sinT, 0.12) * 0.9 + 1)));
      // a lesser road ending at a bigger one: its traffic should stop / give way here
      if ((i === 0 || i === n - 1) && (best.s.rank > r.rank || best.s.ring) && !r.bridge && !r.ring && CLASS_LIST[r.cls] !== 'path') {
        const k = i === 0 ? 1 : n - 2, dx = px - r.pts[2 * k], dz = pz - r.pts[2 * k + 1], dl = Math.hypot(dx, dz) || 1;
        stops.push(px, pz, dx / dl, dz / dl, r.hw, best.s.hw, r.rank, best.s.ring ? -1 : best.s.rank);
      }
      if (best.pos > pos[ri]) continue; // the other road ranks lower: it snaps to this one later
      const target = best.s.h[best.j] + (best.s.h[best.j + 1] - best.s.h[best.j]) * best.t;
      const delta = target - r.h[i];
      if (Math.abs(delta) < 1e-3) continue;
      const L0 = r.len[i];
      for (let k = 0; k < n; k++) {
        const u = Math.abs(r.len[k] - L0) / 30;
        if (u < 1) r.h[k] += delta * (1 - u * u * (3 - 2 * u));
      }
    }
  }
  // lane markings fade out inside junctions
  const JG = 32, jgrid = new Map();
  for (let q = 0; q < junctions.length; q += 3) {
    const gx = Math.floor(junctions[q] / JG), gz = Math.floor(junctions[q + 1] / JG);
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
      const k = key(gx + dx, gz + dz);
      (jgrid.get(k) || jgrid.set(k, []).get(k)).push(q);
    }
  }
  for (const r of roads) {
    for (let i = 0; i < r.pts.length / 2; i++) {
      const x = r.pts[2 * i], z = r.pts[2 * i + 1], list = jgrid.get(key(Math.floor(x / JG), Math.floor(z / JG)));
      if (!list) continue;
      let f = 1;
      for (const q of list) f = Math.min(f, Math.max(0, Math.min(1, (Math.hypot(junctions[q] - x, junctions[q + 1] - z) - junctions[q + 2]) / 4)));
      r.fade[i] = f;
    }
  }
  const rings = [];
  for (const r of roads) if (r.ring) {
    const n = r.pts.length / 2; let cx = 0, cz = 0, rad = 0, hy = 0;
    for (let i = 0; i < n; i++) { cx += r.pts[2 * i]; cz += r.pts[2 * i + 1]; hy += r.h[i]; }
    cx /= n; cz /= n; hy /= n;
    for (let i = 0; i < n; i++) rad += Math.hypot(r.pts[2 * i] - cx, r.pts[2 * i + 1] - cz) / n;
    rings.push(cx, cz, rad, r.hw, hy);
  }
  return { junctions, stops, rings };
}

// cut terrain to just below road heights and blend it outwards; returns per-cell road influence (0..1)
function flatten(roads, H0, H, c) {
  const wBest = new Float32Array(G * G), dBest = new Float32Array(G * G).fill(1e9), target = new Float32Array(G * G);
  for (const r of roads) {
    if (r.bridge) continue;
    const inner = r.hw + 0.9 * c, outer = inner + BLEND, n = r.pts.length / 2;
    for (let s = 0; s < n - 1; s++) {
      const ax = r.pts[2 * s], az = r.pts[2 * s + 1], bx = r.pts[2 * s + 2], bz = r.pts[2 * s + 3];
      const ha = r.h[s] - CUT, hb = r.h[s + 1] - CUT, dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz || 1e-6;
      const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - outer) / c + M)), i1 = Math.min(G - 1, Math.ceil((Math.max(ax, bx) + outer) / c + M));
      const j0 = Math.max(0, Math.floor((Math.min(az, bz) - outer) / c + M)), j1 = Math.min(G - 1, Math.ceil((Math.max(az, bz) + outer) / c + M));
      for (let gj = j0; gj <= j1; gj++) {
        const pz = (gj - M) * c;
        for (let gi = i0; gi <= i1; gi++) {
          const px = (gi - M) * c;
          let t = ((px - ax) * dx + (pz - az) * dz) / l2; t = t < 0 ? 0 : t > 1 ? 1 : t;
          const qx = ax + dx * t - px, qz = az + dz * t - pz, d = Math.sqrt(qx * qx + qz * qz);
          if (d >= outer) continue;
          let w = 1;
          if (d > inner) { const u = (d - inner) / BLEND; w = 1 - u * u * (3 - 2 * u); }
          const idx = gj * G + gi;
          if (w > wBest[idx] + 1e-4 || (w > wBest[idx] - 1e-4 && d < dBest[idx])) {
            wBest[idx] = w; dBest[idx] = d; target[idx] = ha + (hb - ha) * t;
          }
        }
      }
    }
  }
  for (let i = 0; i < G * G; i++) if (wBest[i] > 0) H[i] = H0[i] + (target[i] - H0[i]) * wBest[i];
  return wBest;
}

// ------------------------------------------------ colours
function hash2(x, y) {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
function noise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
}
const lin = v => Math.pow(v, 2.2);
const mix = (a, b, t) => a + (b - a) * t;
const smooth = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };

function terrainColor(h, ny, road, gx, gy, out, o) {
  const n1 = noise(gx / 24, gy / 24) * 0.65 + noise(gx / 5, gy / 5) * 0.35;
  let r = mix(0.36, 0.6, n1), g = mix(0.48, 0.56, n1), b = mix(0.22, 0.33, n1);       // grass, lush..dry
  const rock = smooth(0.86, 0.7, ny);
  r = mix(r, 0.55, rock); g = mix(g, 0.52, rock); b = mix(b, 0.48, rock);              // steep rock
  const alp = smooth(3000, 3400, h) * 0.7;
  r = mix(r, 0.56, alp); g = mix(g, 0.53, alp); b = mix(b, 0.42, alp);                 // alpine
  const snow = smooth(3600, 3900, h) * smooth(0.65, 0.8, ny);
  r = mix(r, 1, snow); g = mix(g, 1, snow); b = mix(b, 1, snow);                       // snow
  const sh = Math.min(1, road) * 0.55;
  r = mix(r, 0.55, sh); g = mix(g, 0.49, sh); b = mix(b, 0.38, sh);                    // road shoulders
  out[o] = lin(r); out[o + 1] = lin(g); out[o + 2] = lin(b);
}

// ------------------------------------------------ meshes
function buildTerrain(H, infl, step, c, tx, ty, ox, oz) {
  const n = 256 / step + 1, vGrid = n * n, vCount = vGrid + 4 * n, sc = 2 * step * c;
  const positions = new Float32Array(vCount * 3), normals = new Float32Array(vCount * 3);
  const colors = new Float32Array(vCount * 3), uvs = new Float32Array(vCount * 2);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const idx = (M + j * step) * G + M + i * step, v = j * n + i, h = H[idx];
    const x = i * step * c, z = j * step * c;
    positions[v * 3] = x; positions[v * 3 + 1] = h; positions[v * 3 + 2] = z;
    uvs[v * 2] = (ox + x) / TEX_M; uvs[v * 2 + 1] = (oz + z) / TEX_M;
    const dx = (H[idx + step] - H[idx - step]) / sc, dz = (H[idx + step * G] - H[idx - step * G]) / sc;
    const inv = 1 / Math.sqrt(dx * dx + 1 + dz * dz);
    normals[v * 3] = -dx * inv; normals[v * 3 + 1] = inv; normals[v * 3 + 2] = -dz * inv;
    terrainColor(h, inv, infl ? infl[idx] : 0, tx * 256 + i * step, ty * 256 + j * step, colors, v * 3);
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6 + 4 * (n - 1) * 12);
  let ii = 0;
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = j * n + i, b = a + 1, cc = a + n, d = cc + 1;
    indices[ii++] = a; indices[ii++] = cc; indices[ii++] = d;
    indices[ii++] = a; indices[ii++] = d; indices[ii++] = b;
  }
  // skirts hide cracks between neighbouring tiles of different detail
  const skirt = 8 + step * 12, edges = [[], [], [], []];
  for (let k = 0; k < n; k++) { edges[0].push(k); edges[1].push((n - 1) * n + k); edges[2].push(k * n); edges[3].push(k * n + n - 1); }
  let sv = vGrid;
  for (const e of edges) {
    const base = sv;
    for (const src of e) {
      for (let q = 0; q < 3; q++) { positions[sv * 3 + q] = positions[src * 3 + q]; normals[sv * 3 + q] = normals[src * 3 + q]; colors[sv * 3 + q] = colors[src * 3 + q]; }
      positions[sv * 3 + 1] -= skirt;
      uvs[sv * 2] = uvs[src * 2]; uvs[sv * 2 + 1] = uvs[src * 2 + 1];
      sv++;
    }
    for (let k = 0; k < n - 1; k++) {
      const t0 = e[k], t1 = e[k + 1], s0 = base + k, s1 = base + k + 1;
      indices[ii++] = t0; indices[ii++] = t1; indices[ii++] = s0; indices[ii++] = t1; indices[ii++] = s1; indices[ii++] = s0;
      indices[ii++] = t0; indices[ii++] = s0; indices[ii++] = t1; indices[ii++] = t1; indices[ii++] = s0; indices[ii++] = s1;
    }
  }
  return { positions, normals, colors, uvs, indices };
}

// split a road into the pieces inside this tile, so neighbouring tiles never draw it twice
function clipPieces(r, S) {
  const E = 0.3, n = r.pts.length / 2;
  const inb = q => q.x >= -E && q.x <= S + E && q.z >= -E && q.z <= S + E;
  const pt = i => ({ x: r.pts[2 * i], z: r.pts[2 * i + 1], h: r.h[i], len: r.len[i], fade: r.fade[i] });
  const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t, h: a.h + (b.h - a.h) * t,
                               len: a.len + (b.len - a.len) * t, fade: a.fade + (b.fade - a.fade) * t });
  const edge = (inside, outside) => {
    let lo = 0, hi = 1;
    for (let k = 0; k < 14; k++) { const m = (lo + hi) / 2; if (inb(lerp(inside, outside, m))) lo = m; else hi = m; }
    return lerp(inside, outside, lo);
  };
  const pieces = []; let cur = null;
  for (let i = 0; i < n; i++) {
    const p = pt(i);
    if (inb(p)) {
      if (!cur) { cur = { pts: [], startCap: i === 0, endCap: false }; if (i > 0) cur.pts.push(edge(p, pt(i - 1))); }
      cur.pts.push(p);
    } else if (cur) { cur.pts.push(edge(pt(i - 1), p)); pieces.push(cur); cur = null; }
  }
  if (cur) { cur.endCap = true; pieces.push(cur); }
  return pieces.filter(pc => pc.pts.length >= 2);
}

function buildRoads(roads, H, c, S) {
  const R = { pos: [], nor: [], rd: [], rs: [], idx: [] };
  const V = { pos: [], nor: [], vt: [], idx: [] };
  const rv = (x, y, z, n, rd, rs) => { R.pos.push(x, y, z); R.nor.push(n[0], n[1], n[2]); R.rd.push(rd[0], rd[1], rd[2], rd[3]); R.rs.push(rs[0], rs[1], rs[2], rs[3]); return R.pos.length / 3 - 1; };
  const vv = (x, y, z, n, t) => { V.pos.push(x, y, z); V.nor.push(n[0], n[1], n[2]); V.vt.push(t); return V.pos.length / 3 - 1; };
  const UP = [0, 1, 0];

  for (const r of roads) {
    const W = 2 * r.hw, lift = ROAD_LIFT + r.rank * 0.003;
    const rs = [r.surf, r.lanes, r.shoulder, r.centre + 4 * r.edges + 8 * (r.oneway ? 1 : 0)];
    for (const piece of clipPieces(r, S)) {
      const P = piece.pts, n = P.length, base = R.pos.length / 3, vbase = V.pos.length / 3;
      for (let i = 0; i < n; i++) {
        const p = P[i], ip = P[Math.max(0, i - 1)], inx = P[Math.min(n - 1, i + 1)];
        let ax = 0, az = 0, bx = 0, bz = 0;
        if (i > 0) { ax = p.x - ip.x; az = p.z - ip.z; const l = Math.hypot(ax, az) || 1; ax /= l; az /= l; }
        if (i < n - 1) { bx = inx.x - p.x; bz = inx.z - p.z; const l = Math.hypot(bx, bz) || 1; bx /= l; bz /= l; }
        if (i === 0) { ax = bx; az = bz; }
        if (i === n - 1) { bx = ax; bz = az; }
        let tx = ax + bx, tz = az + bz; const tl = Math.hypot(tx, tz) || 1; tx /= tl; tz /= tl;
        const miter = Math.min(2, 1 / Math.max(0.5, tx * bx + tz * bz));
        const ux = -tz * miter, uz = tx * miter; // left perpendicular, miter-scaled
        const grade = (inx.h - ip.h) / Math.max(0.01, inx.len - ip.len);
        const gl = 1 / Math.sqrt(1 + grade * grade);
        const nn = [-grade * tx * gl, gl, -grade * tz * gl];
        const y = p.h + lift;
        rv(p.x + ux * r.hw, y, p.z + uz * r.hw, nn, [0, p.len, W, p.fade], rs);
        rv(p.x - ux * r.hw, y, p.z - uz * r.hw, nn, [W, p.len, W, p.fade], rs);
        if (!r.bridge) {
          for (const sgn of [1, -1]) {
            const ox = p.x + sgn * ux * (r.hw + VERGE), oz = p.z + sgn * uz * (r.hw + VERGE);
            vv(p.x + sgn * ux * r.hw, y - 0.012, p.z + sgn * uz * r.hw, nn, 0);
            vv(ox, sampleTri(H, ox, oz, c) + 0.02, oz, normalAt(H, ox, oz, c), 1);
          }
        }
      }
      for (let i = 0; i < n - 1; i++) {
        const L0 = base + 2 * i;
        R.idx.push(L0 + 1, L0, L0 + 2, L0 + 1, L0 + 2, L0 + 3);
        if (!r.bridge) {
          const v0 = vbase + 4 * i, v1 = v0 + 4; // per point: [L inner, L outer, R inner, R outer]
          V.idx.push(v0, v0 + 1, v1, v0 + 1, v1 + 1, v1);
          V.idx.push(v0 + 2, v1 + 2, v0 + 3, v0 + 3, v1 + 2, v1 + 3);
        }
      }
      // round caps at real road ends fill junction corners and dead ends
      for (const [use, p] of [[piece.startCap, P[0]], [piece.endCap, P[n - 1]]]) {
        if (!use) continue;
        const y = p.h + lift, seg = 16;
        const cv = rv(p.x, y, p.z, UP, [W / 2, p.len, W, 0], rs);
        for (let s = 0; s <= seg; s++) {
          const a = s / seg * Math.PI * 2;
          rv(p.x + Math.cos(a) * r.hw, y, p.z + Math.sin(a) * r.hw, UP, [W / 2, p.len, W, 0], rs);
        }
        for (let s = 0; s < seg; s++) R.idx.push(cv, cv + s + 2, cv + s + 1);
        if (!r.bridge) {
          const vb = V.pos.length / 3;
          for (let s = 0; s <= seg; s++) {
            const a = s / seg * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
            vv(p.x + ca * r.hw, y - 0.012, p.z + sa * r.hw, UP, 0);
            const ox = p.x + ca * (r.hw + VERGE), oz = p.z + sa * (r.hw + VERGE);
            vv(ox, sampleTri(H, ox, oz, c) + 0.02, oz, normalAt(H, ox, oz, c), 1);
          }
          for (let s = 0; s < seg; s++) { const a = vb + 2 * s; V.idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
        }
      }
    }
  }
  if (!R.idx.length) return null;
  return {
    road: { positions: new Float32Array(R.pos), normals: new Float32Array(R.nor), rd: new Float32Array(R.rd), rs: new Float32Array(R.rs), indices: new Uint32Array(R.idx) },
    verge: { positions: new Float32Array(V.pos), normals: new Float32Array(V.nor), vt: new Float32Array(V.vt), indices: new Uint32Array(V.idx) },
  };
}

function buildSegs(roads) {
  let count = 0;
  for (const r of roads) count += r.pts.length / 2 - 1;
  const segs = new Float32Array(count * SEG);
  let o = 0;
  roads.forEach((r, rid) => {
    const lift = ROAD_LIFT + r.rank * 0.003;
    for (let i = 0; i < r.pts.length / 2 - 1; i++) {
      segs.set([r.pts[2 * i], r.pts[2 * i + 1], r.pts[2 * i + 2], r.pts[2 * i + 3], r.hw, r.surf, r.rank, r.cls,
        r.bridge ? 1 : 0, r.h[i] + lift, r.h[i + 1] + lift, r.oneway ? -r.lanes : r.lanes, rid], o);
      o += SEG;
    }
  });
  return segs;
}

// ------------------------------------------------ grade separation
// OSM says a way is a bridge but not how high it is, and a 30 m DEM is too coarse to show the embankments that
// lift it. So: wherever a bridge crosses a road or railway, push the two apart to a real clearance (the bridge up,
// the road below down a little); bridges clear the water they span; roads crossing streams sit just above them.
// Each change spreads along the connected road network as a smooth ramp, so approaches climb to the deck.
function gradeSeparate(roads, wm, c) {
  if (!roads.length) return;
  const nR = roads.length, base = new Int32Array(nR + 1);
  for (let r = 0; r < nR; r++) base[r + 1] = base[r] + roads[r].pts.length / 2;
  const NV = base[nR], vr = new Int32Array(NV), vk = new Int32Array(NV);
  for (let r = 0; r < nR; r++) for (let k = 0; k < roads[r].pts.length / 2; k++) { vr[base[r] + k] = r; vk[base[r] + k] = k; }
  const isRail = r => CLASS_LIST[r.cls] === 'rail';

  // links: the end of one road lying on another (continuations and T-junctions), at about the same height
  const VC = 4, vgrid = new Map(), vkey = (x, z) => Math.floor(x / VC) * 100003 + Math.floor(z / VC);
  for (let v = 0; v < NV; v++) {
    const r = roads[vr[v]], k = vk[v], q = vkey(r.pts[2 * k], r.pts[2 * k + 1]);
    (vgrid.get(q) || vgrid.set(q, []).get(q)).push(v);
  }
  const links = new Map();
  const link = (a, b, d) => { (links.get(a) || links.set(a, []).get(a)).push(b, d); (links.get(b) || links.set(b, []).get(b)).push(a, d); };
  for (let r = 0; r < nR; r++) {
    const R = roads[r], n = R.pts.length / 2;
    for (const k of n > 1 ? [0, n - 1] : [0]) {
      const x = R.pts[2 * k], z = R.pts[2 * k + 1], v = base[r] + k;
      for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) {
        const list = vgrid.get(vkey(x + dx * VC, z + dz * VC));
        if (!list) continue;
        for (const u of list) {
          if (vr[u] === r) continue;
          const Q = roads[vr[u]], ku = vk[u], d = Math.hypot(Q.pts[2 * ku] - x, Q.pts[2 * ku + 1] - z);
          if (d < 2.5 && Math.abs(Q.h[ku] - R.h[k]) < 2) link(v, u, d);
        }
      }
    }
  }

  // raise/lower by `delta` everything within `flat` metres (along the network) of a vertex, fading over `ramp`
  const dist = new Float64Array(NV).fill(Infinity), touched = [], heap = [];   // 64-bit: float32 rounding loops on zero-length links
  const push = (d, v) => { heap.push([d, v]); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, rr = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (rr < heap.length && heap[rr][0] < heap[m][0]) m = rr; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
  function spread(r0, k0, delta, flat, ramp, block) {
    const v0 = base[r0] + k0, lim = flat + ramp;
    dist[v0] = 0; touched.push(v0); push(0, v0);
    while (heap.length) {
      const [d, v] = pop();
      if (d > dist[v]) continue;
      const r = vr[v], k = vk[v], R = roads[r], n = R.pts.length / 2;
      const step = (u, w) => { if (block && block.has(vr[u])) return; const nd = d + w; if (nd < dist[u] && nd <= lim) { dist[u] = nd; touched.push(u); push(nd, u); } };
      if (k > 0) step(v - 1, R.len[k] - R.len[k - 1]);
      if (k < n - 1) step(v + 1, R.len[k + 1] - R.len[k]);
      const L = links.get(v);
      if (L) for (let i = 0; i < L.length; i += 2) step(L[i], L[i + 1]);
    }
    for (const v of touched) {
      const d = dist[v];
      if (d === Infinity) continue;
      const u = d <= flat ? 0 : (d - flat) / ramp, w = u >= 1 ? 0 : 1 - u * u * (3 - 2 * u);
      roads[vr[v]].h[vk[v]] += delta * w;
      dist[v] = Infinity;
    }
    touched.length = 0;
  }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  // --- bridges over roads and railways
  const SC = 16, sgrid = new Map(), skey = (x, z) => x * 100003 + z;
  roads.forEach((r, ri) => {
    for (let i = 0; i < r.pts.length / 2 - 1; i++) {
      const ax = r.pts[2 * i], az = r.pts[2 * i + 1], bx = r.pts[2 * i + 2], bz = r.pts[2 * i + 3];
      for (let gz = Math.floor(Math.min(az, bz) / SC); gz <= Math.floor(Math.max(az, bz) / SC); gz++)
        for (let gx = Math.floor(Math.min(ax, bx) / SC); gx <= Math.floor(Math.max(ax, bx) / SC); gx++)
          (sgrid.get(skey(gx, gz)) || sgrid.set(skey(gx, gz), []).get(skey(gx, gz))).push(ri, i);
    }
  });
  const crossings = [], seen = new Set();
  roads.forEach((B, bi) => {
    if (!B.bridge) return;
    const nb = B.pts.length / 2;
    for (let i = 0; i < nb - 1; i++) {
      const ax = B.pts[2 * i], az = B.pts[2 * i + 1], bx = B.pts[2 * i + 2], bz = B.pts[2 * i + 3];
      for (let gz = Math.floor(Math.min(az, bz) / SC); gz <= Math.floor(Math.max(az, bz) / SC); gz++)
        for (let gx = Math.floor(Math.min(ax, bx) / SC); gx <= Math.floor(Math.max(ax, bx) / SC); gx++) {
          const list = sgrid.get(skey(gx, gz));
          if (!list) continue;
          for (let q = 0; q < list.length; q += 2) {
            const ni = list[q], j = list[q + 1], N = roads[ni];
            // a bridge over a ground road, or over another bridge when OSM's layer says which is on top
            if (ni === bi || (N.bridge && N.layer === B.layer)) continue;
            const cx = N.pts[2 * j], cz = N.pts[2 * j + 1], dx = N.pts[2 * j + 2], dz = N.pts[2 * j + 3];
            const den = (bx - ax) * (dz - cz) - (bz - az) * (dx - cx);
            if (Math.abs(den) < 1e-9) continue;
            const u = ((cx - ax) * (dz - cz) - (cz - az) * (dx - cx)) / den, v = ((cx - ax) * (bz - az) - (cz - az) * (bx - ax)) / den;
            if (u < 0 || u > 1 || v < 0 || v > 1) continue;
            const Lb = B.len[i] + (B.len[i + 1] - B.len[i]) * u, Ln = N.len[j] + (N.len[j + 1] - N.len[j]) * v;
            // touching at an end is a junction, not a crossing
            if (Lb < 2 || Lb > B.len[nb - 1] - 2 || Ln < 2 || Ln > N.len[N.len.length - 1] - 2) continue;
            const key = Math.min(bi, ni) + ':' + Math.max(bi, ni);
            if (seen.has(key)) continue;
            seen.add(key);
            const l1 = Math.hypot(bx - ax, bz - az) || 1, l2 = Math.hypot(dx - cx, dz - cz) || 1;
            const sin = Math.abs(den) / (l1 * l2), kb = u < 0.5 ? i : i + 1, kn = v < 0.5 ? j : j + 1;
            if (N.layer > B.layer) crossings.push({ bi: ni, kb: kn, ni: bi, kn: kb, sin });
            else crossings.push({ bi, kb, ni, kn, sin });
          }
        }
    }
  });
  for (let it = 0; it < 2; it++) for (const x of crossings) {
    const B = roads[x.bi], N = roads[x.ni];
    const need = isRail(N) ? 7 : N.rank <= 3 ? 5 : 6.2;
    const deficit = need - (B.h[x.kb] - N.h[x.kn]);
    if (deficit < 0.2) continue;
    const fo = isRail(N) ? 0.8 : isRail(B) ? 0.35 : 0.6, up = deficit * fo, down = deficit - up;
    const across = (w) => (w + 4) / Math.max(x.sin, 0.35);
    spread(x.bi, x.kb, up, across(N.hw), isRail(B) ? clamp(60 * up, 120, 240) : clamp(22 * up, 50, 140), new Set([x.ni]));
    if (down > 0.05) spread(x.ni, x.kn, -down, across(B.hw), isRail(N) ? clamp(60 * down, 100, 220) : clamp(22 * down, 40, 120), new Set([x.bi]));
  }

  // --- water: bridges clear it, roads crossing a stream sit just above it
  if (!wm) return;
  const cell = (x, z) => { const gi = Math.round(x / c + M), gj = Math.round(z / c + M); return gi < 0 || gj < 0 || gi >= G || gj >= G ? -1 : gj * G + gi; };
  roads.forEach((R, ri) => {
    const n = R.pts.length / 2;
    let run = null;
    const runs = [];
    for (let k = 0; k < n; k++) {
      const q = cell(R.pts[2 * k], R.pts[2 * k + 1]);
      if (q >= 0 && wm.isW[q]) { if (!run) run = { k0: k, k1: k, sea: false, lvl: -Infinity }; run.k1 = k; run.sea = run.sea || wm.isW[q] === 2; run.lvl = Math.max(run.lvl, wm.L[q]); }
      else if (run) { runs.push(run); run = null; }
    }
    if (run) runs.push(run);
    for (const w of runs) {
      const span = R.len[w.k1] - R.len[w.k0];
      let clr, flat, ramp;
      if (R.bridge) {
        clr = w.sea ? 5 : span < 15 ? 1.8 : span < 40 ? 3 : 4.5;
        flat = span / 2 + 3; ramp = isRail(R) ? 150 : 70;
      } else {
        if (w.sea || span > 60) continue;          // causeways / seafront roads: leave them alone
        clr = 0.5; flat = span / 2 + 2; ramp = 25;
      }
      let worst = 0, wk = w.k0;
      for (let k = w.k0; k <= w.k1; k++) { const d = w.lvl + clr - R.h[k]; if (d > worst) { worst = d; wk = k; } }
      if (worst < 0.1) continue;
      spread(ri, Math.round((w.k0 + w.k1) / 2), worst, flat, ramp, null);
    }
  });
}

// ------------------------------------------------ bridge landmarks
// Put each bridge landmark on its road (nearest real bridge deck, model x along it) so the building pass can
// clear OSM "buildings" that sit on the deck - Tower Bridge's towers and walkways are mapped that way.
function alignBridgeLandmarks(lms, roads) {
  for (const L of lms) {
    if (!L.bridge) continue;
    let best = null;
    for (const r of roads) {
      if (!r.bridge || CLASS_LIST[r.cls] === 'path') continue;
      for (let i = 0; i < r.pts.length / 2 - 1; i++) {
        const ax = r.pts[2 * i], az = r.pts[2 * i + 1], dx = r.pts[2 * i + 2] - ax, dz = r.pts[2 * i + 3] - az, l2 = dx * dx + dz * dz || 1e-9;
        const t = Math.max(0, Math.min(1, ((L.x - ax) * dx + (L.z - az) * dz) / l2)), qx = ax + dx * t, qz = az + dz * t, d = Math.hypot(L.x - qx, L.z - qz);
        if (d < 80 && (!best || d < best.d)) best = { d, x: qx, z: qz, phi: Math.atan2(dz, dx) };   // same reach as the main-thread snap
      }
    }
    if (!best) { L.bridgeOk = false; continue; }
    let rot = -best.phi;
    if (Math.abs(Math.atan2(Math.sin(rot - L.rot), Math.cos(rot - L.rot))) > Math.PI / 2) rot += Math.PI;
    L.x = best.x; L.z = best.z; L.rot = rot; L.bridgeOk = true;
  }
}

// ------------------------------------------------ landmark pads
// A landmark stands on a level pad: the ground under its footprint (plus a small apron) is set to the middle
// height found there and blends back into the natural slope over ~20 m. Done on the raw DEM before roads are
// built, so roads, water and props all follow it. Without it, summit statues sank into the hillside (the base
// goes at the lowest ground) and slopes ran straight through plinths and plazas. Carvings hung off a mountain
// (anchor "centre") keep the natural ground.
function padLandmarks(H, lms, c, wm) {
  const wet = k => !!(wm && wm.isW[k]);
  for (const L of lms) {
    if (L.anchor === 'centre' || L.bridge) continue;
    const cs = Math.cos(L.rot), sn = Math.sin(L.rot);
    const ax = L.pax || L.hx + 4, az = L.paz || L.hz + 4, BL = 20, R = Math.hypot(ax, az) + BL;
    const i0 = Math.max(0, Math.floor((L.x - R) / c + M)), i1 = Math.min(G - 1, Math.ceil((L.x + R) / c + M));
    const j0 = Math.max(0, Math.floor((L.z - R) / c + M)), j1 = Math.min(G - 1, Math.ceil((L.z + R) / c + M));
    if (i0 > i1 || j0 > j1) continue;
    // model-space distance outside the footprint rectangle (0 inside)
    const outside = (gi, gj) => {
      const ex = (gi - M) * c - L.x, ez = (gj - M) * c - L.z, mx = ex * cs - ez * sn, mz = ex * sn + ez * cs;
      return Math.hypot(Math.max(0, Math.abs(mx) - ax), Math.max(0, Math.abs(mz) - az));
    };
    const inside = [];
    let wetN = 0;
    for (let gj = j0; gj <= j1; gj++) for (let gi = i0; gi <= i1; gi++) if (outside(gi, gj) === 0) { const k = gj * G + gi; if (wet(k)) wetN++; else inside.push(H[k]); }
    if (wetN > inside.length * 3) continue;          // mostly over water (a span, a pier): leave the ground alone
    if (!inside.length) {           // footprint smaller than a cell: use the ground at the centre
      const gi = Math.round(L.x / c + M), gj = Math.round(L.z / c + M);
      if (gi < 0 || gj < 0 || gi >= G || gj >= G) continue;
      inside.push(H[gj * G + gi]);
    }
    inside.sort((a, b) => a - b);
    const lvl = inside[inside.length >> 1];
    for (let gj = j0; gj <= j1; gj++) for (let gi = i0; gi <= i1; gi++) {
      const d = outside(gi, gj);
      if (d >= BL) continue;
      const u = d / BL, w = 1 - u * u * (3 - 2 * u), k = gj * G + gi;
      if (wet(k)) continue;                          // never fill in water
      H[k] += (lvl - H[k]) * w;
    }
  }
}

// ------------------------------------------------ bridges & culverts
// Solid structures under/around bridge decks (the deck surface itself is the road mesh) and concrete
// culverts where small streams pass under roads. Non-indexed, vertex-coloured; drawn double-sided so the
// face normal always ends up pointing at the viewer.
function mesher() {
  const pos = [], nor = [], col = [];
  const lin = c => [Math.pow(c[0], 2.2), Math.pow(c[1], 2.2), Math.pow(c[2], 2.2)];
  const tri = (a, b, cc, color) => {
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [cc[0] - a[0], cc[1] - a[1], cc[2] - a[2]];
    let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(n[0], n[1], n[2]) || 1; n = [n[0] / l, n[1] / l, n[2] / l];
    const cl = lin(color);
    for (const v of [a, b, cc]) { pos.push(v[0], v[1], v[2]); nor.push(n[0], n[1], n[2]); col.push(cl[0], cl[1], cl[2]); }
  };
  const quad = (a, b, cc, d, color) => { tri(a, b, cc, color); tri(a, cc, d, color); };
  // box from a centre and three (unit) axes with half sizes
  const boxAxes = (c0, ax, ay, az, hx, hy, hz, color) => {
    const P = (sx, sy, sz) => [c0[0] + ax[0] * hx * sx + ay[0] * hy * sy + az[0] * hz * sz, c0[1] + ax[1] * hx * sx + ay[1] * hy * sy + az[1] * hz * sz, c0[2] + ax[2] * hx * sx + ay[2] * hy * sy + az[2] * hz * sz];
    const f = [[[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]], [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, -1]],
               [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]], [[-1, -1, 1], [-1, -1, -1], [1, -1, -1], [1, -1, 1]],
               [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]]];
    for (const q of f) quad(P(...q[0]), P(...q[1]), P(...q[2]), P(...q[3]), color);
  };
  const box = (cx, cy, cz, w, h, d, yaw, color) => boxAxes([cx, cy, cz], [Math.cos(yaw), 0, -Math.sin(yaw)], [0, 1, 0], [Math.sin(yaw), 0, Math.cos(yaw)], w / 2, h / 2, d / 2, color);
  const beam = (A, B, t, color) => {
    const d = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], L = Math.hypot(d[0], d[1], d[2]) || 1, az = [d[0] / L, d[1] / L, d[2] / L];
    const up = Math.abs(az[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    let ax = [up[1] * az[2] - up[2] * az[1], up[2] * az[0] - up[0] * az[2], up[0] * az[1] - up[1] * az[0]];
    const la = Math.hypot(...ax); ax = ax.map(v => v / la);
    const ay = [az[1] * ax[2] - az[2] * ax[1], az[2] * ax[0] - az[0] * ax[2], az[0] * ax[1] - az[1] * ax[0]];
    boxAxes([(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2], ax, ay, az, t / 2, t / 2, L / 2, color);
  };
  const cylinder = (A, B, r, seg, color, cap) => {
    const d = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], L = Math.hypot(...d), az = d.map(v => v / L);
    const up = Math.abs(az[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    let ax = [up[1] * az[2] - up[2] * az[1], up[2] * az[0] - up[0] * az[2], up[0] * az[1] - up[1] * az[0]];
    const la = Math.hypot(...ax); ax = ax.map(v => v / la);
    const ay = [az[1] * ax[2] - az[2] * ax[1], az[2] * ax[0] - az[0] * ax[2], az[0] * ax[1] - az[1] * ax[0]];
    const ring = (C0, k) => { const a = k / seg * Math.PI * 2, ca = Math.cos(a) * r, sa = Math.sin(a) * r; return [C0[0] + ax[0] * ca + ay[0] * sa, C0[1] + ax[1] * ca + ay[1] * sa, C0[2] + ax[2] * ca + ay[2] * sa]; };
    for (let k = 0; k < seg; k++) quad(ring(A, k), ring(B, k), ring(B, k + 1), ring(A, k + 1), color);
    if (cap) for (const [C0, cc] of [[A, cap], [B, cap]]) for (let k = 0; k < seg; k++) tri(C0, ring(C0, k), ring(C0, k + 1), cc);
  };
  // sweep a 2D profile (u across to the right, v up from the deck) along a polyline
  const extrude = (P, profile, color, capEnds) => {
    const at = (p, u, v) => [p.x - p.tz * u, p.y + v, p.z + p.tx * u];
    for (let i = 0; i < P.length - 1; i++) for (let k = 0; k < profile.length; k++) {
      const [u0, v0] = profile[k], [u1, v1] = profile[(k + 1) % profile.length];
      quad(at(P[i], u0, v0), at(P[i], u1, v1), at(P[i + 1], u1, v1), at(P[i + 1], u0, v0), color);
    }
    if (capEnds) for (const p of [P[0], P[P.length - 1]]) for (let k = 1; k < profile.length - 1; k++)
      tri(at(p, ...profile[0]), at(p, ...profile[k]), at(p, ...profile[k + 1]), color);
  };
  return { tri, quad, box, beam, cylinder, extrude, empty: () => !pos.length,
           out: () => ({ positions: new Float32Array(pos), normals: new Float32Array(nor), colors: new Float32Array(col) }) };
}

const CONCRETE = [0.66, 0.65, 0.62], CONCRETE_DARK = [0.5, 0.49, 0.47], GALV = [0.66, 0.68, 0.7];
function buildStructures(roads, waterway, H, c, S) {
  const m = mesher(), colSegs = [];
  // is a ground-level road / railway within `pad` metres of (x, z)?
  const onGroundRoad = (x, z, pad) => {
    for (const q of roads) {
      if (q.bridge) continue;
      for (let i = 0; i < q.pts.length / 2 - 1; i++) {
        const ax = q.pts[2 * i], az = q.pts[2 * i + 1], dx = q.pts[2 * i + 2] - ax, dz = q.pts[2 * i + 3] - az;
        if (Math.abs(ax - x) > 60 || Math.abs(az - z) > 60) continue;
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz || 1e-9)));
        if (Math.hypot(ax + dx * t - x, az + dz * t - z) < q.hw + pad) return true;
      }
    }
    return false;
  };
  // raised grass islands in roundabouts
  for (const r of roads) if (r.ring) {
    const n = r.pts.length / 2; let cx = 0, cz = 0, rad = 0, hy = 0;
    for (let i = 0; i < n; i++) { cx += r.pts[2 * i]; cz += r.pts[2 * i + 1]; hy += r.h[i]; }
    cx /= n; cz /= n; hy /= n;
    for (let i = 0; i < n; i++) rad += Math.hypot(r.pts[2 * i] - cx, r.pts[2 * i + 1] - cz) / n;
    const ir = rad - r.hw - 0.3;
    if (ir > 1.5 && cx >= 0 && cx < S && cz >= 0 && cz < S) m.cylinder([cx, hy - 0.6, cz], [cx, hy + 0.2, cz], ir, 28, CONCRETE, [0.3, 0.42, 0.2]);
  }
  for (const r of roads) {
    if (!r.bridge) continue;
    const rail = CLASS_LIST[r.cls] === 'rail';
    const lift = ROAD_LIFT + r.rank * 0.003, hw = r.hw;
    const sty = hash2(Math.round(r.pts[0]), Math.round(r.pts[1]));
    const total = r.len[r.len.length - 1];
    const style = rail ? (sty < 0.5 ? 'truss' : 'girder') : r.rank >= 6 ? 'concrete' : sty < 0.5 ? 'girder' : sty < 0.68 && total < 80 ? 'truss' : 'concrete';
    const paint = [[0.36, 0.44, 0.4], [0.48, 0.33, 0.24], [0.3, 0.38, 0.5], [0.55, 0.56, 0.55]][Math.floor(hash2(Math.round(r.pts[0] * 3), 7) * 4)];
    const T = r.rank >= 6 ? 1.3 : 0.65;
    for (const piece of clipPieces(r, S)) {
      const P = piece.pts.map((p, i, arr) => {
        const a = arr[Math.max(0, i - 1)], b = arr[Math.min(arr.length - 1, i + 1)], l = Math.hypot(b.x - a.x, b.z - a.z) || 1;
        return { x: p.x, z: p.z, y: p.h + lift, tx: (b.x - a.x) / l, tz: (b.z - a.z) / l, len: p.len };
      });
      if (P.length < 2) continue;
      // deck slab
      m.extrude(P, [[-hw - 0.4, -0.03], [hw + 0.4, -0.03], [hw + 0.4, -T], [-hw - 0.4, -T]], CONCRETE, true);
      // parapets: concrete barriers on big roads, steel railings otherwise
      if (r.rank >= 6 || style === 'concrete') {
        m.extrude(P, [[-hw - 0.4, -0.03], [-hw + 0.05, -0.03], [-hw - 0.05, 0.25], [-hw - 0.15, 0.85], [-hw - 0.4, 0.85]], CONCRETE, true);
        m.extrude(P, [[hw + 0.4, 0.85], [hw + 0.15, 0.85], [hw + 0.05, 0.25], [hw - 0.05, -0.03], [hw + 0.4, -0.03]], CONCRETE, true);
      } else {
        for (const sgn of [-1, 1]) {
          m.extrude(P, [[sgn * (hw + 0.25) - 0.04, 0.95], [sgn * (hw + 0.25) + 0.04, 0.95], [sgn * (hw + 0.25) + 0.04, 1.05], [sgn * (hw + 0.25) - 0.04, 1.05]], GALV, true);
          m.extrude(P, [[sgn * (hw + 0.25) - 0.04, 0.5], [sgn * (hw + 0.25) + 0.04, 0.5], [sgn * (hw + 0.25) + 0.04, 0.6], [sgn * (hw + 0.25) - 0.04, 0.6]], GALV, true);
        }
      }
      // walk the deck for posts, girders, truss members and piers
      const span = style === 'concrete' ? 32 : 24, L0 = P[0].len, L1 = P[P.length - 1].len;
      let j = 0;
      const pt = d => { while (j < P.length - 2 && P[j + 1].len < d) j++; while (j > 0 && P[j].len > d) j--; const a = P[j], b = P[j + 1], f = Math.max(0, Math.min(1, (d - a.len) / ((b.len - a.len) || 1))); return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, z: a.z + (b.z - a.z) * f, tx: a.tx, tz: a.tz }; };
      const W = (p, u, v) => [p.x - p.tz * u, p.y + v, p.z + p.tx * u];
      if (style === 'girder') {
        const nG = Math.max(3, Math.round(2 * hw / 2.4));
        for (let g = 0; g < nG; g++) {
          const u = -hw + 0.6 + g * (2 * hw - 1.2) / (nG - 1);
          m.extrude(P, [[u - 0.06, -T], [u + 0.06, -T], [u + 0.06, -T - 1.0], [u - 0.06, -T - 1.0]], paint, true);
          m.extrude(P, [[u - 0.22, -T - 1.0], [u + 0.22, -T - 1.0], [u + 0.22, -T - 1.1], [u - 0.22, -T - 1.1]], paint, true);
        }
      }
      if (style === 'truss') {
        const top = 3.4;
        for (const sgn of [-1, 1]) {
          const u = sgn * (hw + 0.45);
          m.extrude(P, [[u - 0.15, top - 0.15], [u + 0.15, top - 0.15], [u + 0.15, top + 0.15], [u - 0.15, top + 0.15]], paint, true);
          m.extrude(P, [[u - 0.15, -0.1], [u + 0.15, -0.1], [u + 0.15, 0.25], [u - 0.15, 0.25]], paint, true);
          let k = 0;
          for (let d = L0; d <= L1 + 0.01; d += 5, k++) {
            const a = pt(d);
            m.beam(W(a, u, 0.1), W(a, u, top), 0.2, paint);
            if (d + 5 <= L1 + 0.01) { const b = pt(d + 5); m.beam(k % 2 ? W(a, u, 0.1) : W(a, u, top), k % 2 ? W(b, u, top) : W(b, u, 0.1), 0.16, paint); }
          }
        }
        for (let d = L0; d <= L1 + 0.01; d += 10) { const a = pt(d); m.beam(W(a, -hw - 0.45, top), W(a, hw + 0.45, top), 0.16, paint); }
      }
      if (style !== 'concrete' && r.rank < 6) for (let d = L0; d <= L1; d += 2) {
        const a = pt(d);
        for (const sgn of [-1, 1]) m.box(...W(a, sgn * (hw + 0.25), 0.5), 0.1, 1.05, 0.1, Math.atan2(a.tx, a.tz), GALV);
      }
      // piers between spans, abutments at real ends
      const under = T + (style === 'girder' ? 1.1 : 0);
      const nSpan = Math.max(1, Math.round((L1 - L0) / span));
      for (let k = 1; k < nSpan; k++) {
        // slide a pier along the deck until it is clear of any road or railway below
        let a = null;
        for (const shift of [0, 4, -4, 8, -8, 12, -12]) {
          const dd = L0 + (L1 - L0) * k / nSpan + shift;
          if (dd <= L0 + 3 || dd >= L1 - 3) continue;
          const q = pt(dd);
          if (!onGroundRoad(q.x, q.z, r.hw * 0.6 + 2)) { a = q; break; }
        }
        if (!a) continue;
        const gy = sampleTri(H, a.x, a.z, c), topY = a.y - under, hgt = topY - gy;
        if (hgt < 1.2) continue;
        const yaw = Math.atan2(a.tx, a.tz);
        if (r.rank >= 6) {
          m.box(a.x, gy - 0.5 + (hgt + 0.5) / 2, a.z, 2 * hw * 0.6, hgt + 0.5, 1.4, yaw, CONCRETE_DARK);
          m.box(a.x, topY - 0.45, a.z, 2 * hw + 0.6, 0.9, 1.8, yaw, CONCRETE);
        } else {
          for (const sgn of [-0.5, 0.5]) { const p = W(a, sgn * hw, 0); m.cylinder([p[0], gy - 0.5, p[2]], [p[0], topY, p[2]], 0.45, 10, CONCRETE_DARK); }
          m.box(a.x, topY - 0.35, a.z, 2 * hw + 0.4, 0.7, 1.0, yaw, CONCRETE);
        }
      }
      for (const [use, a] of [[piece.startCap, P[0]], [piece.endCap, P[P.length - 1]]]) {
        if (!use) continue;
        const gy = sampleTri(H, a.x, a.z, c), hgt = a.y - gy;
        m.box(a.x, a.y - T - Math.max(1, hgt) / 2 - 0.2, a.z, 2 * hw + 1.4, Math.max(1, hgt) + 0.4, 2.2, Math.atan2(a.tx, a.tz), CONCRETE_DARK);
      }
      // edge collisions (with deck height so cars underneath don't hit them)
      for (let i = 0; i < P.length - 1; i++) for (const sgn of [-1, 1]) {
        const a = W(P[i], sgn * (hw + 0.2), 0), b = W(P[i + 1], sgn * (hw + 0.2), 0);
        colSegs.push(a[0], a[2], b[0], b[2], a[1]);
      }
    }
  }
  // culverts where a stream / ditch / drain passes under a road
  if (waterway) {
    const k = S / waterway.extent;
    const segs = [];
    for (const r of roads) if (!r.bridge && CLASS_LIST[r.cls] !== 'rail') for (let i = 0; i < r.pts.length / 2 - 1; i++) segs.push(r, i);
    for (const fb of waterway.features) {
      const f = readFeature(fb, waterway), w = WATERWAY_WIDTH[f.props.class];
      if (f.type !== 2 || !w || w > 4 || f.props.brunnel) continue;
      for (const part of f.parts) for (let a = 0; a < part.length - 2; a += 2) {
        const ax = part[a] * k, az = part[a + 1] * k, bx = part[a + 2] * k, bz = part[a + 3] * k;
        for (let q = 0; q < segs.length; q += 2) {
          const r = segs[q], i = segs[q + 1];
          const cx = r.pts[2 * i], cz = r.pts[2 * i + 1], dx = r.pts[2 * i + 2], dz = r.pts[2 * i + 3];
          const den = (bx - ax) * (dz - cz) - (bz - az) * (dx - cx);
          if (Math.abs(den) < 1e-9) continue;
          const t = ((cx - ax) * (dz - cz) - (cz - az) * (dx - cx)) / den, u = ((cx - ax) * (bz - az) - (cz - az) * (bx - ax)) / den;
          if (t < 0 || t > 1 || u < 0 || u > 1) continue;
          const X = ax + (bx - ax) * t, Zp = az + (bz - az) * t;
          if (X < 0 || X >= S || Zp < 0 || Zp >= S) continue;
          const roadY = r.h[i] + (r.h[i + 1] - r.h[i]) * u, rp = w <= 2 ? 0.45 : 0.75;
          const wl = Math.hypot(bx - ax, bz - az) || 1, wx = (bx - ax) / wl, wz = (bz - az) / wl;
          const reach = r.hw + 2.6, cy = roadY - 1.1 - rp;
          for (const sgn of [-1, 1]) {
            const ex = X + wx * reach * sgn, ez = Zp + wz * reach * sgn, ey = Math.min(cy, sampleTri(H, ex, ez, c) + rp * 0.6);
            m.cylinder([X + wx * (reach - 3) * sgn, ey, Zp + wz * (reach - 3) * sgn], [ex + wx * 0.4 * sgn, ey, ez + wz * 0.4 * sgn], rp + 0.08, 12, CONCRETE, [0.05, 0.05, 0.05]);
            m.box(ex, ey + 0.1, ez, 2 * rp + 1.6, 2 * rp + 0.9, 0.3, Math.atan2(wx, wz), CONCRETE_DARK);
          }
        }
      }
    }
  }
  return m.empty() ? null : { mesh: m.out(), col: new Float32Array(colSegs) };
}

// ------------------------------------------------ land classification + aerial photo
// Sources (Microsoft Planetary Computer mosaics, ids registered by the page):
//   US:        NAIP photo (~1 m) + NAIP NDVI (from its infrared band) + USDA Cropland Data Layer
//   elsewhere: Sentinel-2 photo + Sentinel-2 NDVI + ESA WorldCover
const PC = 'https://planetarycomputer.microsoft.com/api/data/v1/mosaic/';
const enc = encodeURIComponent;
// ground material ids (also texture-array layers 0..6; 7 = water, photo only)
const GRASS = 0, CROP = 1, SOIL = 2, FOREST = 3, MUD = 4, GRAVEL = 5, URBAN = 6, WATER = 7;
const N = 512; // classification grid per tile (~3.7 m at 40° lat)

function cdlMat(c) {
  if (c === 111 || c === 83 || c === 92) return WATER;
  if (c === 112 || c === 131 || c === 65) return GRAVEL;
  if (c === 121) return GRASS;                                  // developed, open space (lawns, parks)
  if (c === 82 || c === 122 || c === 123 || c === 124) return URBAN;
  if (c === 141 || c === 142 || c === 143 || c === 63 || c === 190 || c === 87) return FOREST;
  if (c === 152 || c === 64) return SOIL;                       // shrubland: dry ground between shrubs
  if (c === 176 || c === 62 || c === 36 || c === 37 || c === 58 || c === 59 || c === 60) return GRASS;
  if (c === 195) return MUD;                                    // herbaceous wetland
  if (c === 61) return SOIL;                                    // fallow / idle cropland
  if (c === 0 || c === 81 || c === 88 || c === 255) return -1;  // no data
  return CROP;                                                  // every actual crop
}
function wcMat(c) {
  return { 10: FOREST, 20: SOIL, 30: GRASS, 40: CROP, 50: URBAN, 60: GRAVEL, 70: GRAVEL, 80: WATER, 90: MUD, 95: FOREST, 100: GRASS }[c] ?? -1;
}
// sharpen the coarse class with the photo's own NDVI (n) and brightness (lum)
function refine(m, n, lum) {
  if (Number.isNaN(n)) return m < 0 ? GRASS : m;
  if (m < 0) m = n > 0.5 ? FOREST : n > 0.2 ? GRASS : SOIL;
  const wet = n < -0.08 && lum < 0.22;
  switch (m) {
    case CROP: return wet ? WATER : CROP;
    case GRASS: return wet ? WATER : n < 0.08 ? (lum > 0.45 ? GRAVEL : SOIL) : GRASS;
    case SOIL: return wet ? WATER : n > 0.4 ? GRASS : SOIL;
    case FOREST: return n < 0.12 ? (lum > 0.35 ? GRAVEL : SOIL) : FOREST;
    case URBAN: return n > 0.35 ? GRASS : URBAN;
    case WATER: return n > 0.2 ? MUD : WATER;
    case MUD: return n > 0.4 ? GRASS : MUD;
    case GRAVEL: return n > 0.35 ? GRASS : GRAVEL;
  }
  return m;
}

async function fetchPixels(url, size) {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url);
      if (res.status === 404 || res.status === 204) return null;
      if (!res.ok) throw new Error(res.status);
      const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
      const ctx = new OffscreenCanvas(size, size).getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bmp, 0, 0, size, size); bmp.close();
      return ctx.getImageData(0, 0, size, size).data;
    } catch { await sleep(600 * (attempt + 1)); }
  }
  return null;
}
function validFraction(px, byAlpha) {
  let ok = 0, n = 0;
  for (let i = 0; i < px.length; i += 4 * 37, n++) if (byAlpha ? px[i + 3] > 0 : px[i] + px[i + 1] + px[i + 2] > 24) ok++;
  return ok / n;
}

async function buildImagery(tx, ty, lod, img) {
  const ids = img.ids, path = `tiles/WebMercatorQuad/${Z}/${tx}/${ty}`;
  const naipPhoto = s => ids.naip && `${PC}${ids.naip}/${path}@${s}x.jpg?collection=naip&assets=image&asset_bidx=image%7C1%2C2%2C3`;
  const naipNdvi = ids.naip && `${PC}${ids.naip}/${path}@2x.png?collection=naip&assets=image&expression=${enc('(image_b4-image_b1)/(image_b4+image_b1)')}&rescale=-1%2C1`;
  const cdl = ids.cdl && `${PC}${ids.cdl}/${path}@1x.png?collection=usda-cdl&assets=cropland`;
  const s2Photo = s => ids.s2 && `${PC}${ids.s2}/${path}@${s}x.jpg?collection=sentinel-2-l2a&assets=visual&nodata=0`;
  const s2Ndvi = ids.s2 && `${PC}${ids.s2}/${path}@2x.png?collection=sentinel-2-l2a&assets=B08&assets=B04&expression=${enc('(B08-B04)/(B08+B04)')}&asset_as_band=true&rescale=-1%2C1`;
  const wc = ids.wc && `${PC}${ids.wc}/${path}@1x.png?collection=esa-worldcover&assets=map`;
  const get = (url, size) => url ? fetchPixels(url, size) : Promise.resolve(null);

  if (lod !== 1) { // far tiles: just a low-res photo
    let photo = img.us ? await get(naipPhoto(1), 256) : null;
    let source = 'NAIP';
    if (!photo || validFraction(photo, false) < 0.5) { photo = await get(s2Photo(1), 256); source = 'Sentinel-2'; }
    if (!photo) return null;
    for (let i = 3; i < photo.length; i += 4) photo[i] = 128;
    return { photo: new Uint8Array(photo.buffer), photoSize: 256, source };
  }

  let photo = null, ndvi = null, cls = null, size = 1024, source = 'NAIP', clsKind = 'cdl';
  if (img.us) [photo, ndvi, cls] = await Promise.all([get(naipPhoto(4), 1024), get(naipNdvi, 512), get(cdl, 256)]);
  if (!photo || !ndvi || validFraction(photo, false) < 0.6) {
    [photo, ndvi] = await Promise.all([get(s2Photo(2), 512), get(s2Ndvi, 512)]);
    size = 512; source = 'Sentinel-2';
  }
  if (!cls || validFraction(cls, true) < 0.5) { cls = await get(wc, 256); clsKind = 'wc'; }
  if (!photo) return null;

  const mat = new Uint8Array(N * N), code = new Uint8Array(N * N);
  const ps = size / N;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    const ci = ((y >> 1) * 256 + (x >> 1)) * 4;
    const c = cls && cls[ci + 3] > 0 ? cls[ci] : 0;
    const n = ndvi && ndvi[i * 4 + 3] > 0 ? ndvi[i * 4] / 255 * 2 - 1 : NaN;
    const pi = (Math.floor(y * ps) * size + Math.floor(x * ps)) * 4;
    const lum = (photo[pi] * 0.3 + photo[pi + 1] * 0.59 + photo[pi + 2] * 0.11) / 255;
    const m0 = c ? (clsKind === 'cdl' ? cdlMat(c) : wcMat(c)) : -1;
    mat[i] = refine(m0, n, lum);
    code[i] = clsKind === 'cdl' ? c : 0;
  }

  // The crop map's 30 m pixels spill over field margins, farmyards and verges: pull crop areas
  // ~7 m in from their edges and turn the margins into grass or bare soil by their greenness.
  const eroded = new Uint8Array(mat);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x;
    if (mat[i] !== CROP) continue;
    let edge = false;
    for (let dy = -2; dy <= 2 && !edge; dy++) for (let dx = -2; dx <= 2; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || xx < 0 || yy >= N || xx >= N) continue;
      if (mat[yy * N + xx] !== CROP) { edge = true; break; }
    }
    if (!edge) continue;
    const n = ndvi && ndvi[i * 4 + 3] > 0 ? ndvi[i * 4] / 255 * 2 - 1 : 0.3;
    eroded[i] = n > 0.25 ? GRASS : SOIL;
  }
  mat.set(eroded);

  // one-hot material weights, softened with a small blur, packed into two RGBA textures
  const W = new Float32Array(N * N * 8), T = new Float32Array(N * N * 8);
  for (let i = 0; i < N * N; i++) W[i * 8 + mat[i]] = 1;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = (y * N + Math.max(0, x - 1)) * 8, b = (y * N + x) * 8, c = (y * N + Math.min(N - 1, x + 1)) * 8;
    for (let k = 0; k < 8; k++) T[b + k] = (W[a + k] + 2 * W[b + k] + W[c + k]) / 4;
  }
  const wa = new Uint8Array(N * N * 4), wb = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = (Math.max(0, y - 1) * N + x) * 8, b = (y * N + x) * 8, c = (Math.min(N - 1, y + 1) * N + x) * 8, o = (y * N + x) * 4;
    for (let k = 0; k < 8; k++) {
      const v = Math.round((T[a + k] + 2 * T[b + k] + T[c + k]) / 4 * 255);
      if (k < 4) wa[o + k] = v; else wb[o + k - 4] = v;
    }
  }

  // photo RGB + NDVI in alpha (used for crop canopy cover)
  const out = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const o = (y * size + x) * 4, ni = (Math.floor(y / ps) * N + Math.floor(x / ps)) * 4;
    out[o] = photo[o]; out[o + 1] = photo[o + 1]; out[o + 2] = photo[o + 2];
    out[o + 3] = ndvi && ndvi[ni + 3] > 0 ? ndvi[ni] : 128;
  }
  return { photo: out, photoSize: size, wa, wb, mat, code, source, clsKind };
}

// ------------------------------------------------ water
// Lakes / rivers / sea from OpenFreeMap polygons + river & stream lines, rasterised onto the height grid.
// The terrain is carved into a bed below the water level and a water surface mesh is built on top.
const WATERWAY_WIDTH = { river: 14, canal: 10, stream: 3, drain: 2, ditch: 2 };
// typical surface current (m/s); OSM waterways are drawn in the direction the water flows
const WATERWAY_SPEED = { river: 1.3, canal: 0.25, stream: 0.9, drain: 0.5, ditch: 0.35 };
function waterMask(vt, H0, c, tileSize) {
  if (!vt.water && !vt.waterway) return null;
  const ctx = new OffscreenCanvas(G, G).getContext('2d', { willReadFrequently: true });
  const toG = (v, k) => v * k / c + M + 0.5; // vertex (gi) sits at canvas pixel centre gi + 0.5
  let drew = false;
  const flowSegs = [];   // [gx0, gz0, gx1, gz1, speed, radius] in grid vertex units
  if (vt.water) {
    const k = tileSize / vt.water.extent;
    for (const fb of vt.water.features) {
      const f = readFeature(fb, vt.water);
      if (f.type !== 3 || f.props.brunnel === 'tunnel') continue;
      ctx.fillStyle = f.props.class === 'ocean' ? '#00ff00' : '#ff0000';
      ctx.beginPath();
      for (const ring of f.parts) {
        ctx.moveTo(toG(ring[0], k), toG(ring[1], k));
        for (let i = 2; i < ring.length; i += 2) ctx.lineTo(toG(ring[i], k), toG(ring[i + 1], k));
        ctx.closePath();
      }
      ctx.fill('evenodd');
      drew = true;
    }
  }
  if (vt.waterway) {
    const k = tileSize / vt.waterway.extent;
    ctx.strokeStyle = '#ff0000'; ctx.lineCap = ctx.lineJoin = 'round';
    for (const fb of vt.waterway.features) {
      const f = readFeature(fb, vt.waterway);
      const w = WATERWAY_WIDTH[f.props.class];
      if (f.type !== 2 || !w || f.props.brunnel === 'tunnel') continue;
      ctx.lineWidth = Math.max(w / c, 1.6);
      ctx.beginPath();
      const sp = WATERWAY_SPEED[f.props.class] || 0.5, rad = Math.max(1.5, w / c / 2 + 1);
      for (const part of f.parts) {
        ctx.moveTo(toG(part[0], k), toG(part[1], k));
        for (let i = 2; i < part.length; i += 2) {
          ctx.lineTo(toG(part[i], k), toG(part[i + 1], k));
          flowSegs.push(toG(part[i - 2], k) - 0.5, toG(part[i - 1], k) - 0.5, toG(part[i], k) - 0.5, toG(part[i + 1], k) - 0.5, sp, rad);
        }
      }
      ctx.stroke();
      drew = true;
    }
  }
  if (!drew) return null;
  const px = ctx.getImageData(0, 0, G, G).data;
  const isW = new Uint8Array(G * G), L = new Float32Array(G * G).fill(NaN), cov = new Float32Array(G * G);
  let any = 0;
  for (let i = 0; i < G * G; i++) {
    cov[i] = Math.max(px[i * 4], px[i * 4 + 1]) / 255;                    // anti-aliased edge coverage
    if (cov[i] < 0.5) continue;
    if (px[i * 4 + 1] >= px[i * 4]) { isW[i] = 2; L[i] = 0; any++; }       // sea level
    else { isW[i] = 1; L[i] = H0[i]; any++; }                              // DEMs are flattened over water
  }
  if (!any) return null;
  // tiny holes in the water (OSM cuts them around bridge piers) would show as little earth islands under every
  // bridge: fill any dry patch smaller than ~2000 m2 that water surrounds completely
  {
    const seen = new Int32Array(G * G), maxCells = Math.max(8, Math.round(2000 / (c * c)));   // 0 unseen, else region id
    let rid = 0;
    for (let s0 = 0; s0 < G * G; s0++) {
      if (isW[s0] || seen[s0]) continue;
      rid++;
      const cells = [s0], ring = new Set(); seen[s0] = rid; let open = false;
      for (let h = 0; h < cells.length && !open; h++) {
        const k = cells[h], i = k % G, j = (k / G) | 0;
        if (i === 0 || j === 0 || i === G - 1 || j === G - 1) { open = true; break; }
        for (const q of [k - 1, k + 1, k - G, k + G]) {
          if (isW[q]) ring.add(q);
          else if (!seen[q]) { seen[q] = rid; cells.push(q); if (cells.length > maxCells) { open = true; break; } }
          else if (seen[q] !== rid) { open = true; break; }     // joins a region already found to be large
        }
      }
      if (open) continue;
      let kind = 1, lv = 0, n = 0;
      for (const q of ring) { if (isW[q] === 2) kind = 2; lv += H0[q]; n++; }
      for (const k of cells) { isW[k] = kind; L[k] = kind === 2 ? 0 : lv / n; cov[k] = 1; }
    }
  }
  // smooth inland water levels a little (DEM noise along rivers)
  for (let pass = 0; pass < 2; pass++) {
    const L2 = new Float32Array(L);
    for (let j = 1; j < G - 1; j++) for (let i = 1; i < G - 1; i++) {
      const k = j * G + i;
      if (isW[k] !== 1) continue;
      let s = 0, n = 0;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) { const q = k + dj * G + di; if (isW[q] === 1) { s += L2[q]; n++; } }
      L[k] = s / n;
    }
  }
  // ---- sane inland water levels. The DEM under a water polygon is only roughly flat: near the banks it climbs
  // the valley sides, and odd polygons (reservoirs on slopes, mis-mapped areas) follow the hillside, so the water
  // sheet climbed hills and gorge walls. Instead:
  //  - lakes and ponds (no river line through them) get one flat level,
  //  - rivers take their level from the middle of the channel, flat from bank to bank,
  //  - nothing may rise faster than a river can (6% per cell, measured along the water),
  //  - water that ends up well below the ground is not really there at that level and is dropped.
  {
    const comp = new Int32Array(G * G).fill(-1), comps = [];
    for (let s = 0; s < G * G; s++) {
      if (isW[s] !== 1 || comp[s] >= 0) continue;
      const cells = [s]; comp[s] = comps.length;
      for (let h = 0; h < cells.length; h++) {
        const k = cells[h], i = k % G;
        for (const q of [i > 0 ? k - 1 : -1, i < G - 1 ? k + 1 : -1, k - G, k + G]) if (q >= 0 && q < G * G && isW[q] === 1 && comp[q] < 0) { comp[q] = comps.length; cells.push(q); }
      }
      comps.push({ cells, river: false });
    }
    for (let o = 0; o < flowSegs.length; o += 6) for (let t = 0; t <= 4; t++) {
      const gi = Math.round(flowSegs[o] + (flowSegs[o + 2] - flowSegs[o]) * t / 4), gj = Math.round(flowSegs[o + 1] + (flowSegs[o + 3] - flowSegs[o + 1]) * t / 4);
      if (gi >= 0 && gj >= 0 && gi < G && gj < G && comp[gj * G + gi] >= 0) comps[comp[gj * G + gi]].river = true;
    }
    // distance from the bank, for carrying river levels out from the middle
    const dist = new Int32Array(G * G).fill(0), q0 = [];
    for (let k = 0; k < G * G; k++) {
      if (isW[k] !== 1) continue;
      const i = k % G, j = (k / G) | 0;
      if (i === 0 || j === 0 || i === G - 1 || j === G - 1 || isW[k - 1] !== 1 || isW[k + 1] !== 1 || isW[k - G] !== 1 || isW[k + G] !== 1) { dist[k] = 1; q0.push(k); }
    }
    for (let h = 0; h < q0.length; h++) {
      const k = q0[h];
      for (const q of [k - 1, k + 1, k - G, k + G]) if (q >= 0 && q < G * G && isW[q] === 1 && !dist[q]) { dist[q] = dist[k] + 1; q0.push(q); }
    }
    for (const cp of comps) {
      if (!cp.river) {
        const hs = cp.cells.map(k => L[k]).sort((a, b) => a - b), lvl = hs[Math.floor(hs.length * 0.2)];
        for (const k of cp.cells) L[k] = lvl;
        continue;
      }
      let maxD = 0;
      for (const k of cp.cells) maxD = Math.max(maxD, dist[k]);
      const byD = Array.from({ length: maxD + 1 }, () => []);
      for (const k of cp.cells) byD[dist[k]].push(k);
      for (let d = maxD - 1; d >= 1; d--) for (const k of byD[d]) {
        let s = 0, n = 0;
        for (const q of [k - 1, k + 1, k - G, k + G, k - G - 1, k - G + 1, k + G - 1, k + G + 1]) if (q >= 0 && q < G * G && isW[q] === 1 && dist[q] > d) { s += L[q]; n++; }
        if (n) L[k] = s / n;
      }
    }
    // even out the DEM's ripples along rivers (each wobble would otherwise read as rapids)
    for (let pass = 0; pass < 12; pass++) {
      const L2 = new Float32Array(L);
      for (let j = 1; j < G - 1; j++) for (let i = 1; i < G - 1; i++) {
        const k = j * G + i;
        if (isW[k] !== 1 || !comps[comp[k]].river) continue;
        let s = 0, n = 0;
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) { const q = k + dj * G + di; if (isW[q] === 1) { s += L2[q]; n++; } }
        L[k] = s / n;
      }
    }
    // slope limit: forward and backward sweeps of L = min(L, neighbour + rise)
    const r1 = 0.06 * c, r2 = r1 * Math.SQRT2;
    for (let it = 0; it < 3; it++) {
      for (let j = 1; j < G - 1; j++) for (let i = 1; i < G - 1; i++) {
        const k = j * G + i;
        if (isW[k] !== 1) continue;
        let m = L[k];
        if (isW[k - 1] === 1) m = Math.min(m, L[k - 1] + r1);
        if (isW[k - G] === 1) m = Math.min(m, L[k - G] + r1);
        if (isW[k - G - 1] === 1) m = Math.min(m, L[k - G - 1] + r2);
        if (isW[k - G + 1] === 1) m = Math.min(m, L[k - G + 1] + r2);
        L[k] = m;
      }
      for (let j = G - 2; j >= 1; j--) for (let i = G - 2; i >= 1; i--) {
        const k = j * G + i;
        if (isW[k] !== 1) continue;
        let m = L[k];
        if (isW[k + 1] === 1) m = Math.min(m, L[k + 1] + r1);
        if (isW[k + G] === 1) m = Math.min(m, L[k + G] + r1);
        if (isW[k + G + 1] === 1) m = Math.min(m, L[k + G + 1] + r2);
        if (isW[k + G - 1] === 1) m = Math.min(m, L[k + G - 1] + r2);
        L[k] = m;
      }
    }
    // peel from the banks inward: a cell whose ground stands well above the level and touches dry land is dropped,
    // which trims sheets climbing a hillside; bulges in the DEM out in the middle of a river stay water
    const high = k => isW[k] === 1 && H0[k] - L[k] > 3;
    const dry = k => { const i = k % G; return (i > 0 && !isW[k - 1]) || (i < G - 1 && !isW[k + 1]) || (k >= G && !isW[k - G]) || (k < G * G - G && !isW[k + G]); };
    const peel = [];
    for (let k = 0; k < G * G; k++) if (high(k) && dry(k)) peel.push(k);
    for (let h = 0; h < peel.length; h++) {
      const k = peel[h];
      if (isW[k] !== 1) continue;
      isW[k] = 0; L[k] = NaN;
      for (const q of [k - 1, k + 1, k - G, k + G]) if (q >= 0 && q < G * G && high(q)) peel.push(q);
    }
  }
  return { isW, L, cov, flowSegs };
}
// carve the bed and build the surface; roads crossing a stream keep the ground (a culvert runs under them)
function buildWater(wm, H, infl, c) {
  const { isW, L, cov, flowSegs } = wm;
  if (infl) for (let k = 0; k < G * G; k++) if (isW[k] === 1 && infl[k] > 0.6) { isW[k] = 0; L[k] = NaN; }
  // distance from the shore (in cells) for every water vertex
  const dist = new Float32Array(G * G).fill(1e9), queue = [];
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const k = j * G + i;
    if (!isW[k]) continue;
    if (i === 0 || j === 0 || i === G - 1 || j === G - 1) continue;
    if (!isW[k - 1] || !isW[k + 1] || !isW[k - G] || !isW[k + G]) { dist[k] = 1; queue.push(k); }
  }
  for (let h = 0; h < queue.length; h++) {
    const k = queue[h], d = dist[k] + 1;
    for (const q of [k - 1, k + 1, k - G, k + G]) if (q >= 0 && q < G * G && isW[q] && dist[q] > d) { dist[q] = d; queue.push(q); }
  }
  // the water surface runs one ring of cells past the shore (terrain hides it there)
  const ring = new Uint8Array(G * G);
  for (let j = 1; j < G - 1; j++) for (let i = 1; i < G - 1; i++) {
    const k = j * G + i;
    if (isW[k]) continue;
    let m = Infinity;
    for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) { const q = k + dj * G + di; if (isW[q]) m = Math.min(m, L[q]); }
    if (m < Infinity) { L[k] = m; ring[k] = 1; }
  }
  // carve the bed (roads keep priority)
  for (let k = 0; k < G * G; k++) {
    if (!isW[k] || (infl && infl[k] > 0.5)) continue;
    const dm = Math.min(dist[k], 60) * c;
    const edge = 0.35 + 0.65 * Math.min(1, (cov[k] - 0.5) * 2);           // shallower right at the edge
    const depth = (isW[k] === 2 ? Math.min(25, 1 + dm * 0.06) : Math.min(4.5, 0.4 + dm * 0.22)) * (dist[k] <= 1 ? edge : 1);
    H[k] = Math.min(H[k], L[k] - depth);
  }
  // bank vertices partly covered by the polygon slope down towards the water, so the shoreline
  // lands where the real edge is instead of stepping along the grid
  for (let k = 0; k < G * G; k++) {
    if (!ring[k] || cov[k] <= 0 || (infl && infl[k] > 0.5)) continue;
    const target = L[k] + 0.3 - cov[k] * 0.9;
    if (H[k] > target) H[k] += (target - H[k]) * Math.min(1, cov[k] * 2.2);
  }
  // current: stamp each waterway line's direction into the cells around it, then spread it across the
  // rest of the river polygon (nearest line wins), slower near the banks and faster down steep reaches
  const flow = new Float32Array(G * G * 2), fw = new Float32Array(G * G), fq = [], fd = new Uint8Array(G * G);
  for (let o = 0; o < flowSegs.length; o += 6) {
    const x0 = flowSegs[o], z0 = flowSegs[o + 1], dx = flowSegs[o + 2] - x0, dz = flowSegs[o + 3] - z0, len = Math.hypot(dx, dz);
    if (len < 1e-6) continue;
    const ux = dx / len, uz = dz / len, sp = flowSegs[o + 4], r = flowSegs[o + 5], R = Math.ceil(r), n = Math.ceil(len * 2);
    for (let q = 0; q <= n; q++) {
      const px = x0 + dx * q / n, pz = z0 + dz * q / n, ci = Math.round(px), cj = Math.round(pz);
      for (let j = cj - R; j <= cj + R; j++) for (let i = ci - R; i <= ci + R; i++) {
        if (i < 0 || j < 0 || i >= G || j >= G || (i - px) ** 2 + (j - pz) ** 2 > r * r) continue;
        const kk = j * G + i;
        if (!isW[kk]) continue;
        flow[kk * 2] += ux * sp; flow[kk * 2 + 1] += uz * sp; fw[kk]++;
      }
    }
  }
  for (let k = 0; k < G * G; k++) if (fw[k]) { flow[k * 2] /= fw[k]; flow[k * 2 + 1] /= fw[k]; fd[k] = 1; fq.push(k); }
  for (let h = 0; h < fq.length; h++) {
    const k = fq[h];
    if (fd[k] > 40) continue;
    const decay = isW[k] === 2 ? 0.9 : 0.995;
    for (const q of [k - 1, k + 1, k - G, k + G]) {
      if (q < 0 || q >= G * G || !isW[q] || fd[q]) continue;
      fd[q] = fd[k] + 1; flow[q * 2] = flow[k * 2] * decay; flow[q * 2 + 1] = flow[k * 2 + 1] * decay; fq.push(q);
    }
  }
  const turb = new Float32Array(G * G);
  // slope of the water surface over +-3 cells, only between points well clear of the banks (DEM levels
  // right at the edge pick up the embankment and would read as rapids)
  const lv = (q, k) => (isW[q] === 1 && dist[q] >= 3 ? L[q] : L[k]);
  for (let j = 3; j < G - 3; j++) for (let i = 3; i < G - 3; i++) {
    const k = j * G + i;
    if (isW[k] !== 1) continue;
    const grad = dist[k] < 3 ? 0 : Math.hypot(lv(k + 3, k) - lv(k - 3, k), lv(k + 3 * G, k) - lv(k - 3 * G, k)) / (6 * c), bank = 0.3 + 0.7 * Math.min(1, (dist[k] - 1) / 3);
    const m = Math.min(3, 1 + Math.max(0, grad - 0.03) * 18) * bank;
    flow[k * 2] *= m; flow[k * 2 + 1] *= m;
    turb[k] = Math.min(1, Math.max(0, grad - 0.03) * 10);   // DEM ripples under a few % are not rapids
  }
  // surface mesh over this tile's own cells
  const vid = new Int32Array(G * G).fill(-1), pos = [], dep = [], wfl = [], idx = [];
  for (let j = M; j <= M + 256; j++) for (let i = M; i <= M + 256; i++) {
    const k = j * G + i;
    if (Number.isNaN(L[k])) continue;
    vid[k] = pos.length / 3;
    pos.push((i - M) * c, L[k] - 0.04, (j - M) * c);
    dep.push(Math.max(0, L[k] - H[k]));
    let sea = isW[k] === 2 ? 1 : 0;
    if (!isW[k]) for (const q of [k - 1, k + 1, k - G, k + G, k - G - 1, k - G + 1, k + G - 1, k + G + 1]) if (isW[q] === 2) sea = 1;
    wfl.push(flow[k * 2], flow[k * 2 + 1], sea, turb[k]);
  }
  for (let j = M; j < M + 256; j++) for (let i = M; i < M + 256; i++) {
    const a = vid[j * G + i], b = vid[j * G + i + 1], cc = vid[(j + 1) * G + i], d = vid[(j + 1) * G + i + 1];
    if (a < 0 || b < 0 || cc < 0 || d < 0) continue;
    idx.push(a, cc, d, a, d, b);
  }
  const level = new Float32Array(G * G).fill(NaN), current = new Int8Array(G * G * 2);
  for (let k = 0; k < G * G; k++) if (isW[k]) {
    level[k] = isW[k] === 2 ? -0.001 : L[k];       // tiny negative marks the sea (for swell)
    current[k * 2] = Math.max(-127, Math.min(127, Math.round(flow[k * 2] * 32)));
    current[k * 2 + 1] = Math.max(-127, Math.min(127, Math.round(flow[k * 2 + 1] * 32)));
  }
  return {
    isW, level, current,
    mesh: idx.length ? { positions: new Float32Array(pos), depth: new Float32Array(dep), wflow: new Float32Array(wfl), indices: new Uint32Array(idx) } : null,
  };
}

// ------------------------------------------------ buildings
// OSM buildings (OpenFreeMap 'building' layer) plus, in the US, FEMA/ORNL USA Structures footprints for
// any structure OSM doesn't have. Extruded walls with a hip roof on house-sized boxes, flat roofs otherwise.
try { importScripts('https://cdn.jsdelivr.net/npm/earcut@2.2.4/dist/earcut.min.js'); } catch (e) { /* fan fallback */ }
const USA_STRUCTURES = 'https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/rest/services/USA_Structures_View/FeatureServer/0/query';
const tileLon = (x) => x / NT * 360 - 180;
const tileLat = (y) => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / NT))) * 180 / Math.PI;
const latToV = lat => { const r = lat * Math.PI / 180; return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2; };

async function fetchUsaStructures(tx, ty, tileSize) {
  const bbox = [tileLon(tx), tileLat(ty + 1), tileLon(tx + 1), tileLat(ty)].map(v => v.toFixed(6)).join(',');
  const out = [];
  for (let offset = 0; offset < 40000; offset += 2000) {
    const url = `${USA_STRUCTURES}?where=1%3D1&geometry=${bbox}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects`
      + `&outFields=HEIGHT,OCC_CLS,OUTBLDG&returnGeometry=true&outSR=4326&geometryPrecision=7&f=geojson&resultRecordCount=2000&resultOffset=${offset}`;
    let json = null;
    for (let attempt = 0; attempt < 2 && !json; attempt++) {
      try { const r = await fetch(url); if (r.ok) json = await r.json(); } catch { await sleep(800); }
    }
    if (!json || !json.features) break;
    for (const f of json.features) {
      if (!f.geometry) continue;
      const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.type === 'MultiPolygon' ? f.geometry.coordinates : [];
      for (const poly of polys) {
        const ring = [];
        for (const [lon, lat] of poly[0]) ring.push(((lon + 180) / 360 * NT - tx) * tileSize, (latToV(lat) * NT - ty) * tileSize);
        out.push({ ring, props: f.properties || {} });
      }
    }
    if (json.features.length < 2000 && !(json.properties && json.properties.exceededTransferLimit)) break;
  }
  return out;
}

function ringArea(r) { let a = 0; for (let i = 0, n = r.length / 2; i < n; i++) { const j = (i + 1) % n; a += r[2 * i] * r[2 * j + 1] - r[2 * j] * r[2 * i + 1]; } return a / 2; }
function ringCentroid(r) { let x = 0, z = 0; const n = r.length / 2; for (let i = 0; i < n; i++) { x += r[2 * i]; z += r[2 * i + 1]; } return [x / n, z / n]; }
function pointInRing(x, z, r) {
  let inside = false;
  for (let i = 0, n = r.length / 2, j = n - 1; i < n; j = i++) {
    const xi = r[2 * i], zi = r[2 * i + 1], xj = r[2 * j], zj = r[2 * j + 1];
    if ((zi > z) !== (zj > z) && x < (xj - xi) * (z - zi) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}
// best-fit rotated rectangle (tested along each edge direction)
function orientedBox(r) {
  const n = r.length / 2; let best = null;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, dx = r[2 * j] - r[2 * i], dz = r[2 * j + 1] - r[2 * i + 1], l = Math.hypot(dx, dz);
    if (l < 0.5) continue;
    const ux = dx / l, uz = dz / l;
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (let k = 0; k < n; k++) { const a = r[2 * k] * ux + r[2 * k + 1] * uz, b = -r[2 * k] * uz + r[2 * k + 1] * ux; a0 = Math.min(a0, a); a1 = Math.max(a1, a); b0 = Math.min(b0, b); b1 = Math.max(b1, b); }
    const area = (a1 - a0) * (b1 - b0);
    if (!best || area < best.area) best = { area, ux, uz, a0, a1, b0, b1 };
  }
  return best;
}

// building types (index is packed into the shader's flag: w = roof + 2 * type)
const BT = { house: 0, apartment: 1, shop: 2, office: 3, warehouse: 4, school: 5, hospital: 6, church: 7, barn: 8, shed: 9, civic: 10, plain: 11 };
const WALL_COLORS = {
  house: [[0.86, 0.82, 0.74], [0.92, 0.9, 0.86], [0.78, 0.74, 0.68], [0.7, 0.62, 0.52], [0.84, 0.8, 0.66], [0.62, 0.36, 0.3], [0.74, 0.76, 0.78], [0.66, 0.72, 0.66], [0.8, 0.72, 0.6]],
  apartment: [[0.88, 0.85, 0.78], [0.93, 0.92, 0.9], [0.74, 0.72, 0.7], [0.6, 0.38, 0.3], [0.82, 0.7, 0.55], [0.7, 0.74, 0.78]],
  shop: [[0.9, 0.87, 0.8], [0.95, 0.94, 0.92], [0.78, 0.76, 0.74], [0.62, 0.4, 0.32], [0.8, 0.72, 0.6]],
  office: [[0.62, 0.64, 0.66], [0.75, 0.75, 0.73], [0.5, 0.53, 0.57], [0.82, 0.8, 0.76]],
  warehouse: [[0.76, 0.77, 0.78], [0.66, 0.68, 0.7], [0.82, 0.79, 0.7], [0.6, 0.68, 0.76], [0.92, 0.92, 0.9], [0.42, 0.52, 0.44]],
  school: [[0.62, 0.36, 0.28], [0.74, 0.58, 0.44], [0.85, 0.8, 0.7], [0.7, 0.48, 0.36]],
  hospital: [[0.94, 0.94, 0.92], [0.9, 0.9, 0.88], [0.86, 0.87, 0.88]],
  church: [[0.7, 0.68, 0.64], [0.93, 0.92, 0.88], [0.6, 0.38, 0.3], [0.8, 0.74, 0.62]],
  barn: [[0.55, 0.14, 0.1], [0.62, 0.18, 0.12], [0.5, 0.45, 0.4], [0.88, 0.86, 0.8], [0.7, 0.72, 0.72]],
  shed: [[0.66, 0.68, 0.7], [0.5, 0.42, 0.34], [0.86, 0.86, 0.84], [0.42, 0.5, 0.42]],
  civic: [[0.82, 0.76, 0.64], [0.66, 0.42, 0.32], [0.86, 0.84, 0.8]],
};
const ROOF_COLORS = {
  pitched: [[0.42, 0.2, 0.15], [0.3, 0.3, 0.32], [0.22, 0.22, 0.24], [0.48, 0.3, 0.22], [0.36, 0.38, 0.4], [0.55, 0.32, 0.22]],
  flat: [[0.5, 0.5, 0.5], [0.38, 0.38, 0.4], [0.6, 0.6, 0.58]],
  metal: [[0.62, 0.64, 0.66], [0.5, 0.52, 0.54], [0.42, 0.2, 0.15], [0.3, 0.38, 0.32]],
};
const BRANDS = [[0.75, 0.12, 0.1], [0.14, 0.3, 0.62], [0.92, 0.45, 0.08], [0.2, 0.45, 0.18], [0.85, 0.72, 0.1], [0.45, 0.15, 0.45], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9]];
function parseColour(c) {
  if (typeof c !== 'string' || !/^#[0-9a-f]{6}$/i.test(c)) return null;
  return [parseInt(c.slice(1, 3), 16) / 255, parseInt(c.slice(3, 5), 16) / 255, parseInt(c.slice(5, 7), 16) / 255];
}
// OpenMapTiles poi class/subclass -> building type
function poiType(p) {
  const c = p.class || '', s = p.subclass || '';
  if (c === 'hospital' || s === 'hospital' || s === 'clinic') return 'hospital';
  if (c === 'school' || c === 'college' || s === 'school' || s === 'university' || s === 'college' || s === 'kindergarten') return 'school';
  if (c === 'place_of_worship' || s === 'place_of_worship') return 'church';
  if (c === 'town_hall' || c === 'police' || c === 'fire_station' || c === 'library' || c === 'post' || s === 'courthouse' || s === 'community_centre') return 'civic';
  if (c === 'lodging' || s === 'hotel') return 'apartment';
  if (s === 'warehouse' || c === 'car' && s === 'car_repair') return 'warehouse';
  if (/shop|grocery|clothing|hardware|furniture|restaurant|fast_food|cafe|bar|beer|bank|fuel|pharmacy|bakery|ice_cream|alcohol|books|gift|jewelry|music|shoe|sports|toys|florist|optician|laundry|hairdresser|car|bicycle|mobile_phone|electronics|supermarket|marketplace|cinema|theatre/.test(c + ' ' + s)) return 'shop';
  if (c === 'office' || s === 'office') return 'office';
  return null;
}
// landuse class -> building type hint
const LANDUSE_TYPE = { hospital: 'hospital', school: 'school', university: 'school', college: 'school', kindergarten: 'school', industrial: 'warehouse',
  railway: 'warehouse', garages: 'shed', commercial: 'com', retail: 'shop', residential: 'res', cemetery: 'church', military: 'warehouse', farmyard: 'barn' };

function buildBuildings(layers, usa, H, c, tileSize, lms) {
  const osmLayer = layers && layers.building;
  const list = [];
  const S = tileSize, inTile = (x, z) => x >= 0 && x < S && z >= 0 && z < S;
  // OSM
  if (osmLayer) {
    const k = tileSize / osmLayer.extent;
    for (const fb of osmLayer.features) {
      const f = readFeature(fb, osmLayer), p = f.props;
      if (f.type !== 3 || p.hide_3d) continue;
      const rings = f.parts.map(r => r.map(v => v * k));
      let cur = null;
      const sign0 = Math.sign(ringArea(rings[0]));
      for (const r of rings) {
        if (r.length < 8) continue;
        if (Math.sign(ringArea(r)) === sign0 || !cur) { cur = { ring: r, holes: [], h: p.render_height || 6, minH: p.render_min_height || 0, colour: parseColour(p.colour), src: 'osm' }; list.push(cur); }
        else cur.holes.push(r);
      }
    }
  }
  const osmCount = list.length;
  if (usa && usa.length) {
    const CELL = 24, grid = new Map(), key = (x, z) => Math.floor(x / CELL) * 100003 + Math.floor(z / CELL);
    for (let b = 0; b < osmCount; b++) {
      const r = list[b].ring; let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      for (let i = 0; i < r.length; i += 2) { x0 = Math.min(x0, r[i]); x1 = Math.max(x1, r[i]); z0 = Math.min(z0, r[i + 1]); z1 = Math.max(z1, r[i + 1]); }
      for (let gz = Math.floor(z0 / CELL); gz <= Math.floor(z1 / CELL); gz++) for (let gx = Math.floor(x0 / CELL); gx <= Math.floor(x1 / CELL); gx++) {
        const kk = gx * 100003 + gz; (grid.get(kk) || grid.set(kk, []).get(kk)).push(b);
      }
    }
    for (const u of usa) {
      const [cx, cz] = ringCentroid(u.ring);
      const cands = grid.get(key(cx, cz)) || [];
      const occ = String(u.props.OCC_CLS || ''), out = u.props.OUTBLDG ? 1 : 0;
      const hit = cands.find(b => pointInRing(cx, cz, list[b].ring));
      if (hit !== undefined) { if (!list[hit].occ) list[hit].occ = occ; continue; }   // OSM shape, USA use class
      const h = u.props.HEIGHT > 2 ? u.props.HEIGHT : 0;
      list.push({ ring: u.ring, holes: [], h, minH: 0, colour: null, src: 'usa', occ, out });
    }
  }

  // ---- what each building is: a POI inside it, then the USA occupancy class, then the land use around it, then size
  const k0 = osmLayer ? tileSize / osmLayer.extent : 1;
  const pois = [];
  if (layers && layers.poi) {
    const L = layers.poi, k = tileSize / L.extent;
    for (const fb of L.features) {
      const f = readFeature(fb, L);
      if (f.type !== 1 || !f.parts.length) continue;
      const t = poiType(f.props);
      if (t) pois.push(f.parts[0][0] * k, f.parts[0][1] * k, t);
    }
  }
  const PG = 32, poiGrid = new Map();
  for (let i = 0; i < pois.length; i += 3) { const kk = Math.floor(pois[i] / PG) * 7919 + Math.floor(pois[i + 1] / PG); (poiGrid.get(kk) || poiGrid.set(kk, []).get(kk)).push(i); }
  const landuse = [];
  if (layers && layers.landuse) {
    const L = layers.landuse, k = tileSize / L.extent;
    for (const fb of L.features) {
      const f = readFeature(fb, L), t = LANDUSE_TYPE[f.props.class];
      if (f.type !== 3 || !t) continue;
      for (const r0 of f.parts) {
        if (r0.length < 8) continue;
        const r = r0.map(v => v * k);
        let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
        for (let i = 0; i < r.length; i += 2) { x0 = Math.min(x0, r[i]); x1 = Math.max(x1, r[i]); z0 = Math.min(z0, r[i + 1]); z1 = Math.max(z1, r[i + 1]); }
        landuse.push({ r, x0, x1, z0, z1, t, area: Math.abs(ringArea(r)) });
      }
    }
    landuse.sort((a, b) => a.area - b.area);   // smallest (most specific) first
  }
  const landuseAt = (x, z) => {
    for (const l of landuse) if (x >= l.x0 && x <= l.x1 && z >= l.z0 && z <= l.z1 && pointInRing(x, z, l.r)) return l.t;
    return null;
  };
  // neighbour density (centroids within ~80 m) separates farmsteads from town
  const DG = 80, dens = new Map(), cents = list.map(b => ringCentroid(b.ring));
  for (const [x, z] of cents) { const kk = Math.floor(x / DG) * 7919 + Math.floor(z / DG); dens.set(kk, (dens.get(kk) || 0) + 1); }
  const densAt = (x, z) => { let n = 0; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) n += dens.get((Math.floor(x / DG) + i) * 7919 + Math.floor(z / DG) + j) || 0; return n; };

  // ---- landmarks: snap to the best matching footprint, then drop every generic building under the model
  const skip = new Uint8Array(list.length), snaps = [];
  for (const L of lms || []) {
    const A = 4 * L.hx * L.hz, R = Math.hypot(L.hx, L.hz) + 25;
    let place = { x: L.x, z: L.z, rot: L.rot };
    if (L.own) {
      // only a single footprint that matches the model's size and lies close by; stadiums and big complexes
      // mapped as many separate parts keep the authored (Wikidata) position
      let best = null;
      for (let b = 0; b < list.length; b++) {
        const [bx, bz] = cents[b], d = Math.hypot(bx - L.x, bz - L.z);
        if (d > 0.5 * R) continue;
        const a = Math.abs(ringArea(list[b].ring));
        const sim = Math.min(a, A) / Math.max(a, A);
        if (sim < 0.35) continue;
        const score = sim - d / (2 * R);
        if (!best || score > best.score) best = { b, score, d };
      }
      if (best) {
        const box = orientedBox(list[best.b].ring);
        if (box) {
          const am = (box.a0 + box.a1) / 2, bm = (box.b0 + box.b1) / 2;
          place.x = am * box.ux - bm * box.uz; place.z = am * box.uz + bm * box.ux;
          // long axes together when both are clearly elongated; the authored heading picks among the options
          const fl = box.a1 - box.a0, fw = box.b1 - box.b0, phi = Math.atan2(box.uz, box.ux) + (fl >= fw ? 0 : Math.PI / 2);
          const mAsp = Math.max(L.hx, L.hz) / Math.min(L.hx, L.hz), fAsp = Math.max(fl, fw) / Math.max(1, Math.min(fl, fw));
          if (mAsp > 1.15 && fAsp > 1.15) {
            const base = L.hx >= L.hz ? -phi : Math.PI / 2 - phi;
            let bestR = base, bd = Infinity;
            for (const r of [base, base + Math.PI]) { const dd = Math.abs(Math.atan2(Math.sin(r - L.rot), Math.cos(r - L.rot))); if (dd < bd) { bd = dd; bestR = r; } }
            place.rot = bestR;
          }
          snaps.push({ qid: L.qid, x: place.x, z: place.z, rot: place.rot });
        }
      }
    }
    // generic buildings whose centre lies under the (slightly enlarged) model base
    const cs = Math.cos(place.rot), sn = Math.sin(place.rot);
    for (let b = 0; b < list.length; b++) {
      const dx = cents[b][0] - place.x, dz = cents[b][1] - place.z;
      if (Math.abs(dx) > R + 10 || Math.abs(dz) > R + 10) continue;
      if (L.bridge && !L.bridgeOk) break;
      const m = L.bridge ? 2 : Math.max(4, 0.12 * Math.max(L.hx, L.hz)), mxx = L.bridge ? 20 : m;
      const under = (x, z) => { const ex = x - place.x, ez = z - place.z, mx = ex * cs - ez * sn, mz = ex * sn + ez * cs; return Math.abs(mx) < L.hx + mxx && Math.abs(mz) < L.hz + m; };
      if (under(cents[b][0], cents[b][1])) { skip[b] = 1; continue; }
      // a tall OSM building right by a tall landmark is the landmark itself (towers, arches mapped as building parts)
      if (!L.bridge && L.h > 40 && list[b].h >= 0.3 * L.h && Math.hypot(cents[b][0] - place.x, cents[b][1] - place.z) < Math.hypot(L.hx, L.hz) + 30) { skip[b] = 1; continue; }
      // or most of its outline lies under the model (stands and wings whose centre is just outside)
      const r = list[b].ring; let inside = 0;
      for (let i = 0; i < r.length; i += 2) if (under(r[i], r[i + 1])) inside++;
      if (inside * 2 >= r.length / 2) skip[b] = 1;
    }
  }

  function classify(b, cx, cz, area) {
    // a named place inside the footprint wins
    const gx = Math.floor(cx / PG), gz = Math.floor(cz / PG);
    for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) {
      const l = poiGrid.get((gx + i) * 7919 + gz + j);
      if (l) for (const q of l) if (pointInRing(pois[q], pois[q + 1], b.ring)) return pois[q + 2];
    }
    const big = area > 2500, tall = b.h >= 14;
    if (b.out) return 'shed';
    if (b.occ) {
      if (b.occ === 'Education') return 'school';
      if (b.occ === 'Government') return 'civic';
      if (b.occ === 'Assembly') return area < 1800 ? 'church' : 'civic';
      if (b.occ === 'Industrial') return area < 60 ? 'shed' : 'warehouse';
      if (b.occ === 'Agriculture') return area < 60 ? 'shed' : 'barn';
      if (b.occ === 'Commercial') return tall ? 'office' : big ? 'warehouse' : 'shop';
      if (b.occ === 'Residential') return area < 40 ? 'shed' : area > 450 || tall ? 'apartment' : 'house';
    }
    const lu = landuseAt(cx, cz);
    if (area < 35) return 'shed';
    if (lu === 'hospital') return area > 250 ? 'hospital' : 'shed';
    if (lu === 'school') return area > 150 ? 'school' : 'shed';
    if (lu === 'church') return 'church';
    if (lu === 'warehouse' || lu === 'shed') return area < 80 ? 'shed' : 'warehouse';
    if (lu === 'barn') return area < 60 ? 'shed' : area < 250 && densAt(cx, cz) < 6 ? 'house' : 'barn';
    if (lu === 'shop') return tall ? 'office' : 'shop';
    if (lu === 'com') return tall ? 'office' : big ? 'warehouse' : 'shop';
    if (lu === 'res') return area > 500 || tall ? 'apartment' : area < 45 ? 'shed' : 'house';
    // no land use mapped: decide by size, height and how built-up the area is
    const d = densAt(cx, cz);
    if (tall) return area > 900 ? 'office' : 'apartment';
    if (big) return 'warehouse';
    if (area < 45) return 'shed';
    if (d < 5 && area > 140) return 'barn';
    if (area < 320) return 'house';
    return d > 25 ? 'shop' : 'warehouse';
  }
  // typical height when the data has none (OSM's default render_height is a placeholder)
  function defaultHeight(type, area, hash, given) {
    const known = given > 0 && !(given === 5 || given === 6);
    if (known) return given;
    switch (type) {
      case 'house': return hash < 0.55 ? 6.2 : 3.8 + hash;
      case 'apartment': return 3.1 * (3 + Math.floor(hash * Math.min(9, 2 + area / 250)));
      case 'office': return 3.6 * (4 + Math.floor(hash * 10));
      case 'hospital': return 3.8 * (3 + Math.floor(hash * 4));
      case 'school': return hash < 0.5 ? 4.5 : 8;
      case 'shop': return area > 900 ? 6.5 : hash < 0.6 ? 4.6 : 7.5;
      case 'warehouse': return 7 + hash * 5;
      case 'church': return 9 + hash * 4;
      case 'barn': return 6 + hash * 3;
      case 'shed': return 2.6 + hash;
      case 'civic': return 8 + hash * 6;
      default: return given || 6;
    }
  }

  const pos = [], nor = [], col = [], bw = [], idx = [];
  const foot = [], footOff = [0];
  const push = (x, y, z, n, rgb, a) => { pos.push(x, y, z); nor.push(n[0], n[1], n[2]); col.push(rgb[0], rgb[1], rgb[2]); bw.push(a[0], a[1], a[2], a[3]); return pos.length / 3 - 1; };
  const tri = (a, b, cc, up) => {
    const ax = pos[3 * a], ay = pos[3 * a + 1], az = pos[3 * a + 2];
    const e1 = [pos[3 * b] - ax, pos[3 * b + 1] - ay, pos[3 * b + 2] - az], e2 = [pos[3 * cc] - ax, pos[3 * cc + 1] - ay, pos[3 * cc + 2] - az];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    if (n[0] * up[0] + n[1] * up[1] + n[2] * up[2] >= 0) idx.push(a, b, cc); else idx.push(a, cc, b);
  };
  const lin = v => v.map(x => Math.pow(x, 2.2));
  const PLAIN = 2 * BT.plain;
  // flat-shaded quad / polygon with an outward normal
  // `hint` is roughly the outward direction; the face and its normal are turned to agree with it
  const face = (vs, rgb, w, hint = [0, 1, 0]) => {
    const e1 = [vs[1][0] - vs[0][0], vs[1][1] - vs[0][1], vs[1][2] - vs[0][2]], e2 = [vs[2][0] - vs[0][0], vs[2][1] - vs[0][1], vs[2][2] - vs[0][2]];
    let nn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const ln = Math.hypot(...nn) || 1; nn = nn.map(v => v / ln);
    if (nn[0] * hint[0] + nn[1] * hint[1] + nn[2] * hint[2] < 0) nn = nn.map(v => -v);
    const ids = vs.map(v => push(v[0], v[1], v[2], nn, rgb, [0, 0, 0, w]));
    for (let q = 1; q < ids.length - 1; q++) tri(ids[0], ids[q], ids[q + 1], nn);
    return nn;
  };
  // oriented box: centre (x, z), axes u (ux, uz) and v = (-uz, ux), half sizes, from y0 to y1
  const box3 = (x, z, ux, uz, hu, hv, y0, y1, rgb, w = PLAIN) => {
    const vx = -uz, vz = ux, P = (a, b) => [x + ux * a + vx * b, z + uz * a + vz * b];
    const c4 = [P(-hu, -hv), P(hu, -hv), P(hu, hv), P(-hu, hv)];
    for (let i = 0; i < 4; i++) {
      const a = c4[i], b = c4[(i + 1) % 4], mx = (a[0] + b[0]) / 2 - x, mz = (a[1] + b[1]) / 2 - z;
      const nn = [mx, 0, mz], l = Math.hypot(mx, mz) || 1; nn[0] /= l; nn[2] /= l;
      const ids = [push(a[0], y0, a[1], nn, rgb, [0, 0, 0, w]), push(b[0], y0, b[1], nn, rgb, [0, 0, 0, w]), push(b[0], y1, b[1], nn, rgb, [0, 0, 0, w]), push(a[0], y1, a[1], nn, rgb, [0, 0, 0, w])];
      tri(ids[0], ids[1], ids[2], nn); tri(ids[0], ids[2], ids[3], nn);
    }
    const UP = [0, 1, 0], t = c4.map(p => push(p[0], y1, p[1], UP, rgb, [0, 0, 0, w]));
    tri(t[0], t[1], t[2], UP); tri(t[0], t[2], t[3], UP);
  };
  const pyramid = (x, z, ux, uz, hu, hv, y0, y1, rgb) => {
    const vx = -uz, vz = ux, P = (a, b) => [x + ux * a + vx * b, y0, z + uz * a + vz * b];
    const c4 = [P(-hu, -hv), P(hu, -hv), P(hu, hv), P(-hu, hv)], top = [x, y1, z];
    for (let i = 0; i < 4; i++) {
      const a = c4[i], b = c4[(i + 1) % 4];
      face([a, b, top], rgb, PLAIN, [(a[0] + b[0]) / 2 - x, 0.3, (a[2] + b[2]) / 2 - z]);
    }
  };

  let count = 0;
  for (let bi = 0; bi < list.length; bi++) {
    const b = list[bi], r = b.ring, n = r.length / 2;
    const [cx, cz] = cents[bi];
    if (!inTile(cx, cz) || skip[bi]) continue;
    const area = Math.abs(ringArea(r));
    if (area < 6) continue;
    let gMin = Infinity, gMax = -Infinity;
    for (let i = 0; i < n; i++) { const g = sampleTri(H, r[2 * i], r[2 * i + 1], c); gMin = Math.min(gMin, g); gMax = Math.max(gMax, g); }
    const hash = hash2(Math.round(cx * 10), Math.round(cz * 10)), hash2b = hash2(Math.round(cz * 7), Math.round(cx * 3));
    const type = b.minH > 0 ? 'office' : classify(b, cx, cz, area);
    const T = BT[type] ?? 0, W2 = 2 * T;
    const h = b.minH > 0 ? b.h : defaultHeight(type, area, hash2b, b.h);
    const pal = WALL_COLORS[type] || WALL_COLORS.house;
    const wallCol = lin(b.colour || pal[Math.floor(hash * pal.length)]);
    const box = orientedBox(r);
    const L = box ? box.a1 - box.a0 : 0, W = box ? box.b1 - box.b0 : 0;
    const rect = box && area / box.area > 0.8;
    // roof shape
    let roof = 'flat';
    if (!b.minH && rect && W > 3) {
      if ((type === 'house' || type === 'shed') && area < 500) roof = hash2b < 0.55 ? 'hip' : 'gable';
      else if (type === 'barn' || type === 'church') roof = 'gable';
      else if (type === 'warehouse' && W < 45) roof = 'lowgable';
      else if ((type === 'school' || type === 'civic') && area < 900 && hash < 0.4) roof = 'hip';
    }
    const rp = roof === 'flat' ? ROOF_COLORS.flat : roof === 'lowgable' || type === 'barn' || type === 'shed' ? ROOF_COLORS.metal : ROOF_COLORS.pitched;
    const roofCol = lin(rp[Math.floor(hash2b * rp.length * 0.999)]);
    const base = b.minH > 0 ? gMin + b.minH : gMin - 0.6;
    const top = gMax + Math.max(2.5, h);
    const minWL = roof === 'hip' || roof === 'gable' ? Math.min(L, W) : 0;
    const ridgeH = roof === 'hip' ? Math.min(minWL * 0.32, 4) : roof === 'gable' ? Math.min(minWL * (type === 'barn' || type === 'church' ? 0.55 : 0.38), type === 'church' ? 9 : 6) : roof === 'lowgable' ? Math.min(W * 0.08, 2.5) : 0;
    const eave = roof === 'flat' ? top : Math.max(gMax + 2.6, top - (roof === 'lowgable' ? 0 : ridgeH));
    const parapet = roof === 'flat' && type !== 'house' && type !== 'shed' && area > 80 ? 0.7 : 0;
    // walls (window height stops at the roof line; a parapet continues the wall above it)
    const sgn = Math.sign(ringArea(r)) || 1;
    let along = 0;
    const wallTop = eave + parapet, wh = eave - base;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, ax = r[2 * i], az = r[2 * i + 1], bx = r[2 * j], bz = r[2 * j + 1];
      const dx = bx - ax, dz = bz - az, l = Math.hypot(dx, dz);
      if (l < 0.05) continue;
      const nn = sgn > 0 ? [dz / l, 0, -dx / l] : [-dz / l, 0, dx / l];
      const th = wallTop - base;
      const v0 = push(ax, base, az, nn, wallCol, [along, 0, wh, W2]), v1 = push(bx, base, bz, nn, wallCol, [along + l, 0, wh, W2]);
      const v2 = push(bx, wallTop, bz, nn, wallCol, [along + l, th, wh, W2]), v3 = push(ax, wallTop, az, nn, wallCol, [along, th, wh, W2]);
      tri(v0, v1, v2, nn); tri(v0, v2, v3, nn);
      // shop fronts: a coloured fascia band above the windows, standing slightly proud of the wall
      if (type === 'shop' && l > 3 && wh > 3.8) {
        const brand = lin(BRANDS[Math.floor(hash2(i + 3, Math.round(cx)) * BRANDS.length)]), o = 0.12, y0 = base + 0.6 + 3.3, y1 = y0 + 0.75;
        const ex = nn[0] * o, ez = nn[2] * o;
        const f0 = push(ax + ex, y0, az + ez, nn, brand, [0, 0, 0, PLAIN]), f1 = push(bx + ex, y0, bz + ez, nn, brand, [0, 0, 0, PLAIN]);
        const f2 = push(bx + ex, y1, bz + ez, nn, brand, [0, 0, 0, PLAIN]), f3 = push(ax + ex, y1, az + ez, nn, brand, [0, 0, 0, PLAIN]);
        tri(f0, f1, f2, nn); tri(f0, f2, f3, nn);
      }
      along += l;
    }
    // roof
    const RW = 1 + W2;
    if (roof !== 'flat') {
      const o = roof === 'lowgable' ? 0.2 : 0.4, { ux, uz } = box, a0 = box.a0 - o, a1 = box.a1 + o, b0 = box.b0 - o, b1 = box.b1 + o;
      const P = (a, bb) => [a * ux - bb * uz, a * uz + bb * ux];
      const long = (a1 - a0) >= (b1 - b0);
      const mid = long ? (b0 + b1) / 2 : (a0 + a1) / 2, half = long ? (b1 - b0) / 2 : (a1 - a0) / 2;
      const inset = roof === 'hip' ? half : 0;
      const r0 = long ? P(a0 + inset, mid) : P(mid, b0 + inset), r1 = long ? P(a1 - inset, mid) : P(mid, b1 - inset);
      const cs = [P(a0, b0), P(a1, b0), P(a1, b1), P(a0, b1)];
      const ridgeY = eave + ridgeH, y0r = eave - 0.05;
      const slopes = long ? [[cs[0], cs[1], r1, r0], [cs[2], cs[3], r0, r1]] : [[cs[1], cs[2], r1, r0], [cs[3], cs[0], r0, r1]];
      const ends = long ? [[cs[1], cs[2], r1], [cs[3], cs[0], r0]] : [[cs[0], cs[1], r0], [cs[2], cs[3], r1]];
      for (const f of slopes) face(f.map((p2, q) => [p2[0], q < 2 ? y0r : ridgeY, p2[1]]), roofCol, RW);
      for (const f of ends) {
        const vs = f.map((p2, q) => [p2[0], q < 2 ? y0r : ridgeY, p2[1]]);
        const out = [(vs[0][0] + vs[1][0]) / 2 - cx, 0, (vs[0][2] + vs[1][2]) / 2 - cz];
        if (roof === 'hip') face(vs, roofCol, RW, [out[0], 1, out[2]]);
        else face(vs, wallCol, PLAIN, out);            // gable end: a wall triangle
      }
    } else {
      const flat = r.slice(), holeIdx = [];
      for (const hh of b.holes) { holeIdx.push(flat.length / 2); flat.push(...hh); }
      const UPN = [0, 1, 0];
      const vb = pos.length / 3;
      for (let i = 0; i < flat.length / 2; i++) push(flat[2 * i], top, flat[2 * i + 1], UPN, roofCol, [0, 0, 0, RW]);
      const t = typeof earcut === 'function' ? earcut(flat, holeIdx.length ? holeIdx : null, 2) : Array.from({ length: (n - 2) * 3 }, (_, q) => [0, Math.floor(q / 3) + 1, Math.floor(q / 3) + 2][q % 3]);
      for (let q = 0; q < t.length; q += 3) tri(vb + t[q], vb + t[q + 1], vb + t[q + 2], UPN);
      // rooftop plant on bigger flat roofs
      if (box && area > 150 && type !== 'house' && type !== 'shed') {
        const units = Math.min(6, 1 + Math.floor(area / 600));
        const grey = lin([0.62, 0.63, 0.64]);
        for (let u = 0; u < units; u++) {
          const fa = hash2(u * 7 + 1, Math.round(cx)) * 0.7 + 0.15, fb = hash2(u * 13 + 5, Math.round(cz)) * 0.7 + 0.15;
          const a = box.a0 + (box.a1 - box.a0) * fa, bb = box.b0 + (box.b1 - box.b0) * fb;
          const x = a * box.ux - bb * box.uz, z = a * box.uz + bb * box.ux;
          if (!pointInRing(x, z, r)) continue;
          const s = 0.8 + hash2(u, Math.round(cx * 3)) * 1.4;
          box3(x, z, box.ux, box.uz, s, s * 0.7, top, top + 0.9 + s * 0.4, grey);
        }
      }
    }
    // landmarks
    if (box && type === 'church' && area > 60 && area < 3000) {
      const s = Math.min(Math.min(L, W) * 0.4, 4.5), along0 = box.a1 - s * 0.9, bm = (box.b0 + box.b1) / 2;
      const x = along0 * box.ux - bm * box.uz, z = along0 * box.uz + bm * box.ux, towerTop = eave + ridgeH + 4 + hash * 8;
      box3(x, z, box.ux, box.uz, s * 0.5, s * 0.5, base, towerTop, wallCol);
      pyramid(x, z, box.ux, box.uz, s * 0.55, s * 0.55, towerTop, towerTop + s * 2.4 + hash2b * 4, lin([0.3, 0.32, 0.34]));
    }
    if (box && type === 'hospital' && area > 600 && roof === 'flat') {
      // roof pad with a red cross
      const am = (box.a0 + box.a1) / 2, bm = (box.b0 + box.b1) / 2, x = am * box.ux - bm * box.uz, z = am * box.uz + bm * box.ux;
      if (pointInRing(x, z, r)) {
        const s = Math.min(6, Math.min(L, W) * 0.3);
        box3(x, z, box.ux, box.uz, s, s, top, top + 0.12, lin([0.9, 0.9, 0.88]));
        box3(x, z, box.ux, box.uz, s * 0.7, s * 0.2, top + 0.12, top + 0.16, lin([0.8, 0.08, 0.08]));
        box3(x, z, box.ux, box.uz, s * 0.2, s * 0.7, top + 0.12, top + 0.16, lin([0.8, 0.08, 0.08]));
      }
    }
    foot.push(...r);
    footOff.push(foot.length / 2);
    count++;
  }
  if (!idx.length) return snaps.length ? { count: 0, osm: 0, usa: 0, snaps, mesh: null, foot: new Float32Array(0), footOff: new Uint32Array([0]) } : null;
  return {
    count, osm: osmCount, usa: list.length - osmCount, snaps,
    mesh: { positions: new Float32Array(pos), normals: new Float32Array(nor), colors: new Float32Array(col), bw: new Float32Array(bw), indices: new Uint32Array(idx) },
    foot: new Float32Array(foot), footOff: new Uint32Array(footOff),
  };
}

// ------------------------------------------------ trees
// Individual crowns from the aerial photo: vegetation (NDVI) that is darker than its surroundings is canopy
// (lawns and crops are smooth and bright). A distance transform gives crown centres and sizes.
// ------------------------------------------------ tree species by biome
// Each tree's species is drawn from a blend of biomes; biomes are soft-edged regions (so ranges fade into each other
// over hundreds of km) plus an altitude band. A slow noise field per species makes trees grow in natural stands.
// name, conifer, height = clamp(crownR * k + base, min, max), crown tint (blended with the photo colour)
const SPECIES = [
  ['spruce', 1, 3.6, 3, 6, 40, [0.17, 0.27, 0.2]], ['fir', 1, 3.3, 3, 6, 45, [0.15, 0.29, 0.2]], ['pine', 1, 3.0, 5, 8, 32, [0.25, 0.35, 0.2]],
  ['ponderosa', 1, 3.2, 5, 8, 40, [0.26, 0.36, 0.2]], ['larch', 1, 3.2, 3, 6, 35, [0.42, 0.52, 0.25]], ['cedar', 1, 3.8, 3, 7, 45, [0.18, 0.3, 0.18]],
  ['cypress', 1, 4.5, 2, 6, 22, [0.16, 0.26, 0.15]], ['juniper', 1, 1.6, 1.5, 2.5, 8, [0.3, 0.36, 0.25]], ['oak', 0, 2.2, 2, 5, 26, [0.28, 0.38, 0.18]],
  ['maple', 0, 2.4, 2, 5, 28, [0.3, 0.42, 0.17]], ['beech', 0, 2.6, 2, 6, 32, [0.3, 0.44, 0.18]], ['birch', 0, 3.0, 2, 6, 25, [0.42, 0.56, 0.22]],
  ['poplar', 0, 4.5, 2, 8, 32, [0.36, 0.48, 0.2]], ['willow', 0, 1.8, 2, 5, 18, [0.42, 0.52, 0.25]], ['olive', 0, 1.4, 1.5, 3, 9, [0.42, 0.46, 0.34]],
  ['gum', 0, 3.4, 4, 8, 50, [0.4, 0.45, 0.35]], ['mallee', 0, 1.8, 1.5, 3, 10, [0.42, 0.47, 0.36]], ['acacia', 0, 1.6, 2, 4, 14, [0.42, 0.48, 0.25]],
  ['stonepine', 1, 1.8, 4, 8, 22, [0.2, 0.32, 0.18]], ['palm', 0, 3.0, 5, 6, 26, [0.38, 0.5, 0.22]], ['emergent', 0, 4.5, 15, 30, 60, [0.22, 0.38, 0.16]],
  ['canopy', 0, 3.0, 6, 12, 38, [0.2, 0.36, 0.14]], ['baobab', 0, 2.2, 4, 8, 22, [0.4, 0.45, 0.25]], ['casuarina', 1, 3.0, 3, 6, 25, [0.36, 0.42, 0.3]],
];
const SP = Object.fromEntries(SPECIES.map((s, i) => [s[0], i]));
const BIOMES = {
  boreal: { spruce: 0.35, pine: 0.3, birch: 0.25, larch: 0.1 },
  europe: { oak: 0.22, beech: 0.2, maple: 0.1, birch: 0.12, poplar: 0.05, pine: 0.12, spruce: 0.12, willow: 0.02, larch: 0.05 },
  naEast: { oak: 0.28, maple: 0.25, beech: 0.07, birch: 0.07, pine: 0.18, cedar: 0.08, poplar: 0.05, willow: 0.02 },
  naWest: { ponderosa: 0.3, fir: 0.25, spruce: 0.12, juniper: 0.1, birch: 0.15, poplar: 0.05, pine: 0.03 },
  pnw: { fir: 0.45, cedar: 0.3, spruce: 0.15, maple: 0.1 },
  med: { olive: 0.25, stonepine: 0.2, cypress: 0.2, oak: 0.3, pine: 0.05 },
  aus: { gum: 0.6, mallee: 0.12, acacia: 0.14, casuarina: 0.14 },
  nz: { canopy: 0.35, beech: 0.25, pine: 0.2, cedar: 0.15, palm: 0.05 },
  rain: { canopy: 0.55, emergent: 0.25, palm: 0.2 },
  savanna: { acacia: 0.5, baobab: 0.08, palm: 0.12, canopy: 0.3 },
  subtrop: { oak: 0.35, pine: 0.4, canopy: 0.25 },
  palms: { palm: 1 },
  plains: { poplar: 0.3, oak: 0.25, juniper: 0.2, maple: 0.15, willow: 0.1 },
  japan: { cedar: 0.35, pine: 0.2, maple: 0.2, beech: 0.15, canopy: 0.1 },
  eastAsia: { pine: 0.3, oak: 0.25, maple: 0.15, poplar: 0.15, birch: 0.15 },
  mountain: { spruce: 0.45, fir: 0.35, pine: 0.1, larch: 0.1 },
  generic: { oak: 0.3, maple: 0.2, pine: 0.3, birch: 0.2 },
  tropGeneric: { canopy: 0.5, palm: 0.25, emergent: 0.1, acacia: 0.15 },
};
const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// soft box: 1 inside, fading to 0 over `f` degrees outside the edges
const soft = (v, lo, hi, f) => sm(lo - f, lo + f, v) * (1 - sm(hi - f, hi + f, v));
function biomeMix(lat, lon, alt) {
  const al = Math.abs(lat), box = (a, b, c, d, f = 3) => soft(lat, a, b, f) * soft(lon, c, d, f);
  const tropic = 1 - sm(13, 24, al), boreal = sm(53, 61, al);
  const w = {
    boreal: boreal * (1 - box(-48, -34, 165, 179)),
    europe: box(41, 58, -11, 45, 4) * (1 - boreal),
    naEast: box(26, 50, -98, -60, 5) * (1 - boreal),
    naWest: box(30, 52, -122, -100, 4) * (1 - boreal),
    pnw: box(42, 56, -130, -121, 2) * (1 - boreal),
    med: box(30, 44, -10, 38, 3) * soft(lat, 28, 44, 2) + box(32, 40, -124, -117, 2),
    aus: box(-44, -11, 112, 155, 3),
    nz: box(-48, -34, 165, 179, 2),
    rain: tropic * Math.min(1, box(-17, 11, -82, -34, 4) + box(7, 20, -95, -76, 3) + box(10, 19, -76, -60, 2) + box(-9, 8, 8, 32, 4) + box(-11, 21, 94, 156, 4) + box(6, 22, 72, 94, 3) * 0.6),
    savanna: (1 - sm(18, 30, al)) * Math.min(1, box(-26, 16, -18, 52, 5) + box(-22, -10, 119, 152, 3) + box(-25, -5, -60, -35, 4)),
    subtrop: soft(al, 20, 34, 3) * (1 - box(30, 44, -10, 38, 3)),
    palms: 0.25 * soft(al, 20, 34, 3) * (1 - sm(27, 31, al)) * (1 - box(30, 44, -10, 38, 3)),
    plains: box(28, 49, -104, -95, 3) * (1 - boreal),
    japan: box(30, 46, 128, 146, 2),
    eastAsia: box(22, 50, 100, 128, 4),
    generic: 0.06 * (1 - tropic),
    tropGeneric: 0.06 * tropic,
  };
  // treeline zone: conifers take over on mountains (higher in the tropics)
  const t0 = 2700 - 45 * Math.max(0, al - 22);
  const mount = sm(t0 - 600, t0 + 400, alt);
  const out = new Float32Array(SPECIES.length);
  let tot = 0;
  for (const k in w) tot += w[k];
  for (const k in w) {
    const bw = (w[k] / tot) * (1 - mount);
    if (bw > 1e-4) for (const sp in BIOMES[k]) out[SP[sp]] += bw * BIOMES[k][sp];
  }
  for (const sp in BIOMES.mountain) out[SP[sp]] += mount * BIOMES.mountain[sp];
  return out;
}
function pickSpecies(mix, gx, gz, forestCode, rv) {
  let tot = 0;
  const w = new Float32Array(mix.length);
  for (let i = 0; i < mix.length; i++) {
    if (mix[i] <= 0) continue;
    // stands: each species is locally common or rare following a slow noise field
    let v = mix[i] * (0.15 + 1.7 * noise(gx / 140 + i * 13.7, gz / 140 - i * 7.3));
    if (forestCode === 142 && !SPECIES[i][1]) v *= 0.05;       // USDA evergreen forest
    if ((forestCode === 141 || forestCode === 190) && SPECIES[i][1]) v *= 0.05; // deciduous / woody wetland
    w[i] = v; tot += v;
  }
  let acc = rv * tot;
  for (let i = 0; i < w.length; i++) { acc -= w[i]; if (acc <= 0) return i; }
  return SP.oak;
}

const TREE = 8; // x z radius height r g b species
function detectTrees(im, ctx, H0, c, tileSize, lat, lon, gx0, gz0) {
  const mixCache = new Map();
  const S = im.photoSize, P = im.photo, px = tileSize / S, s2 = im.source !== 'NAIP';
  const lum = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) lum[i] = (P[i * 4] * 0.3 + P[i * 4 + 1] * 0.59 + P[i * 4 + 2] * 0.11) / 255;
  // local mean brightness of *vegetation* over ~50 m (integral images), so bright roads and
  // roofs nearby don't make ordinary grass look like dark tree canopy
  const R = Math.max(2, Math.round(25 / px)), W1 = S + 1;
  const I = new Float64Array(W1 * W1), C = new Float64Array(W1 * W1);
  for (let y = 0; y < S; y++) {
    let row = 0, cnt = 0;
    for (let x = 0; x < S; x++) {
      const i = y * S + x, v = P[i * 4 + 3] / 255 * 2 - 1 > 0.2 ? 1 : 0;
      row += lum[i] * v; cnt += v;
      I[(y + 1) * W1 + x + 1] = I[y * W1 + x + 1] + row;
      C[(y + 1) * W1 + x + 1] = C[y * W1 + x + 1] + cnt;
    }
  }
  const mean = (x, y) => {
    const x0 = Math.max(0, x - R), y0 = Math.max(0, y - R), x1 = Math.min(S, x + R + 1), y1 = Math.min(S, y + R + 1);
    const n = C[y1 * W1 + x1] - C[y0 * W1 + x1] - C[y1 * W1 + x0] + C[y0 * W1 + x0];
    return n > 0 ? (I[y1 * W1 + x1] - I[y0 * W1 + x1] - I[y1 * W1 + x0] + I[y0 * W1 + x0]) / n : 0;
  };
  const T = new Uint8Array(S * S), F = new Uint8Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const i = y * S + x;
    const ndvi = P[i * 4 + 3] / 255 * 2 - 1;
    if (ndvi < 0.2) continue;
    const ci = Math.floor(y * N / S) * N + Math.floor(x * N / S), m = im.mat[ci], cd = im.code[ci];
    const gk = (Math.floor(y * px / c) + M) * G + Math.floor(x * px / c) + M;
    if (m === WATER || (ctx.isW && ctx.isW[gk]) || (ctx.infl && ctx.infl[gk] > 0.3)) continue;
    if (m === FOREST && ndvi > 0.25) F[i] = 1;
    if (s2) { if (m === FOREST && ndvi > 0.35) T[i] = 1; continue; }
    const orchard = (cd >= 66 && cd <= 77) || cd === 204 || (cd >= 210 && cd <= 212);
    if (m === CROP && !orchard) continue;
    if (m === FOREST ? lum[i] < mean(x, y) * 1.02 : ndvi > 0.3 && lum[i] < mean(x, y) * 0.82) T[i] = 1;
  }
  // chamfer distance to the nearest non-canopy pixel
  const D = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) D[i] = T[i] ? 1e6 : 0;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const i = y * S + x; if (!D[i]) continue;
    let d = D[i];
    if (x > 0) d = Math.min(d, D[i - 1] + 1);
    if (y > 0) { d = Math.min(d, D[i - S] + 1); if (x > 0) d = Math.min(d, D[i - S - 1] + 1.414); if (x < S - 1) d = Math.min(d, D[i - S + 1] + 1.414); }
    D[i] = x === 0 || y === 0 ? Math.min(d, 1) : d;
  }
  for (let y = S - 1; y >= 0; y--) for (let x = S - 1; x >= 0; x--) {
    const i = y * S + x; if (!D[i]) continue;
    let d = D[i];
    if (x < S - 1) d = Math.min(d, D[i + 1] + 1);
    if (y < S - 1) { d = Math.min(d, D[i + S] + 1); if (x < S - 1) d = Math.min(d, D[i + S + 1] + 1.414); if (x > 0) d = Math.min(d, D[i + S - 1] + 1.414); }
    D[i] = d;
  }
  // candidates: crown centres (distance maxima) first, then a jittered fill inside big canopy patches
  const cand = [];
  for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
    const i = y * S + x, d = D[i];
    if (d < 1.2) continue;
    if (d < D[i - 1] || d < D[i + 1] || d < D[i - S] || d < D[i + S] || d < D[i - S - 1] || d < D[i + S + 1]) continue;
    cand.push(x, y, Math.min(6.5, (d + 0.3) * px));
  }
  const order = []; for (let q = 0; q < cand.length; q += 3) order.push(q);
  order.sort((a, b) => cand[b + 2] - cand[a + 2]);
  const gs = Math.max(2, Math.round((s2 ? 6 : 5) / px));
  for (let y = 0; y < S; y += gs) for (let x = 0; x < S; x += gs) {
    const jx = Math.min(S - 1, x + Math.floor(hash2(x, y) * gs)), jy = Math.min(S - 1, y + Math.floor(hash2(y + 9, x) * gs));
    const d = D[jy * S + jx];
    if (d >= 1) { cand.push(jx, jy, Math.max(1.8, Math.min(3.8, d * px))); order.push(cand.length - 3); }
  }
  // closed forest: canopy cover over ~30 m decides a second, much denser fill whose acceptance rises steeply with
  // cover, so woods read as a continuous canopy while scattered town trees stay as they are
  const RC = Math.max(2, Math.round(15 / px)), CI = new Float32Array(W1 * W1);
  for (let y = 0; y < S; y++) { let row = 0; for (let x = 0; x < S; x++) { const i = y * S + x; F[i] |= T[i]; row += F[i]; CI[(y + 1) * W1 + x + 1] = CI[y * W1 + x + 1] + row; } }
  const coverAt = (x, y) => {
    const x0 = Math.max(0, x - RC), y0 = Math.max(0, y - RC), x1 = Math.min(S, x + RC + 1), y1 = Math.min(S, y + RC + 1);
    return (CI[y1 * W1 + x1] - CI[y0 * W1 + x1] - CI[y1 * W1 + x0] + CI[y0 * W1 + x0]) / ((x1 - x0) * (y1 - y0));
  };
  const dense = new Set();
  const gd = 3.2 / px;   // may be below one pixel on Sentinel tiles: sample in metres, not pixels
  for (let fy = 0; fy < S; fy += gd) for (let fx = 0; fx < S; fx += gd) {
    const h1 = hash2(Math.round(fx * 17), Math.round(fy * 31)), h2 = hash2(Math.round(fy * 13) + 7, Math.round(fx * 11));
    const jx = Math.min(S - 1, Math.floor(fx + h1 * gd)), jy = Math.min(S - 1, Math.floor(fy + h2 * gd));
    if (!F[jy * S + jx]) continue;
    const cv = coverAt(jx, jy);
    if (cv < 0.35) continue;
    const pk = Math.pow((cv - 0.35) / 0.65, 1.6);
    if (hash2(Math.round(fx * 7) + 3, Math.round(fy * 5) + 1) > pk) continue;
    cand.push(jx + (h1 - 0.5) * 0.9, jy + (h2 - 0.5) * 0.9, 1.5 + hash2(jx + 1, jy + 2) * 1.6);
    dense.add(cand.length - 3); order.push(cand.length - 3);
  }
  // accept with spacing, via a 4 m hash grid
  const CELL = 4, GW = Math.ceil(tileSize / CELL) + 1, occ = new Map(), out = [];
  for (const q of order) {
    const x = (cand[q] + 0.5) * px, z = (cand[q + 1] + 0.5) * px, r = cand[q + 2], fs = dense.has(q) ? 0.55 : 0.75;
    const cx = Math.floor(x / CELL), cz = Math.floor(z / CELL), reach = Math.ceil((r + 6.5) * 0.75 / CELL);
    let ok = true;
    for (let dz = -reach; dz <= reach && ok; dz++) for (let dx = -reach; dx <= reach && ok; dx++) {
      const list = occ.get((cz + dz) * GW + cx + dx);
      if (list) for (const o of list) if (Math.hypot(out[o] - x, out[o + 1] - z) < fs * (r + out[o + 2])) { ok = false; break; }
    }
    if (!ok) continue;
    const key = cz * GW + cx;
    (occ.get(key) || occ.set(key, []).get(key)).push(out.length);
    // species from the local biome blend (altitude in 50 m bands is enough), USDA forest type where known
    const ix = Math.round(cand[q]), iy = Math.round(cand[q + 1]), ci = Math.min(N * N - 1, Math.floor(Math.min(S - 1, iy) * N / S) * N + Math.floor(Math.min(S - 1, ix) * N / S)), cd = im.code[ci];
    const alt = sampleBilinear(H0, x, z, c), ak = Math.round(alt / 50);
    let mix = mixCache.get(ak);
    if (!mix) mixCache.set(ak, mix = biomeMix(lat, lon, ak * 50));
    const sp = pickSpecies(mix, gx0 + x, gz0 + z, cd, hash2(ix * 7 + 3, iy * 13 + 1));
    const SPc = SPECIES[sp], hr = hash2(iy, ix);
    const h = Math.min(SPc[5], Math.max(SPc[4], r * SPc[2] + SPc[3] + (hr - 0.3) * 4));
    let cr = 0, cg = 0, cb = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = Math.min(S - 1, Math.max(0, iy + dy)), xx = Math.min(S - 1, Math.max(0, ix + dx)), o = (yy * S + xx) * 4;
      cr += P[o]; cg += P[o + 1]; cb += P[o + 2]; n++;
    }
    // crown colour: halfway between the photo and the species tone, with a little per-tree variation
    const jit = 0.9 + hash2(ix + 5, iy * 3) * 0.2, tn = SPc[6];
    out.push(x, z, r, h, (cr / n / 255 * 0.5 + tn[0] * 0.5) * jit, (cg / n / 255 * 0.5 + tn[1] * 0.5) * jit, (cb / n / 255 * 0.5 + tn[2] * 0.5) * jit, sp);
    if (out.length >= 320000 * TREE) break;
  }
  const res = new Float32Array(out), tmp = new Float32Array(TREE), nT = res.length / TREE;
  for (let i = nT - 1; i > 0; i--) {
    const j = Math.floor(hash2(i * 3 + 1, i * 7 + 5) * (i + 1)) * TREE, a = i * TREE;
    tmp.set(res.subarray(a, a + TREE)); res.copyWithin(a, j, j + TREE); res.set(tmp, j);
  }
  return res;
}

// ------------------------------------------------ entry
onmessage = async e => {
  const { id, tx, ty, lod, tileSize, ox, oz, url, img, lms } = e.data;
  const imageryP = img ? buildImagery(tx, ty, lod, img).catch(() => null) : Promise.resolve(null);
  const c = tileSize / 256;
  let ctx = null, H0 = null, bctx = null;
  try {
    const vtP = lod === 1 ? getVectorLayers(url, tx, ty) : Promise.resolve(null);
    const usaP = lod === 1 && img && img.us ? fetchUsaStructures(tx, ty, tileSize).catch(() => []) : Promise.resolve([]);
    H0 = await assembleHeights(tx, ty);
    const H = new Float32Array(H0);
    const vt = await vtP;
    if (lod === 1 && !vt) throw new Error('vector tile failed');
    const layers = vt && !vt.empty ? vt : null;
    const wm = layers ? waterMask(layers, H0, c, tileSize) : null;
    if (lms) padLandmarks(H0, lms, c, wm);
    const roads = layers && layers.transportation ? extractRoads(layers.transportation, tileSize, H0, c, lms) : [];
    const conn = roads.length ? connectRoads(roads) : { junctions: [], stops: [], rings: [] };
    gradeSeparate(roads, wm, c);
    if (lms) alignBridgeLandmarks(lms, roads);
    roads.sort((a, b) => a.rank - b.rank);
    const infl = roads.length ? flatten(roads, H0, H, c) : null;
    // landmarks sunk into the ground (memorial pools, plazas): cut the terrain under each pit; the model
    // draws its own water there, so OSM water inside a pit is dropped
    if (lms) for (const L of lms) if (L.pits) for (const [pcx, pcz, phx, phz, prot, depth] of L.pits) {
      const cs = Math.cos(L.rot), sn = Math.sin(L.rot), wx = L.x + pcx * cs + pcz * sn, wz = L.z - pcx * sn + pcz * cs;
      const pr = L.rot - prot * Math.PI / 180, pc = Math.cos(pr), ps = Math.sin(pr), pad = c * 1.05, R = Math.hypot(phx, phz) + pad;
      for (let gj = Math.max(0, Math.floor((wz - R) / c + M)); gj <= Math.min(G - 1, Math.ceil((wz + R) / c + M)); gj++)
        for (let gi = Math.max(0, Math.floor((wx - R) / c + M)); gi <= Math.min(G - 1, Math.ceil((wx + R) / c + M)); gi++) {
          const ex = (gi - M) * c - wx, ez = (gj - M) * c - wz, mx = ex * pc - ez * ps, mz = ex * ps + ez * pc;
          if (Math.abs(mx) < phx + pad && Math.abs(mz) < phz + pad) { const k = gj * G + gi; H[k] = Math.min(H[k], H0[k] - depth); if (wm && wm.isW[k]) { wm.isW[k] = 0; wm.L[k] = NaN; } }
        }
    }
    const water = wm ? buildWater(wm, H, infl, c) : null;
    ctx = { infl, isW: water ? water.isW : null };
    const terrain = buildTerrain(H, infl, lod, c, tx, ty, ox, oz);
    if (lod === 1) bctx = { layers, H: new Float32Array(H), usaP };
    const roadMesh = roads.length ? buildRoads(roads, H, c, tileSize) : null;
    const structures = layers ? buildStructures(roads, layers.waterway, H, c, tileSize) : null;
    const segs = buildSegs(roads);
    const transfer = [H.buffer, segs.buffer, ...Object.values(terrain).map(a => a.buffer)];
    if (roadMesh) for (const part of [roadMesh.road, roadMesh.verge]) transfer.push(...Object.values(part).map(a => a.buffer));
    const waterOut = water ? { level: water.level, current: water.current, mesh: water.mesh } : null;
    if (water) { transfer.push(water.level.buffer, water.current.buffer); if (water.mesh) transfer.push(...Object.values(water.mesh).map(a => a.buffer)); }
    const junc = new Float32Array(conn.junctions), stops = new Float32Array(conn.stops), rings = new Float32Array(conn.rings);
    transfer.push(junc.buffer, stops.buffer, rings.buffer);
    if (structures) transfer.push(...Object.values(structures.mesh).map(a => a.buffer), structures.col.buffer);
    postMessage({ type: 'terrain', id, ok: true, tx, ty, lod, H, terrain, roads: roadMesh, segs, water: waterOut, junc, stops, rings, structures }, transfer);
  } catch (err) {
    postMessage({ type: 'terrain', id, ok: false, tx, ty, lod, error: String(err) });
    return;
  }
  if (bctx) {
    try {
      const b = buildBuildings(bctx.layers, await bctx.usaP, bctx.H, c, tileSize, lms);
      if (b) postMessage({ type: 'buildings', id, tx, ty, ...b }, [...(b.mesh ? Object.values(b.mesh).map(a => a.buffer) : []), b.foot.buffer, b.footOff.buffer]);
    } catch (err) { postMessage({ type: 'log', msg: 'buildings failed ' + err }); }
  }
  const im = await imageryP;
  if (!im) return;
  let trees = null;
  if (im.mat) {
    const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * (ty + 0.5) / NT))) * 180 / Math.PI;
    const lon = (tx + 0.5) / NT * 360 - 180;
    try { trees = detectTrees(im, ctx, H0, c, tileSize, lat, lon, tx * tileSize, ty * tileSize); } catch (err) { trees = null; }
  }
  const transfer = [im.photo.buffer];
  for (const k of ['wa', 'wb', 'mat', 'code']) if (im[k]) transfer.push(im[k].buffer);
  if (trees) transfer.push(trees.buffer);
  postMessage({ type: 'imagery', id, tx, ty, lod, ...im, trees }, transfer);
};
