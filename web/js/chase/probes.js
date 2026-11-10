'use strict';
// Chase mode's mesonet probe cars (web/models/cars/probes/PROBE_*_LOD0/1.glb, the user's models): read straight
// from the glTF binary here, no baking.  Each car comes apart into what moves: the body (everything fixed to it),
// for each wheel its hub (the brake caliper: steers, does not roll) and its rolling part, the steering wheel, the
// anemometer's vane (turns into the wind over the car) and its propeller.  Every vertex keeps its material's colour
// (sRGB) and a class: the paint (recoloured to the player's colour), each group of lights (switched on by the car's
// state, render.js), the glass (dithered so the cabin shows through).
// Model axes here: x forward, y left, z up, metres, origin on the ground at the body's middle (the files' own:
// +Z forward, +Y up, so +X left: a rotation, nothing mirrored).
const Probes = (() => {
  const FILES = { p_suv: 'PROBE_1_SUV', p_pickup: 'PROBE_2_Pickup', p_sedan: 'PROBE_3_Sedan', p_van: 'PROBE_4_Minivan' };
  // vertex classes (render.js CAR_FS)
  const CLASS = { Paint_Primary: 1, Paint_Roof: 2, Livery_Accent: 3, Light_Head: 4, Light_DRL: 5, Light_Tail: 6, Light_Brake: 7,
                  Light_Turn: 8, Light_Reverse: 9, LightBar_A: 10, LightBar_B: 11, LightBar_B_White: 11, Screen_Radar: 12,
                  Glass_Front: 13, Glass_Privacy: 14 };
  const SKIP = new Set(['Collider', 'Decal_Door']);          // the collision hull; the door decal (a textured overlay)
  // textured materials, as one colour (sRGB 0..1): their textures are small details
  const FLAT = { License_Plate: [0.86, 0.87, 0.84], Screen_Radar: [0.05, 0.12, 0.08] };
  const cache = new Map();                                   // kind/lod -> data | promise | null (failed)

  const lin2srgb = v => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
  // column-major 4x4
  const I4 = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  function mul(a, b) {
    const o = new Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  }
  function trs(n) {
    if (n.matrix) return n.matrix.slice();
    const [tx, ty, tz] = n.translation || [0, 0, 0], [x, y, z, w] = n.rotation || [0, 0, 0, 1], [sx, sy, sz] = n.scale || [1, 1, 1];
    return [(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + w * z) * sx, 2 * (x * z - w * y) * sx, 0,
            2 * (x * y - w * z) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + w * x) * sy, 0,
            2 * (x * z + w * y) * sz, 2 * (y * z - w * x) * sz, (1 - 2 * (x * x + y * y)) * sz, 0, tx, ty, tz, 1];
  }
  const xf = (m, v, w = 1) => [m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * w, m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * w,
                               m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * w];
  const toModel = g => [g[2], g[0], g[1]];                  // the file's (x left, y up, z forward) -> x fwd, y left, z up

  function accessor(J, bin, i) {
    const a = J.accessors[i], bv = J.bufferViews[a.bufferView], n = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[a.type];
    const T = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }[a.componentType];
    const off = (bv.byteOffset || 0) + (a.byteOffset || 0), stride = bv.byteStride || n * T.BYTES_PER_ELEMENT;
    if (stride === n * T.BYTES_PER_ELEMENT) return { arr: new T(bin, off, a.count * n), n };
    const out = new T(a.count * n), dv = new DataView(bin);
    const get = { 5121: 'getUint8', 5123: 'getUint16', 5125: 'getUint32', 5126: 'getFloat32' }[a.componentType];
    for (let k = 0; k < a.count; k++) for (let j = 0; j < n; j++) out[k * n + j] = dv[get](off + k * stride + j * T.BYTES_PER_ELEMENT, true);
    return { arr: out, n };
  }

  function parse(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('not a glb');
    const jl = dv.getUint32(12, true), J = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jl)));
    const bo = 20 + jl, bl = dv.getUint32(bo, true), bin = buf.slice(bo + 8, bo + 8 + bl);
    const root = J.nodes[J.scenes[J.scene || 0].nodes[0]], ex = root.extras || {};
    const mats = J.materials.map(m => {
      const f = (m.pbrMetallicRoughness || {}).baseColorFactor || [1, 1, 1, 1];
      const cls = CLASS[m.name] || 0, es = (m.extensions && m.extensions.KHR_materials_emissive_strength) ? m.extensions.KHR_materials_emissive_strength.emissiveStrength || 1 : 1;
      return { name: m.name, cls, skip: SKIP.has(m.name), col: FLAT[m.name] || f.slice(0, 3).map(lin2srgb),
               em: m.emissiveFactor ? m.emissiveFactor.map(v => Math.min(1, v * Math.max(1.5, Math.min(es, 4)))) : null };
    });
    // the parts: by the moving node each mesh hangs under
    const parts = new Map(), pivots = {};
    const part = (key, info) => { if (!parts.has(key)) parts.set(key, { key, ...info, P: [], N: [], C: [], X: [] }); return parts.get(key); };
    function walk(ni, parent, owner) {
      const n = J.nodes[ni], m = mul(parent, trs(n)), name = n.name || '';
      let own = owner;
      if (/^WheelSpin_(FL|FR|RL|RR)$/.test(name)) own = 'spin_' + name.slice(10);
      else if (/^Wheel_(FL|FR|RL|RR)$/.test(name)) own = 'hub_' + name.slice(6);
      else if (name === 'SteeringWheel') own = 'steering';
      else if (name === 'Anemometer_Vane') own = 'vane';
      else if (name === 'Anemometer_Prop') own = 'prop';
      if (own !== owner || name === 'Model') pivots[own] = { o: toModel(xf(m, [0, 0, 0])), x: toModel(xf(m, [1, 0, 0], 0)), y: toModel(xf(m, [0, 1, 0], 0)), z: toModel(xf(m, [0, 0, 1], 0)) };
      if (n.mesh !== undefined && name !== 'COL_Hull') {
        const pt = part(own, {});
        for (const pr of J.meshes[n.mesh].primitives) {
          if ((pr.mode === undefined ? 4 : pr.mode) !== 4 || pr.indices === undefined) continue;
          const mt = mats[pr.material] || { col: [0.5, 0.5, 0.5], cls: 0 };
          if (mt.skip) continue;
          const P = accessor(J, bin, pr.attributes.POSITION).arr, N = pr.attributes.NORMAL !== undefined ? accessor(J, bin, pr.attributes.NORMAL).arr : null;
          const X = accessor(J, bin, pr.indices).arr, base = pt.P.length / 3;
          const nm = m;                                      // (no non-uniform scales in these files)
          for (let k = 0; k < P.length; k += 3) {
            pt.P.push(...toModel(xf(m, [P[k], P[k + 1], P[k + 2]])));
            const q = N ? toModel(xf(nm, [N[k], N[k + 1], N[k + 2]], 0)) : [0, 0, 1], l = Math.hypot(...q) || 1;
            pt.N.push(q[0] / l, q[1] / l, q[2] / l);
            pt.C.push(Math.round(mt.col[0] * 255), Math.round(mt.col[1] * 255), Math.round(mt.col[2] * 255), mt.cls);
          }
          for (let k = 0; k < X.length; k++) pt.X.push(base + X[k]);
        }
      }
      for (const c of n.children || []) walk(c, m, own);
    }
    walk(J.scenes[J.scene || 0].nodes[0], I4(), 'body');
    // each part's vertices from its pivot (the moving node's origin); the body's from the model's origin
    const out = [];
    for (const pt of parts.values()) {
      if (!pt.X.length) continue;
      const pv = pt.key === 'body' ? null : pivots[pt.key];
      const o = pv ? pv.o : [0, 0, 0], P = new Float32Array(pt.P.length);
      for (let k = 0; k < P.length; k += 3) { P[k] = pt.P[k] - o[0]; P[k + 1] = pt.P[k + 1] - o[1]; P[k + 2] = pt.P[k + 2] - o[2]; }
      out.push({ key: pt.key, pivot: o, axes: pv, P, N: new Float32Array(pt.N), C: new Uint8Array(pt.C), X: new Uint32Array(pt.X) });
    }
    // the classes' glow colours (the lights' emissive)
    const em = new Array(16).fill(null);
    for (const m of mats) if (m.cls && m.em) em[m.cls] = m.em;
    const vane = out.find(p => p.key === 'vane');
    let vaneYaw0 = 0;
    if (vane) {
      // which way the vane points at rest: away from its tail (the fin: most of its vertices behind the pivot)
      let cx = 0, cy = 0;
      for (let k = 0; k < vane.P.length; k += 3) { cx += vane.P[k]; cy += vane.P[k + 1]; }
      vaneYaw0 = Math.atan2(-cy, -cx);
    }
    const hubs = {};
    for (const w of ['FL', 'FR', 'RL', 'RR']) if (pivots['hub_' + w]) hubs[w] = pivots['hub_' + w].o;
    let bb = [Infinity, -Infinity, Infinity, -Infinity, 0];
    const body = out.find(p => p.key === 'body');
    for (let k = 0; k < body.P.length; k += 3) {
      bb[0] = Math.min(bb[0], body.P[k]); bb[1] = Math.max(bb[1], body.P[k]);
      bb[2] = Math.min(bb[2], body.P[k + 1]); bb[3] = Math.max(bb[3], body.P[k + 1]); bb[4] = Math.max(bb[4], body.P[k + 2]);
    }
    return { parts: out, em, hubs, vaneYaw0, steerAxis: pivots.steering ? pivots.steering.x : [1, 0, 0],
             propAxis: pivots.prop ? pivots.prop.x : [1, 0, 0], wheelR: ex.wheelRadius || 0.38, bbox: bb, tris: out.reduce((s, p) => s + p.X.length / 3, 0) };
  }

  // the car's model (lod 0: the player's own, 1: everyone else's): its data, or null while it loads (or failed)
  function get(kind, lod) {
    const key = kind + '/' + lod;
    const c = cache.get(key);
    if (c && !(c instanceof Promise)) return c === 'failed' ? null : c;
    if (!c && FILES[kind]) {
      cache.set(key, fetch(`/static/models/cars/probes/${FILES[kind]}_LOD${lod}.glb`)
        .then(r => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.status))))
        .then(b => { const d = parse(b); cache.set(key, d); return d; })
        .catch(e => { console.warn('chase: probe car', key, e); cache.set(key, 'failed'); }));
    }
    return null;
  }
  const isProbe = kind => !!FILES[kind];
  return { get, isProbe, FILES, parse };
})();
window.Probes = Probes;
