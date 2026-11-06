'use strict';
// Chase mode's minimap: a real map underneath (OpenStreetMap, dark OSM or satellite, warped onto the run's map
// plane by tiles.py ground()), and over it any of the viewer's products from the run's newest radar frame -- the
// map products (composite reflectivity, UH, winds ...) or a radar's own sweeps (REF, VEL, SRV, ZDR, CC, SW) from
// the nearest radar or one picked -- in the viewer's colour tables (colors.js).  The game draws the roads, the
// warnings, the tornadoes and the cars on top.
const MiniMap = (() => {
  const GRID = [
    ['cref', 'Composite reflectivity'], ['ref1km', '1 km reflectivity'], ['reflow', 'Lowest-level reflectivity'],
    ['thp', 'Surface θ′ (cold pools)'], ['wmax', 'Max updraft'], ['uh25', '2–5 km updraft helicity'],
    ['wspd', 'Lowest-level wind'], ['vort', 'Lowest-level vorticity'], ['rain', 'Accumulated rain'],
    ['uhtrack', 'Storm tracks (run-max UH)'], ['windtrack', 'Wind swaths (run-max)'], ['ir', 'Satellite IR (simulated)'],
  ];
  const RADAR = [['REF', 'Radar reflectivity'], ['VEL', 'Radar velocity'], ['SRV', 'Storm-relative velocity'],
                 ['ZDR', 'Differential reflectivity'], ['RHO', 'Correlation coefficient'], ['SW', 'Spectrum width']];
  const SHORT = { cref: 'CREF', ref1km: 'REF 1 km', reflow: 'REF low', thp: 'θ′', wmax: 'W max', uh25: 'UH 2–5', wspd: 'Wind',
                  vort: 'Vort', rain: 'Rain', uhtrack: 'UH tracks', windtrack: 'Wind swaths', ir: 'IR', REF: 'REF', VEL: 'VEL',
                  SRV: 'SRV', ZDR: 'ZDR', RHO: 'CC', SW: 'SW' };
  const st = {
    base: { key: '', img: null, box: null, busy: false },
    data: { key: '', busy: false, kind: null, ppi: null, patches: null, icao: null, at: 0, err: '' },
    frames: { n: 0, at: 0, radars: [] },
    layer: { cv: null, key: '' },
  };
  const hdr = () => (window.ChaseAuth ? window.ChaseAuth() : {});
  const isRadar = p => RADAR.some(r => r[0] === p);

  // ---------------------------------------------------------------- the map picture
  // a box (world km) twice the range around the car, fetched again when the car nears its edge
  function wantBase(base, cx, cy, rangeKm) {
    const b = st.base;
    if (base === 'none') return null;
    const side = rangeKm * 2.2;
    const far = !b.box || Math.abs(cx - (b.box[0] + b.box[2]) / 2) > side * 0.27 || Math.abs(cy - (b.box[1] + b.box[3]) / 2) > side * 0.27 ||
                Math.abs((b.box[2] - b.box[0]) - side) > 1e-6;
    const k = `${base}/${rangeKm}`;
    if ((far || b.key !== k) && !b.busy) {
      b.busy = true;
      const box = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2].map(v => +v.toFixed(3));
      const src = base.startsWith('sat') ? 'sat' : 'osm', labels = base === 'satlabels' ? 1 : 0;
      fetch(`/api/volume/ground?x0=${box[0]}&y0=${box[1]}&x1=${box[2]}&y1=${box[3]}&src=${src}&labels=${labels}&px=900`, { headers: hdr() })
        .then(r => (r.ok ? r.blob() : Promise.reject(new Error(r.status)))).then(createImageBitmap)
        .then(img => { Object.assign(b, { img, box, key: k }); })
        .catch(() => { b.key = k; b.box = box; b.img = null; })
        .finally(() => { b.busy = false; });
    }
    return b.img && b.key === k ? b : (b.img ? b : null);
  }

  // ---------------------------------------------------------------- the product
  async function refresh(product, site, cx, cy) {
    const d = st.data, f = st.frames;
    if (product === 'none' || d.busy) return;
    const now = performance.now();
    if (now - f.at > 5000) {
      f.at = now;
      try {
        const fr = await (await fetch('/api/frames', { headers: hdr() })).json();
        f.n = fr.length; f.radars = fr.length ? fr[fr.length - 1].radars : []; f.t = fr.length ? fr[fr.length - 1].t : null;
      } catch (e) { /* next time */ }
    }
    if (!f.n) { d.err = 'no radar frames yet'; return; }
    let icao = null;
    if (isRadar(product)) {
      icao = site !== 'nearest' && f.radars.includes(site) ? site : nearestRadar(cx, cy, f.radars);
      if (!icao) { d.err = 'no radar in this run'; return; }
    }
    const k = `${product}/${icao}/${f.n}`;
    if (k === d.key) return;
    d.busy = true;
    try {
      if (icao) {
        const r = await fetch(`/api/ppi?i=-1&icao=${icao}&moment=${product}`, { headers: hdr() });
        if (!r.ok) throw new Error(`${icao} ${product}: not in this frame`);
        const { header, data } = unpack(await r.arrayBuffer());
        const pal = Colors.DEFAULT_PALETTES[Colors.PRODUCT_PALETTE[product]];
        Object.assign(d, { kind: 'ppi', ppi: { m: header, codes: data, lut: Colors.buildLUT(pal, header.scale, header.offset) }, patches: null, icao });
      } else {
        const r = await fetch(`/api/frame?i=-1&product=${product}`, { headers: hdr() });
        if (!r.ok) throw new Error(`${product}: not in this frame`);
        const { header, data } = unpack(await r.arrayBuffer());
        const [scale, offset] = header.codec, pal = Colors.DEFAULT_PALETTES[Colors.PRODUCT_PALETTE[product]];
        const lut = Colors.buildLUT(pal, scale, offset);
        const patches = header.patches.map(p => {
          const cv = Object.assign(document.createElement('canvas'), { width: p.nx, height: p.ny });
          const g = cv.getContext('2d'), img = g.createImageData(p.nx, p.ny), codes = data.subarray(p.offset, p.offset + p.nx * p.ny);
          for (let i = 0; i < codes.length; i++) {
            const c = codes[i] * 4, o = i * 4;
            img.data[o] = lut[c]; img.data[o + 1] = lut[c + 1]; img.data[o + 2] = lut[c + 2]; img.data[o + 3] = lut[c + 3];
          }
          g.putImageData(img, 0, 0);
          return { x0: p.x0 / 1000, y0: p.y0 / 1000, x1: p.x1 / 1000, y1: p.y1 / 1000, cv };
        });
        Object.assign(d, { kind: 'grid', patches, ppi: null, icao: null });
      }
      d.key = k; d.err = ''; st.layer.key = '';
    } catch (e) { d.err = e.message; d.key = k; }
    d.busy = false;
  }
  function nearestRadar(cx, cy, list) {
    const rs = (Chase.st.run && Chase.st.run.radars) || [];
    let best = null, bd = 1e18;
    for (const r of rs) {
      if (!list.includes(r.icao)) continue;
      const d = Math.hypot(r.x / 1000 - cx, r.y / 1000 - cy);
      if (d < bd) { bd = d; best = r.icao; }
    }
    return best || list[0] || null;
  }

  // a radar's sweep under the minimap (S px a side, range R km around cx, cy), at half the minimap's pixels
  function ppiLayer(S, R, cx, cy) {
    const P = st.data.ppi, m = P.m, L = st.layer;
    const step = 2, w = Math.ceil(S / step), key = `${st.data.key}/${S}/${R}/${cx.toFixed(2)}/${cy.toFixed(2)}`;
    if (L.key === key && L.cv) return L.cv;
    if (!L.cv || L.cv.width !== w) L.cv = Object.assign(document.createElement('canvas'), { width: w, height: w });
    const g = L.cv.getContext('2d'), img = g.createImageData(w, w), px = img.data, lut = P.lut, codes = P.codes;
    const rx = m.x / 1000, ry = m.y / 1000, nrad = m.nrad, ng = m.ngate, first = m.first / 1000, gate = m.gate / 1000;
    const cosE = Math.cos(m.elev * Math.PI / 180), daz = 360 / nrad, conv = m.conv || 0, kmpp = 2 * R / w;
    for (let j = 0; j < w; j++) {
      const wy = cy + R - (j + 0.5) * kmpp - ry;
      for (let i = 0; i < w; i++) {
        const wx = cx - R + (i + 0.5) * kmpp - rx;
        const r = Math.hypot(wx, wy) / cosE, gi = Math.floor((r - first) / gate + 0.5);
        if (gi < 0 || gi >= ng) continue;
        let az = Math.atan2(wx, wy) * 57.29578 + conv;
        if (az < 0) az += 360; if (az >= 360) az -= 360;
        const code = codes[(Math.floor(az / daz) % nrad) * ng + gi];
        if (!code) continue;
        const o = (j * w + i) * 4, q = code * 4;
        px[o] = lut[q]; px[o + 1] = lut[q + 1]; px[o + 2] = lut[q + 2]; px[o + 3] = lut[q + 3];
      }
    }
    g.putImageData(img, 0, 0);
    L.key = key;
    return L.cv;
  }

  // under the minimap: the map, the product (opacity a); P(x m, y m) -> screen
  function draw(g, x0, y0, S, cx, cy, R, opt) {
    const sc = S / (2 * R), Pk = (x, y) => [x0 + S / 2 + (x - cx) * sc, y0 + S / 2 - (y - cy) * sc];
    const b = wantBase(opt.base, cx, cy, R);
    if (b && b.img) {
      const [ax, ay] = Pk(b.box[0], b.box[3]), [bx, by] = Pk(b.box[2], b.box[1]);
      g.save();
      if (opt.base === 'osmdark') g.filter = 'invert(1) hue-rotate(180deg) brightness(0.85) contrast(0.85) saturate(0.6)';
      g.drawImage(b.img, ax, ay, bx - ax, by - ay);
      g.restore();
      g.fillStyle = `rgba(0,0,0,${opt.dim})`; g.fillRect(x0, y0, S, S);
    }
    refresh(opt.product, opt.site, cx, cy);
    const d = st.data;
    if (opt.product !== 'none' && d.key.startsWith(opt.product + '/')) {
      g.save(); g.globalAlpha = opt.opacity;
      if (d.kind === 'ppi' && d.ppi) {
        g.imageSmoothingEnabled = true;
        g.drawImage(ppiLayer(S, R, cx, cy), x0, y0, S, S);
      } else if (d.kind === 'grid' && d.patches) {
        g.imageSmoothingEnabled = false;
        for (const p of d.patches) {
          // canvas row 0 is the patch's south edge: drawn flipped
          const [ax, ay] = Pk(p.x0, p.y0), [bx, by] = Pk(p.x1, p.y1);
          g.save(); g.translate(ax, ay); g.scale((bx - ax) / p.cv.width, (by - ay) / p.cv.height);
          g.drawImage(p.cv, 0, 0); g.restore();
        }
      }
      g.restore();
    }
  }
  function label(product) {
    const d = st.data;
    if (product === 'none') return '';
    return `${d.icao && isRadar(product) ? d.icao + ' ' : ''}${SHORT[product] || product}${d.err ? ' · ' + d.err : ''}`;
  }

  return { draw, label, GRID, RADAR, st, radars: () => st.frames.radars };
})();
window.MiniMap = MiniMap;
