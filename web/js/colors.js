'use strict';
// Color tables: built-in defaults and GR2Analyst/GRLevelX .pal tables, turned into
// 256-entry lookup tables for 8-bit codes (value = (code - offset) / scale, code 0 = no data).

const DEFAULT_PALETTES = {
  dbz: { units: 'dBZ', stops: [
    { v: -10, c: [40, 40, 50, 0] }, { v: 0, c: [60, 60, 75, 120] }, { v: 5, c: [100, 100, 120, 200] },
    { v: 10, c: [4, 233, 231, 255] }, { v: 15, c: [1, 159, 244, 255] }, { v: 20, c: [3, 0, 244, 255] },
    { v: 25, c: [2, 253, 2, 255] }, { v: 30, c: [1, 197, 1, 255] }, { v: 35, c: [0, 142, 0, 255] },
    { v: 40, c: [253, 248, 2, 255] }, { v: 45, c: [229, 188, 0, 255] }, { v: 50, c: [253, 149, 0, 255] },
    { v: 55, c: [253, 0, 0, 255] }, { v: 60, c: [212, 0, 0, 255] }, { v: 65, c: [188, 0, 0, 255] },
    { v: 70, c: [248, 0, 253, 255] }, { v: 75, c: [152, 84, 198, 255] }, { v: 80, c: [253, 253, 253, 255] },
  ] },
  vel: { units: 'm/s', stops: [
    { v: -64, c: [255, 0, 255, 255] }, { v: -50, c: [128, 255, 255, 255] }, { v: -40, c: [0, 255, 128, 255] },
    { v: -25, c: [0, 200, 0, 255] }, { v: -10, c: [0, 110, 0, 255] }, { v: -1, c: [70, 100, 70, 255] },
    { v: 0, c: [110, 110, 110, 255] }, { v: 1, c: [100, 70, 70, 255] }, { v: 10, c: [120, 0, 0, 255] },
    { v: 25, c: [220, 0, 0, 255] }, { v: 40, c: [255, 130, 130, 255] }, { v: 50, c: [255, 220, 120, 255] },
    { v: 64, c: [255, 255, 255, 255] },
  ] },
  zdr: { units: 'dB', stops: [
    { v: -2, c: [60, 60, 60, 255] }, { v: -0.5, c: [120, 120, 120, 255] }, { v: 0, c: [160, 160, 160, 255] },
    { v: 0.5, c: [30, 30, 200, 255] }, { v: 1, c: [40, 150, 240, 255] }, { v: 1.5, c: [40, 220, 220, 255] },
    { v: 2, c: [40, 200, 40, 255] }, { v: 2.5, c: [250, 250, 30, 255] }, { v: 3, c: [250, 160, 20, 255] },
    { v: 4, c: [230, 20, 20, 255] }, { v: 5, c: [250, 120, 230, 255] }, { v: 6, c: [255, 255, 255, 255] },
  ] },
  cc: { units: '', stops: [
    { v: 0.2, c: [20, 20, 60, 255] }, { v: 0.6, c: [60, 60, 200, 255] }, { v: 0.75, c: [40, 160, 230, 255] },
    { v: 0.85, c: [40, 200, 60, 255] }, { v: 0.9, c: [200, 230, 40, 255] }, { v: 0.95, c: [250, 170, 30, 255] },
    { v: 0.97, c: [230, 40, 40, 255] }, { v: 1.0, c: [190, 20, 140, 255] }, { v: 1.05, c: [255, 220, 255, 255] },
  ] },
  sw: { units: 'm/s', stops: [
    { v: 0, c: [30, 30, 30, 255] }, { v: 2, c: [90, 90, 90, 255] }, { v: 4, c: [40, 120, 220, 255] },
    { v: 6, c: [40, 200, 80, 255] }, { v: 8, c: [240, 220, 40, 255] }, { v: 10, c: [240, 110, 30, 255] },
    { v: 14, c: [230, 30, 30, 255] }, { v: 20, c: [255, 255, 255, 255] },
  ] },
  thp: { units: 'K', stops: [
    { v: -12, c: [120, 0, 160, 255] }, { v: -8, c: [40, 40, 200, 255] }, { v: -5, c: [30, 130, 230, 255] },
    { v: -3, c: [60, 200, 230, 255] }, { v: -1.5, c: [170, 230, 240, 255] }, { v: -0.5, c: [235, 240, 245, 120] },
    { v: 0, c: [240, 240, 240, 0] }, { v: 0.5, c: [250, 230, 210, 120] }, { v: 2, c: [250, 170, 120, 255] },
    { v: 5, c: [220, 60, 40, 255] },
  ] },
  w: { units: 'm/s', stops: [
    { v: 0, c: [0, 0, 0, 0] }, { v: 2, c: [70, 70, 110, 120] }, { v: 5, c: [60, 120, 220, 255] },
    { v: 10, c: [40, 200, 120, 255] }, { v: 20, c: [240, 230, 40, 255] }, { v: 30, c: [245, 140, 30, 255] },
    { v: 45, c: [230, 30, 30, 255] }, { v: 60, c: [250, 120, 240, 255] }, { v: 80, c: [255, 255, 255, 255] },
  ] },
  uh: { units: 'm2/s2', stops: [
    { v: 0, c: [0, 0, 0, 0] }, { v: 25, c: [90, 90, 120, 140] }, { v: 75, c: [60, 150, 230, 255] },
    { v: 150, c: [60, 220, 90, 255] }, { v: 300, c: [245, 230, 40, 255] }, { v: 500, c: [245, 130, 30, 255] },
    { v: 800, c: [230, 30, 30, 255] }, { v: 1200, c: [250, 110, 240, 255] }, { v: 2000, c: [255, 255, 255, 255] },
  ] },
  wspd: { units: 'm/s', stops: [
    { v: 0, c: [20, 20, 40, 0] }, { v: 5, c: [50, 70, 120, 160] }, { v: 10, c: [40, 140, 210, 255] },
    { v: 15, c: [50, 200, 120, 255] }, { v: 20, c: [240, 230, 40, 255] }, { v: 25, c: [245, 150, 30, 255] },
    { v: 32, c: [230, 30, 30, 255] }, { v: 40, c: [250, 100, 240, 255] }, { v: 55, c: [255, 255, 255, 255] },
  ] },
  vort: { units: '1/s', stops: [
    { v: -0.03, c: [40, 40, 220, 255] }, { v: -0.01, c: [90, 150, 240, 255] }, { v: -0.002, c: [160, 200, 240, 0] },
    { v: 0.002, c: [240, 200, 160, 0] }, { v: 0.01, c: [245, 150, 60, 255] }, { v: 0.03, c: [230, 30, 30, 255] },
    { v: 0.06, c: [255, 120, 255, 255] }, { v: 0.1, c: [255, 255, 255, 255] }, { v: 0.25, c: [110, 255, 255, 255] },
  ] },
  rain: { units: 'log2(1+mm)', stops: [
    { v: 0, c: [0, 0, 0, 0] }, { v: 1, c: [60, 160, 60, 200] }, { v: 3, c: [40, 200, 60, 255] },
    { v: 5, c: [240, 220, 40, 255] }, { v: 6.5, c: [240, 120, 30, 255] }, { v: 8, c: [220, 30, 200, 255] },
  ] },
};

// hurricane runs: the simulated IR in SatPainter's Tropical Tidbits table (irtable.js, -90..+40 C every 0.5 C,
// held flat beyond its ends), the SST, sea-level pressure and the 10 m wind by Saffir-Simpson thresholds
if (window.IR_TT) {
  const hex = h => [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16), 255];
  DEFAULT_PALETTES.ir = { units: '°C', stops: [{ v: -120, c: hex(IR_TT[0]) }]
    .concat(IR_TT.map((h, i) => ({ v: -90 + 0.5 * i, c: hex(h) }))) };
}
DEFAULT_PALETTES.sst = { units: '°C', stops: [
  { v: 0, c: [40, 20, 90, 255] }, { v: 10, c: [40, 70, 180, 255] }, { v: 18, c: [40, 150, 220, 255] },
  { v: 22, c: [60, 200, 190, 255] }, { v: 25, c: [120, 210, 90, 255] }, { v: 26.5, c: [235, 230, 60, 255] },
  { v: 28, c: [245, 160, 40, 255] }, { v: 29, c: [235, 80, 30, 255] }, { v: 30, c: [200, 20, 40, 255] },
  { v: 31, c: [190, 30, 160, 255] }, { v: 32, c: [255, 210, 255, 255] },
] };
DEFAULT_PALETTES.slp = { units: 'hPa', stops: [
  { v: 880, c: [255, 255, 255, 255] }, { v: 900, c: [200, 60, 220, 255] }, { v: 930, c: [200, 30, 40, 255] },
  { v: 960, c: [245, 140, 30, 255] }, { v: 985, c: [240, 225, 60, 255] }, { v: 1000, c: [80, 190, 120, 255] },
  { v: 1008, c: [60, 120, 200, 255] }, { v: 1015, c: [40, 50, 110, 255] }, { v: 1030, c: [20, 20, 40, 255] },
] };
DEFAULT_PALETTES.hwind = { units: 'm/s', stops: [
  { v: 0, c: [20, 30, 60, 0] }, { v: 5, c: [40, 70, 130, 150] }, { v: 10, c: [50, 130, 210, 255] },
  { v: 17.5, c: [60, 200, 110, 255] }, { v: 25.7, c: [240, 230, 50, 255] },
  { v: 32.9, c: [250, 160, 40, 255] }, { v: 42.7, c: [235, 70, 40, 255] }, { v: 49.4, c: [200, 20, 60, 255] },
  { v: 58.1, c: [210, 60, 210, 255] }, { v: 70.5, c: [255, 255, 255, 255] },
] };

// which default palette each product uses
const PRODUCT_PALETTE = {
  cref: 'dbz', ref1km: 'dbz', reflow: 'dbz', REF: 'dbz', VEL: 'vel', SRV: 'vel', SW: 'sw', ZDR: 'zdr', RHO: 'cc',
  thp: 'thp', wmax: 'w', uh25: 'uh', wspd: 'wspd', vort: 'vort', rain: 'rain', uhtrack: 'uh', windtrack: 'wspd',
  ir: 'ir', sst: 'sst', slp: 'slp', wind10: 'hwind',
};
// GR palette "Product:" keys that fit each family
const PAL_FAMILY = { dbz: ['BR', 'REF', 'DBZ', 'CR'], vel: ['BV', 'SRV', 'VEL', 'SR'], zdr: ['ZDR'], cc: ['CC', 'RHO'], sw: ['SW'] };

function colorAt(stops, v) {
  if (!stops.length) return [0, 0, 0, 0];
  if (v < stops[0].v) return null;
  for (let i = 0; i < stops.length; i++) {
    const s = stops[i];
    const n = stops[i + 1];
    if (!n || v < n.v) {
      if (!n) return s.c;
      if (s.solid) return s.c;
      const c2 = s.c2 || n.c;
      const t = (v - s.v) / Math.max(n.v - s.v, 1e-9);
      return [0, 1, 2, 3].map(k => s.c[k] + (c2[k] - s.c[k]) * t);
    }
  }
  return stops[stops.length - 1].c;
}

// LUT for codes: value = (code - offset) / scale; code 0 transparent
function buildLUT(pal, scale, offset, minAlpha = 0) {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let code = 1; code < 256; code++) {
    const v = (code - offset) / scale;
    const c = colorAt(pal.stops, v);
    if (!c) continue;
    lut[code * 4] = c[0];
    lut[code * 4 + 1] = c[1];
    lut[code * 4 + 2] = c[2];
    lut[code * 4 + 3] = Math.max(c[3] === undefined ? 255 : c[3], minAlpha);
  }
  if (offset === 129) {             // velocity / spectrum width: code 1 is range folded (RF), purple as in GR2
    lut[4] = 119; lut[5] = 0; lut[6] = 125; lut[7] = 255;
  }
  return lut;
}

// GR tables in the units the codes carry (GR velocity tables are in knots by default)
function adaptGR(pal, family) {
  let stops = pal.stops.map(s => ({ ...s }));
  const units = (pal.units || '').toLowerCase();
  // a GR Scale: converts m/s into the table's units, so with one the Units: line is only a label
  if (pal.scale && pal.scale !== 1) {
    stops = stops.map(s => ({ ...s, v: s.v / pal.scale }));
  } else if (family === 'vel' && units.includes('mph')) {
    stops = stops.map(s => ({ ...s, v: s.v * 0.44704 }));
  } else if (family === 'vel' && (units.includes('kt') || units.includes('knot') || !units)) {
    stops = stops.map(s => ({ ...s, v: s.v * 0.514444 }));
  }
  return { units: pal.units, stops };
}

function legendGradient(pal, lo, hi, n = 64) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const v = lo + (hi - lo) * i / (n - 1);
    out.push(colorAt(pal.stops, v) || [0, 0, 0, 0]);
  }
  return out;
}

window.Colors = { DEFAULT_PALETTES, PRODUCT_PALETTE, PAL_FAMILY, buildLUT, adaptGR, colorAt, legendGradient };
