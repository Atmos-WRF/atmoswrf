'use strict';
// Chase mode's cars, made from code (no model files): tapered boxes for the body, the glasshouse and roof, bumpers,
// lights and the chase gear on the roof; a spoked wheel.  Body axes from the car's centre of gravity: x forward,
// y left, z up, metres.  Each vertex: position, normal, sRGB colour (9 floats).
const CarMesh = (() => {
  // a six-sided solid from its eight corners: [x0 y0 z0] (rear-right-bottom) ... in the order
  // rrb, rlb, frb, flb, rrt, rlt, frt, flt (r/f: rear/front, r/l: right/left, b/t: bottom/top)
  function hexa(out, P, col, skip = '') {
    const c = [0, 1, 2].map(i => P.reduce((s, p) => s + p[i], 0) / 8);
    const faces = { b: [0, 1, 3, 2], t: [4, 6, 7, 5], r: [0, 2, 6, 4], l: [1, 5, 7, 3], k: [0, 4, 5, 1], f: [2, 3, 7, 6] };
    for (const [name, f] of Object.entries(faces)) {
      if (skip.includes(name)) continue;
      quad(out, P[f[0]], P[f[1]], P[f[2]], P[f[3]], col, c);
    }
  }
  function tri(out, a, b, c, col, centre) {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const l = Math.hypot(...n);
    if (l < 1e-9) return;
    n = n.map(x => x / l);
    const m = [(a[0] + b[0] + c[0]) / 3 - centre[0], (a[1] + b[1] + c[1]) / 3 - centre[1], (a[2] + b[2] + c[2]) / 3 - centre[2]];
    if (n[0] * m[0] + n[1] * m[1] + n[2] * m[2] < 0) { n = n.map(x => -x); [b, c] = [c, b]; }   // outward, counter-clockwise
    for (const p of [a, b, c]) out.push(p[0], p[1], p[2], n[0], n[1], n[2], col[0], col[1], col[2]);
  }
  function quad(out, a, b, c, d, col, centre) { tri(out, a, b, c, col, centre); tri(out, a, c, d, col, centre); }
  // an axis-aligned box, optionally narrower / shorter at its top (taper: [dx front, dx rear, dy each side])
  function box(out, x0, x1, y0, y1, z0, z1, col, top = [0, 0, 0], skip = '') {
    const [tf, tr, ty] = top;
    hexa(out, [[x0, y0, z0], [x0, y1, z0], [x1, y0, z0], [x1, y1, z0],
               [x0 + tr, y0 + ty, z1], [x0 + tr, y1 - ty, z1], [x1 - tf, y0 + ty, z1], [x1 - tf, y1 - ty, z1]], col, skip);
  }

  const srgb = c => c.map(v => v / 255);

  function build(S, kind, color) {
    const body = [];
    const [Ld, Wd, Hd] = S.dims, cg = S.cgH, a = S.cgFront, b = S.wheelbase - a;
    const xF = a + (Ld - S.wheelbase) / 2, xR = -b - (Ld - S.wheelbase) / 2;
    const z = h => h - cg;                                  // heights above the ground -> from the CG
    const y0 = -Wd / 2, y1 = Wd / 2;
    const C = srgb(color), dark = [0.06, 0.06, 0.07], glass = [0.08, 0.10, 0.13], chrome = [0.55, 0.56, 0.58];
    const black = [0.04, 0.04, 0.045], red = [0.55, 0.05, 0.04], amber = [0.85, 0.55, 0.12], white = [0.9, 0.9, 0.85];
    const gc = S.ground, R = S.wheelR;
    if (kind === 'truck') {
      // an armoured intercept vehicle: slab sides, a sloped armour nose, a turret-like observation pod on top
      const belt = gc + 1.25;
      box(body, xR, xF - 0.9, y0, y1, z(gc + 0.1), z(belt), C);
      box(body, xF - 0.9, xF, y0 + 0.05, y1 - 0.05, z(gc + 0.25), z(belt - 0.25), C, [0.35, 0, 0.05]);
      box(body, xR + 0.1, xF - 1.0, y0 + 0.08, y1 - 0.08, z(belt), z(Hd - 0.12), C, [0.55, 0.05, 0.12]);
      box(body, xF - 1.65, xF - 1.05, y0 + 0.2, y1 - 0.2, z(belt + 0.05), z(Hd - 0.25), glass, [0.38, 0, 0.02]);
      box(body, xR + 0.9, xR + 2.6, -0.55, 0.55, z(Hd - 0.12), z(Hd + 0.28), C.map(v => v * 0.8), [0.2, 0.2, 0.15]);
      box(body, xR + 1.5, xR + 2.0, -0.12, 0.12, z(Hd + 0.28), z(Hd + 0.95), dark);     // the instrument mast
      box(body, xR + 1.3, xR + 2.2, -0.35, 0.35, z(Hd + 0.95), z(Hd + 1.05), chrome);
      box(body, xF - 0.05, xF + 0.12, y0 + 0.1, y1 - 0.1, z(gc + 0.1), z(gc + 0.55), dark);
      box(body, xR - 0.1, xR + 0.02, y0 + 0.1, y1 - 0.1, z(gc + 0.1), z(gc + 0.5), dark);
      for (const s of [-1, 1]) {
        box(body, xF - 0.02, xF + 0.03, s > 0 ? y1 - 0.45 : y0 + 0.15, s > 0 ? y1 - 0.15 : y0 + 0.45, z(belt - 0.6), z(belt - 0.42), white);
        box(body, xR - 0.03, xR + 0.02, s > 0 ? y1 - 0.35 : y0 + 0.1, s > 0 ? y1 - 0.1 : y0 + 0.35, z(belt - 0.4), z(belt - 0.2), red);
        // skirts low over the wheels: the armour keeps the wind from getting under it
        box(body, xR + 0.3, xF - 0.3, s > 0 ? y1 - 0.02 : y0 - 0.04, s > 0 ? y1 + 0.04 : y0 + 0.02, z(gc - 0.08), z(gc + 0.35), C.map(v => v * 0.7));
      }
    } else {
      const suv = kind === 'suv';
      const belt = gc + (suv ? 0.86 : 0.58);
      const hoodL = suv ? 1.25 : 1.35, deck = suv ? 0.18 : 0.9;
      // the lower body, the nose sloping down to the bumper
      box(body, xR, xF - 0.25, y0, y1, z(gc + 0.12), z(belt), C);
      box(body, xF - 0.25, xF, y0 + 0.04, y1 - 0.04, z(gc + 0.18), z(belt - 0.12), C, [0.18, 0, 0.03]);
      // the glasshouse (dark glass), the pillars as its edges, the roof over it
      const wsBase = xF - hoodL, rear = xR + deck, roofF = wsBase - (suv ? 0.75 : 0.95), roofR = rear + (suv ? 0.08 : 0.55);
      const top = Hd - 0.06;
      hexa(body, [[rear, y0 + 0.06, z(belt)], [rear, y1 - 0.06, z(belt)], [wsBase, y0 + 0.06, z(belt)], [wsBase, y1 - 0.06, z(belt)],
                  [roofR, y0 + 0.2, z(top)], [roofR, y1 - 0.2, z(top)], [roofF, y0 + 0.2, z(top)], [roofF, y1 - 0.2, z(top)]], glass);
      hexa(body, [[roofR - 0.02, y0 + 0.18, z(top - 0.01)], [roofR - 0.02, y1 - 0.18, z(top - 0.01)], [roofF + 0.02, y0 + 0.18, z(top - 0.01)],
                  [roofF + 0.02, y1 - 0.18, z(top - 0.01)], [roofR + 0.03, y0 + 0.21, z(Hd)], [roofR + 0.03, y1 - 0.21, z(Hd)],
                  [roofF - 0.03, y0 + 0.21, z(Hd)], [roofF - 0.03, y1 - 0.21, z(Hd)]], C);
      // B pillars
      const bx = (roofF + roofR) / 2 + (suv ? 0.25 : 0.1);
      for (const s of [-1, 1]) {
        const ya = s > 0 ? y1 - 0.065 : y0 + 0.05, yb = s > 0 ? y1 - 0.05 : y0 + 0.065;
        hexa(body, [[bx - 0.06, ya, z(belt)], [bx - 0.06, yb, z(belt)], [bx + 0.06, ya, z(belt)], [bx + 0.06, yb, z(belt)],
                    [bx - 0.06, ya - s * 0.14, z(top)], [bx - 0.06, yb - s * 0.14, z(top)], [bx + 0.06, ya - s * 0.14, z(top)], [bx + 0.06, yb - s * 0.14, z(top)]], C);
      }
      if (!suv) box(body, xR, rear, y0 + 0.04, y1 - 0.04, z(belt), z(belt + 0.06), C);     // the boot lid's lip
      // bumpers, grille, lights
      box(body, xF - 0.08, xF + 0.06, y0 + 0.05, y1 - 0.05, z(gc + 0.1), z(gc + 0.38), dark);
      box(body, xR - 0.06, xR + 0.08, y0 + 0.05, y1 - 0.05, z(gc + 0.1), z(gc + 0.36), dark);
      box(body, xF - 0.06, xF + 0.005, -0.42, 0.42, z(belt - 0.36), z(belt - 0.16), black);
      for (const s of [-1, 1]) {
        const yo = s > 0 ? [y1 - 0.42, y1 - 0.1] : [y0 + 0.1, y0 + 0.42];
        box(body, xF - 0.08, xF + 0.008, yo[0], yo[1], z(belt - 0.3), z(belt - 0.17), white);
        box(body, xR - 0.01, xR + 0.05, s > 0 ? y1 - 0.3 : y0 + 0.04, s > 0 ? y1 - 0.04 : y0 + 0.3, z(belt - 0.28), z(belt - 0.06), red);
        box(body, xF - 0.3, xF - 0.15, s > 0 ? y1 - 0.01 : y0 - 0.02, s > 0 ? y1 + 0.02 : y0 + 0.01, z(belt - 0.2), z(belt - 0.12), amber);
        box(body, wsBase - 0.05, wsBase + 0.12, s > 0 ? y1 - 0.02 : y0 - 0.16, s > 0 ? y1 + 0.16 : y0 + 0.02, z(belt + 0.02), z(belt + 0.15), C);  // mirrors
      }
      if (suv) {
        // the chaser's roof: rack, a light bar, an anemometer mast and a dome camera
        for (const xx of [roofR + 0.25, roofF - 0.25]) box(body, xx - 0.04, xx + 0.04, y0 + 0.2, y1 - 0.2, z(Hd), z(Hd + 0.07), black);
        box(body, roofF - 0.32, roofF - 0.18, y0 + 0.28, y1 - 0.28, z(Hd + 0.07), z(Hd + 0.15), amber);
        box(body, roofR + 0.2, roofR + 0.26, 0.48, 0.54, z(Hd + 0.07), z(Hd + 0.85), dark);
        box(body, roofR + 0.12, roofR + 0.34, 0.4, 0.62, z(Hd + 0.85), z(Hd + 0.9), chrome);
        box(body, roofR + 0.45, roofR + 0.75, -0.15, 0.15, z(Hd + 0.07), z(Hd + 0.27), black, [0.08, 0.08, 0.08]);
      }
      // dark wheel arches under the body's edge
      for (const xa of [a, -b]) for (const s of [-1, 1]) {
        const yo = s > 0 ? [y1 - 0.02, y1 + 0.005] : [y0 - 0.005, y0 + 0.02];
        box(body, xa - R - 0.08, xa + R + 0.08, yo[0], yo[1], z(R * 1.15), z(Math.min(belt - 0.05, R * 2 + 0.12)), black);
      }
    }
    // the wheel: a tyre and a spoked rim (local: axle along y, centre at 0)
    const wheel = [];
    const N = 16, w = kind === 'truck' ? 0.34 : 0.25, rr = R * 0.62;
    for (let i = 0; i < N; i++) {
      const a0 = 2 * Math.PI * i / N, a1 = 2 * Math.PI * (i + 1) / N;
      const p = (r, a_, y) => [r * Math.cos(a_), y, r * Math.sin(a_)];
      // tread
      quad(wheel, p(R, a0, -w / 2), p(R, a1, -w / 2), p(R, a1, w / 2), p(R, a0, w / 2), [0.07, 0.07, 0.07], [0, 0, 0]);
      for (const sd of [-1, 1]) {
        const y = sd * w / 2, ctr = [0, sd * w * 2, 0];
        // sidewall ring, the rim, alternate spokes darker so the spin shows
        quad(wheel, p(R, a0, y), p(R, a1, y), p(rr, a1, y), p(rr, a0, y), [0.09, 0.09, 0.09], [0, -sd * 5, 0]);
        tri(wheel, [0, y, 0], p(rr, a0, y), p(rr, a1, y), i % 4 < 2 ? chrome : [0.18, 0.18, 0.19], [0, -sd * 5, 0]);
      }
    }
    return { body: new Float32Array(body), wheel: new Float32Array(wheel) };
  }

  return { build };
})();
window.CarMesh = CarMesh;
