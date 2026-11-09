'use strict';
// Chase mode on foot (F: out of the car and back in): a person who walks (W A S D, relative to the camera), runs
// (Shift) and jumps (Space) on the same ground as the car, can't walk through the car, buildings or trunks, and stands
// in the model's wind: a strong wind leans on them and slows them, past ~40 m/s it knocks them down and slides them
// along the ground, and a tornado's winds can lift them.  Drawn as a jointed figure (boxes; carmesh.js's kit) whose
// arms and legs swing with the stride.  World axes: x east, y north, z up, metres; the figure's: x forward, y left,
// z up from between the feet.
const Walker = (() => {
  const G = 9.81, RHO = 1.2, MASS = 75, CDA = 0.7, RAD = 0.32;
  const WALK = 1.6, RUN = 5.6, JUMP = 4.2;
  const KNOCK = 8.5;                              // m/s^2 of wind push that knocks a person down (~40 m/s of wind)

  // ---------------------------------------------------------------- the figure's parts (built once per colour set)
  const meshes = new Map();
  function parts(jacket) {
    const key = jacket.join(',');
    if (meshes.has(key)) return meshes.get(key);
    const B = CarMesh.box, J = jacket.map(v => v / 255);
    const skin = [0.78, 0.6, 0.48], jeans = [0.16, 0.2, 0.32], boot = [0.12, 0.1, 0.09], hair = [0.17, 0.12, 0.08];
    const cap = J.map(v => v * 0.75);
    const mk = f => { const a = []; f(a); return new Float32Array(a); };
    const m = {
      // pivot at the hips' middle (z 0.95)
      torso: mk(a => { B(a, -0.11, 0.11, -0.2, 0.2, 0.0, 0.55, J, [0.01, 0.01, 0.03]); B(a, -0.12, 0.12, -0.19, 0.19, -0.08, 0.04, jeans); }),
      // pivot at the neck (on top of the torso)
      head: mk(a => {
        B(a, -0.04, 0.04, -0.05, 0.05, 0.0, 0.06, skin);
        B(a, -0.1, 0.11, -0.095, 0.095, 0.06, 0.29, skin, [0.02, 0.02, 0.01]);
        B(a, -0.11, 0.09, -0.1, 0.1, 0.22, 0.31, hair);
        B(a, -0.115, 0.12, -0.105, 0.105, 0.27, 0.33, cap); B(a, 0.08, 0.2, -0.09, 0.09, 0.27, 0.29, cap);   // a ball cap
      }),
      // pivots at the joint above, hanging down -z
      upperArm: mk(a => B(a, -0.055, 0.055, -0.055, 0.055, -0.3, 0.02, J)),
      lowerArm: mk(a => { B(a, -0.045, 0.045, -0.045, 0.045, -0.24, 0.0, J); B(a, -0.04, 0.05, -0.035, 0.035, -0.33, -0.24, skin); }),
      upperLeg: mk(a => B(a, -0.075, 0.075, -0.07, 0.07, -0.46, 0.02, jeans)),
      lowerLeg: mk(a => { B(a, -0.06, 0.06, -0.055, 0.055, -0.42, 0.0, jeans); B(a, -0.07, 0.17, -0.055, 0.055, -0.5, -0.42, boot); }),
    };
    meshes.set(key, m);
    return m;
  }

  // ---------------------------------------------------------------- rotations: matrices whose rows are axes in world
  const mul = (A, B) => [0, 1, 2].map(i => [0, 1, 2].map(j => A[0][j] * B[i][0] + A[1][j] * B[i][1] + A[2][j] * B[i][2]));
  const yawM = a => [[Math.cos(a), Math.sin(a), 0], [-Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
  // about the body's y axis (left): + swings the part's lower end forward (its top back)
  const pitchM = a => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
  // about the body's x axis (forward): + leans its top to the left
  const rollM = a => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
  const apply = (M, v) => [0, 1, 2].map(i => M[0][i] * v[0] + M[1][i] * v[1] + M[2][i] * v[2]);
  const addv = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

  class Person {
    constructor(x, y, z, yaw) {
      this.p = [x, y, z]; this.v = [0, 0, 0]; this.yaw = yaw;
      this.phase = 0; this.air = false; this.down = 0; this.lean = 0; this.gait = 0; this.windA = 0;
    }
    get speed() { return Math.hypot(this.v[0], this.v[1]); }

    // ctl: { dir: [x, y] wanted (world, length 0..1), run, jump }; env: { ground(x, y) -> { h }, push(x, y, rad) ->
    // [dx, dy] | null, wind: [u, v] m/s, car }
    update(dt, ctl, env) {
      const g = env.ground(this.p[0], this.p[1]), gh = g ? g.h : this.p[2];
      const onGround = this.p[2] <= gh + 0.04 && this.v[2] <= 0.01;
      // the wind on a standing person (or one lying flat: a third of it)
      const rel = [env.wind[0] - this.v[0], env.wind[1] - this.v[1]], wr = Math.hypot(...rel);
      const k = 0.5 * RHO * CDA / MASS * (this.down > 0 ? 0.35 : 1);
      const aw = [k * wr * rel[0], k * wr * rel[1]];
      this.windA = Math.hypot(...aw);
      if (onGround && this.down <= 0 && this.windA > KNOCK) { this.down = 2.5; if (say) say('Knocked flat by the wind!'); }
      if (this.down > 0) this.down = Math.max(0, this.down - dt * (this.windA > KNOCK ? 0.2 : 1));
      const control = this.down > 0 ? 0 : 1 / (1 + Math.max(0, this.windA - 2) * 0.25);
      if (onGround) {
        // walking: the velocity follows what is wanted (less of it, leaning into a strong wind); friction holds the
        // feet against the wind until it is past what the ground can hold
        // (into a headwind you get slower: the wind's push along where you go takes off from your pace)
        const along = ctl.dir[0] * aw[0] + ctl.dir[1] * aw[1];
        const sp = (ctl.run ? RUN : WALK) * Math.max(0.25, 1 + Math.min(0, along) * 0.12), want = [ctl.dir[0] * sp, ctl.dir[1] * sp];
        const rate = this.down > 0 ? 0 : 10 * control;
        const f = 1 - Math.exp(-rate * dt);
        this.v[0] += (want[0] - this.v[0]) * f; this.v[1] += (want[1] - this.v[1]) * f;
        const grip = (this.down > 0 ? 0.55 : 0.8) * G;
        const slide = Math.max(0, this.windA - grip);
        if (slide > 0) { this.v[0] += aw[0] / this.windA * slide * dt; this.v[1] += aw[1] / this.windA * slide * dt; }
        else if (this.down > 0) { const fr = Math.exp(-6 * dt); this.v[0] *= fr; this.v[1] *= fr; }
        // a tornado's winds lift what they push hardest
        if (this.windA > 2.2 * G) { this.v[2] = (this.windA - 2.2 * G) * 0.25; this.air = true; this.down = 3; }
        else if (ctl.jump && this.down <= 0) { this.v[2] = JUMP; this.air = true; }
        else { this.v[2] = 0; this.p[2] = gh; this.air = false; }
      } else {
        this.v[0] += aw[0] * dt; this.v[1] += aw[1] * dt;
        this.v[2] -= G * dt;
        this.air = true;
      }
      this.p = addv(this.p, this.v.map(x => x * dt));
      const g2 = env.ground(this.p[0], this.p[1]);
      if (g2 && this.p[2] < g2.h) {
        if (this.v[2] < -9) this.down = Math.max(this.down, 2);
        this.p[2] = g2.h; this.v[2] = 0; this.air = false;
      }
      // out of walls and trunks, and out of the car
      const pu = env.push ? env.push(this.p[0], this.p[1], RAD) : null;
      if (pu) this.bump(pu);
      if (env.car) this.carBump(env.car);
      // facing where they go (or, standing, where they were going)
      const s = this.speed;
      if (s > 0.3 && this.down <= 0) {
        const head = Math.atan2(this.v[1], this.v[0]);
        this.yaw += Math.atan2(Math.sin(head - this.yaw), Math.cos(head - this.yaw)) * (1 - Math.exp(-dt * 10));
      }
      this.gait += ((onGround && this.down <= 0 ? Math.min(1, s / WALK) : 0) - this.gait) * (1 - Math.exp(-dt * 8));
      this.phase += s * dt / (s > 3 ? 2.2 : 1.35) * Math.PI * 2;
      // leaning into the wind
      const lw = this.down > 0 ? 0 : Math.min(0.35, this.windA * 0.04);
      this.lean += (lw - this.lean) * (1 - Math.exp(-dt * 3));
      this.leanDir = Math.atan2(-aw[1], -aw[0]);
    }
    bump(d) {
      const l = Math.hypot(d[0], d[1]);
      if (l < 1e-6) return;
      this.p[0] += d[0]; this.p[1] += d[1];
      const nx = d[0] / l, ny = d[1] / l, vn = this.v[0] * nx + this.v[1] * ny;
      if (vn < 0) { this.v[0] -= nx * vn; this.v[1] -= ny * vn; }
    }
    // the car as a box (its length and width, up to its roof): pushed out along the nearest side
    carBump(c) {
      const S = c.spec, M = c.matrix, d = [this.p[0] - c.p[0], this.p[1] - c.p[1], this.p[2] - c.p[2]];
      const bx = d[0] * M[0][0] + d[1] * M[0][1] + d[2] * M[0][2], by = d[0] * M[1][0] + d[1] * M[1][1] + d[2] * M[1][2];
      const bz = d[0] * M[2][0] + d[1] * M[2][1] + d[2] * M[2][2];
      const a = S.cgFront, b = S.wheelbase - a, x1 = a + (S.dims[0] - S.wheelbase) / 2 + RAD, x0 = -b - (S.dims[0] - S.wheelbase) / 2 - RAD;
      const hw = S.dims[1] / 2 + RAD, top = S.dims[2] - S.cgH;
      if (bx < x0 || bx > x1 || by < -hw || by > hw || bz > top) return;
      const pen = [x1 - bx, bx - x0, hw - by, by + hw];
      const i = pen.indexOf(Math.min(...pen)), dir = [[1, 0], [-1, 0], [0, 1], [0, -1]][i];
      const push = [M[0][0] * dir[0] * pen[i] + M[1][0] * dir[1] * pen[i], M[0][1] * dir[0] * pen[i] + M[1][1] * dir[1] * pen[i]];
      this.bump(push);
    }

    // the figure's parts for render.js: [{ data, M (rows: axes in world), p (pivot, world m) }]
    figure(jacket) {
      const m = parts(jacket), ph = this.phase, sw = this.gait * (this.speed > 3 ? 0.9 : 0.5);
      const swing = Math.sin(ph) * sw, knee = s => Math.max(0, Math.sin(ph + s)) * sw * 1.3;
      let base = yawM(this.yaw);
      // lying down (knocked flat): the whole figure tipped onto its back along the wind
      if (this.down > 0 && !this.air) base = mul(base, pitchM(-Math.PI / 2 * Math.min(1, this.down)));
      else if (this.lean > 0.01) {
        // leaning into the wind: tilted toward where it comes from
        const rel = this.leanDir - this.yaw;
        base = mul(base, mul(pitchM(-this.lean * Math.cos(rel)), rollM(this.lean * Math.sin(rel))));
      }
      const flat = this.down > 0 && !this.air;
      const hips = flat ? addv(this.p, [0, 0, 0.13]) : addv(this.p, apply(base, [0, 0, 0.95]));
      const torsoM = mul(base, pitchM(-0.05 * this.gait));
      const out = [{ data: m.torso, M: torsoM, p: hips }];
      const neck = addv(hips, apply(torsoM, [0, 0, 0.55]));
      out.push({ data: m.head, M: torsoM, p: neck });
      for (const s of [1, -1]) {
        // arms: from the shoulders, swinging against the legs
        const sh = addv(hips, apply(torsoM, [0, s * 0.255, 0.5]));
        const ua = mul(torsoM, mul(rollM(-s * 0.08), pitchM(-swing * s * 1.1)));
        out.push({ data: m.upperArm, M: ua, p: sh });
        const el = addv(sh, apply(ua, [0, 0, -0.3]));
        out.push({ data: m.lowerArm, M: mul(ua, pitchM(0.25 + 0.6 * this.gait * (this.speed > 3 ? 1 : 0.4))), p: el });
        // legs: from the hips
        const hp = addv(hips, apply(base, [0, s * 0.1, 0]));
        const ul = mul(base, pitchM(swing * s));
        out.push({ data: m.upperLeg, M: ul, p: hp });
        const kn = addv(hp, apply(ul, [0, 0, -0.46]));
        out.push({ data: m.lowerLeg, M: mul(ul, pitchM(-knee(s > 0 ? 0 : Math.PI))), p: kn });
      }
      return out;
    }
  }
  let say = null;
  return { Person, RAD, setSay: f => { say = f; } };
})();
window.Walker = Walker;
