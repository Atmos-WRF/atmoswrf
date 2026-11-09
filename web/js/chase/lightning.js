'use strict';
// Chase mode's lightning, from the model: the flash rate in each storm column (arhtmodelviewer/chase.py lightning():
// McCaul et al. 2009, the graupel flux at -15 C and the column's ice) polled every few seconds, flashes drawn from
// it as a Poisson process (each column at its own rate), about one in four to the ground (the usual IC:CG of ~3).
// A flash: 1-4 strokes over a few tenths of a second, lighting the cloud round where it happens (at the -15 C level,
// render.js flashAt) and everything near (flashC); a cloud-to-ground one also a channel drawn from the cloud base to
// the ground, flickering with its strokes.  Thunder after the sound's travel time (sound.js thunder), strikes on
// the minimap for a minute and a half.  World metres; times performance.now() ms.
const Lightning = (() => {
  const BOX_KM = 160, POLL_MS = 4000, CG_SHARE = 0.25, KEEP_MS = 90000, NEAR_KM = 80;
  const st = { cells: [], cum: null, total: 0, busy: false, last: 0, err: '', flashes: [], marks: [], t: null, next: 0 };

  async function poll(x, y) {
    const now = performance.now();
    if (st.busy || now - st.last < POLL_MS) return;
    st.busy = true; st.last = now;
    try {
      const h = BOX_KM / 2, b = [x / 1000 - h, y / 1000 - h, x / 1000 + h, y / 1000 + h].map(v => v.toFixed(2));
      const r = await fetch(`/api/c/lightning?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}`, { headers: window.ChaseAuth ? window.ChaseAuth() : {} });
      if (r.status === 404) { const t = await r.text(); st.err = /no run/.test(t) ? '' : 'lightning needs the viewer restarted (new in this version)'; st.cells = []; st.total = 0; }
      else if (!r.ok) throw new Error(await r.text());
      else {
        const d = await r.json();
        st.cells = d.cells; st.t = d.t; st.err = '';
        const cum = new Float64Array(d.cells.length);
        let s = 0;
        d.cells.forEach((c, i) => { s += c[2]; cum[i] = s; });
        st.cum = cum; st.total = s;
      }
    } catch (e) { st.err = e.message; }
    st.busy = false;
  }

  // a cloud-to-ground channel: from the cloud base down, a jagged walk with a branch or two (world m)
  function channel(x, y, zTop, zGround) {
    const pts = [[x, y, zTop]], br = [];
    let px = x, py = y, pz = zTop;
    while (pz > zGround) {
      const dz = 40 + Math.random() * 90;
      px += (Math.random() - 0.5) * dz * 0.9; py += (Math.random() - 0.5) * dz * 0.9; pz = Math.max(zGround, pz - dz);
      pts.push([px, py, pz]);
      if (Math.random() < 0.12 && pz > zGround + 150) {
        const b = [[px, py, pz]];
        let bx = px, by = py, bz = pz;
        const ang = Math.random() * 6.283, n = 3 + Math.floor(Math.random() * 5);
        for (let k = 0; k < n; k++) { bz -= 40 + Math.random() * 60; bx += Math.cos(ang) * 50 + (Math.random() - 0.5) * 40; by += Math.sin(ang) * 50 + (Math.random() - 0.5) * 40; b.push([bx, by, bz]); }
        br.push(b);
      }
    }
    return { pts, br };
  }

  // the flashes now: new ones drawn from the rates (rate.total per s), old ones let go; g(x, y) the ground height,
  // base(x, y) the cloud base over the ground (m) or null
  function step(now, car, g, base, onFlash) {
    if (!st.next) st.next = now;
    const R = st.total;
    let budget = 8;
    while (R > 0 && st.next <= now && budget-- > 0) {
      st.next += -Math.log(1 - Math.random()) / R * 1000;
      const u = Math.random() * R, cum = st.cum;
      let lo = 0, hi = cum.length - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < u) lo = m + 1; else hi = m; }
      const c = st.cells[lo];
      if (!c) continue;
      const dx = 900 * (Math.random() - 0.5), dy = 900 * (Math.random() - 0.5);
      const x = c[0] * 1000 + dx, y = c[1] * 1000 + dy;
      const d = Math.hypot(x - car[0], y - car[1]);
      const cg = Math.random() < CG_SHARE, gz = g(x, y);
      const f = { t0: st.next, x, y, z: gz + c[3], d, cg, top: c[4], strokes: [], I: 0 };
      const n = cg ? 1 + Math.floor(Math.random() * 4) : 1 + Math.floor(Math.random() * 2.5);
      let t = 0;
      for (let k = 0; k < n; k++) { f.strokes.push(t); t += 40 + Math.random() * 110; }
      f.dur = t + 120;
      f.peak = cg ? 1 : 0.55 + 0.4 * Math.random();
      if (cg) {
        const cb = base(x, y);
        f.ch = channel(x, y, gz + (cb !== null && cb > 200 ? cb : 1500) + 300, gz);
        f.gz = gz;
      }
      if (d < NEAR_KM * 1000) st.flashes.push(f);
      st.marks.push({ t: st.next, x, y, cg });
      if (onFlash && d < 30000) onFlash(f);
    }
    if (st.next < now - 2000) st.next = now;          // (no backlog after a pause)
    st.flashes = st.flashes.filter(f => now - f.t0 < f.dur);
    while (st.marks.length && now - st.marks[0].t > KEEP_MS) st.marks.shift();
  }
  // how bright flash f is now (0..1): each stroke a sharp peak decaying in ~60 ms, a glow between them
  function bright(f, now) {
    const t = now - f.t0;
    if (t < 0 || t > f.dur) return 0;
    let b = 0;
    for (const s of f.strokes) if (t >= s) b = Math.max(b, Math.exp(-(t - s) / 55));
    return f.peak * Math.max(b, 0.12 * (t < f.dur - 120 ? 1 : 0));
  }
  // render.js's flash: the brightest flash now, where it is and how bright, and the light round the eye
  function light(now, eye) {
    let best = null, bb = 0, amb = 0;
    for (const f of st.flashes) {
      const b = bright(f, now);
      if (b <= 0) continue;
      const d = Math.hypot(f.x - eye[0], f.y - eye[1], (f.z - eye[2]) * 0.5) / 1000;
      amb += b * 0.22 / (1 + (d / 5) * (d / 5));
      const v = b / (1 + d / 20);
      if (v > bb) { bb = v; best = f; }
    }
    const a = Math.min(amb, 2.5);
    return { p: best ? [best.x, best.y, best.z] : [0, 0, 0], I: best ? bright(best, now) * 30 : 0, amb: [a * 0.78, a * 0.83, a * 1.0] };
  }
  // the channels of the cloud-to-ground flashes for render.js (drawBolts): through the rain they fade with
  // distance (rainI 0..1 at the eye)
  function bolts(now, eye, rainI) {
    const out = [];
    for (const f of st.flashes) {
      if (!f.ch) continue;
      const b = bright(f, now);
      if (b < 0.05) continue;
      const d = Math.hypot(f.x - eye[0], f.y - eye[1]), I = b / (1 + rainI * d / 3000);
      const w = Math.max(0.8, d * 0.0011);
      out.push({ pts: f.ch.pts, I, w });
      for (const br of f.ch.br) out.push({ pts: br, I: I * 0.6, w: w * 0.7 });
    }
    return out;
  }
  // (the channels on a 2-D canvas: project(world) -> [px, py] or null)
  function draw(g, now, project) {
    for (const f of st.flashes) {
      if (!f.ch) continue;
      const b = bright(f, now);
      if (b < 0.08) continue;
      const line = pts => {
        g.beginPath();
        let on = false;
        for (const p of pts) {
          const s = project(p);
          if (!s) { on = false; continue; }
          if (on) g.lineTo(s[0], s[1]); else { g.moveTo(s[0], s[1]); on = true; }
        }
        g.stroke();
      };
      const near = Math.max(0.6, Math.min(3.2, 8000 / Math.max(f.d, 500)));
      g.save();
      g.globalCompositeOperation = 'lighter';
      g.lineJoin = 'round'; g.lineCap = 'round';
      g.shadowColor = 'rgba(170,190,255,0.9)'; g.shadowBlur = 18 * Math.min(1.5, near);
      g.strokeStyle = `rgba(200,215,255,${0.55 * b})`; g.lineWidth = 4 * near; line(f.ch.pts);
      g.shadowBlur = 0;
      g.strokeStyle = `rgba(255,255,255,${Math.min(1, 1.2 * b)})`; g.lineWidth = 1.4 * near; line(f.ch.pts);
      g.strokeStyle = `rgba(225,232,255,${0.6 * b})`; g.lineWidth = 0.8 * near;
      for (const br of f.ch.br) line(br);
      g.restore();
    }
  }
  // the strikes on a map (P(x, y) -> [px, py])
  function map(g, P, now) {
    for (const m of st.marks) {
      const age = (now - m.t) / KEEP_MS, s = P(m.x, m.y);
      g.globalAlpha = Math.max(0, 1 - age);
      if (m.cg) {
        g.strokeStyle = age < 0.1 ? '#ffffff' : '#ffd84a'; g.lineWidth = 2;
        g.beginPath(); g.moveTo(s[0] - 4, s[1]); g.lineTo(s[0] + 4, s[1]); g.moveTo(s[0], s[1] - 4); g.lineTo(s[0], s[1] + 4); g.stroke();
      } else { g.fillStyle = '#9ad0ff'; g.fillRect(s[0] - 1, s[1] - 1, 2, 2); }
    }
    g.globalAlpha = 1;
  }
  return { poll, step, light, bolts, draw, map, bright, state: st };
})();
window.Lightning = Lightning;
