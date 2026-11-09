'use strict';
// Chase mode's multiplayer (arhtmodelviewer/chase.py): the PC running the viewer hosts; friends on the network or
// over Tailscale open http://<host>:8830/chase and join with the host's code.  Every page sends its car about
// 12 times a second (POST /api/c/sync) and gets everyone else's back, stamped with the host's clock; the other
// cars are drawn a little in the past (DELAY), smoothly between the states received, carried on along their
// velocity for a moment when a packet is late.  Every page draws the storm, the roads and the buildings itself
// from what the host serves.
const Net = (() => {
  const RATE = 80, DELAY = 0.15, KEEP = 3;           // ms between syncs, s behind the host's clock, s of states kept
  const local = ['127.0.0.1', 'localhost', '[::1]', '::1'].includes(location.hostname) && !new URLSearchParams(location.search).has('join');
  const st = { token: null, id: null, offset: null, busy: false, last: 0, others: new Map(), hosting: false, code: null,
               urls: [], err: '', rtt: 0, joined: false };
  window.ChaseAuth = () => (st.token ? { 'X-Chase-Token': st.token } : {});

  async function post(path, body) {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', ...window.ChaseAuth() }, body: JSON.stringify(body) });
    let d = null;
    try { d = await r.json(); } catch (e) { d = null; }
    if (!r.ok) throw new Error((d && d.error) || `${r.status}`);
    return d;
  }

  async function join(name, kind, color, code) {
    const d = await post('/api/c/join', { name, kind, color, code });
    st.token = d.token; st.id = d.id; st.joined = true;
    window.addEventListener('pagehide', () => {
      try { fetch('/api/c/leave', { method: 'POST', keepalive: true, headers: window.ChaseAuth() }); } catch (e) { /* closing */ }
    });
    return d;
  }

  // the host's switch (this PC only)
  async function host(on, newCode) {
    const d = await post('/api/c/host', { on, new_code: !!newCode });
    st.hosting = d.on; st.code = d.code; st.urls = d.urls || [];
    return d;
  }
  async function hostStatus() {
    try {
      const r = await fetch('/api/c/host');
      if (r.ok) { const d = await r.json(); st.hosting = d.on; st.code = d.code; st.urls = d.urls || []; st.roster = d.players; st.tunnel = d.tunnel; }
    } catch (e) { /* not this PC */ }
  }

  const r2 = v => Math.round(v * 100) / 100, r4 = v => Math.round(v * 1e4) / 1e4;
  // our car out, theirs in
  async function tick(car, me) {
    if (!st.joined || st.busy || performance.now() - st.last < RATE) return;
    st.busy = true; st.last = performance.now();
    const s = {
      p: car.p.map(r2), q: car.q.map(r4), v: car.v.map(r2),
      st: car.wheels.slice(0, 2).map(w => r4(w.steer)), sp: car.wheels.map(w => r2(w.spin % 1000)), L: car.wheels.map(w => r4(w.L)),
      g: car.gear, b: car.in.brk > 0.1 ? 1 : 0,
    };
    // height above the sender's own ground: each page's ground can sit at its own level (the detailed map's
    // offset is taken where that page started, and the basic map is another surface), so heights go by this
    if (Number.isFinite(me.hg)) s.hg = r2(me.hg);
    // on foot: [x, y, height above the ground, yaw, stride phase, gait, down, lean, lean direction, in the air, speed]
    if (me.f) s.f = me.f;
    const t0 = performance.now();
    try {
      const d = await post('/api/c/sync', { s, name: me.name, kind: me.kind, color: me.color });
      const t1 = performance.now();
      st.rtt += ((t1 - t0) - st.rtt) * 0.2;
      // the host's clock against ours: the smallest one-way estimate wins (the least delayed reply)
      const est = d.now - (t0 + t1) / 2000;
      st.offset = st.offset === null ? est : Math.abs(est - st.offset) > 1 ? est : st.offset + (est - st.offset) * 0.05;
      const seen = new Set();
      for (const p of d.players) {
        seen.add(p.id);
        let o = st.others.get(p.id);
        if (!o) st.others.set(p.id, o = { buf: [] });
        Object.assign(o, { id: p.id, name: p.name, kind: p.kind, color: p.color });
        if (!o.buf.length || o.buf[o.buf.length - 1].ts < p.ts) o.buf.push({ ts: p.ts, s: p.s });
        while (o.buf.length > 2 && o.buf[1].ts < p.ts - KEEP) o.buf.shift();
      }
      for (const id of [...st.others.keys()]) if (!seen.has(id)) st.others.delete(id);
      st.err = '';
      if (!d.hosting && !local) st.err = 'The host stopped hosting';
    } catch (e) { st.err = e.message; }
    st.busy = false;
  }

  const lerp = (a, b, t) => a + (b - a) * t;
  // every other car now: { id, name, kind, color, p, q, v, steer[2], spin[4], L[4], brake }
  function others() {
    if (st.offset === null) return [];
    const T = performance.now() / 1000 + st.offset - DELAY, out = [];
    for (const o of st.others.values()) {
      const b = o.buf;
      if (!b.length || !b[0].s) continue;
      let i = b.length - 1;
      while (i > 0 && b[i].ts > T) i--;
      const A = b[i], B = b[Math.min(i + 1, b.length - 1)];
      let s;
      if (A === B || T >= B.ts) {                     // past the newest: carried on along its velocity (0.4 s at most)
        const dt = Math.min(0.4, Math.max(0, T - A.ts));
        s = { ...A.s, p: A.s.p.map((v, k) => v + A.s.v[k] * dt) };
      } else {
        const t = Math.max(0, Math.min(1, (T - A.ts) / (B.ts - A.ts)));
        let q2 = B.s.q;
        if (A.s.q.reduce((a, v, k) => a + v * q2[k], 0) < 0) q2 = q2.map(v => -v);
        const q = A.s.q.map((v, k) => lerp(v, q2[k], t)), ql = Math.hypot(...q);
        s = { p: A.s.p.map((v, k) => lerp(v, B.s.p[k], t)), q: q.map(v => v / ql), v: A.s.v.map((v, k) => lerp(v, B.s.v[k], t)),
              st: A.s.st.map((v, k) => lerp(v, B.s.st[k], t)), sp: A.s.sp.map((v, k) => lerp(v, B.s.sp[k] < v - 500 ? B.s.sp[k] + 1000 : B.s.sp[k], t)),
              L: A.s.L.map((v, k) => lerp(v, B.s.L[k], t)), g: B.s.g, b: B.s.b,
              hg: Number.isFinite(A.s.hg) && Number.isFinite(B.s.hg) ? lerp(A.s.hg, B.s.hg, t) : B.s.hg,
              f: A.s.f && B.s.f ? B.s.f.map((v, k) => (k < 3 ? lerp(A.s.f[k], v, t)
                                                     : k === 3 ? A.s.f[3] + Math.atan2(Math.sin(v - A.s.f[3]), Math.cos(v - A.s.f[3])) * t
                                                     : k === 4 && v >= A.s.f[4] ? lerp(A.s.f[4], v, t) : v)) : B.s.f };
      }
      out.push({ id: o.id, name: o.name, kind: o.kind, color: o.color, ...s });
    }
    return out;
  }

  // the internet link (a Cloudflare quick tunnel on the host): friends need only a browser
  async function tunnel(on) { st.tunnel = await post('/api/c/tunnel', { on }); return st.tunnel; }

  return { st, local, join, host, hostStatus, tunnel, tick, others };
})();
window.Net = Net;
