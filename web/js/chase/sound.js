'use strict';
// Chase mode's sound, made in the browser (Web Audio, no recordings):
//  - the engine: its firing pulses at the revs (a saw, a square an octave down, through a soft clipper and a
//    low-pass that opens with the throttle), intake noise, the tyres squealing past their grip, the road's rumble;
//  - the wind: the model's ground-relative wind at the car (chase.py air(), gusting about its mean towards the
//    gust): a broadband rush that grows and brightens with it, a howl (two resonances) from a gale up, and the
//    deep roar of a tornado's winds -- or of a tornado near the car -- from about 35 m/s;
//  - the rain: a hiss and the drops' patter at the model's rain rate, on the roof and muffled from inside the car;
//    snow falls nearly silent (it only damps the rain's share).
// update(s) once a frame with the state; start() from a click (browsers start audio only on a gesture).
const Sound = (() => {
  let ac = null, out = null, cabin = null, n = {}, drop = null, nextDrop = 0, gustT = 0;
  const lvl = { wind: 0, rain: 0 };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

  // looped noise: white, pink (Voss-McCartney-ish, Paul Kellet's filter) or brown
  function noise(kind, secs = 4) {
    const len = Math.round(ac.sampleRate * secs), b = ac.createBuffer(1, len, ac.sampleRate), d = b.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, last = 0;
    for (let i = 0; i < len; i++) {
      const w = Math.random() * 2 - 1;
      if (kind === 'white') d[i] = w * 0.5;
      else if (kind === 'pink') {
        b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.96900 * b2 + w * 0.1538520;
        b3 = 0.86650 * b3 + w * 0.3104856; b4 = 0.55000 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.0168980;
        d[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11; b6 = w * 0.115926;
      } else { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; }
    }
    // no click at the loop's seam
    const f = Math.round(ac.sampleRate * 0.05);
    for (let i = 0; i < f; i++) { const a = i / f; d[i] = d[i] * a + d[len - f + i] * (1 - a); }
    return b;
  }
  function src(buf) { const s = ac.createBufferSource(); s.buffer = buf; s.loop = true; s.start(ac.currentTime + Math.random() * 0.1, Math.random() * buf.duration); return s; }
  const gain = v => { const g = ac.createGain(); g.gain.value = v; return g; };
  function filt(type, f, Q = 0.7) { const x = ac.createBiquadFilter(); x.type = type; x.frequency.value = f; x.Q.value = Q; return x; }
  const chain = (...a) => { for (let i = 0; i < a.length - 1; i++) a[i].connect(a[i + 1]); return a[a.length - 1]; };
  function shaper(k) {
    const s = ac.createWaveShaper(), c = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) { const x = i / 511.5 - 1; c[i] = Math.tanh(k * x) / Math.tanh(k); }
    s.curve = c; return s;
  }

  function start() {
    if (ac) { if (ac.state === 'suspended') ac.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ac = new AC();
    out = gain(0.7); out.connect(ac.destination);
    const comp = ac.createDynamicsCompressor();
    comp.threshold.value = -14; comp.ratio.value = 4; comp.connect(out);
    // what comes from outside the car goes through the cabin (muffled from the hood and driver's seat)
    cabin = filt('lowpass', 20000, 0.5); cabin.connect(comp);
    const W = noise('white'), P = noise('pink'), B = noise('brown');
    // ---- the engine
    n.saw = ac.createOscillator(); n.saw.type = 'sawtooth';
    n.sq = ac.createOscillator(); n.sq.type = 'square';
    n.sub = ac.createOscillator(); n.sub.type = 'sine';
    n.mixE = gain(1);
    chain(n.saw, gain(0.55), n.mixE); chain(n.sq, gain(0.35), n.mixE); chain(n.sub, gain(0.5), n.mixE);
    n.drive = shaper(2.5);
    n.eLP = filt('lowpass', 600, 1.2);
    n.eG = gain(0);
    chain(n.mixE, n.drive, n.eLP, n.eG, comp);
    n.intake = filt('bandpass', 400, 1.5); n.inG = gain(0);
    chain(src(W), n.intake, n.inG, comp);
    for (const o of [n.saw, n.sq, n.sub]) o.start();
    // ---- the tyres and the road
    n.sq1 = filt('bandpass', 1150, 7); n.sq2 = filt('bandpass', 2350, 9); n.sqG = gain(0);
    const tw = src(W); tw.connect(n.sq1); tw.connect(n.sq2); n.sq1.connect(n.sqG); n.sq2.connect(n.sqG); n.sqG.connect(comp);
    n.roadLP = filt('lowpass', 220, 0.8); n.roadG = gain(0);
    chain(src(B), n.roadLP, n.roadG, comp);
    // ---- the wind
    n.rushLP = filt('lowpass', 400, 0.6); n.rushHP = filt('highpass', 60, 0.5); n.rushG = gain(0);
    chain(src(P), n.rushHP, n.rushLP, n.rushG, cabin);
    n.howl1 = filt('bandpass', 420, 10); n.howl2 = filt('bandpass', 760, 12); n.howlG = gain(0);
    const hp = src(P); hp.connect(n.howl1); hp.connect(n.howl2); n.howl1.connect(n.howlG); n.howl2.connect(n.howlG); n.howlG.connect(cabin);
    n.roarLP = filt('lowpass', 110, 0.9); n.roarG = gain(0);
    chain(src(B), n.roarLP, n.roarG, comp);              // the roar comes through the car's body: not muffled
    // ---- the rain
    n.hissHP = filt('highpass', 1400, 0.5); n.hissLP = filt('lowpass', 9000, 0.5); n.hissG = gain(0);
    chain(src(W), n.hissHP, n.hissLP, n.hissG, cabin);
    n.patG = gain(1); n.patG.connect(comp);
    // one drop: a tick of noise that dies away in a few ms
    const len = Math.round(ac.sampleRate * 0.03);
    drop = ac.createBuffer(1, len, ac.sampleRate);
    const d = drop.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ac.sampleRate * 0.0035));
  }

  // a smooth wander in -1..1 (for the gusts)
  const wander = t => 0.55 * Math.sin(t * 0.37) + 0.3 * Math.sin(t * 0.93 + 1.3) + 0.15 * Math.sin(t * 2.1 + 0.4);

  // s: { rpm, idle, redline, thr, cyl, speed m/s, slip (the most-slipping tyre, 1 = its peak), rough, wind m/s,
  //      gust m/s, tornado (0..1: a tornado's roar from nearby), rain 0..1, snow 0..1, inside (hood / driver view),
  //      vol 0..1, dt s }
  function update(s) {
    if (!ac || ac.state !== 'running') return;
    const t = ac.currentTime, k = 0.06;            // the time constant every level glides with
    const set = (p, v, tc = k) => p.setTargetAtTime(v, t, tc);
    set(out.gain, clamp(s.vol, 0, 1) * 0.9, 0.05);
    set(cabin.frequency, s.inside ? 1300 : 18000, 0.15);
    // ---- the engine
    const rf = clamp((s.rpm - s.idle) / (s.redline - s.idle), 0, 1);
    const fire = s.rpm / 60 * (s.cyl / 2);
    set(n.saw.frequency, fire, 0.02); set(n.sq.frequency, fire / 2, 0.02); set(n.sub.frequency, fire / 2, 0.02);
    set(n.eLP.frequency, 180 + fire * 2.5 + 2600 * s.thr * (0.4 + rf), 0.03);
    set(n.eG.gain, (0.05 + 0.13 * s.thr + 0.06 * rf) * (s.inside ? 1.1 : 0.85), 0.03);
    set(n.intake.frequency, 300 + fire * 3, 0.03); set(n.inG.gain, 0.05 * s.thr * (0.3 + rf), 0.04);
    set(n.sqG.gain, 0.22 * smooth(1.0, 1.6, s.slip) * smooth(1, 6, s.speed), 0.04);
    set(n.roadG.gain, (0.05 + 0.25 * s.rough) * smooth(0.5, 35, s.speed), 0.1);
    set(n.roadLP.frequency, 140 + 6 * s.speed, 0.1);
    // ---- the wind: gusting between the mean and the gust
    gustT += s.dt;
    const g = clamp(wander(gustT * (0.6 + s.wind / 40)), -1, 1) * 0.5 + 0.5;
    const W = s.wind + (Math.max(s.gust, s.wind) - s.wind) * g;
    lvl.wind = W;
    set(n.rushG.gain, 0.5 * Math.pow(smooth(0, 45, W), 1.3), 0.12);
    set(n.rushLP.frequency, 260 + 55 * W, 0.12);
    set(n.howlG.gain, 0.32 * smooth(12, 38, W), 0.15);
    set(n.howl1.frequency, 300 + 9 * W + 40 * g, 0.2); set(n.howl2.frequency, 560 + 14 * W + 70 * g, 0.2);
    const roar = Math.max(smooth(32, 80, W), clamp(s.tornado, 0, 1));
    set(n.roarG.gain, 0.9 * roar, 0.25); set(n.roarLP.frequency, 80 + 70 * roar, 0.25);
    // ---- the rain (snow damps it)
    const R = clamp(s.rain, 0, 1) * (1 - 0.85 * clamp(s.snow, 0, 1));
    lvl.rain = R;
    set(n.hissG.gain, 0.32 * R, 0.3);
    // the patter: Poisson drops, a few hundred a second at the heaviest (on the roof from inside: lower, louder)
    const rate = (s.inside ? 260 : 140) * R * R + 6 * R;
    if (nextDrop < t) nextDrop = t;
    let budget = 60;
    while (rate > 0.5 && nextDrop < t + 0.12 && budget-- > 0) {
      nextDrop += -Math.log(1 - Math.random()) / rate;
      const b = ac.createBufferSource(); b.buffer = drop;
      b.playbackRate.value = s.inside ? 0.35 + 0.3 * Math.random() : 0.7 + 0.8 * Math.random();
      const gd = gain((s.inside ? 0.3 : 0.12) * (0.3 + Math.random()));
      chain(b, gd, n.patG); b.start(nextDrop);
    }
  }

  // thunder from a flash d m away, heard d / 343 s later: close ones a crack first, then the rumble rolling for
  // several seconds, darker and weaker with distance (nothing past ~25 km); in the cloud quieter than to the ground
  let thBrown = null, thWhite = null;
  function thunder(d, cg) {
    if (!ac || ac.state !== 'running' || d > 25000) return;
    if (!thBrown) { thBrown = noise('brown', 12); thWhite = noise('white', 2); }
    const t0 = ac.currentTime + d / 343, len = 5 + Math.min(7, d / 2200) + Math.random() * 2;
    const A = Math.min(1.1, 1.5 / (1 + d / 1600)) * (cg ? 1 : 0.6);
    const r = ac.createBufferSource(); r.buffer = thBrown;
    const lp = filt('lowpass', Math.max(110, 2600 / (1 + d / 1500)), 0.5), g = gain(0);
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(A * 0.8, t0 + 0.06 + d / 9000);
    let t = t0 + 0.2 + d / 9000;
    for (let k = 0; k < 5 && t < t0 + len - 0.5; k++) {
      t += 0.3 + Math.random() * 1.3;
      g.gain.linearRampToValueAtTime(A * (0.2 + 0.7 * Math.random()) * Math.exp(-(t - t0) / (len * 0.5)), Math.min(t, t0 + len - 0.4));
    }
    g.gain.linearRampToValueAtTime(0, t0 + len);
    chain(r, lp, g, cabin);
    r.start(t0, Math.random() * 5); r.stop(t0 + len + 0.1);
    if (cg && d < 3000) {                                 // the crack of a close strike
      const c = ac.createBufferSource(); c.buffer = thWhite;
      const hp = filt('highpass', 600, 0.5), gc = gain(0), Ac = 0.9 / (1 + d / 500);
      gc.gain.setValueAtTime(0, t0); gc.gain.linearRampToValueAtTime(Ac, t0 + 0.004);
      gc.gain.exponentialRampToValueAtTime(0.001, t0 + 0.35);
      chain(c, hp, gc, cabin);
      c.start(t0, Math.random()); c.stop(t0 + 0.4);
    }
  }

  function toggle() { if (!ac) return true; if (ac.state === 'running') { ac.suspend(); return false; } ac.resume(); return true; }
  return { start, update, toggle, thunder, levels: lvl, ctx: () => ac, nodes: n };
})();
window.Sound = Sound;
