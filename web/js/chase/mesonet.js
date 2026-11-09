'use strict';
// Chase mode's mesonet probe: the car carries one, as storm chasers' mobile mesonets do -- the model's air where you
// are (arhtmodelviewer/chase.py air(): its finest grid's lowest level, the wind taken down to 10 m): temperature,
// dew point, wind and gust, direction, pressure at the ground, and the rain at the car.  Logged once a second
// (model time from the air's replies, the car's place), up to two hours; J shows the graphs of the last minutes,
// Shift+J saves the whole log as CSV.
const Mesonet = (() => {
  const KEEP = 7200, WINDOW_S = 600;
  const st = { log: [], on: false, lastT: null, lastWall: 0 };

  // one sample: a = the air reply ({ t, td, u, v, gust, p }), mt = model time s, x, y world m, rain 0..1
  function record(a, mt, x, y, rain) {
    if (!a || !Number.isFinite(a.t)) return;
    const now = performance.now();
    if (now - st.lastWall < 1000) return;
    st.lastWall = now;
    const W = Math.hypot(a.u, a.v), dir = (Math.atan2(-a.u, -a.v) * 180 / Math.PI + 360) % 360;
    st.log.push({ wall: Date.now(), mt, x, y, t: a.t, td: a.td, W, gust: Math.max(a.gust || W, W), dir, p: Number.isFinite(a.p) ? a.p : null, rain });
    if (st.log.length > KEEP) st.log.shift();
  }

  // the panel: units 'mph' (deg F, mph) or 'km/h' (deg C, km/h)
  function draw(g, x0, y0, w, h, units) {
    if (!st.on) return;
    const L = st.log, mph = units === 'mph';
    g.save();
    g.fillStyle = 'rgba(8,12,18,.78)'; roundRect(g, x0, y0, w, h, 10); g.fill();
    g.strokeStyle = 'rgba(255,255,255,.12)'; g.lineWidth = 1; g.stroke();
    g.font = '600 13px Segoe UI'; g.fillStyle = '#e6edf3'; g.textAlign = 'left';
    g.fillText('Mesonet probe', x0 + 12, y0 + 20);
    g.font = '11px Segoe UI'; g.fillStyle = '#8fa0b0'; g.textAlign = 'right';
    g.fillText(`${L.length ? fmtDur(L.length) + ' logged' : 'waiting for the model…'} · J hide · Shift+J save CSV`, x0 + w - 12, y0 + 20);
    if (L.length < 2) { g.restore(); return; }
    const end = L[L.length - 1].wall, t0 = end - WINDOW_S * 1000;
    const S = L.filter(s => s.wall >= t0);
    const strips = 4, top = y0 + 30, sh = (h - 40) / strips, X = s => x0 + 46 + (w - 58) * (s.wall - t0) / (WINDOW_S * 1000);
    const T = v => (mph ? v * 9 / 5 + 32 : v), Wv = v => v * (mph ? 2.23694 : 3.6);
    const strip = (i, series, fmtNow, label) => {
      const ya = top + i * sh, yb = ya + sh - 6;
      g.strokeStyle = 'rgba(255,255,255,.08)'; g.beginPath(); g.moveTo(x0 + 46, yb + 3); g.lineTo(x0 + w - 12, yb + 3); g.stroke();
      let lo = Infinity, hi = -Infinity;
      for (const sr of series) for (const s of S) { const v = sr.f(s); if (v !== null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
      if (!Number.isFinite(lo)) return;
      const pad = Math.max((hi - lo) * 0.12, sr0(series[0].minSpan) / 2);
      lo -= pad; hi += pad;
      const Y = v => yb - (v - lo) / (hi - lo) * (yb - ya - 4);
      for (const sr of series) {
        g.strokeStyle = sr.c; g.fillStyle = sr.c; g.lineWidth = 1.6;
        if (sr.dots) { for (const s of S) { const v = sr.f(s); if (v !== null) g.fillRect(X(s) - 1, Y(v) - 1, 2, 2); } continue; }
        g.beginPath(); let on = false;
        for (const s of S) { const v = sr.f(s); if (v === null) { on = false; continue; } if (on) g.lineTo(X(s), Y(v)); else { g.moveTo(X(s), Y(v)); on = true; } }
        g.stroke();
      }
      g.font = '10px Segoe UI'; g.fillStyle = '#7f8e9c'; g.textAlign = 'right';
      g.fillText(hi.toFixed(hi - lo < 5 ? 1 : 0), x0 + 42, ya + 9); g.fillText(lo.toFixed(hi - lo < 5 ? 1 : 0), x0 + 42, yb + 2);
      g.textAlign = 'left'; g.font = '600 11px Segoe UI';
      const tw = g.measureText(label + '  ' + fmtNow).width;
      g.fillStyle = 'rgba(8,12,18,.75)'; g.fillRect(x0 + 47, ya, tw + 7, 13);
      g.fillStyle = '#dfe7ee'; g.fillText(label + '  ' + fmtNow, x0 + 50, ya + 10);
    };
    const sr0 = v => v || 1;
    const last = L[L.length - 1], ago = s => S.find(q => q.wall >= last.wall - 60000) || s;
    const dp = last.p !== null && ago(last).p !== null ? last.p - ago(last).p : null;
    strip(0, [{ f: s => T(s.t), c: '#ff6b5b', minSpan: 2 }, { f: s => (s.td === null ? null : T(s.td)), c: '#5bd68a' }],
          `${T(last.t).toFixed(1)}° / ${last.td === null ? '–' : T(last.td).toFixed(1) + '°'}`, `T / Td (${mph ? '°F' : '°C'})`);
    strip(1, [{ f: s => Wv(s.gust), c: 'rgba(120,190,255,.55)', minSpan: 5 }, { f: s => Wv(s.W), c: '#6fb8ff' }],
          `${Math.round(Wv(last.W))} G ${Math.round(Wv(last.gust))}`, `Wind (${mph ? 'mph' : 'km/h'})`);
    strip(2, [{ f: s => s.dir, c: '#ffd166', dots: true, minSpan: 60 }], `${Math.round(last.dir)}° (from)`, 'Direction');
    strip(3, [{ f: s => s.p, c: '#c792ff', minSpan: 1 }],
          last.p === null ? '–' : `${last.p.toFixed(1)} hPa${dp !== null ? ` (${dp >= 0 ? '+' : ''}${dp.toFixed(1)} in 1 min)` : ''}`, 'Pressure');
    g.font = '10px Segoe UI'; g.fillStyle = '#7f8e9c'; g.textAlign = 'left';
    g.fillText(`-${WINDOW_S / 60} min`, x0 + 46, y0 + h - 4); g.textAlign = 'right'; g.fillText('now', x0 + w - 12, y0 + h - 4);
    g.restore();
  }
  const fmtDur = n => (n < 60 ? `${n} s` : n < 3600 ? `${Math.floor(n / 60)} min` : `${Math.floor(n / 3600)} h ${Math.floor(n / 60) % 60} min`);
  function roundRect(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }

  // the log as CSV (world km for the place; lat/lon with the run's projection where there is one)
  function csv() {
    const head = 'wall_time_utc,model_time_s,x_km,y_km,temp_c,dewpoint_c,wind_ms,gust_ms,dir_deg,pressure_hpa,rain_0_1';
    const rows = st.log.map(s => [new Date(s.wall).toISOString(), s.mt === null ? '' : s.mt.toFixed(0), (s.x / 1000).toFixed(3), (s.y / 1000).toFixed(3),
      s.t.toFixed(2), s.td === null ? '' : s.td.toFixed(2), s.W.toFixed(2), s.gust.toFixed(2), s.dir.toFixed(0), s.p === null ? '' : s.p.toFixed(2), s.rain.toFixed(2)].join(','));
    return [head, ...rows].join('\n');
  }
  function save() {
    if (!st.log.length) return false;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv()], { type: 'text/csv' }));
    a.download = `mesonet-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`;
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    return true;
  }
  return { record, draw, csv, save, state: st, toggle: () => (st.on = !st.on) };
})();
window.Mesonet = Mesonet;
