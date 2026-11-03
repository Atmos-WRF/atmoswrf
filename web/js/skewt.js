'use strict';
// Small skew-T (1050-100 hPa) and hodograph (0-12 km) previews of the sounding.

function drawSkewT(canvas, snd) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 180, H = canvas.clientHeight || 180;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = '#0a0d12'; g.fillRect(0, 0, W, H);
  if (!snd) return;
  const pTop = 100, pBot = 1050, skew = 0.9;
  const yOf = p => H * Math.log(p / pTop) / Math.log(pBot / pTop);
  const xOf = (T, p) => W * (0.5 + (T + 10) / 110) + skew * (H - yOf(p));
  g.lineWidth = 0.6;
  g.strokeStyle = '#26303d';
  for (const p of [1000, 850, 700, 500, 300, 200, 100]) { g.beginPath(); g.moveTo(0, yOf(p)); g.lineTo(W, yOf(p)); g.stroke(); }
  g.strokeStyle = '#2b3542';
  for (let T = -100; T <= 50; T += 10) {
    g.beginPath(); g.moveTo(xOf(T, pBot), yOf(pBot)); g.lineTo(xOf(T, pTop), yOf(pTop)); g.stroke();
  }
  g.strokeStyle = '#4a5a70'; g.setLineDash([2, 2]);
  g.beginPath(); g.moveTo(xOf(0, pBot), yOf(pBot)); g.lineTo(xOf(0, pTop), yOf(pTop)); g.stroke();
  g.setLineDash([]);
  const line = (vals, color) => {
    g.strokeStyle = color; g.lineWidth = 1.6; g.beginPath();
    let first = true;
    for (let i = 0; i < snd.p.length; i++) {
      if (snd.p[i] < pTop) break;
      const x = xOf(vals[i], snd.p[i]), y = yOf(snd.p[i]);
      if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y);
    }
    g.stroke();
  };
  line(snd.Td, '#3ccf7a');
  line(snd.T, '#ef5b5b');
  g.fillStyle = '#8a96a8'; g.font = '9px Segoe UI';
  for (const p of [850, 700, 500, 300]) g.fillText(p, 2, yOf(p) - 2);
}

function drawHodo(canvas, snd, diag) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 132, H = canvas.clientHeight || 132;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = '#0a0d12'; g.fillRect(0, 0, W, H);
  if (!snd) return;
  let vmax = 20;
  const z0 = snd.z[0];
  for (let i = 0; i < snd.z.length; i++) if (snd.z[i] - z0 < 12000) vmax = Math.max(vmax, Math.hypot(snd.u[i], snd.v[i]));
  vmax = Math.ceil(vmax / 10) * 10;
  const s = (W / 2 - 4) / vmax, cx = W / 2, cy = H / 2;
  g.strokeStyle = '#26303d';
  for (let r = 10; r <= vmax; r += 10) { g.beginPath(); g.arc(cx, cy, r * s, 0, 2 * Math.PI); g.stroke(); }
  g.beginPath(); g.moveTo(0, cy); g.lineTo(W, cy); g.moveTo(cx, 0); g.lineTo(cx, H); g.stroke();
  const colors = [[1000, '#c060ff'], [3000, '#ef5b5b'], [6000, '#3ccf7a'], [9000, '#f2d24b'], [12000, '#4ad8e8']];
  g.lineWidth = 1.8;
  for (let i = 1; i < snd.z.length; i++) {
    const h = snd.z[i] - z0;
    if (h > 12000) break;
    g.strokeStyle = (colors.find(c => h <= c[0]) || colors[4])[1];
    g.beginPath(); g.moveTo(cx + snd.u[i - 1] * s, cy - snd.v[i - 1] * s); g.lineTo(cx + snd.u[i] * s, cy - snd.v[i] * s); g.stroke();
  }
  if (diag && diag.rm) {
    for (const [m, c] of [[diag.rm, '#ff7070'], [diag.lm, '#70a0ff']]) {
      g.fillStyle = c; g.beginPath(); g.arc(cx + m[0] * s, cy - m[1] * s, 3, 0, 2 * Math.PI); g.fill();
    }
  }
  g.fillStyle = '#8a96a8'; g.font = '9px Segoe UI'; g.fillText(vmax + ' m/s', 3, 10);
}

window.SkewT = { drawSkewT, drawHodo };
