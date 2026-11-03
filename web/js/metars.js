'use strict';
// Simulated METARs (arhtmodelviewer/metars.py) on the map: a station plot at each real ASOS site near the run
// (wind barb, temperature red, dewpoint green, gust orange, weather blue), the observation of the frame shown
// (the latest while live).  Click a station: its METAR decoded, the run's peak gust there and its last hours.
const Metar = (() => {
  let data = null;             // { t, stations } of the time loaded
  let key = '';                // run, version and time of what is loaded / loading
  let busy = false;
  let open = null;             // the icao whose card is open
  const KT = 0.514444;

  function frameT() {
    const fr = S.frames.length ? S.frames[S.frameIdx < 0 ? S.frames.length - 1 : Math.min(S.frameIdx, S.frames.length - 1)] : null;
    return fr && !S.live ? fr.t : null;
  }
  function on() { return $('showMetars').checked && S.run && S.run.metars_version; }

  async function load() {
    if (!on() || busy) return;
    const t = frameT();
    const k = `${S.run.run_id}/${S.run.metars_version}/${t === null ? 'live' : Math.floor(t / 600)}`;
    if (k === key) return;
    busy = true;
    try {
      data = await getJSON('/api/metars' + (t === null ? '' : `?t=${t}`));
      key = k;
      map.redraw();
      if (open) card(open, true);
    } catch (e) { /* the next poll tries again */ }
    busy = false;
  }

  const fmtT = (c, us) => Math.round(us ? c * 9 / 5 + 32 : c);

  // the barb: staff toward where the wind comes from, feathers on its clockwise side
  function barb(g, x, y, dir, kt) {
    if (kt < 1) {
      g.beginPath(); g.arc(x, y, 4.5, 0, 7); g.stroke();
      return;
    }
    const a = dir * Math.PI / 180;
    const ux = Math.sin(a), uy = -Math.cos(a);            // along the staff (screen: y down)
    const px = -uy, py = ux;                              // clockwise of it
    const L = 22, step = 4.2, F = 11;
    const fx = 0.42 * ux + 0.91 * px, fy = 0.42 * uy + 0.91 * py;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + ux * L, y + uy * L);
    let d = L, left = Math.round(kt / 5) * 5;
    const tris = [];
    while (left >= 50) {
      const bx = x + ux * d, by = y + uy * d;
      tris.push([bx, by, bx - ux * 6, by - uy * 6, bx + px * F + ux * 1.5, by + py * F + uy * 1.5]);
      d -= 8; left -= 50;
    }
    while (left >= 10) {
      const bx = x + ux * d, by = y + uy * d;
      g.moveTo(bx, by); g.lineTo(bx + fx * F, by + fy * F);
      d -= step; left -= 10;
    }
    if (left >= 5) {
      if (d === L) d -= step;
      const bx = x + ux * d, by = y + uy * d;
      g.moveTo(bx, by); g.lineTo(bx + fx * F * 0.55, by + fy * F * 0.55);
    }
    g.stroke();
    for (const t of tris) {
      g.beginPath(); g.moveTo(t[0], t[1]); g.lineTo(t[2], t[3]); g.lineTo(t[4], t[5]); g.closePath(); g.fill();
    }
  }

  function draw(g, m) {
    if (!on()) return;
    load();
    if (!data || !data.stations.length) return;
    const W = g.canvas.width, H = g.canvas.height;
    // thin the plots out when zoomed out: one per ~46 px cell, stations in or near storms first
    const taken = new Set();
    const list = data.stations.slice().sort((a, b) => (b.wx ? 1 : 0) - (a.wx ? 1 : 0) || b.kt - a.kt);
    g.save();
    g.lineCap = 'round'; g.lineJoin = 'round';
    g.font = 'bold 11px Segoe UI'; g.textBaseline = 'middle';
    for (const s of list) {
      const [x, y] = m.toScreen(s.x / 1000, s.y / 1000);
      if (x < -30 || y < -30 || x > W + 30 || y > H + 30) continue;
      const cell = Math.floor(x / 46) + ',' + Math.floor(y / 46);
      if (taken.has(cell)) continue;
      taken.add(cell);
      s._sx = x; s._sy = y;
      g.strokeStyle = 'rgba(0,0,0,.75)'; g.fillStyle = 'rgba(0,0,0,.75)'; g.lineWidth = 3.6;
      barb(g, x, y, s.dir, s.kt);
      g.strokeStyle = '#fff'; g.fillStyle = '#fff'; g.lineWidth = 1.5;
      barb(g, x, y, s.dir, s.kt);
      g.beginPath(); g.arc(x, y, 2.6, 0, 7); g.fill();
      const txt = (str, tx, ty, col, align) => {
        g.textAlign = align;
        g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.8)'; g.strokeText(str, tx, ty);
        g.fillStyle = col; g.fillText(str, tx, ty);
      };
      txt(fmtT(s.t_c, s.us), x - 7, y - 9, '#ff6b6b', 'right');
      txt(fmtT(s.td_c, s.us), x - 7, y + 9, '#5ee07a', 'right');
      if (s.gust) txt('G' + s.gust, x + 8, y + 9, '#ffbe3c', 'left');
      if (s.wx) txt(s.wx.split(' ')[0], x - 7, y + 22, '#7cc8ff', 'right');
    }
    g.restore();
  }

  function hit(x, y) {
    if (!on() || !data) return null;
    const [px, py] = map.toScreen(x, y);
    let best = null, bd = 20;            // the whole plot: the station, its barb's foot and the numbers left of it
    for (const s of data.stations) {
      if (s._sx === undefined) continue;
      const d = Math.hypot(s._sx - 6 - px, s._sy - py);
      if (d < bd) { bd = d; best = s; }
    }
    return best;
  }

  const kt2 = kt => `${Math.round(kt)} kt · ${Math.round(kt * 1.15078)} mph`;
  const temp = c => `${Math.round(c * 9 / 5 + 32)}°F · ${c.toFixed(1)}°C`;
  const COVER = { FEW: 'few', SCT: 'scattered', BKN: 'broken', OVC: 'overcast' };

  async function card(icao, quiet) {
    const el = $('metarCard');
    if (!icao) { el.classList.add('hidden'); open = null; return; }
    const t = frameT();
    let d;
    try { d = await getJSON(`/api/metar?icao=${icao}` + (t === null ? '' : `&t=${t}`)); } catch (e) {
      if (!quiet) { el.classList.add('hidden'); open = null; }
      return;
    }
    open = icao;
    const o = d.latest;
    const when = ms => new Date(ms).toISOString().slice(11, 16) + 'Z';
    const at = tt => when(d.start_ms + tt * 1000);
    const row = (a, b) => `<div class="cyc-row"><span>${a}</span><span>${b}</span></div>`;
    const sky = o.sky.length ? o.sky.map(([c, f]) => `${COVER[c]} ${Math.round(f / 100) * 100} ft`).join(', ') : 'clear below 12,000 ft';
    const wind = o.kt === 0 && !o.gust ? 'calm' : `${String(o.dir).padStart(3, '0')}° at ${kt2(o.kt)}` + (o.gust ? `, gusting ${o.gust} kt` : '');
    const hist = d.history.map(h => `<tr><td>${at(h.t)}</td><td>${h.kt === 0 && !h.gust ? 'calm' : String(h.dir).padStart(3, '0') + '/' + h.kt + (h.gust ? 'G' + h.gust : '')}</td>` +
      `<td>${Math.round(h.peak10)}</td><td>${fmtT(h.t_c, d.us)}/${fmtT(h.td_c, d.us)}</td><td>${h.slp.toFixed(1)}</td><td>${h.wx || ''}</td></tr>`).join('');
    $('metarCard').innerHTML = `<div class="cyc-head"><span class="cyc-badge" style="background:#7cc8ff">METAR</span>
      ${d.icao} <span class="metar-name">${d.name}</span><button class="cyc-x" title="Close">✕</button></div>
      <div class="metar-raw">${o.metar}</div>
      ${row('Observed', at(o.t) + (o.in_storm ? ' · in the storm grid' : ' · environment'))}
      ${row('Wind (2-min mean)', wind)}
      ${row('Peak gust, last 10 min', `${kt2(o.peak10)} from ${String(o.peak10_dir).padStart(3, '0')}°`)}
      ${row('Peak gust, last hour', `${kt2(o.pk[0])} at ${at(o.pk[2])}`)}
      ${row('<b>Max gust this run</b>', `<b>${kt2(d.best[0])}</b> from ${String(d.best[1]).padStart(3, '0')}° at ${at(d.best[2])}`)}
      ${row('Temperature', temp(o.t_c))}
      ${row('Dewpoint', temp(o.td_c) + ` · RH ${o.rh}%`)}
      ${row('Sea-level pressure', `${o.slp.toFixed(1)} hPa · altimeter ${o.alt.toFixed(2)} inHg`)}
      ${row('Visibility', `${o.vis >= 10 ? '10+' : o.vis.toFixed(2).replace(/\.?0+$/, '')} mi`)}
      ${row('Weather', o.wx || 'none')}
      ${row('Sky (ceilometer)', sky)}
      ${o.tops_ft ? row('Cloud tops overhead', `${Math.round(o.tops_ft / 1000)},000 ft`) : ''}
      ${o.dbz !== null && o.dbz > 5 ? row('Reflectivity overhead', `${o.dbz.toFixed(0)} dBZ · ${o.rate.toFixed(1)} mm/h`) : ''}
      ${row('Rain, last hour / run', `${o.p1h.toFixed(2)} in / ${o.ptot.toFixed(2)} in`)}
      ${o.tor_km !== null && o.tor_km < 60 ? row('Nearest tornado', `${o.tor_km.toFixed(1)} km`) : ''}
      ${row('Elevation', `${Math.round(d.elev)} m (the model ground is ${Math.round(d.model_ground)} m)`)}
      <table class="metar-hist"><tr><th>Time</th><th>Wind</th><th>Pk</th><th>${d.us ? '°F' : '°C'}</th><th>SLP</th><th>Wx</th></tr>${hist}</table>`;
    const s = data && data.stations.find(q => q.icao === icao);
    if (s && s._sx !== undefined && !quiet) {
      el.style.left = Math.max(8, Math.min(s._sx + 22, el.parentElement.clientWidth - 400)) + 'px';
      el.style.top = Math.max(s._sy - 40, 60) + 'px';
    }
    el.classList.remove('hidden');
  }

  function onPoll() { if (on()) load(); }

  function init() {
    $('showMetars').onchange = () => { key = ''; if (!$('showMetars').checked) card(null); map.redraw(); };
    $('metarCard').addEventListener('click', e => { if (e.target.closest('.cyc-x')) card(null); });
  }

  return { init, draw, hit, card, onPoll };
})();
window.Metar = Metar;
