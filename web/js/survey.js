// The tornado list and damage survey (arhtmodelviewer/survey.py): a window with every tornado of the run,
// strongest first, with the map's EF rating; downloads (Google Earth KML with the tracks coloured by EF
// as on the map, CSV, Markdown, GR2Analyst placefile) and the live placefile's address.  Opens by itself when a run with
// tornadoes ends.  Click a tornado: the map goes to its track.
const Survey = (() => {
  const $ = id => document.getElementById(id);
  const EF_RGB = ['#7fdcff', '#3ccf7a', '#f5d33a', '#f2913a', '#e8413c', '#c13ae8'];     // the map's (app.js EF_COLORS)
  let lastStatus = null, lastRun = null, shownFor = null, current = null;

  function build() {
    if ($('svOverlay')) return;
    const d = document.createElement('div');
    d.id = 'svOverlay'; d.className = 'modal hidden'; d.setAttribute('role', 'dialog');
    d.innerHTML = `<div class="modal-box sv-box">
      <div class="modal-head"><b>Tornado list</b><span id="svTitle" class="meta"></span><span class="grow"></span>
        <button id="svRefresh" title="Read the run's tornadoes again">Refresh</button>
        <button id="svClose" class="icon" title="Close (Esc)" aria-label="Close">✕</button></div>
      <div class="sv-body"><p id="svSummary" class="hint"></p>
        <div class="sv-tablewrap"><table id="svTable"><thead><tr><th>#</th><th class="l">Tornado</th><th>EF</th>
          <th>Wind</th><th>Vrot</th><th>Min CC</th><th class="l">Where</th><th>Start</th><th>Min</th><th>Miles</th></tr></thead>
          <tbody></tbody></table></div>
        <p class="hint">The same rating as the tracks on the map, as an NWS office rates a tornado off the radar: along
          each track the 0.5° Vrot gives the EF (33 kt EF1, 42 EF2, 52 EF3, 63 EF4, 74+ EF5), a debris signature (CC under
          0.80) makes it at least EF1 (under 0.60: EF2), and without one it is held to EF3; a tornado is rated its highest. Wind: the model's peak, within that EF's range. Named after the biggest town on the track
          (else the nearest). The damage survey is the track itself, each stretch in its EF's colour, as on the map.
          Click a tornado to see its track.</p></div>
      <div class="modal-foot"><span class="hint">GR2Analyst live placefile: <code id="svPlace"></code></span>
        <span class="grow"></span>
        <button data-sv="kml" title="Google Earth: every track with its facts, coloured by EF along it">Google Earth (KML)</button>
        <button data-sv="placefile" title="A GR2Analyst placefile of the damage survey (tracks coloured by EF, labels) to keep">Placefile</button>
        <button data-sv="csv">CSV</button><button data-sv="md">Markdown</button></div></div>`;
    document.body.appendChild(d);
    $('svClose').onclick = close;
    $('svRefresh').onclick = () => open(current);
    d.addEventListener('mousedown', e => { if (e.target === d) close(); });
    document.addEventListener('keydown', e => { if (e.code === 'Escape' && !d.classList.contains('hidden')) close(); });
    d.querySelectorAll('[data-sv]').forEach(b => b.onclick = () => {
      const q = new URLSearchParams({ kind: b.dataset.sv }); if (current) q.set('run', current);
      location.href = '/api/survey/download?' + q;
    });
    $('svPlace').textContent = location.origin + '/poll/survey.txt';
  }

  async function open(runId) {
    build();
    current = runId || null;
    $('svOverlay').classList.remove('hidden');
    $('svSummary').textContent = 'Reading the run…';
    const body = $('svTable').querySelector('tbody');
    body.innerHTML = '';
    let d;
    try {
      const r = await fetch('/api/survey' + (runId ? '?run=' + encodeURIComponent(runId) : ''));
      d = await r.json();
      if (!r.ok) throw new Error(d.error || r.statusText);
    } catch (e) { $('svSummary').textContent = 'No tornado list: ' + e.message; return; }
    $('svTitle').textContent = 'run ' + d.run_id;
    $('svSummary').textContent = d.summary;
    body.innerHTML = d.rows.map(r => `<tr data-i="${r.rank - 1}">
      <td>${r.rank}</td><td class="l"><b>${esc(r.name)}</b>${r.active ? ' <span class="sv-live">on the ground</span>' : ''}</td>
      <td><span class="sv-ef" style="background:${EF_RGB[r.ef]}">EF${r.ef}</span></td><td>${r.wind_mph} mph</td>
      <td>${r.vrot_kt} kt</td><td>${r.cc == null ? '—' : r.cc.toFixed(2) + (r.tds ? ' TDS' : '')}</td>
      <td class="l">${esc(r.where)}</td><td>${esc(r.start_utc)}</td><td>${r.minutes}</td><td>${r.miles}</td></tr>`).join('')
      || '<tr><td colspan="10" class="l">No tornadoes in this run.</td></tr>';
    body.querySelectorAll('tr[data-i]').forEach(tr => tr.onclick = () => {
      const r = d.rows[+tr.dataset.i];
      const w = Math.max((r.x1 - r.x0) / 1000, 4), h = Math.max((r.y1 - r.y0) / 1000, 4);
      const sc = Math.min(map.W / (w * 1.3), map.H / (h * 1.3), 60);
      map.center((r.x0 + r.x1) / 2000, (r.y0 + r.y1) / 2000, sc);
      close();
    });
  }

  function close() { const d = $('svOverlay'); if (d) d.classList.add('hidden'); }
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // a run that ends with tornadoes opens the list once
  function onPoll(st) {
    const ended = st && ['stopped', 'error'].includes(st.status);
    if (st && st.run_id === lastRun && ended && lastStatus && !['stopped', 'error'].includes(lastStatus)
        && (st.tornado_tracks || []).length && shownFor !== st.run_id) {
      shownFor = st.run_id;
      setTimeout(() => open(st.run_id), 1500);      // the run's state.json is written as it ends
    }
    lastStatus = st ? st.status : null; lastRun = st ? st.run_id : null;
  }
  return { open, close, onPoll };
})();
