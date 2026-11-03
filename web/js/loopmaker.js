// The loop maker (arhtmodelviewer/loops.py): an animated GIF of one radar's product at one tilt over a range of
// its volumes, on the map area in view, in any color table of the viewer's library (or the built-in one).
const LoopMaker = (() => {
  const $ = id => document.getElementById(id);
  let src = [], run = null, timer = null;

  function build() {
    if ($('lmOverlay')) return;
    const d = document.createElement('div');
    d.id = 'lmOverlay'; d.className = 'modal hidden'; d.setAttribute('role', 'dialog');
    d.innerHTML = `<div class="modal-box lm-box">
      <div class="modal-head"><b>Loop maker</b><span id="lmRun" class="meta"></span><span class="grow"></span>
        <button id="lmClose" class="icon" title="Close (Esc)" aria-label="Close">✕</button></div>
      <div class="lm-body">
        <div class="lm-form">
          <label>Radar <select id="lmSite"></select></label>
          <label>Product <select id="lmProd"><option value="BR">Reflectivity (BR)</option><option value="BV">Velocity (BV)</option>
            <option value="CC">Correlation coefficient (CC)</option><option value="ZDR">ZDR</option><option value="SW">Spectrum width</option>
            <option value="PHI">PHI</option></select></label>
          <label>Tilt <select id="lmTilt"></select></label>
          <label>From <select id="lmFirst"></select></label>
          <label>To <select id="lmLast"></select></label>
          <label title="Any .pal in the viewer's color-table library (Files tab adds folders)">Color table <select id="lmPal"></select></label>
          <label>Size <select id="lmSize"><option value="960x540">960 × 540</option><option value="1280x720" selected>1280 × 720</option>
            <option value="1920x1080">1920 × 1080</option><option value="1080x1080">1080 × 1080</option></select></label>
          <label>Frame time <select id="lmMs"><option value="120">0.12 s</option><option value="200">0.2 s</option>
            <option value="250" selected>0.25 s</option><option value="400">0.4 s</option><option value="700">0.7 s</option></select></label>
          <label class="chk"><input id="lmTracks" type="checkbox" checked> Tornado tracks</label>
          <p class="hint">The loop covers the map area in view now: pan and zoom the map to frame it first. Each frame
            is that volume's sweep nearest the tilt, from where the radar stood (a DOW that moved is drawn from each spot).</p>
          <div class="row"><button id="lmMake" class="primary">Make the loop</button></div>
          <p id="lmMsg" class="hint"></p>
        </div>
        <div class="lm-view"><img id="lmImg" alt=""><a id="lmDl" class="hidden" href="#">Download the GIF</a></div>
      </div></div>`;
    document.body.appendChild(d);
    $('lmClose').onclick = close;
    d.addEventListener('mousedown', e => { if (e.target === d) close(); });
    document.addEventListener('keydown', e => { if (e.code === 'Escape' && !d.classList.contains('hidden')) close(); });
    $('lmSite').onchange = fillSite;
    $('lmProd').onchange = fillPal;
    $('lmMake').onclick = make;
  }

  async function open(runId) {
    build();
    $('lmOverlay').classList.remove('hidden');
    $('lmMsg').textContent = 'Reading the run’s radars…';
    try {
      const r = await fetch('/api/loop/sources' + (runId ? '?run=' + encodeURIComponent(runId) : ''));
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || r.statusText);
      src = d.sources; run = d.run;
    } catch (e) { $('lmMsg').textContent = 'No radar data: ' + e.message; return; }
    $('lmRun').textContent = 'run ' + run;
    $('lmSite').innerHTML = src.map(s => `<option value="${s.site}">${s.site} (${s.volumes.length} volumes)</option>`).join('');
    if (S.radar && src.some(s => s.site === S.radar)) $('lmSite').value = S.radar;
    fillSite(); await fillPal();
    $('lmMsg').textContent = src.length ? '' : 'This run has no Level II files yet.';
  }

  function fillSite() {
    const s = src.find(q => q.site === $('lmSite').value);
    if (!s) return;
    $('lmTilt').innerHTML = s.tilts.map(t => `<option value="${t}">${t.toFixed(1)}°</option>`).join('');
    const opts = s.volumes.map(v => `<option value="${v.name}">${v.time}</option>`).join('');
    $('lmFirst').innerHTML = opts; $('lmLast').innerHTML = opts;
    $('lmFirst').selectedIndex = 0; $('lmLast').selectedIndex = s.volumes.length - 1;
  }

  async function fillPal() {
    let pals = [];
    try { pals = await (await fetch('/api/palettes')).json(); } catch (e) { }
    const p = $('lmProd').value;
    const want = { BR: ['BR', 'REF'], BV: ['BV', 'VEL', 'SRV'], CC: ['CC', 'RHO'], ZDR: ['ZDR'], SW: ['SW'], PHI: ['PHI'] }[p];
    $('lmPal').innerHTML = '<option value="">Built-in</option>' + pals.filter(q => want.includes(String(q.product).toUpperCase()))
      .map(q => `<option value="${q.path.replace(/"/g, '&quot;')}">${q.name}</option>`).join('');
  }

  async function make() {
    const [w, h] = $('lmSize').value.split('x').map(Number);
    const body = { run, site: $('lmSite').value, product: $('lmProd').value, tilt: +$('lmTilt').value,
      first: $('lmFirst').value, last: $('lmLast').value, palette: $('lmPal').value || null, size: [w, h],
      frame_ms: +$('lmMs').value, tracks: $('lmTracks').checked,
      box: { cx: map.cx, cy: map.cy, w: map.W / map.scale } };
    $('lmMake').disabled = true; $('lmDl').classList.add('hidden');
    $('lmMsg').textContent = 'Starting…';
    let job;
    try {
      const r = await fetch('/api/loop/make', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || r.statusText);
      job = d.job;
    } catch (e) { $('lmMsg').textContent = 'Could not start: ' + e.message; $('lmMake').disabled = false; return; }
    clearInterval(timer);
    timer = setInterval(async () => {
      const s = await (await fetch('/api/loop/status?job=' + job)).json();
      if (s.state === 'running') { $('lmMsg').textContent = `Drawing frame ${s.done} of ${s.total || '…'}`; return; }
      clearInterval(timer); $('lmMake').disabled = false;
      if (s.state === 'done') {
        const url = '/api/loop/file?name=' + encodeURIComponent(s.name);
        $('lmImg').src = url + '&t=' + Date.now();
        $('lmDl').href = url + '&download=1'; $('lmDl').classList.remove('hidden');
        $('lmMsg').textContent = `Done: ${s.total} frames (output\\loops\\${s.name})`;
      } else $('lmMsg').textContent = 'The loop failed: ' + (s.error || s.state);
    }, 800);
  }

  function close() { const d = $('lmOverlay'); if (d) d.classList.add('hidden'); }
  return { open, close };
})();
