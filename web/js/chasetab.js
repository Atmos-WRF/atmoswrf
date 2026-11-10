'use strict';
// The main page's Chase tab: opens chase mode (/chase) and runs its hosting from here (arhtmodelviewer/chase.py
// through /api/c/host and /api/c/tunnel, this PC only): friends on or off, the code, the links, the internet link
// and who is in.  Polled every 3 s while the tab is open.
const ChaseTab = (() => {
  const $ = id => document.getElementById(id);
  let timer = null, busy = false, acts = 0;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  async function post(path, body) {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }
  function link(u) {
    return `<div class="row buttons"><a href="${esc(u)}" target="_blank" rel="noopener" style="flex:1;word-break:break-all">${esc(u)}</a>` +
           `<button data-copy="${esc(u)}" title="Copy the link">Copy</button></div>`;
  }
  function show(d) {
    $('chHost').checked = !!d.on;
    $('chCodeRow').classList.toggle('hidden', !d.on);
    $('chCode').textContent = d.code || '';
    $('chUrls').innerHTML = d.on ? ((d.urls || []).map(link).join('') || '<span class="hint">No network address found on this PC.</span>') : '';
    const t = d.tunnel || {};
    $('chTunnel').checked = t.state === 'on' || t.state === 'starting';
    $('chTunnel').disabled = !d.on;
    $('chTunnelNote').innerHTML = t.state === 'on' && t.url ? link(t.url)
      : t.state === 'starting' ? 'Starting the internet link…'
      : t.error ? esc(t.error) : d.on ? '' : 'Switch on Let friends join first.';
    const P = d.players || [];
    $('chPlayers').innerHTML = P.length
      ? P.map(p => `<div>${esc(p.name)}${p.local ? ' <span class="hint">(this PC)</span>' : ''}</div>`).join('')
      : '<span class="hint">Nobody yet.</span>';
  }
  async function refresh() {
    if (busy) return;
    busy = true;
    try {
      const st = await fetch('/api/state').then(r => r.json()).catch(() => ({}));
      $('chRun').textContent = !st.status || st.status === 'none'
        ? 'No run yet: launch one (or open a saved run) and chase mode drives in it. With no run going it shows the newest saved 3-D snapshot.'
        : st.status === 'running' || st.status === 'waiting' ? 'The run is going: chase mode follows it live.'
        : 'The run has stopped: chase mode shows its last 3-D snapshot.';
      const n = acts, r = await fetch('/api/c/host');
      if (r.status === 404) $('chTunnelNote').textContent = 'Hosting needs the viewer restarted (new in this version).';
      else if (r.ok) { const d = await r.json(); if (n === acts) show(d); }   // not if a click changed it meanwhile
    } catch (e) { /* the server is restarting */ }
    busy = false;
  }
  async function act(f) {
    acts++;
    try { show(await f()); } catch (e) { $('chTunnelNote').textContent = e.message; }
  }
  function init() {
    $('chOpen').onclick = () => window.open('/chase', '_blank');
    $('chHost').onchange = e => act(() => post('/api/c/host', { on: e.target.checked }));
    $('chNewCode').onclick = () => act(() => post('/api/c/host', { new_code: true }));
    $('chTunnel').onchange = e => act(async () => {
      await post('/api/c/tunnel', { on: e.target.checked });
      return fetch('/api/c/host').then(r => r.json());
    });
    $('tab-chase').addEventListener('click', e => {
      const b = e.target.closest('[data-copy]');
      if (b) navigator.clipboard.writeText(b.dataset.copy).then(() => { b.textContent = 'Copied'; setTimeout(() => (b.textContent = 'Copy'), 1200); });
    });
    const tab = document.querySelector('.tabs button[data-tab="chase"]');
    const watch = () => {
      const on = tab.classList.contains('on');
      if (on && !timer) { refresh(); timer = setInterval(refresh, 3000); }
      if (!on && timer) { clearInterval(timer); timer = null; }
    };
    document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => setTimeout(watch, 0)));
  }
  document.addEventListener('DOMContentLoaded', init);
  return { refresh };
})();
