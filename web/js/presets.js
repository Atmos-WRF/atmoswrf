'use strict';
// Saved setups (setups.py): the next run's settings under a name, as a file in the viewer's setups
// folder with the sounding inside, so a setup can be sent to anyone and opened in their viewer.

const Presets = (() => {
  let items = [];
  let noteTimer = null;
  const el = id => document.getElementById(id);
  const gridLabel = k => (S.defaults && S.defaults.grids[k] && S.defaults.grids[k].label) || k || '';
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

  function describe(s) {
    const dom = s.domain || {};
    const parts = [s.sounding + (s.changes.length ? ` (+${s.changes.length} change${s.changes.length === 1 ? '' : 's'})` : ''),
      gridLabel(s.grid), dom.mode === 'fixed' ? `fixed ${dom.size_km} km` : dom.max_km ? `patches up to ${dom.max_km} km` : 'unlimited'];
    if (s.storms) parts.push(plural(s.storms, 'storm'));
    if (s.radars.length) parts.push(s.radars.join(', '));
    if (s.boundaries) parts.push(plural(s.boundaries, 'front'));
    if (s.zones) parts.push(plural(s.zones, 'initiation area'));
    if (s.nudgers) parts.push(plural(s.nudgers, 'nudger'));
    if (s.auto) parts.push('OutbreakLab mode');
    return parts.join(' · ');
  }

  function note(text) {
    el('psNote').textContent = text;
    clearTimeout(noteTimer);
    noteTimer = setTimeout(() => { el('psNote').textContent = ''; }, 8000);
  }

  async function refresh() {
    try { items = await getJSON('/api/setups'); } catch (e) { return; }
    setHTML('psList', items.map(it =>
      `<div class="item col" data-name="${esc(it.name)}"><div><b>${esc(it.name)}</b><br><span class="meta">` +
      `${esc(describe(it.summary))}<br>saved ${esc(String(it.saved).replace('T', ' ').slice(0, 16))}</span></div>` +
      '<span class="btns"><button data-ps="load" title="Make it the setup for the next run">Load</button>' +
      '<button data-ps="export" title="Download the file, to send it to someone">Export</button>' +
      '<button data-ps="show" title="Show the file in its folder (to drag it into a message)">Show file</button>' +
      '<button data-ps="del" title="Delete this saved setup">Delete</button></span></div>').join('') ||
      '<div class="hint">No saved setups yet.</div>');
  }

  // a name to start from: the sounding's own (its file's or title, without the changes) and the grid
  const suggested = settings => {
    const sp = settings.sounding || {};
    const t = sp.title || (sp.file ? sp.file.split(/[\\/]/).pop().replace(/\.[^.]+$/, '') : sp.analytic ? 'Weisman-Klemp' : 'my setup');
    return `${t} ${gridLabel(settings.grid)}`.replace(/[<>:"/\\|?*]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 70);
  };

  // the setup for the next run (what Start new run would use), or the settings of the run on the map
  async function save(fromRun) {
    const settings = fromRun ? S.run && S.run.settings : readSetupFromForm();
    if (!settings) { alert('There is no run to save.'); return; }
    const name = prompt(fromRun ? "Save this run's settings as:" : 'Save the setup as:', suggested(settings));
    if (!name || !name.trim()) return;
    if (items.some(i => i.name.toLowerCase() === name.trim().toLowerCase()) &&
        !confirm(`"${name.trim()}" is saved already. Replace it?`)) return;
    try {
      const got = await getJSON('/api/setups/save', { name: name.trim(), settings });
      await refresh();
      if (got.warnings.length) alert(got.warnings.join('\n\n'));
      note(`Saved "${got.name}" (in the viewer's setups folder).`);
    } catch (e) { alert('Save: ' + e.message); }
  }

  async function load(name) {
    if (runActive()) {
      if (!confirm(`Loading "${name}" stops the current run (its frames and Level II files are kept). Go on?`)) return;
      try { await api('/api/run/stop', {}); S.run.status = 'stopped'; } catch (e) { alert('Stop: ' + e.message); return; }
    }
    try {
      const d = await getJSON('/api/setups/get?name=' + encodeURIComponent(name));
      await useSetup(d.settings);
      note(`Loaded "${d.name}". Launch run starts it.`);
    } catch (e) { alert('Load: ' + e.message); }
  }

  function exportFile(name) {
    const a = document.createElement('a');
    a.href = '/api/setups/file?name=' + encodeURIComponent(name);
    a.download = name + '.json';
    document.body.appendChild(a); a.click(); a.remove();
    note(`"${name}.json" goes to your Downloads folder: send it to anyone, who opens it with Open a setup file….`);
  }

  async function show(name) {
    try { await getJSON('/api/setups/reveal', name ? { name } : {}); } catch (e) { alert('Show: ' + e.message); }
  }

  async function remove(name) {
    if (!confirm(`Delete the saved setup "${name}"? (Its file is removed from the setups folder.)`)) return;
    try { await getJSON('/api/setups/delete', { name }); await refresh(); } catch (e) { alert('Delete: ' + e.message); }
  }

  // setups someone sent (or a run's settings.json): copied into the setups folder
  async function open(files) {
    const done = [], failed = [], warns = [];
    for (const f of files) {
      try {
        if (f.size > 8e6) throw new Error('too big for a setup');
        const got = await getJSON('/api/setups/import', { name: f.name.replace(/\.json$/i, ''), text: await f.text() });
        done.push(got.name);
        warns.push(...got.warnings.map(w => `${got.name}: ${w}`));
      } catch (e) { failed.push(`${f.name}: ${e.message}`); }
    }
    await refresh();
    if (failed.length) alert('Not opened:\n' + failed.join('\n'));
    if (warns.length) alert(warns.join('\n\n'));
    if (done.length === 1 && confirm(`Opened "${done[0]}". Load it now?`)) await load(done[0]);
    else if (done.length) note(`Opened ${plural(done.length, 'setup')}: ${done.join(', ')}.`);
  }

  function init() {
    el('psSave').onclick = () => save(false);
    el('psSaveRun').onclick = () => save(true);
    el('psFolder').onclick = () => show(null);
    el('psOpen').onchange = async e => { await open([...e.target.files]); e.target.value = ''; };
    el('psList').addEventListener('click', e => {
      const b = e.target.closest('button[data-ps]');
      if (!b) return;
      const name = b.closest('[data-name]').dataset.name;
      ({ load, export: exportFile, show, del: remove })[b.dataset.ps](name);
    });
    document.querySelector('.tabs button[data-tab="files"]').addEventListener('click', refresh);
    refresh();
  }

  // the "Save this run's settings" button only while there is a run to save
  function update() { el('psSaveRun').classList.toggle('hidden', !(S.run && S.run.settings) || !!S.detached); }

  return { init, refresh, save, update };
})();

window.Presets = Presets;
