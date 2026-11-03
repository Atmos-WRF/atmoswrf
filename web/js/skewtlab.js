'use strict';
// Skew-T Lab (web/skewt-lab, a copy of the standalone editor) in an overlay.  It gets the setup's sounding
// as the sounding editor does: the profile itself, without the Setup panel's changes, as SHARPpy text
// (SndEdit.sharppyText), read with Skew-T Lab's own reader.  "Use this sounding" takes its SHARPpy export
// and loads it as pasted text is (loadSounding({text})), so the Setup panel's changes apply on top again
// and the parameters, plots and run settings follow.

const SkewTLabView = (() => {
  const SRC = '/static/skewt-lab/index.html';
  let isOpen = false, ready = null, startText = '', baseTitle = '';
  const el = id => document.getElementById(id);

  const frame = () => el('stlFrame');
  const lab = () => { const w = frame().contentWindow; return w && w.SkewTLab && w.Sharppy ? w : null; };

  function say(text, bad) {
    el('stlMsg').textContent = text || '';
    el('stlMsg').classList.toggle('bad', !!bad);
  }

  // the iframe is loaded on first use and kept, so Skew-T Lab's own settings (parcel, brush) persist
  function whenReady() {
    if (lab()) return Promise.resolve(lab());
    if (!ready) {
      ready = new Promise((resolve, reject) => {
        const f = frame();
        const t = setTimeout(() => { ready = null; reject(new Error('Skew-T Lab did not load.')); }, 20000);
        f.addEventListener('load', function done() {
          f.removeEventListener('load', done);
          clearTimeout(t);
          const w = lab();
          if (!w) { ready = null; reject(new Error('Skew-T Lab loaded without its SkewTLab handle.')); return; }
          hookKeys(w);
          resolve(w);
        });
        f.src = SRC;
      });
    }
    return ready;
  }

  // Esc inside the iframe closes its own dialog first, then the overlay
  function hookKeys(w) {
    w.document.addEventListener('keydown', e => {
      if (!isOpen || e.key !== 'Escape') return;
      const m = w.document.getElementById('modal');
      if (m && !m.hidden) { m.hidden = true; return; }
      e.preventDefault();
      close(false);
    });
  }

  async function open() {
    if (!S.snd) { alert('Load a sounding first.'); return; }
    el('stlOverlay').classList.remove('hidden');
    isOpen = true;
    el('stlUse').disabled = true;
    say('Loading the sounding…');
    let text;
    try {
      // the profile itself: the Setup panel's changes apply on top of whatever comes back
      const base = await getJSON('/api/sounding/load', { spec: { ...baseSpec(S.setup.sounding), ice_cap: false } });
      baseTitle = base.sounding.title || S.snd.sounding.title || 'Sounding';
      text = SndEdit.sharppyText(base.sounding, baseTitle);
    } catch (e) { say('Could not read the sounding: ' + e.message, true); return; }
    let w;
    try { w = await whenReady(); } catch (e) { say(e.message, true); return; }
    if (!isOpen) return;
    try {
      const r = w.Sharppy.fromText(text);
      r.meta.station = 'ARHT';
      r.meta.time = '000000/0000';
      w.SkewTLab.load({ levels: r.levels, meta: r.meta });
      const pre = w.document.getElementById('preset');
      if (pre) pre.value = '';
      const res = w.document.getElementById('resolution');
      if (res) res.value = '';
    } catch (e) { say('Skew-T Lab could not read the sounding: ' + e.message, true); return; }
    el('stlTitle').textContent = baseTitle;
    startText = w.SkewTLab.exportText();
    el('stlUse').disabled = false;
    say(`${w.SkewTLab.state.prof.levels.length} levels. The Atmosphere tab's adjustments (mix, scale winds, moisture, cap, jet, winds aloft) apply on top, as they do to any sounding.`);
    w.SkewTLab.relayout();
    setTimeout(() => { if (isOpen && lab()) lab().SkewTLab.relayout(); }, 60);
    frame().focus();
  }

  const edited = () => { const w = lab(); return !!w && !!startText && w.SkewTLab.exportText() !== startText; };

  function close(force) {
    if (!isOpen) return;
    if (!force && edited() && !confirm('Close Skew-T Lab and lose your changes to the sounding?')) return;
    isOpen = false;
    el('stlOverlay').classList.add('hidden');
  }

  async function use() {
    const w = lab();
    if (!w) { say('Skew-T Lab is not loaded.', true); return; }
    if (!edited()) { close(true); return; }          // unchanged: keep the sounding as it was (its file)
    let text;
    try {
      text = w.SkewTLab.exportText();
      w.Sharppy.fromText(text);                      // throws on a profile it cannot read back
    } catch (e) { say('Could not export the sounding: ' + e.message, true); return; }
    const title = /\((edited|Skew-T Lab)\)$/.test(baseTitle) ? baseTitle : `${baseTitle} (Skew-T Lab)`;
    el('stlUse').disabled = true;
    say('Applying…');
    const ok = await loadSounding({ text, title }, e => say('The viewer could not use this sounding: ' + e.message, true));
    el('stlUse').disabled = false;
    if (ok) { $('sndText').value = text; close(true); }
  }

  function init() {
    el('stlClose').onclick = () => close(false);
    el('stlCancel').onclick = () => close(false);
    el('stlUse').onclick = use;
    el('stlReload').onclick = () => { if (!edited() || confirm('Discard your changes and load the setup\'s sounding again?')) open(); };
    el('stlOverlay').addEventListener('mousedown', e => { if (e.target === el('stlOverlay')) close(false); });
    // while it is open the map's keys (Space, arrows) are off; Esc closes it
    document.addEventListener('keydown', e => {
      if (!isOpen) return;
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(false); return; }
      if (!['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) e.stopPropagation();
    }, true);
  }

  return { init, open, close, isOpen: () => isOpen };
})();

window.SkewTLabView = SkewTLabView;
