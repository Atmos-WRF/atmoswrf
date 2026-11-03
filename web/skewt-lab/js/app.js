/* app.js - state, panels, history and the export/import dialogs. */
(function () {
  var T = Thermo;
  var $ = function (s) { return document.querySelector(s); };

  var state = {
    prof: null, params: null, selected: 0,
    opts: { parcel: 'ml', stormMode: 'rm', brush: 0 },
    hist: [], hidx: -1, pendingSnapshot: null
  };

  var skewt = new SkewTView($('#skewt'), {
    onEdit: onEdit, onDragStart: beginEdit, onDragEnd: commitEdit,
    onSelect: function (i) { state.selected = i; renderLevels(); },
    onAdd: addLevelAt, onRemove: removeLevel
  });
  var hodo = new HodoView($('#hodo'), {
    onEdit: onEdit, onDragStart: beginEdit, onDragEnd: commitEdit,
    onSelect: function (i) { state.selected = i; renderLevels(); }
  });

  /* ---------------- state plumbing ---------------- */
  function load(sample) {
    state.prof = new Profile.Sounding(sample.levels, sample.meta);
    state.selected = 0;
    state.hist = []; state.hidx = -1;
    snapshot();
    rebuild();
  }
  function rebuild() {
    state.prof.build();
    state.params = Params.compute(state.prof, state.opts);
    render();
  }
  var pending = 0;
  function onEdit() {                       // called on every pointermove of a drag
    if (pending) return;
    pending = setTimeout(function () { pending = 0; rebuild(); }, 8);
  }
  function render() {
    state.opts.selected = state.selected;
    skewt.setState(state.prof, state.params, state.opts);
    hodo.setState(state.prof, state.params, state.opts);
    renderParams();
    renderHodoFoot();
    if (!state.dragging) renderLevels();    // don't stomp on inputs mid-drag
    $('#undo').disabled = state.hidx <= 0;
    $('#redo').disabled = state.hidx >= state.hist.length - 1;
  }

  /* ---------------- history ---------------- */
  function snapshot() {
    var snap = { levels: state.prof.copyLevels(), elev: state.prof.meta.elev };
    state.hist = state.hist.slice(0, state.hidx + 1);
    state.hist.push(snap);
    if (state.hist.length > 120) state.hist.shift();
    state.hidx = state.hist.length - 1;
  }
  function beginEdit() { state.pendingSnapshot = true; state.dragging = true; }
  function commitEdit() {
    state.dragging = false;
    if (state.pendingSnapshot) { state.pendingSnapshot = false; snapshot(); }
    rebuild();
  }
  function restore(i) {
    if (i < 0 || i >= state.hist.length) return;
    state.hidx = i;
    var s = state.hist[i];
    state.prof.levels = s.levels.map(function (l) {
      return { p: l.p, t: l.t, td: l.td, dir: l.dir, spd: l.spd };
    });
    state.prof.meta.elev = s.elev;
    rebuild();
  }

  /* ---------------- level edits ---------------- */
  function addLevelAt(p) {
    var prof = state.prof;
    var s = Profile.sample(prof.levels, p);
    var ds = T.dirspd(s.u, s.v);
    snapshot();
    prof.levels.push({ p: p, t: s.t, td: s.td, dir: ds[0], spd: ds[1] });
    prof.levels.sort(function (a, b) { return b.p - a.p; });
    state.selected = prof.levels.findIndex(function (l) { return l.p === p; });
    snapshot();
    rebuild();
  }
  function removeLevel(i) {
    if (state.prof.levels.length <= 4) return;
    snapshot();
    state.prof.levels.splice(i, 1);
    state.selected = Math.min(state.selected, state.prof.levels.length - 1);
    snapshot();
    rebuild();
  }

  /* ---------------- parameter panel ---------------- */
  function f(x, n, dash) {
    if (x === null || x === undefined || !isFinite(x)) return dash || '––';
    return x.toFixed(n === undefined ? 0 : n);
  }
  function lvl(v, t) {                       // t = ascending thresholds
    if (v === null || v === undefined || !isFinite(v)) return 0;
    var n = 0;
    for (var i = 0; i < t.length; i++) if (v >= t[i]) n = i + 1;
    return n;
  }
  function row(k, v, u, l) {
    return '<div class="prow"><span class="k">' + k + '</span>' +
           '<span class="v l' + (l || 0) + '">' + v +
           (u ? '<span class="u">' + u + '</span>' : '') + '</span></div>';
  }
  function group(title, rows) {
    return '<div class="pgroup"><h3>' + title + '</h3>' + rows.join('') + '</div>';
  }

  var CAPE_T = [500, 1500, 2500, 4000];
  var SRH_T = [100, 200, 350, 500];
  var SHEAR_T = [20, 35, 50, 65];

  function renderParams() {
    var P = state.params, prof = state.prof, out = [];

    var pr = [];
    [['sfc', 'SB'], ['ml', 'ML'], ['mu', 'MU']].forEach(function (k) {
      var p = P[k[0]], n = k[1];
      pr.push(row(n + 'CAPE', f(p.cape), 'J/kg', lvl(p.cape, CAPE_T)));
      pr.push(row(n + 'CIN', f(p.cin), 'J/kg', lvl(-p.cin, [25, 75, 150, 250])));
      pr.push(row(n + 'LCL', f(p.lclz), 'm', lvl(-p.lclz, [-1500, -1200, -900, -600])));
      pr.push(row(n + 'LFC', f(p.lfcz), 'm', 0));
      pr.push(row(n + 'EL', f(p.elz), 'm', 0));
      pr.push(row(n + 'LI', f(p.li, 1), '°C', lvl(-p.li, [2, 5, 8, 11])));
    });
    out.push(group('Parcels', pr));

    out.push(group('Moisture &amp; lapse rates', [
      row('Precipitable water', f(P.pwat, 2), 'in', lvl(P.pwat, [0.75, 1.25, 1.75, 2.1])),
      row('ML mixing ratio', f(P.mlMixRatio, 1), 'g/kg', lvl(P.mlMixRatio, [10, 13, 16, 19])),
      row('Sfc / ML RH', f(P.sfcRH) + ' / ' + f(P.mlRH), '%', 0),
      row('0–3 km lapse rate', f(P.lr03, 1), '°C/km', lvl(P.lr03, [6, 7.5, 8.5, 9.3])),
      row('3–6 km lapse rate', f(P.lr36, 1), '°C/km', lvl(P.lr36, [6, 7, 7.5, 8])),
      row('700–500 lapse rate', f(P.lr75, 1), '°C/km', lvl(P.lr75, [6.5, 7.5, 8.2, 8.8])),
      row('850–500 lapse rate', f(P.lr85, 1), '°C/km', lvl(P.lr85, [6, 6.8, 7.4, 8]))
    ]));

    out.push(group('Levels &amp; downdrafts', [
      row('Freezing level', f(P.fzl), 'm', 0),
      row('Wet-bulb zero', f(P.wbz), 'm', 0),
      row('−20 °C level', f(P.m20), 'm', 0),
      row('Hail growth (−10/−30)', f(P.m10) + ' – ' + f(P.m30), 'm', 0),
      row('DCAPE', f(P.dcape), 'J/kg', lvl(P.dcape, [500, 800, 1100, 1400])),
      row('Convective temp', f(P.tconv, 1), '°C', 0),
      row('Forecast max temp', f(P.maxTemp, 1), '°C', 0)
    ]));

    out.push(group('Legacy indices', [
      row('K index', f(P.kindex), '', lvl(P.kindex, [20, 30, 35, 40])),
      row('Total totals', f(P.totaltotals), '', lvl(P.totaltotals, [45, 50, 55, 60])),
      row('SWEAT', f(P.sweat), '', lvl(P.sweat, [250, 350, 450, 600])),
      row('Showalter', f(P.showalter, 1), '°C', lvl(-P.showalter, [0, 3, 6, 9]))
    ]));

    out.push(group('Shear', [
      row('0–500 m', f(P.shr500), 'kt', lvl(P.shr500, [10, 18, 25, 35])),
      row('0–1 km', f(P.shr1), 'kt', lvl(P.shr1, [15, 25, 35, 45])),
      row('0–3 km', f(P.shr3), 'kt', lvl(P.shr3, [20, 30, 40, 50])),
      row('0–6 km', f(P.shr6), 'kt', lvl(P.shr6, SHEAR_T)),
      row('0–8 km', f(P.shr8), 'kt', lvl(P.shr8, SHEAR_T)),
      row('Effective bulk shear', f(P.ebwd), 'kt', lvl(P.ebwd, SHEAR_T))
    ]));

    var eff = P.eff;
    out.push(group('Helicity &amp; storm relative', [
      row('SRH 0–500 m', f(P.srh500), 'm²/s²', lvl(Math.abs(P.srh500), [50, 100, 175, 250])),
      row('SRH 0–1 km', f(P.srh1), 'm²/s²', lvl(Math.abs(P.srh1), SRH_T)),
      row('SRH 0–3 km', f(P.srh3), 'm²/s²', lvl(Math.abs(P.srh3), SRH_T)),
      row('Effective SRH', f(P.esrh), 'm²/s²', lvl(Math.abs(P.esrh), SRH_T)),
      row('Effective inflow', eff.botP === null ? 'none'
        : f(eff.botZ) + ' – ' + f(eff.topZ), eff.botP === null ? '' : 'm', 0),
      row('Critical angle', f(P.critAngle), '°',
        (P.critAngle !== null && P.critAngle > 55 && P.critAngle < 125) ? 2 : 0),
      row('SR wind 0–1 km', f(P.srw1), 'kt', 0),
      row('SR wind 4–6 km', f(P.srw46), 'kt', lvl(P.srw46, [15, 25, 35, 45])),
      row('SR wind 9–11 km', f(P.srw911), 'kt', 0)
    ]));

    out.push(group('Composites', [
      row('STP (fixed layer)', f(P.stpFixed, 1), '', lvl(P.stpFixed, [0.5, 1, 3, 6])),
      row('STP (effective)', f(P.stpEff, 1), '', lvl(P.stpEff, [0.5, 1, 3, 6])),
      row('Supercell composite', f(P.scp, 1), '', lvl(P.scp, [1, 4, 10, 20])),
      row('Sig. hail (SHIP)', f(P.ship, 1), '', lvl(P.ship, [0.5, 1, 2, 3])),
      row('EHI 0–1 km', f(P.ehi1, 1), '', lvl(P.ehi1, [0.5, 1, 2, 3])),
      row('EHI 0–3 km', f(P.ehi3, 1), '', lvl(P.ehi3, [0.5, 1, 2, 3])),
      row('Derecho composite', f(P.dcp, 1), '', lvl(P.dcp, [0.5, 1, 2, 4]))
    ]));

    $('#page-params').innerHTML = out.join('');
  }

  /* The always-visible vitals strip: the handful of numbers you actually watch
     while dragging, so they never scroll out of the panel. */
  function renderHodoFoot() {
    var P = state.params;
    function pad3(x) { return ('00' + Math.round(x) % 360).slice(-3); }
    function ds(v) { var d = T.dirspd(v[0], v[1]); return pad3(d[0]) + '/' + f(d[1]); }
    function cell(k, v, l) {
      return '<div class="vit"><span class="vk">' + k + '</span>' +
             '<span class="vv l' + (l || 0) + '">' + v + '</span></div>';
    }
    var ml = P.ml;
    $('#hodoFoot').innerHTML =
      cell('MLCAPE', f(ml.cape), lvl(ml.cape, CAPE_T)) +
      cell('MLCIN', f(ml.cin), lvl(-ml.cin, [25, 75, 150, 250])) +
      cell('MLLCL', f(ml.lclz), lvl(-ml.lclz, [-1500, -1200, -900, -600])) +
      cell('SHR 0–6', f(P.shr6), lvl(P.shr6, SHEAR_T)) +
      cell('SRH 0–1', f(P.srh1), lvl(Math.abs(P.srh1), SRH_T)) +
      cell('ESRH', f(P.esrh), lvl(Math.abs(P.esrh), SRH_T)) +
      cell('PWAT', f(P.pwat, 2), lvl(P.pwat, [0.75, 1.25, 1.75, 2.1])) +
      cell('DCAPE', f(P.dcape), lvl(P.dcape, [500, 800, 1100, 1400])) +
      cell('STP eff', f(P.stpEff, 1), lvl(P.stpEff, [0.5, 1, 3, 6])) +
      cell('SCP', f(P.scp, 1), lvl(P.scp, [1, 4, 10, 20]));
    void ds;                                   // storm motions are labelled on the hodograph
  }

  /* ---------------- level table ---------------- */
  function renderLevels() {
    var L = state.prof.levels, prof = state.prof;
    $('#lvlCount').textContent = L.length + ' levels';
    // Building a few hundred rows of inputs is not free; skip it while the
    // Levels tab is hidden and catch up when it is opened.
    if (!$('#page-levels').classList.contains('active')) {
      state.levelsStale = true;
      $('#elev').value = Math.round(prof.meta.elev);
      return;
    }
    state.levelsStale = false;
    var rows = L.map(function (l, i) {
      var agl = l.z - prof.zsfc;
      return '<tr data-i="' + i + '"' + (i === state.selected ? ' class="sel"' : '') + '>' +
        '<td><input class="num" data-k="p" data-i="' + i + '" value="' + l.p.toFixed(1) + '"></td>' +
        '<td class="z">' + agl.toFixed(0) + '</td>' +
        '<td class="ct"><input class="num" data-k="t" data-i="' + i + '" value="' + l.t.toFixed(1) + '"></td>' +
        '<td class="cd"><input class="num" data-k="td" data-i="' + i + '" value="' + l.td.toFixed(1) + '"></td>' +
        '<td><input class="num" data-k="dir" data-i="' + i + '" value="' + l.dir.toFixed(0) + '"></td>' +
        '<td><input class="num" data-k="spd" data-i="' + i + '" value="' + l.spd.toFixed(0) + '"></td>' +
        '</tr>';
    });
    $('#lvlTable tbody').innerHTML = rows.join('');
    $('#elev').value = Math.round(prof.meta.elev);
  }

  $('#lvlTable').addEventListener('change', function (e) {
    var el = e.target;
    if (!el.classList.contains('num')) return;
    var i = +el.dataset.i, k = el.dataset.k, v = parseFloat(el.value);
    if (!isFinite(v)) { renderLevels(); return; }
    snapshot();
    var l = state.prof.levels[i];
    if (k === 'p') { l.p = Math.min(1100, Math.max(50, v)); }
    else if (k === 't') { l.t = v; if (l.td > l.t) l.td = l.t; }
    else if (k === 'td') { l.td = Math.min(v, l.t); }
    else if (k === 'dir') { l.dir = ((v % 360) + 360) % 360; }
    else if (k === 'spd') { l.spd = Math.max(0, v); }
    state.prof.levels.sort(function (a, b) { return b.p - a.p; });
    snapshot();
    rebuild();
  });
  $('#lvlTable').addEventListener('focusin', function (e) {
    if (e.target.dataset && e.target.dataset.i !== undefined) {
      state.selected = +e.target.dataset.i;
      skewt.draw(); hodo.draw();
      var tr = $('#lvlTable').querySelector('tr.sel');
      if (tr) tr.classList.remove('sel');
      var row2 = e.target.closest('tr');
      if (row2) row2.classList.add('sel');
    }
  });
  $('#elev').addEventListener('change', function () {
    var v = parseFloat(this.value);
    if (!isFinite(v)) return;
    snapshot();
    state.prof.meta.elev = v;
    snapshot();
    rebuild();
  });
  /* ---------------- resolution / smoothing ---------------- */
  var RESOLUTIONS = {
    raob:   { kind: 'raob' },
    bufr25: { kind: 'bufr', dp: 25 },
    bufr10: { kind: 'bufr', dp: 10 },
    bufr5:  { kind: 'bufr', dp: 5 },
    bufr2:  { kind: 'bufr', dp: 2 },
    bufr1:  { kind: 'bufr', dp: 1 }
  };

  $('#resolution').addEventListener('change', function () {
    var spec = RESOLUTIONS[this.value];
    if (!spec) return;
    snapshot();
    state.prof.levels = Profile.resample(state.prof.levels, spec);
    state.selected = Math.min(state.selected, state.prof.levels.length - 1);
    snapshot();
    rebuild();
  });

  $('#smoothT').addEventListener('click', function () {
    snapshot();
    state.prof.levels = Profile.smooth(state.prof.levels);
    snapshot();
    rebuild();
  });
  $('#smoothW').addEventListener('click', function () {
    snapshot();
    state.prof.levels = Profile.smoothWind(state.prof.levels);
    snapshot();
    rebuild();
  });

  $('#addLvl').addEventListener('click', function () {
    var L = state.prof.levels, i = state.selected;
    var p = i < L.length - 1 ? (L[i].p + L[i + 1].p) / 2 : L[i].p - 25;
    if (p <= 20) return;
    addLevelAt(p);
  });
  $('#delLvl').addEventListener('click', function () { removeLevel(state.selected); });

  /* ---------------- controls ---------------- */
  var sel = $('#preset');
  Samples.list.forEach(function (s, i) {
    var o = document.createElement('option');
    o.value = String(i); o.textContent = s.name;
    sel.appendChild(o);
  });
  sel.addEventListener('change', function () { load(Samples.list[+this.value]); });

  $('#parcel').addEventListener('change', function () {
    state.opts.parcel = this.value; render();
  });
  $('#storm').addEventListener('change', function () {
    state.opts.stormMode = this.value; rebuild();
  });
  $('#brush').addEventListener('input', function () {
    state.opts.brush = +this.value;
    $('#brushVal').textContent = this.value === '0' ? 'off' : (+this.value / 1000).toFixed(1) + ' km';
    skewt.opts.brush = state.opts.brush;
    hodo.opts.brush = state.opts.brush;
  });
  $('#showWb').addEventListener('change', function () {
    skewt.opts.showWetbulb = this.checked; skewt.draw();
  });
  $('#showTv').addEventListener('change', function () {
    skewt.opts.showVirtual = this.checked; skewt.draw();
  });
  $('#undo').addEventListener('click', function () { restore(state.hidx - 1); });
  $('#redo').addEventListener('click', function () { restore(state.hidx + 1); });

  document.addEventListener('keydown', function (e) {
    if (/input|textarea|select/i.test(e.target.tagName)) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) restore(state.hidx + 1); else restore(state.hidx - 1);
    }
  });

  document.querySelectorAll('.tab').forEach(function (t) {
    t.addEventListener('click', function () {
      document.querySelectorAll('.tab').forEach(function (x) { x.classList.remove('active'); });
      document.querySelectorAll('.tabpage').forEach(function (x) { x.classList.remove('active'); });
      t.classList.add('active');
      $('#page-' + t.dataset.tab).classList.add('active');
      if (t.dataset.tab === 'levels' && state.levelsStale) renderLevels();
    });
  });

  /* ---------------- modal ---------------- */
  var modal = $('#modal');
  function openModal(title, tools, body, foot) {
    $('#modalTitle').textContent = title;
    $('#modalTools').innerHTML = tools;
    $('#modalText').value = body;
    $('#modalFoot').innerHTML = foot;
    modal.hidden = false;
  }
  $('#modalClose').addEventListener('click', function () { modal.hidden = true; });
  modal.addEventListener('click', function (e) { if (e.target === modal) modal.hidden = true; });

  /* Saving a file: the hosted build goes through the viewer's downloads
     capability (which asks the viewer first); a standalone copy uses a blob. */
  var downloadsApi = null;
  (function probeDownloads() {
    if (!window.claude || typeof window.claude.use !== 'function') return;
    Promise.resolve(window.claude.use('downloads'))
      .then(function (api) { downloadsApi = api; }, function () { downloadsApi = null; });
  })();
  function canSaveFile() { return !!downloadsApi || window.self === window.top; }
  function saveFile(filename, text) {
    if (downloadsApi) {
      downloadsApi.save({ filename: filename, data: text }).then(function () {
        msg('Saved ' + filename);
      }, function (e) {
        msg(e && e.code === 'declined' ? 'Save cancelled.'
          : 'Could not save: ' + ((e && e.message) || e), e && e.code !== 'declined');
      });
      return;
    }
    var blob = new Blob([text], { type: 'text/plain' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
    msg('Saved ' + filename);
  }
  function exportFilename() {
    return (state.prof.meta.station || 'sounding') + '_' +
           (state.prof.meta.time || '').replace(/[^0-9]/g, '') +
           (exportFormat() === 'bufkit' ? '.buf' : '.txt');
  }

  function exportFormat() {
    var sel = $('#optFormat');
    return sel ? sel.value : 'spc';
  }
  function exportText() {
    if (exportFormat() === 'bufkit') {
      return Bufkit.toText(state.prof, state.params, {});
    }
    var withSummary = $('#optSummary') && $('#optSummary').checked;
    var txt = Sharppy.toText(state.prof, {});
    if (withSummary) txt += '\n\n' + Sharppy.summaryText(state.prof, state.params);
    return txt;
  }
  function refreshExport() { $('#modalText').value = exportText(); }

  /* A textarea hands back its value with newlines normalised to LF, which
     would quietly strip the CRLFs BUFKIT needs - SHARPpy splits members on
     '\r\n\r\n\r\n' and would see one unterminated member. Take whatever is in
     the box (so hand edits survive) and put the line endings back. */
  function finalText() {
    var txt = $('#modalText').value;
    if (exportFormat() === 'bufkit') {
      txt = txt.replace(/\r\n/g, '\n').replace(/\n/g, '\r\n');
    }
    return txt;
  }

  $('#exportBtn').addEventListener('click', function () {
    var m = state.prof.meta;
    openModal('Export sounding',
      '<label>Format <select id="optFormat">' +
      '<option value="spc">SHARPpy / SPC text (.txt)</option>' +
      '<option value="bufkit">BUFKIT (.buf)</option>' +
      '</select></label>' +
      '<label id="optSummaryWrap"><input type="checkbox" id="optSummary"> append parameter summary as # comments</label>' +
      '<label>Station <input type="text" id="optStation" class="num" value="' + m.station + '" size="6"></label>' +
      '<label>Time <input type="text" id="optTime" class="num" value="' + m.time + '" size="12"></label>',
      '',
      '<span class="msg" id="modalMsg"></span>' +
      (canSaveFile() ? '<button class="btn" id="dlBtn">Save .txt</button>' : '') +
      '<button class="btn primary" id="copyBtn">Copy to clipboard</button>');
    refreshExport();
    $('#optSummary').addEventListener('change', refreshExport);
    $('#optFormat').addEventListener('change', function () {
      // the # comment block is an SPC-text convenience; BUFKIT has no room for it
      $('#optSummaryWrap').style.display = this.value === 'bufkit' ? 'none' : '';
      refreshExport();
    });
    $('#optStation').addEventListener('input', function () {
      state.prof.meta.station = this.value.trim() || 'EDIT'; refreshExport();
    });
    $('#optTime').addEventListener('input', function () {
      state.prof.meta.time = this.value.trim() || '000000/0000'; refreshExport();
    });
    $('#copyBtn').addEventListener('click', function () {
      var ta = $('#modalText');
      navigator.clipboard.writeText(finalText()).then(function () {
        msg('Copied ' + ta.value.split('\n').length + ' lines to the clipboard.');
      }, function () {
        ta.select(); document.execCommand('copy');
        msg('Copied.');
      });
    });
    if ($('#dlBtn')) $('#dlBtn').addEventListener('click', function () {
      saveFile(exportFilename(), finalText());
    });
  });

  $('#importBtn').addEventListener('click', function () {
    openModal('Import sounding',
      '<label class="filebtn">Choose file…<input type="file" id="optFile" ' +
      'accept=".txt,.buf,.bufr,.dat,.snd,text/plain"></label>' +
      '<span>…or paste below. SPC text and BUFKIT are detected automatically.</span>' +
      '<span id="bufSteps"></span>',
      '',
      '<span class="msg" id="modalMsg"></span>' +
      '<button class="btn primary" id="loadBtn">Load sounding</button>');
    $('#modalText').placeholder =
      '%TITLE%\n OUN   990503/2350\n\n%RAW%\n' +
      '1000.00,  357.00,   28.00,   22.50,  160.00,   20.00\n...\n%END%\n\n' +
      '…or a BUFKIT file (STID = … / SNPARM = … / PRES TMPC TMWC …)';

    $('#optFile').addEventListener('change', function () {
      var file = this.files && this.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function () {
        $('#modalText').value = String(reader.result);
        msg('Read ' + file.name + ' (' + Math.round(file.size / 1024) + ' kB) — press Load.');
      };
      reader.onerror = function () { msg('Could not read that file.', true); };
      reader.readAsText(file);
    });

    $('#loadBtn').addEventListener('click', function () { doImport(); });
  });

  /* BUFKIT files hold one sounding per forecast hour, so offer the list. */
  function doImport(stepIdx) {
    var text = $('#modalText').value;
    try {
      if (Bufkit.looksLikeBufkit(text)) {
        var b = Bufkit.fromText(text);
        var idx = stepIdx || 0;
        var step = b.times[idx];
        load({ levels: step.levels,
               meta: { station: b.meta.station, time: step.time, elev: step.elev,
                       lat: b.meta.lat, lon: b.meta.lon } });
        $('#resolution').value = '';            // imported spacing is its own thing
        if (b.times.length > 1) {
          renderStepPicker(b, idx);
          msg('Loaded ' + step.time + ' (' + step.levels.length + ' levels) — ' +
              b.times.length + ' forecast hours in this file.');
        } else {
          modal.hidden = true;
        }
      } else {
        var r = Sharppy.fromText(text);
        load({ levels: r.levels, meta: r.meta });
        $('#resolution').value = '';
        modal.hidden = true;
      }
    } catch (err) {
      msg(err.message, true);
    }
  }

  function renderStepPicker(b, idx) {
    var opts = b.times.map(function (t, i) {
      return '<option value="' + i + '"' + (i === idx ? ' selected' : '') + '>' +
             t.time + ' (' + t.levels.length + ')</option>';
    }).join('');
    $('#bufSteps').innerHTML = '<label>Forecast hour <select id="bufStep">' + opts + '</select></label>';
    $('#bufStep').addEventListener('change', function () { doImport(+this.value); });
  }

  function msg(text, isErr) {
    var el = $('#modalMsg');
    if (!el) return;
    el.textContent = text;
    el.className = 'msg' + (isErr ? ' err' : '');
    clearTimeout(msg._t);
    msg._t = setTimeout(function () { el.textContent = ''; }, 4000);
  }

  /* ---------------- boot ---------------- */
  var roPending = 0;
  function relayout() {                       // timers, not rAF: a hidden tab still lays out
    if (roPending) return;
    roPending = setTimeout(function () {
      roPending = 0; skewt.resize(); hodo.resize();
    }, 16);
  }
  var ro = new ResizeObserver(relayout);
  [$('#skewt'), $('#hodo'), document.querySelector('.skewt-pane'),
   document.querySelector('.side'), document.querySelector('main')]
    .forEach(function (el) { if (el) ro.observe(el); });
  window.addEventListener('resize', relayout);

  load(Samples.list[0]);
  skewt.resize(); hodo.resize();

  // handle for the console: SkewTLab.state.prof, .rebuild(), .load(...)
  window.SkewTLab = {
    state: state, skewt: skewt, hodo: hodo,
    rebuild: rebuild, load: load, relayout: relayout,
    exportText: function () { return Sharppy.toText(state.prof, {}); }
  };
})();
