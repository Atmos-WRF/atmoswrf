/* sharppy.js - read and write the SPC / SHARPpy plain-text sounding format.

   %TITLE%
    OUN   990503/2350

      LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD
   -------------------------------------------------------------------
   %RAW%
    1000.00,    111.00,     26.20,     22.20,     170.00,     15.00
   %END%
*/
var Sharppy = (function () {
  var MISSING = -9999.00;

  function pad(v, w) {
    var s = (v === null || v === undefined || !isFinite(v)) ? MISSING.toFixed(2) : v.toFixed(2);
    while (s.length < w) s = ' ' + s;
    return s;
  }

  function toText(prof, opts) {
    opts = opts || {};
    var m = prof.meta, L = prof.levels;
    var lines = [];
    lines.push('%TITLE%');
    lines.push(' ' + (m.station || 'EDIT') + '   ' + (m.time || '000000/0000'));
    lines.push('');
    lines.push('   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD');
    lines.push('-------------------------------------------------------------------');
    lines.push('%RAW%');
    for (var i = 0; i < L.length; i++) {
      var l = L[i];
      lines.push([pad(l.p, 8), pad(l.z, 10), pad(l.t, 10), pad(l.td, 10),
                  pad(l.dir, 10), pad(l.spd, 10)].join(','));
    }
    lines.push('%END%');
    if (opts.comment) {
      lines.push('');
      opts.comment.split('\n').forEach(function (c) { lines.push('# ' + c); });
    }
    return lines.join('\n');
  }

  /* Parse the same format back.  Tolerant of whitespace-separated files and of
     a missing %RAW% block (falls back to any 6-number rows it can find). */
  function fromText(text) {
    var raw = text.replace(/\r/g, '');
    var meta = { station: 'EDIT', time: '000000/0000', elev: 0 };

    var tm = raw.match(/%TITLE%\s*\n\s*(\S+)\s+(\S+)/);
    if (tm) { meta.station = tm[1]; meta.time = tm[2]; }

    var body = raw;
    var start = raw.indexOf('%RAW%');
    if (start >= 0) {
      var end = raw.indexOf('%END%', start);
      body = raw.slice(start + 5, end < 0 ? raw.length : end);
    }

    var levels = [], firstZ = null;
    body.split('\n').forEach(function (line) {
      line = line.trim();
      if (!line || line[0] === '%' || line[0] === '#' || line[0] === '-') return;
      var parts = line.split(/[,\s]+/).map(parseFloat).filter(function (x) { return !isNaN(x); });
      if (parts.length < 6) return;
      var p = parts[0], z = parts[1], t = parts[2], td = parts[3], d = parts[4], s = parts[5];
      if (p <= 0 || p > 1100) return;
      if (t <= MISSING + 1) return;
      if (td <= MISSING + 1) td = t - 30;
      if (d <= MISSING + 1) d = 0;
      if (s <= MISSING + 1) s = 0;
      if (firstZ === null && z > MISSING + 1) firstZ = z;
      levels.push({ p: p, t: t, td: Math.min(td, t), dir: (d % 360 + 360) % 360, spd: Math.abs(s) });
    });
    if (firstZ !== null) meta.elev = firstZ;
    if (levels.length < 3) throw new Error('Could not find at least 3 usable levels in that text.');
    levels.sort(function (a, b) { return b.p - a.p; });
    // drop duplicate pressures
    var out = [levels[0]];
    for (var i = 1; i < levels.length; i++) {
      if (Math.abs(levels[i].p - out[out.length - 1].p) > 0.05) out.push(levels[i]);
    }
    return { levels: out, meta: meta };
  }

  /* A SHARPpy-style text summary of the derived parameters, handy for pasting
     into a discussion alongside the raw data. */
  function summaryText(prof, P) {
    var f = function (x, n) {
      return (x === null || x === undefined || !isFinite(x)) ? 'M'
        : x.toFixed(n === undefined ? 0 : n);
    };
    var row = function (a, b) { return ('  ' + a).padEnd(26) + b; };
    var L = [];
    L.push('#  SHARPpy-style summary  (' + prof.meta.station + ' ' + prof.meta.time + ')');
    L.push('#  --- THERMODYNAMICS ---');
    [['sfc', 'SFC'], ['ml', 'ML'], ['mu', 'MU']].forEach(function (k) {
      var p = P[k[0]];
      L.push('#' + row(k[1] + ' CAPE/CIN', f(p.cape) + ' / ' + f(p.cin) + ' J/kg' +
        '   LCL ' + f(p.lclz) + ' m   LFC ' + f(p.lfcz) + ' m   EL ' + f(p.elz) +
        ' m   LI ' + f(p.li, 1)));
    });
    L.push('#' + row('PWAT', f(P.pwat, 2) + ' in'));
    L.push('#' + row('LR 0-3 / 700-500', f(P.lr03, 1) + ' / ' + f(P.lr75, 1) + ' C/km'));
    L.push('#' + row('DCAPE', f(P.dcape) + ' J/kg'));
    L.push('#' + row('FZL / WBZ', f(P.fzl) + ' / ' + f(P.wbz) + ' m'));
    L.push('#  --- KINEMATICS ---');
    L.push('#' + row('Shear 0-1/0-3/0-6', f(P.shr1) + ' / ' + f(P.shr3) + ' / ' + f(P.shr6) + ' kt'));
    L.push('#' + row('SRH 0-1 / 0-3 (RM)', f(P.srh1rm) + ' / ' + f(P.srh3rm) + ' m2/s2'));
    L.push('#' + row('Eff inflow', (P.eff.botP === null ? 'none'
      : f(P.eff.botZ) + '-' + f(P.eff.topZ) + ' m') + '   ESRH ' + f(P.esrh) +
      '   EBWD ' + f(P.ebwd) + ' kt'));
    var ds = function (v) { var d = Thermo.dirspd(v[0], v[1]); return f(d[0]) + '/' + f(d[1]) + 'kt'; };
    L.push('#' + row('Bunkers RM / LM', ds(P.bunkers.rm) + '   ' + ds(P.bunkers.lm)));
    L.push('#  --- COMPOSITES ---');
    L.push('#' + row('STP (fix / eff)', f(P.stpFixed, 1) + ' / ' + f(P.stpEff, 1)));
    L.push('#' + row('SCP / SHIP', f(P.scp, 1) + ' / ' + f(P.ship, 1)));
    L.push('#' + row('EHI 0-1 / 0-3', f(P.ehi1, 1) + ' / ' + f(P.ehi3, 1)));
    return L.join('\n');
  }

  return { toText: toText, fromText: fromText, summaryText: summaryText, MISSING: MISSING };
})();
