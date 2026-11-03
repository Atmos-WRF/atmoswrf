/* bufkit.js - read and write the BUFKIT model-sounding format.

   Layout expected by SHARPpy's sharppy/io/buf_decoder.py:

     <member name>                                  <- data[0]
     SNPARM = PRES;TMPC;TMWC;DWPC;THTE;DRCT;SKNT;OMEG;CFRL;HGHT
     STNPRM = SHOW;LIFT;SWET;KINX;LCLP;PWAT;TOTL;CAPE;LCLT;CINS;EQLV;LFCT;BRCH
     <blank>
     STID = OUN STNM = 723570 TIME = 990503/2350
     SLAT = 35.18 SLON = -97.44 SELV = 357
     STIM = 0
     <blank>
     SHOW = .. LIFT = .. SWET = .. KINX = ..
     ... (station parameter block)
     <blank>
     PRES TMPC TMWC DWPC THTE DRCT SKNT OMEG
     CFRL HGHT
     <pres tmpc tmwc dwpc thte drct sknt omeg>      <- rows come in pairs
     <cfrl hght>
     ...
     STN YYMMDD/HHMM ...                            <- surface block, no blank before it
     <one surface row>
     <CRLF CRLF CRLF>                               <- member terminator

   Three details the decoder depends on, all of which bit us in testing:

   * Lines must be CRLF and the file must end with three of them, because the
     decoder splits members on '\r\n\r\n\r\n' and drops the last piece.
   * The SNPARM line has to be there. The decoder scans for lines containing
     'HGHT' to find data blocks, matches SNPARM first, and then unconditionally
     throws away that first (bogus) block with data_idxs[1:]. Omit SNPARM and
     it eats your first real sounding instead.
   * A blank line goes before each STID but NOT before the STN surface header:
     blocks ending at a STID are cut at i-1, blocks ending at STN are cut at i.
     Get it wrong and the block has an odd line count, which breaks the
     two-lines-per-level pairing.
*/
var Bufkit = (function () {
  var T = Thermo;
  var MISSING = -9999.00;
  var CRLF = '\r\n';

  var SNPARM = 'SNPARM = PRES;TMPC;TMWC;DWPC;THTE;DRCT;SKNT;OMEG;CFRL;HGHT';
  var STNPRM = 'STNPRM = SHOW;LIFT;SWET;KINX;LCLP;PWAT;TOTL;CAPE;LCLT;CINS;EQLV;LFCT;BRCH';
  var SFC_HDR = [
    'STN YYMMDD/HHMM PMSL PRES SKTC STC1 SNFL WTNS',
    'P01M C01M STC2 LCLD MCLD HCLD',
    'SNRA UWND VWND R01M BFGR T2MS',
    'Q2MS WXTS WXTP WXTZ WXTR USTM',
    'VSTM HLCY SLLH WSYM CDBP VSBK',
    'TD2M'
  ];

  function f2(x) {
    return (x === null || x === undefined || !isFinite(x)) ? MISSING.toFixed(2) : x.toFixed(2);
  }

  /* ---------------- write ---------------- */
  function toText(prof, P, opts) {
    opts = opts || {};
    var m = prof.meta, L = prof.levels, i;
    var out = [];

    out.push(opts.member || 'SKEWTLAB');
    out.push(SNPARM);
    out.push(STNPRM);
    out.push('');

    var stnm = (m.wmo || '723570');
    out.push('STID = ' + (m.station || 'EDIT') + ' STNM = ' + stnm +
             ' TIME = ' + (m.time || '000000/0000'));
    out.push('SLAT = ' + (m.lat != null ? m.lat : 35.18).toFixed(2) +
             ' SLON = ' + (m.lon != null ? m.lon : -97.44).toFixed(2) +
             ' SELV = ' + Math.round(m.elev || 0));
    out.push('STIM = 0');
    out.push('');

    // Station parameter block. SHARPpy ignores these, but BUFKIT itself and
    // other readers show them, so fill in what we actually computed.
    var sfc = P ? P.sfc : null;
    out.push('SHOW = ' + f2(P ? P.showalter : null) +
             ' LIFT = ' + f2(sfc ? sfc.li : null) +
             ' SWET = ' + f2(P ? P.sweat : null) +
             ' KINX = ' + f2(P ? P.kindex : null));
    out.push('LCLP = ' + f2(sfc ? sfc.lclp : null) +
             ' PWAT = ' + f2(P ? P.pwat * 25.4 : null) +      // BUFKIT uses mm
             ' TOTL = ' + f2(P ? P.totaltotals : null) +
             ' CAPE = ' + f2(sfc ? sfc.cape : null));
    out.push('LCLT = ' + f2(sfc ? sfc.lclt + T.ZK : null) +   // kelvin
             ' CINS = ' + f2(sfc ? sfc.cin : null) +
             ' EQLV = ' + f2(sfc && sfc.elp ? sfc.elp : null) +
             ' LFCT = ' + f2(sfc && sfc.lfcp ? sfc.lfcp : null));
    out.push('BRCH = ' + f2(null));
    out.push('');

    out.push('PRES TMPC TMWC DWPC THTE DRCT SKNT OMEG');
    out.push('CFRL HGHT');
    for (i = 0; i < L.length; i++) {
      var l = L[i];
      var tw = T.wetbulb(l.p, l.t, l.td);
      var the = T.thetae(l.p, l.t, l.td);
      out.push([f2(l.p), f2(l.t), f2(tw), f2(l.td), f2(the),
                f2(l.dir), f2(l.spd), f2(0)].join(' '));
      out.push([f2(0), f2(l.z)].join(' '));
    }

    // Surface block terminates the last sounding chunk - no blank line first.
    SFC_HDR.forEach(function (h) { out.push(h); });
    var s = L[0];
    var uv = T.uv(s.dir, s.spd);
    out.push([stnm, (m.time || '000000/0000'), f2(s.p), f2(s.p), f2(s.t),
              f2(MISSING), f2(0), f2(MISSING)].join(' '));
    out.push([f2(0), f2(0), f2(MISSING), f2(0), f2(0), f2(0)].join(' '));
    out.push([f2(0), f2(uv[0] * T.KT2MS), f2(uv[1] * T.KT2MS), f2(0), f2(MISSING), f2(s.t)].join(' '));
    out.push([f2(MISSING), f2(0), f2(0), f2(0), f2(0), f2(MISSING)].join(' '));
    out.push([f2(MISSING), f2(P ? P.srh3 : null), f2(MISSING), f2(0), f2(MISSING), f2(MISSING)].join(' '));
    out.push(f2(s.td));

    // Member terminator: the body, then three CRLFs.
    return out.join(CRLF) + CRLF + CRLF + CRLF;
  }

  /* ---------------- read ---------------- */
  /* Returns { times: [{time, levels, meta}], meta } - a BUFKIT file normally
     holds one sounding per forecast hour. */
  function fromText(text) {
    var raw = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    var lines = raw.split('\n');
    var i;

    // Locate every station header; each one begins a sounding.
    var heads = [];
    for (i = 0; i < lines.length; i++) {
      if (lines[i].indexOf('STID') >= 0 && /STID\s*=/.test(lines[i])) heads.push(i);
    }
    if (!heads.length) throw new Error('No STID header found - is this a BUFKIT file?');

    var meta = { station: 'BUFR', time: '000000/0000', elev: 0, lat: 35.18, lon: -97.44 };
    var times = [];

    for (var h = 0; h < heads.length; h++) {
      var start = heads[h];
      var end = (h + 1 < heads.length) ? heads[h + 1] : lines.length;

      var info = parseHead(lines, start);
      if (h === 0) {
        meta.station = info.station || 'BUFR';
        meta.lat = info.lat; meta.lon = info.lon; meta.elev = info.elev;
      }

      // Data rows start after the 'CFRL HGHT' continuation of the column header.
      var dstart = -1;
      for (i = start; i < end; i++) {
        if (/(^|\s)HGHT(\s|$)/.test(lines[i]) && lines[i].indexOf('=') < 0 &&
            lines[i].indexOf(';') < 0) { dstart = i + 1; break; }
      }
      if (dstart < 0) continue;

      var levels = [];
      for (i = dstart; i + 1 < end; i += 2) {
        var a = lines[i].trim().split(/\s+/).map(Number);
        var b = lines[i + 1].trim().split(/\s+/).map(Number);
        if (a.length < 8 || b.length < 1) break;
        if (!isFinite(a[0]) || a[0] <= 0 || a[0] > 1100) break;
        var hght = (b.length === 1) ? b[0] : b[1];
        var t = a[1], td = a[3], dir = a[5], spd = a[6];
        if (t <= MISSING + 1) continue;
        if (td <= MISSING + 1) td = t - 30;
        if (dir <= MISSING + 1) dir = 0;
        if (spd <= MISSING + 1) spd = 0;
        levels.push({ p: a[0], t: t, td: Math.min(td, t),
                      dir: ((dir % 360) + 360) % 360, spd: Math.abs(spd), z: hght });
      }
      if (levels.length < 3) continue;
      levels.sort(function (x, y) { return y.p - x.p; });
      times.push({ time: info.time, levels: levels,
                   elev: isFinite(info.elev) ? info.elev : levels[0].z });
    }

    if (!times.length) throw new Error('Found BUFKIT headers but no usable level data.');
    meta.time = times[0].time;
    meta.elev = times[0].elev;
    return { times: times, meta: meta };
  }

  function parseHead(lines, i) {
    var spl = lines[i].trim().split(/\s+/);
    var station, time;
    if (spl[2] === 'STNM') { station = ''; time = spl[7]; }
    else { station = spl[2]; time = spl[8]; }

    var lat = 35.18, lon = -97.44, elev = 0;
    var ll = lines[i + 1] || '';
    var m = ll.match(/SLAT\s*=\s*(-?[\d.]+)\s+SLON\s*=\s*(-?[\d.]+)\s+SELV\s*=\s*(-?[\d.]+)/);
    if (m) { lat = +m[1]; lon = +m[2]; elev = +m[3]; }
    return { station: station, time: time || '000000/0000', lat: lat, lon: lon, elev: elev };
  }

  function looksLikeBufkit(text) {
    return /STID\s*=/.test(text) && /SNPARM|CFRL|STNPRM/.test(text);
  }

  return { toText: toText, fromText: fromText, looksLikeBufkit: looksLikeBufkit };
})();
