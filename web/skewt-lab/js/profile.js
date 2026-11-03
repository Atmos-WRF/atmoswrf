/* profile.js - a sounding: levels, hydrostatic heights, a fine analysis grid,
   and parcel lifting (CAPE / CIN / LCL / LFC / EL). */
var Profile = (function () {
  var T = Thermo;

  function clone(levels) {
    return levels.map(function (l) {
      return { p: l.p, t: l.t, td: l.td, dir: l.dir, spd: l.spd };
    });
  }

  function Sounding(levels, meta) {
    this.levels = clone(levels).sort(function (a, b) { return b.p - a.p; });
    this.meta = Object.assign({ station: 'EDIT', time: '000000/0000', elev: 0,
                                lat: 35.18, lon: -97.44 }, meta || {});
    this.build();
  }

  /* Build the analysis grid: every level pressure plus a 5 hPa filler, then
     hydrostatic heights integrated with virtual temperature. */
  Sounding.prototype.build = function () {
    var L = this.levels, n = L.length, i;
    var psfc = L[0].p, ptop = L[n - 1].p;
    var set = {};
    for (i = 0; i < n; i++) set[L[i].p.toFixed(4)] = L[i].p;
    for (var p = Math.floor(psfc / 5) * 5; p >= ptop; p -= 5) {
      if (p < ptop || p > psfc) continue;
      set[p.toFixed(4)] = p;
    }
    var ps = Object.keys(set).map(function (k) { return set[k]; })
                             .sort(function (a, b) { return b - a; });

    var g = { p: ps, t: [], td: [], u: [], v: [], z: [], tv: [], w: [] };
    for (i = 0; i < ps.length; i++) {
      var s = sample(L, ps[i]);
      g.t.push(s.t); g.td.push(s.td); g.u.push(s.u); g.v.push(s.v);
      var w = T.mixratio(ps[i], s.td);
      g.w.push(w);
      g.tv.push(T.virtualTemp(s.t, w));
    }
    g.z[0] = this.meta.elev;
    for (i = 1; i < ps.length; i++) {
      var tvbar = 0.5 * (g.tv[i] + g.tv[i - 1]) + T.ZK;
      g.z[i] = g.z[i - 1] + (T.Rd / T.G) * tvbar * Math.log(ps[i - 1] / ps[i]);
    }
    this.grid = g;
    this.psfc = psfc; this.ptop = ptop;
    this.zsfc = g.z[0]; this.ztop = g.z[g.z.length - 1];
    this._parcels = {}; this._eff = null;
    for (i = 0; i < n; i++) L[i].z = this.zAtP(L[i].p);
  };

  // linear-in-log-p interpolation of the level data
  function sample(L, p) {
    var n = L.length;
    if (p >= L[0].p) return lev(L[0]);
    if (p <= L[n - 1].p) return lev(L[n - 1]);
    for (var i = 1; i < n; i++) {
      if (p >= L[i].p) {
        var a = L[i - 1], b = L[i];
        var f = Math.log(p / a.p) / Math.log(b.p / a.p);
        var ua = T.uv(a.dir, a.spd), ub = T.uv(b.dir, b.spd);
        return { t: a.t + f * (b.t - a.t), td: a.td + f * (b.td - a.td),
                 u: ua[0] + f * (ub[0] - ua[0]), v: ua[1] + f * (ub[1] - ua[1]) };
      }
    }
    return lev(L[n - 1]);
  }
  function lev(l) { var c = T.uv(l.dir, l.spd); return { t: l.t, td: l.td, u: c[0], v: c[1] }; }

  /* --- grid lookups --- */
  Sounding.prototype.gidx = function (p) {
    var P = this.grid.p, lo = 0, hi = P.length - 1;
    if (p >= P[0]) return 0;
    if (p <= P[hi]) return hi - 1;
    while (hi - lo > 1) { var m = (lo + hi) >> 1; if (P[m] >= p) lo = m; else hi = m; }
    return lo;
  };
  Sounding.prototype.interpP = function (p, key) {
    var P = this.grid.p, A = this.grid[key];
    if (p >= P[0]) return A[0];
    if (p <= P[P.length - 1]) return A[A.length - 1];
    var i = this.gidx(p);
    var f = Math.log(p / P[i]) / Math.log(P[i + 1] / P[i]);
    return A[i] + f * (A[i + 1] - A[i]);
  };
  Sounding.prototype.zAtP = function (p) { return this.interpP(p, 'z'); };
  Sounding.prototype.pAtZ = function (z) {
    var Z = this.grid.z, P = this.grid.p;
    if (z <= Z[0]) return P[0];
    if (z >= Z[Z.length - 1]) return P[P.length - 1];
    var lo = 0, hi = Z.length - 1;
    while (hi - lo > 1) { var m = (lo + hi) >> 1; if (Z[m] <= z) lo = m; else hi = m; }
    var f = (z - Z[lo]) / (Z[hi] - Z[lo]);
    return Math.exp(Math.log(P[lo]) + f * (Math.log(P[hi]) - Math.log(P[lo])));
  };
  Sounding.prototype.pAtAgl = function (agl) { return this.pAtZ(this.zsfc + agl); };
  Sounding.prototype.windAtAgl = function (agl) {
    var p = this.pAtAgl(agl);
    return [this.interpP(p, 'u'), this.interpP(p, 'v')];
  };

  /* ---------------- parcel lifting ---------------- */
  function liftParcel(prof, p0, t0, td0, label, quick) {
    var g = prof.grid, i;
    if (td0 > t0) td0 = t0;
    var w0 = T.mixratio(p0, td0);
    var th0 = T.thetaK(p0, t0);
    var plcl = T.lclPres(p0, t0, td0);
    var tlcl = T.lclTempK(t0, td0) - T.ZK;
    var the = T.thetae(p0, t0, td0);

    var ps = [p0];
    if (plcl < p0 && plcl > prof.ptop) ps.push(plcl);
    for (i = 0; i < g.p.length; i++) if (g.p[i] < p0 - 1e-6) ps.push(g.p[i]);
    ps.sort(function (a, b) { return b - a; });

    var tp = [], tvp = [], B = [], guess = t0;
    for (i = 0; i < ps.length; i++) {
      var p = ps[i], tParcel, wParcel;
      if (p >= plcl) { tParcel = T.tempFromThetaK(th0, p); wParcel = w0; }
      else {
        tParcel = T.tempAtSatThetae(the, p, guess);
        guess = tParcel;
        wParcel = T.mixratio(p, tParcel);
      }
      tp.push(tParcel);
      tvp.push(T.virtualTemp(tParcel, wParcel));
      B.push(tvp[i] - prof.interpP(p, 'tv'));
    }

    var segs = [];
    for (i = 0; i < ps.length - 1; i++) {
      var pb = ps[i], pt = ps[i + 1], bb = B[i], bt = B[i + 1];
      if (pt >= pb) continue;
      if (bb * bt < 0) {
        var f = bb / (bb - bt);
        var pc = Math.exp(Math.log(pb) + f * (Math.log(pt) - Math.log(pb)));
        segs.push({ pbot: pb, ptop: pc, area: T.Rd * 0.5 * bb * Math.log(pb / pc) });
        segs.push({ pbot: pc, ptop: pt, area: T.Rd * 0.5 * bt * Math.log(pc / pt) });
      } else {
        segs.push({ pbot: pb, ptop: pt, area: T.Rd * 0.5 * (bb + bt) * Math.log(pb / pt) });
      }
    }

    var lfcp = null, elp = null, cape = 0, cin = 0;
    for (i = 0; i < segs.length; i++) {
      if (segs[i].area > 0 && segs[i].pbot <= plcl + 1e-6) { lfcp = segs[i].pbot; break; }
    }
    if (lfcp !== null) {
      for (i = segs.length - 1; i >= 0; i--) {
        if (segs[i].area > 0 && segs[i].ptop < lfcp) { elp = segs[i].ptop; break; }
      }
      for (i = 0; i < segs.length; i++) {
        if (segs[i].pbot <= lfcp && segs[i].area > 0) cape += segs[i].area;
        if (segs[i].ptop >= lfcp - 1e-9 && segs[i].area < 0) cin += segs[i].area;
      }
    }
    // With no LFC the parcel never becomes buoyant, so there is no meaningful
    // inhibition to quote - report 0 rather than the whole column's negative area.

    function capeBetween(pbot, ptp) {
      var s = 0;
      for (var k = 0; k < segs.length; k++) {
        var sg = segs[k];
        if (sg.area <= 0) continue;
        var b = Math.min(sg.pbot, pbot), t2 = Math.max(sg.ptop, ptp);
        if (b <= t2) continue;
        s += sg.area * (Math.log(b / t2) / Math.log(sg.pbot / sg.ptop));
      }
      return s;
    }

    var res = {
      label: label || '', p0: p0, t0: t0, td0: td0, w0: w0, thetae: the,
      z0: prof.zAtP(p0) - prof.zsfc,
      lclp: plcl, lclt: tlcl, lclz: prof.zAtP(plcl) - prof.zsfc,
      lfcp: lfcp, lfcz: lfcp ? prof.zAtP(lfcp) - prof.zsfc : null,
      elp: elp, elz: elp ? prof.zAtP(elp) - prof.zsfc : null,
      elt: elp ? prof.interpP(elp, 't') : null,
      cape: cape, cin: cin, ps: ps, tp: tp, B: B, segs: segs,
      capeBetween: capeBetween
    };
    if (quick) return res;
    res.li = (prof.ptop <= 500 && p0 >= 500) ? (prof.interpP(500, 't') - traceT(res, 500)) : null;
    res.cape3 = capeBetween(prof.psfc, prof.pAtAgl(3000));
    res.cape6 = capeBetween(prof.psfc, prof.pAtAgl(6000));
    if (elp) {
      var acc = 0, mpl = null;
      for (i = 0; i < segs.length; i++) {
        if (segs[i].pbot > lfcp) continue;
        acc += segs[i].area;
        if (acc <= 0 && segs[i].ptop < elp) { mpl = segs[i].ptop; break; }
      }
      res.mplp = mpl; res.mplz = mpl ? prof.zAtP(mpl) - prof.zsfc : null;
    }
    return res;
  }

  function traceT(res, p) {
    var ps = res.ps;
    if (p >= ps[0]) return res.tp[0];
    for (var i = 1; i < ps.length; i++) {
      if (p >= ps[i]) {
        var f = Math.log(p / ps[i - 1]) / Math.log(ps[i] / ps[i - 1]);
        return res.tp[i - 1] + f * (res.tp[i] - res.tp[i - 1]);
      }
    }
    return res.tp[res.tp.length - 1];
  }
  Sounding.prototype.traceT = function (res, p) { return traceT(res, p); };

  /* Named parcels, memoised until the next build(). */
  Sounding.prototype.parcel = function (kind) {
    if (this._parcels[kind]) return this._parcels[kind];
    var L = this.levels, g = this.grid, r, i;
    if (kind === 'sfc') {
      r = liftParcel(this, L[0].p, L[0].t, L[0].td, 'Surface');
    } else if (kind === 'ml') {
      var pb = this.psfc, pt = this.psfc - 100, sth = 0, sw = 0, cnt = 0;
      for (i = 0; i < g.p.length; i++) {
        if (g.p[i] > pb || g.p[i] < pt) continue;
        sth += T.thetaK(g.p[i], g.t[i]); sw += g.w[i]; cnt++;
      }
      var t0 = T.tempFromThetaK(sth / cnt, pb), td0 = T.tempFromMixratio(sw / cnt, pb);
      r = liftParcel(this, pb, t0, Math.min(td0, t0), 'Mixed Layer 100mb');
    } else if (kind === 'mu') {
      var best = -1e9, bi = 0;
      for (i = 0; i < g.p.length; i++) {
        if (g.p[i] < this.psfc - 300) break;
        var e = T.thetae(g.p[i], g.t[i], g.td[i]);
        if (e > best) { best = e; bi = i; }
      }
      r = liftParcel(this, g.p[bi], g.t[bi], g.td[bi], 'Most Unstable');
    } else if (kind === 'fcst') {
      r = liftParcel(this, this.psfc, this.maxTemp(), L[0].td, 'Forecast Surface');
    } else {
      var eff = this.effectiveInflow();
      if (eff.botP === null) r = this.parcel('ml');
      else r = liftParcel(this, eff.botP, this.interpP(eff.botP, 't'),
                          this.interpP(eff.botP, 'td'), 'Effective Inflow Base');
    }
    this._parcels[kind] = r;
    return r;
  };

  Sounding.prototype.liftFrom = function (p0, t0, td0, label) {
    return liftParcel(this, p0, t0, td0, label);
  };

  /* Rough forecast max surface temperature: dry adiabat from 100 mb up. */
  Sounding.prototype.maxTemp = function () {
    var p = this.psfc - 100;
    return T.tempFromThetaK(T.thetaK(p, this.interpP(p, 't')), this.psfc);
  };

  /* Effective inflow layer: contiguous levels with CAPE >= 100, CIN >= -250. */
  Sounding.prototype.effectiveInflow = function () {
    if (this._eff) return this._eff;
    var g = this.grid, botP = null, topP = null;
    var limit = Math.max(this.psfc - 400, this.ptop), lastP = Infinity;
    for (var i = 0; i < g.p.length; i++) {
      var p = g.p[i];
      if (p < limit) break;
      // 10 hPa scan to find the base, 5 hPa after, however dense the grid is
      if (lastP - p < (botP === null ? 9.99 : 4.99)) continue;
      lastP = p;
      var r = liftParcel(this, p, g.t[i], g.td[i], '', true);
      var ok = r.cape >= 100 && r.cin >= -250;
      if (ok) { if (botP === null) botP = p; topP = p; }
      else if (botP !== null) break;
    }
    this._eff = {
      botP: botP, topP: topP,
      botZ: botP === null ? null : this.zAtP(botP) - this.zsfc,
      topZ: topP === null ? null : this.zAtP(topP) - this.zsfc,
      depth: botP === null ? 0 : this.zAtP(topP) - this.zAtP(botP)
    };
    return this._eff;
  };

  Sounding.prototype.copyLevels = function () { return clone(this.levels); };

  /* ---------------- resampling ---------------- */
  /* Mandatory + significant pressures a radiosonde would actually report. */
  var RAOB_P = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550,
                500, 450, 400, 350, 300, 250, 200, 150, 100];

  /* Put a profile onto a different set of pressures without changing its
     shape - everything is interpolated the same way the analysis grid is, so
     the curve you see is untouched; only the number of editable points moves.
       spec: {kind:'raob'}  or  {kind:'bufr', dp: 10}
  */
  function resample(levels, spec) {
    var psfc = levels[0].p, ptop = levels[levels.length - 1].p;
    var want = [], i;

    if (spec.kind === 'raob') {
      want.push(psfc);
      for (i = 0; i < RAOB_P.length; i++) {
        if (RAOB_P[i] < psfc - 1 && RAOB_P[i] > ptop + 1) want.push(RAOB_P[i]);
      }
      want.push(ptop);
    } else {
      var dp = spec.dp || 10;
      for (var p = psfc; p > ptop + dp * 0.5; p -= dp) want.push(p);
      want.push(ptop);
    }

    /* Both grids walk straight past thin features - the peak of a cap
       inversion or the top of a moist layer can sit between two samples and
       get clipped, which moves CAPE. Keep every level that is a local
       extremum in T or Td: these are exactly a radiosonde's significant
       levels, and they make the resample close to lossless. */
    for (i = 1; i < levels.length - 1; i++) {
      var a = levels[i - 1], b = levels[i], c = levels[i + 1];
      if ((b.t - a.t) * (c.t - b.t) < 0 || (b.td - a.td) * (c.td - b.td) < 0) {
        want.push(b.p);
      }
    }
    want.sort(function (x, y) { return y - x; });

    var out = [];
    for (i = 0; i < want.length; i++) {
      var q = want[i];
      if (i > 0 && Math.abs(q - out[out.length - 1].p) < Math.min(0.8, (spec.dp || 10) * 0.3)) continue;
      var s = sample(levels, q);
      var ds = T.dirspd(s.u, s.v);
      out.push({ p: q, t: s.t, td: Math.min(s.td, s.t), dir: ds[0], spd: ds[1] });
    }
    return out;
  }

  /* One light 1-2-1 pass in log-pressure over temperature and dewpoint.
     Endpoints are left alone so the surface and the top of the sounding hold. */
  function smooth(levels, fields) {
    fields = fields || ['t', 'td'];
    var n = levels.length, out = clone(levels), i, k;
    for (k = 0; k < fields.length; k++) {
      var f = fields[k];
      for (i = 1; i < n - 1; i++) {
        out[i][f] = 0.25 * levels[i - 1][f] + 0.5 * levels[i][f] + 0.25 * levels[i + 1][f];
      }
    }
    for (i = 0; i < n; i++) if (out[i].td > out[i].t) out[i].td = out[i].t;
    return out;
  }

  /* Winds smooth in u/v, not in direction - averaging degrees wraps badly. */
  function smoothWind(levels) {
    var n = levels.length, uv = levels.map(function (l) { return T.uv(l.dir, l.spd); });
    var out = clone(levels);
    for (var i = 1; i < n - 1; i++) {
      var u = 0.25 * uv[i - 1][0] + 0.5 * uv[i][0] + 0.25 * uv[i + 1][0];
      var v = 0.25 * uv[i - 1][1] + 0.5 * uv[i][1] + 0.25 * uv[i + 1][1];
      var ds = T.dirspd(u, v);
      out[i].dir = ds[0]; out[i].spd = ds[1];
    }
    return out;
  }

  return { Sounding: Sounding, sample: sample, resample: resample,
           smooth: smooth, smoothWind: smoothWind, RAOB_P: RAOB_P };
})();
