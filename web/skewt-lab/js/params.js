/* params.js - derived sounding parameters: kinematics, thermodynamics,
   and the usual SPC composite indices. Winds are knots unless noted. */
var Params = (function () {
  var T = Thermo;
  var KT = T.KT2MS;

  /* ---------- wind helpers ---------- */
  function meanWind(prof, pbot, ptop) {       // mass-weighted, knots
    var g = prof.grid, su = 0, sv = 0, sw = 0;
    for (var i = 0; i < g.p.length - 1; i++) {
      var pa = g.p[i], pb = g.p[i + 1];
      if (pa > pbot) { if (pb > pbot) continue; pa = pbot; }
      if (pb < ptop) { if (pa < ptop) continue; pb = ptop; }
      var dp = pa - pb; if (dp <= 0) continue;
      su += dp * 0.5 * (prof.interpP(pa, 'u') + prof.interpP(pb, 'u'));
      sv += dp * 0.5 * (prof.interpP(pa, 'v') + prof.interpP(pb, 'v'));
      sw += dp;
    }
    if (sw === 0) return [prof.interpP(pbot, 'u'), prof.interpP(pbot, 'v')];
    return [su / sw, sv / sw];
  }
  function meanWindAgl(prof, a, b) {
    return meanWind(prof, prof.pAtAgl(a), prof.pAtAgl(b));
  }
  function shearVec(prof, aglBot, aglTop) {   // knots
    var a = prof.windAtAgl(aglBot), b = prof.windAtAgl(aglTop);
    return [b[0] - a[0], b[1] - a[1]];
  }
  function mag(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1]); }

  function layerPts(prof, aglBot, aglTop) {   // [z, u, v] rows, m and knots
    var g = prof.grid, zb = prof.zsfc + aglBot, zt = prof.zsfc + aglTop, out = [];
    var pb = prof.pAtZ(zb), pt = prof.pAtZ(zt);
    out.push([zb, prof.interpP(pb, 'u'), prof.interpP(pb, 'v')]);
    for (var i = 0; i < g.p.length; i++) {
      if (g.z[i] > zb + 1 && g.z[i] < zt - 1) out.push([g.z[i], g.u[i], g.v[i]]);
    }
    out.push([zt, prof.interpP(pt, 'u'), prof.interpP(pt, 'v')]);
    return out;
  }
  function helicity(prof, aglBot, aglTop, cu, cv) {   // m2/s2; cu,cv in knots
    var pts = layerPts(prof, aglBot, aglTop), s = 0;
    for (var i = 0; i < pts.length - 1; i++) {
      var u1 = (pts[i][1] - cu) * KT, v1 = (pts[i][2] - cv) * KT;
      var u2 = (pts[i + 1][1] - cu) * KT, v2 = (pts[i + 1][2] - cv) * KT;
      s += (u2 * v1) - (u1 * v2);
    }
    return s;
  }
  function bunkers(prof) {
    var mw = meanWindAgl(prof, 0, 6000);
    var sh = shearVec(prof, 0, 6000);
    var m = mag(sh);
    var D = 7.5 / KT;                         // 7.5 m/s in knots
    if (m < 1e-6) return { rm: mw.slice(), lm: mw.slice(), mean: mw };
    return {
      rm: [mw[0] + D * sh[1] / m, mw[1] - D * sh[0] / m],
      lm: [mw[0] - D * sh[1] / m, mw[1] + D * sh[0] / m],
      mean: mw
    };
  }
  function criticalAngle(prof, cu, cv) {
    var sh = shearVec(prof, 0, 500);
    var sfc = prof.windAtAgl(0);
    var sr = [sfc[0] - cu, sfc[1] - cv];
    if (mag(sh) < 1e-6 || mag(sr) < 1e-6) return null;
    var a = Math.atan2(sh[1], sh[0]) - Math.atan2(sr[1], sr[0]);
    a = Math.abs(a * 180 / Math.PI) % 360;
    if (a > 180) a = 360 - a;
    return a;
  }

  /* ---------- thermodynamic odds and ends ---------- */
  function pwat(prof) {                        // inches
    var g = prof.grid, s = 0;
    for (var i = 0; i < g.p.length - 1; i++) {
      s += 0.5 * (g.w[i] + g.w[i + 1]) * (g.p[i] - g.p[i + 1]);
    }
    return s * 0.010197 / 25.4;
  }
  function crossing(prof, key, value, fromTop) { // height AGL where field crosses value
    var g = prof.grid, i;
    if (fromTop) {
      for (i = g.p.length - 1; i > 0; i--) {
        var a = g[key][i], b = g[key][i - 1];
        if ((a - value) * (b - value) <= 0 && a !== b) {
          var f = (value - a) / (b - a);
          return g.z[i] + f * (g.z[i - 1] - g.z[i]) - prof.zsfc;
        }
      }
    } else {
      for (i = 0; i < g.p.length - 1; i++) {
        var c = g[key][i], d = g[key][i + 1];
        if ((c - value) * (d - value) <= 0 && c !== d) {
          var f2 = (value - c) / (d - c);
          return g.z[i] + f2 * (g.z[i + 1] - g.z[i]) - prof.zsfc;
        }
      }
    }
    return null;
  }
  function lapse(prof, zb, zt) {               // C/km over AGL heights
    var pa = prof.pAtAgl(zb), pb = prof.pAtAgl(zt);
    return (prof.interpP(pa, 't') - prof.interpP(pb, 't')) / (zt - zb) * 1000;
  }
  function lapseP(prof, pb, pt) {
    var dz = prof.zAtP(pt) - prof.zAtP(pb);
    return (prof.interpP(pb, 't') - prof.interpP(pt, 't')) / dz * 1000;
  }

  /* Downdraft CAPE: lowest-theta-e parcel in the bottom 400 mb brought down
     moist-adiabatically from its wet-bulb temperature. */
  function dcape(prof) {
    var g = prof.grid, best = 1e9, bi = 0;
    for (var i = 0; i < g.p.length; i++) {
      if (g.p[i] < prof.psfc - 400) break;
      var e = T.thetae(g.p[i], g.t[i], g.td[i]);
      if (e < best) { best = e; bi = i; }
    }
    var p0 = g.p[bi];
    var the = T.thetae(p0, g.t[bi], g.td[bi]);
    var sum = 0, guess = g.t[bi];
    for (i = bi; i > 0; i--) {
      var pa = g.p[i], pb = g.p[i - 1];
      var ta = T.tempAtSatThetae(the, pa, guess);
      var tb = T.tempAtSatThetae(the, pb, ta); guess = tb;
      var ba = T.virtualTemp(ta, T.mixratio(pa, ta)) - g.tv[i];
      var bb = T.virtualTemp(tb, T.mixratio(pb, tb)) - g.tv[i - 1];
      sum += T.Rd * 0.5 * (ba + bb) * Math.log(pb / pa);
    }
    // sum is the (negative) buoyant work on the way down; DCAPE is its magnitude
    return { dcape: Math.max(0, -sum), srcP: p0, srcZ: prof.zAtP(p0) - prof.zsfc };
  }

  /* Convective temperature and CCL from the lowest-100mb mean mixing ratio. */
  function convectiveTemp(prof) {
    var g = prof.grid, sw = 0, cnt = 0;
    for (var i = 0; i < g.p.length; i++) {
      if (g.p[i] < prof.psfc - 100) break;
      sw += g.w[i]; cnt++;
    }
    var wbar = sw / Math.max(cnt, 1);
    for (i = 0; i < g.p.length; i++) {
      if (T.tempFromMixratio(wbar, g.p[i]) >= g.t[i]) {
        var cclp = g.p[i];
        return { cclp: cclp, cclz: g.z[i] - prof.zsfc,
                 tconv: T.tempFromThetaK(T.thetaK(cclp, g.t[i]), prof.psfc) };
      }
    }
    return { cclp: null, cclz: null, tconv: null };
  }

  function sweat(prof) {
    var td850 = prof.interpP(850, 'td'), t850 = prof.interpP(850, 't');
    var t500 = prof.interpP(500, 't');
    var tt = t850 + td850 - 2 * t500;
    var w8 = T.dirspd(prof.interpP(850, 'u'), prof.interpP(850, 'v'));
    var w5 = T.dirspd(prof.interpP(500, 'u'), prof.interpP(500, 'v'));
    var term1 = td850 > 0 ? 12 * td850 : 0;
    var term2 = tt > 49 ? 20 * (tt - 49) : 0;
    var sh = 0;
    var d8 = w8[0], d5 = w5[0];
    if (d8 >= 130 && d8 <= 250 && d5 >= 210 && d5 <= 310 &&
        (d5 - d8) > 0 && w8[1] >= 15 && w5[1] >= 15) {
      sh = 125 * (Math.sin((d5 - d8) * Math.PI / 180) + 0.2);
    }
    return term1 + term2 + 2 * w8[1] + w5[1] + sh;
  }

  /* ---------- composites ---------- */
  function stp(cape, lclz, srh, shr6kt, cin) {
    var s6 = shr6kt * KT;
    var lclT = lclz < 1000 ? 1 : (lclz > 2000 ? 0 : (2000 - lclz) / 1000);
    var shrT = s6 > 30 ? 1.5 : (s6 < 12.5 ? 0 : s6 / 20);
    var cinT = cin > -50 ? 1 : (cin < -200 ? 0 : (200 + cin) / 150);
    return (cape / 1500) * lclT * (srh / 150) * shrT * cinT;
  }
  function scp(mucape, esrh, ebwdKt) {
    var e = ebwdKt * KT;
    var shrT = e > 20 ? 1 : (e < 10 ? 0 : e / 20);
    return (mucape / 1000) * (esrh / 50) * shrT;
  }
  function ship(mucape, mumr, lr75, t500, shr6kt, fzl) {
    var s6 = Math.min(Math.max(shr6kt * KT, 7), 27);
    var mr = Math.min(Math.max(mumr, 11), 13.6);
    var t5 = Math.min(t500, -5.5);
    var v = (mucape * mr * lr75 * (-t5) * s6) / 42000000;
    if (mucape < 1300) v *= mucape / 1300;
    if (lr75 < 5.8) v *= lr75 / 5.8;
    if (fzl < 2400) v *= fzl / 2400;
    return Math.max(v, 0);
  }

  /* ---------- everything ---------- */
  function compute(prof, opts) {
    opts = opts || {};
    var g = prof.grid, out = {};
    var sfc = prof.parcel('sfc'), ml = prof.parcel('ml'), mu = prof.parcel('mu');
    var eff = prof.effectiveInflow();
    out.sfc = sfc; out.ml = ml; out.mu = mu; out.eff = eff;

    var bk = bunkers(prof);
    out.bunkers = bk;
    var sm;
    if (opts.stormMode === 'lm') sm = bk.lm;
    else if (opts.stormMode === 'mean') sm = bk.mean;
    else if (opts.stormMode === 'user' && opts.stormUser) sm = opts.stormUser.slice();
    else sm = bk.rm;
    out.storm = sm;
    out.stormDS = T.dirspd(sm[0], sm[1]);

    out.shr1 = mag(shearVec(prof, 0, 1000));
    out.shr3 = mag(shearVec(prof, 0, 3000));
    out.shr6 = mag(shearVec(prof, 0, 6000));
    out.shr8 = mag(shearVec(prof, 0, 8000));
    out.shr500 = mag(shearVec(prof, 0, 500));
    out.mw6 = T.dirspd(bk.mean[0], bk.mean[1]);

    out.srh500 = helicity(prof, 0, 500, sm[0], sm[1]);
    out.srh1 = helicity(prof, 0, 1000, sm[0], sm[1]);
    out.srh3 = helicity(prof, 0, 3000, sm[0], sm[1]);
    out.srh1rm = helicity(prof, 0, 1000, bk.rm[0], bk.rm[1]);
    out.srh1lm = helicity(prof, 0, 1000, bk.lm[0], bk.lm[1]);
    out.srh3rm = helicity(prof, 0, 3000, bk.rm[0], bk.rm[1]);
    out.srh3lm = helicity(prof, 0, 3000, bk.lm[0], bk.lm[1]);
    out.critAngle = criticalAngle(prof, sm[0], sm[1]);

    // effective layer kinematics
    if (eff.botP !== null && mu.elz) {
      out.esrh = helicity(prof, eff.botZ, Math.max(eff.topZ, eff.botZ + 1), sm[0], sm[1]);
      var half = eff.botZ + 0.5 * (mu.elz - eff.botZ);
      var a = prof.windAtAgl(eff.botZ), b = prof.windAtAgl(half);
      out.ebwd = mag([b[0] - a[0], b[1] - a[1]]);
    } else { out.esrh = 0; out.ebwd = 0; }

    // storm-relative winds
    function srw(a, b) {
      var m = meanWindAgl(prof, a, b);
      return mag([m[0] - sm[0], m[1] - sm[1]]);
    }
    out.srw1 = srw(0, 1000);
    out.srw2 = srw(0, 2000);
    out.srw46 = srw(4000, 6000);
    out.srw911 = srw(9000, 11000);

    // thermodynamics
    out.pwat = pwat(prof);
    out.fzl = crossing(prof, 't', 0);
    if (!g.wb) {
      g.wb = [];
      for (var wi = 0; wi < g.p.length; wi++) g.wb.push(T.wetbulb(g.p[wi], g.t[wi], g.td[wi]));
    }
    out.wbz = crossing(prof, 'wb', 0);
    out.m10 = crossing(prof, 't', -10);
    out.m20 = crossing(prof, 't', -20);
    out.m30 = crossing(prof, 't', -30);
    out.lr03 = lapse(prof, 0, 3000);
    out.lr36 = lapse(prof, 3000, 6000);
    out.lr75 = lapseP(prof, 700, 500);
    out.lr85 = lapseP(prof, 850, 500);
    out.t500 = prof.interpP(500, 't');
    out.t700 = prof.interpP(700, 't');
    out.t850 = prof.interpP(850, 't');
    out.td850 = prof.interpP(850, 'td');
    out.td700 = prof.interpP(700, 'td');
    out.kindex = (out.t850 - out.t500) + out.td850 - (out.t700 - out.td700);
    out.totaltotals = out.t850 + out.td850 - 2 * out.t500;
    out.sweat = sweat(prof);
    var sh = prof.liftFrom(850, out.t850, out.td850, 'Showalter');
    out.showalter = out.t500 - prof.traceT(sh, 500);
    var dc = dcape(prof);
    out.dcape = dc.dcape; out.dcapeSrc = dc.srcZ;
    var ct = convectiveTemp(prof);
    out.tconv = ct.tconv; out.cclz = ct.cclz;
    out.maxTemp = prof.maxTemp();
    out.sfcRH = T.relh(prof.psfc, g.t[0], g.td[0]);
    out.mlRH = (function () {
      var s = 0, c = 0;
      for (var i = 0; i < g.p.length; i++) {
        if (g.p[i] < prof.psfc - 100) break;
        s += T.relh(g.p[i], g.t[i], g.td[i]); c++;
      }
      return s / Math.max(c, 1);
    })();
    out.mlMixRatio = (function () {
      var s = 0, c = 0;
      for (var i = 0; i < g.p.length; i++) {
        if (g.p[i] < prof.psfc - 100) break;
        s += g.w[i]; c++;
      }
      return s / Math.max(c, 1);
    })();

    // composites
    out.stpFixed = stp(sfc.cape, sfc.lclz, out.srh1rm, out.shr6, sfc.cin);
    out.stpEff = (eff.botP !== null)
      ? stp(ml.cape, ml.lclz, out.esrh, out.ebwd, ml.cin) : 0;
    out.scp = scp(mu.cape, out.esrh, out.ebwd);
    out.ship = ship(mu.cape, mu.w0, out.lr75, out.t500, out.shr6, out.fzl || 0);
    out.ehi1 = sfc.cape * out.srh1rm / 160000;
    out.ehi3 = sfc.cape * out.srh3rm / 160000;
    out.dcp = (out.dcape / 980) * (mu.cape / 2000) * (out.shr6 * KT / 20) *
              (mag(bk.mean) * KT / 16);
    out.kolev = null;
    return out;
  }

  return {
    compute: compute, meanWind: meanWind, meanWindAgl: meanWindAgl,
    shearVec: shearVec, helicity: helicity, bunkers: bunkers, mag: mag,
    layerPts: layerPts, pwat: pwat, dcape: dcape
  };
})();
