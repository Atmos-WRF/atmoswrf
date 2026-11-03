/* thermo.js - moist thermodynamics for sounding analysis.
   Temperatures in Celsius unless the name says K. Pressure in hPa.
   Mixing ratio in g/kg. Follows Bolton (1980) where practical. */
var Thermo = (function () {
  var Rd = 287.04, Rv = 461.5, Cp = 1005.7, G = 9.80665;
  var EPS = Rd / Rv;          // 0.62197
  var ROCP = Rd / Cp;         // 0.28541
  var ZK = 273.15;

  function esat(t) {                       // saturation vapour pressure, hPa
    return 6.112 * Math.exp(17.67 * t / (t + 243.5));
  }
  function mixratio(p, t) {                // g/kg from temperature (use Td for actual w)
    var e = esat(t);
    if (e >= p * 0.999) e = p * 0.999;
    return 621.97 * e / (p - e);
  }
  function tempFromMixratio(w, p) {        // inverse of mixratio, returns C
    var e = p * w / (621.97 + w);
    if (e < 1e-9) e = 1e-9;
    var ln = Math.log(e / 6.112);
    return 243.5 * ln / (17.67 - ln);
  }
  function relh(p, t, td) {
    return 100 * mixratio(p, td) / mixratio(p, t);
  }
  function tempFromRH(p, t, rh) {          // dewpoint from RH%
    return tempFromMixratio(mixratio(p, t) * rh / 100, p);
  }
  function virtualTemp(t, w) {             // t C, w g/kg -> C
    var tk = t + ZK, wk = w / 1000;
    return tk * (1 + wk / EPS) / (1 + wk) - ZK;
  }
  function virtualTempTd(p, t, td) {
    return virtualTemp(t, mixratio(p, td));
  }
  function thetaK(p, t, p0) {              // potential temperature, K
    return (t + ZK) * Math.pow((p0 || 1000) / p, ROCP);
  }
  function tempFromThetaK(th, p, p0) {
    return th * Math.pow(p / (p0 || 1000), ROCP) - ZK;
  }
  function lclTempK(t, td) {               // Bolton (1980) eq. 15
    var tk = t + ZK, tdk = td + ZK;
    if (tdk > tk) tdk = tk;
    return 1 / (1 / (tdk - 56) + Math.log(tk / tdk) / 800) + 56;
  }
  function lclPres(p, t, td) {
    return p * Math.pow(lclTempK(t, td) / (t + ZK), 1 / ROCP);
  }
  function thetae(p, t, td) {              // Bolton (1980) eq. 43, K
    var w = mixratio(p, td);
    var tl = lclTempK(t, td);
    var th = (t + ZK) * Math.pow(1000 / p, 0.2854 * (1 - 0.28e-3 * w));
    return th * Math.exp((3.376 / tl - 0.00254) * w * (1 + 0.81e-3 * w));
  }
  function satThetae(p, t) { return thetae(p, t, t); }

  /* Temperature on the moist adiabat with the given theta-e, at pressure p.
     Newton iteration with an optional warm start keeps this cheap enough to
     run inside a drag loop. */
  function tempAtSatThetae(the, p, guess) {
    var t = (guess === undefined || !isFinite(guess)) ? 5 : guess;
    for (var i = 0; i < 25; i++) {
      var f = satThetae(p, t) - the;
      if (Math.abs(f) < 0.002) return t;
      var d = (satThetae(p, t + 0.1) - satThetae(p, t - 0.1)) / 0.2;
      if (!isFinite(d) || Math.abs(d) < 1e-6) break;
      var step = f / d;
      if (step > 25) step = 25; else if (step < -25) step = -25;
      t -= step;
      if (t < -140) t = -140; else if (t > 80) t = 80;
    }
    // fall back to bisection if Newton wandered off
    var lo = -140, hi = 80;
    for (var j = 0; j < 60; j++) {
      var mid = 0.5 * (lo + hi);
      if (satThetae(p, mid) < the) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }
  function wetbulb(p, t, td) {
    return tempAtSatThetae(thetae(p, t, td), p, td);
  }
  function wetbulbPotentialTempC(p, t, td) {
    return tempAtSatThetae(thetae(p, t, td), 1000, t);
  }

  /* ---- wind helpers ---- */
  function uv(dir, spd) {                  // met direction (from) -> u,v
    var r = dir * Math.PI / 180;
    return [-spd * Math.sin(r), -spd * Math.cos(r)];
  }
  function dirspd(u, v) {
    var spd = Math.sqrt(u * u + v * v);
    var dir = (270 - Math.atan2(v, u) * 180 / Math.PI) % 360;
    if (dir < 0) dir += 360;
    return [dir, spd];
  }
  var KT2MS = 0.514444, MS2KT = 1 / 0.514444;

  return {
    Rd: Rd, Rv: Rv, Cp: Cp, G: G, EPS: EPS, ROCP: ROCP, ZK: ZK,
    KT2MS: KT2MS, MS2KT: MS2KT,
    esat: esat, mixratio: mixratio, tempFromMixratio: tempFromMixratio,
    relh: relh, tempFromRH: tempFromRH,
    virtualTemp: virtualTemp, virtualTempTd: virtualTempTd,
    thetaK: thetaK, tempFromThetaK: tempFromThetaK,
    lclTempK: lclTempK, lclPres: lclPres,
    thetae: thetae, satThetae: satThetae, tempAtSatThetae: tempAtSatThetae,
    wetbulb: wetbulb, wetbulbPotentialTempC: wetbulbPotentialTempC,
    uv: uv, dirspd: dirspd
  };
})();
