/* skewt.js - skew-T log-P canvas with draggable temperature / dewpoint points. */
var Draw = {
  barb: function (ctx, x, y, dir, spd, color, len) {
    len = len || 26;
    var r = dir * Math.PI / 180;
    var ux = Math.sin(r), uy = -Math.cos(r);          // toward where the wind is from
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color;
    ctx.lineWidth = 1.4; ctx.lineCap = 'round';
    if (spd < 2.5) {                                   // calm
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 6.2832); ctx.stroke();
      ctx.restore(); return;
    }
    var tipx = x + ux * len, tipy = y + uy * len;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tipx, tipy); ctx.stroke();
    var px = -uy, py = ux;                             // perpendicular
    var rem = Math.round(spd / 5) * 5;
    var pos = 0, step = 5.5, blen = 11;
    while (rem >= 50) {
      var ax = tipx - ux * pos, ay = tipy - uy * pos;
      var bx = tipx - ux * (pos + step * 1.3), by = tipy - uy * (pos + step * 1.3);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax + px * blen, ay + py * blen);
      ctx.lineTo(bx, by);
      ctx.closePath(); ctx.fill();
      pos += step * 1.7; rem -= 50;
    }
    while (rem >= 10) {
      var cx = tipx - ux * pos, cy = tipy - uy * pos;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + px * blen - ux * step * 0.5, cy + py * blen - uy * step * 0.5);
      ctx.stroke();
      pos += step; rem -= 10;
    }
    if (rem >= 5) {
      if (pos === 0) pos = step;                       // don't hang a half barb off the tip
      var dx = tipx - ux * pos, dy = tipy - uy * pos;
      ctx.beginPath(); ctx.moveTo(dx, dy);
      ctx.lineTo(dx + px * blen * 0.5 - ux * step * 0.25, dy + py * blen * 0.5 - uy * step * 0.25);
      ctx.stroke();
    }
    ctx.restore();
  }
};

var SkewTView = (function () {
  var T = Thermo;
  var PBOT = 1050, PTOP = 100;
  var TMIN = -45, TMAX = 45;
  /* The plot is a fixed square with isotherms at 45 degrees, the standard
     skew-T aspect, whatever shape the pane is. Zooming scales that square
     about the cursor; the frame stays put and the diagram moves inside it. */
  var ZOOM_MAX = 24;
  var PLINES = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100];

  function View(canvas, hooks) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.hooks = hooks || {};
    this.pad = { l: 44, r: 88, t: 16, b: 34 };
    this.bg = document.createElement('canvas');
    this.drag = null; this.hover = null; this.hoverInfo = null;
    this.opts = { parcel: 'ml', showWetbulb: false, showVirtual: false, brush: 0 };
    this.zoom = { k: 1, ox: 0, oy: 0 };
    this.pan = null;
    this.bind();
  }

  View.prototype.setState = function (prof, params, opts) {
    this.prof = prof; this.params = params;
    if (opts) Object.assign(this.opts, opts);
    this.draw();
  };

  /* ---------- geometry ---------- */
  View.prototype.layout = function () {
    var r = this.canvas.getBoundingClientRect();
    this.w = Math.max(320, r.width); this.h = Math.max(300, r.height);
    var aw = this.w - this.pad.l - this.pad.r, ah = this.h - this.pad.t - this.pad.b;
    var side = Math.max(200, Math.min(aw, ah));
    this.pw = this.ph = side;
    this.x0 = this.pad.l + Math.max(0, (aw - side) / 2);
    this.y0 = this.pad.t + Math.max(0, (ah - side) / 2);
    this.ppd = this.pw / (TMAX - TMIN);
    this.skewPx = this.ph;
    this.barbX = this.x0 + this.pw + 32;
    this.clampZoom();
  };
  View.prototype.clampZoom = function () {
    var z = this.zoom;
    z.k = Math.max(1, Math.min(ZOOM_MAX, z.k));
    if (!this.pw) return;
    z.ox = Math.min(0, Math.max(this.pw * (1 - z.k), z.ox));
    z.oy = Math.min(0, Math.max(this.ph * (1 - z.k), z.oy));
  };
  View.prototype.zoomAt = function (mx, my, factor) {
    var z = this.zoom, k0 = z.k;
    var k1 = Math.max(1, Math.min(ZOOM_MAX, k0 * factor));
    if (k1 === k0) return;
    var u = (mx - this.x0 - z.ox) / k0, v = (my - this.y0 - z.oy) / k0;
    z.k = k1;
    z.ox = mx - this.x0 - u * k1;
    z.oy = my - this.y0 - v * k1;
    this.clampZoom();
    this.draw();
  };
  View.prototype.resetZoom = function () {
    this.zoom = { k: 1, ox: 0, oy: 0 };
    this.draw();
  };
  View.prototype.frac = function (p) { return Math.log(PBOT / p) / Math.log(PBOT / PTOP); };
  View.prototype.yP = function (p) {
    return this.y0 + this.zoom.oy + this.ph * this.zoom.k * (1 - this.frac(p));
  };
  View.prototype.pY = function (y) {
    var f = 1 - (y - this.y0 - this.zoom.oy) / (this.ph * this.zoom.k);
    return PBOT * Math.pow(PTOP / PBOT, f);
  };
  View.prototype.xT = function (t, p) {
    return this.x0 + this.zoom.ox +
           this.zoom.k * ((t - TMIN) * this.ppd + this.frac(p) * this.skewPx);
  };
  View.prototype.tX = function (x, p) {
    return TMIN + ((x - this.x0 - this.zoom.ox) / this.zoom.k - this.frac(p) * this.skewPx) / this.ppd;
  };
  View.prototype.inPlot = function (x, y) {
    return x >= this.x0 - 1 && x <= this.x0 + this.pw + 1 &&
           y >= this.y0 - 1 && y <= this.y0 + this.ph + 1;
  };
  View.prototype.key = function () {
    var z = this.zoom;
    return [this.w, this.h, window.devicePixelRatio || 1, z.k, z.ox, z.oy].join('x');
  };

  /* ---------- static background ---------- */
  View.prototype.buildBg = function () {
    var dpr = window.devicePixelRatio || 1;
    this.bg.width = Math.round(this.w * dpr); this.bg.height = Math.round(this.h * dpr);
    var c = this.bg.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);
    var self = this, k = this.zoom.k;
    var dp = Math.max(1, Math.round(20 / k));               // curve sampling
    var pTopV = this.pY(this.y0), pBotV = this.pY(this.y0 + this.ph);
    var pHi = Math.min(PBOT, pBotV * 1.02), pLo = Math.max(PTOP, pTopV / 1.02);

    c.fillStyle = getCss('--plot-bg', '#0e1116');
    c.fillRect(this.x0, this.y0, this.pw, this.ph);

    c.save();
    c.beginPath(); c.rect(this.x0, this.y0, this.pw, this.ph); c.clip();

    function path(pts, style, width, dash) {
      if (pts.length < 2) return;
      c.beginPath();
      c.moveTo(pts[0][0], pts[0][1]);
      for (var i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]);
      c.strokeStyle = style; c.lineWidth = width || 1;
      c.setLineDash(dash || []); c.stroke(); c.setLineDash([]);
    }

    // mixing ratio
    var mr = [0.4, 1, 2, 3, 5, 8, 12, 16, 20, 25];
    mr.forEach(function (w) {
      var pts = [];
      for (var p = Math.min(pHi, 1050); p >= Math.max(pLo, 400) - dp; p -= dp) pts.push([self.xT(T.tempFromMixratio(w, p), p), self.yP(p)]);
      path(pts, getCss('--mixr', '#3f8f5e'), 1, [3, 3]);
    });

    // dry adiabats
    var thStep = k >= 4 ? 2 : (k >= 2 ? 5 : 10);
    for (var th = -30; th <= 200; th += thStep) {
      var pts = [];
      for (var p = pHi; p >= pLo - dp; p -= dp) pts.push([self.xT(T.tempFromThetaK(th + 273.15, p), p), self.yP(p)]);
      path(pts, getCss('--dryad', '#7a5a33'), 1);
    }

    // moist adiabats
    var twStep = k >= 4 ? 1 : (k >= 2 ? 2 : 4);
    for (var tw = -20; tw <= 36; tw += twStep) {
      var the = T.satThetae(1000, tw), guess = tw, pts2 = [];
      for (var p2 = 1050; p2 >= 100; p2 -= dp) {
        guess = T.tempAtSatThetae(the, p2, guess);
        if (p2 <= pHi && p2 >= pLo - dp) pts2.push([self.xT(guess, p2), self.yP(p2)]);
      }
      path(pts2, getCss('--moistad', '#2f6f74'), 1, [6, 4]);
    }

    // isotherms
    var tStep = this.tStep();
    for (var t = -140; t <= 60; t += tStep) {
      var a = [self.xT(t, PBOT), self.yP(PBOT)], b = [self.xT(t, PTOP), self.yP(PTOP)];
      var zero = (t === 0), major = (t % 10 === 0);
      path([a, b], zero ? getCss('--iso0', '#4f7fd0') : getCss('--iso', '#3a4250'),
           zero ? 1.7 : (major ? 1 : 0.6));
    }

    // isobars
    var plines = this.isobars();
    plines.forEach(function (p) {
      var y = self.yP(p);
      path([[self.x0, y], [self.x0 + self.pw, y]], getCss('--isobar', '#3a4250'),
           PLINES.indexOf(p) >= 0 ? 1 : 0.6);
    });
    c.restore();

    // frame
    c.strokeStyle = getCss('--frame', '#5a6577'); c.lineWidth = 1.2;
    c.strokeRect(this.x0 + 0.5, this.y0 + 0.5, this.pw, this.ph);

    // labels
    c.fillStyle = getCss('--axis-text', '#93a0b4');
    c.font = '11px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    c.textAlign = 'right'; c.textBaseline = 'middle';
    var lastLab = -1e9;
    plines.slice().reverse().forEach(function (p) {
      var y = self.yP(p);
      if (y < self.y0 - 1 || y > self.y0 + self.ph + 1 || y - lastLab < 14) return;
      lastLab = y;
      c.fillText(String(p), self.x0 - 6, y);
    });

    // isotherms are labelled where they cross the bottom and top of the frame
    var lStep = tStep * (this.ppd * k * tStep < 22 ? 2 : 1);
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (var ti = -140; ti <= 60; ti += lStep) {
      var x = this.xT(ti, pBotV);
      if (x < this.x0 - 2 || x > this.x0 + this.pw + 2) continue;
      c.fillText(String(ti), x, this.y0 + this.ph + 6);
    }
    c.textBaseline = 'bottom';
    for (ti = -140; ti <= 60; ti += lStep) {
      var xt = this.xT(ti, pTopV);
      if (xt < this.x0 + 12 || xt > this.x0 + this.pw - 12) continue;
      c.fillText(String(ti), xt, this.y0 - 2);
    }
    c.textAlign = 'center'; c.textBaseline = 'top';
    c.fillStyle = getCss('--axis-dim', '#66748a');
    c.fillText('temperature  (°C)', this.x0 + this.pw / 2, this.y0 + this.ph + 19);
    c.save();
    c.translate(this.x0 - 44, this.y0 + this.ph / 2); c.rotate(-Math.PI / 2);
    c.textBaseline = 'top'; c.fillText('pressure  (hPa)', 0, 0);
    c.restore();

    this.bgKey = this.key();
  };
  View.prototype.tStep = function () {
    var k = this.zoom.k;
    return k >= 8 ? 1 : (k >= 4 ? 2 : (k >= 2 ? 5 : 10));
  };
  /* Standard isobars, filled in with 50 / 25 / 10 / 5 hPa lines as you zoom. */
  View.prototype.isobars = function () {
    var k = this.zoom.k;
    var step = k >= 10 ? 5 : (k >= 5 ? 10 : (k >= 3 ? 25 : (k >= 1.6 ? 50 : 0)));
    if (!step) return PLINES.slice();
    var out = PLINES.slice();
    for (var p = 1050; p >= PTOP; p -= step) if (out.indexOf(p) < 0) out.push(p);
    return out.sort(function (a, b) { return b - a; });
  };

  /* ---------- main draw ---------- */
  View.prototype.resize = function () {
    this.layout();
    var dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.buildBg();
    this.draw();
  };

  View.prototype.draw = function () {
    if (!this.prof) return;
    this.layout();
    var dpr = window.devicePixelRatio || 1;
    if (this.bgKey !== this.key()) {
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.buildBg();
    }
    var c = this.ctx, self = this, prof = this.prof, P = this.params;
    c.clearRect(0, 0, this.w, this.h);
    c.drawImage(this.bg, 0, 0, this.w, this.h);

    // ask the profile directly: params.eff is the inflow layer, not a parcel
    var parcel = prof.parcel(this.opts.parcel);

    c.save();
    c.beginPath(); c.rect(this.x0, this.y0, this.pw, this.ph); c.clip();

    // ---- CAPE / CIN shading ----
    if (parcel) this.shade(c, prof, parcel);

    // ---- effective inflow band ----
    if (P && P.eff && P.eff.botP !== null) {
      var yb = this.yP(P.eff.botP), yt = this.yP(P.eff.topP);
      c.fillStyle = 'rgba(120,190,255,0.09)';
      c.fillRect(this.x0, yt, 26, yb - yt);
      c.strokeStyle = 'rgba(120,190,255,0.55)'; c.lineWidth = 1.4;
      c.beginPath();
      c.moveTo(this.x0 + 22, yt); c.lineTo(this.x0 + 4, yt);
      c.lineTo(this.x0 + 4, yb); c.lineTo(this.x0 + 22, yb);
      c.stroke();
    }

    // ---- height ticks (km AGL) ----
    c.font = '10px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    c.fillStyle = getCss('--hgt-text', '#7d8ba0');
    c.textAlign = 'left'; c.textBaseline = 'middle';
    for (var km = 1; km <= 15; km++) {
      var pk = prof.pAtAgl(km * 1000);
      if (pk <= PTOP + 2 || pk >= prof.psfc) continue;
      var yk = this.yP(pk);
      c.strokeStyle = 'rgba(140,155,175,0.35)';
      c.beginPath(); c.moveTo(this.x0 + 28, yk); c.lineTo(this.x0 + 36, yk); c.stroke();
      c.fillText(km + 'km', this.x0 + 39, yk);
    }

    // ---- environment curves ----
    var g = prof.grid;
    if (this.opts.showWetbulb && g.wb) {
      this.curve(c, g.p, g.wb, getCss('--wetbulb', '#4fd0c8'), 1.4, [4, 3]);
    }
    if (this.opts.showVirtual) {
      this.curve(c, g.p, g.tv, getCss('--virt', '#ff9d6b'), 1.2, [2, 3]);
    }

    // ---- parcel trace ----
    if (parcel) {
      c.beginPath();
      for (var i = 0; i < parcel.ps.length; i++) {
        var xx = this.xT(parcel.tp[i], parcel.ps[i]), yy = this.yP(parcel.ps[i]);
        if (i === 0) c.moveTo(xx, yy); else c.lineTo(xx, yy);
      }
      c.strokeStyle = getCss('--parcel', '#ffffff'); c.lineWidth = 1.6;
      c.setLineDash([5, 4]); c.stroke(); c.setLineDash([]);
    }

    // ---- T and Td ----
    this.curve(c, prof.levels.map(l => l.p), prof.levels.map(l => l.td), getCss('--td', '#3fd07a'), 2.6);
    this.curve(c, prof.levels.map(l => l.p), prof.levels.map(l => l.t), getCss('--t', '#ff5b5b'), 2.6);

    c.restore();

    // ---- level markers ----
    // A dot per level stops being readable once they are a few pixels apart,
    // so at high resolution thin them out and keep full-size handles only for
    // the points actually in play. Hit testing still covers every level.
    var L = prof.levels;
    var spacing = this.ph * this.zoom.k / Math.max(L.length - 1, 1);
    var step = Math.max(1, Math.ceil(10 / Math.max(spacing, 0.5)));
    var dense = step > 1;
    var tdCol = getCss('--td', '#3fd07a'), tCol = getCss('--t', '#ff5b5b');

    for (i = 0; i < L.length; i++) {
      // BUFKIT profiles run well past 100 hPa; those levels are off the top of
      // the diagram and their handles would float outside the frame.
      if (L[i].p < PTOP || L[i].p > PBOT) continue;
      var ly = this.yP(L[i].p);
      if (ly < this.y0 - 6 || ly > this.y0 + this.ph + 6) continue;
      var live = this.isLive(i);
      if (!live && dense && (i % step) !== 0 && i !== 0 && i !== L.length - 1) continue;
      var r = live ? 0 : (dense ? 1.7 : (spacing < 16 ? 2.8 : 3.6));
      var xd = this.xT(L[i].td, L[i].p), xt = this.xT(L[i].t, L[i].p);
      if (this.inPlot(xd, ly)) this.handle(c, xd, ly, tdCol, 'td', i, r);
      if (this.inPlot(xt, ly)) this.handle(c, xt, ly, tCol, 't', i, r);
    }

    // ---- special levels ----
    if (parcel) {
      this.tag(c, parcel.lclp, 'LCL', getCss('--lcl', '#9fe8ff'));
      if (parcel.lfcp) this.tag(c, parcel.lfcp, 'LFC', getCss('--lfc', '#ffd166'));
      if (parcel.elp) this.tag(c, parcel.elp, 'EL', getCss('--el', '#c39bff'));
    }
    if (P && P.fzl !== null) this.tag(c, prof.pAtAgl(P.fzl), '0°C', 'rgba(120,180,255,0.75)', true);

    // ---- wind barbs ----
    c.save();
    var lastY = -1e9;
    for (i = 0; i < L.length; i++) {
      var y = this.yP(L[i].p);
      if (y < this.y0 - 2 || y > this.y0 + this.ph + 2) continue;
      if (Math.abs(y - lastY) < 21) continue;
      lastY = y;
      Draw.barb(c, this.barbX, y, L[i].dir, L[i].spd, getCss('--barb', '#c9d4e4'), 24);
    }
    c.restore();

    // ---- readout ----
    if (this.hoverInfo) this.readout(c, this.hoverInfo);
    if (parcel) this.legend(c, parcel);
    if (this.zoom.k > 1.001) this.zoomBadge(c);
  };

  View.prototype.zoomBadge = function (c) {
    c.save();
    c.font = '10.5px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    var s = '×' + this.zoom.k.toFixed(1) + '   drag to pan · double-click to reset';
    var w = c.measureText(s).width + 14;
    var x = this.x0 + this.pw - w - 6, y = this.y0 + this.ph - 25;
    c.fillStyle = 'rgba(10,14,20,0.78)';
    c.fillRect(x, y, w, 19);
    c.strokeStyle = 'rgba(79,157,255,0.45)'; c.strokeRect(x + 0.5, y + 0.5, w, 19);
    c.fillStyle = '#cfe0f7'; c.textAlign = 'left'; c.textBaseline = 'middle';
    c.fillText(s, x + 7, y + 10);
    c.restore();
  };

  View.prototype.curve = function (c, ps, ts, color, width, dash) {
    c.beginPath();
    for (var i = 0; i < ps.length; i++) {
      var x = this.xT(ts[i], ps[i]), y = this.yP(ps[i]);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.strokeStyle = color; c.lineWidth = width;
    c.lineJoin = 'round'; c.setLineDash(dash || []);
    c.stroke(); c.setLineDash([]);
  };

  /* Fill between the parcel trace and the environment: warm (CAPE) above the
     LFC, cool (CIN) at or below it.  Bands are split at each crossing so the
     polygons close cleanly. */
  View.prototype.shade = function (c, prof, parcel) {
    var ps = parcel.ps, i;
    var P = [], TP = [], TE = [];
    for (i = 0; i < ps.length; i++) {
      var te = prof.interpP(ps[i], 't');
      P.push(ps[i]); TP.push(parcel.tp[i]); TE.push(te);
      if (i < ps.length - 1) {
        var d0 = parcel.tp[i] - te;
        var te1 = prof.interpP(ps[i + 1], 't');
        var d1 = parcel.tp[i + 1] - te1;
        if (d0 * d1 < 0) {
          var f = d0 / (d0 - d1);
          var pc = Math.exp(Math.log(ps[i]) + f * (Math.log(ps[i + 1]) - Math.log(ps[i])));
          var tc = parcel.tp[i] + f * (parcel.tp[i + 1] - parcel.tp[i]);
          P.push(pc); TP.push(tc); TE.push(tc);
        }
      }
    }
    var lfc = parcel.lfcp || parcel.lclp;
    var i0 = 0;
    while (i0 < P.length - 1) {
      var s = Math.sign((TP[i0] - TE[i0]) + (TP[i0 + 1] - TE[i0 + 1]));
      var i1 = i0 + 1;
      while (i1 < P.length - 1 &&
             Math.sign((TP[i1] - TE[i1]) + (TP[i1 + 1] - TE[i1 + 1])) === s) i1++;
      var keep = s > 0 ? (P[i1] <= lfc + 0.5) : (P[i1] >= lfc - 0.5);
      if (s !== 0 && keep) {
        c.beginPath();
        c.moveTo(this.xT(TE[i0], P[i0]), this.yP(P[i0]));
        for (i = i0; i <= i1; i++) c.lineTo(this.xT(TE[i], P[i]), this.yP(P[i]));
        for (i = i1; i >= i0; i--) c.lineTo(this.xT(TP[i], P[i]), this.yP(P[i]));
        c.closePath();
        c.fillStyle = s > 0 ? 'rgba(255,72,72,0.22)' : 'rgba(88,148,255,0.22)';
        c.fill();
      }
      i0 = i1;
    }
  };

  /* Is this level the one being dragged, hovered, or selected in the table? */
  View.prototype.isLive = function (idx) {
    return (this.drag && this.drag.idx === idx) ||
           (this.hover && this.hover.idx === idx) ||
           (this.opts.selected === idx);
  };

  View.prototype.handle = function (c, x, y, color, kind, idx, radius) {
    var hot = (this.drag && this.drag.idx === idx && this.drag.kind === kind) ||
              (this.hover && this.hover.idx === idx && this.hover.kind === kind);
    var r = hot ? 6 : (radius || 3.6);
    if (r <= 0) r = this.opts.selected === idx ? 4.2 : 3.6;
    c.beginPath();
    c.arc(x, y, r, 0, 6.2832);
    c.fillStyle = hot ? '#fff' : color;
    c.fill();
    c.lineWidth = 1.4; c.strokeStyle = color; c.stroke();
  };

  View.prototype.tag = function (c, p, text, color, dim) {
    if (!p || p > PBOT || p < PTOP) return;
    var y = this.yP(p), x = this.x0 + this.pw;
    if (y < this.y0 + 10 || y > this.y0 + this.ph) return;
    c.save();
    c.strokeStyle = color; c.lineWidth = dim ? 1 : 1.3;
    c.setLineDash([4, 3]);
    c.beginPath(); c.moveTo(this.x0 + 82, y); c.lineTo(x - 4, y); c.stroke();
    c.setLineDash([]);
    c.fillStyle = color;
    c.font = 'bold 10px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    c.textAlign = 'right'; c.textBaseline = 'bottom';
    c.fillText(text, x - 6, y - 2);
    c.restore();
  };

  View.prototype.legend = function (c, parcel) {
    c.save();
    c.font = '11px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    c.textAlign = 'left'; c.textBaseline = 'top';
    var lines = [
      parcel.label,
      'CAPE ' + parcel.cape.toFixed(0) + '   CIN ' + parcel.cin.toFixed(0)
    ];
    var w = 0;
    lines.forEach(function (s) { w = Math.max(w, c.measureText(s).width); });
    var bx = this.x0 + 8, by = this.y0 + 8;
    c.fillStyle = 'rgba(10,14,20,0.72)';
    c.fillRect(bx, by, w + 14, 32);
    c.strokeStyle = 'rgba(255,255,255,0.12)'; c.strokeRect(bx + 0.5, by + 0.5, w + 14, 32);
    c.fillStyle = '#cfd8e6'; c.fillText(lines[0], bx + 7, by + 4);
    c.fillStyle = '#ff9a9a'; c.fillText(lines[1], bx + 7, by + 17);
    c.restore();
  };

  View.prototype.readout = function (c, info) {
    c.save();
    c.font = '11px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    var s = info.text;
    var w = c.measureText(s).width + 14;
    var x = Math.min(info.x + 12, this.x0 + this.pw - w - 2);
    var y = Math.max(info.y - 26, this.y0 + 2);
    c.fillStyle = 'rgba(10,14,20,0.85)';
    c.fillRect(x, y, w, 19);
    c.strokeStyle = 'rgba(255,255,255,0.15)'; c.strokeRect(x + 0.5, y + 0.5, w, 19);
    c.fillStyle = '#dfe6f0'; c.textAlign = 'left'; c.textBaseline = 'middle';
    c.fillText(s, x + 7, y + 10);
    c.restore();
  };

  /* ---------- interaction ---------- */
  View.prototype.pick = function (mx, my) {
    var L = this.prof.levels, best = null, bd = 13 * 13;
    for (var i = 0; i < L.length; i++) {
      if (L[i].p < PTOP || L[i].p > PBOT) continue;   // not on the diagram
      var y = this.yP(L[i].p);
      if (y < this.y0 - 6 || y > this.y0 + this.ph + 6) continue;
      var kinds = [['t', L[i].t], ['td', L[i].td]];
      for (var k = 0; k < 2; k++) {
        var x = this.xT(kinds[k][1], L[i].p);
        if (!this.inPlot(x, y)) continue;
        var d = (x - mx) * (x - mx) + (y - my) * (y - my);
        if (d < bd) { bd = d; best = { kind: kinds[k][0], idx: i }; }
      }
    }
    return best;
  };

  View.prototype.bind = function () {
    var self = this, cv = this.canvas;
    function pos(e) {
      var r = cv.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    }
    function startPan(e, m) {
      cv.setPointerCapture(e.pointerId);
      self.pan = { x: m[0], y: m[1], ox: self.zoom.ox, oy: self.zoom.oy };
      cv.style.cursor = 'grabbing';
      e.preventDefault();
    }
    cv.addEventListener('pointerdown', function (e) {
      if (!self.prof) return;
      var m = pos(e);
      if (e.button === 1 || e.button === 2) { startPan(e, m); return; }
      var hit = self.pick(m[0], m[1]);
      if (e.altKey && hit) {                       // alt-click removes a level
        if (self.prof.levels.length > 4 && self.hooks.onRemove) self.hooks.onRemove(hit.idx);
        return;
      }
      if (!hit) {
        if (e.shiftKey && self.hooks.onAdd) {      // shift-click adds one
          var p = self.pY(m[1]);
          if (p < self.prof.psfc && p > self.prof.ptop) self.hooks.onAdd(p);
        } else if (self.zoom.k > 1.001 && self.inPlot(m[0], m[1])) {
          startPan(e, m);                           // drag empty space to pan
        }
        return;
      }
      cv.setPointerCapture(e.pointerId);
      self.drag = { kind: hit.kind, idx: hit.idx, snapshot: self.prof.copyLevels() };
      if (self.hooks.onDragStart) self.hooks.onDragStart();
      if (self.hooks.onSelect) self.hooks.onSelect(hit.idx);
      self.draw();
      e.preventDefault();
    });
    cv.addEventListener('pointermove', function (e) {
      if (!self.prof) return;
      var m = pos(e);
      if (self.pan) {
        self.zoom.ox = self.pan.ox + m[0] - self.pan.x;
        self.zoom.oy = self.pan.oy + m[1] - self.pan.y;
        self.clampZoom();
        self.hoverInfo = null;
        self.draw();
        return;
      }
      if (self.drag) {
        var L = self.prof.levels, i = self.drag.idx, snap = self.drag.snapshot;
        var newT = self.tX(m[0], L[i].p);
        newT = Math.max(-120, Math.min(60, newT));
        var delta = newT - snap[i][self.drag.kind];
        var brush = self.opts.brush || 0;
        for (var j = 0; j < L.length; j++) {
          var wgt = j === i ? 1 : (brush > 0 ? hann(Math.abs(L[j].z - L[i].z), brush) : 0);
          if (wgt <= 0) continue;
          var v = snap[j][self.drag.kind] + delta * wgt;
          if (self.drag.kind === 't') {
            L[j].t = v;
            if (L[j].td > L[j].t) L[j].td = L[j].t;
          } else {
            L[j].td = Math.min(v, L[j].t);
          }
        }
        if (self.hooks.onEdit) self.hooks.onEdit();
        return;
      }
      var hit = self.pick(m[0], m[1]);
      var changed = JSON.stringify(hit) !== JSON.stringify(self.hover);
      self.hover = hit;
      cv.style.cursor = hit ? 'ew-resize' : (e.shiftKey ? 'copy' :
                        (self.zoom.k > 1.001 ? 'grab' : 'crosshair'));
      if (m[0] > self.x0 && m[0] < self.x0 + self.pw && m[1] > self.y0 && m[1] < self.y0 + self.ph) {
        var p = self.pY(m[1]), t = self.tX(m[0], p);
        var z = self.prof.zAtP(p) - self.prof.zsfc;
        self.hoverInfo = {
          x: m[0], y: m[1],
          text: p.toFixed(0) + ' hPa  ' + t.toFixed(1) + '°C  ' +
                (z >= 0 ? z.toFixed(0) : '--') + ' m'
        };
        changed = true;
      } else if (self.hoverInfo) { self.hoverInfo = null; changed = true; }
      if (changed) self.draw();
    });
    function end(e) {
      if (self.pan) { self.pan = null; cv.style.cursor = 'grab'; }
      if (self.drag) {
        self.drag = null;
        if (self.hooks.onDragEnd) self.hooks.onDragEnd();
      }
      self.draw();
    }
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', function () {
      if (!self.drag) { self.hover = null; self.hoverInfo = null; self.draw(); }
    });
    cv.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    cv.addEventListener('wheel', function (e) {
      if (!self.prof) return;
      var m = pos(e);
      if (!self.inPlot(m[0], m[1])) return;
      e.preventDefault();
      var d = e.deltaY * (e.deltaMode === 1 ? 16 : (e.deltaMode === 2 ? 400 : 1));
      self.zoomAt(m[0], m[1], Math.exp(-d * 0.0018));
    }, { passive: false });
    cv.addEventListener('dblclick', function (e) {
      var m = pos(e);
      if (!self.pick(m[0], m[1])) self.resetZoom();
    });
  };

  function hann(d, r) {
    if (d >= r) return 0;
    return 0.5 * (1 + Math.cos(Math.PI * d / r));
  }
  function getCss(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) || fallback;
  }

  return View;
})();
