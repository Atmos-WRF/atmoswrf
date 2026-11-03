/* hodo.js - hodograph canvas with draggable wind points. */
var HodoView = (function () {
  var T = Thermo;
  var BANDS = [
    { top: 1000, color: '#ff5b5b', label: '0-1' },
    { top: 3000, color: '#ffa94d', label: '1-3' },
    { top: 6000, color: '#3fd07a', label: '3-6' },
    { top: 9000, color: '#4fb8ff', label: '6-9' },
    { top: 12000, color: '#b18cff', label: '9-12' }
  ];

  function View(canvas, hooks) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.hooks = hooks || {};
    this.opts = { brush: 0, ring: 20, maxZ: 12000, autoScale: true, scale: 70 };
    this.drag = null; this.hover = null;
    this.bind();
  }

  View.prototype.setState = function (prof, params, opts) {
    this.prof = prof; this.params = params;
    if (opts) Object.assign(this.opts, opts);
    this.draw();
  };

  /* Fit the visible window to the trace, the storm motions and the origin,
     rather than always centring on 0/0 - a fast hodograph otherwise leaves
     three quarters of the box empty. */
  View.prototype.layout = function () {
    var r = this.canvas.getBoundingClientRect();
    this.w = Math.max(200, r.width); this.h = Math.max(200, r.height);
    var maxR = Math.min(this.w, this.h) / 2 - 14;

    var lo = [-12, -12], hi = [12, 12];          // always keep the origin in view
    if (this.prof) {
      var L = this.prof.levels, pts = [];
      for (var i = 0; i < L.length; i++) {
        if (L[i].z - this.prof.zsfc > this.opts.maxZ) break;
        pts.push(T.uv(L[i].dir, L[i].spd));
      }
      if (this.params) {
        pts.push(this.params.bunkers.rm, this.params.bunkers.lm, this.params.bunkers.mean);
        if (this.params.storm) pts.push(this.params.storm);
      }
      for (i = 0; i < pts.length; i++) {
        lo[0] = Math.min(lo[0], pts[i][0]); hi[0] = Math.max(hi[0], pts[i][0]);
        lo[1] = Math.min(lo[1], pts[i][1]); hi[1] = Math.max(hi[1], pts[i][1]);
      }
    }
    this.ou = (lo[0] + hi[0]) / 2; this.ov = (lo[1] + hi[1]) / 2;
    var half = Math.max((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2) * 1.1;
    this.half = Math.max(half, 20);
    if (!this.opts.autoScale) {
      this.half = this.opts.scale; this.ou = 0; this.ov = 0;
    }
    this.ppk = maxR / this.half;
    this.cx = this.w / 2; this.cy = this.h / 2;
    this.scale = this.half + Math.max(Math.abs(this.ou), Math.abs(this.ov));
    this.opts.ring = this.half > 55 ? 20 : (this.half > 28 ? 10 : 5);
  };
  View.prototype.X = function (u) { return this.cx + (u - this.ou) * this.ppk; };
  View.prototype.Y = function (v) { return this.cy - (v - this.ov) * this.ppk; };
  View.prototype.U = function (x) { return (x - this.cx) / this.ppk + this.ou; };
  View.prototype.V = function (y) { return (this.cy - y) / this.ppk + this.ov; };

  View.prototype.resize = function () {
    var dpr = window.devicePixelRatio || 1;
    this.layout();
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  };

  View.prototype.draw = function () {
    if (!this.prof) return;
    var dpr = window.devicePixelRatio || 1;
    this.layout();
    if (this.canvas.width !== Math.round(this.w * dpr)) {
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    var c = this.ctx, self = this, prof = this.prof, P = this.params;
    c.clearRect(0, 0, this.w, this.h);
    c.fillStyle = css('--plot-bg', '#0e1116');
    c.fillRect(0, 0, this.w, this.h);

    // rings, centred on the true origin
    var ox = this.X(0), oy = this.Y(0);
    c.font = '9px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.save();
    c.beginPath(); c.rect(0, 0, this.w, this.h); c.clip();
    for (var r = this.opts.ring; r <= this.scale + 0.1; r += this.opts.ring) {
      c.beginPath(); c.arc(ox, oy, r * this.ppk, 0, 6.2832);
      c.strokeStyle = css('--ring', '#333c4a'); c.lineWidth = 1; c.stroke();
      c.fillStyle = css('--axis-dim', '#66748a');
      c.fillText(String(r), ox + r * this.ppk - 10, oy - 8);
    }
    c.strokeStyle = css('--ring', '#333c4a');
    c.beginPath();
    c.moveTo(0, oy); c.lineTo(this.w, oy);
    c.moveTo(ox, 0); c.lineTo(ox, this.h);
    c.stroke();
    c.restore();

    // storm-relative helicity area for 0-3 km
    if (P && P.storm) {
      var pts = Params.layerPts(prof, 0, 3000);
      c.beginPath();
      c.moveTo(this.X(P.storm[0]), this.Y(P.storm[1]));
      for (var i = 0; i < pts.length; i++) c.lineTo(this.X(pts[i][1]), this.Y(pts[i][2]));
      c.closePath();
      c.fillStyle = 'rgba(120,190,255,0.10)'; c.fill();
    }

    // the trace, coloured by height band
    var L = prof.levels;
    var prev = null, prevZ = null;
    for (i = 0; i < L.length; i++) {
      var z = L[i].z - prof.zsfc;
      if (z > this.opts.maxZ) {
        if (prev && prevZ < this.opts.maxZ) {
          var w = prof.windAtAgl(this.opts.maxZ);
          seg(c, this, prev, w, bandColor(this.opts.maxZ));
        }
        break;
      }
      var cur = T.uv(L[i].dir, L[i].spd);
      if (prev) seg(c, this, prev, cur, bandColor(z));
      prev = cur; prevZ = z;
    }

    // Bunkers + mean wind + current storm motion
    if (P) {
      this.marker(c, P.bunkers.rm, '#ff6b6b', 'RM');
      this.marker(c, P.bunkers.lm, '#5aa9ff', 'LM');
      this.marker(c, P.bunkers.mean, '#c8d3e2', 'MW', true);
      if (P.storm) {
        c.beginPath();
        c.arc(this.X(P.storm[0]), this.Y(P.storm[1]), 9, 0, 6.2832);
        c.strokeStyle = '#ffd166'; c.lineWidth = 1.6; c.setLineDash([3, 3]);
        c.stroke(); c.setLineDash([]);
      }
    }

    // draggable points
    for (i = 0; i < L.length; i++) {
      var zz = L[i].z - prof.zsfc;
      if (zz > this.opts.maxZ) break;
      var q = T.uv(L[i].dir, L[i].spd);
      var hot = (this.drag && this.drag.idx === i) || (this.hover === i);
      c.beginPath();
      c.arc(this.X(q[0]), this.Y(q[1]), hot ? 6 : 3.6, 0, 6.2832);
      c.fillStyle = hot ? '#fff' : bandColor(zz);
      c.fill();
      c.lineWidth = 1.3; c.strokeStyle = bandColor(zz); c.stroke();
    }

    // km labels along the trace
    c.font = '9px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    c.fillStyle = '#9fb0c6';
    for (var km = 1; km <= this.opts.maxZ / 1000; km++) {
      var wv = prof.windAtAgl(km * 1000);
      if (prof.pAtAgl(km * 1000) <= prof.ptop) break;
      c.beginPath();
      c.arc(this.X(wv[0]), this.Y(wv[1]), 2, 0, 6.2832);
      c.fillStyle = '#e6edf7'; c.fill();
      c.fillStyle = '#9fb0c6';
      c.textAlign = 'left'; c.textBaseline = 'bottom';
      c.fillText(String(km), this.X(wv[0]) + 5, this.Y(wv[1]) - 3);
    }

    if (this.hoverInfo) {
      c.font = '10px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
      var s = this.hoverInfo;
      var tw = c.measureText(s).width + 12;
      c.fillStyle = 'rgba(10,14,20,0.85)';
      c.fillRect(6, this.h - 24, tw, 18);
      c.fillStyle = '#dfe6f0'; c.textAlign = 'left'; c.textBaseline = 'middle';
      c.fillText(s, 12, this.h - 15);
    }

    c.strokeStyle = css('--frame', '#5a6577'); c.lineWidth = 1;
    c.strokeRect(0.5, 0.5, this.w - 1, this.h - 1);
  };

  function seg(c, self, a, b, color) {
    c.beginPath();
    c.moveTo(self.X(a[0]), self.Y(a[1]));
    c.lineTo(self.X(b[0]), self.Y(b[1]));
    c.strokeStyle = color; c.lineWidth = 2.6; c.lineCap = 'round'; c.stroke();
  }
  function bandColor(z) {
    for (var i = 0; i < BANDS.length; i++) if (z <= BANDS[i].top) return BANDS[i].color;
    return BANDS[BANDS.length - 1].color;
  }

  View.prototype.marker = function (c, uv, color, label, square) {
    var x = this.X(uv[0]), y = this.Y(uv[1]);
    c.save();
    c.strokeStyle = color; c.fillStyle = color; c.lineWidth = 1.6;
    if (square) { c.strokeRect(x - 4, y - 4, 8, 8); }
    else { c.beginPath(); c.arc(x, y, 4.5, 0, 6.2832); c.stroke(); }
    c.font = 'bold 9px "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
    c.textAlign = 'left'; c.textBaseline = 'top';
    c.fillText(label, x + 6, y + 3);
    c.restore();
  };

  View.prototype.pick = function (mx, my) {
    var L = this.prof.levels, best = -1, bd = 13 * 13;
    for (var i = 0; i < L.length; i++) {
      if (L[i].z - this.prof.zsfc > this.opts.maxZ) break;
      var q = T.uv(L[i].dir, L[i].spd);
      var dx = this.X(q[0]) - mx, dy = this.Y(q[1]) - my;
      var d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best < 0 ? null : best;
  };

  View.prototype.bind = function () {
    var self = this, cv = this.canvas;
    function pos(e) {
      var r = cv.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    }
    cv.addEventListener('pointerdown', function (e) {
      if (!self.prof) return;
      var m = pos(e), hit = self.pick(m[0], m[1]);
      if (hit === null) return;
      cv.setPointerCapture(e.pointerId);
      self.drag = { idx: hit, snapshot: self.prof.copyLevels() };
      if (self.hooks.onDragStart) self.hooks.onDragStart();
      if (self.hooks.onSelect) self.hooks.onSelect(hit);
      self.draw();
      e.preventDefault();
    });
    cv.addEventListener('pointermove', function (e) {
      if (!self.prof) return;
      var m = pos(e);
      if (self.drag) {
        var L = self.prof.levels, i = self.drag.idx, snap = self.drag.snapshot;
        var nu = self.U(m[0]), nv = self.V(m[1]);
        if (e.shiftKey) {                             // shift keeps the direction
          var keep = T.uv(snap[i].dir, 1);
          var mag = Math.sqrt(nu * nu + nv * nv);
          nu = keep[0] * mag; nv = keep[1] * mag;
        }
        var old = T.uv(snap[i].dir, snap[i].spd);
        var du = nu - old[0], dv = nv - old[1];
        var brush = self.opts.brush || 0;
        for (var j = 0; j < L.length; j++) {
          var wgt = j === i ? 1 : (brush > 0 ? hann(Math.abs(L[j].z - L[i].z), brush) : 0);
          if (wgt <= 0) continue;
          var o = T.uv(snap[j].dir, snap[j].spd);
          var ds = T.dirspd(o[0] + du * wgt, o[1] + dv * wgt);
          L[j].dir = ds[0]; L[j].spd = ds[1];
        }
        if (self.hooks.onEdit) self.hooks.onEdit();
        return;
      }
      var hit = self.pick(m[0], m[1]);
      var changed = hit !== self.hover;
      self.hover = hit;
      cv.style.cursor = hit === null ? 'crosshair' : 'move';
      var u = self.U(m[0]), v = self.V(m[1]);
      var ds2 = T.dirspd(u, v);
      var info = ds2[0].toFixed(0).padStart(3, '0') + '° / ' + ds2[1].toFixed(0) + ' kt';
      if (info !== self.hoverInfo) { self.hoverInfo = info; changed = true; }
      if (changed) self.draw();
    });
    function end() {
      if (self.drag) { self.drag = null; if (self.hooks.onDragEnd) self.hooks.onDragEnd(); }
      self.draw();
    }
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', function () {
      if (!self.drag) { self.hover = null; self.hoverInfo = null; self.draw(); }
    });
  };

  function hann(d, r) { return d >= r ? 0 : 0.5 * (1 + Math.cos(Math.PI * d / r)); }
  function css(name, fb) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) || fb;
  }

  View.BANDS = BANDS;
  return View;
})();
