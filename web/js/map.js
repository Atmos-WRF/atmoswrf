'use strict';
// The map: world coordinates are Lambert-conformal kilometres (the model's own plane),
// shown with pan / zoom.  Draws grid products (one raster per patch), radar PPIs (polar
// gates resampled to screen pixels), map lines and overlays supplied by the app.

class MapView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.cx = 0; this.cy = 0; this.scale = 2.0;        // px per km
    this.proj = null;
    this.lines = null; this.paths = null;
    this.raster = null;                                // {patches:[{x0,y0,x1,y1,canvas}], smooth}
    this.ppi = null;                                   // {meta, codes, lut}
    this._ppiCache = null;
    this.showCounties = true;
    this.overlay = null;                               // fn(ctx, map)
    this.onClick = null; this.onHover = null; this.onDragRect = null; this.onDblClick = null;
    // onPress(x, y, event) may grab the mouse for an object on the map: it returns
    // {move(x, y), end(x, y)} (world km) instead of panning; grabbable(x, y) shows a move cursor
    this.onPress = null; this.grabbable = null;
    this._grab = null;
    this.tool = 'pan';
    this._drag = null;
    this._dirty = true;
    this._bind();
    const loop = () => {
      try { if (this._dirty) { this._dirty = false; this._draw(); } }
      catch (e) { console.error(e); }
      finally { requestAnimationFrame(loop); }
    };
    requestAnimationFrame(loop);
    new ResizeObserver(() => this.redraw()).observe(canvas);
  }

  redraw() { this._dirty = true; }

  // ---------------------------------------------------------------- projection
  setProjection(p) {
    this.proj = p;
  }
  fwd(lat, lon) {             // degrees -> km
    const p = this.proj, d2r = Math.PI / 180;
    let dlon = ((lon - p.lon0 + 540) % 360) - 180;
    const rho = p.R * p.F / Math.pow(Math.tan(Math.PI / 4 + lat * d2r / 2), p.n);
    const th = p.n * dlon * d2r;
    return [rho * Math.sin(th) / 1000, (p.rho0 - rho * Math.cos(th)) / 1000];
  }
  inv(xk, yk) {               // km -> [lat, lon]
    const p = this.proj, x = xk * 1000, y = yk * 1000;
    const dy = p.rho0 - y, sg = p.n > 0 ? 1 : -1;
    const rho = sg * Math.hypot(x, dy);
    const th = Math.atan2(sg * x, sg * dy);
    const lat = 2 * Math.atan(Math.pow(p.R * p.F / rho, 1 / p.n)) - Math.PI / 2;
    return [lat * 180 / Math.PI, p.lon0 + th / p.n * 180 / Math.PI];
  }

  // ---------------------------------------------------------------- view
  get W() { return this.canvas.clientWidth; }
  get H() { return this.canvas.clientHeight; }
  toScreen(x, y) { return [(x - this.cx) * this.scale + this.W / 2, this.H / 2 - (y - this.cy) * this.scale]; }
  toWorld(px, py) { return [(px - this.W / 2) / this.scale + this.cx, (this.H / 2 - py) / this.scale + this.cy]; }
  center(x, y, scale) { this.cx = x; this.cy = y; if (scale) this.scale = scale; this.redraw(); }

  setLines(lines) {
    this.lines = lines;
    this.paths = {};
    for (const [name, polys] of Object.entries(lines || {})) {
      const p = new Path2D();
      for (const a of polys) {
        p.moveTo(a[0], a[1]);
        for (let i = 2; i < a.length; i += 2) p.lineTo(a[i], a[i + 1]);
      }
      this.paths[name] = p;
    }
    this.redraw();
  }

  setRaster(r) { this.raster = r; this.redraw(); }
  setPPI(p) { this.ppi = p; this._ppiCache = null; this.redraw(); }

  // ---------------------------------------------------------------- map tiles (arhtmodelviewer/tiles.py)
  // The background: 'plain' (the dark map), 'osm' / 'osmdark' (OpenStreetMap, light or darkened here), 'sat'
  // (Esri World Imagery) or 'satlabels' (with roads and place names over the radar).  Web-Mercator tiles are
  // warped onto the Lambert plane piece by piece (each tile cut into n x n quads, each drawn with its own affine
  // map), so they line up with the county lines and the radar; a tile still loading is stood in for by the
  // nearest coarser one already here.  dim: 0..0.8 of darkening over the imagery, so the radar stands out.
  setBase(name, dim) {
    this.base = name || 'plain';
    if (dim !== undefined) this.baseDim = dim;
    this.redraw();
  }

  static BASES = {
    osm: { under: ['osm'], over: [], filter: 'none', attribution: '© OpenStreetMap contributors' },
    osmdark: { under: ['osm'], over: [], filter: 'invert(1) hue-rotate(180deg) brightness(0.9) contrast(0.85) saturate(0.6)',
               attribution: '© OpenStreetMap contributors' },
    sat: { under: ['sat'], over: [], filter: 'none',
           attribution: 'Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community' },
    satlabels: { under: ['sat'], over: ['roads', 'places'], filter: 'none',
                 attribution: 'Imagery, roads and labels © Esri, Maxar, Earthstar Geographics, and the GIS User Community' },
  };

  _tile(src, z, x, y) {
    const T = this._tiles || (this._tiles = new Map());
    const key = `${src}/${z}/${x}/${y}`;
    let t = T.get(key);
    if (t) { T.delete(key); T.set(key, t); return t; }         // most recently used last
    t = { img: new Image(), ok: false };
    this._inflight = this._inflight || 0;
    if (this._inflight > 16) return null;                    // not now: asked again on a later frame
    this._inflight++;
    t.img.onload = () => { t.ok = true; this._inflight--; this.redraw(); };
    t.img.onerror = () => { t.bad = true; this._inflight--; };
    t.img.src = `/tiles/${src}/${z}/${x}/${y}`;
    T.set(key, t);
    while (T.size > 700) T.delete(T.keys().next().value);
    return t;
  }

  _tileLL(z, tx, ty) {             // tile coordinates (fractional) at zoom z -> [lat, lon]
    const n = 2 ** z;
    return [Math.atan(Math.sinh(Math.PI * (1 - 2 * ty / n))) * 180 / Math.PI, tx / n * 360 - 180];
  }

  _drawTileImage(g, dpr, img, z, tx, ty, span, sx, sy, ss) {
    // the part (sx, sy, ss px) of img that covers tile-space [tx, tx+span] x [ty, ty+span] at zoom z
    const nq = z < 6 ? 6 : z < 9 ? 3 : 2, d = span / nq, sd = ss / nq;
    for (let j = 0; j < nq; j++) for (let i = 0; i < nq; i++) {
      const p = (a, b) => { const [la, lo] = this._tileLL(z, tx + a * d, ty + b * d); return this.toScreen(...this.fwd(la, lo)); };
      const p00 = p(i, j), p10 = p(i + 1, j), p01 = p(i, j + 1);
      const ax = (p10[0] - p00[0]) / sd, ay = (p10[1] - p00[1]) / sd, bx = (p01[0] - p00[0]) / sd, by = (p01[1] - p00[1]) / sd;
      g.setTransform(dpr * ax, dpr * ay, dpr * bx, dpr * by, dpr * p00[0], dpr * p00[1]);
      g.drawImage(img, sx + i * sd, sy + j * sd, sd, sd, -0.3, -0.3, sd + 0.6, sd + 0.6);   // a hair of overlap: no seams
    }
  }

  _drawTiles(g, dpr, layers, filter) {
    if (!this.proj || !layers.length) return;
    const W = this.W, H = this.H;
    // the view's corners and edges in lat / lon, and the zoom whose pixels are about the screen's
    const pts = [];
    for (let a = 0; a <= 4; a++) for (let b = 0; b <= 4; b++) pts.push(this.inv(...this.toWorld(a * W / 4, b * H / 4)));
    const clat = this.inv(this.cx, this.cy)[0];
    const mpp = 1000 / this.scale / Math.min(dpr, 2);
    let z = Math.round(Math.log2(156543.03 * Math.cos(clat * Math.PI / 180) / mpp));
    z = Math.max(2, Math.min(18, z));
    const tileXY = (la, lo, zz) => { const n = 2 ** zz, lr = Math.max(-85, Math.min(85, la)) * Math.PI / 180;
      return [(lo + 180) / 360 * n, (1 - Math.log(Math.tan(lr) + 1 / Math.cos(lr)) / Math.PI) / 2 * n]; };
    let x0, x1, y0, y1;
    for (;;) {
      const xy = pts.map(([la, lo]) => tileXY(la, lo, z));
      x0 = Math.floor(Math.min(...xy.map(q => q[0]))); x1 = Math.floor(Math.max(...xy.map(q => q[0])));
      y0 = Math.floor(Math.min(...xy.map(q => q[1]))); y1 = Math.floor(Math.max(...xy.map(q => q[1])));
      if ((x1 - x0 + 1) * (y1 - y0 + 1) <= 160 || z <= 2) break;
      z--;
    }
    const n = 2 ** z;
    g.save();
    g.filter = filter || 'none';
    for (const src of layers) {
      for (let y = Math.max(0, y0); y <= Math.min(n - 1, y1); y++) for (let x = x0; x <= x1; x++) {
        const xx = ((x % n) + n) % n;
        const t = this._tile(src, z, xx, y);
        if (t && t.ok) { this._drawTileImage(g, dpr, t.img, z, x, y, 1, 0, 0, t.img.width); continue; }
        // a coarser tile already here stands in
        for (let up = 1; up <= 4 && z - up >= 0; up++) {
          const f = 2 ** up, px = Math.floor(xx / f), py = Math.floor(y / f);
          const k = this._tiles && this._tiles.get(`${src}/${z - up}/${px}/${py}`);
          if (k && k.ok) {
            const ss = k.img.width / f;
            this._drawTileImage(g, dpr, k.img, z, x, y, 1, (xx - px * f) * ss, (y - py * f) * ss, ss);
            break;
          }
        }
      }
    }
    g.restore();
  }

  // ---------------------------------------------------------------- input
  _bind() {
    const c = this.canvas;
    c.addEventListener('wheel', e => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const [wx, wy] = this.toWorld(e.clientX - r.left, e.clientY - r.top);
      const f = Math.exp(-e.deltaY * 0.0015);
      this.scale = Math.min(200, Math.max(0.05, this.scale * f));
      const [nx, ny] = this.toWorld(e.clientX - r.left, e.clientY - r.top);
      this.cx += wx - nx; this.cy += wy - ny;
      this.redraw();
    }, { passive: false });
    c.addEventListener('mousedown', e => {
      const r = c.getBoundingClientRect();
      if (e.button === 0 && this.onPress) {
        const g = this.onPress(...this.toWorld(e.clientX - r.left, e.clientY - r.top), e);
        if (g) { this._grab = g; e.preventDefault(); return; }
      }
      this._drag = { x: e.clientX - r.left, y: e.clientY - r.top, cx: this.cx, cy: this.cy, moved: false };
    });
    window.addEventListener('mousemove', e => {
      const r = c.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      if (this._grab) { this._grab.move(...this.toWorld(px, py)); return; }
      if (!this._drag && this.grabbable && px >= 0 && py >= 0 && px <= this.W && py <= this.H) {
        c.style.cursor = this.grabbable(...this.toWorld(px, py)) ? 'move' : '';
      }
      if (this._drag) {
        const d = this._drag;
        if (Math.abs(px - d.x) + Math.abs(py - d.y) > 3) d.moved = true;
        if ((this.tool === 'zone' || this.tool === 'fire' || this.tool === 'volume' || this.tool === 'tornado')) { d.ex = px; d.ey = py; this.redraw(); }
        else if (this.tool === 'brush') {
          d.path = d.path || [[d.x, d.y]];
          const [lx, ly] = d.path[d.path.length - 1];
          if (Math.hypot(px - lx, py - ly) > 4) { d.path.push([px, py]); this.redraw(); }
        }
        else if (d.moved) {
          this.cx = d.cx - (px - d.x) / this.scale;
          this.cy = d.cy + (py - d.y) / this.scale;
          this.redraw();
        }
      }
      if (px >= 0 && py >= 0 && px <= this.W && py <= this.H && this.onHover) {
        this.onHover(...this.toWorld(px, py));
      }
    });
    c.addEventListener('dblclick', e => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      if (this.onDblClick) this.onDblClick(...this.toWorld(e.clientX - r.left, e.clientY - r.top));
    });
    window.addEventListener('mouseup', e => {
      if (this._grab) {
        const g = this._grab, r = c.getBoundingClientRect();
        this._grab = null;
        g.end(...this.toWorld(e.clientX - r.left, e.clientY - r.top));
        return;
      }
      const d = this._drag;
      this._drag = null;
      if (!d) return;
      const r = c.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      if (this.tool === 'brush' && d.moved && d.path && d.path.length > 1 && this.onDragPath) {
        this.onDragPath(d.path.map(([a, b]) => this.toWorld(a, b)));
        this.redraw();
      } else if ((this.tool === 'zone' || this.tool === 'fire' || this.tool === 'volume' || this.tool === 'tornado') && d.moved && this.onDragRect) {
        const [x0, y0] = this.toWorld(d.x, d.y), [x1, y1] = this.toWorld(px, py);
        this.onDragRect(Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1));
        this.redraw();
      } else if (!d.moved && this.onClick) {
        this.onClick(...this.toWorld(px, py));
      }
    });
  }

  // ---------------------------------------------------------------- drawing
  _draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = this.W, H = this.H;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    }
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#0b0e13'; g.fillRect(0, 0, W, H);
    const s = this.scale, ox = W / 2 - this.cx * s, oy = H / 2 + this.cy * s;
    const base = MapView.BASES[this.base];
    if (base) {
      this._drawTiles(g, dpr, base.under, base.filter);
      const dim = this.baseDim ?? (this.base === 'osm' ? 0 : 0.25);
      if (dim > 0) { g.setTransform(dpr, 0, 0, dpr, 0, 0); g.fillStyle = `rgba(11,14,19,${dim})`; g.fillRect(0, 0, W, H); }
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    // world (km) -> screen: x' = s x + ox, y' = -s y + oy
    if (!base && this.paths && this.paths.lakes) {
      g.save(); g.setTransform(dpr * s, 0, 0, -dpr * s, dpr * ox, dpr * oy);
      g.lineWidth = 0.8 / s; g.strokeStyle = '#1f3550'; g.stroke(this.paths.lakes); g.restore();
    }
    // grid products
    if (this.raster) {
      g.save(); g.setTransform(dpr * s, 0, 0, -dpr * s, dpr * ox, dpr * oy);
      g.imageSmoothingEnabled = !!this.raster.smooth;
      for (const p of this.raster.patches) {
        if (!p.canvas) continue;
        // image row 0 is the patch's south edge; the y-flip puts it at the bottom
        g.drawImage(p.canvas, p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0);
      }
      g.restore();
    }
    if (this.ppi) this._drawPPI(g, dpr);
    if (base && base.over.length) { this._drawTiles(g, dpr, base.over, 'none'); g.setTransform(dpr, 0, 0, dpr, 0, 0); }
    // map lines
    if (this.paths) {
      g.save(); g.setTransform(dpr * s, 0, 0, -dpr * s, dpr * ox, dpr * oy);
      g.lineJoin = 'round';
      if (this.showCounties && s > 0.9 && this.paths.counties) {
        g.lineWidth = 0.6 / s; g.strokeStyle = s > 3 ? '#4a5568' : '#343d4b'; g.stroke(this.paths.counties);
      }
      if (this.paths.coast) { g.lineWidth = 1.0 / s; g.strokeStyle = '#6f87a8'; g.stroke(this.paths.coast); }
      if (this.paths.states) { g.lineWidth = 1.3 / s; g.strokeStyle = '#a3afc0'; g.stroke(this.paths.states); }
      if (this.paths.borders) { g.lineWidth = 1.6 / s; g.strokeStyle = '#c8cfda'; g.stroke(this.paths.borders); }
      g.restore();
    }
    if (this.overlay) { g.save(); this.overlay(g, this); g.restore(); }
    if (base) {                    // the tiles' attribution, as their terms ask
      g.save(); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.font = '10px Segoe UI, sans-serif';
      const tw = g.measureText(base.attribution).width + 10;
      g.fillStyle = 'rgba(0,0,0,.55)'; g.fillRect(W - tw, H - 15, tw, 15);
      g.fillStyle = '#d8dee8'; g.fillText(base.attribution, W - tw + 5, H - 4); g.restore();
    }
    // hodograph brush stroke being painted: as wide as the brush
    if (this._drag && this.tool === 'brush' && this._drag.path) {
      g.save(); g.strokeStyle = 'rgba(167,139,250,.45)'; g.lineCap = 'round'; g.lineJoin = 'round';
      g.lineWidth = Math.max(4, 2 * (this.brushKm || 30) * this.scale);
      g.beginPath(); this._drag.path.forEach(([a, b], i) => i ? g.lineTo(a, b) : g.moveTo(a, b)); g.stroke(); g.restore();
    }
    // zone drag preview
    if (this._drag && (this.tool === 'zone' || this.tool === 'fire' || this.tool === 'volume' || this.tool === 'tornado') && this._drag.ex !== undefined) {
      const d = this._drag;
      g.strokeStyle = '#f2b84b'; g.setLineDash([5, 4]); g.lineWidth = 1.5;
      g.strokeRect(Math.min(d.x, d.ex), Math.min(d.y, d.ey), Math.abs(d.ex - d.x), Math.abs(d.ey - d.y));
      g.setLineDash([]);
    }
  }

  // Polar radar data -> screen pixels (rendered at half resolution, cached per view).  Smooth:
  // each pixel's value interpolated between the 4 nearest gates (radials x range), like
  // GR2Analyst's smoothing; gates without data drop out, and a pixel with fewer than 2 of
  // them takes the nearest gate.  The codes are linear in the value, so interpolating them is.
  _drawPPI(g, dpr) {
    const P = this.ppi, m = P.meta;
    const W = this.W, H = this.H;
    const key = [this.cx, this.cy, this.scale, W, H, !!P.smooth].join(',');
    if (!this._ppiCache || this._ppiCache.key !== key) {
      const step = this.scale > 6 ? 1 : 2;
      const w = Math.ceil(W / step), h = Math.ceil(H / step);
      const cv = this._ppiCache && this._ppiCache.canvas.width === w && this._ppiCache.canvas.height === h
        ? this._ppiCache.canvas : Object.assign(document.createElement('canvas'), { width: w, height: h });
      const cg = cv.getContext('2d');
      const img = cg.createImageData(w, h);
      const px = img.data, lut = P.lut, codes = P.codes;
      const rx = m.x / 1000, ry = m.y / 1000;
      const nrad = m.nrad, ng = m.ngate, first = m.first / 1000, gate = m.gate / 1000;
      const cosE = Math.cos(m.elev * Math.PI / 180), daz = 360 / nrad, conv = m.conv || 0;
      const rmax = first + gate * ng;
      for (let j = 0; j < h; j++) {
        const wy = (H / 2 - (j + 0.5) * step) / this.scale + this.cy - ry;
        for (let i = 0; i < w; i++) {
          const wx = ((i + 0.5) * step - W / 2) / this.scale + this.cx - rx;
          const sgr = Math.sqrt(wx * wx + wy * wy);
          if (sgr > rmax) continue;
          const r = sgr / cosE;
          const gi = Math.floor((r - first) / gate + 0.5);
          if (gi < 0 || gi >= ng) continue;
          let az = Math.atan2(wx, wy) * 57.29578 + conv;
          if (az < 0) az += 360; if (az >= 360) az -= 360;
          const ri = Math.floor(az / daz) % nrad;
          let code = codes[ri * ng + gi];
          if (P.smooth && code !== 1) {    // RF gates (code 1) stay RF
            // gate centres at first + k*gate, radial centres at (k + 0.5)*daz
            const fg = (r - first) / gate, fa = az / daz - 0.5;
            const g0 = Math.floor(fg), a0 = Math.floor(fa), tg = fg - g0, ta = fa - a0;
            let sum = 0, wsum = 0, n = 0;
            for (let da = 0; da < 2; da++) {
              const rr = ((a0 + da) % nrad + nrad) % nrad;
              for (let dg = 0; dg < 2; dg++) {
                const gg = g0 + dg;
                if (gg < 0 || gg >= ng) continue;
                const c = codes[rr * ng + gg];
                if (c < 2) continue;
                const wgt = (da ? ta : 1 - ta) * (dg ? tg : 1 - tg);
                sum += wgt * c; wsum += wgt; n++;
              }
            }
            if (n >= 2 && wsum > 0) code = Math.max(2, Math.min(255, Math.round(sum / wsum)));
          }
          if (!code) continue;
          const o = (j * w + i) * 4, q = code * 4;
          px[o] = lut[q]; px[o + 1] = lut[q + 1]; px[o + 2] = lut[q + 2]; px[o + 3] = lut[q + 3];
        }
      }
      cg.putImageData(img, 0, 0);
      this._ppiCache = { key, canvas: cv, step };
    }
    g.save();
    g.imageSmoothingEnabled = this._ppiCache.step > 1;
    g.drawImage(this._ppiCache.canvas, 0, 0, W, H);
    g.restore();
  }
}

window.MapView = MapView;
