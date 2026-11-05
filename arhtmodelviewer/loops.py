"""Radar loops from a run's Level II files: one radar (a WSR-88D, a TDWR, a DOW...), one product, one tilt, a
range of its volumes, on a map area, in a GR color table, as an animated GIF -- the viewer's loop maker.

Each frame is that volume's sweep nearest the tilt asked for, drawn gate by gate (nearest radial and gate) from the
volume's own site (a DOW that moved is drawn from wherever it stood), with county and state lines, towns, the
radar's mark and optionally the run's tornado tracks; a bar on top gives the radar, product, tilt and time.
Jobs run in the background (Loops.start / Loops.status); the GIFs go to output/loops."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import PROJECT, geo, level2, palettes

OUT = PROJECT / "output" / "loops"
PRODUCTS = {"BR": ("REF", "Base reflectivity", "dBZ"), "BV": ("VEL", "Base velocity", "m/s"),
            "SW": ("SW", "Spectrum width", "m/s"), "ZDR": ("ZDR", "Differential reflectivity", "dB"),
            "CC": ("RHO", "Correlation coefficient", ""), "PHI": ("PHI", "Differential phase", "deg")}


_ARCH = re.compile(r"^([A-Z0-9]{4})(\d{8})_(\d{6})_V\d\d$")
BG = (14, 16, 20)


def volume_time(name: str) -> dt.datetime:
    m = _ARCH.match(name)
    return dt.datetime.strptime(m[2] + m[3], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)


def sources(run_dir: Path) -> list[dict]:
    """The run's radars with Level II files: id, volumes (name, time), and the tilts of its newest volume."""
    out = []
    for d in sorted(p for p in Path(run_dir).iterdir() if p.is_dir()):
        vols = sorted(p.name for p in d.iterdir() if _ARCH.match(p.name) and p.name.startswith(d.name))
        if not vols:
            continue
        try:
            v = level2.read_volume((d / vols[-1]).read_bytes())
            tilts = sorted({round(float(s["elev"]), 1) for s in v["sweeps"].values()})
        except Exception:
            tilts = []
        out.append({"site": d.name, "tilts": tilts,
                    "volumes": [{"name": n, "time": volume_time(n).strftime("%Y-%m-%d %H:%M:%SZ")} for n in vols]})
    return out


# the tables a loop takes when none is picked: the NWS look, in GR's format
BUILTIN_PAL = {
    "BR": "Product: BR\nUnits: dBZ\nColor: 5 4 233 231\nColor: 10 1 159 244\nColor: 15 3 0 244\nColor: 20 2 253 2\n"
          "Color: 25 1 197 1\nColor: 30 0 142 0\nColor: 35 253 248 2\nColor: 40 229 188 0\nColor: 45 253 149 0\n"
          "Color: 50 253 0 0\nColor: 55 212 0 0\nColor: 60 188 0 0\nColor: 65 248 0 253\nColor: 70 152 84 198\n"
          "Color: 75 253 253 253\n",
    "BV": "Product: BV\nUnits: KTS\nScale: 1.943844\nColor: -120 252 0 130 109 0 150\nColor: -100 110 0 150 20 20 255\n"
          "Color: -80 20 20 255 0 220 255\nColor: -50 0 220 255 0 140 0\nColor: -30 0 140 0 0 250 0\n"
          "Color: -5 120 120 120 120 120 120\nColor: 5 230 0 0 140 0 0\nColor: 30 140 0 0 255 70 70\n"
          "Color: 50 255 70 70 255 250 0\nColor: 80 255 250 0 255 140 255\nColor: 100 255 140 255 255 255 255\nColor: 120 255 255 255\n",
    "SW": "Product: SW\nUnits: m/s\nColor: 0 60 60 60\nColor: 3 120 120 120\nColor: 6 0 180 0\nColor: 9 255 255 0\n"
          "Color: 12 255 128 0\nColor: 15 255 0 0\nColor: 20 255 0 255\n",
    "ZDR": "Product: ZDR\nUnits: dB\nColor: -4 0 0 0\nColor: -2 100 100 100\nColor: 0 200 200 200\nColor: 0.5 0 0 255\n"
           "Color: 1 0 200 255\nColor: 1.5 0 255 0\nColor: 2 255 255 0\nColor: 3 255 128 0\nColor: 4 255 0 0\n"
           "Color: 5 255 0 255\nColor: 8 255 255 255\n",
    "CC": "Product: CC\nUnits:\nColor: 0.2 20 0 50\nColor: 0.45 0 0 200\nColor: 0.65 0 200 255\nColor: 0.8 0 255 0\n"
          "Color: 0.9 255 255 0\nColor: 0.95 255 128 0\nColor: 0.97 255 0 0\nColor: 0.99 160 0 100\nColor: 1.05 255 200 255\n",
    "PHI": "Product: PHI\nUnits: deg\nColor: 0 0 0 120\nColor: 60 0 150 255\nColor: 120 0 220 0\nColor: 180 255 255 0\n"
           "Color: 240 255 100 0\nColor: 300 255 0 0\nColor: 360 255 255 255\n",
}


def _palette(path: str | None, product: str) -> dict:
    p = Path(path) if path else None
    if p is not None and p.is_file():
        return palettes.parse_pal(p.read_text(errors="replace"))
    return palettes.parse_pal(BUILTIN_PAL[product])


def colorize(vals: np.ndarray, pal: dict):
    """RGB (h, w, 3) and the mask of drawn pixels, the GR way: a stop's color runs to its second color (or the
    next stop's) across its interval; solid stops are flat; nothing below the first stop."""
    v = vals * float(pal.get("scale") or 1.0) + float(pal.get("offset") or 0.0)
    stops = pal["stops"]
    rgb = np.zeros(vals.shape + (3,), np.float32)
    drawn = np.isfinite(v) & (v >= stops[0]["v"])
    for i, s in enumerate(stops):
        b = stops[i + 1]["v"] if i + 1 < len(stops) else np.inf
        m = drawn & (v >= s["v"]) & (v < b)
        if not m.any():
            continue
        ca = np.asarray(s["c"][:3], np.float32)
        if s.get("solid") or not np.isfinite(b):
            rgb[m] = ca
            continue
        cb = np.asarray((s["c2"] or stops[i + 1]["c"])[:3], np.float32)
        t = np.clip((v[m] - s["v"]) / (b - s["v"]), 0.0, 1.0)[:, None]
        rgb[m] = ca + (cb - ca) * t
    return rgb.astype(np.uint8), drawn


def make(run_dir, site: str, product: str, tilt: float, first: str | None, last: str | None, box: dict,
         palette_path: str | None = None, size=(1280, 720), frame_ms: int = 250, tracks: bool = True,
         progress=None, frames_only: bool = False):
    """A GIF of ``site``'s ``product`` at ``tilt`` from volume ``first`` to ``last`` (names; None: the first /
    the last) over ``box`` ({cx, cy, w, h} world km, the map's view).  Returns its path (``frames_only``: the
    frames, PIL images, and no file -- the phone remote's picture)."""
    from PIL import Image, ImageDraw, ImageFont
    run_dir = Path(run_dir)
    mom, label, _ = PRODUCTS[product]
    pal = _palette(palette_path, product)
    st = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").is_file() else {}
    settings = st.get("settings") or json.loads((run_dir / "settings.json").read_text())
    proj = geo.Lambert(float(settings["map"]["lat0"]), float(settings["map"]["lon0"]))
    names = sorted(p.name for p in (run_dir / site).iterdir() if _ARCH.match(p.name) and p.name.startswith(site))
    i0 = names.index(first) if first in names else 0
    i1 = names.index(last) if last in names else len(names) - 1
    names = names[min(i0, i1):max(i0, i1) + 1]
    if not names:
        raise ValueError(f"{site} has no volumes in that range")
    W, H = int(size[0]), int(size[1])
    cx, cy = float(box["cx"]) * 1000.0, float(box["cy"]) * 1000.0
    half_w = float(box["w"]) * 500.0
    half_h = half_w * H / W
    X = cx + (np.arange(W) + 0.5 - W / 2) / W * 2 * half_w
    Y = cy - (np.arange(H) + 0.5 - H / 2) / H * 2 * half_h
    XX, YY = np.meshgrid(X, Y)
    lat, lon = proj.inv(XX.ravel(), YY.ravel())
    la = np.radians(np.asarray(lat, np.float64).reshape(XX.shape))
    lo = np.radians(np.asarray(lon, np.float64).reshape(XX.shape))

    def to_px(xkm, ykm):
        return (xkm * 1000 - cx) / (2 * half_w) * W + W / 2, H / 2 - (ykm * 1000 - cy) / (2 * half_h) * H

    # the map: county and state lines, towns, tornado tracks (once)
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    lines = geo.map_lines(proj, 2600.0)
    for layer, col, wd in (("counties", (120, 120, 120, 140), 1), ("states", (230, 230, 230, 230), 2)):
        for ln in lines.get(layer, []):
            xy = [to_px(ln[i], ln[i + 1]) for i in range(0, len(ln), 2)]
            if any(-50 < x < W + 50 and -50 < y < H + 50 for x, y in xy):
                d.line(xy, fill=col, width=wd)
    try:
        font, big = ImageFont.truetype("segoeui.ttf", 15), ImageFont.truetype("segoeuib.ttf", 19)
    except OSError:
        font = big = ImageFont.load_default()
    minpop = 300 if half_w < 60000 else 3000 if half_w < 200000 else 30000
    for c in geo.cities():
        if c["pop"] < minpop:
            continue
        x, y = proj.fwd(np.array([c["lat"]]), np.array([c["lon"]]))
        px, py = to_px(float(np.ravel(x)[0]) / 1000, float(np.ravel(y)[0]) / 1000)
        if 0 < px < W and 34 < py < H:
            d.rectangle([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255, 230))
            d.text((px + 5, py - 9), c["name"], fill=(255, 255, 255, 230), font=font)
    if tracks:
        for tr in st.get("tornado_tracks") or []:
            xy = [to_px(p[1] / 1000, p[2] / 1000) for p in tr["pts"]]
            if len(xy) > 1:
                d.line(xy, fill=(255, 255, 255, 170), width=1)

    frames = []
    for n, name in enumerate(names):
        v = level2.read_volume((run_dir / site / name).read_bytes())
        # the tilt nearest the one asked for that holds the moment (a split cut's surveillance sweep has no
        # velocity, its Doppler sweep -- on a 1990s radar -- no reflectivity)
        sws = [s for s in v["sweeps"].values() if mom in s["moments"]] or list(v["sweeps"].values())
        sw = min(sws, key=lambda s: abs(float(s["elev"]) - tilt))
        a = sw["moments"].get(mom)
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = BG
        # where the radar is: the volume says (message 31), else the run's radar list (a 1990s radar's message 1
        # radials carry no position), else the map's centre
        here = next(((float(r["lat"]), float(r["lon"])) for r in st.get("radars") or [] if r.get("icao") == site), None)
        slat, slon = sw.get("site") or here or (settings["map"]["lat0"], settings["map"]["lon0"])
        if a is not None:
            first_g, gate = sw.get("gates", {}).get(mom, (2125, 250))
            la0, lo0 = math.radians(slat), math.radians(slon)
            dlon = lo - lo0
            az = (np.degrees(np.arctan2(np.sin(dlon) * np.cos(la), math.cos(la0) * np.sin(la)
                                        - math.sin(la0) * np.cos(la) * np.cos(dlon))) + 360.0) % 360.0
            rng = 6371000.0 * np.arccos(np.clip(math.sin(la0) * np.sin(la) + math.cos(la0) * np.cos(la) * np.cos(dlon), -1, 1))
            rng = rng / math.cos(math.radians(float(sw["elev"])))
            azs = np.asarray(sw["az"], np.float64)
            order = np.argsort(azs)
            nrad, ngate = a.shape
            ia = np.searchsorted(azs[order], az) % nrad
            cand = np.stack([order[(ia - 1) % nrad], order[ia % nrad]])
            dif = np.abs(((azs[cand] - az[None]) + 180.0) % 360.0 - 180.0)
            ray = np.where(dif[0] < dif[1], cand[0], cand[1])
            g = np.round((rng - first_g) / gate).astype(np.int64)
            step = 360.0 / max(nrad, 1)
            ok = (g >= 0) & (g < ngate) & (np.minimum(dif[0], dif[1]) < 0.75 * step + 0.05)
            vals = np.full(az.shape, np.nan, np.float32)
            vals[ok] = a[ray[ok], g[ok]]
            rgb, drawn = colorize(vals, pal)
            img[drawn] = rgb[drawn]
        fr = Image.fromarray(img).convert("RGBA")
        fr.alpha_composite(base)
        dd = ImageDraw.Draw(fr)
        sx, sy = proj.fwd(np.array([slat]), np.array([slon]))
        px, py = to_px(float(np.ravel(sx)[0]) / 1000, float(np.ravel(sy)[0]) / 1000)
        if 0 < px < W and 0 < py < H:
            dd.rectangle([px - 5, py - 5, px + 5, py + 5], outline=(255, 210, 80, 255), width=2)
            dd.text((px + 8, py - 22), site, fill=(255, 210, 80, 255), font=big)
        dd.rectangle([0, 0, W, 32], fill=(0, 0, 0, 200))
        units = pal.get("units") or PRODUCTS[product][2]
        dd.text((10, 5), f"{site}  {label}{f' ({units})' if units else ''}  {float(sw['elev']):.1f}°   "
                         f"{volume_time(name):%Y-%m-%d %H:%M:%S} UTC", fill=(255, 255, 255, 255), font=big)
        frames.append(fr.convert("RGB"))
        if progress:
            progress(n + 1, len(names))
    if frames_only:
        return frames
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{run_dir.name} {site} {product} {tilt:.1f} {names[0][4:19]}-{names[-1][13:19]}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[int(frame_ms)] * (len(frames) - 1) + [1500],
                   loop=0)
    return out


class Loops:
    """Background loop jobs: start() returns a job id, status() its progress and, when done, its file."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, **kw) -> str:
        jid = f"{int(time.time() * 1000):x}"
        job = {"id": jid, "state": "running", "done": 0, "total": 0, "file": None, "error": None}
        with self.lock:
            self.jobs[jid] = job

        def prog(n, total):
            job["done"], job["total"] = n, total

        def run():
            try:
                job["file"] = str(make(progress=prog, **kw))
                job["state"] = "done"
            except Exception as e:
                job["state"], job["error"] = "error", str(e)
        threading.Thread(target=run, name=f"loop-{jid}", daemon=True).start()
        return jid

    def status(self, jid: str) -> dict:
        with self.lock:
            j = dict(self.jobs.get(jid) or {"state": "unknown"})
        if j.get("file"):
            j["name"] = Path(j["file"]).name
        return j


LOOPS = Loops()
