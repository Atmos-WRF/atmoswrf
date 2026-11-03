"""A run's tornado list and damage survey, from its state (state.json / Simulation.info()).

The list: every tornado, named after the biggest town it went through (else the town nearest its path; a second
one for a town is #2), rated as an NWS office does before a survey -- peak 0.5-degree Vrot for the rating, a
debris signature (low CC) as a floor, no EF4-5 without one -- with its estimated peak wind, path length, start and
end.  Listed strongest first.  (output/loop_tools/tor_list.py of 2026-10-28, folded into the viewer.)

The survey: each tornado's damage swath as nested EF contours, from the model's own winds along the track: a core
of radius 150-400 m (the radar's sub-grid tornado, radar.tornado_winds) with the wind falling as r^-0.8 outside it
(the same profile), so the EF-k contour's half width is where that wind drops to EF-k's threshold (radar.EF_WIND).
The rating caps the contours (the survey's rating is the list's).  No damage indicators.

Exports: KML (Google Earth: tracks with their facts, swath polygons by EF) and a GRLevelX placefile (polygons and
outlines in the NWS survey colours, labels)."""
from __future__ import annotations

import csv
import datetime as dt
import io
import math
from xml.sax.saxutils import escape

import numpy as np

from . import geo
from .radar import EF_WIND

KT = 1.943844
# peak 0.5 deg Vrot (kt) -> EF (WSR-88D tornado-intensity studies: Smith et al. 2015, 2020)
VROT_EF = [(74.0, 5), (63.0, 4), (52.0, 3), (42.0, 2), (33.0, 1)]
TDS_CC = 0.80                 # a tornadic debris signature: CC this low in the tornado's echo
TDS_STRONG = 0.60             # a strong one
EF_MPH = {0: (65, 85), 1: (86, 110), 2: (111, 135), 3: (136, 165), 4: (166, 200), 5: (201, 250)}
VROT_BAND = {0: (20.0, 33.0), 1: (33.0, 42.0), 2: (42.0, 52.0), 3: (52.0, 63.0), 4: (63.0, 74.0), 5: (74.0, 110.0)}
# the NWS damage survey colours (r, g, b)
EF_RGB = {0: (0, 180, 230), 1: (0, 200, 0), 2: (255, 255, 0), 3: (255, 128, 0), 4: (255, 0, 0), 5: (255, 0, 255)}
CORE_R = (150.0, 400.0)       # m: the core radius, weakest .. strongest tornado (35 .. 80 m/s)
DECAY = 0.8                   # the wind outside the core falls as r^-DECAY
SWATH_MAX = 2500.0            # m: no half width wider


def ef_radar(vrot_ms, cc):
    kt = (vrot_ms or 0.0) * KT
    ef = next((e for v, e in VROT_EF if kt >= v), 0)
    tds = cc is not None and cc <= TDS_CC
    if tds:
        ef = max(ef, 1)
    if cc is not None and cc <= TDS_STRONG:
        ef = max(ef, 2)
    if not tds:
        ef = min(ef, 3)
    return ef, kt, tds


def ef_wind_mph(ef, kt):
    lo, hi = EF_MPH[ef]
    a, b = VROT_BAND[ef]
    f = min(max((kt - a) / (b - a), 0.0), 1.0)
    return int(5 * round((lo + f * (hi - lo)) / 5.0))


def _compass(dx, dy):
    return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][
        int(((math.degrees(math.atan2(dx, dy)) + 360) % 360 + 11.25) // 22.5) % 16]


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    u = np.clip(((px - ax) * dx + (py - ay) * dy) / L, 0.0, 1.0) if L > 0 else 0.0
    return np.hypot(px - (ax + u * dx), py - (ay + u * dy))


class Survey:
    """The list and the swaths of one run's state."""

    def __init__(self, state: dict):
        self.state = state
        self.run_id = state.get("run_id", "run")
        p = state["proj"]
        self.proj = geo.Lambert(p["lat0"], p["lon0"], p.get("truelat1", 33.0), p.get("truelat2", 45.0))
        self.start = dt.datetime.fromisoformat(state["settings"]["start_time"].replace("Z", "+00:00"))
        cs = [c for c in geo.cities() if c["pop"] >= 100]
        self._cs = cs
        self._cx, self._cy = self.proj.fwd(np.array([c["lat"] for c in cs]), np.array([c["lon"] for c in cs]))
        self._pop = np.array([c["pop"] for c in cs])
        self._town_r = np.clip(0.35 + 0.45 * np.log10(np.maximum(self._pop, 300.0) / 1000.0), 0.3, 2.5) * 1000.0 + 500.0
        self.rows = self._rows()

    # ---------------------------------------------------------------- the list
    def _towns_on(self, P):
        cx, cy = self._cx, self._cy
        sel = np.where((cx > P[:, 0].min() - 60e3) & (cx < P[:, 0].max() + 60e3) &
                       (cy > P[:, 1].min() - 60e3) & (cy < P[:, 1].max() + 60e3))[0]
        if not len(sel):
            return None, None, None
        if len(P) == 1:
            d = np.hypot(cx[sel] - P[0, 0], cy[sel] - P[0, 1])
        else:
            d = np.full(len(sel), np.inf)
            for a, b in zip(P[:-1], P[1:]):
                d = np.minimum(d, _seg_dist(cx[sel], cy[sel], a[0], a[1], b[0], b[1]))
        hit = sel[d <= self._town_r[sel]]
        big = int(hit[np.argmax(self._pop[hit])]) if len(hit) else None
        k = int(np.argmin(d))
        return big, int(sel[k]), float(d[k])

    def _rows(self):
        rows = []
        for tor in sorted(self.state.get("tornado_tracks") or [], key=lambda t: t["t0"]):
            P = np.array([(p[1], p[2]) for p in tor["pts"]], np.float64)
            big, near, dist = self._towns_on(P)
            k = big if big is not None else near
            town = f"{self._cs[k]['name']}, {self._cs[k]['state']}" if k is not None else "open country"
            ef, kt, tds = ef_radar(tor.get("vrot"), tor.get("cc"))
            length = float(np.hypot(*np.diff(P, axis=0).T).sum()) / 1609.344 if len(P) > 1 else 0.0
            la, lo = self.proj.inv(P[:, 0], P[:, 1])
            la, lo = np.ravel(la).astype(float), np.ravel(lo).astype(float)
            where = ("through town" if big is not None else
                     f"{dist / 1609.344:.0f} mi {_compass(P.mean(0)[0] - self._cx[k], P.mean(0)[1] - self._cy[k])} of town"
                     if k is not None else "")
            rows.append({
                "id": tor["id"], "town": town, "where": where, "ef": ef, "wind_mph": ef_wind_mph(ef, kt),
                "vrot_kt": round(kt), "cc": tor.get("cc"), "tds": tds, "model_ef": tor.get("ef"),
                "model_wind_mph": round(float(tor.get("wind") or 0.0) * 2.23694),
                "start_utc": (self.start + dt.timedelta(seconds=tor["t0"])).strftime("%Y-%m-%d %H:%MZ"),
                "end_utc": (self.start + dt.timedelta(seconds=tor["t1"])).strftime("%H:%MZ"),
                "minutes": round((tor["t1"] - tor["t0"]) / 60.0, 1), "miles": round(length, 1),
                "start_latlon": f"{la[0]:.3f}, {lo[0]:.3f}", "end_latlon": f"{la[-1]:.3f}, {lo[-1]:.3f}",
                "active": bool(tor.get("active")) and self.state.get("status") in ("running", "paused", "waiting"), "path": [[round(a, 5), round(b, 5)] for a, b in zip(la, lo)],
                "x0": float(P[:, 0].min()), "y0": float(P[:, 1].min()), "x1": float(P[:, 0].max()), "y1": float(P[:, 1].max()),
                "_tor": tor})
        seen = {}
        for r in rows:
            seen[r["town"]] = seen.get(r["town"], 0) + 1
            r["name"] = r["town"] if seen[r["town"]] == 1 else f"{r['town']} #{seen[r['town']]}"
        for r in rows:
            if seen[r["town"]] > 1 and r["name"] == r["town"]:
                r["name"] = f"{r['town']} #1"
        rows.sort(key=lambda r: (-r["ef"], -r["vrot_kt"], -r["miles"]))
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        return rows

    # ---------------------------------------------------------------- the swaths
    def swaths(self, r) -> dict:
        """{EF: [rings]}: each ring a closed list of (lat, lon) around the stretch of track where the wind reaches
        that rating, EF0 outermost."""
        pts = r["_tor"]["pts"]
        P = np.array([(p[1], p[2]) for p in pts], np.float64)
        wind = np.array([p[4] if len(p) > 4 else EF_WIND[max(int(p[3]), 0)] + 5.0 for p in pts], np.float64)
        if len(P) < 2:
            return {}
        top = int(r["ef"])
        d = np.gradient(P, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n /= np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-6)[:, None]
        s = np.clip((wind - 35.0) / 45.0, 0.0, 1.0)
        rc = CORE_R[0] + (CORE_R[1] - CORE_R[0]) * s
        out = {}
        for k in range(0, top + 1):
            vk = EF_WIND[k]
            hw = np.where(wind >= vk, np.minimum(rc * (np.maximum(wind, vk) / vk) ** (1.0 / DECAY), SWATH_MAX), 0.0)
            if k == top:                              # the rating's own contour: at least where the track is that strong
                hw = np.where(hw > 0, hw, 0.0)
            rings = []
            on = hw > 0
            i = 0
            while i < len(P):
                if not on[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(P) and on[j + 1]:
                    j += 1
                rings.append(self._ring(P[i:j + 1], n[i:j + 1], hw[i:j + 1]))
                i = j + 1
            if rings:
                out[k] = rings
        return out

    def _ring(self, P, n, hw):
        left = P + n * hw[:, None]
        right = P - n * hw[:, None]
        # round ends: half circles at the stretch's first and last points
        def cap(c, nn, w, sign):
            a0 = math.atan2(nn[1], nn[0])
            return [(c[0] + w * math.cos(a0 + sign * t), c[1] + w * math.sin(a0 + sign * t))
                    for t in np.linspace(0, math.pi, 9)[1:-1]]
        ring = list(map(tuple, left)) + cap(P[-1], n[-1], hw[-1], -1.0) + list(map(tuple, right[::-1])) + \
            cap(P[0], -n[0], hw[0], -1.0)
        ring.append(ring[0])
        xy = np.array(ring)
        la, lo = self.proj.inv(xy[:, 0], xy[:, 1])
        return [(round(float(a), 5), round(float(b), 5)) for a, b in zip(np.ravel(la), np.ravel(lo))]

    # ---------------------------------------------------------------- exports
    def summary(self) -> str:
        from collections import Counter
        n = Counter(r["ef"] for r in self.rows)
        return (f"{len(self.rows)} tornadoes: " + ", ".join(f"{n.get(e, 0)} EF{e}" for e in range(6))
                + (f". Longest track {max(r['miles'] for r in self.rows):.1f} mi." if self.rows else "."))

    def table(self) -> list:
        """The rows for the page (no internals), with each one's swaths."""
        out = []
        for r in self.rows:
            d = {k: v for k, v in r.items() if not k.startswith("_")}
            d["swaths"] = {str(k): v for k, v in self.swaths(r).items()}
            out.append(d)
        return out

    def csv(self) -> str:
        cols = ["rank", "name", "ef", "wind_mph", "vrot_kt", "cc", "tds", "where", "start_utc", "end_utc", "minutes",
                "miles", "start_latlon", "end_latlon", "model_ef", "model_wind_mph", "id"]
        f = io.StringIO()
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(self.rows)
        return f.getvalue()

    def kml(self) -> str:
        def colour(ef, a="ff"):
            r_, g, b = EF_RGB[ef]
            return f"{a}{b:02x}{g:02x}{r_:02x}"
        styles = "".join(
            f'<Style id="ef{e}"><LineStyle><color>{colour(e)}</color><width>{2 + e * 0.6:.1f}</width></LineStyle>'
            f'<PolyStyle><color>{colour(e, "66")}</color></PolyStyle>'
            f'<IconStyle><color>{colour(e)}</color><scale>{0.8 + 0.1 * e:.1f}</scale><Icon><href>'
            'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>'
            for e in range(6))

        def desc(r):
            cc = "—" if r["cc"] is None else f"{r['cc']:.2f}" + (" (debris signature)" if r["tds"] else "")
            items = [("Rank", f"{r['rank']} of {len(self.rows)}"), ("Rating", f"EF{r['ef']}"),
                     ("Estimated peak wind", f"{r['wind_mph']} mph"), ("Peak Vrot (0.5°)", f"{r['vrot_kt']} kt"),
                     ("Lowest CC", cc), ("Where", r["where"]), ("Start", r["start_utc"]), ("End", r["end_utc"]),
                     ("Duration", f"{r['minutes']:g} min"), ("Path length", f"{r['miles']:g} mi"),
                     ("Model peak wind", f"{r['model_wind_mph']} mph")]
            return "<![CDATA[<table>" + "".join(f"<tr><td><b>{escape(a)}</b></td><td>{escape(str(b))}</td></tr>"
                                                for a, b in items) + "</table>]]>"
        body = []
        for r in self.rows:
            label = f"{r['name']} (EF{r['ef']}, {r['wind_mph']} mph)"
            coords = " ".join(f"{lo:.5f},{la:.5f},0" for la, lo in r["path"])
            sw = self.swaths(r)
            polys = "".join(
                f"<Placemark><name>EF{k} damage</name><styleUrl>#ef{k}</styleUrl><Polygon><tessellate>1</tessellate>"
                f"<outerBoundaryIs><LinearRing><coordinates>{' '.join(f'{lo:.5f},{la:.5f},0' for la, lo in ring)}"
                "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
                for k in sorted(sw) for ring in sw[k])
            la0, lo0 = r["path"][0]
            body.append(f"<Folder><name>{escape(label)}</name>"
                        f"<Placemark><name>{escape(label)}</name><description>{desc(r)}</description>"
                        f"<styleUrl>#ef{r['ef']}</styleUrl><Point><coordinates>{lo0:.5f},{la0:.5f},0</coordinates></Point></Placemark>"
                        f"<Placemark><name>{escape(label)} track</name><styleUrl>#ef{r['ef']}</styleUrl><LineString>"
                        f"<tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString></Placemark>"
                        f"<Folder><name>Damage swath</name>{polys}</Folder></Folder>")
        return ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                f"<name>{escape('Damage survey, run ' + self.run_id)}</name><description>{escape(self.summary())}</description>"
                f"{styles}{''.join(body)}</Document></kml>\n")

    def placefile(self, refresh: int = 60) -> str:
        """GRLevelX placefile: each swath's EF contours filled (outermost first, so the stronger lie on top) and
        outlined, the track, and a label at its start."""
        lines = [f"Title: Atmos-WRF damage survey ({self.run_id})", f"RefreshSeconds: {refresh}",
                 "Threshold: 999", ""]
        for r in self.rows:
            sw = self.swaths(r)
            for k in sorted(sw):
                cr, cg, cb = EF_RGB[k]
                for ring in sw[k]:
                    lines.append("Polygon:")
                    lines.append(f" {ring[0][0]:.5f}, {ring[0][1]:.5f}, {cr}, {cg}, {cb}, 70")
                    lines += [f" {la:.5f}, {lo:.5f}" for la, lo in ring[1:]]
                    lines.append("End:")
                    lines.append(f"Color: {cr} {cg} {cb} 255")
                    lines.append(f'Line: 2, 0, "EF{k} damage, {r["name"]}"')
                    lines += [f" {la:.5f}, {lo:.5f}" for la, lo in ring]
                    lines.append("End:")
            cr, cg, cb = EF_RGB[r["ef"]]
            lines.append("Color: 255 255 255 255")
            lines.append(f'Line: 1, 0, "{r["name"]} EF{r["ef"]} track"')
            lines += [f" {la:.5f}, {lo:.5f}" for la, lo in r["path"]]
            lines.append("End:")
            la0, lo0 = r["path"][0]
            lines.append(f"Color: {cr} {cg} {cb} 255")
            lines.append(f'Text: {la0:.5f}, {lo0:.5f}, 1, "EF{r["ef"]}", "{r["name"]}: EF{r["ef"]}, {r["wind_mph"]} mph, '
                         f'{r["miles"]:g} mi, {r["start_utc"]}"')
            lines.append("")
        return "\n".join(lines) + "\n"

    def markdown(self) -> str:
        out = [f"# Tornadoes of run {self.run_id}", "", self.summary(), "",
               "| # | Tornado | EF | Wind | Vrot | Min CC | Where | Start | Min | Miles |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in self.rows:
            cc = "—" if r["cc"] is None else f"{r['cc']:.2f}" + (" TDS" if r["tds"] else "")
            out.append(f"| {r['rank']} | {r['name']} | EF{r['ef']} | {r['wind_mph']} mph | {r['vrot_kt']} kt | {cc} | "
                       f"{r['where']} | {r['start_utc']} | {r['minutes']:g} | {r['miles']:g} |")
        return "\n".join(out) + "\n"

    def write(self, folder) -> list:
        """The list (csv, md), the survey (kml, placefile) into ``folder``."""
        from pathlib import Path
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        files = {"tornadoes.csv": self.csv(), "tornadoes.md": self.markdown(), "damage_survey.kml": self.kml(),
                 "damage_survey_placefile.txt": self.placefile(refresh=0)}
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8")
        return [str(d / n) for n in files]
