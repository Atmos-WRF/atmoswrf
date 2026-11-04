"""A run's tornado list and damage survey, from its state (state.json / Simulation.info()).

One rating everywhere: the map's.  Every point of a tornado's track carries the viewer's EF estimate
(radar.ef_estimate: the model's peak wind, the radar's 0.5-degree Vrot and the debris signature's CC drop, as
Simulation._record_tornadoes makes it), the map colours the track by it, and here the tornado's rating is its
highest and the survey is that same track in the same colours.  The list: every tornado named after the biggest
town it went through (else the town nearest its path; a second one for a town is #2), with its estimated peak
wind (the model's, held within its rating's EF range), peak Vrot, lowest CC, path length, start and end;
strongest first.

The survey: each track as a line, each stretch coloured and widened by its EF (as on the map), no polygons and no
damage indicators.  Exports: KML (Google Earth: the tracks with their facts) and a GRLevelX placefile (the
tracks, labels).  CSV and Markdown of the list."""
from __future__ import annotations

import csv
import datetime as dt
import io
import math
from xml.sax.saxutils import escape

import numpy as np

from . import geo

KT = 1.943844
TDS_CC = 0.80                 # a tornadic debris signature: CC this low in the tornado's echo (shown; rated in ef_estimate)
EF_MPH = {0: (65, 85), 1: (86, 110), 2: (111, 135), 3: (136, 165), 4: (166, 200), 5: (201, 250)}
# the map's EF colours (web/js/app.js EF_COLORS), (r, g, b)
EF_RGB = {0: (127, 220, 255), 1: (60, 207, 122), 2: (245, 211, 58), 3: (242, 145, 58), 4: (232, 65, 60), 5: (193, 58, 232)}
SMOOTH = 3                    # Chaikin passes over the track (the map's too)


def track_ef(tor) -> int:
    """The tornado's rating: the highest EF along its track, as the map labels it."""
    return max([max(int(p[3]), 0) for p in tor.get("pts") or []] or [max(int(tor.get("ef") or 0), 0)])


def wind_mph(ef, model_ms) -> int:
    """Estimated peak wind: the model's peak, held within the rating's EF range, to 5 mph."""
    lo, hi = EF_MPH[ef]
    v = 5 * round((model_ms or 0.0) * 2.23694 / 5.0)
    return int(min(max(v, 5 * math.ceil(lo / 5.0)), 5 * math.floor(hi / 5.0)))     # to 5 mph, inside the range


def _chaikin(P, E, n=SMOOTH):
    """The map's smoothing (app.js chaikin): each new point keeps the EF of the original point it came from."""
    for _ in range(n):
        if len(P) < 3:
            break
        a, b = P[:-1], P[1:]
        Q = np.empty((2 * len(a) + 2, 2))
        Q[0], Q[-1] = P[0], P[-1]
        Q[1:-1:2] = 0.75 * a + 0.25 * b
        Q[2:-1:2] = 0.25 * a + 0.75 * b
        F = np.empty(len(Q), int)
        F[0], F[-1] = E[0], E[-1]
        F[1:-1:2], F[2:-1:2] = E[:-1], E[1:]
        P, E = Q, F
    return P, E


def _coords(pts):
    return " ".join(f"{lo:.5f},{la:.5f},0" for la, lo in pts)


def _compass(dx, dy):
    return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][
        int(((math.degrees(math.atan2(dx, dy)) + 360) % 360 + 11.25) // 22.5) % 16]


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    u = np.clip(((px - ax) * dx + (py - ay) * dy) / L, 0.0, 1.0) if L > 0 else 0.0
    return np.hypot(px - (ax + u * dx), py - (ay + u * dy))


class Survey:
    """The list and the survey lines of one run's state."""

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
            ef, kt = track_ef(tor), (tor.get("vrot") or 0.0) * KT
            tds = tor.get("cc") is not None and tor["cc"] <= TDS_CC
            peak = max([float(p[4]) for p in tor["pts"] if len(p) > 4] or [float(tor.get("wind") or 0.0)])
            length = float(np.hypot(*np.diff(P, axis=0).T).sum()) / 1609.344 if len(P) > 1 else 0.0
            la, lo = self.proj.inv(P[:, 0], P[:, 1])
            la, lo = np.ravel(la).astype(float), np.ravel(lo).astype(float)
            where = ("through town" if big is not None else
                     f"{dist / 1609.344:.0f} mi {_compass(P.mean(0)[0] - self._cx[k], P.mean(0)[1] - self._cy[k])} of town"
                     if k is not None else "")
            rows.append({
                "id": tor["id"], "town": town, "where": where, "ef": ef, "wind_mph": wind_mph(ef, peak),
                "vrot_kt": round(kt), "cc": tor.get("cc"), "tds": tds, "model_ef": tor.get("ef"),
                "model_wind_mph": round(peak * 2.23694),
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

    # ---------------------------------------------------------------- the survey line
    def segments(self, r) -> list:
        """The track smoothed as the map draws it, cut where its EF changes: [(EF, [(lat, lon), ...]), ...] in
        order along the track, each piece starting where the one before ended."""
        pts = r["_tor"]["pts"]
        P = np.array([(p[1], p[2]) for p in pts], np.float64)
        E = np.array([max(int(p[3]), 0) for p in pts], int)
        if len(P) < 2:
            return []
        P, E = _chaikin(P, E)
        la, lo = self.proj.inv(P[:, 0], P[:, 1])
        L = [(round(float(a), 5), round(float(b), 5)) for a, b in zip(np.ravel(la), np.ravel(lo))]
        out, i = [], 0
        while i < len(L) - 1:
            j = i
            while j + 1 < len(L) - 1 and E[j + 1] == E[i]:
                j += 1
            out.append((int(E[i]), L[i:j + 2]))
            i = j + 1
        return out

    # ---------------------------------------------------------------- exports
    def summary(self) -> str:
        from collections import Counter
        n = Counter(r["ef"] for r in self.rows)
        return (f"{len(self.rows)} tornadoes: " + ", ".join(f"{n.get(e, 0)} EF{e}" for e in range(6))
                + (f". Longest track {max(r['miles'] for r in self.rows):.1f} mi." if self.rows else "."))

    def table(self) -> list:
        """The rows for the page (no internals), with each one's survey line."""
        out = []
        for r in self.rows:
            d = {k: v for k, v in r.items() if not k.startswith("_")}
            d["segments"] = self.segments(r)
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
            f'<Style id="ef{e}"><LineStyle><color>{colour(e)}</color><width>{2 + 1.2 * e:.1f}</width></LineStyle>'
            f'<IconStyle><color>{colour(e)}</color><scale>{0.8 + 0.1 * e:.1f}</scale><Icon><href>'
            'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>'
            for e in range(6)) + "".join(
            f'<Style id="case{e}"><LineStyle><color>8c000000</color><width>{4 + 1.2 * e:.1f}</width></LineStyle></Style>'
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
            segs = self.segments(r)
            whole = [q for k, (_, s) in enumerate(segs) for q in (s if k == 0 else s[1:])] or r["path"]
            pieces = "".join(
                f"<Placemark><name>EF{k}</name><styleUrl>#ef{k}</styleUrl><LineString><tessellate>1</tessellate>"
                f"<coordinates>{_coords(s)}</coordinates></LineString></Placemark>" for k, s in segs)
            la0, lo0 = r["path"][0]
            body.append(f"<Folder><name>{escape(label)}</name>"
                        f"<Placemark><name>{escape(label)}</name><description>{desc(r)}</description>"
                        f"<styleUrl>#ef{r['ef']}</styleUrl><Point><coordinates>{lo0:.5f},{la0:.5f},0</coordinates></Point></Placemark>"
                        f"<Folder><name>Track by EF</name>"
                        f"<Placemark><name>casing</name><styleUrl>#case{r['ef']}</styleUrl><LineString>"
                        f"<tessellate>1</tessellate><coordinates>{_coords(whole)}</coordinates></LineString></Placemark>"
                        f"{pieces}</Folder></Folder>")
        return ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                f"<name>{escape('Damage survey, run ' + self.run_id)}</name><description>{escape(self.summary())}</description>"
                f"{styles}{''.join(body)}</Document></kml>\n")

    def placefile(self, refresh: int = 60) -> str:
        """GRLevelX placefile: each track as on the map -- a dark casing, then each stretch in its EF's colour and
        width -- and a label at its start."""
        lines = [f"Title: Atmos-WRF damage survey ({self.run_id})", f"RefreshSeconds: {refresh}",
                 "Threshold: 999", ""]
        for r in self.rows:
            segs = self.segments(r)
            if not segs:
                continue
            whole = [q for k, (_, s) in enumerate(segs) for q in (s if k == 0 else s[1:])]
            lines.append("Color: 0 0 0 140")
            lines.append(f'Line: {4 + round(1.2 * r["ef"])}, 0, "{r["name"]} EF{r["ef"]}"')
            lines += [f" {la:.5f}, {lo:.5f}" for la, lo in whole]
            lines.append("End:")
            for k, s in segs:
                cr, cg, cb = EF_RGB[k]
                lines.append(f"Color: {cr} {cg} {cb} 255")
                lines.append(f'Line: {2 + round(1.2 * k)}, 0, "{r["name"]}: EF{k} here, EF{r["ef"]} overall"')
                lines += [f" {la:.5f}, {lo:.5f}" for la, lo in s]
                lines.append("End:")
            la0, lo0 = r["path"][0]
            cr, cg, cb = EF_RGB[r["ef"]]
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
