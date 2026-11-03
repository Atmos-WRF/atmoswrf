"""Automatic warnings, the way a forecaster would issue them on the run's storms: a polygon downstream
along each storm's motion, valid for a while, for

* SPS  special weather statement: strong, not yet severe (40+ mph wind, half-inch hail)
* SVR  severe thunderstorm warning: 58+ mph wind or 1"+ hail; Considerable from 70 mph / 1.75",
       Destructive from 80 mph / 2.75"
* TOR  tornado warning: radar indicated (strong low-level rotation under a mature storm) or observed
       (a confirmed tornado); PDS (considerable damage) for an EF2+ tornado; a tornado emergency for
       an EF3+ one heading into a town, or EF4+ anywhere

All of it from the model, every WARN_EVERY s, for storms followed at least MIN_AGE: the lowest level's
peak wind, the 0-3 km reflectivity for hail (Thompson's graupel reads 66-69 dBZ aloft in most storms,
which would make every one a baseball-hail storm), the low-level rotation (radar.lowlevel_vrot) and the tornado tracker's confirmed tornadoes, within
STORM_R of each followed updraft.  A storm keeps its warning while it stays inside the polygon; once it
leaves it, or the warning runs out while it still qualifies, a new one is issued ahead of it.  A tornado
warning is upgraded to PDS / emergency in place, as a severe weather statement would.  Drawn in
GR2Analyst's NWS colours (a colour line in a black casing, heavier as the product escalates) as a
placefile with a TimeRange per warning in the polling feed's clock.
"""
from __future__ import annotations

import datetime as dt
import math
import random
import time

import numpy as np

WARN_EVERY = 60.0           # s of model time between looks at the storms
STORM_R = 12000.0           # m around each updraft
MS_MPH = 2.23694
MIN_AGE = 600.0             # s a storm has been followed before it is warned
# hail size (in) from the 0-3 km maximum reflectivity (dBZ)
HAIL = ((70.0, 2.75), (67.0, 2.50), (65.0, 1.75), (62.0, 1.25), (60.0, 1.00), (57.0, 0.75), (53.0, 0.50))
RANK = {"SPS": 0, "SVR": 1, "TOR": 2}
# (minutes valid, back half width km, front half width km, km behind the anchor): SPS / SVR around the
# storm's forward-flank precipitation, TOR around the tornado or the low-level rotation
SHAPE = {"SPS": (30, 14.0, 22.0, 12.0), "SVR": (30, 14.0, 24.0, 12.0), "TOR": (20, 5.0, 10.0, 4.0)}
TOR_LEN = (12000.0, 22000.0)    # m: a tornado warning's length ahead of the tornado, whatever the storm's speed
TOR_R = 5000.0              # m: a storm's own low-level rotation, around its updraft (a left split next to a
                            # rotating right mover must not take the neighbour's meso for its own)
TOR_SAME = 10000.0          # m: a tornado warning this close to the circulation already covers it
FFD_R = 25000.0             # m around the updraft where its forward flank's echo is looked for
FFD_DBZ = 40.0              # dBZ at the lowest level: the forward flank's core
TOR_EXTEND = 0.6            # chance a still-tornadic storm's warning is extended in its last 5 minutes
TOR_EXTEND_LEFT = 300.0     # s
TOR_VROT = 12.0             # m/s of low-level Vrot for a radar-indicated tornado warning
TOR_W = 20.0                # m/s: under a mature updraft
TOR_AGE = 1200.0            # s
EMERGENCY_TOWN = 2500       # people: a town an EF3+ tornado is heading into
COLORS = {"SPS": (255, 228, 181), "SVR": (255, 255, 0), "TOR": (255, 0, 0), "TORR": (255, 0, 255)}
TITLES = {"SPS": "SPECIAL WEATHER STATEMENT", "SVR": "SEVERE THUNDERSTORM WARNING", "TOR": "TORNADO WARNING"}


def hail_size(dbz: float) -> float:
    return next((s for d, s in HAIL if dbz >= d), 0.0)


def hail_name(inch: float) -> str:
    return {0.5: "half inch", 0.75: "penny size", 1.0: "quarter size", 1.25: "half dollar size", 1.75: "golf ball size",
            2.5: "tennis ball size", 2.75: "baseball size"}.get(inch, f"{inch:.2f} inch")


def _inside(poly: np.ndarray, x: float, y: float) -> bool:
    n, c = len(poly), False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


class WarningCenter:
    def __init__(self, sim):
        self.sim = sim
        self.items: list[dict] = []
        self._next = 1
        self._last = -1e9
        self._rng = random.Random(int(sim.settings.get("seed", 1) or 1) * 31 + 7)
        self.rm = None              # the Bunkers right mover: a storm's motion until its track gives one

    def _motion_default(self):
        if self.rm is None:
            try:
                from .sounding import diagnostics
                self.rm = tuple(diagnostics(self.sim.sounding)["rm"])
            except Exception:
                self.rm = (10.0, 5.0)
        return self.rm

    # ------------------------------------------------------------ the storms
    def _assess(self, fields: dict, t: float) -> list[dict]:
        import cupy as cp
        from . import radar as rd
        sim, eng = self.sim, self.sim.engine
        self._motion_default()
        z_h = np.asarray(sim.env.z_h)
        k2, k7 = 0, int(np.searchsorted(z_h, 3000.0))
        vrots = {}
        out = []
        tors = [c for c in sim.tornadoes.items if c.found and c.seen >= rd.TORNADO_CONFIRM]
        live = [tr for tr in eng.tracks.live if t - tr.last[0] <= 180.0]
        # each tornado belongs to the nearest storm only
        owner = {}
        for c in tors:
            near = min(live, key=lambda tr: math.hypot(c.x - tr.last[1], c.y - tr.last[2]), default=None)
            if near is not None and math.hypot(c.x - near.last[1], c.y - near.last[2]) < STORM_R:
                owner.setdefault(near.id, []).append(c)
        for tr in eng.tracks.live:
            lt, x, y, w = tr.last
            if t - lt > 180.0 or t - tr.born < MIN_AGE:
                continue
            f = next((q for q in fields.values() if q.x0 <= x < q.x0 + q.nx * q.dx and q.y0 <= y < q.y0 + q.ny * q.dx), None)
            if f is None:
                continue
            dx = f.dx
            i, j, n = int((x - f.x0) / dx), int((y - f.y0) / dx), int(STORM_R / dx)
            sl = (slice(max(j - n, 0), j + n + 1), slice(max(i - n, 0), i + n + 1))
            wind = float(cp.hypot(f.uc[0][sl], f.vc[0][sl]).max())
            dbz = float(f.dbz[k2:k7 + 1][(slice(None),) + sl].max())
            if id(f) not in vrots:
                vrots[id(f)] = rd.lowlevel_vrot(f.uc, f.vc, z_h, dx)
            nr = max(int(TOR_R / dx), 1)
            vrot = float(vrots[id(f)][max(j - nr, 0):j + nr + 1, max(i - nr, 0):i + nr + 1].max())   # cyclonic only
            tor = owner.get(tr.id, [])
            twind = max((c.ve if c.ve is not None else rd.DEBRIS_V0 + c.s * (rd.DEBRIS_V1 - rd.DEBRIS_V0)) for c in tor) if tor else 0.0
            vx, vy = tr.velocity(default=self.rm)
            # the forward flank: the reflectivity-weighted middle of the lowest level's 40+ dBZ echo near it
            nf = int(FFD_R / dx)
            j0, i0 = max(j - nf, 0), max(i - nf, 0)
            low = f.dbz[0][j0:j + nf + 1, i0:i + nf + 1]
            zl = cp.where(low >= FFD_DBZ, 10.0 ** (0.1 * low), 0.0)
            tot = float(zl.sum())
            if tot > 0.0:
                jj, ii = cp.mgrid[0:low.shape[0], 0:low.shape[1]]
                cx = f.x0 + (i0 + float((zl * ii).sum()) / tot + 0.5) * dx
                cy = f.y0 + (j0 + float((zl * jj).sum()) / tot + 0.5) * dx
            else:
                cx, cy = x, y
            # a tornado warning sits on the tornado, else on the storm's updraft and its rotation
            best = max(tor, key=lambda c: c.s) if tor else None
            tx, ty = (best.x, best.y) if best is not None else (x, y)
            out.append(dict(storm=tr.id, x=x, y=y, vx=vx, vy=vy, w=w, age=t - tr.born, wind=wind, dbz=dbz,
                            hail=hail_size(dbz), vrot=vrot, tornado=bool(tor), twind=twind, cx=cx, cy=cy,
                            tx=float(tx), ty=float(ty)))
        return out

    # ------------------------------------------------------------ products
    def _product(self, s: dict) -> tuple[str, list] | None:
        k = self.sim.storm_scale(s["x"], s["y"])          # low-CAPE storms: sim.STORM_W_GAIN
        if s["tornado"] or (s["vrot"] >= TOR_VROT and s["w"] >= TOR_W * k and s["age"] >= TOR_AGE * k):
            tags = ["observed" if s["tornado"] else "radar indicated"]
            if s["tornado"] and s["twind"] >= 50.0:
                tags.append("PDS")
            return "TOR", tags
        mph, hail = s["wind"] * MS_MPH, s["hail"]
        if mph >= 58 or hail >= 1.0:
            tag = "destructive" if (mph >= 80 or hail >= 2.75) else "considerable" if (mph >= 70 or hail >= 1.75) else ""
            return "SVR", [tag] if tag else []
        if mph >= 40 or hail >= 0.5:
            return "SPS", []
        return None

    def _polygon(self, kind: str, s: dict) -> np.ndarray:
        minutes, hb, hf, behind = SHAPE[kind]
        ax, ay = (s["tx"], s["ty"]) if kind == "TOR" else (s["cx"], s["cy"])
        sp = math.hypot(s["vx"], s["vy"])
        ux, uy = (s["vx"] / sp, s["vy"] / sp) if sp > 1.0 else (self.rm[0] / max(math.hypot(*self.rm), 1e-6),
                                                                self.rm[1] / max(math.hypot(*self.rm), 1e-6))
        if kind == "TOR":
            L = min(max(0.7 * max(sp, 8.0) * minutes * 60.0, TOR_LEN[0]), TOR_LEN[1])
        else:
            L = max(sp, 8.0) * minutes * 60.0 + 10000.0
        nx, ny = -uy, ux
        bx, by = ax - ux * behind * 1000.0, ay - uy * behind * 1000.0
        fx, fy = ax + ux * L, ay + uy * L
        return np.array([[bx + nx * hb * 1e3, by + ny * hb * 1e3], [fx + nx * hf * 1e3, fy + ny * hf * 1e3],
                         [fx - nx * hf * 1e3, fy - ny * hf * 1e3], [bx - nx * hb * 1e3, by - ny * hb * 1e3]])

    def _emergency(self, poly: np.ndarray, s: dict) -> bool:
        if s["twind"] >= 74.0:
            return True
        if s["twind"] < 61.0:
            return False
        towns = getattr(self.sim.env, "towns", None)
        if towns is None or not len(towns):
            return False
        big = towns[towns[:, 2] >= EMERGENCY_TOWN]
        return any(_inside(poly, float(tx), float(ty)) for tx, ty, _ in big)

    def step(self, fields: dict, t: float) -> None:
        if t - self._last < WARN_EVERY:
            return
        self._last = t
        now = time.time()
        for w in self.items:                              # run out
            if w["active"] and t >= w["t1"]:
                w["active"], w["wall1"] = False, now
        try:
            storms = self._assess(fields, t)
        except Exception as exc:                          # never stop a run over a warning
            self.sim.engine.note("warning", f"warnings skipped this minute: {exc}")
            return
        for s in storms:
            got = self._product(s)
            if got is None:
                continue
            kind, tags = got
            px, py = (s["tx"], s["ty"]) if kind == "TOR" else (s["cx"], s["cy"])
            mine = [w for w in self.items if w["active"] and w["storm"] == s["storm"] and w["kind"] == kind]
            # a storm already inside a warning of this kind (or a higher one, for SPS / SVR) is covered by
            # it, whichever storm it was issued for: one polygon for a cluster, as forecasters draw them
            cover = [w for w in self.items if w["active"] and (_inside(w["poly"], px, py) or (
                         kind == "TOR" and w["kind"] == "TOR"
                         and math.hypot(w["anchor"][0] - px, w["anchor"][1] - py) < TOR_SAME))
                     and (w["kind"] == kind or (kind != "TOR" and RANK[w["kind"]] > RANK[kind]))]
            if cover and not (mine and _inside(mine[-1]["poly"], px, py)):
                if kind == "TOR":
                    mine = [w for w in cover if w["kind"] == "TOR"]
                else:
                    continue
            # an upgrade ends the storm's lower products (its SPS once it is warned, both once a tornado is)
            for low in self.items:
                if low["active"] and low["storm"] == s["storm"] and RANK[low["kind"]] < RANK[kind]:
                    self._end(low, now)
            if mine and _inside(mine[-1]["poly"], px, py):
                w = mine[-1]
                new_tags = self._merged_tags(kind, w["tags"], tags, w["poly"], s)
                if kind == "TOR" and w["t1"] - t <= TOR_EXTEND_LEFT and self._rng.random() < TOR_EXTEND:
                    self._issue(kind, sorted(set(new_tags) | {"extended"}), s, t, now, replaces=w)
                elif self._rank_tags(kind, new_tags) > self._rank_tags(kind, w["tags"]):
                    # a higher tag (considerable, destructive, observed, PDS, emergency) is a new warning on
                    # the same polygon until the old one's end; the old one disappears
                    self._issue(kind, new_tags, s, t, now, replaces=w, poly=w["poly"], t1=w["t1"])
                else:
                    w["hazard"] = self._hazard(kind, s, w["tags"])
                continue
            for old in mine:                            # left its polygon: a new one ahead of it
                self._end(old, now)
            if kind == "TOR" and self._emergency(self._polygon(kind, s), s):
                tags = tags + ["emergency"]
            self._issue(kind, tags, s, t, now)

    # ------------------------------------------------------------ issuing
    @staticmethod
    def _end(w: dict, now: float) -> None:
        w["active"], w["wall1"] = False, now

    def _issue(self, kind: str, tags: list, s: dict, t: float, now: float, replaces: dict | None = None,
               poly=None, t1: float | None = None) -> None:
        if replaces is not None:
            self._end(replaces, now)
        poly = self._polygon(kind, s) if poly is None else poly
        t1 = t + SHAPE[kind][0] * 60.0 if t1 is None else t1
        anchor = (s["tx"], s["ty"]) if kind == "TOR" else (s["cx"], s["cy"])
        w = dict(id=self._next, kind=kind, tags=list(tags), storm=s["storm"], t0=t, t1=t1, wall0=now, wall1=None,
                 active=True, poly=poly, anchor=anchor, hazard=self._hazard(kind, s, tags))
        self.items.append(w)
        self._next += 1
        how = ""
        if replaces is not None:
            how = f" (extends #{replaces['id']})" if "extended" in tags else f" (replaces #{replaces['id']})"
        self.sim.engine.note("warning", f"{self.code(w)} #{w['id']} for storm {s['storm']}{how} ({w['hazard']})")

    def _merged_tags(self, kind: str, old: list, new: list, poly, s: dict) -> list:
        up = set(old) | set(new)
        up.discard("extended")
        if kind == "TOR":
            if "observed" in up:
                up.discard("radar indicated")
            if self._emergency(poly, s):
                up.add("emergency")
        elif kind == "SVR" and "destructive" in up:
            up.discard("considerable")
        return sorted(up)

    @staticmethod
    def _rank_tags(kind: str, tags: list) -> int:
        if kind == "TOR":
            return 3 if "emergency" in tags else 2 if "PDS" in tags else 1 if "observed" in tags else 0
        if kind == "SVR":
            return 2 if "destructive" in tags else 1 if "considerable" in tags else 0
        return 0

    @staticmethod
    def _hazard(kind: str, s: dict, tags: list) -> str:
        mph = int(round(s["wind"] * MS_MPH / 5.0) * 5)
        hail = hail_name(s["hail"]) + " hail" if s["hail"] else ""
        if kind == "TOR":
            return ("tornado" + (" (observed)" if "observed" in tags else " (radar indicated rotation)")
                    + (f", {hail}" if hail else ""))
        parts = []
        if mph >= 40:
            parts.append(f"{mph} mph wind gusts")
        if hail:
            parts.append(hail)
        return " and ".join(parts) or "strong winds"

    @staticmethod
    def code(w: dict) -> str:
        if w["kind"] == "TOR":
            return "TORE" if "emergency" in w["tags"] else "TORP" if "PDS" in w["tags"] else \
                "TORR" if "observed" in w["tags"] else "TOR"
        if w["kind"] == "SVR":
            return "SVRD" if "destructive" in w["tags"] else "SVRC" if "considerable" in w["tags"] else "SVR"
        return "SPS"

    # ------------------------------------------------------------ output
    def to_json(self) -> list:
        return [dict(id=w["id"], code=self.code(w), kind=w["kind"], tags=w["tags"], storm=w["storm"], t0=w["t0"],
                     t1=w["t1"], active=w["active"], hazard=w["hazard"],
                     poly=[[round(float(x) / 1000, 2), round(float(y) / 1000, 2)] for x, y in w["poly"]])
                for w in self.items]

    def placefile(self, proj, start_epoch: float, speed: float, refresh: int = 15) -> str:
        """GRLevelX placefile: every warning of the run, each shown only while it was valid."""
        now = time.time()
        at = lambda e: dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=float(e))   # before 1970 too
        iso = lambda e: at(e).strftime("%Y-%m-%dT%H:%M:%SZ")
        hhmm = lambda tm: at(start_epoch + tm).strftime("%H%MZ")
        sim_t = float(getattr(self.sim.engine, "t", 0.0)) if self.sim.engine else 0.0
        lines = ["; Atmos-WRF: automatic warnings (SPS, SVR, TOR, PDS, tornado emergency)",
                 "Title: Atmos-WRF warnings", f"RefreshSeconds: {int(refresh)}", "Threshold: 999"]
        order = {"SPS": 0, "SVR": 1, "TOR": 2}
        for w in sorted(self.items, key=lambda w: order[w["kind"]]):
            code = self.code(w)
            if w["active"]:
                end = now + max(w["t1"] - sim_t, 0.0) / (speed if speed > 0 else 1.0) + 5.0
            else:
                end = w["wall1"] or now
            tier = {"SPS": 0, "SVR": 1, "SVRC": 2, "SVRD": 3, "TOR": 1, "TORR": 2, "TORP": 3, "TORE": 4}[code]
            color = COLORS["TORR" if code in ("TORR", "TORP", "TORE") else w["kind"]]
            width, casing = (2, 1) if tier == 0 else (3, tier if tier < 4 else 3)
            if code == "TORE":
                width = 4
            label = (f"{TITLES[w['kind']]}" + (" (EXTENDED)" if "extended" in w["tags"] else "")
                     + (" - PARTICULARLY DANGEROUS SITUATION" if code == "TORP" else "")
                     + (" - TORNADO EMERGENCY" if code == "TORE" else "")
                     + (f" ({w['tags'][0].upper()})" if code in ("SVRC", "SVRD") else "")
                     + f"\\nStorm {w['storm']}, #{w['id']}  {hhmm(w['t0'])} - {hhmm(w['t1'])}\\nHAZARD...{w['hazard']}")
            lat, lon = proj.inv(w["poly"][:, 0], w["poly"][:, 1])
            pts = [f"{float(a):.4f},{float(b):.4f}" for a, b in zip(lat, lon)] + [f"{float(lat[0]):.4f},{float(lon[0]):.4f}"]
            lines.append(f"TimeRange: {iso(w['wall0'])} {iso(end)}")
            lines.append("Color: 0 0 0")
            lines.append(f'Line: {width + 2 * casing}, 0, "{label}"')
            lines += pts + ["End:"]
            lines.append(f"Color: {color[0]} {color[1]} {color[2]}")
            lines.append(f'Line: {width}, 0, "{label}"')
            lines += pts + ["End:"]
        return "\n".join(lines) + "\n"
