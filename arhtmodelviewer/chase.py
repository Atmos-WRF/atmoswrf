"""Chase mode (web/chase.html at /chase): driving under the run's storms, alone or with friends.

The buildings: FEMA's USA Structures footprints (ArcGIS Online), or OpenStreetMap's (OpenFreeMap, Overpass) where
FEMA has none (outside the US), fetched in cells (BUILDING_CELL degrees) by a background thread and kept in
output/cache/osm/structures, like landscape.py's roads; a request returns the footprints there are (world km) with
a height and a kind, and how many cells are still on their way.

The air at the cars (air()): the model's lowest-level wind (ground-relative, at 10 m), gust, temperature and
moisture at each point, from the finest grid holding it (the tornado zoom where it is), for the wind's sound and the
rain or snow.

Multiplayer: the PC running the viewer hosts.  Hosting is off until switched on from the chase page on this PC;
then a six-digit join code lets friends on the network (or over a VPN such as Tailscale) open /chase, join and
drive.  A joined page gets a token; with it, and only while hosting is on, other computers may read what the
page draws (the run's state, the cloud volume, the land cover, the roads and buildings, the map) and post their
car to /api/c/sync, which answers with everyone else's -- nothing that changes the run or the files.  Wrong codes
are slowed down per address."""
from __future__ import annotations

import atexit
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

from . import osmfetch
from .landscape import OSM_CACHE

# ---------------------------------------------------------------- buildings
BUILDING_CELL = 0.05             # degrees
BUILDING_DIR = OSM_CACHE / "structures"          # FEMA's (OpenStreetMap's outside the US); "buildings": OSM's only
FEMA_URL = ("https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/"
            "FeatureServer/0/query")
FEMA_PAGE, FEMA_THREADS, FEMA_TIMEOUT = 2000, 4, 60.0
# kinds: 0 house, 1 shop / office, 2 farm / industrial, 3 tall (flats, hotel), 4 shed / garage, 5 civic (church, school)
_KIND = {"house": 0, "detached": 0, "residential": 0, "semidetached_house": 0, "bungalow": 0, "terrace": 0,
         "static_caravan": 0, "cabin": 0, "farm": 0, "yes": 0,
         "commercial": 1, "retail": 1, "office": 1, "supermarket": 1, "kiosk": 1, "service": 1,
         "industrial": 2, "warehouse": 2, "barn": 2, "farm_auxiliary": 2, "agricultural": 2, "hangar": 2,
         "greenhouse": 2, "storage_tank": 2, "silo": 2, "manufacture": 2,
         "apartments": 3, "hotel": 3, "dormitory": 3, "hospital": 3,
         "garage": 4, "garages": 4, "shed": 4, "carport": 4, "roof": 4, "hut": 4,
         "church": 5, "school": 5, "public": 5, "civic": 5, "government": 5, "university": 5, "college": 5,
         "fire_station": 5, "sports_hall": 5}
_HEIGHT = {0: 5.5, 1: 6.5, 2: 7.0, 3: 12.0, 4: 3.2, 5: 9.0}


def _num(s) -> float | None:
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(s or ""))
    return float(m.group(1)) if m else None


def _b_file(la: float, lo: float):
    return BUILDING_DIR / f"{la:+08.2f}_{lo:+08.3f}.json"


def _fema_get(params: dict) -> dict:
    url = FEMA_URL + "?" + urllib.parse.urlencode(params)
    err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Atmos-WRF chase mode"})
            with urllib.request.urlopen(req, timeout=FEMA_TIMEOUT) as r:
                d = json.load(r)
            if "error" in d:
                raise RuntimeError(str(d["error"])[:200])
            return d
        except Exception as e:                   # a busy moment on ArcGIS Online: again, a little later
            err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FEMA USA Structures: {err}")


def _fema_kind(a: dict, area: float, h: float) -> int:
    """A kind (see _KIND) from the structure's FEMA occupancy class and type."""
    occ, prim = (a.get("OCC_CLS") or "").lower(), (a.get("PRIM_OCC") or "").lower()
    if a.get("OUTBLDG") not in (None, "", 0, "0", "N", "No", "NO"):
        return 4
    if occ == "residential":
        if "multi" in prim or "lodging" in prim or "dormitor" in prim or "nursing" in prim:
            return 3 if h >= 9.0 or area > 1200.0 else 0
        return 4 if area < 30.0 else 0
    if occ == "commercial":
        return 3 if h >= 12.0 else 1
    if occ in ("industrial", "agriculture"):
        return 2
    if occ in ("education", "government", "religious", "assembly"):
        return 5
    return _kind_by_size(area, h)


def fema_buildings(la: float, lo: float, step: float) -> list:
    """The buildings of a cell from FEMA's USA Structures (every structure over 450 sq ft in the US and its
    territories, footprints from imagery and lidar, with an occupancy class and a height), as chase's files have
    them: [[height m, kind, [lat ...], [lon ...]], ...].  Pages of FEMA_PAGE, several at a time."""
    from concurrent.futures import ThreadPoolExecutor
    base = {"where": "1=1", "geometry": f"{lo:.5f},{la:.5f},{lo + step:.5f},{la + step:.5f}",
            "geometryType": "esriGeometryEnvelope", "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
            "f": "json"}
    n = int(_fema_get({**base, "returnCountOnly": "true"}).get("count") or 0)
    if n == 0:
        return []
    q = {**base, "outFields": "OCC_CLS,PRIM_OCC,HEIGHT,OUTBLDG,SQMETERS", "returnGeometry": "true", "outSR": "4326",
         "geometryPrecision": "6", "orderByFields": "OBJECTID", "resultRecordCount": str(FEMA_PAGE)}
    offsets = list(range(0, n, FEMA_PAGE))
    with ThreadPoolExecutor(max_workers=FEMA_THREADS) as ex:
        pages = list(ex.map(lambda o: _fema_get({**q, "resultOffset": str(o)}), offsets))
    out = []
    for d in pages:
        for f in d.get("features", []):
            g, a = f.get("geometry") or {}, f.get("attributes") or {}
            rings = g.get("rings") or []
            if not rings or len(rings[0]) < 4:
                continue
            ring = rings[0]                         # the outline (inner rings: courtyards, left out)
            area = float(a.get("SQMETERS") or 0.0)
            h0 = a.get("HEIGHT")
            h = float(h0) if h0 not in (None, "") and float(h0) >= 2.5 else None
            k = _fema_kind(a, area, h or 0.0)
            if h is None:
                h = _HEIGHT[k]
            elif k == 0:
                h = max(h, 4.5)                     # room for the gable roof over the walls
            h = min(h, 300.0)
            out.append([round(h, 1), k, [round(p[1], 6) for p in ring], [round(p[0], 6) for p in ring]])
    return out


def _b_fetch(la: float, lo: float) -> None:
    try:                                         # FEMA's USA Structures first (the US); OpenStreetMap elsewhere
        out = fema_buildings(la, lo, BUILDING_CELL)
    except Exception:
        out = None
    if out:
        _b_write(la, lo, out)
        return
    q = (f'[out:json][timeout:150];way["building"]({la:.4f},{lo:.4f},{la + BUILDING_CELL:.4f},'
         f'{lo + BUILDING_CELL:.4f});out geom tags;')
    try:                                         # OpenFreeMap's tiles first (fast, a CDN), Overpass if it fails
        out = osmfetch.ofm_buildings(la, lo, BUILDING_CELL, _kind_by_size)
    except Exception:
        out = None
    if out is not None:
        _b_write(la, lo, out)
        return
    d = osmfetch.overpass(q)
    out = []
    for w in d.get("elements", []):
        g, tags = w.get("geometry"), w.get("tags", {})
        if not g or len(g) < 4:
            continue
        b = tags.get("building", "yes")
        if b in ("no", "construction", "ruins", "collapsed"):
            continue
        k = _KIND.get(b, 0)
        h = _num(tags.get("height"))
        if h is None and _num(tags.get("building:levels")) is not None:
            h = 3.1 * _num(tags["building:levels"]) + (1.5 if k == 0 else 0.5)
        h = min(max(h if h is not None else _HEIGHT[k], 2.5), 300.0)
        out.append([round(h, 1), k, [round(p["lat"], 6) for p in g], [round(p["lon"], 6) for p in g]])
    _b_write(la, lo, out)


def _b_write(la: float, lo: float, out: list) -> None:
    f = _b_file(la, lo)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f.name + ".part")
    part.write_text(json.dumps(out))
    part.replace(f)


def _kind_by_size(area: float, h: float) -> int:
    """A building's kind from its footprint (m^2) and height when the map has no type (OpenFreeMap)."""
    if h >= 12.0:
        return 3
    if area < 30.0:
        return 4
    if area <= 450.0:
        return 0
    if area >= 1500.0:
        return 2 if h <= 9.0 else 1
    return 1


_BUILDINGS = osmfetch.CellQueue("buildings", _b_fetch)


def buildings(proj, box_km) -> dict:
    """The buildings in the box (world km): {"b": [[height m, kind, [x, y, x, y, ...] km], ...], "pending": n}; cells
    not fetched yet are queued (nearest the box's middle first) and counted."""
    x0, y0, x1, y1 = (float(v) for v in box_km)
    xs = np.array([x0, x1, x0, x1, (x0 + x1) / 2, (x0 + x1) / 2, x0, x1]) * 1000.0
    ys = np.array([y0, y0, y1, y1, y0, y1, (y0 + y1) / 2, (y0 + y1) / 2]) * 1000.0
    lat, lon = (np.asarray(v) for v in proj.inv(xs, ys))
    S = BUILDING_CELL
    clat, clon = float(lat[4:6].mean()), float(lon[6:8].mean())
    cells = [(round(a * S, 2), round(b * S, 3))
             for a in range(int(math.floor(lat.min() / S)), int(math.floor(lat.max() / S)) + 1)
             for b in range(int(math.floor(lon.min() / S)), int(math.floor(lon.max() / S)) + 1)]
    if len(cells) > 100:
        raise ValueError("the box is too big for the buildings")
    cells.sort(key=lambda c: (c[0] + S / 2 - clat) ** 2 + ((c[1] + S / 2 - clon) * 0.8) ** 2)
    out, pending = [], 0
    for c in cells:
        f = _b_file(*c)
        if not f.is_file():
            pending += _BUILDINGS.want(c)
            continue
        items = json.loads(f.read_text())
        if not items:
            continue
        # one projection for the whole cell
        n = [len(it[2]) for it in items]
        la = np.concatenate([np.asarray(it[2]) for it in items])
        lo = np.concatenate([np.asarray(it[3]) for it in items])
        X, Y = proj.fwd(la, lo)
        X, Y = np.asarray(X) / 1000.0, np.asarray(Y) / 1000.0
        i = 0
        for it, m in zip(items, n):
            bx, by = X[i:i + m], Y[i:i + m]
            i += m
            if bx.max() < x0 or bx.min() > x1 or by.max() < y0 or by.min() > y1:
                continue
            xy = np.empty(2 * m)
            xy[0::2], xy[1::2] = bx, by
            out.append([it[0], it[1], np.round(xy, 5).tolist()])
    return {"b": out, "pending": pending, "busy": bool(pending) and osmfetch.busy()}


# ---------------------------------------------------------------- the gas stations
# OpenStreetMap's fuel stations (amenity=fuel: OpenMapTiles' poi layer, class "fuel", subclass "fuel" -- not the EV
# chargers), from OpenFreeMap's z14 tiles in FUEL_CELL-degree cells, kept in output/cache/osm/fuel: chase mode builds
# a station at each (a canopy over the pumps, a store you can walk into) and routes to the nearest
FUEL_CELL = 0.1
FUEL_DIR = OSM_CACHE / "fuel"


def _f_file(la: float, lo: float):
    return FUEL_DIR / f"{la:+08.2f}_{lo:+08.2f}.json"


def _f_fetch(la: float, lo: float) -> None:
    out, seen = [], set()
    for x, y, L, ext, box in osmfetch._cell_tiles(la, lo, FUEL_CELL, 14, {"poi"}):
        for f in L.get("poi", {}).get("features", []):
            t = f["tags"]
            if f["type"] != 1 or t.get("class") != "fuel" or t.get("subclass", "fuel") != "fuel":
                continue
            for part in f["parts"]:
                for px, py in part:
                    if not (0 <= px < ext and 0 <= py < ext and box[0] <= px < box[2] and box[1] <= py < box[3]):
                        continue                             # in another tile's or cell's share
                    lat, lon = osmfetch._lonlat(14, x + px / ext, y + py / ext)
                    key = (round(lat, 5), round(lon, 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append([round(lat, 6), round(lon, 6), str(t.get("name") or t.get("brand") or "")[:40]])
    f = _f_file(la, lo)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f.name + ".part")
    part.write_text(json.dumps(out))
    part.replace(f)


_FUEL = osmfetch.CellQueue("fuel", _f_fetch)


def fuel(proj, box_km) -> dict:
    """The gas stations in the box (world km): {"fuel": [[x, y km, name], ...], "pending": n}."""
    x0, y0, x1, y1 = (float(v) for v in box_km)
    xs = np.array([x0, x1, x0, x1]) * 1000.0
    ys = np.array([y0, y0, y1, y1]) * 1000.0
    lat, lon = (np.asarray(v) for v in proj.inv(xs, ys))
    S = FUEL_CELL
    cells = [(round(a * S, 2), round(b * S, 2))
             for a in range(int(math.floor(lat.min() / S)), int(math.floor(lat.max() / S)) + 1)
             for b in range(int(math.floor(lon.min() / S)), int(math.floor(lon.max() / S)) + 1)]
    if len(cells) > 120:
        raise ValueError("the box is too big for the gas stations")
    out, pending = [], 0
    for c in cells:
        f = _f_file(*c)
        if not f.is_file():
            pending += _FUEL.want(c)
            continue
        items = json.loads(f.read_text())
        if not items:
            continue
        X, Y = proj.fwd(np.array([it[0] for it in items]), np.array([it[1] for it in items]))
        for it, xx, yy in zip(items, np.atleast_1d(X), np.atleast_1d(Y)):
            xk, yk = float(xx) / 1000.0, float(yy) / 1000.0
            if x0 <= xk <= x1 and y0 <= yk <= y1:
                out.append([round(xk, 5), round(yk, 5), it[2]])
    return {"fuel": out, "pending": pending}


# ---------------------------------------------------------------- the air at the cars
AIR_MAX = 16                     # points per request


# the model's 10 m wind round the first point (air(grid=True)): GRID_HALF cells each way of the finest grid holding
# it, at most GRID_SPAN m across -- chase mode's car reads its wind from it every physics step (a tornado's winds
# change over tens of metres: one point twice a second is too coarse and too late near one)
GRID_HALF = 20
GRID_SPAN = 2400.0


def air(sim, pts, grid: bool = False) -> list:
    """The model's air at ground points pts [(x, y) world m]: per point {"u", "v" (m/s at 10 m, ground-relative),
    "gust" (m/s), "t" (deg C), "td" (dew point, deg C), "inside" (a storm patch holds it)}, the finest grid holding
    the point (the tornado zoom where it is), the environment's lowest level elsewhere.  ``grid``: the first point's
    also carries "grid" {x0, y0 (its first cell's centre, world m), dx, nx, ny, u, v (row by row from the south,
    m/s at 10 m)}: that grid's lowest level round it, taken down to 10 m as the point is.  Caller holds sim.lock."""
    from .metars import GUST_MIN, GUST_SIGMA, _Station
    net, env = sim.metars, sim.env
    z0 = float(env.physics.z0)
    sts = []
    for k, (x, y) in enumerate(pts[:AIR_MAX]):
        hg = 0.0
        if getattr(env, "terrain", None) is not None:
            try:
                hg = float(np.asarray(env.terrain.height(np.array([x]), np.array([y])), float).ravel()[0])
            except Exception:
                pass
        zz = z0
        if env.roughness is not None:
            try:
                zz = float(np.asarray(env.roughness(np.array([x]), np.array([y])), float).ravel()[0])
            except Exception:
                pass
        sts.append(_Station(k, f"car{k}", float(x), float(y), max(zz, 1e-4), hg))
    if not sts:
        return []
    where = net._where(sts)
    vals, by_patch = {}, {}
    for s in sts:
        if s.icao in where:
            p, j, i = where[s.icao]
            by_patch.setdefault(id(p), (p, []))[1].append((s, j, i))
    for p, lst in by_patch.values():
        a = net._gather(p, np.array([q[1] for q in lst]), np.array([q[2] for q in lst]), net._fields)
        z1, zt = float(p.env.z_h[0]), float(p.env.z_w[-1])
        for k, (s, _, _) in enumerate(lst):
            vals[s.icao] = tuple(a[:6, k]) + (z1 * (1.0 - s.hg / zt), True)
    near = None
    if grid and sts and sts[0].icao in where:
        p, j, i = where[sts[0].icao]
        dx = float(p.env.grid.dx)
        h = max(2, min(GRID_HALF, int(GRID_SPAN / (2.0 * dx))))
        i0, i1 = max(0, i - h), min(p.rect.nx - 1, i + h)
        j0, j1 = max(0, j - h), min(p.rect.ny - 1, j + h)
        JJ, II = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1), indexing="ij")
        g = net._gather(p, JJ.ravel(), II.ravel(), net._fields)
        near = (p, dx, i0, j0, i1 - i0 + 1, j1 - j0 + 1, g[0], g[1])
    rest = [s for s in sts if s.icao not in vals]
    if rest:
        u, v, th, qv, pp, tke = net._environment(rest)
        z1 = float(env.z_h[0])
        for k, s in enumerate(rest):
            vals[s.icao] = (u[k], v[k], th[k], qv[k], pp[k], tke[k], z1, False)
    out = []
    for s in sts:
        u, v, th, qv, pp, tke, z1, inside = (float(c) if not isinstance(c, bool) else c for c in vals[s.icao])
        fac = 1.0 if z1 <= 10.0 else min(max(math.log(10.0 / s.z0) / math.log(z1 / s.z0), 0.4), 1.0)
        u10, v10 = u * fac, v * fac
        spd = math.hypot(u10, v10)
        sig = math.sqrt(max(tke, 0.0) * 2.0 / 3.0) * fac if tke >= 0 else 0.0
        gust = spd + max(GUST_SIGMA * sig, GUST_MIN * spd)
        tk = th * (pp / 1e5) ** 0.2857
        e = max(qv, 1e-7) * pp / (0.622 + max(qv, 1e-7)) / 100.0        # vapour pressure, hPa
        td = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
        ok = all(math.isfinite(c) for c in (u10, v10, gust, tk, td))
        d = {"u": round(u10, 2), "v": round(v10, 2), "gust": round(gust, 2), "t": round(tk - 273.15, 2),
             "td": round(min(td, tk - 273.15), 2), "inside": bool(inside)} if ok else None
        if d is not None and near is not None and s is sts[0]:
            p, dx, i0, j0, nx, ny, gu, gv = near
            gu, gv = np.nan_to_num(gu * fac), np.nan_to_num(gv * fac)          # down to 10 m as the point is
            d["grid"] = {"x0": p.world_x0() + (i0 + 0.5) * dx, "y0": p.world_y0() + (j0 + 0.5) * dx, "dx": dx,
                         "nx": int(nx), "ny": int(ny), "u": np.round(gu, 2).tolist(), "v": np.round(gv, 2).tolist()}
        out.append(d)
    return out


# ---------------------------------------------------------------- multiplayer
_LOCK = threading.Lock()
_HOST = {"on": False, "code": None}
_PLAYERS: dict[str, dict] = {}                  # token -> {id, name, kind, color, s (state), ts, addr}
_FAILS: dict[str, list[float]] = {}
_NEXT = [1]
STALE = 6.0                                     # s without a sync: gone from the others' lists
# what a joined page on another computer may read (GET) with its token while hosting is on
READ = ("/api/state", "/api/pmw/targets", "/api/c/air", "/api/volume", "/api/volume/times", "/api/volume/roads",
        "/api/volume/landcover", "/api/volume/ground", "/api/volume/buildings", "/api/volume/fuel", "/api/frames", "/api/frame", "/api/ppi")
# (and the detailed map's module, tile worker and landmarks, and the cars' models)
STATIC = re.compile(r"^/(chase|static/chase\.html|static/js/(volume|landscape|colors)\.js|static/js/chase/[a-z0-9_]+\.js"
                    r"|static/js/chase/world/(world|tileworker)\.js|static/js/chase/world/landmarks\.json"
                    r"|static/models/cars/[a-z0-9_]+\.json)$")


def host(on: bool | None = None, new_code: bool = False) -> dict:
    if on is False:
        tunnel(False)
    with _LOCK:
        if on is not None:
            _HOST["on"] = bool(on)
            if not on:
                for t in [t for t, p in _PLAYERS.items() if not p.get("local")]:
                    _PLAYERS.pop(t)
        if _HOST["code"] is None or new_code:
            _HOST["code"] = f"{secrets.randbelow(10 ** 6):06d}"
            if new_code:
                for t in [t for t, p in _PLAYERS.items() if not p.get("local")]:
                    _PLAYERS.pop(t)
        return {"on": _HOST["on"], "code": _HOST["code"], "players": _roster()}


def _roster() -> list:
    now = time.monotonic()
    return [{"id": p["id"], "name": p["name"], "local": bool(p.get("local"))} for p in _PLAYERS.values()
            if now - p["ts"] < STALE]


def _clean_name(s) -> str:
    s = re.sub(r"[^\w .'\-]", "", str(s or ""), flags=re.UNICODE).strip()[:20]
    return s or "Chaser"


def join(body: dict, addr: str, local: bool) -> dict:
    """A page joins: on this PC without a code, from elsewhere with the join code while hosting is on."""
    now = time.time()
    if not local:
        with _LOCK:
            recent = [t for t in _FAILS.get(addr, []) if now - t < 300.0]
            _FAILS[addr] = recent
            if len(recent) >= 8:
                raise PermissionError("too many wrong codes: wait a few minutes")
            on, code = _HOST["on"], _HOST["code"]
        if not on:
            raise PermissionError("the host is not taking players: they switch it on in their chase page (Menu, Multiplayer)")
        if not code or not secrets.compare_digest(str(body.get("code", "")).strip(), code):
            with _LOCK:
                _FAILS.setdefault(addr, []).append(now)
            time.sleep(0.6)
            raise PermissionError("wrong join code")
    tok = secrets.token_urlsafe(24)
    with _LOCK:
        pid = _NEXT[0]
        _NEXT[0] += 1
        _PLAYERS[tok] = {"id": pid, "name": _clean_name(body.get("name")), "kind": str(body.get("kind", "suv"))[:10],
                         "color": _color(body.get("color")), "s": None, "ts": time.monotonic(), "addr": addr,
                         "local": local}
    return {"token": tok, "id": pid}


def _color(c) -> list:
    try:
        return [max(0, min(255, int(v))) for v in list(c)[:3]] if c and len(c) >= 3 else [200, 40, 40]
    except (TypeError, ValueError):
        return [200, 40, 40]


def player(token: str | None) -> dict | None:
    if not token:
        return None
    with _LOCK:
        p = _PLAYERS.get(token)
    if p is None or (not p.get("local") and not _HOST["on"]):
        return None
    return p


def sync(token: str | None, body: dict) -> dict:
    """A car's state in, everyone else's out (server time stamps, s)."""
    p = player(token)
    if p is None:
        raise PermissionError("not joined (or hosting was switched off)")
    s = body.get("s")
    now = time.monotonic()
    with _LOCK:
        if isinstance(s, dict) and len(json.dumps(s)) < 2000:
            p["s"] = s
        for k in ("kind",):
            if body.get(k):
                p[k] = str(body[k])[:10]
        if body.get("color"):
            p["color"] = _color(body["color"])
        if body.get("name"):
            p["name"] = _clean_name(body["name"])
        p["ts"] = now
        others = [{"id": q["id"], "name": q["name"], "kind": q["kind"], "color": q["color"], "s": q["s"], "ts": q["ts"]}
                  for t, q in _PLAYERS.items() if t != token and q["s"] is not None and now - q["ts"] < STALE]
        for t in [t for t, q in _PLAYERS.items() if now - q["ts"] > 600.0]:
            _PLAYERS.pop(t)
    return {"now": now, "me": p["id"], "players": others, "hosting": _HOST["on"]}


def leave(token: str | None) -> dict:
    with _LOCK:
        _PLAYERS.pop(token or "", None)
    return {"ok": True}


def allowed(path: str, method: str, token: str | None) -> bool:
    """May another computer have this?  The chase page and its scripts, the join, and -- with a joined page's
    token -- what the page reads and its own syncs; only while hosting is on."""
    if not _HOST["on"]:
        return False
    if method == "GET" and STATIC.match(path):
        return True
    if method == "POST" and path == "/api/c/join":
        return True
    if player(token) is None:
        return False
    if method == "POST":
        return path in ("/api/c/sync", "/api/c/leave")
    return method == "GET" and path in READ


# ---------------------------------------------------------------- the internet link
# A free Cloudflare quick tunnel (no account): https://<words>.trycloudflare.com reaches this viewer, so friends
# need only a browser.  Its requests reach the server from this PC's own address, so the server spots them
# (server._tunneled: the forwarding headers and the Host) and lets them have chase mode only, like any other
# computer.  cloudflared is found on the PATH or fetched once (GitHub, ~60 MB) into output/cache/tools.
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
_TUN = {"proc": None, "url": "", "state": "off", "error": ""}
_TUN_LOCK = threading.Lock()
_TUN_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _cloudflared() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    exe = OSM_CACHE.parent / "tools" / "cloudflared.exe"
    if not exe.is_file():
        exe.parent.mkdir(parents=True, exist_ok=True)
        part = exe.with_suffix(".part")
        with urllib.request.urlopen(CLOUDFLARED_URL, timeout=180) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(part, exe)
    return str(exe)


def _tunnel_run(port: int) -> None:
    try:
        exe = _cloudflared()
        flags = 0x08000000 if os.name == "nt" else 0              # no console window
        # HTTP/2 over TCP 443: QUIC needs UDP, which some networks block
        proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{port}"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags)
        with _TUN_LOCK:
            if _TUN["state"] != "starting":                         # switched off while it started
                proc.terminate()
                return
            _TUN["proc"] = proc
        t0 = time.time()
        for line in proc.stdout:
            m = _TUN_RE.search(line)
            if m:
                with _TUN_LOCK:
                    _TUN.update(url=m.group(0), state="on", error="")
                for _ in proc.stdout:                               # keep its output drained while it runs
                    pass
                break
            if time.time() - t0 > 90:
                raise RuntimeError("the tunnel did not start (no internet, or cloudflared was blocked)")
        with _TUN_LOCK:
            if _TUN["proc"] is proc and _TUN["state"] != "off":
                _TUN.update(state="error", error=_TUN["error"] or "the tunnel closed", url="")
    except Exception as e:
        with _TUN_LOCK:
            _TUN.update(state="error", error=str(e), url="")


def tunnel(on: bool, port: int | None = None) -> dict:
    """Start (needs hosting on) or stop the internet link; its state."""
    with _TUN_LOCK:
        proc = _TUN["proc"]
        if not on:
            _TUN.update(proc=None, url="", state="off", error="")
            if proc is not None and proc.poll() is None:
                proc.terminate()
            return dict(_tunnel_view())
        if _TUN["state"] in ("starting", "on") and (proc is None or proc.poll() is None):
            return dict(_tunnel_view())
        if not _HOST["on"]:
            raise PermissionError("switch on Let friends join first")
        _TUN.update(proc=None, url="", state="starting", error="")
    threading.Thread(target=_tunnel_run, args=(int(port or 8830),), name="chase-tunnel", daemon=True).start()
    return tunnel_state()


def _tunnel_view() -> dict:
    return {"state": _TUN["state"], "url": _TUN["url"], "error": _TUN["error"]}


def tunnel_state() -> dict:
    with _TUN_LOCK:
        return _tunnel_view()


@atexit.register
def _tunnel_close() -> None:
    p = _TUN.get("proc")
    if p is not None and p.poll() is None:
        p.terminate()
