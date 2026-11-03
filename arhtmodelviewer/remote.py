"""The phone / tablet remote: a small page (web/remote.html at /remote) other devices on the network -- or over a
VPN such as Tailscale -- can open while the PC runs, to watch a run (radar picture, status, tornadoes, warnings,
log) and, with the control PIN, pause, resume or stop it.

Two PINs: WATCH (to share: it only looks) and CONTROL (yours).  A PIN is exchanged once for a token the page keeps;
everything under /api/r/ needs one.  Off until switched on in the viewer (Radar tab, Phone remote); the PINs and the
switch live in output/cache/remote.json.  Failed PINs are slowed down per address."""
from __future__ import annotations

import io
import json
import secrets
import threading
import time
from pathlib import Path

from . import PROJECT

CONFIG = PROJECT / "output" / "cache" / "remote.json"
TOKEN_LIFE = 7 * 24 * 3600.0
_LOCK = threading.Lock()
_TOKENS: dict[str, tuple[str, float]] = {}        # token -> (role, expires)
_FAILS: dict[str, list[float]] = {}               # address -> recent failed-PIN times
_IMG: dict = {}                                   # (volume, product, view, size) -> png bytes (the newest few)


def config() -> dict:
    try:
        c = json.loads(CONFIG.read_text())
    except Exception:
        c = {}
    changed = False
    for k in ("watch_pin", "control_pin"):
        if not str(c.get(k, "")).isdigit() or len(str(c.get(k))) < 4:
            c[k] = f"{secrets.randbelow(10 ** 6):06d}"
            changed = True
    c.setdefault("on", False)
    if changed:
        save(c)
    return c


def save(c: dict) -> dict:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"on": bool(c.get("on")), "watch_pin": str(c["watch_pin"]),
                                  "control_pin": str(c["control_pin"])}))
    return c


def update(body: dict) -> dict:
    c = config()
    if "on" in body:
        c["on"] = bool(body["on"])
    for k in ("watch_pin", "control_pin"):
        v = str(body.get(k, "")).strip()
        if v:
            if not v.isdigit() or not 4 <= len(v) <= 12:
                raise ValueError("a PIN is 4-12 digits")
            c[k] = v
    if c["watch_pin"] == c["control_pin"]:
        raise ValueError("the watch and control PINs must differ")
    if body.get("new_pins"):
        c["watch_pin"] = f"{secrets.randbelow(10 ** 6):06d}"
        c["control_pin"] = f"{secrets.randbelow(10 ** 6):06d}"
    if body.get("new_pins") or any(body.get(k) for k in ("watch_pin", "control_pin")):
        with _LOCK:
            _TOKENS.clear()                       # new PINs: everyone signs in again
    return save(c)


def login(pin: str, addr: str) -> dict:
    now = time.time()
    with _LOCK:
        recent = [t for t in _FAILS.get(addr, []) if now - t < 300.0]
        _FAILS[addr] = recent
        if len(recent) >= 8:
            raise PermissionError("too many wrong PINs: wait a few minutes")
    c = config()
    if not c.get("on"):
        raise PermissionError("the remote is switched off in the viewer")
    role = ("control" if secrets.compare_digest(str(pin), c["control_pin"]) else
            "watch" if secrets.compare_digest(str(pin), c["watch_pin"]) else None)
    if role is None:
        with _LOCK:
            _FAILS.setdefault(addr, []).append(now)
        time.sleep(0.6)
        raise PermissionError("wrong PIN")
    tok = secrets.token_urlsafe(24)
    with _LOCK:
        _TOKENS[tok] = (role, now + TOKEN_LIFE)
    return {"token": tok, "role": role}


def role_of(token: str | None) -> str | None:
    if not token or not config().get("on"):
        return None
    with _LOCK:
        got = _TOKENS.get(token)
        if got is None or got[1] < time.time():
            _TOKENS.pop(token, None)
            return None
        return got[0]


def summary(sim) -> dict:
    """What the remote page shows of the run on the map."""
    if sim is None:
        return {"status": "none"}
    got = sim.lock.acquire(timeout=3.0)
    try:
        st = sim.info()
    finally:
        if got:
            sim.lock.release()
    tors = st.get("tornado_tracks") or []
    warns = st.get("warnings") or []
    return {"run_id": st.get("run_id"), "status": st.get("status"), "error": (st.get("error") or "")[:400],
            "t": st.get("t"), "model_ms": st.get("model_ms"), "speed": st.get("speed"),
            "radars": [{"icao": r["icao"], "kind": r.get("kind"), "volumes": r.get("volumes")} for r in st.get("radars") or []],
            "display_radar": st.get("display_radar"),
            "tornadoes": [{"id": t["id"], "ef": t.get("ef"), "active": t.get("active"), "wind": t.get("wind"),
                           "t0": t.get("t0"), "t1": t.get("t1")} for t in tors],
            "warnings": [{"code": w.get("code") or w.get("kind"), "hazard": (w.get("hazard") or "")[:160],
                          "active": w.get("active"), "t0": w.get("t0"), "t1": w.get("t1"), "storm": w.get("storm")}
                         for w in (warns if isinstance(warns, list) else [])][-12:],
            "events": [{"t": e.get("t"), "kind": e.get("kind"), "text": e.get("text", "")[:200]}
                       for e in (st.get("events") or [])[-25:]],
            "storms": len([t for t in (st.get("tracks") or []) if t.get("ended") is None])}


def picture(run_dir: Path, site: str, product: str, view: str, size=(900, 900)) -> bytes:
    """A PNG of the radar's newest volume (loops.make, one frame), cached per volume.  ``view``: "storm" (60 km
    on the strongest tornado or else the radar), or a range in km around the radar ("60", "150", "230")."""
    from . import loops, geo
    import numpy as np
    d = Path(run_dir) / site
    names = sorted(p.name for p in d.iterdir() if loops._ARCH.match(p.name) and p.name.startswith(site)) if d.is_dir() else []
    if not names:
        raise ValueError(f"{site} has no volumes yet")
    key = (str(run_dir), names[-1], product, view, tuple(size))
    with _LOCK:
        if key in _IMG:
            return _IMG[key]
    st = json.loads((Path(run_dir) / "state.json").read_text()) if (Path(run_dir) / "state.json").is_file() else {}
    settings = st.get("settings") or json.loads((Path(run_dir) / "settings.json").read_text())
    proj = geo.Lambert(float(settings["map"]["lat0"]), float(settings["map"]["lon0"]))
    r = next((q for q in st.get("radars") or [] if q.get("icao") == site), None)
    if r is not None:
        cx, cy = r["x"] / 1000.0, r["y"] / 1000.0
    else:
        from . import level2
        v = level2.read_volume((d / names[-1]).read_bytes())
        la, lo = v["sweeps"][min(v["sweeps"])].get("site", (settings["map"]["lat0"], settings["map"]["lon0"]))
        x, y = proj.fwd(np.array([la]), np.array([lo]))
        cx, cy = float(np.ravel(x)[0]) / 1000.0, float(np.ravel(y)[0]) / 1000.0
    w = 120.0
    if view == "storm":
        live = [t for t in st.get("tornado_tracks") or [] if t.get("active")] or (st.get("tornado_tracks") or [])[-1:]
        if live:
            p = live[-1]["pts"][-1]
            cx, cy, w = p[1] / 1000.0, p[2] / 1000.0, 60.0
    else:
        w = 2.0 * float(view)
    frames = loops.make(run_dir, site, product, 0.5, names[-1], names[-1], {"cx": cx, "cy": cy, "w": w},
                        size=size, frames_only=True)
    buf = io.BytesIO()
    frames[0].save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    with _LOCK:
        if len(_IMG) > 24:
            _IMG.clear()
        _IMG[key] = data
    return data
