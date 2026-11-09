"""The live tornado from an outside large-eddy simulation (the Unity tornado model, through its AtmosBridge), passed
on to chase mode.

The bridge POSTs a frame to /api/tornado3d several times a second: b"TOR1", a little-endian uint32 header length,
the header (JSON: where the box stands in this run's coordinates, its size and grids, the codes, the look), then the
media and wind volumes as bytes (see AtmosBridge.cs).  Only the newest frame is kept, compressed once here, and every
chase page (this PC's and joined friends') reads it with GET /api/tornado3d?since=<seq>: the frame (deflate,
X-Tornado-Seq), or 204 with X-Tornado-State "same" (nothing newer) or "off" (none for STALE s: the bridge stopped).
"""
from __future__ import annotations

import json
import struct
import threading
import time
import zlib

STALE = 3.0                     # s without a frame: no tornado
MAX_BYTES = 64 * 1024 * 1024    # a frame bigger than this is refused

_LOCK = threading.Lock()
_LATEST = {"seq": 0, "blob": None, "at": 0.0, "head": None}


def put(raw: bytes) -> dict:
    """Keeps a frame from the bridge (checked, compressed once). Its header, or ValueError."""
    if len(raw) < 8 or raw[:4] != b"TOR1":
        raise ValueError("not a tornado frame (no TOR1 tag)")
    if len(raw) > MAX_BYTES:
        raise ValueError("frame too large")
    (hl,) = struct.unpack_from("<I", raw, 4)
    if 8 + hl > len(raw):
        raise ValueError("header runs past the frame")
    head = json.loads(raw[8:8 + hl].decode("utf-8"))
    w, h, d = (int(v) for v in head["tex"])
    vw, vh, vd = (int(v) for v in head["vtex"])
    if 8 + hl + 4 * (w * h * d + vw * vh * vd) != len(raw):
        raise ValueError("frame size does not match its grids")
    blob = zlib.compress(raw, 1)
    with _LOCK:
        _LATEST["seq"] += 1
        _LATEST.update(blob=blob, at=time.monotonic(), head=head)
        return {"seq": _LATEST["seq"], "bytes": len(raw), "compressed": len(blob)}


def get(since: int | None):
    """(state, seq, compressed frame or None): state "new", "same" or "off"."""
    with _LOCK:
        seq, blob, at = _LATEST["seq"], _LATEST["blob"], _LATEST["at"]
    if blob is None or time.monotonic() - at > STALE:
        return "off", seq, None
    if since is not None and since == seq:
        return "same", seq, None
    return "new", seq, blob


def status() -> dict:
    with _LOCK:
        head, at, seq = _LATEST["head"], _LATEST["at"], _LATEST["seq"]
    live = head is not None and time.monotonic() - at <= STALE
    return {"live": live, "seq": seq, "age": None if head is None else round(time.monotonic() - at, 2),
            "x": head.get("x") if live else None, "y": head.get("y") if live else None}
