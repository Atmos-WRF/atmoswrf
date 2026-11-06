"""A small Mapbox Vector Tile reader (protobuf, no packages): the layers of a .pbf tile, each feature's tags and its
geometry in tile pixels (0..extent, y down), as OpenFreeMap serves OpenStreetMap (osmfetch.py)."""
from __future__ import annotations

import gzip
import struct


def _varint(b: bytes, i: int):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b: bytes, i: int = 0, end: int | None = None):
    """(field number, wire type, value) of a message: varints as ints, length-delimited as (start, stop)."""
    end = len(b) if end is None else end
    while i < end:
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 1:
            v, i = b[i:i + 8], i + 8
        elif w == 2:
            n, i = _varint(b, i)
            v, i = (i, i + n), i + n
        elif w == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"protobuf wire type {w}")
        yield f, w, v


def _packed(b: bytes, span) -> list:
    out, i = [], span[0]
    while i < span[1]:
        v, i = _varint(b, i)
        out.append(v)
    return out


def _value(b: bytes, span):
    for f, w, v in _fields(b, *span):
        if f == 1:
            return b[v[0]:v[1]].decode("utf-8", "replace")
        if f == 2:
            return struct.unpack("<f", v)[0]
        if f == 3:
            return struct.unpack("<d", v)[0]
        if f in (4, 5):
            return v
        if f == 6:
            return (v >> 1) ^ -(v & 1)
        if f == 7:
            return bool(v)
    return None


def _geometry(cmds: list) -> list:
    """The parts (lists of [x, y]) of a feature's geometry: lines, or rings (closed)."""
    parts, cur, x, y, i = [], None, 0, 0, 0
    while i < len(cmds):
        c = cmds[i]
        i += 1
        op, n = c & 7, c >> 3
        if op == 7:                                   # ClosePath
            if cur:
                cur.append(cur[0])
            continue
        for _ in range(n):
            dx, dy = cmds[i], cmds[i + 1]
            i += 2
            x += (dx >> 1) ^ -(dx & 1)
            y += (dy >> 1) ^ -(dy & 1)
            if op == 1:                               # MoveTo: a new part
                cur = [[x, y]]
                parts.append(cur)
            else:
                cur.append([x, y])
    return parts


def ring_area(r: list) -> float:
    """The surveyor's formula in tile pixels (y down): positive for an outer ring, negative for a hole."""
    a = 0.0
    for (x0, y0), (x1, y1) in zip(r, r[1:]):
        a += x0 * y1 - x1 * y0
    return a / 2.0


def read(data: bytes, layers=None) -> dict:
    """{layer name: {"extent": n, "features": [{"type": 1|2|3, "tags": {...}, "parts": [[[x, y], ...], ...]}]}} --
    only the layers named (all when None)."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    out = {}
    for f, w, v in _fields(data):
        if f != 3 or w != 2:
            continue
        name, extent, keys, vals, feats = None, 4096, [], [], []
        for lf, lw, lv in _fields(data, *v):
            if lf == 1:
                name = data[lv[0]:lv[1]].decode()
            elif lf == 2:
                feats.append(lv)
            elif lf == 3:
                keys.append(data[lv[0]:lv[1]].decode())
            elif lf == 4:
                vals.append(lv)
            elif lf == 5:
                extent = lv
        if layers is not None and name not in layers:
            continue
        values = [_value(data, s) for s in vals]
        fl = []
        for span in feats:
            typ, tags, geom = 0, {}, []
            for ff, fw, fv in _fields(data, *span):
                if ff == 2:
                    t = _packed(data, fv)
                    tags = {keys[t[k]]: values[t[k + 1]] for k in range(0, len(t) - 1, 2)}
                elif ff == 3:
                    typ = fv
                elif ff == 4:
                    geom = _packed(data, fv)
            fl.append({"type": typ, "tags": tags, "parts": _geometry(geom)})
        out[name] = {"extent": extent, "features": fl}
    return out
