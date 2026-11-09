"""Bakes glTF car models (.glb: a node named "body", four named "wheel-*-*", a base-colour texture or factors) into
chase mode's car format (web/models/cars/<kind>.json), read by web/js/chase/carmesh.js:

  body:  triangles (p: x left, y up, z forward, m in the model's units; n: normals; c: sRGB colour from the texture
         at each vertex; paint: 1 where it is the car's paint, which chase mode tints to the player's colour)
  wheel: one wheel, centred, its axle along x, mirrored so it has a rim on both sides (one mesh serves all four)
  axle:  the rear and front axles' z, wheelR / wheelY: the wheel's radius and its centre's height (the ground at y 0)

Chase mode stretches each to its car's physics (wheelbase, track, wheel radius): carmesh.js.

python bake_cars.py <out dir> kind=path.glb [kind=path.glb ...]

The defaults (suv, sedan, truck) are Kenney's Car Kit 3.1 (CC0): https://kenney.nl/assets/car-kit"""
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def load(path):
    b = Path(path).read_bytes()
    jl = struct.unpack_from("<I", b, 12)[0]
    j = json.loads(b[20:20 + jl])
    off = 20 + jl
    binl = struct.unpack_from("<I", b, off)[0]
    blob = b[off + 8:off + 8 + binl]
    return j, blob


def accessor(j, blob, i):
    a = j["accessors"][i]
    v = j["bufferViews"][a["bufferView"]]
    dt, n = COMP[a["componentType"]], NCOMP[a["type"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride", 0)
    item = np.dtype(dt).itemsize * n
    if stride and stride != item:
        raw = np.frombuffer(blob, np.uint8, count=stride * a["count"], offset=start).reshape(a["count"], stride)[:, :item]
        out = raw.copy().view(dt).reshape(a["count"], n)
    else:
        out = np.frombuffer(blob, dt, count=a["count"] * n, offset=start).reshape(a["count"], n)
    if a.get("normalized"):
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out.astype(np.float32) if dt is np.float32 or a.get("normalized") else out


def texture_image(j, blob, mat, folder):
    pbr = mat.get("pbrMetallicRoughness", {})
    if "baseColorTexture" not in pbr:
        return None
    img = j["images"][j["textures"][pbr["baseColorTexture"]["index"]]["source"]]
    if "bufferView" in img:
        v = j["bufferViews"][img["bufferView"]]
        data = blob[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.float32) / 255.0
    if img.get("uri") and not img["uri"].startswith("data:"):
        # an image file next to the model (the kit's shared Textures/colormap.png)
        for base in (folder, folder.parent):
            p = base / img["uri"]
            if p.is_file():
                return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        raise SystemExit(f"texture {img['uri']} not found next to the model")
    return None


def node_world(j):
    """Each node's translation in the model (parents' translations added; the kit's cars have no rotations)."""
    T = {}
    def walk(i, t):
        n = j["nodes"][i]
        tt = t + np.array(n.get("translation", [0, 0, 0]), np.float64)
        T[i] = tt
        for c in n.get("children", []):
            walk(c, tt)
    roots = j["scenes"][j.get("scene", 0)]["nodes"]
    for r in roots:
        walk(r, np.zeros(3))
    return T


def mesh_tris(j, blob, mi, t, tex_cache, folder):
    """A mesh's triangles: positions (+ t), normals, colours, flattened (3 per triangle)."""
    P, N, C = [], [], []
    for prim in j["meshes"][mi]["primitives"]:
        pos = accessor(j, blob, prim["attributes"]["POSITION"]) + t
        nor = accessor(j, blob, prim["attributes"]["NORMAL"])
        mat = j["materials"][prim["material"]] if "material" in prim else {}
        key = prim.get("material")
        if key not in tex_cache:
            tex_cache[key] = texture_image(j, blob, mat, folder)
        img = tex_cache[key]
        fac = np.array(mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1])[:3], np.float32)
        if img is not None and "TEXCOORD_0" in prim["attributes"]:
            uv = accessor(j, blob, prim["attributes"]["TEXCOORD_0"])
            h, w = img.shape[:2]
            px = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
            py = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
            col = img[py, px] * fac
        else:
            col = np.tile(fac, (len(pos), 1))
        idx = accessor(j, blob, prim["indices"]).ravel() if "indices" in prim else np.arange(len(pos))
        P.append(pos[idx]); N.append(nor[idx]); C.append(col[idx])
    return np.concatenate(P), np.concatenate(N), np.concatenate(C)


def bake(path):
    j, blob = load(path)
    T = node_world(j)
    tex = {}
    folder = Path(path).parent
    body = None
    wheels = {}
    for i, n in enumerate(j["nodes"]):
        if "mesh" not in n:
            continue
        name = (n.get("name") or "").lower()
        if name == "body":
            body = mesh_tris(j, blob, n["mesh"], T[i], tex, folder)
        elif name.startswith("wheel-") and name.count("-") == 2:
            wheels[name] = (T[i], n["mesh"])
    if body is None or "wheel-front-left" not in wheels or "wheel-back-left" not in wheels:
        raise SystemExit(f"{path}: needs a 'body' node and wheel-front-left / wheel-back-left")
    P, N, C = body
    # the paint: the hue covering most of the body (by triangle area) among the coloured parts -- not the grey
    # underbody, bumpers and trim, the pale glass or the near-black tyres; the kit's palette shades each colour, so a
    # band of hue (+-22 deg) round the commonest one
    tri = P.reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    tc = C.reshape(-1, 3, 3)[:, 0]
    mx, mn = tc.max(axis=1), tc.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    r, g, b = tc[:, 0], tc[:, 1], tc[:, 2]
    d = np.maximum(mx - mn, 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    coloured = (sat > 0.3) & (mx > 0.18)
    if coloured.any():
        hist = np.zeros(360)
        for h, a in zip(hue[coloured], area[coloured]):
            for k in range(-12, 13):
                hist[int(h + k) % 360] += a
        h0 = int(np.argmax(hist))
        dh = np.abs((hue - h0 + 180) % 360 - 180)
        is_paint = coloured & (dh <= 22)
    else:                                          # a grey or white car: its commonest bright colour
        keys = np.round(tc * 255).astype(int) @ np.array([65536, 256, 1])
        best = max(np.unique(keys), key=lambda k: area[keys == k].sum() if mx[keys == k][0] > 0.3 else -1)
        is_paint = keys == best
    paint = np.repeat(is_paint.astype(np.uint8), 3)
    pc = (tc[is_paint] * area[is_paint, None]).sum(axis=0) / max(area[is_paint].sum(), 1e-9)
    best = int(pc[0] * 255) * 65536 + int(pc[1] * 255) * 256 + int(pc[2] * 255)
    # the wheel: the left front one, centred, mirrored onto both sides
    (wt, wm) = wheels["wheel-front-left"]
    WP, WN, WC = mesh_tris(j, blob, wm, np.zeros(3), tex, folder)
    lo, hi = WP.min(axis=0), WP.max(axis=0)
    centre = np.array([(lo[0] + hi[0]) / 2, 0.0, 0.0])
    WP = WP - centre
    MP = WP * np.array([-1, 1, 1]); MN = WN * np.array([-1, 1, 1])
    MP = MP.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3); MN = MN.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
    MC = WC.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
    WP, WN, WC = np.concatenate([WP, MP]), np.concatenate([WN, MN]), np.concatenate([WC, MC])
    zf = wheels["wheel-front-left"][0][2]
    zr = wheels["wheel-back-left"][0][2]
    r4 = lambda a: [round(float(v), 4) for v in np.asarray(a).ravel()]
    return {
        "src": Path(path).name,
        "axle": [float(zr), float(zf)], "wheelR": float((hi[1] - lo[1]) / 2), "wheelY": float(wheels["wheel-front-left"][0][1]),
        "wheelW": float(hi[0] - lo[0]), "wheelX": float(abs(wt[0] + centre[0])),
        "width": float(P[:, 0].max() - P[:, 0].min()), "height": float(P[:, 1].max()),
        "body": {"p": r4(P), "n": r4(N), "c": r4(C), "paint": paint.tolist()},
        "wheel": {"p": r4(WP), "n": r4(WN), "c": r4(WC)},
        "paintRGB": [int(best // 65536), int(best // 256 % 256), int(best % 256)],
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[2:]:
        kind, path = arg.split("=", 1)
        d = bake(path)
        (out / f"{kind}.json").write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        print(kind, d["src"], "body tris", len(d["body"]["paint"]) // 3, "paint", d["paintRGB"], "axles", d["axle"],
              "wheelR", round(d["wheelR"], 3), "size", (out / f"{kind}.json").stat().st_size)
