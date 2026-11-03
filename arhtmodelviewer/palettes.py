"""GR2Analyst / GRLevelX / Supercell-Wx color tables (.pal): the parser.
Finding them on the computer is library.py's job."""
from __future__ import annotations


def parse_pal(text: str) -> dict:
    """Parse a GR color table into {product, units, scale, offset, stops}.
    A stop is {v, c: [r,g,b,a], c2: [r,g,b,a] | None, solid: bool}."""
    product, units, scale, offset = "", "", 1.0, 0.0
    stops = []
    rf = None
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        vals = val.replace(",", " ").split()
        try:
            if key == "product":
                product = val.strip().upper()
            elif key == "units":
                units = val.strip()
            elif key == "scale":
                scale = float(vals[0])
            elif key == "offset":
                offset = float(vals[0])
            elif key == "rf":
                rf = [int(float(x)) for x in vals[:3]] + [255]
            elif key in ("color", "color4", "solidcolor", "solidcolor4"):
                four = key.endswith("4")
                n = 4 if four else 3
                v = float(vals[0])
                nums = [int(float(x)) for x in vals[1:]]
                c = nums[:n] + ([] if four else [255])
                c2 = None
                if len(nums) >= 2 * n:
                    c2 = nums[n:2 * n] + ([] if four else [255])
                stops.append({"v": v, "c": c, "c2": c2, "solid": key.startswith("solid")})
        except (ValueError, IndexError):
            continue
    stops.sort(key=lambda s: s["v"])
    return {"product": product, "units": units, "scale": scale, "offset": offset,
            "stops": stops, "rf": rf}
