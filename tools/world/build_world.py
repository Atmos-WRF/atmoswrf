"""Builds chase mode's detailed map (web/js/chase/world/) from the friend's single-page "Real Roads Drive" (index.html):
its tile worker, its landmark data and its world (terrain, roads, water, buildings, trees, grass, props, landmarks,
birds) as an ES module, without its car, camera, minimap, navigation, search, collection, HUD or main loop.

python build_world.py <index.html> <web/js/chase/world>"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
lines = src.split("\n")


def find(pat, start=0):
    for i in range(start, len(lines)):
        if re.search(pat, lines[i]):
            return i
    raise SystemExit(f"not found: {pat}")


# ---- the tile worker
w0 = find(r'<script id="tileWorker"')
w1 = find(r"^</script>", w0)
worker = "\n".join(lines[w0 + 1:w1]) + "\n"
# Tree species: the subtropical belt (20-34 deg, any longitude, palms a fifth of it) put palms in Oklahoma and
# Arkansas; palms now only where winters are mild (fading out from 27 to 31 deg), and the Great Plains get their
# own mix: cottonwood (poplar), oak (post / blackjack), eastern redcedar (juniper), elm and hackberry (maple), willow
WORKER_REPL = [
    (r"  subtrop: \{ oak: 0\.3, pine: 0\.3, palm: 0\.2, canopy: 0\.2 \},\n",
     "  subtrop: { oak: 0.35, pine: 0.4, canopy: 0.25 },\n"
     "  palms: { palm: 1 },\n"
     "  plains: { poplar: 0.3, oak: 0.25, juniper: 0.2, maple: 0.15, willow: 0.1 },\n"),
    (r"    subtrop: soft\(al, 20, 34, 3\) \* \(1 - box\(30, 44, -10, 38, 3\)\),\n",
     "    subtrop: soft(al, 20, 34, 3) * (1 - box(30, 44, -10, 38, 3)),\n"
     "    palms: 0.25 * soft(al, 20, 34, 3) * (1 - sm(27, 31, al)) * (1 - box(30, 44, -10, 38, 3)),\n"
     "    plains: box(28, 49, -104, -95, 3) * (1 - boreal),\n"),
]
for pat, rep in WORKER_REPL:
    worker, n = re.subn(pat, rep, worker)
    if n != 1:
        raise SystemExit(f"worker replacement {pat[:60]!r}: {n} matches")
(out / "tileworker.js").write_text("// The detailed map's tile worker (from Real Roads Drive): terrain, roads, water, structures,\n"
                                   "// buildings, imagery and trees for one z14 tile at a time.\n" + worker, encoding="utf-8")

# ---- the landmarks
d0 = find(r'<script type="application/json" id="landmarkData">')
d1 = find(r"^</script>", d0)
(out / "landmarks.json").write_text("\n".join(lines[d0 + 1:d1]).strip() + "\n", encoding="utf-8")

# ---- the world: from the module's start to the car, plus the vehicle state the world reads (V, SURF_PHYS)
m0 = find(r'<script type="module">')
car = find(r"^// =+ car$", m0)
v0 = find(r"^// Arcade vehicle\.", car)
v1 = find(r"^const DRAFT", v0)
body = "\n".join(lines[m0 + 1:car]) + "\n" + "\n".join(lines[v0:v1 + 1]) + "\n"

REPL = [
    # where the map starts: given by chase mode
    # (OpenDrive, 2026-11-09: comment lines, DEFAULT_START and START_FACE round it -- a default start on
    # Going-to-the-Sun Road that only its own car and welcome toast read)
    (r"const params = new URLSearchParams\(location\.search\);\n"
     r"(?://[^\n]*\n)*(?:const DEFAULT_START = [^\n]*\n)?"
     r"const START_LAT = parseFloat\(params\.get\('lat'\) \?\? '[-\d.]+'\);\n"
     r"const START_LON = parseFloat\(params\.get\('lon'\) \?\? '[-\d.]+'\);"
     r"(?:\nconst START_FACE = [^\n]*)?",
     "const CW = globalThis.CHASE_WORLD;\nconst START_LAT = CW.lat, START_LON = CW.lon;\n"
     "const DEFAULT_START = false, START_FACE = null;"),
    # the renderer: chase mode's canvas and WebGL context, drawn into a target chase mode composites
    (r"const renderer = new THREE\.WebGLRenderer\(\{ antialias: true \}\);\n"
     r"renderer\.setPixelRatio\([^\n]*\);\n"
     r"renderer\.setSize\(innerWidth, innerHeight\);\n",
     "const renderer = new THREE.WebGLRenderer({ canvas: CW.gl.canvas, context: CW.gl, antialias: false });\n"
     "renderer.autoClear = true;\nrenderer.setClearColor(0x000000, 0);\n"),
    (r"document\.body\.prepend\(renderer\.domElement\);\n", ""),
    # no sky or haze of its own: the storm's renderer draws them over it
    (r"scene\.background = SKY;", "scene.background = null;"),
    (r"scene\.fog = new THREE\.Fog\([^\n]*\);", "scene.fog = null;"),
    (r"const camera = new THREE\.PerspectiveCamera\(65, innerWidth / innerHeight,",
     "const camera = new THREE.PerspectiveCamera(65, 1,"),
    (r"addEventListener\('resize', \(\) => \{\n  camera\.aspect = innerWidth / innerHeight;\n"
     r"  camera\.updateProjectionMatrix\(\);\n  renderer\.setSize\(innerWidth, innerHeight\);\n\}\);\n", ""),
    (r"scene\.add\(new THREE\.HemisphereLight\(([^)]*)\)\);",
     r"const hemi = new THREE.HemisphereLight(\1);\nscene.add(hemi);"),
    # the worker and the landmarks: files of their own
    (r"new Worker\(URL\.createObjectURL\(new Blob\(\[document\.getElementById\('tileWorker'\)\.textContent\], "
     r"\{ type: 'text/javascript' \}\)\)\)", "new Worker(CW.workerUrl)"),
    (r"JSON\.parse\(document\.getElementById\('landmarkData'\)\.textContent\)", "(CW.landmarks || [])"),
]
for pat, rep in REPL:
    body, n = re.subn(pat, rep, body)
    if n != 1:
        raise SystemExit(f"replacement {pat[:60]!r}: {n} matches")

api = (Path(__file__).parent / "world_api.js").read_text(encoding="utf-8")
head = ("// Chase mode's detailed map, from Real Roads Drive: real terrain (AWS Terrain Tiles), aerial photos and land\n"
        "// classes (NAIP / Sentinel-2, USDA CDL, ESA WorldCover), roads with their lanes and markings (OpenFreeMap),\n"
        "// bridges, water, buildings (OpenStreetMap, FEMA USA Structures), trees, grass and crops, roadside props, farms,\n"
        "// landmarks and birds -- its own car, camera, minimap, navigation, search, collection and HUD left out.\n"
        "// Built by build_world.py; chase mode drives it through the API at the end (window.ChaseWorld).\n")
(out / "world.js").write_text(head + body + "\n" + api, encoding="utf-8")
print("ok", len(body), "chars of world;", (out / "landmarks.json").stat().st_size, "bytes of landmarks")
