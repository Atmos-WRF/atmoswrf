"""GR2Analyst's own site list (radars.gis) and the viewer's world radars.

GR2Analyst places a Level II volume by the four-character site id in its header, looked up in its
radars.gis: a site that table does not know is ignored (the file opens, nothing draws).  The WSR-88D
and TDWR sites are in it already; install() adds the international ones of data/radars.csv (types 4
and 5) as WSR-88D-type sites.  The table is under Program Files, so the install runs elevated (one
Windows prompt), backs the table up once (radars.gis.arht-backup), replaces any earlier lines of the
same ids and leaves it read-only as it found it.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import PROJECT, geo

CANDIDATES = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "GRLevelX" / v / "GIS" / "radars.gis"
              for v in ("GR2Analyst_3", "GR2Analyst_2", "GR2Analyst")]


def gis_path() -> Path | None:
    return next((p for p in CANDIDATES if p.is_file()), None)


def _known(path: Path) -> set[str]:
    ids = set()
    for ln in path.read_text(encoding="latin-1").splitlines():
        p = ln.split("|")
        if len(p) >= 8:
            ids.add(p[0].strip().upper())
    return ids


def world_sites() -> list[dict]:
    return [s for s in geo.radar_sites() if s["type"] in (4, 5)]


def status() -> dict:
    path = gis_path()
    if path is None:
        return {"found": False, "world": len(world_sites())}
    missing = [s for s in world_sites() if s["icao"] not in _known(path)]
    return {"found": True, "path": str(path), "world": len(world_sites()), "missing": len(missing)}


def _line(s: dict) -> str:
    """A site as GR2Analyst's own lines are: the latitude unpadded ("41.59070"; a padded " 41.59070" left the
    DOWs unknown to it -- the file opened, nothing loaded), the longitude in 10 columns, the elevation in 4."""
    i = s["icao"].lower()
    name = s["name"].replace("|", "/")[:40]
    return f"{i}|{i}|{s['lat']:.5f}|{s['lon']:10.5f}|{int(round(s['elev'])):4d}|1|{s['state'][:2]}|{name}"


def install() -> dict:
    """Add the world radars GR2Analyst does not know yet (elevated; waits for the prompt)."""
    path = gis_path()
    if path is None:
        raise ValueError("GR2Analyst's radars.gis was not found (is GR2Analyst installed?)")
    sites = world_sites()
    if not sites:
        return status()
    _write_sites(path, sites, "gr2_world_sites")
    return status()


def set_dow(icao: str, lat: float, lon: float, elev: float) -> dict:
    """Put a DOW's parking spot in GR2Analyst's table (elevated; waits for the prompt).  GR2Analyst places it
    from there, not from its volumes' headers, and reads the table when it starts: it has to be restarted to
    draw the DOW at its new place."""
    path = gis_path()
    if path is None:
        raise ValueError("GR2Analyst's radars.gis was not found (is GR2Analyst installed?)")
    icao = icao.upper()
    _write_sites(path, [{"icao": icao, "lat": lat, "lon": lon, "elev": elev, "state": "US",
                         "name": f"{icao} (Doppler on Wheels)"}], "gr2_dow_site")
    p = path.read_text(encoding="latin-1")
    ok = any(ln.split("|")[0].strip().upper() == icao and abs(float(ln.split("|")[2]) - lat) < 1e-4
             for ln in p.splitlines() if len(ln.split("|")) >= 8)
    return {"found": True, "written": ok, "gr2_running": gr2_running()}


def gr2_running() -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq gr2analyst.exe", "/NH"], capture_output=True,
                             text=True, timeout=20).stdout
        return "gr2analyst.exe" in out.lower()
    except Exception:
        return False


def _write_sites(path: Path, sites: list[dict], tag: str) -> None:
    """Replace (or add) these sites' lines in radars.gis: one elevated PowerShell, one prompt, waited for."""
    work = PROJECT / "output" / "cache"
    work.mkdir(parents=True, exist_ok=True)
    lines = work / f"{tag}.txt"
    # as bytes: write_text turned each \r\n into \r\r\n, and PowerShell read a blank line after every site
    lines.write_bytes(("\r\n".join(_line(s) for s in sites) + "\r\n").encode("ascii", errors="replace"))
    ids = ",".join(s["icao"].lower() for s in sites)
    script = work / f"{tag}.ps1"
    script.write_text(rf"""$ErrorActionPreference = 'Stop'
$path = '{path}'
$backup = "$path.arht-backup"
if (-not (Test-Path $backup)) {{ Copy-Item $path $backup }}
$ro = (Get-Item $path).IsReadOnly
Set-ItemProperty $path -Name IsReadOnly -Value $false
$ids = @{{}}; foreach ($i in '{ids}'.Split(',')) {{ $ids[$i] = 1 }}
$keep = Get-Content $path | Where-Object {{ $_.Trim() -ne '' -and -not $ids.ContainsKey(($_ -split '\|')[0].Trim().ToLower()) }}
$add = Get-Content '{lines}' | Where-Object {{ $_.Trim() -ne '' }}
Set-Content -Path $path -Value ($keep + $add) -Encoding ASCII
if ($ro) {{ Set-ItemProperty $path -Name IsReadOnly -Value $true }}
""", encoding="utf-8")
    # one UAC prompt; waits until the elevated PowerShell has written the table
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList "
                    f"'-NoProfile -ExecutionPolicy Bypass -File \"{script}\"'"], check=False, timeout=600)
