"""GR2Analyst's polling feed: a run's Level II volumes, re-timed to when the viewer made them.

GR2Analyst polls a folder laid out like Iowa State's Level II server (grlevel2.cfg,
<SITE>/dir.list with one "size name" line per volume, oldest first, and the volumes).  It reads
each volume's time from its name and keeps only the last few hours ("Retain files for", 2 h or
more), so the model's own dates (3 May 1999 for the default setup) never show up in it.  The feed
lists each volume under the time it was written (its file's modification time, at least SPACING
after the volume before, never ahead of the clock: a fast run's volumes wait their turn, at most MAX_LAG,
and one that would wait longer is skipped) and hands GR2Analyst a copy with every time
inside moved to match (level2.retime), the way NWS's displaced-real-time simulations do with
l2munger.  The copies are made when first listed and kept in output/cache/poll/<run>/<SITE>/
(the newest run's only); the run's own files keep the model's times.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import threading
import time
from pathlib import Path

from . import PROJECT, level2

CACHE = PROJECT / "output" / "cache" / "poll"
LISTED = 30                 # newest volumes in a dir.list (GR2Analyst keeps at most 24 in memory)
STALE_PART = 120.0          # s: a .part file this old is a write that died, not one in progress
# s between listed volumes.  GR2Analyst fetches only the newest volume each poll (10 s), so two listed
# within one poll (a fast run, or GR2Analyst busy downloading several sites) dropped one from its loop
SPACING = 20
# s a volume may wait for its turn.  A run making volumes faster than one every SPACING (fast models, several
# radars) queued them without end: the feed fell an hour behind the model.  Such a run's feed lists every Nth
# volume (N from the pace the run makes them, so a steady loop instead of bursts of skips; the same volumes at
# every site); the rest stay in the run's folder.  A volume that would still wait longer is left out as well
MAX_LAG = 90
MAX_STRIDE = 8
_ARCHIVE = re.compile(r"^([A-Z0-9]{4})(\d{8})_(\d{6})_V\d\d$")
_POLL = re.compile(r"^([A-Z0-9]{4})_(\d{8})_(\d{6})(?:\.bz2)?$")
_LOCK = threading.Lock()


def poll_name(site: str, t: int) -> str:
    """Iowa State's style (KTLX_20261023_190833.bz2), which GR2Analyst reads the time from."""
    return f"{site}_{dt.datetime.fromtimestamp(t, dt.timezone.utc):%Y%m%d_%H%M%S}.bz2"


def volumes(run_dir: Path | None, site: str, since: float = 0.0) -> list[tuple[Path, int]]:
    """The site's complete volumes, oldest first, each with the second (since 1970) it is listed
    under.  A volume still being written holds back the ones after it, so that every volume keeps
    the same time from one listing to the next.  ``since``: only those written from then on (a DOW's
    current deployment; its earlier volumes were made from another place)."""
    d = run_dir / site if run_dir is not None else None
    if d is None or not d.is_dir():
        return []
    files, busy, now = [], [], time.time()
    for p in d.iterdir():
        m = _ARCHIVE.match(p.name)
        try:
            if m and m[1] == site and p.is_file():
                mt = p.stat().st_mtime
                if mt >= since:
                    files.append((p.name, p, mt))
            elif p.name.startswith(".") and p.name.endswith(".part") and now - p.stat().st_mtime < STALE_PART:
                busy.append(p.name[1:-len(".part")])
        except OSError:                  # renamed or removed while listed
            continue
    files.sort()
    if busy:
        files = [f for f in files if f[0] < min(busy)]
    # every decision uses only the volumes before it, so each volume keeps its place from one listing to the next
    out, prev, last, pace, n, k = [], 0, None, float(SPACING), 1, 0
    for _, p, mtime in files:
        if last is not None:
            pace = 0.7 * pace + 0.3 * max(mtime - last, 0.5)
        last = mtime
        want = min(max(int(SPACING / pace + 0.25), 1), MAX_STRIDE)    # a little slower than the feed: the lag takes the rest
        lag = (prev + SPACING - int(mtime)) if prev else 0
        if want > n or (want < n and lag < MAX_LAG / 2):     # thin out at once, fill in again once caught up
            n = want
        k += 1
        if (k - 1) % n:
            continue
        t = max(int(mtime), prev + SPACING) if prev else int(mtime)
        if t - int(mtime) > MAX_LAG:
            continue
        prev = t
        out.append((p, t))
    return out


def _copy(run_dir: Path, site: str, src: Path, t: int) -> Path:
    """The re-timed copy of ``src`` listed at ``t``, made the first time it is asked for."""
    out = CACHE / run_dir.name / site / poll_name(site, t)
    if out.is_file():
        return out
    with _LOCK:
        if out.is_file():
            return out
        if not out.parent.is_dir():
            # a new run: drop the copies made for the others
            if CACHE.is_dir():
                for old in CACHE.iterdir():
                    if old.name != run_dir.name:
                        shutil.rmtree(old, ignore_errors=True)
            out.parent.mkdir(parents=True, exist_ok=True)
        data = level2.retime(src.read_bytes(), t * 1000)
        part = out.with_name(f".{out.name}.part")
        part.write_bytes(data)
        os.replace(part, out)
    return out


def dir_list(run_dir: Path | None, site: str, since: float = 0.0) -> tuple[bytes, int, int]:
    """(dir.list, the newest volume's time or 0, volumes listed): the newest LISTED volumes.
    Volumes made faster than one every SPACING seconds wait for their time to come: a name ahead of the clock would sort after the
    next run's first volumes in GR2Analyst."""
    now = time.time()
    vols = [v for v in volumes(run_dir, site, since) if v[1] <= now][-LISTED:]
    lines = []
    for src, t in vols:
        c = _copy(run_dir, site, src, t)
        lines.append(f"{c.stat().st_size} {c.name}\n")
    if vols:
        keep = {poll_name(site, t) for _, t in vols}
        for f in (CACHE / run_dir.name / site).iterdir():
            # copies that dropped off the list; one being sent stays (Windows keeps it open)
            if f.name not in keep and not f.name.startswith("."):
                try:
                    f.unlink()
                except OSError:
                    pass
    return "".join(lines).encode(), (vols[-1][1] if vols else 0), len(vols)


def volume_file(run_dir: Path | None, site: str, name: str, since: float = 0.0) -> tuple[Path | None, float]:
    """The file to send for ``name`` and its Last-Modified: a re-timed copy for a polling name,
    the run's own file (model times) for an archive name; (None, 0) if there is none."""
    if run_dir is None:
        return None, 0.0
    m = _ARCHIVE.match(name)
    if m and m[1] == site:
        f = run_dir / site / name
        return (f, f.stat().st_mtime) if f.is_file() else (None, 0.0)
    m = _POLL.match(name)
    if not m or m[1] != site:
        return None, 0.0
    want = f"{site}_{m[2]}_{m[3]}.bz2"
    for src, t in volumes(run_dir, site, since):
        if poll_name(site, t) == want:
            return _copy(run_dir, site, src, t), float(t)
    return None, 0.0
