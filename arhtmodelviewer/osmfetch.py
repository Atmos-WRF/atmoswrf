"""OpenStreetMap through the Overpass API, for the 3-D view's and chase mode's roads (landscape.py) and chase mode's
buildings (chase.py): cells fetched in the background and kept on disk by their owners.

Overpass is a shared, often busy service (two queries at a time per address, servers that stall for minutes), so:
  - a queue per kind of cell that always fetches the cell asked for most recently (where the players are now),
    and forgets cells nobody has asked for in FORGET s;
  - short timeouts, a server that fails rests a while, and when every server has failed the whole thing waits
    (BACKOFF, doubling to BACKOFF_MAX) instead of chewing through the queue;
  - status(): how many cells are waiting and whether Overpass is answering, for the page to say so."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .tiles import UA

SERVERS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter")
TIMEOUT = 100.0                  # s: a query's longest wait for its answer
FORGET = 600.0                   # s: a cell nobody asked for since is dropped from the queue
BACKOFF, BACKOFF_MAX = 45.0, 300.0
_SLOTS = threading.BoundedSemaphore(2)          # Overpass gives an address two queries at a time
_G = threading.Lock()
_STATE = {"rest": {}, "down_until": 0.0, "backoff": BACKOFF, "ok_at": 0.0, "fail_at": 0.0, "error": ""}


def overpass(query: str) -> dict:
    """The answer to an Overpass QL query (JSON), from the first server that is not resting; raises when none
    answers (and then the whole service rests)."""
    with _SLOTS:
        err = None
        now = time.time()
        with _G:
            order = sorted(SERVERS, key=lambda s: _STATE["rest"].get(s, 0.0) > now)
        for url in order:
            with _G:
                if _STATE["rest"].get(url, 0.0) > time.time():
                    continue
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    d = json.load(r)
                if d.get("remark") and "runtime error" in str(d["remark"]).lower():
                    raise RuntimeError(d["remark"][:200])
                with _G:
                    _STATE.update(ok_at=time.time(), backoff=BACKOFF, error="")
                return d
            except Exception as e:                       # busy, refusing or down: rest it, the next server
                err = e
                code = e.code if isinstance(e, urllib.error.HTTPError) else None
                with _G:
                    _STATE["rest"][url] = time.time() + (120.0 if code in (429, 503, 504) else 60.0)
        with _G:
            _STATE["down_until"] = time.time() + _STATE["backoff"]
            _STATE["backoff"] = min(_STATE["backoff"] * 2, BACKOFF_MAX)
            _STATE.update(fail_at=time.time(), error=str(err)[:200])
        raise RuntimeError(f"Overpass: {err}")


def busy() -> bool:
    """Overpass is not answering right now (every server failed lately)."""
    with _G:
        return _STATE["fail_at"] > _STATE["ok_at"] and time.time() - _STATE["fail_at"] < 600.0


class CellQueue:
    """Cells to fetch with fetch(*cell) (which writes its file), newest wish first, one worker thread."""

    def __init__(self, name: str, fetch):
        self.name, self.fetch = name, fetch
        self.wanted: dict = {}                     # cell -> when it was last asked for
        self.running: set = set()
        self.failed: dict = {}                     # cell -> not before (a cell that failed waits its turn)
        self.lock = threading.Lock()
        self.worker = None

    def want(self, cell) -> bool:
        """Ask for a cell (not on disk): True while it is waiting or being fetched."""
        with self.lock:
            self.wanted[cell] = time.time()
            if self.worker is None:
                self.worker = threading.Thread(target=self._work, name=f"osm-{self.name}", daemon=True)
                self.worker.start()
        return True

    def _work(self):
        while True:
            with self.lock:
                now = time.time()
                for c in [c for c, t in self.wanted.items() if now - t > FORGET]:
                    self.wanted.pop(c)
                todo = [c for c in self.wanted if c not in self.running and self.failed.get(c, 0.0) <= now]
                if not todo and self.wanted:            # only cells that just failed: a short nap, then again
                    nap = min(self.failed.get(c, now) for c in self.wanted) - now
                    if nap > 0:
                        self.lock.release()
                        try:
                            time.sleep(min(nap, 10.0))
                        finally:
                            self.lock.acquire()
                        continue
                if not todo:
                    self.worker = None
                    return
                cell = max(todo, key=lambda c: self.wanted[c])
                self.running.add(cell)
            wait = _STATE["down_until"] - time.time()
            if wait > 0:                               # Overpass is resting: so do we
                time.sleep(min(wait, 15.0))
                with self.lock:
                    self.running.discard(cell)
                continue
            try:
                self.fetch(*cell)
                with self.lock:
                    self.wanted.pop(cell, None)
                    self.failed.pop(cell, None)
            except Exception:
                with self.lock:                         # stays wanted: again after the others, in a minute or so
                    self.failed[cell] = time.time() + 75.0
            with self.lock:
                self.running.discard(cell)
