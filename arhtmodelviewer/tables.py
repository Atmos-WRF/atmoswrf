"""Download the lookup tables of ArWen's Thompson and TEMPO microphysics into runtime\\tables,
where the viewer reads them, from their official releases.  gpuwm checks every file against its
size and SHA-256 pins before installing it; files already there and valid are kept, so running
this again downloads nothing.

    python -m arhtmodelviewer.tables          Thompson (mp 8 and 28, ~315 MB, FahrenheitResearch/arwen
                                        release v1.0.0) and the NOAA/HRRR Thompson set (the
                                        "wrf_39_noaa" suites, ~330 MB, arwen release v2.9.2)
    python -m arhtmodelviewer.tables --tempo  also TEMPO (mp 900, ~930 MB, NCAR/TEMPO tempo_v3.1.0)
"""
from __future__ import annotations

import os
import sys

import arhtmodelviewer  # noqa: F401  -- names the table roots: runtime\tables\...


def main() -> int:
    from gpuwm.cli import main as gpuwm

    print("Thompson microphysics tables -> " + os.environ["GPUWM_THOMPSON_TABLE_ROOT"], flush=True)
    code = gpuwm(["fetch-tables", "--thompson-fork",
                  "--thompson-fork-root", os.environ["GPUWM_THOMPSON_FORK_TABLE_ROOT"]])
    if "--tempo" in sys.argv:
        from gpuwm.tempo_assets import ensure_tempo_tables
        print("TEMPO microphysics tables -> " + os.environ["GPUWM_TEMPO_TABLE_ROOT"], flush=True)
        try:
            print(f"TEMPO tables: verified {ensure_tempo_tables()}")
        except Exception as exc:
            print(f"TEMPO tables: REFUSED: {exc}")
            code = code or 2
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
