"""NEXRAD Level II (Archive II, message 31) writer and reader.

File layout (RDA/RPG ICD 2620002 / 2620010):
  * 24-byte volume header: "AR2V0006." + "001" + modified Julian date (4) +
    milliseconds of day (4) + ICAO (4);
  * LDM records, each a 4-byte big-endian control word (the compressed length,
    negative for the volume's last record) followed by a bzip2 stream;
  * the first record is the metadata record: 134 message segments of 2432
    bytes (12-byte CTM header + 16-byte message header + data) carrying
    message 3 (performance), 5 (VCP) and 2 (RDA status);
  * the rest hold message-31 radials, 120 per record.

``read_volume`` decodes what ``encode_volume`` writes (and real message-31
files) so exports can be checked without third-party readers.
"""
from __future__ import annotations

import bz2
import datetime as dt
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SEG = 2432
_POOL = ThreadPoolExecutor(max_workers=max(2, min(12, (os.cpu_count() or 4) - 2)))

DATA_NAMES = {"REF": b"DREF", "VEL": b"DVEL", "SW": b"DSW ", "ZDR": b"DZDR",
              "PHI": b"DPHI", "RHO": b"DRHO"}
ORDER = ["REF", "VEL", "SW", "ZDR", "PHI", "RHO"]


# Level II counts days from 1 January 1970 (day 1), unsigned in each radial: a day before 1970 (ERA5 runs since
# 1940) cannot be written.  Such a volume is dated whole 28-year calendar cycles later (1901-2099 repeat every
# 28 years: the same weekday and leap years), its clock times unchanged: 11 Apr 1965 is written 11 Apr 1993.
CYCLE_MS = 10227 * 86400000             # 28 Gregorian years


def l2_shift_ms(epoch_ms: int) -> int:
    """What a time before 1970 is moved by to be written (0 from 1970 on)."""
    return 0 if epoch_ms >= 0 else -(epoch_ms // CYCLE_MS) * CYCLE_MS


def utc(epoch_ms: float) -> dt.datetime:
    """UTC datetime of ms since 1970, before 1970 too (Windows' fromtimestamp refuses negative times)."""
    return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(milliseconds=float(epoch_ms))


def _mjd(epoch_ms: int):
    days = int(epoch_ms // 86400000)
    return days + 1, int(epoch_ms - days * 86400000)


def _msg_header(buf, off, size_hw, mtype, seq, epoch_ms, nseg=1, seg=1):
    """12-byte CTM (left zero) + 16-byte message header at ``off``."""
    jd, ms = _mjd(epoch_ms)
    struct.pack_into(">HBBHHIHH", buf, off + 12, size_hw, 8, mtype, seq & 0xFFFF, jd, ms, nseg, seg)


def _angle_code(deg: float) -> int:
    """Coded binary angle (16 bit, 360/65536 deg per count)."""
    return int(round((deg % 360.0) * 65536.0 / 360.0)) & 0xFFFF


def _msg5(cuts, vcp: int, epoch_ms: int, seq: int, vel_res: int = 2) -> bytes:
    buf = bytearray(SEG)
    n = len(cuts)
    length = 22 + 46 * n                       # bytes of message-5 data
    _msg_header(buf, 0, (16 + length) // 2, 5, seq, epoch_ms)
    o = 28
    # msg size (hw), pattern type (2 = constant elevation), pattern number,
    # number of cuts, VCP version 1, clutter map group 1, velocity resolution
    # (2 = 0.5 m/s, 4 = 1 m/s), pulse width 2 (short), 4 spare bytes, sequencing,
    # supplemental data, spare
    struct.pack_into(">HHHHBBBB", buf, o, length // 2, 2, vcp, n, 1, 1, vel_res, 2)
    o += 22
    for c in cuts:
        # channel config, waveform (1 CS, 2 CD, 4 batch), super-res bits
        # (1 half-degree azimuths, 2 quarter-km REF, 4 300-km Doppler, 8 dual pol)
        wf = {"surv": 1, "dop": 2, "both": 4}[c.kind]
        sres = (0x0B if c.kind == "surv" else 0x07 if c.kind == "dop" else 0x0E) if c.super_res else 0x08
        prf_num = 1 if c.kind == "surv" else 5
        az_rate = min(int(round(360.0 / c.seconds * 16384.0 / 22.5)), 0xFFFF)   # BAMS deg/s (to ~90)
        struct.pack_into(">HBBBBHH", buf, o, _angle_code(c.elev), 0, wf, sres, prf_num,
                         15 if c.kind == "surv" else 52, az_rate)
        for k in range(6):                     # thresholds (dB * 8): REF, VEL, SW, ZDR, PHI, RHO
            struct.pack_into(">h", buf, o + 10 + 2 * k, 28)
        for k, edge in enumerate((0x1558, 0x9558, 0xEE38)):
            struct.pack_into(">HHH", buf, o + 22 + 8 * k, edge, prf_num, 0x1F)
        o += 46
    return bytes(buf)


def _msg2(vcp: int, epoch_ms: int, seq: int) -> bytes:
    buf = bytearray(SEG)
    jd, ms = _mjd(epoch_ms)
    hw = [
        0x0002,         # RDA status: operate
        0x0002,         # operability: online
        0x0002,         # control: remote
        0x0000,         # aux power
        0x0000,         # average transmitter power
        0x0000,         # horizontal reflectivity calibration correction
        0x001C,         # data transmission enabled: REF, VEL, SW
        vcp,            # VCP
        0x0000,         # control authorization
        1902,           # RDA build 19.02 (x100)
        0x0004,         # operational mode: operational
        0x0002,         # super-resolution: enabled
        0x0002,         # clutter mitigation decision: enabled
        0x0002,         # AVSET
        0x0000,         # alarm summary
        0x0000,         # command acknowledgement
        0x0000,         # channel control
        0x0000,         # spot blanking
        jd, 0,          # bypass map date / time
        jd, 0,          # clutter filter map date / time
        0x0000,         # vertical reflectivity calibration correction
        0x0000,         # transition power source
        0x0002,         # RMS control
        0x0002,         # performance check
    ]
    struct.pack_into(">" + "H" * len(hw), buf, 28, *[h & 0xFFFF for h in hw])
    _msg_header(buf, 0, (16 + 2 * len(hw)) // 2, 2, seq, epoch_ms)
    return bytes(buf)


def _metadata(cuts, vcp, epoch_ms, vel_res: int = 2) -> bytes:
    segs = [bytes(SEG)] * 131
    m3 = bytearray(SEG)
    _msg_header(m3, 0, 488, 3, 1, epoch_ms)
    segs.append(bytes(m3))
    segs.append(_msg5(cuts, vcp, epoch_ms, 2, vel_res))
    segs.append(_msg2(vcp, epoch_ms, 3))
    return b"".join(segs)


def _radials(sweep, elev_num, nsweeps, seq0, icao, lat, lon, height_m, vcp):
    """All message-31 radials of one sweep as a (nrad, L) uint8 array."""
    cut = sweep["cut"]
    nrad = int(sweep["nrad"])
    moments = sweep["moments"]               # name -> (codes (nrad, ng), ng, bits, scale, offset)
    present = [m for m in ORDER if m in moments]
    first_gate = int(sweep.get("first_gate", 2125))
    gate = int(sweep.get("gate", 250))
    t0 = int(sweep["start_ms"])
    dur = float(sweep["dur_ms"])
    az0 = float(sweep.get("az0", 0.0))
    daz = 360.0 / nrad
    sizes = {m: moments[m][1] * moments[m][2] // 8 for m in present}
    hdr = 68
    blocks = 44 + 12 + 28 + sum(28 + sizes[m] for m in present)
    rad_len = hdr + blocks
    rad_len += rad_len & 1
    L = 28 + rad_len
    buf = bytearray(L)
    _msg_header(buf, 0, (16 + rad_len) // 2, 31, 0, t0)
    d = 28
    buf[d:d + 4] = icao.encode("ascii")[:4].ljust(4, b" ")
    ptrs = []
    p = hdr
    # volume constants
    ptrs.append(p)
    buf[d + p:d + p + 4] = b"RVOL"
    struct.pack_into(">HBBffhHfffffHH", buf, d + p + 4, 44, 1, 0, lat, lon, int(round(height_m)), 20,
                     -43.0, 750.0, 750.0, 0.0, 60.0, vcp, 0)
    p += 44
    ptrs.append(p)
    buf[d + p:d + p + 4] = b"RELV"
    struct.pack_into(">Hhf", buf, d + p + 4, 12, -12, -43.0)
    p += 12
    ptrs.append(p)
    buf[d + p:d + p + 4] = b"RRAD"
    unamb = int(round(getattr(cut, "unamb_m", 4.66e5 if cut.kind == "surv" else 1.17e5) / 100.0))   # 0.1 km
    struct.pack_into(">HhffhHff", buf, d + p + 4, 28, unamb, -80.0, -80.0,
                     int(round(float(sweep.get("nyquist", cut.nyquist)) * 100)), 0, -43.0, -43.0)
    p += 28
    data_at = {}
    for m in present:
        codes, ng, bits, scale, offset = moments[m]
        ptrs.append(p)
        buf[d + p:d + p + 4] = DATA_NAMES[m]
        struct.pack_into(">IHHHHhBBff", buf, d + p + 4, 0, ng, first_gate, gate, 16, 16, 0, bits,
                         float(scale), float(offset))
        data_at[m] = d + p + 28
        p += 28 + sizes[m]
    # message-31 data header (offsets from the radar id):
    #  0 id, 4 time, 8 date, 10 azimuth number, 12 azimuth, 16 compression,
    #  17 spare, 18 radial length, 20 azimuth spacing, 21 radial status,
    #  22 elevation number, 23 cut sector, 24 elevation, 28 spot blanking,
    #  29 azimuth indexing, 30 block count, 32 block pointers
    struct.pack_into(">BBH", buf, d + 16, 0, 0, rad_len)
    struct.pack_into(">BBBB", buf, d + 20, 1 if nrad == 720 else 2, 1, elev_num, 1)
    struct.pack_into(">f", buf, d + 24, float(cut.elev))
    struct.pack_into(">BBH", buf, d + 28, 0, 50 if nrad == 720 else 100, len(ptrs))
    for k, pp in enumerate(ptrs):
        struct.pack_into(">I", buf, d + 32 + 4 * k, pp)
    arr = np.tile(np.frombuffer(bytes(buf), dtype=np.uint8), (nrad, 1))
    ri = np.arange(nrad)
    tms = t0 + np.rint(dur * ri / nrad).astype(np.int64)
    days = tms // 86400000
    jd = (days + 1).astype(">u2")
    jms = (tms - days * 86400000).astype(">u4")
    seq = ((seq0 + ri) & 0xFFFF).astype(">u2")
    az = ((az0 + (ri + 0.5) * daz) % 360.0).astype(">f4")
    status = np.ones(nrad, dtype=np.uint8)
    status[0] = 3 if elev_num == 1 else 0
    status[-1] = 4 if elev_num == nsweeps else 2

    def put(off, vals):
        v = np.ascontiguousarray(vals)
        arr[:, off:off + v.itemsize] = v.view(np.uint8).reshape(nrad, v.itemsize)

    put(12 + 4, seq)
    put(12 + 6, jd)
    put(12 + 8, jms)
    put(d + 4, jms)
    put(d + 8, jd)
    put(d + 10, (ri + 1).astype(">u2"))
    put(d + 12, az)
    arr[:, d + 21] = status
    for m in present:
        codes, ng, bits, scale, offset = moments[m]
        raw = np.ascontiguousarray(codes)
        if bits == 16:
            raw = raw.astype(">u2")
        nbytes = sizes[m]
        arr[:, data_at[m]:data_at[m] + nbytes] = raw.view(np.uint8).reshape(nrad, nbytes)
    return arr


def encode_volume(icao: str, lat: float, lon: float, height_m: float, vcp: int,
                  start_ms: int, sweeps: list) -> bytes:
    """``sweeps``: dicts with cut (radar.Cut), nrad, start_ms, dur_ms, az0 and
    moments {name: (codes (nrad, ngates), ngates, bits, scale, offset)}."""
    shift = l2_shift_ms(int(start_ms))
    if shift:                                       # before 1970: dated 28-year cycles later (l2_shift_ms)
        start_ms = int(start_ms) + shift
        sweeps = [dict(s, start_ms=int(s["start_ms"]) + shift) for s in sweeps]
    cuts = [s["cut"] for s in sweeps]
    # the whole volume's velocity in 1 m/s steps when its sweeps carry them (dealiased winds past 63.5 m/s);
    # every sweep alike, as message 5 says -- GR2Analyst drops the velocity of sweeps that differ from it
    one = any(s["moments"].get("VEL", (None,) * 5)[3] == 1.0 for s in sweeps)
    records = [_metadata(cuts, vcp, start_ms, 4 if one else 2)]
    seq = 10
    for e, s in enumerate(sweeps):
        arr = _radials(s, e + 1, len(sweeps), seq, icao, lat, lon, height_m, vcp)
        seq += arr.shape[0]
        for i in range(0, arr.shape[0], 120):
            records.append(arr[i:i + 120].tobytes())
    jd, ms = _mjd(start_ms)
    out = [b"AR2V0006.001" + struct.pack(">ii", jd, ms) + icao.encode("ascii")[:4].ljust(4, b" ")]
    comp = list(_POOL.map(lambda r: bz2.compress(r, 5), records))
    for i, c in enumerate(comp):
        out.append(struct.pack(">i", -len(c) if i == len(comp) - 1 else len(c)))
        out.append(c)
    return b"".join(out)


def archive_name(icao: str, epoch_ms: int) -> str:
    t = utc(int(epoch_ms) + l2_shift_ms(int(epoch_ms)))
    return "%s%s_V06" % (icao.upper(), t.strftime("%Y%m%d_%H%M%S"))


def _shift_date(buf, off_jd, off_ms, shift_ms, jd_fmt):
    """Move the (Julian date, ms of day) pair at ``off_jd`` / ``off_ms`` by ``shift_ms``."""
    (jd,) = struct.unpack_from(jd_fmt, buf, off_jd)
    (ms,) = struct.unpack_from(">I", buf, off_ms)
    if jd <= 0:                                 # an empty segment: no time to move
        return
    jd, ms = _mjd((jd - 1) * 86400000 + ms + shift_ms)
    struct.pack_into(jd_fmt, buf, off_jd, jd)
    struct.pack_into(">I", buf, off_ms, ms)


def retime(data: bytes, end_ms: int) -> bytes:
    """The same volume with every time in it (volume header, message headers, radials)
    moved so that its last radial is at ``end_ms`` (ms since 1970), like NWS's l2munger.
    GR2Analyst polls only volumes from the last few hours: the polling feed hands it the
    model's volumes re-timed to when the viewer made them."""
    if not data.startswith(b"AR2V"):
        raise ValueError("not an Archive II file")
    pos, recs = 24, []
    while pos + 4 <= len(data):
        (n,) = struct.unpack(">i", data[pos:pos + 4])
        recs.append(data[pos + 4:pos + 4 + abs(n)])
        pos += 4 + abs(n)
        if n < 0:
            break
    raw = [bytearray(r) for r in _POOL.map(bz2.decompress, recs)]
    radials, segments = [], []                  # offsets of message-31 and fixed-size messages
    last = None
    for k, rec in enumerate(raw):
        p = 0
        while p + 28 <= len(rec):
            size_hw, _, mtype = struct.unpack_from(">HBB", rec, p + 12)
            if mtype != 31:
                if mtype:
                    segments.append((k, p))
                p += SEG
                continue
            radials.append((k, p))
            jd, ms = struct.unpack_from(">HI", rec, p + 18)
            t = (jd - 1) * 86400000 + ms
            last = t if last is None else max(last, t)
            p += 12 + 2 * size_hw
    if last is None:
        raise ValueError("no radials in the volume")
    shift = int(end_ms) - last
    for k, p in segments:
        _shift_date(raw[k], p + 18, p + 20, shift, ">H")
    for k, p in radials:
        _shift_date(raw[k], p + 18, p + 20, shift, ">H")
        _shift_date(raw[k], p + 28 + 8, p + 28 + 4, shift, ">H")
    hdr = bytearray(data[:24])
    _shift_date(hdr, 12, 16, shift, ">i")
    comp = list(_POOL.map(lambda r: bz2.compress(bytes(r), 5), raw))
    out = [bytes(hdr)]
    for i, c in enumerate(comp):
        out.append(struct.pack(">i", -len(c) if i == len(comp) - 1 else len(c)))
        out.append(c)
    return b"".join(out)


# ------------------------------------------------------------------ reader
def read_volume(data: bytes) -> dict:
    """Decode a message-31 Archive II volume (enough to verify exports)."""
    hdr = data[:24]
    if not hdr.startswith(b"AR2V"):
        raise ValueError("not an Archive II file")
    icao = hdr[20:24].decode("ascii", "replace")
    pos = 24
    raw = []
    while pos + 4 <= len(data):
        (n,) = struct.unpack(">i", data[pos:pos + 4])
        pos += 4
        size = abs(n)
        raw.append(bz2.decompress(data[pos:pos + size]))
        pos += size
        if n < 0:
            break
    meta = raw[0]
    msgs = {}
    for k in range(len(meta) // SEG):
        seg = meta[k * SEG:(k + 1) * SEG]
        mtype = seg[12 + 3]
        if mtype:
            msgs[mtype] = seg
    vcp = None
    ncuts = None
    if 5 in msgs:
        _, _, vcp, ncuts = struct.unpack(">HHHH", msgs[5][28:36])
    sweeps = {}
    for rec in raw[1:]:
        p = 0
        while p + 28 <= len(rec):
            size_hw, chan, mtype = struct.unpack(">HBB", rec[p + 12:p + 16])
            if mtype != 31:
                p += SEG
                continue
            d = p + 28
            az = struct.unpack(">f", rec[d + 12:d + 16])[0]
            elnum = rec[d + 22]
            elev = struct.unpack(">f", rec[d + 24:d + 28])[0]
            nblk = struct.unpack(">H", rec[d + 30:d + 32])[0]
            ptrs = struct.unpack(">" + "I" * nblk, rec[d + 32:d + 32 + 4 * nblk])
            sw = sweeps.setdefault(elnum, {"elev": elev, "az": [], "moments": {}})
            sw["az"].append(az)
            for ptr in ptrs:
                name = rec[d + ptr:d + ptr + 4]
                if name[:1] != b"D":
                    if name == b"RRAD":
                        sw["nyquist"] = struct.unpack(">h", rec[d + ptr + 16:d + ptr + 18])[0] / 100.0
                    if name == b"RVOL":
                        lat, lon = struct.unpack(">ff", rec[d + ptr + 8:d + ptr + 16])
                        sw["site"] = (lat, lon)
                    continue
                ng, first, gate = struct.unpack(">HHH", rec[d + ptr + 8:d + ptr + 14])
                bits = rec[d + ptr + 19]
                scale, offset = struct.unpack(">ff", rec[d + ptr + 20:d + ptr + 28])
                nb = ng * bits // 8
                g = np.frombuffer(rec[d + ptr + 28:d + ptr + 28 + nb], dtype=">u2" if bits == 16 else np.uint8)
                vals = np.where(g >= 2, (g.astype(np.float32) - offset) / scale, np.nan)
                mname = name[1:].decode().strip()
                sw["moments"].setdefault(mname, []).append(vals)
                sw.setdefault("gates", {})[mname] = (first, gate)        # first gate and spacing (m)
            p += 12 + 2 * size_hw
    for sw in sweeps.values():
        sw["az"] = np.array(sw["az"])
        sw["moments"] = {k: np.array(v) for k, v in sw["moments"].items()}
    return {"icao": icao, "vcp": vcp, "ncuts": ncuts, "sweeps": sweeps}
