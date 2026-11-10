"""Model forcings applied to ArWen patches between dynamics steps.

* Storm starts: Naylor & Gilmore (2012) updraft nudging (as used in CM1),
  carried along with the storm motion so the young storm is never left
  behind by its own push, and the classic WRF/CM1 warm bubble.
* Relaxation rim: WRF's lateral relax zone idea applied at every patch
  edge, toward the environment, so open boundaries stay quiet.
* Environment drift correction: a horizontally uniform large-scale forcing
  that holds the environment profile steady against surface drag, mixing
  and Coriolis (see ``Engine._drift_correction``).
* Activity map: where the atmosphere is disturbed, for patch management.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_KERNELS = {}

_SRC = r"""
__device__ __forceinline__ float sstep(float a, float b, float x) {
    float t = fminf(fmaxf((x - a) / (b - a), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Smooth 3-D value noise in [-1, 1] on a unit lattice (seeded): the thermals' ragged shapes.
__device__ __forceinline__ float lattice(int x, int y, int z, int s) {
    unsigned int h = (unsigned int)x * 374761393u + (unsigned int)y * 668265263u
                   + (unsigned int)z * 2246822519u + (unsigned int)s * 3266489917u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0xffffffu) / 8388607.5f - 1.0f;
}
__device__ float vnoise(float x, float y, float z, int s) {
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float ux = x - fx, uy = y - fy, uz = z - fz;
    ux = ux * ux * (3.0f - 2.0f * ux); uy = uy * uy * (3.0f - 2.0f * uy); uz = uz * uz * (3.0f - 2.0f * uz);
    float a = lattice(ix, iy, iz, s) + ux * (lattice(ix + 1, iy, iz, s) - lattice(ix, iy, iz, s));
    float b = lattice(ix, iy + 1, iz, s) + ux * (lattice(ix + 1, iy + 1, iz, s) - lattice(ix, iy + 1, iz, s));
    float c = lattice(ix, iy, iz + 1, s) + ux * (lattice(ix + 1, iy, iz + 1, s) - lattice(ix, iy, iz + 1, s));
    float d = lattice(ix, iy + 1, iz + 1, s) + ux * (lattice(ix + 1, iy + 1, iz + 1, s) - lattice(ix, iy + 1, iz + 1, s));
    float e = a + uy * (b - a), f = c + uy * (d - c);
    return e + uz * (f - e);
}

// Updraft nudging (Naylor & Gilmore 2012), grown from nothing over the ramp on a fixed footprint.
// ev: NEV x 20 floats
//   x0, y0 (m, at t0), vx, vy (m/s), t0, t1 (s), radius (m), zc (m), half depth (m), wmax (m/s),
//   ramp (s), tau (s), then its shape: aspect (>= 1, stretched along angle, area kept), angle (rad),
//   lobes (0..0.5: a lumpy outline), lobe phase (rad), tilt x, tilt y (m of lean per m up),
//   noise (0..1: 3-D noise roughening the outline and the strength inside), noise seed
extern "C" __global__ void nudge_w(float* w, const float* zw, const float* ev, int nev,
                                   int nz1, int ny, int nx, float xw, float ys, float dx,
                                   float t, float dt)
{
    long long n = (long long)nz1 * ny * nx;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % nx;
    int j = (id / nx) % ny;
    int k = id / ((long long)nx * ny);
    if (k == 0 || k == nz1 - 1) return;
    float x = xw + (i + 0.5f) * dx;
    float y = ys + (j + 0.5f) * dx;
    float z = zw[k];
    float best = -1.0e30f, rate = 0.0f;
    for (int e = 0; e < nev; ++e) {
        const float* E = ev + 20 * e;
        float t0 = E[4], t1 = E[5];
        if (t < t0 || t > t1) continue;
        float g = sstep(t0, t0 + fmaxf(E[10], 1.0f), t);
        float h = 1.0f - sstep(t1 - 0.2f * (t1 - t0), t1, t);
        float rz = (z - E[7]) / E[8];
        if (fabsf(rz) >= 1.3f) continue;
        // aspect 0: the classic push (1.6 and before), widening from half its radius and jumping to 40 % at once
        bool classic = E[12] < 0.5f;
        float xc = E[0] + E[2] * (t - t0) + E[16] * (z - E[7]);
        float yc = E[1] + E[3] * (t - t0) + E[17] * (z - E[7]);
        float ddx = x - xc, ddy = y - yc;
        float R = classic ? E[6] * (0.5f + 0.5f * g) : E[6];
        if (ddx * ddx + ddy * ddy > 4.0f * R * R * fmaxf(E[12], 1.0f)) continue;
        float ca = cosf(E[13]), sa = sinf(E[13]), st = sqrtf(fmaxf(E[12], 1.0f));
        float u = (ddx * ca + ddy * sa) / st;
        float v = (-ddx * sa + ddy * ca) * st;
        float lobe = 1.0f;
        if (E[14] > 0.0f) {
            float th = atan2f(v, u), ph = E[15];
            lobe = 1.0f + E[14] * (0.55f * cosf(2.0f * th + ph) + 0.35f * cosf(3.0f * th + 2.3f * ph)
                                   + 0.25f * cosf(5.0f * th + 4.1f * ph));
            lobe = fmaxf(lobe, 0.45f);
        }
        float rh = (u * u + v * v) / (R * R * lobe * lobe);
        float r = sqrtf(rh + rz * rz);
        float amp = 1.0f;
        if (E[18] > 0.0f) {
            // the noise lattice rides with the thermal; ~0.45 R across, ~0.7 half depth tall
            int sd = (int)E[19];
            float L = 0.45f * R, Lz = 0.7f * E[8];
            float px = (x - E[0] - E[2] * (t - t0)) / L, py = (y - E[1] - E[3] * (t - t0)) / L, pz = z / Lz;
            r *= 1.0f + 0.35f * E[18] * vnoise(px, py, pz, sd);
            amp = fmaxf(1.0f + 0.6f * E[18] * vnoise(1.7f * px + 31.0f, 1.7f * py + 17.0f, 1.7f * pz, sd + 7), 0.2f);
        }
        if (r >= 1.0f) continue;
        float c = cosf(1.5707963f * r);
        float target = E[9] * (classic ? 0.4f + 0.6f * g : g) * amp * c * c;
        if (target > best) { best = target; rate = h * fminf(dt / fmaxf(E[11], dt), 1.0f); }
    }
    if (rate > 0.0f) {
        float wv = w[id];
        if (wv < best) w[id] = wv + rate * (best - wv);
    }
}

// Warm bubble added to theta' once: amp * cos^2(pi/2 r) in an ellipsoid.
extern "C" __global__ void add_bubble(float* thp, const float* zh, int nz, int ny, int nx,
                                      float xw, float ys, float dx, float xc, float yc,
                                      float zc, float rad, float hdep, float amp)
{
    long long n = (long long)nz * ny * nx;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % nx;
    int j = (id / nx) % ny;
    int k = id / ((long long)nx * ny);
    float x = xw + (i + 0.5f) * dx - xc;
    float y = ys + (j + 0.5f) * dx - yc;
    float rz = (zh[k] - zc) / hdep;
    float r = sqrtf((x * x + y * y) / (rad * rad) + rz * rz);
    if (r < 1.0f) {
        float c = cosf(1.5707963f * r);
        thp[id] += amp * c * c;
    }
}

// Storm persistence, the inflow's cap: around deep updrafts, warm anomalies of
// sinking, cloud-free air in the capping layer (z_lo..z_hi) relax back toward
// the environment (theta' = the air-mass target, 0 without air masses).  That
// warming is the forward flank's precipitation-dragged descent in a stable
// layer, which in a steady environment builds up ahead of the storm and caps
// its own inflow.  Updrafts, clouds and cold air are never touched.
extern "C" __global__ void ease_cap(float* thp, const float* w, const float* qc, const float* zh,
                                    const float* cores, int nc, int nz, int ny, int nx,
                                    float xw, float ys, float dx, float r_in, float r_out,
                                    float z_lo, float z_hi, float rate)
{
    long long n = (long long)nz * ny * nx;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % nx;
    int j = (id / nx) % ny;
    int k = id / ((long long)nx * ny);
    float z = zh[k];
    if (z < z_lo || z > z_hi) return;
    float th = thp[id];
    if (th <= 0.0f) return;
    long long plane = (long long)nx * ny;
    float wv = 0.5f * (w[id] + w[id + plane]);
    if (wv > 0.25f || qc[id] > 1.0e-5f) return;
    float x = xw + (i + 0.5f) * dx;
    float y = ys + (j + 0.5f) * dx;
    float best = 0.0f;
    for (int c = 0; c < nc; ++c) {
        float ddx = x - cores[3 * c], ddy = y - cores[3 * c + 1];
        float d = sqrtf(ddx * ddx + ddy * ddy);
        if (d <= r_in || d >= r_out) continue;
        float t = (d - 0.7f * r_out) / (0.3f * r_out);
        float taper = t <= 0.0f ? 1.0f : (1.0f - t) * (1.0f - t);
        best = fmaxf(best, cores[3 * c + 2] * taper);
    }
    // vertical taper: full strength in the middle of the layer
    float zr = (z - z_lo) / (z_hi - z_lo);
    float vz = sinf(3.14159265f * zr);
    if (best > 0.0f) thp[id] = th - rate * best * vz * th;
}

// Storm persistence: gentle low-level lift under existing deep updrafts.
// cores: NC x 3 floats (x, y world m, strength 0..1).  Target w(z) peaks at
// zpk with half-width zhw; horizontal Gaussian radius rad.
extern "C" __global__ void keep_alive(float* w, const float* zw, const float* cores, int nc,
                                      int nz1, int ny, int nx, float xw, float ys, float dx,
                                      float amp, float rad, float zpk, float zhw, float rate)
{
    long long n = (long long)nz1 * ny * nx;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % nx;
    int j = (id / nx) % ny;
    int k = id / ((long long)nx * ny);
    if (k == 0 || k == nz1 - 1) return;
    float z = zw[k];
    float rz = (z - zpk) / zhw;
    if (rz * rz >= 1.0f) return;
    float vz = cosf(1.5707963f * rz);
    vz *= vz;
    float x = xw + (i + 0.5f) * dx;
    float y = ys + (j + 0.5f) * dx;
    float best = 0.0f;
    float r2max = 9.0f * rad * rad;
    for (int c = 0; c < nc; ++c) {
        float ddx = x - cores[3 * c], ddy = y - cores[3 * c + 1];
        float d2 = ddx * ddx + ddy * ddy;
        if (d2 > r2max) continue;
        float tgt = amp * cores[3 * c + 2] * expf(-d2 / (rad * rad)) * vz;
        best = fmaxf(best, tgt);
    }
    if (best > 0.05f) {
        float wv = w[id];
        if (wv < best) w[id] = wv + rate * (best - wv);
    }
}

// Storm persistence, the updraft root: in w levels k0..k0+nk-1 (0.5-2.5 km), near a
// deep updraft's root (cores: NC x 3, strength 0..1), rising air (w > w_min) that is
// colder than its surroundings (virtual theta' below the surroundings' mean at that
// level, surr: NC x nk x 2 = theta', qv) gets an upward acceleration of gain x its
// negative buoyancy -- the lift the mesocyclone's low pressure gives that air while
// the storm is strong.  Sinking or still air and buoyant air are never touched.
extern "C" __global__ void lift_root(float* w, const float* thp, const float* qv, const float* th0,
                                     const float* cores, const float* surr, int nc, int k0, int nk,
                                     int ny, int nx, float xw, float ys, float dx, float gain,
                                     float rad, float w_min, float a_max, float dt)
{
    long long plane = (long long)nx * ny;
    long long n = (long long)nk * plane;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % nx;
    int j = (id / nx) % ny;
    int kk = id / plane;
    int k = k0 + kk;
    long long iw = (long long)k * plane + (long long)j * nx + i;    // w level k = mass level k's bottom face
    float wv = w[iw];
    if (wv <= w_min) return;
    float x = xw + (i + 0.5f) * dx;
    float y = ys + (j + 0.5f) * dx;
    float best = 0.0f;
    int cb = -1;
    float r2max = 9.0f * rad * rad;
    for (int c = 0; c < nc; ++c) {
        float ddx = x - cores[3 * c], ddy = y - cores[3 * c + 1];
        float d2 = ddx * ddx + ddy * ddy;
        if (d2 > r2max) continue;
        float g = cores[3 * c + 2] * expf(-d2 / (rad * rad));
        if (g > best) { best = g; cb = c; }
    }
    if (cb < 0 || best < 0.02f) return;
    float th = 0.5f * (thp[iw - plane] + thp[iw]);
    float q = 0.5f * (qv[iw - plane] + qv[iw]);
    const float* S = surr + 2 * ((long long)cb * nk + kk);
    float t0 = th0[k];
    float def = (S[0] - th) + 0.61f * t0 * (S[1] - q);             // K of virtual theta
    if (!(def > 0.0f)) return;                                        // buoyant air (or NaN): left alone
    float vz = sinf(3.14159265f * (kk + 0.5f) / nk);                 // full in the middle of the layer
    float a = fminf(gain * best * vz * 9.81f * def / t0, a_max);
    w[iw] = wv + a * dt;
}

// Relaxation rim toward the environment (env[k], or 0 when env == nullptr).
// The field has fy x fx horizontal points; xs/ys mark face staggering.
extern "C" __global__ void relax_rim(float* f, const float* env, int nk, int fy, int fx,
                                     int xs, int ys, float width, float coef,
                                     int open_w, int open_e, int open_s, int open_n)
{
    long long n = (long long)nk * fy * fx;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % fx;
    int j = (id / fx) % fy;
    int k = id / ((long long)fx * fy);
    float off_x = xs ? 0.0f : 0.5f;
    float off_y = ys ? 0.0f : 0.5f;
    float nxc = xs ? (float)(fx - 1) : (float)fx;
    float nyc = ys ? (float)(fy - 1) : (float)fy;
    float d = 1.0e9f;
    float xi = i + off_x, yj = j + off_y;
    if (open_w) d = fminf(d, xi);
    if (open_e) d = fminf(d, nxc - xi);
    if (open_s) d = fminf(d, yj);
    if (open_n) d = fminf(d, nyc - yj);
    if (d >= width) return;
    float a = 1.0f - d / width;
    float c = coef * a * a;
    float target = env ? env[k] : 0.0f;
    f[id] += c * (target - f[id]);
}

// Air masses (boundaries.py): per-column maps (dT, H, dq, Hq, du, dv, Hw, nudge),
// each ny x nx at mass points.  The target is the environment + the air mass;
// the weight is the larger of the edge rim's and "keep the air mass"'s (low levels,
// deep inside an air mass).  kind: 0 thp, 1 qv, 2 u, 3 v.
__device__ __forceinline__ float am_prof(float z, float H)
{
    H = fmaxf(H, 50.0f);
    float t = fminf(fmaxf((z - 0.7f * H) / (0.45f * H), 0.0f), 1.0f);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float am_map(const float* maps, int c, int ny, int nx, int j, int i,
                                        int xs, int ys)
{
    const float* m = maps + (long long)c * ny * nx;
    if (xs) {
        int a = max(i - 1, 0), b = min(i, nx - 1);
        return 0.5f * (m[(long long)j * nx + a] + m[(long long)j * nx + b]);
    }
    if (ys) {
        int a = max(j - 1, 0), b = min(j, ny - 1);
        return 0.5f * (m[(long long)a * nx + i] + m[(long long)b * nx + i]);
    }
    return m[(long long)j * nx + i];
}

extern "C" __global__ void relax_airmass(float* f, const float* env, const float* maps, const float* zh,
                                         int kind, int nk, int ny, int nx, int fy, int fx, int xs, int ys,
                                         float width, float coef, float keep_coef, int rim_on, float hold_coef)
{
    long long n = (long long)nk * fy * fx;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % fx;
    int j = (id / fx) % fy;
    int k = id / ((long long)fx * fy);
    float c = 0.0f;
    float hc = 0.0f;
    if (rim_on) {
        float xi = i + (xs ? 0.0f : 0.5f), yj = j + (ys ? 0.0f : 0.5f);
        float nxc = xs ? (float)(fx - 1) : (float)fx;
        float nyc = ys ? (float)(fy - 1) : (float)fy;
        float d = fminf(fminf(xi, nxc - xi), fminf(yj, nyc - yj));
        if (d < width) {
            float a = 1.0f - d / width;
            c = coef * a * a;
        }
    }
    float z = zh[k];
    float H = am_map(maps, kind == 0 ? 1 : (kind == 1 ? 3 : 6), ny, nx, j, i, xs, ys);
    float nud = am_map(maps, kind >= 2 ? 8 : 7, ny, nx, j, i, xs, ys);   // winds: the wind-only keep too
    float top = 1.15f * H;
    float t = fminf(fmaxf((z - top - 300.0f) / 1000.0f, 0.0f), 1.0f);
    float taper = 1.0f - t * t * (3.0f - 2.0f * t);
    if (nud > 0.0f && keep_coef > 0.0f)
        c = fmaxf(c, keep_coef * nud * taper);
    float pc = 0.0f;
    if (kind <= 1 && hold_coef > 0.0f)
        hc = hold_coef * am_map(maps, 9, ny, nx, j, i, xs, ys) * taper;
    if (kind == 0 && hold_coef > 0.0f)
        pc = hold_coef * am_map(maps, 10, ny, nx, j, i, xs, ys) * taper;
    if (c <= 0.0f && hc <= 0.0f && pc <= 0.0f) return;
    float target;
    float p = am_prof(z, H);
    if (kind == 0)      target = am_map(maps, 0, ny, nx, j, i, xs, ys) * p;
    else if (kind == 1) target = env[k] * (1.0f - am_map(maps, 2, ny, nx, j, i, xs, ys) * p);
    else if (kind == 2) target = env[k] + am_map(maps, 4, ny, nx, j, i, xs, ys) * p;
    else                target = env[k] + am_map(maps, 5, ny, nx, j, i, xs, ys) * p;
    // the frontal zone's hold is one-sided: warmer / moister air than the air mass is pulled back, a storm's
    // colder outflow is left alone; a stationary boundary's pin also warms its own cool air pushed past the line
    // (only air at most the air mass's |dT| + 0.5 K below the target: colder is storm outflow)
    if (hc > c && f[id] > target) c = hc;
    if (pc > c && f[id] < target && f[id] > target + am_map(maps, 11, ny, nx, j, i, xs, ys) - 0.5f) c = pc;
    if (c <= 0.0f) return;
    f[id] += c * (target - f[id]);
}

// Synoptic mode (synoptic.SynopticAirMass): 3-D targets (nk, cy, cx) on a coarse grid, one point every
// ``stride`` mass cells (at cell centre q*stride + stride/2), interpolated bilinearly.  The weight is the
// edge rim's, or below keep_top the interior keep (the day's heating and moistening), skipped where the
// storms have changed the air: guard_mode 1, f below target - guard (a cold pool); 2, f below
// target * (1 - guard) (dried outflow).  Terrain patches with boundaries also pass the air masses' own keep
// as on flat ground (relax_airmass): am_w (weight) and am_h (depth, m) per coarse column at rate am_coef,
// the frontal zone's one-sided hold am_hold at hold_coef, and a stationary boundary's pin (am_pin, down to
// am_pin_lo below the target) (0 pointers / rates: none).
extern "C" __global__ void relax_target3d(float* f, const float* tgt, const float* zh, int nk, int fy, int fx,
                                          int nx, int ny, int cy, int cx, float stride, float width, float coef,
                                          float keep_coef, float keep_top, int rim_on, int guard_mode, float guard,
                                          const float* am_w, const float* am_h, float am_coef,
                                          const float* am_hold, float hold_coef,
                                          const float* am_pin, const float* am_pin_lo)
{
    long long n = (long long)nk * fy * fx;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int i = id % fx;
    int j = (id / fx) % fy;
    int k = id / ((long long)fx * fy);
    int xs = fx == nx + 1, ys = fy == ny + 1;
    float px = i + (xs ? 0.0f : 0.5f), py = j + (ys ? 0.0f : 0.5f);
    float c = 0.0f;
    if (rim_on) {
        float nxc = xs ? (float)(fx - 1) : (float)fx;
        float nyc = ys ? (float)(fy - 1) : (float)fy;
        float d = fminf(fminf(px, nxc - px), fminf(py, nyc - py));
        if (d < width) {
            float a = 1.0f - d / width;
            c = coef * a * a;
        }
    }
    float z = zh[k];
    float ck = 0.0f;
    if (keep_coef > 0.0f && z < keep_top) {
        float t = z / keep_top;
        ck = keep_coef * (1.0f - t * t * (3.0f - 2.0f * t));
    }
    float gx = fminf(fmaxf(px / stride - 0.5f, 0.0f), (float)(cx - 1));
    float gy = fminf(fmaxf(py / stride - 0.5f, 0.0f), (float)(cy - 1));
    int i0 = (int)gx, j0 = (int)gy;
    int i1 = min(i0 + 1, cx - 1), j1 = min(j0 + 1, cy - 1);
    float tx = gx - i0, ty = gy - j0;
    float hc = 0.0f, pc = 0.0f, plo = 0.0f;
    if (am_w != 0 && (am_coef > 0.0f || hold_coef > 0.0f)) {
        float H = (am_h[j0 * cx + i0] * (1.0f - tx) + am_h[j0 * cx + i1] * tx) * (1.0f - ty)
                + (am_h[j1 * cx + i0] * (1.0f - tx) + am_h[j1 * cx + i1] * tx) * ty;
        float t = fminf(fmaxf((z - 1.15f * H - 300.0f) / 1000.0f, 0.0f), 1.0f);
        float taper = 1.0f - t * t * (3.0f - 2.0f * t);
        float aw = (am_w[j0 * cx + i0] * (1.0f - tx) + am_w[j0 * cx + i1] * tx) * (1.0f - ty)
                 + (am_w[j1 * cx + i0] * (1.0f - tx) + am_w[j1 * cx + i1] * tx) * ty;
        c = fmaxf(c, am_coef * aw * taper);
        if (am_hold != 0 && hold_coef > 0.0f) {
            float ah = (am_hold[j0 * cx + i0] * (1.0f - tx) + am_hold[j0 * cx + i1] * tx) * (1.0f - ty)
                     + (am_hold[j1 * cx + i0] * (1.0f - tx) + am_hold[j1 * cx + i1] * tx) * ty;
            hc = hold_coef * ah * taper;
        }
        if (am_pin != 0 && hold_coef > 0.0f) {
            float ap = (am_pin[j0 * cx + i0] * (1.0f - tx) + am_pin[j0 * cx + i1] * tx) * (1.0f - ty)
                     + (am_pin[j1 * cx + i0] * (1.0f - tx) + am_pin[j1 * cx + i1] * tx) * ty;
            pc = hold_coef * ap * taper;
            plo = fminf(fminf(am_pin_lo[j0 * cx + i0], am_pin_lo[j0 * cx + i1]),
                        fminf(am_pin_lo[j1 * cx + i0], am_pin_lo[j1 * cx + i1]));
        }
    }
    if (c <= 0.0f && ck <= 0.0f && hc <= 0.0f && pc <= 0.0f) return;
    const float* T = tgt + (long long)k * cy * cx;
    float target = (T[j0 * cx + i0] * (1.0f - tx) + T[j0 * cx + i1] * tx) * (1.0f - ty)
                 + (T[j1 * cx + i0] * (1.0f - tx) + T[j1 * cx + i1] * tx) * ty;
    if (ck > c) {
        int skip = 0;
        if (guard_mode == 1) skip = f[id] < target - guard;
        else if (guard_mode == 2) skip = f[id] < target * (1.0f - guard);
        if (!skip) c = ck;
    }
    if (hc > c && f[id] > target) c = hc;
    if (pc > c && f[id] < target && f[id] > target + plo - 0.5f) c = pc;
    if (c <= 0.0f) return;
    f[id] += c * (target - f[id]);
}
"""


def _kernel(name):
    import cupy as cp
    if not _KERNELS:
        mod = cp.RawModule(code=_SRC, options=("--std=c++14",))
        for n in ("nudge_w", "add_bubble", "relax_rim", "keep_alive", "relax_airmass", "ease_cap", "lift_root",
                  "relax_target3d"):
            _KERNELS[n] = mod.get_function(n)
    return _KERNELS[name]


def _grid(n, tpb=256):
    return ((int(n) + tpb - 1) // tpb,), (tpb,)


# ------------------------------------------------------------- storm starts
@dataclass
class StormStart:
    """One storm start, in world metres and run seconds."""
    x: float
    y: float
    t0: float
    kind: str = "nudge"            # "nudge" (Naylor & Gilmore), "bubble" or "vortex" (hurricane.apply_vortex)
    radius: float = 10000.0        # horizontal radius (m)
    zc: float = 1500.0             # centre height (m)
    depth: float = 1500.0          # vertical half depth (m)
    strength: float = 10.0         # nudging target w (m/s) or bubble amplitude (K)
    duration: float = 900.0        # nudging duration (s)
    vx: float = 0.0                # motion of the nudging region (m/s)
    vy: float = 0.0
    ramp: float = 300.0            # nudging grows over this many seconds
    tau: float = 10.0              # nudging time scale (s)
    id: int = 0
    applied: bool = False          # bubbles: already added
    why: str = ""                  # what started it ("warm sector", "cold front #1", "tower" = a satellite)
    waiting: bool = False          # no GPU memory for its patch yet (the engine tries again)
    retry_at: float = 0.0          # model time of the next try (0: never had to wait)
    rmw: float = 0.0               # vortex: radius of maximum wind (m); radius = its outer radius, strength = vmax
    # a nudge's shape (thermal clusters, sim._thermal_cluster): stretched aspect : 1 along angle (rad), a lumpy
    # outline (lobes 0..0.5), leaning tilt_x / tilt_y m per m up, 3-D noise (0..1) roughening it, with its seed
    aspect: float = 0.0            # 0: the classic round push (nudge_w), >= 1: a shaped one
    angle: float = 0.0
    lobes: float = 0.0
    lobe_phase: float = 0.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    noise: float = 0.0
    seed: int = 0

    @property
    def t1(self) -> float:
        return self.t0 + (self.duration if self.kind == "nudge" else 0.0)

    def position(self, t: float):
        dt = max(0.0, t - self.t0) if self.kind == "nudge" else 0.0
        return self.x + self.vx * dt, self.y + self.vy * dt

    def reach(self) -> float:
        """Horizontal half-size of the region it touches (m), with its drift."""
        drift = math.hypot(self.vx, self.vy) * (self.duration if self.kind == "nudge" else 0.0)
        return self.radius + drift

    def to_json(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y, "t0": self.t0, "kind": self.kind,
                "radius": self.radius, "strength": self.strength, "duration": self.duration,
                "vx": self.vx, "vy": self.vy, "why": self.why, "waiting": self.waiting, "rmw": self.rmw}


def apply_nudging(patch, starts: list[StormStart], t: float, dt: float) -> None:
    import cupy as cp
    act = [s for s in starts if s.kind == "nudge" and s.t0 <= t <= s.t1]
    if not act:
        return
    sh = lambda s, n, d: getattr(s, n, d)           # starts saved before the shapes have none
    ev = np.array([[s.x, s.y, s.vx, s.vy, s.t0, s.t1, s.radius, s.zc, s.depth, s.strength, s.ramp, s.tau,
                    sh(s, "aspect", 0.0), sh(s, "angle", 0.0), sh(s, "lobes", 0.0), sh(s, "lobe_phase", 0.0),
                    sh(s, "tilt_x", 0.0), sh(s, "tilt_y", 0.0), sh(s, "noise", 0.0), sh(s, "seed", 0)]
                   for s in act], dtype=np.float32)
    env = patch.env
    zw = _cached_device(env, "z_w", env.z_w)
    w = patch.state.w
    nz1, ny, nx = w.shape
    _kernel("nudge_w")(*_grid(w.size), (w, zw, cp.asarray(ev), np.int32(len(act)), np.int32(nz1),
                                        np.int32(ny), np.int32(nx), np.float32(patch.world_x0()),
                                        np.float32(patch.world_y0()), np.float32(env.grid.dx),
                                        np.float32(t), np.float32(dt)))


def apply_bubble(patch, s: StormStart) -> None:
    env = patch.env
    zh = _cached_device(env, "z_h", env.z_h)
    thp = patch.state.thp
    nz, ny, nx = thp.shape
    _kernel("add_bubble")(*_grid(thp.size), (thp, zh, np.int32(nz), np.int32(ny), np.int32(nx),
                                             np.float32(patch.world_x0()), np.float32(patch.world_y0()),
                                             np.float32(env.grid.dx), np.float32(s.x), np.float32(s.y),
                                             np.float32(s.zc), np.float32(s.radius),
                                             np.float32(s.depth), np.float32(s.strength)))


def env_targets(env):
    """(theta', qv) per half level the rim and air-mass relaxation aim at: the environment's, plus what
    large-scale ascent has done to it so far (largescale.py)."""
    qv0 = getattr(env, "qv_env_h", None)
    if qv0 is None:
        qv0 = env.qv_f(env.z_h)
    dth = getattr(env, "ls_dth", None)
    if dth is None or not np.any(dth):
        return _cached_device(env, "zero_h", np.zeros_like(env.z_h)), _cached_device(env, "qv_env", qv0)
    return (_cached_device(env, "ls_thp", dth),
            _cached_device(env, "ls_qv", np.maximum(np.asarray(qv0, dtype=np.float64) + env.ls_dqv, 1e-7)))


def _cached_device(env, key, host):
    import cupy as cp
    cache = env.__dict__.setdefault("_dev_cache", {})
    if key not in cache:
        cache[key] = cp.asarray(np.asarray(host, dtype=np.float32))
    return cache[key]


# ------------------------------------------------------- storm persistence
def updraft_cores(patch, z_lo=3500.0, z_hi=6500.0, w_on=8.0, w_full=20.0, sep_km=8.0, max_cores=64,
                  root_lo=800.0, root_hi=2500.0, root_min=1.0, any_root=False):
    """Roots of deep updrafts: local maxima of the 0.8-2.5 km max w that have a
    deep (3.5-6.5 km) updraft within ``sep_km``.  In shear the updraft tilts,
    so its root sits on the inflow side of the mid-level core; lifting there
    feeds the storm instead of its downdraft.  Returns an (n, 3) float32 host
    array (x, y world m, strength 0..1 from the deep updraft).  ``any_root``
    also returns roots without a deep updraft now (strength 0), for CoreMemory."""
    import cupy as cp
    from cupyx.scipy import ndimage

    st, env = patch.state, patch.env
    k0 = int(np.searchsorted(env.z_w, z_lo))
    k1 = max(int(np.searchsorted(env.z_w, z_hi)), k0 + 1)
    r0 = max(int(np.searchsorted(env.z_w, root_lo)), 1)
    r1 = max(int(np.searchsorted(env.z_w, root_hi)), r0 + 1)
    size = max(3, int(round(sep_km * 1000.0 / env.grid.dx)) | 1)
    wmid = ndimage.maximum_filter(st.w[k0:k1].max(axis=0), size=size)   # deep updraft nearby
    wlow = st.w[r0:r1].max(axis=0)
    peak = (wlow == ndimage.maximum_filter(wlow, size=size)) & (wlow > root_min)
    if not any_root:
        peak &= wmid > w_on
    jj, ii = cp.nonzero(peak)
    if jj.size == 0:
        return np.zeros((0, 3), np.float32)
    vals = wmid[jj, ii]
    order = cp.argsort(-(vals + 1.0e-3 * wlow[jj, ii]))[:max_cores]
    jj, ii, vals = cp.asnumpy(jj[order]), cp.asnumpy(ii[order]), cp.asnumpy(vals[order])
    dx = env.grid.dx
    x = patch.world_x0() + (ii + 0.5) * dx
    y = patch.world_y0() + (jj + 0.5) * dx
    s = np.clip((vals - w_on) / (w_full - w_on), 0.0, 1.0)
    return np.stack([x, y, s], axis=1).astype(np.float32)


class CoreMemory:
    """Deep updrafts remembered after they weaken, so storm persistence keeps
    helping a storm through a weak spell instead of fading with it.  A root found
    now continues the nearest remembered core within ``match_km`` and gets the
    larger of its own strength and that core's, decayed by exp(-age / tau).  Only
    roots that exist now are returned (a storm that is gone is not lifted again),
    and roots of no remembered core count only with a deep updraft of their own.
    Positions are kept relative to the grid, which moves with the storms."""

    def __init__(self, tau: float = 1800.0, match_km: float = 6.0, forget: float = 600.0):
        self.tau, self.match, self.forget = float(tau), match_km * 1000.0, float(forget)
        self.xy = np.zeros((0, 2))
        self.s = np.zeros(0)
        self.seen = np.zeros(0)
        self.t = None

    def update(self, found: dict, t: float, offset) -> dict:
        """found: {patch id: (n, 3) roots (world x, y, strength now)} ->
        the same with remembered strengths, weak roots of no storm dropped."""
        decay = math.exp(-max(t - self.t, 0.0) / self.tau) if self.t is not None else 1.0
        self.t = t
        s_old = self.s * decay
        used = np.zeros(len(s_old), bool)
        ox, oy = offset
        roots = sorted(((pid, r) for pid, arr in found.items() for r in arr), key=lambda pr: -float(pr[1][2]))
        out, xy, s_new = {pid: [] for pid in found}, [], []
        for pid, r in roots:
            g = np.array([float(r[0]) - ox, float(r[1]) - oy])
            s = float(r[2])
            if len(s_old):
                d = np.hypot(self.xy[:, 0] - g[0], self.xy[:, 1] - g[1])
                d[used] = np.inf
                m = int(np.argmin(d))
                if d[m] <= self.match:
                    used[m] = True
                    s = max(s, float(s_old[m]))
            if s <= 0.02:
                continue
            xy.append(g)
            s_new.append(s)
            out[pid].append((r[0], r[1], s))
        # remembered cores without a root now wait a while (the root may come back)
        keep = ~used & (t - self.seen <= self.forget) & (s_old > 0.02)
        self.xy = np.concatenate([np.asarray(xy, float).reshape(-1, 2), self.xy[keep]])
        self.s = np.concatenate([np.asarray(s_new, float), s_old[keep]])
        self.seen = np.concatenate([np.full(len(s_new), float(t)), self.seen[keep]])
        return {pid: np.asarray(v, np.float32).reshape(-1, 3) for pid, v in out.items()}


def root_surroundings(patch, cores: np.ndarray, z_lo: float = 500.0, z_hi: float = 2500.0,
                      half_km: float = 12.0, max_cores: int = 16):
    """For lift_root: the w levels of the root layer and, for each core, the mean
    theta' and qv at those levels over a box of +-half_km around it (its
    surroundings).  Returns (k0, nk, cores, surr) or None."""
    import cupy as cp
    if cores is None or len(cores) == 0:
        return None
    cores = cores[:max_cores]
    env, st = patch.env, patch.state
    k0 = max(int(np.searchsorted(env.z_w, z_lo)), 1)
    k1 = min(max(int(np.searchsorted(env.z_w, z_hi)), k0 + 1), env.grid.nz)
    nk = k1 - k0
    dx = env.grid.dx
    ny, nx = st.thp.shape[1:]
    h = max(2, int(round(half_km * 1000.0 / dx)))
    th = 0.5 * (st.thp[k0 - 1:k1 - 1] + st.thp[k0:k1])
    qv = 0.5 * (st.qv[k0 - 1:k1 - 1] + st.qv[k0:k1])
    parts = []
    for x, y, _ in cores:
        i = int((x - patch.world_x0()) / dx)
        j = int((y - patch.world_y0()) / dx)
        js, is_ = slice(max(j - h, 0), min(j + h + 1, ny)), slice(max(i - h, 0), min(i + h + 1, nx))
        parts.append(cp.stack([th[:, js, is_].mean(axis=(1, 2)), qv[:, js, is_].mean(axis=(1, 2))], axis=1))
    surr = cp.ascontiguousarray(cp.stack(parts).astype(cp.float32))          # (nc, nk, 2)
    return k0, nk, cp.asarray(np.ascontiguousarray(cores, np.float32)), surr


def apply_root_lift(patch, root, dt: float, gain: float, radius: float = 4000.0,
                    w_min: float = 0.5, a_max: float = 0.15) -> None:
    """Lift the cold, rising air of deep updrafts' roots by ``gain`` x its negative
    buoyancy (``root`` from root_surroundings)."""
    if root is None or gain <= 0:
        return
    k0, nk, cores, surr = root
    env, st = patch.env, patch.state
    th0 = _cached_device(env, "theta_w", env.theta_f(np.asarray(env.z_w)))
    ny, nx = st.thp.shape[1:]
    _kernel("lift_root")(*_grid(nk * ny * nx), (
        st.w, st.thp, st.qv, th0, cores, surr, np.int32(len(cores)), np.int32(k0), np.int32(nk),
        np.int32(ny), np.int32(nx), np.float32(patch.world_x0()), np.float32(patch.world_y0()),
        np.float32(env.grid.dx), np.float32(gain), np.float32(radius), np.float32(w_min),
        np.float32(a_max), np.float32(dt)))


def apply_keep_alive(patch, cores: np.ndarray, dt: float, amp: float, radius: float = 4000.0,
                     z_peak: float = 1200.0, z_half: float = 1100.0, tau: float = 60.0) -> None:
    """Lift the roots of existing deep updrafts (target ``amp`` m/s near
    ``z_peak``), never slowing a stronger updraft."""
    import cupy as cp
    if cores is None or len(cores) == 0 or amp <= 0:
        return
    env = patch.env
    zw = _cached_device(env, "z_w", env.z_w)
    w = patch.state.w
    nz1, ny, nx = w.shape
    _kernel("keep_alive")(*_grid(w.size), (
        w, zw, cp.asarray(cores), np.int32(len(cores)), np.int32(nz1), np.int32(ny), np.int32(nx),
        np.float32(patch.world_x0()), np.float32(patch.world_y0()), np.float32(env.grid.dx),
        np.float32(amp), np.float32(radius), np.float32(z_peak), np.float32(z_half),
        np.float32(min(dt / tau, 1.0))))


def _has_fronts(patch) -> bool:
    """The patch has air-mass maps of real boundaries (not only the warm sector's texture), or the
    synoptic mode's simulated atmosphere."""
    am = patch.env.airmass
    return getattr(patch, "am_maps", None) is not None and (bool(getattr(am, "boundaries", None))
                                                            or getattr(am, "synoptic", False)
                                                            or getattr(am, "pockets", None) is not None)


def _synoptic_level0(patch, maps):
    """The synoptic target theta' of the lowest level at every mass point (nearest coarse point)."""
    import cupy as cp
    s = int(round(maps["stride"]))
    t = maps["thp"][0]
    return cp.repeat(cp.repeat(t, s, axis=0), s, axis=1)[:patch.rect.ny, :patch.rect.nx]


def apply_ease_cap(patch, cores: np.ndarray, dt: float, strength: float, tau: float = 300.0,
                   r_in: float = 5000.0, r_out: float = 35000.0, z_lo: float = 600.0,
                   z_hi: float = 3000.0) -> None:
    """Around the storm's deep updrafts, relax the capping layer's warm,
    sinking, cloud-free anomalies toward the environment (time scale tau at
    strength 1).  Without air masses only: with them the cap layer's own
    target is not zero, so it is left alone."""
    import cupy as cp
    if cores is None or len(cores) == 0 or strength <= 0 or _has_fronts(patch):
        return
    env, st = patch.env, patch.state
    zh = _cached_device(env, "z_h", env.z_h)
    nz, ny, nx = st.thp.shape
    _kernel("ease_cap")(*_grid(st.thp.size), (
        st.thp, st.w, st.qc, zh, cp.asarray(cores), np.int32(len(cores)), np.int32(nz), np.int32(ny),
        np.int32(nx), np.float32(patch.world_x0()), np.float32(patch.world_y0()), np.float32(env.grid.dx),
        np.float32(r_in), np.float32(r_out), np.float32(z_lo), np.float32(z_hi),
        np.float32(min(dt / tau, 1.0) * strength)))


# ---------------------------------------------------------- relaxation rim
RIM_FIELDS_ENV = ("u", "v", "thp", "qv")
RIM_FIELDS_ZERO = ("w", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng")
AEROSOL_FIELDS = ("nwfa", "nifa")


def hydrometeor_fields(st) -> tuple:
    """Every cloud / precipitation field (mass, number, volume) of the state's
    microphysics scheme -- not vapour, aerosols or turbulence."""
    from gpuwm.core.moist import _extra_water_species
    names = ("qc", "qr") + tuple(_extra_water_species(st))
    return tuple(n for n in dict.fromkeys(names)
                 if n not in AEROSOL_FIELDS and getattr(st, n, None) is not None)


def precipitation(st, k1: int):
    """Precipitating condensate (kg/kg) in levels below k1, whatever the scheme."""
    q = None
    for n in ("qr", "qs", "qg", "qh"):
        a = getattr(st, n, None)
        if a is not None:
            q = a[:k1] if q is None else q + a[:k1]
    if getattr(st, "qir", None) is not None:          # P3: rimed / large ice lives in qi
        q = q + st.qi[:k1]
    return q


def apply_rim(patch, dt: float, width_cells: float, tau_edge: float,
              open_sides=(True, True, True, True)) -> None:
    """Relax toward the environment within ``width_cells`` of each patch edge
    (strongest, time scale ``tau_edge``, at the edge itself)."""
    import cupy as cp
    env, st = patch.env, patch.state
    coef = np.float32(min(dt / tau_edge, 1.0))
    ow, oe, os_, on = (np.int32(bool(v)) for v in open_sides)
    th_t, qv_t = env_targets(env)
    profiles = {
        "u": _cached_device(env, "u_env", env.u_env),
        "v": _cached_device(env, "v_env", env.v_env),
        "thp": th_t,
        "qv": qv_t,
    }
    nx, ny = patch.rect.nx, patch.rect.ny
    maps = getattr(patch, "am_maps", None)
    for name in RIM_FIELDS_ENV + ("w",) + hydrometeor_fields(st):
        f = getattr(st, name, None)
        if f is None:
            continue
        if maps is not None and name in RIM_FIELDS_ENV:
            continue                        # apply_airmass relaxes these toward the air masses
        nk, fy, fx = f.shape
        xs = np.int32(fx == nx + 1)
        ys = np.int32(fy == ny + 1)
        prof = profiles.get(name)
        if name == "qv" and prof is not None and prof.shape[0] != nk:
            prof = None
        _kernel("relax_rim")(*_grid(f.size), (f, prof if prof is not None else np.intp(0),
                                              np.int32(nk), np.int32(fy), np.int32(fx), xs, ys,
                                              np.float32(width_cells), coef, ow, oe, os_, on))


def shape_signed_distance(poly, nx: int, ny: int) -> np.ndarray:
    """Signed distance (cells; + inside) of every mass cell of an nx x ny box from the outline ``poly``
    [(fx, fy), ...] given as fractions of the box (host, (ny, nx))."""
    P = np.asarray(poly, dtype=np.float64) * np.array([nx, ny], dtype=np.float64)
    x = np.arange(nx, dtype=np.float64) + 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    X, Y = np.meshgrid(x, y)
    inside = np.zeros((ny, nx), bool)
    d2 = np.full((ny, nx), np.inf)
    A, B = P, np.roll(P, -1, axis=0)
    for (ax, ay), (bx, by) in zip(A, B):
        # even-odd crossing of a ray to +x, and the distance to the segment
        if by != ay:
            inside ^= ((ay > Y) != (by > Y)) & (X < (bx - ax) * (Y - ay) / (by - ay) + ax)
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey
        t = np.clip(((X - ax) * ex + (Y - ay) * ey) / L2, 0.0, 1.0) if L2 > 0 else np.zeros_like(X)
        d2 = np.minimum(d2, (X - ax - t * ex) ** 2 + (Y - ay - t * ey) ** 2)
    d = np.sqrt(d2)
    return np.where(inside, d, -d)


def shape_inner_rect(poly, n: int = 96) -> tuple:
    """The largest axis-aligned rectangle inside the outline ``poly`` ((fx, fy) fractions of its box), as
    fractions (fx0, fy0, fx1, fy1): where a shaped fixed domain keeps its storms (engine.Engine._shaped)."""
    inside = shape_signed_distance(poly, n, n) > 0.5
    best, h = (0, 0, 0, 0, 0), np.zeros(n, int)
    for j in range(n):                       # the largest rectangle under each row's histogram
        h = np.where(inside[j], h + 1, 0)
        stack = []
        for i in range(n + 1):
            cur = h[i] if i < n else 0
            start = i
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                if hh * (i - s) > best[0]:
                    best = (hh * (i - s), s, j - hh + 1, i, j + 1)
                start = s
            stack.append((start, cur))
    if best[0] == 0:
        return 0.0, 0.0, 1.0, 1.0
    _, i0, j0, i1, j1 = best
    return i0 / n, j0 / n, i1 / n, j1 / n


def apply_shape(patch, poly, dt: float, width_cells: float, tau_edge: float) -> None:
    """A fixed domain of another shape than its box (an ellipse, an outline drawn on the map): the cells outside
    the outline are relaxed toward the environment (+ air masses) like the edge rim -- at its full rate outside,
    easing to nothing ``width_cells`` inside the outline -- and their clouds, rain and vertical motion toward
    none, so storms live only inside it."""
    import cupy as cp
    env, st = patch.env, patch.state
    nx, ny = patch.rect.nx, patch.rect.ny
    key = (nx, ny, tuple(map(tuple, poly)), float(width_cells))
    cache = getattr(patch, "_shape_c", None)
    if cache is None or cache[0] != key:
        sd = shape_signed_distance(poly, nx, ny)
        a = np.clip(1.0 - sd / max(width_cells, 1.0), 0.0, 1.0)
        cache = patch._shape_c = (key, cp.asarray((a * a).astype(np.float32)), {})
    _, wmass, targets = cache
    coef = float(min(dt / tau_edge, 1.0))

    def stag(a, fy, fx):                     # a mass-cell map (..., ny, nx) onto a staggered field's shape
        if a.shape[-1] != fx:
            a = cp.concatenate([a[..., :1], 0.5 * (a[..., 1:] + a[..., :-1]), a[..., -1:]], axis=-1)
        if a.shape[-2] != fy:
            a = cp.concatenate([a[..., :1, :], 0.5 * (a[..., 1:, :] + a[..., :-1, :]), a[..., -1:, :]], axis=-2)
        return a

    tg = _shape_targets(patch, targets)
    for name in RIM_FIELDS_ENV + ("w",) + hydrometeor_fields(st):
        f = getattr(st, name, None)
        if f is None:
            continue
        nk, fy, fx = f.shape
        c = (coef * stag(wmass, fy, fx))[None].astype(f.dtype)
        t = tg.get(name)
        if t is None:
            f -= c * f                                          # w and condensate: toward none
            continue
        if t.ndim == 1:
            if t.shape[0] != nk:
                continue
            t = t[:, None, None]
        else:
            if t.shape[0] != nk:
                continue
            t = stag(t, fy, fx)
        f += c * (t.astype(f.dtype) - f)


def _shape_targets(patch, cache: dict) -> dict:
    """What the outside of a shaped domain is relaxed to: the 3-D rim targets (terrain, pockets, simulated
    weather), the air masses' columns (renewed with the patch's maps), or the environment's profiles."""
    import cupy as cp
    env = patch.env
    maps = getattr(patch, "am_maps", None)
    nx, ny = patch.rect.nx, patch.rect.ny
    if isinstance(maps, dict):
        if cache.get("maps") is not maps:
            s = int(round(maps["stride"]))
            cache.clear()
            cache["maps"] = maps
            cache["t"] = {n: cp.repeat(cp.repeat(maps[n], s, axis=-2), s, axis=-1)[..., :ny, :nx]
                          for n in RIM_FIELDS_ENV if maps.get(n) is not None}
        return cache["t"]
    th_t, qv_t = env_targets(env)
    prof = {"u": _cached_device(env, "u_env", env.u_env), "v": _cached_device(env, "v_env", env.v_env),
            "thp": th_t, "qv": qv_t}
    if maps is None or not env.airmass:
        return prof
    at = getattr(patch, "am_time", None)
    if cache.get("am_time") != at or cache.get("rect") != (patch.rect.i0, patch.rect.j0, nx, ny):
        cache.clear()
        cache["am_time"], cache["rect"] = at, (patch.rect.i0, patch.rect.j0, nx, ny)
        X, Y = env.ground_xy(patch.rect, t=env.clock)
        thp, qv, _, _ = env.airmass_columns(X, Y, env.clock, xp=cp)
        u = cp.asarray(env.u_env)[:, None, None] + env.airmass_wind(X, Y, env.clock, "u", xp=cp)
        v = cp.asarray(env.v_env)[:, None, None] + env.airmass_wind(X, Y, env.clock, "v", xp=cp)
        cache["t"] = {"thp": thp.astype(cp.float32), "qv": qv.astype(cp.float32),
                      "u": u.astype(cp.float32), "v": v.astype(cp.float32)}
    return cache["t"]


AIRMASS_KINDS = {"thp": 0, "qv": 1, "u": 2, "v": 3}
AIRMASS_HOLD_TAU = 120.0    # s: the frontal zone's one-sided hold (warm air blown across a kept boundary)
WIND_HOLD_TAU = 200.0       # s: the air masses' winds (backed winds ahead of boundaries) held at the large scale
WIND_HOLD_KM = 20.0         # ... smoothed over this box, so storms keep their own inflow and outflow


def apply_wind_hold(patch, dt: float) -> None:
    """Relax the box-smoothed low-level wind toward the air masses' winds (the backed, strengthened wind
    ahead of boundaries, kept air masses) where they are held: friction and mixing veer and slow the
    backing within ~20 min, the pointwise keep is too gentle to restore it, and a pointwise hold strong
    enough would also damp the storms' own flow."""
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter
    maps = getattr(patch, "am_maps", None)
    if maps is None:
        return
    env, st = patch.env, patch.state
    nx, ny = patch.rect.nx, patch.rect.ny
    zh = np.asarray(env.z_h)
    if isinstance(maps, dict):
        am = maps.get("am")
        if not am:
            return
        s = int(round(maps["stride"]))

        def up(a):
            return cp.repeat(cp.repeat(a, s, axis=-2), s, axis=-1)[..., :ny, :nx]
        w2, H = up(am["u"][0]), up(am["u"][1])
        tgt = {n: (lambda k, n=n: up(maps[n][:k])) for n in ("u", "v")}
    else:
        w2, H, du, dv = maps[8], maps[6], maps[4], maps[5]
        from .boundaries import prof
        zz = cp.asarray(zh, dtype=cp.float32)[:, None, None]
        env_w = {"u": cp.asarray(env.u_env, dtype=cp.float32), "v": cp.asarray(env.v_env, dtype=cp.float32)}
        tgt = {n: (lambda k, n=n, d=d: env_w[n][:k, None, None] + d[None] * prof(zz[:k], H[None]))
               for n, d in (("u", du), ("v", dv))}
    if float(w2.max()) <= 0.0:
        return
    kmax = max(1, min(int(np.searchsorted(zh, 1.15 * float(H.max()) + 1300.0)), len(zh)))
    zz = cp.asarray(zh[:kmax], dtype=cp.float32)[:, None, None]
    t = cp.clip((zz - 1.15 * H[None] - 300.0) / 1000.0, 0.0, 1.0)
    weight = (min(dt / WIND_HOLD_TAU, 1.0) * w2[None] * (1.0 - t * t * (3.0 - 2.0 * t))).astype(cp.float32)
    size = max(3, int(round(WIND_HOLD_KM * 1000.0 / env.grid.dx)) | 1)
    for n, ax in (("u", 2), ("v", 1)):
        f = getattr(st, n)
        lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[0] = hi[0] = slice(0, kmax); lo[ax] = slice(0, -1); hi[ax] = slice(1, None)
        fm = 0.5 * (f[tuple(lo)] + f[tuple(hi)])
        inc = weight * (tgt[n](kmax) - uniform_filter(fm, size=(1, size, size), mode="nearest"))
        mid = [slice(0, kmax), slice(None), slice(None)]
        mid[ax] = slice(1, -1)
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[ax] = slice(0, -1); b[ax] = slice(1, None)
        f[tuple(mid)] += (0.5 * (inc[tuple(a)] + inc[tuple(b)])).astype(f.dtype)
        e0 = [slice(0, kmax), slice(None), slice(None)]; e0[ax] = 0
        e1 = [slice(0, kmax), slice(None), slice(None)]; e1[ax] = -1
        i0 = [slice(None)] * 3; i0[ax] = 0
        i1 = [slice(None)] * 3; i1[ax] = -1
        f[tuple(e0)] += inc[tuple(i0)].astype(f.dtype)
        f[tuple(e1)] += inc[tuple(i1)].astype(f.dtype)


def apply_airmass(patch, dt: float, width_cells: float, tau_edge: float, tau_keep: float,
                  rim: bool = True) -> None:
    """Relax u, v, theta', qv toward the environment + air masses: in the edge
    rim (like apply_rim) and, with time scale ``tau_keep``, in the low levels
    deep inside air masses that are kept ("keep the air mass")."""
    maps = getattr(patch, "am_maps", None)
    if maps is None:
        return
    if isinstance(maps, dict):
        _apply_synoptic(patch, maps, dt, width_cells, tau_edge, tau_keep, rim)
        return
    env, st = patch.env, patch.state
    th_t, qv_t = env_targets(env)
    profiles = {
        "u": _cached_device(env, "u_env", env.u_env),
        "v": _cached_device(env, "v_env", env.v_env),
        "thp": th_t,
        "qv": qv_t,
    }
    zh = _cached_device(env, "z_h", env.z_h)
    nx, ny = patch.rect.nx, patch.rect.ny
    coef = np.float32(min(dt / tau_edge, 1.0))
    keep = np.float32(min(dt / tau_keep, 1.0) if tau_keep > 0 else 0.0)
    hold = np.float32(min(dt / AIRMASS_HOLD_TAU, 1.0) if maps.shape[0] > 11 else 0.0)
    for name, kind in AIRMASS_KINDS.items():
        f = getattr(st, name, None)
        if f is None:
            continue
        nk, fy, fx = f.shape
        _kernel("relax_airmass")(*_grid(f.size), (
            f, profiles[name], maps, zh, np.int32(kind), np.int32(nk), np.int32(ny), np.int32(nx),
            np.int32(fy), np.int32(fx), np.int32(fx == nx + 1), np.int32(fy == ny + 1),
            np.float32(width_cells), coef, keep, np.int32(bool(rim)), hold))


# synoptic keep, per field: (x the engine's keep rate, top (m), guard mode, guard).  theta' and qv in the
# lowest 1.5 / 1.2 km follow the simulated day (heating, moistening) over ~2 h, except in cold pools and
# storm outflow; the winds only at the edges -- inside, the storms' own flow is left alone.
SYNOPTIC_KEEP = {"thp": (0.15, 1500.0, 1, 1.5), "qv": (0.10, 1200.0, 2, 0.15), "u": (0.0, 0.0, 0, 0.0),
                 "v": (0.0, 0.0, 0, 0.0)}


def _apply_synoptic(patch, maps: dict, dt: float, width_cells: float, tau_edge: float, tau_keep: float,
                    rim: bool) -> None:
    env, st = patch.env, patch.state
    zh = _cached_device(env, "z_h", env.z_h)
    nx, ny = patch.rect.nx, patch.rect.ny
    coef = np.float32(min(dt / tau_edge, 1.0))
    keep = min(dt / tau_keep, 1.0) if tau_keep > 0 and maps.get("keep", True) else 0.0
    hold = min(dt / AIRMASS_HOLD_TAU, 1.0)
    am = maps.get("am")                     # terrain patches: the boundaries' own keep (weight, depth, hold)
    none = np.intp(0)
    for name, (kc, top, gm, g) in SYNOPTIC_KEEP.items():
        f = getattr(st, name, None)
        tgt = maps.get(name)
        if f is None or tgt is None or tgt.shape[0] != f.shape[0]:
            continue
        nk, fy, fx = f.shape
        aw, ah, ahold, apin, aplo = am[name] if am else (None,) * 5
        _kernel("relax_target3d")(*_grid(f.size), (
            f, tgt, zh, np.int32(nk), np.int32(fy), np.int32(fx), np.int32(nx), np.int32(ny),
            np.int32(maps["cy"]), np.int32(maps["cx"]), np.float32(maps["stride"]), np.float32(width_cells),
            coef, np.float32(keep * kc), np.float32(top), np.int32(bool(rim)), np.int32(gm), np.float32(g),
            aw if aw is not None else none, ah if ah is not None else none,
            np.float32(keep if aw is not None else 0.0),
            ahold if ahold is not None else none, np.float32(hold if ahold is not None else 0.0),
            apin if apin is not None else none, aplo if aplo is not None else none))


# ------------------------------------------------------------- activity
def activity_map(patch, w_thresh=3.0, q_thresh=1.0e-4, cold_thresh=-1.5, z_top=9000.0):
    """2-D bool map (device) of columns that belong to a storm: precipitation
    below ~9 km, convective up/downdrafts below ~9 km, or a surface cold pool.
    Gravity waves (weak w, mostly aloft) and the anvil's thin ice do not
    count, so a patch is sized to its storms, not to the ripples around
    them.  Where air masses are present, their own cold air is not a cold
    pool, and the shallow lift along a front is not a storm: only w above
    3 km and heavier precipitation count."""
    import cupy as cp
    st, env = patch.state, patch.env
    maps = getattr(patch, "am_maps", None) if _has_fronts(patch) else None
    kq = max(int(np.searchsorted(env.z_h, z_top)), 2)
    kw = max(int(np.searchsorted(env.z_w, z_top)), 2)
    kw0 = 1
    if maps is not None:
        kw0 = max(int(np.searchsorted(env.z_w, 3000.0)), 1)
        q_thresh = max(q_thresh, 3.0e-4)
    act = cp.abs(st.w[kw0:kw]).max(axis=0) > w_thresh
    act |= precipitation(st, kq).max(axis=0) > q_thresh
    if isinstance(maps, dict):
        # the simulated atmosphere's own cold air (behind a front, a night's inversion) is not a cold pool
        act |= (st.thp[0] - _synoptic_level0(patch, maps)) < cold_thresh
    elif maps is not None:
        z0 = float(env.z_h[0])
        H = cp.maximum(maps[1], 50.0)
        t = cp.clip((z0 - 0.7 * H) / (0.45 * H), 0.0, 1.0)
        target = maps[0] * (1.0 - t * t * (3.0 - 2.0 * t))
        act |= (st.thp[0] - target) < cold_thresh
    else:
        act |= st.thp[0] < cold_thresh
    return act


def column_max(a, axis=0):
    return a.max(axis=axis)
