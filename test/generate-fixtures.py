#!/usr/bin/env python3
"""Independent reference fixtures for svi-vol-surface.

- density_g:   Durrleman's g(k) recovered from Black call prices,
               g = K·sqrt(2πw)·exp(d₋²/2)·∂²C/∂K² (F = 1, undiscounted), with the
               second strike derivative taken numerically by mpmath at 40 digits.
               This checks the closed-form g against the risk-neutral density it
               is supposed to sign.
- butterfly:   verdicts for random raw slices from numpy: w ≥ 0 (closed form),
               wing slope b(1+|ρ|) ≤ 2 (Lee), and min g ≥ -1e-10 on a 200k-point
               grid k = m + σ·sinh(u) reaching |k − m| = 1e4.
- calendar:    verdicts for random slice pairs: far wing slopes ≥ near ones, and
               min(w_far − w_near) ≥ -1e-10 on a 1.4M-point grid to |k| = 1e5.
- calibration: noisy smiles fitted by scipy.optimize.least_squares from 162
               starting points; the best SSE found is the target.

Usage: python -I test/generate-fixtures.py   (needs numpy, scipy, mpmath)
Writes test/fixtures.json.
"""
import json
import random

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import least_squares

out = {}

# --- density_g -------------------------------------------------------------
mp.mp.dps = 40
random.seed(11)


def w_mp(p, k):
    a, b, r, m, s = p
    return a + b * (r * (k - m) + mp.sqrt((k - m) ** 2 + s * s))


def call(p, K):
    k = mp.log(K)
    sw = mp.sqrt(w_mp(p, k))
    d1 = -k / sw + sw / 2
    return mp.ncdf(d1) - K * mp.ncdf(d1 - sw)


rows = []
while len(rows) < 80:
    p = [mp.mpf(x) for x in (random.uniform(-0.05, 0.1), random.uniform(0.01, 1.2), random.uniform(-0.95, 0.95),
                             random.uniform(-0.4, 0.4), random.uniform(0.02, 0.6))]
    if p[0] + p[1] * p[4] * mp.sqrt(1 - p[2] ** 2) <= 0.002:
        continue
    k = mp.mpf(random.uniform(-1.5, 1.5))
    K = mp.e**k
    w = w_mp(p, k)
    dm = -k / mp.sqrt(w) - mp.sqrt(w) / 2
    g = K * mp.sqrt(2 * mp.pi * w) * mp.exp(dm * dm / 2) * mp.diff(lambda x: call(p, x), K, 2)
    rows.append(dict(a=float(p[0]), b=float(p[1]), rho=float(p[2]), m=float(p[3]), sigma=float(p[4]),
                     k=float(k), g=float(g)))
out["density_g"] = rows

# --- butterfly -------------------------------------------------------------
rng = np.random.default_rng(7)
rows = []
for _ in range(600):
    a, b, r, m = rng.uniform(-0.1, 0.2), rng.uniform(0, 3), rng.uniform(-0.99, 0.99), rng.uniform(-0.5, 0.5)
    s = 10 ** rng.uniform(-2.5, 0)
    U = np.arcsinh(1e4 / s)
    k = m + s * np.sinh(np.linspace(-U, U, 200001))
    x = k - m
    R = np.sqrt(x * x + s * s)
    w = a + b * (r * x + R)
    wp = b * (r + x / R)
    wpp = b * s * s / R**3
    with np.errstate(all="ignore"):
        g = (1 - k * wp / (2 * w)) ** 2 - wp * wp / 4 * (1 / w + 0.25) + wpp / 2
    gmin = float(np.min(np.where(w > 0, g, np.inf)))
    minw = a + b * s * np.sqrt(1 - r * r)
    rows.append(dict(a=a, b=b, rho=r, m=m, sigma=s, minTotalVariance=float(minw),
                     arbitrageFree=bool(minw >= 0 and b * (1 + abs(r)) <= 2 and gmin >= -1e-10)))
out["butterfly"] = rows

# --- calendar --------------------------------------------------------------
rng = np.random.default_rng(9)
kk = np.concatenate([np.linspace(-5, 5, 1000001),
                     np.sign(np.linspace(-1, 1, 400001)) * 10 ** np.abs(np.linspace(-5, 5, 400001))])


def w_np(p, k):
    return p["a"] + p["b"] * (p["rho"] * (k - p["m"]) + np.sqrt((k - p["m"]) ** 2 + p["sigma"] ** 2))


rows = []
for _ in range(400):
    near = dict(a=rng.uniform(0, 0.05), b=rng.uniform(0.02, 0.4), rho=rng.uniform(-0.9, 0.9),
                m=rng.uniform(-0.3, 0.3), sigma=rng.uniform(0.01, 0.5))
    far = dict(a=near["a"] + rng.uniform(-0.01, 0.04), b=near["b"] * rng.uniform(0.8, 1.4),
               rho=float(np.clip(near["rho"] + rng.uniform(-0.2, 0.2), -0.95, 0.95)),
               m=near["m"] + rng.uniform(-0.2, 0.2), sigma=near["sigma"] * rng.uniform(0.5, 2))
    slopes = (far["b"] * (1 + far["rho"]) >= near["b"] * (1 + near["rho"]) - 1e-12
              and far["b"] * (1 - far["rho"]) >= near["b"] * (1 - near["rho"]) - 1e-12)
    gap = w_np(far, kk) - w_np(near, kk)
    rows.append(dict(near=near, far=far, arbitrageFree=bool(slopes and gap.min() >= -1e-10)))
out["calendar"] = rows

# --- calibration -----------------------------------------------------------
rng = np.random.default_rng(42)
rows = []
for _ in range(12):
    T = rng.uniform(0.05, 3)
    tr = [0.005 + 0.05 * rng.random() * T, (0.02 + 0.4 * rng.random()) * np.sqrt(T), rng.uniform(-0.9, 0.7),
          rng.uniform(-0.3, 0.2), rng.uniform(0.03, 0.53)]
    ks = np.linspace(-1.2, 0.8, 15)
    wq = (tr[0] + tr[1] * (tr[2] * (ks - tr[3]) + np.sqrt((ks - tr[3]) ** 2 + tr[4] ** 2))) * (
        1 + 0.02 * rng.standard_normal(len(ks)))

    def res(p):
        a, b, r, m, s = p
        return a + b * (r * (ks - m) + np.sqrt((ks - m) ** 2 + s * s)) - wq

    best = None
    for a0 in [0.0, 0.5 * wq.min()]:
        for m0 in [-0.6, -0.1, 0.4]:
            for s0 in [0.05, 0.2, 0.6]:
                for r0 in [-0.7, 0.0, 0.5]:
                    sol = least_squares(res, [a0, 0.1, r0, m0, s0],
                                        bounds=([-1, 0, -0.999999999, -3, 1e-6], [1, 10, 0.999999999, 3, 5]),
                                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
                    sse = float(np.sum(sol.fun**2))
                    if best is None or sse < best:
                        best = sse
    rows.append(dict(k=[float(x) for x in ks], totalVariance=[float(x) for x in wq], sse=best))
out["calibration"] = rows

out["_meta"] = {"generator": "test/generate-fixtures.py", "numpy": np.__version__, "scipy": scipy.__version__,
                "mpmath": mp.__version__}
with open("test/fixtures.json", "w") as f:
    json.dump(out, f)
print({k: len(v) for k, v in out.items() if k != "_meta"},
      "butterfly free:", sum(r["arbitrageFree"] for r in out["butterfly"]),
      "calendar free:", sum(r["arbitrageFree"] for r in out["calendar"]))
