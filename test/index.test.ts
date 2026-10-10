import { describe, expect, it } from "vitest";
import {
  calibrateRaw,
  checkButterfly,
  checkCalendar,
  d2TotalVariance,
  dTotalVariance,
  durrlemanG,
  impliedVol,
  jumpWingsToRaw,
  minTotalVariance,
  naturalToRaw,
  rawToJumpWings,
  rawToNatural,
  totalVariance,
} from "../src/index.js";
import fixtures from "./fixtures.json";

// A well-behaved arbitrage-free slice.
const CLEAN = { a: 0.04, b: 0.1, rho: -0.3, m: 0, sigma: 0.2 };
// Axel Vogt parameters (Gatheral & Jacquier 2014) — a KNOWN butterfly-arbitrage slice.
const VOGT = { a: -0.041, b: 0.1331, rho: 0.306, m: 0.3586, sigma: 0.4153 };

describe("total variance and vol", () => {
  it("w(0) with m=0 is a + b·σ", () => {
    expect(totalVariance(CLEAN, 0)).toBeCloseTo(0.04 + 0.1 * 0.2, 12);
  });

  it("implied vol is sqrt(w/t)", () => {
    const t = 0.5;
    expect(impliedVol(CLEAN, 0.1, t)).toBeCloseTo(Math.sqrt(totalVariance(CLEAN, 0.1) / t), 12);
  });

  it("total variance is positive across the range for a clean slice", () => {
    for (let k = -2; k <= 2; k += 0.1) expect(totalVariance(CLEAN, k)).toBeGreaterThan(0);
  });
});

describe("analytic derivatives match finite differences", () => {
  const h = 1e-6;
  it("w'(k) matches a central difference", () => {
    for (const k of [-1, -0.3, 0, 0.5, 1.2]) {
      const fd = (totalVariance(CLEAN, k + h) - totalVariance(CLEAN, k - h)) / (2 * h);
      expect(dTotalVariance(CLEAN, k)).toBeCloseTo(fd, 6);
    }
  });

  it("w''(k) matches a second central difference", () => {
    for (const k of [-1, -0.3, 0, 0.5, 1.2]) {
      const fd =
        (totalVariance(CLEAN, k + h) - 2 * totalVariance(CLEAN, k) + totalVariance(CLEAN, k - h)) /
        (h * h);
      expect(d2TotalVariance(CLEAN, k)).toBeCloseTo(fd, 3);
    }
  });
});

describe("butterfly (static) arbitrage", () => {
  it("passes a clean slice", () => {
    const check = checkButterfly(CLEAN);
    expect(check.arbitrageFree).toBe(true);
    expect(check.minG).toBeGreaterThanOrEqual(-1e-10);
    expect(check.wingConditionOk).toBe(true);
  });

  it("flags the Vogt slice (published butterfly-arbitrage example)", () => {
    const check = checkButterfly(VOGT);
    expect(check.arbitrageFree).toBe(false);
    expect(check.minG).toBeLessThan(0); // g(k) < 0 somewhere ⇒ arbitrage
  });

  it(`g(k) matches the density implied by Black call prices (${fixtures.density_g.length} mpmath cases)`, () => {
    for (const f of fixtures.density_g) {
      expect(Math.abs(durrlemanG(f, f.k) - f.g), JSON.stringify(f)).toBeLessThan(1e-12 * Math.max(1, Math.abs(f.g)));
    }
  });

  it(`agrees with an independent dense-grid verdict on ${fixtures.butterfly.length} random slices`, () => {
    for (const f of fixtures.butterfly) {
      const check = checkButterfly(f);
      expect(check.arbitrageFree, JSON.stringify(f)).toBe(f.arbitrageFree);
      expect(check.minTotalVariance).toBeCloseTo(f.minTotalVariance, 14);
    }
  });

  it("rejects negative total variance (regression: g looked positive there)", () => {
    const p = { a: -0.05, b: 0.01, rho: 0, m: 0, sigma: 0.5 };
    expect(minTotalVariance(p)).toBeCloseTo(-0.045, 14);
    const check = checkButterfly(p);
    expect(check.arbitrageFree).toBe(false);
    expect(check.minTotalVariance).toBeLessThan(0);
  });

  it("enforces Lee's wing bound b(1+|ρ|) ≤ 2 (regression: g < 0 beyond the old ±2 scan)", () => {
    const p = { a: 0.16291528291204724, b: 1.562132037040839, rho: 0.3728075399635393, m: -0.46650294585456276, sigma: 0.7979912178907502 };
    const check = checkButterfly(p);
    expect(check.wingConditionOk).toBe(false); // slope 2.14
    expect(check.arbitrageFree).toBe(false);
    expect(check.minG).toBeLessThan(0);
    expect(Math.abs(check.minK - p.m)).toBeGreaterThan(2);
  });

  it("resolves narrow dips the old 1000-point grid stepped over", () => {
    // σ = 0.003: the whole curvature sits in a window narrower than the old 0.004 grid step
    const p = { a: 0.01, b: 0.4, rho: -0.2, m: 0.0013, sigma: 0.003 };
    const check = checkButterfly(p);
    const dense = Array.from({ length: 200001 }, (_, i) => p.m - 0.05 + i * 5e-7).reduce(
      (min, k) => Math.min(min, durrlemanG(p, k)),
      Infinity,
    );
    expect(check.minG).toBeLessThanOrEqual(dense + 1e-12);
    expect(check.arbitrageFree).toBe(dense >= -1e-10);
  });
});

describe("calendar arbitrage", () => {
  it("passes when the far slice dominates the near slice", () => {
    const near = { a: 0.02, b: 0.1, rho: -0.3, m: 0, sigma: 0.2 };
    const far = { a: 0.05, b: 0.12, rho: -0.3, m: 0, sigma: 0.2 };
    expect(checkCalendar(near, far).arbitrageFree).toBe(true);
  });

  it(`agrees with an independent dense-grid verdict on ${fixtures.calendar.length} random pairs`, () => {
    for (const f of fixtures.calendar) {
      expect(checkCalendar(f.near, f.far).arbitrageFree, JSON.stringify(f)).toBe(f.arbitrageFree);
    }
  });

  it("flags slices that cross outside [−2, 2] (regression)", () => {
    // The far slice is above near on [−2, 2] (old verdict: free, minGap 0.0167)
    // but has a shallower left wing, so it falls below near far out.
    const near = { a: 0.024247216348864405, b: 0.04475859500064086, rho: -0.8898731087770928, m: 0.1983728621827542, sigma: 0.491818099700226 };
    const far = { a: 0.05347770898249839, b: 0.04428245727374849, rho: -0.8077396269413113, m: 0.11804529146205289, sigma: 0.7923793166557772 };
    for (let k = -2; k <= 2; k += 0.01) expect(totalVariance(far, k)).toBeGreaterThan(totalVariance(near, k));
    expect(totalVariance(far, -50)).toBeLessThan(totalVariance(near, -50));
    const check = checkCalendar(near, far);
    expect(check.arbitrageFree).toBe(false);
    expect(check.minGap).toBeLessThan(0);
  });

  it("flags a crossing (far dips below near)", () => {
    const near = { a: 0.05, b: 0.1, rho: -0.3, m: 0, sigma: 0.2 };
    const far = { a: 0.04, b: 0.1, rho: -0.3, m: 0, sigma: 0.2 };
    const check = checkCalendar(near, far);
    expect(check.arbitrageFree).toBe(false);
    expect(check.minGap).toBeLessThan(0);
  });
});

describe("parametrization conversions", () => {
  it("raw → natural → raw round-trips", () => {
    const nat = rawToNatural(CLEAN);
    const back = naturalToRaw(nat);
    expect(back.a).toBeCloseTo(CLEAN.a, 12);
    expect(back.b).toBeCloseTo(CLEAN.b, 12);
    expect(back.rho).toBeCloseTo(CLEAN.rho, 12);
    expect(back.m).toBeCloseTo(CLEAN.m, 12);
    expect(back.sigma).toBeCloseTo(CLEAN.sigma, 12);
  });

  it("jump-wings → raw inverts raw → jump-wings", () => {
    const slices = [CLEAN, VOGT, { a: 0.01, b: 0.3, rho: 0.5, m: -0.4, sigma: 0.05 }, { a: 0.02, b: 0.2, rho: -0.7, m: 0.3, sigma: 0.4 }];
    for (const p of slices) {
      for (const t of [0.1, 1, 3]) {
        const back = jumpWingsToRaw(rawToJumpWings(p, t), t);
        for (const key of ["a", "b", "rho", "m", "sigma"] as const) {
          expect(back[key], `${key} t=${t}`).toBeCloseTo(p[key], 11);
        }
      }
    }
  });

  it("jump-wings → raw rejects parameters no slice can have", () => {
    const jw = rawToJumpWings(CLEAN, 1);
    expect(() => jumpWingsToRaw({ ...jw, vt: jw.v + 0.01 }, 1)).toThrow(RangeError); // min above ATM
    expect(() => jumpWingsToRaw({ ...jw, psi: 10 }, 1)).toThrow(RangeError); // |β| ≥ 1
    expect(() => jumpWingsToRaw({ ...jw, p: -1 }, 1)).toThrow(RangeError);
    expect(() => jumpWingsToRaw(jw, 0)).toThrow(RangeError);
  });

  it("jump-wings ATM variance equals w(0)/t", () => {
    const t = 1.5;
    const jw = rawToJumpWings(CLEAN, t);
    expect(jw.v).toBeCloseTo(totalVariance(CLEAN, 0) / t, 12);
    expect(jw.p).toBeGreaterThan(0);
    expect(jw.c).toBeGreaterThan(0);
  });
});

describe("calibration", () => {
  it("recovers raw SVI parameters from a synthetic smile", () => {
    const truth = { a: 0.03, b: 0.15, rho: -0.4, m: 0.05, sigma: 0.18 };
    const ks = [-0.8, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.8];
    const quotes = ks.map((k) => ({ k, totalVariance: totalVariance(truth, k) }));
    const fit = calibrateRaw(quotes);
    expect(fit.rmse).toBeLessThan(1e-5);
    for (const q of quotes) {
      expect(totalVariance(fit, q.k)).toBeCloseTo(q.totalVariance, 5);
    }
  });

  it(`reaches scipy's multi-start least-squares optimum on ${fixtures.calibration.length} noisy smiles`, () => {
    for (const f of fixtures.calibration) {
      const fit = calibrateRaw(f.k.map((k, i) => ({ k, totalVariance: f.totalVariance[i]! })));
      const sse = fit.rmse ** 2 * f.k.length;
      expect(sse / f.sse).toBeLessThan(1 + 1e-6);
    }
  });

  it("fits exact SVI smiles, including narrow and steep ones the old optimizer missed", () => {
    // Regression: the old 5-D Nelder–Mead stopped at rmse 4e-4 on this smile, and
    // threw a RangeError (ρ rounded to −1) on others.
    const truths = [
      { a: 0.009446666512785017, b: 0.09103860242658654, rho: 0.13760154668130042, m: -0.12212653980689428, sigma: 0.049245465527868584 },
      { a: 0.02609893456457726, b: 0.2953480779677846, rho: 0.42270936580500995, m: -0.18240278599895665, sigma: 0.486375715535309 },
      { a: 0.002, b: 0.3, rho: -0.98, m: 0.1, sigma: 0.02 },
    ];
    const ks = Array.from({ length: 21 }, (_, i) => -1.2 + 0.1 * i);
    for (const truth of truths) {
      const fit = calibrateRaw(ks.map((k) => ({ k, totalVariance: totalVariance(truth, k) })));
      expect(fit.rmse).toBeLessThan(1e-8);
      expect(fit.rho).toBeCloseTo(truth.rho, 5);
      expect(fit.m).toBeCloseTo(truth.m, 5);
      expect(fit.sigma).toBeCloseTo(truth.sigma, 5);
    }
  });

  it("requires at least five quotes", () => {
    expect(() =>
      calibrateRaw([
        { k: 0, totalVariance: 0.04 },
        { k: 0.1, totalVariance: 0.05 },
      ]),
    ).toThrow(RangeError);
  });
});

describe("validation", () => {
  it("rejects invalid raw parameters", () => {
    expect(() => totalVariance({ ...CLEAN, b: -1 }, 0)).toThrow(RangeError);
    expect(() => totalVariance({ ...CLEAN, rho: 1 }, 0)).toThrow(RangeError);
    expect(() => totalVariance({ ...CLEAN, sigma: 0 }, 0)).toThrow(RangeError);
    expect(() => impliedVol(CLEAN, 0, 0)).toThrow(RangeError);
  });

  it("rejects NaN / non-finite parameters in every function", () => {
    const bad = [{ ...CLEAN, rho: Number.NaN }, { ...CLEAN, a: Number.NaN }, { ...CLEAN, m: Infinity }, { ...CLEAN, b: Infinity }];
    for (const p of bad) {
      expect(() => totalVariance(p, 0)).toThrow(RangeError);
      expect(() => dTotalVariance(p, 0)).toThrow(RangeError);
      expect(() => d2TotalVariance(p, 0)).toThrow(RangeError);
      expect(() => durrlemanG(p, 0)).toThrow(RangeError);
      expect(() => checkButterfly(p)).toThrow(RangeError);
    }
    expect(() => naturalToRaw({ delta: 0, mu: 0, rho: 1.5, omega: 0.1, zeta: 1 })).toThrow(RangeError);
    expect(() =>
      calibrateRaw([0, 1, 2, 3, 4].map((i) => ({ k: i / 10, totalVariance: i === 2 ? Number.NaN : 0.04 }))),
    ).toThrow(RangeError);
  });
});
