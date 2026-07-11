import { describe, expect, it } from "vitest";
import {
  calibrateRaw,
  checkButterfly,
  checkCalendar,
  d2TotalVariance,
  dTotalVariance,
  impliedVol,
  naturalToRaw,
  rawToJumpWings,
  rawToNatural,
  totalVariance,
} from "../src/index.js";

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
});

describe("calendar arbitrage", () => {
  it("passes when the far slice dominates the near slice", () => {
    const near = { a: 0.02, b: 0.1, rho: -0.3, m: 0, sigma: 0.2 };
    const far = { a: 0.05, b: 0.12, rho: -0.3, m: 0, sigma: 0.2 };
    expect(checkCalendar(near, far).arbitrageFree).toBe(true);
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
});
