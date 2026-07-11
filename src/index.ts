/**
 * svi-vol-surface — Gatheral's SVI volatility-surface parametrization.
 *
 * The raw, natural, and jump-wings SVI forms and conversions between them;
 * total implied variance and Black implied vol from a slice; the Durrleman
 * butterfly-arbitrage function g(k); a static (butterfly) arbitrage check; a
 * calendar-arbitrage check between two slices; and least-squares calibration
 * of a raw SVI slice to a market smile.
 *
 * Convention: k is log-moneyness ln(K/F); a slice returns TOTAL implied
 * variance w(k) = σ_BS(k)² · T. Implied vol is sqrt(w/T).
 *
 * References: Gatheral (2004) "A parsimonious arbitrage-free implied volatility
 * parameterization"; Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility
 * surfaces".
 */

/** Raw SVI parameters. w(k) = a + b·( ρ(k−m) + √((k−m)² + σ²) ). */
export interface RawSVI {
  a: number;
  b: number;
  rho: number;
  m: number;
  sigma: number;
}

/** Natural SVI parameters. */
export interface NaturalSVI {
  delta: number;
  mu: number;
  rho: number;
  omega: number;
  zeta: number;
}

/** Jump-wings SVI parameters, tied to a maturity t. */
export interface JumpWingsSVI {
  /** ATM total variance v·t (here stored as the variance v). */
  v: number;
  /** ATM skew (psi). */
  psi: number;
  /** Left (put) wing slope. */
  p: number;
  /** Right (call) wing slope. */
  c: number;
  /** Minimum implied variance. */
  vt: number;
}

function validateRaw(p: RawSVI): void {
  if (!(p.b >= 0)) throw new RangeError(`b must be ≥ 0, got ${p.b}`);
  if (p.rho <= -1 || p.rho >= 1) throw new RangeError(`rho must be in (-1,1), got ${p.rho}`);
  if (!(p.sigma > 0)) throw new RangeError(`sigma must be > 0, got ${p.sigma}`);
}

/**
 * Total implied variance w(k) for a raw SVI slice at log-moneyness k.
 *
 * ```ts
 * totalVariance({ a: 0.04, b: 0.4, rho: -0.4, m: 0, sigma: 0.1 }, 0); // ≈ 0.056
 * ```
 */
export function totalVariance(p: RawSVI, k: number): number {
  validateRaw(p);
  const km = k - p.m;
  const w = p.a + p.b * (p.rho * km + Math.sqrt(km * km + p.sigma * p.sigma));
  return w;
}

/** Black implied volatility from a raw slice: sqrt(w(k) / t). */
export function impliedVol(p: RawSVI, k: number, t: number): number {
  if (!(t > 0)) throw new RangeError(`t must be > 0, got ${t}`);
  const w = totalVariance(p, k);
  if (w < 0) throw new RangeError(`negative total variance at k=${k} (a slice with arbitrage)`);
  return Math.sqrt(w / t);
}

/** First derivative w'(k) of total variance. */
export function dTotalVariance(p: RawSVI, k: number): number {
  const km = k - p.m;
  return p.b * (p.rho + km / Math.sqrt(km * km + p.sigma * p.sigma));
}

/** Second derivative w''(k) of total variance. */
export function d2TotalVariance(p: RawSVI, k: number): number {
  const km = k - p.m;
  const s2 = p.sigma * p.sigma;
  const root = Math.sqrt(km * km + s2);
  return (p.b * s2) / (root * root * root);
}

/**
 * Durrleman's g function. A slice is free of butterfly (static) arbitrage iff
 * g(k) ≥ 0 for all k (and the wings satisfy b(1+|ρ|) ≤ 4).
 */
export function durrlemanG(p: RawSVI, k: number): number {
  const w = totalVariance(p, k);
  const wp = dTotalVariance(p, k);
  const wpp = d2TotalVariance(p, k);
  const a = 1 - (k * wp) / (2 * w);
  return a * a - (wp * wp / 4) * (1 / w + 0.25) + wpp / 2;
}

export interface ButterflyCheck {
  arbitrageFree: boolean;
  /** Minimum of g(k) over the scanned grid (≥ 0 ⇒ butterfly-free). */
  minG: number;
  /** k at which the minimum occurred. */
  minK: number;
  /** Whether the wing condition b(1+|ρ|) ≤ 4 holds. */
  wingConditionOk: boolean;
}

/**
 * Scan for butterfly (static) arbitrage: g(k) ≥ 0 on a grid of log-moneyness
 * plus the necessary wing condition. `range` and `steps` control the scan.
 */
export function checkButterfly(p: RawSVI, range = 2, steps = 1000): ButterflyCheck {
  validateRaw(p);
  let minG = Infinity;
  let minK = 0;
  for (let i = 0; i <= steps; i++) {
    const k = p.m - range + (2 * range * i) / steps;
    const g = durrlemanG(p, k);
    if (g < minG) {
      minG = g;
      minK = k;
    }
  }
  const wingConditionOk = p.b * (1 + Math.abs(p.rho)) <= 4;
  return { arbitrageFree: minG >= -1e-10 && wingConditionOk, minG, minK, wingConditionOk };
}

/**
 * Calendar-arbitrage check between two slices at increasing maturities. A
 * surface is calendar-arbitrage-free when total variance is non-decreasing in
 * maturity at every log-moneyness: w₂(k) ≥ w₁(k) for t₂ > t₁.
 */
export interface CalendarCheck {
  arbitrageFree: boolean;
  /** Minimum of w₂(k) − w₁(k) over the grid (≥ 0 ⇒ calendar-free). */
  minGap: number;
  minK: number;
}

export function checkCalendar(
  near: RawSVI,
  far: RawSVI,
  range = 2,
  steps = 1000,
): CalendarCheck {
  validateRaw(near);
  validateRaw(far);
  let minGap = Infinity;
  let minK = 0;
  for (let i = 0; i <= steps; i++) {
    const k = -range + (2 * range * i) / steps;
    const gap = totalVariance(far, k) - totalVariance(near, k);
    if (gap < minGap) {
      minGap = gap;
      minK = k;
    }
  }
  return { arbitrageFree: minGap >= -1e-10, minGap, minK };
}

// ---------------------------------------------------------------------------
// Parametrization conversions (Gatheral & Jacquier 2014, §3)
// ---------------------------------------------------------------------------

/** Raw → natural SVI. */
export function rawToNatural(p: RawSVI): NaturalSVI {
  validateRaw(p);
  const { a, b, rho, m, sigma } = p;
  const omega = (2 * b * sigma) / Math.sqrt(1 - rho * rho);
  const delta = a - (omega / 2) * (1 - rho * rho);
  const mu = m + (rho * sigma) / Math.sqrt(1 - rho * rho);
  const zeta = Math.sqrt(1 - rho * rho) / sigma;
  return { delta, mu, rho, omega, zeta };
}

/** Natural → raw SVI. */
export function naturalToRaw(n: NaturalSVI): RawSVI {
  const { delta, mu, rho, omega, zeta } = n;
  if (!(zeta > 0)) throw new RangeError(`zeta must be > 0, got ${zeta}`);
  const b = (omega * zeta) / 2;
  const sigma = Math.sqrt(1 - rho * rho) / zeta;
  const m = mu - (rho * sigma) / Math.sqrt(1 - rho * rho);
  const a = delta + (omega / 2) * (1 - rho * rho);
  return { a, b, rho, m, sigma };
}

/**
 * Raw → jump-wings SVI for a slice at maturity `t`. Jump-wings expresses the
 * slice in terms of directly observable quantities (ATM variance/skew, wings).
 */
export function rawToJumpWings(p: RawSVI, t: number): JumpWingsSVI {
  validateRaw(p);
  if (!(t > 0)) throw new RangeError(`t must be > 0, got ${t}`);
  const { a, b, rho, m, sigma } = p;
  const root = Math.sqrt(m * m + sigma * sigma);
  const wt = a + b * (-rho * m + root); // total variance at k=0
  const v = wt / t;
  const psi = (b / (2 * Math.sqrt(wt))) * (-m / root + rho);
  const pWing = (b * (1 - rho)) / Math.sqrt(wt);
  const cWing = (b * (1 + rho)) / Math.sqrt(wt);
  const vt = (a + b * sigma * Math.sqrt(1 - rho * rho)) / t;
  return { v, psi, p: pWing, c: cWing, vt };
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

export interface SmileQuote {
  /** Log-moneyness ln(K/F). */
  k: number;
  /** Observed total implied variance at k (σ_BS² · t). */
  totalVariance: number;
  weight?: number;
}

export interface SviCalibrationResult extends RawSVI {
  rmse: number;
  iterations: number;
  butterflyFree: boolean;
}

/**
 * Calibrate a raw SVI slice to a smile (total-variance quotes) by least
 * squares (Nelder–Mead), with (b, σ) kept positive and ρ in (−1, 1).
 */
export function calibrateRaw(
  quotes: readonly SmileQuote[],
  opts: { maxIter?: number; initial?: Partial<RawSVI> } = {},
): SviCalibrationResult {
  if (quotes.length < 5) throw new RangeError("SVI calibration needs at least 5 quotes");
  const ks = quotes.map((q) => q.k);
  const ws = quotes.map((q) => q.totalVariance);
  const wMin = Math.min(...ws);
  const kMid = ks[Math.floor(ks.length / 2)]!;

  // Initial guess.
  const a0 = opts.initial?.a ?? Math.max(wMin * 0.5, 1e-4);
  const b0 = opts.initial?.b ?? 0.1;
  const rho0 = opts.initial?.rho ?? -0.3;
  const m0 = opts.initial?.m ?? kMid;
  const sigma0 = opts.initial?.sigma ?? 0.1;

  // Unconstrained coords: a=v0, b=exp(v1), rho=tanh(v2), m=v3, sigma=exp(v4).
  const objective = (v: number[]): number => {
    const p: RawSVI = {
      a: v[0]!,
      b: Math.exp(v[1]!),
      rho: Math.tanh(v[2]!),
      m: v[3]!,
      sigma: Math.exp(v[4]!),
    };
    let sse = 0;
    for (let i = 0; i < quotes.length; i++) {
      const model = p.a + p.b * (p.rho * (ks[i]! - p.m) + Math.hypot(ks[i]! - p.m, p.sigma));
      const e = model - ws[i]!;
      sse += (quotes[i]!.weight ?? 1) * e * e;
    }
    return sse;
  };

  const start = [a0, Math.log(b0), Math.atanh(clamp(rho0, -0.999, 0.999)), m0, Math.log(sigma0)];
  const { point, iterations } = nelderMead(objective, start, opts.maxIter ?? 4000);
  const result: RawSVI = {
    a: point[0]!,
    b: Math.exp(point[1]!),
    rho: Math.tanh(point[2]!),
    m: point[3]!,
    sigma: Math.exp(point[4]!),
  };
  const totalW = quotes.reduce((s, q) => s + (q.weight ?? 1), 0);
  const rmse = Math.sqrt(objective(start.map((_, i) => point[i]!)) / totalW);
  return { ...result, rmse, iterations, butterflyFree: checkButterfly(result).arbitrageFree };
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/** Minimal Nelder–Mead simplex minimizer. */
function nelderMead(
  f: (x: number[]) => number,
  x0: number[],
  maxIter: number,
): { point: number[]; value: number; iterations: number } {
  const n = x0.length;
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const p = x0.slice();
    const xi = p[i]!;
    p[i] = xi + (xi !== 0 ? 0.1 * xi : 0.1);
    simplex.push(p);
  }
  let fvals = simplex.map(f);
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const order = fvals.map((_, i) => i).sort((a, b) => fvals[a]! - fvals[b]!);
    const s = order.map((i) => simplex[i]!);
    const fs = order.map((i) => fvals[i]!);
    for (let i = 0; i < s.length; i++) {
      simplex[i] = s[i]!;
      fvals[i] = fs[i]!;
    }
    if (Math.abs(fvals[n]! - fvals[0]!) < 1e-16) break;
    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j]! += simplex[i]![j]!;
    for (let j = 0; j < n; j++) centroid[j]! /= n;
    const worst = simplex[n]!;
    const reflected = centroid.map((c, j) => c + (c - worst[j]!));
    const fr = f(reflected);
    if (fr < fvals[0]!) {
      const expanded = centroid.map((c, j) => c + 2 * (reflected[j]! - c));
      const fe = f(expanded);
      simplex[n] = fe < fr ? expanded : reflected;
      fvals[n] = fe < fr ? fe : fr;
    } else if (fr < fvals[n - 1]!) {
      simplex[n] = reflected;
      fvals[n] = fr;
    } else {
      const contracted = centroid.map((c, j) => c + 0.5 * (worst[j]! - c));
      const fc = f(contracted);
      if (fc < fvals[n]!) {
        simplex[n] = contracted;
        fvals[n] = fc;
      } else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[0]!.map((b, j) => b + 0.5 * (simplex[i]![j]! - b));
          fvals[i] = f(simplex[i]!);
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < fvals.length; i++) if (fvals[i]! < fvals[best]!) best = i;
  return { point: simplex[best]!, value: fvals[best]!, iterations: iter };
}
