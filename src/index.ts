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
  /** ATM variance v (the ATM total variance is v·t). */
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
  if (!Number.isFinite(p.a)) throw new RangeError(`a must be finite, got ${p.a}`);
  if (!(p.b >= 0 && p.b < Infinity)) throw new RangeError(`b must be finite and ≥ 0, got ${p.b}`);
  if (!(p.rho > -1 && p.rho < 1)) throw new RangeError(`rho must be in (-1,1), got ${p.rho}`);
  if (!Number.isFinite(p.m)) throw new RangeError(`m must be finite, got ${p.m}`);
  if (!(p.sigma > 0 && p.sigma < Infinity)) {
    throw new RangeError(`sigma must be finite and > 0, got ${p.sigma}`);
  }
}

/** Minimum total variance of a raw slice, a + bσ√(1−ρ²), attained at k = m − ρσ/√(1−ρ²). */
export function minTotalVariance(p: RawSVI): number {
  validateRaw(p);
  return p.a + p.b * p.sigma * Math.sqrt(1 - p.rho * p.rho);
}

/**
 * Log-moneyness grid k = m + σ·sinh(u), uniform in u. It is dense where a
 * slice curves (|k − m| ≲ σ) and reaches |k − m| = `reach` in the wings, so
 * features of width ~σ and the asymptotic regime are both sampled.
 */
function wideGrid(m: number, sigma: number, reach = 1e4, perSide = 2000): number[] {
  const U = Math.asinh(reach / sigma);
  const out: number[] = [];
  for (let i = -perSide; i <= perSide; i++) out.push(m + sigma * Math.sinh((U * i) / perSide));
  return out;
}

/**
 * Minimum of f over a sorted grid, with every interior local minimum refined
 * by golden-section search between its neighbours.
 */
function gridMin(f: (k: number) => number, ks: number[]): { min: number; at: number } {
  const vals = ks.map(f);
  let min = Infinity;
  let at = ks[0]!;
  for (let i = 0; i < ks.length; i++) {
    if (vals[i]! < min) {
      min = vals[i]!;
      at = ks[i]!;
    }
  }
  for (let i = 1; i < ks.length - 1; i++) {
    if (!(vals[i]! <= vals[i - 1]! && vals[i]! <= vals[i + 1]!)) continue;
    let lo = ks[i - 1]!;
    let hi = ks[i + 1]!;
    const r = (Math.sqrt(5) - 1) / 2;
    let x1 = hi - r * (hi - lo);
    let x2 = lo + r * (hi - lo);
    let f1 = f(x1);
    let f2 = f(x2);
    for (let it = 0; it < 60 && hi - lo > 1e-14 * (1 + Math.abs(lo)); it++) {
      if (f1 < f2) {
        hi = x2;
        x2 = x1;
        f2 = f1;
        x1 = hi - r * (hi - lo);
        f1 = f(x1);
      } else {
        lo = x1;
        x1 = x2;
        f1 = f2;
        x2 = lo + r * (hi - lo);
        f2 = f(x2);
      }
    }
    for (const [x, fx] of [[x1, f1], [x2, f2]] as const) {
      if (fx < min) {
        min = fx;
        at = x;
      }
    }
  }
  return { min, at };
}

/**
 * Total implied variance w(k) for a raw SVI slice at log-moneyness k.
 *
 * ```ts
 * totalVariance({ a: 0.04, b: 0.4, rho: -0.4, m: 0, sigma: 0.1 }, 0); // 0.08
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
  validateRaw(p);
  const km = k - p.m;
  return p.b * (p.rho + km / Math.sqrt(km * km + p.sigma * p.sigma));
}

/** Second derivative w''(k) of total variance. */
export function d2TotalVariance(p: RawSVI, k: number): number {
  validateRaw(p);
  const km = k - p.m;
  const s2 = p.sigma * p.sigma;
  const root = Math.sqrt(km * km + s2);
  return (p.b * s2) / (root * root * root);
}

/**
 * Durrleman's g function. Where w(k) > 0, the risk-neutral density at k has
 * the sign of g(k), so a slice is free of butterfly (static) arbitrage iff
 * w > 0 and g(k) ≥ 0 for all k. (g is meaningless where w ≤ 0.)
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
  /** Minimum of g(k) found (≥ 0 ⇒ butterfly-free where w > 0). */
  minG: number;
  /** k at which the minimum occurred. */
  minK: number;
  /**
   * Whether the asymptotic wing slopes b(1 ± ρ) are ≤ 2, i.e. b(1+|ρ|) ≤ 2
   * (Lee's moment formula; with a steeper wing g(k) → 1/4 − slope²/16 < 0).
   */
  wingConditionOk: boolean;
  /** Minimum total variance a + bσ√(1−ρ²); must be ≥ 0. */
  minTotalVariance: number;
}

/**
 * Check a slice for butterfly (static) arbitrage: total variance must be
 * non-negative (checked in closed form), the wings must satisfy Lee's bound
 * b(1+|ρ|) ≤ 2, and g(k) must be ≥ 0. g is minimized over a grid that is
 * dense near m (resolution well below σ) and reaches |k − m| = 10⁴, with each
 * local minimum refined by golden-section search; `range`/`steps` add a
 * uniform grid on [m − range, m + range] on top of that.
 */
export function checkButterfly(p: RawSVI, range = 2, steps = 1000): ButterflyCheck {
  validateRaw(p);
  const minW = minTotalVariance(p);
  const ks = wideGrid(p.m, p.sigma);
  for (let i = 0; i <= steps; i++) ks.push(p.m - range + (2 * range * i) / steps);
  ks.sort((x, y) => x - y);
  // g is undefined where w ≤ 0; such slices fail the variance test anyway.
  const g = (k: number): number => (totalVariance(p, k) > 0 ? durrlemanG(p, k) : Infinity);
  const { min, at } = gridMin(g, ks);
  const wingConditionOk = p.b * (1 + Math.abs(p.rho)) <= 2;
  return {
    arbitrageFree: minW >= 0 && min >= -1e-10 && wingConditionOk,
    minG: min,
    minK: at,
    wingConditionOk,
    minTotalVariance: minW,
  };
}

/**
 * Calendar-arbitrage check between two slices at increasing maturities. A
 * surface is calendar-arbitrage-free when total variance is non-decreasing in
 * maturity at every log-moneyness: w₂(k) ≥ w₁(k) for t₂ > t₁.
 */
export interface CalendarCheck {
  arbitrageFree: boolean;
  /** Minimum of w₂(k) − w₁(k) found (≥ 0 ⇒ calendar-free). */
  minGap: number;
  minK: number;
}

/**
 * Check w_far(k) ≥ w_near(k) for all k. The far slice's wing slopes must be at
 * least the near slice's (otherwise the slices cross far out in a wing), and
 * the gap is minimized over grids dense around both slices' m and reaching
 * |k| ≈ 10⁴, with local minima refined; `range`/`steps` add a uniform grid on
 * [−range, range].
 */
export function checkCalendar(
  near: RawSVI,
  far: RawSVI,
  range = 2,
  steps = 1000,
): CalendarCheck {
  validateRaw(near);
  validateRaw(far);
  const ks = [...wideGrid(near.m, near.sigma), ...wideGrid(far.m, far.sigma)];
  for (let i = 0; i <= steps; i++) ks.push(-range + (2 * range * i) / steps);
  ks.sort((x, y) => x - y);
  const gap = (k: number): number => totalVariance(far, k) - totalVariance(near, k);
  const { min, at } = gridMin(gap, ks);
  const tol = 1e-12;
  const wingsOk =
    far.b * (1 + far.rho) >= near.b * (1 + near.rho) - tol &&
    far.b * (1 - far.rho) >= near.b * (1 - near.rho) - tol;
  return { arbitrageFree: wingsOk && min >= -1e-10, minGap: min, minK: at };
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
  if (!(zeta > 0 && zeta < Infinity)) throw new RangeError(`zeta must be finite and > 0, got ${zeta}`);
  if (!(omega >= 0 && omega < Infinity)) throw new RangeError(`omega must be finite and ≥ 0, got ${omega}`);
  if (!(rho > -1 && rho < 1)) throw new RangeError(`rho must be in (-1,1), got ${rho}`);
  if (!Number.isFinite(delta) || !Number.isFinite(mu)) {
    throw new RangeError(`delta and mu must be finite, got ${delta}, ${mu}`);
  }
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
  if (!(wt > 0)) throw new RangeError(`ATM total variance must be > 0 for jump-wings, got ${wt}`);
  const v = wt / t;
  const psi = (b / (2 * Math.sqrt(wt))) * (-m / root + rho);
  const pWing = (b * (1 - rho)) / Math.sqrt(wt);
  const cWing = (b * (1 + rho)) / Math.sqrt(wt);
  const vt = (a + b * sigma * Math.sqrt(1 - rho * rho)) / t;
  return { v, psi, p: pWing, c: cWing, vt };
}

/**
 * Jump-wings → raw SVI for a slice at maturity `t` (the inverse of
 * {@link rawToJumpWings}; Gatheral & Jacquier 2014, Lemma 3.2). Throws when
 * no raw slice has the given jump-wings parameters. The map is ill-conditioned
 * when the variance minimum is close to k = 0 (ψ ≈ 0, v ≈ vt): σ and m are then
 * recovered only to a few digits fewer than the inputs.
 */
export function jumpWingsToRaw(jw: JumpWingsSVI, t: number): RawSVI {
  if (!(t > 0 && t < Infinity)) throw new RangeError(`t must be > 0, got ${t}`);
  const { v, psi, p, c, vt } = jw;
  if (!(v > 0 && v < Infinity)) throw new RangeError(`v must be finite and > 0, got ${v}`);
  if (!(p >= 0 && c >= 0 && p + c > 0 && p + c < Infinity)) {
    throw new RangeError(`wing slopes p, c must be ≥ 0 with p + c > 0, got p=${p}, c=${c}`);
  }
  if (!Number.isFinite(psi) || !Number.isFinite(vt)) {
    throw new RangeError(`psi and vt must be finite, got ${psi}, ${vt}`);
  }
  const wt = v * t;
  const sw = Math.sqrt(wt);
  const b = (sw / 2) * (c + p);
  const rho = 1 - (p * sw) / b;
  // β = m/√(m² + σ²) ∈ (−1, 1).
  const beta = rho - (2 * psi * sw) / b;
  if (!(beta > -1 && beta < 1)) {
    throw new RangeError(`inconsistent jump-wings parameters: |ψ| too large for the wings (β=${beta})`);
  }
  if (rho === beta) {
    // m/√(m²+σ²) = ρ puts the variance minimum at k = 0, where σ is not identified.
    throw new RangeError("ψ = 0 puts the variance minimum at k = 0, where σ is not determined by (v, vt)");
  }
  if (!(v > vt)) throw new RangeError(`v must exceed the minimum variance vt, got v=${v}, vt=${vt}`);
  // σ = (v − ṽ)t·√(1−β²) / (b(1 − ρβ − √(1−β²)√(1−ρ²))), with the denominator
  // rewritten as (β − ρ)²/(1 − ρβ + √(1−β²)√(1−ρ²)) to avoid cancellation.
  const sb = Math.sqrt(1 - beta * beta);
  const sr = Math.sqrt(1 - rho * rho);
  const sigma = ((v - vt) * t * sb * (1 - rho * beta + sb * sr)) / (b * (beta - rho) * (beta - rho));
  const m = (sigma * beta) / sb;
  const a = vt * t - b * sigma * sr;
  return { a, b, rho, m, sigma };
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
 * Calibrate a raw SVI slice to a smile (total-variance quotes) by weighted
 * least squares, using the quasi-explicit method of Zeliade (2009): with
 * y = (k − m)/σ the slice is w = a + d·y + c·√(y² + 1) (d = bσρ, c = bσ), which
 * is linear in (a, d, c). For each (m, σ) the best (a, d, c) subject to
 * |d| ≤ c is solved exactly, and only (m, σ) are searched numerically
 * (multi-start Nelder–Mead). `initial.m` / `initial.sigma` add a starting
 * point; `initial.a`, `b`, `rho` are not needed (they are solved exactly). The
 * fit is not constrained to be arbitrage-free; `butterflyFree` reports whether
 * it is.
 */
export function calibrateRaw(
  quotes: readonly SmileQuote[],
  opts: { maxIter?: number; initial?: Partial<RawSVI> } = {},
): SviCalibrationResult {
  if (quotes.length < 5) throw new RangeError("SVI calibration needs at least 5 quotes");
  for (const q of quotes) {
    if (!Number.isFinite(q.k)) throw new RangeError(`quote k must be finite, got ${q.k}`);
    if (!Number.isFinite(q.totalVariance)) {
      throw new RangeError(`quote totalVariance must be finite, got ${q.totalVariance}`);
    }
    const w = q.weight ?? 1;
    if (!(w >= 0 && w < Infinity)) throw new RangeError(`quote weight must be ≥ 0, got ${w}`);
  }
  const ks = quotes.map((q) => q.k);
  const ws = quotes.map((q) => q.totalVariance);
  const sw = quotes.map((q) => Math.sqrt(q.weight ?? 1));
  const kMin = Math.min(...ks);
  const kMax = Math.max(...ks);
  const span = Math.max(kMax - kMin, 1e-8);

  // Inner problem: weighted LS for (a, d, c) given (m, σ), with |d| ≤ ρmax·c.
  const RHO_MAX = 1 - 1e-9;
  const inner = (m: number, sigma: number): { sse: number; a: number; d: number; c: number } => {
    const y = ks.map((k) => (k - m) / sigma);
    const r = y.map((yi) => Math.sqrt(yi * yi + 1));
    let best = { sse: Infinity, a: 0, d: 0, c: 0 };
    const consider = (a: number, d: number, c: number): void => {
      if (!(c >= 0 && Math.abs(d) <= RHO_MAX * c + 1e-300)) return;
      let sse = 0;
      for (let i = 0; i < ks.length; i++) {
        const e = sw[i]! * (a + d * y[i]! + c * r[i]! - ws[i]!);
        sse += e * e;
      }
      if (sse < best.sse) best = { sse, a, d, c };
    };
    // Interior: unconstrained 3-parameter fit.
    const full = lstsq([ks.map(() => 1), y, r], ws, sw);
    if (full) consider(full[0]!, full[1]!, full[2]!);
    // Faces d = ±ρmax·c: 2-parameter fits in (a, c).
    for (const sgn of [1, -1]) {
      const col = y.map((yi, i) => r[i]! + sgn * RHO_MAX * yi);
      const face = lstsq([ks.map(() => 1), col], ws, sw);
      if (face) consider(face[0]!, sgn * RHO_MAX * face[1]!, face[1]!);
    }
    // Edge c = d = 0: flat smile.
    const flat = lstsq([ks.map(() => 1)], ws, sw);
    if (flat) consider(flat[0]!, 0, 0);
    return best;
  };

  // Outer search over (m, ln σ).
  const objective = (v: number[]): number => {
    const sigma = Math.exp(v[1]!);
    if (!(sigma > 0 && sigma < Infinity) || !Number.isFinite(v[0]!)) return Infinity;
    return inner(v[0]!, sigma).sse;
  };
  const starts: number[][] = [];
  if (opts.initial?.m !== undefined || opts.initial?.sigma !== undefined) {
    starts.push([opts.initial.m ?? (kMin + kMax) / 2, Math.log(opts.initial.sigma ?? 0.1 * span)]);
  }
  for (const fm of [0.25, 0.5, 0.75]) {
    for (const fs of [0.02, 0.1, 0.5]) starts.push([kMin + fm * span, Math.log(fs * span)]);
  }
  const maxIter = opts.maxIter ?? 4000;
  const budget = Math.max(1, Math.floor(maxIter / starts.length));
  let point = starts[0]!;
  let value = Infinity;
  let iterations = 0;
  for (const x0 of starts) {
    const r = nelderMead(objective, x0, budget);
    iterations += r.iterations;
    if (r.value < value) {
      point = r.point;
      value = r.value;
    }
  }
  const m = point[0]!;
  const sigma = Math.exp(point[1]!);
  const { a, d, c } = inner(m, sigma);
  const result: RawSVI = { a, b: c / sigma, rho: c > 0 ? d / c : 0, m, sigma };
  const totalW = quotes.reduce((acc, q) => acc + (q.weight ?? 1), 0);
  const rmse = Math.sqrt(value / totalW);
  return { ...result, rmse, iterations, butterflyFree: checkButterfly(result).arbitrageFree };
}

/**
 * Weighted linear least squares via modified Gram–Schmidt QR. `cols` are the
 * design-matrix columns. Returns null when the columns are rank-deficient.
 */
function lstsq(cols: number[][], rhs: number[], sw: number[]): number[] | null {
  const p = cols.length;
  const n = rhs.length;
  const q = cols.map((col) => col.map((v, i) => v * sw[i]!));
  const R: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let t = 0; t < n; t++) dot += q[i]![t]! * q[j]![t]!;
      R[i]![j] = dot;
      for (let t = 0; t < n; t++) q[j]![t]! -= dot * q[i]![t]!;
    }
    let norm = 0;
    for (let t = 0; t < n; t++) norm += q[j]![t]! * q[j]![t]!;
    norm = Math.sqrt(norm);
    let colNorm = 0;
    for (let t = 0; t < n; t++) colNorm += (cols[j]![t]! * sw[t]!) ** 2;
    if (!(norm > 1e-12 * Math.sqrt(colNorm))) return null;
    R[j]![j] = norm;
    for (let t = 0; t < n; t++) q[j]![t]! /= norm;
  }
  // Project the right-hand side the same (modified) way, for stability.
  const b = rhs.map((v, t) => v * sw[t]!);
  const qtb = q.map((col) => {
    let dot = 0;
    for (let t = 0; t < n; t++) dot += col[t]! * b[t]!;
    for (let t = 0; t < n; t++) b[t]! -= dot * col[t]!;
    return dot;
  });
  const x = new Array<number>(p).fill(0);
  for (let j = p - 1; j >= 0; j--) {
    let v = qtb[j]!;
    for (let i = j + 1; i < p; i++) v -= R[j]![i]! * x[i]!;
    x[j] = v / R[j]![j]!;
  }
  return x;
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
