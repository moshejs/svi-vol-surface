# svi-vol-surface

Gatheral's **SVI** implied-volatility surface — the raw / natural / jump-wings parametrizations, total implied variance, **butterfly & calendar arbitrage checks**, and smile **calibration**. Zero dependencies.

```
npm install svi-vol-surface
```

## Why

SVI is the standard parametric form for an equity implied-volatility smile, largely because its no-arbitrage conditions are well understood (Gatheral & Jacquier, 2014). npm had no SVI implementation at all. This one parametrizes a slice, converts between the three standard forms, checks both static (butterfly) and calendar arbitrage, and calibrates to market quotes.

```ts
import { totalVariance, checkButterfly, calibrateRaw } from "svi-vol-surface";

const slice = { a: 0.04, b: 0.1, rho: -0.3, m: 0, sigma: 0.2 };
totalVariance(slice, 0.1);        // w(k) = σ²·t at log-moneyness 0.1

checkButterfly(slice).arbitrageFree; // true — g(k) ≥ 0 everywhere

// Fit a slice to a market smile (total-variance quotes):
const fit = calibrateRaw(quotes);
fit.rmse; fit.butterflyFree;
```

## API

Log-moneyness `k = ln(K/F)`; a slice returns **total** implied variance `w(k) = σ_BS(k)²·t`.

- **`totalVariance(raw, k)`** — `a + b(ρ(k−m) + √((k−m)² + σ²))`; **`impliedVol(raw, k, t)`** = `√(w/t)`.
- **`dTotalVariance`**, **`d2TotalVariance`** — analytic w′(k), w″(k).
- **`minTotalVariance(raw)`** — `a + bσ√(1−ρ²)`, the smallest total variance of the slice.
- **`durrlemanG(raw, k)`** — Durrleman's butterfly function. Where w > 0 the risk-neutral density has the sign of g, so the slice has no static arbitrage iff w > 0 and g ≥ 0 everywhere.
- **`checkButterfly(raw)`** — requires `minTotalVariance ≥ 0`, Lee's wing bound `b(1+|ρ|) ≤ 2`, and g(k) ≥ 0. g is minimized over a grid that is dense near `m` and reaches |k − m| = 10⁴, with each local minimum refined. Returns `arbitrageFree`, `minG`, `minK`, `wingConditionOk`, `minTotalVariance`.
- **`checkCalendar(near, far)`** — `w_far(k) ≥ w_near(k)` for all k: the far wings must be at least as steep, and the gap is minimized over the same kind of wide, refined grid.
- **`rawToNatural` / `naturalToRaw` / `rawToJumpWings` / `jumpWingsToRaw`** — conversions between the three parametrizations (Gatheral & Jacquier 2014, §3).
- **`calibrateRaw(quotes, opts?)`** — weighted least-squares fit of a raw slice using the quasi-explicit method (Zeliade 2009): exact constrained linear least squares for (a, bρσ, bσ) inside a multi-start search over (m, σ). Returns params + `rmse` + `butterflyFree`. The fit itself is not constrained to be arbitrage-free.

Invalid parameters (b < 0, |ρ| ≥ 1, σ ≤ 0, or any NaN / non-finite value) throw `RangeError`.

## Correctness

Fixtures come from independent references ([`test/generate-fixtures.py`](test/generate-fixtures.py)):

- **g(k)**: 80 values of Durrleman's g recovered from the risk-neutral density of Black call prices (40-digit `mpmath` second derivative), matching the closed form to 1e-12.
- **Butterfly verdicts**: 600 random slices judged by a separate numpy implementation on a 200k-point grid. An audit over 20,000 slices agreed on every one.
- **Calendar verdicts**: 400 random slice pairs judged on a 1.4M-point grid out to |k| = 10⁵. An audit over 4,000 pairs agreed on every one.
- **Calibration**: 12 noisy smiles where `calibrateRaw` reaches the best SSE that `scipy.optimize.least_squares` finds from 162 starting points. An audit over 60 smiles matched to 1e-11, and 900 exact synthetic smiles were recovered to RMSE < 2e-9.

Also tested: the **Axel Vogt slice from Gatheral & Jacquier (2014)**, a published example with butterfly arbitrage, is flagged. Analytic derivatives are checked against finite differences, and natural and jump-wings conversions round-trip.

## Related

Pairs with [`hagan-sabr`](https://github.com/moshejs/hagan-sabr) (the SABR smile model). Part of a quant/fixed-income toolkit: [`compounded-sofr`](https://github.com/moshejs/compounded-sofr) · [`day-count`](https://github.com/moshejs/day-count) · [`instrument-identifiers`](https://github.com/moshejs/instrument-identifiers).

## Author

Built by **[Moshe Malka](https://moshemalka.com)** — engineering leader in New York City. Studio work at [Quentin.Code](https://www.quentin.software/).

MIT © Moshe Malka
