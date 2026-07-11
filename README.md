# svi-vol-surface

Gatheral's **SVI** implied-volatility surface — the raw / natural / jump-wings parametrizations, total implied variance, **butterfly & calendar arbitrage checks**, and smile **calibration**. Zero dependencies.

```
npm install svi-vol-surface
```

## Why

SVI is the standard parametric form for an equity implied-volatility smile, and the reason it's used is that its no-arbitrage conditions are known in closed form (Gatheral & Jacquier, 2014). npm had no SVI implementation at all. This one parametrizes a slice, converts between the three standard forms, checks both static (butterfly) and calendar arbitrage, and calibrates to market quotes.

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
- **`durrlemanG(raw, k)`** — Durrleman's butterfly function (≥ 0 everywhere ⇔ no static arbitrage).
- **`checkButterfly(raw)`** — scans g(k) and the wing condition `b(1+|ρ|) ≤ 4`.
- **`checkCalendar(near, far)`** — total variance non-decreasing in maturity.
- **`rawToNatural` / `naturalToRaw` / `rawToJumpWings`** — the three parametrizations.
- **`calibrateRaw(quotes, opts?)`** — least-squares fit (Nelder–Mead) of a raw slice; returns params + `rmse` + `butterflyFree`.

Invalid parameters (b < 0, |ρ| ≥ 1, σ ≤ 0) throw `RangeError`.

## Correctness

The static-arbitrage check is validated against the **Axel Vogt parameters from Gatheral & Jacquier (2014)** — a published example that *does* contain butterfly arbitrage — which `checkButterfly` correctly flags (g(k) < 0), while a clean slice passes. Analytic derivatives are checked against finite differences, parametrization conversions round-trip, and calibration recovers known parameters from a synthetic smile.

## Related

Pairs with [`hagan-sabr`](https://github.com/moshejs/hagan-sabr) (the SABR smile model). Part of a quant/fixed-income toolkit: [`compounded-sofr`](https://github.com/moshejs/compounded-sofr) · [`day-count`](https://github.com/moshejs/day-count) · [`instrument-identifiers`](https://github.com/moshejs/instrument-identifiers).

## Author

Built by **[Moshe Malka](https://moshemalka.com)** — engineering leader in New York City. Studio work at [Quentin.Code](https://www.quentin.software/).

MIT © Moshe Malka
