# Risk Report: Leverage & Position Sizing for Donchian Breakout

Scope: backtesting only, on public historical BTC/USDT data (Binance spot OHLCV + real USDM
perpetual funding rates). No live or testnet orders were placed to produce this report; nothing
here touches `src/exchange.py` or `testnet_bot.py`.

Covers the two walk-forward-validated Donchian breakout variants from
`validate_donchian_upgrade.py`:

- **donchian_25_10** (entry 25, exit 10) — best all-around, test Sharpe 1.46, CAGR 46%, max
  drawdown -25.6% (unleveraged, 1x).
- **donchian_70_20** (entry 70, exit 20) — lower risk, test Sharpe 0.97, CAGR 26.6%, max
  drawdown -31.6% (unleveraged, 1x), fewer/more selective trades.

Mechanics (daily-rebalanced exposure, intrabar liquidation vs. the day's low, real funding cost,
flat 0.5% maintenance-margin approximation rather than Binance's real tiered bracket table) are
exactly as documented in `src/leverage_backtest.py` — read that docstring before trusting a
liquidation count here as precise. All numbers below use the full cached price history
(2018-01-01 through the latest cached candle) unless a segment is labeled otherwise.

## 1. Flat-leverage sweep on the upgraded parameters

Re-run of `run_leverage.py` (now includes `donchian_25_10` / `donchian_70_20` in its
`STRATEGIES` dict) at 1x-20x. Original `donchian_20_10` included for reference — note its numbers
here differ slightly from the number quoted in earlier notes because the price cache has grown
(now extends to 2026) since that run.

| strategy | leverage | total return % | CAGR % | Sharpe | max DD % | liquidations |
|---|---:|---:|---:|---:|---:|---:|
| donchian_20_10 | 1x | 650.63 | 26.20 | 0.79 | -68.32 | 0 |
| donchian_20_10 | 2x | 1409.28 | 36.79 | 0.79 | -92.67 | 0 |
| donchian_20_10 | 3x | 683.21 | 26.82 | 0.79 | -98.81 | 0 |
| donchian_20_10 | 5x | -99.99 | -67.25 | 0.72 | -100.00 | 2 |
| donchian_20_10 | 10x | -100.00 | -99.99 | 0.67 | -100.00 | 27 |
| donchian_20_10 | 20x | -100.00 | -100.00 | 0.72 | -100.00 | 147 |
| **donchian_25_10** | 1x | 891.56 | 30.32 | 0.90 | -52.48 | 0 |
| **donchian_25_10** | 2x | 2925.08 | 48.23 | 0.90 | -79.83 | 0 |
| **donchian_25_10** | 3x | 2666.90 | 46.71 | 0.90 | -92.51 | 0 |
| **donchian_25_10** | 5x | -99.88 | -54.02 | 0.82 | -100.00 | 2 |
| **donchian_25_10** | 10x | -100.00 | -99.98 | 0.76 | -100.00 | 25 |
| **donchian_25_10** | 20x | -100.00 | -100.00 | 0.84 | -100.00 | 130 |
| **donchian_70_20** | 1x | 994.55 | 31.81 | 0.98 | -34.72 | 0 |
| **donchian_70_20** | 2x | 4320.69 | 54.86 | 0.98 | -61.80 | 0 |
| **donchian_70_20** | 3x | 6395.63 | 61.90 | 0.98 | -84.92 | 0 |
| **donchian_70_20** | 5x | -98.48 | -38.33 | 0.89 | -99.99 | 2 |
| **donchian_70_20** | 10x | -100.00 | -99.92 | 0.78 | -100.00 | 22 |
| **donchian_70_20** | 20x | -100.00 | -100.00 | 0.64 | -100.00 | 132 |

**Finding: the "no liquidations through low leverage" pattern holds for both upgraded
variants, and is actually stronger than the original 20/10 result** — for all three parameter
sets (20/10, 25/10, 70/20), leverage up to and including **3x produced zero liquidations** over
the full backtest, not just 2x. 5x+ breaks down identically across all three: 2 liquidations at
5x, 20-30+ at 10x, 130+ at 20x. This is not specific to 20/10; it looks like a property of
Donchian breakout's exit discipline (it cuts losing positions on a new N-day low well before a
single-day move would plausibly blow through a 2-3x maintenance margin buffer), not a lucky
artifact of one parameter pair.

**But "zero liquidations" is not the same as "safe."** Max drawdown at 3x is brutal even without
a technical liquidation: -92.5% (25/10) and -84.9% (70/20). Nobody holds through a 90%+
peak-to-trough drawdown in practice — margin calls, psychology, and the flat-maintenance-margin
approximation understating real intrabar risk all make 3x a backtest curiosity, not a usable
setting. Total return is also **not** monotonically increasing with leverage: for 20/10 and
25/10, 2x beats 3x on total return (compounding drag from daily rebalancing eats the extra
leverage); for 70/20, 3x happens to beat 2x on this specific historical path. That inconsistency
is itself a reason not to lean on "which leverage had the highest total return" as a decision
rule — Sharpe and drawdown are the more stable signals, and both say the same thing: **2x is
where the risk/return tradeoff is still sane; every step past that gets rapidly worse for
comparatively little extra Sharpe.**

## 2. Volatility-targeted position sizing

Implemented in `src/leverage_backtest.py` as `compute_target_leverage` +
`run_vol_targeted_backtest`, alongside the existing flat-leverage `run_leveraged_backtest` (same
file, same daily-rebalance/intrabar-liquidation/real-funding mechanics — only the leverage input
changes from a scalar to a per-day schedule). Method and assumptions, stated the same way the
rest of this module states them:

- Realized vol = rolling 20-day standard deviation of daily close-to-close returns, annualized by
  `sqrt(365)`.
- `leverage(t) = target_annual_vol / realized_vol(t)`, computed from data through day *t-1* (same
  one-bar lag the rest of this repo uses for signals) and clipped to `[0, 5]`.
- This targets **volatility**, not drawdown or tail risk directly. It's backward-looking: a
  regime change (calm → violent) only shows up in the rolling window *after* a few bad days have
  already happened, so it dampens ordinary chop more than it protects against a single sharp move
  — a standard, known limitation of simple realized-vol targeting, not specific to this
  implementation.

Vol-target schedules were calibrated (manual sweep, see `run_vol_target.py`'s docstring) so their
**average leverage while in a position**, measured over the full period, lands close to flat 2x
and flat 3x — the point of comparison the brief asked for ("comparable average exposure").
Calibration values used: `target_annual_vol = 1.0` → ~2x avg leverage, `1.5` → ~3x avg leverage,
for both variants.

## 3. Vol-target vs. flat leverage: full period and walk-forward test split

`run_vol_target.py` runs both schemes on both variants, over the full period and over the
2023-01-01+ walk-forward test split (same split as `validate_donchian_upgrade.py`) — the standard
this project already holds every other result to.

| strategy | segment | scheme | avg leverage | total return % | CAGR % | Sharpe | max DD % | liqs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| donchian_25_10 | full | flat 2x | 2.00 | 2925.08 | 48.23 | 0.90 | -79.83 | 0 |
| donchian_25_10 | full | flat 3x | 3.00 | 2666.90 | 46.71 | 0.90 | -92.51 | 0 |
| donchian_25_10 | full | vol-target ~2x | 2.09 | 7110.34 | 63.86 | 1.00 | -83.97 | 0 |
| donchian_25_10 | full | vol-target ~3x | 3.06 | 7596.92 | 65.10 | 0.99 | -95.03 | 0 |
| donchian_25_10 | **test (23+)** | flat 2x | 2.00 | 681.73 | 75.38 | **1.25** | **-54.88** | 0 |
| donchian_25_10 | **test (23+)** | flat 3x | 3.00 | 1298.03 | 105.57 | **1.25** | **-70.80** | 0 |
| donchian_25_10 | **test (23+)** | vol-target ~2x | 2.48 | 611.64 | 70.94 | 1.08 | -72.80 | 0 |
| donchian_25_10 | **test (23+)** | vol-target ~3x | 3.60 | 1025.76 | 93.76 | 1.13 | -84.03 | 0 |
| donchian_70_20 | full | flat 2x | 2.00 | 4320.69 | 54.86 | 0.98 | -61.80 | 0 |
| donchian_70_20 | full | flat 3x | 3.00 | 6395.63 | 61.90 | 0.98 | -84.92 | 0 |
| donchian_70_20 | full | vol-target ~2x | 2.02 | 4868.15 | 56.96 | 0.97 | -69.90 | 0 |
| donchian_70_20 | full | vol-target ~3x | 2.96 | 8649.04 | 67.56 | 0.99 | -82.89 | 0 |
| donchian_70_20 | **test (23+)** | flat 2x | 2.00 | 260.28 | 41.93 | **0.89** | **-54.65** | 0 |
| donchian_70_20 | **test (23+)** | flat 3x | 3.00 | 341.75 | 50.06 | **0.89** | **-71.44** | 0 |
| donchian_70_20 | **test (23+)** | vol-target ~2x | 2.37 | 207.28 | 35.89 | 0.78 | -69.90 | 0 |
| donchian_70_20 | **test (23+)** | vol-target ~3x | 3.49 | 276.09 | 43.61 | 0.84 | -82.89 | 0 |

**Honest answer: no, vol-targeting as implemented does not clearly help — and on the metric
that matters most here (the walk-forward test split), it's worse on both Sharpe and max
drawdown, for both variants, at matched average exposure.**

- **Full period**: vol-targeting shows a higher Sharpe and much higher total return than flat
  leverage at the same average exposure (e.g. donchian_25_10 ~2x: Sharpe 1.00 vs. 0.90, total
  return 7110% vs. 2925%) — but *worse* max drawdown in every case (-83.97% vs. -79.83% there).
  This looks like it's driven by the 2020-2021 bull run: vol-targeting happened to lever up
  during calm uptrending stretches of that period, which is exactly the kind of full-period
  result the team has already learned to distrust (same pattern as the 25/20 Donchian pick that
  looked best full-period but didn't hold up on the test split).
- **Test split (2023+)**: vol-targeting underperforms flat leverage on **Sharpe** (1.08 vs. 1.25
  and 1.13 vs. 1.25 for donchian_25_10; 0.78 vs. 0.89 and 0.84 vs. 0.89 for donchian_70_20) *and*
  on **max drawdown** (worse in all four matched comparisons) *and* on total return in 3 of 4
  comparisons. It does not reduce liquidation risk either — both schemes had zero liquidations at
  this exposure, so there was no liquidation risk left to reduce.
- A robustness check with a faster 10-day (instead of 20-day) realized-vol window narrows the gap
  slightly (Sharpe 1.20 vs. flat's 1.25 for donchian_25_10 ~2x on the test split) but the
  direction doesn't flip — drawdown is still meaningfully worse (-61.99% vs. flat's -54.88%).
- **Mechanism**: this is the backward-looking-vol assumption stated in the docstring showing up
  concretely. At matched *average* exposure, vol-targeting spends calm stretches above the flat
  baseline's leverage (sometimes well above — realized-vol windows this data hits values near the
  5x cap ~1% of active days) and only cuts back once the rolling window has already absorbed a
  spike. On this data, the calm-before-a-drawdown pattern happened often enough that the extra
  leverage during calm periods cost more (in drawdown) than the reduced leverage during
  already-volatile periods saved.

This is a negative result for naive realized-vol targeting on this strategy/data, not a
confirmation of the hoped-for improvement — reported as plainly as the positive findings above,
per this project's own standard.

## 4. Recommendation

- **Safe/practical leverage range for both donchian_25_10 and donchian_70_20: 1x-2x.** 2x is
  where the backtest still shows a sane risk/return tradeoff (max DD in the -60% to -80% range,
  zero liquidations, Sharpe roughly in line with 1x). This generalizes the original 20/10 finding
  — it is not specific to those parameters.
- **3x is a backtest curiosity, not a usable setting**, despite zero liquidations at 3x across
  all three parameter sets tested. A -85% to -98% max drawdown is not survivable in practice
  (margin calls, psychology, and the flat 0.5% maintenance-margin approximation likely
  understating real liquidation risk relative to Binance's actual tiered bracket table — see
  `src/leverage_backtest.py`'s docstring). Treat any "3x survives" claim as a statement about this
  specific backtest's liquidation model, not a live-safety guarantee.
- **5x and above: hard no.** Every parameter set tested gets liquidated repeatedly at 5x+
  (2 liquidations at 5x scaling to 130+ at 20x) — this is not sensitive to which Donchian
  variant is used.
- **Do not adopt the vol-targeted sizing implemented here as a live risk control.** It looked
  better on the full period but was worse on Sharpe and drawdown on the walk-forward test split
  for both variants — exactly the split this project treats as the one that counts. If the team
  wants to revisit volatility-aware sizing later, a faster-reacting vol estimator (EWMA/GARCH
  instead of a rolling window) or capping leverage *below* the flat baseline instead of matching
  its average would be more defensible starting points than what's here — but neither has been
  tested, and this report is not recommending them, only naming them as the more promising
  direction if the team wants to keep pushing on this.
- **If the team ever moves toward live paper-trading** (via the separate, out-of-scope
  `testnet_bot.py`): use **flat 2x leverage, not vol-targeted sizing, and not 3x**, plus a
  hard max-drawdown circuit breaker independent of the leverage mechanism itself (this backtest's
  maintenance-margin model is a simplification, and a real tiered bracket table plus real-time
  mark price could liquidate earlier than this data suggests). That combination — a conservative
  flat multiplier and an explicit kill switch — is a more defensible starting point than either
  a higher flat multiplier or the naive vol-targeting scheme tested here.

## Reproducing these numbers

```bash
python3 run_leverage.py --strategy donchian_25_10   # section 1
python3 run_leverage.py --strategy donchian_70_20   # section 1
python3 run_vol_target.py                            # sections 2-3
```

Outputs: `results/leverage_metrics_donchian_{25_10,70_20}.json`,
`results/leverage_equity_curve_donchian_{25_10,70_20}.png`, `results/vol_target_comparison.json`.
