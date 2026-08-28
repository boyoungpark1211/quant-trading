"""Compare flat leverage against volatility-targeted leverage sizing (see
`src.leverage_backtest.compute_target_leverage`) on the walk-forward-validated Donchian variants
(donchian_25_10, donchian_70_20), over both the full backtest period and the walk-forward test
split (2023-01-01 -> now, same split used in validate_donchian_upgrade.py).

Vol-target schedules are calibrated (via a manual sweep, not fit inside this script) so their
average leverage-while-in-position over the *full* period lands close to flat 2x and flat 3x --
"comparable average exposure" per the brief. The same target_annual_vol is then applied unchanged
to the train/test split; a real deployment can't recalibrate its target using data from the
future either, and the realized-vol series is computed once over the full price history so the
test segment isn't cold-started with zero volatility context at 2023-01-01.
"""

import json
from pathlib import Path

from src.data import fetch_daily_funding_rate, fetch_ohlcv
from src.leverage_backtest import (
    compute_target_leverage,
    run_leveraged_backtest,
    run_vol_targeted_backtest,
)
from src.strategies import donchian_breakout

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_DATE = "2023-01-01"
VOL_LOOKBACK = 20
MAX_LEVERAGE = 5.0

STRATEGIES = {
    "donchian_25_10": lambda df: donchian_breakout(df, entry_lookback=25, exit_lookback=10),
    "donchian_70_20": lambda df: donchian_breakout(df, entry_lookback=70, exit_lookback=20),
}

# target_annual_vol -> lands at roughly the stated avg leverage-while-in-position, calibrated
# against the full period for each strategy (donchian_25_10: 1.0 -> ~2.09x, 1.5 -> ~3.06x;
# donchian_70_20: 1.0 -> ~2.02x, 1.5 -> ~2.96x).
VOL_TARGETS = {
    "donchian_25_10": {"vol-target ~2x": 1.0, "vol-target ~3x": 1.5},
    "donchian_70_20": {"vol-target ~2x": 1.0, "vol-target ~3x": 1.5},
}


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    funding = fetch_daily_funding_rate("BTC/USDT", since="2019-09-01")
    test_df = df[df.index >= SPLIT_DATE]

    all_results = {}
    for strat_name, strat_fn in STRATEGIES.items():
        full_position = strat_fn(df)

        # Compute each vol-target leverage schedule once over the *full* price history so the
        # test segment inherits proper vol warm-up instead of restarting the rolling window.
        full_target_leverage = {
            label: compute_target_leverage(
                df, target_annual_vol=tv, vol_lookback=VOL_LOOKBACK, min_leverage=0.0, max_leverage=MAX_LEVERAGE
            )
            for label, tv in VOL_TARGETS[strat_name].items()
        }

        print(f"\n=== {strat_name} ===")
        print(
            f"{'segment':<12}{'scheme':<18}{'avg_lev':>8}{'total_ret%':>13}{'cagr%':>9}"
            f"{'sharpe':>8}{'max_dd%':>9}{'liqs':>6}"
        )
        for seg_name, seg_df in [("full", df), ("test (23+)", test_df)]:
            seg_position = full_position.reindex(seg_df.index)
            seg_funding = funding.reindex(seg_df.index)

            schemes = []
            for lev in (2, 3):
                res = run_leveraged_backtest(seg_df, seg_position, leverage=lev, funding_daily=seg_funding)
                schemes.append((f"flat {lev}x", float(lev), res))
            for label, target_leverage in full_target_leverage.items():
                seg_target_leverage = target_leverage.reindex(seg_df.index)
                res = run_vol_targeted_backtest(seg_df, seg_position, seg_target_leverage, funding_daily=seg_funding)
                schemes.append((label, res.metrics.get("avg_leverage_when_in_position", 0.0), res))

            for scheme_name, avg_lev, res in schemes:
                m = res.metrics
                print(
                    f"{seg_name:<12}{scheme_name:<18}{avg_lev:>8.2f}{m['total_return_pct']:>13}"
                    f"{m['cagr_pct']:>9}{m['sharpe']:>8}{m['max_drawdown_pct']:>9}{m['n_liquidations']:>6}"
                )
                all_results.setdefault(strat_name, {}).setdefault(seg_name, {})[scheme_name] = {
                    "avg_leverage": round(avg_lev, 2),
                    **m,
                }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "vol_target_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
