"""Sweep parameters for the three Donchian whipsaw filters (vol-regime, breakout-margin,
confirmation-bar) on top of the validated donchian_25_10 base rule. Same discipline as
sweep_donchian.py / sweep_sma.py: search on the TRAIN split only, then validate_donchian_filters.py
checks the winners against held-out test data and the 2022/2020 bear windows before anything
gets called an improvement.

Search objective is train-period Sharpe on the FULL train window (2018 .. 2023-01-01), not just
the 2022 sub-window -- optimizing directly on the exact window we're trying to fix would be
textbook overfitting to the one bad year, the same mistake this project has flagged before.
"""

import pandas as pd

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import (
    donchian_breakout,
    donchian_breakout_confirmed,
    donchian_breakout_margin_filtered,
    donchian_breakout_vol_filtered,
)

SPLIT_DATE = "2023-01-01"
ENTRY, EXIT = 25, 10  # the validated base pick (see validate_donchian_upgrade.py)

VOL_RANK_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
VOL_RANK_LOOKBACKS = [126, 252]
MARGIN_PCTS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
CONFIRM_BARS = [2, 3, 4, 5]


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    train_df = df[df.index < SPLIT_DATE]

    base_position = donchian_breakout(df, entry_lookback=ENTRY, exit_lookback=EXIT)
    base_train = run_backtest(train_df, base_position.reindex(train_df.index)).metrics
    print(f"base donchian_{ENTRY}_{EXIT}  train Sharpe={base_train['sharpe']}  "
          f"CAGR={base_train['cagr_pct']}%  MDD={base_train['max_drawdown_pct']}%  trades={base_train['n_trades']}\n")

    rows = []

    print("=== vol-regime filter ===")
    for lb in VOL_RANK_LOOKBACKS:
        for th in VOL_RANK_THRESHOLDS:
            pos = donchian_breakout_vol_filtered(
                df, entry_lookback=ENTRY, exit_lookback=EXIT, vol_rank_lookback=lb, vol_rank_threshold=th
            )
            m = run_backtest(train_df, pos.reindex(train_df.index)).metrics
            rows.append({"filter": "vol", "lookback": lb, "threshold": th, **m})
            print(f"  lookback={lb:>4} threshold={th:.1f}  sharpe={m['sharpe']:>6} cagr={m['cagr_pct']:>7}% "
                  f"mdd={m['max_drawdown_pct']:>7}% trades={m['n_trades']:>4}")

    print("\n=== breakout-margin filter ===")
    for pct in MARGIN_PCTS:
        pos = donchian_breakout_margin_filtered(df, entry_lookback=ENTRY, exit_lookback=EXIT, min_breakout_margin_pct=pct)
        m = run_backtest(train_df, pos.reindex(train_df.index)).metrics
        rows.append({"filter": "margin", "min_pct": pct, **m})
        print(f"  min_margin_pct={pct:.2f}  sharpe={m['sharpe']:>6} cagr={m['cagr_pct']:>7}% "
              f"mdd={m['max_drawdown_pct']:>7}% trades={m['n_trades']:>4}")

    print("\n=== confirmation-bar filter ===")
    for bars in CONFIRM_BARS:
        pos = donchian_breakout_confirmed(df, entry_lookback=ENTRY, exit_lookback=EXIT, confirm_bars=bars)
        m = run_backtest(train_df, pos.reindex(train_df.index)).metrics
        rows.append({"filter": "confirm", "confirm_bars": bars, **m})
        print(f"  confirm_bars={bars}  sharpe={m['sharpe']:>6} cagr={m['cagr_pct']:>7}% "
              f"mdd={m['max_drawdown_pct']:>7}% trades={m['n_trades']:>4}")

    out = pd.DataFrame(rows)
    out.to_csv("results/donchian_filter_sweep_train.csv", index=False)
    print("\nSaved results/donchian_filter_sweep_train.csv")


if __name__ == "__main__":
    main()
