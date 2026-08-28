"""Validate the whipsaw-filter candidates picked from sweep_donchian_filters.py's TRAIN sweep.

A filter only counts as a genuine fix if it:
  1. holds up on the walk-forward TEST split (2023-01-01 onward), not just train
  2. actually reduces the 2022 grinding-bear whipsaw loss (the problem it's meant to fix)
  3. doesn't meaningfully damage the 2020 COVID V-crash result (where unfiltered Donchian
     already does well) or the full walk-forward test-period edge

Candidates below are the train-sweep standouts from results/donchian_filter_sweep_train.csv,
not every combination tested -- picking the single best train number per filter type (plus one
runner-up for the confirmation filter, since it was the clearest train winner) and checking
whether it survives, rather than cherry-picking after the fact.
"""

import json
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import (
    donchian_breakout,
    donchian_breakout_confirmed,
    donchian_breakout_margin_filtered,
    donchian_breakout_vol_filtered,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_DATE = "2023-01-01"
ENTRY, EXIT = 25, 10

BEAR_WINDOWS = {
    "2020 COVID crash": ("2020-02-01", "2020-04-30"),
    "2022 bear (Luna/FTX)": ("2022-01-01", "2022-12-31"),
}

CANDIDATES = {
    "donchian_25_10 (baseline, unfiltered)": lambda df: donchian_breakout(df, entry_lookback=ENTRY, exit_lookback=EXIT),
    "confirm_bars=3": lambda df: donchian_breakout_confirmed(df, entry_lookback=ENTRY, exit_lookback=EXIT, confirm_bars=3),
    "confirm_bars=4": lambda df: donchian_breakout_confirmed(df, entry_lookback=ENTRY, exit_lookback=EXIT, confirm_bars=4),
    "vol_filter lb=126 th=0.3": lambda df: donchian_breakout_vol_filtered(
        df, entry_lookback=ENTRY, exit_lookback=EXIT, vol_rank_lookback=126, vol_rank_threshold=0.3
    ),
    "vol_filter lb=126 th=0.4": lambda df: donchian_breakout_vol_filtered(
        df, entry_lookback=ENTRY, exit_lookback=EXIT, vol_rank_lookback=126, vol_rank_threshold=0.4
    ),
    "margin_filter pct=3.0": lambda df: donchian_breakout_margin_filtered(
        df, entry_lookback=ENTRY, exit_lookback=EXIT, min_breakout_margin_pct=3.0
    ),
}


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2017-01-01")
    train_df = df[(df.index >= "2018-01-01") & (df.index < SPLIT_DATE)]
    test_df = df[df.index >= SPLIT_DATE]

    results = {}
    positions = {name: fn(df) for name, fn in CANDIDATES.items()}

    print(f"=== Walk-forward: train (2018 .. {SPLIT_DATE}) vs test ({SPLIT_DATE} .. 2026) ===")
    print(f"{'candidate':<32} | {'train Sharpe':>12}{'train CAGR%':>13}{'train MDD%':>12} | "
          f"{'test Sharpe':>12}{'test CAGR%':>12}{'test MDD%':>11}{'test trades':>13}")
    for name, pos in positions.items():
        train_m = run_backtest(train_df, pos.reindex(train_df.index)).metrics
        test_m = run_backtest(test_df, pos.reindex(test_df.index)).metrics
        results[name] = {"train": train_m, "test": test_m, "windows": {}}
        print(
            f"{name:<32} | {train_m['sharpe']:>12}{train_m['cagr_pct']:>13}{train_m['max_drawdown_pct']:>12} | "
            f"{test_m['sharpe']:>12}{test_m['cagr_pct']:>12}{test_m['max_drawdown_pct']:>11}{test_m['n_trades']:>13}"
        )

    for window_name, (start, end) in BEAR_WINDOWS.items():
        print(f"\n=== {window_name} ({start} .. {end}) ===")
        window_df = df[(df.index >= start) & (df.index <= end)]
        window_price_change = (window_df["close"].iloc[-1] / window_df["close"].iloc[0] - 1) * 100
        print(f"BTC price change over window: {window_price_change:.1f}%")
        print(f"{'candidate':<32} {'return%':>10} {'max_dd%':>9} {'trades':>7}")
        for name, pos in positions.items():
            pos_in_window = pos.reindex(window_df.index)
            m = run_backtest(window_df, pos_in_window).metrics
            results[name]["windows"][window_name] = m
            print(f"{name:<32} {m['total_return_pct']:>10} {m['max_drawdown_pct']:>9} {m['n_trades']:>7}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "donchian_filter_validation.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {RESULTS_DIR / 'donchian_filter_validation.json'}")


if __name__ == "__main__":
    main()
