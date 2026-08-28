"""Parameter sweep + train/test split for the scalping strategy. The single RSI(2)/0.3%/0.2%
config already showed a loss with real trades -- before trying anything "stronger," check
whether ANY parameter region in this family has genuine edge, and whether it survives on data
the sweep never saw. Grid search alone, without the test-set check, is how you'd fool yourself
into shipping an overfit config.
"""

import itertools
import json
from pathlib import Path

import pandas as pd

from src.data import fetch_ohlcv
from src.scalping_backtest import run_scalp_backtest

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_TRADES = 15
TRAIN_FRACTION = 0.7

GRID = {
    "rsi_entry": [10, 15, 20],
    "take_profit_pct": [0.002, 0.004],
    "stop_loss_pct": [0.0015, 0.003],
    "max_hold_bars": [20, 45],
    "trend_ema_period": [None, 120],
}


def main() -> None:
    df = fetch_ohlcv("upbit", "BTC/KRW", timeframe="1m", since="2026-07-10")
    split_i = int(len(df) * TRAIN_FRACTION)
    train_df, test_df = df.iloc[:split_i], df.iloc[split_i:]
    print(f"Total: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
    print(f"Train: {len(train_df)} candles ({train_df.index[0]} -> {train_df.index[-1]})")
    print(f"Test:  {len(test_df)} candles ({test_df.index[0]} -> {test_df.index[-1]})\n")

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"Sweeping {len(combos)} parameter combinations on TRAIN only...")

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        result = run_scalp_backtest(train_df, starting_capital=10_000, **params)
        m = result.metrics
        if m["n_trades"] < MIN_TRADES:
            continue
        rows.append({**params, **m})

    sweep_df = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    sweep_df.to_csv(RESULTS_DIR / "scalp_sweep_train.csv", index=False)

    n_profitable = (sweep_df["total_return_pct"] > 0).sum()
    print(f"\n{len(sweep_df)} combos had >= {MIN_TRADES} trades on train.")
    print(f"Profitable on train: {n_profitable} / {len(sweep_df)} ({n_profitable/len(sweep_df)*100:.0f}%)")
    print("\nTop 5 on TRAIN:")
    print(sweep_df.head(5).to_string(index=False))

    print("\n\nValidating top 5 TRAIN configs on unseen TEST data:")
    print(f"{'rank':<5}{'rsi':<5}{'tp%':<6}{'sl%':<6}{'hold':<6}{'trend':<7} | "
          f"{'train_ret%':>11}{'train_n':>9} | {'test_ret%':>10}{'test_n':>9}{'test_win%':>11}")
    validation_rows = []
    for rank, row in enumerate(sweep_df.head(5).itertuples(), start=1):
        params = {k: getattr(row, k) for k in keys}
        test_result = run_scalp_backtest(test_df, starting_capital=10_000, **params)
        tm = test_result.metrics
        print(
            f"{rank:<5}{params['rsi_entry']:<5}{params['take_profit_pct']*100:<6.1f}"
            f"{params['stop_loss_pct']*100:<6.2f}{params['max_hold_bars']:<6}"
            f"{str(params['trend_ema_period']):<7} | "
            f"{row.total_return_pct:>11}{row.n_trades:>9} | "
            f"{tm['total_return_pct']:>10}{tm['n_trades']:>9}{tm['win_rate_pct']:>11}"
        )
        validation_rows.append({"rank": rank, **params, "train_metrics": row._asdict(), "test_metrics": tm})

    with open(RESULTS_DIR / "scalp_sweep_validation.json", "w") as f:
        json.dump(validation_rows, f, indent=2, default=str)
    print(f"\nSaved results/scalp_sweep_train.csv and results/scalp_sweep_validation.json")


if __name__ == "__main__":
    main()
