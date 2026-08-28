"""Full statistical validation of the momentum-spike scalp on a full year of 1-minute BTC/KRW
data (~105k candles, ~10x more than the earlier 7-week run). Sweeps on train, then for the top
train configs runs a formal significance test on TEST trades: a one-sample t-test (H0: mean
trade return = 0) and a bootstrap 95% confidence interval on the mean. A config only counts as
validated if BOTH say the edge is real -- not just "the average happened to be positive."
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data import fetch_ohlcv
from src.scalping_backtest import run_scalp_backtest
from src.scalping_signals import momentum_spike_signal

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_TRADES_TRAIN = 15
MIN_TRADES_TEST = 30  # need enough test trades for the significance test to mean anything
TRAIN_FRACTION = 0.7
N_BOOTSTRAP = 5000
TOP_N_TO_VALIDATE = 15

GRID = {
    "roc_lookback_bars": [1, 2, 3, 5],
    "min_move_pct": [0.0015, 0.002, 0.003],
    "volume_mult": [None, 2.0, 3.0],
    "take_profit_pct": [0.003, 0.005],
    "stop_loss_pct": [0.002, 0.003],
    "max_hold_bars": [10, 20, 30],
}


def bootstrap_ci(returns: np.ndarray, n_boot: int = N_BOOTSTRAP, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(returns)
    boot_means = np.array([rng.choice(returns, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return lo, hi


def main() -> None:
    df = fetch_ohlcv("upbit", "BTC/KRW", timeframe="1m", since="2025-08-29")
    split_i = int(len(df) * TRAIN_FRACTION)
    train_df, test_df = df.iloc[:split_i], df.iloc[split_i:]
    print(f"Total: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
    print(f"Train: {len(train_df)} candles ({train_df.index[0]} -> {train_df.index[-1]})")
    print(f"Test:  {len(test_df)} candles ({test_df.index[0]} -> {test_df.index[-1]})\n")

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"Sweeping {len(combos)} combinations on TRAIN...")

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        signal = momentum_spike_signal(
            train_df,
            roc_lookback_bars=params["roc_lookback_bars"],
            min_move_pct=params["min_move_pct"],
            volume_mult=params["volume_mult"],
        )
        result = run_scalp_backtest(
            train_df,
            signal,
            starting_capital=10_000,
            take_profit_pct=params["take_profit_pct"],
            stop_loss_pct=params["stop_loss_pct"],
            max_hold_bars=params["max_hold_bars"],
        )
        m = result.metrics
        if m["n_trades"] < MIN_TRADES_TRAIN:
            continue
        rows.append({**params, **m})

    sweep_df = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    sweep_df.to_csv(RESULTS_DIR / "scalp_spike_1y_sweep_train.csv", index=False)
    n_profitable = (sweep_df["total_return_pct"] > 0).sum()
    print(f"{len(sweep_df)} combos had >= {MIN_TRADES_TRAIN} trades on train.")
    print(f"Profitable on train: {n_profitable} / {len(sweep_df)} ({n_profitable/len(sweep_df)*100:.0f}%)\n")

    print(f"Statistically validating top {TOP_N_TO_VALIDATE} TRAIN configs on TEST (t-test + bootstrap 95% CI)...")
    print(
        f"{'rank':<5}{'roc':<5}{'move%':<7}{'vol':<6}{'tp%':<6}{'sl%':<6}{'hold':<6} | "
        f"{'test_n':>7}{'test_ret%':>10}{'mean/trade%':>12} | {'p-value':>9}{'95% CI (mean%)':>22}{'verdict':>14}"
    )

    validated = []
    for rank, row in enumerate(sweep_df.head(TOP_N_TO_VALIDATE).itertuples(), start=1):
        params = {k: getattr(row, k) for k in keys}
        signal = momentum_spike_signal(
            test_df,
            roc_lookback_bars=params["roc_lookback_bars"],
            min_move_pct=params["min_move_pct"],
            volume_mult=params["volume_mult"],
        )
        test_result = run_scalp_backtest(
            test_df,
            signal,
            starting_capital=10_000,
            take_profit_pct=params["take_profit_pct"],
            stop_loss_pct=params["stop_loss_pct"],
            max_hold_bars=params["max_hold_bars"],
        )
        tm = test_result.metrics
        trades = test_result.trades

        if len(trades) < MIN_TRADES_TEST:
            verdict = f"only {len(trades)} test trades"
            print(
                f"{rank:<5}{params['roc_lookback_bars']:<5}{params['min_move_pct']*100:<7.2f}"
                f"{str(params['volume_mult']):<6}{params['take_profit_pct']*100:<6.1f}"
                f"{params['stop_loss_pct']*100:<6.2f}{params['max_hold_bars']:<6} | "
                f"{len(trades):>7}{'--':>10}{'--':>12} | {'--':>9}{'--':>22}{verdict:>14}"
            )
            continue

        returns = trades["return_pct"].to_numpy()
        t_stat, p_value = stats.ttest_1samp(returns, 0.0)
        ci_lo, ci_hi = bootstrap_ci(returns)
        significant_positive = p_value < 0.05 and ci_lo > 0
        verdict = "SIGNIFICANT+" if significant_positive else ("not sig." if p_value >= 0.05 else "sig. NEGATIVE")

        print(
            f"{rank:<5}{params['roc_lookback_bars']:<5}{params['min_move_pct']*100:<7.2f}"
            f"{str(params['volume_mult']):<6}{params['take_profit_pct']*100:<6.1f}"
            f"{params['stop_loss_pct']*100:<6.2f}{params['max_hold_bars']:<6} | "
            f"{len(trades):>7}{tm['total_return_pct']:>10}{returns.mean():>12.4f} | "
            f"{p_value:>9.4f}{f'[{ci_lo:.4f}, {ci_hi:.4f}]':>22}{verdict:>14}"
        )
        validated.append(
            {
                "rank": rank,
                **params,
                "test_n_trades": len(trades),
                "test_total_return_pct": tm["total_return_pct"],
                "test_mean_return_pct": float(returns.mean()),
                "p_value": float(p_value),
                "ci_95_lo": float(ci_lo),
                "ci_95_hi": float(ci_hi),
                "verdict": verdict,
            }
        )

    with open(RESULTS_DIR / "scalp_spike_1y_statistical_validation.json", "w") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)

    n_significant = sum(1 for v in validated if v["verdict"] == "SIGNIFICANT+")
    print(f"\n{n_significant} / {len(validated)} tested configs show a statistically significant positive edge (p<0.05 AND 95% CI excludes zero).")
    print("Saved results/scalp_spike_1y_sweep_train.csv and results/scalp_spike_1y_statistical_validation.json")


if __name__ == "__main__":
    main()
