"""Parameter sweep: is SMA 20/50 and 50/200 actually good, or did we just get lucky with two
picks? Grid-search fast/slow and check whether performance is a broad, robust plateau or a
narrow spike (a narrow spike surrounded by mediocre neighbors is the signature of overfitting).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import sma_crossover

RESULTS_DIR = Path(__file__).resolve().parent / "results"

FAST_RANGE = [5, 10, 15, 20, 30, 40, 50, 60]
SLOW_RANGE = [30, 50, 75, 100, 150, 200, 250, 300]


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")

    sharpe_grid = pd.DataFrame(index=FAST_RANGE, columns=SLOW_RANGE, dtype=float)
    cagr_grid = pd.DataFrame(index=FAST_RANGE, columns=SLOW_RANGE, dtype=float)

    for fast in FAST_RANGE:
        for slow in SLOW_RANGE:
            if fast >= slow:
                continue
            position = sma_crossover(df, fast=fast, slow=slow)
            result = run_backtest(df, position)
            sharpe_grid.loc[fast, slow] = result.metrics["sharpe"]
            cagr_grid.loc[fast, slow] = result.metrics["cagr_pct"]

    sharpe_grid.to_csv(RESULTS_DIR / "sma_sweep_sharpe.csv")
    cagr_grid.to_csv(RESULTS_DIR / "sma_sweep_cagr.csv")

    valid = sharpe_grid.values[~np.isnan(sharpe_grid.values.astype(float))]
    print(f"Tested {len(valid)} (fast, slow) combinations")
    print(f"Sharpe range: {np.nanmin(sharpe_grid.values):.2f} to {np.nanmax(sharpe_grid.values):.2f}")
    print(f"Sharpe > 0.6 (beats buy&hold's 0.64): {(valid > 0.6).sum()} / {len(valid)} combos ({(valid > 0.6).mean()*100:.0f}%)")
    print(f"Sharpe > 0.7: {(valid > 0.7).sum()} / {len(valid)} combos ({(valid > 0.7).mean()*100:.0f}%)")
    print("\nSharpe grid (rows=fast, cols=slow):")
    print(sharpe_grid.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(sharpe_grid.values.astype(float), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1.0)
    ax.set_xticks(range(len(SLOW_RANGE)))
    ax.set_xticklabels(SLOW_RANGE)
    ax.set_yticks(range(len(FAST_RANGE)))
    ax.set_yticklabels(FAST_RANGE)
    ax.set_xlabel("slow SMA period")
    ax.set_ylabel("fast SMA period")
    ax.set_title("Sharpe ratio across SMA crossover parameters (BTC/USDT, 2018-2026)")
    for i, fast in enumerate(FAST_RANGE):
        for j, slow in enumerate(SLOW_RANGE):
            v = sharpe_grid.loc[fast, slow]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Sharpe")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sma_sweep_heatmap.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'sma_sweep_heatmap.png'}")


if __name__ == "__main__":
    main()
