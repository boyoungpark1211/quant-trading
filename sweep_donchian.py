"""Parameter sweep for Donchian breakout -- this was only ever tested at 20/10. Before trusting
it further, check whether that pick is a broad robust plateau (like SMA's sweep showed) or a
lucky spike (the failure mode we've now seen twice with the scalping hypotheses).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import donchian_breakout

RESULTS_DIR = Path(__file__).resolve().parent / "results"

ENTRY_RANGE = [10, 15, 20, 25, 30, 40, 55, 70]
EXIT_RANGE = [5, 10, 15, 20, 30, 40]


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")

    sharpe_grid = pd.DataFrame(index=ENTRY_RANGE, columns=EXIT_RANGE, dtype=float)
    cagr_grid = pd.DataFrame(index=ENTRY_RANGE, columns=EXIT_RANGE, dtype=float)
    dd_grid = pd.DataFrame(index=ENTRY_RANGE, columns=EXIT_RANGE, dtype=float)

    for entry in ENTRY_RANGE:
        for exit_ in EXIT_RANGE:
            if exit_ >= entry:
                continue
            position = donchian_breakout(df, entry_lookback=entry, exit_lookback=exit_)
            result = run_backtest(df, position)
            sharpe_grid.loc[entry, exit_] = result.metrics["sharpe"]
            cagr_grid.loc[entry, exit_] = result.metrics["cagr_pct"]
            dd_grid.loc[entry, exit_] = result.metrics["max_drawdown_pct"]

    sharpe_grid.to_csv(RESULTS_DIR / "donchian_sweep_sharpe.csv")

    valid = sharpe_grid.values[~np.isnan(sharpe_grid.values.astype(float))]
    print(f"Tested {len(valid)} (entry, exit) combinations")
    print(f"Sharpe range: {np.nanmin(sharpe_grid.values):.2f} to {np.nanmax(sharpe_grid.values):.2f}")
    print(f"Sharpe > 0.64 (beats buy&hold): {(valid > 0.64).sum()} / {len(valid)} ({(valid > 0.64).mean()*100:.0f}%)")
    print(f"Sharpe > 0.79 (beats original 20/10 pick): {(valid > 0.79).sum()} / {len(valid)} ({(valid > 0.79).mean()*100:.0f}%)")
    print("\nSharpe grid (rows=entry_lookback, cols=exit_lookback):")
    print(sharpe_grid.round(2).to_string())
    print("\nMax drawdown %% grid:")
    print(dd_grid.round(1).to_string())

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sharpe_grid.values.astype(float), cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=1.2)
    ax.set_xticks(range(len(EXIT_RANGE)))
    ax.set_xticklabels(EXIT_RANGE)
    ax.set_yticks(range(len(ENTRY_RANGE)))
    ax.set_yticklabels(ENTRY_RANGE)
    ax.set_xlabel("exit lookback (days)")
    ax.set_ylabel("entry lookback (days)")
    ax.set_title("Sharpe ratio across Donchian breakout parameters (BTC/USDT, 2018-2026)")
    for i, entry in enumerate(ENTRY_RANGE):
        for j, exit_ in enumerate(EXIT_RANGE):
            v = sharpe_grid.loc[entry, exit_]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Sharpe")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "donchian_sweep_heatmap.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'donchian_sweep_heatmap.png'}")


if __name__ == "__main__":
    main()
