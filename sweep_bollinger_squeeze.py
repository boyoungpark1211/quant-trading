"""Parameter sweep for bollinger_squeeze_breakout, on the TRAIN split only (2018 .. 2023-01-01)
-- same discipline as sweep_donchian.py: pick a candidate on train, then walk-forward validate
it on the held-out test split separately (see validate_bollinger_squeeze.py), never pick on the
full period and call it done.

bb_period/bb_std are left at their textbook-standard values (20, 2.0) -- see the docstring in
src/strategies.py for why -- and only the squeeze-specific parameters (squeeze_lookback,
squeeze_percentile) are grid-searched here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import bollinger_squeeze_breakout, buy_and_hold

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_DATE = "2023-01-01"

LOOKBACK_RANGE = [40, 60, 90, 120, 180, 250]
PERCENTILE_RANGE = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    train_df = df[df.index < SPLIT_DATE]

    bh_train = run_backtest(train_df, buy_and_hold(train_df)).metrics
    print(f"buy_and_hold (train): sharpe={bh_train['sharpe']} cagr%={bh_train['cagr_pct']}\n")

    sharpe_grid = pd.DataFrame(index=LOOKBACK_RANGE, columns=PERCENTILE_RANGE, dtype=float)
    cagr_grid = pd.DataFrame(index=LOOKBACK_RANGE, columns=PERCENTILE_RANGE, dtype=float)
    trades_grid = pd.DataFrame(index=LOOKBACK_RANGE, columns=PERCENTILE_RANGE, dtype=float)

    for lookback in LOOKBACK_RANGE:
        for pct in PERCENTILE_RANGE:
            # compute signal on the FULL series so the rolling windows have proper warm-up,
            # then backtest only the train slice -- same convention as compare_all.py
            position = bollinger_squeeze_breakout(df, squeeze_lookback=lookback, squeeze_percentile=pct)
            result = run_backtest(train_df, position.reindex(train_df.index))
            sharpe_grid.loc[lookback, pct] = result.metrics["sharpe"]
            cagr_grid.loc[lookback, pct] = result.metrics["cagr_pct"]
            trades_grid.loc[lookback, pct] = result.metrics["n_trades"]

    sharpe_grid.to_csv(RESULTS_DIR / "bollinger_squeeze_sweep_sharpe_train.csv")

    valid = sharpe_grid.values[~np.isnan(sharpe_grid.values.astype(float))]
    print(f"Tested {len(valid)} (squeeze_lookback, squeeze_percentile) combinations on train")
    print(f"Sharpe range: {np.nanmin(sharpe_grid.values):.2f} to {np.nanmax(sharpe_grid.values):.2f}")
    print(
        f"Sharpe > buy&hold train ({bh_train['sharpe']}): "
        f"{(valid > bh_train['sharpe']).sum()} / {len(valid)} ({(valid > bh_train['sharpe']).mean()*100:.0f}%)"
    )
    print("\nSharpe grid (rows=squeeze_lookback, cols=squeeze_percentile):")
    print(sharpe_grid.round(2).to_string())
    print("\nCAGR%% grid:")
    print(cagr_grid.round(1).to_string())
    print("\nTrade-count grid:")
    print(trades_grid.round(0).to_string())

    best = sharpe_grid.stack().idxmax()
    print(f"\nBest on train: squeeze_lookback={best[0]}, squeeze_percentile={best[1]}, "
          f"sharpe={sharpe_grid.loc[best]:.2f}, cagr%={cagr_grid.loc[best]:.1f}, "
          f"trades={trades_grid.loc[best]:.0f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(sharpe_grid.values.astype(float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(PERCENTILE_RANGE)))
    ax.set_xticklabels(PERCENTILE_RANGE)
    ax.set_yticks(range(len(LOOKBACK_RANGE)))
    ax.set_yticklabels(LOOKBACK_RANGE)
    ax.set_xlabel("squeeze_percentile")
    ax.set_ylabel("squeeze_lookback (days)")
    ax.set_title("Sharpe: Bollinger squeeze breakout, BTC/USDT train (2018..2023)")
    for i, lookback in enumerate(LOOKBACK_RANGE):
        for j, pct in enumerate(PERCENTILE_RANGE):
            v = sharpe_grid.loc[lookback, pct]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Sharpe")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bollinger_squeeze_sweep_heatmap.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'bollinger_squeeze_sweep_heatmap.png'}")


if __name__ == "__main__":
    main()
