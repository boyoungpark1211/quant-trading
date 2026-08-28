"""Walk-forward validation for bollinger_squeeze_breakout candidates picked from the TRAIN-only
sweep (sweep_bollinger_squeeze.py). Same standard as validate_donchian_upgrade.py: a parameter
pick only earns trust if it holds up on the held-out test split, not just on the period it was
selected from. Also compares directly against buy_and_hold and the donchian_25_10 baseline that
has already been validated in this repo.
"""

import json
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import bollinger_squeeze_breakout, buy_and_hold, donchian_breakout

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPLIT_DATE = "2023-01-01"

# picked from the train-only sweep: the full-train-best pick, a pick from the broad plateau at
# lookback=250 (not just the single best cell), the function's own defaults, and the highest
# train-CAGR pick -- covering different corners of the grid rather than only the single winner.
CANDIDATES = {
    "squeeze_60_0.30 (best train sharpe)": (60, 0.30),
    "squeeze_250_0.15 (broad plateau pick)": (250, 0.15),
    "squeeze_120_0.20 (function defaults)": (120, 0.20),
    "squeeze_40_0.40 (highest train CAGR)": (40, 0.40),
}


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    train_df = df[df.index < SPLIT_DATE]
    test_df = df[df.index >= SPLIT_DATE]

    bh_train = run_backtest(train_df, buy_and_hold(train_df)).metrics
    bh_test = run_backtest(test_df, buy_and_hold(test_df)).metrics
    print(
        f"buy_and_hold              train: sharpe={bh_train['sharpe']:.2f} cagr={bh_train['cagr_pct']}%  "
        f"|  test: sharpe={bh_test['sharpe']:.2f} cagr={bh_test['cagr_pct']}%"
    )

    donchian_pos = donchian_breakout(df, entry_lookback=25, exit_lookback=10)
    dc_train = run_backtest(train_df, donchian_pos.reindex(train_df.index)).metrics
    dc_test = run_backtest(test_df, donchian_pos.reindex(test_df.index)).metrics
    print(
        f"donchian_25_10 (baseline) train: sharpe={dc_train['sharpe']:.2f} cagr={dc_train['cagr_pct']}%  "
        f"|  test: sharpe={dc_test['sharpe']:.2f} cagr={dc_test['cagr_pct']}%\n"
    )

    results = {}
    print(
        f"{'strategy':<38} | {'train Sharpe':>12}{'train CAGR%':>13}{'train MDD%':>12} | "
        f"{'test Sharpe':>12}{'test CAGR%':>12}{'test MDD%':>11}{'test trades':>13}"
    )
    for name, (lookback, pct) in CANDIDATES.items():
        full_position = bollinger_squeeze_breakout(df, squeeze_lookback=lookback, squeeze_percentile=pct)
        train_m = run_backtest(train_df, full_position.reindex(train_df.index)).metrics
        test_m = run_backtest(test_df, full_position.reindex(test_df.index)).metrics
        results[name] = {"train": train_m, "test": test_m}
        print(
            f"{name:<38} | {train_m['sharpe']:>12}{train_m['cagr_pct']:>13}{train_m['max_drawdown_pct']:>12} | "
            f"{test_m['sharpe']:>12}{test_m['cagr_pct']:>12}{test_m['max_drawdown_pct']:>11}{test_m['n_trades']:>13}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bollinger_squeeze_validation.json", "w") as f:
        json.dump({"buy_and_hold": {"train": bh_train, "test": bh_test},
                    "donchian_25_10": {"train": dc_train, "test": dc_test},
                    **results}, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'bollinger_squeeze_validation.json'}")


if __name__ == "__main__":
    main()
