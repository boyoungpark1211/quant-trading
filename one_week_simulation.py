"""₩1,000,000 starting capital, 10 different 1-week windows, using the actual validated model
(donchian_breakout_confirmed 25/10, confirm_bars=3) and its real historical position at each
date -- not a hypothetical, the strategy's real signal on real BTC/USDT price action.

Start dates are picked evenly spaced across the full available history (not hand-picked), so
this isn't cherry-picking favorable or unfavorable weeks.

EXECUTION_LAG=0: same-day execution (captures the signal-confirming day's own return too) --
see run_backtest's docstring for why this is a best-case comparison, not a live-tradeable rule.
Set to 1 for the realistic T+1 convention used everywhere else in this repo.
"""

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.strategies import donchian_breakout_confirmed

STARTING_CAPITAL_KRW = 1_000_000
N_WINDOWS = 10
WINDOW_DAYS = 7
EXECUTION_LAG = 0


def main() -> None:
    df = fetch_ohlcv("binance", "BTC/USDT", timeframe="1d", since="2018-01-01")
    position = donchian_breakout_confirmed(df, entry_lookback=25, exit_lookback=10, confirm_bars=3)

    warmup = 30  # entry_lookback(25) + confirm_bars(3) + a little buffer
    valid_start_range = (warmup, len(df) - WINDOW_DAYS - 1)
    start_indices = np.linspace(valid_start_range[0], valid_start_range[1], N_WINDOWS, dtype=int)

    print(f"Model: donchian_breakout_confirmed(entry=25, exit=10, confirm_bars=3) on BTC/USDT")
    print(f"Starting capital: {STARTING_CAPITAL_KRW:,}원, 10 evenly-spaced 1-week windows\n")
    print(f"{'#':<3}{'start date':<12}{'end date':<12}{'in market?':<12}{'return%':>9}{'final krw':>14}")

    rows = []
    for i, start_idx in enumerate(start_indices, start=1):
        window_df = df.iloc[start_idx - 1 : start_idx + WINDOW_DAYS]
        window_position = position.reindex(window_df.index)
        result = run_backtest(window_df, window_position, execution_lag=EXECUTION_LAG)

        week_return = result.equity_curve.iloc[-1] - 1
        final_krw = STARTING_CAPITAL_KRW * result.equity_curve.iloc[-1]
        was_in_market = bool(window_position.iloc[1])  # position on the week's first real day

        start_date = window_df.index[1].date()  # first real day of the week (context day excluded)
        end_date = window_df.index[-1].date()

        print(
            f"{i:<3}{str(start_date):<12}{str(end_date):<12}{('long' if was_in_market else 'flat'):<12}"
            f"{week_return*100:>9.2f}{final_krw:>14,.0f}"
        )
        rows.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "was_in_market": was_in_market,
                "return_pct": week_return * 100,
                "final_krw": final_krw,
            }
        )

    returns = pd.Series([r["return_pct"] for r in rows])
    print(f"\nAcross 10 weeks: mean {returns.mean():+.2f}%, median {returns.median():+.2f}%, "
          f"best {returns.max():+.2f}%, worst {returns.min():+.2f}%")
    print(f"Weeks with the strategy actually in the market: {sum(r['was_in_market'] for r in rows)}/10")
    print(f"Weeks flat (out of market, 0% by construction): {sum(not r['was_in_market'] and r['return_pct']==0 for r in rows)}/10")


if __name__ == "__main__":
    main()
