# Quant Trading — Strategy Research & Backtest Engine

Crypto strategy research, built to be exchange-agnostic (via [ccxt](https://github.com/ccxt/ccxt)) so the same
backtest code works against Binance, Upbit, Bithumb, etc.

**Scope today: research and backtesting only. No live trading, no exchange API keys, no real capital.**
Turning any strategy here into live execution is a separate, deliberate step — see "Going live" below.

## Structure

```
src/
  data.py         # OHLCV fetch + local CSV cache via ccxt (public endpoints, no API key needed)
  backtest.py      # vectorized backtest engine + performance metrics
  strategies.py    # strategy functions: price data in, position signal out
run.py             # entry point — fetch data, run a strategy, print metrics, save an equity curve
results/           # metrics.json + equity_curve.png from the last run (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the baseline backtest

```bash
source .venv/bin/activate
python3 run.py
```

Fetches BTC/USDT daily candles from Binance's public API, runs an SMA crossover strategy, prints
performance metrics, and writes `results/equity_curve.png` + `results/metrics.json`.

## Adding a strategy

Add a function to `src/strategies.py` with the signature `(df: pd.DataFrame, **params) -> pd.Series`
returning a position series (1 = long, 0 = flat) indexed the same as `df`. Wire it up in `run.py`.

## Going live (not built yet, deliberately)

This repo only ever reads public market data and never touches an exchange account. If a strategy
here looks worth trading for real:
1. Paper-trade it first — replay live prices through the same backtest logic without placing real orders.
2. Any real execution requires you to personally hold and configure the exchange API keys, in an
   environment you control — that step is out of scope for what gets built here by default.
