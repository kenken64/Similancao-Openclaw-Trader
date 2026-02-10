# 🐆 SimilanCao Trader

MA Crossover strategy bot for BTC/USDT perpetual futures on Binance.  
Based on successful manual trades from Feb 10, 2026.

## Strategy

- **SHORT**: Price bounces up to MA25/MA99 resistance in bearish alignment (MA7 < MA25 < MA99)
- **LONG**: Price dips to 24h low support + bounce candle + volume spike
- **Timeframe**: 15m candles, checked every 15 seconds
- **Risk/Reward**: Minimum 1:2

## Setup

```bash
cd /Users/kennethphang/projects/similancao-trader
cp .env.example .env
# Edit .env with your Binance API keys and settings
pip install -r requirements.txt
```

## Usage

```bash
# Paper trading (default, safe)
python bot.py

# Live trading (real orders!)
python bot.py --live
```

## Files

| File | Description |
|---|---|
| `bot.py` | Main loop — fetches data, evaluates signals, places orders |
| `strategy.py` | MA crossover signal detection logic |
| `risk_manager.py` | Position sizing, SL/TP, drawdown tracking |
| `binance_client.py` | Binance Futures API wrapper |
| `config.py` | Loads settings from `.env` |

## Config (.env)

Key settings:
- `LEVERAGE` — Default 25x
- `POSITION_SIZE_USDT` — Notional size per trade (default 5000)
- `MAX_DRAWDOWN_PCT` — Stop bot if drawdown exceeds this (default 20%)
- `DRY_RUN` — `true` for paper trading, `false` for live
- `OPENCLAW_ALERT_ENABLED` — WhatsApp alerts via OpenClaw

## Alerts

Sends trade alerts to WhatsApp via OpenClaw gateway at `localhost:4778`.
