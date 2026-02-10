# 🐆 SimiLancao OpenClaw Trader

AI-powered MA crossover strategy bot for BTC/USDT perpetual futures on Binance with OpenClaw AI advisor integration and real-time web dashboard.

Based on successful manual trades from Feb 10, 2026.

## ✨ Features

- 🤖 **OpenClaw AI Advisor** - Every trade validated by AI before execution
- 📊 **Real-time Web Dashboard** - Monitor trades, PnL, and statistics on port 8080
- 💾 **SQLite Trade History** - Automatic tracking of all trades with PnL calculation
- 📱 **WhatsApp Alerts** - Real-time notifications via OpenClaw gateway
- 🎯 **MA Crossover Strategy** - Proven moving average crossover signals
- 🛡️ **Risk Management** - Automatic SL/TP, position sizing, drawdown protection
- 🧪 **Dry-run Mode** - Safe paper trading for testing strategies

## 📈 Strategy

### SHORT Signals
- Price bounces to MA25/MA99 resistance
- Bearish MA alignment (MA7 < MA25 < MA99)
- Sufficient volume confirmation
- Favorable funding rate

### LONG Signals
- Price dips to 24h low support
- Bounce candle with volume spike
- Bullish MA alignment
- Support level confirmation

**Parameters:**
- **Timeframe**: 15m candles, checked every 15 seconds
- **Risk/Reward**: Minimum 1:2
- **MA Periods**: 7, 25, 99
- **Leverage**: 25x (configurable)

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/kenken64/Similancao-Openclaw-Trader.git
cd Similancao-Openclaw-Trader

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required settings:**
```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here
SYMBOL=BTCUSDT
POSITION_SIZE_USDT=1000
DRY_RUN=true
```

**Important:**
- Whitelist your IP in Binance API settings
- Enable "Reading" and "Futures" permissions
- Never enable withdrawals

### 3. Run the Bot

```bash
# Dry-run mode (paper trading - safe!)
./run_with_dashboard.sh

# Live trading (real orders!)
./run_with_dashboard.sh --live
```

**Dashboard:** http://localhost:8080

## 🧠 OpenClaw AI Advisor

The bot consults OpenClaw AI before executing any trade:

### Flow
1. Bot detects MA crossover signal
2. Sends trade analysis request to OpenClaw: `POST http://localhost:4778/api/trade/analyze`
3. OpenClaw AI analyzes market conditions
4. Returns approval/rejection with reasoning
5. Bot executes only if approved

### API Integration

**Endpoint:** `POST /api/trade/analyze`

**Request:**
```json
{
  "symbol": "BTCUSDT",
  "direction": "SHORT",
  "entry_price": 69500.0,
  "stop_loss": 69700.0,
  "take_profit": 68900.0,
  "confidence": 0.75,
  "reason": "MA resistance bounce in bearish alignment",
  "current_price": 69487.4,
  "funding_rate": -0.0001
}
```

**Expected Response:**
```json
{
  "approve": true,
  "reason": "Strong bearish setup with MA alignment"
}
```

**Configuration:**
```env
ADVISOR_MODE=true
ADVISOR_TIMEOUT_SECONDS=120
OPENCLAW_GATEWAY_URL=http://localhost:4778
```

## 📊 Dashboard

Access the real-time dashboard at **http://localhost:8080**

**Features:**
- Live bot status and balance
- Current open position
- Trading statistics (win rate, total PnL, avg PnL)
- Recent 10 trades with entry/exit prices
- Real-time activity logs
- Auto-refresh every 2 seconds

## 💾 Trade History

### View Trades (CLI)

```bash
# Recent 20 trades
python view_trades.py

# All trades
python view_trades.py --all

# Trading statistics
python view_trades.py --stats

# Current open trade
python view_trades.py --open
```

### API Endpoints

- `GET /api/trades/recent?limit=20` - Recent trades
- `GET /api/trades/statistics` - Win rate, PnL stats
- `GET /api/trades/open` - Current open trade

### Database

Trades are stored in `trades.db` (SQLite) with:
- Entry/exit prices and timestamps
- PnL calculation (USDT and percentage)
- Trade direction, confidence, reason
- Advisor approval status
- Funding rate at entry

**Note:** Database file is excluded from git commits (contains sensitive data)

## 📁 Project Structure

```
similancao-trader/
├── bot.py                    # Main trading bot with AI advisor
├── dashboard.py              # Web dashboard (port 8080)
├── strategy.py               # MA crossover signal detection
├── risk_manager.py          # Position sizing, SL/TP, drawdown
├── binance_client.py        # Binance Futures API wrapper
├── config.py                # Configuration loader
├── trade_history.py         # SQLite trade tracking
├── view_trades.py           # CLI tool for viewing trades
├── run_with_dashboard.sh    # Startup script
├── templates/
│   └── dashboard.html       # Dashboard UI
├── .env                     # Configuration (not in git)
├── .env.example             # Configuration template
├── trades.db                # Trade history (not in git)
└── requirements.txt         # Python dependencies
```

## ⚙️ Configuration (.env)

### Trading Settings
```env
SYMBOL=BTCUSDT
TIMEFRAME=15m
LEVERAGE=25
POSITION_SIZE_USDT=1000
```

### Moving Averages
```env
MA_FAST=7
MA_MID=25
MA_SLOW=99
```

### Risk Management
```env
MAX_DRAWDOWN_PCT=20
RISK_REWARD_RATIO=2.0
STOP_LOSS_BUFFER_PCT=0.15
VOLUME_FILTER_RATIO=0.8
```

### Bot Settings
```env
CHECK_INTERVAL_SECONDS=15
DRY_RUN=true
LOG_LEVEL=INFO
```

### OpenClaw Integration
```env
OPENCLAW_GATEWAY_URL=http://localhost:4778
OPENCLAW_ALERT_ENABLED=true
ADVISOR_MODE=true
ADVISOR_TIMEOUT_SECONDS=120
```

## 🛡️ Safety Features

- **Dry-run Mode**: Test strategies with simulated trades
- **Drawdown Protection**: Auto-halt if losses exceed 20%
- **AI Approval**: OpenClaw validates every trade
- **Stop-Loss/Take-Profit**: Automatic risk management
- **Volume Filtering**: Avoid low-liquidity trades
- **Funding Rate Check**: Skip trades with unfavorable funding
- **Position Limit**: One position at a time

## 📱 Alerts

Sends WhatsApp notifications via OpenClaw for:
- Trade entries (with entry, SL, TP prices)
- Trade exits (TP/SL hits)
- AI approval/rejection decisions
- Bot start/stop events
- Error alerts

## 🧪 Testing

```bash
# Run in dry-run mode first
./run_with_dashboard.sh

# Monitor the dashboard
open http://localhost:8080

# Check logs
tail -f bot.log

# View trade history
python view_trades.py --stats
```

## ⚠️ Important Notes

### Security
- ✅ Never commit `.env` or `trades.db` to git
- ✅ Whitelist your IP in Binance API settings
- ✅ Use API keys with minimal permissions (Reading + Futures only)
- ✅ Never enable withdrawal permissions
- ✅ Rotate API keys if exposed

### Risk Warning
- 🚨 Trading crypto futures is highly risky
- 🚨 Only trade with money you can afford to lose
- 🚨 Test thoroughly in dry-run mode first
- 🚨 Start with small position sizes
- 🚨 Monitor the bot regularly

### Binance API Setup
1. Create API key at Binance.com → Profile → API Management
2. Enable "Reading" and "Enable Futures"
3. Whitelist your IP address (get with `curl ifconfig.me`)
4. Never enable "Enable Withdrawals"
5. Set "Futures" as allowed API

## 📊 Performance Tracking

View detailed statistics:
```bash
python view_trades.py --stats
```

Output:
```
📊 TRADING STATISTICS
============================================================
Total Trades:     42
Winning Trades:   28 (66.7%)
Losing Trades:    14
Total PnL:        +347.50 USDT
Average PnL:      +8.27 USDT
Best Trade:       +45.20 USDT
Worst Trade:      -23.10 USDT
============================================================
```

## 🔧 Troubleshooting

### Bot won't start
```bash
# Check if ports are available
lsof -i :8080

# Check API connection
source venv/bin/activate
python -c "from binance.client import Client; print('OK')"
```

### API errors
- Verify IP is whitelisted in Binance
- Check API key has correct permissions
- Ensure API key is not expired or restricted

### OpenClaw not responding
- Check OpenClaw is running on port 4778
- Verify `/api/trade/analyze` endpoint is implemented
- Set `ADVISOR_MODE=false` to disable (not recommended)

## 🤝 Contributing

This is a personal trading bot. Use at your own risk.

## 📄 License

Private project - All rights reserved.

## 🙏 Credits

Built with:
- [python-binance](https://github.com/sammchardy/python-binance) - Binance API wrapper
- [Flask](https://flask.palletsprojects.com/) - Web dashboard
- OpenClaw - AI advisor integration
- Claude Opus 4.6 - Development assistance

---

**⚠️ Disclaimer:** This bot is for educational purposes. Cryptocurrency trading carries significant risk. The authors are not responsible for any financial losses incurred through the use of this software.
