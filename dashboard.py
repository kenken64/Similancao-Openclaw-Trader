#!/usr/bin/env python3
"""
Web Dashboard for SimilanCao Trader
Displays bot status, recent trades, and logs in real-time
"""
import os
import json
import logging
import time
import threading
from datetime import datetime

logger = logging.getLogger(__name__)
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from trade_history import TradeHistory
from binance_client import BinanceFuturesClient
from backtest import run_backtest

app = Flask(__name__)
CORS(app)

# Initialize trade history
trade_history = TradeHistory()

# Lazy-initialized Binance client for live queries
_binance_client = None
_cached_balance = 0.0
_balance_last_fetched = 0
_cached_position = None
_position_last_fetched = 0

def _get_binance_client():
    global _binance_client
    if _binance_client is None:
        try:
            _binance_client = BinanceFuturesClient(dry_run=True)
        except Exception:
            return None
    return _binance_client

def _get_live_balance():
    """Fetch live balance from Binance, cached for 30 seconds."""
    global _cached_balance, _balance_last_fetched
    now = time.time()
    if now - _balance_last_fetched < 30:
        return _cached_balance
    client = _get_binance_client()
    if not client:
        return _cached_balance
    try:
        _cached_balance = client.get_live_balance()
        _balance_last_fetched = now
    except Exception:
        pass
    return _cached_balance

def _get_live_position():
    """Fetch live position from Binance, cached for 10 seconds."""
    global _cached_position, _position_last_fetched
    now = time.time()
    if now - _position_last_fetched < 10:
        return _cached_position
    client = _get_binance_client()
    if not client:
        return _cached_position
    try:
        _cached_position = client.get_live_position()
        _position_last_fetched = now
    except Exception:
        pass
    return _cached_position

# Shared state
bot_state = {
    "status": "Starting...",
    "balance": 0.0,
    "position": None,
    "position_pnl": None,
    "last_update": None,
    "logs": [],
    "trades": [],
    "mode": "DRY-RUN"
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """API endpoint for current bot status"""
    state = dict(bot_state)
    live_bal = _get_live_balance()
    if live_bal > 0:
        state['balance'] = live_bal
    pos = _get_live_position()
    if pos:
        state['position'] = f"{pos['side']} {pos['size']} @ {pos['entry_price']}"
        pnl = pos['unrealized_pnl']
        state['position_pnl'] = f"{pnl:+.2f} USDT | {pos['side']} {pos['size']} @ {pos['entry_price']}"
        state['status'] = "Position Opened"
    return jsonify(state)

@app.route('/api/logs')
def get_logs():
    """API endpoint for recent logs"""
    try:
        with open('bot.log', 'r') as f:
            lines = f.readlines()
            recent_logs = lines[-50:]  # Last 50 lines
            return jsonify({"logs": recent_logs})
    except FileNotFoundError:
        return jsonify({"logs": ["Log file not found"]})

@app.route('/api/trades/recent')
def get_recent_trades():
    """API endpoint for recent trades"""
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    trades = trade_history.get_recent_trades(limit=limit, offset=offset)
    total = trade_history.get_trades_count()
    return jsonify({"trades": trades, "total": total})

@app.route('/api/trades/statistics')
def get_statistics():
    """API endpoint for trading statistics"""
    stats = trade_history.get_statistics()
    return jsonify(stats)

@app.route('/api/trades/open')
def get_open_trade():
    """API endpoint for current open trade"""
    trade = trade_history.get_open_trade()
    return jsonify({"trade": trade})

@app.route('/api/advisor/recent')
def get_recent_advisor_decisions():
    """API endpoint for recent OpenClaw advisor decisions"""
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)
    decisions = trade_history.get_recent_advisor_decisions(limit=limit, offset=offset)
    total = trade_history.get_advisor_decisions_count()
    return jsonify({"decisions": decisions, "total": total})

@app.route('/api/advisor/statistics')
def get_advisor_statistics():
    """API endpoint for advisor statistics"""
    stats = trade_history.get_advisor_statistics()
    return jsonify(stats)

@app.route('/backtest')
def backtest_page():
    """Backtest dashboard page"""
    return render_template('backtest.html')

@app.route('/api/backtest/run', methods=['POST'])
def run_backtest_api():
    """Run backtest on historical data and return results."""
    data = request.get_json(silent=True) or {}
    days = min(int(data.get('days', 7)), 10)  # cap at 10 days
    candle_count = days * 96  # 96 fifteen-minute candles per day

    client = _get_binance_client()
    if not client:
        return jsonify({"error": "Binance client unavailable"}), 503

    try:
        df = client.get_klines(limit=candle_count, interval="15m")
    except Exception as e:
        return jsonify({"error": f"Failed to fetch klines: {e}"}), 500

    # Compute low_24h from last 96 candles as fallback
    low_24h = df["low"].tail(96).min()

    result = run_backtest(df, low_24h)

    # Save to SQLite for bot reference
    try:
        trade_history.save_backtest(days, result)
    except Exception as e:
        logger.warning(f"Failed to save backtest: {e}")

    return jsonify(result)

@app.route('/api/backtest/latest')
def get_latest_backtest():
    """API endpoint for most recent backtest results."""
    result = trade_history.get_latest_backtest()
    if not result:
        return jsonify({"error": "No backtest runs found"}), 404
    return jsonify(result)

@app.route('/api/backtest/runs')
def get_backtest_runs():
    """API endpoint for backtest run history."""
    limit = request.args.get('limit', 10, type=int)
    runs = trade_history.get_backtest_runs(limit=limit)
    return jsonify({"runs": runs})

def update_state_from_log():
    """Parse log file to update dashboard state"""
    while True:
        try:
            if os.path.exists('bot.log'):
                with open('bot.log', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        bot_state['last_update'] = datetime.now().isoformat()

                        # Search entire log for last balance update (not just recent lines)
                        for line in reversed(lines):
                            if 'Starting balance' in line:
                                try:
                                    bal = line.split('balance: ')[1].split(' USDT')[0]
                                    bot_state['balance'] = float(bal)
                                    break  # Found it, stop searching
                                except:
                                    pass

                        # Scan for mode (search more lines since errors can push it out)
                        for line in lines[-50:]:
                            if 'LIVE' in line and '🔴' in line:
                                bot_state['mode'] = "LIVE"
                            elif 'DRY-RUN' in line and '🧪' in line:
                                bot_state['mode'] = "DRY-RUN"

                        # Parse recent activity for status updates
                        for line in lines[-20:]:
                            if 'OPENING' in line and ('SHORT' in line or 'LONG' in line):
                                bot_state['status'] = "Position Opened"
                            elif 'closed' in line.lower() and ('position' in line.lower() or 'tp' in line.lower() or 'sl' in line.lower()):
                                bot_state['status'] = "Running - Monitoring Market"
                                bot_state['position'] = None
                                bot_state['position_pnl'] = None
                            elif '📌 Existing position:' in line:
                                try:
                                    parts = line.split('Existing position: ')[1].strip()
                                    bot_state['position'] = parts
                                    bot_state['status'] = "Position Opened"
                                except:
                                    pass
                            elif 'Position PnL:' in line:
                                try:
                                    # e.g. "Position PnL: -5.45 USDT | LONG 0.069 @ 69139.1"
                                    pnl_part = line.split('Position PnL: ')[1].strip()
                                    bot_state['position_pnl'] = pnl_part
                                    # Extract position info after the pipe
                                    if '|' in pnl_part:
                                        pos_info = pnl_part.split('|')[1].strip()
                                        bot_state['position'] = pos_info
                                    bot_state['status'] = "Position Opened"
                                except:
                                    pass
                            elif '📍' in line and 'Price:' in line:
                                # Bot is actively checking market
                                if bot_state['status'] != "Position Opened":
                                    bot_state['status'] = "Running - Monitoring Market"
                                if 'Pos: No' in line:
                                    bot_state['position'] = None
                                    bot_state['position_pnl'] = None
                                elif 'Pos: Yes' in line and not bot_state['position']:
                                    bot_state['status'] = "Position Opened"
                            elif 'HALTING' in line or 'stopped' in line.lower():
                                bot_state['status'] = "Stopped"
        except Exception as e:
            print(f"Error updating state: {e}")

        time.sleep(2)  # Update every 2 seconds

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Start background thread to monitor logs
    monitor_thread = threading.Thread(target=update_state_from_log, daemon=True)
    monitor_thread.start()

    print("=" * 60)
    print("🐆 SimilanCao Trader Dashboard")
    print("   Dashboard URL: http://localhost:8080")
    print("=" * 60)

    app.run(host='0.0.0.0', port=8080, debug=False)
