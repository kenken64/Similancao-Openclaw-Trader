"""Backtesting engine — replays historical candles through the strategy."""
import logging
import pandas as pd
from config import Config
from strategy import analyze

logger = logging.getLogger(__name__)


def run_backtest(df: pd.DataFrame, low_24h: float) -> dict:
    """Run backtest over historical candle data.

    Iterates candle-by-candle, calling analyze() on each slice to generate
    signals, then simulates trade entries/exits based on SL/TP hits.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume, open_time).
        low_24h: Fallback 24h low (used only until we have 96 candles for rolling).

    Returns:
        Dict with trades, candles (for chart), and summary statistics.
    """
    trades = []
    open_trade = None
    start_idx = Config.MA_SLOW + 5  # Need enough candles for MA99 + buffer

    for i in range(start_idx, len(df)):
        candle = df.iloc[i]

        # Check if open trade hits SL or TP on this candle
        if open_trade is not None:
            hit = _check_exit(open_trade, candle, i, df)
            if hit:
                trades.append(hit)
                open_trade = None

        # Only look for new signals if no position is open
        if open_trade is None:
            # Rolling 24h low: use last 96 candles (24h of 15m data)
            lookback_start = max(0, i - 95)
            rolling_low = df["low"].iloc[lookback_start:i + 1].min()

            slice_df = df.iloc[:i + 1].copy()
            signal = analyze(slice_df, rolling_low)

            if signal and signal.direction in ("LONG", "SHORT"):
                open_trade = {
                    "entry_idx": i,
                    "entry_time": _candle_time_str(candle),
                    "entry_price": signal.entry_price,
                    "direction": signal.direction,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "confidence": signal.confidence,
                    "reason": signal.reason,
                    "exit_idx": None,
                    "exit_time": None,
                    "exit_price": None,
                    "exit_type": None,
                    "pnl": None,
                }

    # If a trade is still open at end of data, include it as-is
    if open_trade is not None:
        trades.append(open_trade)

    # Build candles list for chart rendering
    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": int(row["open_time"].timestamp()) if hasattr(row["open_time"], "timestamp") else int(row["open_time"]),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
        })

    # Summary statistics
    closed = [t for t in trades if t["exit_type"] is not None]
    wins = [t for t in closed if t["pnl"] and t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] and t["pnl"] <= 0]
    still_open = [t for t in trades if t["exit_type"] is None]
    total_pnl = sum(t["pnl"] for t in closed if t["pnl"])

    summary = {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "open": len(still_open),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(closed), 2) if closed else 0.0,
    }

    return {"trades": trades, "candles": candles, "summary": summary}


def _check_exit(trade: dict, candle, candle_idx: int, df: pd.DataFrame):
    """Check if the current candle triggers SL or TP for an open trade.

    Returns a completed trade dict if hit, or None.
    """
    sl = trade["stop_loss"]
    tp = trade["take_profit"]
    direction = trade["direction"]

    if direction == "LONG":
        sl_hit = candle["low"] <= sl
        tp_hit = candle["high"] >= tp
    else:  # SHORT
        sl_hit = candle["high"] >= sl
        tp_hit = candle["low"] <= tp

    if not sl_hit and not tp_hit:
        return None

    # Determine which was hit (if both, assume SL hit first for conservative estimate)
    if sl_hit and tp_hit:
        exit_type = "SL"
        exit_price = sl
    elif tp_hit:
        exit_type = "TP"
        exit_price = tp
    else:
        exit_type = "SL"
        exit_price = sl

    # PnL calculation
    if direction == "LONG":
        pnl_pct = (exit_price - trade["entry_price"]) / trade["entry_price"]
    else:
        pnl_pct = (trade["entry_price"] - exit_price) / trade["entry_price"]

    pnl = pnl_pct * Config.POSITION_SIZE_USDT * Config.LEVERAGE

    result = dict(trade)
    result["exit_idx"] = candle_idx
    result["exit_time"] = _candle_time_str(candle)
    result["exit_price"] = exit_price
    result["exit_type"] = exit_type
    result["pnl"] = round(pnl, 2)
    return result


def _candle_time_str(candle) -> str:
    """Extract ISO timestamp string from a candle row."""
    t = candle["open_time"]
    if hasattr(t, "isoformat"):
        return t.isoformat()
    return str(t)
