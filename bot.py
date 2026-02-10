#!/usr/bin/env python3
"""
SimilanCao Trader — MA Crossover Strategy Bot
Based on successful BTC/USDT futures trades from Feb 10, 2026.

Usage:
    python bot.py              # dry-run mode (default)
    python bot.py --live       # live trading (real orders)
    python bot.py --dry-run    # explicit dry-run
"""
import os
import sys
import time
import logging
import argparse
import requests
from datetime import datetime
from config import Config
from binance_client import BinanceFuturesClient
from strategy import analyze
from risk_manager import RiskManager

# === Logging Setup ===
log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler("bot.log", encoding="utf-8"),
]
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=log_fmt, handlers=handlers)
logger = logging.getLogger("similancao")


def send_alert(message: str):
    """Send WhatsApp alert via OpenClaw gateway."""
    if not Config.OPENCLAW_ALERT_ENABLED:
        return
    try:
        requests.post(
            f"{Config.OPENCLAW_GATEWAY_URL}/api/message/send",
            json={"message": message},
            timeout=5,
        )
    except Exception:
        pass  # Non-critical


def ask_advisor(signal, price, funding_rate) -> bool:
    """Ask OpenClaw AI to analyze trade before execution.

    Sends trade details to OpenClaw API for AI analysis.
    Returns True if approved, False if rejected or error.
    """
    if not Config.ADVISOR_MODE:
        return True  # Skip advisor, auto-approve

    logger.info(f"🧠 Asking OpenClaw AI for trade analysis...")

    try:
        # Prepare trade data for OpenClaw analysis
        trade_data = {
            "symbol": Config.SYMBOL,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "current_price": price,
            "funding_rate": funding_rate,
            "timestamp": datetime.now().isoformat()
        }

        # Send to OpenClaw for AI analysis
        response = requests.post(
            f"{Config.OPENCLAW_GATEWAY_URL}/api/trade/analyze",
            json=trade_data,
            timeout=Config.ADVISOR_TIMEOUT_SECONDS
        )

        if response.status_code == 200:
            result = response.json()
            approved = result.get("approve", False)
            reason = result.get("reason", "No reason provided")

            if approved:
                logger.info(f"✅ OpenClaw APPROVED: {reason}")
                # Send notification
                send_alert(
                    f"✅ Trade APPROVED by OpenClaw AI\n"
                    f"{'🔴' if signal.direction == 'SHORT' else '🟢'} {signal.direction} {Config.SYMBOL} @ {signal.entry_price:.1f}\n"
                    f"Reason: {reason}"
                )
                return True
            else:
                logger.info(f"❌ OpenClaw REJECTED: {reason}")
                # Send notification
                send_alert(
                    f"❌ Trade REJECTED by OpenClaw AI\n"
                    f"Signal: {signal.direction} @ {signal.entry_price:.1f}\n"
                    f"Reason: {reason}"
                )
                return False
        else:
            logger.warning(f"OpenClaw API error: HTTP {response.status_code}")
            # Fail-safe: reject on API error
            send_alert(f"⚠️ OpenClaw API error (HTTP {response.status_code}) - trade skipped")
            return False

    except requests.Timeout:
        logger.warning("⏰ OpenClaw API timeout — trade SKIPPED")
        send_alert("⏰ OpenClaw timeout - trade skipped for safety")
        return False
    except Exception as e:
        logger.error(f"OpenClaw advisor error: {e}")
        send_alert(f"❌ OpenClaw error: {e} - trade skipped")
        return False  # Fail-safe: don't trade if advisor is broken


def main():
    parser = argparse.ArgumentParser(description="SimilanCao Trader")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Paper trading (default)")
    args = parser.parse_args()

    dry_run = not args.live
    mode = "🧪 DRY-RUN" if dry_run else "🔴 LIVE"

    logger.info("=" * 60)
    logger.info(f"🐆 SimilanCao Trader — {mode}")
    logger.info(f"   Symbol: {Config.SYMBOL} | TF: {Config.TIMEFRAME} | Leverage: {Config.LEVERAGE}x")
    logger.info(f"   Position size: {Config.POSITION_SIZE_USDT} USDT | RR: 1:{Config.RISK_REWARD_RATIO}")
    logger.info(f"   MA periods: {Config.MA_FAST}/{Config.MA_MID}/{Config.MA_SLOW}")
    logger.info("=" * 60)

    if not dry_run:
        send_alert(f"🐆 SimilanCao Trader LIVE — {Config.SYMBOL} {Config.LEVERAGE}x")

    client = BinanceFuturesClient(dry_run=dry_run)
    balance = client.get_account_balance()
    risk = RiskManager(initial_balance=balance)
    logger.info(f"💰 Starting balance: {balance:.2f} USDT")

    # Check if there's already an open position (live mode)
    if not dry_run:
        pos = client.get_position()
        if pos:
            risk.has_open_position = True
            logger.info(f"📌 Existing position: {pos['side']} {pos['size']} @ {pos['entry_price']}")

    iteration = 0
    while True:
        try:
            iteration += 1
            now = datetime.now().strftime("%H:%M:%S")

            # Fetch data
            df = client.get_klines(limit=150)
            ticker = client.get_24h_ticker()
            mark_info = client.get_mark_price()
            price = df.iloc[-1]["close"]
            low_24h = float(ticker["lowPrice"])
            funding_rate = float(mark_info.get("lastFundingRate", 0))

            if iteration % 20 == 1:  # Log status every ~5 min
                logger.info(
                    f"📍 [{now}] Price: {price:.1f} | 24hLow: {low_24h:.1f} | "
                    f"Funding: {funding_rate:.4f} | Pos: {'Yes' if risk.has_open_position else 'No'}"
                )

            # Dry-run: check if simulated position hit SL/TP
            if dry_run and risk.has_open_position:
                exit_type = risk.check_dry_run_exit(price)
                if exit_type:
                    pos = risk.dry_run_position
                    if exit_type == "TP":
                        pnl = abs(pos["tp"] - pos["entry"]) * pos["qty"]
                        logger.info(f"🎯 [DRY-RUN] Take-profit HIT! {pos['direction']} closed @ {price:.1f} | PnL: +{pnl:.2f} USDT")
                        send_alert(f"🎯 TP hit! {pos['direction']} closed +{pnl:.2f} USDT")
                    else:
                        pnl = abs(pos["sl"] - pos["entry"]) * pos["qty"]
                        logger.info(f"🛑 [DRY-RUN] Stop-loss HIT! {pos['direction']} closed @ {price:.1f} | PnL: -{pnl:.2f} USDT")
                        send_alert(f"🛑 SL hit! {pos['direction']} closed -{pnl:.2f} USDT")
                    risk.set_position_closed()

            # Live: check if position was closed externally
            if not dry_run and risk.has_open_position:
                pos = client.get_position()
                if not pos:
                    logger.info("📭 Position closed (externally or by SL/TP)")
                    risk.set_position_closed()

            # Look for new signals if no position
            if risk.can_open_position():
                signal = analyze(df, low_24h)

                if signal and signal.confidence >= 0.5:
                    # Funding rate check
                    if not risk.check_funding_rate(funding_rate, signal.direction):
                        time.sleep(Config.CHECK_INTERVAL)
                        continue

                    qty = risk.calculate_quantity(price)

                    logger.info(f"{'🔴' if signal.direction == 'SHORT' else '🟢'} SIGNAL: {signal.direction}")
                    logger.info(f"   Reason: {signal.reason}")
                    logger.info(f"   Entry: {signal.entry_price:.1f} | SL: {signal.stop_loss:.1f} | TP: {signal.take_profit:.1f}")
                    logger.info(f"   Qty: {qty} | Confidence: {signal.confidence:.0%}")

                    # Ask advisor (Similancao) for approval
                    if not ask_advisor(signal, price, funding_rate):
                        logger.info("⏭️ Trade skipped by advisor")
                        time.sleep(Config.CHECK_INTERVAL)
                        continue

                    logger.info(f"{'🔴' if signal.direction == 'SHORT' else '🟢'} OPENING {signal.direction}")

                    # Place orders
                    order_side = "SELL" if signal.direction == "SHORT" else "BUY"
                    close_side = "BUY" if signal.direction == "SHORT" else "SELL"

                    result = client.place_market_order(order_side, qty)
                    if result:
                        client.place_stop_loss(close_side, qty, signal.stop_loss)
                        client.place_take_profit(close_side, qty, signal.take_profit)
                        risk.set_position_open(
                            signal.direction, signal.entry_price,
                            signal.stop_loss, signal.take_profit, qty
                        )

                        alert = (
                            f"{'🔴' if signal.direction == 'SHORT' else '🟢'} {signal.direction} {Config.SYMBOL}\n"
                            f"Entry: {signal.entry_price:.1f} | SL: {signal.stop_loss:.1f} | TP: {signal.take_profit:.1f}\n"
                            f"Reason: {signal.reason}"
                        )
                        send_alert(alert)

            # Balance/drawdown check
            if not dry_run and iteration % 40 == 0:
                bal = client.get_account_balance()
                if not risk.update_balance(bal):
                    logger.critical("HALTING due to max drawdown")
                    send_alert("🛑 Bot HALTED — max drawdown reached!")
                    break

        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            time.sleep(30)
            continue

        time.sleep(Config.CHECK_INTERVAL)

    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
