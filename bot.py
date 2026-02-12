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
import pandas as pd
import requests
from datetime import datetime
from config import Config
from binance_client import BinanceFuturesClient
from strategy import analyze, analyze_daily
from risk_manager import RiskManager
from trade_history import TradeHistory

# === Logging Setup ===
log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler("bot.log", encoding="utf-8"),
]
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), format=log_fmt, handlers=handlers)
logger = logging.getLogger("similancao")


def _openclaw_headers():
    """Build auth headers for OpenClaw gateway."""
    headers = {"Content-Type": "application/json"}
    if Config.OPENCLAW_GATEWAY_TOKEN:
        headers["Authorization"] = f"Bearer {Config.OPENCLAW_GATEWAY_TOKEN}"
    return headers


def send_alert(message: str):
    """Send WhatsApp alert via OpenClaw gateway /v1/chat/completions."""
    if not Config.OPENCLAW_ALERT_ENABLED:
        return
    try:
        requests.post(
            f"{Config.OPENCLAW_GATEWAY_URL}/v1/chat/completions",
            headers=_openclaw_headers(),
            json={
                "model": "openclaw:main",
                "messages": [{"role": "user", "content": f"Send this WhatsApp alert to me: {message}"}],
                "stream": False,
            },
            timeout=15,
        )
    except Exception:
        pass  # Non-critical


def ask_advisor(signal, price, funding_rate, trade_history, daily_context: str = "") -> tuple:
    """Ask OpenClaw AI to analyze trade before execution.

    Sends trade details to OpenClaw /v1/chat/completions for AI analysis.
    Returns (approved: bool, advisor_log_id: int or None, advisor_sl: float or None, advisor_tp: float or None)
    """
    if not Config.ADVISOR_MODE:
        return True, None, None, None  # Skip advisor, auto-approve

    logger.info(f"🧠 Asking OpenClaw AI for trade analysis...")

    start_time = time.time()
    try:
        # Build analysis prompt for OpenClaw AI
        prompt = (
            f"Analyze this crypto futures trade and decide whether to APPROVE or REJECT it. "
            f"If approved, also provide your recommended Stop Loss (sl) and Take Profit (tp) prices. "
            f"Respond ONLY with valid JSON: {{\"approve\": true/false, \"reason\": \"your reason\", \"sl\": number, \"tp\": number}}\n\n"
            f"Trade Details:\n"
            f"- Symbol: {Config.SYMBOL}\n"
            f"- Direction: {signal.direction}\n"
            f"- Entry Price: {signal.entry_price}\n"
            f"- Stop Loss (bot suggestion): {signal.stop_loss}\n"
            f"- Take Profit (bot suggestion): {signal.take_profit}\n"
            f"- Confidence: {signal.confidence}\n"
            f"- Signal Reason: {signal.reason}\n"
            f"- Current Price: {price}\n"
            f"- Funding Rate: {funding_rate}\n"
            f"- Leverage: {Config.LEVERAGE}x\n"
            f"- Position Size: {Config.POSITION_SIZE_USDT} USDT\n"
            f"- Timestamp: {datetime.now().isoformat()}\n\n"
            f"Daily (1D) Timeframe Context:\n"
            f"{daily_context if daily_context else 'N/A'}\n\n"
            f"Consider risk/reward ratio, market conditions, funding rate, and the daily timeframe context above. "
            f"If the 15m signal conflicts with the daily trend or is near a major daily S/R level, weigh that heavily. "
            f"You may keep the bot's suggested SL/TP or adjust them based on your analysis. "
            f"Respond with JSON only."
        )

        # Send to OpenClaw via /v1/chat/completions
        response = requests.post(
            f"{Config.OPENCLAW_GATEWAY_URL}/v1/chat/completions",
            headers=_openclaw_headers(),
            json={
                "model": "openclaw:main",
                "messages": [
                    {"role": "system", "content": "You are a crypto trading risk advisor. Analyze trades and respond with JSON only: {\"approve\": true/false, \"reason\": \"explanation\", \"sl\": stop_loss_price, \"tp\": take_profit_price}. Always provide sl and tp values when approving."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
            },
            timeout=Config.ADVISOR_TIMEOUT_SECONDS
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            result = response.json()
            # Parse OpenAI-compatible response
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            import json as _json
            try:
                parsed = _json.loads(content)
            except _json.JSONDecodeError:
                # Try to extract JSON from response text
                import re
                match = re.search(r'\{[^}]+\}', content)
                if match:
                    parsed = _json.loads(match.group())
                else:
                    parsed = {"approve": False, "reason": f"Could not parse advisor response: {content[:200]}"}
            approved = parsed.get("approve", False)
            reason = parsed.get("reason", "No reason provided")
            advisor_sl = parsed.get("sl")
            advisor_tp = parsed.get("tp")

            # Validate advisor SL/TP — convert to float if present
            try:
                advisor_sl = float(advisor_sl) if advisor_sl is not None else None
                advisor_tp = float(advisor_tp) if advisor_tp is not None else None
            except (ValueError, TypeError):
                advisor_sl = None
                advisor_tp = None

            if approved and advisor_sl and advisor_tp:
                logger.info(f"🧠 Advisor SL: {advisor_sl:.1f} | TP: {advisor_tp:.1f}")

            # Log advisor decision to database
            advisor_log_id = trade_history.log_advisor_decision(
                symbol=Config.SYMBOL,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                signal_reason=signal.reason,
                funding_rate=funding_rate,
                approved=approved,
                advisor_reason=reason,
                response_time_ms=response_time_ms,
                advisor_sl=advisor_sl,
                advisor_tp=advisor_tp
            )

            if approved:
                sl_tp_info = ""
                if advisor_sl and advisor_tp:
                    sl_tp_info = f"\nAdvisor SL: {advisor_sl:.1f} | TP: {advisor_tp:.1f}"
                logger.info(f"✅ OpenClaw APPROVED: {reason}")
                # Send notification
                send_alert(
                    f"✅ Trade APPROVED by OpenClaw AI\n"
                    f"{'🔴' if signal.direction == 'SHORT' else '🟢'} {signal.direction} {Config.SYMBOL} @ {signal.entry_price:.1f}\n"
                    f"Reason: {reason}{sl_tp_info}"
                )
                return True, advisor_log_id, advisor_sl, advisor_tp
            else:
                logger.info(f"❌ OpenClaw REJECTED: {reason}")
                # Send notification
                send_alert(
                    f"❌ Trade REJECTED by OpenClaw AI\n"
                    f"Signal: {signal.direction} @ {signal.entry_price:.1f}\n"
                    f"Reason: {reason}"
                )
                return False, advisor_log_id, None, None
        else:
            logger.warning(f"OpenClaw API error: HTTP {response.status_code}")
            # Log error decision
            advisor_log_id = trade_history.log_advisor_decision(
                symbol=Config.SYMBOL,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                signal_reason=signal.reason,
                funding_rate=funding_rate,
                approved=False,
                advisor_reason=f"API Error: HTTP {response.status_code}",
                response_time_ms=response_time_ms
            )
            send_alert(f"⚠️ OpenClaw API error (HTTP {response.status_code}) - trade skipped")
            return False, advisor_log_id, None, None

    except requests.Timeout:
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.warning("⏰ OpenClaw API timeout — trade SKIPPED")
        # Log timeout decision
        advisor_log_id = trade_history.log_advisor_decision(
            symbol=Config.SYMBOL,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            signal_reason=signal.reason,
            funding_rate=funding_rate,
            approved=False,
            advisor_reason="Timeout",
            response_time_ms=response_time_ms
        )
        send_alert("⏰ OpenClaw timeout - trade skipped for safety")
        return False, advisor_log_id, None, None
    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"OpenClaw advisor error: {e}")
        # Log error decision
        advisor_log_id = trade_history.log_advisor_decision(
            symbol=Config.SYMBOL,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            signal_reason=signal.reason,
            funding_rate=funding_rate,
            approved=False,
            advisor_reason=f"Error: {str(e)}",
            response_time_ms=response_time_ms
        )
        send_alert(f"❌ OpenClaw error: {e} - trade skipped")
        return False, advisor_log_id, None, None


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
    trade_history = TradeHistory()
    logger.info(f"💰 Starting balance: {balance:.2f} USDT")

    # Trade cooldown tracking
    last_trade_time = 0

    # Track current trade ID for exits — recover from database
    current_trade_id = None
    open_trade = trade_history.get_open_trade()
    if open_trade:
        current_trade_id = open_trade['id']
        logger.info(f"📎 Recovered open trade #{current_trade_id} from database")
    # Software stop-loss tracking (when exchange rejects STOP_MARKET)
    software_sl = None  # {"price": float, "side": str, "qty": float, "direction": str}
    # Software take-profit tracking (when exchange rejects TAKE_PROFIT_MARKET)
    software_tp = None  # {"price": float, "side": str, "qty": float, "direction": str}

    # Check if there's already an open position (live mode)
    if not dry_run:
        pos = client.get_position()
        if pos:
            risk.has_open_position = True
            pnl = pos.get('unrealized_pnl', 0)
            pnl_sign = '+' if pnl >= 0 else ''
            logger.info(f"📌 Existing position: {pos['side']} {pos['size']} @ {pos['entry_price']} | PnL: {pnl_sign}{pnl:.2f} USDT")
        elif current_trade_id:
            # No position on Binance but DB has open trade — close it
            logger.info(f"📭 No position found but trade #{current_trade_id} is open in DB — closing it")
            exit_price = float(client.get_klines(limit=1)['close'].iloc[-1])
            try:
                user_trades = client.get_recent_user_trades(limit=10)
                if user_trades:
                    recent_closes = [t for t in user_trades if float(t.get('realizedPnl', 0)) != 0]
                    if recent_closes:
                        last_close = recent_closes[-1]
                        exit_price = float(last_close['price'])
                        realized_pnl = sum(float(t['realizedPnl']) for t in recent_closes
                                           if t['time'] >= recent_closes[-1]['time'] - 1000)
                        trade_history.record_exit(current_trade_id, exit_price, "EXTERNAL", realized_pnl)
                        logger.info(f"📊 Closed stale trade #{current_trade_id}: PnL {realized_pnl:+.2f} USDT")
                        current_trade_id = None
            except Exception as e:
                logger.warning(f"Could not close stale trade: {e}")
                trade_history.record_exit(current_trade_id, exit_price, "EXTERNAL", None)
                current_trade_id = None

    # Daily timeframe cache (refreshed every hour)
    daily_analysis_cache = None
    daily_cache_time = 0

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

            # Fetch daily klines (cached for 1 hour)
            if time.time() - daily_cache_time > 3600:
                try:
                    df_daily = client.get_klines(limit=90, interval="1d")
                    daily_analysis_cache = analyze_daily(df_daily)
                    daily_cache_time = time.time()
                    logger.info(f"📅 Daily analysis: {daily_analysis_cache['daily_summary']}")
                except Exception as e:
                    logger.warning(f"Daily kline fetch failed: {e}")
                    daily_analysis_cache = None

            if iteration % 4 == 1:  # Log status every ~5 min
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
                        pnl = -abs(pos["sl"] - pos["entry"]) * pos["qty"]
                        logger.info(f"🛑 [DRY-RUN] Stop-loss HIT! {pos['direction']} closed @ {price:.1f} | PnL: -{abs(pnl):.2f} USDT")
                        send_alert(f"🛑 SL hit! {pos['direction']} closed {pnl:.2f} USDT")

                    # Record exit in trade history
                    if current_trade_id:
                        trade_history.record_exit(current_trade_id, price, exit_type, pnl)
                        current_trade_id = None

                    risk.set_position_closed()
                    last_trade_time = time.time()

            # Live: SOFTWARE STOP-LOSS CHECK
            if not dry_run and risk.has_open_position and software_sl:
                sl_price = software_sl["price"]
                sl_direction = software_sl["direction"]
                sl_hit = False
                if sl_direction == "LONG" and price <= sl_price:
                    sl_hit = True
                elif sl_direction == "SHORT" and price >= sl_price:
                    sl_hit = True

                if sl_hit:
                    logger.warning(f"🛑 SOFTWARE SL TRIGGERED! Price {price:.1f} hit SL {sl_price:.1f}")
                    close_result = client.close_position_market(software_sl["side"], software_sl["qty"])
                    if close_result:
                        pnl_est = (sl_price - risk.dry_run_position["entry"]) * software_sl["qty"] if sl_direction == "LONG" else (risk.dry_run_position["entry"] - sl_price) * software_sl["qty"]
                        send_alert(f"🛑 Software SL hit! {sl_direction} closed @ {price:.1f} | Est PnL: {pnl_est:+.2f} USDT")
                        if current_trade_id:
                            trade_history.record_exit(current_trade_id, price, "SOFTWARE_SL", pnl_est)
                            current_trade_id = None
                        risk.set_position_closed()
                        last_trade_time = time.time()
                        software_sl = None

            # Live: SOFTWARE TAKE-PROFIT CHECK
            if not dry_run and risk.has_open_position and software_tp:
                tp_price = software_tp["price"]
                tp_direction = software_tp["direction"]
                tp_hit = False
                if tp_direction == "LONG" and price >= tp_price:
                    tp_hit = True
                elif tp_direction == "SHORT" and price <= tp_price:
                    tp_hit = True

                if tp_hit:
                    logger.info(f"🎯 SOFTWARE TP TRIGGERED! Price {price:.1f} hit TP {tp_price:.1f}")
                    close_result = client.close_position_market(software_tp["side"], software_tp["qty"])
                    if close_result:
                        entry = risk.dry_run_position["entry"] if risk.dry_run_position else 0
                        pnl_est = (tp_price - entry) * software_tp["qty"] if tp_direction == "LONG" else (entry - tp_price) * software_tp["qty"]
                        send_alert(f"🎯 Software TP hit! {tp_direction} closed @ {price:.1f} | Est PnL: {pnl_est:+.2f} USDT")
                        if current_trade_id:
                            trade_history.record_exit(current_trade_id, price, "SOFTWARE_TP", pnl_est)
                            current_trade_id = None
                        risk.set_position_closed()
                        last_trade_time = time.time()
                        software_tp = None
                        software_sl = None  # Clear SL too

            # Live: check if position was closed externally
            if not dry_run and risk.has_open_position:
                pos = client.get_position()
                if pos and iteration % 4 == 1:
                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_sign = '+' if pnl >= 0 else ''
                    pnl_emoji = '🟢' if pnl >= 0 else '🔴'
                    logger.info(f"{pnl_emoji} Position PnL: {pnl_sign}{pnl:.2f} USDT | {pos['side']} {pos['size']} @ {pos['entry_price']}")
                if not pos:
                    logger.info("📭 Position closed (externally or by SL/TP)")
                    software_sl = None  # Clear software SL
                    software_tp = None  # Clear software TP
                    # Fetch actual exit price and realized PnL from Binance
                    if current_trade_id:
                        exit_price = price
                        realized_pnl = None
                        try:
                            user_trades = client.get_recent_user_trades(limit=10)
                            if user_trades:
                                # Sum realized PnL from recent closing trades
                                recent_closes = [t for t in user_trades if float(t.get('realizedPnl', 0)) != 0]
                                if recent_closes:
                                    last_close = recent_closes[-1]
                                    exit_price = float(last_close['price'])
                                    realized_pnl = sum(float(t['realizedPnl']) for t in recent_closes
                                                       if t['time'] >= recent_closes[-1]['time'] - 1000)
                                    logger.info(f"📊 Binance realized PnL: {realized_pnl:+.2f} USDT @ {exit_price}")
                        except Exception as e:
                            logger.warning(f"Could not fetch realized PnL: {e}")
                        trade_history.record_exit(current_trade_id, exit_price, "EXTERNAL", realized_pnl)
                        current_trade_id = None
                    risk.set_position_closed()
                    last_trade_time = time.time()

            # Hard check: verify no open position on Binance before trading
            if not dry_run and not risk.has_open_position:
                pos = client.get_position()
                if pos:
                    risk.has_open_position = True
                    logger.info(f"📌 Hard check: found open position on Binance — skipping signal scan")

            # Look for new signals if no position
            if risk.can_open_position():
                # Trade cooldown check
                cooldown_remaining = Config.TRADE_COOLDOWN_SECONDS - (time.time() - last_trade_time)
                if cooldown_remaining > 0:
                    if iteration % 4 == 1:
                        logger.info(f"⏳ Trade cooldown: {cooldown_remaining:.0f}s remaining")
                    time.sleep(Config.CHECK_INTERVAL)
                    continue

                # Fetch 1h klines for trend filter
                trend_1h = None
                try:
                    df_1h = client.get_klines(interval='1h', limit=30)
                    ma7_1h = df_1h['close'].rolling(7).mean().iloc[-1]
                    ma25_1h = df_1h['close'].rolling(25).mean().iloc[-1]
                    if not pd.isna(ma7_1h) and not pd.isna(ma25_1h):
                        trend_1h = "BULLISH" if ma7_1h > ma25_1h else "BEARISH"
                except Exception as e:
                    logger.warning(f"1h trend fetch failed: {e}")

                signal = analyze(df, low_24h)

                if signal and signal.confidence >= 0.65:
                    # 1h trend filter: skip counter-trend signals
                    if trend_1h:
                        if signal.direction == "LONG" and trend_1h == "BEARISH":
                            logger.info(f"⏭️ Skipping LONG signal — 1h trend is BEARISH (MA7 < MA25)")
                            time.sleep(Config.CHECK_INTERVAL)
                            continue
                        if signal.direction == "SHORT" and trend_1h == "BULLISH":
                            logger.info(f"⏭️ Skipping SHORT signal — 1h trend is BULLISH (MA7 > MA25)")
                            time.sleep(Config.CHECK_INTERVAL)
                            continue
                    # Funding rate check
                    if not risk.check_funding_rate(funding_rate, signal.direction):
                        time.sleep(Config.CHECK_INTERVAL)
                        continue

                    qty = risk.calculate_quantity(price)

                    logger.info(f"{'🔴' if signal.direction == 'SHORT' else '🟢'} SIGNAL: {signal.direction}")
                    logger.info(f"   Reason: {signal.reason}")
                    logger.info(f"   Entry: {signal.entry_price:.1f} | SL: {signal.stop_loss:.1f} | TP: {signal.take_profit:.1f}")
                    logger.info(f"   Qty: {qty} | Confidence: {signal.confidence:.0%}")

                    # Ask advisor (OpenClaw AI) for approval — include daily context
                    daily_ctx = daily_analysis_cache["daily_summary"] if daily_analysis_cache else ""
                    approved, advisor_log_id, advisor_sl, advisor_tp = ask_advisor(signal, price, funding_rate, trade_history, daily_context=daily_ctx)
                    if not approved:
                        logger.info("⏭️ Trade skipped by advisor")
                        time.sleep(Config.CHECK_INTERVAL)
                        continue

                    # Use advisor's SL/TP if provided, otherwise use signal's
                    final_sl = advisor_sl if advisor_sl else signal.stop_loss
                    final_tp = advisor_tp if advisor_tp else signal.take_profit
                    if advisor_sl or advisor_tp:
                        logger.info(f"🧠 Using advisor levels — SL: {final_sl:.1f} | TP: {final_tp:.1f}")

                    logger.info(f"{'🔴' if signal.direction == 'SHORT' else '🟢'} OPENING {signal.direction}")

                    # Place orders
                    order_side = "SELL" if signal.direction == "SHORT" else "BUY"
                    close_side = "BUY" if signal.direction == "SHORT" else "SELL"

                    result = client.place_market_order(order_side, qty)
                    if result:
                        sl_result = client.place_stop_loss(close_side, qty, final_sl)
                        tp_result = client.place_take_profit(close_side, qty, final_tp)

                        # Check if we need software SL
                        if sl_result and sl_result.get("software_sl"):
                            software_sl = {
                                "price": final_sl,
                                "side": close_side,
                                "qty": qty,
                                "direction": signal.direction
                            }
                            logger.info(f"⚠️ Using SOFTWARE SL @ {final_sl} (checking every {Config.CHECK_INTERVAL}s)")
                            send_alert(f"⚠️ Using software stop-loss @ {final_sl} (exchange SL blocked by proxy)")
                        else:
                            software_sl = None

                        # Check if we need software TP
                        if tp_result and tp_result.get("software_tp"):
                            software_tp = {
                                "price": final_tp,
                                "side": close_side,
                                "qty": qty,
                                "direction": signal.direction
                            }
                            logger.info(f"⚠️ Using SOFTWARE TP @ {final_tp} (checking every {Config.CHECK_INTERVAL}s)")
                            send_alert(f"⚠️ Using software take-profit @ {final_tp} (exchange TP blocked by proxy)")
                        else:
                            software_tp = None

                        risk.set_position_open(
                            signal.direction, signal.entry_price,
                            final_sl, final_tp, qty
                        )

                        # Record trade entry in database
                        current_trade_id = trade_history.record_entry(
                            symbol=Config.SYMBOL,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            quantity=qty,
                            stop_loss=final_sl,
                            take_profit=final_tp,
                            mode="LIVE" if not dry_run else "DRY-RUN",
                            reason=signal.reason,
                            advisor_approved=True,
                            funding_rate=funding_rate,
                            confidence=signal.confidence
                        )

                        last_trade_time = time.time()

                        # Link advisor decision to executed trade
                        if advisor_log_id:
                            trade_history.link_advisor_to_trade(advisor_log_id, current_trade_id)

                        alert = (
                            f"{'🔴' if signal.direction == 'SHORT' else '🟢'} {signal.direction} {Config.SYMBOL}\n"
                            f"Entry: {signal.entry_price:.1f} | SL: {final_sl:.1f} | TP: {final_tp:.1f}\n"
                            f"Reason: {signal.reason}"
                        )
                        send_alert(alert)

            # Balance/drawdown check
            if not dry_run and iteration % 20 == 0:
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
