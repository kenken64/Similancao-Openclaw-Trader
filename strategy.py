"""MA Crossover Short Strategy — 15m timeframe
Based on successful trades from Feb 10, 2026.

SHORT when: price bounces UP to MA25/MA99 resistance + bearish MA alignment (MA7 < MA25 < MA99)
LONG when: price dips to 24h low support + bounce candle + volume spike
"""
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    direction: str  # "SHORT", "LONG", or "NONE"
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    confidence: float  # 0-1


def compute_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA columns to dataframe."""
    df["ma7"] = df["close"].rolling(Config.MA_FAST).mean()
    df["ma25"] = df["close"].rolling(Config.MA_MID).mean()
    df["ma99"] = df["close"].rolling(Config.MA_SLOW).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


def check_bearish_alignment(row) -> bool:
    """MA7 < MA25 < MA99 = bearish alignment."""
    return row["ma7"] < row["ma25"] < row["ma99"]


def check_bullish_alignment(row) -> bool:
    """MA7 > MA25 > MA99 = bullish alignment."""
    return row["ma7"] > row["ma25"] > row["ma99"]


def analyze(df: pd.DataFrame, low_24h: float) -> Optional[Signal]:
    """Analyze latest candle data and return a Signal or None.
    
    Args:
        df: DataFrame with OHLCV data (needs at least 100 rows)
        low_24h: 24-hour low price from ticker
    """
    if len(df) < Config.MA_SLOW + 5:
        logger.warning("Not enough candle data for analysis")
        return None

    df = compute_mas(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    buffer_pct = Config.STOP_LOSS_BUFFER_PCT / 100

    # Volume filter: skip if volume is too low
    if latest["volume"] < latest["vol_ma20"] * Config.VOLUME_FILTER_RATIO:
        logger.info(f"📊 Volume too low ({latest['volume']:.0f} vs avg {latest['vol_ma20']:.0f}) — skipping")
        return None

    reasons = []

    # === SHORT SIGNAL ===
    # Bearish MA alignment + price touching MA25 or MA99 from below
    if check_bearish_alignment(latest):
        touching_ma25 = abs(price - latest["ma25"]) / price < 0.003  # within 0.3%
        touching_ma99 = abs(price - latest["ma99"]) / price < 0.005  # within 0.5%
        price_rose = latest["close"] > prev["close"]  # bounced up

        if (touching_ma25 or touching_ma99) and price_rose:
            resistance = latest["ma99"] if touching_ma99 else latest["ma25"]
            sl = latest["ma99"] * (1 + buffer_pct)
            risk = sl - price
            tp = price - (risk * Config.RISK_REWARD_RATIO)

            confidence = 0.6
            reasons.append("Bearish MA alignment (MA7<MA25<MA99)")
            if touching_ma99:
                reasons.append(f"Price near MA99 resistance ({latest['ma99']:.1f})")
                confidence += 0.15
            if touching_ma25:
                reasons.append(f"Price near MA25 resistance ({latest['ma25']:.1f})")
                confidence += 0.1
            if latest["volume"] > latest["vol_ma20"] * 1.2:
                reasons.append("Above-average volume")
                confidence += 0.1

            reason_str = " | ".join(reasons)
            logger.info(f"🔴 SHORT signal: {reason_str}")
            logger.info(f"   Entry: {price:.1f} | SL: {sl:.1f} | TP: {tp:.1f} | Conf: {confidence:.0%}")

            return Signal(
                direction="SHORT", entry_price=price,
                stop_loss=round(sl, 1), take_profit=round(tp, 1),
                reason=reason_str, confidence=min(confidence, 1.0)
            )

    # === LONG SIGNAL ===
    # Price near 24h low + bounce candle + volume spike
    near_24h_low = (price - low_24h) / price < 0.005  # within 0.5% of 24h low
    bounce_candle = latest["close"] > latest["open"] and prev["close"] < prev["open"]  # green after red
    volume_spike = latest["volume"] > latest["vol_ma20"] * 1.5

    if near_24h_low and bounce_candle:
        sl = low_24h * (1 - buffer_pct)
        risk = price - sl
        tp = price + (risk * Config.RISK_REWARD_RATIO)

        confidence = 0.5
        reasons.append(f"Price near 24h low support ({low_24h:.1f})")
        reasons.append("Bounce candle (green after red)")
        if volume_spike:
            reasons.append("Volume spike on bounce")
            confidence += 0.2
        if check_bullish_alignment(latest):
            confidence += 0.1

        reason_str = " | ".join(reasons)
        logger.info(f"🟢 LONG signal: {reason_str}")
        logger.info(f"   Entry: {price:.1f} | SL: {sl:.1f} | TP: {tp:.1f} | Conf: {confidence:.0%}")

        return Signal(
            direction="LONG", entry_price=price,
            stop_loss=round(sl, 1), take_profit=round(tp, 1),
            reason=reason_str, confidence=min(confidence, 1.0)
        )

    # No signal
    logger.debug(
        f"No signal | Price: {price:.1f} | MA7: {latest['ma7']:.1f} "
        f"MA25: {latest['ma25']:.1f} MA99: {latest['ma99']:.1f} | 24hLow: {low_24h:.1f}"
    )
    return None
