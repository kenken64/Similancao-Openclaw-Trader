"""Enhanced MA Crossover Strategy — 15m timeframe
Based on successful trades from Feb 10, 2026.

SHORT when: price bounces UP to MA25/MA99 resistance + bearish MA alignment (MA7 < MA25 < MA99)
LONG when: price dips to 24h low support + bounce candle + volume spike

Enhanced signals (Feb 11):
- RSI oversold/overbought confirmation
- MA7/MA25 crossover signals
- Higher low / lower high pattern detection
- Swing high/low support & resistance
- Stacked confidence scoring for multi-confirmation setups
"""
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
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


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA columns to dataframe."""
    df["ma7"] = df["close"].rolling(Config.MA_FAST).mean()
    df["ma25"] = df["close"].rolling(Config.MA_MID).mean()
    df["ma99"] = df["close"].rolling(Config.MA_SLOW).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


def compute_rsi(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    """Compute RSI and add as column."""
    period = period or Config.RSI_PERIOD
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[float], List[float]]:
    """Find recent swing highs and swing lows.
    
    Returns (swing_highs, swing_lows) — lists of price levels.
    """
    highs = []
    lows = []
    # Need at least lookback*2+1 candles for a swing point
    start = max(lookback, 0)
    end = len(df) - lookback
    for i in range(start, end):
        window_high = df["high"].iloc[i - lookback:i + lookback + 1]
        window_low = df["low"].iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            highs.append(df["high"].iloc[i])
        if df["low"].iloc[i] == window_low.min():
            lows.append(df["low"].iloc[i])
    return highs, lows


def detect_higher_lows(df: pd.DataFrame, lookback: int = 20) -> bool:
    """Check if recent swing lows are ascending (bullish structure)."""
    _, swing_lows = find_swing_points(df.tail(lookback + 10), lookback=3)
    if len(swing_lows) >= 2:
        return swing_lows[-1] > swing_lows[-2]
    return False


def detect_lower_highs(df: pd.DataFrame, lookback: int = 20) -> bool:
    """Check if recent swing highs are descending (bearish structure)."""
    swing_highs, _ = find_swing_points(df.tail(lookback + 10), lookback=3)
    if len(swing_highs) >= 2:
        return swing_highs[-1] < swing_highs[-2]
    return False


def find_nearest_support_resistance(price: float, swing_highs: List[float], swing_lows: List[float],
                                     threshold_pct: float = 0.01) -> Tuple[Optional[float], Optional[float]]:
    """Find nearest support (below price) and resistance (above price) from swing points."""
    all_levels = sorted(set(swing_highs + swing_lows))
    support = None
    resistance = None
    for level in all_levels:
        if level < price * (1 - 0.001):  # below price
            support = level
        elif level > price * (1 + 0.001) and resistance is None:  # above price
            resistance = level
    return support, resistance


# ---------------------------------------------------------------------------
# Alignment checks
# ---------------------------------------------------------------------------

def check_bearish_alignment(row) -> bool:
    """MA7 < MA25 < MA99 = bearish alignment."""
    return row["ma7"] < row["ma25"] < row["ma99"]


def check_bullish_alignment(row) -> bool:
    """MA7 > MA25 > MA99 = bullish alignment."""
    return row["ma7"] > row["ma25"] > row["ma99"]


def check_ma_crossover(df: pd.DataFrame) -> Optional[str]:
    """Check for MA7/MA25 crossover on the last two candles.
    
    Returns "BULLISH", "BEARISH", or None.
    """
    if len(df) < 3:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    # Bullish crossover: MA7 crosses above MA25
    if prev["ma7"] <= prev["ma25"] and curr["ma7"] > curr["ma25"]:
        return "BULLISH"
    # Bearish crossover: MA7 crosses below MA25
    if prev["ma7"] >= prev["ma25"] and curr["ma7"] < curr["ma25"]:
        return "BEARISH"
    return None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

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
    df = compute_rsi(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50.0
    buffer_pct = Config.STOP_LOSS_BUFFER_PCT / 100

    # Volume filter: skip if volume is too low
    if latest["volume"] < latest["vol_ma20"] * Config.VOLUME_FILTER_RATIO:
        logger.info(f"📊 Volume too low ({latest['volume']:.0f} vs avg {latest['vol_ma20']:.0f}) — skipping")
        return None

    # Pre-compute swing points and structure
    swing_highs, swing_lows = find_swing_points(df, lookback=5)
    support, resistance = find_nearest_support_resistance(price, swing_highs, swing_lows)
    ma_cross = check_ma_crossover(df)
    has_higher_lows = detect_higher_lows(df)
    has_lower_highs = detect_lower_highs(df)
    bounce_candle = latest["close"] > latest["open"] and prev["close"] < prev["open"]
    rejection_candle = latest["close"] < latest["open"] and prev["close"] > prev["open"]
    volume_spike = latest["volume"] > latest["vol_ma20"] * 1.5
    above_avg_volume = latest["volume"] > latest["vol_ma20"] * 1.2

    # Collect candidate signals — pick the best one
    best_signal: Optional[Signal] = None

    # =======================================================================
    # SHORT SIGNALS
    # =======================================================================

    # --- Original: Bearish alignment + price at MA resistance ---
    if check_bearish_alignment(latest):
        touching_ma25 = abs(price - latest["ma25"]) / price < 0.003
        touching_ma99 = abs(price - latest["ma99"]) / price < 0.005
        price_rose = latest["close"] > prev["close"]

        if (touching_ma25 or touching_ma99) and price_rose:
            reasons = []
            confidence = 0.6
            reasons.append("Bearish MA alignment (MA7<MA25<MA99)")
            if touching_ma99:
                reasons.append(f"Price near MA99 resistance ({latest['ma99']:.1f})")
                confidence += 0.15
            if touching_ma25:
                reasons.append(f"Price near MA25 resistance ({latest['ma25']:.1f})")
                confidence += 0.1
            if above_avg_volume:
                reasons.append("Above-average volume")
                confidence += 0.05
            if volume_spike:
                reasons.append("Volume spike")
                confidence += 0.05
            if rsi > 70:
                reasons.append(f"RSI overbought ({rsi:.1f})")
                confidence += 0.1
            elif rsi > 60:
                reasons.append(f"RSI elevated ({rsi:.1f})")
                confidence += 0.05
            if has_lower_highs:
                reasons.append("Lower highs pattern (bearish structure)")
                confidence += 0.05
            if resistance and abs(price - resistance) / price < 0.003:
                reasons.append(f"Near swing resistance ({resistance:.1f})")
                confidence += 0.05

            sl = latest["ma99"] * (1 + buffer_pct)
            risk = sl - price
            tp = price - (risk * Config.RISK_REWARD_RATIO)
            sig = Signal(
                direction="SHORT", entry_price=price,
                stop_loss=round(sl, 1), take_profit=round(tp, 1),
                reason=" | ".join(reasons), confidence=min(confidence, 1.0)
            )
            if best_signal is None or sig.confidence > best_signal.confidence:
                best_signal = sig

    # --- NEW: Bearish MA crossover signal ---
    if ma_cross == "BEARISH":
        reasons = ["MA7/MA25 bearish crossover"]
        confidence = 0.55
        if rsi > 70:
            reasons.append(f"RSI overbought ({rsi:.1f})")
            confidence += 0.15
        elif rsi > 60:
            reasons.append(f"RSI elevated ({rsi:.1f})")
            confidence += 0.05
        if has_lower_highs:
            reasons.append("Lower highs pattern")
            confidence += 0.1
        if rejection_candle:
            reasons.append("Rejection candle")
            confidence += 0.1
        if above_avg_volume:
            reasons.append("Above-average volume")
            confidence += 0.05
        if volume_spike:
            reasons.append("Volume spike on crossover")
            confidence += 0.05
        if check_bearish_alignment(latest):
            reasons.append("Bearish MA alignment")
            confidence += 0.05

        sl = latest["ma25"] * (1 + buffer_pct * 2)
        risk = sl - price
        if risk > 0:
            tp = price - (risk * Config.RISK_REWARD_RATIO)
            sig = Signal(
                direction="SHORT", entry_price=price,
                stop_loss=round(sl, 1), take_profit=round(tp, 1),
                reason=" | ".join(reasons), confidence=min(confidence, 1.0)
            )
            if best_signal is None or sig.confidence > best_signal.confidence:
                best_signal = sig

    # =======================================================================
    # LONG SIGNALS
    # =======================================================================

    # --- Original: Price near 24h low + bounce candle ---
    near_24h_low = (price - low_24h) / price < 0.005
    if near_24h_low and bounce_candle:
        reasons = []
        confidence = 0.55  # bumped from 0.5
        reasons.append(f"Price near 24h low support ({low_24h:.1f})")
        reasons.append("Bounce candle (green after red)")
        if volume_spike:
            reasons.append("Volume spike on bounce")
            confidence += 0.15
        elif above_avg_volume:
            reasons.append("Above-average volume")
            confidence += 0.05
        if rsi < 30:
            reasons.append(f"RSI oversold ({rsi:.1f})")
            confidence += 0.15
        elif rsi < 40:
            reasons.append(f"RSI low ({rsi:.1f})")
            confidence += 0.05
        if has_higher_lows:
            reasons.append("Higher lows pattern (bullish structure)")
            confidence += 0.1
        if check_bullish_alignment(latest):
            reasons.append("Bullish MA alignment")
            confidence += 0.05

        sl = low_24h * (1 - buffer_pct)
        risk = price - sl
        tp = price + (risk * Config.RISK_REWARD_RATIO)
        sig = Signal(
            direction="LONG", entry_price=price,
            stop_loss=round(sl, 1), take_profit=round(tp, 1),
            reason=" | ".join(reasons), confidence=min(confidence, 1.0)
        )
        if best_signal is None or sig.confidence > best_signal.confidence:
            best_signal = sig

    # --- NEW: Swing support bounce (not just 24h low) ---
    if support and not near_24h_low:
        near_support = (price - support) / price < 0.005
        if near_support and bounce_candle:
            reasons = [f"Price near swing support ({support:.1f})", "Bounce candle"]
            confidence = 0.55
            if volume_spike:
                reasons.append("Volume spike on bounce")
                confidence += 0.15
            elif above_avg_volume:
                reasons.append("Above-average volume")
                confidence += 0.05
            if rsi < 30:
                reasons.append(f"RSI oversold ({rsi:.1f})")
                confidence += 0.15
            elif rsi < 40:
                reasons.append(f"RSI low ({rsi:.1f})")
                confidence += 0.05
            if has_higher_lows:
                reasons.append("Higher lows pattern")
                confidence += 0.1
            if check_bullish_alignment(latest):
                reasons.append("Bullish MA alignment")
                confidence += 0.05

            sl = support * (1 - buffer_pct)
            risk = price - sl
            if risk > 0:
                tp = price + (risk * Config.RISK_REWARD_RATIO)
                sig = Signal(
                    direction="LONG", entry_price=price,
                    stop_loss=round(sl, 1), take_profit=round(tp, 1),
                    reason=" | ".join(reasons), confidence=min(confidence, 1.0)
                )
                if best_signal is None or sig.confidence > best_signal.confidence:
                    best_signal = sig

    # --- NEW: Bullish MA crossover signal ---
    if ma_cross == "BULLISH":
        reasons = ["MA7/MA25 bullish crossover"]
        confidence = 0.55
        if rsi < 30:
            reasons.append(f"RSI oversold ({rsi:.1f})")
            confidence += 0.15
        elif rsi < 40:
            reasons.append(f"RSI low ({rsi:.1f})")
            confidence += 0.05
        if has_higher_lows:
            reasons.append("Higher lows pattern")
            confidence += 0.1
        if bounce_candle:
            reasons.append("Bounce candle")
            confidence += 0.1
        if above_avg_volume:
            reasons.append("Above-average volume")
            confidence += 0.05
        if volume_spike:
            reasons.append("Volume spike on crossover")
            confidence += 0.05
        if check_bullish_alignment(latest):
            reasons.append("Bullish MA alignment")
            confidence += 0.05

        sl_ref = support if support else low_24h
        sl = sl_ref * (1 - buffer_pct)
        risk = price - sl
        if risk > 0:
            tp = price + (risk * Config.RISK_REWARD_RATIO)
            sig = Signal(
                direction="LONG", entry_price=price,
                stop_loss=round(sl, 1), take_profit=round(tp, 1),
                reason=" | ".join(reasons), confidence=min(confidence, 1.0)
            )
            if best_signal is None or sig.confidence > best_signal.confidence:
                best_signal = sig

    # --- NEW: RSI extreme + confirmation (standalone) ---
    if best_signal is None:
        if rsi < 25 and bounce_candle:
            reasons = [f"RSI deeply oversold ({rsi:.1f})", "Bounce candle"]
            confidence = 0.6
            if volume_spike:
                reasons.append("Volume spike")
                confidence += 0.15
            if has_higher_lows:
                reasons.append("Higher lows pattern")
                confidence += 0.1
            sl_ref = support if support else low_24h
            sl = sl_ref * (1 - buffer_pct)
            risk = price - sl
            if risk > 0:
                tp = price + (risk * Config.RISK_REWARD_RATIO)
                best_signal = Signal(
                    direction="LONG", entry_price=price,
                    stop_loss=round(sl, 1), take_profit=round(tp, 1),
                    reason=" | ".join(reasons), confidence=min(confidence, 1.0)
                )

        elif rsi > 75 and rejection_candle:
            reasons = [f"RSI deeply overbought ({rsi:.1f})", "Rejection candle"]
            confidence = 0.6
            if volume_spike:
                reasons.append("Volume spike")
                confidence += 0.15
            if has_lower_highs:
                reasons.append("Lower highs pattern")
                confidence += 0.1
            sl_ref = resistance if resistance else latest["ma99"]
            sl = sl_ref * (1 + buffer_pct)
            risk = sl - price
            if risk > 0:
                tp = price - (risk * Config.RISK_REWARD_RATIO)
                best_signal = Signal(
                    direction="SHORT", entry_price=price,
                    stop_loss=round(sl, 1), take_profit=round(tp, 1),
                    reason=" | ".join(reasons), confidence=min(confidence, 1.0)
                )

    # --- Log result ---
    if best_signal:
        icon = "🟢" if best_signal.direction == "LONG" else "🔴"
        logger.info(f"{icon} {best_signal.direction} signal: {best_signal.reason}")
        logger.info(
            f"   Entry: {best_signal.entry_price:.1f} | SL: {best_signal.stop_loss:.1f} "
            f"| TP: {best_signal.take_profit:.1f} | Conf: {best_signal.confidence:.0%}"
        )
        return best_signal

    # No signal
    logger.debug(
        f"No signal | Price: {price:.1f} | RSI: {rsi:.1f} | MA7: {latest['ma7']:.1f} "
        f"MA25: {latest['ma25']:.1f} MA99: {latest['ma99']:.1f} | 24hLow: {low_24h:.1f}"
    )
    return None
