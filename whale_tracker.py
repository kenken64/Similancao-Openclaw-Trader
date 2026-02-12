"""Whale / Smart Money tracking via Binance Futures data endpoints."""
import logging
import time
import requests
from config import Config

logger = logging.getLogger(__name__)

_session = requests.Session()
_base_url = Config.PROXY_URL.rstrip("/") if Config.PROXY_URL else "https://fapi.binance.com"
if Config.PROXY_API_KEY:
    _session.headers.update({"X-Proxy-Api-Key": Config.PROXY_API_KEY})

# Paths that returned 404 — skip them to avoid spamming logs
_dead_paths: set = set()


def _timestamp_params(params: dict = None) -> dict:
    p = params or {}
    p["timestamp"] = int(time.time() * 1000)
    return p


def _get(path: str, params: dict) -> list:
    """GET with retry, returns JSON list or empty list on failure."""
    if path in _dead_paths:
        return []
    for attempt in range(3):
        try:
            resp = _session.get(f"{_base_url}{path}", params=params)
            if resp.status_code == 404:
                logger.info(f"Whale tracker: {path} not available (404) — disabling")
                _dead_paths.add(path)
                return []
            if resp.status_code == 502 and attempt < 2:
                time.sleep(2)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            logger.warning(f"Whale tracker GET {path} failed: {e}")
            return []


def get_whale_signals() -> dict:
    """Fetch whale/smart-money data and return a signal dict.

    Returns dict with keys:
        top_trader_long_pct, taker_buy_ratio, oi_change_pct, signal
    """
    symbol = Config.SYMBOL
    base_params = {"symbol": symbol, "period": "15m", "limit": 5}

    # Fetch all endpoints
    position_ratio = _get("/futures/data/topLongShortPositionRatio", _timestamp_params(dict(base_params)))
    account_ratio = _get("/futures/data/topLongShortAccountRatio", _timestamp_params(dict(base_params)))
    taker_ratio = _get("/futures/data/takerlongshortRatio", _timestamp_params(dict(base_params)))
    oi_hist = _get("/futures/data/openInterestHist", _timestamp_params(dict(base_params)))

    # Parse top trader long percentage (average of position and account ratios)
    top_trader_long_pct = 50.0
    try:
        ratios = []
        if position_ratio:
            ratios.append(float(position_ratio[-1].get("longAccount", 0.5)) * 100)
        if account_ratio:
            ratios.append(float(account_ratio[-1].get("longAccount", 0.5)) * 100)
        if ratios:
            top_trader_long_pct = sum(ratios) / len(ratios)
    except (IndexError, KeyError, ValueError) as e:
        logger.warning(f"Whale: failed to parse top trader ratio: {e}")

    # Parse taker buy/sell ratio
    taker_buy_ratio = 1.0
    try:
        if taker_ratio:
            taker_buy_ratio = float(taker_ratio[-1].get("buySellRatio", 1.0))
    except (IndexError, KeyError, ValueError) as e:
        logger.warning(f"Whale: failed to parse taker ratio: {e}")

    # Parse open interest change
    oi_change_pct = 0.0
    try:
        if oi_hist and len(oi_hist) >= 2:
            oi_latest = float(oi_hist[-1].get("sumOpenInterest", 0))
            oi_oldest = float(oi_hist[0].get("sumOpenInterest", 0))
            if oi_oldest > 0:
                oi_change_pct = ((oi_latest - oi_oldest) / oi_oldest) * 100
    except (IndexError, KeyError, ValueError) as e:
        logger.warning(f"Whale: failed to parse OI change: {e}")

    # Determine signal
    if top_trader_long_pct > 55 and taker_buy_ratio > 1.0:
        signal = "BULLISH"
    elif top_trader_long_pct < 45 and taker_buy_ratio < 1.0:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    result = {
        "top_trader_long_pct": round(top_trader_long_pct, 1),
        "taker_buy_ratio": round(taker_buy_ratio, 3),
        "oi_change_pct": round(oi_change_pct, 2),
        "signal": signal,
    }
    logger.info(
        f"🐋 Whale signals: Long={result['top_trader_long_pct']}% | "
        f"Taker={result['taker_buy_ratio']} | OI Δ={result['oi_change_pct']}% | {signal}"
    )
    return result
