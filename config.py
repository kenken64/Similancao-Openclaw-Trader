"""Configuration loader from .env"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Binance API
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    # Binance API Proxy
    PROXY_URL = os.getenv("BINANCE_PROXY_URL", "")
    PROXY_API_KEY = os.getenv("BINANCE_PROXY_API_KEY", "")

    # Trading
    SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
    TIMEFRAME = os.getenv("TIMEFRAME", "15m")
    LEVERAGE = int(os.getenv("LEVERAGE", "25"))
    POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "5000"))

    # MA Periods
    MA_FAST = int(os.getenv("MA_FAST", "7"))
    MA_MID = int(os.getenv("MA_MID", "25"))
    MA_SLOW = int(os.getenv("MA_SLOW", "99"))

    # Risk
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "20"))
    RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", "2.0"))
    STOP_LOSS_BUFFER_PCT = float(os.getenv("STOP_LOSS_BUFFER_PCT", "0.15"))

    # RSI
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

    # Volume
    VOLUME_FILTER_RATIO = float(os.getenv("VOLUME_FILTER_RATIO", "0.8"))

    # Bot
    CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "15"))
    DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Alerts
    OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
    OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
    OPENCLAW_ALERT_ENABLED = os.getenv("OPENCLAW_ALERT_ENABLED", "true").lower() == "true"

    # Advisor Mode — ask Similancao before trading
    ADVISOR_MODE = os.getenv("ADVISOR_MODE", "true").lower() == "true"
    ADVISOR_TIMEOUT_SECONDS = int(os.getenv("ADVISOR_TIMEOUT_SECONDS", "120"))
