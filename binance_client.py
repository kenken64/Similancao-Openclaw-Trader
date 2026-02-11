"""Binance Futures USDT-M API wrapper — direct HTTP via proxy"""
import hashlib
import hmac
import logging
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode
import requests
import pandas as pd
from config import Config

logger = logging.getLogger(__name__)


class BinanceFuturesClient:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.symbol = Config.SYMBOL
        self.base_url = Config.PROXY_URL.rstrip("/") if Config.PROXY_URL else "https://fapi.binance.com"
        self.session = requests.Session()
        if Config.PROXY_API_KEY:
            self.session.headers.update({"X-Proxy-Api-Key": Config.PROXY_API_KEY})
        logger.info(f"Using API endpoint: {self.base_url}")

        # Verify connectivity
        resp = self.session.get(f"{self.base_url}/fapi/v1/ping")
        resp.raise_for_status()

        if not dry_run:
            self._setup_leverage()

    def _timestamp_params(self, params: dict = None) -> dict:
        """Add timestamp only (proxy handles signing for GET requests)."""
        p = params or {}
        p["timestamp"] = int(time.time() * 1000)
        return p

    def _signed_params(self, params: dict = None) -> dict:
        """Add timestamp and HMAC signature (for POST requests the proxy doesn't sign)."""
        p = self._timestamp_params(params)
        query_string = urlencode(p)
        signature = hmac.new(
            Config.API_SECRET.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        p["signature"] = signature
        return p

    def _request_with_retry(self, method: str, path: str, **kwargs) -> Any:
        for attempt in range(3):
            resp = self.session.request(method, f"{self.base_url}{path}", **kwargs)
            if resp.status_code == 502 and attempt < 2:
                import time as _time
                _time.sleep(2)
                continue
            resp.raise_for_status()
            return resp.json()

    def _get(self, path: str, params: dict = None) -> Any:
        return self._request_with_retry("GET", path, params=params)

    def _post(self, path: str, data: dict = None) -> Any:
        return self._request_with_retry("POST", path, data=data)

    def _delete(self, path: str, params: dict = None) -> Any:
        return self._request_with_retry("DELETE", path, params=params)

    def _setup_leverage(self):
        """Set leverage and margin type on Binance."""
        try:
            self._post("/fapi/v1/leverage", self._signed_params({
                "symbol": self.symbol, "leverage": Config.LEVERAGE
            }))
            logger.info(f"✅ Leverage set to {Config.LEVERAGE}x for {self.symbol}")
        except Exception as e:
            logger.warning(f"Leverage setup: {e}")
        try:
            self._post("/fapi/v1/marginType", self._signed_params({
                "symbol": self.symbol, "marginType": "ISOLATED"
            }))
        except Exception:
            pass  # Already set

    def get_klines(self, limit: int = 150) -> pd.DataFrame:
        """Fetch kline/candlestick data as DataFrame."""
        klines = self._get("/fapi/v1/klines", {
            "symbol": self.symbol, "interval": Config.TIMEFRAME, "limit": limit
        })
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df

    def get_24h_ticker(self) -> Dict[str, Any]:
        """Get 24h price statistics."""
        return self._get("/fapi/v1/ticker/24hr", {"symbol": self.symbol})

    def get_mark_price(self) -> Dict[str, Any]:
        """Get mark price and funding rate."""
        return self._get("/fapi/v1/premiumIndex", {"symbol": self.symbol})

    def get_position(self) -> Optional[Dict[str, Any]]:
        """Get current open position for symbol, or None."""
        if self.dry_run:
            return None
        try:
            positions = self._get("/fapi/v2/positionRisk",
                                  self._timestamp_params({"symbol": self.symbol}))
            for p in positions:
                amt = float(p.get("positionAmt", 0))
                if amt != 0:
                    return {
                        "side": "LONG" if amt > 0 else "SHORT",
                        "size": abs(amt),
                        "entry_price": float(p["entryPrice"]),
                        "unrealized_pnl": float(p["unRealizedProfit"]),
                        "leverage": int(p["leverage"]),
                    }
        except Exception as e:
            logger.error(f"Error fetching position: {e}")
        return None

    def get_open_orders(self) -> List[Dict]:
        """Get open orders for symbol."""
        if self.dry_run:
            return []
        try:
            return self._get("/fapi/v1/openOrders",
                             self._timestamp_params({"symbol": self.symbol}))
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    def get_account_balance(self) -> float:
        """Get USDT futures balance."""
        if self.dry_run:
            return 10000.0  # Simulated
        return self.get_live_balance()

    def get_live_balance(self) -> float:
        """Get real USDT futures balance from Binance, regardless of dry_run mode."""
        try:
            balances = self._get("/fapi/v2/balance", self._timestamp_params())
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["balance"])
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
        return 0.0

    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """Place a market order. side = 'BUY' or 'SELL'."""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] Market {side} {quantity} {self.symbol}")
            return {"orderId": "dry-run", "side": side, "quantity": quantity}
        try:
            order = self._post("/fapi/v1/order", self._signed_params({
                "symbol": self.symbol, "side": side, "type": "MARKET",
                "quantity": quantity
            }))
            logger.info(f"✅ Market {side} {quantity} {self.symbol} — ID: {order['orderId']}")
            return order
        except Exception as e:
            logger.error(f"❌ Order failed: {e}")
            return None

    def place_stop_loss(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Place stop-market order."""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] Stop-loss {side} @ {stop_price}")
            return {"orderId": "dry-run-sl"}
        try:
            order = self._post("/fapi/v1/order", self._signed_params({
                "symbol": self.symbol, "side": side, "type": "STOP_MARKET",
                "stopPrice": round(stop_price, 1), "closePosition": "true"
            }))
            logger.info(f"✅ Stop-loss set @ {stop_price}")
            return order
        except Exception as e:
            logger.error(f"❌ Stop-loss failed: {e}")
            return None

    def place_take_profit(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Place take-profit market order."""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] Take-profit {side} @ {stop_price}")
            return {"orderId": "dry-run-tp"}
        try:
            order = self._post("/fapi/v1/order", self._signed_params({
                "symbol": self.symbol, "side": side, "type": "TAKE_PROFIT_MARKET",
                "stopPrice": round(stop_price, 1), "closePosition": "true"
            }))
            logger.info(f"✅ Take-profit set @ {stop_price}")
            return order
        except Exception as e:
            logger.error(f"❌ Take-profit failed: {e}")
            return None

    def cancel_all_orders(self):
        """Cancel all open orders for the symbol."""
        if self.dry_run:
            logger.info("🧪 [DRY-RUN] Cancel all orders")
            return
        try:
            self._delete("/fapi/v1/allOpenOrders",
                         self._timestamp_params({"symbol": self.symbol}))
            logger.info("✅ All orders cancelled")
        except Exception as e:
            logger.error(f"❌ Cancel orders failed: {e}")
