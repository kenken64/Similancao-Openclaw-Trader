"""Binance Futures USDT-M API wrapper"""
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import Config

logger = logging.getLogger(__name__)


class BinanceFuturesClient:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.client = Client(Config.API_KEY, Config.API_SECRET)
        self.symbol = Config.SYMBOL
        if not dry_run:
            self._setup_leverage()

    def _setup_leverage(self):
        """Set leverage and margin type on Binance."""
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, leverage=Config.LEVERAGE
            )
            logger.info(f"✅ Leverage set to {Config.LEVERAGE}x for {self.symbol}")
        except BinanceAPIException as e:
            logger.warning(f"Leverage setup: {e}")
        try:
            self.client.futures_change_margin_type(
                symbol=self.symbol, marginType="ISOLATED"
            )
        except BinanceAPIException:
            pass  # Already set

    def get_klines(self, limit: int = 150) -> pd.DataFrame:
        """Fetch kline/candlestick data as DataFrame."""
        klines = self.client.futures_klines(
            symbol=self.symbol, interval=Config.TIMEFRAME, limit=limit
        )
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
        return self.client.futures_ticker(symbol=self.symbol)

    def get_mark_price(self) -> Dict[str, Any]:
        """Get mark price and funding rate."""
        return self.client.futures_mark_price(symbol=self.symbol)

    def get_position(self) -> Optional[Dict[str, Any]]:
        """Get current open position for symbol, or None."""
        if self.dry_run:
            return None
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
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
        except BinanceAPIException as e:
            logger.error(f"Error fetching position: {e}")
        return None

    def get_open_orders(self) -> List[Dict]:
        """Get open orders for symbol."""
        if self.dry_run:
            return []
        try:
            return self.client.futures_get_open_orders(symbol=self.symbol)
        except BinanceAPIException as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    def get_account_balance(self) -> float:
        """Get USDT futures balance."""
        if self.dry_run:
            return 10000.0  # Simulated
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["balance"])
        except BinanceAPIException as e:
            logger.error(f"Error fetching balance: {e}")
        return 0.0

    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """Place a market order. side = 'BUY' or 'SELL'."""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] Market {side} {quantity} {self.symbol}")
            return {"orderId": "dry-run", "side": side, "quantity": quantity}
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol, side=side, type="MARKET", quantity=quantity
            )
            logger.info(f"✅ Market {side} {quantity} {self.symbol} — ID: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Order failed: {e}")
            return None

    def place_stop_loss(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Place stop-market order. side should be closing side (SELL to close LONG, BUY to close SHORT)."""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] Stop-loss {side} @ {stop_price}")
            return {"orderId": "dry-run-sl"}
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol, side=side, type="STOP_MARKET",
                stopPrice=round(stop_price, 1), closePosition="true"
            )
            logger.info(f"✅ Stop-loss set @ {stop_price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Stop-loss failed: {e}")
            return None

    def place_take_profit(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Place take-profit market order."""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] Take-profit {side} @ {stop_price}")
            return {"orderId": "dry-run-tp"}
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol, side=side, type="TAKE_PROFIT_MARKET",
                stopPrice=round(stop_price, 1), closePosition="true"
            )
            logger.info(f"✅ Take-profit set @ {stop_price}")
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Take-profit failed: {e}")
            return None

    def cancel_all_orders(self):
        """Cancel all open orders for the symbol."""
        if self.dry_run:
            logger.info("🧪 [DRY-RUN] Cancel all orders")
            return
        try:
            self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            logger.info("✅ All orders cancelled")
        except BinanceAPIException as e:
            logger.error(f"❌ Cancel orders failed: {e}")
