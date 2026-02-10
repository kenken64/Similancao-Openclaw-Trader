"""Risk management: position sizing, drawdown tracking, funding rate checks."""
import logging
from typing import Optional, Dict
from config import Config

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.has_open_position = False
        self.dry_run_position: Optional[Dict] = None  # Track simulated position

    def update_balance(self, current_balance: float):
        """Track peak balance and check max drawdown."""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        drawdown_pct = ((self.peak_balance - current_balance) / self.peak_balance) * 100
        if drawdown_pct >= Config.MAX_DRAWDOWN_PCT:
            logger.critical(
                f"🛑 MAX DRAWDOWN HIT: {drawdown_pct:.1f}% (limit: {Config.MAX_DRAWDOWN_PCT}%) — HALTING"
            )
            return False  # Should stop trading
        if drawdown_pct > Config.MAX_DRAWDOWN_PCT * 0.7:
            logger.warning(f"⚠️ Drawdown warning: {drawdown_pct:.1f}%")
        return True

    def calculate_quantity(self, price: float) -> float:
        """Calculate order quantity from notional position size."""
        qty = Config.POSITION_SIZE_USDT / price
        # Round to Binance precision (3 decimals for BTC)
        return round(qty, 3)

    def can_open_position(self) -> bool:
        """Check if we're allowed to open a new position."""
        if self.has_open_position:
            logger.info("⏳ Already have an open position — skipping signal")
            return False
        return True

    def check_funding_rate(self, funding_rate: float, direction: str) -> bool:
        """Warn if funding rate works against the trade direction.
        Returns True if OK to trade, False if funding is dangerously high."""
        rate = abs(funding_rate)
        if rate > 0.01:  # >1% funding — extreme
            logger.warning(f"⚠️ Extreme funding rate: {funding_rate:.4f} — skipping trade")
            return False
        # Warn if funding works against us
        if direction == "SHORT" and funding_rate < -0.005:
            logger.warning(f"⚠️ Negative funding ({funding_rate:.4f}) = shorts pay longs")
        elif direction == "LONG" and funding_rate > 0.005:
            logger.warning(f"⚠️ High positive funding ({funding_rate:.4f}) = longs pay shorts")
        return True

    def set_position_open(self, direction: str, entry: float, sl: float, tp: float, qty: float):
        self.has_open_position = True
        self.dry_run_position = {
            "direction": direction, "entry": entry,
            "sl": sl, "tp": tp, "qty": qty
        }

    def set_position_closed(self):
        self.has_open_position = False
        self.dry_run_position = None

    def check_dry_run_exit(self, current_price: float) -> Optional[str]:
        """In dry-run mode, check if SL or TP is hit. Returns 'SL', 'TP', or None."""
        pos = self.dry_run_position
        if not pos:
            return None

        if pos["direction"] == "SHORT":
            if current_price >= pos["sl"]:
                return "SL"
            if current_price <= pos["tp"]:
                return "TP"
        elif pos["direction"] == "LONG":
            if current_price <= pos["sl"]:
                return "SL"
            if current_price >= pos["tp"]:
                return "TP"
        return None
