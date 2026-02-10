"""Trade history tracking with SQLite database"""
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TradeHistory:
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_type TEXT,
                    pnl REAL,
                    pnl_percentage REAL,
                    mode TEXT NOT NULL,
                    reason TEXT,
                    advisor_approved BOOLEAN,
                    advisor_reason TEXT,
                    funding_rate REAL,
                    confidence REAL,
                    status TEXT DEFAULT 'open'
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entry_time
                ON trades(entry_time DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON trades(status)
            """)

            logger.info(f"✅ Trade history database initialized: {self.db_path}")

    def record_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        mode: str,
        reason: str,
        advisor_approved: bool = True,
        advisor_reason: str = None,
        funding_rate: float = 0.0,
        confidence: float = 0.0
    ) -> int:
        """Record a new trade entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    symbol, direction, entry_time, entry_price, quantity,
                    stop_loss, take_profit, mode, reason, advisor_approved,
                    advisor_reason, funding_rate, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """, (
                symbol, direction, datetime.now(), entry_price, quantity,
                stop_loss, take_profit, mode, reason, advisor_approved,
                advisor_reason, funding_rate, confidence
            ))
            trade_id = cursor.lastrowid
            logger.info(f"📝 Trade #{trade_id} recorded: {direction} {symbol} @ {entry_price}")
            return trade_id

    def record_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_type: str,
        pnl: float = None
    ):
        """Record trade exit"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get trade details to calculate PnL if not provided
            cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            trade = cursor.row_factory(cursor, cursor.fetchone())

            if not trade:
                logger.error(f"Trade #{trade_id} not found")
                return

            # Calculate PnL if not provided
            if pnl is None:
                if trade['direction'] == 'LONG':
                    pnl = (exit_price - trade['entry_price']) * trade['quantity']
                else:  # SHORT
                    pnl = (trade['entry_price'] - exit_price) * trade['quantity']

            pnl_percentage = (pnl / (trade['entry_price'] * trade['quantity'])) * 100

            cursor.execute("""
                UPDATE trades
                SET exit_time = ?, exit_price = ?, exit_type = ?,
                    pnl = ?, pnl_percentage = ?, status = 'closed'
                WHERE id = ?
            """, (datetime.now(), exit_price, exit_type, pnl, pnl_percentage, trade_id))

            emoji = "🎯" if pnl > 0 else "🛑"
            logger.info(f"{emoji} Trade #{trade_id} closed: {exit_type} @ {exit_price} | PnL: {pnl:+.2f} USDT ({pnl_percentage:+.2f}%)")

    def get_open_trade(self) -> Optional[Dict[str, Any]]:
        """Get currently open trade"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                WHERE status = 'open'
                ORDER BY entry_time DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                ORDER BY entry_time DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total trades
            cursor.execute("SELECT COUNT(*) as total FROM trades WHERE status = 'closed'")
            total_trades = cursor.fetchone()[0]

            if total_trades == 0:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0
                }

            # Win/Loss statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                WHERE status = 'closed'
            """)
            row = cursor.fetchone()

            return {
                "total_trades": row[0],
                "winning_trades": row[1] or 0,
                "losing_trades": row[2] or 0,
                "win_rate": (row[1] / row[0] * 100) if row[0] > 0 else 0.0,
                "total_pnl": row[3] or 0.0,
                "avg_pnl": row[4] or 0.0,
                "best_trade": row[5] or 0.0,
                "worst_trade": row[6] or 0.0
            }

    def get_trades_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get trades within date range"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                WHERE entry_time BETWEEN ? AND ?
                ORDER BY entry_time DESC
            """, (start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]
