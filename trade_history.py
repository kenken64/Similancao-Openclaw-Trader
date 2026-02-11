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

            # OpenClaw advisor log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advisor_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    confidence REAL,
                    signal_reason TEXT,
                    funding_rate REAL,
                    approved BOOLEAN NOT NULL,
                    advisor_reason TEXT,
                    response_time_ms INTEGER,
                    trade_executed BOOLEAN DEFAULT FALSE,
                    trade_id INTEGER,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
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
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_advisor_timestamp
                ON advisor_log(timestamp DESC)
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

    def get_recent_trades(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get recent trades"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                ORDER BY entry_time DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_count(self) -> int:
        """Get total number of trades"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            return cursor.fetchone()[0]

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

    def log_advisor_decision(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        signal_reason: str,
        funding_rate: float,
        approved: bool,
        advisor_reason: str,
        response_time_ms: int = 0
    ) -> int:
        """Log OpenClaw advisor decision"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO advisor_log (
                    timestamp, symbol, direction, entry_price, stop_loss,
                    take_profit, confidence, signal_reason, funding_rate,
                    approved, advisor_reason, response_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(), symbol, direction, entry_price, stop_loss,
                take_profit, confidence, signal_reason, funding_rate,
                approved, advisor_reason, response_time_ms
            ))
            advisor_log_id = cursor.lastrowid

            status = "✅ APPROVED" if approved else "❌ REJECTED"
            logger.info(f"📝 Advisor decision logged #{advisor_log_id}: {status} - {advisor_reason}")
            return advisor_log_id

    def link_advisor_to_trade(self, advisor_log_id: int, trade_id: int):
        """Link an advisor log entry to an executed trade"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE advisor_log
                SET trade_executed = TRUE, trade_id = ?
                WHERE id = ?
            """, (trade_id, advisor_log_id))

    def get_recent_advisor_decisions(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get recent OpenClaw advisor decisions"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM advisor_log
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def get_advisor_decisions_count(self) -> int:
        """Get total number of advisor decisions"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM advisor_log")
            return cursor.fetchone()[0]

    def get_advisor_statistics(self) -> Dict[str, Any]:
        """Get advisor decision statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_decisions,
                    SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) as approved_count,
                    SUM(CASE WHEN approved = 0 THEN 1 ELSE 0 END) as rejected_count,
                    SUM(CASE WHEN trade_executed = 1 THEN 1 ELSE 0 END) as executed_count,
                    AVG(response_time_ms) as avg_response_time
                FROM advisor_log
            """)
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return {
                    "total_decisions": 0,
                    "approved_count": 0,
                    "rejected_count": 0,
                    "executed_count": 0,
                    "approval_rate": 0.0,
                    "execution_rate": 0.0,
                    "avg_response_time_ms": 0
                }

            total = row[0]
            approved = row[1] or 0
            rejected = row[2] or 0
            executed = row[3] or 0

            return {
                "total_decisions": total,
                "approved_count": approved,
                "rejected_count": rejected,
                "executed_count": executed,
                "approval_rate": (approved / total * 100) if total > 0 else 0.0,
                "execution_rate": (executed / approved * 100) if approved > 0 else 0.0,
                "avg_response_time_ms": int(row[4] or 0)
            }
