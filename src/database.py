"""
SQLite Database Layer for Candle Caching, Decisions, Portfolio State, and Backtest Runs.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger

from src.models import (
    Candle,
    MarketState,
    PortfolioSnapshot,
    RawTick,
    RiskDecision,
    StrategyRunSummary,
    StrategySignal,
    TradeResult,
)


class Database:
    """Thread-safe SQLite database manager for the trading system."""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create all required tables and indexes idempotently."""
        with self._lock, self.get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS candles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pair        TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                open        REAL NOT NULL,
                high        REAL NOT NULL,
                low         REAL NOT NULL,
                close       REAL NOT NULL,
                volume      REAL NOT NULL,
                source      TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_pair_ts ON candles (pair, timestamp);

            CREATE TABLE IF NOT EXISTS raw_ticks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pair        TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                price       REAL NOT NULL,
                volume_24h  REAL,
                best_bid    REAL,
                best_ask    REAL,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_raw_ticks_pair_ts ON raw_ticks (pair, timestamp);

            CREATE TABLE IF NOT EXISTS strategy_runs (
                run_id          TEXT PRIMARY KEY,
                strategy_id     TEXT NOT NULL,
                mode            TEXT NOT NULL,
                pair            TEXT NOT NULL,
                start_time      TEXT NOT NULL,
                end_time        TEXT,
                initial_cash    REAL NOT NULL,
                final_value     REAL,
                total_trades    INTEGER DEFAULT 0,
                winning_trades  INTEGER DEFAULT 0,
                losing_trades   INTEGER DEFAULT 0,
                win_rate        REAL DEFAULT 0.0,
                total_pnl_pct   REAL DEFAULT 0.0,
                max_drawdown    REAL DEFAULT 0.0,
                profit_factor   REAL DEFAULT 0.0,
                sharpe_ratio    REAL DEFAULT 0.0,
                config_snapshot TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                strategy_id     TEXT NOT NULL,
                pair            TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                market_state    TEXT NOT NULL,
                action          TEXT NOT NULL,
                conviction      REAL NOT NULL,
                risk_level      REAL NOT NULL,
                risk_outcome    TEXT NOT NULL,
                risk_reason     TEXT,
                mode            TEXT NOT NULL,
                latency_ms      REAL DEFAULT 0.0,
                raw_response    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_run_strat ON decisions (run_id, strategy_id);

            CREATE TABLE IF NOT EXISTS paper_trades (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                TEXT NOT NULL,
                strategy_id           TEXT NOT NULL,
                decision_id           INTEGER,
                pair                  TEXT NOT NULL,
                side                  TEXT NOT NULL,
                quantity              REAL NOT NULL,
                fill_price            REAL NOT NULL,
                fill_value_usd        REAL NOT NULL,
                fee_usd               REAL NOT NULL,
                slippage_pct          REAL DEFAULT 0.0,
                entry_trade_id        INTEGER,
                pnl_usd               REAL,
                pnl_pct               REAL,
                timestamp             TEXT NOT NULL,
                portfolio_value_after REAL
            );
            CREATE INDEX IF NOT EXISTS idx_paper_trades_run ON paper_trades (run_id, strategy_id);

            CREATE TABLE IF NOT EXISTS live_trades (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                pair                TEXT NOT NULL,
                decision_id         INTEGER,
                coinbase_order_id   TEXT NOT NULL UNIQUE,
                side                TEXT NOT NULL,
                status              TEXT NOT NULL,
                quantity_requested  REAL NOT NULL,
                quantity_filled     REAL,
                avg_fill_price      REAL,
                total_fees_usd      REAL,
                entry_trade_id      INTEGER,
                pnl_usd             REAL,
                pnl_pct             REAL,
                submitted_at        TEXT NOT NULL,
                filled_at           TEXT,
                raw_response        TEXT
            );

            CREATE TABLE IF NOT EXISTS paper_portfolios (
                run_id          TEXT NOT NULL,
                strategy_id     TEXT NOT NULL,
                cash_usd        REAL NOT NULL,
                coin_quantity   REAL NOT NULL DEFAULT 0.0,
                avg_entry_price REAL,
                open_trade_id   INTEGER,
                high_water_mark REAL NOT NULL,
                total_trades    INTEGER DEFAULT 0,
                updated_at      TEXT NOT NULL,
                PRIMARY KEY (run_id, strategy_id)
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                strategy_id     TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                cash_usd        REAL NOT NULL,
                coin_quantity   REAL NOT NULL,
                coin_price      REAL NOT NULL,
                total_value_usd REAL NOT NULL,
                daily_pnl_usd   REAL,
                daily_pnl_pct   REAL
            );
            CREATE INDEX IF NOT EXISTS idx_portfolio_snap_run ON portfolio_snapshots (run_id, timestamp);

            CREATE TABLE IF NOT EXISTS config_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                config_json TEXT NOT NULL,
                reason      TEXT
            );
            """)

    # ---------------- Candle Operations ----------------
    def insert_candles(self, candles: List[Candle]) -> int:
        """Insert a batch of candles, ignoring duplicates. Returns number of inserted candles."""
        if not candles:
            return 0
        rows = [
            (
                c.pair,
                c.timestamp,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.source,
            )
            for c in candles
        ]
        with self._lock, self.get_connection() as conn:
            cur = conn.executemany(
                """
                INSERT OR IGNORE INTO candles (pair, timestamp, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return cur.rowcount

    def get_candles_df(
        self,
        pair: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch candles for a pair as a sorted pandas DataFrame."""
        query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE pair = ?"
        params: List[Any] = [pair]
        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)
        query += " ORDER BY timestamp ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._lock, self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df.set_index("timestamp_dt", inplace=False)
        return df

    def get_latest_candles_df(
        self, pair: str, n: int = 200, before_ts: Optional[float] = None
    ) -> pd.DataFrame:
        """Get the last N candles up to an optional timestamp (strictly no lookahead)."""
        query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE pair = ?"
        params: List[Any] = [pair]
        if before_ts is not None:
            query += " AND timestamp <= ?"
            params.append(before_ts)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(n)

        with self._lock, self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df

    def get_candle_time_range(self, pair: str) -> Tuple[Optional[float], Optional[float]]:
        """Return (min_ts, max_ts) of stored candles for a pair."""
        with self._lock, self.get_connection() as conn:
            cur = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE pair = ?",
                (pair,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return float(row[0]), float(row[1])
        return None, None

    def find_missing_candle_ranges(
        self, pair: str, start_ts: float, end_ts: float, granularity: int = 60
    ) -> List[Tuple[float, float]]:
        """Identify missing time intervals in SQLite candles within [start_ts, end_ts]."""
        with self._lock, self.get_connection() as conn:
            cur = conn.execute(
                """
                SELECT timestamp FROM candles
                WHERE pair = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (pair, start_ts, end_ts),
            )
            timestamps = [row["timestamp"] for row in cur.fetchall()]

        if not timestamps:
            return [(start_ts, end_ts)]

        missing_ranges: List[Tuple[float, float]] = []
        # Check start gap
        if timestamps[0] - start_ts >= granularity:
            missing_ranges.append((start_ts, timestamps[0] - granularity))

        # Check internal gaps
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap > granularity * 1.5:
                missing_ranges.append((timestamps[i] + granularity, timestamps[i + 1] - granularity))

        # Check end gap
        if end_ts - timestamps[-1] >= granularity:
            missing_ranges.append((timestamps[-1] + granularity, end_ts))

        return missing_ranges

    # ---------------- Raw Ticks ----------------
    def insert_tick(self, tick: RawTick) -> None:
        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO raw_ticks (pair, timestamp, price, volume_24h, best_bid, best_ask)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tick.pair,
                    tick.timestamp,
                    tick.price,
                    tick.volume_24h,
                    tick.best_bid,
                    tick.best_ask,
                ),
            )
            conn.commit()

    def prune_raw_ticks(self, retention_days: int = 7) -> int:
        cutoff = datetime.now(timezone.utc).timestamp() - (retention_days * 86400)
        with self._lock, self.get_connection() as conn:
            cur = conn.execute("DELETE FROM raw_ticks WHERE timestamp < ?", (cutoff,))
            conn.commit()
            return cur.rowcount

    # ---------------- Strategy Runs ----------------
    def record_strategy_run(self, summary: StrategyRunSummary) -> None:
        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO strategy_runs (
                    run_id, strategy_id, mode, pair, start_time, end_time, initial_cash,
                    final_value, total_trades, winning_trades, losing_trades, win_rate,
                    total_pnl_pct, max_drawdown, profit_factor, sharpe_ratio, config_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    end_time=excluded.end_time,
                    final_value=excluded.final_value,
                    total_trades=excluded.total_trades,
                    winning_trades=excluded.winning_trades,
                    losing_trades=excluded.losing_trades,
                    win_rate=excluded.win_rate,
                    total_pnl_pct=excluded.total_pnl_pct,
                    max_drawdown=excluded.max_drawdown,
                    profit_factor=excluded.profit_factor,
                    sharpe_ratio=excluded.sharpe_ratio
                """,
                (
                    summary.run_id,
                    summary.strategy_id,
                    summary.mode,
                    summary.pair,
                    summary.start_time,
                    summary.end_time,
                    summary.initial_cash,
                    summary.final_value,
                    summary.total_trades,
                    summary.winning_trades,
                    summary.losing_trades,
                    summary.win_rate,
                    summary.total_pnl_pct,
                    summary.max_drawdown,
                    summary.profit_factor,
                    summary.sharpe_ratio,
                    summary.config_snapshot,
                ),
            )
            conn.commit()

    def get_strategy_run(self, run_id: str) -> Optional[StrategyRunSummary]:
        with self._lock, self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM strategy_runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            if row:
                return StrategyRunSummary(**dict(row))
        return None

    def list_strategy_runs(self, limit: int = 50) -> List[StrategyRunSummary]:
        with self._lock, self.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM strategy_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return [StrategyRunSummary(**dict(row)) for row in cur.fetchall()]

    # ---------------- Decisions ----------------
    def record_decision(
        self,
        run_id: str,
        strategy_id: str,
        pair: str,
        timestamp: datetime,
        market_state: MarketState,
        signal: StrategySignal,
        risk: RiskDecision,
        mode: str,
    ) -> int:
        with self._lock, self.get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO decisions (
                    run_id, strategy_id, pair, timestamp, market_state, action,
                    conviction, risk_level, risk_outcome, risk_reason, mode, latency_ms, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    strategy_id,
                    pair,
                    timestamp.isoformat(),
                    market_state.model_dump_json(),
                    signal.action,
                    signal.conviction,
                    signal.risk_level,
                    risk.outcome,
                    risk.reason,
                    mode,
                    signal.latency_ms,
                    json.dumps(signal.raw_response) if signal.raw_response else None,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    # ---------------- Trades ----------------
    def record_trade(self, trade: TradeResult) -> int:
        with self._lock, self.get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO paper_trades (
                    run_id, strategy_id, decision_id, pair, side, quantity,
                    fill_price, fill_value_usd, fee_usd, slippage_pct, entry_trade_id,
                    pnl_usd, pnl_pct, timestamp, portfolio_value_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.run_id,
                    trade.strategy_id,
                    trade.id,
                    trade.pair,
                    trade.side,
                    trade.quantity,
                    trade.fill_price,
                    trade.fill_value_usd,
                    trade.fee_usd,
                    trade.slippage_pct,
                    trade.entry_trade_id,
                    trade.pnl_usd,
                    trade.pnl_pct,
                    trade.timestamp.isoformat(),
                    trade.portfolio_value_after,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def get_run_trades(
        self, run_id: str, strategy_id: Optional[str] = None
    ) -> List[TradeResult]:
        query = "SELECT * FROM paper_trades WHERE run_id = ?"
        params: List[Any] = [run_id]
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY id ASC"

        with self._lock, self.get_connection() as conn:
            cur = conn.execute(query, params)
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["timestamp"] = datetime.fromisoformat(d["timestamp"])
                results.append(TradeResult(**d))
            return results

    # ---------------- Portfolio Operations ----------------
    def get_or_create_portfolio(
        self, run_id: str, strategy_id: str, initial_cash: float, coin_price: float = 0.0
    ) -> PortfolioSnapshot:
        with self._lock, self.get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM paper_portfolios WHERE run_id = ? AND strategy_id = ?",
                (run_id, strategy_id),
            )
            row = cur.fetchone()
            now = datetime.now(timezone.utc)
            if row:
                coin_qty = float(row["coin_quantity"])
                cash = float(row["cash_usd"])
                tot = cash + (coin_qty * coin_price)
                return PortfolioSnapshot(
                    run_id=run_id,
                    strategy_id=strategy_id,
                    timestamp=datetime.fromisoformat(row["updated_at"]),
                    cash_usd=cash,
                    coin_quantity=coin_qty,
                    coin_price=coin_price,
                    avg_entry_price=row["avg_entry_price"],
                    open_trade_id=row["open_trade_id"],
                    total_value_usd=tot,
                    high_water_mark=float(row["high_water_mark"]),
                    total_trades=int(row["total_trades"] or 0),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO paper_portfolios (
                        run_id, strategy_id, cash_usd, coin_quantity, high_water_mark, total_trades, updated_at
                    ) VALUES (?, ?, ?, 0.0, ?, 0, ?)
                    """,
                    (run_id, strategy_id, initial_cash, initial_cash, now.isoformat()),
                )
                conn.commit()
                return PortfolioSnapshot(
                    run_id=run_id,
                    strategy_id=strategy_id,
                    timestamp=now,
                    cash_usd=initial_cash,
                    coin_quantity=0.0,
                    coin_price=coin_price,
                    total_value_usd=initial_cash,
                    high_water_mark=initial_cash,
                    total_trades=0,
                )

    def update_portfolio(self, portfolio: PortfolioSnapshot) -> None:
        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO paper_portfolios (
                    run_id, strategy_id, cash_usd, coin_quantity, avg_entry_price,
                    open_trade_id, high_water_mark, total_trades, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, strategy_id) DO UPDATE SET
                    cash_usd=excluded.cash_usd,
                    coin_quantity=excluded.coin_quantity,
                    avg_entry_price=excluded.avg_entry_price,
                    open_trade_id=excluded.open_trade_id,
                    high_water_mark=excluded.high_water_mark,
                    total_trades=excluded.total_trades,
                    updated_at=excluded.updated_at
                """,
                (
                    portfolio.run_id,
                    portfolio.strategy_id,
                    portfolio.cash_usd,
                    portfolio.coin_quantity,
                    portfolio.avg_entry_price,
                    portfolio.open_trade_id,
                    portfolio.high_water_mark,
                    portfolio.total_trades,
                    portfolio.timestamp.isoformat(),
                ),
            )
            conn.commit()

    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_snapshots (
                    run_id, strategy_id, timestamp, cash_usd, coin_quantity,
                    coin_price, total_value_usd, daily_pnl_usd, daily_pnl_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.run_id,
                    snapshot.strategy_id,
                    snapshot.timestamp.isoformat(),
                    snapshot.cash_usd,
                    snapshot.coin_quantity,
                    snapshot.coin_price,
                    snapshot.total_value_usd,
                    snapshot.daily_pnl_usd,
                    snapshot.daily_pnl_pct,
                ),
            )
            conn.commit()

    def get_portfolio_snapshots(
        self, run_id: str, strategy_id: Optional[str] = None
    ) -> List[PortfolioSnapshot]:
        query = "SELECT * FROM portfolio_snapshots WHERE run_id = ?"
        params: List[Any] = [run_id]
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY timestamp ASC"

        with self._lock, self.get_connection() as conn:
            cur = conn.execute(query, params)
            snaps = []
            for row in cur.fetchall():
                d = dict(row)
                d["timestamp"] = datetime.fromisoformat(d["timestamp"])
                d["high_water_mark"] = d["total_value_usd"]
                snaps.append(PortfolioSnapshot(**d))
            return snaps
