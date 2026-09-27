"""
Unit tests for the SQLite Database layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.database import Database
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


def test_database_initialization(temp_db: Database) -> None:
    """Assert all tables are created properly."""
    with temp_db.get_connection() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row["name"] for row in cur.fetchall()]
        expected_tables = [
            "candles",
            "config_history",
            "decisions",
            "live_trades",
            "paper_portfolios",
            "paper_trades",
            "portfolio_snapshots",
            "raw_ticks",
            "strategy_runs",
        ]
        for t in expected_tables:
            assert t in tables, f"Expected table '{t}' was not found in database."


def test_candle_insert_and_retrieval(temp_db: Database) -> None:
    """Test batch insertion, duplicate ignoring, and range queries."""
    candles = [
        Candle(pair="BTC-USD", timestamp=1000.0, open=100.0, high=105.0, low=99.0, close=102.0, volume=10.0),
        Candle(pair="BTC-USD", timestamp=1060.0, open=102.0, high=108.0, low=101.0, close=107.0, volume=12.0),
        Candle(pair="BTC-USD", timestamp=1120.0, open=107.0, high=110.0, low=106.0, close=109.0, volume=15.0),
    ]
    inserted = temp_db.insert_candles(candles)
    assert inserted == 3

    # Re-inserting exact duplicates should be ignored
    inserted_again = temp_db.insert_candles(candles)
    assert inserted_again == 0

    df = temp_db.get_candles_df("BTC-USD", start_ts=1000.0, end_ts=1120.0)
    assert len(df) == 3
    assert list(df["close"]) == [102.0, 107.0, 109.0]

    # Test get_latest_candles_df with before_ts cutoff
    latest_cutoff = temp_db.get_latest_candles_df("BTC-USD", n=2, before_ts=1060.0)
    assert len(latest_cutoff) == 2
    assert float(latest_cutoff.iloc[-1]["timestamp"]) == 1060.0


def test_missing_candle_ranges(temp_db: Database) -> None:
    """Test missing time gap detection in stored candles."""
    candles = [
        Candle(pair="BTC-USD", timestamp=1000.0, open=100, high=101, low=99, close=100, volume=1),
        Candle(pair="BTC-USD", timestamp=1060.0, open=100, high=101, low=99, close=100, volume=1),
        # Gap here: 1120, 1180 missing
        Candle(pair="BTC-USD", timestamp=1240.0, open=100, high=101, low=99, close=100, volume=1),
    ]
    temp_db.insert_candles(candles)

    # Search for gaps between 900 and 1300
    gaps = temp_db.find_missing_candle_ranges("BTC-USD", start_ts=940.0, end_ts=1300.0, granularity=60)
    assert len(gaps) >= 2
    # Should flag the start gap and the internal gap
    assert any(g[0] <= 940.0 for g in gaps)
    assert any(g[0] <= 1120.0 <= g[1] for g in gaps)


def test_raw_ticks_and_pruning(temp_db: Database) -> None:
    """Test tick recording and pruning older than retention window."""
    now_ts = datetime.now(timezone.utc).timestamp()
    old_ts = now_ts - (10 * 86400)  # 10 days ago

    temp_db.insert_tick(RawTick(pair="BTC-USD", timestamp=old_ts, price=60000.0))
    temp_db.insert_tick(RawTick(pair="BTC-USD", timestamp=now_ts, price=65000.0))

    pruned = temp_db.prune_raw_ticks(retention_days=7)
    assert pruned == 1


def test_portfolio_lifecycle(temp_db: Database) -> None:
    """Test portfolio creation, update, and snapshotting."""
    port = temp_db.get_or_create_portfolio(
        run_id="run-1", strategy_id="jev", initial_cash=10000.0, coin_price=50000.0
    )
    assert port.cash_usd == 10000.0
    assert port.coin_quantity == 0.0
    assert port.total_value_usd == 10000.0

    # Update portfolio after trade
    port.cash_usd = 9000.0
    port.coin_quantity = 0.02
    port.avg_entry_price = 50000.0
    temp_db.update_portfolio(port)

    # Re-fetch and verify persistence
    port_reloaded = temp_db.get_or_create_portfolio(
        run_id="run-1", strategy_id="jev", initial_cash=10000.0, coin_price=51000.0
    )
    assert port_reloaded.cash_usd == 9000.0
    assert port_reloaded.coin_quantity == 0.02
    assert port_reloaded.total_value_usd == 9000.0 + (0.02 * 51000.0)


def test_decisions_and_trades_logging(temp_db: Database) -> None:
    """Test recording decisions and trades."""
    now = datetime.now(timezone.utc)
    dummy_state = MarketState(
        pair="BTC-USD",
        timestamp=now,
        current_price=65000.0,
    )
    signal = StrategySignal(
        strategy_id="jev", action="buy", conviction=8.0, risk_level=3.0
    )
    risk = RiskDecision(approved=True, outcome="APPROVED")

    dec_id = temp_db.record_decision(
        run_id="run-1",
        strategy_id="jev",
        pair="BTC-USD",
        timestamp=now,
        market_state=dummy_state,
        signal=signal,
        risk=risk,
        mode="paper",
    )
    assert dec_id > 0

    trade = TradeResult(
        id=dec_id,
        run_id="run-1",
        strategy_id="jev",
        pair="BTC-USD",
        side="buy",
        quantity=0.01,
        fill_price=65000.0,
        fill_value_usd=650.0,
        fee_usd=3.9,
        timestamp=now,
        portfolio_value_after=10000.0,
    )
    trade_id = temp_db.record_trade(trade)
    assert trade_id > 0

    trades = temp_db.get_run_trades("run-1", "jev")
    assert len(trades) == 1
    assert trades[0].quantity == 0.01
    assert trades[0].fill_price == 65000.0
