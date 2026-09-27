"""
End-to-End (E2E) Test: Historical Replay & Multi-Strategy Comparative Backtesting.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.database import Database
from src.models import AppConfig
from src.replay_engine import HistoricalReplayEngine
from tests.conftest import generate_synthetic_candles


@pytest.mark.asyncio
async def test_e2e_multi_strategy_historical_replay(temp_db: Database) -> None:
    """
    Run full end-to-end historical backtest from an arbitrary start time
    across 3 strategies (jev, ema_macd, rsi_bb) on synthetic candle history.
    """
    # 1. Generate 2 days of synthetic 1-minute candles (2880 candles)
    start_history_dt = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    candles = generate_synthetic_candles(
        pair="BTC-USD", start_dt=start_history_dt, num_candles=2880, base_price=65000.0
    )
    temp_db.insert_candles(candles)

    # 2. Select replay start time: 1 day into history (so 1440 candles lookback buffer available)
    replay_start_dt = start_history_dt.timestamp() + (1440 * 60)
    replay_end_dt = start_history_dt.timestamp() + (2000 * 60)

    # 3. Configure replay engine
    config = AppConfig()
    replay_engine = HistoricalReplayEngine(db=temp_db, config=config)

    strategies_to_test = ["jev", "ema_macd", "rsi_bb"]

    # 4. Execute the multi-strategy replay
    summaries, comparison_df = await replay_engine.run_replay(
        start_time=replay_start_dt,
        end_time=replay_end_dt,
        strategies=strategies_to_test,
        pair="BTC-USD",
        initial_cash=10000.0,
        auto_backfill=False,  # Use cached candles
    )

    # 5. Assertions on results
    assert len(summaries) == 3
    assert set(summaries.keys()) == set(strategies_to_test)

    for strat_id, summary in summaries.items():
        assert summary.strategy_id == strat_id
        assert summary.initial_cash == 10000.0
        assert summary.final_value > 0.0
        assert summary.mode == "replay"
        assert summary.pair == "BTC-USD"
        # Verify persistence in strategy_runs table
        stored_summary = temp_db.get_strategy_run(summary.run_id)
        assert stored_summary is not None
        assert stored_summary.run_id == summary.run_id

    # 6. Verify decisions were written to SQLite
    with temp_db.get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*), strategy_id FROM decisions GROUP BY strategy_id")
        decision_counts = {row["strategy_id"]: row[0] for row in cur.fetchall()}
        for s in strategies_to_test:
            assert s in decision_counts
            assert decision_counts[s] > 0

    # 7. Verify comparison dataframe
    assert len(comparison_df) == 3
    assert "Strategy" in comparison_df.columns
    assert "Total P&L (%)" in comparison_df.columns
    assert "Win Rate (%)" in comparison_df.columns
