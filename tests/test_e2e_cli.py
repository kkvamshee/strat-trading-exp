"""
End-to-End (E2E) Test: CLI Execution and Command Handlers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pytest

from main import cmd_compare, cmd_replay
from src.database import Database
from src.models import StrategyRunSummary
from tests.conftest import generate_synthetic_candles


@pytest.mark.asyncio
async def test_cli_replay_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test executing the replay and compare CLI commands end-to-end."""
    db_path = str(tmp_path / "cli_test.db")
    db = Database(db_path)

    # Insert historical candles
    start_history_dt = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc)
    candles = generate_synthetic_candles(pair="BTC-USD", start_dt=start_history_dt, num_candles=500)
    db.insert_candles(candles)

    # Create temporary config
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(f"""
system:
  default_pair: "BTC-USD"
  db_path: "{db_path}"
  log_path: "{tmp_path}/bot.log"
  log_level: "ERROR"

features:
  candle_lookback: 50

risk:
  min_conviction: 6.0
  cooldown_minutes: 2

paper:
  default_cash_usd: 10000.0

replay:
  default_start_time: "{start_history_dt.timestamp() + (100 * 60)}"
""")

    # 1. Run replay via CLI handler
    start_ts = start_history_dt.timestamp() + (100 * 60)
    end_ts = start_history_dt.timestamp() + (200 * 60)
    replay_args = argparse.Namespace(
        config=str(config_file),
        strategies="ema_macd,rsi_bb",
        pair="BTC-USD",
        start_time=str(start_ts),
        end_time=str(end_ts),
        initial_cash=10000.0,
        no_backfill=True,
    )
    await cmd_replay(replay_args)

    # Verify that runs were recorded in the test DB
    runs = db.list_strategy_runs()
    assert len(runs) >= 2

    # 2. Run compare via CLI handler
    compare_args = argparse.Namespace(
        config=str(config_file),
        runs=None,
        limit=10,
    )
    await cmd_compare(compare_args)

    captured = capsys.readouterr()
    assert "Historical Strategy Runs Comparison" in captured.out or "Comparative Strategy Leaderboard" in captured.out
