"""
Unit tests for the Performance Tracker.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.models import PortfolioSnapshot, TradeResult
from src.performance_tracker import PerformanceTracker


def test_performance_summary_calculation() -> None:
    """Test win rate, profit factor, max drawdown, and PnL calculation."""
    now = datetime.now(timezone.utc)

    # 3 trades: 2 winning, 1 losing
    trades = [
        TradeResult(
            run_id="r1",
            strategy_id="s1",
            pair="BTC-USD",
            side="buy",
            quantity=0.1,
            fill_price=50000.0,
            fill_value_usd=5000.0,
            fee_usd=30.0,
            timestamp=now,
            portfolio_value_after=10000.0,
        ),
        TradeResult(
            run_id="r1",
            strategy_id="s1",
            pair="BTC-USD",
            side="sell",
            quantity=0.1,
            fill_price=52000.0,
            fill_value_usd=5200.0,
            fee_usd=31.2,
            pnl_usd=138.8,
            pnl_pct=2.77,
            timestamp=now,
            portfolio_value_after=10138.8,
        ),
        TradeResult(
            run_id="r1",
            strategy_id="s1",
            pair="BTC-USD",
            side="sell",
            quantity=0.1,
            fill_price=49000.0,
            fill_value_usd=4900.0,
            fee_usd=29.4,
            pnl_usd=-129.4,
            pnl_pct=-2.58,
            timestamp=now,
            portfolio_value_after=10009.4,
        ),
    ]

    snapshots = [
        PortfolioSnapshot(run_id="r1", strategy_id="s1", timestamp=now, cash_usd=10000.0, coin_price=50000.0, total_value_usd=10000.0, high_water_mark=10000.0),
        PortfolioSnapshot(run_id="r1", strategy_id="s1", timestamp=now, cash_usd=10200.0, coin_price=52000.0, total_value_usd=10200.0, high_water_mark=10200.0),
        PortfolioSnapshot(run_id="r1", strategy_id="s1", timestamp=now, cash_usd=9800.0, coin_price=49000.0, total_value_usd=9800.0, high_water_mark=10200.0),
    ]

    summary = PerformanceTracker.calculate_summary(
        run_id="r1",
        strategy_id="s1",
        mode="replay",
        pair="BTC-USD",
        start_time="2026-09-25T00:00:00Z",
        end_time="2026-09-27T00:00:00Z",
        initial_cash=10000.0,
        final_value=10500.0,
        trades=trades,
        snapshots=snapshots,
    )

    assert summary.total_trades == 2  # 2 closed sells
    assert summary.winning_trades == 1
    assert summary.losing_trades == 1
    assert summary.win_rate == 50.0
    assert summary.total_pnl_pct == 5.0  # (10500 - 10000) / 10000
    assert summary.profit_factor == pytest.approx(138.8 / 129.4, abs=0.05)
    # Drawdown from 10,200 to 9,800: (10200 - 9800) / 10200 = ~3.92%
    assert summary.max_drawdown == pytest.approx(3.92, abs=0.1)


def test_build_comparison_dataframe() -> None:
    """Test leaderboard dataframe generation and sorting."""
    now_str = "2026-09-25T00:00:00Z"
    s1 = PerformanceTracker.calculate_summary(
        run_id="r1", strategy_id="strat_a", mode="replay", pair="BTC-USD",
        start_time=now_str, end_time=now_str, initial_cash=10000.0, final_value=10200.0, trades=[]
    )
    s2 = PerformanceTracker.calculate_summary(
        run_id="r2", strategy_id="strat_b", mode="replay", pair="BTC-USD",
        start_time=now_str, end_time=now_str, initial_cash=10000.0, final_value=11000.0, trades=[]
    )

    df = PerformanceTracker.build_comparison_dataframe([s1, s2])
    assert len(df) == 2
    # strat_b (+10%) should be ranked first before strat_a (+2%)
    assert df.iloc[0]["Strategy"] == "strat_b"
    assert df.iloc[1]["Strategy"] == "strat_a"
