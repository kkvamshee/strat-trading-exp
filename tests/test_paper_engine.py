"""
Unit tests for the Multi-Portfolio Paper Trading Simulator.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.database import Database
from src.models import OrderIntent, PaperConfig
from src.paper_engine import PaperTradingEngine


def test_paper_buy_execution(temp_db: Database) -> None:
    """Test realistic paper BUY execution with taker fee and slippage."""
    config = PaperConfig(default_cash_usd=10000.0, taker_fee_pct=0.006, slippage_pct=0.0005)
    engine = PaperTradingEngine(db=temp_db, config=config)
    now = datetime.now(timezone.utc)

    intent = OrderIntent(
        pair="BTC-USD",
        side="buy",
        quantity=0.1,
        price=50000.0,
    )
    trade = engine.execute_order(
        intent=intent, run_id="r1", strategy_id="s1", timestamp=now, decision_id=1
    )
    assert trade is not None
    assert trade.side == "buy"
    # Upward slippage: 50,000 * 1.0005 = 50,025
    assert trade.fill_price == pytest.approx(50025.0)
    # Value: 0.1 * 50,025 = 5,002.50
    assert trade.fill_value_usd == pytest.approx(5002.50)
    # Fee: 5,002.50 * 0.006 = 30.015
    assert trade.fee_usd == pytest.approx(30.015)

    port = engine.get_portfolio("r1", "s1", current_price=50000.0)
    # Remaining cash: 10,000 - 5,002.50 - 30.015 = 4,967.485
    assert port.cash_usd == pytest.approx(4967.485)
    assert port.coin_quantity == pytest.approx(0.1)
    assert port.avg_entry_price == pytest.approx(50025.0)


def test_paper_sell_and_pnl_calculation(temp_db: Database) -> None:
    """Test paper SELL execution and profit calculation."""
    config = PaperConfig(default_cash_usd=10000.0, taker_fee_pct=0.006, slippage_pct=0.0005)
    engine = PaperTradingEngine(db=temp_db, config=config)
    now = datetime.now(timezone.utc)

    # 1. Buy at 50,000
    buy_intent = OrderIntent(pair="BTC-USD", side="buy", quantity=0.1, price=50000.0)
    engine.execute_order(intent=buy_intent, run_id="r1", strategy_id="s1", timestamp=now)

    # 2. Sell at 55,000 (10% higher)
    sell_intent = OrderIntent(pair="BTC-USD", side="sell", quantity=0.1, price=55000.0)
    sell_trade = engine.execute_order(intent=sell_intent, run_id="r1", strategy_id="s1", timestamp=now)

    assert sell_trade is not None
    assert sell_trade.side == "sell"
    # Downward slippage: 55,000 * 0.9995 = 54,972.50
    assert sell_trade.fill_price == pytest.approx(54972.50)
    assert sell_trade.pnl_usd is not None
    # Gross gain is around $500 minus fees on both ends (~$60), net P&L should be ~+$436
    assert sell_trade.pnl_usd > 400.0
    assert sell_trade.pnl_pct > 8.0

    port = engine.get_portfolio("r1", "s1", current_price=55000.0)
    assert port.coin_quantity == 0.0
    assert port.avg_entry_price is None
    assert port.cash_usd > 10400.0  # Profited


def test_multi_portfolio_isolation(temp_db: Database) -> None:
    """Test that two concurrent strategies maintain completely isolated balances."""
    engine = PaperTradingEngine(db=temp_db)
    now = datetime.now(timezone.utc)

    # Strategy A buys
    intent_a = OrderIntent(pair="BTC-USD", side="buy", quantity=0.05, price=60000.0)
    engine.execute_order(intent_a, run_id="session-1", strategy_id="strat_a", timestamp=now)

    port_a = engine.get_portfolio("session-1", "strat_a", current_price=60000.0)
    port_b = engine.get_portfolio("session-1", "strat_b", current_price=60000.0)

    # Strategy A has coins; Strategy B remains 100% cash
    assert port_a.coin_quantity == pytest.approx(0.05)
    assert port_a.cash_usd < 10000.0

    assert port_b.coin_quantity == 0.0
    assert port_b.cash_usd == 10000.0
