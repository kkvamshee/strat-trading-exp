"""
Unit tests for the Strategy Framework (Jev, EMA/MACD, RSI/BB, Momentum).
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.models import MarketState, PortfolioSnapshot
from src.strategies.jev_strategy import JevStrategy
from src.strategies.registry import build_strategy, list_available_strategies
from src.strategies.technical_strategies import (
    EmaMacdStrategy,
    MomentumStrategy,
    RsiBollingerStrategy,
)


@pytest.fixture
def mock_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        run_id="test-run",
        strategy_id="test",
        timestamp=datetime.now(timezone.utc),
        cash_usd=10000.0,
        coin_quantity=0.0,
        coin_price=60000.0,
        total_value_usd=10000.0,
        high_water_mark=10000.0,
    )


@pytest.mark.asyncio
async def test_jev_strategy(mock_portfolio: PortfolioSnapshot) -> None:
    """Test JevStrategy prompt formatting and heuristic fallback."""
    strat = JevStrategy(mock_mode=True)
    state = MarketState(
        pair="BTC-USD",
        timestamp=datetime.now(timezone.utc),
        current_price=65000.0,
        change_5m_pct=0.8,
        change_15m_pct=1.5,
        rsi_14=62.0,
        macd=15.0,
        macd_signal=10.0,
        macd_hist=5.0,
        ema_9=65200.0,
        ema_21=64800.0,
        ema_50=64000.0,
        volume_ratio=1.4,
    )

    prompt = strat.format_market_state_prompt(state)
    assert "CRYPTO TECHNICAL STATE" in prompt
    assert "BTC-USD" in prompt
    assert "$65,000.00" in prompt
    assert "EMA9" in prompt

    signal = await strat.generate_signal(state, mock_portfolio)
    assert signal.strategy_id == "jev"
    assert signal.action in ("buy", "sell", "hold")
    assert 0.0 <= signal.conviction <= 10.0
    assert 0.0 <= signal.risk_level <= 10.0


@pytest.mark.asyncio
async def test_ema_macd_strategy(mock_portfolio: PortfolioSnapshot) -> None:
    """Test EmaMacdStrategy generates buy on bullish cross and sell on bearish cross."""
    strat = EmaMacdStrategy()

    # Bullish state
    bullish = MarketState(
        pair="BTC-USD",
        timestamp=datetime.now(timezone.utc),
        current_price=65000.0,
        ema_9=65500.0,
        ema_21=64500.0,
        macd=10.0,
        macd_signal=5.0,
        macd_hist=5.0,
    )
    sig_buy = await strat.generate_signal(bullish, mock_portfolio)
    assert sig_buy.action == "buy"
    assert sig_buy.conviction >= 6.5

    # Bearish state
    bearish = MarketState(
        pair="BTC-USD",
        timestamp=datetime.now(timezone.utc),
        current_price=64000.0,
        ema_9=63500.0,
        ema_21=64500.0,
        macd=-5.0,
        macd_signal=0.0,
        macd_hist=-5.0,
    )
    sig_sell = await strat.generate_signal(bearish, mock_portfolio)
    assert sig_sell.action == "sell"


@pytest.mark.asyncio
async def test_rsi_bb_strategy(mock_portfolio: PortfolioSnapshot) -> None:
    """Test mean-reversion buy on oversold dip and sell on overbought rally."""
    strat = RsiBollingerStrategy(rsi_oversold=30.0, rsi_overbought=70.0)

    oversold = MarketState(
        pair="BTC-USD",
        timestamp=datetime.now(timezone.utc),
        current_price=59000.0,
        rsi_14=25.0,
        bb_percent=0.10,
    )
    sig_buy = await strat.generate_signal(oversold, mock_portfolio)
    assert sig_buy.action == "buy"

    overbought = MarketState(
        pair="BTC-USD",
        timestamp=datetime.now(timezone.utc),
        current_price=71000.0,
        rsi_14=75.0,
        bb_percent=0.92,
    )
    sig_sell = await strat.generate_signal(overbought, mock_portfolio)
    assert sig_sell.action == "sell"


@pytest.mark.asyncio
async def test_momentum_strategy(mock_portfolio: PortfolioSnapshot) -> None:
    """Test volume-supported breakout strategy."""
    strat = MomentumStrategy(threshold_pct=0.5, volume_factor=1.3)

    breakout = MarketState(
        pair="BTC-USD",
        timestamp=datetime.now(timezone.utc),
        current_price=66000.0,
        change_15m_pct=0.8,
        volume_ratio=1.6,
        rsi_14=60.0,
    )
    sig = await strat.generate_signal(breakout, mock_portfolio)
    assert sig.action == "buy"
    assert sig.conviction >= 7.0


def test_strategy_registry() -> None:
    """Verify registry dynamically instantiates all available strategies."""
    available = list_available_strategies()
    assert "jev" in available
    assert "ema_macd" in available
    assert "rsi_bb" in available
    assert "momentum" in available

    for name in available:
        s = build_strategy(name)
        assert s is not None
        assert s.strategy_id != ""

    with pytest.raises(ValueError):
        build_strategy("non_existent_strat")
