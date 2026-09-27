"""
Unit tests for the Unified Strategy Brain Component.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest

from src.brain import StrategyBrain
from src.models import (
    FeatureConfig,
    PortfolioSnapshot,
    RiskConfig,
    StrategySignal,
)
from src.strategies.technical_strategies import EmaMacdStrategy
from tests.conftest import generate_synthetic_candles


@pytest.fixture
def candle_dataframe() -> pd.DataFrame:
    """Fixture providing a pandas DataFrame of 200 candles."""
    candles = generate_synthetic_candles(num_candles=200)
    data = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    return pd.DataFrame(data)


def test_brain_feature_computation(candle_dataframe: pd.DataFrame) -> None:
    """Test that all technical indicators and features are accurately computed."""
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy)

    now = datetime.now(timezone.utc)
    state = brain.build_market_state("BTC-USD", now, candle_dataframe)

    assert state.pair == "BTC-USD"
    assert state.current_price > 0
    # RSI must be between 0 and 100
    assert 0.0 <= state.rsi_14 <= 100.0
    # Bollinger %B is a float
    assert isinstance(state.bb_percent, float)
    assert state.bb_upper > state.bb_lower
    # EMAs must be reasonable
    assert state.ema_9 > 0
    assert state.ema_21 > 0
    assert state.ema_50 > 0
    # ATR regime must be valid
    assert state.atr_regime in ("low", "medium", "high")
    assert state.candle_count == 200


def test_brain_risk_rules_conviction() -> None:
    """Test rejection when conviction is below minimum."""
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy, risk_config=RiskConfig(min_conviction=6.0))

    now = datetime.now(timezone.utc)
    port = PortfolioSnapshot(
        run_id="r1",
        strategy_id="ema_macd",
        timestamp=now,
        cash_usd=10000.0,
        coin_quantity=0.0,
        coin_price=60000.0,
        total_value_usd=10000.0,
        high_water_mark=10000.0,
    )

    low_conv_signal = StrategySignal(
        strategy_id="ema_macd", action="buy", conviction=4.5, risk_level=3.0
    )
    risk_dec = brain.apply_risk_rules(low_conv_signal, port, now)
    assert not risk_dec.approved
    assert risk_dec.outcome == "SKIP_CONVICTION"


def test_brain_risk_rules_risk_level() -> None:
    """Test rejection when risk rating is above threshold."""
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy, risk_config=RiskConfig(max_risk_level=7.0))

    now = datetime.now(timezone.utc)
    port = PortfolioSnapshot(
        run_id="r1",
        strategy_id="ema_macd",
        timestamp=now,
        cash_usd=10000.0,
        coin_quantity=0.0,
        coin_price=60000.0,
        total_value_usd=10000.0,
        high_water_mark=10000.0,
    )

    high_risk_signal = StrategySignal(
        strategy_id="ema_macd", action="buy", conviction=8.0, risk_level=8.5
    )
    risk_dec = brain.apply_risk_rules(high_risk_signal, port, now)
    assert not risk_dec.approved
    assert risk_dec.outcome == "SKIP_RISK"


def test_brain_risk_rules_daily_loss_cap() -> None:
    """Test circuit breaker tripping when daily loss cap is breached."""
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy, risk_config=RiskConfig(daily_loss_cap_pct=3.0))

    now = datetime.now(timezone.utc)
    port = PortfolioSnapshot(
        run_id="r1",
        strategy_id="ema_macd",
        timestamp=now,
        cash_usd=9600.0,
        coin_quantity=0.0,
        coin_price=60000.0,
        total_value_usd=9600.0,
        high_water_mark=10000.0,
        daily_pnl_pct=-3.5,  # Down 3.5% today
    )

    valid_signal = StrategySignal(
        strategy_id="ema_macd", action="buy", conviction=8.5, risk_level=2.0
    )
    risk_dec = brain.apply_risk_rules(valid_signal, port, now)
    assert not risk_dec.approved
    assert risk_dec.outcome == "DAILY_LOSS_CAP"


def test_brain_risk_rules_cooldown() -> None:
    """Test cooldown enforcement between trades."""
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy, risk_config=RiskConfig(cooldown_minutes=5))

    t0 = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    brain.last_trade_time = t0

    port = PortfolioSnapshot(
        run_id="r1",
        strategy_id="ema_macd",
        timestamp=t0,
        cash_usd=10000.0,
        coin_quantity=0.0,
        coin_price=60000.0,
        total_value_usd=10000.0,
        high_water_mark=10000.0,
    )

    # 2 minutes later (within 5-minute cooldown)
    t_soon = t0 + timedelta(minutes=2)
    sig = StrategySignal(strategy_id="ema_macd", action="buy", conviction=8.0, risk_level=2.0)
    risk_dec = brain.apply_risk_rules(sig, port, t_soon)
    assert not risk_dec.approved
    assert risk_dec.outcome == "COOLDOWN"

    # 6 minutes later (cooldown expired)
    t_later = t0 + timedelta(minutes=6)
    risk_dec_ok = brain.apply_risk_rules(sig, port, t_later)
    assert risk_dec_ok.approved
    assert risk_dec_ok.outcome == "APPROVED"


def test_brain_position_guardrails() -> None:
    """Test that double buys and naked sells are blocked."""
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy)
    now = datetime.now(timezone.utc)

    # 1. Double buy guard
    holding_port = PortfolioSnapshot(
        run_id="r1",
        strategy_id="ema_macd",
        timestamp=now,
        cash_usd=5000.0,
        coin_quantity=0.1,  # Already holding 0.1 coin
        coin_price=60000.0,
        total_value_usd=11000.0,
        high_water_mark=11000.0,
    )
    buy_sig = StrategySignal(strategy_id="ema_macd", action="buy", conviction=8.0, risk_level=2.0)
    res_buy = brain.apply_risk_rules(buy_sig, holding_port, now)
    assert not res_buy.approved
    assert res_buy.outcome == "POSITION_LIMIT"

    # 2. Naked sell guard
    empty_port = PortfolioSnapshot(
        run_id="r1",
        strategy_id="ema_macd",
        timestamp=now,
        cash_usd=10000.0,
        coin_quantity=0.0,  # 0 coins to sell
        coin_price=60000.0,
        total_value_usd=10000.0,
        high_water_mark=10000.0,
    )
    sell_sig = StrategySignal(strategy_id="ema_macd", action="sell", conviction=8.0, risk_level=2.0)
    res_sell = brain.apply_risk_rules(sell_sig, empty_port, now)
    assert not res_sell.approved
    assert res_sell.outcome == "NO_POSITION"


@pytest.mark.asyncio
async def test_brain_evaluate_end_to_end(candle_dataframe: pd.DataFrame) -> None:
    """Test full master pipeline evaluate() execution."""
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy)

    now = datetime.now(timezone.utc)
    port = PortfolioSnapshot(
        run_id="r1",
        strategy_id="ema_macd",
        timestamp=now,
        cash_usd=10000.0,
        coin_quantity=0.0,
        coin_price=float(candle_dataframe.iloc[-1]["close"]),
        total_value_usd=10000.0,
        high_water_mark=10000.0,
    )

    eval_result = await brain.evaluate("BTC-USD", now, candle_dataframe, port)
    assert eval_result.state is not None
    assert eval_result.signal is not None
    assert eval_result.risk is not None
    assert eval_result.signal.strategy_id == "ema_macd"
