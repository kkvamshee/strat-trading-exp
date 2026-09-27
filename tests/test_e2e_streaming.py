"""
End-to-End (E2E) Test: Streaming Paper Trading Lifecycle.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.brain import StrategyBrain
from src.database import Database
from src.models import AppConfig, Candle
from src.paper_engine import PaperTradingEngine
from src.strategies.technical_strategies import EmaMacdStrategy
from tests.conftest import generate_synthetic_candles


@pytest.mark.asyncio
async def test_e2e_paper_streaming_lifecycle(temp_db: Database) -> None:
    """Simulate streaming candle processing, brain evaluation, and paper execution."""
    config = AppConfig()
    strategy = EmaMacdStrategy()
    brain = StrategyBrain(strategy=strategy, risk_config=config.risk, feature_config=config.features)
    paper_engine = PaperTradingEngine(db=temp_db, config=config.paper)

    run_id = "test-stream-e2e"
    pair = "BTC-USD"

    # Pre-populate 50 candles for indicator warm-up
    warmup_candles = generate_synthetic_candles(pair=pair, num_candles=60)
    temp_db.insert_candles(warmup_candles)

    # Simulate arrival of 10 incoming live candles
    current_ts = warmup_candles[-1].timestamp + 60
    base_price = warmup_candles[-1].close

    for i in range(10):
        # Create a sharp uptrend to trigger buy signals
        base_price *= 1.005
        candle = Candle(
            pair=pair,
            timestamp=current_ts,
            open=base_price * 0.999,
            high=base_price * 1.002,
            low=base_price * 0.998,
            close=base_price,
            volume=50.0,
            source="websocket",
        )
        temp_db.insert_candles([candle])

        now_dt = datetime.fromtimestamp(candle.timestamp, tz=timezone.utc)
        candles_df = temp_db.get_latest_candles_df(pair, n=config.features.candle_lookback)
        portfolio = paper_engine.get_portfolio(run_id, strategy.strategy_id, current_price=candle.close)

        eval_res = await brain.evaluate(pair, now_dt, candles_df, portfolio)

        dec_id = temp_db.record_decision(
            run_id=run_id,
            strategy_id=strategy.strategy_id,
            pair=pair,
            timestamp=now_dt,
            market_state=eval_res.state,
            signal=eval_res.signal,
            risk=eval_res.risk,
            mode="paper",
        )

        if eval_res.order_intent:
            paper_engine.execute_order(
                intent=eval_res.order_intent,
                run_id=run_id,
                strategy_id=strategy.strategy_id,
                timestamp=now_dt,
                decision_id=dec_id,
            )

        current_ts += 60

    # Assertions
    port_final = paper_engine.get_portfolio(run_id, strategy.strategy_id, current_price=base_price)
    trades = temp_db.get_run_trades(run_id, strategy.strategy_id)

    assert port_final.total_value_usd > 0
    # Decisions were recorded for every candle
    with temp_db.get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM decisions WHERE run_id = ?", (run_id,))
        assert cur.fetchone()[0] == 10
