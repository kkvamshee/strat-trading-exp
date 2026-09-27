"""
Technical & Quantitative Baseline Strategies for Comparative Testing.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.models import MarketState, PortfolioSnapshot, StrategySignal
from src.strategies.base import BaseStrategy


class EmaMacdStrategy(BaseStrategy):
    """
    Trend-following strategy utilizing EMA crossovers and MACD confirmation.
    """

    def __init__(
        self,
        strategy_id: str = "ema_macd",
        name: str = "EMA/MACD Trend Follower",
        fast_period: int = 9,
        slow_period: int = 21,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(strategy_id=strategy_id, name=name, parameters=parameters)
        self.fast_period = fast_period
        self.slow_period = slow_period

    async def generate_signal(
        self, state: MarketState, portfolio: PortfolioSnapshot
    ) -> StrategySignal:
        t0 = time.perf_counter()

        bullish_crossover = state.ema_9 > state.ema_21
        macd_positive = state.macd_hist > 0 and state.macd > state.macd_signal
        bearish_crossover = state.ema_9 < state.ema_21
        macd_negative = state.macd_hist < 0 and state.macd < state.macd_signal

        action = "hold"
        conviction = 4.0
        risk_level = 5.0

        if bullish_crossover and macd_positive:
            action = "buy"
            # Scale conviction based on MACD histogram strength
            hist_scale = min(abs(state.macd_hist) * 10, 3.0)
            conviction = min(6.5 + hist_scale, 9.5)
            risk_level = 4.5
        elif bearish_crossover or macd_negative:
            action = "sell"
            hist_scale = min(abs(state.macd_hist) * 10, 3.0)
            conviction = min(6.5 + hist_scale, 9.5)
            risk_level = 5.0
        else:
            action = "hold"
            conviction = 3.0
            risk_level = 5.0

        latency = (time.perf_counter() - t0) * 1000
        return StrategySignal(
            strategy_id=self.strategy_id,
            action=action,  # type: ignore
            conviction=conviction,
            risk_level=risk_level,
            is_trending=bullish_crossover or bearish_crossover,
            rationale=f"EMA9 ({state.ema_9:.2f}) vs EMA21 ({state.ema_21:.2f}), MACD Hist {state.macd_hist:+.4f}",
            latency_ms=latency,
        )


class RsiBollingerStrategy(BaseStrategy):
    """
    Mean-reversion strategy buying oversold dips and selling overbought peaks.
    """

    def __init__(
        self,
        strategy_id: str = "rsi_bb",
        name: str = "RSI & Bollinger Mean Reversion",
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(strategy_id=strategy_id, name=name, parameters=parameters)
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    async def generate_signal(
        self, state: MarketState, portfolio: PortfolioSnapshot
    ) -> StrategySignal:
        t0 = time.perf_counter()

        action = "hold"
        conviction = 4.0
        risk_level = 5.0

        # Oversold condition: RSI < oversold threshold and price near lower Bollinger Band
        if state.rsi_14 < self.rsi_oversold and state.bb_percent < 0.20:
            action = "buy"
            conviction = min(7.0 + ((self.rsi_oversold - state.rsi_14) / 10.0), 9.5)
            risk_level = 4.0
        # Overbought condition: RSI > overbought threshold or price near upper Bollinger Band
        elif state.rsi_14 > self.rsi_overbought or state.bb_percent > 0.85:
            action = "sell"
            conviction = min(7.0 + ((state.rsi_14 - self.rsi_overbought) / 10.0), 9.5)
            risk_level = 4.5
        else:
            action = "hold"
            conviction = 3.0
            risk_level = 5.0

        latency = (time.perf_counter() - t0) * 1000
        return StrategySignal(
            strategy_id=self.strategy_id,
            action=action,  # type: ignore
            conviction=conviction,
            risk_level=risk_level,
            is_trending=False,
            rationale=f"RSI={state.rsi_14:.1f}, BB%={state.bb_percent:.2f}",
            latency_ms=latency,
        )


class MomentumStrategy(BaseStrategy):
    """
    Momentum breakout strategy detecting sudden price surges backed by abnormal volume.
    """

    def __init__(
        self,
        strategy_id: str = "momentum",
        name: str = "Volume-Supported Momentum",
        threshold_pct: float = 0.4,
        volume_factor: float = 1.3,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(strategy_id=strategy_id, name=name, parameters=parameters)
        self.threshold_pct = threshold_pct
        self.volume_factor = volume_factor

    async def generate_signal(
        self, state: MarketState, portfolio: PortfolioSnapshot
    ) -> StrategySignal:
        t0 = time.perf_counter()

        action = "hold"
        conviction = 3.5
        risk_level = 5.0

        surge_up = (
            state.change_15m_pct > self.threshold_pct and state.volume_ratio > self.volume_factor
        )
        surge_down = state.change_15m_pct < -self.threshold_pct

        if surge_up and state.rsi_14 < 72:
            action = "buy"
            conviction = min(7.2 + (state.volume_ratio - 1.0), 9.0)
            risk_level = 4.8
        elif surge_down or state.rsi_14 > 78:
            action = "sell"
            conviction = 7.5
            risk_level = 5.2
        else:
            action = "hold"
            conviction = 3.0
            risk_level = 5.0

        latency = (time.perf_counter() - t0) * 1000
        return StrategySignal(
            strategy_id=self.strategy_id,
            action=action,  # type: ignore
            conviction=conviction,
            risk_level=risk_level,
            is_trending=True,
            rationale=f"15m Change: {state.change_15m_pct:+.2f}%, Vol Ratio: {state.volume_ratio:.2f}x",
            latency_ms=latency,
        )
