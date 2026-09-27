"""
Jev Strategy — TypeSafe AI System One decision model integration.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional
from loguru import logger

from src.models import MarketState, PortfolioSnapshot, StrategySignal
from src.strategies.base import BaseStrategy

try:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score
    HAS_TYPESAFE_SDK = True
except ImportError:
    HAS_TYPESAFE_SDK = False


class JevStrategy(BaseStrategy):
    """
    Evaluates market state using TypeSafe AI's Jev (System One) model,
    with an intelligent fallback simulator when running offline or without API keys.
    """

    def __init__(
        self,
        strategy_id: str = "jev",
        name: str = "Jev System One Strategy",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        model: str = "system-one",
        timeout_sec: float = 10.0,
        mock_mode: bool = False,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(strategy_id=strategy_id, name=name, parameters=parameters)
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.api_endpoint = api_endpoint
        self.model = model
        self.timeout_sec = timeout_sec
        self.mock_mode = mock_mode or (not self.api_key or self.api_key.startswith("your_"))
        self._client: Optional[Any] = None

    def _get_client(self) -> Optional[Any]:
        if not HAS_TYPESAFE_SDK or self.mock_mode or not self.api_key:
            return None
        if self._client is None:
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.api_endpoint:
                kwargs["base_url"] = self.api_endpoint
            self._client = AsyncTypeSafeClient(**kwargs)
        return self._client

    def format_market_state_prompt(self, state: MarketState) -> str:
        """Construct dense, readable market state context for Jev."""
        rsi_label = (
            "overbought" if state.rsi_14 > 70 else "oversold" if state.rsi_14 < 30 else "neutral"
        )
        return f"""CRYPTO TECHNICAL STATE — {state.pair} — {state.timestamp.isoformat()}

Price: ${state.current_price:,.2f}
Momentum:
  5m Change: {state.change_5m_pct:+.2f}%
  15m Change: {state.change_15m_pct:+.2f}%
  1h Change: {state.change_1h_pct:+.2f}%
  RSI(14): {state.rsi_14:.1f} ({rsi_label})
  MACD: {state.macd:.4f}, Signal: {state.macd_signal:.4f}, Hist: {state.macd_hist:+.4f}

Volatility & Bands:
  Bollinger %B: {state.bb_percent:.2f} (0=lower, 1=upper)
  ATR(14): {state.atr_14:.2f} (Regime: {state.atr_regime})

Trend & Structure:
  EMA9: {state.ema_9:.2f}, EMA21: {state.ema_21:.2f}, EMA50: {state.ema_50:.2f}
  VWAP: {state.vwap:.2f}
  Volume Ratio: {state.volume_ratio:.2f}x (vs 20-bar avg)
"""

    async def generate_signal(
        self, state: MarketState, portfolio: PortfolioSnapshot
    ) -> StrategySignal:
        start_time = time.perf_counter()
        prompt = self.format_market_state_prompt(state)
        client = self._get_client()

        if client and HAS_TYPESAFE_SDK:
            try:
                questions = {
                    "action": Choice(
                        instructions="Given the market state, select the appropriate trading action.",
                        criteria={"buy": "Initiate long", "sell": "Close or short", "hold": "No action"},
                    ),
                    "conviction": Score(
                        instructions="Confidence score in the directional move.",
                        criteria=[f"Level {i}" for i in range(11)],
                    ),
                    "is_trending": Noul(
                        instructions="Is the market currently in a strong directional trend?"
                    ),
                    "risk_level": Score(
                        instructions="Estimated trade execution risk right now.",
                        criteria=[f"Level {i}" for i in range(11)],
                    ),
                }

                resp = await client.system_one(
                    state=prompt,
                    questions=questions,
                    model=self.model,
                    timeout=self.timeout_sec,
                )
                latency = (time.perf_counter() - start_time) * 1000

                answers = resp.answers if hasattr(resp, "answers") else {}
                raw_action = answers.get("action")
                action_val = str(getattr(raw_action, "choice", raw_action) or "hold").lower()
                if action_val not in ("buy", "sell", "hold"):
                    action_val = "hold"

                raw_conv = answers.get("conviction")
                conv_score = float(getattr(raw_conv, "score", 5.0) or 5.0)

                raw_risk = answers.get("risk_level")
                risk_score = float(getattr(raw_risk, "score", 5.0) or 5.0)

                raw_trend = answers.get("is_trending")
                is_trending = bool(getattr(raw_trend, "result", False))

                return StrategySignal(
                    strategy_id=self.strategy_id,
                    action=action_val,  # type: ignore
                    conviction=min(max(conv_score, 0.0), 10.0),
                    risk_level=min(max(risk_score, 0.0), 10.0),
                    is_trending=is_trending,
                    rationale=f"Jev System One decision: {action_val} (conviction: {conv_score:.1f})",
                    latency_ms=latency,
                    raw_response={"model": getattr(resp, "model", "jev"), "usage": getattr(resp, "usage", {})},
                )
            except Exception as e:
                logger.warning(f"TypeSafe API query error ({e}); falling back to heuristic engine.")

        # Fallback heuristic / Mock simulation when no API key is available
        latency = (time.perf_counter() - start_time) * 1000
        return self._heuristic_signal(state, latency)

    def _heuristic_signal(self, state: MarketState, latency: float) -> StrategySignal:
        """
        Calibrated technical heuristic mimicking System One decision making
        for tests or offline backtesting runs without API charges.
        """
        action = "hold"
        conviction = 4.0
        risk_level = 5.0
        is_trending = False

        # Strong trend + momentum alignment
        bullish_ema = state.ema_9 > state.ema_21 > state.ema_50
        bearish_ema = state.ema_9 < state.ema_21 < state.ema_50
        bullish_macd = state.macd_hist > 0 and state.macd > state.macd_signal
        bearish_macd = state.macd_hist < 0 and state.macd < state.macd_signal

        if bullish_ema and bullish_macd and state.rsi_14 < 68 and state.volume_ratio > 1.1:
            action = "buy"
            conviction = 7.8
            risk_level = 4.2
            is_trending = True
        elif state.rsi_14 < 28 and state.bb_percent < 0.1:  # Deep oversold bounce
            action = "buy"
            conviction = 7.2
            risk_level = 5.5
            is_trending = False
        elif bearish_ema and bearish_macd and state.rsi_14 > 35:
            action = "sell"
            conviction = 7.5
            risk_level = 4.5
            is_trending = True
        elif state.rsi_14 > 75 and state.bb_percent > 0.9:  # Overbought exhaustion
            action = "sell"
            conviction = 7.0
            risk_level = 5.8
            is_trending = False
        else:
            action = "hold"
            conviction = 3.5
            risk_level = 5.0

        return StrategySignal(
            strategy_id=self.strategy_id,
            action=action,  # type: ignore
            conviction=conviction,
            risk_level=risk_level,
            is_trending=is_trending,
            rationale=f"Jev Heuristic/Simulated: {action} (conviction: {conviction:.1f})",
            latency_ms=latency,
            raw_response={"mode": "simulated_jev"},
        )
