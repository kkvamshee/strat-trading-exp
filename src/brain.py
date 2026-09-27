"""
Unified Strategy Brain — State/Feature Computation, Strategy Dispatch, and Deterministic Risk Gating.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from src.models import (
    EvaluationResult,
    FeatureConfig,
    MarketState,
    OrderIntent,
    PortfolioSnapshot,
    RiskConfig,
    RiskDecision,
    StrategySignal,
)
from src.strategies.base import BaseStrategy


class StrategyBrain:
    """
    Unified Brain component condensing:
    1. Technical Indicator & Feature Computation (State Builder)
    2. Strategy Signal Dispatching (Strategy Engine)
    3. Deterministic Risk Filtering & Sizing (Risk Manager)
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        risk_config: Optional[RiskConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        self.strategy = strategy
        self.risk_config = risk_config or RiskConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.last_trade_time: Optional[datetime] = None

    # ---------------- 1. Feature Computation (State Builder) ----------------
    def build_market_state(
        self,
        pair: str,
        timestamp: datetime,
        df: pd.DataFrame,
    ) -> MarketState:
        """Calculate technical indicators on historical candle slice up to current timestamp."""
        if df.empty:
            raise ValueError(f"Cannot build market state for {pair}: candle DataFrame is empty.")

        # Ensure sorted and correct types
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        n = len(df)
        curr_price = float(close.iloc[-1])

        # Momentum: % changes over windows (1m candles)
        change_5m = float(((curr_price - close.iloc[-5]) / close.iloc[-5]) * 100) if n >= 5 else 0.0
        change_15m = float(((curr_price - close.iloc[-15]) / close.iloc[-15]) * 100) if n >= 15 else 0.0
        change_1h = float(((curr_price - close.iloc[-60]) / close.iloc[-60]) * 100) if n >= 60 else 0.0

        # Technical Indicators using robust vectorized formulas
        # 1. RSI(14)
        rsi = self._compute_rsi(close, length=self.feature_config.rsi_length)

        # 2. MACD(12, 26, 9)
        macd, macd_signal, macd_hist = self._compute_macd(
            close,
            fast=self.feature_config.macd_fast,
            slow=self.feature_config.macd_slow,
            signal=self.feature_config.macd_signal,
        )

        # 3. Bollinger Bands(20, 2)
        bb_upper, bb_mid, bb_lower, bb_pct = self._compute_bollinger(
            close,
            length=self.feature_config.bb_length,
            std=self.feature_config.bb_std,
        )

        # 4. ATR(14)
        atr, atr_regime = self._compute_atr(
            high, low, close, length=self.feature_config.atr_length
        )

        # 5. EMAs (9, 21, 50)
        ema_9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1]) if n >= 9 else curr_price
        ema_21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1]) if n >= 21 else curr_price
        ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if n >= 50 else curr_price

        # 6. VWAP
        typical_price = (high + low + close) / 3.0
        cum_vol = volume.cumsum()
        vwap = (
            float((typical_price * volume).cumsum().iloc[-1] / cum_vol.iloc[-1])
            if cum_vol.iloc[-1] > 0
            else curr_price
        )

        # 7. Volume Ratio (current vs 20-bar rolling average)
        vol_rolling_avg = float(volume.tail(20).mean()) if n >= 5 else 1.0
        curr_vol = float(volume.iloc[-1])
        vol_ratio = (curr_vol / vol_rolling_avg) if vol_rolling_avg > 0 else 1.0

        return MarketState(
            pair=pair,
            timestamp=timestamp,
            current_price=curr_price,
            change_5m_pct=change_5m,
            change_15m_pct=change_15m,
            change_1h_pct=change_1h,
            rsi_14=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            bb_percent=bb_pct,
            bb_upper=bb_upper,
            bb_middle=bb_mid,
            bb_lower=bb_lower,
            atr_14=atr,
            atr_regime=atr_regime,
            ema_9=ema_9,
            ema_21=ema_21,
            ema_50=ema_50,
            vwap=vwap,
            volume_ratio=vol_ratio,
            candle_count=n,
        )

    # ---------------- 2. Risk Gate & Sizing (Risk Manager) ----------------
    def apply_risk_rules(
        self,
        signal: StrategySignal,
        portfolio: PortfolioSnapshot,
        timestamp: datetime,
    ) -> RiskDecision:
        """
        Evaluate deterministic risk gate.
        Returns RiskDecision(approved=True/False, outcome=..., reason=...).
        """
        if signal.action == "hold":
            return RiskDecision(approved=False, outcome="HOLD", reason="Strategy signaled HOLD")

        # Rule 1: Conviction Threshold
        if signal.conviction < self.risk_config.min_conviction:
            return RiskDecision(
                approved=False,
                outcome="SKIP_CONVICTION",
                reason=f"Conviction {signal.conviction:.1f} < threshold {self.risk_config.min_conviction:.1f}",
            )

        # Rule 2: Risk Level Cap
        if signal.risk_level > self.risk_config.max_risk_level:
            return RiskDecision(
                approved=False,
                outcome="SKIP_RISK",
                reason=f"Risk level {signal.risk_level:.1f} > maximum {self.risk_config.max_risk_level:.1f}",
            )

        # Rule 3: Daily Loss Cap
        daily_loss = portfolio.daily_pnl_pct or 0.0
        if daily_loss <= -abs(self.risk_config.daily_loss_cap_pct):
            return RiskDecision(
                approved=False,
                outcome="DAILY_LOSS_CAP",
                reason=f"Daily loss cap reached ({daily_loss:.2f}% <= -{self.risk_config.daily_loss_cap_pct}%)",
            )

        # Rule 4: Trade Cooldown Period
        if self.last_trade_time is not None:
            time_since_last_sec = (timestamp - self.last_trade_time).total_seconds()
            required_cooldown_sec = self.risk_config.cooldown_minutes * 60
            if time_since_last_sec < required_cooldown_sec:
                rem_min = (required_cooldown_sec - time_since_last_sec) / 60
                return RiskDecision(
                    approved=False,
                    outcome="COOLDOWN",
                    reason=f"Trade cooldown active ({rem_min:.1f} min remaining)",
                )

        # Rule 5: Position availability
        if signal.action == "buy" and portfolio.coin_quantity > 0.00001:
            return RiskDecision(
                approved=False,
                outcome="POSITION_LIMIT",
                reason=f"Already holding active position ({portfolio.coin_quantity:.6f} coins)",
            )

        if signal.action == "sell" and portfolio.coin_quantity <= 0.00001:
            return RiskDecision(
                approved=False,
                outcome="NO_POSITION",
                reason="No open position available to sell",
            )

        # Rule 6: Cash availability for BUY
        if signal.action == "buy" and portfolio.cash_usd < 10.0:
            return RiskDecision(
                approved=False,
                outcome="INSUFFICIENT_FUNDS",
                reason=f"Available cash ${portfolio.cash_usd:.2f} too low to place buy order",
            )

        # All risk rules passed
        return RiskDecision(approved=True, outcome="APPROVED", reason="All risk checks passed")

    def compute_order_intent(
        self,
        signal: StrategySignal,
        risk: RiskDecision,
        state: MarketState,
        portfolio: PortfolioSnapshot,
    ) -> Optional[OrderIntent]:
        """Calculate order sizing and intent for approved signals."""
        if not risk.approved:
            return None

        price = state.current_price
        if price <= 0:
            return None

        if signal.action == "buy":
            # Sizing: fraction of cash, capped by max_order_usd
            budget = min(
                self.risk_config.max_order_usd,
                portfolio.cash_usd * self.risk_config.order_size_pct,
            )
            # Reserve 1% for taker fee buffer
            budget = min(budget, portfolio.cash_usd * 0.99)
            if budget < 5.0:
                return None
            quantity = budget / price
            return OrderIntent(
                pair=state.pair,
                side="buy",
                quantity=quantity,
                price=price,
                order_type="market",
            )
        elif signal.action == "sell":
            if portfolio.coin_quantity <= 0:
                return None
            return OrderIntent(
                pair=state.pair,
                side="sell",
                quantity=portfolio.coin_quantity,
                price=price,
                order_type="market",
            )

        return None

    # ---------------- 3. Master Pipeline Method ----------------
    async def evaluate(
        self,
        pair: str,
        timestamp: datetime,
        candles_df: pd.DataFrame,
        portfolio: PortfolioSnapshot,
    ) -> EvaluationResult:
        """
        Execute the unified Brain pipeline:
        Features -> Strategy -> Risk -> Order Intent.
        """
        # Step 1: Feature Building
        state = self.build_market_state(pair, timestamp, candles_df)

        # Step 2: Strategy Dispatch
        signal = await self.strategy.generate_signal(state, portfolio)

        # Step 3: Risk Gate
        risk = self.apply_risk_rules(signal, portfolio, timestamp)

        # Step 4: Sizing & Order Intent
        order_intent = self.compute_order_intent(signal, risk, state, portfolio)

        # If an order is generated, mark trade timestamp for cooldown
        if order_intent is not None:
            self.last_trade_time = timestamp

        return EvaluationResult(
            timestamp=timestamp,
            state=state,
            signal=signal,
            risk=risk,
            order_intent=order_intent,
        )

    # ---------------- Helper Indicator Calculations ----------------
    def _compute_rsi(self, close: pd.Series, length: int = 14) -> float:
        if len(close) < length + 1:
            return 50.0
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=length, min_periods=length).mean()
        avg_loss = loss.rolling(window=length, min_periods=length).mean()
        last_gain = float(avg_gain.iloc[-1])
        last_loss = float(avg_loss.iloc[-1])
        if last_loss == 0.0:
            return 100.0 if last_gain > 0 else 50.0
        rs = last_gain / last_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _compute_macd(
        self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[float, float, float]:
        if len(close) < slow:
            return 0.0, 0.0, 0.0
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - sig_line
        return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float(hist.iloc[-1])

    def _compute_bollinger(
        self, close: pd.Series, length: int = 20, std: float = 2.0
    ) -> tuple[float, float, float, float]:
        if len(close) < length:
            last = float(close.iloc[-1])
            return last * 1.02, last, last * 0.98, 0.5
        rolling_mean = close.rolling(window=length).mean()
        rolling_std = close.rolling(window=length).std(ddof=0)
        mid = float(rolling_mean.iloc[-1])
        s = float(rolling_std.iloc[-1])
        upper = mid + (std * s)
        lower = mid - (std * s)
        curr = float(close.iloc[-1])
        width = upper - lower
        pct = (curr - lower) / width if width > 0 else 0.5
        return upper, mid, lower, float(pct)

    def _compute_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
    ) -> tuple[float, str]:
        if len(close) < length + 1:
            return 0.0, "medium"
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = float(tr.rolling(window=length).mean().iloc[-1])
        curr_price = float(close.iloc[-1])
        atr_pct = (atr_val / curr_price) * 100 if curr_price > 0 else 0.0
        regime = "low" if atr_pct < 0.2 else "high" if atr_pct > 0.8 else "medium"
        return atr_val, regime
