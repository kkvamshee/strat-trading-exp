"""
Pydantic data models for market data, brain evaluation, trading signals, portfolios, and configurations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Candle(BaseModel):
    pair: str
    timestamp: float  # Unix timestamp in seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "websocket"  # "websocket" | "rest_backfill"


class RawTick(BaseModel):
    pair: str
    timestamp: float
    price: float
    volume_24h: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None


class MarketState(BaseModel):
    pair: str
    timestamp: datetime
    current_price: float
    change_5m_pct: float = 0.0
    change_15m_pct: float = 0.0
    change_1h_pct: float = 0.0
    rsi_14: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    bb_percent: float = 0.5  # 0.0 = lower band, 1.0 = upper band
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    atr_14: float = 0.0
    atr_regime: Literal["low", "medium", "high"] = "medium"
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    vwap: float = 0.0
    volume_ratio: float = 1.0  # current vol vs rolling avg vol
    candle_count: int = 0


class StrategySignal(BaseModel):
    strategy_id: str
    action: Literal["buy", "sell", "hold"]
    conviction: float = Field(default=0.0, ge=0.0, le=10.0)
    risk_level: float = Field(default=5.0, ge=0.0, le=10.0)
    is_trending: Optional[bool] = None
    rationale: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


class RiskDecision(BaseModel):
    approved: bool
    outcome: str  # "APPROVED" | "SKIP_CONVICTION" | "SKIP_RISK" | "DAILY_LOSS_CAP" | "COOLDOWN" | "POSITION_LIMIT" | "NO_POSITION" | "INSUFFICIENT_FUNDS"
    reason: Optional[str] = None


class OrderIntent(BaseModel):
    pair: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    order_type: Literal["market", "limit"] = "market"


class EvaluationResult(BaseModel):
    timestamp: datetime
    state: MarketState
    signal: StrategySignal
    risk: RiskDecision
    order_intent: Optional[OrderIntent] = None


class TradeResult(BaseModel):
    id: Optional[int] = None
    run_id: str
    strategy_id: str
    pair: str
    side: Literal["buy", "sell"]
    quantity: float
    fill_price: float
    fill_value_usd: float
    fee_usd: float
    slippage_pct: float = 0.0
    entry_trade_id: Optional[int] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    timestamp: datetime
    portfolio_value_after: float


class PortfolioSnapshot(BaseModel):
    run_id: str
    strategy_id: str
    timestamp: datetime
    cash_usd: float
    coin_quantity: float = 0.0
    coin_price: float
    avg_entry_price: Optional[float] = None
    open_trade_id: Optional[int] = None
    total_value_usd: float
    high_water_mark: float
    daily_pnl_usd: Optional[float] = 0.0
    daily_pnl_pct: Optional[float] = 0.0
    total_trades: int = 0


class StrategyRunSummary(BaseModel):
    run_id: str
    strategy_id: str
    mode: str  # "replay" | "paper" | "live"
    pair: str
    start_time: str
    end_time: Optional[str] = None
    initial_cash: float
    final_value: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    config_snapshot: Optional[str] = None


# Configuration Models
class SystemConfig(BaseModel):
    default_pair: str = "BTC-USD"
    db_path: str = "data/trading.db"
    log_path: str = "logs/bot.log"
    log_level: str = "INFO"


class FeatureConfig(BaseModel):
    candle_lookback: int = 200
    rsi_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_length: int = 20
    bb_std: float = 2.0
    atr_length: int = 14


class RiskConfig(BaseModel):
    min_conviction: float = 6.0
    max_risk_level: float = 7.0
    daily_loss_cap_pct: float = 3.0
    cooldown_minutes: int = 5
    max_order_usd: float = 100.0
    order_size_pct: float = 0.10


class PaperConfig(BaseModel):
    default_cash_usd: float = 10000.0
    taker_fee_pct: float = 0.006
    slippage_pct: float = 0.0005


class LiveConfig(BaseModel):
    order_size_usd: float = 100.0
    max_order_size_usd: float = 500.0
    timeout_sec: int = 30


class ReplayConfig(BaseModel):
    default_start_time: str = "2d"
    step_interval_sec: int = 60
    snapshot_interval_sec: int = 300


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    strategies: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
