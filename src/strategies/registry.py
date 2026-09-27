"""
Strategy Registry & Factory for Dynamic Strategy Loading.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.strategies.base import BaseStrategy
from src.strategies.jev_strategy import JevStrategy
from src.strategies.technical_strategies import (
    EmaMacdStrategy,
    MomentumStrategy,
    RsiBollingerStrategy,
)

STRATEGY_REGISTRY = {
    "jev": JevStrategy,
    "jev_system_one": JevStrategy,
    "typesafe": JevStrategy,
    "ema_macd": EmaMacdStrategy,
    "ema": EmaMacdStrategy,
    "rsi_bb": RsiBollingerStrategy,
    "rsi": RsiBollingerStrategy,
    "momentum": MomentumStrategy,
}


def build_strategy(name: str, config: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    """Build a strategy instance by name with optional custom parameters."""
    normalized = name.strip().lower()
    strategy_cls = STRATEGY_REGISTRY.get(normalized)
    if not strategy_cls:
        available = ", ".join(list(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")

    params = config or {}
    return strategy_cls(**params)


def list_available_strategies() -> List[str]:
    """Return unique primary strategy identifiers."""
    return ["jev", "ema_macd", "rsi_bb", "momentum"]
