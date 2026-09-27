"""
Base Strategy Interface for the Modular Strategy Trading System.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.models import MarketState, PortfolioSnapshot, StrategySignal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies (AI, quantitative, rule-based)."""

    def __init__(
        self,
        strategy_id: str,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.parameters = parameters or {}

    @abstractmethod
    async def generate_signal(
        self, state: MarketState, portfolio: PortfolioSnapshot
    ) -> StrategySignal:
        """
        Evaluate current market state and portfolio to produce an action,
        conviction rating (0-10), and perceived risk level (0-10).
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id='{self.strategy_id}' name='{self.name}'>"
