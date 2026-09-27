"""
Multi-Portfolio Paper Trading Simulator with realistic fee and slippage modeling.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple
from loguru import logger

from src.database import Database
from src.models import (
    OrderIntent,
    PaperConfig,
    PortfolioSnapshot,
    TradeResult,
)


class PaperTradingEngine:
    """Simulates realistic order fills with fee and slippage deduction across isolated portfolios."""

    def __init__(self, db: Database, config: Optional[PaperConfig] = None):
        self.db = db
        self.config = config or PaperConfig()
        # In-memory cache for active run portfolios: (run_id, strategy_id) -> PortfolioSnapshot
        self._portfolios: Dict[Tuple[str, str], PortfolioSnapshot] = {}

    def get_portfolio(
        self,
        run_id: str,
        strategy_id: str,
        current_price: float = 0.0,
        initial_cash: Optional[float] = None,
    ) -> PortfolioSnapshot:
        """Fetch or initialize the portfolio for a strategy run."""
        key = (run_id, strategy_id)
        if key not in self._portfolios:
            init_cash = initial_cash or self.config.default_cash_usd
            snapshot = self.db.get_or_create_portfolio(
                run_id=run_id,
                strategy_id=strategy_id,
                initial_cash=init_cash,
                coin_price=current_price,
            )
            self._portfolios[key] = snapshot

        # Refresh total value with latest price
        p = self._portfolios[key]
        if current_price > 0:
            p.coin_price = current_price
            p.total_value_usd = p.cash_usd + (p.coin_quantity * current_price)
            if p.total_value_usd > p.high_water_mark:
                p.high_water_mark = p.total_value_usd
        return p

    def execute_order(
        self,
        intent: OrderIntent,
        run_id: str,
        strategy_id: str,
        timestamp: datetime,
        decision_id: Optional[int] = None,
    ) -> Optional[TradeResult]:
        """Execute a simulated fill, apply fees and slippage, and persist updates."""
        portfolio = self.get_portfolio(run_id, strategy_id, current_price=intent.price)
        slippage = self.config.slippage_pct
        taker_fee = self.config.taker_fee_pct

        if intent.side == "buy":
            # Apply upward slippage for market buy
            fill_price = intent.price * (1.0 + slippage)
            gross_value = intent.quantity * fill_price
            fee = gross_value * taker_fee
            total_cost = gross_value + fee

            # Ensure budget doesn't exceed cash
            if total_cost > portfolio.cash_usd:
                if portfolio.cash_usd < 5.0:
                    logger.warning(
                        f"Paper execution rejected for {strategy_id}: Insufficient cash (${portfolio.cash_usd:.2f})"
                    )
                    return None
                # Downsize quantity to fit available cash
                gross_value = portfolio.cash_usd / (1.0 + taker_fee)
                fee = gross_value * taker_fee
                total_cost = gross_value + fee
                quantity = gross_value / fill_price
            else:
                quantity = intent.quantity

            # Update portfolio balances
            portfolio.cash_usd -= total_cost
            prev_qty = portfolio.coin_quantity
            new_qty = prev_qty + quantity
            if new_qty > 0:
                prev_cost = prev_qty * (portfolio.avg_entry_price or fill_price)
                portfolio.avg_entry_price = (prev_cost + gross_value) / new_qty
            portfolio.coin_quantity = new_qty
            portfolio.total_trades += 1
            portfolio.timestamp = timestamp
            portfolio.total_value_usd = portfolio.cash_usd + (portfolio.coin_quantity * fill_price)

            trade = TradeResult(
                id=decision_id,
                run_id=run_id,
                strategy_id=strategy_id,
                pair=intent.pair,
                side="buy",
                quantity=quantity,
                fill_price=fill_price,
                fill_value_usd=gross_value,
                fee_usd=fee,
                slippage_pct=slippage,
                timestamp=timestamp,
                portfolio_value_after=portfolio.total_value_usd,
            )
            trade_id = self.db.record_trade(trade)
            portfolio.open_trade_id = trade_id
            self.db.update_portfolio(portfolio)
            logger.info(
                f"[{strategy_id}] PAPER BUY {quantity:.6f} {intent.pair} @ ${fill_price:,.2f} "
                f"(fee: ${fee:.2f}, port: ${portfolio.total_value_usd:,.2f})"
            )
            return trade

        elif intent.side == "sell":
            if portfolio.coin_quantity <= 0:
                logger.warning(f"Paper execution rejected for {strategy_id}: No position to sell.")
                return None

            quantity = min(intent.quantity, portfolio.coin_quantity)
            # Apply downward slippage for market sell
            fill_price = intent.price * (1.0 - slippage)
            gross_value = quantity * fill_price
            fee = gross_value * taker_fee
            net_proceeds = gross_value - fee

            cost_basis = quantity * (portfolio.avg_entry_price or fill_price)
            pnl_usd = net_proceeds - cost_basis
            pnl_pct = (pnl_usd / cost_basis * 100.0) if cost_basis > 0 else 0.0

            # Update portfolio balances
            portfolio.cash_usd += net_proceeds
            portfolio.coin_quantity -= quantity
            if portfolio.coin_quantity <= 0.00001:
                portfolio.coin_quantity = 0.0
                portfolio.avg_entry_price = None

            portfolio.total_trades += 1
            portfolio.timestamp = timestamp
            portfolio.total_value_usd = portfolio.cash_usd + (portfolio.coin_quantity * fill_price)

            trade = TradeResult(
                id=decision_id,
                run_id=run_id,
                strategy_id=strategy_id,
                pair=intent.pair,
                side="sell",
                quantity=quantity,
                fill_price=fill_price,
                fill_value_usd=gross_value,
                fee_usd=fee,
                slippage_pct=slippage,
                entry_trade_id=portfolio.open_trade_id,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                timestamp=timestamp,
                portfolio_value_after=portfolio.total_value_usd,
            )
            portfolio.open_trade_id = None
            self.db.record_trade(trade)
            self.db.update_portfolio(portfolio)
            logger.info(
                f"[{strategy_id}] PAPER SELL {quantity:.6f} {intent.pair} @ ${fill_price:,.2f} "
                f"P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%), port: ${portfolio.total_value_usd:,.2f}"
            )
            return trade

        return None
