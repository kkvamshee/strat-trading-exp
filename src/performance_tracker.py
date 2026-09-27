"""
Performance Tracker & Multi-Strategy Comparator.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from src.models import (
    PortfolioSnapshot,
    StrategyRunSummary,
    TradeResult,
)


class PerformanceTracker:
    """Calculates quantitative performance analytics and comparative leaderboards."""

    @staticmethod
    def calculate_summary(
        run_id: str,
        strategy_id: str,
        mode: str,
        pair: str,
        start_time: str,
        end_time: str,
        initial_cash: float,
        final_value: float,
        trades: List[TradeResult],
        snapshots: Optional[List[PortfolioSnapshot]] = None,
        config_snapshot: Optional[str] = None,
    ) -> StrategyRunSummary:
        """Compute full statistical summary for a strategy execution run."""
        closed_trades = [t for t in trades if t.side == "sell" and t.pnl_usd is not None]
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if (t.pnl_usd or 0) > 0])
        losing_trades = len([t for t in closed_trades if (t.pnl_usd or 0) < 0])

        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        total_pnl_pct = ((final_value - initial_cash) / initial_cash * 100.0) if initial_cash > 0 else 0.0

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in closed_trades if (t.pnl_usd or 0) > 0)
        gross_loss = abs(sum(t.pnl_usd for t in closed_trades if (t.pnl_usd or 0) < 0))
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = 99.9  # No losses
        else:
            profit_factor = 0.0

        # Max drawdown calculation
        max_drawdown = 0.0
        if snapshots and len(snapshots) > 1:
            values = pd.Series([s.total_value_usd for s in snapshots])
            running_max = values.cummax()
            drawdowns = (running_max - values) / running_max * 100.0
            max_drawdown = float(drawdowns.max())
        elif total_trades > 0:
            # Approximate from equity after trades
            vals = pd.Series([initial_cash] + [t.portfolio_value_after for t in trades])
            running_max = vals.cummax()
            drawdowns = (running_max - vals) / running_max * 100.0
            max_drawdown = float(drawdowns.max())

        # Sharpe ratio
        sharpe_ratio = 0.0
        if closed_trades:
            returns = [t.pnl_pct or 0.0 for t in closed_trades]
            if len(returns) > 1:
                std = float(np.std(returns))
                mean = float(np.mean(returns))
                if std > 0:
                    sharpe_ratio = round(mean / std, 2)

        return StrategyRunSummary(
            run_id=run_id,
            strategy_id=strategy_id,
            mode=mode,
            pair=pair,
            start_time=start_time,
            end_time=end_time,
            initial_cash=initial_cash,
            final_value=final_value,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_pnl_pct=round(total_pnl_pct, 2),
            max_drawdown=round(max_drawdown, 2),
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            config_snapshot=config_snapshot,
        )

    @staticmethod
    def build_comparison_dataframe(summaries: List[StrategyRunSummary]) -> pd.DataFrame:
        """Convert a list of run summaries into a comparison DataFrame sorted by Total PnL %."""
        records = []
        for s in summaries:
            records.append({
                "Strategy": s.strategy_id,
                "Run ID": s.run_id,
                "Final Value ($)": f"${s.final_value:,.2f}",
                "Total P&L (%)": f"{s.total_pnl_pct:+.2f}%",
                "Trades": s.total_trades,
                "Win Rate (%)": f"{s.win_rate:.1f}%",
                "Max DD (%)": f"{s.max_drawdown:.2f}%",
                "Profit Factor": s.profit_factor,
                "Sharpe": s.sharpe_ratio,
            })
        df = pd.DataFrame(records)
        if not df.empty and "Total P&L (%)" in df.columns:
            # Sort descending by numeric total pnl
            df["_pnl_num"] = [s.total_pnl_pct for s in summaries]
            df = df.sort_values("_pnl_num", ascending=False).drop(columns=["_pnl_num"]).reset_index(drop=True)
        return df
