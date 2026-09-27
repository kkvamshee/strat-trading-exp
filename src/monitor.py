"""
Terminal Monitoring Dashboard, Watchdog, and Visual Reporting using Rich.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.models import (
    MarketState,
    PortfolioSnapshot,
    RiskDecision,
    StrategyRunSummary,
    StrategySignal,
)

console = Console()


class TerminalMonitor:
    """Provides styled terminal outputs and dashboards."""

    @staticmethod
    def print_banner(text: str = "MODULAR STRATEGY TRADING SYSTEM v2.0") -> None:
        console.print(
            Panel(
                f"[bold cyan]{text}[/bold cyan]\n"
                "[dim]Unified Brain | Multi-Strategy Framework | Historical Replay | Paper & Live[/dim]",
                border_style="cyan",
            )
        )

    @staticmethod
    def render_comparison_table(df: pd.DataFrame, title: str = "Strategy Comparative Performance") -> None:
        """Display a formatted Rich table comparing multiple strategy backtests."""
        if df.empty:
            console.print("[yellow]No comparison data available.[/yellow]")
            return

        table = Table(title=f"[bold green]{title}[/bold green]", border_style="blue")
        for col in df.columns:
            table.add_column(str(col), justify="center" if col not in ("Strategy", "Run ID") else "left")

        for _, row in df.iterrows():
            formatted_cells = []
            for col in df.columns:
                val = str(row[col])
                if "P&L" in col:
                    if val.startswith("+"):
                        val = f"[bold green]{val}[/bold green]"
                    elif val.startswith("-"):
                        val = f"[bold red]{val}[/bold red]"
                elif col == "Strategy":
                    val = f"[bold yellow]{val}[/bold yellow]"
                formatted_cells.append(val)
            table.add_row(*formatted_cells)

        console.print(table)

    @staticmethod
    def render_summary(summary: StrategyRunSummary) -> None:
        """Print detailed summary panel for a single run."""
        pnl_color = "green" if summary.total_pnl_pct >= 0 else "red"
        text = (
            f"[bold]Strategy:[/bold] {summary.strategy_id}   [bold]Run ID:[/bold] {summary.run_id}\n"
            f"[bold]Pair:[/bold] {summary.pair}   [bold]Mode:[/bold] {summary.mode}\n"
            f"[bold]Time Range:[/bold] {summary.start_time} -> {summary.end_time}\n"
            f"[bold]Initial Cash:[/bold] ${summary.initial_cash:,.2f} -> [bold]Final Value:[/bold] ${summary.final_value:,.2f}\n"
            f"[bold]Total P&L:[/bold] [{pnl_color}]{summary.total_pnl_pct:+.2f}%[/{pnl_color}]\n"
            f"[bold]Total Trades:[/bold] {summary.total_trades} (Win Rate: [cyan]{summary.win_rate:.1f}%[/cyan])\n"
            f"[bold]Winning / Losing:[/bold] [green]{summary.winning_trades}[/green] / [red]{summary.losing_trades}[/red]\n"
            f"[bold]Max Drawdown:[/bold] [red]{summary.max_drawdown:.2f}%[/red]   "
            f"[bold]Profit Factor:[/bold] {summary.profit_factor}   "
            f"[bold]Sharpe Ratio:[/bold] {summary.sharpe_ratio}"
        )
        console.print(Panel(text, title=f"Run Summary: {summary.strategy_id}", border_style=pnl_color))

    @staticmethod
    def render_live_status(
        pair: str,
        current_price: float,
        state: Optional[MarketState] = None,
        signal: Optional[StrategySignal] = None,
        risk: Optional[RiskDecision] = None,
        portfolio: Optional[PortfolioSnapshot] = None,
    ) -> None:
        """Render a single-cycle status update for live or paper streaming."""
        table = Table(show_header=False, border_style="dim")
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Pair & Price", f"{pair} @ ${current_price:,.2f}")
        if state:
            table.add_row(
                "Technical State",
                f"RSI={state.rsi_14:.1f} | MACD Hist={state.macd_hist:+.4f} | BB%={state.bb_percent:.2f}",
            )
        if signal:
            act_col = "green" if signal.action == "buy" else "red" if signal.action == "sell" else "yellow"
            table.add_row(
                "Strategy Signal",
                f"[{act_col}]{signal.action.upper()}[/{act_col}] (Conviction: {signal.conviction:.1f}/10, Risk: {signal.risk_level:.1f}/10)",
            )
        if risk:
            r_col = "green" if risk.approved else "magenta"
            table.add_row("Risk Gate", f"[{r_col}]{risk.outcome}[/{r_col}] ({risk.reason or 'Passed'})")
        if portfolio:
            pnl = portfolio.total_value_usd - portfolio.high_water_mark
            table.add_row(
                "Virtual Portfolio",
                f"Cash: ${portfolio.cash_usd:,.2f} | Coins: {portfolio.coin_quantity:.6f} | Total: ${portfolio.total_value_usd:,.2f}",
            )

        console.print(table)
