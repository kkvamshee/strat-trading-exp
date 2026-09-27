"""
Historical Replay & Multi-Strategy Backtesting Engine.
Executes simulation from any arbitrary start time across past market data without lookahead bias.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
from loguru import logger

from src.brain import StrategyBrain
from src.coinbase_connector import CoinbaseConnector
from src.database import Database
from src.models import (
    AppConfig,
    FeatureConfig,
    PortfolioSnapshot,
    RiskConfig,
    StrategyRunSummary,
)
from src.paper_engine import PaperTradingEngine
from src.performance_tracker import PerformanceTracker
from src.strategies.base import BaseStrategy
from src.strategies.registry import build_strategy


def parse_time_argument(val: Union[str, float, datetime]) -> float:
    """
    Parse a time argument which can be:
    - Relative duration like '2d', '48h', '30m', '1w'
    - ISO 8601 string like '2026-09-25T00:00:00Z'
    - Unix timestamp float/int
    - datetime object
    """
    now_ts = datetime.now(timezone.utc).timestamp()
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.timestamp()

    val_str = str(val).strip().lower()

    # Numeric string (e.g. '1790300400.0')
    try:
        return float(val_str)
    except ValueError:
        pass

    # Relative match: e.g. 2d, 12h, 30m, 1w
    match = re.match(r"^(\d+)\s*([mhdws])$", val_str)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        multipliers = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
        }
        sec_offset = amount * multipliers[unit]
        return now_ts - sec_offset

    # ISO format match
    try:
        dt = datetime.fromisoformat(val_str.replace("z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        raise ValueError(
            f"Cannot parse time '{val}'. Use relative format like '2d', '48h' or ISO timestamp like '2026-09-25T00:00:00Z'."
        )


class HistoricalReplayEngine:
    """Runs one or multiple strategies against historical data starting from any arbitrary time."""

    def __init__(
        self,
        db: Database,
        config: Optional[AppConfig] = None,
        connector: Optional[CoinbaseConnector] = None,
    ):
        self.db = db
        self.config = config or AppConfig()
        self.connector = connector or CoinbaseConnector()
        self.paper_engine = PaperTradingEngine(db=self.db, config=self.config.paper)

    async def run_replay(
        self,
        start_time: Union[str, float, datetime],
        strategies: List[Union[str, BaseStrategy]],
        end_time: Optional[Union[str, float, datetime]] = None,
        pair: Optional[str] = None,
        initial_cash: Optional[float] = None,
        step_interval_sec: int = 60,
        snapshot_interval_sec: int = 300,
        auto_backfill: bool = True,
        progress_callback: Optional[Callable[[datetime, int, int, Dict[str, float]], Any]] = None,
    ) -> Tuple[Dict[str, StrategyRunSummary], pd.DataFrame]:
        """
        Run historical replay across past data from start_time to end_time.
        Tests all specified strategies concurrently on identical market data.
        """
        pair = pair or self.config.system.default_pair
        initial_cash = initial_cash or self.config.paper.default_cash_usd
        start_ts = parse_time_argument(start_time)
        end_ts = parse_time_argument(end_time) if end_time is not None else datetime.now(timezone.utc).timestamp()

        if start_ts >= end_ts:
            raise ValueError(f"start_time ({start_ts}) must be before end_time ({end_ts})")

        # Calculate buffer needed for indicators (e.g. 200 candles before start_time)
        lookback_sec = self.config.features.candle_lookback * 60
        data_start_ts = start_ts - lookback_sec

        logger.info(
            f"Initializing Replay for {pair}: "
            f"Start={datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()} -> "
            f"End={datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()}"
        )

        # Ensure historical data is available in SQLite
        if auto_backfill:
            await self.connector.ensure_candles_available(
                pair, data_start_ts, end_ts, self.db, granularity=60
            )

        # Load candle history into memory for ultra-fast replay
        candles_df = self.db.get_candles_df(pair, start_ts=data_start_ts, end_ts=end_ts)
        if candles_df.empty or len(candles_df) < 10:
            raise ValueError(
                f"Insufficient historical candle data for {pair} between {data_start_ts} and {end_ts}. "
                f"Found {len(candles_df)} candles."
            )

        # Prepare strategy instances and brains
        session_id = f"replay-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        active_brains: Dict[str, Tuple[StrategyBrain, str]] = {}  # strategy_id -> (brain, run_id)

        for strat_input in strategies:
            if isinstance(strat_input, str):
                strat_obj = build_strategy(strat_input, self.config.strategies.get(strat_input))
            else:
                strat_obj = strat_input

            run_id = f"{session_id}-{strat_obj.strategy_id}"
            brain = StrategyBrain(
                strategy=strat_obj,
                risk_config=self.config.risk,
                feature_config=self.config.features,
            )
            # Initialize portfolio in database
            self.paper_engine.get_portfolio(run_id, strat_obj.strategy_id, initial_cash=initial_cash)
            active_brains[strat_obj.strategy_id] = (brain, run_id)

        # Filter timestamps where simulation will evaluate (from start_ts forward)
        eval_candles = candles_df[candles_df["timestamp"] >= start_ts].sort_values("timestamp")
        total_steps = len(eval_candles)
        last_snapshot_ts = 0.0

        logger.info(f"Stepping through {total_steps} historical intervals for {len(active_brains)} strategies...")

        for idx, (_, row) in enumerate(eval_candles.iterrows()):
            t_curr = float(row["timestamp"])
            dt_curr = datetime.fromtimestamp(t_curr, tz=timezone.utc)
            curr_price = float(row["close"])

            # Slice candles up to current timestamp T (strictly no lookahead bias)
            history_slice = candles_df[candles_df["timestamp"] <= t_curr].tail(
                self.config.features.candle_lookback
            )

            # Evaluate each strategy
            for strat_id, (brain, run_id) in active_brains.items():
                portfolio = self.paper_engine.get_portfolio(
                    run_id, strat_id, current_price=curr_price
                )

                eval_result = await brain.evaluate(pair, dt_curr, history_slice, portfolio)

                # Persist decision to DB
                dec_id = self.db.record_decision(
                    run_id=run_id,
                    strategy_id=strat_id,
                    pair=pair,
                    timestamp=dt_curr,
                    market_state=eval_result.state,
                    signal=eval_result.signal,
                    risk=eval_result.risk,
                    mode="replay",
                )

                # Execute simulated trade if approved
                if eval_result.order_intent:
                    self.paper_engine.execute_order(
                        intent=eval_result.order_intent,
                        run_id=run_id,
                        strategy_id=strat_id,
                        timestamp=dt_curr,
                        decision_id=dec_id,
                    )

                # Periodic portfolio snapshot
                if t_curr - last_snapshot_ts >= snapshot_interval_sec or idx == total_steps - 1:
                    snap = PortfolioSnapshot(
                        run_id=run_id,
                        strategy_id=strat_id,
                        timestamp=dt_curr,
                        cash_usd=portfolio.cash_usd,
                        coin_quantity=portfolio.coin_quantity,
                        coin_price=curr_price,
                        total_value_usd=portfolio.cash_usd + (portfolio.coin_quantity * curr_price),
                        high_water_mark=portfolio.high_water_mark,
                        daily_pnl_usd=0.0,
                        daily_pnl_pct=0.0,
                    )
                    self.db.save_portfolio_snapshot(snap)

            if t_curr - last_snapshot_ts >= snapshot_interval_sec:
                last_snapshot_ts = t_curr

            if progress_callback:
                p_stats = {
                    s_id: self.paper_engine.get_portfolio(r_id, s_id, curr_price).total_value_usd
                    for s_id, (_, r_id) in active_brains.items()
                }
                res = progress_callback(dt_curr, total_steps, idx + 1, p_stats)
                if asyncio.iscoroutine(res):
                    await res

        # Finalize and calculate metrics for each strategy
        summaries: Dict[str, StrategyRunSummary] = {}
        for strat_id, (_, run_id) in active_brains.items():
            final_port = self.paper_engine.get_portfolio(run_id, strat_id)
            trades = self.db.get_run_trades(run_id, strat_id)
            snapshots = self.db.get_portfolio_snapshots(run_id, strat_id)

            summary = PerformanceTracker.calculate_summary(
                run_id=run_id,
                strategy_id=strat_id,
                mode="replay",
                pair=pair,
                start_time=datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
                end_time=datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
                initial_cash=initial_cash,
                final_value=final_port.total_value_usd,
                trades=trades,
                snapshots=snapshots,
            )
            self.db.record_strategy_run(summary)
            summaries[strat_id] = summary

        comparison_df = PerformanceTracker.build_comparison_dataframe(list(summaries.values()))
        logger.info(f"Replay complete! Evaluated {len(summaries)} strategies across {total_steps} periods.")
        return summaries, comparison_df
