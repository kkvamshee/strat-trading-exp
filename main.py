"""
Main Entry Point & CLI for the Modular Strategy Trading System.
Provides commands for:
  - replay: Backtest & compare multiple strategies from any arbitrary start time
  - stream: Real-time paper or live trading with Coinbase WebSocket
  - compare: Compare historical strategy performance runs
  - backfill: Pre-fetch historical candles into SQLite
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from src.brain import StrategyBrain
from src.coinbase_connector import CoinbaseConnector
from src.config import load_config
from src.database import Database
from src.models import Candle, OrderIntent
from src.monitor import TerminalMonitor
from src.order_executor import LiveOrderExecutor
from src.paper_engine import PaperTradingEngine
from src.performance_tracker import PerformanceTracker
from src.replay_engine import HistoricalReplayEngine, parse_time_argument
from src.strategies.registry import build_strategy, list_available_strategies


def setup_logging(log_path: str = "logs/bot.log", level: str = "INFO") -> None:
    """Configure Loguru structured logging to stdout and rotating file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{message}</cyan>",
        level=level,
    )
    logger.add(
        log_path,
        rotation="50 MB",
        retention="10 days",
        level="DEBUG",
        enqueue=True,
    )


async def cmd_replay(args: argparse.Namespace) -> None:
    """Run historical backtest from an arbitrary start time on one or more strategies."""
    config = load_config(args.config)
    setup_logging(config.system.log_path, config.system.log_level)
    TerminalMonitor.print_banner("HISTORICAL REPLAY & MULTI-STRATEGY BACKTEST")

    db = Database(config.system.db_path)
    connector = CoinbaseConnector(
        api_key=os.getenv("COINBASE_API_KEY"),
        api_secret=os.getenv("COINBASE_API_SECRET"),
    )
    replay_engine = HistoricalReplayEngine(db=db, config=config, connector=connector)

    # Determine strategies
    raw_strats = args.strategies.strip()
    if raw_strats.lower() == "all":
        strat_names = list_available_strategies()
    else:
        strat_names = [s.strip() for s in raw_strats.split(",") if s.strip()]

    pair = args.pair or config.system.default_pair
    start_time = args.start_time or config.replay.default_start_time
    initial_cash = args.initial_cash or config.paper.default_cash_usd

    logger.info(f"Target Strategies: {strat_names} | Pair: {pair} | Start: {start_time}")

    summaries, comparison_df = await replay_engine.run_replay(
        start_time=start_time,
        end_time=args.end_time,
        strategies=strat_names,
        pair=pair,
        initial_cash=initial_cash,
        auto_backfill=not args.no_backfill,
    )

    # Print results
    TerminalMonitor.render_comparison_table(
        comparison_df, title=f"Comparative Strategy Leaderboard ({pair})"
    )
    for s_id, summary in summaries.items():
        TerminalMonitor.render_summary(summary)


async def cmd_stream(args: argparse.Namespace) -> None:
    """Stream live WebSocket data and execute paper or live trades."""
    config = load_config(args.config)
    setup_logging(config.system.log_path, config.system.log_level)
    mode = args.mode.lower()
    TerminalMonitor.print_banner(f"STREAMING EXECUTION [{mode.upper()} MODE]")

    db = Database(config.system.db_path)
    connector = CoinbaseConnector(
        api_key=os.getenv("COINBASE_API_KEY"),
        api_secret=os.getenv("COINBASE_API_SECRET"),
    )
    pair = args.pair or config.system.default_pair
    strategy_name = args.strategy
    run_id = f"stream-{mode}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Catchup replay if requested
    if args.catchup_from:
        logger.info(f"Warming up and catching up from {args.catchup_from} ago...")
        replay_engine = HistoricalReplayEngine(db=db, config=config, connector=connector)
        await replay_engine.run_replay(
            start_time=args.catchup_from,
            strategies=[strategy_name],
            pair=pair,
            auto_backfill=True,
        )
        logger.info("Catch-up complete. Seamlessly connecting to live WebSocket stream...")

    strat_obj = build_strategy(strategy_name, config.strategies.get(strategy_name))
    brain = StrategyBrain(
        strategy=strat_obj,
        risk_config=config.risk,
        feature_config=config.features,
    )

    paper_engine = PaperTradingEngine(db=db, config=config.paper)
    live_executor = (
        LiveOrderExecutor(db=db, config=config.live) if mode == "live" else None
    )

    async def handle_candle(candle: Candle) -> None:
        try:
            now_dt = datetime.fromtimestamp(candle.timestamp, tz=timezone.utc)
            candles_df = db.get_latest_candles_df(pair, n=config.features.candle_lookback)
            if len(candles_df) < 20:
                logger.warning(f"Waiting for candle history ({len(candles_df)} / 20)...")
                return

            portfolio = paper_engine.get_portfolio(run_id, strat_obj.strategy_id, current_price=candle.close)

            eval_res = await brain.evaluate(pair, now_dt, candles_df, portfolio)

            dec_id = db.record_decision(
                run_id=run_id,
                strategy_id=strat_obj.strategy_id,
                pair=pair,
                timestamp=now_dt,
                market_state=eval_res.state,
                signal=eval_res.signal,
                risk=eval_res.risk,
                mode=mode,
            )

            TerminalMonitor.render_live_status(
                pair=pair,
                current_price=candle.close,
                state=eval_res.state,
                signal=eval_res.signal,
                risk=eval_res.risk,
                portfolio=portfolio,
            )

            if eval_res.order_intent:
                if mode == "live" and live_executor:
                    await live_executor.execute_live_order(
                        intent=eval_res.order_intent,
                        run_id=run_id,
                        strategy_id=strat_obj.strategy_id,
                        decision_id=dec_id,
                    )
                else:
                    paper_engine.execute_order(
                        intent=eval_res.order_intent,
                        run_id=run_id,
                        strategy_id=strat_obj.strategy_id,
                        timestamp=now_dt,
                        decision_id=dec_id,
                    )

        except Exception as e:
            logger.error(f"Error processing candle: {e}")

    await connector.start_stream(pair=pair, on_candle=handle_candle, db=db)


async def cmd_compare(args: argparse.Namespace) -> None:
    """Print comparative table of past strategy runs."""
    config = load_config(args.config)
    db = Database(config.system.db_path)
    runs = db.list_strategy_runs(limit=args.limit)

    if not runs:
        print("No strategy runs recorded in database yet.")
        return

    if args.runs:
        filter_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
        runs = [r for r in runs if r.run_id in filter_ids or r.strategy_id in filter_ids]

    df = PerformanceTracker.build_comparison_dataframe(runs)
    TerminalMonitor.render_comparison_table(df, title="Historical Strategy Runs Comparison")


async def cmd_backfill(args: argparse.Namespace) -> None:
    """Backfill historical candles into SQLite via Coinbase REST."""
    config = load_config(args.config)
    setup_logging(config.system.log_path, config.system.log_level)
    db = Database(config.system.db_path)
    connector = CoinbaseConnector()
    pair = args.pair or config.system.default_pair
    days = args.days or 2
    end_ts = datetime.now(timezone.utc).timestamp()
    start_ts = end_ts - (days * 86400)

    logger.info(f"Backfilling {days} days of 1m candles for {pair}...")
    count = await connector.ensure_candles_available(pair, start_ts, end_ts, db)
    logger.info(f"Backfill complete! Saved {count} candles to {config.system.db_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modular Strategy Trading System — Multi-Strategy, Historical Replay & Paper/Live Trading"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Replay Command
    p_replay = subparsers.add_parser(
        "replay", help="Backtest and compare multiple strategies from any start time"
    )
    p_replay.add_argument("-s", "--start-time", type=str, default="2d", help="Start time: e.g. '2d', '48h', '2026-09-25T00:00:00Z'")
    p_replay.add_argument("-e", "--end-time", type=str, default=None, help="End time (defaults to now)")
    p_replay.add_argument("--strategies", type=str, default="jev,ema_macd,rsi_bb", help="Comma-separated strategy names or 'all'")
    p_replay.add_argument("-p", "--pair", type=str, default=None, help="Coin pair (e.g. BTC-USD)")
    p_replay.add_argument("--initial-cash", type=float, default=None, help="Starting cash per strategy")
    p_replay.add_argument("--no-backfill", action="store_true", help="Do not auto-backfill missing candles")

    # Stream Command
    p_stream = subparsers.add_parser("stream", help="Live WebSocket streaming (paper or live mode)")
    p_stream.add_argument("-m", "--mode", choices=["paper", "live"], default="paper", help="Execution mode")
    p_stream.add_argument("--strategy", type=str, default="jev", help="Active strategy to run")
    p_stream.add_argument("-p", "--pair", type=str, default=None, help="Coin pair (e.g. BTC-USD)")
    p_stream.add_argument("--catchup-from", type=str, default=None, help="Warm-up / replay duration before streaming (e.g. '6h')")

    # Compare Command
    p_compare = subparsers.add_parser("compare", help="Compare past strategy runs")
    p_compare.add_argument("--runs", type=str, default=None, help="Filter by specific run IDs or strategy names")
    p_compare.add_argument("--limit", type=int, default=20, help="Max runs to display")

    # Backfill Command
    p_backfill = subparsers.add_parser("backfill", help="Fetch historical candles from Coinbase REST")
    p_backfill.add_argument("-p", "--pair", type=str, default=None, help="Coin pair (e.g. BTC-USD)")
    p_backfill.add_argument("-d", "--days", type=int, default=2, help="Number of days to backfill")

    args = parser.parse_args()

    if args.command == "replay":
        asyncio.run(cmd_replay(args))
    elif args.command == "stream":
        asyncio.run(cmd_stream(args))
    elif args.command == "compare":
        asyncio.run(cmd_compare(args))
    elif args.command == "backfill":
        asyncio.run(cmd_backfill(args))


if __name__ == "__main__":
    main()
