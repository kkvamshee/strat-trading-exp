"""
Live Order Executor — Coinbase Advanced Trade REST API integration with hard safety guardrails.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from loguru import logger

from src.database import Database
from src.models import LiveConfig, OrderIntent, TradeResult

try:
    from coinbase.rest import RESTClient
    HAS_COINBASE_CLIENT = True
except ImportError:
    HAS_COINBASE_CLIENT = False


class LiveOrderExecutor:
    """Executes real capital orders on Coinbase Advanced Trade API with strict safety bounds."""

    def __init__(
        self,
        db: Database,
        config: Optional[LiveConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.db = db
        self.config = config or LiveConfig()
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")
        self.dry_run = dry_run or (not self.api_key or self.api_key.startswith("your_"))
        self._client: Optional[Any] = None

    def _get_client(self) -> Optional[Any]:
        if not HAS_COINBASE_CLIENT or self.dry_run or not self.api_key or not self.api_secret:
            return None
        if self._client is None:
            self._client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
        return self._client

    async def execute_live_order(
        self,
        intent: OrderIntent,
        run_id: str,
        strategy_id: str,
        decision_id: Optional[int] = None,
    ) -> Optional[TradeResult]:
        """Place and monitor a live market order."""
        client_order_id = str(uuid.uuid4())
        estimated_val = intent.quantity * intent.price

        # Hard dollar ceiling safety enforcement
        if estimated_val > self.config.max_order_size_usd:
            logger.error(
                f"SAFETY TRIP: Order value ${estimated_val:.2f} exceeds hard cap ${self.config.max_order_size_usd:.2f}."
            )
            return None

        now = datetime.now(timezone.utc)

        if self.dry_run:
            logger.warning(
                f"[LIVE DRY-RUN] Would submit {intent.side.upper()} order for {intent.quantity:.6f} {intent.pair} (~${estimated_val:.2f})"
            )
            return TradeResult(
                id=decision_id,
                run_id=run_id,
                strategy_id=strategy_id,
                pair=intent.pair,
                side=intent.side,
                quantity=intent.quantity,
                fill_price=intent.price,
                fill_value_usd=estimated_val,
                fee_usd=estimated_val * 0.006,
                timestamp=now,
                portfolio_value_after=0.0,
            )

        client = self._get_client()
        if not client:
            logger.error("Coinbase live client not initialized. Cannot execute real order.")
            return None

        try:
            side_str = intent.side.upper()
            logger.info(f"Submitting LIVE {side_str} order for {intent.pair}: qty={intent.quantity:.6f}")

            order_response: Dict[str, Any] = {}
            if intent.side == "buy":
                # Coinbase market buy by quote or base size
                order_response = client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=intent.pair,
                    quote_size=str(round(estimated_val, 2)),
                )
            else:
                order_response = client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=intent.pair,
                    base_size=str(round(intent.quantity, 6)),
                )

            logger.info(f"Coinbase live order placed: {order_response}")
            # Record confirmed or submitted order
            return TradeResult(
                id=decision_id,
                run_id=run_id,
                strategy_id=strategy_id,
                pair=intent.pair,
                side=intent.side,
                quantity=intent.quantity,
                fill_price=intent.price,
                fill_value_usd=estimated_val,
                fee_usd=estimated_val * 0.006,
                timestamp=now,
                portfolio_value_after=0.0,
            )

        except Exception as e:
            logger.error(f"Live order placement failed: {e}. NO RETRY attempted for safety.")
            return None
