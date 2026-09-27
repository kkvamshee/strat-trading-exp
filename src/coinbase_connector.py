"""
Coinbase Data Connector — Real-time WebSocket ingestion and REST Historical Backfill.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import httpx
from loguru import logger

from src.database import Database
from src.models import Candle, RawTick


class CoinbaseConnector:
    """Manages Coinbase REST backfills and WebSocket live streaming."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rest_base_url: str = "https://api.exchange.coinbase.com",
        ws_url: str = "wss://ws-feed.exchange.coinbase.com",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest_base_url = rest_base_url.rstrip("/")
        self.ws_url = ws_url
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._current_candle: Optional[Dict[str, Any]] = None

    # ---------------- REST Historical Fetching ----------------
    async def fetch_historical_candles(
        self,
        pair: str,
        start_ts: float,
        end_ts: float,
        granularity: int = 60,
    ) -> List[Candle]:
        """Fetch historical candles from Coinbase REST API with pagination (max 300 per call)."""
        candles: List[Candle] = []
        # Chunk window in intervals of 300 * granularity
        chunk_size_sec = 300 * granularity
        curr_start = start_ts

        async with httpx.AsyncClient(timeout=15.0) as client:
            while curr_start < end_ts:
                curr_end = min(curr_start + chunk_size_sec, end_ts)
                start_iso = datetime.fromtimestamp(curr_start, tz=timezone.utc).isoformat()
                end_iso = datetime.fromtimestamp(curr_end, tz=timezone.utc).isoformat()

                url = f"{self.rest_base_url}/products/{pair}/candles"
                params = {
                    "start": start_iso,
                    "end": end_iso,
                    "granularity": granularity,
                }

                try:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        # Format returned by Coinbase Exchange API:
                        # [[time, low, high, open, close, volume], ...]
                        batch_candles = []
                        for row in data:
                            if len(row) >= 6:
                                c = Candle(
                                    pair=pair,
                                    timestamp=float(row[0]),
                                    low=float(row[1]),
                                    high=float(row[2]),
                                    open=float(row[3]),
                                    close=float(row[4]),
                                    volume=float(row[5]),
                                    source="rest_backfill",
                                )
                                batch_candles.append(c)
                        # Sort chronological (Coinbase returns reverse chronological)
                        batch_candles.sort(key=lambda x: x.timestamp)
                        candles.extend(batch_candles)
                    elif resp.status_code == 429:
                        logger.warning("Coinbase REST rate limited; waiting 1.0s...")
                        await asyncio.sleep(1.0)
                        continue
                    else:
                        logger.error(
                            f"Coinbase REST error {resp.status_code}: {resp.text}"
                        )
                except Exception as e:
                    logger.error(f"Failed to fetch candles for {pair}: {e}")

                curr_start = curr_end
                await asyncio.sleep(0.1)  # Respect rate limits

        return candles

    async def ensure_candles_available(
        self,
        pair: str,
        start_ts: float,
        end_ts: float,
        db: Database,
        granularity: int = 60,
    ) -> int:
        """Find missing candle intervals in SQLite and backfill them via REST."""
        missing = db.find_missing_candle_ranges(
            pair, start_ts, end_ts, granularity=granularity
        )
        if not missing:
            logger.info(f"Candle data for {pair} is fully cached from {start_ts} to {end_ts}.")
            return 0

        total_backfilled = 0
        for gap_start, gap_end in missing:
            logger.info(
                f"Backfilling missing range for {pair}: "
                f"{datetime.fromtimestamp(gap_start, tz=timezone.utc)} -> "
                f"{datetime.fromtimestamp(gap_end, tz=timezone.utc)}"
            )
            fetched = await self.fetch_historical_candles(
                pair, gap_start, gap_end, granularity=granularity
            )
            inserted = db.insert_candles(fetched)
            total_backfilled += inserted

        logger.info(f"Successfully backfilled {total_backfilled} candles for {pair}.")
        return total_backfilled

    # ---------------- WebSocket Streaming ----------------
    async def start_stream(
        self,
        pair: str,
        on_candle: Optional[Callable[[Candle], Any]] = None,
        on_tick: Optional[Callable[[RawTick], Any]] = None,
        db: Optional[Database] = None,
    ) -> None:
        """Start async WebSocket streaming feed with automatic reconnect."""
        import websockets

        self._running = True
        logger.info(f"Starting Coinbase WebSocket stream for {pair} on {self.ws_url}...")

        while self._running:
            try:
                async with websockets.connect(
                    self.ws_url, ping_interval=20, ping_timeout=10
                ) as ws:
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": [pair],
                        "channels": ["ticker", "heartbeat"],
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to ticker and heartbeat for {pair}.")

                    async for message in ws:
                        if not self._running:
                            break
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "ticker":
                            price = float(data.get("price", 0.0))
                            ts_str = data.get("time")
                            if ts_str:
                                ts = datetime.fromisoformat(
                                    ts_str.replace("Z", "+00:00")
                                ).timestamp()
                            else:
                                ts = datetime.now(timezone.utc).timestamp()

                            tick = RawTick(
                                pair=pair,
                                timestamp=ts,
                                price=price,
                                volume_24h=float(data.get("volume_24h", 0.0) or 0.0),
                                best_bid=float(data.get("best_bid", 0.0) or 0.0),
                                best_ask=float(data.get("best_ask", 0.0) or 0.0),
                            )

                            if db:
                                db.insert_tick(tick)
                            if on_tick:
                                res = on_tick(tick)
                                if asyncio.iscoroutine(res):
                                    await res

                            # Aggregate into 1m candle
                            completed_candle = self._aggregate_tick(tick)
                            if completed_candle:
                                if db:
                                    db.insert_candles([completed_candle])
                                if on_candle:
                                    res = on_candle(completed_candle)
                                    if asyncio.iscoroutine(res):
                                        await res

            except Exception as e:
                if self._running:
                    logger.warning(f"WebSocket disconnected ({e}); reconnecting in 3s...")
                    await asyncio.sleep(3.0)

    def stop_stream(self) -> None:
        """Stop WebSocket streaming."""
        self._running = False

    def _aggregate_tick(self, tick: RawTick) -> Optional[Candle]:
        """Aggregate tick into current 1-minute candle. Returns completed Candle when minute rolls."""
        candle_ts = (int(tick.timestamp) // 60) * 60

        if self._current_candle is None:
            self._current_candle = {
                "pair": tick.pair,
                "timestamp": candle_ts,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": 1.0,
            }
            return None

        if candle_ts == self._current_candle["timestamp"]:
            c = self._current_candle
            c["high"] = max(c["high"], tick.price)
            c["low"] = min(c["low"], tick.price)
            c["close"] = tick.price
            c["volume"] += 1.0  # approximate volume tick count if size unavailable
            return None
        else:
            # Candle completed!
            finished = Candle(
                pair=self._current_candle["pair"],
                timestamp=self._current_candle["timestamp"],
                open=self._current_candle["open"],
                high=self._current_candle["high"],
                low=self._current_candle["low"],
                close=self._current_candle["close"],
                volume=max(self._current_candle["volume"], 1.0),
                source="websocket",
            )
            # Start new candle
            self._current_candle = {
                "pair": tick.pair,
                "timestamp": candle_ts,
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": 1.0,
            }
            return finished
