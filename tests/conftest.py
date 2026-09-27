"""
Pytest fixtures and synthetic market data generators.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.database import Database
from src.models import Candle


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    """Fixture providing a clean temporary SQLite database."""
    db_file = tmp_path / "test_trading.db"
    return Database(str(db_file))


def generate_synthetic_candles(
    pair: str = "BTC-USD",
    start_dt: datetime = datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc),
    num_candles: int = 2880,  # 2 days of 1-minute candles
    base_price: float = 65000.0,
) -> List[Candle]:
    """
    Generate realistic synthetic crypto OHLCV 1-minute candles
    with cyclical swings, trend periods, and volume spikes.
    """
    candles: List[Candle] = []
    price = base_price
    t = start_dt.timestamp()

    np.random.seed(42)  # Deterministic for tests

    for i in range(num_candles):
        # Sine wave regime + random walk
        cycle = math.sin(i / 120.0) * 150.0
        shock = np.random.normal(0, 25.0)
        drift = 0.5 if i < 1440 else -0.4  # Uptrend day 1, downtrend day 2
        
        open_p = price
        close_p = max(open_p + shock + drift + (cycle * 0.05), 100.0)
        high_p = max(open_p, close_p) + abs(np.random.normal(10, 5))
        low_p = min(open_p, close_p) - abs(np.random.normal(10, 5))
        volume = float(np.random.uniform(5.0, 50.0))
        if i % 60 == 0:  # Occasional volume surge
            volume *= 3.0

        candles.append(
            Candle(
                pair=pair,
                timestamp=t,
                open=round(open_p, 2),
                high=round(high_p, 2),
                low=round(low_p, 2),
                close=round(close_p, 2),
                volume=round(volume, 2),
                source="test_synthetic",
            )
        )
        price = close_p
        t += 60  # Step 1 minute

    return candles
