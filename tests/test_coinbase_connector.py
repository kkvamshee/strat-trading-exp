"""
Unit tests for the Coinbase Connector and Tick Aggregator.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.coinbase_connector import CoinbaseConnector
from src.models import RawTick
from src.replay_engine import parse_time_argument


def test_time_argument_parser() -> None:
    """Test parsing relative durations and ISO strings."""
    now_ts = datetime.now(timezone.utc).timestamp()

    # Relative 2 days
    ts_2d = parse_time_argument("2d")
    assert abs((now_ts - 172800) - ts_2d) < 5.0

    # Relative 12 hours
    ts_12h = parse_time_argument("12h")
    assert abs((now_ts - 43200) - ts_12h) < 5.0

    # ISO string
    iso_ts = parse_time_argument("2026-09-25T00:00:00Z")
    assert iso_ts == datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc).timestamp()

    # Invalid string raises ValueError
    with pytest.raises(ValueError):
        parse_time_argument("invalid_format_xyz")


def test_tick_aggregator() -> None:
    """Test aggregating high-frequency ticks into 1-minute OHLCV candles."""
    connector = CoinbaseConnector()

    # 3 ticks in same minute (1020..1079)
    tick1 = RawTick(pair="BTC-USD", timestamp=1025.0, price=65000.0)
    tick2 = RawTick(pair="BTC-USD", timestamp=1035.0, price=65100.0)
    tick3 = RawTick(pair="BTC-USD", timestamp=1055.0, price=64950.0)

    # First tick starts candle
    assert connector._aggregate_tick(tick1) is None
    # Subsequent ticks in same minute update candle
    assert connector._aggregate_tick(tick2) is None
    assert connector._aggregate_tick(tick3) is None

    # Next tick in subsequent minute (1080..1139) completes previous candle and returns it
    tick4 = RawTick(pair="BTC-USD", timestamp=1085.0, price=65200.0)
    completed = connector._aggregate_tick(tick4)

    assert completed is not None
    assert completed.pair == "BTC-USD"
    assert completed.timestamp == 1020.0
    assert completed.open == 65000.0
    assert completed.high == 65100.0
    assert completed.low == 64950.0
    assert completed.close == 64950.0
    assert completed.volume == 3.0
