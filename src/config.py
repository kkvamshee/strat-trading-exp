"""
Configuration loader and environment manager.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

from src.models import AppConfig

# Load environment variables from .env
load_dotenv()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file and merge with environment overrides."""
    if config_path is None:
        # Default locations to search
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break

    config_dict: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_dict = loaded

    # Allow environment variable overrides for paths & pairs
    if os.getenv("TRADING_PAIR"):
        config_dict.setdefault("system", {})["default_pair"] = os.getenv("TRADING_PAIR")
    if os.getenv("TRADING_DB_PATH"):
        config_dict.setdefault("system", {})["db_path"] = os.getenv("TRADING_DB_PATH")

    return AppConfig.model_validate(config_dict)


def get_api_key(service: str) -> Optional[str]:
    """Retrieve service API key from environment."""
    mapping = {
        "typesafe": "TYPESAFE_API_KEY",
        "coinbase_key": "COINBASE_API_KEY",
        "coinbase_secret": "COINBASE_API_SECRET",
    }
    env_var = mapping.get(service.lower(), service.upper())
    return os.getenv(env_var)
