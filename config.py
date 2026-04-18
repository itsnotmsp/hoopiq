"""Configuration constants and filter profiles."""
from __future__ import annotations
from typing import Any

CHAINS: dict[str, dict[str, str]] = {
    "solana":    {"id": "solana",    "name": "Solana",    "symbol": "SOL",  "gecko_id": "solana"},
    "base":      {"id": "base",      "name": "Base",      "symbol": "ETH",  "gecko_id": "base"},
    "ethereum":  {"id": "ethereum",  "name": "Ethereum",  "symbol": "ETH",  "gecko_id": "eth"},
    "bsc":       {"id": "bsc",       "name": "BSC",       "symbol": "BNB",  "gecko_id": "bsc"},
    "arbitrum":  {"id": "arbitrum",  "name": "Arbitrum",  "symbol": "ETH",  "gecko_id": "arbitrum"},
    "polygon":   {"id": "polygon",   "name": "Polygon",   "symbol": "MATIC","gecko_id": "polygon_pos"},
    "optimism":  {"id": "optimism",  "name": "Optimism",  "symbol": "ETH",  "gecko_id": "optimism"},
    "avalanche": {"id": "avalanche", "name": "Avalanche", "symbol": "AVAX", "gecko_id": "avax"},
}

DEFAULT_CHAINS = ["solana", "base", "ethereum", "bsc", "arbitrum"]

SCAN_PROFILES: dict[str, dict[str, Any]] = {
    "discovery": {
        "min_liquidity_usd": 8_000,
        "min_volume_h24": 10_000,
        "min_txns_h1": 5,
        "description": "Wide net - early gems, degen plays, micro-caps",
    },
    "balanced": {
        "min_liquidity_usd": 20_000,
        "min_volume_h24": 40_000,
        "min_txns_h1": 25,
        "description": "General scanning - mix of safety and opportunity",
    },
    "strict": {
        "min_liquidity_usd": 35_000,
        "min_volume_h24": 90_000,
        "min_txns_h1": 50,
        "description": "Established tokens only - blue-chip filtering",
    },
    "pump": {
        "min_liquidity_usd": 5_000,
        "min_volume_h24": 8_000,
        "min_txns_h1": 10,
        "description": "Pump runners - catches tokens pumping hard in short windows",
    },
}

# Chain-specific liquidity multipliers
CHAIN_MULTIPLIERS: dict[str, float] = {
    "solana":    1.0,
    "base":      0.8,
    "ethereum":  1.5,
    "bsc":       0.7,
    "arbitrum":  0.9,
    "polygon":   0.6,
    "optimism":  0.7,
    "avalanche": 0.7,
}

DEFAULT_LIMIT = 20
DEFAULT_INTERVAL = 7
DEFAULT_PROFILE = "balanced"

# API base URLs
DEXSCREENER_BASE = "https://api.dexscreener.com"
GECKO_TERMINAL_BASE = "https://api.geckoterminal.com/api/v2"
BLOCKSCOUT_ETH_BASE = "https://eth.blockscout.com/api/v2"
BLOCKSCOUT_BASE_BASE = "https://base.blockscout.com/api/v2"
HONEYPOT_BASE = "https://api.honeypot.is/v1"
MORALIS_BASE = "https://deep-index.moralis.io/api/v2.2"

# Cache TTL in seconds
CACHE_TTL_PAIRS = 20
CACHE_TTL_HOLDERS = 900   # 15 minutes
CACHE_TTL_BOOSTS = 30

# Rate limits (requests per minute)
DEXSCREENER_RPM_SLOW = 60
DEXSCREENER_RPM_FAST = 300

# Scoring component weights (must sum to 1.0)
SCORING_WEIGHTS: dict[str, float] = {
    "volume_velocity":    0.20,
    "txn_velocity":       0.15,
    "relative_strength":  0.15,
    "breakout_readiness": 0.10,
    "boost_velocity":     0.10,
    "momentum_decay":     0.10,
    "liquidity_depth":    0.10,
    "flow_pressure":      0.10,
}

# Pump-mode weights — heavily favour velocity, buy pressure, momentum
# Ignores liquidity depth and boost (irrelevant for fast pumps)
PUMP_SCORING_WEIGHTS: dict[str, float] = {
    "volume_velocity":    0.30,   # is volume exploding right now?
    "txn_velocity":       0.25,   # are txns spiking?
    "flow_pressure":      0.20,   # one-sided buying?
    "momentum_decay":     0.15,   # is the pump still going?
    "relative_strength":  0.10,   # price moving up?
    "breakout_readiness": 0.00,
    "boost_velocity":     0.00,
    "liquidity_depth":    0.00,
}

# Score thresholds
SCORE_HOT = 80
SCORE_INTERESTING = 60
SCORE_MODERATE = 40

# AI token keywords for ai-top command
AI_KEYWORDS = [
    "ai", "gpt", "llm", "neural", "deep", "brain", "cogni", "intel",
    "smart", "learn", "predict", "agent", "agi", "sentient", "mind",
    "algo", "bard", "claude", "degen", "robo",
]

# State directory
import os
STATE_DIR = os.path.expanduser("~/.dexscreener-cli")
PRESETS_FILE = os.path.join(STATE_DIR, "presets.json")
TASKS_FILE = os.path.join(STATE_DIR, "tasks.json")
TASK_RUNS_FILE = os.path.join(STATE_DIR, "task_runs.json")
CONFIG_FILE = os.path.join(STATE_DIR, "config.json")
