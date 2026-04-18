"""8-component token scoring engine (0-100 scale)."""
from __future__ import annotations

import math
from typing import Optional

from typing import Optional

from .config import SCORING_WEIGHTS
from .models import HotTokenCandidate, PairSnapshot


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Sigmoid squash to 0-1."""
    return 1.0 / (1.0 + math.exp(-k * x))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Individual component scorers (return 0.0–1.0) ────────────────────────────

def score_volume_velocity(snap: PairSnapshot, chain_avg_h24: float = 50_000) -> float:
    """How fast volume is growing relative to chain average."""
    if snap.volume_h24 <= 0:
        return 0.0
    # Acceleration: is h1 pace above h24 average pace?
    h24_hourly_avg = snap.volume_h24 / 24
    accel = (snap.volume_h1 / h24_hourly_avg - 1.0) if h24_hourly_avg > 0 else 0.0
    # Absolute size score (log scale)
    size_score = _clamp(math.log10(max(snap.volume_h24, 1)) / 7.0)
    # Acceleration score
    accel_score = _clamp(_sigmoid(accel, k=1.5))
    return (size_score * 0.4 + accel_score * 0.6)


def score_txn_velocity(snap: PairSnapshot) -> float:
    """Rate of transaction count over h1."""
    txns_h1 = snap.txns_h1_total
    txns_h24_hourly = snap.txns_h24_total / 24 if snap.txns_h24_total > 0 else 1
    ratio = txns_h1 / txns_h24_hourly
    # Log scale for absolute txn count
    abs_score = _clamp(math.log10(max(txns_h1, 1)) / 3.0)
    velocity_score = _clamp(_sigmoid(ratio - 1.0, k=1.0))
    return abs_score * 0.4 + velocity_score * 0.6


def score_relative_strength(snap: PairSnapshot) -> float:
    """Price performance vs baseline (using price_change signals)."""
    h1 = snap.price_change_h1
    h6 = snap.price_change_h6
    h24 = snap.price_change_h24

    # Weight recent more heavily
    combined = h1 * 0.5 + h6 * 0.3 + h24 * 0.2
    # Sigmoid around 0 with positive bias
    return _clamp(_sigmoid(combined / 20.0))


def score_breakout_readiness(snap: PairSnapshot) -> float:
    """Price compression / coiling before a breakout.
    High h24 volatility but LOW h1 volatility = compression.
    """
    h1 = abs(snap.price_change_h1)
    h24 = abs(snap.price_change_h24)
    if h24 < 5:
        return 0.3  # Not enough movement to signal anything
    # Compression ratio: if h1 is small relative to h24
    compression_ratio = 1.0 - _clamp(h1 / max(h24, 1))
    # Recent upward bias
    direction_bonus = 0.2 if snap.price_change_h1 > 0 else 0.0
    return _clamp(compression_ratio * 0.8 + direction_bonus)


def score_boost_velocity(snap: PairSnapshot) -> float:
    """Boost activity score."""
    boost = snap.boost_count
    if boost == 0:
        return 0.0
    return _clamp(math.log10(boost + 1) / 2.5)


def score_momentum_decay(snap: PairSnapshot) -> float:
    """Whether momentum is sustaining (high) or fading (low)."""
    if snap.volume_h24 <= 0:
        return 0.0
    # h1 vs h6: h1 high relative to h6/6 means sustaining
    h6_hourly = snap.volume_h6 / 6 if snap.volume_h6 > 0 else 0
    h24_hourly = snap.volume_h24 / 24 if snap.volume_h24 > 0 else 0
    if h24_hourly == 0:
        return 0.0
    recent = snap.volume_h1
    prior = h6_hourly
    if prior == 0:
        return 0.5
    decay_ratio = recent / prior
    return _clamp(_sigmoid(math.log(max(decay_ratio, 0.01)), k=2.0))


def score_liquidity_depth(snap: PairSnapshot) -> float:
    """Pool liquidity health."""
    liq = snap.liquidity_usd
    if liq <= 0:
        return 0.0
    # Log scale 3 = $1k, 6 = $1M
    raw = (math.log10(liq) - 3) / 4.0
    return _clamp(raw)


def score_flow_pressure(snap: PairSnapshot) -> float:
    """Buy vs sell transaction imbalance."""
    total = snap.txns_h1_total
    if total == 0:
        return 0.5
    buy_ratio = snap.txns_h1_buys / total
    # Score: 0.5 = neutral, 1.0 = all buys
    return _clamp(buy_ratio)


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_token(
    snap: PairSnapshot,
    chain_avg_h24: float = 50_000,
    weights: Optional[dict[str, float]] = None,
) -> tuple[float, dict[str, float]]:
    """Score a token 0-100. Returns (score, components)."""
    w = weights or SCORING_WEIGHTS
    components = {
        "volume_velocity":    score_volume_velocity(snap, chain_avg_h24),
        "txn_velocity":       score_txn_velocity(snap),
        "relative_strength":  score_relative_strength(snap),
        "breakout_readiness": score_breakout_readiness(snap),
        "boost_velocity":     score_boost_velocity(snap),
        "momentum_decay":     score_momentum_decay(snap),
        "liquidity_depth":    score_liquidity_depth(snap),
        "flow_pressure":      score_flow_pressure(snap),
    }
    weighted = sum(components[k] * w.get(k, 0) for k in components)
    total = round(weighted * 100, 1)
    return total, components


def rank_candidates(candidates: list[HotTokenCandidate]) -> list[HotTokenCandidate]:
    """Sort candidates by score descending and assign ranks."""
    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    for i, c in enumerate(sorted_c):
        c.rank = i + 1
    return sorted_c


def score_label(score: float) -> str:
    if score >= 80:
        return "🔥 HOT"
    elif score >= 60:
        return "⚡ WARM"
    elif score >= 40:
        return "📈 MILD"
    else:
        return "💤 COLD"
