"""Token discovery, scanning, and filtering."""
from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from .client import DexscreenerClient
from .config import (
    AI_KEYWORDS, CHAINS, CHAIN_MULTIPLIERS, DEFAULT_CHAINS,
    GECKO_NETWORK_MAP, SCAN_PROFILES,
)
from .holders import enrich_with_holders
from .models import HotTokenCandidate, PairSnapshot
from .scoring import rank_candidates, score_token

# Chains that GeckoTerminal supports for trending/new pools
GECKO_SUPPORTED = {
    "solana": "solana",
    "ethereum": "eth",
    "base": "base",
    "bsc": "bsc",
    "arbitrum": "arbitrum",
    "polygon": "polygon_pos",
    "optimism": "optimism",
    "avalanche": "avax",
}

# Symbols that should never appear as results —
# wrapped tokens, stablecoins, native gas tokens, LP tokens, etc.
BLOCKLIST_SYMBOLS: set[str] = {
    # Wrapped natives
    "WETH", "WBTC", "WBNB", "WMATIC", "WAVAX", "WSOL", "WFTM", "WONE",
    "WGLMR", "WCRO", "WKAVA", "WROSE", "WMOVR",
    # Stablecoins
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "FRAX",
    "LUSD", "SUSD", "USDD", "USDJ", "USDN", "CUSD", "XUSD", "USDX",
    "HUSD", "OUSD", "USDK", "USDQ", "MIM", "DOLA", "ALUSD", "MUSD",
    "CRVUSD", "PYUSD", "FDUSD", "USDE", "EURC", "EURS", "EURT",
    # Native gas tokens
    "ETH", "BNB", "MATIC", "AVAX", "FTM", "ONE", "CRO", "SOL",
    "ARB", "OP", "GLMR", "MOVR",
    # Common LP / receipt tokens
    "CAKE-LP", "UNI-V2", "SLP", "VELO",
    # Well-known DeFi bluechips
    "WSTETH", "STETH", "RETH", "CBETH", "SFRXETH",
    "LINK", "AAVE", "UNI", "COMP", "MKR", "SNX", "CRV", "CVX",
    "LDO", "RPL", "FXS", "BAL", "SUSHI", "1INCH", "YFI",
}

# Token NAMES that scammers use to impersonate real projects.
# Match is done on exact name (case-insensitive) or starts-with.
BLOCKLIST_NAMES: set[str] = {
    # Layer 1s
    "solana", "ethereum", "bitcoin", "binance coin", "avalanche",
    "polygon", "cardano", "polkadot", "tron", "near protocol",
    "near", "algorand", "cosmos", "aptos", "sui", "arbitrum",
    "optimism", "fantom",
    # Stablecoin full names
    "tether", "usd coin", "binance usd", "dai stablecoin",
    "true usd", "frax", "magic internet money",
    # Major DeFi
    "wrapped bitcoin", "wrapped ether", "wrapped ethereum",
    "wrapped solana", "wrapped bnb", "lido staked ether",
    "chainlink", "uniswap", "aave", "maker",
}

# If a token name CONTAINS any of these substrings it is blocked.
BLOCKLIST_NAME_FRAGMENTS: set[str] = {
    "wrapped", "bridged", "usd coin", "tether", "staked eth",
    "staked ether", "(wormhole)", "(bridged)", "(base)",
    "cetoken", ".e)", " lp", "lp token",
}


# ── Momentum Tracker ──────────────────────────────────────────────────────────

@dataclass
class TokenSnapshot:
    """A single point-in-time record of a token's metrics."""
    ts: float
    txns_h1: int
    buys_h1: int
    sells_h1: int
    volume_h1: float
    price_usd: float
    buy_pressure: float
    liquidity_usd: float
    holder_count: Optional[int] = None


@dataclass
class MomentumEntry:
    """Tracks a token across multiple scans to detect acceleration."""
    ca: str
    chain: str
    symbol: str
    history: list[TokenSnapshot] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    alerted: bool = False
    exit_alerted: bool = False
    liq_drop_alerted: bool = False
    buy_burst_alerted: bool = False
    # Peak values for exit detection
    peak_buy_pressure: float = 0.0
    peak_liquidity: float = 0.0
    # Baseline buy txns (average of first 3 observations) for burst comparison
    baseline_buys: float = 0.0
    _baseline_samples: int = 0

    def add(self, snap: PairSnapshot) -> None:
        s = TokenSnapshot(
            ts=time.time(),
            txns_h1=snap.txns_h1_total,
            buys_h1=snap.txns_h1_buys,
            sells_h1=snap.txns_h1_sells,
            volume_h1=snap.volume_h1,
            price_usd=snap.price_usd,
            buy_pressure=snap.buy_pressure,
            liquidity_usd=snap.liquidity_usd,
            holder_count=snap.holder_count,
        )
        self.history.append(s)
        if len(self.history) > 20:
            self.history = self.history[-20:]
        # Track peaks
        if s.buy_pressure > self.peak_buy_pressure:
            self.peak_buy_pressure = s.buy_pressure
        if s.liquidity_usd > self.peak_liquidity:
            self.peak_liquidity = s.liquidity_usd
        # Build baseline from first 3 observations (quiet state)
        if self._baseline_samples < 3:
            total = self.baseline_buys * self._baseline_samples + s.buys_h1
            self._baseline_samples += 1
            self.baseline_buys = total / self._baseline_samples

    @property
    def obs_count(self) -> int:
        return len(self.history)

    @property
    def latest(self) -> Optional[TokenSnapshot]:
        return self.history[-1] if self.history else None

    def delta(self) -> Optional[dict]:
        if len(self.history) < 2:
            return None
        prev = self.history[-2]
        curr = self.history[-1]
        dt = max(curr.ts - prev.ts, 1.0)

        txn_pct   = ((curr.txns_h1 - prev.txns_h1) / max(prev.txns_h1, 1)) * 100
        vol_pct   = ((curr.volume_h1 - prev.volume_h1) / max(prev.volume_h1, 0.01)) * 100
        price_pct = ((curr.price_usd - prev.price_usd) / max(prev.price_usd, 1e-12)) * 100
        liq_pct   = ((curr.liquidity_usd - prev.liquidity_usd) / max(prev.liquidity_usd, 1.0)) * 100
        bp_delta  = curr.buy_pressure - prev.buy_pressure

        return {
            "txn_pct":      txn_pct,
            "vol_pct":      vol_pct,
            "price_pct":    price_pct,
            "liq_pct":      liq_pct,
            "bp_delta":     bp_delta,
            "buy_pressure": curr.buy_pressure,
            "liquidity_usd": curr.liquidity_usd,
            "dt_seconds":   dt,
        }

    # ── 1. Ignition score ────────────────────────────────────────────────────

    def ignition_score(self) -> float:
        d = self.delta()
        if d is None:
            return 0.0
        score = 0.0
        score += min(35.0, max(0.0, d["txn_pct"] / 2.0))    # txn spike
        score += min(30.0, max(0.0, d["vol_pct"] / 3.0))    # vol spike
        score += min(20.0, max(0.0, d["price_pct"] * 2.0))  # price up
        bp = d["buy_pressure"]
        bp_pts = 15.0 if bp >= 0.70 else 10.0 if bp >= 0.60 else 5.0 if bp >= 0.55 else 0.0
        if d["bp_delta"] > 0.05:
            bp_pts = min(15.0, bp_pts + 5.0)
        score += bp_pts
        return min(100.0, score)

    # ── 2. Liquidity drop detection ──────────────────────────────────────────

    def liquidity_drop(self) -> Optional[dict]:
        """
        Returns alert dict if liquidity dropped significantly.
        Compares current liq to peak liq seen, and to previous scan.
        """
        if not self.history or self.peak_liquidity <= 0:
            return None
        curr_liq = self.history[-1].liquidity_usd
        # Drop from peak (catches slow rug over multiple scans)
        peak_drop_pct = ((self.peak_liquidity - curr_liq) / self.peak_liquidity) * 100
        # Drop from last scan (catches sudden rug pull)
        scan_drop_pct = 0.0
        if len(self.history) >= 2:
            prev_liq = self.history[-2].liquidity_usd
            if prev_liq > 0:
                scan_drop_pct = ((prev_liq - curr_liq) / prev_liq) * 100

        # Alert thresholds
        if scan_drop_pct >= 20:   # sudden 20%+ drop in one scan = rug warning
            severity = "🚨 RUG ALERT"
        elif scan_drop_pct >= 10 or peak_drop_pct >= 30:
            severity = "⚠️ LIQ DROP"
        else:
            return None

        return {
            "severity":      severity,
            "scan_drop_pct": scan_drop_pct,
            "peak_drop_pct": peak_drop_pct,
            "curr_liq":      curr_liq,
            "peak_liq":      self.peak_liquidity,
        }

    # ── 3. Exit signal detection ─────────────────────────────────────────────

    def exit_signal(self) -> Optional[dict]:
        """
        Returns alert dict if momentum is collapsing after a pump.
        Only fires if we've seen the token building up (peak buy pressure was good).
        """
        if self.obs_count < 3:
            return None
        if self.peak_buy_pressure < 0.60:
            return None   # never pumped properly, don't generate exit noise

        d = self.delta()
        if not d:
            return None

        curr_bp = d["buy_pressure"]
        reasons = []

        # Buy pressure collapsed from peak
        bp_drop = self.peak_buy_pressure - curr_bp
        if bp_drop >= 0.20:
            reasons.append(f"buys dropped {int(bp_drop*100)}% from peak")

        # Current buy pressure below 50% (sellers dominating)
        if curr_bp < 0.50:
            reasons.append(f"buys now {int(curr_bp*100)}% (sellers winning)")

        # Txns falling sharply
        if d["txn_pct"] < -25:
            reasons.append(f"txns down {abs(d['txn_pct']):.0f}%")

        # Volume collapsing
        if d["vol_pct"] < -30:
            reasons.append(f"vol down {abs(d['vol_pct']):.0f}%")

        if len(reasons) >= 2:   # need 2+ signals to avoid false exits
            return {
                "reasons":      reasons,
                "buy_pressure": curr_bp,
                "peak_bp":      self.peak_buy_pressure,
                "txn_pct":      d["txn_pct"],
            }
        return None

    # ── 4. Wallet concentration (uses holder_count from snapshot) ────────────

    def concentration_warning(self, snap: PairSnapshot) -> Optional[str]:
        """
        Warns if holder count is very low (concentrated ownership = easy dump).
        Returns warning string or None.
        """
        holders = snap.holder_count
        if not holders:
            return None
        if holders < 20:
            return f"⚠️ Only {holders} holders — very concentrated"
        if holders < 50:
            return f"⚠️ {holders} holders — concentrated"
        return None

    # ── 5. Buy burst detection ────────────────────────────────────────────────

    def buy_burst(self) -> Optional[dict]:
        """
        Detects a sudden abnormal spike in BUY transactions.
        Compares current buys_h1 to:
          - previous scan (instant spike)
          - established baseline (sustained abnormal buying)

        Triggers when:
          - Buys jump 3x+ from previous scan  (sudden burst)
          - Buys jump 5x+ from baseline       (massive vs normal state)
          - AND buy pressure is >= 60%         (not just noise)
        """
        if len(self.history) < 2:
            return None

        curr = self.history[-1]
        prev = self.history[-2]

        curr_buys = curr.buys_h1
        prev_buys = max(prev.buys_h1, 1)
        baseline  = max(self.baseline_buys, 1)

        # Multipliers
        vs_prev     = curr_buys / prev_buys
        vs_baseline = curr_buys / baseline

        # Must be real buying (not just noise)
        if curr.buy_pressure < 0.60:
            return None

        # Must have minimum activity
        if curr_buys < 10:
            return None

        # Determine severity
        if vs_prev >= 5.0 or vs_baseline >= 8.0:
            severity = "🚀 MASSIVE BUY BURST"
        elif vs_prev >= 3.0 or vs_baseline >= 5.0:
            severity = "⚡ BUY BURST"
        else:
            return None

        return {
            "severity":     severity,
            "curr_buys":    curr_buys,
            "prev_buys":    prev.buys_h1,
            "baseline_buys": self.baseline_buys,
            "vs_prev":      vs_prev,
            "vs_baseline":  vs_baseline,
            "buy_pressure": curr.buy_pressure,
            "volume_h1":    curr.volume_h1,
            "price_usd":    curr.price_usd,
        }

    def trend(self) -> str:
        if len(self.history) < 2:
            return "watching"
        d = self.delta()
        if not d:
            return "watching"
        # Check exit first
        if self.exit_signal():
            return "📉 EXIT"
        liq = self.liquidity_drop()
        if liq:
            return liq["severity"]
        ig = self.ignition_score()
        if ig >= 70:
            return "🔥 IGNITING"
        if ig >= 45:
            return "⚡ HEATING"
        if ig >= 20:
            return "📈 BUILDING"
        if d["txn_pct"] < -20:
            return "〰 FADING"
        return "〰 FLAT"


class MomentumTracker:
    """
    Tracks tokens across multiple scans.
    Detects ignition, liquidity drops, exit signals, and concentration warnings.
    """

    def __init__(self, max_age_minutes: int = 120) -> None:
        self._entries: dict[str, MomentumEntry] = {}
        self.max_age_minutes = max_age_minutes

    def _key(self, chain: str, ca: str) -> str:
        return f"{chain}:{ca.lower()}"

    def update(self, snap: PairSnapshot) -> MomentumEntry:
        key = self._key(snap.chain_id, snap.base_token_address)
        if key not in self._entries:
            self._entries[key] = MomentumEntry(
                ca=snap.base_token_address,
                chain=snap.chain_id,
                symbol=snap.base_token_symbol,
            )
        entry = self._entries[key]
        entry.add(snap)
        return entry

    def get(self, chain: str, ca: str) -> Optional[MomentumEntry]:
        return self._entries.get(self._key(chain, ca))

    def purge_old(self) -> None:
        cutoff = time.time() - self.max_age_minutes * 60
        stale = [k for k, e in self._entries.items()
                 if e.history and e.history[-1].ts < cutoff]
        for k in stale:
            del self._entries[k]
        # Reset buy burst flag every 5 minutes so repeat bursts can re-alert
        burst_cooldown = time.time() - 300
        for entry in self._entries.values():
            if entry.buy_burst_alerted:
                last_ts = entry.history[-1].ts if entry.history else 0
                if last_ts < burst_cooldown:
                    entry.buy_burst_alerted = False

    def igniting(self, min_score: float = 50.0) -> list[tuple[MomentumEntry, float]]:
        result = []
        for entry in self._entries.values():
            ig = entry.ignition_score()
            if ig >= min_score:
                result.append((entry, ig))
        return sorted(result, key=lambda x: x[1], reverse=True)

    def check_warnings(self, candidates: list[HotTokenCandidate]) -> list[dict]:
        """
        Check all tracked tokens for danger signals.
        Returns list of warning dicts ready to be sent as alerts.
        """
        warnings = []
        for c in candidates:
            entry = self.get(c.chain_id, c.address)
            if not entry or entry.obs_count < 2:
                continue

            # ── Liquidity drop ────────────────────────────────────────────
            liq = entry.liquidity_drop()
            if liq and not entry.liq_drop_alerted:
                warnings.append({
                    "type":    "liq_drop",
                    "token":   c,
                    "entry":   entry,
                    "data":    liq,
                    "message": (
                        f"{liq['severity']}: {c.symbol} ({c.chain_id.upper()})\n"
                        f"Liq dropped {liq['scan_drop_pct']:.0f}% this scan "
                        f"/ {liq['peak_drop_pct']:.0f}% from peak\n"
                        f"Now: ${liq['curr_liq']:,.0f} | Peak: ${liq['peak_liq']:,.0f}"
                    ),
                })
                entry.liq_drop_alerted = True

            # ── Exit signal ───────────────────────────────────────────────
            ext = entry.exit_signal()
            if ext and not entry.exit_alerted:
                reasons_str = " · ".join(ext["reasons"])
                warnings.append({
                    "type":    "exit",
                    "token":   c,
                    "entry":   entry,
                    "data":    ext,
                    "message": (
                        f"📉 EXIT SIGNAL: {c.symbol} ({c.chain_id.upper()})\n"
                        f"{reasons_str}\n"
                        f"Buy pressure: {int(ext['buy_pressure']*100)}% "
                        f"(was {int(ext['peak_bp']*100)}% at peak)"
                    ),
                })
                entry.exit_alerted = True

            # ── Concentration warning ─────────────────────────────────────
            conc = entry.concentration_warning(c.snapshot)
            if conc:
                warnings.append({
                    "type":    "concentration",
                    "token":   c,
                    "entry":   entry,
                    "data":    {"warning": conc},
                    "message": f"{conc}: {c.symbol} ({c.chain_id.upper()})",
                })

            # ── Buy burst ─────────────────────────────────────────────────
            burst = entry.buy_burst()
            if burst and not entry.buy_burst_alerted:
                warnings.append({
                    "type":  "buy_burst",
                    "token": c,
                    "entry": entry,
                    "data":  burst,
                })
                entry.buy_burst_alerted = True

        return warnings

    @property
    def tracked_count(self) -> int:
        return len(self._entries)


class ScanFilter:
    """Token filter parameters."""

    def __init__(
        self,
        profile: str = "balanced",
        min_liquidity_usd: Optional[float] = None,
        min_volume_h24: Optional[float] = None,
        min_txns_h1: Optional[int] = None,
        max_age_hours: Optional[float] = None,
        require_boost: bool = False,
        min_price_change_h1: Optional[float] = None,
        min_volume_acceleration: Optional[float] = None,
        min_buy_pressure: Optional[float] = None,
        pump_mode: bool = False,
    ) -> None:
        defaults = SCAN_PROFILES.get(profile, SCAN_PROFILES["balanced"])
        self.min_liquidity_usd = min_liquidity_usd or defaults["min_liquidity_usd"]
        self.min_volume_h24 = min_volume_h24 or defaults["min_volume_h24"]
        self.min_txns_h1 = min_txns_h1 or defaults["min_txns_h1"]
        self.max_age_hours = max_age_hours
        self.require_boost = require_boost
        self.pump_mode = pump_mode or (profile == "pump")

        # Pump-specific filters with sensible defaults in pump mode
        if self.pump_mode:
            self.min_price_change_h1 = min_price_change_h1 if min_price_change_h1 is not None else 10.0
            self.min_volume_acceleration = min_volume_acceleration if min_volume_acceleration is not None else 2.0
            self.min_buy_pressure = min_buy_pressure if min_buy_pressure is not None else 0.55
            self.max_age_hours = max_age_hours or 6.0  # only fresh tokens
        else:
            self.min_price_change_h1 = min_price_change_h1
            self.min_volume_acceleration = min_volume_acceleration
            self.min_buy_pressure = min_buy_pressure

    def apply_chain_multiplier(self, chain: str) -> "ScanFilter":
        m = CHAIN_MULTIPLIERS.get(chain, 1.0)
        f = ScanFilter.__new__(ScanFilter)
        f.min_liquidity_usd = self.min_liquidity_usd * m
        f.min_volume_h24 = self.min_volume_h24 * m
        f.min_txns_h1 = max(1, int(self.min_txns_h1 * m))
        f.max_age_hours = self.max_age_hours
        f.require_boost = self.require_boost
        f.pump_mode = self.pump_mode
        f.min_price_change_h1 = self.min_price_change_h1
        f.min_volume_acceleration = self.min_volume_acceleration
        f.min_buy_pressure = self.min_buy_pressure
        return f

    def passes(self, snap: PairSnapshot) -> bool:
        sym = snap.base_token_symbol.upper()
        name_lower = snap.base_token_name.lower().strip()

        # ── Symbol blocklist ─────────────────────────────────────────────────
        if sym in BLOCKLIST_SYMBOLS:
            return False

        # ── Exact name blocklist (impersonators) ─────────────────────────────
        if name_lower in BLOCKLIST_NAMES:
            return False

        # ── Name fragment blocklist ──────────────────────────────────────────
        if any(frag in name_lower for frag in BLOCKLIST_NAME_FRAGMENTS):
            return False

        # ── Require age data — no created_at = old/unknown token ────────────
        # Tokens with no age info are almost always old tokens, not new runners
        if snap.created_at is None:
            return False

        # ── Standard filters ─────────────────────────────────────────────────
        if snap.liquidity_usd < self.min_liquidity_usd:
            return False
        if snap.volume_h24 < self.min_volume_h24:
            return False
        if snap.txns_h1_total < self.min_txns_h1:
            return False
        if self.require_boost and snap.boost_count == 0:
            return False

        # ── Age filter ───────────────────────────────────────────────────────
        if self.max_age_hours:
            import time
            age_hours = (time.time() * 1000 - snap.created_at) / 3_600_000
            if age_hours > self.max_age_hours:
                return False

        # ── Minimum buy pressure (default 50% — net buying) ─────────────────
        effective_min_bp = self.min_buy_pressure if self.min_buy_pressure is not None else 0.50
        if snap.buy_pressure < effective_min_bp:
            return False

        # ── Pump-specific filters ────────────────────────────────────────────
        if self.min_price_change_h1 is not None:
            if snap.price_change_h1 < self.min_price_change_h1:
                return False
        if self.min_volume_acceleration is not None:
            if snap.volume_acceleration < self.min_volume_acceleration:
                return False

        return True


class Scanner:
    """Orchestrates token discovery across chains."""

    def __init__(self, client: DexscreenerClient, holders: bool = False) -> None:
        self.client = client
        self.fetch_holders = holders

    async def _get_boost_addresses(self) -> dict[str, set[str]]:
        """Return boosted token addresses grouped by chain."""
        result: dict[str, set[str]] = {}
        try:
            boosts = await self.client.get_latest_boosts()
            for b in boosts:
                chain = b.get("chainId", "")
                addr = b.get("tokenAddress", "")
                if chain and addr:
                    result.setdefault(chain, set()).add(addr.lower())
        except Exception:
            pass
        return result

    async def _fetch_chain_pairs(
        self, chain: str, flt: ScanFilter, boost_addrs: set[str]
    ) -> list[PairSnapshot]:
        """Fetch candidate pairs for a single chain."""
        pairs: list[PairSnapshot] = []
        seen: set[str] = set()

        gecko_net = GECKO_SUPPORTED.get(chain)

        async def _add_raw(raw_list: list[dict]) -> None:
            for raw in raw_list:
                if raw.get("chainId") != chain:
                    continue
                snap = self.client.parse_pair(raw)
                if snap and snap.pair_address not in seen:
                    seen.add(snap.pair_address)
                    pairs.append(snap)

        # Strategy 1: boosted tokens for this chain
        chain_boost = boost_addrs  # already filtered per chain
        if chain_boost:
            tasks = [self.client.get_pairs_by_token(chain, addr) for addr in list(chain_boost)[:20]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    await _add_raw(res)

        # Strategy 2: GeckoTerminal trending pools
        if gecko_net:
            try:
                trending = await self.client.get_trending_pools(gecko_net)
                # Extract pair addresses and look up on dexscreener
                for pool in trending[:15]:
                    attrs = pool.get("attributes", {})
                    addr = attrs.get("address", "")
                    if addr and addr not in seen:
                        raw = await self.client.get_pair(chain, addr)
                        if raw:
                            await _add_raw([raw])
            except Exception:
                pass

        # Strategy 3: search by chain name for popular tokens
        try:
            search_results = await self.client.search_pairs(chain)
            await _add_raw(search_results[:30])
        except Exception:
            pass

        return pairs

    async def scan_hot(
        self,
        chains: Optional[list[str]] = None,
        flt: Optional[ScanFilter] = None,
        limit: int = 20,
        with_holders: bool = False,
    ) -> list[HotTokenCandidate]:
        chains = chains or DEFAULT_CHAINS
        flt = flt or ScanFilter()

        # Get boost data for all chains
        all_boosts = await self._get_boost_addresses()

        # Fetch pairs per chain concurrently
        chain_tasks = [
            self._fetch_chain_pairs(chain, flt.apply_chain_multiplier(chain), all_boosts.get(chain, set()))
            for chain in chains
        ]
        chain_results = await asyncio.gather(*chain_tasks, return_exceptions=True)

        # Mark boosted pairs
        all_pairs: list[PairSnapshot] = []
        for chain, result in zip(chains, chain_results):
            if isinstance(result, list):
                boost_set = all_boosts.get(chain, set())
                for snap in result:
                    if snap.base_token_address.lower() in boost_set:
                        snap.boost_count = max(snap.boost_count, 1)
                all_pairs.extend(result)

        # Apply filters
        chain_avg = self._compute_chain_avg(all_pairs)
        filtered = [p for p in all_pairs if flt.passes(p)]

        # Optionally fetch holders
        if with_holders or self.fetch_holders:
            by_chain: dict[str, list[PairSnapshot]] = {}
            for p in filtered:
                by_chain.setdefault(p.chain_id, []).append(p)
            holder_tasks = [enrich_with_holders(snaps, chain) for chain, snaps in by_chain.items()]
            await asyncio.gather(*holder_tasks, return_exceptions=True)

        # Score and rank
        candidates = []
        weights = None
        if flt.pump_mode:
            from .config import PUMP_SCORING_WEIGHTS
            weights = PUMP_SCORING_WEIGHTS
        for snap in filtered:
            score, components = score_token(snap, chain_avg.get(snap.chain_id, 50_000), weights=weights)
            candidates.append(HotTokenCandidate(snapshot=snap, score=score, score_components=components))

        ranked = rank_candidates(candidates)
        return ranked[:limit]

    def _compute_chain_avg(self, pairs: list[PairSnapshot]) -> dict[str, float]:
        by_chain: dict[str, list[float]] = {}
        for p in pairs:
            if p.volume_h24 > 0:
                by_chain.setdefault(p.chain_id, []).append(p.volume_h24)
        return {
            chain: statistics.median(vols)
            for chain, vols in by_chain.items()
            if vols
        }

    async def scan_new_runners(
        self,
        chain: str = "solana",
        flt: Optional[ScanFilter] = None,
        limit: int = 20,
    ) -> list[HotTokenCandidate]:
        """Scan newly launched tokens with momentum."""
        flt = flt or ScanFilter(profile="discovery")
        chain_flt = flt.apply_chain_multiplier(chain)

        gecko_net = GECKO_SUPPORTED.get(chain)
        pairs: list[PairSnapshot] = []

        if gecko_net:
            try:
                pools = await self.client.get_new_pools(gecko_net)
                for pool in pools[:30]:
                    attrs = pool.get("attributes", {})
                    addr = attrs.get("address", "")
                    if addr:
                        raw = await self.client.get_pair(chain, addr)
                        if raw:
                            snap = self.client.parse_pair(raw)
                            if snap:
                                pairs.append(snap)
            except Exception:
                pass

        filtered = [p for p in pairs if chain_flt.passes(p) and p.price_change_h1 > 0]

        candidates = []
        for snap in filtered:
            score, components = score_token(snap)
            candidates.append(HotTokenCandidate(snapshot=snap, score=score, score_components=components))

        return rank_candidates(candidates)[:limit]

    async def scan_new_launches(
        self,
        chains: Optional[list[str]] = None,
        max_age_minutes: int = 30,
        min_liquidity_usd: float = 2_000,
        min_txns: int = 3,
        limit: int = 30,
        tracker: Optional[MomentumTracker] = None,
    ) -> list[HotTokenCandidate]:
        """
        Ultra-fresh token launch scanner with momentum tracking.
        - Fetches tokens < max_age_minutes old
        - Feeds each snapshot into the MomentumTracker
        - Scores based on acceleration between scans, not just current state
        """
        chains = chains or ["solana", "base"]
        now_ms = time.time() * 1000
        max_age_ms = max_age_minutes * 60 * 1000

        seen: set[str] = set()
        all_snaps: list[PairSnapshot] = []

        async def _collect(chain: str) -> None:
            gecko_net = GECKO_SUPPORTED.get(chain)
            if not gecko_net:
                return
            try:
                # 3 pages for wider coverage
                pages = await asyncio.gather(*[
                    self.client.get_new_pools(gecko_net, page=p)
                    for p in range(1, 4)
                ], return_exceptions=True)

                for page_pools in pages:
                    if not isinstance(page_pools, list):
                        continue
                    for pool in page_pools:
                        attrs = pool.get("attributes", {})
                        addr = attrs.get("address", "")
                        if not addr or addr in seen:
                            continue
                        seen.add(addr)

                        raw = await self.client.get_pair(chain, addr)
                        if not raw:
                            continue
                        snap = self.client.parse_pair(raw)
                        if not snap or snap.created_at is None:
                            continue

                        # Age check
                        age_ms = now_ms - snap.created_at
                        if age_ms > max_age_ms or age_ms < 0:
                            continue

                        # Minimum activity
                        if snap.liquidity_usd < min_liquidity_usd:
                            continue
                        if snap.txns_h1_total < min_txns:
                            continue

                        # Blocklist checks
                        sym = snap.base_token_symbol.upper()
                        name_lower = snap.base_token_name.lower()
                        if sym in BLOCKLIST_SYMBOLS:
                            continue
                        if name_lower in BLOCKLIST_NAMES:
                            continue
                        if any(f in name_lower for f in BLOCKLIST_NAME_FRAGMENTS):
                            continue

                        all_snaps.append(snap)
            except Exception:
                pass

        await asyncio.gather(*[_collect(c) for c in chains])

        # Fetch holders for concentration check (best-effort)
        if all_snaps:
            by_chain: dict[str, list[PairSnapshot]] = {}
            for s in all_snaps:
                by_chain.setdefault(s.chain_id, []).append(s)
            await asyncio.gather(
                *[enrich_with_holders(snaps, ch) for ch, snaps in by_chain.items()],
                return_exceptions=True,
            )

        # Feed all snaps into the tracker and build candidates
        from .config import PUMP_SCORING_WEIGHTS
        candidates = []

        for snap in all_snaps:
            # Base pump score
            base_score, components = score_token(snap, weights=PUMP_SCORING_WEIGHTS)

            # Momentum / ignition bonus from tracker
            ignition_bonus = 0.0
            entry = None
            if tracker is not None:
                entry = tracker.update(snap)
                ignition_bonus = entry.ignition_score()

            # Final score: blend base score with ignition signal
            # First observation: use base score only
            # 2+ observations: ignition score dominates (it's more accurate)
            obs = entry.obs_count if entry else 1
            if obs >= 2:
                final_score = base_score * 0.3 + ignition_bonus * 0.7
            else:
                final_score = base_score

            final_score = min(100.0, final_score)

            c = HotTokenCandidate(
                snapshot=snap,
                score=round(final_score, 1),
                score_components={
                    **components,
                    "ignition": ignition_bonus / 100.0,
                    "observations": float(obs),
                },
            )
            candidates.append(c)

        # Sort: igniting tokens first, then by age (freshest first within same tier)
        def _sort_key(c: HotTokenCandidate) -> tuple:
            age_mins = (now_ms - (c.snapshot.created_at or 0)) / 60_000
            tier = 0 if c.score >= 65 else 1 if c.score >= 40 else 2
            return (tier, age_mins)

        candidates.sort(key=_sort_key)
        for i, c in enumerate(candidates):
            c.rank = i + 1

        # Clean up stale entries
        if tracker:
            tracker.purge_old()

        return candidates[:limit]

    async def scan_alpha_drops(
        self,
        chains: Optional[list[str]] = None,
        flt: Optional[ScanFilter] = None,
        limit: int = 20,
        min_score: float = 60.0,
    ) -> list[HotTokenCandidate]:
        """Alpha-grade drops with breakout potential."""
        chains = chains or ["solana", "base"]
        results = await self.scan_hot(chains=chains, flt=flt, limit=limit * 3)
        alpha = [c for c in results if c.score >= min_score and c.snapshot.price_change_h1 > 5]
        return alpha[:limit]

    async def scan_ai_tokens(
        self,
        chains: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[HotTokenCandidate]:
        """AI-themed token leaderboard."""
        chains = chains or DEFAULT_CHAINS
        tasks = [self.client.search_pairs(kw) for kw in ["ai", "gpt", "neural", "agent"]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen: set[str] = set()
        pairs: list[PairSnapshot] = []
        for res in results:
            if isinstance(res, list):
                for raw in res:
                    if raw.get("chainId") in chains:
                        snap = self.client.parse_pair(raw)
                        if snap and snap.pair_address not in seen:
                            # Check if name/symbol has AI keywords
                            text = (snap.base_token_name + snap.base_token_symbol).lower()
                            if any(kw in text for kw in AI_KEYWORDS):
                                seen.add(snap.pair_address)
                                pairs.append(snap)

        candidates = []
        for snap in pairs:
            score, components = score_token(snap)
            candidates.append(HotTokenCandidate(snapshot=snap, score=score, score_components=components))

        return rank_candidates(candidates)[:limit]

    async def search(self, query: str, chains: Optional[list[str]] = None) -> list[HotTokenCandidate]:
        """Search pairs by name/symbol/address."""
        raw_pairs = await self.client.search_pairs(query)
        if not raw_pairs:
            return []

        seen: set[str] = set()
        candidates = []
        for raw in raw_pairs:
            if chains and raw.get("chainId") not in chains:
                continue
            snap = self.client.parse_pair(raw)
            if snap and snap.pair_address not in seen:
                seen.add(snap.pair_address)
                score, components = score_token(snap)
                candidates.append(HotTokenCandidate(snapshot=snap, score=score, score_components=components))

        return rank_candidates(candidates)

    async def inspect_token(self, address: str, chain: Optional[str] = None) -> Optional[HotTokenCandidate]:
        """Deep-dive on a specific token."""
        # Try to find the best pair for this token
        search_results = await self.client.search_pairs(address)
        if not search_results:
            return None

        # Filter by chain if provided
        if chain:
            search_results = [r for r in search_results if r.get("chainId") == chain]

        if not search_results:
            return None

        # Pick the pair with highest liquidity
        best_raw = max(search_results, key=lambda r: float(r.get("liquidity", {}).get("usd") or 0))
        snap = self.client.parse_pair(best_raw)
        if not snap:
            return None

        # Enrich with holders
        from .holders import get_holder_count
        holders = await get_holder_count(snap.chain_id, snap.base_token_address)
        snap.holder_count = holders

        score, components = score_token(snap)
        return HotTokenCandidate(snapshot=snap, score=score, score_components=components)
