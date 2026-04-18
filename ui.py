"""Terminal UI rendering using Rich."""
from __future__ import annotations

import os
import time
from typing import Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .config import SCAN_PROFILES, SCORE_HOT, SCORE_INTERESTING, SCORE_MODERATE
from .models import HotTokenCandidate
from .scoring import score_label

console = Console()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_usd(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.2f}"


def _fmt_price(v: float) -> str:
    """Always show price in scientific notation (powers of 10)."""
    if v == 0:
        return "$0"
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:.4f}"
    # Scientific: e.g. $6.51×10⁻⁴
    exp = int(f"{v:.2e}".split("e")[1])
    mantissa = v / (10 ** exp)
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp_str = str(exp).translate(superscripts)
    return f"${mantissa:.2f}×10{exp_str}"


def _fmt_pct(v: float) -> Text:
    color = "green" if v >= 0 else "red"
    return Text(f"{v:+.1f}%", style=color)


def _score_color(score: float) -> str:
    if score >= SCORE_HOT:
        return "bright_red"
    if score >= SCORE_INTERESTING:
        return "yellow"
    if score >= SCORE_MODERATE:
        return "cyan"
    return "dim"


def _chain_badge(chain: str) -> Text:
    colors = {
        "solana": "bright_magenta",
        "base":   "bright_blue",
        "ethereum": "cyan",
        "bsc":    "yellow",
        "arbitrum": "blue",
        "polygon": "magenta",
        "optimism": "red",
        "avalanche": "bright_red",
    }
    color = colors.get(chain, "white")
    abbrevs = {
        "solana": "SOL", "base": "BASE", "ethereum": "ETH",
        "bsc": "BSC", "arbitrum": "ARB", "polygon": "POLY",
        "optimism": "OP", "avalanche": "AVAX",
    }
    return Text(abbrevs.get(chain, chain.upper()[:4]), style=f"bold {color}")


def _terminal_width() -> int:
    env_w = os.environ.get("DS_TABLE_WIDTH")
    if env_w:
        try:
            return int(env_w)
        except ValueError:
            pass
    return os.get_terminal_size().columns if hasattr(os, "get_terminal_size") else 120


def _compact_mode() -> bool:
    return os.environ.get("DS_TABLE_MODE", "").lower() == "compact"


def _fmt_age(created_at_ms: Optional[int]) -> Text:
    """Format token age from a unix timestamp in milliseconds."""
    if not created_at_ms:
        return Text("—", style="dim")
    age_s = time.time() - (created_at_ms / 1000)
    if age_s < 0:
        return Text("—", style="dim")
    if age_s < 3600:
        label = f"{int(age_s // 60)}m"
        style = "bright_green"
    elif age_s < 86400:
        label = f"{int(age_s // 3600)}h"
        style = "green"
    elif age_s < 7 * 86400:
        label = f"{int(age_s // 86400)}d"
        style = "yellow"
    else:
        label = f"{int(age_s // 86400)}d"
        style = "dim"
    return Text(label, style=style)


# ── Main hot-token table ──────────────────────────────────────────────────────

def build_hot_table(
    candidates: list[HotTokenCandidate],
    title: str = "🔥 Hot Tokens",
    compact: bool = False,
) -> Table:
    compact = compact or _compact_mode()
    width = _terminal_width()
    show_holders = width >= 140 and not compact

    # Filter out tokens older than 24 hours
    now_ms = time.time() * 1000
    filtered = []
    for c in candidates:
        created = c.snapshot.created_at
        if created is None or (now_ms - created) <= 86_400_000:
            filtered.append(c)
    removed = len(candidates) - len(filtered)

    caption = f"[dim]Updated {time.strftime('%H:%M:%S')} · {len(filtered)} tokens"
    if removed:
        caption += f" · {removed} >24h filtered"
    caption += " · Ctrl+C to stop[/dim]"

    tbl = Table(
        title=title,
        box=box.ROUNDED,
        border_style="bright_black",
        show_footer=False,
        expand=False,
        title_style="bold bright_white",
        caption=caption,
    )

    tbl.add_column("#",      style="dim", width=3,  justify="right")
    tbl.add_column("Chain",  width=5,                justify="center")
    tbl.add_column("Symbol", style="bold white", width=12)
    tbl.add_column("Price",  justify="right", width=14)
    tbl.add_column("1h %",   justify="right", width=8)
    tbl.add_column("24h %",  justify="right", width=8)
    tbl.add_column("MCap",   justify="right", width=9)
    tbl.add_column("Liq",    justify="right", width=9)
    tbl.add_column("Vol24h", justify="right", width=9)
    tbl.add_column("Txns/h", justify="right", width=7)
    tbl.add_column("Age",    justify="right", width=6)
    if show_holders:
        tbl.add_column("Holders", justify="right", width=9)
    # Score: kept — it's the core ranking signal (0–100 composite of 8 signals).
    # Higher = more momentum. Used to sort the whole table.
    tbl.add_column("Score",  justify="right", width=6)

    for c in filtered:
        row_style = "on grey7" if c.score >= SCORE_HOT else ""
        holders_str = f"{c.holder_count:,}" if c.holder_count else "—"
        mcap_str = _fmt_usd(c.snapshot.market_cap) if c.snapshot.market_cap else "—"

        cells: list = [
            str(c.rank),
            _chain_badge(c.chain_id),
            c.symbol[:12],
            _fmt_price(c.price_usd),
            _fmt_pct(c.price_change_h1),
            _fmt_pct(c.price_change_h24),
            Text(mcap_str, style="cyan" if c.snapshot.market_cap else "dim"),
            _fmt_usd(c.liquidity_usd),
            _fmt_usd(c.volume_h24),
            str(c.txns_h1),
            _fmt_age(c.snapshot.created_at),
        ]
        if show_holders:
            cells.append(holders_str)
        cells.append(Text(f"{c.score:.0f}", style=_score_color(c.score)))

        tbl.add_row(*cells, style=row_style)

    return tbl


# ── New launches table ────────────────────────────────────────────────────────

def build_launches_table(
    candidates: list[HotTokenCandidate],
    max_age_minutes: int = 30,
    title: str = "🚀 New Launches",
    show_trend: bool = False,
    tracker=None,
) -> Table:
    """Special table for ultra-fresh tokens — Age first, trend column when tracking."""
    caption = f"[dim]Updated {time.strftime('%H:%M:%S')} · tokens < {max_age_minutes}m old · Ctrl+C to stop[/dim]"

    tbl = Table(
        title=title,
        box=box.ROUNDED,
        border_style="bright_black",
        show_footer=False,
        expand=False,
        title_style="bold bright_white",
        caption=caption,
    )

    tbl.add_column("#",      style="dim", width=3,  justify="right")
    tbl.add_column("Age",    width=7,  justify="right")
    tbl.add_column("Chain",  width=5,  justify="center")
    tbl.add_column("Symbol", style="bold white", width=12)
    tbl.add_column("Price",  justify="right", width=14)
    tbl.add_column("1h %",   justify="right", width=8)
    tbl.add_column("MCap",   justify="right", width=9)
    tbl.add_column("Liq",    justify="right", width=9)
    tbl.add_column("Vol1h",  justify="right", width=9)
    tbl.add_column("Txns/h", justify="right", width=7)
    tbl.add_column("Buys%",  justify="right", width=6)
    if show_trend:
        tbl.add_column("Trend",   width=14)
        tbl.add_column("Obs",     width=4, justify="right")
    tbl.add_column("Score",  justify="right", width=6)

    for c in candidates:
        snap = c.snapshot
        age = _fmt_age(snap.created_at)

        age_ms = (time.time() * 1000 - (snap.created_at or 0))
        if age_ms < 600_000:       # < 10 min — brightest
            row_style = "on dark_green"
        elif age_ms < 1_800_000:   # < 30 min
            row_style = "on grey7"
        else:
            row_style = ""

        mcap_str = _fmt_usd(snap.market_cap) if snap.market_cap else "—"
        buy_pct = int(snap.buy_pressure * 100)
        bp_color = "bright_green" if buy_pct >= 65 else "green" if buy_pct >= 50 else "red"

        cells: list = [
            str(c.rank),
            age,
            _chain_badge(c.chain_id),
            c.symbol[:12],
            _fmt_price(c.price_usd),
            _fmt_pct(c.price_change_h1),
            Text(mcap_str, style="cyan" if snap.market_cap else "dim"),
            _fmt_usd(c.liquidity_usd),
            _fmt_usd(snap.volume_h1),
            str(c.txns_h1),
            Text(f"{buy_pct}%", style=bp_color),
        ]

        if show_trend and tracker is not None:
            entry = tracker.get(c.chain_id, c.address)
            if entry:
                trend_str = entry.trend()
                ig = entry.ignition_score()
                obs = entry.obs_count
                if ig >= 65:
                    trend_style = "bold bright_red"
                elif ig >= 45:
                    trend_style = "bold yellow"
                elif ig >= 20:
                    trend_style = "cyan"
                else:
                    trend_style = "dim"
                cells.append(Text(trend_str, style=trend_style))
                cells.append(Text(str(obs), style="dim"))
            else:
                cells.append(Text("watching", style="dim"))
                cells.append(Text("1", style="dim"))

        cells.append(Text(f"{c.score:.0f}", style=_score_color(c.score)))
        tbl.add_row(*cells, style=row_style)

    return tbl

def render_inspect(c: HotTokenCandidate) -> None:
    snap = c.snapshot
    lines = []

    lines.append(f"[bold bright_white]{snap.base_token_name}[/] ([bright_cyan]{snap.base_token_symbol}[/])")
    lines.append(f"[dim]Address:[/] {snap.base_token_address}")
    lines.append(f"[dim]Chain:[/]   {snap.chain_id} ({snap.dex_id})")
    lines.append(f"[dim]Pair:[/]    {snap.pair_address}")
    lines.append("")
    lines.append(f"[bold]Price:[/]        {_fmt_price(snap.price_usd)}")
    lines.append(f"[bold]Liquidity:[/]    {_fmt_usd(snap.liquidity_usd)}")
    lines.append(f"[bold]Market Cap:[/]   {_fmt_usd(snap.market_cap) if snap.market_cap else '—'}")
    lines.append(f"[bold]FDV:[/]          {_fmt_usd(snap.fdv) if snap.fdv else '—'}")
    lines.append("")
    lines.append(f"[bold]Volume 1h:[/]    {_fmt_usd(snap.volume_h1)}")
    lines.append(f"[bold]Volume 6h:[/]    {_fmt_usd(snap.volume_h6)}")
    lines.append(f"[bold]Volume 24h:[/]   {_fmt_usd(snap.volume_h24)}")
    lines.append("")
    lines.append(f"[bold]Price 1h:[/]     {snap.price_change_h1:+.2f}%")
    lines.append(f"[bold]Price 6h:[/]     {snap.price_change_h6:+.2f}%")
    lines.append(f"[bold]Price 24h:[/]    {snap.price_change_h24:+.2f}%")
    lines.append("")
    lines.append(f"[bold]Txns/h:[/]       {snap.txns_h1_total} ({snap.txns_h1_buys}↑ / {snap.txns_h1_sells}↓)")
    lines.append(f"[bold]Buy pressure:[/] {snap.buy_pressure:.0%}")
    if snap.holder_count:
        lines.append(f"[bold]Holders:[/]      {snap.holder_count:,}")
    if snap.boost_count:
        lines.append(f"[bold]Boosts:[/]       {snap.boost_count}")
    lines.append("")
    lines.append(f"[bold bright_yellow]Score: {c.score:.1f}/100[/]  {score_label(c.score)}")
    lines.append("")
    lines.append("[bold]Score breakdown:[/]")
    for k, v in c.score_components.items():
        bar = "█" * int(v * 20)
        lines.append(f"  {k:<22} {v:.2f}  [cyan]{bar}[/]")
    lines.append("")
    lines.append(f"[link={snap.url}]{snap.url}[/link]")

    panel = Panel("\n".join(lines), title="[bold]Token Inspect[/]", border_style="cyan")
    console.print(panel)


# ── Search results table ──────────────────────────────────────────────────────

def render_search(candidates: list[HotTokenCandidate], query: str) -> None:
    tbl = build_hot_table(candidates, title=f"🔍 Search: {query}")
    console.print(tbl)


# ── Profiles table ────────────────────────────────────────────────────────────

def render_profiles(chains: Optional[list[str]] = None) -> None:
    from .config import CHAIN_MULTIPLIERS

    tbl = Table(title="📊 Scan Profiles", box=box.SIMPLE_HEAVY, border_style="bright_black")
    tbl.add_column("Profile", style="bold", width=12)
    tbl.add_column("Description", width=40)
    tbl.add_column("Min Liquidity", justify="right")
    tbl.add_column("Min Vol 24h", justify="right")
    tbl.add_column("Min Txns/h", justify="right")

    for name, p in SCAN_PROFILES.items():
        tbl.add_row(
            name,
            p["description"],
            _fmt_usd(p["min_liquidity_usd"]),
            _fmt_usd(p["min_volume_h24"]),
            str(p["min_txns_h1"]),
        )
    console.print(tbl)

    if chains:
        console.print()
        chain_tbl = Table(title="Chain multipliers", box=box.SIMPLE, border_style="dim")
        chain_tbl.add_column("Chain")
        chain_tbl.add_column("Multiplier", justify="right")
        for chain in chains:
            m = CHAIN_MULTIPLIERS.get(chain, 1.0)
            chain_tbl.add_row(chain, f"×{m:.1f}")
        console.print(chain_tbl)


# ── Live watch rendering ──────────────────────────────────────────────────────

def make_live_header(chains: list[str], sort: str, interval: int) -> Text:
    chain_str = " · ".join(c.upper() for c in chains)
    return Text(
        f"⚡ Live Scanner | {chain_str} | Sort: {sort} | Refresh: {interval}s | Ctrl+C to stop",
        style="bold bright_cyan",
    )


def spinner_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


# ── Doctor output ─────────────────────────────────────────────────────────────

async def run_doctor() -> None:
    import httpx
    from .config import DEXSCREENER_BASE, GECKO_TERMINAL_BASE

    console.print("\n[bold]🩺 Dexscreener CLI Doctor[/bold]\n")

    checks = [
        ("Dexscreener API",    f"{DEXSCREENER_BASE}/latest/dex/search?q=sol"),
        ("GeckoTerminal API",  f"{GECKO_TERMINAL_BASE}/networks/solana/trending_pools"),
        ("Honeypot.is API",    "https://api.honeypot.is/v1/IsHoneypot?address=So11111111111111111111111111111111111111112"),
        ("Blockscout API",     "https://eth.blockscout.com/api/v2/stats"),
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in checks:
            try:
                r = await client.get(url)
                status = "✅ OK" if r.status_code < 400 else f"⚠️  HTTP {r.status_code}"
                style = "green" if r.status_code < 400 else "yellow"
            except Exception as e:
                status = f"❌ {type(e).__name__}"
                style = "red"
            console.print(f"  [{style}]{status}[/{style}]  {name}")

    # Check state dir
    from .config import STATE_DIR
    import os
    state_ok = os.path.isdir(STATE_DIR)
    console.print(f"  {'✅' if state_ok else '⚠️ '} State dir: {STATE_DIR}")

    # Check Moralis key
    import os as _os
    moralis = _os.environ.get("MORALIS_API_KEY", "")
    if moralis:
        console.print(f"  ✅ Moralis key configured")
    else:
        console.print(f"  [dim]ℹ️  No MORALIS_API_KEY (optional)[/dim]")

    console.print()
