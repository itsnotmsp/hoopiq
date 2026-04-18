"""Dexscreener CLI - all commands."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box

load_dotenv()

app = typer.Typer(
    name="ds",
    help="🔥 Dexscreener Visual CLI + MCP Scanner",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()

# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def _get_client():
    from .client import DexscreenerClient
    return DexscreenerClient()


def _chains_arg(chains_str: Optional[str]) -> Optional[list[str]]:
    if not chains_str:
        return None
    return [c.strip().lower() for c in chains_str.split(",") if c.strip()]


def _load_preset_overrides(preset_name: Optional[str]) -> dict:
    if not preset_name:
        return {}
    from .state import get_preset
    p = get_preset(preset_name)
    if not p:
        console.print(f"[yellow]⚠️  Preset '{preset_name}' not found, using defaults.[/yellow]")
        return {}
    return p


def _build_filter(
    preset: Optional[str] = None,
    profile: str = "balanced",
    min_liquidity_usd: Optional[float] = None,
    min_txns_h1: Optional[int] = None,
):
    from .scanner import ScanFilter
    overrides = _load_preset_overrides(preset)
    return ScanFilter(
        profile=overrides.get("profile", profile),
        min_liquidity_usd=min_liquidity_usd or overrides.get("min_liquidity_usd"),
        min_txns_h1=min_txns_h1 or overrides.get("min_txns_h1"),
    )


def _resolve_chains(chains_str: Optional[str], preset: Optional[str]) -> Optional[list[str]]:
    if chains_str:
        return _chains_arg(chains_str)
    overrides = _load_preset_overrides(preset)
    return overrides.get("chains")


# ── hot ───────────────────────────────────────────────────────────────────────

@app.command()
def hot(
    chains: Optional[str] = typer.Option(None, "--chains", "-c", help="Chains: solana,base,ethereum,bsc,arbitrum"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results"),
    profile: str = typer.Option("balanced", "--profile", "-p", help="Filter profile: discovery/balanced/strict/pump"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Use a saved preset"),
    min_liquidity_usd: Optional[float] = typer.Option(None, "--min-liquidity-usd"),
    min_txns_h1: Optional[int] = typer.Option(None, "--min-txns-h1"),
    pump: bool = typer.Option(False, "--pump", help="Pump mode: catches tokens pumping hard right now"),
    output_json: bool = typer.Option(False, "--json", help="JSON output"),
    holders: bool = typer.Option(False, "--holders", help="Fetch holder counts"),
):
    """Scan hot tokens across configured chains."""
    from .client import DexscreenerClient
    from .scanner import Scanner, ScanFilter
    from .ui import build_hot_table

    resolved_chains = _resolve_chains(chains, preset)
    overrides = _load_preset_overrides(preset)
    use_pump = pump or profile == "pump"
    effective_profile = "pump" if use_pump else overrides.get("profile", profile)

    flt = ScanFilter(
        profile=effective_profile,
        min_liquidity_usd=min_liquidity_usd or overrides.get("min_liquidity_usd"),
        min_txns_h1=min_txns_h1 or overrides.get("min_txns_h1"),
        pump_mode=use_pump,
    )

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client, holders=holders)
            mode_str = "🚀 Pump Mode" if use_pump else "scanning"
            with console.status(f"[cyan]{mode_str}..."):
                results = await scanner.scan_hot(chains=resolved_chains, flt=flt, limit=limit, with_holders=holders)
            return results
        finally:
            await client.close()

    results = _run(_go())

    if output_json:
        data = [{"rank": c.rank, "symbol": c.symbol, "chain": c.chain_id, "score": c.score,
                 "price_usd": c.price_usd, "volume_h24": c.volume_h24, "url": c.url} for c in results]
        console.print_json(json.dumps(data))
        return

    if not results:
        console.print("[yellow]No tokens found. Try --pump or --profile discovery[/yellow]")
        return

    title = "🚀 Pump Runners" if use_pump else f"🔥 Hot Tokens ({len(results)})"
    tbl = build_hot_table(results, title=title)
    console.print(tbl)


# ── search ────────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Token name, symbol, or address"),
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    output_json: bool = typer.Option(False, "--json"),
):
    """Search tokens by name, symbol, or address."""
    from .client import DexscreenerClient
    from .scanner import Scanner
    from .ui import build_hot_table

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client)
            with console.status(f"[cyan]Searching: {query}..."):
                return await scanner.search(query, chains=_chains_arg(chains))
        finally:
            await client.close()

    results = _run(_go())

    if output_json:
        data = [{"rank": c.rank, "symbol": c.symbol, "chain": c.chain_id, "score": c.score,
                 "price_usd": c.price_usd, "url": c.url} for c in results]
        console.print_json(json.dumps(data))
        return

    if not results:
        console.print(f"[yellow]No results for '{query}'[/yellow]")
        return

    tbl = build_hot_table(results, title=f"🔍 Search: {query}")
    console.print(tbl)


# ── inspect ───────────────────────────────────────────────────────────────────

@app.command()
def inspect(
    address: str = typer.Argument(..., help="Token contract address"),
    chain: Optional[str] = typer.Option(None, "--chain", "-c"),
    output_json: bool = typer.Option(False, "--json"),
):
    """Deep-dive analysis on a specific token."""
    from .client import DexscreenerClient
    from .scanner import Scanner
    from .ui import render_inspect

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client)
            with console.status("[cyan]Fetching token data..."):
                return await scanner.inspect_token(address, chain=chain)
        finally:
            await client.close()

    result = _run(_go())

    if not result:
        console.print(f"[red]Token not found: {address}[/red]")
        raise typer.Exit(1)

    if output_json:
        snap = result.snapshot
        d = {"symbol": result.symbol, "name": result.name, "chain": result.chain_id,
             "address": result.address, "score": result.score,
             "price_usd": result.price_usd, "liquidity_usd": result.liquidity_usd,
             "volume_h24": result.volume_h24, "score_components": result.score_components}
        console.print_json(json.dumps(d))
        return

    render_inspect(result)


# ── top-new ───────────────────────────────────────────────────────────────────

@app.command(name="top-new")
def top_new(
    chain: str = typer.Option("solana", "--chain", "-c"),
    limit: int = typer.Option(20, "--limit", "-l"),
    output_json: bool = typer.Option(False, "--json"),
):
    """Top new tokens by 24h volume."""
    from .client import DexscreenerClient
    from .scanner import Scanner, ScanFilter
    from .ui import build_hot_table

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client)
            flt = ScanFilter(profile="discovery")
            with console.status(f"[cyan]Fetching new tokens on {chain}..."):
                return await scanner.scan_new_runners(chain=chain, flt=flt, limit=limit)
        finally:
            await client.close()

    results = _run(_go())
    if output_json:
        console.print_json(json.dumps([{"symbol": c.symbol, "score": c.score, "url": c.url} for c in results]))
        return
    if not results:
        console.print("[yellow]No new tokens found.[/yellow]")
        return
    tbl = build_hot_table(results, title=f"🆕 Top New Tokens — {chain.upper()}")
    console.print(tbl)


# ── new-runners ───────────────────────────────────────────────────────────────

@app.command(name="new-runners")
def new_runners(
    chain: str = typer.Option("solana", "--chain", "-c"),
    limit: int = typer.Option(20, "--limit", "-l"),
    output_json: bool = typer.Option(False, "--json"),
):
    """Fresh token runners with momentum scoring."""
    from .client import DexscreenerClient
    from .scanner import Scanner, ScanFilter
    from .ui import build_hot_table

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client)
            flt = ScanFilter(profile="discovery")
            with console.status(f"[cyan]Scanning new runners on {chain}..."):
                return await scanner.scan_new_runners(chain=chain, flt=flt, limit=limit)
        finally:
            await client.close()

    results = _run(_go())
    if output_json:
        console.print_json(json.dumps([{"symbol": c.symbol, "score": c.score, "url": c.url} for c in results]))
        return
    if not results:
        console.print("[yellow]No new runners found.[/yellow]")
        return
    tbl = build_hot_table(results, title=f"🚀 New Runners — {chain.upper()}")
    console.print(tbl)


# ── burst-watch ───────────────────────────────────────────────────────────────

@app.command(name="burst-watch")
def burst_watch(
    chains: Optional[str] = typer.Option(None, "--chains", "-c", help="Chains to watch (default: solana,base,bsc,ethereum)"),
    interval: int = typer.Option(20, "--interval", "-i", help="Scan every N seconds"),
    limit: int = typer.Option(50, "--limit", "-l", help="Tokens to track per scan"),
    profile: str = typer.Option("discovery", "--profile", "-p"),
    min_liquidity_usd: float = typer.Option(5_000, "--min-liquidity-usd"),
    telegram_bot_token: Optional[str] = typer.Option(None, "--telegram-bot-token"),
    telegram_chat_id: Optional[str] = typer.Option(None, "--telegram-chat-id"),
    burst_multiplier: float = typer.Option(3.0, "--burst-multiplier", help="Buys must spike this many times vs last scan"),
    min_buy_pressure: float = typer.Option(0.60, "--min-buy-pressure", help="Minimum buy ratio (0.6 = 60%% buys)"),
    min_buys: int = typer.Option(10, "--min-buys", help="Minimum buy count to avoid noise"),
):
    """
    Watch ALL tokens for sudden buy bursts and alert on Telegram.
    Not limited to new launches — scans the full hot token universe.
    """
    from .client import DexscreenerClient
    from .scanner import Scanner, ScanFilter, MomentumTracker
    from .alerts import send_warning_alerts
    from .ui import build_hot_table

    resolved = _chains_arg(chains) or ["solana", "base", "bsc", "ethereum"]
    tg_enabled = bool(telegram_bot_token and telegram_chat_id)

    # Persistent tracker across all scans
    tracker = MomentumTracker(max_age_minutes=240)

    # Override burst thresholds from CLI args
    import dexscreener_cli.scanner as _scanner_mod
    _original_burst = _scanner_mod.MomentumEntry.buy_burst

    def _custom_burst(self):
        if len(self.history) < 2:
            return None
        curr = self.history[-1]
        prev = self.history[-2]
        curr_buys = curr.buys_h1
        prev_buys = max(prev.buys_h1, 1)
        baseline  = max(self.baseline_buys, 1)
        vs_prev     = curr_buys / prev_buys
        vs_baseline = curr_buys / baseline
        if curr.buy_pressure < min_buy_pressure:
            return None
        if curr_buys < min_buys:
            return None
        if vs_prev >= burst_multiplier * 1.5 or vs_baseline >= burst_multiplier * 2.5:
            severity = "🚀 MASSIVE BUY BURST"
        elif vs_prev >= burst_multiplier or vs_baseline >= burst_multiplier * 1.5:
            severity = "⚡ BUY BURST"
        else:
            return None
        return {
            "severity":      severity,
            "curr_buys":     curr_buys,
            "prev_buys":     prev.buys_h1,
            "baseline_buys": self.baseline_buys,
            "vs_prev":       vs_prev,
            "vs_baseline":   vs_baseline,
            "buy_pressure":  curr.buy_pressure,
            "volume_h1":     curr.volume_h1,
            "price_usd":     curr.price_usd,
        }

    _scanner_mod.MomentumEntry.buy_burst = _custom_burst

    flt = ScanFilter(
        profile=profile,
        min_liquidity_usd=min_liquidity_usd,
        pump_mode=False,
    )

    scan_count = 0
    burst_count = 0

    async def _go():
        nonlocal scan_count, burst_count
        client = DexscreenerClient()
        scanner = Scanner(client)
        try:
            with Live(console=console, refresh_per_second=1, screen=False) as live:
                while True:
                    try:
                        scan_count += 1

                        # Scan all tokens
                        results = await scanner.scan_hot(
                            chains=resolved,
                            flt=flt,
                            limit=limit,
                        )

                        # Feed every token into tracker
                        for c in results:
                            tracker.update(c.snapshot)

                        # Check for buy bursts
                        bursts = []
                        for c in results:
                            entry = tracker.get(c.chain_id, c.address)
                            if not entry or entry.obs_count < 2:
                                continue
                            burst = entry.buy_burst()
                            if burst and not entry.buy_burst_alerted:
                                bursts.append({"type": "buy_burst", "token": c, "entry": entry, "data": burst})
                                entry.buy_burst_alerted = True
                                burst_count += 1

                        # Send Telegram alerts
                        if tg_enabled and bursts:
                            await send_warning_alerts(bursts, telegram_bot_token, telegram_chat_id)

                        # Update display
                        chain_str = ",".join(resolved).upper()
                        tbl = build_hot_table(
                            results[:20],
                            title=(
                                f"⚡ Burst Watch | {chain_str} | "
                                f"scan #{scan_count} | {interval}s | "
                                f"tracking {tracker.tracked_count} tokens | "
                                f"bursts caught: {burst_count}"
                            ),
                        )
                        live.update(tbl)
                        tracker.purge_old()

                    except Exception as e:
                        live.update(Panel(f"[red]Error: {e}[/red]"))
                    await asyncio.sleep(interval)

        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            _scanner_mod.MomentumEntry.buy_burst = _original_burst
            await client.close()

    if not tg_enabled:
        console.print("[yellow]⚠️  No Telegram configured — alerts will show in terminal only[/yellow]")
        console.print("[dim]Add --telegram-bot-token and --telegram-chat-id to get phone alerts[/dim]\n")

    console.print(f"[cyan]Starting burst watch on {','.join(resolved).upper()}...[/cyan]")
    console.print(f"[dim]Burst threshold: {burst_multiplier}× spike | Min buy pressure: {int(min_buy_pressure*100)}% | Min buys: {min_buys}[/dim]\n")

    try:
        _run(_go())
    except KeyboardInterrupt:
        console.print(f"\n[dim]Stopped. Caught {burst_count} burst(s) across {scan_count} scans.[/dim]")


# ── alpha-drops ───────────────────────────────────────────────────────────────

@app.command(name="alpha-drops")
def alpha_drops(
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    limit: int = typer.Option(20, "--limit", "-l"),
    min_score: float = typer.Option(60.0, "--min-score"),
    output_json: bool = typer.Option(False, "--json"),
):
    """Alpha-grade new drops with breakout scoring."""
    from .client import DexscreenerClient
    from .scanner import Scanner
    from .ui import build_hot_table

    resolved = _chains_arg(chains) or ["solana", "base"]

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client)
            with console.status("[cyan]Scanning for alpha drops..."):
                return await scanner.scan_alpha_drops(chains=resolved, limit=limit, min_score=min_score)
        finally:
            await client.close()

    results = _run(_go())
    if output_json:
        console.print_json(json.dumps([{"symbol": c.symbol, "score": c.score, "url": c.url} for c in results]))
        return
    if not results:
        console.print("[yellow]No alpha drops found. Try --min-score 40[/yellow]")
        return
    tbl = build_hot_table(results, title="⚡ Alpha Drops")
    console.print(tbl)


@app.command(name="launches")
def launches(
    chains: Optional[str] = typer.Option(None, "--chains", "-c", help="Chains: solana,base,bsc,ethereum"),
    max_age_minutes: int = typer.Option(30, "--max-age-minutes", "-m", help="Only tokens launched within this many minutes"),
    min_liquidity_usd: float = typer.Option(2_000, "--min-liquidity-usd"),
    min_txns: int = typer.Option(5, "--min-txns"),
    limit: int = typer.Option(30, "--limit", "-l"),
    output_json: bool = typer.Option(False, "--json"),
):
    """Scan brand-new token launches (under 30 minutes old by default)."""
    from .client import DexscreenerClient
    from .scanner import Scanner
    from .ui import build_launches_table

    resolved = _chains_arg(chains) or ["solana", "base"]

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client)
            with console.status(f"[cyan]Scanning new launches (last {max_age_minutes}m)..."):
                return await scanner.scan_new_launches(
                    chains=resolved,
                    max_age_minutes=max_age_minutes,
                    min_liquidity_usd=min_liquidity_usd,
                    min_txns=min_txns,
                    limit=limit,
                )
        finally:
            await client.close()

    results = _run(_go())

    if output_json:
        console.print_json(json.dumps([{
            "rank": c.rank, "symbol": c.symbol, "chain": c.chain_id,
            "score": c.score, "price_usd": c.price_usd,
            "created_at": c.snapshot.created_at, "url": c.url,
        } for c in results]))
        return

    if not results:
        console.print(f"[yellow]No new launches found in the last {max_age_minutes} minutes.[/yellow]")
        console.print("[dim]Try --max-age-minutes 60 or --min-liquidity-usd 500[/dim]")
        return

    tbl = build_launches_table(results, max_age_minutes=max_age_minutes)
    console.print(tbl)


@app.command(name="launches-watch")
def launches_watch(
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    max_age_minutes: int = typer.Option(30, "--max-age-minutes", "-m"),
    min_liquidity_usd: float = typer.Option(2_000, "--min-liquidity-usd"),
    min_txns: int = typer.Option(3, "--min-txns"),
    limit: int = typer.Option(30, "--limit", "-l"),
    interval: int = typer.Option(20, "--interval", "-i", help="Refresh every N seconds (lower = faster ignition detection)"),
    telegram_bot_token: Optional[str] = typer.Option(None, "--telegram-bot-token"),
    telegram_chat_id: Optional[str] = typer.Option(None, "--telegram-chat-id"),
    alert_min_score: float = typer.Option(55.0, "--alert-min-score"),
    alert_min_liquidity: float = typer.Option(3_000, "--alert-min-liquidity"),
    alert_min_observations: int = typer.Option(2, "--alert-min-obs", help="Require N scans before alerting (avoids false positives)"),
):
    """
    Live new launch monitor with ignition detection.
    Watches tokens across multiple scans and alerts the moment momentum ignites.
    """
    from .client import DexscreenerClient
    from .scanner import Scanner, MomentumTracker
    from .ui import build_launches_table
    from .alerts import send_telegram
    from .task_runner import _ca_key, _should_alert, _mark_alerted

    resolved = _chains_arg(chains) or ["solana", "base"]
    tg_enabled = bool(telegram_bot_token and telegram_chat_id)

    # Persistent tracker — lives for the whole session
    tracker = MomentumTracker(max_age_minutes=max_age_minutes + 30)

    async def _go():
        client = DexscreenerClient()
        scanner = Scanner(client)
        scan_count = 0
        try:
            with Live(console=console, refresh_per_second=1, screen=False) as live:
                while True:
                    try:
                        scan_count += 1
                        results = await scanner.scan_new_launches(
                            chains=resolved,
                            max_age_minutes=max_age_minutes,
                            min_liquidity_usd=min_liquidity_usd,
                            min_txns=min_txns,
                            limit=limit,
                            tracker=tracker,
                        )

                        tbl = build_launches_table(
                            results,
                            max_age_minutes=max_age_minutes,
                            title=(
                                f"🚀 Launch Tracker | {','.join(resolved).upper()} | "
                                f"scan #{scan_count} | {interval}s | "
                                f"watching {tracker.tracked_count} tokens"
                            ),
                            show_trend=True,
                            tracker=tracker,
                        )
                        live.update(tbl)

                        if tg_enabled and results:
                            # ── Danger / exit / rug warnings ─────────────────
                            from .alerts import send_warning_alerts
                            warnings = tracker.check_warnings(results)
                            if warnings:
                                await send_warning_alerts(
                                    warnings,
                                    telegram_bot_token,
                                    telegram_chat_id,
                                )

                        # Telegram: alert on igniting tokens with enough observations
                        if tg_enabled and results:
                            alert_tokens = []
                            for c in results:
                                if c.score < alert_min_score:
                                    continue
                                if c.liquidity_usd < alert_min_liquidity:
                                    continue
                                # Require minimum observations to avoid false positives
                                entry = tracker.get(c.chain_id, c.address)
                                if not entry or entry.obs_count < alert_min_observations:
                                    continue
                                should, reason = _should_alert(c)
                                if should:
                                    alert_tokens.append((c, reason))

                            if alert_tokens:
                                spike_reasons = {}
                                for c, reason in alert_tokens:
                                    entry = tracker.get(c.chain_id, c.address)
                                    trend = entry.trend() if entry else ""
                                    spike_reasons[f"{c.chain_id}:{c.address.lower()}"] = trend
                                    _mark_alerted(c)

                                await send_telegram(
                                    telegram_bot_token,
                                    telegram_chat_id,
                                    [c for c, _ in alert_tokens],
                                    title=f"🚀 Ignition Detected! ({len(alert_tokens)} token{'s' if len(alert_tokens) > 1 else ''})",
                                    spike_reasons=spike_reasons,
                                )

                    except Exception as e:
                        live.update(Panel(f"[red]Error: {e}[/red]"))
                    await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await client.close()

    try:
        _run(_go())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ── ai-top ────────────────────────────────────────────────────────────────────

@app.command(name="ai-top")
def ai_top(
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    limit: int = typer.Option(20, "--limit", "-l"),
    output_json: bool = typer.Option(False, "--json"),
):
    """AI-themed token leaderboard."""
    from .client import DexscreenerClient
    from .scanner import Scanner
    from .ui import build_hot_table

    resolved = _chains_arg(chains)

    async def _go():
        client = DexscreenerClient()
        try:
            scanner = Scanner(client)
            with console.status("[cyan]Scanning AI tokens..."):
                return await scanner.scan_ai_tokens(chains=resolved, limit=limit)
        finally:
            await client.close()

    results = _run(_go())
    if output_json:
        console.print_json(json.dumps([{"symbol": c.symbol, "score": c.score, "url": c.url} for c in results]))
        return
    if not results:
        console.print("[yellow]No AI tokens found.[/yellow]")
        return
    tbl = build_hot_table(results, title="🤖 AI Token Leaderboard")
    console.print(tbl)


# ── watch ─────────────────────────────────────────────────────────────────────

@app.command()
def watch(
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    limit: int = typer.Option(20, "--limit", "-l"),
    interval: int = typer.Option(7, "--interval", "-i"),
    profile: str = typer.Option("balanced", "--profile", "-p"),
    preset: Optional[str] = typer.Option(None, "--preset"),
    pump: bool = typer.Option(False, "--pump", help="Pump mode: only tokens pumping hard right now"),
):
    """Live hot runner board — auto-refreshes."""
    from .client import DexscreenerClient
    from .scanner import Scanner, ScanFilter
    from .ui import build_hot_table

    resolved_chains = _resolve_chains(chains, preset)
    overrides = _load_preset_overrides(preset)
    use_pump = pump or profile == "pump"
    effective_profile = "pump" if use_pump else overrides.get("profile", profile)
    flt = ScanFilter(profile=effective_profile, pump_mode=use_pump)

    async def _go():
        client = DexscreenerClient()
        scanner = Scanner(client)
        try:
            with Live(console=console, refresh_per_second=1, screen=False) as live:
                while True:
                    try:
                        results = await scanner.scan_hot(chains=resolved_chains, flt=flt, limit=limit)
                        if results:
                            chain_str = ",".join(resolved_chains) if resolved_chains else "all"
                            mode = "🚀 PUMP" if use_pump else "⚡ Live"
                            tbl = build_hot_table(results, title=f"{mode}: {chain_str.upper()} | {interval}s refresh")
                            live.update(tbl)
                    except Exception as e:
                        live.update(Panel(f"[red]Error: {e}[/red]"))
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            await client.close()

    try:
        _run(_go())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ── new-runners-watch ──────────────────────────────────────────────────────────

@app.command(name="new-runners-watch")
def new_runners_watch(
    chain: str = typer.Option("solana", "--chain", "-c"),
    interval: int = typer.Option(6, "--interval", "-i"),
    limit: int = typer.Option(20, "--limit", "-l"),
    watch_chains: Optional[str] = typer.Option(None, "--watch-chains"),
    no_screen: bool = typer.Option(False, "--no-screen"),
):
    """Live new runner tracker with keyboard controls."""
    from .client import DexscreenerClient
    from .scanner import Scanner, ScanFilter
    from .ui import build_hot_table
    from .watch_controls import KeyboardController, SortMode, copy_to_clipboard

    current_chain = chain
    sort = SortMode()
    selected_idx = 0
    last_results = []
    kb = KeyboardController()

    switchable_chains = _chains_arg(watch_chains) if watch_chains else [chain]

    def switch_chain(key: str):
        nonlocal current_chain
        idx = ord(key) - ord("1")
        if 0 <= idx < len(switchable_chains):
            current_chain = switchable_chains[idx]

    def select_up(_):
        nonlocal selected_idx
        selected_idx = max(0, selected_idx - 1)

    def select_down(_):
        nonlocal selected_idx
        selected_idx = min(len(last_results) - 1, selected_idx + 1)

    def copy_addr(_):
        if last_results and selected_idx < len(last_results):
            addr = last_results[selected_idx].address
            if copy_to_clipboard(addr):
                console.print(f"\n[green]Copied: {addr}[/green]")

    for i in range(9):
        kb.on(str(i + 1), switch_chain)
    kb.on("j", select_down)
    kb.on("k", select_up)
    kb.on("s", lambda _: sort.next())
    kb.on("c", copy_addr)
    kb.start()

    async def _go():
        nonlocal last_results
        client = DexscreenerClient()
        scanner = Scanner(client)
        flt = ScanFilter(profile="discovery")
        try:
            with Live(console=console, refresh_per_second=2, screen=not no_screen) as live:
                while True:
                    await kb.process()
                    try:
                        results = await scanner.scan_new_runners(chain=current_chain, flt=flt, limit=limit)
                        last_results = results
                        tbl = build_hot_table(
                            results,
                            title=f"🚀 New Runners [{current_chain.upper()}] | Sort: {sort.label} | s=sort j/k=select c=copy",
                        )
                        live.update(tbl)
                    except Exception as e:
                        live.update(Panel(f"[red]{e}[/red]"))
                    await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            kb.stop()
            await client.close()

    try:
        _run(_go())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ── alpha-drops-watch ──────────────────────────────────────────────────────────

@app.command(name="alpha-drops-watch")
def alpha_drops_watch(
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    interval: int = typer.Option(6, "--interval", "-i"),
    limit: int = typer.Option(20, "--limit", "-l"),
    discord_webhook_url: Optional[str] = typer.Option(None, "--discord-webhook-url"),
    alert_min_score: float = typer.Option(70.0, "--alert-min-score"),
    alert_cooldown_seconds: int = typer.Option(120, "--alert-cooldown-seconds"),
    no_screen: bool = typer.Option(False, "--no-screen"),
):
    """Live alpha drop scanner with optional Discord/Telegram alerts."""
    from .client import DexscreenerClient
    from .scanner import Scanner
    from .ui import build_hot_table
    from .alerts import send_alerts

    resolved = _chains_arg(chains) or ["solana", "base"]
    last_alert_time = 0.0

    async def _go():
        nonlocal last_alert_time
        client = DexscreenerClient()
        scanner = Scanner(client)
        try:
            with Live(console=console, refresh_per_second=1, screen=not no_screen) as live:
                while True:
                    try:
                        results = await scanner.scan_alpha_drops(chains=resolved, limit=limit, min_score=alert_min_score * 0.8)
                        tbl = build_hot_table(results, title=f"⚡ Alpha Drops Watch | {','.join(resolved).upper()}")
                        live.update(tbl)

                        # Fire alerts if configured
                        if discord_webhook_url:
                            alert_candidates = [c for c in results if c.score >= alert_min_score]
                            if alert_candidates and (time.time() - last_alert_time) >= alert_cooldown_seconds:
                                await send_alerts(
                                    alert_candidates,
                                    {"discord_webhook_url": discord_webhook_url},
                                    title="⚡ Alpha Drop Alert",
                                )
                                last_alert_time = time.time()
                    except Exception as e:
                        live.update(Panel(f"[red]{e}[/red]"))
                    await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await client.close()

    try:
        _run(_go())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ── setup ─────────────────────────────────────────────────────────────────────

@app.command()
def setup():
    """Interactive 5-question setup wizard."""
    from .config import DEFAULT_CHAINS, SCAN_PROFILES
    from .state import save_preset

    console.print(Panel("[bold bright_cyan]🔧 Dexscreener CLI Setup Wizard[/bold bright_cyan]", expand=False))
    console.print()

    # 1. Chains
    console.print("[bold]1. Which chains do you want to scan?[/bold]")
    console.print("   Options: solana, base, ethereum, bsc, arbitrum, polygon, optimism, avalanche")
    chains_input = Prompt.ask("   Chains (comma-separated)", default="solana,base,ethereum,bsc,arbitrum")
    chains = [c.strip().lower() for c in chains_input.split(",") if c.strip()]

    # 2. Profile
    console.print()
    console.print("[bold]2. Which scan profile?[/bold]")
    for name, p in SCAN_PROFILES.items():
        console.print(f"   [cyan]{name}[/cyan] — {p['description']}")
    profile = Prompt.ask("   Profile", choices=list(SCAN_PROFILES.keys()), default="balanced")

    # 3. Limit
    console.print()
    limit_str = Prompt.ask("[bold]3. How many results per scan?[/bold]", default="20")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 20

    # 4. Refresh interval
    console.print()
    interval_str = Prompt.ask("[bold]4. Live refresh interval (seconds)?[/bold]", default="7")
    try:
        interval = int(interval_str)
    except ValueError:
        interval = 7

    # 5. Moralis key
    console.print()
    console.print("[bold]5. Moralis API key?[/bold] [dim](optional, unlocks holder data — get free at moralis.io)[/dim]")
    moralis = Prompt.ask("   Key (or Enter to skip)", default="")

    # Save as 'default' preset
    preset = {
        "chains": chains,
        "profile": profile,
        "limit": limit,
        "interval": interval,
        "description": "Default profile (created by setup wizard)",
    }
    save_preset("default", preset)

    # Optionally write Moralis key to .env
    if moralis:
        env_path = Path(".env")
        existing = env_path.read_text() if env_path.exists() else ""
        if "MORALIS_API_KEY" not in existing:
            with open(env_path, "a") as f:
                f.write(f"\nMORALIS_API_KEY={moralis}\n")
        console.print("[green]✅ Moralis key saved to .env[/green]")

    console.print()
    console.print(Panel(
        f"[green]✅ Setup complete![/green]\n\n"
        f"Chains: [cyan]{', '.join(chains)}[/cyan]\n"
        f"Profile: [cyan]{profile}[/cyan]\n"
        f"Limit: [cyan]{limit}[/cyan]\n"
        f"Interval: [cyan]{interval}s[/cyan]\n\n"
        f"Run [bold]ds hot[/bold] to start scanning!",
        title="Setup Summary",
        expand=False,
    ))


# ── doctor ────────────────────────────────────────────────────────────────────

@app.command()
def doctor():
    """Diagnose issues and verify your setup."""
    from .ui import run_doctor
    _run(run_doctor())


# ── update ────────────────────────────────────────────────────────────────────

@app.command()
def update():
    """Pull latest code and reinstall."""
    import subprocess
    console.print("[cyan]Pulling latest code...[/cyan]")
    r1 = subprocess.run(["git", "pull"], capture_output=True, text=True)
    console.print(r1.stdout or r1.stderr)
    console.print("[cyan]Reinstalling...[/cyan]")
    r2 = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], capture_output=True, text=True)
    console.print(r2.stdout[-500:] if r2.stdout else r2.stderr[-500:])
    console.print("[green]✅ Update complete.[/green]")


# ── profiles ──────────────────────────────────────────────────────────────────

@app.command()
def profiles(
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
):
    """Show built-in filter thresholds per chain."""
    from .ui import render_profiles
    render_profiles(_chains_arg(chains))


# ── preset sub-commands ───────────────────────────────────────────────────────

preset_app = typer.Typer(name="preset", help="Manage scan presets", no_args_is_help=True)
app.add_typer(preset_app)


@preset_app.command("save")
def preset_save(
    name: str = typer.Argument(...),
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    profile: str = typer.Option("balanced", "--profile", "-p"),
    limit: int = typer.Option(20, "--limit", "-l"),
    min_liquidity_usd: Optional[float] = typer.Option(None, "--min-liquidity-usd"),
    min_volume_h24: Optional[float] = typer.Option(None, "--min-volume-h24"),
    min_txns_h1: Optional[int] = typer.Option(None, "--min-txns-h1"),
):
    """Save a custom scan preset."""
    from .state import save_preset as _save

    preset = {
        "profile": profile,
        "limit": limit,
    }
    if chains:
        preset["chains"] = _chains_arg(chains)
    if min_liquidity_usd:
        preset["min_liquidity_usd"] = min_liquidity_usd
    if min_volume_h24:
        preset["min_volume_h24"] = min_volume_h24
    if min_txns_h1:
        preset["min_txns_h1"] = min_txns_h1

    _save(name, preset)
    console.print(f"[green]✅ Preset '{name}' saved.[/green]")


@preset_app.command("list")
def preset_list():
    """List all saved presets."""
    from .state import list_presets as _list

    presets = _list()
    if not presets:
        console.print("[dim]No presets saved yet. Use 'ds preset save <name>'[/dim]")
        return

    tbl = Table(title="📋 Saved Presets", box=box.SIMPLE_HEAVY)
    tbl.add_column("Name", style="bold")
    tbl.add_column("Profile")
    tbl.add_column("Chains")
    tbl.add_column("Limit", justify="right")
    for name, p in presets.items():
        chains_str = ",".join(p.get("chains") or []) or "—"
        tbl.add_row(name, p.get("profile", "balanced"), chains_str, str(p.get("limit", 20)))
    console.print(tbl)


@preset_app.command("show")
def preset_show(name: str = typer.Argument(...)):
    """Show details of a preset."""
    from .state import get_preset

    p = get_preset(name)
    if not p:
        console.print(f"[red]Preset '{name}' not found.[/red]")
        raise typer.Exit(1)
    console.print_json(json.dumps(p, indent=2))


@preset_app.command("delete")
def preset_delete(name: str = typer.Argument(...)):
    """Delete a preset."""
    from .state import delete_preset

    if delete_preset(name):
        console.print(f"[green]Deleted preset '{name}'.[/green]")
    else:
        console.print(f"[red]Preset '{name}' not found.[/red]")


# ── task sub-commands ──────────────────────────────────────────────────────────

task_app = typer.Typer(name="task", help="Manage scheduled scan tasks", no_args_is_help=True)
app.add_typer(task_app)


@task_app.command("create")
def task_create(
    name: str = typer.Argument(...),
    chains: Optional[str] = typer.Option(None, "--chains", "-c"),
    profile: str = typer.Option("balanced", "--profile"),
    preset: Optional[str] = typer.Option(None, "--preset"),
    limit: int = typer.Option(10, "--limit"),
    interval_seconds: int = typer.Option(300, "--interval-seconds"),
):
    """Create a scheduled scan task."""
    from .state import create_task as _create

    task = {
        "chains": _resolve_chains(chains, preset),
        "profile": profile,
        "limit": limit,
        "interval_seconds": interval_seconds,
    }
    if preset:
        task["preset"] = preset
    _create(name, task)
    console.print(f"[green]✅ Task '{name}' created (every {interval_seconds}s).[/green]")


@task_app.command("list")
def task_list():
    """List all tasks."""
    from .state import list_tasks as _list
    import time

    tasks = _list()
    if not tasks:
        console.print("[dim]No tasks. Use 'ds task create <name>'[/dim]")
        return

    tbl = Table(title="📅 Scan Tasks", box=box.SIMPLE_HEAVY)
    tbl.add_column("Name", style="bold")
    tbl.add_column("Interval")
    tbl.add_column("Chains")
    tbl.add_column("Last Run")
    tbl.add_column("Alerts", justify="center")

    for name, t in tasks.items():
        chains_str = ",".join(t.get("chains") or []) or "all"
        last = t.get("last_run")
        last_str = time.strftime("%H:%M:%S", time.localtime(last)) if last else "never"
        has_discord = "✅" if t.get("discord_webhook_url") else ""
        has_tg = "✅" if t.get("telegram_bot_token") else ""
        alerts = f"Discord{has_discord} TG{has_tg}"
        tbl.add_row(name, f"{t.get('interval_seconds', 300)}s", chains_str, last_str, alerts)
    console.print(tbl)


@task_app.command("run")
def task_run(name: str = typer.Argument(...)):
    """Run a task once immediately."""
    from .state import get_task
    from .task_runner import run_task as _run_task
    from .ui import build_hot_table

    task = get_task(name)
    if not task:
        console.print(f"[red]Task '{name}' not found.[/red]")
        raise typer.Exit(1)

    async def _go():
        with console.status(f"[cyan]Running task: {name}..."):
            return await _run_task(name, task)

    results = _run(_go())
    if results:
        tbl = build_hot_table(results, title=f"Task: {name}")
        console.print(tbl)
    else:
        console.print("[yellow]No results.[/yellow]")


@task_app.command("daemon")
def task_daemon(
    all_tasks: bool = typer.Option(False, "--all", help="Run all due tasks in loop"),
):
    """Run the task scheduler daemon."""
    from .task_runner import daemon_loop

    console.print("[cyan]Starting task daemon... Press Ctrl+C to stop.[/cyan]")
    try:
        _run(daemon_loop(verbose=True))
    except KeyboardInterrupt:
        console.print("\n[dim]Daemon stopped.[/dim]")


@task_app.command("configure")
def task_configure(
    name: str = typer.Argument(...),
    discord_webhook_url: Optional[str] = typer.Option(None, "--discord-webhook-url"),
    telegram_bot_token: Optional[str] = typer.Option(None, "--telegram-bot-token"),
    telegram_chat_id: Optional[str] = typer.Option(None, "--telegram-chat-id"),
    webhook_url: Optional[str] = typer.Option(None, "--webhook-url"),
    alert_min_score: Optional[float] = typer.Option(None, "--alert-min-score"),
    alert_cooldown_seconds: Optional[int] = typer.Option(None, "--alert-cooldown-seconds"),
):
    """Configure alerts for a task."""
    from .state import update_task

    updates: dict = {}
    if discord_webhook_url:
        updates["discord_webhook_url"] = discord_webhook_url
    if telegram_bot_token:
        updates["telegram_bot_token"] = telegram_bot_token
    if telegram_chat_id:
        updates["telegram_chat_id"] = telegram_chat_id
    if webhook_url:
        updates["webhook_url"] = webhook_url
    if alert_min_score is not None:
        updates["alert_min_score"] = alert_min_score
    if alert_cooldown_seconds is not None:
        updates["alert_cooldown_seconds"] = alert_cooldown_seconds

    if update_task(name, updates):
        console.print(f"[green]✅ Task '{name}' configured.[/green]")
    else:
        console.print(f"[red]Task '{name}' not found.[/red]")


@task_app.command("test-alert")
def task_test_alert(name: str = typer.Argument(...)):
    """Send a test alert for a task."""
    async def _go():
        with console.status("[cyan]Sending test alert..."):
            return await test_task_alert(name)

    from .task_runner import test_task_alert
    results = _run(_go())
    if results.get("error"):
        console.print(f"[red]Task '{name}' not found.[/red]")
    else:
        for channel, ok in results.items():
            status = "✅" if ok else "❌"
            console.print(f"  {status} {channel}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
