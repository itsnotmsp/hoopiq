"""Alert delivery: Discord webhooks, Telegram bots, generic webhooks."""
from __future__ import annotations

import json
import time
from typing import Optional

import httpx

from .models import HotTokenCandidate


def _fmt_token(c: HotTokenCandidate) -> str:
    pc = c.price_change_h1
    arrow = "📈" if pc >= 0 else "📉"
    return (
        f"{arrow} **{c.symbol}** ({c.chain_id.upper()})\n"
        f"Score: {c.score:.0f}/100 | Price: ${c.price_usd:.6g}\n"
        f"Liq: ${c.liquidity_usd:,.0f} | Vol24h: ${c.volume_h24:,.0f}\n"
        f"1h: {pc:+.1f}% | Txns/h: {c.txns_h1}\n"
        f"{c.url}"
    )


async def send_discord(webhook_url: str, candidates: list[HotTokenCandidate], title: str = "🔥 Hot Tokens") -> bool:
    """Send alert to Discord via webhook."""
    if not candidates:
        return True
    embeds = []
    for c in candidates[:10]:
        pc = c.price_change_h1
        color = 0x00FF88 if pc >= 0 else 0xFF4444
        embeds.append({
            "title": f"{c.symbol} ({c.chain_id.upper()})",
            "description": (
                f"**Score:** {c.score:.0f}/100\n"
                f"**Price:** ${c.price_usd:.6g}\n"
                f"**Liquidity:** ${c.liquidity_usd:,.0f}\n"
                f"**Vol 24h:** ${c.volume_h24:,.0f}\n"
                f"**1h change:** {pc:+.1f}%\n"
                f"**Txns/h:** {c.txns_h1}"
            ),
            "color": color,
            "url": c.url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    payload = {
        "username": "Dexscreener CLI",
        "avatar_url": "https://dexscreener.com/favicon.ico",
        "content": f"**{title}** — {len(candidates)} signal(s) found",
        "embeds": embeds[:10],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(webhook_url, json=payload)
            return r.status_code in (200, 204)
    except Exception:
        return False


def _fmt_usd_short(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.2f}"


def _fmt_price_sci(v: float) -> str:
    """Scientific notation price: $6.51×10⁻⁴"""
    if v == 0:
        return "$0"
    if v >= 1:
        return f"${v:,.4f}"
    exp = int(f"{v:.2e}".split("e")[1])
    mantissa = v / (10 ** exp)
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"${mantissa:.2f}×10{str(exp).translate(sup)}"


def _fmt_age_short(created_at_ms: Optional[int]) -> str:
    if not created_at_ms:
        return "—"
    secs = time.time() - created_at_ms / 1000
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def _score_bar(score: float) -> str:
    """Visual score bar out of 10 blocks."""
    filled = round(score / 10)
    bar = "█" * filled + "░" * (10 - filled)
    if score >= 80:
        emoji = "🔥"
    elif score >= 60:
        emoji = "⚡"
    elif score >= 40:
        emoji = "📈"
    else:
        emoji = "💤"
    return f"{emoji} {bar} {score:.0f}/100"


def _build_token_card(c: HotTokenCandidate, spike_reason: str = "") -> str:
    """Build a rich token card matching the MonstaScan/STBot style."""
    snap = c.snapshot

    # Address line — full address in monospace
    addr = snap.base_token_address
    addr_display = f"<code>{addr}</code>"

    # Chain badge
    chain_icons = {
        "solana": "◎", "ethereum": "Ξ", "base": "🔵",
        "bsc": "🟡", "arbitrum": "🔷", "polygon": "🟣",
        "optimism": "🔴", "avalanche": "🔺",
    }
    chain_icon = chain_icons.get(snap.chain_id, "●")

    # Price change emoji
    pc1 = snap.price_change_h1
    pc24 = snap.price_change_h24
    arr1 = "📈" if pc1 >= 0 else "📉"
    arr24 = "📈" if pc24 >= 0 else "📉"

    # Buy pressure
    total_txns = snap.txns_h1_total
    buy_pct = int(snap.buy_pressure * 100)
    if buy_pct >= 70:
        bp_emoji = "🟢"
    elif buy_pct >= 50:
        bp_emoji = "🟡"
    else:
        bp_emoji = "🔴"

    # Market cap
    mcap = _fmt_usd_short(snap.market_cap) if snap.market_cap else "—"

    # Dex name capitalised
    dex_name = (snap.dex_id or "unknown").replace("-", " ").title()

    # Holder count
    holders_str = f"{snap.holder_count:,}" if snap.holder_count else "—"

    # Age
    age = _fmt_age_short(snap.created_at)

    lines = [
        f"{chain_icon} {addr_display}",
        f"├ <b>{snap.base_token_name} ({snap.base_token_symbol})</b>",
        f"├ USD: <b>{_fmt_price_sci(snap.price_usd)}</b>",
        f"├ MC: {mcap}",
        f"├ Vol: {_fmt_usd_short(snap.volume_h24)}",
        f"├ Liq: {_fmt_usd_short(snap.liquidity_usd)}",
        f"├ Seen: {age}",
        f"├ Dex: {dex_name}",
        f"├ 1h: {arr1} {pc1:+.1f}%  24h: {arr24} {pc24:+.1f}%",
        f"├ Txns/h: {total_txns}  {bp_emoji} Buys: {buy_pct}%",
        f"├ Holders: {holders_str}",
        f"├ Score: {_score_bar(c.score)}",
        f"│",
    ]
    if spike_reason and ("spike" in spike_reason or "IGNIT" in spike_reason or "HEAT" in spike_reason):
        lines.append(f"├ ⚡ <b>{spike_reason}</b>")
        lines.append(f"│")
    lines.append(f'└ <a href="{c.url}">📊 View on Dexscreener</a>')
    return "\n".join(lines)


async def send_telegram(
    bot_token: str,
    chat_id: str,
    candidates: list[HotTokenCandidate],
    title: str = "🔥 Hot Tokens",
    spike_reasons: Optional[dict[str, str]] = None,
) -> bool:
    """Send rich token cards to Telegram chat, one message per token."""
    if not candidates:
        return True

    spike_reasons = spike_reasons or {}
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    all_ok = True

    # Send header message first
    header = f"<b>{title}</b> — {len(candidates)} signal(s) found"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(url, json={
                "chat_id": chat_id,
                "text": header,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            })
        except Exception:
            pass

        # Send one card per token (up to 5)
        for c in candidates[:5]:
            reason = spike_reasons.get(f"{c.chain_id}:{c.address.lower()}", "")
            card = _build_token_card(c, spike_reason=reason)
            try:
                r = await client.post(url, json={
                    "chat_id": chat_id,
                    "text": card,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                })
                if r.status_code != 200:
                    all_ok = False
            except Exception:
                all_ok = False

    return all_ok


async def send_webhook(
    url: str,
    candidates: list[HotTokenCandidate],
    title: str = "Hot Tokens",
) -> bool:
    """Send alert to a generic JSON webhook."""
    payload = {
        "title": title,
        "timestamp": time.time(),
        "count": len(candidates),
        "tokens": [
            {
                "symbol": c.symbol,
                "name": c.name,
                "chain": c.chain_id,
                "address": c.address,
                "score": c.score,
                "price_usd": c.price_usd,
                "liquidity_usd": c.liquidity_usd,
                "volume_h24": c.volume_h24,
                "price_change_h1": c.price_change_h1,
                "txns_h1": c.txns_h1,
                "url": c.url,
            }
            for c in candidates
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload)
            return r.status_code < 400
    except Exception:
        return False


def _filter_by_age(candidates: list[HotTokenCandidate], max_hours: float = 24.0) -> list[HotTokenCandidate]:
    """Drop tokens older than max_hours. Tokens with no age data pass through."""
    now_ms = time.time() * 1000
    max_ms = max_hours * 3_600_000
    result = []
    for c in candidates:
        created = c.snapshot.created_at
        if created is None:
            result.append(c)  # no age data — let it through
        elif (now_ms - created) <= max_ms:
            result.append(c)
    return result


async def send_alerts(
    candidates: list[HotTokenCandidate],
    config: dict,
    title: str = "🔥 Dexscreener Alert",
    max_age_hours: float = 24.0,
    spike_reasons: Optional[dict[str, str]] = None,
) -> dict[str, bool]:
    """Dispatch alerts to all configured channels."""

    # Filter out old tokens before sending anywhere
    filtered = _filter_by_age(candidates, max_age_hours)
    if not filtered:
        return {}

    results: dict[str, bool] = {}
    discord_url = config.get("discord_webhook_url")
    if discord_url:
        results["discord"] = await send_discord(discord_url, filtered, title)

    tg_token = config.get("telegram_bot_token")
    tg_chat = config.get("telegram_chat_id")
    if tg_token and tg_chat:
        results["telegram"] = await send_telegram(
            tg_token, tg_chat, filtered, title,
            spike_reasons=spike_reasons,
        )

    webhook_url = config.get("webhook_url")
    if webhook_url:
        results["webhook"] = await send_webhook(webhook_url, filtered, title)

    return results


async def send_warning_alerts(
    warnings: list[dict],
    bot_token: str,
    chat_id: str,
) -> bool:
    """Send danger/exit/rug warnings to Telegram. One message per warning."""
    if not warnings:
        return True

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    all_ok = True
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=10.0) as client:
        for w in warnings:
            c = w["token"]
            dedup_key = f"{w['type']}:{c.chain_id}:{c.address.lower()}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            snap = c.snapshot
            wtype = w["type"]

            if wtype == "liq_drop":
                d = w["data"]
                lines = [
                    f"<b>{d['severity']}</b>",
                    f"├ <b>{snap.base_token_name} ({snap.base_token_symbol})</b>",
                    f"├ Chain: {c.chain_id.upper()}",
                    f"├ <code>{snap.base_token_address}</code>",
                    f"├ Liq now:  <b>${d['curr_liq']:,.0f}</b>",
                    f"├ Peak liq: ${d['peak_liq']:,.0f}",
                    f"├ Drop this scan: <b>{d['scan_drop_pct']:.0f}%</b>",
                    f"├ Drop from peak: {d['peak_drop_pct']:.0f}%",
                    f"│",
                    f'└ <a href="{c.url}">📊 View on Dexscreener</a>',
                ]
            elif wtype == "exit":
                d = w["data"]
                reasons_str = "\n├ ".join(d["reasons"])
                lines = [
                    f"<b>📉 EXIT SIGNAL: {snap.base_token_symbol}</b>",
                    f"├ <b>{snap.base_token_name}</b> ({c.chain_id.upper()})",
                    f"├ <code>{snap.base_token_address}</code>",
                    f"├ {reasons_str}",
                    f"├ Buys now: <b>{int(d['buy_pressure']*100)}%</b>  (peak: {int(d['peak_bp']*100)}%)",
                    f"├ Price: {_fmt_price_sci(snap.price_usd)}",
                    f"├ Liq: ${snap.liquidity_usd:,.0f}",
                    f"│",
                    f'└ <a href="{c.url}">📊 View on Dexscreener</a>',
                ]
            elif wtype == "concentration":
                d = w["data"]
                lines = [
                    f"<b>⚠️ CONCENTRATION WARNING</b>",
                    f"├ <b>{snap.base_token_name} ({snap.base_token_symbol})</b>",
                    f"├ Chain: {c.chain_id.upper()}",
                    f"├ <code>{snap.base_token_address}</code>",
                    f"├ {d['warning']}",
                    f"├ Liq: ${snap.liquidity_usd:,.0f}",
                    f"│",
                    f'└ <a href="{c.url}">📊 View on Dexscreener</a>',
                ]

            elif wtype == "buy_burst":
                d = w["data"]
                lines = [
                    f"<b>{d['severity']}</b>",
                    f"├ <b>{snap.base_token_name} ({snap.base_token_symbol})</b>",
                    f"├ Chain: {c.chain_id.upper()}",
                    f"├ <code>{snap.base_token_address}</code>",
                    f"│",
                    f"├ Buys this scan: <b>{d['curr_buys']}</b>",
                    f"├ Buys last scan: {d['prev_buys']}",
                    f"├ vs last scan:   <b>{d['vs_prev']:.1f}×</b>",
                    f"├ vs baseline:    <b>{d['vs_baseline']:.1f}×</b>",
                    f"├ Buy pressure:   <b>{int(d['buy_pressure']*100)}%</b>",
                    f"├ Vol 1h:  {_fmt_usd_short(d['volume_h1'])}",
                    f"├ Price:   {_fmt_price_sci(d['price_usd'])}",
                    f"├ Liq:     ${snap.liquidity_usd:,.0f}",
                    f"│",
                    f'└ <a href="{c.url}">📊 View on Dexscreener</a>',
                ]
            else:
                lines = [w.get("message", "Unknown warning")]

            text = "\n".join(lines)
            try:
                r = await client.post(url, json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                })
                if r.status_code != 200:
                    all_ok = False
            except Exception:
                all_ok = False

    return all_ok
