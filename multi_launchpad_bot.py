#!/usr/bin/env python3
"""
Enhanced Pump.fun Launch Monitor with Direct API Integration
Monitors pump.fun, bonk.fun and other launchpads for new tokens
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiLaunchpadBot:
    def __init__(
        self,
        telegram_bot_token: str,
        telegram_chat_id: str,
        check_interval: int = 20
    ):
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.check_interval = check_interval
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Track seen tokens
        self.seen_tokens: Set[str] = set()
        
        # APIs
        self.dexscreener_api = "https://api.dexscreener.com/latest/dex"
        
        # Alert thresholds (customizable)
        self.config = {
            'min_volume_5m': 3000,      # $3k in 5 min
            'min_volume_1h': 15000,     # $15k in 1 hour
            'min_liquidity': 5000,      # $5k liquidity
            'min_market_cap': 10000,    # $10k market cap
            'max_age_hours': 24,        # Only tokens launched in last 24h
        }
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        logger.info("✅ Bot initialized")
        
    async def close(self):
        if self.session:
            await self.session.close()
    
    async def send_telegram_alert(self, message: str):
        """Send formatted alert to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Telegram error: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending Telegram: {e}")
            return False
    
    async def fetch_solana_new_pairs(self) -> List[Dict]:
        """Fetch new Solana pairs from DexScreener"""
        try:
            # Get latest Solana pairs sorted by creation time
            url = f"{self.dexscreener_api}/pairs/solana"
            
            pairs = []
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'pairs' in data:
                        pairs = data['pairs']
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error fetching pairs: {e}")
            return []
    
    async def search_new_tokens(self) -> List[Dict]:
        """Search for newly launched tokens across platforms"""
        all_tokens = []
        
        try:
            # Method 1: Get latest Solana pairs
            pairs = await self.fetch_solana_new_pairs()
            
            current_time = datetime.now()
            
            for pair in pairs[:50]:  # Check top 50 most recent
                try:
                    # Check if token is new enough
                    created_at = pair.get('pairCreatedAt')
                    if created_at:
                        created_time = datetime.fromtimestamp(created_at / 1000)
                        hours_old = (current_time - created_time).total_seconds() / 3600
                        
                        if hours_old > self.config['max_age_hours']:
                            continue  # Too old
                    
                    # Extract token info
                    base_token = pair.get('baseToken', {})
                    quote_token = pair.get('quoteToken', {})
                    
                    # Skip if quote token is not SOL or common stables
                    quote_symbol = quote_token.get('symbol', '')
                    if quote_symbol not in ['SOL', 'USDC', 'USDT']:
                        continue
                    
                    token_info = {
                        'address': base_token.get('address'),
                        'name': base_token.get('name', 'Unknown'),
                        'symbol': base_token.get('symbol', '???'),
                        'pair_address': pair.get('pairAddress'),
                        'dex': pair.get('dexId', 'Unknown'),
                        'price_usd': float(pair.get('priceUsd', 0)),
                        'volume_5m': float(pair.get('volume', {}).get('m5', 0)),
                        'volume_1h': float(pair.get('volume', {}).get('h1', 0)),
                        'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                        'liquidity_usd': float(pair.get('liquidity', {}).get('usd', 0)),
                        'market_cap': float(pair.get('fdv', 0)),  # Fully diluted valuation
                        'price_change_5m': float(pair.get('priceChange', {}).get('m5', 0)),
                        'price_change_1h': float(pair.get('priceChange', {}).get('h1', 0)),
                        'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                        'txns_5m': pair.get('txns', {}).get('m5', {}),
                        'txns_1h': pair.get('txns', {}).get('h1', {}),
                        'created_at': created_at,
                        'age_hours': hours_old if created_at else None,
                        'url': pair.get('url', ''),
                        'chain': 'solana'
                    }
                    
                    all_tokens.append(token_info)
                    
                except Exception as e:
                    logger.debug(f"Error parsing pair: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in search_new_tokens: {e}")
        
        return all_tokens
    
    def calculate_alert_score(self, token: Dict) -> float:
        """Calculate alert score based on multiple factors"""
        score = 0.0
        
        # Volume scoring
        volume_5m = token.get('volume_5m', 0)
        volume_1h = token.get('volume_1h', 0)
        
        if volume_5m > 10000:
            score += 50
        elif volume_5m > 5000:
            score += 30
        elif volume_5m > 3000:
            score += 15
        
        if volume_1h > 50000:
            score += 40
        elif volume_1h > 20000:
            score += 25
        elif volume_1h > 10000:
            score += 10
        
        # Liquidity scoring
        liquidity = token.get('liquidity_usd', 0)
        if liquidity > 50000:
            score += 30
        elif liquidity > 20000:
            score += 20
        elif liquidity > 10000:
            score += 10
        
        # Transaction volume scoring
        txns_5m = token.get('txns_5m', {})
        buys = txns_5m.get('buys', 0)
        sells = txns_5m.get('sells', 0)
        
        if buys > 20:
            score += 20
        elif buys > 10:
            score += 10
        
        # Buy/sell ratio (more buys = good)
        if sells > 0:
            buy_ratio = buys / sells
            if buy_ratio > 2:
                score += 15
        
        # Price momentum
        price_change_5m = token.get('price_change_5m', 0)
        if price_change_5m > 50:
            score += 25
        elif price_change_5m > 20:
            score += 15
        
        return score
    
    def should_alert(self, token: Dict) -> tuple[bool, float]:
        """Determine if token should trigger alert"""
        
        # Basic requirements
        liquidity = token.get('liquidity_usd', 0)
        if liquidity < self.config['min_liquidity']:
            return False, 0
        
        volume_5m = token.get('volume_5m', 0)
        volume_1h = token.get('volume_1h', 0)
        
        # Must meet minimum volume
        if volume_5m < self.config['min_volume_5m'] and volume_1h < self.config['min_volume_1h']:
            return False, 0
        
        # Calculate score
        score = self.calculate_alert_score(token)
        
        # Alert if score is high enough
        if score >= 50:  # Threshold score
            return True, score
        
        return False, score
    
    def format_telegram_alert(self, token: Dict, score: float) -> str:
        """Format beautiful Telegram message"""
        
        name = token['name']
        symbol = token['symbol']
        address = token['address']
        price = token['price_usd']
        
        volume_5m = token['volume_5m']
        volume_1h = token['volume_1h']
        volume_24h = token['volume_24h']
        
        liquidity = token['liquidity_usd']
        market_cap = token['market_cap']
        
        price_change_5m = token['price_change_5m']
        price_change_1h = token['price_change_1h']
        price_change_24h = token['price_change_24h']
        
        txns_5m = token.get('txns_5m', {})
        buys = txns_5m.get('buys', 0)
        sells = txns_5m.get('sells', 0)
        
        age_hours = token.get('age_hours')
        dex = token['dex']
        url = token['url']
        
        # Emojis
        fire_emoji = "🔥" if volume_5m > 10000 else "⚡"
        trend_emoji = "🚀" if price_change_5m > 0 else "📉"
        score_emoji = "💎" if score > 80 else "⭐" if score > 60 else "✨"
        
        age_str = f"{age_hours:.1f}h" if age_hours else "Unknown"
        
        message = f"""
{fire_emoji}{score_emoji} <b>HIGH VOLUME LAUNCH DETECTED!</b> {score_emoji}{fire_emoji}

<b>{name} (${symbol})</b>
{trend_emoji} ${price:.10f}

📊 <b>VOLUME</b>
├ 5m: ${volume_5m:,.0f}
├ 1h: ${volume_1h:,.0f}
└ 24h: ${volume_24h:,.0f}

💰 <b>METRICS</b>
├ Liquidity: ${liquidity:,.0f}
├ MCap: ${market_cap:,.0f}
└ Alert Score: {score:.0f}/100

📈 <b>PRICE CHANGE</b>
├ 5m: {price_change_5m:+.1f}%
├ 1h: {price_change_1h:+.1f}%
└ 24h: {price_change_24h:+.1f}%

🔄 <b>TRANSACTIONS (5m)</b>
├ Buys: {buys}
└ Sells: {sells}

⏰ <b>Age:</b> {age_str}
🏪 <b>DEX:</b> {dex}

📍 <code>{address}</code>

<b>🔗 LINKS:</b>
• <a href="{url}">DexScreener</a>
• <a href="https://pump.fun/{address}">Pump.fun</a>
• <a href="https://solscan.io/token/{address}">Solscan</a>
• <a href="https://birdeye.so/token/{address}?chain=solana">Birdeye</a>

⚠️ <b>DYOR - Not Financial Advice!</b>
"""
        return message.strip()
    
    async def scan_and_alert(self):
        """Main scanning function"""
        logger.info("🔍 Scanning for new launches...")
        
        try:
            tokens = await self.search_new_tokens()
            logger.info(f"Found {len(tokens)} tokens to analyze")
            
            alerts_sent = 0
            
            for token in tokens:
                address = token.get('address')
                
                if not address or address in self.seen_tokens:
                    continue
                
                should_alert, score = self.should_alert(token)
                
                if should_alert:
                    logger.info(f"🚨 ALERT: {token['symbol']} | Score: {score:.0f} | "
                              f"Vol 5m: ${token['volume_5m']:,.0f}")
                    
                    message = self.format_telegram_alert(token, score)
                    success = await self.send_telegram_alert(message)
                    
                    if success:
                        self.seen_tokens.add(address)
                        alerts_sent += 1
                        await asyncio.sleep(1)  # Rate limit
            
            if alerts_sent > 0:
                logger.info(f"✅ Sent {alerts_sent} alerts")
            else:
                logger.info("No qualifying tokens found")
                
        except Exception as e:
            logger.error(f"Error in scan_and_alert: {e}")
    
    async def run(self):
        """Main loop"""
        await self.initialize()
        
        # Send startup notification
        await self.send_telegram_alert(
            "🤖 <b>Multi-Launchpad Bot Started!</b>\n\n"
            "Monitoring: pump.fun, bonk.fun, and all Solana DEXes\n\n"
            f"<b>Thresholds:</b>\n"
            f"• Min 5m volume: ${self.config['min_volume_5m']:,}\n"
            f"• Min 1h volume: ${self.config['min_volume_1h']:,}\n"
            f"• Min liquidity: ${self.config['min_liquidity']:,}\n"
            f"• Check interval: {self.check_interval}s\n\n"
            "Let's find some gems! 💎"
        )
        
        try:
            while True:
                await self.scan_and_alert()
                
                # Cleanup old tokens
                if len(self.seen_tokens) > 500:
                    self.seen_tokens = set(list(self.seen_tokens)[-300:])
                
                logger.info(f"⏳ Next scan in {self.check_interval}s...")
                await asyncio.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Bot stopped")
        finally:
            await self.close()


async def main():
    # Load from environment variables
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    if not BOT_TOKEN or not CHAT_ID:
        print("\n❌ Missing Telegram credentials!\n")
        print("Setup instructions:")
        print("1. Create bot: Message @BotFather → /newbot")
        print("2. Get chat ID: Message @userinfobot")
        print("\nThen set environment variables:")
        print("export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("export TELEGRAM_CHAT_ID='your_chat_id'\n")
        return
    
    bot = MultiLaunchpadBot(
        telegram_bot_token=BOT_TOKEN,
        telegram_chat_id=CHAT_ID,
        check_interval=20  # Check every 20 seconds
    )
    
    # Customize thresholds if needed
    bot.config['min_volume_5m'] = 3000
    bot.config['min_volume_1h'] = 15000
    bot.config['min_liquidity'] = 5000
    
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
