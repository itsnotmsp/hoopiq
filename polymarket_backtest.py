#!/usr/bin/env python3
"""
Polymarket 5-Minute BTC Strategy Backtester
============================================
All-in-one script with web interface.

QUICK START:
1. Save this file as: polymarket_backtest.py
2. Run: python3 polymarket_backtest.py
3. Open: http://localhost:8080

Or with your API key:
python3 polymarket_backtest.py --api-key pdm_XIiAZF1cWc8Vfp5nztw1WzDb6AVLBSuI
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import requests
except ImportError:
    print("❌ Error: 'requests' library not installed")
    print("\nPlease run:")
    print("  pip3 install requests")
    print("\nThen run this script again.")
    sys.exit(1)

class BacktestHandler(BaseHTTPRequestHandler):
    api_key = None
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_html()
        elif self.path.startswith('/api/backtest'):
            self.run_backtest_api()
        else:
            self.send_error(404)
    
    def serve_html(self):
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Polymarket Backtester</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card { backdrop-filter: blur(10px); background: rgba(255,255,255,0.1); }
        .gradient-text { background: linear-gradient(to right, #fbbf24, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
</head>
<body class="min-h-screen p-8">
    <div class="max-w-4xl mx-auto">
        <div class="text-center mb-8">
            <h1 class="text-6xl font-black text-white mb-2">POLYMARKET</h1>
            <p class="text-2xl font-bold gradient-text">5-Minute Strategy Backtester</p>
            <p class="text-white/80 mt-2">Real Historical Data Analysis</p>
        </div>
        
        <div class="card rounded-2xl p-6 mb-6 border border-white/20">
            <h2 class="text-xl font-bold text-white mb-4">⚙️ Configuration</h2>
            <div class="grid grid-cols-3 gap-4 mb-4">
                <div>
                    <label class="block text-white/80 text-sm mb-2">Days Back</label>
                    <input type="number" id="days" value="7" min="1" max="30" 
                           class="w-full px-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white">
                </div>
                <div>
                    <label class="block text-white/80 text-sm mb-2">Min Price ($)</label>
                    <input type="number" id="minPrice" value="0.03" step="0.01" 
                           class="w-full px-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white">
                </div>
                <div>
                    <label class="block text-white/80 text-sm mb-2">Max Price ($)</label>
                    <input type="number" id="maxPrice" value="0.07" step="0.01" 
                           class="w-full px-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white">
                </div>
            </div>
            <button onclick="runBacktest()" id="runBtn" 
                    class="w-full bg-gradient-to-r from-amber-400 to-orange-500 text-white font-bold py-4 rounded-xl hover:shadow-xl transform hover:scale-105 transition-all">
                🚀 RUN BACKTEST
            </button>
        </div>
        
        <div id="progress" class="card rounded-2xl p-6 mb-6 border border-white/20 hidden">
            <div class="flex items-center justify-center gap-4">
                <div class="animate-spin w-8 h-8 border-4 border-white border-t-transparent rounded-full"></div>
                <p id="progressText" class="text-white font-semibold"></p>
            </div>
        </div>
        
        <div id="error" class="bg-red-500/20 border border-red-500 rounded-2xl p-4 mb-6 hidden">
            <p id="errorText" class="text-white"></p>
        </div>
        
        <div id="results" class="hidden">
            <div class="grid grid-cols-4 gap-4 mb-6">
                <div class="card rounded-xl p-4 border border-white/20">
                    <div class="text-white/60 text-xs mb-1">TRADES</div>
                    <div id="totalTrades" class="text-3xl font-bold text-white">0</div>
                </div>
                <div class="card rounded-xl p-4 border border-emerald-400/50">
                    <div class="text-white/60 text-xs mb-1">WIN RATE</div>
                    <div id="winRate" class="text-3xl font-bold text-emerald-400">0%</div>
                </div>
                <div class="card rounded-xl p-4 border border-white/20">
                    <div class="text-white/60 text-xs mb-1">PROFIT</div>
                    <div id="profit" class="text-3xl font-bold text-white">$0</div>
                </div>
                <div class="card rounded-xl p-4 border border-amber-400/50">
                    <div class="text-white/60 text-xs mb-1">ROI</div>
                    <div id="roi" class="text-3xl font-bold text-amber-400">0%</div>
                </div>
            </div>
            
            <div class="card rounded-2xl p-6 border border-white/20">
                <h3 class="text-xl font-bold text-white mb-4">📊 Performance Analysis</h3>
                <div class="space-y-3">
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-white/80">Wins</span>
                        <span id="wins" class="font-bold text-emerald-400">0</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-white/80">Losses</span>
                        <span id="losses" class="font-bold text-rose-400">0</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-white/80">Average Win</span>
                        <span id="avgWin" class="font-bold text-emerald-400">$0</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-white/80">Average Loss</span>
                        <span id="avgLoss" class="font-bold text-rose-400">$0</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-white/80">Expected (Article)</span>
                        <span class="font-bold text-white">8.8%</span>
                    </div>
                    <div class="flex justify-between py-2">
                        <span class="text-white/80">Difference</span>
                        <span id="difference" class="font-bold text-white">0%</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function runBacktest() {
            const btn = document.getElementById('runBtn');
            const progress = document.getElementById('progress');
            const results = document.getElementById('results');
            const error = document.getElementById('error');
            
            btn.disabled = true;
            progress.classList.remove('hidden');
            results.classList.add('hidden');
            error.classList.add('hidden');
            
            const days = document.getElementById('days').value;
            const minPrice = document.getElementById('minPrice').value;
            const maxPrice = document.getElementById('maxPrice').value;
            
            document.getElementById('progressText').textContent = 'Fetching markets...';
            
            try {
                const response = await fetch(`/api/backtest?days=${days}&minPrice=${minPrice}&maxPrice=${maxPrice}`);
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('errorText').textContent = data.error;
                    error.classList.remove('hidden');
                    return;
                }
                
                document.getElementById('totalTrades').textContent = data.totalTrades;
                document.getElementById('winRate').textContent = data.winRate.toFixed(1) + '%';
                document.getElementById('profit').textContent = '$' + data.totalProfit.toFixed(2);
                document.getElementById('roi').textContent = data.roi.toFixed(1) + '%';
                document.getElementById('wins').textContent = data.wins;
                document.getElementById('losses').textContent = data.losses;
                document.getElementById('avgWin').textContent = '$' + data.avgWin.toFixed(2);
                document.getElementById('avgLoss').textContent = '$' + Math.abs(data.avgLoss).toFixed(2);
                
                const diff = data.winRate - 8.8;
                const diffEl = document.getElementById('difference');
                diffEl.textContent = (diff >= 0 ? '+' : '') + diff.toFixed(1) + '%';
                diffEl.className = diff >= 0 ? 'font-bold text-emerald-400' : 'font-bold text-rose-400';
                
                results.classList.remove('hidden');
            } catch (err) {
                document.getElementById('errorText').textContent = 'Error: ' + err.message;
                error.classList.remove('hidden');
            } finally {
                progress.classList.add('hidden');
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def run_backtest_api(self):
        try:
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            
            days = int(query.get('days', ['7'])[0])
            min_price = float(query.get('minPrice', ['0.03'])[0])
            max_price = float(query.get('maxPrice', ['0.07'])[0])
            
            print(f"\n🔍 Starting backtest: {days} days, ${min_price:.3f}-${max_price:.3f}")
            
            result = self.fetch_and_analyze(days, min_price, max_price)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def fetch_and_analyze(self, days, min_price, max_price):
        base_url = "https://api.polybacktest.com"
        headers = {"X-API-Key": self.api_key}
        
        # Fetch markets
        markets = []
        offset = 0
        start_date = datetime.now() - timedelta(days=days)
        
        print(f"📥 Fetching 5m markets...")
        while True:
            resp = requests.get(f"{base_url}/v1/markets", headers=headers, 
                              params={"limit": 100, "offset": offset, "market_type": "5m", "resolved": True}, 
                              timeout=30)
            
            if resp.status_code != 200:
                raise Exception(f"API Error {resp.status_code}")
            
            data = resp.json()
            batch = data.get('markets', [])
            if not batch:
                break
            
            for m in batch:
                mdate = datetime.fromisoformat(m['start_time'].replace('Z', '+00:00'))
                if mdate >= start_date:
                    markets.append(m)
            
            print(f"  Found {len(markets)} markets...")
            if len(batch) < 100:
                break
            offset += 100
        
        print(f"✓ Total markets: {len(markets)}")
        
        # Analyze
        trades = []
        for i, market in enumerate(markets, 1):
            if i % 10 == 0:
                print(f"  Analyzing {i}/{len(markets)}...")
            
            try:
                resp = requests.get(f"{base_url}/v1/markets/{market['market_id']}/snapshots",
                                  headers=headers, params={"limit": 1000}, timeout=30)
                if resp.status_code != 200:
                    continue
                
                snapshots = resp.json().get('snapshots', [])
                winner = market.get('winner')
                if not winner:
                    continue
                
                for snap in snapshots:
                    for side, price in [('UP', snap.get('price_up')), ('DOWN', snap.get('price_down'))]:
                        if price and min_price <= price <= max_price:
                            tokens = int(10 / price)
                            cost = tokens * price
                            won = winner.lower() == side.lower()
                            payout = tokens if won else 0
                            trades.append({'won': won, 'profit': payout - cost, 'cost': cost})
            except:
                continue
        
        if not trades:
            return {'error': 'No opportunities found'}
        
        wins = sum(1 for t in trades if t['won'])
        losses = len(trades) - wins
        total_profit = sum(t['profit'] for t in trades)
        total_cost = sum(t['cost'] for t in trades)
        win_rate = (wins / len(trades)) * 100
        roi = (total_profit / total_cost) * 100
        
        winning = [t for t in trades if t['won']]
        losing = [t for t in trades if not t['won']]
        avg_win = sum(t['profit'] for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t['profit'] for t in losing) / len(losing) if losing else 0
        
        print(f"\n✅ Results: {len(trades)} trades, {win_rate:.1f}% win rate, ${total_profit:.2f} profit")
        
        return {
            'totalTrades': len(trades),
            'wins': wins,
            'losses': losses,
            'winRate': win_rate,
            'totalProfit': total_profit,
            'roi': roi,
            'avgWin': avg_win,
            'avgLoss': avg_loss
        }
    
    def log_message(self, format, *args):
        pass

def main():
    parser = argparse.ArgumentParser(description='Polymarket Backtester')
    parser.add_argument('--api-key', type=str, default='pdm_XIiAZF1cWc8Vfp5nztw1WzDb6AVLBSuI',
                       help='PolyBackTest API key')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    args = parser.parse_args()
    BacktestHandler.api_key = args.api_key
    
    server = HTTPServer(('localhost', args.port), BacktestHandler)
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║         🚀 POLYMARKET BACKTEST SERVER RUNNING 🚀         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

✅ Server is ready!

🌐 Open in your browser:
   👉 http://localhost:{args.port}

📊 What happens next:
   1. Configure your backtest settings
   2. Click "Run Backtest"
   3. View real results from PolyBackTest API

⚡ Testing the 5-minute BTC strategy with real data!

Press Ctrl+C to stop the server.
""")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped. Thanks for using Polymarket Backtester!")

if __name__ == "__main__":
    main()
