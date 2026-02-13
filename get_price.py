import os
import sys

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import MarketOrderRequest
# from alpaca.trading.enums import OrderSide, TimeInForce

api_key = os.environ.get("ALPACA_API_KEY")
secret_key = os.environ.get("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    print("Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables.")
    sys.exit(1)

# --- 最新の株価（仲値）を取得 ---
data_client = StockHistoricalDataClient(api_key, secret_key)

try:
    request = StockLatestQuoteRequest(symbol_or_symbols="NVDA")
    quotes = data_client.get_stock_latest_quote(request)
    quote = quotes["NVDA"]
except Exception as e:
    print(f"Failed to get quote: {e}")
    sys.exit(1)

mid_price = (quote.bid_price + quote.ask_price) / 2

print("=== NVDA Latest Quote ===")
print(f"Bid:      ${quote.bid_price:.2f}")
print(f"Ask:      ${quote.ask_price:.2f}")
print(f"Mid:      ${mid_price:.2f}")
print(f"Time:     {quote.timestamp}")

# --- NVDA を1株だけ成行注文で買う（コメントアウト） ---
# trading_client = TradingClient(api_key, secret_key, paper=True)
#
# order_request = MarketOrderRequest(
#     symbol="NVDA",
#     qty=1,
#     side=OrderSide.BUY,
#     time_in_force=TimeInForce.DAY,
# )
#
# order = trading_client.submit_order(order_request)
# print(f"\n=== Order Submitted ===")
# print(f"Order ID:  {order.id}")
# print(f"Symbol:    {order.symbol}")
# print(f"Side:      {order.side}")
# print(f"Qty:       {order.qty}")
# print(f"Type:      {order.type}")
# print(f"Status:    {order.status}")
