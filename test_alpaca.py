import os
import sys

from alpaca.trading.client import TradingClient

api_key = os.environ.get("ALPACA_API_KEY")
secret_key = os.environ.get("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    print("Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables.")
    sys.exit(1)

# paper=True でペーパートレード用エンドポイントに接続
client = TradingClient(api_key, secret_key, paper=True)

try:
    account = client.get_account()
except Exception as e:
    print(f"API connection failed: {e}")
    sys.exit(1)

print("=== Alpaca API Connection Successful ===")
print(f"Account ID:     {account.id}")
print(f"Status:         {account.status}")
print(f"Currency:       {account.currency}")
print(f"Cash:           ${account.cash}")
print(f"Portfolio Value: ${account.portfolio_value}")
print(f"Buying Power:   ${account.buying_power}")
print(f"Equity:         ${account.equity}")
