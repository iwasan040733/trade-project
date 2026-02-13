"""各エントリー条件のヒット状況を診断する。"""
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import config
from indicators import calc_pivot_points, calc_psychological_levels

ET = ZoneInfo("America/New_York")
client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

start = datetime.now(timezone.utc) - timedelta(days=35)

# 5分足
req = StockBarsRequest(symbol_or_symbols="NVDA", timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=start)
df = client.get_stock_bars(req).df
if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index(level=0, drop=True)

# 日足
req_d = StockBarsRequest(symbol_or_symbols="NVDA", timeframe=TimeFrame.Day,
                         start=datetime.now(timezone.utc) - timedelta(days=95))
df_daily = client.get_stock_bars(req_d).df
if isinstance(df_daily.index, pd.MultiIndex):
    df_daily = df_daily.reset_index(level=0, drop=True)

# 1h足
req_h = StockBarsRequest(symbol_or_symbols="NVDA", timeframe=TimeFrame.Hour, start=start)
df_1h = client.get_stock_bars(req_h).df
if isinstance(df_1h.index, pd.MultiIndex):
    df_1h = df_1h.reset_index(level=0, drop=True)
ma_1h_20 = df_1h["close"].astype(float).rolling(20).mean()

close = df["close"].astype(float)
open_ = df["open"].astype(float)
rsi = ta.rsi(close, length=14)

# RSI 30 クロスオーバー + 陽線
crossovers = []
for i in range(1, len(df)):
    ts = df.index[i]
    ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
    t = ts_et.time()
    if not (dtime(9, 30) <= t < dtime(15, 30) and ts_et.weekday() < 5):
        continue
    if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
        continue
    r_cur = float(rsi.iloc[i])
    r_prev = float(rsi.iloc[i - 1])
    if r_prev <= 30 and r_cur > 30 and float(close.iloc[i]) > float(open_.iloc[i]):
        crossovers.append((i, ts, ts_et))

print(f"RSI 30 crossover + bullish candle: {len(crossovers)} 回\n")

for i, ts, ts_et in crossovers:
    price = float(close.iloc[i])

    # サポートレベル算出
    prev_data = df_daily[df_daily.index.normalize() < ts.normalize()]
    if prev_data.empty:
        continue
    prev_bar = prev_data.iloc[-1]
    levels = calc_pivot_points(prev_bar)
    levels.update(calc_psychological_levels(float(prev_bar["close"])))
    c = prev_data["close"].astype(float)
    if len(c) >= 50:
        levels["sma50"] = round(float(c.rolling(50).mean().iloc[-1]), 4)

    # サポート接近チェック
    nearest = None
    nearest_dist = float("inf")
    for name, lvl in levels.items():
        if lvl <= 0 or lvl > price * 1.005:
            continue
        d = abs(price - lvl) / lvl
        if d < nearest_dist:
            nearest_dist = d
            nearest = (name, lvl)

    # 1h 20MA チェック
    prior_1h = ma_1h_20[ma_1h_20.index <= ts].dropna()
    above_ma = price > float(prior_1h.iloc[-1]) if not prior_1h.empty else False

    support_ok = nearest_dist <= 0.005 if nearest else False
    support_1pct = nearest_dist <= 0.01 if nearest else False
    support_2pct = nearest_dist <= 0.02 if nearest else False

    status = "ENTRY" if (support_ok and above_ma) else "-----"
    ma_str = "MA:ok" if above_ma else "MA:NG"
    if nearest:
        sup_str = f"{nearest_dist*100:.2f}% {nearest[0]}(${nearest[1]:.0f})"
    else:
        sup_str = "N/A"

    s05 = "Y" if support_ok else "N"
    s10 = "Y" if support_1pct else "N"
    s20 = "Y" if support_2pct else "N"

    print(
        f"[{status}] {ts_et.strftime('%m/%d %H:%M')} ${price:.2f}  "
        f"{ma_str}  sup={sup_str}  "
        f"0.5%={s05} 1%={s10} 2%={s20}"
    )
