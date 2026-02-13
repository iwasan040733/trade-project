"""Discord に Embed + Buy ボタン付きテスト通知を送るスクリプト"""

import sys
import asyncio
from datetime import datetime, timedelta, timezone

import discord

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from indicators import (
    calc_pivot_points, calc_indicators, check_proximity,
    calc_sma_levels, calc_vwap, calc_psychological_levels,
)
from trade_bot import build_embed, BuyConfirmView, fetch_asset_info, data_client

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    channel = bot.get_channel(config.DISCORD_CHANNEL_ID)
    if channel is None:
        print(f"ERROR: Channel {config.DISCORD_CHANNEL_ID} not found")
        await bot.close()
        return

    dc = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    # 5分足
    bars = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="NVDA",
        timeframe=TimeFrame.Minute,
        start=datetime.now(timezone.utc) - timedelta(days=3),
        limit=100,
    ))
    df = bars.df
    if hasattr(df.index, "levels"):
        df = df.reset_index(level=0, drop=True)

    # 前日日足
    daily = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="NVDA",
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=5),
        limit=2,
    ))
    dfd = daily.df
    if hasattr(dfd.index, "levels"):
        dfd = dfd.reset_index(level=0, drop=True)

    # 日足(SMA用)
    daily_long = dc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="NVDA",
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=300),
        limit=250,
    ))
    dl = daily_long.df
    if hasattr(dl.index, "levels"):
        dl = dl.reset_index(level=0, drop=True)

    # 全節目を統合
    indicators = calc_indicators(df)
    price = indicators["price"]

    levels = {}
    levels.update(calc_pivot_points(dfd.iloc[-2]))
    levels.update(calc_sma_levels(dl))
    levels.update(calc_vwap(df))
    levels.update(calc_psychological_levels(price))

    asset_info = fetch_asset_info("NVDA")

    # テスト用: 5%閾値で接近リストを作る
    nearby = check_proximity(price, levels, 0.05)
    nearby.sort(key=lambda x: x["distance_pct"])

    embed = build_embed("NVDA", indicators, nearby, asset_info)
    view = BuyConfirmView(symbol="NVDA", qty=config.BUY_QTY)
    await channel.send(embed=embed, view=view)
    print("Test notification sent! Check Discord.")
    print("Waiting for button interaction (5min)...")
    await asyncio.sleep(300)
    await bot.close()


if not config.DISCORD_BOT_TOKEN:
    print("Error: DISCORD_BOT_KEY not set")
    sys.exit(1)

bot.run(config.DISCORD_BOT_TOKEN)
