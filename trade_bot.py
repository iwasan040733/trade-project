"""
AUTO_SYMBOLS 自動売買 + プレマーケット・スクリーナー通知 Bot

3つの非同期ループが並行動作:
  - monitor_loop    (5分)  : WATCHLIST銘柄の節目監視 → Discord通知（手動承認）
  - screener_loop   (8:30 ET 1日1回) : 高ボラ銘柄スクリーニング → 通知のみ（自動売買には使わない）
  - auto_trade_loop (30秒) : AUTO_SYMBOLS のエントリー判定 + ポジション監視・決済

使い方:
  export ALPACA_API_KEY="..."
  export ALPACA_SECRET_KEY="..."
  export DISCORD_BOT_KEY="..."
  export DISCORD_CHANNEL_ID="..."
  ./venv/bin/python trade_bot.py
"""

import asyncio
import os
import signal
import sys
import logging
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import discord
from discord.ext import tasks
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config
from indicators import (
    calc_pivot_points, calc_indicators, check_proximity,
    calc_sma_levels, calc_vwap, calc_psychological_levels,
)
from screener import StockScreener
from auto_trader import AutoTrader

# ============================================================
#  ログ設定（コンソール + ファイル出力）
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# コンソール出力
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

# ファイル出力（日付ローテーション、30日分保持）
from logging.handlers import TimedRotatingFileHandler
_file_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "trade_bot.log"),
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_console_handler, _file_handler],
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ============================================================
#  Alpaca クライアント
# ============================================================
data_client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
trading_client = TradingClient(
    config.ALPACA_API_KEY,
    config.ALPACA_SECRET_KEY,
    paper=config.ALPACA_PAPER,
)

# ============================================================
#  自動売買エンジン & スクリーナー
# ============================================================
screener = StockScreener(data_client)
auto_trader = AutoTrader(data_client, trading_client)

# ============================================================
#  Discord Bot
# ============================================================
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


# ----------------------------------------------------------
#  データ取得
# ----------------------------------------------------------
def fetch_5min_bars(symbol: str, limit: int = 100) -> pd.DataFrame:
    """直近の5分足バーを取得して DataFrame で返す。"""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime.now(timezone.utc) - timedelta(days=3),
        limit=limit,
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    # マルチインデックスの場合はリセット
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    return df


def fetch_prev_daily_bar(symbol: str) -> pd.Series:
    """前日の日足バーを取得する。"""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=5),
        limit=2,
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    return df.iloc[-2]  # 前日


def fetch_daily_bars(symbol: str, limit: int = 250) -> pd.DataFrame:
    """SMA 算出用に日足バーを取得する（200日分以上）。"""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=limit + 50),
        limit=limit,
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    return df


def fetch_asset_info(symbol: str) -> dict:
    """Alpaca の get_asset で銘柄情報を取得。"""
    try:
        asset = trading_client.get_asset(symbol)
        return {
            "name": asset.name,
            "exchange": asset.exchange,
            "tradable": asset.tradable,
            "easy_to_borrow": getattr(asset, "easy_to_borrow", None),
        }
    except Exception as e:
        log.warning(f"get_asset failed: {e}")
        return {}


# ----------------------------------------------------------
#  Discord Embed 作成
# ----------------------------------------------------------
def build_embed(
    symbol: str,
    indicators: dict,
    nearby_levels: list,
    asset_info: dict,
) -> discord.Embed:
    """通知用の Discord Embed を組み立てる。"""
    score = indicators["total_score"]
    max_score = indicators["max_score"]
    price = indicators["price"]

    # スコアに応じた色
    if score >= 4:
        color = discord.Color.green()
    elif score >= 2:
        color = discord.Color.gold()
    else:
        color = discord.Color.red()

    embed = discord.Embed(
        title=f"📊 {symbol} 買いシグナル検出",
        description=f"**現在価格: ${price:.2f}**\n買い推奨スコア: **{score} / {max_score}**",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # 節目への接近情報
    levels_text = "\n".join(
        f"• **{lv['name']}** (${lv['level']:.2f}) — "
        f"{'↑' if lv.get('direction') == 'above' else '↓'} {lv['distance_pct']:.2f}%"
        for lv in nearby_levels
    )
    embed.add_field(name="🎯 接近中の節目", value=levels_text or "なし", inline=False)

    # スコア内訳
    scores = indicators["scores"]
    details = indicators["details"]

    score_lines = []
    # ボリンジャーバンド
    bb = details.get("bb")
    bb_status = f"${bb['lower']:.2f} / ${bb['mid']:.2f} / ${bb['upper']:.2f}" if bb else "N/A"
    score_lines.append(f"{'✅' if scores['bb'] else '⬜'} BB(-2σ以下): {bb_status}")

    # RSI
    rsi = details.get("rsi")
    score_lines.append(f"{'✅' if scores['rsi'] else '⬜'} RSI(≤30): {rsi if rsi else 'N/A'}")

    # 一目均衡表
    ichi = details.get("ichimoku")
    ichi_text = f"SA={ichi['span_a']:.2f} SB={ichi['span_b']:.2f} ({ichi['position']})" if ichi else "N/A"
    score_lines.append(f"{'✅' if scores['ichimoku'] else '⬜'} 一目均衡表: {ichi_text}")

    # 出来高
    vol = details.get("volume")
    vol_text = f"{vol['current']:,} (平均: {vol['avg5']:,})" if vol else "N/A"
    score_lines.append(f"{'✅' if scores['volume'] else '⬜'} 出来高増加: {vol_text}")

    # ボラティリティ
    volatility = details.get("volatility")
    vol_pct = f"ATR={volatility['atr']:.2f} ({volatility['pct']:.2f}%)" if volatility else "N/A"
    score_lines.append(f"{'✅' if scores['volatility'] else '⬜'} 注目度(Vol): {vol_pct}")

    embed.add_field(name="📈 スコア内訳", value="\n".join(score_lines), inline=False)

    # 銘柄情報
    if asset_info:
        info_text = f"{asset_info.get('name', '')} ({asset_info.get('exchange', '')})"
        embed.add_field(name="ℹ️ 銘柄情報", value=info_text, inline=False)

    embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Trading")
    return embed


# ----------------------------------------------------------
#  Buy ボタン View
# ----------------------------------------------------------
class BuyConfirmView(discord.ui.View):
    """「承認（Buy）」ボタンを含む View。"""

    def __init__(self, symbol: str, qty: int):
        super().__init__(timeout=300)  # 5分でタイムアウト
        self.symbol = symbol
        self.qty = qty
        self.order_id = None  # 送信済み注文IDを保持

    @discord.ui.button(label="承認（Buy）", style=discord.ButtonStyle.green, emoji="💰")
    async def buy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """ボタンが押されたら成行注文を実行する。"""
        await interaction.response.defer()

        try:
            # 最新の気配値を取得して表示
            quote_req = StockLatestQuoteRequest(symbol_or_symbols=self.symbol)
            quotes = data_client.get_stock_latest_quote(quote_req)
            quote = quotes[self.symbol]
            mid = (float(quote.bid_price) + float(quote.ask_price)) / 2

            # 成行注文を送信
            order_request = MarketOrderRequest(
                symbol=self.symbol,
                qty=self.qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = trading_client.submit_order(order_request)
            self.order_id = order.id

            result_embed = discord.Embed(
                title="✅ 注文を送信しました",
                color=discord.Color.green(),
            )
            result_embed.add_field(name="銘柄", value=self.symbol, inline=True)
            result_embed.add_field(name="数量", value=str(self.qty), inline=True)
            result_embed.add_field(name="参考仲値", value=f"${mid:.2f}", inline=True)
            result_embed.add_field(name="注文ID", value=str(order.id), inline=False)
            result_embed.add_field(name="ステータス", value=str(order.status), inline=True)

            button.disabled = True
            button.label = "注文済み"
            button.style = discord.ButtonStyle.grey
            await interaction.edit_original_response(embed=result_embed, view=self)

        except Exception as e:
            log.error(f"Order failed: {e}")
            error_embed = discord.Embed(
                title="❌ 注文に失敗しました",
                description=str(e),
                color=discord.Color.red(),
            )
            await interaction.followup.send(embed=error_embed)

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.grey, emoji="❌")
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """キャンセルボタン。未約定の注文があれば取り消す。"""
        await interaction.response.defer()

        cancel_results = []

        # Buy ボタンで送信済みの注文があればキャンセル
        if self.order_id:
            try:
                trading_client.cancel_order_by_id(str(self.order_id))
                cancel_results.append(f"注文 `{self.order_id}` をキャンセルしました。")
            except Exception as e:
                cancel_results.append(f"注文 `{self.order_id}` のキャンセル失敗: {e}")
        else:
            # Buy ボタンが押される前のキャンセル（通知自体を無視）
            cancel_results.append("注文は未送信です。通知を閉じました。")

        for child in self.children:
            child.disabled = True

        cancel_embed = discord.Embed(
            title="❌ キャンセル",
            description="\n".join(cancel_results),
            color=discord.Color.light_grey(),
        )
        await interaction.edit_original_response(embed=cancel_embed, view=self)
        self.stop()


# ----------------------------------------------------------
#  監視ループ（WATCHLIST — 手動承認）
# ----------------------------------------------------------
@tasks.loop(seconds=config.CHECK_INTERVAL_SECONDS)
async def monitor_loop():
    """WATCHLIST銘柄の節目監視ループ。5分ごとに実行。"""
    for symbol in config.WATCHLIST:
        log.info(f"[Monitor] Checking {symbol}...")
        try:
            # 5分足・前日日足・日足(SMA用)を取得
            df_5min = fetch_5min_bars(symbol)
            prev_daily = fetch_prev_daily_bar(symbol)
            daily_df = fetch_daily_bars(symbol)

            # 全節目を統合
            levels = {}
            levels.update(calc_pivot_points(prev_daily))
            levels.update(calc_sma_levels(daily_df))
            levels.update(calc_vwap(df_5min))

            # テクニカル指標算出
            indicators = calc_indicators(df_5min)
            current_price = indicators["price"]

            levels.update(calc_psychological_levels(current_price))

            # 節目への接近チェック
            nearby = check_proximity(current_price, levels, config.PROXIMITY_THRESHOLD)

            if not nearby:
                log.info(f"  ${current_price:.2f} — 節目への接近なし")
                continue

            log.info(f"  🎯 節目接近検出: {[n['name'] for n in nearby]}")

            # 銘柄情報取得
            asset_info = fetch_asset_info(symbol)

            # Discord に通知
            channel = bot.get_channel(config.DISCORD_CHANNEL_ID)
            if channel is None:
                log.error(f"Channel {config.DISCORD_CHANNEL_ID} not found")
                continue

            embed = build_embed(symbol, indicators, nearby, asset_info)
            view = BuyConfirmView(symbol=symbol, qty=config.BUY_QTY)
            await channel.send(embed=embed, view=view)
            log.info(f"  Discord notification sent for {symbol}.")

        except Exception as e:
            log.error(f"Monitor error ({symbol}): {e}", exc_info=True)


@monitor_loop.before_loop
async def before_monitor():
    """Bot が ready になるまで待機。"""
    await bot.wait_until_ready()
    log.info("Monitor loop started.")


# ----------------------------------------------------------
#  スクリーナーループ（毎日 8:30 ET — プレマーケット通知のみ）
# ----------------------------------------------------------
@tasks.loop(time=dtime(hour=config.SCREENER_HOUR_ET, minute=config.SCREENER_MINUTE_ET, tzinfo=ET))
async def screener_loop():
    """プレマーケット・スクリーニング。取引開始1時間前に1回実行（通知のみ）。"""
    log.info("[Screener] プレマーケット・スクリーニング開始...")

    try:
        candidates = screener.screen()

        if candidates:
            log.info(f"[Screener] {len(candidates)} 銘柄が高ボラ候補（通知のみ）")

            channel = bot.get_channel(config.DISCORD_CHANNEL_ID)
            if channel:
                embed = discord.Embed(
                    title="🔍 プレマーケット・スクリーニング",
                    description=(
                        f"**{len(candidates)}** 銘柄が高ボラティリティ候補\n"
                        f"※ 参考情報のみ — 自動売買対象は AUTO_SYMBOLS"
                    ),
                    color=discord.Color.purple(),
                    timestamp=datetime.now(timezone.utc),
                )

                # 上位10銘柄 + 節目安値を表示
                lines = []
                for c in candidates[:10]:
                    # 現在価格以下のサポートレベルを近い順に抽出
                    supports_below = []
                    for name, level in c.support_levels.items():
                        if 0 < level <= c.current_price * 1.005:
                            dist_pct = (c.current_price - level) / c.current_price * 100
                            supports_below.append((name, level, dist_pct))
                    supports_below.sort(key=lambda x: x[2])

                    sup_text = " / ".join(
                        f"{n}=${v:.0f}(-{d:.1f}%)" for n, v, d in supports_below[:4]
                    )
                    lines.append(
                        f"**{c.symbol}** ATR%={c.atr_pct}% (${c.current_price:.2f})\n"
                        f"  節目: {sup_text or 'なし'}"
                    )

                embed.add_field(
                    name="上位候補 + 節目安値",
                    value="\n".join(lines) or "なし",
                    inline=False,
                )
                embed.add_field(
                    name="AUTO_SYMBOLS (フォールバック)",
                    value=", ".join(config.AUTO_SYMBOLS_FALLBACK),
                    inline=True,
                )
                embed.set_footer(
                    text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} | "
                         f"自動売買は AUTO_SYMBOLS のみ"
                )
                await channel.send(embed=embed)
        else:
            log.info("[Screener] 条件を満たす銘柄なし")

    except Exception as e:
        log.error(f"Screener error: {e}", exc_info=True)


@screener_loop.before_loop
async def before_screener():
    await bot.wait_until_ready()
    log.info(
        f"Screener loop started "
        f"(daily at {config.SCREENER_HOUR_ET}:{config.SCREENER_MINUTE_ET:02d} ET)."
    )


# ----------------------------------------------------------
#  自動売買ループ（30秒ごと）
# ----------------------------------------------------------
@tasks.loop(seconds=config.AUTO_TRADE_INTERVAL_SECONDS)
async def auto_trade_loop():
    """自動売買ループ。30秒ごとにエントリー判定 + ポジション監視。"""
    try:
        # エントリー判定
        new_positions = auto_trader.check_entries()
        for position in new_positions:
            log.info(f"[AutoTrade] エントリー: {position.symbol} ${position.entry_price:.2f}")
            channel = bot.get_channel(config.DISCORD_CHANNEL_ID)
            if channel:
                embed = auto_trader.build_entry_embed(position)
                await channel.send(embed=embed)

        # ポジション監視・決済
        closed_positions = auto_trader.check_exits()
        for position in closed_positions:
            log.info(
                f"[AutoTrade] 決済: {position.symbol} "
                f"reason={position.close_reason} PnL=${position.pnl:+.2f}"
            )
            channel = bot.get_channel(config.DISCORD_CHANNEL_ID)
            if channel:
                embed = auto_trader.build_exit_embed(position)
                await channel.send(embed=embed)

    except Exception as e:
        log.error(f"AutoTrade error: {e}", exc_info=True)


@auto_trade_loop.before_loop
async def before_auto_trade():
    await bot.wait_until_ready()
    log.info("AutoTrade loop started.")


# ----------------------------------------------------------
#  グレースフルシャットダウン
# ----------------------------------------------------------
async def graceful_shutdown():
    """シグナル受信時: 全決済 → サマリー送信 → 停止通知 → Bot終了。"""
    log.info("[Shutdown] グレースフルシャットダウン開始...")

    # ループを停止
    if monitor_loop.is_running():
        monitor_loop.cancel()
    screener_loop.cancel()
    auto_trade_loop.cancel()

    channel = bot.get_channel(config.DISCORD_CHANNEL_ID)

    # オープンポジションを全決済
    closed = auto_trader.close_all_positions()
    if closed:
        log.info(f"[Shutdown] {len(closed)} ポジションを決済")
        if channel:
            for pos in closed:
                embed = auto_trader.build_exit_embed(pos)
                await channel.send(embed=embed)

    # 日次サマリーを送信
    if channel:
        summary_embed = auto_trader.build_daily_summary_embed()
        await channel.send(embed=summary_embed)

    # 停止通知
    if channel:
        embed = discord.Embed(
            title="🔴 Bot 停止",
            description="自動売買システムを停止しました",
            color=discord.Color.dark_grey(),
            timestamp=datetime.now(timezone.utc),
        )
        embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Trading")
        await channel.send(embed=embed)

    log.info("[Shutdown] Discord 通知完了。Bot を終了します。")
    await bot.close()


# ----------------------------------------------------------
#  Bot イベント
# ----------------------------------------------------------
@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    log.info(f"SNIPER_SYMBOLS: {config.SNIPER_SYMBOLS}")
    log.info(f"Channel: {config.DISCORD_CHANNEL_ID}")
    log.info(f"Entry window: 9:{config.MARKET_OPEN_MINUTE + config.ENTRY_BUFFER_MINUTES_OPEN:02d}~"
             f"{config.MARKET_CLOSE_HOUR}:{config.MARKET_CLOSE_MINUTE:02d} "
             f"(- {config.ENTRY_BUFFER_MINUTES_CLOSE}min) ET")
    log.info(f"AutoTrade interval: {config.AUTO_TRADE_INTERVAL_SECONDS}s")
    log.info(f"Account: ${config.ACCOUNT_SIZE} / Position: ${config.POSITION_SIZE} / Max: {config.MAX_POSITIONS}")
    log.info(f"Paper: {config.ALPACA_PAPER}")

    # ループ起動を最優先（Discord通知より前に行う）
    # monitor_loop は無効化（買いシグナル通知は停止中）
    # if not monitor_loop.is_running():
    #     monitor_loop.start()
    if not screener_loop.is_running():
        screener_loop.start()
    if not auto_trade_loop.is_running():
        auto_trade_loop.start()

    # シグナルハンドラを登録（SIGTERM / SIGINT でグレースフル停止）
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(graceful_shutdown()))

    # 起動通知を Discord に送信（失敗しても取引ループには影響しない）
    try:
        channel = bot.get_channel(config.DISCORD_CHANNEL_ID)
        if channel:
            embed = discord.Embed(
                title="🚀 Bot 起動",
                description="自動売買システムが稼働を開始しました",
                color=discord.Color.blue(),
                timestamp=datetime.now(timezone.utc),
            )
            embed.add_field(
                name="SNIPER銘柄",
                value=", ".join(config.SNIPER_SYMBOLS),
                inline=False,
            )
            embed.add_field(
                name="リスク管理",
                value=f"資金${config.ACCOUNT_SIZE:,} / 1銘柄${config.POSITION_SIZE:,}",
                inline=True,
            )
            embed.add_field(name="最大ポジション", value=f"{config.MAX_POSITIONS}銘柄", inline=True)
            embed.add_field(
                name="損切り / トレーリング",
                value=f"SL=ATR×{config.BREAKOUT_STOP_ATR_MULT} / Trail=ATR×{config.BREAKOUT_TRAILING_ATR_MULT}",
                inline=True,
            )
            embed.add_field(
                name="エントリー時間帯",
                value=f"9:{config.MARKET_OPEN_MINUTE + config.ENTRY_BUFFER_MINUTES_OPEN:02d}"
                      f"~15:{60 - config.ENTRY_BUFFER_MINUTES_CLOSE:02d} ET",
                inline=True,
            )
            embed.add_field(
                name="フィルター",
                value=f"ADX≥{config.BREAKOUT_ADX_THRESHOLD} / VolR≥{config.BREAKOUT_VOL_SPIKE_MULT} / ATR拡大",
                inline=True,
            )
            embed.add_field(
                name="AutoTrade間隔",
                value=f"{config.AUTO_TRADE_INTERVAL_SECONDS}秒",
                inline=True,
            )
            # 市場レジーム設定
            qqq_status = "有効" if config.QQQ_FILTER_ENABLED else "無効"
            vix_status = f"有効(+{config.VIX_PANIC_THRESHOLD*100:.0f}%)" if config.VIX_PANIC_ENABLED else "無効"
            embed.add_field(
                name="市場レジーム",
                value=f"QQQ bullish/bearish={qqq_status} / VIXパニック={vix_status}",
                inline=False,
            )
            embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Trading")
            await channel.send(embed=embed)
    except Exception as e:
        log.error(f"起動通知の送信に失敗しました（取引ループは正常稼働中）: {e}")


# ----------------------------------------------------------
#  エントリーポイント
# ----------------------------------------------------------
def main():
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        print("Error: ALPACA_API_KEY / ALPACA_SECRET_KEY が未設定です。")
        sys.exit(1)
    if not config.DISCORD_BOT_TOKEN:
        print("Error: DISCORD_BOT_KEY が未設定です。")
        sys.exit(1)
    if config.DISCORD_CHANNEL_ID == 0:
        print("Error: DISCORD_CHANNEL_ID が未設定です。")
        sys.exit(1)

    bot.run(config.DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
