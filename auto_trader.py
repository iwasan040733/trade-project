"""自動売買エンジン — スナイパー型ブレイクアウト/ブレイクダウン

バックテスト (backtest_short.py) と同一ロジック:
  ロング: 始値 + 前日レンジ × K 上抜け + EMA20上 + QQQ bullish + ADX≥40 +DI>-DI + VolR≥2.0 + ATR拡大
  ショート: 始値 - 前日レンジ × K 下抜け + EMA20下 + QQQ bearish + ADX≥40 -DI>+DI + VolR≥2.0 + ATR拡大
  出口: SL ATR×1.5 / トレーリング ATR×5.0 / EOD 15:55 ET
  制限: 1銘柄1日1回
"""

import logging
import math
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo

import discord
import pandas as pd
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config
from models import AutoPosition
from indicators import calc_qqq_regime_hybrid

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def is_market_open() -> bool:
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    market_open = time(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    market_close = time(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    return market_open <= now_et.time() < market_close


def is_entry_window() -> bool:
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    entry_start = time(
        config.MARKET_OPEN_HOUR,
        config.MARKET_OPEN_MINUTE + config.ENTRY_BUFFER_MINUTES_OPEN,
    )
    entry_end = time(15, 30)  # 15:30 ET
    return entry_start <= now_et.time() < entry_end


class AutoTrader:
    """スナイパー型ブレイクアウト自動売買エンジン。"""

    def __init__(
        self,
        data_client: StockHistoricalDataClient,
        trading_client: TradingClient,
    ):
        self.data_client = data_client
        self.trading_client = trading_client
        self.positions: list[AutoPosition] = []

        # 1日1回制限: {symbol: "YYYY-MM-DD"}
        self._daily_trade_done: dict[str, str] = {}

        # QQQ レジーム
        self._qqq_ratio: float = 0.5
        self._qqq_regime: str | None = None
        self._qqq_regime_source: str = "rolling"
        self._qqq_regime_lock_until: datetime | None = None
        self._qqq_cache_time: datetime | None = None

        # VIX パニック
        self._vix_panic: bool = False
        self._vix_panic_date: datetime | None = None

        # 決算ブラックアウトキャッシュ
        self._earnings_cache: dict[str, tuple] = {}

    # ----------------------------------------------------------
    #  データ取得
    # ----------------------------------------------------------
    def _fetch_5min_bars(self, symbol: str, hours: int = 4) -> pd.DataFrame | None:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=datetime.now(timezone.utc) - timedelta(hours=hours),
            )
            df = self.data_client.get_stock_bars(req).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            return df
        except Exception as e:
            log.debug(f"{symbol} 5分足取得失敗: {e}")
            return None

    def _fetch_daily_bars(self, symbol: str, limit: int = 10) -> pd.DataFrame | None:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now(timezone.utc) - timedelta(days=int(limit * 1.5) + 5),
                limit=limit,
            )
            df = self.data_client.get_stock_bars(req).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            return df
        except Exception as e:
            log.debug(f"{symbol} 日足取得失敗: {e}")
            return None

    def _get_latest_price(self, symbol: str) -> float | None:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(req)
            q = quotes[symbol]
            bid, ask = float(q.bid_price), float(q.ask_price)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return ask if ask > 0 else bid
        except Exception as e:
            log.debug(f"{symbol} 気配値取得失敗: {e}")
            return None

    # ----------------------------------------------------------
    #  決算ブラックアウト
    # ----------------------------------------------------------
    def _check_earnings_blackout(self, symbol: str) -> bool:
        today = datetime.now(ET).date()
        if symbol in self._earnings_cache:
            cache_date, is_blackout = self._earnings_cache[symbol]
            if cache_date == today:
                return is_blackout
        is_blackout = self._fetch_earnings_status(symbol)
        self._earnings_cache[symbol] = (today, is_blackout)
        return is_blackout

    def _fetch_earnings_status(self, symbol: str) -> bool:
        try:
            import yfinance as yf
            cal = yf.Ticker(symbol).calendar
            if cal is None:
                return False
            earnings_dates = list(cal.get("Earnings Date", [])) if isinstance(cal, dict) else []
            now = datetime.now(timezone.utc)
            for d in earnings_dates:
                dt = pd.Timestamp(d) if hasattr(d, "hour") else pd.Timestamp(datetime.combine(d, time(16, 0)))
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                if abs((dt - now).total_seconds()) / 3600 <= config.EARNINGS_BLACKOUT_HOURS:
                    log.info(f"[Earnings] {symbol} 決算ブラックアウト")
                    return True
            return False
        except Exception:
            return False

    # ----------------------------------------------------------
    #  市場レジーム (QQQ)
    # ----------------------------------------------------------
    def _update_market_regime(self) -> None:
        if not config.QQQ_FILTER_ENABLED:
            self._qqq_regime = None
            return

        now = datetime.now(timezone.utc)
        if self._qqq_cache_time and (now - self._qqq_cache_time).total_seconds() < config.QQQ_CACHE_SECONDS:
            return

        try:
            df = self._fetch_5min_bars("QQQ", hours=4)
            if df is None or df.empty:
                return

            new_regime, source, ratio = calc_qqq_regime_hybrid(
                df,
                window=config.QQQ_BULLISH_RATIO_WINDOW,
                cb_bars=config.QQQ_CB_BARS,
                cb_threshold_down=config.QQQ_CB_THRESHOLD_DOWN,
                cb_threshold_up=config.QQQ_CB_THRESHOLD_UP,
                gray_zone_low=config.QQQ_GRAY_ZONE_LOW,
                gray_zone_high=config.QQQ_GRAY_ZONE_HIGH,
            )
            self._qqq_ratio = ratio

            # レジームロック（チャタリング防止）
            if (
                self._qqq_regime is not None
                and new_regime != self._qqq_regime
                and self._qqq_regime_lock_until
                and now < self._qqq_regime_lock_until
            ):
                lock_remaining = int((self._qqq_regime_lock_until - now).total_seconds())
                log.info(
                    f"[Regime] ロック中({lock_remaining}秒残) "
                    f"維持={self._qqq_regime} 却下={new_regime}[{source}]"
                )
            else:
                if self._qqq_regime != new_regime:
                    self._qqq_regime_lock_until = now + timedelta(seconds=config.QQQ_REGIME_LOCK_SECONDS)
                self._qqq_regime = new_regime
                self._qqq_regime_source = source

            self._qqq_cache_time = now
            price = float(df["close"].astype(float).iloc[-1])
            log.info(
                f"[Regime] QQQ=${price:.2f} ratio={ratio:.0%} "
                f"→ {self._qqq_regime} [{self._qqq_regime_source}]"
            )
        except Exception as e:
            log.warning(f"[Regime] QQQ判定失敗: {e}")

    # ----------------------------------------------------------
    #  VIX パニック
    # ----------------------------------------------------------
    def _check_vix_panic(self) -> bool:
        if not config.VIX_PANIC_ENABLED:
            return False
        today = datetime.now(timezone.utc).date()
        if self._vix_panic and self._vix_panic_date == today:
            return True
        try:
            import yfinance as yf
            hist = yf.Ticker(config.VIX_SYMBOL).history(period="5d")
            if hist is None or len(hist) < 2:
                return False
            prev_close = float(hist["Close"].iloc[-2])
            curr = float(hist["Close"].iloc[-1])
            if prev_close <= 0:
                return False
            change_pct = (curr - prev_close) / prev_close
            self._vix_panic_date = today
            self._vix_panic = change_pct >= config.VIX_PANIC_THRESHOLD
            if self._vix_panic:
                log.warning(f"[VIX] パニックモード VIX={curr:.2f} 前日比={change_pct*100:+.1f}%")
            return self._vix_panic
        except Exception:
            return False

    # ----------------------------------------------------------
    #  空売り可否
    # ----------------------------------------------------------
    def _check_shortable(self, symbol: str) -> bool:
        try:
            asset = self.trading_client.get_asset(symbol)
            return bool(asset.shortable and asset.easy_to_borrow)
        except Exception:
            return False

    # ----------------------------------------------------------
    #  ポジションサイジング（固定額 $2,000/銘柄）
    # ----------------------------------------------------------
    def _calc_qty(self, price: float) -> int:
        return max(1, math.floor(config.POSITION_SIZE / price))

    @property
    def open_position_count(self) -> int:
        return sum(1 for p in self.positions if not p.closed)

    # ----------------------------------------------------------
    #  エントリー: スナイパー型ブレイクアウト/ブレイクダウン
    # ----------------------------------------------------------
    def check_entries(self) -> list[AutoPosition]:
        if not is_entry_window():
            return []
        if self.open_position_count >= config.MAX_POSITIONS:
            return []
        if self._check_vix_panic():
            log.info("[AutoTrade] VIXパニック — 全エントリー停止")
            return []

        self._update_market_regime()
        today_str = datetime.now(ET).strftime("%Y-%m-%d")

        new_positions = []
        for sym in config.SNIPER_SYMBOLS:
            if self.open_position_count >= config.MAX_POSITIONS:
                break
            # 1日1回制限
            if self._daily_trade_done.get(sym) == today_str:
                continue
            # 既にポジションあり
            if any(p.symbol == sym and not p.closed for p in self.positions):
                continue

            pos = self._try_breakout_entry(sym, today_str)
            if pos is not None:
                new_positions.append(pos)

        return new_positions

    def _try_breakout_entry(self, symbol: str, today_str: str) -> AutoPosition | None:
        # 決算ブラックアウト
        if self._check_earnings_blackout(symbol):
            return None

        # 5分足データ取得
        df5 = self._fetch_5min_bars(symbol, hours=4)
        if df5 is None or len(df5) < 30:
            return None

        # 日足データ取得（前日レンジ計算用）
        daily = self._fetch_daily_bars(symbol, limit=5)
        if daily is None or daily.empty:
            return None

        c5 = df5["close"].astype(float)
        h5 = df5["high"].astype(float)
        l5 = df5["low"].astype(float)
        v5 = df5["volume"].astype(float)

        price = float(c5.iloc[-1])

        # --- 前日レンジからブレイクアウト/ブレイクダウンレベル計算 ---
        today_norm = pd.Timestamp.now(tz="UTC").normalize()
        prev_daily = daily[daily.index.normalize() < today_norm]
        if prev_daily.empty:
            return None
        prev_bar = prev_daily.iloc[-1]
        prev_range = float(prev_bar["high"]) - float(prev_bar["low"])

        # 当日始値（5分足の最初のopen）
        dates = df5.index.normalize()
        today_bars = df5[dates == dates[-1]]
        if today_bars.empty:
            return None
        today_open = float(today_bars["open"].astype(float).iloc[0])

        breakout_level = today_open + prev_range * config.BREAKOUT_K
        breakdown_level = today_open - prev_range * config.BREAKOUT_K

        # --- EMA20 ---
        ema20 = c5.ewm(span=config.BREAKOUT_EMA_PERIOD, adjust=False).mean()
        ema20_val = float(ema20.iloc[-1])

        # --- ADX / DI ---
        adx_result = ta.adx(h5, l5, c5, length=config.BREAKOUT_ADX_PERIOD)
        if adx_result is None:
            return None
        adx_col = f"ADX_{config.BREAKOUT_ADX_PERIOD}"
        dmp_col = f"DMP_{config.BREAKOUT_ADX_PERIOD}"
        dmn_col = f"DMN_{config.BREAKOUT_ADX_PERIOD}"
        if pd.isna(adx_result[adx_col].iloc[-1]):
            return None
        adx_val = float(adx_result[adx_col].iloc[-1])
        di_plus = float(adx_result[dmp_col].iloc[-1])
        di_minus = float(adx_result[dmn_col].iloc[-1])

        # --- RSI ---
        rsi_series = ta.rsi(c5, length=14)
        rsi_val = float(rsi_series.iloc[-1]) if rsi_series is not None and not pd.isna(rsi_series.iloc[-1]) else 50.0

        # --- ATR ---
        atr_series = ta.atr(h5, l5, c5, length=14)
        if atr_series is None or pd.isna(atr_series.iloc[-1]):
            return None
        atr_v = float(atr_series.iloc[-1])

        # --- QQQ レジーム ---
        regime = self._qqq_regime
        ratio = self._qqq_ratio

        # ===== ロング: ブレイクアウト =====
        entry_side = None
        if price > breakout_level and price > ema20_val:
            if regime == "bullish":
                if adx_val >= config.BREAKOUT_ADX_THRESHOLD and di_plus > di_minus:
                    entry_side = "long"
                else:
                    log.debug(f"{symbol} ロング ADX不適合 ADX={adx_val:.1f} +DI={di_plus:.1f} -DI={di_minus:.1f}")
            else:
                log.debug(f"{symbol} ロング QQQ非ブル regime={regime} ratio={ratio:.0%}")

        # ===== ショート: ブレイクダウン =====
        if entry_side is None and price < breakdown_level and price < ema20_val:
            if regime == "bearish":
                if adx_val >= config.BREAKOUT_ADX_THRESHOLD and di_minus > di_plus:
                    entry_side = "short"
                else:
                    log.debug(f"{symbol} ショート ADX不適合 ADX={adx_val:.1f} +DI={di_plus:.1f} -DI={di_minus:.1f}")
            else:
                log.debug(f"{symbol} ショート QQQ非ベア regime={regime} ratio={ratio:.0%}")

        if entry_side is None:
            return None

        # === Volume Spike ===
        short_window = config.BREAKOUT_VOL_SPIKE_SHORT
        long_window = config.BREAKOUT_VOL_SPIKE_LONG
        if len(v5) < long_window:
            return None
        vol_short_avg = float(v5.iloc[-short_window:].mean())
        vol_long_avg = float(v5.iloc[-long_window:].mean())
        if vol_long_avg <= 0:
            return None
        vol_ratio = vol_short_avg / vol_long_avg
        if vol_ratio < config.BREAKOUT_VOL_SPIKE_MULT:
            log.debug(f"{symbol} 出来高不足 VolR={vol_ratio:.2f} < {config.BREAKOUT_VOL_SPIKE_MULT}")
            return None

        # === ATR拡大 ===
        if config.BREAKOUT_ATR_EXPANSION:
            bars_per_day = 78
            lookback = min(bars_per_day, len(atr_series))
            atr_day_avg = float(atr_series.iloc[-lookback:].dropna().mean()) if lookback > 0 else 0
            if atr_day_avg > 0 and atr_v <= atr_day_avg:
                log.debug(f"{symbol} ATR縮小 ATR={atr_v:.2f} <= avg={atr_day_avg:.2f}")
                return None

        # === ショート: インバースETF代替 or 空売り可否チェック ===
        if entry_side == "short":
            if symbol in config.SYMBOL_SHORT_SUBSTITUTE:
                sub_symbol = config.SYMBOL_SHORT_SUBSTITUTE[symbol]
                return self._try_inverse_entry(
                    signal_symbol=symbol, trade_symbol=sub_symbol, today_str=today_str
                )
            elif not self._check_shortable(symbol):
                log.info(f"{symbol} 空売り不可 — スキップ")
                return None

        # === 全フィルター通過 → エントリー ===
        self._daily_trade_done[symbol] = today_str
        qty = self._calc_qty(price)
        sl_dist = atr_v * config.BREAKOUT_STOP_ATR_MULT
        tp_dist = atr_v * config.BREAKOUT_TP_ATR_MULT if config.BREAKOUT_TP_ATR_MULT > 0 else 0

        if entry_side == "long":
            sl_price = round(price - sl_dist, 2)
            tp_price = round(price + tp_dist, 2) if tp_dist > 0 else 0.0
            level_label = f"L_bo_{breakout_level:.0f}"
        else:
            sl_price = round(price + sl_dist, 2)
            tp_price = round(price - tp_dist, 2) if tp_dist > 0 else 0.0
            level_label = f"S_bd_{breakdown_level:.0f}"

        log.info(
            f"[{entry_side.upper()} ENTRY] {symbol} ${price:.2f} "
            f"ADX={adx_val:.1f} +DI={di_plus:.1f} -DI={di_minus:.1f} "
            f"VolR={vol_ratio:.2f} ATR=${atr_v:.2f} RSI={rsi_val:.0f} "
            f"QQQ={ratio:.0%} SL=${sl_price:.2f}"
        )

        return self._submit_entry_order(
            symbol=symbol, side=entry_side, price=price, qty=qty,
            sl_price=sl_price, tp_price=tp_price,
            atr_value=atr_v, level_label=level_label,
            adx_val=adx_val, vol_ratio=vol_ratio, rsi_val=rsi_val,
        )

    def _try_inverse_entry(
        self, signal_symbol: str, trade_symbol: str, today_str: str
    ) -> AutoPosition | None:
        """ショート代替: シグナルシンボルのブレイクダウン検知時に、インバースETFをロングで買う。"""
        # 既にインバースETFのポジションあり
        if any(p.symbol == trade_symbol and not p.closed for p in self.positions):
            return None

        df5 = self._fetch_5min_bars(trade_symbol, hours=4)
        if df5 is None or len(df5) < 30:
            log.warning(f"{trade_symbol} インバースETFデータ不足 — スキップ")
            return None

        c5 = df5["close"].astype(float)
        h5 = df5["high"].astype(float)
        l5 = df5["low"].astype(float)
        price = float(c5.iloc[-1])

        atr_series = ta.atr(h5, l5, c5, length=14)
        if atr_series is None or pd.isna(atr_series.iloc[-1]):
            return None
        atr_v = float(atr_series.iloc[-1])

        self._daily_trade_done[signal_symbol] = today_str
        qty = self._calc_qty(price)
        sl_dist = atr_v * config.BREAKOUT_STOP_ATR_MULT
        sl_price = round(price - sl_dist, 2)
        level_label = f"L_inv_{signal_symbol}"

        log.info(
            f"[SHORT→INVERSE] {signal_symbol}ショート → {trade_symbol}ロング "
            f"${price:.2f} ATR=${atr_v:.2f} SL=${sl_price:.2f}"
        )

        return self._submit_entry_order(
            symbol=trade_symbol, side="long", price=price, qty=qty,
            sl_price=sl_price, tp_price=0.0,
            atr_value=atr_v, level_label=level_label,
            adx_val=0.0, vol_ratio=0.0, rsi_val=50.0,
        )

    # ----------------------------------------------------------
    #  注文送信
    # ----------------------------------------------------------
    def _submit_entry_order(
        self, *, symbol: str, side: str, price: float, qty: int,
        sl_price: float, tp_price: float, atr_value: float,
        level_label: str, adx_val: float, vol_ratio: float, rsi_val: float,
    ) -> AutoPosition | None:
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        try:
            order = self.trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY,
            ))
        except Exception as e:
            log.error(f"{symbol} 注文失敗: {e}")
            return None

        position = AutoPosition(
            symbol=symbol, entry_price=price, qty=qty, order_id=str(order.id),
            take_profit_price=tp_price, stop_loss_price=sl_price,
            support_level=0.0, support_name=level_label,
            atr_pct=round((atr_value / price) * 100, 2),
            atr_value=atr_value, side=side, strategy="breakout",
            highest_price=price if side == "long" else 0.0,
            lowest_price=price if side == "short" else 0.0,
        )
        self.positions.append(position)

        side_jp = "ロング" if side == "long" else "ショート"
        log.info(f"注文実行: {symbol} [{side_jp}] qty={qty} entry=${price:.2f} SL=${sl_price:.2f}")
        return position

    # ----------------------------------------------------------
    #  ポジション監視・決済
    # ----------------------------------------------------------
    def check_exits(self) -> list[AutoPosition]:
        closed_positions = []

        # EOD 15:55 ET → 全決済
        now_et = datetime.now(ET)
        if now_et.time() >= time(15, 55) and now_et.time() < time(16, 0):
            if any(not p.closed for p in self.positions):
                log.info("[AutoTrade] EOD 15:55 — 全ポジション決済")
                return self.close_all_positions()

        for position in self.positions:
            if position.closed:
                continue

            current_price = self._get_latest_price(position.symbol)
            if current_price is None:
                continue

            position.update_trailing_stop(current_price)
            exit_reason = position.check_exit(current_price)
            if exit_reason is None:
                continue

            success = self._close_position(position, current_price, exit_reason)
            if success:
                closed_positions.append(position)

        return closed_positions

    def _close_position(self, position: AutoPosition, price: float, reason: str) -> bool:
        close_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
        try:
            self.trading_client.submit_order(MarketOrderRequest(
                symbol=position.symbol, qty=position.qty,
                side=close_side, time_in_force=TimeInForce.DAY,
            ))
        except Exception as e:
            log.error(f"{position.symbol} 決済失敗: {e}")
            return False

        position.closed = True
        position.close_reason = reason
        position.close_price = price
        if position.side == "short":
            position.pnl = (position.entry_price - price) * position.qty
        else:
            position.pnl = (price - position.entry_price) * position.qty

        log.info(
            f"決済: {position.symbol} [{position.side}] reason={reason} "
            f"${position.entry_price:.2f}→${price:.2f} PnL=${position.pnl:+.2f}"
        )
        return True

    def close_all_positions(self) -> list[AutoPosition]:
        closed = []
        for p in self.positions:
            if p.closed:
                continue
            price = self._get_latest_price(p.symbol) or p.entry_price
            if self._close_position(p, price, "end_of_day"):
                closed.append(p)
        return closed

    # ----------------------------------------------------------
    #  Discord 通知
    # ----------------------------------------------------------
    def build_entry_embed(self, position: AutoPosition) -> discord.Embed:
        is_short = position.side == "short"
        side_label = "ショート" if is_short else "ロング"
        color = discord.Color.orange() if is_short else discord.Color.blue()

        embed = discord.Embed(
            title=f"🎯 スナイパー: {position.symbol} [{side_label}] エントリー",
            color=color, timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name="価格", value=f"${position.entry_price:.2f}", inline=True)
        embed.add_field(name="数量", value=str(position.qty), inline=True)
        embed.add_field(name="金額", value=f"${position.entry_price * position.qty:.2f}", inline=True)
        embed.add_field(name="損切り", value=f"${position.stop_loss_price:.2f}", inline=True)
        embed.add_field(name="トレーリング", value=f"ATR×{config.BREAKOUT_TRAILING_ATR_MULT}", inline=True)
        embed.add_field(name="ATR%", value=f"{position.atr_pct:.1f}%", inline=True)
        embed.add_field(name="レベル", value=position.support_name, inline=True)
        embed.add_field(name="QQQ", value=f"{self._qqq_regime} ({self._qqq_ratio:.0%}) [{self._qqq_regime_source}]", inline=True)
        embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Sniper")
        return embed

    def build_exit_embed(self, position: AutoPosition) -> discord.Embed:
        pnl = position.pnl
        if position.side == "short":
            pnl_pct = ((position.entry_price - position.close_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.close_price - position.entry_price) / position.entry_price) * 100

        color = discord.Color.green() if pnl >= 0 else discord.Color.red()
        title = f"{'💰' if pnl >= 0 else '🔻'} {position.symbol} [{position.side}] 決済"

        reason_map = {
            "take_profit": "利確", "stop_loss": "損切り",
            "trailing_stop": "トレーリング", "end_of_day": "EOD決済",
        }
        embed = discord.Embed(title=title, color=color, timestamp=datetime.now(timezone.utc))
        embed.add_field(name="Entry", value=f"${position.entry_price:.2f}", inline=True)
        embed.add_field(name="Exit", value=f"${position.close_price:.2f}", inline=True)
        embed.add_field(name="PnL", value=f"${pnl:+.2f} ({pnl_pct:+.1f}%)", inline=True)
        embed.add_field(name="理由", value=reason_map.get(position.close_reason, position.close_reason), inline=True)
        embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Sniper")
        return embed

    def build_status_summary(self) -> str:
        open_pos = [p for p in self.positions if not p.closed]
        if not open_pos:
            return "オープンポジション: なし"
        lines = [f"オープンポジション: {len(open_pos)}/{config.MAX_POSITIONS}"]
        for p in open_pos:
            price = self._get_latest_price(p.symbol)
            tag = "S" if p.side == "short" else "L"
            if price:
                pnl = (p.entry_price - price) * p.qty if p.side == "short" else (price - p.entry_price) * p.qty
                lines.append(f"  [{tag}] {p.symbol}: ${p.entry_price:.2f}→${price:.2f} PnL=${pnl:+.2f}")
            else:
                lines.append(f"  [{tag}] {p.symbol}: ${p.entry_price:.2f}")
        return "\n".join(lines)

    def build_daily_summary_embed(self) -> discord.Embed:
        closed_pos = [p for p in self.positions if p.closed]
        total_pnl = sum(p.pnl for p in closed_pos)
        wins = [p for p in closed_pos if p.pnl >= 0]
        losses = [p for p in closed_pos if p.pnl < 0]
        color = discord.Color.green() if total_pnl >= 0 else discord.Color.red()

        embed = discord.Embed(
            title="📋 スナイパー日次サマリー", color=color,
            timestamp=datetime.now(timezone.utc),
        )
        win_rate = (len(wins) / len(closed_pos) * 100) if closed_pos else 0
        embed.add_field(name="収支", value=(
            f"取引数: {len(closed_pos)}\n"
            f"勝ち/負け: {len(wins)}/{len(losses)}\n"
            f"勝率: {win_rate:.0f}%\n"
            f"総損益: **${total_pnl:+.2f}**"
        ), inline=False)

        if closed_pos:
            reason_map = {"take_profit": "利確", "stop_loss": "損切り", "trailing_stop": "トレーリング", "end_of_day": "EOD"}
            lines = []
            for p in closed_pos:
                pnl_pct = ((p.entry_price - p.close_price) / p.entry_price * 100) if p.side == "short" else ((p.close_price - p.entry_price) / p.entry_price * 100)
                tag = "S" if p.side == "short" else "L"
                r = reason_map.get(p.close_reason, p.close_reason)
                lines.append(f"[{tag}] **{p.symbol}** ${p.pnl:+.2f} ({pnl_pct:+.1f}%) [{r}]")
            text = "\n".join(lines)
            if len(text) > 1024:
                text = text[:1020] + "..."
            embed.add_field(name="取引履歴", value=text, inline=False)

        embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Sniper")
        return embed
