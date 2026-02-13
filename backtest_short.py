"""バックテスト: グレーゾーン + カナリア戦略比較

3モード比較:
  A) ロング（従来 — レジーム制御なし）
  B) 新戦略: グレーゾーン制御ロング + VWAP条件ショート + カナリア
  各サブ: B-Long, B-Short(regime), B-Canary を個別集計

使い方:
    python backtest_short.py
    python backtest_short.py --days 5
    python backtest_short.py --symbols NVDA,TSLA,AAPL --days 3
"""

import argparse
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from indicators import (
    calc_atr_3day,
    calc_pivot_points,
    calc_resistance_levels,
    calc_psychological_levels,
    calc_dynamic_take_profit,
    calc_running_vwap,
    calc_qqq_bullish_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


# ============================================================
#  BTPosition
# ============================================================
@dataclass
class BTPosition:
    symbol: str
    side: str
    strategy: str  # "long", "short", "canary"
    entry_price: float
    entry_time: pd.Timestamp
    qty: int
    take_profit_price: float
    stop_loss_price: float
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_activated: bool = False
    level_name: str = ""
    atr_pct: float = 0.0
    rsi_at_entry: float = 0.0
    vwap_at_entry: float = 0.0  # canary: VWAP上抜けで損切り

    closed: bool = False
    close_price: float = 0.0
    close_time: pd.Timestamp | None = None
    close_reason: str = ""
    pnl: float = 0.0

    def update_trailing(self, bar_high: float, bar_low: float) -> None:
        if self.side == "long":
            if bar_high > self.highest_price:
                self.highest_price = bar_high
            if not self.trailing_activated:
                if (bar_high - self.entry_price) / self.entry_price >= config.TRAILING_ACTIVATE_PCT:
                    self.trailing_activated = True
            if self.trailing_activated:
                ns = self.highest_price * (1 - config.TRAILING_RETURN_PCT)
                if ns > self.trailing_stop_price:
                    self.trailing_stop_price = ns
        else:
            if self.lowest_price <= 0 or bar_low < self.lowest_price:
                self.lowest_price = bar_low
            if not self.trailing_activated:
                if (self.entry_price - bar_low) / self.entry_price >= config.TRAILING_ACTIVATE_PCT:
                    self.trailing_activated = True
            if self.trailing_activated:
                ns = self.lowest_price * (1 + config.TRAILING_RETURN_PCT)
                if self.trailing_stop_price <= 0 or ns < self.trailing_stop_price:
                    self.trailing_stop_price = ns

    def check_exit(self, bh: float, bl: float, cur_vwap: float = 0.0) -> tuple[str | None, float]:
        if self.side == "long":
            if bl <= self.stop_loss_price:
                return "stop_loss", self.stop_loss_price
            if bh >= self.take_profit_price:
                return "take_profit", self.take_profit_price
            if self.trailing_stop_price > 0 and bl <= self.trailing_stop_price:
                return "trailing_stop", self.trailing_stop_price
        else:
            if bh >= self.stop_loss_price:
                return "stop_loss", self.stop_loss_price
            if bl <= self.take_profit_price:
                return "take_profit", self.take_profit_price
            # canary: VWAP上抜け
            if self.strategy == "canary" and cur_vwap > 0:
                bar_close = (bh + bl) / 2  # 近似
                if bar_close > cur_vwap:
                    return "vwap_cross", bar_close
            if self.trailing_stop_price > 0 and bh >= self.trailing_stop_price:
                return "trailing_stop", self.trailing_stop_price
        return None, 0.0

    def calc_pnl(self, exit_price: float) -> float:
        if self.side == "short":
            return (self.entry_price - exit_price) * self.qty
        return (exit_price - self.entry_price) * self.qty


# ============================================================
#  データ取得
# ============================================================
def fetch_bars(client, symbol, timeframe, days):
    start = datetime.now(timezone.utc) - timedelta(days=days + 5)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start)
    df = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    return df


def calc_support_for_date(daily_df, target_date):
    prev = daily_df[daily_df.index.normalize() < target_date.normalize()]
    if prev.empty:
        return {}
    levels = {}
    levels.update(calc_pivot_points(prev.iloc[-1]))
    levels.update(calc_psychological_levels(float(prev.iloc[-1]["close"])))
    c = prev["close"].astype(float)
    if len(c) >= 50:
        levels["sma50"] = round(float(c.rolling(50).mean().iloc[-1]), 4)
    if len(prev) >= 2:
        levels["prev2_low"] = round(float(prev.iloc[-2]["low"]), 4)
    return levels


def calc_resistance_for_date(daily_df, target_date, price):
    prev = daily_df[daily_df.index.normalize() < target_date.normalize()]
    if prev.empty:
        return {}
    levels = {}
    levels.update(calc_resistance_levels(prev.iloc[-1]))
    for n, v in calc_psychological_levels(price).items():
        if v >= price:
            levels[n] = v
    c = prev["close"].astype(float)
    if len(c) >= 50:
        sv = float(c.rolling(50).mean().iloc[-1])
        if sv >= price:
            levels["sma50"] = round(sv, 4)
    return levels


# ============================================================
#  メインバックテスト
# ============================================================
def run_backtest(symbols, days, canary_symbols=None):
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    if canary_symbols is None:
        canary_symbols = list(config.RETAIL_FAVORITES)

    # --- QQQ データ ---
    log.info("QQQ データ取得中...")
    qqq_5min = fetch_bars(client, "QQQ", TimeFrame(5, TimeFrameUnit.Minute), days)
    qqq_daily = fetch_bars(client, "QQQ", TimeFrame.Day, days + 10)
    log.info(f"QQQ 5分足={len(qqq_5min)}本")

    # QQQ ブリッシュ比率
    ratio_series = calc_qqq_bullish_ratio(qqq_5min, window=config.QQQ_BULLISH_RATIO_WINDOW)
    qqq_close = qqq_5min["close"].astype(float)

    # QQQ 前日終値（日ごと）
    qqq_prev_close_map = {}
    for i in range(1, len(qqq_daily)):
        d = qqq_daily.index[i].normalize()
        qqq_prev_close_map[d] = float(qqq_daily["close"].astype(float).iloc[i - 1])

    # QQQ 日中始値（日ごと）
    qqq_day_open_map = {}
    qqq_dates = qqq_5min.index.normalize()
    for d in qqq_dates.unique():
        day_data = qqq_5min[qqq_dates == d]
        if not day_data.empty:
            qqq_day_open_map[d] = float(day_data["open"].astype(float).iloc[0])

    # --- 全銘柄データ取得 ---
    all_symbols = list(set(symbols) | set(canary_symbols))
    sym_data = {}
    for sym in all_symbols:
        try:
            d5 = fetch_bars(client, sym, TimeFrame(5, TimeFrameUnit.Minute), days)
            dd = fetch_bars(client, sym, TimeFrame.Day, days + 200)
            if d5.empty:
                continue
            sym_data[sym] = {"5min": d5, "daily": dd}
            log.info(f"  {sym}: 5min={len(d5)} daily={len(dd)}")
        except Exception as e:
            log.warning(f"  {sym} 取得失敗: {e}")

    # --- 結果格納 ---
    trades_a_long = []      # A: 従来ロング
    trades_b_long = []      # B: グレーゾーン制御ロング
    trades_b_short = []     # B: レジームショート
    trades_b_canary = []    # B: カナリア

    market_open = dtime(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    market_close = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    entry_start = dtime(
        config.MARKET_OPEN_HOUR,
        config.MARKET_OPEN_MINUTE + config.ENTRY_BUFFER_MINUTES_OPEN,
    )
    eod_min = config.EOD_CLOSE_MINUTES_BEFORE

    # 各銘柄シミュレーション
    for sym in symbols:
        if sym not in sym_data:
            continue

        df5 = sym_data[sym]["5min"]
        dfd = sym_data[sym]["daily"]

        c5 = df5["close"].astype(float)
        h5 = df5["high"].astype(float)
        l5 = df5["low"].astype(float)
        o5 = df5["open"].astype(float)
        rsi5 = ta.rsi(c5, length=14)
        atr5 = ta.atr(h5, l5, c5, length=14)
        sma_d50 = dfd["close"].astype(float).rolling(50).mean()
        vwap5 = calc_running_vwap(df5)

        # 日始値マップ
        d5_dates = df5.index.normalize()
        day_open = {}
        for d in d5_dates.unique():
            dd_data = df5[d5_dates == d]
            if not dd_data.empty:
                day_open[d] = float(dd_data["open"].astype(float).iloc[0])

        sup_cache = {}
        res_cache = {}

        open_a = None
        open_b_long = None
        open_b_short = None

        for i in range(20, len(df5)):
            ts = df5.index[i]
            price = float(c5.iloc[i])
            bh = float(h5.iloc[i])
            bl = float(l5.iloc[i])
            bo = float(o5.iloc[i])

            ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
            bt = ts_et.time()
            if bt < market_open or bt >= market_close or ts_et.weekday() >= 5:
                continue

            eod_t = (datetime.combine(ts_et.date(), market_close, tzinfo=ET) - timedelta(minutes=eod_min)).time()

            # レジーム取得
            prior_r = ratio_series[ratio_series.index <= ts]
            ratio = float(prior_r.iloc[-1]) if not prior_r.empty else 0.5

            cur_vwap = float(vwap5.iloc[i]) if not pd.isna(vwap5.iloc[i]) else 0.0

            # グレーゾーン判定
            in_gray = config.QQQ_GRAY_ZONE_LOW <= ratio <= config.QQQ_GRAY_ZONE_HIGH
            sl_tighten = config.GRAY_ZONE_SL_TIGHTEN_PCT if in_gray else 0.0

            # レジーム文字列
            if ratio > config.QQQ_GRAY_ZONE_HIGH:
                regime = "bullish"
            elif ratio < config.QQQ_GRAY_ZONE_LOW:
                regime = "bearish"
            else:
                regime = "gray"

            # --- EOD決済 ---
            if bt >= eod_t:
                for pos, tlist in [(open_a, trades_a_long), (open_b_long, trades_b_long), (open_b_short, trades_b_short)]:
                    if pos is not None and not pos.closed:
                        pos.closed = True
                        pos.close_price = price
                        pos.close_time = ts
                        pos.close_reason = "end_of_day"
                        pos.pnl = pos.calc_pnl(price)
                        tlist.append(pos)
                open_a = open_b_long = open_b_short = None
                continue

            # --- ポジション監視 ---
            for pos, tlist in [(open_a, trades_a_long), (open_b_long, trades_b_long), (open_b_short, trades_b_short)]:
                if pos is None or pos.closed:
                    continue
                pos.update_trailing(bh, bl)
                reason, ep = pos.check_exit(bh, bl, cur_vwap)
                if reason:
                    pos.closed = True
                    pos.close_price = ep
                    pos.close_time = ts
                    pos.close_reason = reason
                    pos.pnl = pos.calc_pnl(ep)
                    tlist.append(pos)
            if open_a is not None and open_a.closed:
                open_a = None
            if open_b_long is not None and open_b_long.closed:
                open_b_long = None
            if open_b_short is not None and open_b_short.closed:
                open_b_short = None

            # --- エントリー ---
            if bt < entry_start or bt >= dtime(15, 30):
                continue
            if pd.isna(rsi5.iloc[i]) or pd.isna(atr5.iloc[i]):
                continue
            rsi_c = float(rsi5.iloc[i])
            rsi_p = float(rsi5.iloc[i - 1]) if not pd.isna(rsi5.iloc[i - 1]) else None
            atr_v = float(atr5.iloc[i])
            if rsi_p is None:
                continue

            dk = ts_et.strftime("%Y-%m-%d")
            prior_sma = sma_d50[sma_d50.index.normalize() <= ts.normalize()].dropna()
            above_sma50 = not prior_sma.empty and price > float(prior_sma.iloc[-1])

            prior_d = dfd[dfd.index.normalize() < ts.normalize()]
            atr_daily = calc_atr_3day(prior_d) if len(prior_d) >= 3 else None
            tp_pct = calc_dynamic_take_profit(atr_daily, price, config.TAKE_PROFIT_MIN, config.TAKE_PROFIT_MAX) if atr_daily else config.TAKE_PROFIT_MIN
            max_stop = price * config.STOP_LOSS_MAX_PCT
            sl_dist = min(atr_v * config.STOP_LOSS_ATR_MULT, max_stop)
            qty = max(1, math.floor(config.POSITION_SIZE / price))

            # === A: 従来ロング ===
            if open_a is None and above_sma50:
                if rsi_p <= config.ENTRY_RSI_THRESHOLD and rsi_c > config.ENTRY_RSI_THRESHOLD and price > bo:
                    if dk not in sup_cache:
                        sup_cache[dk] = calc_support_for_date(dfd, ts)
                    nearest = _find_nearest_level(price, sup_cache[dk], config.ENTRY_PROXIMITY_THRESHOLD, "support")
                    if nearest:
                        open_a = BTPosition(
                            symbol=sym, side="long", strategy="long",
                            entry_price=price, entry_time=ts, qty=qty,
                            take_profit_price=round(price * (1 + tp_pct), 2),
                            stop_loss_price=round(price - sl_dist, 2),
                            highest_price=price, level_name=nearest[0],
                            atr_pct=round((atr_v / price) * 100, 2), rsi_at_entry=round(rsi_c, 1),
                        )

            # === B: グレーゾーン制御ロング ===
            if open_b_long is None and regime == "bullish" and above_sma50:
                if rsi_p <= config.ENTRY_RSI_THRESHOLD and rsi_c > config.ENTRY_RSI_THRESHOLD and price > bo:
                    if dk not in sup_cache:
                        sup_cache[dk] = calc_support_for_date(dfd, ts)
                    nearest = _find_nearest_level(price, sup_cache[dk], config.ENTRY_PROXIMITY_THRESHOLD, "support")
                    if nearest:
                        adj_sl = sl_dist * (1 - sl_tighten)
                        open_b_long = BTPosition(
                            symbol=sym, side="long", strategy="long",
                            entry_price=price, entry_time=ts, qty=qty,
                            take_profit_price=round(price * (1 + tp_pct), 2),
                            stop_loss_price=round(price - adj_sl, 2),
                            highest_price=price, level_name=nearest[0],
                            atr_pct=round((atr_v / price) * 100, 2), rsi_at_entry=round(rsi_c, 1),
                        )

            # === B: レジームショート（bearish、またはgray+VWAP下抜け） ===
            short_ok = False
            if regime == "bearish":
                short_ok = True
            elif regime == "gray" and config.QQQ_WEAK_BEAR_LOW <= ratio <= config.QQQ_WEAK_BEAR_HIGH:
                if cur_vwap > 0 and price < cur_vwap:
                    short_ok = True

            if open_b_short is None and short_ok:
                if rsi_p >= config.SHORT_RSI_THRESHOLD and rsi_c < config.SHORT_RSI_THRESHOLD and price < bo:
                    # 改善4: 確認バー（直前バーも陰線）
                    prev_bearish = True
                    if config.SHORT_CONFIRM_PREV_BEARISH and i >= 2:
                        prev_o = float(o5.iloc[i - 1])
                        prev_c = float(c5.iloc[i - 1])
                        prev_bearish = prev_c < prev_o
                    if not prev_bearish:
                        continue
                    if dk not in res_cache:
                        res_cache[dk] = calc_resistance_for_date(dfd, ts, price)
                    nearest = _find_nearest_level(price, res_cache[dk], config.SHORT_PROXIMITY_THRESHOLD, "resistance")
                    if nearest:
                        adj_sl = sl_dist * (1 - sl_tighten)
                        open_b_short = BTPosition(
                            symbol=sym, side="short", strategy="short",
                            entry_price=price, entry_time=ts, qty=qty,
                            take_profit_price=round(price * (1 - tp_pct), 2),
                            stop_loss_price=round(price + adj_sl, 2),
                            lowest_price=price, level_name=nearest[0],
                            atr_pct=round((atr_v / price) * 100, 2), rsi_at_entry=round(rsi_c, 1),
                        )

        # 未決済
        if not df5.empty:
            lp = float(c5.iloc[-1])
            lt = df5.index[-1]
            for pos, tl in [(open_a, trades_a_long), (open_b_long, trades_b_long), (open_b_short, trades_b_short)]:
                if pos is not None and not pos.closed:
                    pos.closed = True
                    pos.close_price = lp
                    pos.close_time = lt
                    pos.close_reason = "backtest_end"
                    pos.pnl = pos.calc_pnl(lp)
                    tl.append(pos)

    # --- QQQ VWAP / RSI 事前計算（カナリア警戒モード用） ---
    qqq_vwap_series = calc_running_vwap(qqq_5min)
    qqq_rsi_series = ta.rsi(qqq_close, length=14)

    # --- カナリア戦略 ---
    canary_targets = [s for s in canary_symbols if s in sym_data]
    # 改善3: カナリアエントリーカットオフ時刻
    canary_cutoff_t = (
        datetime.combine(datetime.now().date(), market_close)
        - timedelta(minutes=config.CANARY_ENTRY_CUTOFF_MINUTES)
    ).time()

    for sym in canary_targets:
        df5 = sym_data[sym]["5min"]
        c5 = df5["close"].astype(float)
        h5 = df5["high"].astype(float)
        l5 = df5["low"].astype(float)
        o5 = df5["open"].astype(float)
        rsi5 = ta.rsi(c5, length=14)
        vwap5 = calc_running_vwap(df5)
        d5_dates = df5.index.normalize()

        day_open = {}
        for d in d5_dates.unique():
            dd_data = df5[d5_dates == d]
            if not dd_data.empty:
                day_open[d] = float(dd_data["open"].astype(float).iloc[0])

        open_canary = None
        for i in range(20, len(df5)):
            ts = df5.index[i]
            price = float(c5.iloc[i])
            bh = float(h5.iloc[i])
            bl = float(l5.iloc[i])
            ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
            bt = ts_et.time()
            if bt < market_open or bt >= market_close or ts_et.weekday() >= 5:
                continue

            cur_vwap = float(vwap5.iloc[i]) if not pd.isna(vwap5.iloc[i]) else 0.0
            eod_t = (datetime.combine(ts_et.date(), market_close, tzinfo=ET) - timedelta(minutes=eod_min)).time()

            # EOD
            if bt >= eod_t and open_canary and not open_canary.closed:
                open_canary.closed = True
                open_canary.close_price = price
                open_canary.close_time = ts
                open_canary.close_reason = "end_of_day"
                open_canary.pnl = open_canary.calc_pnl(price)
                trades_b_canary.append(open_canary)
                open_canary = None
                continue

            # 監視
            if open_canary and not open_canary.closed:
                open_canary.update_trailing(bh, bl)
                reason, ep = open_canary.check_exit(bh, bl, cur_vwap)
                if reason:
                    open_canary.closed = True
                    open_canary.close_price = ep
                    open_canary.close_time = ts
                    open_canary.close_reason = reason
                    open_canary.pnl = open_canary.calc_pnl(ep)
                    trades_b_canary.append(open_canary)
                    open_canary = None

            # エントリー
            if bt < entry_start or bt >= dtime(15, 30):
                continue
            # 改善3: カナリアエントリーカットオフ（引けN分前は禁止）
            if bt >= canary_cutoff_t:
                continue
            if open_canary is not None:
                continue
            # 改善5: カナリア同時ポジション上限（バックテストでは銘柄横断の合計）
            canary_open_count = sum(1 for t in trades_b_canary if not t.closed)
            if open_canary is not None:
                canary_open_count += 1
            if canary_open_count >= config.CANARY_MAX_POSITIONS:
                continue
            if pd.isna(rsi5.iloc[i]):
                continue
            rsi_c = float(rsi5.iloc[i])
            rsi_p = float(rsi5.iloc[i - 1]) if i > 0 and not pd.isna(rsi5.iloc[i - 1]) else None
            if rsi_p is None:
                continue

            # QQQ 警戒モード判定（緩和版）
            prior_qqq = qqq_close[qqq_close.index <= ts]
            if len(prior_qqq) < 2:
                continue
            qqq_cur = float(prior_qqq.iloc[-1])
            qqq_prev_bar = float(prior_qqq.iloc[-2])
            qqq_5min_neg = qqq_cur < qqq_prev_bar

            # 条件A: 直近5分マイナス AND QQQが自身のVWAPを0.5%以上下回る
            cond_a = False
            prior_qqq_vwap = qqq_vwap_series[qqq_vwap_series.index <= ts]
            if not prior_qqq_vwap.empty and not pd.isna(prior_qqq_vwap.iloc[-1]):
                qqq_vwap_val = float(prior_qqq_vwap.iloc[-1])
                if qqq_vwap_val > 0:
                    qqq_vwap_dev = (qqq_cur - qqq_vwap_val) / qqq_vwap_val
                    cond_a = qqq_5min_neg and qqq_vwap_dev <= -0.005

            # 条件B: QQQ RSI(14) < 40
            cond_b = False
            prior_qqq_rsi = qqq_rsi_series[qqq_rsi_series.index <= ts]
            if not prior_qqq_rsi.empty and not pd.isna(prior_qqq_rsi.iloc[-1]):
                qqq_rsi_val = float(prior_qqq_rsi.iloc[-1])
                cond_b = qqq_rsi_val < 40

            alert_mode = cond_a or cond_b

            if not alert_mode:
                continue

            # 条件1: VWAP下抜け
            if cur_vwap <= 0 or price >= cur_vwap:
                continue

            # 条件2: 相対的弱さ
            ts_day = ts.normalize()
            qqq_open_today = qqq_day_open_map.get(ts_day)
            stock_open_today = day_open.get(ts_day)
            if qqq_open_today is None or stock_open_today is None:
                continue
            qqq_ret = (qqq_cur - qqq_open_today) / qqq_open_today
            stock_ret = (price - stock_open_today) / stock_open_today
            if qqq_ret >= 0 or stock_ret >= qqq_ret:
                continue

            # 改善1: 最小乖離フィルター（QQQとの乖離が小さすぎる銘柄を除外）
            divergence = stock_ret - qqq_ret
            if abs(divergence) < config.CANARY_MIN_DIVERGENCE:
                continue

            # 条件3: RSI 60→50 急落（直近N本以内にRSI>=60、現在<50）
            if rsi_c >= config.CANARY_RSI_ENTRY_LOW:
                continue

            # 改善2: RSI下限ガード（売られすぎの銘柄は反発リスクが高い）
            if rsi_c < config.CANARY_RSI_FLOOR:
                continue

            lookback_start = max(0, i - config.CANARY_RSI_LOOKBACK)
            rsi_window = rsi5.iloc[lookback_start:i].dropna()
            if rsi_window.empty or float(rsi_window.max()) < config.CANARY_RSI_ENTRY_HIGH:
                continue

            # カナリア発火ログ（個別銘柄 vs QQQ 乖離）
            alert_reason = "VWAP-0.5%" if cond_a else "RSI<40"
            qqq_rsi_at_entry = float(prior_qqq_rsi.iloc[-1]) if not prior_qqq_rsi.empty and not pd.isna(prior_qqq_rsi.iloc[-1]) else 0.0
            log.info(
                f"[Canary] {sym} 発火 {ts_et.strftime('%m/%d %H:%M')} "
                f"alert={alert_reason} | "
                f"銘柄ret={stock_ret*100:+.2f}% vs QQQ={qqq_ret*100:+.2f}% "
                f"乖離={divergence*100:+.2f}pp | "
                f"RSI={rsi_c:.0f} (window_max={float(rsi_window.max()):.0f}) | "
                f"VWAP=${cur_vwap:.2f} 価格=${price:.2f} | "
                f"QQQ_RSI={qqq_rsi_at_entry:.1f}"
            )

            qty = max(1, math.floor(config.POSITION_SIZE / price))
            open_canary = BTPosition(
                symbol=sym, side="short", strategy="canary",
                entry_price=price, entry_time=ts, qty=qty,
                take_profit_price=round(price * (1 - config.CANARY_TAKE_PROFIT_PCT), 2),
                stop_loss_price=round(price * (1 + config.CANARY_STOP_LOSS_PCT), 2),
                lowest_price=price, level_name="canary:vwap",
                rsi_at_entry=round(rsi_c, 1), vwap_at_entry=round(cur_vwap, 2),
            )

        if open_canary and not open_canary.closed:
            lp = float(c5.iloc[-1])
            open_canary.closed = True
            open_canary.close_price = lp
            open_canary.close_time = df5.index[-1]
            open_canary.close_reason = "backtest_end"
            open_canary.pnl = open_canary.calc_pnl(lp)
            trades_b_canary.append(open_canary)

    # レジーム日次集計
    regime_daily = {}
    for ts, r in ratio_series.items():
        ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
        day = ts_et.strftime("%m/%d")
        if day not in regime_daily:
            regime_daily[day] = []
        regime_daily[day].append(float(r))

    return {
        "a_long": trades_a_long,
        "b_long": trades_b_long,
        "b_short": trades_b_short,
        "b_canary": trades_b_canary,
        "regime_daily": regime_daily,
    }


def _find_nearest_level(price, levels, threshold, direction):
    nearest = None
    nearest_dist = float("inf")
    for name, level in levels.items():
        if level <= 0:
            continue
        if direction == "support" and level > price * 1.005:
            continue
        if direction == "resistance" and level < price * 0.995:
            continue
        dist = abs(price - level) / level
        if dist <= threshold and dist < nearest_dist:
            nearest_dist = dist
            nearest = (name, level)
    return nearest


# ============================================================
#  結果表示
# ============================================================
REASON_JP = {
    "take_profit": "利確", "stop_loss": "損切り", "trailing_stop": "トレーリング",
    "vwap_cross": "VWAP上抜け", "end_of_day": "EOD決済", "backtest_end": "BT終了",
}


def print_trades(trades, label):
    print(f"\n{'=' * 105}")
    print(f"  {label}")
    print(f"{'=' * 105}")
    if not trades:
        print("  取引なし")
        print("=" * 105)
        return

    print(
        f"{'#':>3} {'銘柄':<5} {'戦略':<7} {'日時':>12} {'Entry':>8} {'Exit':>8} "
        f"{'PnL':>8} {'%':>6} {'RSI':>5} {'理由':<10} {'レベル':<12}"
    )
    print("-" * 105)

    for idx, t in enumerate(trades, 1):
        pnl_pct = (t.entry_price - t.close_price) / t.entry_price * 100 if t.side == "short" else (t.close_price - t.entry_price) / t.entry_price * 100
        et = t.entry_time.astimezone(ET) if t.entry_time.tzinfo else t.entry_time
        reason = REASON_JP.get(t.close_reason, t.close_reason)
        print(
            f"{idx:>3} {t.symbol:<5} {t.strategy:<7} {et.strftime('%m/%d %H:%M'):>12} "
            f"${t.entry_price:>7.2f} ${t.close_price:>7.2f} "
            f"${t.pnl:>+7.2f} {pnl_pct:>+5.1f}% "
            f"{t.rsi_at_entry:>4.1f} {reason:<10} {t.level_name:<12}"
        )

    total = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl >= 0]
    losses = [t for t in trades if t.pnl < 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    aw = sum(t.pnl for t in wins) / len(wins) if wins else 0
    al = sum(t.pnl for t in losses) / len(losses) if losses else 0
    pf = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float("inf")

    rc = {}
    for t in trades:
        r = REASON_JP.get(t.close_reason, t.close_reason)
        rc[r] = rc.get(r, 0) + 1

    daily_pnl = {}
    for t in trades:
        et = t.entry_time.astimezone(ET) if t.entry_time.tzinfo else t.entry_time
        d = et.strftime("%m/%d")
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl

    print(f"\n{'-' * 105}")
    print(f"  総取引数: {len(trades)}  勝ち/負け: {len(wins)}/{len(losses)}  勝率: {wr:.1f}%")
    print(f"  総損益: ${total:+.2f}  平均利益: ${aw:+.2f}  平均損失: ${al:+.2f}  PF: {pf:.2f}")
    print(f"  決済理由: {rc}")
    if daily_pnl:
        print("  日別損益:")
        for d, p in sorted(daily_pnl.items()):
            n = max(1, int(abs(p) / 2))
            bar = "\033[32m" + "█" * min(n, 40) + "\033[0m" if p >= 0 else "\033[31m" + "█" * min(n, 40) + "\033[0m"
            print(f"    {d}: ${p:>+8.2f}  {bar}")
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="グレーゾーン + カナリア戦略 バックテスト")
    parser.add_argument("--symbols", default="NVDA,TSLA,AAPL,MSFT,AMD,META,GOOGL,AMZN")
    parser.add_argument("--days", type=int, default=10)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    # カナリア対象: symbols + RETAIL_FAVORITES の和集合
    canary_syms = list(set(symbols) | set(config.RETAIL_FAVORITES[:10]))

    print(f"\n  バックテスト: グレーゾーン + カナリア戦略")
    print(f"  銘柄: {', '.join(symbols)}")
    print(f"  カナリア対象: {', '.join(canary_syms[:10])}{'...' if len(canary_syms)>10 else ''}")
    print(f"  期間: 直近{args.days}日")
    print(f"  グレーゾーン: {config.QQQ_GRAY_ZONE_LOW*100:.0f}%〜{config.QQQ_GRAY_ZONE_HIGH*100:.0f}% (SL-{config.GRAY_ZONE_SL_TIGHTEN_PCT*100:.0f}%)")
    print(f"  VWAP条件ショート: QQQ {config.QQQ_WEAK_BEAR_LOW*100:.0f}%〜{config.QQQ_WEAK_BEAR_HIGH*100:.0f}% + VWAP下抜け")
    print(f"  カナリア: RSI {config.CANARY_RSI_ENTRY_HIGH}→{config.CANARY_RSI_ENTRY_LOW} TP={config.CANARY_TAKE_PROFIT_PCT*100}% SL={config.CANARY_STOP_LOSS_PCT*100}%")

    results = run_backtest(symbols, args.days, canary_syms)

    # レジーム表示
    print(f"\n{'=' * 105}")
    print("  QQQ ブリッシュ比率（日別平均）")
    print(f"{'=' * 105}")
    for day, ratios in sorted(results["regime_daily"].items()):
        avg = sum(ratios) / len(ratios) if ratios else 0
        label = "BULLISH" if avg > config.QQQ_GRAY_ZONE_HIGH else ("BEARISH" if avg < config.QQQ_GRAY_ZONE_LOW else "GRAY")
        bar_len = int(avg * 40)
        print(f"    {day}: avg={avg:.0%} [{label:>7}] {'█' * bar_len}{'░' * (40 - bar_len)}")

    print_trades(results["a_long"], "[A] ロング（従来 — レジーム制御なし）")
    print_trades(results["b_long"], "[B-Long] ロング（bullish時のみ + グレーゾーンSL縮小）")
    print_trades(results["b_short"], "[B-Short] ショート（bearish / gray+VWAP下抜け）")
    print_trades(results["b_canary"], "[B-Canary] カナリア戦略（警戒モード + 高ベータ銘柄）")

    # 統合サマリー
    b_all = results["b_long"] + results["b_short"] + results["b_canary"]

    print(f"\n{'=' * 105}")
    print("  戦略比較サマリー")
    print(f"{'=' * 105}")
    for label, trades in [
        ("A: ロング（従来）", results["a_long"]),
        ("B-Long: グレーゾーン制御ロング", results["b_long"]),
        ("B-Short: レジームショート", results["b_short"]),
        ("B-Canary: カナリア", results["b_canary"]),
        ("B合計: 新戦略全体", b_all),
    ]:
        n = len(trades)
        pnl = sum(t.pnl for t in trades)
        wr = len([t for t in trades if t.pnl >= 0]) / n * 100 if n else 0
        print(f"  {label:40s}  取引={n:>3}  PnL=${pnl:>+8.2f}  勝率={wr:>5.1f}%")
    print(f"{'=' * 105}")


if __name__ == "__main__":
    main()
