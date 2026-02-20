"""バックテスト: ブレイクアウト・スナイパー型（ロング＋ショート / TP+トレーリング）

低頻度・厚利のブレイクアウト/ブレイクダウン戦略:
  ロング: 始値 + 前日レンジ × K 上抜け → ADX>30 +DI>-DI → QQQ bullish
  ショート: 始値 - 前日レンジ × K 下抜け → ADX>30 -DI>+DI → QQQ bearish
  共通: Volume Spike + ATR拡大
  出口: 固定TP ATR×3.0 / SL ATR×1.5 / トレーリング ATR×5.0

使い方:
    python backtest_short.py --symbols COIN,MARA,MSTR --days 365
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
from indicators import calc_qqq_bullish_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


# ============================================================
#  PendingOrder（プルバック待ち指値注文）
# ============================================================
@dataclass
class PendingOrder:
    symbol: str
    side: str  # "long" or "short"
    limit_price: float       # 指値（プルバック目標）
    signal_time: pd.Timestamp
    signal_bar_idx: int      # シグナル発生時のバーindex
    breakout_level: float    # ブレイクアウト/ブレイクダウンライン
    atr_value: float
    rsi_at_signal: float
    adx_at_signal: float
    vol_ratio: float
    ema20_val: float
    level_name: str = ""

    def is_expired(self, current_bar_idx: int) -> bool:
        return (current_bar_idx - self.signal_bar_idx) > config.BREAKOUT_PULLBACK_TIMEOUT_BARS

    def is_filled(self, bar_high: float, bar_low: float) -> bool:
        if self.side == "long":
            return bar_low <= self.limit_price  # 安値が指値以下 → 約定
        else:
            return bar_high >= self.limit_price  # 高値が指値以上 → 約定


# ============================================================
#  BTPosition（ロング・ショート / TP+トレーリング）
# ============================================================
@dataclass
class BTPosition:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: pd.Timestamp
    qty: int
    stop_loss_price: float
    take_profit_price: float = 0.0  # 固定TP（全利確）
    extreme_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_activated: bool = False
    level_name: str = ""
    atr_value: float = 0.0
    rsi_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    vol_ratio: float = 0.0

    closed: bool = False
    close_price: float = 0.0
    close_time: pd.Timestamp | None = None
    close_reason: str = ""
    pnl: float = 0.0

    def update_trailing(self, bar_high: float, bar_low: float) -> None:
        if self.atr_value <= 0:
            return
        if self.side == "long":
            if bar_high > self.extreme_price:
                self.extreme_price = bar_high
            if not self.trailing_activated:
                gain = bar_high - self.entry_price
                if gain >= self.atr_value * config.TRAILING_ACTIVATE_ATR_MULT:
                    self.trailing_activated = True
            if self.trailing_activated:
                ns = self.extreme_price - self.atr_value * config.BREAKOUT_TRAILING_ATR_MULT
                if ns > self.trailing_stop_price:
                    self.trailing_stop_price = ns
        else:
            if self.extreme_price == 0 or bar_low < self.extreme_price:
                self.extreme_price = bar_low
            if not self.trailing_activated:
                gain = self.entry_price - bar_low
                if gain >= self.atr_value * config.TRAILING_ACTIVATE_ATR_MULT:
                    self.trailing_activated = True
            if self.trailing_activated:
                ns = self.extreme_price + self.atr_value * config.BREAKOUT_TRAILING_ATR_MULT
                if self.trailing_stop_price == 0 or ns < self.trailing_stop_price:
                    self.trailing_stop_price = ns

    def check_exit(self, bh: float, bl: float) -> tuple[str | None, float]:
        if self.side == "long":
            # TP（利確）
            if self.take_profit_price > 0 and bh >= self.take_profit_price:
                return "take_profit", self.take_profit_price
            # SL
            if bl <= self.stop_loss_price:
                return "stop_loss", self.stop_loss_price
            # トレーリング
            if self.trailing_stop_price > 0 and bl <= self.trailing_stop_price:
                return "trailing_stop", self.trailing_stop_price
        else:
            if self.take_profit_price > 0 and bl <= self.take_profit_price:
                return "take_profit", self.take_profit_price
            if bh >= self.stop_loss_price:
                return "stop_loss", self.stop_loss_price
            if self.trailing_stop_price > 0 and bh >= self.trailing_stop_price:
                return "trailing_stop", self.trailing_stop_price
        return None, 0.0

    def calc_pnl(self, exit_price: float) -> float:
        if self.side == "long":
            return (exit_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - exit_price) * self.qty


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


# ============================================================
#  メインバックテスト
# ============================================================
def run_backtest(symbols, days, start_date=None, end_date=None, btc_filter_override=None, legacy_regime=False):
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    btc_enabled = btc_filter_override if btc_filter_override is not None else config.BTC_REGIME_ENABLED

    main_tf = TimeFrame(5, TimeFrameUnit.Minute)
    bars_per_day = 78  # 6.5H=390min / 5min
    log.info(f"タイムフレーム: 5min（1日あたり約{bars_per_day}本）")

    # --- QQQ データ（レジーム判定は常に5分足）---
    log.info("QQQ データ取得中...")
    qqq_5min = fetch_bars(client, "QQQ", TimeFrame(5, TimeFrameUnit.Minute), days)
    log.info(f"QQQ 5分足={len(qqq_5min)}本")
    ratio_series = calc_qqq_bullish_ratio(qqq_5min, window=config.QQQ_BULLISH_RATIO_WINDOW)

    # ハイブリッドレジーム用: サーキットブレイカー
    _qqq_close = qqq_5min["close"].astype(float)
    _cb_change_series = _qqq_close.pct_change(periods=config.QQQ_CB_BARS)

    def _qqq_regime_at(ts) -> tuple[str, float]:
        """バー時刻 ts 時点のQQQレジームを返す。legacy_regime=True でローリング比率のみ使用。"""
        prior = ratio_series.index <= ts
        ratio = float(ratio_series[prior].iloc[-1]) if prior.any() else 0.5

        # Layer 1: サーキットブレイカー
        if not legacy_regime:
            cb_prior = _cb_change_series.index <= ts
            if cb_prior.any():
                cb_change = float(_cb_change_series[cb_prior].iloc[-1])
                if not pd.isna(cb_change):
                    if cb_change <= config.QQQ_CB_THRESHOLD_DOWN:
                        return "bearish", ratio
                    if cb_change >= config.QQQ_CB_THRESHOLD_UP:
                        return "bullish", ratio

        # Layer 2: ローリング比率
        if ratio > config.QQQ_GRAY_ZONE_HIGH:
            return "bullish", ratio
        if ratio < config.QQQ_GRAY_ZONE_LOW:
            return "bearish", ratio
        return "gray", ratio

    # --- BTC地合いフィルター ---
    btc_bullish_dates = set()
    if btc_enabled:
        log.info(f"BTC地合いフィルター: {config.BTC_REGIME_SYMBOL} 日足取得中...")
        try:
            bito_daily = fetch_bars(client, config.BTC_REGIME_SYMBOL, TimeFrame.Day, days + 50)
            bito_close = bito_daily["close"].astype(float)
            bito_sma = bito_close.rolling(config.BTC_REGIME_SMA_PERIOD).mean()
            bullish_count = 0
            bearish_count = 0
            for idx in bito_daily.index:
                d = idx.normalize()
                sma_val = bito_sma.loc[idx]
                close_val = bito_close.loc[idx]
                if not pd.isna(sma_val) and float(close_val) > float(sma_val):
                    btc_bullish_dates.add(d)
                    bullish_count += 1
                else:
                    bearish_count += 1
            log.info(f"  BTC regime: bullish={bullish_count}日 bearish={bearish_count}日")
        except Exception as e:
            log.warning(f"  {config.BTC_REGIME_SYMBOL} 取得失敗（フィルター無効化）: {e}")
            btc_enabled = False

    # --- 銘柄データ取得 ---
    sym_data = {}
    for sym in symbols:
        try:
            d5 = fetch_bars(client, sym, main_tf, days)
            dd = fetch_bars(client, sym, TimeFrame.Day, days + 200)
            if d5.empty:
                continue
            sym_data[sym] = {"5min": d5, "daily": dd}
            log.info(f"  {sym}: 5min={len(d5)} daily={len(dd)}")
        except Exception as e:
            log.warning(f"  {sym} 取得失敗: {e}")

    trades = []
    skipped = {
        "long_no_regime": 0, "long_no_adx": 0, "long_no_vol": 0, "long_no_atr": 0, "long_passed": 0,
        "short_no_regime": 0, "short_no_adx": 0, "short_no_vol": 0, "short_no_atr": 0, "short_passed": 0,
        "long_pullback_filled": 0, "long_pullback_expired": 0,
        "short_pullback_filled": 0, "short_pullback_expired": 0,
    }

    market_open = dtime(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    market_close = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    entry_start = dtime(
        config.MARKET_OPEN_HOUR,
        config.MARKET_OPEN_MINUTE + config.ENTRY_BUFFER_MINUTES_OPEN,
    )

    for sym in symbols:
        if sym not in sym_data:
            continue

        df5 = sym_data[sym]["5min"]
        dfd = sym_data[sym]["daily"]

        c5 = df5["close"].astype(float)
        h5 = df5["high"].astype(float)
        l5 = df5["low"].astype(float)
        o5 = df5["open"].astype(float)
        v5 = df5["volume"].astype(float)
        atr5 = ta.atr(h5, l5, c5, length=14)
        rsi5 = ta.rsi(c5, length=14)
        ema20 = c5.ewm(span=config.BREAKOUT_EMA_PERIOD, adjust=False).mean()

        adx_result = ta.adx(h5, l5, c5, length=config.BREAKOUT_ADX_PERIOD)
        adx_col = f"ADX_{config.BREAKOUT_ADX_PERIOD}"
        dmp_col = f"DMP_{config.BREAKOUT_ADX_PERIOD}"
        dmn_col = f"DMN_{config.BREAKOUT_ADX_PERIOD}"

        d5_dates = df5.index.normalize()
        day_open = {}
        for d in d5_dates.unique():
            dd_data = df5[d5_dates == d]
            if not dd_data.empty:
                day_open[d] = float(dd_data["open"].astype(float).iloc[0])

        open_pos = None
        pending_order = None  # プルバック待ち指値注文
        daily_trade_count = {}
        pullback_enabled = config.BREAKOUT_PULLBACK_ENABLED

        for i in range(max(20, bars_per_day), len(df5)):
            ts = df5.index[i]
            price = float(c5.iloc[i])
            bh = float(h5.iloc[i])
            bl = float(l5.iloc[i])

            ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
            bt = ts_et.time()
            if bt < market_open or bt >= market_close or ts_et.weekday() >= 5:
                continue

            # --- EOD: 指値キャンセル + ポジション決済 (15:55 ET) ---
            if bt >= dtime(15, 55):
                if pending_order is not None:
                    skipped[f"{pending_order.side}_pullback_expired"] += 1
                    pending_order = None
                if open_pos is not None and not open_pos.closed:
                    open_pos.closed = True
                    open_pos.close_price = price
                    open_pos.close_time = ts
                    open_pos.close_reason = "end_of_day"
                    open_pos.pnl = open_pos.calc_pnl(price)
                    trades.append(open_pos)
                    open_pos = None
                continue

            # --- ポジション監視 ---
            if open_pos is not None and not open_pos.closed:
                open_pos.update_trailing(bh, bl)
                reason, ep = open_pos.check_exit(bh, bl)
                if reason:
                    open_pos.closed = True
                    open_pos.close_price = ep
                    open_pos.close_time = ts
                    open_pos.close_reason = reason
                    open_pos.pnl = open_pos.calc_pnl(ep)
                    trades.append(open_pos)
                    open_pos = None

            # --- 指値注文の約定チェック ---
            if pending_order is not None and open_pos is None:
                if pending_order.is_expired(i):
                    skipped[f"{pending_order.side}_pullback_expired"] += 1
                    pending_order = None
                elif pending_order.is_filled(bh, bl):
                    # 約定 → ポジション作成
                    fill_price = pending_order.limit_price
                    dk_day = ts_et.strftime("%Y-%m-%d")
                    daily_trade_count[dk_day] = daily_trade_count.get(dk_day, 0) + 1
                    qty = max(1, math.floor(config.POSITION_SIZE / fill_price))
                    sl_dist = pending_order.atr_value * config.BREAKOUT_STOP_ATR_MULT
                    tp_dist = pending_order.atr_value * config.BREAKOUT_TP_ATR_MULT if config.BREAKOUT_TP_ATR_MULT > 0 else 0

                    if pending_order.side == "long":
                        sl_price = round(fill_price - sl_dist, 2)
                        tp_price = round(fill_price + tp_dist, 2) if tp_dist > 0 else 0.0
                    else:
                        sl_price = round(fill_price + sl_dist, 2)
                        tp_price = round(fill_price - tp_dist, 2) if tp_dist > 0 else 0.0

                    bars_waited = i - pending_order.signal_bar_idx
                    log.info(
                        f"[{pending_order.side.upper()} FILL] {sym} {ts_et.strftime('%m/%d %H:%M')} "
                        f"${fill_price:.2f} (待ち{bars_waited}本) ADX={pending_order.adx_at_signal:.1f} "
                        f"VolR={pending_order.vol_ratio:.2f} ATR=${pending_order.atr_value:.2f}"
                    )

                    skipped[f"{pending_order.side}_pullback_filled"] += 1
                    open_pos = BTPosition(
                        symbol=sym, side=pending_order.side,
                        entry_price=fill_price, entry_time=ts, qty=qty,
                        stop_loss_price=sl_price, take_profit_price=tp_price,
                        extreme_price=fill_price, level_name=pending_order.level_name,
                        atr_value=pending_order.atr_value,
                        rsi_at_entry=pending_order.rsi_at_signal,
                        adx_at_entry=pending_order.adx_at_signal,
                        vol_ratio=pending_order.vol_ratio,
                    )
                    pending_order = None

            # --- エントリー判定（シグナル検出）---
            if open_pos is not None or pending_order is not None:
                continue
            if bt < entry_start or bt >= dtime(15, 30):
                continue
            bar_date = ts_et.date()
            if start_date and bar_date < start_date:
                continue
            if end_date and bar_date > end_date:
                continue
            dk_day = ts_et.strftime("%Y-%m-%d")
            if daily_trade_count.get(dk_day, 0) >= config.BREAKOUT_MAX_TRADES_PER_DAY:
                continue
            if pd.isna(atr5.iloc[i]):
                continue

            atr_v = float(atr5.iloc[i])
            prior_d = dfd[dfd.index.normalize() < ts.normalize()]
            if prior_d.empty:
                continue

            ema20_val = float(ema20.iloc[i]) if not pd.isna(ema20.iloc[i]) else None
            today_open_price = day_open.get(ts.normalize())
            if ema20_val is None or today_open_price is None:
                continue

            prev_day_bar = prior_d.iloc[-1]
            prev_range = float(prev_day_bar["high"]) - float(prev_day_bar["low"])
            breakout_level = today_open_price + prev_range * config.BREAKOUT_K
            breakdown_level = today_open_price - prev_range * config.BREAKOUT_K

            if adx_result is None or pd.isna(adx_result[adx_col].iloc[i]):
                continue
            adx_val = float(adx_result[adx_col].iloc[i])
            di_plus = float(adx_result[dmp_col].iloc[i])
            di_minus = float(adx_result[dmn_col].iloc[i])

            regime, ratio = _qqq_regime_at(ts)
            rsi_c = float(rsi5.iloc[i]) if not pd.isna(rsi5.iloc[i]) else 0.0

            # ===== ロング: ブレイクアウト =====
            entry_side = None
            if price > breakout_level and price > ema20_val:
                if regime != "bullish":
                    skipped["long_no_regime"] += 1
                elif adx_val < config.BREAKOUT_ADX_THRESHOLD or di_plus <= di_minus:
                    skipped["long_no_adx"] += 1
                else:
                    entry_side = "long"

            # ===== ショート: ブレイクダウン =====
            if entry_side is None and price < breakdown_level and price < ema20_val:
                if regime != "bearish":
                    skipped["short_no_regime"] += 1
                elif adx_val < config.BREAKOUT_ADX_THRESHOLD or di_minus <= di_plus:
                    skipped["short_no_adx"] += 1
                else:
                    entry_side = "short"

            if entry_side is None:
                continue

            # === 共通フィルター: Volume Spike ===
            short_window = config.BREAKOUT_VOL_SPIKE_SHORT
            long_window = config.BREAKOUT_VOL_SPIKE_LONG
            if i < long_window:
                skipped[f"{entry_side}_no_vol"] += 1
                continue
            vol_short_avg = float(v5.iloc[i - short_window:i].mean())
            vol_long_avg = float(v5.iloc[i - long_window:i].mean())
            if vol_long_avg <= 0:
                skipped[f"{entry_side}_no_vol"] += 1
                continue
            vol_ratio = vol_short_avg / vol_long_avg
            if vol_ratio < config.BREAKOUT_VOL_SPIKE_MULT:
                skipped[f"{entry_side}_no_vol"] += 1
                continue

            # === 共通フィルター: ATR拡大 ===
            if config.BREAKOUT_ATR_EXPANSION:
                atr_lookback = min(bars_per_day, i)
                atr_day_avg = float(atr5.iloc[i - atr_lookback:i].dropna().mean()) if atr_lookback > 0 else 0
                if atr_day_avg > 0 and atr_v <= atr_day_avg:
                    skipped[f"{entry_side}_no_atr"] += 1
                    continue

            # === 全フィルター通過 → エントリー or 指値予約 ===
            skipped[f"{entry_side}_passed"] += 1

            if entry_side == "long":
                level_label = f"L_bo_{breakout_level:.0f}"
            else:
                level_label = f"S_bd_{breakdown_level:.0f}"

            if pullback_enabled:
                # --- プルバック指値: ブレイクアウトライン + ATR×buffer ---
                buf = atr_v * config.BREAKOUT_PULLBACK_BUFFER_ATR
                if entry_side == "long":
                    limit_price = round(breakout_level + buf, 2)
                else:
                    limit_price = round(breakdown_level - buf, 2)

                log.info(
                    f"[{entry_side.upper()} SIGNAL] {sym} {ts_et.strftime('%m/%d %H:%M')} "
                    f"${price:.2f} → 指値${limit_price:.2f} ADX={adx_val:.1f} "
                    f"VolR={vol_ratio:.2f} ATR=${atr_v:.2f} RSI={rsi_c:.0f}"
                )

                pending_order = PendingOrder(
                    symbol=sym, side=entry_side, limit_price=limit_price,
                    signal_time=ts, signal_bar_idx=i,
                    breakout_level=breakout_level if entry_side == "long" else breakdown_level,
                    atr_value=atr_v, rsi_at_signal=round(rsi_c, 1),
                    adx_at_signal=round(adx_val, 1), vol_ratio=round(vol_ratio, 2),
                    ema20_val=ema20_val, level_name=level_label,
                )
            else:
                # --- 成行エントリー（従来） ---
                daily_trade_count[dk_day] = daily_trade_count.get(dk_day, 0) + 1
                qty = max(1, math.floor(config.POSITION_SIZE / price))
                sl_dist = atr_v * config.BREAKOUT_STOP_ATR_MULT
                tp_dist = atr_v * config.BREAKOUT_TP_ATR_MULT if config.BREAKOUT_TP_ATR_MULT > 0 else 0

                if entry_side == "long":
                    sl_price = round(price - sl_dist, 2)
                    tp_price = round(price + tp_dist, 2) if tp_dist > 0 else 0.0
                else:
                    sl_price = round(price + sl_dist, 2)
                    tp_price = round(price - tp_dist, 2) if tp_dist > 0 else 0.0

                log.info(
                    f"[{entry_side.upper()}] {sym} {ts_et.strftime('%m/%d %H:%M')} "
                    f"${price:.2f} ADX={adx_val:.1f} +DI={di_plus:.1f} -DI={di_minus:.1f} "
                    f"VolRatio={vol_ratio:.2f} ATR=${atr_v:.2f} RSI={rsi_c:.0f}"
                )

                open_pos = BTPosition(
                    symbol=sym, side=entry_side, entry_price=price, entry_time=ts, qty=qty,
                    stop_loss_price=sl_price, take_profit_price=tp_price,
                    extreme_price=price, level_name=level_label, atr_value=atr_v,
                    rsi_at_entry=round(rsi_c, 1), adx_at_entry=round(adx_val, 1),
                    vol_ratio=round(vol_ratio, 2),
                )

        # 未決済
        if open_pos is not None and not open_pos.closed:
            lp = float(c5.iloc[-1])
            open_pos.closed = True
            open_pos.close_price = lp
            open_pos.close_time = df5.index[-1]
            open_pos.close_reason = "backtest_end"
            open_pos.pnl = open_pos.calc_pnl(lp)
            trades.append(open_pos)

    # レジーム日次集計
    regime_daily = {}
    for ts, r in ratio_series.items():
        ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
        day = ts_et.strftime("%m/%d")
        if day not in regime_daily:
            regime_daily[day] = []
        regime_daily[day].append(float(r))

    return {"trades": trades, "regime_daily": regime_daily, "skipped": skipped}


# ============================================================
#  結果表示
# ============================================================
REASON_JP = {
    "stop_loss": "損切り", "trailing_stop": "トレーリング",
    "take_profit": "利確", "end_of_day": "EOD決済", "backtest_end": "BT終了",
}


def print_trades(trades):
    print(f"\n{'=' * 130}")
    print(f"  ブレイクアウト・スナイパー（全取引）")
    print(f"{'=' * 130}")
    if not trades:
        print("  取引なし")
        print("=" * 130)
        return

    print(
        f"{'#':>3} {'L/S':<4} {'銘柄':<5} {'日時':>12} {'Entry':>8} {'Exit':>8} "
        f"{'PnL':>9} {'%':>6} {'ADX':>5} {'VolR':>5} {'RSI':>5} {'理由':<10} {'レベル':<14}"
    )
    print("-" * 130)

    for idx, t in enumerate(trades, 1):
        if t.side == "long":
            pnl_pct = (t.close_price - t.entry_price) / t.entry_price * 100
        else:
            pnl_pct = (t.entry_price - t.close_price) / t.entry_price * 100
        et = t.entry_time.astimezone(ET) if t.entry_time.tzinfo else t.entry_time
        reason = REASON_JP.get(t.close_reason, t.close_reason)
        side_label = "L" if t.side == "long" else "S"
        print(
            f"{idx:>3} {side_label:<4} {t.symbol:<5} {et.strftime('%m/%d %H:%M'):>12} "
            f"${t.entry_price:>7.2f} ${t.close_price:>7.2f} "
            f"${t.pnl:>+8.2f} {pnl_pct:>+5.1f}% "
            f"{t.adx_at_entry:>4.1f} {t.vol_ratio:>4.2f} "
            f"{t.rsi_at_entry:>4.1f} {reason:<10} {t.level_name:<14}"
        )

    # --- サイド別サマリー ---
    for side_label, side_name in [("long", "ロング"), ("short", "ショート"), (None, "合計")]:
        if side_label:
            st = [t for t in trades if t.side == side_label]
        else:
            st = trades
        if not st:
            continue

        total = sum(t.pnl for t in st)
        wins = [t for t in st if t.pnl >= 0]
        losses = [t for t in st if t.pnl < 0]
        n = len(st)
        wr = len(wins) / n * 100 if n else 0
        aw = sum(t.pnl for t in wins) / len(wins) if wins else 0
        al = sum(t.pnl for t in losses) / len(losses) if losses else 0
        pf = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float("inf")
        rr = abs(aw / al) if al != 0 else float("inf")
        avg_pnl = total / n if n else 0

        # 決済理由の内訳
        rc = {}
        for t in st:
            r = REASON_JP.get(t.close_reason, t.close_reason)
            rc[r] = rc.get(r, 0) + 1

        print(f"\n  --- {side_name} ---")
        print(f"  取引数: {n}  勝ち/負け: {len(wins)}/{len(losses)}  勝率: {wr:.1f}%")
        print(f"  総損益: ${total:+.2f}  平均利益: ${aw:+.2f}  平均損失: ${al:+.2f}  PF: {pf:.2f}  R:R: 1:{rr:.2f}")
        print(f"  ★ 平均利益/取引: ${avg_pnl:+.2f}  決済: {rc}")

    # 日別損益
    daily_pnl = {}
    for t in trades:
        et = t.entry_time.astimezone(ET) if t.entry_time.tzinfo else t.entry_time
        d = et.strftime("%m/%d")
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl

    if daily_pnl:
        print("\n  日別損益:")
        for d, p in sorted(daily_pnl.items()):
            n_bar = max(1, int(abs(p) / 2))
            bar = "\033[32m" + "█" * min(n_bar, 40) + "\033[0m" if p >= 0 else "\033[31m" + "█" * min(n_bar, 40) + "\033[0m"
            print(f"    {d}: ${p:>+8.2f}  {bar}")
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="ブレイクアウト・スナイパー バックテスト")
    parser.add_argument("--symbols", default="COIN")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--start-date", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument("--btc-filter", type=str, default=None, choices=["on", "off"], help="BTC地合いフィルター on/off")
    parser.add_argument("--legacy-regime", action="store_true", help="QQQレジームをローリング比率のみで判定（CB無効）")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    from datetime import date as date_type
    start_date = date_type.fromisoformat(args.start_date) if args.start_date else None
    end_date = date_type.fromisoformat(args.end_date) if args.end_date else None
    btc_override = None
    if args.btc_filter == "on":
        btc_override = True
    elif args.btc_filter == "off":
        btc_override = False

    btc_label = "ON" if (btc_override if btc_override is not None else config.BTC_REGIME_ENABLED) else "OFF"
    period_label = f"{args.start_date or '(全期間)'} ~ {args.end_date or '(全期間)'}"

    print(f"\n  ブレイクアウト・スナイパー バックテスト（TP+トレーリング）")
    print(f"  銘柄: {', '.join(symbols)}  タイムフレーム: 5min")
    print(f"  データ: 直近{args.days}日  エントリー期間: {period_label}")
    print(f"  ロング: 始値+前日レンジ×{config.BREAKOUT_K} 上抜け + EMA{config.BREAKOUT_EMA_PERIOD}上")
    print(f"  ショート: 始値-前日レンジ×{config.BREAKOUT_K} 下抜け + EMA{config.BREAKOUT_EMA_PERIOD}下")
    print(f"  ADXフィルター: ADX>{config.BREAKOUT_ADX_THRESHOLD} (期間{config.BREAKOUT_ADX_PERIOD})")
    print(f"  出来高スパイク: 直近{config.BREAKOUT_VOL_SPIKE_SHORT}本/{config.BREAKOUT_VOL_SPIKE_LONG}本 >= {config.BREAKOUT_VOL_SPIKE_MULT}x")
    print(f"  ATR拡大: {config.BREAKOUT_ATR_EXPANSION}")
    print(f"  SL=ATR×{config.BREAKOUT_STOP_ATR_MULT}  TP=ATR×{config.BREAKOUT_TP_ATR_MULT}  トレーリング=ATR×{config.BREAKOUT_TRAILING_ATR_MULT}  EOD=15:55")
    print(f"  R:R設計 = 1:{config.BREAKOUT_TP_ATR_MULT/config.BREAKOUT_STOP_ATR_MULT:.1f} （SL{config.BREAKOUT_STOP_ATR_MULT}:TP{config.BREAKOUT_TP_ATR_MULT}）")
    pullback_label = "ON" if config.BREAKOUT_PULLBACK_ENABLED else "OFF"
    print(f"  プルバック指値: {pullback_label}" + (
        f"（バッファ=ATR×{config.BREAKOUT_PULLBACK_BUFFER_ATR}, タイムアウト={config.BREAKOUT_PULLBACK_TIMEOUT_BARS}本）"
        if config.BREAKOUT_PULLBACK_ENABLED else ""
    ))
    print(f"  BTC地合いフィルター: {btc_label}")

    results = run_backtest(symbols, args.days, start_date=start_date, end_date=end_date, btc_filter_override=btc_override, legacy_regime=args.legacy_regime)

    # フィルター統計
    sk = results["skipped"]
    long_total = sk["long_no_regime"] + sk["long_no_adx"] + sk["long_no_vol"] + sk["long_no_atr"] + sk["long_passed"]
    short_total = sk["short_no_regime"] + sk["short_no_adx"] + sk["short_no_vol"] + sk["short_no_atr"] + sk["short_passed"]

    print(f"\n  フィルター統計:")
    if long_total > 0:
        print(f"  【ロング候補】 計{long_total}")
        print(f"    QQQ非ブル:  {sk['long_no_regime']:>5}  ADX不適合: {sk['long_no_adx']:>5}  出来高不足: {sk['long_no_vol']:>5}  ATR縮小: {sk['long_no_atr']:>5}  → シグナル: {sk['long_passed']:>5}")
    if short_total > 0:
        print(f"  【ショート候補】 計{short_total}")
        print(f"    QQQ非ベア:  {sk['short_no_regime']:>5}  ADX不適合: {sk['short_no_adx']:>5}  出来高不足: {sk['short_no_vol']:>5}  ATR縮小: {sk['short_no_atr']:>5}  → シグナル: {sk['short_passed']:>5}")
    if config.BREAKOUT_PULLBACK_ENABLED:
        print(f"  【プルバック指値】")
        print(f"    ロング:  約定{sk['long_pullback_filled']:>3} / 期限切れ{sk['long_pullback_expired']:>3}")
        print(f"    ショート: 約定{sk['short_pullback_filled']:>3} / 期限切れ{sk['short_pullback_expired']:>3}")

    print_trades(results["trades"])


if __name__ == "__main__":
    main()
