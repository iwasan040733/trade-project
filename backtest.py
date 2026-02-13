"""バックテスト: ATR損切り & RSI 30反発戦略

Alpaca から過去N日分の5分足データを取得し、
現在の自動売買ロジックを適用した場合の損益をシミュレーションする。

2つのモードで比較:
  A) 現行ルール（1h 20MA フィルター付き）
  B) MAフィルター緩和版（日足 50SMA の上のみ）

使い方:
    python backtest.py
    python backtest.py --symbol TSLA --days 60
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
    calc_psychological_levels,
    calc_dynamic_take_profit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ============================================================
#  バックテスト用データクラス
# ============================================================

@dataclass
class BTPosition:
    """バックテスト用ポジション。"""
    symbol: str
    entry_price: float
    entry_time: pd.Timestamp
    qty: int
    take_profit_price: float
    stop_loss_price: float
    highest_price: float
    trailing_stop_price: float = 0.0
    trailing_activated: bool = False
    support_name: str = ""
    atr_pct: float = 0.0

    closed: bool = False
    close_price: float = 0.0
    close_time: pd.Timestamp | None = None
    close_reason: str = ""
    pnl: float = 0.0

    def update_trailing_stop(self, price: float) -> None:
        if price > self.highest_price:
            self.highest_price = price
        if not self.trailing_activated:
            gain_pct = (price - self.entry_price) / self.entry_price
            if gain_pct >= config.TRAILING_ACTIVATE_PCT:
                self.trailing_activated = True
        if self.trailing_activated:
            new_stop = self.highest_price * (1 - config.TRAILING_RETURN_PCT)
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop

    def check_exit(self, price: float) -> str | None:
        if price >= self.take_profit_price:
            return "take_profit"
        if price <= self.stop_loss_price:
            return "stop_loss"
        if self.trailing_stop_price > 0 and price <= self.trailing_stop_price:
            return "trailing_stop"
        return None


# ============================================================
#  データ取得
# ============================================================

def fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    days: int,
) -> pd.DataFrame:
    """指定期間のバーデータを取得する。"""
    start = datetime.now(timezone.utc) - timedelta(days=days + 5)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    return df


# ============================================================
#  サポートレベル計算（日ごとに更新）
# ============================================================

def calc_support_levels_for_date(
    daily_df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict:
    """指定日の前日までのデータからサポートレベルを算出する。"""
    prev_data = daily_df[daily_df.index.normalize() < target_date.normalize()]
    if prev_data.empty:
        return {}

    levels = {}

    # ピボットポイント（前日OHLCから）
    prev_bar = prev_data.iloc[-1]
    pivots = calc_pivot_points(prev_bar)
    levels.update(pivots)

    # 心理的節目
    last_close = float(prev_bar["close"])
    psych = calc_psychological_levels(last_close)
    levels.update(psych)

    # 50日 SMA
    close = prev_data["close"].astype(float)
    if len(close) >= 50:
        levels["sma50"] = round(float(close.rolling(50).mean().iloc[-1]), 4)

    # 前々日の安値（追加サポート）
    if len(prev_data) >= 2:
        levels["prev2_low"] = round(float(prev_data.iloc[-2]["low"]), 4)

    return levels


# ============================================================
#  メインバックテストロジック
# ============================================================

def run_backtest(
    symbol: str,
    days: int,
    ma_filter: str = "daily_sma50",
) -> list[BTPosition]:
    """バックテストを実行する。

    Args:
        symbol: 銘柄シンボル
        days: バックテスト期間（日数）
        ma_filter: MAフィルター ("1h_20ma", "daily_sma50", "none")
    """
    log.info(f"=== バックテスト: {symbol} 過去{days}日 [MAフィルター={ma_filter}] ===")

    client = StockHistoricalDataClient(
        config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
    )

    # --- データ取得 ---
    log.info("データ取得中...")
    df_5min = fetch_bars(client, symbol, TimeFrame(5, TimeFrameUnit.Minute), days)
    df_daily = fetch_bars(client, symbol, TimeFrame.Day, days + 200)
    df_1h = fetch_bars(client, symbol, TimeFrame.Hour, days)
    log.info(
        f"  5分足={len(df_5min)}本  日足={len(df_daily)}本  1h足={len(df_1h)}本  "
        f"期間: {df_5min.index[0].strftime('%Y-%m-%d')} ~ {df_5min.index[-1].strftime('%Y-%m-%d')}"
    )

    # --- 指標の事前計算 ---
    close_5m = df_5min["close"].astype(float)
    high_5m = df_5min["high"].astype(float)
    low_5m = df_5min["low"].astype(float)
    open_5m = df_5min["open"].astype(float)

    rsi_5m = ta.rsi(close_5m, length=14)
    atr_5m = ta.atr(high_5m, low_5m, close_5m, length=14)

    # 1時間足 20MA
    close_1h = df_1h["close"].astype(float)
    ma_1h_20 = close_1h.rolling(20).mean()

    # 日足 50SMA
    close_daily = df_daily["close"].astype(float)
    sma_daily_50 = close_daily.rolling(50).mean()

    # --- バックテストループ ---
    all_trades: list[BTPosition] = []
    open_position: BTPosition | None = None
    support_cache: dict[str, dict] = {}

    market_open = dtime(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    market_close = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    eod_minutes = config.EOD_CLOSE_MINUTES_BEFORE

    stats = {"rsi_cross": 0, "bullish": 0, "ma_pass": 0, "support_pass": 0, "entry": 0}

    for i in range(20, len(df_5min)):
        ts = df_5min.index[i]
        price = float(close_5m.iloc[i])

        ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
        bar_time = ts_et.time()

        if bar_time < market_open or bar_time >= market_close:
            continue
        if ts_et.weekday() >= 5:
            continue

        # === ポジション保有中 → エグジット判定 ===
        if open_position is not None:
            eod_threshold = (
                datetime.combine(ts_et.date(), market_close, tzinfo=ET)
                - timedelta(minutes=eod_minutes)
            ).time()

            if bar_time >= eod_threshold:
                open_position.closed = True
                open_position.close_price = price
                open_position.close_time = ts
                open_position.close_reason = "end_of_day"
                open_position.pnl = (price - open_position.entry_price) * open_position.qty
                all_trades.append(open_position)
                open_position = None
                continue

            # 5分足の high/low で SL/TP をチェック（バー内部での到達を考慮）
            bar_high = float(high_5m.iloc[i])
            bar_low = float(low_5m.iloc[i])

            open_position.update_trailing_stop(bar_high)

            # SL は low でチェック、TP は high でチェック
            if bar_low <= open_position.stop_loss_price:
                exit_price = open_position.stop_loss_price
                exit_reason = "stop_loss"
            elif bar_high >= open_position.take_profit_price:
                exit_price = open_position.take_profit_price
                exit_reason = "take_profit"
            elif open_position.trailing_stop_price > 0 and bar_low <= open_position.trailing_stop_price:
                exit_price = open_position.trailing_stop_price
                exit_reason = "trailing_stop"
            else:
                continue

            open_position.closed = True
            open_position.close_price = exit_price
            open_position.close_time = ts
            open_position.close_reason = exit_reason
            open_position.pnl = (exit_price - open_position.entry_price) * open_position.qty
            all_trades.append(open_position)
            open_position = None
            continue

        # === エントリー判定 ===
        if bar_time >= dtime(15, 30):
            continue

        if pd.isna(rsi_5m.iloc[i]) or pd.isna(atr_5m.iloc[i]):
            continue
        rsi_cur = float(rsi_5m.iloc[i])
        rsi_prev = float(rsi_5m.iloc[i - 1]) if not pd.isna(rsi_5m.iloc[i - 1]) else None
        atr_val = float(atr_5m.iloc[i])

        if rsi_prev is None:
            continue

        # 条件1: RSI 30 クロスオーバー
        if not (rsi_prev <= config.ENTRY_RSI_THRESHOLD and rsi_cur > config.ENTRY_RSI_THRESHOLD):
            continue
        stats["rsi_cross"] += 1

        # 条件2: 陽線確認
        if float(close_5m.iloc[i]) <= float(open_5m.iloc[i]):
            continue
        stats["bullish"] += 1

        # 条件3: MAフィルター
        ma_ok = False
        if ma_filter == "1h_20ma":
            prior = ma_1h_20[ma_1h_20.index <= ts].dropna()
            if not prior.empty:
                ma_ok = price > float(prior.iloc[-1])
        elif ma_filter == "daily_sma50":
            prior = sma_daily_50[sma_daily_50.index.normalize() <= ts.normalize()].dropna()
            if not prior.empty:
                ma_ok = price > float(prior.iloc[-1])
        else:
            ma_ok = True

        if not ma_ok:
            continue
        stats["ma_pass"] += 1

        # 条件4: サポート接近（0.5%以内）
        date_key = ts_et.strftime("%Y-%m-%d")
        if date_key not in support_cache:
            support_cache[date_key] = calc_support_levels_for_date(df_daily, ts)
        supports = support_cache[date_key]

        nearest_support = None
        nearest_dist = float("inf")
        for name, level in supports.items():
            if level <= 0 or level > price * 1.005:
                continue
            dist = abs(price - level) / level
            if dist <= config.ENTRY_PROXIMITY_THRESHOLD and dist < nearest_dist:
                nearest_dist = dist
                nearest_support = (name, level)

        if nearest_support is None:
            continue
        stats["support_pass"] += 1
        stats["entry"] += 1

        # === エントリー ===
        qty = max(1, math.floor(config.POSITION_SIZE / price))

        prior_daily = df_daily[df_daily.index.normalize() < ts.normalize()]
        atr_daily = calc_atr_3day(prior_daily) if len(prior_daily) >= 3 else None
        if atr_daily is not None:
            tp_pct = calc_dynamic_take_profit(
                atr_daily, price,
                tp_min=config.TAKE_PROFIT_MIN, tp_max=config.TAKE_PROFIT_MAX,
            )
        else:
            tp_pct = config.TAKE_PROFIT_MIN

        tp_price = price * (1 + tp_pct)

        max_stop = price * config.STOP_LOSS_MAX_PCT
        sl_distance = min(atr_val * config.STOP_LOSS_ATR_MULT, max_stop)
        sl_price = price - sl_distance
        atr_pct = (atr_val / price) * 100

        open_position = BTPosition(
            symbol=symbol,
            entry_price=price,
            entry_time=ts,
            qty=qty,
            take_profit_price=round(tp_price, 2),
            stop_loss_price=round(sl_price, 2),
            highest_price=price,
            support_name=nearest_support[0],
            atr_pct=round(atr_pct, 2),
        )

        log.info(
            f"  ENTRY {ts_et.strftime('%m/%d %H:%M')} "
            f"${price:.2f} RSI={rsi_cur:.1f} "
            f"sup={nearest_support[0]}(${nearest_support[1]:.2f}) "
            f"SL=${sl_price:.2f}(-{sl_distance/price*100:.1f}%) "
            f"TP=${tp_price:.2f}(+{tp_pct*100:.1f}%)"
        )

    # 未決済ポジションの強制決済
    if open_position is not None:
        last_price = float(df_5min["close"].iloc[-1])
        open_position.closed = True
        open_position.close_price = last_price
        open_position.close_time = df_5min.index[-1]
        open_position.close_reason = "backtest_end"
        open_position.pnl = (last_price - open_position.entry_price) * open_position.qty
        all_trades.append(open_position)

    log.info(
        f"  条件通過数: RSIクロス={stats['rsi_cross']} → 陽線={stats['bullish']} "
        f"→ MA={stats['ma_pass']} → サポート={stats['support_pass']}"
    )

    return all_trades


# ============================================================
#  結果表示
# ============================================================

REASON_JP = {
    "take_profit": "利確",
    "stop_loss": "損切り",
    "trailing_stop": "トレーリング",
    "end_of_day": "EOD決済",
    "backtest_end": "BT終了",
}


def print_results(trades: list[BTPosition], symbol: str, label: str = "") -> None:
    """バックテスト結果を表示する。"""
    header = f"  {label}" if label else f"  {symbol}  ATR損切り & RSI30反発戦略"
    print("\n" + "=" * 90)
    print(header)
    print("=" * 90)

    if not trades:
        print("\n  取引なし\n" + "=" * 90)
        return

    # 個別トレード一覧
    print(
        f"\n{'#':>3} {'日時':>12} {'Entry':>8} {'Exit':>8} "
        f"{'PnL':>8} {'%':>6} {'SL幅':>6} {'理由':<10} {'ATR%':>5} {'サポート':<10}"
    )
    print("-" * 90)

    for idx, t in enumerate(trades, 1):
        pnl_pct = (t.close_price - t.entry_price) / t.entry_price * 100
        sl_pct = (t.entry_price - t.stop_loss_price) / t.entry_price * 100
        entry_et = t.entry_time.astimezone(ET) if t.entry_time.tzinfo else t.entry_time
        reason = REASON_JP.get(t.close_reason, t.close_reason)

        print(
            f"{idx:>3} {entry_et.strftime('%m/%d %H:%M'):>12} "
            f"${t.entry_price:>7.2f} ${t.close_price:>7.2f} "
            f"${t.pnl:>+7.2f} {pnl_pct:>+5.1f}% "
            f"{sl_pct:>5.1f}% {reason:<10} "
            f"{t.atr_pct:>4.1f}% {t.support_name:<10}"
        )

    # サマリー
    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl >= 0]
    losses = [t for t in trades if t.pnl < 0]
    win_rate = len(wins) / len(trades) * 100

    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    profit_factor = (
        abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses))
        if losses and sum(t.pnl for t in losses) != 0
        else float("inf")
    )

    # 最大ドローダウン
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t.pnl
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    # 理由別集計
    reason_counts: dict[str, int] = {}
    for t in trades:
        r = REASON_JP.get(t.close_reason, t.close_reason)
        reason_counts[r] = reason_counts.get(r, 0) + 1

    print("\n" + "-" * 90)
    print(f"  総取引数:      {len(trades)}")
    print(f"  勝ち / 負け:   {len(wins)} / {len(losses)}")
    print(f"  勝率:          {win_rate:.1f}%")
    print(f"  総損益:        ${total_pnl:+.2f}")
    print(f"  平均利益:      ${avg_win:+.2f}  平均損失: ${avg_loss:+.2f}")
    print(f"  PF:            {profit_factor:.2f}")
    print(f"  最大DD:        ${max_dd:.2f}")
    print(f"  決済理由:      {reason_counts}")

    # 日別損益
    daily_pnl: dict[str, float] = {}
    for t in trades:
        entry_et = t.entry_time.astimezone(ET) if t.entry_time.tzinfo else t.entry_time
        day = entry_et.strftime("%m/%d")
        daily_pnl[day] = daily_pnl.get(day, 0) + t.pnl

    if daily_pnl:
        print("\n  日別損益:")
        for day, pnl in sorted(daily_pnl.items()):
            n = max(1, int(abs(pnl) / 1))
            bar = "\033[32m" + "█" * n + "\033[0m" if pnl >= 0 else "\033[31m" + "█" * n + "\033[0m"
            print(f"    {day}: ${pnl:>+8.2f}  {bar}")

    print("=" * 90)


# ============================================================
#  メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ATR損切り & RSI30反発 バックテスト")
    parser.add_argument("--symbol", default="NVDA", help="銘柄 (default: NVDA)")
    parser.add_argument("--days", type=int, default=30, help="期間 (default: 30)")
    args = parser.parse_args()

    print(f"\n  パラメータ: SL=5分ATR×{config.STOP_LOSS_ATR_MULT} (cap {config.STOP_LOSS_MAX_PCT*100}%)"
          f"  TP={config.TAKE_PROFIT_MIN*100}~{config.TAKE_PROFIT_MAX*100}%"
          f"  RSI閾値={config.ENTRY_RSI_THRESHOLD}"
          f"  サポート近接={config.ENTRY_PROXIMITY_THRESHOLD*100}%"
          f"  ポジションサイズ=${config.POSITION_SIZE}")

    # --- モードA: 1h 20MA フィルター（現行ルール） ---
    trades_a = run_backtest(args.symbol, args.days, ma_filter="1h_20ma")
    print_results(trades_a, args.symbol, f"[A] 現行ルール（1h 20MAフィルター） — {args.symbol}")

    # --- モードB: 日足 50SMA フィルター（緩和版） ---
    trades_b = run_backtest(args.symbol, args.days, ma_filter="daily_sma50")
    print_results(trades_b, args.symbol, f"[B] 日足50SMAフィルター — {args.symbol}")

    # --- モードC: MAフィルターなし（参考） ---
    trades_c = run_backtest(args.symbol, args.days, ma_filter="none")
    print_results(trades_c, args.symbol, f"[C] MAフィルターなし（参考） — {args.symbol}")

    # --- 比較サマリー ---
    print("\n" + "=" * 90)
    print("  モード比較")
    print("=" * 90)
    for label, trades in [("A: 1h 20MA", trades_a), ("B: 日足50SMA", trades_b), ("C: フィルターなし", trades_c)]:
        n = len(trades)
        pnl = sum(t.pnl for t in trades)
        wr = len([t for t in trades if t.pnl >= 0]) / n * 100 if n else 0
        print(f"  {label:20s}  取引={n:>3}  PnL=${pnl:>+8.2f}  勝率={wr:>5.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
