"""動的銘柄追加バックテスト

固定銘柄（SNIPER_SYMBOLS）を常時監視しつつ、毎営業日 9:30〜10:25 ET の
ドル出来高上位かつ中小型株（価格フィルター）から動的に銘柄を追加監視する。

使い方:
    python backtest_dynamic.py --days 365
    python backtest_dynamic.py --dynamic-top-n 5 --dynamic-max-price 80 --start-date 2025-01-01
    python backtest_dynamic.py --dynamic-top-n 3 --days 180
"""

import argparse
import logging
import math
from datetime import datetime, timedelta, time as dtime, date as date_type
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from indicators import calc_qqq_bullish_ratio
from backtest_short import (
    BTPosition, PendingOrder,
    fetch_bars, fetch_vix_daily,
    print_trades,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# 動的追加の候補ユニバース（中小型・ボラタイル）
# 固定SNIPER_SYMBOLSと重複があってもOK（その場合は常時監視になるだけ）
DEFAULT_DYNAMIC_UNIVERSE = [
    # クリプトマイニング
    "MARA", "MSTR", "CLSK", "WULF", "BTBT", "IREN", "BITF",
    # AI・量子
    "SOUN", "IONQ", "RGTI", "QBTS", "BBAI", "ARQQ",
    # 宇宙・新興
    "ASTS", "RKLB", "LUNR", "ACHR",
    # EV・輸送
    "RIVN", "LCID", "NIO", "GOEV",
    # フィンテック
    "HOOD", "SOFI", "AFRM", "UPST",
    # ゲーム・SNS
    "RBLX", "SNAP",
    # その他中小型
    "SMCI", "RIOT", "PLTR", "APP",
]

DEFAULT_DYNAMIC_TOP_N = 5       # 毎日追加する最大銘柄数
DEFAULT_DYNAMIC_MAX_PRICE = 100.0  # 中小型フィルター: 価格 < $100


# ============================================================
#  メインバックテスト
# ============================================================
def run_dynamic_backtest(
    fixed_symbols,
    dynamic_universe,
    dynamic_top_n,
    dynamic_max_price,
    days,
    start_date=None,
    end_date=None,
    vix_filter_override=None,
):
    """
    固定銘柄 + 動的追加銘柄の組み合わせバックテスト。
    動的追加: 毎日 9:30〜10:25 ET のドル出来高上位 dynamic_top_n 件（価格 < dynamic_max_price）
    """
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    vix_filter_enabled = vix_filter_override if vix_filter_override is not None else config.VIX_FILTER_ENABLED

    main_tf = TimeFrame(5, TimeFrameUnit.Minute)
    bars_per_day = 78

    # --- QQQ レジーム ---
    log.info("QQQ データ取得中...")
    qqq_5min = fetch_bars(client, "QQQ", TimeFrame(5, TimeFrameUnit.Minute), days)
    log.info(f"QQQ 5分足: {len(qqq_5min)} 本")
    ratio_series = calc_qqq_bullish_ratio(qqq_5min, window=config.QQQ_BULLISH_RATIO_WINDOW)
    _qqq_close = qqq_5min["close"].astype(float)
    _cb_change = _qqq_close.pct_change(periods=config.QQQ_CB_BARS)

    def _qqq_regime_at(ts) -> tuple[str, float]:
        prior = ratio_series.index <= ts
        ratio = float(ratio_series[prior].iloc[-1]) if prior.any() else 0.5
        cb_prior = _cb_change.index <= ts
        if cb_prior.any():
            cb_val = float(_cb_change[cb_prior].iloc[-1])
            if not pd.isna(cb_val):
                if cb_val <= config.QQQ_CB_THRESHOLD_DOWN:
                    return "bearish", ratio
                if cb_val >= config.QQQ_CB_THRESHOLD_UP:
                    return "bullish", ratio
        if ratio > config.QQQ_GRAY_ZONE_HIGH:
            return "bullish", ratio
        if ratio < config.QQQ_GRAY_ZONE_LOW:
            return "bearish", ratio
        return "gray", ratio

    # --- VIX ---
    vix_prev_map: dict = {}
    if vix_filter_enabled:
        log.info(f"VIXフィルター: 前日VIX < {config.VIX_FILTER_THRESHOLD} でスキップ")
        vix_prev_map = fetch_vix_daily(days)
        if not vix_prev_map:
            log.warning("VIX 取得失敗 → フィルター無効化")
            vix_filter_enabled = False

    # --- 全シンボル取得（固定 ∪ 動的候補）---
    all_symbols = list(dict.fromkeys(list(fixed_symbols) + list(dynamic_universe)))
    log.info(f"データ取得: 固定{len(fixed_symbols)}銘柄 + 動的候補{len(dynamic_universe)}銘柄 = 計{len(all_symbols)}銘柄")

    sym_data: dict = {}
    for sym in all_symbols:
        try:
            d5 = fetch_bars(client, sym, main_tf, days)
            dd = fetch_bars(client, sym, TimeFrame.Day, days + 200)
            if d5.empty:
                log.warning(f"  {sym}: 5分足データなし → スキップ")
                continue
            sym_data[sym] = {"5min": d5, "daily": dd}
            log.info(f"  {sym}: 5min={len(d5)} daily={len(dd)}")
        except Exception as e:
            log.warning(f"  {sym} 取得失敗: {e}")

    if not sym_data:
        log.error("有効な銘柄データがありません")
        return {"trades": [], "daily_added": {}, "skipped": {}, "vix_filter_enabled": False, "vix_prev_map": {}}

    # --- インジケーター計算 ---
    adx_col = f"ADX_{config.BREAKOUT_ADX_PERIOD}"
    dmp_col = f"DMP_{config.BREAKOUT_ADX_PERIOD}"
    dmn_col = f"DMN_{config.BREAKOUT_ADX_PERIOD}"

    sym_indicators: dict = {}
    for sym, data in sym_data.items():
        df5 = data["5min"]
        c5 = df5["close"].astype(float)
        h5 = df5["high"].astype(float)
        l5 = df5["low"].astype(float)
        v5 = df5["volume"].astype(float)
        atr5 = ta.atr(h5, l5, c5, length=14)
        rsi5 = ta.rsi(c5, length=14)
        ema20 = c5.ewm(span=config.BREAKOUT_EMA_PERIOD, adjust=False).mean()
        adx_result = ta.adx(h5, l5, c5, length=config.BREAKOUT_ADX_PERIOD)

        d5_dates = df5.index.normalize()
        day_open: dict = {}
        for d in d5_dates.unique():
            sub = df5[d5_dates == d]
            if not sub.empty:
                day_open[d] = float(sub["open"].astype(float).iloc[0])

        sym_indicators[sym] = {
            "c5": c5, "h5": h5, "l5": l5, "v5": v5,
            "atr5": atr5, "rsi5": rsi5, "ema20": ema20,
            "adx_result": adx_result, "day_open": day_open,
        }

    # --- 時間定数 ---
    market_open = dtime(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    market_close = dtime(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    from datetime import datetime as _dt
    _open_dt = _dt(2000, 1, 1, config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    entry_start = (_open_dt + timedelta(minutes=config.ENTRY_BUFFER_MINUTES_OPEN)).time()

    # --- 銘柄ごとに 日付 → バー一覧 を事前インデックス化 ---
    sym_date_bars: dict[str, dict] = {}
    for sym, data in sym_data.items():
        df5 = data["5min"]
        sym_date_bars[sym] = {}
        for idx, ts in enumerate(df5.index):
            ts_et = ts.astimezone(ET) if ts.tzinfo else ts.tz_localize("UTC").astimezone(ET)
            if ts_et.weekday() >= 5:
                continue
            d = ts_et.date()
            sym_date_bars[sym].setdefault(d, []).append((ts_et, idx))

    # --- 日次プレセッション（9:30〜10:25 ET）のドル出来高と平均価格を事前計算 ---
    pre_session_data: dict[str, dict] = {}  # sym -> {date: {"dv": float, "avg_price": float}}
    for sym in sym_data:
        v5 = sym_indicators[sym]["v5"]
        c5 = sym_indicators[sym]["c5"]
        pre_session_data[sym] = {}
        for d, bars in sym_date_bars[sym].items():
            dv = 0.0
            price_sum = 0.0
            count = 0
            for ts_et, bar_idx in bars:
                bt = ts_et.time()
                if bt < market_open or bt >= entry_start:
                    continue
                close_v = float(c5.iloc[bar_idx])
                vol_v = float(v5.iloc[bar_idx])
                dv += vol_v * close_v
                price_sum += close_v
                count += 1
            if count > 0:
                pre_session_data[sym][d] = {"dv": dv, "avg_price": price_sum / count}

    # --- 全取引日を収集 ---
    all_dates: list = sorted({d for sym in sym_date_bars for d in sym_date_bars[sym]})

    # 固定銘柄セット（データ取得成功分のみ）
    fixed_set = set(s for s in fixed_symbols if s in sym_data)
    dynamic_candidates = [s for s in dynamic_universe if s in sym_data and s not in fixed_set]

    # --- 状態変数 ---
    open_positions: dict[str, BTPosition] = {}
    daily_side_done: dict[str, dict] = {}
    daily_added: dict[str, list] = {}  # 動的追加された銘柄の記録

    trades: list = []
    skipped = {
        "long_no_regime": 0, "long_no_adx": 0, "long_no_vol": 0, "long_no_atr": 0,
        "long_vix_filter": 0, "long_daily_limit": 0, "long_passed": 0,
        "short_no_regime": 0, "short_no_adx": 0, "short_no_vol": 0, "short_no_atr": 0,
        "short_vix_filter": 0, "short_daily_limit": 0, "short_passed": 0,
        "long_pullback_filled": 0, "long_pullback_expired": 0,
        "short_pullback_filled": 0, "short_pullback_expired": 0,
        "dynamic_added_total": 0,
    }

    # ============================================================
    #  メインループ（日次）
    # ============================================================
    for trade_date in all_dates:
        if start_date and trade_date < start_date:
            continue
        if end_date and trade_date > end_date:
            continue

        dk_day = trade_date.strftime("%Y-%m-%d")

        # --- 動的銘柄選定: 価格フィルター + ドル出来高ランキング ---
        dv_candidates = {}
        for sym in dynamic_candidates:
            ps = pre_session_data[sym].get(trade_date)
            if ps is None:
                continue
            if ps["avg_price"] > dynamic_max_price:
                continue  # 中小型株フィルター: 高額株は除外
            if ps["dv"] <= 0:
                continue
            dv_candidates[sym] = ps["dv"]

        sorted_cands = sorted(dv_candidates.items(), key=lambda x: x[1], reverse=True)
        dynamic_added_today = [s for s, _ in sorted_cands[:dynamic_top_n]]
        daily_added[dk_day] = dynamic_added_today
        skipped["dynamic_added_total"] += len(dynamic_added_today)

        if dynamic_added_today:
            dv_strs = [f"{s}(${dv_candidates[s]/1e6:.1f}M)" for s in dynamic_added_today]
            log.debug(f"{dk_day}: 動的追加: {', '.join(dv_strs)}")

        # --- 今日の監視銘柄 = 固定 ∪ 動的追加 ∪ オープンポジション保有 ---
        syms_today = fixed_set | set(dynamic_added_today) | set(open_positions.keys())

        # --- 時間順バーリスト ---
        all_bars_today: list = []
        for sym in syms_today:
            if sym not in sym_date_bars or trade_date not in sym_date_bars[sym]:
                continue
            for ts_et, bar_idx in sym_date_bars[sym][trade_date]:
                all_bars_today.append((ts_et, sym, bar_idx))
        all_bars_today.sort(key=lambda x: (x[0], x[1]))

        pending_orders: dict[str, PendingOrder] = {}

        # ============================================================
        #  バーループ
        # ============================================================
        for ts_et, sym, i in all_bars_today:
            bt = ts_et.time()
            if bt < market_open or bt >= market_close:
                continue

            ind = sym_indicators[sym]
            c5 = ind["c5"]
            h5 = ind["h5"]
            l5 = ind["l5"]
            v5 = ind["v5"]
            atr5 = ind["atr5"]
            rsi5 = ind["rsi5"]
            ema20 = ind["ema20"]
            adx_result = ind["adx_result"]
            day_open = ind["day_open"]

            ts = sym_data[sym]["5min"].index[i]
            price = float(c5.iloc[i])
            bh = float(h5.iloc[i])
            bl = float(l5.iloc[i])

            # --- EOD 15:55 ---
            if bt >= dtime(15, 55):
                if sym in pending_orders:
                    skipped[f"{pending_orders[sym].side}_pullback_expired"] += 1
                    del pending_orders[sym]
                if sym in open_positions:
                    pos = open_positions.pop(sym)
                    pos.closed = True
                    pos.close_price = price
                    pos.close_time = ts
                    pos.close_reason = "end_of_day"
                    pos.pnl = pos.calc_pnl(price)
                    trades.append(pos)
                continue

            # --- ポジション監視 ---
            if sym in open_positions:
                pos = open_positions[sym]
                pos.update_trailing(bh, bl)
                reason, ep = pos.check_exit(bh, bl)
                if reason:
                    pos.closed = True
                    pos.close_price = ep
                    pos.close_time = ts
                    pos.close_reason = reason
                    pos.pnl = pos.calc_pnl(ep)
                    trades.append(pos)
                    del open_positions[sym]

            # --- ペンディング注文約定チェック ---
            if sym in pending_orders and sym not in open_positions:
                pend = pending_orders[sym]
                if pend.is_expired(i):
                    skipped[f"{pend.side}_pullback_expired"] += 1
                    del pending_orders[sym]
                elif pend.is_filled(bh, bl):
                    fp = pend.limit_price
                    daily_side_done.setdefault(sym, {}).setdefault(dk_day, set()).add(pend.side)
                    qty = max(1, math.floor(config.POSITION_SIZE / fp))
                    sl_dist = pend.atr_value * config.BREAKOUT_STOP_ATR_MULT
                    tp_dist = pend.atr_value * config.BREAKOUT_TP_ATR_MULT if config.BREAKOUT_TP_ATR_MULT > 0 else 0.0
                    if pend.side == "long":
                        sl_price = round(fp - sl_dist, 2)
                        tp_price = round(fp + tp_dist, 2) if tp_dist > 0 else 0.0
                    else:
                        sl_price = round(fp + sl_dist, 2)
                        tp_price = round(fp - tp_dist, 2) if tp_dist > 0 else 0.0
                    open_positions[sym] = BTPosition(
                        symbol=sym, side=pend.side,
                        entry_price=fp, entry_time=ts, qty=qty,
                        stop_loss_price=sl_price, take_profit_price=tp_price,
                        extreme_price=fp, level_name=pend.level_name,
                        atr_value=pend.atr_value, rsi_at_entry=pend.rsi_at_signal,
                        adx_at_entry=pend.adx_at_signal, vol_ratio=pend.vol_ratio,
                    )
                    skipped[f"{pend.side}_pullback_filled"] += 1
                    del pending_orders[sym]

            # --- エントリー判定 ---
            if sym in open_positions or sym in pending_orders:
                continue
            if bt < entry_start or bt >= dtime(15, 30):
                continue

            bar_date = ts_et.date()
            if start_date and bar_date < start_date:
                continue
            if end_date and bar_date > end_date:
                continue

            sym_day_done = daily_side_done.get(sym, {}).get(dk_day, set())
            if sym_day_done >= {"long", "short"}:
                continue

            if pd.isna(atr5.iloc[i]):
                continue

            atr_v = float(atr5.iloc[i])
            dfd = sym_data[sym]["daily"]
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

            entry_side = None
            if price > breakout_level and price > ema20_val:
                if regime != "bullish":
                    skipped["long_no_regime"] += 1
                elif adx_val < config.BREAKOUT_ADX_THRESHOLD or di_plus <= di_minus:
                    skipped["long_no_adx"] += 1
                else:
                    entry_side = "long"

            if entry_side is None and price < breakdown_level and price < ema20_val:
                if regime != "bearish":
                    skipped["short_no_regime"] += 1
                elif adx_val < config.BREAKOUT_ADX_THRESHOLD or di_minus <= di_plus:
                    skipped["short_no_adx"] += 1
                else:
                    entry_side = "short"

            if entry_side is None:
                continue

            if entry_side in sym_day_done:
                skipped[f"{entry_side}_daily_limit"] += 1
                continue

            if vix_filter_enabled and sym in config.VIX_FILTER_SYMBOLS and vix_prev_map:
                vix_val = vix_prev_map.get(bar_date)
                if vix_val is not None and vix_val < config.VIX_FILTER_THRESHOLD:
                    skipped[f"{entry_side}_vix_filter"] += 1
                    continue

            short_w = config.BREAKOUT_VOL_SPIKE_SHORT
            long_w = config.BREAKOUT_VOL_SPIKE_LONG
            if i < long_w:
                skipped[f"{entry_side}_no_vol"] += 1
                continue
            vol_short_avg = float(v5.iloc[i - short_w:i].mean())
            vol_long_avg = float(v5.iloc[i - long_w:i].mean())
            if vol_long_avg <= 0:
                skipped[f"{entry_side}_no_vol"] += 1
                continue
            vol_ratio = vol_short_avg / vol_long_avg
            if vol_ratio < config.BREAKOUT_VOL_SPIKE_MULT:
                skipped[f"{entry_side}_no_vol"] += 1
                continue

            if config.BREAKOUT_ATR_EXPANSION:
                atr_lookback = min(bars_per_day, i)
                atr_day_avg = float(atr5.iloc[i - atr_lookback:i].dropna().mean()) if atr_lookback > 0 else 0
                if atr_day_avg > 0 and atr_v <= atr_day_avg:
                    skipped[f"{entry_side}_no_atr"] += 1
                    continue

            skipped[f"{entry_side}_passed"] += 1
            level_label = (
                f"L_bo_{breakout_level:.0f}" if entry_side == "long"
                else f"S_bd_{breakdown_level:.0f}"
            )
            is_dynamic = sym in set(dynamic_added_today)
            log.info(
                f"[{entry_side.upper()}{'*' if is_dynamic else ''}] {sym} {ts_et.strftime('%m/%d %H:%M')} "
                f"${price:.2f} ADX={adx_val:.1f} +DI={di_plus:.1f} -DI={di_minus:.1f} "
                f"VolR={vol_ratio:.2f} ATR=${atr_v:.2f} RSI={rsi_c:.0f}"
                + (" ← 動的追加" if is_dynamic else "")
            )

            if config.BREAKOUT_PULLBACK_ENABLED:
                buf = atr_v * config.BREAKOUT_PULLBACK_BUFFER_ATR
                lp = (
                    round(breakout_level + buf, 2) if entry_side == "long"
                    else round(breakdown_level - buf, 2)
                )
                pending_orders[sym] = PendingOrder(
                    symbol=sym, side=entry_side, limit_price=lp,
                    signal_time=ts, signal_bar_idx=i,
                    breakout_level=breakout_level if entry_side == "long" else breakdown_level,
                    atr_value=atr_v, rsi_at_signal=round(rsi_c, 1),
                    adx_at_signal=round(adx_val, 1), vol_ratio=round(vol_ratio, 2),
                    ema20_val=ema20_val, level_name=level_label,
                )
            else:
                daily_side_done.setdefault(sym, {}).setdefault(dk_day, set()).add(entry_side)
                qty = max(1, math.floor(config.POSITION_SIZE / price))
                sl_dist = atr_v * config.BREAKOUT_STOP_ATR_MULT
                tp_dist = atr_v * config.BREAKOUT_TP_ATR_MULT if config.BREAKOUT_TP_ATR_MULT > 0 else 0.0
                if entry_side == "long":
                    sl_price = round(price - sl_dist, 2)
                    tp_price = round(price + tp_dist, 2) if tp_dist > 0 else 0.0
                else:
                    sl_price = round(price + sl_dist, 2)
                    tp_price = round(price - tp_dist, 2) if tp_dist > 0 else 0.0

                open_positions[sym] = BTPosition(
                    symbol=sym, side=entry_side, entry_price=price, entry_time=ts, qty=qty,
                    stop_loss_price=sl_price, take_profit_price=tp_price,
                    extreme_price=price, level_name=level_label, atr_value=atr_v,
                    rsi_at_entry=round(rsi_c, 1), adx_at_entry=round(adx_val, 1),
                    vol_ratio=round(vol_ratio, 2),
                )

    # --- バックテスト終了: 未決済ポジション時価決済 ---
    for sym, pos in list(open_positions.items()):
        df5 = sym_data[sym]["5min"]
        c5 = sym_indicators[sym]["c5"]
        lp = float(c5.iloc[-1])
        pos.closed = True
        pos.close_price = lp
        pos.close_time = df5.index[-1]
        pos.close_reason = "backtest_end"
        pos.pnl = pos.calc_pnl(lp)
        trades.append(pos)

    return {
        "trades": trades,
        "daily_added": daily_added,
        "skipped": skipped,
        "vix_filter_enabled": vix_filter_enabled,
        "vix_prev_map": vix_prev_map,
        "fixed_set": fixed_set,
    }


# ============================================================
#  結果表示
# ============================================================
def print_dynamic_summary(daily_added: dict, trades: list, fixed_set: set):
    """動的追加銘柄がどれだけトレードに貢献したかサマリー表示"""
    print(f"\n{'=' * 80}")
    print("  動的追加銘柄サマリー（中小型・出来高フィルター）")
    print(f"{'=' * 80}")

    # 動的追加された全銘柄のカウント
    added_count: dict[str, int] = {}
    for syms in daily_added.values():
        for s in syms:
            added_count[s] = added_count.get(s, 0) + 1

    # 動的銘柄からのトレード抽出
    dynamic_trades = [t for t in trades if t.symbol not in fixed_set]
    fixed_trades = [t for t in trades if t.symbol in fixed_set]

    print(f"  固定銘柄: {len(fixed_set)}銘柄  →  取引数: {len(fixed_trades)}")
    print(f"  動的追加: 計{sum(added_count.values())}回（ユニーク{len(added_count)}銘柄）→  取引数: {len(dynamic_trades)}")

    if added_count:
        top_added = sorted(added_count.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  追加頻度 top10: {', '.join(f'{s}({n}日)' for s, n in top_added)}")

    if dynamic_trades:
        total_pnl = sum(t.pnl for t in dynamic_trades)
        wins = [t for t in dynamic_trades if t.pnl >= 0]
        losses = [t for t in dynamic_trades if t.pnl < 0]
        pf = (
            abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses))
            if losses and sum(t.pnl for t in losses) != 0
            else float("inf")
        )
        print(f"  動的銘柄 損益: ${total_pnl:+.2f}  勝率: {len(wins)}/{len(dynamic_trades)}  PF: {pf:.2f}")
        print(f"  動的銘柄 取引: " + ", ".join(
            f"{'L' if t.side == 'long' else 'S'}{t.symbol}${t.pnl:+.0f}"
            for t in dynamic_trades
        ))

    print("=" * 80)


# ============================================================
#  エントリーポイント
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="固定+動的銘柄追加 バックテスト")
    parser.add_argument(
        "--fixed", type=str, default=None,
        help="固定監視銘柄（カンマ区切り、省略時はSNIPER_SYMBOLS）",
    )
    parser.add_argument(
        "--dynamic-universe", type=str, default=None,
        help="動的追加候補銘柄（カンマ区切り、省略時はデフォルト候補）",
    )
    parser.add_argument(
        "--dynamic-top-n", type=int, default=DEFAULT_DYNAMIC_TOP_N,
        help=f"毎日追加する最大銘柄数（デフォルト: {DEFAULT_DYNAMIC_TOP_N}）",
    )
    parser.add_argument(
        "--dynamic-max-price", type=float, default=DEFAULT_DYNAMIC_MAX_PRICE,
        help=f"中小型フィルター: 価格 < この値の銘柄のみ追加（デフォルト: ${DEFAULT_DYNAMIC_MAX_PRICE}）",
    )
    parser.add_argument("--days", type=int, default=365, help="データ取得日数（デフォルト: 365）")
    parser.add_argument("--start-date", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument(
        "--vix-filter", type=str, default=None, choices=["on", "off"],
        help="VIXフィルター on/off",
    )
    args = parser.parse_args()

    fixed_symbols = (
        [s.strip() for s in args.fixed.split(",")]
        if args.fixed
        else list(config.SNIPER_SYMBOLS)
    )
    dynamic_universe = (
        [s.strip() for s in args.dynamic_universe.split(",")]
        if args.dynamic_universe
        else DEFAULT_DYNAMIC_UNIVERSE
    )
    start_date = date_type.fromisoformat(args.start_date) if args.start_date else None
    end_date = date_type.fromisoformat(args.end_date) if args.end_date else None
    vix_override = {"on": True, "off": False}.get(args.vix_filter)

    vix_label = "ON" if (vix_override if vix_override is not None else config.VIX_FILTER_ENABLED) else "OFF"
    period_label = f"{args.start_date or '(全期間)'} ~ {args.end_date or '(全期間)'}"

    print(f"\n  固定+動的銘柄追加 バックテスト（スナイパー型）")
    print(f"  固定監視: {len(fixed_symbols)} 銘柄（常時）: {fixed_symbols}")
    print(f"  動的候補: {len(dynamic_universe)} 銘柄  → 毎日上位 {args.dynamic_top_n} 件追加（価格 < ${args.dynamic_max_price}）")
    print(f"  選定方法: 9:30〜10:25 ET ドル出来高ランキング × 中小型フィルター")
    print(f"  データ: 直近{args.days}日  |  期間: {period_label}")
    print(f"  ADX>{config.BREAKOUT_ADX_THRESHOLD}  VolSpike≥{config.BREAKOUT_VOL_SPIKE_MULT}x  ATR拡大: {config.BREAKOUT_ATR_EXPANSION}")
    print(f"  SL=ATR×{config.BREAKOUT_STOP_ATR_MULT}  Trail=ATR×{config.BREAKOUT_TRAILING_ATR_MULT}  EOD=15:55 ET")
    print(f"  VIXフィルター: {vix_label}")

    results = run_dynamic_backtest(
        fixed_symbols=fixed_symbols,
        dynamic_universe=dynamic_universe,
        dynamic_top_n=args.dynamic_top_n,
        dynamic_max_price=args.dynamic_max_price,
        days=args.days,
        start_date=start_date,
        end_date=end_date,
        vix_filter_override=vix_override,
    )

    sk = results["skipped"]
    vix_active = results.get("vix_filter_enabled", False)

    long_total = (
        sk["long_no_regime"] + sk["long_no_adx"] + sk["long_no_vol"]
        + sk["long_no_atr"] + sk["long_vix_filter"] + sk["long_passed"]
    )
    short_total = (
        sk["short_no_regime"] + sk["short_no_adx"] + sk["short_no_vol"]
        + sk["short_no_atr"] + sk["short_vix_filter"] + sk["short_passed"]
    )

    print(f"\n  フィルター統計:")
    print(f"  動的追加 延べ件数: {sk['dynamic_added_total']}")
    if long_total > 0:
        vix_str = f"  VIXスキップ: {sk['long_vix_filter']:>5}" if vix_active else ""
        print(f"  【ロング候補】 計{long_total}")
        print(
            f"    QQQ非ブル: {sk['long_no_regime']:>5}  ADX不適合: {sk['long_no_adx']:>5}"
            f"  出来高不足: {sk['long_no_vol']:>5}  ATR縮小: {sk['long_no_atr']:>5}{vix_str}"
            f"  → シグナル: {sk['long_passed']:>5}"
        )
    if short_total > 0:
        vix_str = f"  VIXスキップ: {sk['short_vix_filter']:>5}" if vix_active else ""
        print(f"  【ショート候補】 計{short_total}")
        print(
            f"    QQQ非ベア: {sk['short_no_regime']:>5}  ADX不適合: {sk['short_no_adx']:>5}"
            f"  出来高不足: {sk['short_no_vol']:>5}  ATR縮小: {sk['short_no_atr']:>5}{vix_str}"
            f"  → シグナル: {sk['short_passed']:>5}"
        )

    print_dynamic_summary(results["daily_added"], results["trades"], results["fixed_set"])
    print_trades(results["trades"])


if __name__ == "__main__":
    main()
