"""テクニカル指標の計算とスコアリング"""

from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

import config


def calc_pivot_points(daily_bar: pd.Series) -> dict:
    """前日の OHLC からピボットポイントを算出する。

    Args:
        daily_bar: high, low, close カラムを持つ Series（前日足）

    Returns:
        dict: pivot, s1, s2, prev_low
    """
    h, l, c = float(daily_bar["high"]), float(daily_bar["low"]), float(daily_bar["close"])
    pivot = (h + l + c) / 3
    return {
        "pivot": round(pivot, 4),
        "s1": round(2 * pivot - h, 4),
        "s2": round(pivot - (h - l), 4),
        "prev_low": round(l, 4),
    }


def calc_indicators(df: pd.DataFrame) -> dict:
    """5分足 DataFrame からテクニカル指標を計算する。

    Args:
        df: open, high, low, close, volume カラムを持つ DataFrame（5分足）

    Returns:
        dict: 各指標の値とスコア
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    current_price = float(close.iloc[-1])

    result = {
        "price": current_price,
        "scores": {},
        "details": {},
        "total_score": 0,
        "max_score": 5,
    }

    # --- 1. ボリンジャーバンド (20期間, 2σ) ---
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and len(bb) > 0:
        lower_band = float(bb.iloc[-1].filter(like="BBL").iloc[0])
        upper_band = float(bb.iloc[-1].filter(like="BBU").iloc[0])
        mid_band = float(bb.iloc[-1].filter(like="BBM").iloc[0])
        below_minus2sigma = current_price <= lower_band
        result["details"]["bb"] = {
            "lower": round(lower_band, 2),
            "mid": round(mid_band, 2),
            "upper": round(upper_band, 2),
        }
        result["scores"]["bb"] = 1 if below_minus2sigma else 0
    else:
        result["scores"]["bb"] = 0
        result["details"]["bb"] = None

    # --- 2. RSI (14期間) ---
    rsi_series = ta.rsi(close, length=14)
    if rsi_series is not None and len(rsi_series) > 0:
        rsi_val = float(rsi_series.iloc[-1])
        result["details"]["rsi"] = round(rsi_val, 2)
        result["scores"]["rsi"] = 1 if rsi_val <= 30 else 0
    else:
        result["scores"]["rsi"] = 0
        result["details"]["rsi"] = None

    # --- 3. 一目均衡表 ---
    ichimoku_result = ta.ichimoku(
        df["high"].astype(float),
        df["low"].astype(float),
        close,
    )
    if ichimoku_result is not None:
        ichimoku_df = ichimoku_result[0] if isinstance(ichimoku_result, tuple) else ichimoku_result
        span_a_col = [c for c in ichimoku_df.columns if "ISA" in c]
        span_b_col = [c for c in ichimoku_df.columns if "ISB" in c]
        if span_a_col and span_b_col:
            span_a = float(ichimoku_df[span_a_col[0]].iloc[-1])
            span_b = float(ichimoku_df[span_b_col[0]].iloc[-1])
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
            above_cloud = current_price > cloud_top
            at_cloud_support = cloud_bottom <= current_price <= cloud_top
            result["details"]["ichimoku"] = {
                "span_a": round(span_a, 2),
                "span_b": round(span_b, 2),
                "position": "above" if above_cloud else ("in_cloud" if at_cloud_support else "below"),
            }
            result["scores"]["ichimoku"] = 1 if (above_cloud or at_cloud_support) else 0
        else:
            result["scores"]["ichimoku"] = 0
            result["details"]["ichimoku"] = None
    else:
        result["scores"]["ichimoku"] = 0
        result["details"]["ichimoku"] = None

    # --- 4. 出来高（直近5本の平均と比較） ---
    if len(volume) >= 6:
        avg_vol = float(volume.iloc[-6:-1].mean())
        cur_vol = float(volume.iloc[-1])
        vol_increasing = cur_vol > avg_vol
        result["details"]["volume"] = {
            "current": int(cur_vol),
            "avg5": int(avg_vol),
        }
        result["scores"]["volume"] = 1 if vol_increasing else 0
    else:
        result["scores"]["volume"] = 0
        result["details"]["volume"] = None

    # --- 5. 注目度（ボラティリティ = ATR / 現在値 で評価） ---
    atr = ta.atr(df["high"].astype(float), df["low"].astype(float), close, length=14)
    if atr is not None and len(atr) > 0:
        atr_val = float(atr.iloc[-1])
        volatility_pct = (atr_val / current_price) * 100
        result["details"]["volatility"] = {
            "atr": round(atr_val, 2),
            "pct": round(volatility_pct, 2),
        }
        # ボラティリティが 0.5% 以上なら注目度が高いとみなす
        result["scores"]["volatility"] = 1 if volatility_pct >= 0.5 else 0
    else:
        result["scores"]["volatility"] = 0
        result["details"]["volatility"] = None

    result["total_score"] = sum(result["scores"].values())
    return result


def calc_sma_levels(daily_df: pd.DataFrame) -> dict:
    """日足 DataFrame から 50日・200日 SMA を算出する。

    Args:
        daily_df: close カラムを持つ日足 DataFrame（200本以上推奨）

    Returns:
        dict: sma50, sma200 の値（データ不足の場合は含まない）
    """
    close = daily_df["close"].astype(float)
    levels = {}
    if len(close) >= 50:
        levels["sma50"] = round(float(close.rolling(50).mean().iloc[-1]), 4)
    if len(close) >= 200:
        levels["sma200"] = round(float(close.rolling(200).mean().iloc[-1]), 4)
    return levels


def calc_vwap(intraday_df: pd.DataFrame) -> dict:
    """当日の5分足から VWAP を算出する。

    Args:
        intraday_df: high, low, close, volume カラムを持つ日中足 DataFrame

    Returns:
        dict: vwap の値
    """
    df = intraday_df.copy()
    # 当日分のみ抽出
    if df.index.tz is not None:
        today = pd.Timestamp.now(tz=df.index.tz).normalize()
    else:
        today = pd.Timestamp.now().normalize()
    df_today = df[df.index >= today]
    if df_today.empty:
        # 市場時間外の場合は直近の取引日データを使用
        last_date = df.index[-1].normalize()
        df_today = df[df.index >= last_date]
    if df_today.empty:
        return {}

    typical_price = (
        df_today["high"].astype(float)
        + df_today["low"].astype(float)
        + df_today["close"].astype(float)
    ) / 3
    volume = df_today["volume"].astype(float)
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / cum_vol
    return {"vwap": round(float(vwap.iloc[-1]), 4)}


def calc_psychological_levels(current_price: float) -> dict:
    """現在価格付近の心理的節目（キリ番）を返す。

    $10 刻みで上下最寄りのキリ番を返す。
    """
    step = 10
    lower = int(current_price / step) * step
    upper = lower + step
    levels = {}
    levels[f"round_{lower}"] = float(lower)
    levels[f"round_{upper}"] = float(upper)
    # $50 刻みの大きな節目が近ければ追加
    big_step = 50
    big_lower = int(current_price / big_step) * big_step
    big_upper = big_lower + big_step
    if big_lower != lower:
        levels[f"round_{big_lower}"] = float(big_lower)
    if big_upper != upper:
        levels[f"round_{big_upper}"] = float(big_upper)
    return levels


def check_proximity(current_price: float, levels: dict, threshold: float) -> list:
    """現在価格が節目に接近しているかチェックする。

    Args:
        current_price: 現在の株価
        levels: calc_pivot_points の戻り値
        threshold: 接近判定の閾値（例: 0.005 = 0.5%）

    Returns:
        list[dict]: 接近している節目のリスト
    """
    nearby = []
    for name, level in levels.items():
        if level <= 0:
            continue
        distance_pct = abs(current_price - level) / level
        if distance_pct <= threshold:
            direction = "above" if current_price >= level else "below"
            nearby.append({
                "name": name.upper(),
                "level": level,
                "distance_pct": round(distance_pct * 100, 3),
                "direction": direction,
            })
    return nearby


# ============================================================
#  自動売買向け指標
# ============================================================


def calc_atr_3day(daily_df: pd.DataFrame) -> Optional[float]:
    """直近3日間の ATR を算出する。

    Args:
        daily_df: high, low, close カラムを持つ日足 DataFrame（3本以上必要）

    Returns:
        float: 3日 ATR 値。データ不足の場合は None。
    """
    if len(daily_df) < 3:
        return None
    atr = ta.atr(
        daily_df["high"].astype(float),
        daily_df["low"].astype(float),
        daily_df["close"].astype(float),
        length=3,
    )
    if atr is None or atr.dropna().empty:
        return None
    return float(atr.dropna().iloc[-1])


def calc_atr_5min(df_5min: pd.DataFrame) -> Optional[float]:
    """5分足の ATR(14) を算出する。

    デイトレの損切り幅に使用。日足・1時間足よりきめ細かく、
    直近のボラティリティを正確に反映する。

    Args:
        df_5min: high, low, close カラムを持つ5分足 DataFrame（14本以上必要）

    Returns:
        float: 5分足 ATR(14) 値。データ不足の場合は None。
    """
    if df_5min is None or len(df_5min) < 14:
        return None
    atr = ta.atr(
        df_5min["high"].astype(float),
        df_5min["low"].astype(float),
        df_5min["close"].astype(float),
        length=14,
    )
    if atr is None or atr.dropna().empty:
        return None
    return float(atr.dropna().iloc[-1])


def check_bullish_reversal_1min(
    df_1min: pd.DataFrame,
    support_levels: dict,
    proximity_threshold: float = 0.005,
    rsi_threshold: float = 40,
) -> Optional[dict]:
    """1分足データからエントリーシグナルを判定する。

    エントリー条件（全て満たす）:
      1. 価格がサポートに proximity_threshold 以内に接近
      2. 直近の1分足が陽線（close > open）
      3. RSI クロスオーバー（previous <= rsi_threshold かつ current > rsi_threshold）

    Args:
        df_1min: 1分足 OHLCV DataFrame
        support_levels: {name: price} のサポートレベル辞書
        proximity_threshold: サポートへの接近閾値（デフォルト 0.5%）
        rsi_threshold: RSI 反転判定の閾値（デフォルト 40）

    Returns:
        dict: シグナル情報 {support_name, support_price, price, rsi}。
              シグナルなしの場合は None。
    """
    if len(df_1min) < 16:
        return None

    close = df_1min["close"].astype(float)
    current_price = float(close.iloc[-1])

    # 条件1: サポートへの接近チェック
    nearest_support = None
    nearest_distance = float("inf")
    for name, level in support_levels.items():
        if level <= 0:
            continue
        # サポートは現在価格より下（または同水準）のみ対象
        if level > current_price * 1.005:
            continue
        distance_pct = abs(current_price - level) / level
        if distance_pct <= proximity_threshold and distance_pct < nearest_distance:
            nearest_distance = distance_pct
            nearest_support = (name, level)

    if nearest_support is None:
        return None

    # 条件2: 直近1分足が陽線
    last_open = float(df_1min["open"].iloc[-1])
    last_close = float(df_1min["close"].iloc[-1])
    if last_close <= last_open:
        return None

    # 条件3: RSI の上昇転換
    rsi_series = ta.rsi(close, length=14)
    if rsi_series is None or rsi_series.dropna().empty or len(rsi_series.dropna()) < 2:
        return None

    rsi_current = float(rsi_series.dropna().iloc[-1])
    rsi_prev = float(rsi_series.dropna().iloc[-2])

    if not (rsi_prev <= rsi_threshold and rsi_current > rsi_threshold):
        return None

    return {
        "support_name": nearest_support[0],
        "support_price": nearest_support[1],
        "price": current_price,
        "rsi": round(rsi_current, 2),
    }


def check_above_daily_sma50(
    daily_df: pd.DataFrame, current_price: float
) -> tuple[bool, float]:
    """現在価格が日足50SMAより上にあるかを判定し、乖離率も返す。

    Args:
        daily_df: 日足 OHLCV DataFrame（50本以上必要）
        current_price: 現在のリアルタイム価格

    Returns:
        tuple[bool, float]: (50SMAより上か, 乖離率 %)
    """
    if daily_df is None or len(daily_df) < 50:
        return False, 0.0

    close = daily_df["close"].astype(float)
    sma50 = close.rolling(50).mean().iloc[-1]
    if pd.isna(sma50) or sma50 <= 0:
        return False, 0.0

    sma50 = float(sma50)
    deviation_pct = round((current_price - sma50) / sma50 * 100, 2)
    return current_price > sma50, deviation_pct


def calc_dynamic_take_profit(
    atr_3day: float,
    current_price: float,
    tp_min: float = 0.015,
    tp_max: float = 0.03,
) -> float:
    """3日 ATR に基づいて動的な利確率を算出する。

    ATR% が低い → tp_min (1.5%), ATR% が高い → tp_max (3.0%) にマッピング。
    ATR% の 1%〜4% を tp_min〜tp_max の範囲に線形補間する。

    Args:
        atr_3day: 3日 ATR 値
        current_price: 現在価格
        tp_min: 利確率の下限（デフォルト 0.015）
        tp_max: 利確率の上限（デフォルト 0.03）

    Returns:
        float: 利確率（例: 0.02 = 2%）
    """
    atr_pct = (atr_3day / current_price) * 100  # ATR%

    # ATR% 1%〜4% の範囲を tp_min〜tp_max に線形補間
    atr_low, atr_high = 1.0, 4.0
    if atr_pct <= atr_low:
        return tp_min
    if atr_pct >= atr_high:
        return tp_max

    ratio = (atr_pct - atr_low) / (atr_high - atr_low)
    return tp_min + ratio * (tp_max - tp_min)


# ============================================================
#  急落リバウンド戦略（Crash & Bounce）
# ============================================================


def find_swing_lows(
    daily_df: pd.DataFrame,
    lookback: int = 60,
    order: int = 3,
) -> list[dict]:
    """日足から局所安値（スイングロー）を検出する。

    前後 *order* 本より低い足をスイングローと判定する。

    Args:
        daily_df: high, low, close カラムを持つ日足 DataFrame
        lookback: 遡る日数（デフォルト 60）
        order: 前後何本と比較するか（デフォルト 3）

    Returns:
        list[dict]: ``[{price, date}, ...]`` 日付昇順
    """
    df = daily_df.tail(lookback).copy()
    if len(df) < 2 * order + 1:
        return []

    lows = df["low"].astype(float).values
    swing_lows: list[dict] = []

    for i in range(order, len(lows) - order):
        is_swing = True
        for j in range(1, order + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing = False
                break
        if is_swing:
            idx = df.index[i]
            date_val = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
            swing_lows.append({
                "price": round(float(lows[i]), 4),
                "date": date_val,
            })

    return swing_lows


def check_crash_bounce(
    daily_df: pd.DataFrame,
    current_price: float,
    current_volume: float,
) -> Optional[dict]:
    """急騰実績・急落・スイングロー接近・出来高急増を日足ベースで一括判定する。

    全条件を満たした場合にシグナル辞書を返す。

    Args:
        daily_df: 日足 DataFrame（60本以上推奨）
        current_price: 現在の株価
        current_volume: 当日の出来高

    Returns:
        dict: ``{swing_low_price, drop_pct, volume_ratio, recent_high}``
              条件未達の場合は None
    """
    if daily_df is None or len(daily_df) < 20:
        return None

    close = daily_df["close"].astype(float)
    high = daily_df["high"].astype(float)
    low = daily_df["low"].astype(float)
    volume = daily_df["volume"].astype(float)

    # --- 条件1: 直近の急騰実績 ---
    # 過去20日の最高値が20日SMAより +15% 以上乖離
    recent_20 = high.tail(20)
    sma_20 = close.tail(20).mean()
    recent_high = float(recent_20.max())

    deviation = (recent_high - sma_20) / sma_20
    if deviation < config.CRASH_RECENT_HIGH_SMA_DEVIATION:
        return None

    # --- 条件2: 急落検出 ---
    # 直近5日の最高値から現在価格が -15% 以上下落
    lookback = config.CRASH_DROP_LOOKBACK_DAYS
    recent_high_5d = float(high.tail(lookback).max())
    drop_pct = (recent_high_5d - current_price) / recent_high_5d

    if drop_pct < config.CRASH_DROP_THRESHOLD:
        return None

    # --- 条件3: スイングロー接近 ---
    swing_lows = find_swing_lows(
        daily_df,
        lookback=config.CRASH_SWING_LOW_LOOKBACK_DAYS,
        order=3,
    )
    if not swing_lows:
        return None

    nearest_sl = None
    nearest_dist = float("inf")
    for sl in swing_lows:
        dist = abs(current_price - sl["price"]) / sl["price"]
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_sl = sl

    if nearest_sl is None or nearest_dist > config.CRASH_SWING_LOW_PROXIMITY:
        return None

    # --- 条件4: 出来高急増 ---
    if len(volume) < 6:
        return None
    avg_vol_5d = float(volume.tail(5).mean())
    if avg_vol_5d <= 0:
        return None
    volume_ratio = current_volume / avg_vol_5d
    if volume_ratio < config.CRASH_VOLUME_MULTIPLIER:
        return None

    return {
        "swing_low_price": nearest_sl["price"],
        "drop_pct": round(drop_pct, 4),
        "volume_ratio": round(volume_ratio, 2),
        "recent_high": round(recent_high, 4),
    }


# ============================================================
#  市場レジーム判定 / ショート戦略向け指標
# ============================================================


def calc_5min_20ma(df_5min: pd.DataFrame) -> Optional[float]:
    """5分足 close の20期間SMAを返す。QQQフィルター用。

    Args:
        df_5min: close カラムを持つ5分足 DataFrame（20本以上必要）

    Returns:
        float: 20期間SMA値。データ不足の場合は None。
    """
    if df_5min is None or len(df_5min) < 20:
        return None
    close = df_5min["close"].astype(float)
    sma = close.rolling(20).mean().iloc[-1]
    if pd.isna(sma):
        return None
    return round(float(sma), 4)


def calc_resistance_levels(daily_bar: pd.Series) -> dict:
    """前日の OHLC からレジスタンスレベル（R1, R2, prev_high）を算出する。

    Args:
        daily_bar: high, low, close カラムを持つ Series（前日足）

    Returns:
        dict: r1, r2, prev_high
    """
    h, l, c = float(daily_bar["high"]), float(daily_bar["low"]), float(daily_bar["close"])
    pivot = (h + l + c) / 3
    return {
        "r1": round(2 * pivot - l, 4),
        "r2": round(pivot + (h - l), 4),
        "prev_high": round(h, 4),
    }


def check_bearish_reversal_1min(
    df_1min: pd.DataFrame,
    resistance_levels: dict,
    proximity_threshold: float = 0.005,
    rsi_threshold: float = 65,
) -> Optional[dict]:
    """1分足データからショートエントリーシグナルを判定する。

    check_bullish_reversal_1min のミラー版。
    エントリー条件（全て満たす）:
      1. 価格がレジスタンスに proximity_threshold 以内に接近
      2. 直近の1分足が陰線（close < open）
      3. RSI 下方クロス（previous >= rsi_threshold かつ current < rsi_threshold）

    Args:
        df_1min: 1分足 OHLCV DataFrame
        resistance_levels: {name: price} のレジスタンスレベル辞書
        proximity_threshold: レジスタンスへの接近閾値（デフォルト 0.5%）
        rsi_threshold: RSI 反転判定の閾値（デフォルト 65）

    Returns:
        dict: シグナル情報 {resistance_name, resistance_price, price, rsi}。
              シグナルなしの場合は None。
    """
    if len(df_1min) < 16:
        return None

    close = df_1min["close"].astype(float)
    current_price = float(close.iloc[-1])

    # 条件1: レジスタンスへの接近チェック
    nearest_resistance = None
    nearest_distance = float("inf")
    for name, level in resistance_levels.items():
        if level <= 0:
            continue
        # レジスタンスは現在価格より上（または同水準）のみ対象
        if level < current_price * 0.995:
            continue
        distance_pct = abs(current_price - level) / level
        if distance_pct <= proximity_threshold and distance_pct < nearest_distance:
            nearest_distance = distance_pct
            nearest_resistance = (name, level)

    if nearest_resistance is None:
        return None

    # 条件2: 直近1分足が陰線
    last_open = float(df_1min["open"].iloc[-1])
    last_close = float(df_1min["close"].iloc[-1])
    if last_close >= last_open:
        return None

    # 条件3: RSI の下落転換
    rsi_series = ta.rsi(close, length=14)
    if rsi_series is None or rsi_series.dropna().empty or len(rsi_series.dropna()) < 2:
        return None

    rsi_current = float(rsi_series.dropna().iloc[-1])
    rsi_prev = float(rsi_series.dropna().iloc[-2])

    if not (rsi_prev >= rsi_threshold and rsi_current < rsi_threshold):
        return None

    return {
        "resistance_name": nearest_resistance[0],
        "resistance_price": nearest_resistance[1],
        "price": current_price,
        "rsi": round(rsi_current, 2),
    }


# ============================================================
#  グレーゾーン / カナリア戦略向け指標
# ============================================================


def calc_running_vwap(df_5min: pd.DataFrame) -> pd.Series:
    """5分足データから日中リセットの running VWAP を算出する。

    各バーごとに当日のVWAP値を返す（日替わりでリセット）。

    Args:
        df_5min: high, low, close, volume カラムを持つ5分足 DataFrame

    Returns:
        pd.Series: 各バーのVWAP値（float）
    """
    tp = (
        df_5min["high"].astype(float)
        + df_5min["low"].astype(float)
        + df_5min["close"].astype(float)
    ) / 3
    vol = df_5min["volume"].astype(float)
    tp_vol = tp * vol

    # 日ごとにグループ化してリセット
    dates = df_5min.index.normalize()
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()

    vwap = cum_tp_vol / cum_vol
    vwap = vwap.replace([float("inf"), float("-inf")], float("nan"))
    return vwap


def calc_qqq_bullish_ratio(df_5min: pd.DataFrame, window: int = 20) -> pd.Series:
    """QQQ 5分足の close > 20MA のローリングブリッシュ比率を算出する。

    各バーごとに直近 window 本のうち close > 20MA だった比率（0.0〜1.0）を返す。

    Args:
        df_5min: close カラムを持つ QQQ の5分足 DataFrame
        window: ローリング計算のウィンドウサイズ（デフォルト 20）

    Returns:
        pd.Series: ブリッシュ比率（0.0〜1.0）
    """
    close = df_5min["close"].astype(float)
    ma20 = close.rolling(20).mean()
    is_bullish = (close > ma20).astype(float)
    ratio = is_bullish.rolling(window, min_periods=1).mean()
    return ratio


def calc_qqq_regime_hybrid(
    df_5min: pd.DataFrame,
    *,
    window: int = 20,
    cb_bars: int = 3,
    cb_threshold_down: float = -0.005,
    cb_threshold_up: float = 0.005,
    gray_zone_low: float = 0.45,
    gray_zone_high: float = 0.55,
) -> tuple[str, str, float]:
    """2層ハイブリッドQQQレジーム判定。

    Layer 1 (最優先): サーキットブレイカー — 直近cb_bars本の変化率が閾値を超えたら即時反応
    Layer 2 (ベース): ローリング比率 — 大局トレンドを安定判定（Bullish / Bearish / Gray）

    Args:
        df_5min: close カラムを持つ QQQ の5分足 DataFrame
        window: ローリング比率のウィンドウ（Layer 2）
        cb_bars: サーキットブレイカーで参照する本数（Layer 1）
        cb_threshold_down: 下方向の発動閾値（例: -0.005 = -0.5%）
        cb_threshold_up: 上方向の発動閾値（例: +0.005 = +0.5%）
        gray_zone_low: ローリング比率の bearish 閾値
        gray_zone_high: ローリング比率の bullish 閾値

    Returns:
        tuple(regime, source, ratio)
        - regime: "bullish" | "bearish" | "gray"
        - source: "circuit_breaker" | "rolling"
        - ratio: ローリング比率（0.0〜1.0、表示・ログ用）
    """
    close = df_5min["close"].astype(float)

    # Layer 2: ローリング比率（常に計算、フォールバック兼表示用）
    ratio = float(calc_qqq_bullish_ratio(df_5min, window=window).iloc[-1])

    # Layer 1: サーキットブレイカー（直近N本の変化率）
    if len(close) >= cb_bars + 1:
        ref_price = float(close.iloc[-(cb_bars + 1)])
        if ref_price > 0:
            cb_change = (float(close.iloc[-1]) - ref_price) / ref_price
            if cb_change <= cb_threshold_down:
                return "bearish", "circuit_breaker", ratio
            if cb_change >= cb_threshold_up:
                return "bullish", "circuit_breaker", ratio

    # Layer 2: ローリング比率
    if ratio > gray_zone_high:
        return "bullish", "rolling", ratio
    if ratio < gray_zone_low:
        return "bearish", "rolling", ratio
    return "gray", "rolling", ratio
