"""急落リバウンド戦略のユニットテスト"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from indicators import find_swing_lows, check_crash_bounce


def _make_daily_df(prices: list[dict]) -> pd.DataFrame:
    """テスト用の日足 DataFrame を作成する。"""
    dates = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=len(prices),
        freq="B",  # 営業日
        tz="UTC",
    )
    data = []
    for p in prices:
        data.append({
            "open": p.get("open", p["close"]),
            "high": p.get("high", p["close"] * 1.01),
            "low": p.get("low", p["close"] * 0.99),
            "close": p["close"],
            "volume": p.get("volume", 1_000_000),
        })
    df = pd.DataFrame(data, index=dates)
    return df


# ==========================================================
#  Test 1: find_swing_lows — 基本テスト
# ==========================================================
def test_find_swing_lows_basic():
    """V字の谷を検出できるか"""
    # 下降 → 底 → 上昇 のパターン
    closes = (
        [150 - i * 2 for i in range(10)]   # 150 → 132 (下降)
        + [130]                              # 底
        + [132 + i * 2 for i in range(10)]  # 上昇
    )
    prices = [{"close": c, "low": c - 1, "high": c + 1} for c in closes]
    df = _make_daily_df(prices)

    lows = find_swing_lows(df, lookback=60, order=3)
    assert len(lows) >= 1, f"スイングローが検出されるべき: {lows}"
    # 最安値は 129 (130 - 1)
    assert any(abs(sl["price"] - 129.0) < 1.0 for sl in lows), \
        f"$129付近のスイングローが見つからない: {lows}"
    print(f"  OK: {len(lows)} swing lows found: {lows}")


# ==========================================================
#  Test 2: find_swing_lows — データ不足
# ==========================================================
def test_find_swing_lows_insufficient_data():
    """データ不足時は空リストを返す"""
    prices = [{"close": 100}] * 3
    df = _make_daily_df(prices)
    lows = find_swing_lows(df, lookback=60, order=3)
    assert lows == [], f"データ不足時は空リストのはず: {lows}"
    print("  OK: 空リストを返した")


# ==========================================================
#  Test 3: check_crash_bounce — 全条件を満たすケース
# ==========================================================
def test_check_crash_bounce_signal():
    """急騰→急落→スイングロー接近→出来高急増 のシグナル発火"""
    n = 70
    closes = []

    # Phase 1: 安定期 (day 0-29) — $100前後
    for i in range(30):
        closes.append(100 + np.random.uniform(-2, 2))

    # Phase 2: V字底 (day 30-34) — $90まで下げて戻す（スイングロー作成）
    closes.extend([95, 92, 90, 93, 97])

    # Phase 3: 急騰期 (day 35-54) — $100 → $170
    for i in range(20):
        closes.append(100 + i * 3.5)

    # Phase 4: 急落 (day 55-64) — $170 → $92
    for i in range(10):
        closes.append(170 - i * 8)

    prices = []
    for i, c in enumerate(closes):
        vol = 1_000_000
        if i >= 60:
            vol = 500_000  # 直近5日の平均を低めに
        prices.append({
            "close": c,
            "high": c + 2,
            "low": c - 2,
            "volume": vol,
        })

    df = _make_daily_df(prices)

    # 現在価格: スイングロー($88=$90-2) 付近の $89
    current_price = 89.0
    # 出来高: 5日平均(500k)の 3倍
    current_volume = 1_500_000

    result = check_crash_bounce(df, current_price, current_volume)
    assert result is not None, "シグナルが発火するべき"
    assert "swing_low_price" in result
    assert "drop_pct" in result
    assert result["drop_pct"] > 0.15
    assert result["volume_ratio"] >= 2.0
    print(f"  OK: signal={result}")


# ==========================================================
#  Test 4: check_crash_bounce — 急落なし（条件未達）
# ==========================================================
def test_check_crash_bounce_no_drop():
    """急落がないケースではシグナルなし"""
    n = 30
    prices = [{"close": 100 + i * 0.5, "volume": 1_000_000} for i in range(n)]
    df = _make_daily_df(prices)

    result = check_crash_bounce(df, current_price=115.0, current_volume=3_000_000)
    assert result is None, f"急落なしではNoneのはず: {result}"
    print("  OK: None を返した（急落なし）")


# ==========================================================
#  Test 5: check_crash_bounce — 出来高不足
# ==========================================================
def test_check_crash_bounce_low_volume():
    """出来高が不足していればシグナルなし"""
    n = 70
    closes = []

    for i in range(30):
        closes.append(100)
    closes.extend([95, 92, 90, 93, 97])
    for i in range(20):
        closes.append(100 + i * 3.5)
    for i in range(10):
        closes.append(170 - i * 8)

    # 全ての出来高を高く設定（急増にならない）
    prices = [{
        "close": c, "high": c + 2, "low": c - 2, "volume": 2_000_000
    } for c in closes]
    df = _make_daily_df(prices)

    # 出来高: 5日平均と同じ → 急増でない
    result = check_crash_bounce(df, current_price=89.0, current_volume=2_000_000)
    assert result is None, f"出来高不足ではNoneのはず: {result}"
    print("  OK: None を返した（出来高不足）")


# ==========================================================
#  Run all tests
# ==========================================================
if __name__ == "__main__":
    tests = [
        ("find_swing_lows 基本テスト", test_find_swing_lows_basic),
        ("find_swing_lows データ不足", test_find_swing_lows_insufficient_data),
        ("check_crash_bounce シグナル発火", test_check_crash_bounce_signal),
        ("check_crash_bounce 急落なし", test_check_crash_bounce_no_drop),
        ("check_crash_bounce 出来高不足", test_check_crash_bounce_low_volume),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"結果: {passed}/{passed+failed} passed")
    if failed:
        print(f"  {failed} tests FAILED")
        exit(1)
    else:
        print("  All tests passed!")
