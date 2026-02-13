"""高ボラティリティ銘柄のスクリーニング

Alpaca ScreenerClient で MostActives / MarketMovers を取得し、
出来高・3日 ATR でフィルタリングして売買候補を選定する。
個人投資家に人気のある銘柄を優先的に選出する。
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import (
    MostActivesRequest,
    MarketMoversRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame

import config
from models import TradeCandidate
from indicators import (
    calc_atr_3day,
    calc_pivot_points,
    calc_psychological_levels,
    calc_sma_levels,
    find_swing_lows,
)

log = logging.getLogger(__name__)


def _get_theme(symbol: str) -> str | None:
    """テーマ株ならテーマ名を返す。該当なしなら None。"""
    for theme_name, symbols in config.THEME_STOCKS.items():
        if symbol in symbols:
            return theme_name
    return None


class StockScreener:
    """高ボラティリティ・高出来高銘柄をスクリーニングするクライアント。"""

    def __init__(self, data_client: StockHistoricalDataClient):
        self.screener_client = ScreenerClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
        )
        self.data_client = data_client

    def fetch_candidates(self) -> list[str]:
        """MostActives / MarketMovers + 個人投資家人気銘柄から候補を取得する。"""
        symbols = set()

        # MostActives（出来高上位 = 高出来高銘柄を自然に含む）
        try:
            actives = self.screener_client.get_most_actives(
                MostActivesRequest(top=config.SCREENER_TOP_N)
            )
            for item in actives.most_actives:
                symbols.add(item.symbol)
            log.info(f"MostActives: {len(actives.most_actives)} 銘柄取得")
        except Exception as e:
            log.warning(f"MostActives 取得失敗: {e}")

        # MarketMovers
        try:
            movers = self.screener_client.get_market_movers(
                MarketMoversRequest(top=config.SCREENER_TOP_N)
            )
            for item in movers.gainers:
                symbols.add(item.symbol)
            for item in movers.losers:
                symbols.add(item.symbol)
            log.info(f"MarketMovers: gainers={len(movers.gainers)}, losers={len(movers.losers)}")
        except Exception as e:
            log.warning(f"MarketMovers 取得失敗: {e}")

        # 個人投資家人気銘柄を常に候補に追加
        symbols.update(config.RETAIL_FAVORITES)
        log.info(f"RETAIL_FAVORITES {len(config.RETAIL_FAVORITES)} 銘柄を追加")

        # テーマ株を常に候補に追加
        theme_symbols = set()
        for syms in config.THEME_STOCKS.values():
            theme_symbols.update(syms)
        symbols.update(theme_symbols)
        log.info(f"THEME_STOCKS {len(theme_symbols)} 銘柄を追加")

        # WATCHLIST銘柄は除外
        symbols -= set(config.WATCHLIST)
        log.info(f"候補シンボル合計: {len(symbols)} 銘柄")
        return list(symbols)

    def fetch_daily_bars(self, symbol: str, limit: int = 10) -> pd.DataFrame | None:
        """日足バーを取得する（ATR・出来高計算用）。"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now(timezone.utc) - timedelta(days=limit + 5),
                limit=limit,
            )
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            return df
        except Exception as e:
            log.debug(f"{symbol} 日足取得失敗: {e}")
            return None

    def screen(self) -> list[TradeCandidate]:
        """スクリーニングを実行し、高ボラ・高出来高銘柄の候補リストを返す。"""
        symbols = self.fetch_candidates()
        candidates = []

        for symbol in symbols:
            daily_df = self.fetch_daily_bars(symbol, limit=55)
            if daily_df is None or len(daily_df) < 4:
                continue

            current_price = float(daily_df["close"].iloc[-1])

            # 価格フィルタ
            if current_price < config.SCREENER_MIN_PRICE:
                continue
            if current_price > config.SCREENER_MAX_PRICE:
                continue

            # 出来高フィルタ（直近5日の平均出来高）
            recent_volume = daily_df["volume"].astype(float).tail(5)
            avg_volume = float(recent_volume.mean())
            if avg_volume < config.SCREENER_MIN_VOLUME:
                continue

            # 3日 ATR フィルタ（テーマ株は閾値を緩和）
            atr_val = calc_atr_3day(daily_df)
            if atr_val is None:
                continue
            atr_pct = (atr_val / current_price) * 100
            theme = _get_theme(symbol)
            min_atr = config.SCREENER_MIN_ATR_PCT
            if theme:
                min_atr *= config.THEME_SCREENER_ATR_DISCOUNT
            if atr_pct < min_atr:
                continue

            # サポートレベル算出
            support_levels = {}
            try:
                prev_daily = daily_df.iloc[-2]
                support_levels.update(calc_pivot_points(prev_daily))
                # 前々日安値
                if len(daily_df) >= 3:
                    support_levels["prev2_low"] = round(float(daily_df.iloc[-3]["low"]), 4)
            except Exception:
                pass
            try:
                sma_levels = calc_sma_levels(daily_df)
                support_levels.update(sma_levels)
            except Exception:
                pass
            # 心理的節目（キリ番）
            support_levels.update(calc_psychological_levels(current_price))
            # スイングロー（直近60日の局所安値）
            try:
                swing_lows = find_swing_lows(daily_df, lookback=60, order=3)
                for idx, sl in enumerate(swing_lows[-3:]):  # 直近3つ
                    support_levels[f"swing_low_{idx+1}"] = sl["price"]
            except Exception:
                pass

            candidate = TradeCandidate(
                symbol=symbol,
                atr_pct=round(atr_pct, 2),
                current_price=current_price,
                support_levels=support_levels,
            )
            candidates.append(candidate)

            tag = ""
            if theme:
                tag = f" [テーマ: {theme}]"
            elif symbol in config.RETAIL_FAVORITES:
                tag = " [人気]"
            log.info(
                f"  候補: {symbol}{tag} price=${current_price:.2f} "
                f"ATR%={atr_pct:.2f}% vol={avg_volume:,.0f} "
                f"supports={list(support_levels.keys())}"
            )

        # ソート: テーマ株 > RETAIL_FAVORITES > その他、各段内は ATR% 降順
        def _sort_key(c: TradeCandidate) -> tuple:
            if _get_theme(c.symbol):
                priority = 2
            elif c.symbol in config.RETAIL_FAVORITES:
                priority = 1
            else:
                priority = 0
            return (priority, c.atr_pct)

        candidates.sort(key=_sort_key, reverse=True)
        log.info(f"スクリーニング完了: {len(candidates)} 銘柄が候補")
        return candidates
