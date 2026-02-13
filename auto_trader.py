"""自動売買エンジン

スクリーニング済み候補のエントリー判定、注文実行、
オープンポジションの監視・決済を行う。
"""

import logging
import math
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo

import discord
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config
from models import AutoPosition, TradeCandidate
from indicators import (
    calc_atr_3day,
    calc_atr_5min,
    calc_5min_20ma,
    calc_pivot_points,
    calc_psychological_levels,
    calc_resistance_levels,
    calc_running_vwap,
    calc_qqq_bullish_ratio,
    check_above_daily_sma50,
    check_bearish_reversal_1min,
    check_bullish_reversal_1min,
    check_crash_bounce,
    calc_dynamic_take_profit,
)

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


def is_market_open() -> bool:
    """米国市場がオープン中かどうかを判定する。"""
    now_et = datetime.now(ET)
    # 土日はクローズ（月=0, 日=6）
    if now_et.weekday() >= 5:
        return False
    market_open = time(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE)
    market_close = time(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
    return market_open <= now_et.time() < market_close


def is_entry_window() -> bool:
    """エントリー可能な時間帯かどうかを判定する。

    寄り付き後 N 分（ノイズが多い）と引け前 N 分はエントリーしない。
    プレマーケット・アフターマーケットも不可。
    """
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False

    market_open = datetime.combine(
        now_et.date(),
        time(config.MARKET_OPEN_HOUR, config.MARKET_OPEN_MINUTE),
        tzinfo=ET,
    )
    market_close = datetime.combine(
        now_et.date(),
        time(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE),
        tzinfo=ET,
    )

    entry_start = market_open + timedelta(minutes=config.ENTRY_BUFFER_MINUTES_OPEN)
    entry_end = market_close - timedelta(minutes=config.ENTRY_BUFFER_MINUTES_CLOSE)

    return entry_start <= now_et < entry_end


class AutoTrader:
    """自動売買エンジン。"""

    def __init__(
        self,
        data_client: StockHistoricalDataClient,
        trading_client: TradingClient,
    ):
        self.data_client = data_client
        self.trading_client = trading_client
        self.candidates: list[TradeCandidate] = []
        self.positions: list[AutoPosition] = []

        # 動的銘柄リスト（毎日更新）
        self._auto_symbols: list[str] = list(config.AUTO_SYMBOLS_FALLBACK)
        self._symbol_list_date: datetime | None = None

        # 決算ブラックアウトキャッシュ {symbol: (date, is_blackout)}
        self._earnings_cache: dict[str, tuple] = {}

        # 市場レジーム判定
        self._qqq_regime: str | None = None       # "bullish" / "bearish" / "gray" / None
        self._qqq_cache_time: datetime | None = None
        self._qqq_price: float | None = None
        self._qqq_ma20_value: float | None = None
        self._qqq_bullish_ratio: float = 0.5      # ブリッシュ比率 0.0〜1.0
        self._qqq_prev_close: float | None = None  # QQQ 前日終値
        self._qqq_alert_mode: bool = False         # カナリア警戒モード
        self._qqq_intraday_return: float = 0.0     # QQQ 日中リターン
        self._vix_panic: bool = False
        self._vix_panic_date: datetime | None = None
        self._vix_change_pct: float = 0.0

    # ----------------------------------------------------------
    #  銘柄リスト管理（売買代金上位を日次取得）
    # ----------------------------------------------------------
    def _update_symbol_list(self) -> None:
        """売買代金上位 N 銘柄で自動監視リストを日次更新する。

        MostActives から多めに取得し、株価フィルタ($20以上)で
        小型・ペニー株を除外してから上位 N 銘柄を選定する。
        """
        today = datetime.now(ET).date()
        if self._symbol_list_date and self._symbol_list_date == today:
            return

        try:
            from alpaca.data.historical.screener import ScreenerClient
            from alpaca.data.requests import MostActivesRequest

            sc = ScreenerClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
            actives = sc.get_most_actives(
                MostActivesRequest(top=config.AUTO_SYMBOLS_FETCH_TOP)
            )
            raw_symbols = [item.symbol for item in actives.most_actives]
            log.info(f"[AutoSymbol] MostActives {len(raw_symbols)}銘柄取得: {raw_symbols[:20]}...")

            # 株価フィルタ — $20未満の小型株を除外
            filtered = []
            for sym in raw_symbols:
                price = self._get_latest_price(sym)
                if price is not None and price >= config.AUTO_SYMBOLS_MIN_PRICE:
                    filtered.append(sym)
                elif price is not None:
                    log.debug(f"[AutoSymbol] {sym} 除外 (${price:.2f} < ${config.AUTO_SYMBOLS_MIN_PRICE})")
                if len(filtered) >= config.AUTO_SYMBOLS_COUNT:
                    break

            if len(filtered) >= 5:
                self._auto_symbols = filtered
                log.info(
                    f"[AutoSymbol] フィルタ後{len(filtered)}銘柄を採用: {filtered}"
                )
            else:
                raise ValueError(f"フィルタ後の銘柄数不足: {len(filtered)}")
        except Exception as e:
            self._auto_symbols = list(config.AUTO_SYMBOLS_FALLBACK)
            log.warning(
                f"[AutoSymbol] 取得失敗 ({e}) — フォールバック: {self._auto_symbols}"
            )

        self._symbol_list_date = today

    # ----------------------------------------------------------
    #  決算ブラックアウト
    # ----------------------------------------------------------
    def _check_earnings_blackout(self, symbol: str) -> bool:
        """決算発表の前後 N 時間以内なら True を返す（取引停止）。"""
        today = datetime.now(ET).date()

        # 当日キャッシュがあればそれを使う
        if symbol in self._earnings_cache:
            cache_date, is_blackout = self._earnings_cache[symbol]
            if cache_date == today:
                return is_blackout

        is_blackout = self._fetch_earnings_status(symbol)
        self._earnings_cache[symbol] = (today, is_blackout)
        return is_blackout

    def _fetch_earnings_status(self, symbol: str) -> bool:
        """yfinance で決算日を取得し、ブラックアウト期間内か判定する。"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is None:
                return False

            earnings_dates = []
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date", [])
                earnings_dates = list(raw)

            now = datetime.now(timezone.utc)
            for d in earnings_dates:
                # datetime.date → datetime に変換
                if hasattr(d, "hour"):
                    dt = pd.Timestamp(d)
                else:
                    dt = pd.Timestamp(datetime.combine(d, time(16, 0)))  # 引け後想定
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")

                diff_hours = abs((dt - now).total_seconds()) / 3600
                if diff_hours <= config.EARNINGS_BLACKOUT_HOURS:
                    log.info(
                        f"[Earnings] {symbol} 決算ブラックアウト "
                        f"(earnings: {dt.strftime('%m/%d %H:%M UTC')})"
                    )
                    return True

            return False
        except Exception as e:
            log.debug(f"[Earnings] {symbol} 決算日チェック失敗: {e}")
            return False

    # ----------------------------------------------------------
    #  市場レジーム判定
    # ----------------------------------------------------------
    def _update_market_regime(self) -> None:
        """QQQ 5分足20MAのブリッシュ比率でレジームを判定する（キャッシュ付き）。

        比率ベースの判定:
          - > 55% → bullish (ロングOK)
          - < 45% → bearish (ショートOK)
          - 45〜55% → gray (エントリー制限、ただしVWAP条件で例外あり)
        """
        if not config.QQQ_FILTER_ENABLED:
            self._qqq_regime = None
            return

        now = datetime.now(timezone.utc)
        if (
            self._qqq_cache_time is not None
            and (now - self._qqq_cache_time).total_seconds() < config.QQQ_CACHE_SECONDS
        ):
            return  # キャッシュ有効

        try:
            df_5min = self._fetch_5min_bars("QQQ", limit=50)
            if df_5min is None or df_5min.empty:
                log.warning("[Regime] QQQ 5分足取得失敗")
                return

            ma20 = calc_5min_20ma(df_5min)
            if ma20 is None:
                log.warning("[Regime] QQQ 20MA計算失敗（データ不足）")
                return

            close = float(df_5min["close"].astype(float).iloc[-1])
            self._qqq_price = close
            self._qqq_ma20_value = ma20

            # ブリッシュ比率計算
            ratio_series = calc_qqq_bullish_ratio(
                df_5min, window=config.QQQ_BULLISH_RATIO_WINDOW
            )
            ratio = float(ratio_series.iloc[-1])
            self._qqq_bullish_ratio = ratio

            # レジーム判定
            if ratio > config.QQQ_GRAY_ZONE_HIGH:
                self._qqq_regime = "bullish"
            elif ratio < config.QQQ_GRAY_ZONE_LOW:
                self._qqq_regime = "bearish"
            else:
                self._qqq_regime = "gray"

            # QQQ 前日終値と日中リターン（カナリア警戒モード用）
            self._update_qqq_alert_mode(df_5min, close)

            self._qqq_cache_time = now
            log.info(
                f"[Regime] QQQ=${close:.2f} 20MA=${ma20:.2f} "
                f"ratio={ratio:.0%} → {self._qqq_regime}"
                f"{' [ALERT]' if self._qqq_alert_mode else ''}"
            )
        except Exception as e:
            log.warning(f"[Regime] QQQ レジーム判定失敗: {e}")

    def _update_qqq_alert_mode(self, df_5min: pd.DataFrame, current_price: float) -> None:
        """カナリア警戒モードを判定する。

        条件（いずれかを満たせば警戒モード）:
          A) QQQ直近5分間の騰落率がマイナス かつ QQQが自身の当日VWAPを0.5%以上下回る
          B) QQQ RSI(14) が40を下回った場合
        """
        try:
            import pandas_ta as pta

            # 前日終値を取得（日中リターン計算用）
            if self._qqq_prev_close is None:
                daily_df = self._fetch_daily_bars("QQQ", limit=5)
                if daily_df is not None and len(daily_df) >= 2:
                    today = pd.Timestamp.now(tz="UTC").normalize()
                    prev_data = daily_df[daily_df.index.normalize() < today]
                    if not prev_data.empty:
                        self._qqq_prev_close = float(prev_data.iloc[-1]["close"])

            if self._qqq_prev_close is not None and self._qqq_prev_close > 0:
                self._qqq_intraday_return = (
                    (current_price - self._qqq_prev_close) / self._qqq_prev_close
                )

            # 条件A: 直近5分マイナス AND QQQがVWAPを0.5%以上下回る
            cond_a = False
            if len(df_5min) >= 2:
                prev_close = float(df_5min["close"].astype(float).iloc[-2])
                last_5min_return = (current_price - prev_close) / prev_close

                vwap_series = calc_running_vwap(df_5min)
                if not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]):
                    qqq_vwap = float(vwap_series.iloc[-1])
                    vwap_deviation = (current_price - qqq_vwap) / qqq_vwap
                    cond_a = last_5min_return < 0 and vwap_deviation <= -0.005

            # 条件B: QQQ RSI(14) < 40
            cond_b = False
            close_series = df_5min["close"].astype(float)
            rsi_series = pta.rsi(close_series, length=14)
            if rsi_series is not None and not rsi_series.dropna().empty:
                qqq_rsi = float(rsi_series.dropna().iloc[-1])
                cond_b = qqq_rsi < 40

            self._qqq_alert_mode = cond_a or cond_b
        except Exception as e:
            log.debug(f"[Canary] QQQ alert mode 判定失敗: {e}")
            self._qqq_alert_mode = False

    def _check_vix_panic(self) -> bool:
        """VIX 前日比でパニックモードを判定する（キャッシュ付き）。"""
        if not config.VIX_PANIC_ENABLED:
            return False

        now = datetime.now(timezone.utc)
        today = now.date()

        # パニック検出後は当日中は再チェックしない（True固定）
        if self._vix_panic and self._vix_panic_date == today:
            return True

        # 非パニック時はキャッシュ期間内ならスキップ
        if (
            self._vix_panic_date == today
            and self._qqq_cache_time is not None
            and (now - self._qqq_cache_time).total_seconds() < config.VIX_CACHE_SECONDS
        ):
            return self._vix_panic

        try:
            import yfinance as yf

            ticker = yf.Ticker(config.VIX_SYMBOL)
            hist = ticker.history(period="5d")
            if hist is None or len(hist) < 2:
                log.debug("[VIX] データ不足")
                return False

            prev_close = float(hist["Close"].iloc[-2])
            curr = float(hist["Close"].iloc[-1])
            if prev_close <= 0:
                return False

            change_pct = (curr - prev_close) / prev_close
            self._vix_change_pct = change_pct
            self._vix_panic_date = today

            if change_pct >= config.VIX_PANIC_THRESHOLD:
                self._vix_panic = True
                log.warning(
                    f"[VIX] パニックモード検出 VIX={curr:.2f} "
                    f"前日比={change_pct*100:+.1f}%"
                )
            else:
                self._vix_panic = False
                log.info(
                    f"[VIX] 正常 VIX={curr:.2f} 前日比={change_pct*100:+.1f}%"
                )

            return self._vix_panic
        except Exception as e:
            log.debug(f"[VIX] チェック失敗: {e}")
            return False

    def _check_shortable(self, symbol: str) -> bool:
        """Alpaca API で空売り可能かチェックする。"""
        try:
            asset = self.trading_client.get_asset(symbol)
            return bool(asset.shortable and asset.easy_to_borrow)
        except Exception as e:
            log.debug(f"[Short] {symbol} 空売り可否チェック失敗: {e}")
            return False

    # ----------------------------------------------------------
    #  リスクベース・ポジションサイジング
    # ----------------------------------------------------------
    def _calc_position_qty(
        self, current_price: float, stop_distance: float
    ) -> int:
        """損切り幅から逆算して株数を決定する。

        - リスク金額 = ACCOUNT_SIZE × RISK_PER_TRADE
        - リスク株数 = リスク金額 / 損切り幅
        - 上限: 1銘柄あたり資金の MAX_POSITION_PCT まで
        - 上限: 残余資金を超えない
        """
        equity = config.ACCOUNT_SIZE
        risk_amount = equity * config.RISK_PER_TRADE

        # リスクから逆算
        risk_qty = math.floor(risk_amount / stop_distance) if stop_distance > 0 else 1

        # 1銘柄の最大金額
        max_cost = equity * config.MAX_POSITION_PCT
        # 既存ポジションの投資額を差し引いた残余
        invested = sum(
            p.entry_price * p.qty for p in self.positions if not p.closed
        )
        available = max(0, equity - invested)
        position_cap = min(max_cost, available)
        max_qty = math.floor(position_cap / current_price) if current_price > 0 else 1

        qty = max(1, min(risk_qty, max_qty))

        log.debug(
            f"  sizing: risk=${risk_amount:.0f} / SL_dist=${stop_distance:.2f} "
            f"→ risk_qty={risk_qty} / max_qty={max_qty} → qty={qty} "
            f"(${qty * current_price:.0f})"
        )
        return qty

    # ----------------------------------------------------------
    #  候補管理
    # ----------------------------------------------------------
    def update_candidates(self, new_candidates: list[TradeCandidate]) -> None:
        """スクリーナーからの新しい候補リストで更新する。

        既存候補で期限切れのものを除去し、新候補を追加する。
        """
        # 期限切れ候補を除去
        self.candidates = [c for c in self.candidates if not c.is_expired()]

        # 既存シンボルの集合
        existing = {c.symbol for c in self.candidates}

        # 新候補を追加（重複除外、オープンポジションのシンボルも除外）
        open_symbols = {p.symbol for p in self.positions if not p.closed}
        for candidate in new_candidates:
            if candidate.symbol not in existing and candidate.symbol not in open_symbols:
                self.candidates.append(candidate)

        log.info(f"候補更新: {len(self.candidates)} 銘柄")

    def refresh_auto_symbols(self) -> None:
        """売買代金上位銘柄を常に候補リストに維持する。

        毎日銘柄リストを更新し、既に候補にある銘柄や
        オープンポジションの銘柄はスキップする。
        """
        # 日次で銘柄リストを更新
        self._update_symbol_list()

        existing = {c.symbol for c in self.candidates}
        open_symbols = {p.symbol for p in self.positions if not p.closed}

        for symbol in self._auto_symbols:
            if symbol in existing or symbol in open_symbols:
                continue

            try:
                daily_df = self._fetch_daily_bars(symbol, limit=60)
                if daily_df is None or len(daily_df) < 3:
                    continue

                price = self._get_latest_price(symbol)
                if price is None:
                    continue

                # 前日までのデータでサポートレベルを計算
                today = pd.Timestamp.now(tz="UTC").normalize()
                prev_data = daily_df[daily_df.index.normalize() < today]
                if prev_data.empty:
                    prev_data = daily_df

                support_levels = {}
                prev_bar = prev_data.iloc[-1]
                support_levels.update(calc_pivot_points(prev_bar))
                support_levels.update(calc_psychological_levels(price))

                close = prev_data["close"].astype(float)
                if len(close) >= 50:
                    support_levels["sma50"] = round(
                        float(close.rolling(50).mean().iloc[-1]), 4
                    )
                if len(prev_data) >= 2:
                    support_levels["prev2_low"] = round(
                        float(prev_data.iloc[-2]["low"]), 4
                    )

                atr_daily = calc_atr_3day(prev_data) if len(prev_data) >= 3 else None
                atr_pct = round((atr_daily / price * 100), 2) if atr_daily else 0.0

                candidate = TradeCandidate(
                    symbol=symbol,
                    atr_pct=atr_pct,
                    current_price=price,
                    support_levels=support_levels,
                )
                self.candidates.append(candidate)
                log.info(
                    f"[AutoSymbol] {symbol} 候補追加 "
                    f"price=${price:.2f} ATR%={atr_pct:.1f}% "
                    f"levels={list(support_levels.keys())}"
                )
            except Exception as e:
                log.debug(f"[AutoSymbol] {symbol} 候補追加失敗: {e}")

    @property
    def open_position_count(self) -> int:
        return sum(1 for p in self.positions if not p.closed)

    # ----------------------------------------------------------
    #  データ取得
    # ----------------------------------------------------------
    def _fetch_1min_bars(self, symbol: str, limit: int = 30) -> pd.DataFrame | None:
        """1分足バーを取得する。"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now(timezone.utc) - timedelta(hours=2),
                limit=limit,
            )
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            return df
        except Exception as e:
            log.debug(f"{symbol} 1分足取得失敗: {e}")
            return None

    def _fetch_5min_bars(self, symbol: str, limit: int = 30) -> pd.DataFrame | None:
        """5分足バーを取得する（損切りATR計算用）。"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(amount=5, unit=TimeFrameUnit.Minute),
                start=datetime.now(timezone.utc) - timedelta(hours=4),
                limit=limit,
            )
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            return df
        except Exception as e:
            log.debug(f"{symbol} 5分足取得失敗: {e}")
            return None

    def _fetch_daily_bars(self, symbol: str, limit: int = 10) -> pd.DataFrame | None:
        """日足バーを取得する。"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now(timezone.utc) - timedelta(days=int(limit * 1.5) + 5),
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

    def _get_latest_price(self, symbol: str) -> float | None:
        """最新の仲値を取得する。"""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(req)
            quote = quotes[symbol]
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return ask if ask > 0 else bid
        except Exception as e:
            log.debug(f"{symbol} 気配値取得失敗: {e}")
            return None

    # ----------------------------------------------------------
    #  エントリー判定・実行
    # ----------------------------------------------------------
    def check_entries(self) -> list[AutoPosition]:
        """全候補に対してエントリー条件をチェックし、条件を満たした銘柄で注文を実行する。"""
        if not is_entry_window():
            return []

        if self.open_position_count >= config.MAX_POSITIONS:
            return []

        # VIXパニック → 全エントリー停止
        if self._check_vix_panic():
            log.info("[AutoTrade] VIXパニックモード — 全エントリー停止")
            return []

        # QQQレジーム判定（ブリッシュ比率ベース）
        self._update_market_regime()
        regime = self._qqq_regime       # "bullish" / "bearish" / "gray" / None
        ratio = self._qqq_bullish_ratio  # 0.0〜1.0

        # AUTO_SYMBOLS の候補を補充
        self.refresh_auto_symbols()

        new_positions = []

        # --- カナリア戦略: 警戒モード中にRETAIL_FAVORITES対象 ---
        if (
            config.CANARY_ENABLED
            and self._qqq_alert_mode
            and self.open_position_count < config.MAX_POSITIONS
        ):
            canary_pos = self._try_canary_entries()
            new_positions.extend(canary_pos)

        # --- 通常候補のエントリー ---
        for candidate in list(self.candidates):
            if self.open_position_count >= config.MAX_POSITIONS:
                break

            position = self._try_entry(candidate, regime, ratio)
            if position is not None:
                new_positions.append(position)
                self.candidates = [c for c in self.candidates if c.symbol != candidate.symbol]

        return new_positions

    def _try_entry(
        self, candidate: TradeCandidate,
        regime: str | None = None,
        ratio: float = 0.5,
    ) -> AutoPosition | None:
        """単一候補に対してエントリーを試行する。

        グレーゾーン（45〜55%）: ロング・ショートともに制限。
        ただし弱気ゾーン（40〜50%）＋ VWAP下抜け → ショート許可。
        """
        symbol = candidate.symbol

        # 決算ブラックアウトチェック
        if self._check_earnings_blackout(symbol):
            log.debug(f"{symbol} 決算ブラックアウト中 — スキップ")
            return None

        # 1分足・5分足を取得
        df_1min = self._fetch_1min_bars(symbol)
        if df_1min is None or df_1min.empty:
            return None
        df_5min = self._fetch_5min_bars(symbol)

        current_price = float(df_1min["close"].astype(float).iloc[-1])

        # 日足データ取得（50SMA + ATR計算用、50本以上必要）
        daily_df = self._fetch_daily_bars(symbol, limit=60)

        # 日足50SMAフィルター
        above_sma50, sma50_dev = check_above_daily_sma50(daily_df, current_price)

        # グレーゾーンでのSL縮小率
        in_gray_zone = (
            config.QQQ_GRAY_ZONE_LOW <= ratio <= config.QQQ_GRAY_ZONE_HIGH
        )
        sl_tighten = config.GRAY_ZONE_SL_TIGHTEN_PCT if in_gray_zone else 0.0

        # --- bullish: ロング戦略 ---
        if regime == "bullish":
            if above_sma50:
                signal = check_bullish_reversal_1min(
                    df_1min,
                    candidate.support_levels,
                    proximity_threshold=config.ENTRY_PROXIMITY_THRESHOLD,
                    rsi_threshold=config.ENTRY_RSI_THRESHOLD,
                )
                if signal is not None:
                    return self._execute_normal_entry(
                        candidate, signal, df_5min, daily_df, sma50_dev,
                        sl_tighten=sl_tighten,
                    )
            else:
                log.debug(f"{symbol} daily 50SMA below — 通常戦略スキップ")

            if config.CRASH_BOUNCE_ENABLED:
                result = self._try_crash_bounce_entry(
                    candidate, df_1min, df_5min, sma50_dev,
                )
                if result is not None:
                    return result

        # --- gray: グレーゾーン → 原則エントリー停止 ---
        elif regime == "gray":
            log.debug(
                f"{symbol} グレーゾーン(ratio={ratio:.0%}) — エントリー制限中"
            )
            # ただし弱気寄り（40〜50%）+ VWAP下抜け → ショート許可
            if (
                config.SHORT_ENABLED
                and config.QQQ_WEAK_BEAR_LOW <= ratio <= config.QQQ_WEAK_BEAR_HIGH
            ):
                below_vwap = self._is_below_vwap(symbol, df_5min, current_price)
                if below_vwap:
                    log.info(
                        f"{symbol} グレーゾーンだがVWAP下抜け — ショート試行"
                    )
                    return self._try_short_entry(
                        candidate, df_1min, df_5min, daily_df, sma50_dev,
                        sl_tighten=sl_tighten,
                    )

        # --- bearish: ショート戦略 ---
        elif regime == "bearish" and config.SHORT_ENABLED:
            return self._try_short_entry(
                candidate, df_1min, df_5min, daily_df, sma50_dev,
            )

        # --- regime is None (フィルター無効): 従来ロジック ---
        elif regime is None:
            if above_sma50:
                signal = check_bullish_reversal_1min(
                    df_1min,
                    candidate.support_levels,
                    proximity_threshold=config.ENTRY_PROXIMITY_THRESHOLD,
                    rsi_threshold=config.ENTRY_RSI_THRESHOLD,
                )
                if signal is not None:
                    return self._execute_normal_entry(
                        candidate, signal, df_5min, daily_df, sma50_dev,
                    )
            if config.CRASH_BOUNCE_ENABLED:
                result = self._try_crash_bounce_entry(
                    candidate, df_1min, df_5min, sma50_dev,
                )
                if result is not None:
                    return result

        return None

    def _is_below_vwap(
        self, symbol: str, df_5min: pd.DataFrame | None, current_price: float
    ) -> bool:
        """個別銘柄がVWAPを下回っているかチェックする。"""
        if df_5min is None or df_5min.empty:
            return False
        try:
            vwap_series = calc_running_vwap(df_5min)
            if vwap_series.empty or pd.isna(vwap_series.iloc[-1]):
                return False
            vwap = float(vwap_series.iloc[-1])
            return current_price < vwap
        except Exception:
            return False

    def _execute_normal_entry(
        self, candidate: TradeCandidate, signal: dict,
        df_5min: pd.DataFrame | None = None,
        daily_df: pd.DataFrame | None = None,
        sma50_deviation: float = 0.0,
        sl_tighten: float = 0.0,
    ) -> AutoPosition | None:
        """通常戦略のエントリーを実行する。"""
        symbol = candidate.symbol

        log.info(
            f"エントリーシグナル検出: {symbol} "
            f"price=${signal['price']:.2f} RSI={signal['rsi']} "
            f"support={signal['support_name']}(${signal['support_price']:.2f}) "
            f"SMA50乖離={sma50_deviation:+.2f}%"
        )

        current_price = signal["price"]

        # 日足ATR → 利確
        atr_daily = calc_atr_3day(daily_df) if daily_df is not None else None

        if atr_daily is None:
            take_profit_pct = config.TAKE_PROFIT_MIN
        else:
            take_profit_pct = calc_dynamic_take_profit(
                atr_daily, current_price,
                tp_min=config.TAKE_PROFIT_MIN,
                tp_max=config.TAKE_PROFIT_MAX,
            )

        # 損切り: 5分足 ATR(14) × 2.0、上限 2%
        atr_5m = calc_atr_5min(df_5min)
        max_stop = current_price * config.STOP_LOSS_MAX_PCT
        if atr_5m is not None:
            stop_distance = min(atr_5m * config.STOP_LOSS_ATR_MULT, max_stop)
            atr_pct = (atr_5m / current_price) * 100
        else:
            stop_distance = max_stop
            atr_pct = 0.0

        # グレーゾーンでのSL縮小
        if sl_tighten > 0:
            stop_distance *= (1 - sl_tighten)

        # リスクベースのポジションサイズ（損切り幅から逆算）
        qty = self._calc_position_qty(current_price, stop_distance)

        take_profit_price = current_price * (1 + take_profit_pct)
        stop_loss_price = current_price - stop_distance

        return self._submit_entry_order(
            symbol=symbol,
            current_price=current_price,
            qty=qty,
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
            stop_loss_price=stop_loss_price,
            support_level=signal["support_price"],
            support_name=signal["support_name"],
            atr_pct=atr_pct,
            sma50_deviation=sma50_deviation,
            strategy="normal",
        )

    def _try_crash_bounce_entry(
        self, candidate: TradeCandidate, df_1min: pd.DataFrame,
        df_5min: pd.DataFrame | None = None,
        sma50_deviation: float = 0.0,
    ) -> AutoPosition | None:
        """急落リバウンド戦略のエントリーを試行する。"""
        symbol = candidate.symbol

        # 日足を取得（スイングロー検出に60日以上必要）
        daily_df = self._fetch_daily_bars(
            symbol, limit=config.CRASH_SWING_LOW_LOOKBACK_DAYS + 10
        )
        if daily_df is None or daily_df.empty:
            return None

        # 現在価格と当日出来高を1分足から取得
        current_price = float(df_1min["close"].astype(float).iloc[-1])
        current_volume = float(df_1min["volume"].astype(float).sum())

        # 急落リバウンド条件チェック（日足ベース）
        crash_signal = check_crash_bounce(daily_df, current_price, current_volume)
        if crash_signal is None:
            return None

        # 1分足で陽線確認のみ（RSI条件は不要）
        last_open = float(df_1min["open"].iloc[-1])
        last_close = float(df_1min["close"].iloc[-1])
        if last_close <= last_open:
            return None

        log.info(
            f"急落リバウンドシグナル検出: {symbol} "
            f"price=${current_price:.2f} "
            f"drop={crash_signal['drop_pct']*100:.1f}% "
            f"swing_low=${crash_signal['swing_low_price']:.2f} "
            f"vol_ratio={crash_signal['volume_ratio']:.1f}x "
            f"recent_high=${crash_signal['recent_high']:.2f}"
        )

        # 急落リバウンド用エグジット設定（通常より広め）
        take_profit_pct = config.CRASH_TAKE_PROFIT_PCT
        take_profit_price = current_price * (1 + take_profit_pct)

        # 損切り: 5分足 ATR(14) × 2.0、上限 2%
        atr_5m = calc_atr_5min(df_5min)
        max_stop = current_price * config.STOP_LOSS_MAX_PCT
        if atr_5m is not None:
            stop_distance = min(atr_5m * config.STOP_LOSS_ATR_MULT, max_stop)
            atr_pct = (atr_5m / current_price) * 100
        else:
            stop_distance = max_stop
            atr_pct = 0.0
        stop_loss_price = current_price - stop_distance

        # リスクベースのポジションサイズ（損切り幅から逆算）
        qty = self._calc_position_qty(current_price, stop_distance)

        support_name = f"crash_bounce:swing_low_${crash_signal['swing_low_price']:.2f}"

        return self._submit_entry_order(
            symbol=symbol,
            current_price=current_price,
            qty=qty,
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
            stop_loss_price=stop_loss_price,
            support_level=crash_signal["swing_low_price"],
            support_name=support_name,
            atr_pct=atr_pct,
            sma50_deviation=sma50_deviation,
        )

    def _try_short_entry(
        self, candidate: TradeCandidate, df_1min: pd.DataFrame,
        df_5min: pd.DataFrame | None = None,
        daily_df: pd.DataFrame | None = None,
        sma50_deviation: float = 0.0,
        sl_tighten: float = 0.0,
    ) -> AutoPosition | None:
        """ショート（空売り）エントリーを試行する。"""
        symbol = candidate.symbol

        # 空売り可能性を事前確認
        if not self._check_shortable(symbol):
            log.debug(f"[Short] {symbol} 空売り不可 — スキップ")
            return None

        current_price = float(df_1min["close"].astype(float).iloc[-1])

        # レジスタンスレベル計算
        resistance_levels = {}

        # 前日OHLCからR1, R2, prev_high
        if daily_df is not None and len(daily_df) >= 2:
            today = pd.Timestamp.now(tz="UTC").normalize()
            prev_data = daily_df[daily_df.index.normalize() < today]
            if prev_data.empty:
                prev_data = daily_df
            prev_bar = prev_data.iloc[-1]
            resistance_levels.update(calc_resistance_levels(prev_bar))

        # キリ番（上方のみ）
        psych = calc_psychological_levels(current_price)
        for name, level in psych.items():
            if level >= current_price:
                resistance_levels[name] = level

        # SMA50（価格より上なら抵抗線）
        if daily_df is not None:
            close = daily_df["close"].astype(float)
            if len(close) >= 50:
                sma50_val = float(close.rolling(50).mean().iloc[-1])
                if sma50_val >= current_price:
                    resistance_levels["sma50"] = round(sma50_val, 4)

        # ベアリッシュリバーサルシグナル判定
        signal = check_bearish_reversal_1min(
            df_1min,
            resistance_levels,
            proximity_threshold=config.SHORT_PROXIMITY_THRESHOLD,
            rsi_threshold=config.SHORT_RSI_THRESHOLD,
        )
        if signal is None:
            return None

        # 改善4: 確認バー（直前バーも陰線であることを確認）
        if config.SHORT_CONFIRM_PREV_BEARISH and len(df_1min) >= 2:
            prev_open = float(df_1min["open"].iloc[-2])
            prev_close = float(df_1min["close"].iloc[-2])
            if prev_close >= prev_open:
                log.debug(f"[Short] {symbol} 直前バーが陽線 — 確認失敗")
                return None

        return self._execute_short_entry(
            candidate, signal, df_5min, daily_df, sma50_deviation,
            sl_tighten=sl_tighten,
        )

    def _execute_short_entry(
        self, candidate: TradeCandidate, signal: dict,
        df_5min: pd.DataFrame | None = None,
        daily_df: pd.DataFrame | None = None,
        sma50_deviation: float = 0.0,
        sl_tighten: float = 0.0,
    ) -> AutoPosition | None:
        """ショート戦略のエントリーを実行する。"""
        symbol = candidate.symbol

        log.info(
            f"ショートエントリーシグナル検出: {symbol} "
            f"price=${signal['price']:.2f} RSI={signal['rsi']} "
            f"resistance={signal['resistance_name']}(${signal['resistance_price']:.2f}) "
            f"SMA50乖離={sma50_deviation:+.2f}%"
        )

        current_price = signal["price"]

        # 日足ATR → 利確（ショートなので下方）
        atr_daily = calc_atr_3day(daily_df) if daily_df is not None else None

        if atr_daily is None:
            take_profit_pct = config.TAKE_PROFIT_MIN
        else:
            take_profit_pct = calc_dynamic_take_profit(
                atr_daily, current_price,
                tp_min=config.TAKE_PROFIT_MIN,
                tp_max=config.TAKE_PROFIT_MAX,
            )

        # 損切り: 5分足 ATR(14) × 2.0、上限 2%（ショートは上方）
        atr_5m = calc_atr_5min(df_5min)
        max_stop = current_price * config.STOP_LOSS_MAX_PCT
        if atr_5m is not None:
            stop_distance = min(atr_5m * config.STOP_LOSS_ATR_MULT, max_stop)
            atr_pct = (atr_5m / current_price) * 100
        else:
            stop_distance = max_stop
            atr_pct = 0.0

        # グレーゾーンでのSL縮小
        if sl_tighten > 0:
            stop_distance *= (1 - sl_tighten)

        # リスクベースのポジションサイズ
        qty = self._calc_position_qty(current_price, stop_distance)

        # TP = 下方、SL = 上方
        take_profit_price = current_price * (1 - take_profit_pct)
        stop_loss_price = current_price + stop_distance

        return self._submit_entry_order(
            symbol=symbol,
            current_price=current_price,
            qty=qty,
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
            stop_loss_price=stop_loss_price,
            support_level=signal["resistance_price"],
            support_name=signal["resistance_name"],
            atr_pct=atr_pct,
            sma50_deviation=sma50_deviation,
            side="short",
            strategy="short",
        )

    # ----------------------------------------------------------
    #  カナリア戦略（Canary in the Coal Mine）
    # ----------------------------------------------------------
    def _try_canary_entries(self) -> list[AutoPosition]:
        """警戒モード中にRETAIL_FAVORITES高ベータ銘柄をショートする。

        条件:
          1. VWAP下抜け
          2. 銘柄の下落率がQQQの下落率を上回る（最小乖離フィルター）
          3. RSI(14)が60以上から50以下に急落（RSI下限ガードあり）
        制限:
          - カナリア同時ポジション上限（CANARY_MAX_POSITIONS）
          - 引けN分前はエントリー禁止（CANARY_ENTRY_CUTOFF_MINUTES）
        """
        # 時間制限: 引けN分前はカナリアエントリー禁止
        now_et = datetime.now(ET)
        market_close_dt = datetime.combine(
            now_et.date(),
            time(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE),
            tzinfo=ET,
        )
        canary_cutoff = market_close_dt - timedelta(
            minutes=config.CANARY_ENTRY_CUTOFF_MINUTES
        )
        if now_et >= canary_cutoff:
            log.debug("[Canary] 引け前カットオフ — エントリー停止")
            return []

        # カナリア同時ポジション上限
        canary_open = sum(
            1 for p in self.positions
            if not p.closed and p.strategy == "canary"
        )
        if canary_open >= config.CANARY_MAX_POSITIONS:
            log.debug(f"[Canary] 同時ポジション上限({config.CANARY_MAX_POSITIONS})到達")
            return []

        new_positions = []
        open_symbols = {p.symbol for p in self.positions if not p.closed}

        for symbol in config.RETAIL_FAVORITES:
            if self.open_position_count >= config.MAX_POSITIONS:
                break
            if canary_open + len(new_positions) >= config.CANARY_MAX_POSITIONS:
                break
            if symbol in open_symbols:
                continue

            try:
                pos = self._try_single_canary(symbol)
                if pos is not None:
                    new_positions.append(pos)
                    open_symbols.add(symbol)
            except Exception as e:
                log.debug(f"[Canary] {symbol} 失敗: {e}")

        return new_positions

    def _try_single_canary(self, symbol: str) -> AutoPosition | None:
        """単一銘柄のカナリア戦略エントリーを試行する。"""
        # 空売り可否
        if not self._check_shortable(symbol):
            return None

        df_5min = self._fetch_5min_bars(symbol, limit=30)
        if df_5min is None or df_5min.empty:
            return None

        close = df_5min["close"].astype(float)
        current_price = float(close.iloc[-1])

        # 条件1: VWAP下抜け
        vwap_series = calc_running_vwap(df_5min)
        if vwap_series.empty or pd.isna(vwap_series.iloc[-1]):
            return None
        current_vwap = float(vwap_series.iloc[-1])
        if current_price >= current_vwap:
            return None

        # 条件2: 相対的弱さ（銘柄の下落率 > QQQの下落率）
        # 当日始値からのリターンで比較
        import pandas_ta as ta
        dates = df_5min.index.normalize()
        today = dates[-1]
        today_data = df_5min[dates == today]
        if today_data.empty:
            return None
        stock_open = float(today_data["open"].astype(float).iloc[0])
        stock_return = (current_price - stock_open) / stock_open

        # QQQの日中リターンと比較
        if self._qqq_intraday_return >= 0:
            return None  # QQQ上昇中は対象外
        if stock_return >= self._qqq_intraday_return:
            return None  # QQQより強い（弱さ不足）

        # 改善1: 最小乖離フィルター（QQQとの乖離が小さすぎる銘柄を除外）
        divergence = stock_return - self._qqq_intraday_return
        if abs(divergence) < config.CANARY_MIN_DIVERGENCE:
            log.debug(
                f"[Canary] {symbol} 乖離不足 "
                f"({divergence*100:+.2f}pp < {config.CANARY_MIN_DIVERGENCE*100:.1f}pp)"
            )
            return None

        # 条件3: RSI 60以上→50以下に急落（直近N本以内にRSI>=60、現在<50）
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is None or rsi_series.dropna().empty or len(rsi_series.dropna()) < 2:
            return None
        rsi_cur = float(rsi_series.dropna().iloc[-1])

        if rsi_cur >= config.CANARY_RSI_ENTRY_LOW:
            return None

        # 改善2: RSI下限ガード（売られすぎの銘柄は反発リスクが高い）
        if rsi_cur < config.CANARY_RSI_FLOOR:
            log.debug(
                f"[Canary] {symbol} RSI売られすぎ "
                f"(RSI={rsi_cur:.0f} < {config.CANARY_RSI_FLOOR})"
            )
            return None

        # 直近N本のRSI最大値が閾値以上か
        lookback = config.CANARY_RSI_LOOKBACK
        rsi_window = rsi_series.dropna().iloc[-(lookback + 1):-1]
        if rsi_window.empty or float(rsi_window.max()) < config.CANARY_RSI_ENTRY_HIGH:
            return None
        rsi_prev = float(rsi_window.max())

        log.info(
            f"[Canary] シグナル検出: {symbol} "
            f"price=${current_price:.2f} VWAP=${current_vwap:.2f} "
            f"stock_ret={stock_return*100:+.2f}% vs QQQ={self._qqq_intraday_return*100:+.2f}% "
            f"RSI={rsi_prev:.0f}→{rsi_cur:.0f}"
        )

        # TP 1%, SL 0.5%
        tp_price = current_price * (1 - config.CANARY_TAKE_PROFIT_PCT)
        sl_price = current_price * (1 + config.CANARY_STOP_LOSS_PCT)

        # リスクベースのポジションサイズ（損切り幅から逆算）
        stop_distance = current_price * config.CANARY_STOP_LOSS_PCT
        qty = self._calc_position_qty(current_price, stop_distance)

        return self._submit_entry_order(
            symbol=symbol,
            current_price=current_price,
            qty=qty,
            take_profit_price=tp_price,
            take_profit_pct=config.CANARY_TAKE_PROFIT_PCT,
            stop_loss_price=sl_price,
            support_level=current_vwap,
            support_name="canary:vwap",
            atr_pct=0.0,
            sma50_deviation=0.0,
            side="short",
            strategy="canary",
            vwap_at_entry=current_vwap,
        )

    # ----------------------------------------------------------
    #  注文送信
    # ----------------------------------------------------------
    def _submit_entry_order(
        self,
        *,
        symbol: str,
        current_price: float,
        qty: int,
        take_profit_price: float,
        take_profit_pct: float,
        stop_loss_price: float,
        support_level: float,
        support_name: str,
        atr_pct: float = 0.0,
        sma50_deviation: float = 0.0,
        side: str = "long",
        strategy: str = "",
        vwap_at_entry: float = 0.0,
    ) -> AutoPosition | None:
        """成行注文を送信し AutoPosition を返す。逆指値SL注文も同時送信（ブラケット）。"""
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        side_label = "買い" if side == "long" else "売り（空売り）"

        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
        except Exception as e:
            log.error(f"{symbol} {side_label}注文送信失敗: {e}")
            return None

        # ブラケット注文: 逆指値SL注文を同時送信（ロング=SELL STOP / ショート=BUY STOP）
        stop_order_id = ""
        stop_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        stop_label = "SELL STOP" if side == "long" else "BUY STOP"
        try:
            stop_request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=stop_side,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_loss_price, 2),
            )
            stop_order = self.trading_client.submit_order(stop_request)
            stop_order_id = str(stop_order.id)
            log.info(
                f"  逆指値SL注文送信: {symbol} {stop_label} "
                f"${stop_loss_price:.2f} (ID={stop_order_id})"
            )
        except Exception as e:
            log.warning(f"{symbol} 逆指値SL注文失敗（手動SL監視で継続）: {e}")

        position = AutoPosition(
            symbol=symbol,
            entry_price=current_price,
            qty=qty,
            order_id=str(order.id),
            take_profit_price=round(take_profit_price, 2),
            stop_loss_price=round(stop_loss_price, 2),
            support_level=support_level,
            support_name=support_name,
            atr_pct=round(atr_pct, 2),
            sma50_deviation=round(sma50_deviation, 2),
            side=side,
            strategy=strategy,
            stop_order_id=stop_order_id,
            vwap_at_entry=round(vwap_at_entry, 2),
            highest_price=current_price if side == "long" else 0.0,
            lowest_price=current_price if side == "short" else 0.0,
        )
        self.positions.append(position)

        log.info(
            f"{side_label}注文実行: {symbol} [{strategy}] qty={qty} "
            f"entry=${current_price:.2f} "
            f"TP=${take_profit_price:.2f}({take_profit_pct*100:.1f}%) "
            f"SL=${stop_loss_price:.2f}"
        )
        return position

    # ----------------------------------------------------------
    #  ポジション監視・決済
    # ----------------------------------------------------------
    def check_exits(self) -> list[AutoPosition]:
        """オープンポジションのエグジット条件をチェックし、該当があれば決済する。"""
        closed_positions = []

        # 市場クローズ N 分前なら全ポジション決済
        now_et = datetime.now(ET)
        close_time = time(config.MARKET_CLOSE_HOUR, config.MARKET_CLOSE_MINUTE)
        close_dt = datetime.combine(now_et.date(), close_time, tzinfo=ET)
        eod_threshold = close_dt - timedelta(minutes=config.EOD_CLOSE_MINUTES_BEFORE)
        if now_et >= eod_threshold and now_et < close_dt:
            log.info("[AutoTrade] 市場クローズ前 — 全ポジション決済開始")
            return self.close_all_positions()

        for position in self.positions:
            if position.closed:
                continue

            current_price = self._get_latest_price(position.symbol)
            if current_price is None:
                continue

            # トレーリングストップを更新
            position.update_trailing_stop(current_price)

            # エグジット条件チェック
            exit_reason = position.check_exit(current_price)
            if exit_reason is None:
                continue

            # 決済注文を送信
            success = self._close_position(position, current_price, exit_reason)
            if success:
                closed_positions.append(position)

        return closed_positions

    def _close_position(
        self, position: AutoPosition, current_price: float, reason: str
    ) -> bool:
        """ポジションを決済する。"""
        # ショートの逆指値SL注文が残っていればキャンセル
        if position.stop_order_id:
            try:
                self.trading_client.cancel_order_by_id(position.stop_order_id)
                log.info(f"  逆指値SL注文キャンセル: {position.stop_order_id}")
            except Exception as e:
                log.debug(f"  逆指値SL注文キャンセル失敗（既に約定の可能性）: {e}")

        # ロング決済=SELL、ショート決済=BUY（買い戻し）
        close_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY

        try:
            order_request = MarketOrderRequest(
                symbol=position.symbol,
                qty=position.qty,
                side=close_side,
                time_in_force=TimeInForce.DAY,
            )
            self.trading_client.submit_order(order_request)
        except Exception as e:
            log.error(f"{position.symbol} 決済注文失敗: {e}")
            return False

        position.closed = True
        position.close_reason = reason
        position.close_price = current_price

        # PnL: ロング (close-entry)*qty、ショート (entry-close)*qty
        if position.side == "short":
            position.pnl = (position.entry_price - current_price) * position.qty
        else:
            position.pnl = (current_price - position.entry_price) * position.qty

        side_label = "ショート決済" if position.side == "short" else "決済"
        log.info(
            f"{side_label}実行: {position.symbol} reason={reason} "
            f"entry=${position.entry_price:.2f} exit=${current_price:.2f} "
            f"PnL=${position.pnl:+.2f}"
        )
        return True

    # ----------------------------------------------------------
    #  Discord 通知
    # ----------------------------------------------------------
    def build_entry_embed(self, position: AutoPosition) -> discord.Embed:
        """エントリー通知用の Embed を作成する。"""
        is_short = position.side == "short"
        side_label = "ショート" if is_short else "ロング"
        color = discord.Color.orange() if is_short else discord.Color.blue()
        price_label = "売値" if is_short else "買値"
        level_label = "レジスタンス" if is_short else "サポート"

        embed = discord.Embed(
            title=f"🤖 自動売買: {position.symbol} [{side_label}] エントリー",
            color=color,
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name=price_label, value=f"${position.entry_price:.2f}", inline=True)
        embed.add_field(name="数量", value=str(position.qty), inline=True)
        embed.add_field(
            name="金額",
            value=f"${position.entry_price * position.qty:.2f}",
            inline=True,
        )
        embed.add_field(
            name="利確",
            value=f"${position.take_profit_price:.2f}",
            inline=True,
        )
        embed.add_field(
            name="損切り",
            value=f"${position.stop_loss_price:.2f}",
            inline=True,
        )
        embed.add_field(
            name=level_label,
            value=f"{position.support_name} (${position.support_level:.2f})",
            inline=True,
        )
        embed.add_field(
            name="トレーリング",
            value=f"+{config.TRAILING_ACTIVATE_PCT*100:.1f}%発動→{config.TRAILING_RETURN_PCT*100:.1f}%幅",
            inline=True,
        )
        embed.add_field(
            name="ATR%",
            value=f"{position.atr_pct:.1f}%",
            inline=True,
        )
        embed.add_field(
            name="SMA50乖離",
            value=f"{position.sma50_deviation:+.2f}%",
            inline=True,
        )

        # 市場レジーム情報
        regime_parts = []
        if self._qqq_regime:
            regime_parts.append(f"QQQ={self._qqq_regime}")
        if self._vix_panic:
            regime_parts.append("VIX=パニック")
        if regime_parts:
            embed.add_field(
                name="市場レジーム",
                value=" / ".join(regime_parts),
                inline=True,
            )

        embed.add_field(name="注文ID", value=position.order_id, inline=False)
        embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Auto Trading")
        return embed

    def build_exit_embed(self, position: AutoPosition) -> discord.Embed:
        """エグジット通知用の Embed を作成する。"""
        is_short = position.side == "short"
        pnl = position.pnl

        # PnL% の計算（side対応）
        if is_short:
            pnl_pct = ((position.entry_price - position.close_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.close_price - position.entry_price) / position.entry_price) * 100

        side_label = "ショート" if is_short else "ロング"

        if pnl >= 0:
            color = discord.Color.green()
            title = f"💰 自動売買: {position.symbol} [{side_label}] 利確"
        else:
            color = discord.Color.red()
            title = f"🔻 自動売買: {position.symbol} [{side_label}] 損切り"

        reason_map = {
            "take_profit": "利確ライン到達",
            "stop_loss": "損切りライン到達",
            "support_break": f"サポート割れ ({position.support_name})",
            "resistance_break": f"レジスタンス上抜け ({position.support_name})",
            "trailing_stop": "トレーリングストップ",
            "vwap_cross": "VWAP上抜け（カナリア）",
        }

        # ラベル
        entry_label = "売値（空売り）" if is_short else "買値"
        exit_label = "買戻し" if is_short else "売値"
        extremum_label = "最安値" if is_short else "最高値"
        extremum_val = position.lowest_price if is_short else position.highest_price

        embed = discord.Embed(
            title=title,
            color=color,
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name=entry_label, value=f"${position.entry_price:.2f}", inline=True)
        embed.add_field(name=exit_label, value=f"${position.close_price:.2f}", inline=True)
        embed.add_field(name="PnL", value=f"${pnl:+.2f} ({pnl_pct:+.1f}%)", inline=True)
        embed.add_field(
            name="決済理由",
            value=reason_map.get(position.close_reason, position.close_reason),
            inline=True,
        )
        embed.add_field(name="数量", value=str(position.qty), inline=True)
        embed.add_field(
            name=extremum_label,
            value=f"${extremum_val:.2f}",
            inline=True,
        )
        embed.add_field(
            name="ATR%",
            value=f"{position.atr_pct:.1f}%",
            inline=True,
        )
        embed.add_field(
            name="SMA50乖離",
            value=f"{position.sma50_deviation:+.2f}%",
            inline=True,
        )
        embed.set_footer(text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Auto Trading")
        return embed

    def build_status_summary(self) -> str:
        """現在のポジション状況のサマリーテキストを返す。"""
        open_pos = [p for p in self.positions if not p.closed]
        if not open_pos:
            return "オープンポジション: なし"

        lines = [f"オープンポジション: {len(open_pos)}/{config.MAX_POSITIONS}"]
        for p in open_pos:
            price = self._get_latest_price(p.symbol)
            side_tag = "S" if p.side == "short" else "L"
            if price:
                if p.side == "short":
                    pnl = (p.entry_price - price) * p.qty
                else:
                    pnl = (price - p.entry_price) * p.qty
                lines.append(
                    f"  [{side_tag}] {p.symbol}: ${p.entry_price:.2f}→${price:.2f} "
                    f"PnL=${pnl:+.2f} TS=${p.trailing_stop_price:.2f}"
                )
            else:
                lines.append(f"  [{side_tag}] {p.symbol}: ${p.entry_price:.2f} (価格取得不可)")
        return "\n".join(lines)

    # ----------------------------------------------------------
    #  全決済（シャットダウン時）
    # ----------------------------------------------------------
    def close_all_positions(self) -> list[AutoPosition]:
        """全オープンポジションを成行で決済する。"""
        closed = []
        for position in self.positions:
            if position.closed:
                continue

            current_price = self._get_latest_price(position.symbol)
            if current_price is None:
                current_price = position.entry_price  # フォールバック

            success = self._close_position(position, current_price, "end_of_day")
            if success:
                closed.append(position)

        return closed

    # ----------------------------------------------------------
    #  日次サマリー Embed（シャットダウン時）
    # ----------------------------------------------------------
    def build_daily_summary_embed(self) -> discord.Embed:
        """本日の全取引履歴と収支サマリーの Embed を作成する。"""
        closed_pos = [p for p in self.positions if p.closed]
        total_pnl = sum(p.pnl for p in closed_pos)
        wins = [p for p in closed_pos if p.pnl >= 0]
        losses = [p for p in closed_pos if p.pnl < 0]

        if total_pnl >= 0:
            color = discord.Color.green()
        else:
            color = discord.Color.red()

        embed = discord.Embed(
            title="📋 本日の取引サマリー",
            color=color,
            timestamp=datetime.now(timezone.utc),
        )

        # 収支サマリー
        win_rate = (len(wins) / len(closed_pos) * 100) if closed_pos else 0
        summary_lines = [
            f"総取引数: **{len(closed_pos)}**",
            f"勝ち: **{len(wins)}** / 負け: **{len(losses)}**",
            f"勝率: **{win_rate:.0f}%**",
            f"総損益: **${total_pnl:+.2f}**",
        ]
        embed.add_field(
            name="💰 収支",
            value="\n".join(summary_lines),
            inline=False,
        )

        # 取引履歴
        if closed_pos:
            reason_map = {
                "take_profit": "利確",
                "stop_loss": "損切り",
                "support_break": "サポート割れ",
                "resistance_break": "レジスタンス上抜け",
                "trailing_stop": "トレーリング",
                "vwap_cross": "VWAP上抜け",
                "end_of_day": "終了時決済",
            }
            history_lines = []
            for p in closed_pos:
                if p.side == "short":
                    pnl_pct = ((p.entry_price - p.close_price) / p.entry_price) * 100
                else:
                    pnl_pct = ((p.close_price - p.entry_price) / p.entry_price) * 100
                reason = reason_map.get(p.close_reason, p.close_reason)
                side_tag = "S" if p.side == "short" else "L"
                history_lines.append(
                    f"[{side_tag}] **{p.symbol}** x{p.qty}: "
                    f"${p.entry_price:.2f} → ${p.close_price:.2f} "
                    f"(**${p.pnl:+.2f}** / {pnl_pct:+.1f}%) [{reason}]"
                )
            # Discord Embed のフィールドは1024文字制限があるので分割
            history_text = "\n".join(history_lines)
            if len(history_text) > 1024:
                history_text = history_text[:1020] + "..."
            embed.add_field(
                name="📜 取引履歴",
                value=history_text,
                inline=False,
            )
        else:
            embed.add_field(
                name="📜 取引履歴",
                value="本日の自動売買取引はありませんでした",
                inline=False,
            )

        embed.set_footer(
            text=f"{'🟢 Paper' if config.ALPACA_PAPER else '🔴 Live'} Auto Trading"
        )
        return embed
