"""自動売買システムのデータモデル"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import config


@dataclass
class TradeCandidate:
    """スクリーナーが選定した売買候補銘柄。"""

    symbol: str
    atr_pct: float  # 3日ATR / 現在価格 (%)
    current_price: float
    support_levels: dict = field(default_factory=dict)  # {name: price}
    is_crash_bounce: bool = False  # 急落リバウンド戦略の候補
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """候補の有効期限チェック。"""
        age = (datetime.now(timezone.utc) - self.added_at).total_seconds() / 3600
        return age > max_age_hours


@dataclass
class AutoPosition:
    """自動売買でオープン中のポジション。"""

    symbol: str
    entry_price: float
    qty: int
    order_id: str
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # エグジット設定
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    support_level: float = 0.0  # エントリーのトリガーとなったサポート価格
    support_name: str = ""

    # ボラティリティ
    atr_pct: float = 0.0    # ATR / 価格 (%)
    atr_value: float = 0.0  # 5分足ATR(14)のドル値（トレーリング幅計算用）

    # 日足50SMA乖離率
    sma50_deviation: float = 0.0  # 50SMAからの乖離率 (%)

    # ロング / ショート
    side: str = "long"  # "long" or "short"
    strategy: str = ""  # "normal", "crash_bounce", "short", "canary"

    # ショート用: 逆指値ストップ注文ID（ブラケット）
    stop_order_id: str = ""

    # カナリア戦略用: VWAP上抜けで損切り
    vwap_at_entry: float = 0.0

    # トレーリングストップ（ATRベース: ATR×1.5で発動 → ATR×2.0幅で追従）
    trailing_activated: bool = False
    highest_price: float = 0.0  # ポジション保有中の最高値（ロング用）
    lowest_price: float = 0.0   # ポジション保有中の最安値（ショート用）
    trailing_stop_price: float = 0.0  # 現在のトレーリングストップ価格

    # ステータス
    closed: bool = False
    close_reason: str = ""
    close_price: float = 0.0
    pnl: float = 0.0

    def update_trailing_stop(self, current_price: float) -> None:
        """現在価格に基づいてトレーリングストップを更新する。

        ATR × 1.5 の含み益でトレーリングストップが発動し、
        最高値（ロング）/最安値（ショート）から ATR × 2.0 幅で追従する。
        """
        if self.side == "short":
            self._update_trailing_stop_short(current_price)
        else:
            self._update_trailing_stop_long(current_price)

    def _update_trailing_stop_long(self, current_price: float) -> None:
        """ロング用トレーリングストップ（ブレイクアウト設定: ATR×5.0）。"""
        if current_price > self.highest_price:
            self.highest_price = current_price

        if not self.trailing_activated:
            if self.atr_value > 0:
                gain = current_price - self.entry_price
                if gain >= self.atr_value * config.TRAILING_ACTIVATE_ATR_MULT:
                    self.trailing_activated = True

        if self.trailing_activated and self.atr_value > 0:
            trail_width = self.atr_value * config.BREAKOUT_TRAILING_ATR_MULT
            new_stop = self.highest_price - trail_width
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop

    def _update_trailing_stop_short(self, current_price: float) -> None:
        """ショート用トレーリングストップ（ブレイクアウト設定: ATR×5.0）。"""
        if self.lowest_price <= 0 or current_price < self.lowest_price:
            self.lowest_price = current_price

        if not self.trailing_activated:
            if self.atr_value > 0:
                gain = self.entry_price - current_price
                if gain >= self.atr_value * config.TRAILING_ACTIVATE_ATR_MULT:
                    self.trailing_activated = True

        if self.trailing_activated and self.atr_value > 0:
            trail_width = self.atr_value * config.BREAKOUT_TRAILING_ATR_MULT
            new_stop = self.lowest_price + trail_width
            if self.trailing_stop_price <= 0 or new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop

    def check_exit(self, current_price: float, current_vwap: float = 0.0) -> Optional[str]:
        """エグジット条件をチェックし、該当する理由を返す。"""
        if self.side == "short":
            return self._check_exit_short(current_price, current_vwap)
        return self._check_exit_long(current_price)

    def _check_exit_long(self, current_price: float) -> Optional[str]:
        """ロング用エグジット判定。"""
        if self.take_profit_price > 0 and current_price >= self.take_profit_price:
            return "take_profit"
        if current_price <= self.stop_loss_price:
            return "stop_loss"
        if self.trailing_stop_price > 0 and current_price <= self.trailing_stop_price:
            return "trailing_stop"
        return None

    def _check_exit_short(self, current_price: float, current_vwap: float = 0.0) -> Optional[str]:
        """ショート用エグジット判定。"""
        if self.take_profit_price > 0 and current_price <= self.take_profit_price:
            return "take_profit"
        if current_price >= self.stop_loss_price:
            return "stop_loss"
        if self.trailing_stop_price > 0 and current_price >= self.trailing_stop_price:
            return "trailing_stop"
        return None
