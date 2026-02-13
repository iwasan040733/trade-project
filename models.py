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
    atr_pct: float = 0.0  # ATR / 価格 (%)

    # 日足50SMA乖離率
    sma50_deviation: float = 0.0  # 50SMAからの乖離率 (%)

    # ロング / ショート
    side: str = "long"  # "long" or "short"
    strategy: str = ""  # "normal", "crash_bounce", "short", "canary"

    # ショート用: 逆指値ストップ注文ID（ブラケット）
    stop_order_id: str = ""

    # カナリア戦略用: VWAP上抜けで損切り
    vwap_at_entry: float = 0.0

    # トレーリングストップ（+0.5%発動 → 0.3%幅で追従）
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

        +0.5% の含み益でトレーリングストップが発動し、
        最高値（ロング）/最安値（ショート）から 0.3% 戻った時点で決済される。
        """
        if self.side == "short":
            self._update_trailing_stop_short(current_price)
        else:
            self._update_trailing_stop_long(current_price)

    def _update_trailing_stop_long(self, current_price: float) -> None:
        """ロング用トレーリングストップ。"""
        if current_price > self.highest_price:
            self.highest_price = current_price

        if not self.trailing_activated:
            gain_pct = (current_price - self.entry_price) / self.entry_price
            if gain_pct >= config.TRAILING_ACTIVATE_PCT:
                self.trailing_activated = True

        if self.trailing_activated:
            new_stop = self.highest_price * (1 - config.TRAILING_RETURN_PCT)
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop

    def _update_trailing_stop_short(self, current_price: float) -> None:
        """ショート用トレーリングストップ（最安値追跡、上方のみ移動）。"""
        if self.lowest_price <= 0 or current_price < self.lowest_price:
            self.lowest_price = current_price

        if not self.trailing_activated:
            gain_pct = (self.entry_price - current_price) / self.entry_price
            if gain_pct >= config.TRAILING_ACTIVATE_PCT:
                self.trailing_activated = True

        if self.trailing_activated:
            new_stop = self.lowest_price * (1 + config.TRAILING_RETURN_PCT)
            # ショートは下方のみ移動（ストップ価格を下げる）
            if self.trailing_stop_price <= 0 or new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop

    def check_exit(self, current_price: float, current_vwap: float = 0.0) -> Optional[str]:
        """エグジット条件をチェックし、該当する理由を返す。

        Args:
            current_price: 現在価格
            current_vwap: 現在のVWAP（カナリア戦略のVWAP上抜け判定用）

        Returns:
            str: 'take_profit', 'stop_loss', 'support_break',
                 'resistance_break', 'trailing_stop', 'vwap_cross'
            None: エグジット条件に該当しない
        """
        if self.side == "short":
            return self._check_exit_short(current_price, current_vwap)
        return self._check_exit_long(current_price)

    def _check_exit_long(self, current_price: float) -> Optional[str]:
        """ロング用エグジット判定。"""
        if current_price >= self.take_profit_price:
            return "take_profit"
        if current_price <= self.stop_loss_price:
            return "stop_loss"
        if self.support_level > 0 and current_price < self.support_level:
            return "support_break"
        if self.trailing_stop_price > 0 and current_price <= self.trailing_stop_price:
            return "trailing_stop"
        return None

    def _check_exit_short(self, current_price: float, current_vwap: float = 0.0) -> Optional[str]:
        """ショート用エグジット判定（方向が逆）。"""
        # 利確: 価格が下がったら（TP は entry より下）
        if self.take_profit_price > 0 and current_price <= self.take_profit_price:
            return "take_profit"
        # 損切り: 価格が上がったら（SL は entry より上）
        if current_price >= self.stop_loss_price:
            return "stop_loss"
        # カナリア戦略: VWAP上抜けで損切り
        if self.strategy == "canary" and self.vwap_at_entry > 0 and current_vwap > 0:
            if current_price > current_vwap:
                return "vwap_cross"
        # レジスタンス上抜け（support_level をレジスタンスとして使用）
        if self.support_level > 0 and current_price > self.support_level:
            return "resistance_break"
        # トレーリングストップ: 価格が上がったら
        if self.trailing_stop_price > 0 and current_price >= self.trailing_stop_price:
            return "trailing_stop"
        return None
