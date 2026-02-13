import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#  Alpaca API
# ============================================================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = True  # True=ペーパートレード, False=本番

# ============================================================
#  Discord
# ============================================================
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
# Bot がメッセージを送信するチャンネルID（Discordの開発者モードで取得）
DISCORD_CHANNEL_ID = int(os.environ.get("DISCORD_CHANNEL_ID", "0"))

# ============================================================
#  監視設定（手動承認 — WATCHLIST銘柄向け）
# ============================================================
TARGET_SYMBOL = "NVDA"             # 後方互換用（WATCHLIST優先）
WATCHLIST = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN"]
CHECK_INTERVAL_SECONDS = 300       # 5分ごとに監視
PROXIMITY_THRESHOLD = 0.005        # 節目まで 0.5% 以内で通知
BUY_QTY = 1                        # 買い注文の株数（手動）

# ============================================================
#  自動売買設定
# ============================================================
ACCOUNT_SIZE = 6700                # 運用資金 $6,700（100万円相当）
RISK_PER_TRADE = 0.01              # 1トレードのリスク = 資金の 1%（$67）
MAX_POSITIONS = 3                  # 最大同時ポジション数
MAX_POSITION_PCT = 0.35            # 1銘柄の最大ポジション比率 35%

POSITION_SIZE = 1000               # バックテスト用（本番は RISK_PER_TRADE で自動計算）

# 市場時間（米国東部時間）— この時間外はエントリーしない
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
EOD_CLOSE_MINUTES_BEFORE = 30     # 市場クローズ 30 分前に全ポジション決済（15:30 ET）

# 利確 / 損切り
TAKE_PROFIT_MIN = 0.015            # 利確下限 1.5%
TAKE_PROFIT_MAX = 0.03             # 利確上限 3.0%
STOP_LOSS_ATR_MULT = 2.0           # 損切り = 5分足 ATR(14) × 2.0
STOP_LOSS_MAX_PCT = 0.02           # 損切り上限 2.0%（キャップ）
TRAILING_ACTIVATE_PCT = 0.005      # トレーリングストップ発動: +0.5% の含み益
TRAILING_RETURN_PCT = 0.003        # トレーリングストップ: 最高値から 0.3% 戻りで決済

# スクリーナー（取引開始1時間前に1回だけ実行 → 通知のみ、自動売買には使わない）
SCREENER_HOUR_ET = 8               # 実行時刻（米国東部時間）8:30 AM
SCREENER_MINUTE_ET = 30
SCREENER_TOP_N = 20                # MostActives/MarketMovers の取得件数
SCREENER_MIN_ATR_PCT = 1.5         # 候補に残す最低 ATR%（3日ATR/価格）
SCREENER_MIN_PRICE = 5.0           # 最低株価（ペニー銘柄除外）
SCREENER_MAX_PRICE = 500.0         # 最高株価
SCREENER_MIN_VOLUME = 1_000_000    # 最低出来高（直近日足）

# 個人投資家に人気の銘柄（スクリーニングで優先）
RETAIL_FAVORITES = [
    "ALAB",   # Astera Labs
    "PLTR",   # Palantir
    "SOFI",   # SoFi Technologies
    "HOOD",   # Robinhood
    "COIN",   # Coinbase
    "RIVN",   # Rivian
    "LCID",   # Lucid Motors
    "NIO",    # NIO
    "MARA",   # Marathon Digital
    "RIOT",   # Riot Platforms
    "SQ",     # Block (Square)
    "RBLX",   # Roblox
    "SNAP",   # Snap
    "ARM",    # ARM Holdings
    "SMCI",   # Super Micro Computer
    "IONQ",   # IonQ
    "RGTI",   # Rigetti Computing
    "SOUN",   # SoundHound AI
    "UPST",   # Upstart
    "AFRM",   # Affirm
]

# 注目テーマ株（手動更新 — 旬のテーマごとに銘柄を定義）
THEME_STOCKS = {
    "メモリ/HDD": ["SNDK", "WDC", "STX", "MU", "NAND"],
    "AI半導体": ["NVDA", "AMD", "AVGO", "MRVL", "ALAB"],
    "量子コンピュータ": ["IONQ", "RGTI", "QBTS"],
}

# テーマ株のATR%閾値を通常の半分に緩和
THEME_SCREENER_ATR_DISCOUNT = 0.5

# 自動売買ループ
AUTO_TRADE_INTERVAL_SECONDS = 30   # 30秒ごとに判定

# 自動監視銘柄（毎日売買代金上位10銘柄を自動取得。取得失敗時はフォールバック）
AUTO_SYMBOLS_COUNT = 10
AUTO_SYMBOLS_FETCH_TOP = 50            # MostActives から多めに取得
AUTO_SYMBOLS_MIN_PRICE = 20.0          # $20以上の銘柄のみ（小型・ペニー株除外）
AUTO_SYMBOLS_FALLBACK = ["NVDA", "TSLA", "AAPL", "AMD", "META", "MSFT", "GOOGL", "AMZN"]

# 決算ブラックアウト（決算発表前後は自動停止）
EARNINGS_BLACKOUT_HOURS = 24       # 決算前後 24 時間は取引停止

# エントリー時間制限（ノイズの多い寄り付き・引けを回避）
ENTRY_BUFFER_MINUTES_OPEN = 15    # 寄り付き後 15 分はエントリーしない（9:45 ET〜）
ENTRY_BUFFER_MINUTES_CLOSE = 30   # 引け前 30 分はエントリーしない（〜15:30 ET）

# エントリー条件
ENTRY_PROXIMITY_THRESHOLD = 0.005  # サポートまで 0.5% 以内
ENTRY_RSI_THRESHOLD = 35           # RSI がこの値以下から上昇に転じる

# ============================================================
#  急落リバウンド戦略（Crash & Bounce）
# ============================================================
CRASH_BOUNCE_ENABLED = True
CRASH_RECENT_HIGH_SMA_DEVIATION = 0.15  # 急騰判定: 高値が20SMAから15%以上乖離
CRASH_DROP_THRESHOLD = 0.15              # 急落判定: 直近高値から-15%
CRASH_DROP_LOOKBACK_DAYS = 5             # 急落を検出する期間
CRASH_SWING_LOW_LOOKBACK_DAYS = 60       # スイングロー探索期間
CRASH_SWING_LOW_PROXIMITY = 0.02         # スイングローへの接近閾値 2%
CRASH_VOLUME_MULTIPLIER = 2.0            # 出来高急増倍率
CRASH_TAKE_PROFIT_PCT = 0.05             # 利確 5%（通常より大きめ）
CRASH_STOP_LOSS_PCT = 0.03              # 損切り 3%（スイングロー割れ）

# ============================================================
#  市場レジーム判定（Market Regime Detection）
# ============================================================
QQQ_FILTER_ENABLED = True
QQQ_MA_PERIOD = 20                    # QQQ 5分足の移動平均期間
QQQ_CACHE_SECONDS = 60                # QQQ データのキャッシュ（秒）

# ブリッシュ比率（rolling window）
QQQ_BULLISH_RATIO_WINDOW = 20        # 直近20本（5分足≒1.5時間）
QQQ_GRAY_ZONE_LOW = 0.45             # グレーゾーン下限（45%）
QQQ_GRAY_ZONE_HIGH = 0.55            # グレーゾーン上限（55%）
QQQ_WEAK_BEAR_LOW = 0.40             # 弱気ゾーン下限（VWAP下抜けでショート許可）
QQQ_WEAK_BEAR_HIGH = 0.50            # 弱気ゾーン上限
GRAY_ZONE_SL_TIGHTEN_PCT = 0.20      # グレーゾーンでのSL縮小率 20%

VIX_PANIC_ENABLED = True
VIX_PANIC_THRESHOLD = 0.10            # VIX 前日比 +10% でパニックモード
VIX_SYMBOL = "^VIX"                   # yfinance 用シンボル
VIX_CACHE_SECONDS = 300               # VIX キャッシュ（5分）

# ============================================================
#  ショート戦略
# ============================================================
SHORT_ENABLED = True
SHORT_RSI_THRESHOLD = 65              # RSI >= 65 から下落で売りエントリー
SHORT_PROXIMITY_THRESHOLD = 0.005     # レジスタンスまで 0.5% 以内
SHORT_CONFIRM_PREV_BEARISH = True     # 直前バーも陰線であることを確認

# ============================================================
#  カナリア戦略（Canary in the Coal Mine）
# ============================================================
CANARY_ENABLED = True
CANARY_RSI_ENTRY_HIGH = 60           # RSI がこの値以上から
CANARY_RSI_ENTRY_LOW = 50            # この値以下に急落でエントリー
CANARY_RSI_LOOKBACK = 3              # RSI高値の探索バー数（5分足×3=15分以内）
CANARY_RSI_FLOOR = 35                # RSI下限（これ以下は売られすぎでスキップ）
CANARY_MIN_DIVERGENCE = 0.005        # QQQとの最小乖離（0.5pp以上弱い場合のみ）
CANARY_ENTRY_CUTOFF_MINUTES = 90     # 引け N 分前はカナリアエントリー禁止（14:30 ET）
CANARY_MAX_POSITIONS = 2             # カナリア同時ポジション上限
CANARY_TAKE_PROFIT_PCT = 0.01        # 利確 1.0%（確実に拾う）
CANARY_STOP_LOSS_PCT = 0.005         # 損切り 0.5%（VWAPを再度上抜けor0.5%上昇）
