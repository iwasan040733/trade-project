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

POSITION_SIZE = 2000               # バックテスト用（3銘柄分散: $2,000 × 3）

# 市場時間（米国東部時間）— この時間外はエントリーしない
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
EOD_CLOSE_MINUTES_BEFORE = 30     # 市場クローズ 30 分前に全ポジション決済（15:30 ET）

# 利確 / 損切り（ATRベース R:R設計）
STOP_LOSS_ATR_MULT = 1.5           # 損切り = 5分足 ATR(14) × 1.5
STOP_LOSS_MAX_PCT = 0.02           # 損切り上限 2.0%（キャップ）
TAKE_PROFIT_RR_MULT = 3.0          # 利確 = 損切り幅 × 3.0（R:R = 1:3）

# トレーリングストップ（ATRベース）
TRAILING_ATR_MULT = 2.0            # トレーリング幅 = ATR(14) × 2.0
TRAILING_ACTIVATE_ATR_MULT = 1.0   # トレーリング発動: ATR × 1.0 の含み益

# エントリーATRフィルター（低ボラ回避）
ENTRY_MIN_ATR_PCT = 0.003          # 最低 ATR% = 0.3%（値幅が取れないタイミングを回避）

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
    "AI半導体": ["NVDA", "AMD", "AVGO", "MRVL"],
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
AUTO_SYMBOLS_FALLBACK = ["COIN", "MARA", "MSTR", "SOXL", "SMCI"]

# スナイパー型 固定銘柄（自動売買で使用）
SNIPER_SYMBOLS = ["COIN", "MARA", "MSTR", "SOXL", "SMCI"]

# ショート代替シンボル: ショートシグナル発生時にインバースETFをロング買いで代替エントリー
# 例: SOXLのショートシグナル → SOXSをロング買い（空売り不要）
SYMBOL_SHORT_SUBSTITUTE: dict[str, str] = {"SOXL": "SOXS"}

# 決算ブラックアウト（決算発表前後は自動停止）
EARNINGS_BLACKOUT_HOURS = 24       # 決算前後 24 時間は取引停止

# エントリー時間制限（ノイズの多い寄り付き・引けを回避）
ENTRY_BUFFER_MINUTES_OPEN = 15    # 寄り付き後 15 分はエントリーしない（9:45 ET〜）
ENTRY_BUFFER_MINUTES_CLOSE = 30   # 引け前 30 分はエントリーしない（〜15:30 ET）

# エントリー条件
ENTRY_PROXIMITY_THRESHOLD = 0.005  # サポートまで 0.5% 以内
ENTRY_RSI_THRESHOLD = 30           # RSI がこの値以下から上昇に転じる
ENTRY_VOLUME_MULT = 1.5            # 出来高フィルター: 直近20本平均の1.5倍以上

# ============================================================
#  BTC地合いフィルター（Crypto Regime Filter）
# ============================================================
BTC_REGIME_ENABLED = False
BTC_REGIME_SYMBOL = "BITO"            # ProShares Bitcoin Strategy ETF
BTC_REGIME_SMA_PERIOD = 20            # 日足20SMA: 価格がこの下なら Bearish → ロング禁止

# ============================================================
#  ボラティリティブレイクアウト（Volatility Breakout）— スナイパー型
# ============================================================
BREAKOUT_ENABLED = True
BREAKOUT_K = 0.5                      # 前日レンジ × K を始値に加算
BREAKOUT_EMA_PERIOD = 20              # フィルター: 5分足20EMA上のみ
BREAKOUT_STOP_ATR_MULT = 1.5          # 損切り = ATR(14) × 1.5（ノイズ耐性強化）
BREAKOUT_TRAILING_ATR_MULT = 5.0      # トレーリング = ATR(14) × 5.0（ホームラン狙い）
BREAKOUT_TP_ATR_MULT = 0.0            # 固定TP無効（トレーリングのみ）
BREAKOUT_EOD_MINUTES = 5              # 引け5分前（15:55 ET）に強制決済
BREAKOUT_MAX_TRADES_PER_DAY = 1       # 1日1回制限（往復ビンタ排除）

# スナイパーフィルター
BREAKOUT_ADX_PERIOD = 14              # ADX 計算期間
BREAKOUT_ADX_THRESHOLD = 40           # ADX > 40 で強トレンドのみ（厳選）
BREAKOUT_VOL_SPIKE_SHORT = 5          # 出来高スパイク: 直近N本の平均
BREAKOUT_VOL_SPIKE_LONG = 20          # 出来高スパイク: 過去N本の平均
BREAKOUT_VOL_SPIKE_MULT = 2.0         # 直近5本 / 過去20本 >= 2.0倍（厳選）
BREAKOUT_ATR_EXPANSION = True         # ATR拡大フィルター: 現在ATR > 1日平均ATR

# プルバックエントリー（指値注文）
BREAKOUT_PULLBACK_ENABLED = False     # True=プルバック待ち指値, False=成行
BREAKOUT_PULLBACK_BUFFER_ATR = 0.3    # 指値 = ブレイクアウトライン + ATR×0.3（少し余裕）
BREAKOUT_PULLBACK_TIMEOUT_BARS = 12   # 12本（5分×12=1時間）以内に約定しなければキャンセル

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

# サーキットブレイカー（急な反転に即時反応）
QQQ_CB_BARS = 3                       # 直近N本の変化率で判定（3本=15分）
QQQ_CB_THRESHOLD_DOWN = -0.005        # 下方向閾値 -0.5% → bearish
QQQ_CB_THRESHOLD_UP = 0.005           # 上方向閾値 +0.5% → bullish

# レジームロック（チャタリング防止）
QQQ_REGIME_LOCK_SECONDS = 900         # レジーム変更後15分間はロック

VIX_PANIC_ENABLED = True
VIX_PANIC_THRESHOLD = 0.10            # VIX 前日比 +10% でパニックモード
VIX_SYMBOL = "^VIX"                   # yfinance 用シンボル
VIX_CACHE_SECONDS = 300               # VIX キャッシュ（5分）

