#!/bin/bash
# trade_bot 起動 → 終了後に Mac をスリープするラッパースクリプト
# launchd から呼び出される（平日 23:22 JST）

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$PROJECT_DIR/logs/bot_stdout.log"

# 多重起動防止
if pgrep -f trade_bot.py > /dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Wrapper] Bot already running, exiting." >> "$LOG"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [Wrapper] Starting trade_bot.py..." >> "$LOG"

cd "$PROJECT_DIR"
source venv/bin/activate
python trade_bot.py >> "$LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') [Wrapper] Bot exited. Sleeping in 120s..." >> "$LOG"
sleep 120

echo "$(date '+%Y-%m-%d %H:%M:%S') [Wrapper] Going to sleep." >> "$LOG"
pmset sleepnow
