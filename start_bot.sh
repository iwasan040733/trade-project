#!/bin/bash
# trade_bot 起動スクリプト（既に動いていたら何もしない）

PROJ_DIR="/Users/iwamitomoyuki/claude/trade-project"
LOG_FILE="$PROJ_DIR/trade_bot.log"
PID_FILE="$PROJ_DIR/trade_bot.pid"

# 既に動いていたらスキップ
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "$(date): Bot is already running (PID $(cat "$PID_FILE"))" >> "$LOG_FILE"
    exit 0
fi

# 環境変数（.envから読み込み）
set -a
source "$PROJ_DIR/.env"
set +a

echo "$(date): Starting bot..." >> "$LOG_FILE"
nohup "$PROJ_DIR/venv/bin/python" "$PROJ_DIR/trade_bot.py" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "$(date): Bot started (PID $!)" >> "$LOG_FILE"
