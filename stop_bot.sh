#!/bin/bash
# trade_bot グレースフル停止スクリプト
# SIGTERM → 全決済・サマリー送信・停止通知 → 終了を待つ → タイムアウトで強制終了

PROJ_DIR="/Users/iwamitomoyuki/claude/trade-project"
LOG_FILE="$PROJ_DIR/trade_bot.log"
PID_FILE="$PROJ_DIR/trade_bot.pid"
GRACE_PERIOD=30  # グレースフルシャットダウンの猶予（秒）

stop_pid() {
    local PID=$1
    echo "$(date): Sending SIGTERM to Bot (PID $PID)..." >> "$LOG_FILE"
    kill "$PID"

    # 猶予時間内にプロセスが終了するのを待つ
    for i in $(seq 1 $GRACE_PERIOD); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "$(date): Bot stopped gracefully (PID $PID)" >> "$LOG_FILE"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    # タイムアウト: 強制終了
    echo "$(date): Grace period expired. Force killing Bot (PID $PID)..." >> "$LOG_FILE"
    kill -9 "$PID" 2>/dev/null
    rm -f "$PID_FILE"
}

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        stop_pid "$PID"
    else
        echo "$(date): Bot was not running (stale PID $PID)" >> "$LOG_FILE"
        rm -f "$PID_FILE"
    fi
else
    # PIDファイルがない場合、プロセス名で探す
    PID=$(pgrep -f "python.*trade_bot.py" | head -1)
    if [ -n "$PID" ]; then
        stop_pid "$PID"
    else
        echo "$(date): Bot is not running" >> "$LOG_FILE"
    fi
fi
