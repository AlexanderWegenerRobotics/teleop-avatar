#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AVATAR="$SCRIPT_DIR/build/avatar"
STREAMER="$SCRIPT_DIR/build/avatar_pipeline"

# --- sanity checks ---
if [ ! -f "$AVATAR" ]; then
    echo "[ERROR]: avatar binary not found at $AVATAR"
    exit 1
fi
if [ ! -f "$STREAMER" ]; then
    echo "[ERROR]: avatar_streamer binary not found at $STREAMER"
    exit 1
fi

cleanup() {
    echo ""
    echo "[LAUNCH]: Shutting down..."
    kill "$STREAMER_PID" 2>/dev/null
    kill "$AVATAR_PID" 2>/dev/null
    wait "$STREAMER_PID" 2>/dev/null
    wait "$AVATAR_PID" 2>/dev/null
    echo "[LAUNCH]: All processes stopped."
}
trap cleanup SIGINT SIGTERM

echo "[LAUNCH]: Starting avatar..."
"$AVATAR" &
AVATAR_PID=$!

sleep 2.0

echo "[LAUNCH]: Starting avatar_streamer..."
"$STREAMER" &
STREAMER_PID=$!

echo "[LAUNCH]: avatar PID=$AVATAR_PID | streamer PID=$STREAMER_PID"
echo "[LAUNCH]: Press Ctrl+C to stop both."

# exit if either process dies
while kill -0 "$AVATAR_PID" 2>/dev/null && kill -0 "$STREAMER_PID" 2>/dev/null; do
    sleep 1
done

# only report unexpected exits if we weren't interrupted
if [ "$INTERRUPTED" != "1" ]; then
    if ! kill -0 "$AVATAR_PID" 2>/dev/null; then
        echo "[LAUNCH]: avatar exited unexpectedly."
    fi
    if ! kill -0 "$STREAMER_PID" 2>/dev/null; then
        echo "[LAUNCH]: video_streamer exited unexpectedly."
    fi
    cleanup
fi

cleanup