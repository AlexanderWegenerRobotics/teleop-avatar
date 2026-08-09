#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- optional role override, forwarded as-is to both avatar and
#     avatar_pipeline (see twin/role.hpp parseRoleFlag). Omit to use
#     whatever config.yaml's role: key says -- unchanged from before this
#     flag existed. Usage: ./launch.sh --twin   |   ./launch.sh --avatar
ROLE_ARG="$1"
if [ -n "$ROLE_ARG" ] && [ "$ROLE_ARG" != "--twin" ] && [ "$ROLE_ARG" != "--avatar" ]; then
    echo "[ERROR]: unrecognized argument \"$ROLE_ARG\" (expected --twin or --avatar)"
    exit 1
fi

# --- locate binaries (Release preferred, Debug fallback, then plain build/) ---
AVATAR=""
STREAMER=""
for CONFIG in Release Debug ""; do
    if [ -z "$AVATAR" ]; then
        CANDIDATE_DIR="$SCRIPT_DIR/build"
        [ -n "$CONFIG" ] && CANDIDATE_DIR="$SCRIPT_DIR/build/$CONFIG"
        if [ -f "$CANDIDATE_DIR/avatar" ]; then
            AVATAR="$CANDIDATE_DIR/avatar"
            STREAMER="$CANDIDATE_DIR/avatar_pipeline"
            WORK_DIR="$CANDIDATE_DIR"
        fi
    fi
done

# --- add MuJoCo lib dir to LD_LIBRARY_PATH if not already there ---
for MJ_CANDIDATE in "$HOME/Workspace/tools/mujoco" "$HOME/Documents/awegener/dev/mujoco" "$MUJOCO_ROOT"; do
    if [ -n "$MJ_CANDIDATE" ] && [ -d "$MJ_CANDIDATE/lib" ]; then
        case ":$LD_LIBRARY_PATH:" in
            *":$MJ_CANDIDATE/lib:"*) ;;
            *) export LD_LIBRARY_PATH="$MJ_CANDIDATE/lib:$LD_LIBRARY_PATH" ;;
        esac
        break
    fi
done

# --- sanity checks ---
if [ -z "$AVATAR" ]; then
    echo "[ERROR]: avatar binary not found in build/Release, build/Debug, or build/"
    exit 1
fi
if [ ! -f "$STREAMER" ]; then
    echo "[ERROR]: avatar_pipeline binary not found at $STREAMER"
    exit 1
fi

LOG_OUT="$SCRIPT_DIR/avatar_stdout.log"
LOG_ERR="$SCRIPT_DIR/avatar_stderr.log"
CRASH_LOG="$SCRIPT_DIR/avatar_crash.log"

if [ -z "$ROLE_ARG" ]; then
    echo "[LAUNCH]: Starting avatar (role: config.yaml default) (stdout: avatar_stdout.log  stderr: avatar_stderr.log)"
else
    echo "[LAUNCH]: Starting avatar (role: $ROLE_ARG) (stdout: avatar_stdout.log  stderr: avatar_stderr.log)"
fi

show_log() {
    local path="$1" label="$2" max_lines="${3:-30}"
    if [ -f "$path" ]; then
        local lines
        lines="$(tail -n "$max_lines" "$path")"
        if [ -n "$lines" ]; then
            echo "--- $label ---"
            echo "$lines" | sed 's/^/  /'
            echo "---"
        fi
    fi
}

INTERRUPTED=0
cleanup() {
    INTERRUPTED=1
    echo ""
    echo "[LAUNCH]: Shutting down..."
    kill "$STREAMER_PID" 2>/dev/null
    kill "$AVATAR_PID" 2>/dev/null
    wait "$STREAMER_PID" 2>/dev/null
    wait "$AVATAR_PID" 2>/dev/null
    echo "[LAUNCH]: All processes stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

cd "$WORK_DIR" || exit 1

if [ -n "$ROLE_ARG" ]; then
    "$AVATAR" "$ROLE_ARG" > "$LOG_OUT" 2> "$LOG_ERR" &
else
    "$AVATAR" > "$LOG_OUT" 2> "$LOG_ERR" &
fi
AVATAR_PID=$!
if ! kill -0 "$AVATAR_PID" 2>/dev/null; then
    echo "[ERROR]: Failed to start avatar"
    exit 1
fi
echo "[LAUNCH]: avatar PID=$AVATAR_PID"

sleep 2

if [ -n "$ROLE_ARG" ]; then
    "$STREAMER" "$ROLE_ARG" &
else
    "$STREAMER" &
fi
STREAMER_PID=$!
if ! kill -0 "$STREAMER_PID" 2>/dev/null; then
    echo "[ERROR]: Failed to start avatar_pipeline"
    kill "$AVATAR_PID" 2>/dev/null
    exit 1
fi
echo "[LAUNCH]: streamer PID=$STREAMER_PID"
echo "[LAUNCH]: Press Ctrl+C to stop both."

# --- monitor: exit when either process dies ---
while true; do
    if ! kill -0 "$AVATAR_PID" 2>/dev/null; then
        wait "$AVATAR_PID"
        CODE=$?
        echo ""
        if [ "$CODE" -eq 0 ]; then
            echo "[LAUNCH]: avatar exited cleanly."
        else
            echo "[LAUNCH]: avatar exited with error code $CODE."
        fi
        show_log "$LOG_OUT" "avatar stdout"
        show_log "$LOG_ERR" "avatar stderr"
        [ -f "$CRASH_LOG" ] && show_log "$CRASH_LOG" "crash log" 100
        break
    fi
    if ! kill -0 "$STREAMER_PID" 2>/dev/null; then
        wait "$STREAMER_PID"
        ST_CODE=$?
        echo "[LAUNCH]: avatar_pipeline exited with code $ST_CODE."
        break
    fi
    sleep 0.5
done

cleanup