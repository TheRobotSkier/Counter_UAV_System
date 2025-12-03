#!/bin/bash
# ----------------------------------------------------------------------
# record.sh  —  Start JACK, run Python recorder, stop JACK afterwards
# ----------------------------------------------------------------------

# --- Configuration ---
DEVICE="hw:USB"
RATE=48000
FRAMES=256
PERIODS=3
# Automatically resolve the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC_SCRIPT="${SCRIPT_DIR}/JACK_record_4ch.py"

# --- Functions ---
start_jack() {
    if pgrep -x jackd >/dev/null; then
        echo "[INFO] JACK is already running."
    else
        echo "[INFO] Starting JACK..."
        jackd -R -d alsa -d "$DEVICE" -r "$RATE" -p "$FRAMES" -n "$PERIODS" >/tmp/jack.log 2>&1 &
        JACK_PID=$!
        sleep 3
        if pgrep -x jackd >/dev/null; then
            echo "[INFO] JACK started successfully (PID $JACK_PID)."
        else
            echo "[ERROR] Failed to start JACK. See /tmp/jack.log"
            exit 1
        fi
    fi
}

stop_jack() {
    if pgrep -x jackd >/dev/null; then
        echo "[INFO] Stopping JACK..."
        killall jackd
        sleep 1
        echo "[INFO] JACK stopped."
    else
        echo "[INFO] JACK was not running."
    fi
}

# --- Main sequence ---
start_jack
echo "[INFO] Running recorder..."
python3 "$REC_SCRIPT"
echo "[INFO] Recorder finished."
stop_jack
