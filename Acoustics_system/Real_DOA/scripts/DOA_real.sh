#!/bin/bash
# ----------------------------------------------------------------------
# DOA_real.sh
#
# Launch real-time DOA estimation using JACK + main_real.py.
# Automatically starts and stops JACK if needed.
# ----------------------------------------------------------------------

# -------------------------------
# Configuration
# -------------------------------
DEVICE="hw:USB"       # ALSA device for your 4-channel interface
RATE=48000            # Sample rate
FRAMES=256            # JACK buffer size
PERIODS=3             # JACK periods
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROGRAM="${PROJECT_DIR}/programs/main_real.py"

# -------------------------------
# Helper: start JACK if needed
# -------------------------------
start_jack() {
    if pgrep -x jackd >/dev/null; then
        echo "[INFO] JACK already running."
        JACK_STARTED=false
    else
        echo "[INFO] Starting JACK..."
        jackd -R -d alsa -d "$DEVICE" -r "$RATE" -p "$FRAMES" -n "$PERIODS" > /tmp/jack.log 2>&1 &
        sleep 3

        if pgrep -x jackd >/dev/null; then
            echo "[INFO] JACK started successfully."
            JACK_STARTED=true
        else
            echo "[ERROR] Failed to start JACK. See /tmp/jack.log"
            exit 1
        fi
    fi
}

# -------------------------------
# Helper: stop JACK only if we started it
# -------------------------------
stop_jack() {
    if [ "$JACK_STARTED" = true ]; then
        echo "[INFO] Stopping JACK..."
        killall jackd 2>/dev/null
        sleep 1
        echo "[INFO] JACK stopped."
    else
        echo "[INFO] Leaving JACK running (was not started by this script)."
    fi
}

# -------------------------------
# MAIN
# -------------------------------
echo "------------------------------------------------------------"
echo " Real-Time DOA Estimation Launcher"
echo "------------------------------------------------------------"

start_jack

echo "[INFO] Running main_real.py..."
python3 "$PROGRAM"

echo "[INFO] main_real.py finished."

stop_jack

echo "[INFO] DOA_real.sh complete."
