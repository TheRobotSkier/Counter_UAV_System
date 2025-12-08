#!/usr/bin/env bash
# --------------------------------------------------------------
# test_jack_hardware.sh
#
# Starts JACK for testing, and ALWAYS shuts it down on exit.
# --------------------------------------------------------------

DEVICE="hw:USB"
RATE=96000
FRAMES=256
PERIODS=3

PID_FILE="/tmp/jack_test.pid"

echo "--------------------------------------------------"
echo " JACK + Hardware Test"
echo " Device: $DEVICE"
echo " Rate:   $RATE"
echo " Frames: $FRAMES"
echo " Periods:$PERIODS"
echo "--------------------------------------------------"

cleanup() {
    echo ""
    echo "[INFO] Stopping JACK..."
    if [[ -f "$PID_FILE" ]]; then
        kill $(cat "$PID_FILE") 2>/dev/null
        rm "$PID_FILE"
    else
        # fallback: kill any jackd owned by this user
        pkill -u "$USER" jackd 2>/dev/null
    fi
    echo "[INFO] JACK stopped."
    exit 0
}

# Run cleanup if user presses Ctrl+C or script exits
trap cleanup INT TERM EXIT

# Stop existing JACK instance (only for testing)
if pgrep -x "jackd" > /dev/null; then
    echo "[INFO] Stopping existing JACK..."
    pkill -u "$USER" jackd
    sleep 1
fi

echo "[INFO] Starting JACK..."
jackd -R -d alsa -d "$DEVICE" -r "$RATE" -p "$FRAMES" -n "$PERIODS" \
    >/tmp/jack_test.log 2>&1 &

JACK_PID=$!
echo $JACK_PID > "$PID_FILE"

sleep 3

if kill -0 $JACK_PID 2>/dev/null; then
    echo "[SUCCESS] JACK started successfully."
    echo "[INFO] Log file at /tmp/jack_test.log"
else
    echo "[ERROR] JACK failed to start."
    rm "$PID_FILE"
    exit 1
fi

echo ""
echo "--------------------------------------------------"
echo " JACK is running."
echo " Press Ctrl+C to stop."
echo "--------------------------------------------------"

# Idle forever; trap will handle Ctrl+C
while true; do
    sleep 1
done
