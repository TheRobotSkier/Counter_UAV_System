#!/usr/bin/env bash
# CUAV Acoustic Logging Launcher

DEVICE="hw:USB"
RATE=96000
FRAMES=256
PERIODS=3
JACK_STARTED=false

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "--------------------------------------------------"
echo " CUAV Acoustic Logging Launcher"
echo "--------------------------------------------------"

# Start JACK if needed
if pgrep -x "jackd" >/dev/null; then
    echo "[INFO] JACK already running."
else
    echo "[INFO] Starting JACK..."
    jackd -R -d alsa -d "$DEVICE" -r "$RATE" -p "$FRAMES" -n "$PERIODS" \
        >/tmp/jack_log.log 2>&1 &
    sleep 3

    if pgrep -x "jackd" >/dev/null; then
        JACK_STARTED=true
        echo "[INFO] JACK started successfully."
    else
        echo "[ERROR] Failed to start JACK."
        exit 1
    fi
fi

# Source workspace
source "$WS_DIR/install/setup.bash"

echo "[INFO] Launching DOA Logging System..."
ros2 launch cuav_bringup cuav_logging.launch.py

echo "[INFO] ROS2 launch exited."

# Stop JACK if we started it
if [ "$JACK_STARTED" = true ]; then
    echo "[INFO] Stopping JACK..."
    killall jackd
else
    echo "[INFO] JACK left running."
fi
