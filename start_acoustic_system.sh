#!/usr/bin/env bash
# --------------------------------------------------------------
# start_acoustic_system.sh
#
# Starts JACK (if not already running) and launches
# full ROS2 acoustic system (DOA + serial interface).
# --------------------------------------------------------------

DEVICE="hw:USB"
RATE=96000
FRAMES=256
PERIODS=3

echo "--------------------------------------------------"
echo "  COUNTER-UAV Acoustic System Launcher"
echo "--------------------------------------------------"

# Check if jackd is running
if pgrep -x "jackd" > /dev/null
then
    echo "[INFO] JACK already running."
    JACK_STARTED=false
else
    echo "[INFO] Starting JACK..."
    jackd -R -d alsa -d "$DEVICE" -r "$RATE" -p "$FRAMES" -n "$PERIODS" \
        > /tmp/jack.log 2>&1 &
    sleep 3

    if pgrep -x "jackd" > /dev/null
    then
        echo "[INFO] JACK started successfully."
        JACK_STARTED=true
    else
        echo "[ERROR] Failed to start JACK. Check /tmp/jack.log"
        exit 1
    fi
fi

# Source ROS2 workspace relative to this script
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WS_DIR/install/setup.bash"


echo "[INFO] Launching ROS2 full system..."
ros2 launch acoustic_processing full_system.launch.py

echo "[INFO] ROS2 launch ended."

# Stop JACK if we started it
if [ "$JACK_STARTED" = true ]
then
    echo "[INFO] Stopping JACK..."
    killall jackd
else
    echo "[INFO] JACK left running."
fi

echo "[INFO] Done."
