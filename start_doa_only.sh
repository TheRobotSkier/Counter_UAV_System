#!/usr/bin/env bash
# ---------------------------------------------------------
# Start only the DOA node and JACK
# ---------------------------------------------------------

DEVICE="hw:USB"
RATE=96000
FRAMES=256
PERIODS=3

echo "--------------------------------------------------"
echo "  COUNTER-UAV DOA Only Launcher"
echo "--------------------------------------------------"

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
        echo "[INFO] JACK started."
        JACK_STARTED=true
    else
        echo "[ERROR] Unable to start JACK."
        exit 1
    fi
fi

# Source ROS2 workspace
source "$(dirname "$0")/install/setup.bash"

echo "[INFO] Launching DOA-only system..."
ros2 launch acoustic_processing doa_only.launch.py

if [ "$JACK_STARTED" = true ]
then
    echo "[INFO] Stopping JACK..."
    killall jackd
else
    echo "[INFO] JACK left running."
fi

echo "[INFO] DOA-only finished."
