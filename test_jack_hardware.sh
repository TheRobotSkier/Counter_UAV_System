#!/usr/bin/env bash
# --------------------------------------------------------------
# test_jack_hardware.sh
#
# Starts JACK alone (no ROS2), using your project’s audio device.
# Lets you verify that JACK + ALSA + the microphone array hardware
# are working correctly.
# --------------------------------------------------------------

DEVICE="hw:USB"
RATE=96000
FRAMES=256
PERIODS=3

echo "--------------------------------------------------"
echo " JACK + Hardware Test"
echo " Device: $DEVICE"
echo " Rate:   $RATE"
echo " Frames: $FRAMES"
echo " Periods:$PERIODS"
echo "--------------------------------------------------"

# Stop existing JACK session
if pgrep -x "jackd" > /dev/null; then
    echo "[INFO] Stopping existing JACK instance..."
    killall jackd
    sleep 1
fi

echo "[INFO] Starting JACK..."
jackd -R -d alsa -d "$DEVICE" -r "$RATE" -p "$FRAMES" -n "$PERIODS" \
    > /tmp/jack_test.log 2>&1 &

sleep 3

if pgrep -x "jackd" > /dev/null; then
    echo "[SUCCESS] JACK started successfully."
    echo "[INFO] Log file at /tmp/jack_test.log"
else
    echo "[ERROR] JACK failed to start."
    echo "        See /tmp/jack_test.log for details."
    exit 1
fi

echo ""
echo "--------------------------------------------------"
echo " JACK is running."
echo " Press Ctrl+C to stop."
echo "--------------------------------------------------"

# Wait and keep script alive
while true; do sleep 1; done
