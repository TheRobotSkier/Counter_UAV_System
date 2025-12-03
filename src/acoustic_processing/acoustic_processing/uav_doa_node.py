#!/usr/bin/env python3
"""
uav_doa_node.py

Real-time direction-of-arrival (DOA) estimation using a 4-channel microphone
array and JACK for low-latency audio capture. This program:

1) Initializes ROS2 and publishes DOA estimates to a ROS2 topic: /acoustic_doa
2) Optionally requests speed of sound from a ROS2 service: /get_speed_of_sound 
3) Connects to JACK and receives 4-channel streaming audio
4) Precompute SRP-PHAT τ-grid based on array geometry and speed of sound
5) Buffers audio into overlapping frames (default: 100 ms frames, 50% overlap)
6) Performs DOA estimation per frame using SRP-PHAT until stopped by user
7) Graceful shutdown on user request or Ctrl+C

ROS2 message type: counter_uav_msgs/msg/DOA
Fields:
    - float32 azimuth
    - float32 elevation
    - bool uav_detected
    The UAV classifier is not implemented yet; it always publishes False.
    
ROS2 service type: counter_uav_msgs/srv/SpeedOfSound
Behavior:
    - Requests speed of sound measurement from Arduino via service call
    - Arduino responds with computed speed of sound based on temperature
      and humidity readings.



This script uses:
    - config.py                (global DOA and array parameters)
    - processing/doa_core.py   (SRP-PHAT core routines)
"""

import time
import threading
import queue
import numpy as np
import jack

# -------------------------------------------------------------------------
# Import shared config and processing modules
# -------------------------------------------------------------------------
from acoustic_processing import config
from acoustic_processing import doa_core

# -------------------------------------------------------------------------
#  ROS2 imports
# -------------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from counter_uav_msgs.msg import DOA
from counter_uav_msgs.srv import SpeedOfSound


# -------------------------------------------------------------------------
#  ROS2 DOA Publisher Node
# -------------------------------------------------------------------------
class DOAPublisher(Node):
    """
    ROS2 node for publishing DOA estimates.

    Publishes messages of type counter_uav_msgs/msg/DOA:
        - azimuth (deg)
        - elevation (deg)
        - uav_detected (bool)
    """

    def __init__(self):
        super().__init__('doa_publisher')
        self.pub = self.create_publisher(DOA, '/acoustic_doa', 10)

    def publish_doa(self, az, el, uav_flag):
        """
        Publish a DOA message.

        Parameters
        ----------
        az : float
            Azimuth estimate in degrees
        el : float
            Elevation estimate in degrees
        uav_flag : bool
            UAV classification result (placeholder)
        """
        msg = DOA()
        msg.azimuth = float(az)
        msg.elevation = float(el)
        msg.uav_detected = bool(uav_flag)
        self.pub.publish(msg)


# -------------------------------------------------------------------------
#  Utility: build list of microphone pairs
# -------------------------------------------------------------------------
def build_all_pairs(num_mics: int) -> list[tuple[int, int]]:
    """
    Build list of all unique microphone index pairs (i, j).

    Parameters
    ----------
    num_mics : int
        Number of microphones (e.g., 4)

    Returns
    -------
    pairs : list[(i,j)]
        List of unique microphone pairs
    """
    pairs = []
    for i in range(num_mics):
        for j in range(i + 1, num_mics):
            pairs.append((i, j))
    return pairs


# -------------------------------------------------------------------------
#  Utility: wait for Enter key to stop processing
# -------------------------------------------------------------------------
def wait_for_quit(stop_flag: dict) -> None:
    """
    Wait for the user to press Enter, then set stop flag.

    Parameters
    ----------
    stop_flag : dict
        Dictionary containing "stop": bool
    """
    input("Press Enter to stop DOA processing...\n")
    stop_flag["stop"] = True


# -------------------------------------------------------------------------
#  Main real-time DOA program
# -------------------------------------------------------------------------
def main() -> None:
    """
    Main real-time DOA processing function.

    High-level processing steps
    ---------------------------
    1) Initialize ROS2 and start a background spinning thread
         - Required so the DOA publisher and (later) service clients
           can operate while the main thread runs JACK and DOA processing.

    2) Determine the speed of sound:
         - Either use a fixed value from config.py
         - Or call the ROS2 SpeedOfSound service, which triggers the
           Arduino to measure temperature/humidity and return c(t,h).

    3) Initialize JACK for multichannel audio input
         - Register input ports
         - Setup callback → Python queue (non-real-time safe)

    4) Precompute SRP-PHAT τ-grid
         - Depends on array geometry + speed of sound
         - Computed ONCE per start-up

    5) Setup JACK input queue
         - Buffer audio
         - Extract frames with optional 50% overlap

    6) Perform DOA estimation per frame using SRP-PHAT until stopped by user
         - Compute DOA via SRP-PHAT
         - Publish DOA to ROS2

    7) Graceful shutdown on user request or Ctrl+C:
         - Stop threads
         - Deactivate JACK
         - Shutdown ROS2
    """

    # =========================================================================
    # 1) Initialize ROS2: node + spinning thread
    # =========================================================================
    # This launches the ROS2 client library and creates a node responsible
    # for publishing DOA estimates.  The node must be actively "spun" in order
    # for timers, publishers, subscribers, and service clients to function.
    #
    # Because our main thread is busy running JACK + audio processing,
    # we spin ROS2 in a background daemon thread.
    rclpy.init()
    doa_publisher = DOAPublisher()

    def ros_spin():
        """
        Background thread function:
        Continuously spins the 'doa_publisher' node so ROS2 callbacks
        (e.g., service completion, internal timers) are processed.
        """
        rclpy.spin(doa_publisher)

    # Start ROS spinning in a separate thread.
    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()
    print("[INFO] ROS2 publisher node started and spinning.")


    # =========================================================================
    # 2) Acquire speed of sound (fixed value OR service call)
    # =========================================================================
    print("[INFO] Determining speed of sound...")

    # -------------------------------------------------------------------------
    # 2A) FIXED MODE:
    #     Simply read the value from config.FIXED_SPEED_OF_SOUND.
    # -------------------------------------------------------------------------
    if config.SPEED_OF_SOUND_MODE == "fixed":
        c = float(config.FIXED_SPEED_OF_SOUND)
        print(f"[INFO] Using FIXED speed of sound: {c:.3f} m/s")

    # -------------------------------------------------------------------------
    # 2B) SERVICE MODE:
    #     Query the ROS2 SpeedOfSound service
    #     - The service is expected to be provided by serial_interface_node.py (serial_bridge)
    #     - This request instructs the Arduino (via the service node) to:
    #           * perform a batch of temperature/humidity readings
    #           * compute the speed of sound
    #           * return the result
    # -------------------------------------------------------------------------
    elif config.SPEED_OF_SOUND_MODE == "service":
        print("[INFO] Requesting speed of sound from ROS2 service...")

        # ---------------------------------------------------------------------
        # Create a ROS2 service client
        # ---------------------------------------------------------------------
        # The client is attached to the same DOA publisher node, so spinning
        # the 'doa_publisher' node will drive completion of any async service
        # calls.  The service type is counter_uav_msgs/srv/SpeedOfSound.
        client = doa_publisher.create_client(
            SpeedOfSound,
            config.SPEED_OF_SOUND_SERVICE_NAME
        )

        # ---------------------------------------------------------------------
        # Wait until the service becomes available
        # ---------------------------------------------------------------------
        while not client.wait_for_service(timeout_sec=0.5):
            print("[INFO] Waiting for speed_of_sound service to become available...")

        # ---------------------------------------------------------------------
        # Create a service request object
        # ---------------------------------------------------------------------
        # The SpeedOfSound service request has NO fields, so this object
        # is simply an empty container.  It still must be created explicitly.
        request = SpeedOfSound.Request()

        # ---------------------------------------------------------------------
        # Send the request asynchronously
        # ---------------------------------------------------------------------
        # The 'call_async' method returns a Future.  We must wait for the
        # future to complete before retrieving the result.
        future = client.call_async(request)

        # ---------------------------------------------------------------------
        # Spin the ROS node until the future completes
        # ---------------------------------------------------------------------
        # This blocks the main thread until the service response arrives,
        # but ROS callbacks (including service handling) are processed
        # in the background spinning thread 'ros_spin'.
        rclpy.spin_until_future_complete(doa_publisher, future)
        result = future.result()

        # ---------------------------------------------------------------------
        # Validate the service result and extract speed of sound
        # ---------------------------------------------------------------------
        if result is None:
            print("[ERROR] Speed-of-sound service FAILED → using fallback value")
            c = float(config.FIXED_SPEED_OF_SOUND)
        else:
            c = float(result.speed_of_sound)
            print(f"[INFO] Speed of sound from service: {c:.3f} m/s")

    else:
        raise ValueError(
            f"Invalid SPEED_OF_SOUND_MODE: {config.SPEED_OF_SOUND_MODE}"
        )

    # -------------------------------------------------------------------------
    # Store the final speed of sound into the global config
    # This ensures all downstream DOA computations use the correct value.
    # -------------------------------------------------------------------------
    config.SPEED_OF_SOUND = c

    # =========================================================================
    # 3) Setup JACK & parameters
    # =========================================================================
    NUM_CHANNELS = config.NUM_CHANNELS

    client = jack.Client("doa_realtime")
    fs = client.samplerate
    blocksize = client.blocksize

    FRAME_LEN = int(round(config.FRAME_DUR_SEC * fs))
    HOP_LEN = FRAME_LEN // 2 if config.OVERLAP_50 else FRAME_LEN

    print(f"[INFO] JACK samplerate: {fs} Hz")
    print(f"[INFO] Frame length: {FRAME_LEN} samples")
    print(f"[INFO] Hop length:   {HOP_LEN} samples")

    # =========================================================================
    # 4) Precompute SRP-PHAT geometry (τ-grid)
    # =========================================================================
    pairs = build_all_pairs(NUM_CHANNELS)

    tau_grid, max_tdoa_sec = doa_core.precompute_tau_grid(
        config.MIC_POSITIONS,
        pairs,
        config.AZIMUTHS,
        config.ELEVATIONS,
        config.SPEED_OF_SOUND
    )

    print(f"[INFO] Precomputed τ-grid for {len(pairs)} mic pairs.")
    print(f"[INFO] Max TDOA = {max_tdoa_sec*1e3:.3f} ms\n")

    # =========================================================================
    # 5) Setup JACK input queue
    # =========================================================================
    # Initialize buffering: JACK callback → Python queue
    
    inports = [client.inports.register(f"in_{i+1}") for i in range(NUM_CHANNELS)]

    q_blocks: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
    cb_stats = {"dropped": 0}

    @client.set_process_callback
    def process(frames: int):
        """
        JACK realtime callback: push incoming audio blocks into a queue.
        """
        try:
            cols = [
                np.frombuffer(p.get_array(), dtype=np.float32).copy()
                for p in inports
            ]
            block = np.stack(cols, axis=1)
            q_blocks.put_nowait(block)
        except queue.Full:
            cb_stats["dropped"] += 1

    client.activate()
    for i in range(NUM_CHANNELS):
        try:
            client.connect(f"system:capture_{i+1}", f"{client.name}:in_{i+1}")
        except jack.JackError:
            print(f"[WARN] Could not connect capture_{i+1}")

    print("[INFO] Real-time DOA started.\n")

    # =========================================================================
    # 6) Processing thread
    # =========================================================================
    stop_flag = {"stop": False}

    def processing_loop():
        """
        Main processing loop running in a separate thread.

        - Pulls audio blocks from queue
        - Builds rolling frame buffer
        - Performs DOA estimation per frame
        - Publishes DOA to ROS2
        """
        buffer = np.zeros((0, NUM_CHANNELS), dtype=np.float32)
        next_frame_start = 0
        frame_idx = 0

        while not stop_flag["stop"] or not q_blocks.empty():

            try:
                block = q_blocks.get(timeout=0.25)
            except queue.Empty:
                continue

            buffer = np.vstack((buffer, block))

            while next_frame_start + FRAME_LEN <= buffer.shape[0]:

                frame = buffer[next_frame_start: next_frame_start + FRAME_LEN]
                next_frame_start += HOP_LEN

                # --- DOA estimation ---
                best_az, best_el = doa_core.doa_from_frame(
                    frame,
                    fs,
                    pairs,
                    tau_grid,
                    max_tdoa_sec,
                    config.AZIMUTHS,
                    config.ELEVATIONS,
                    interp=config.INTERP_GCC
                )

                frame_idx += 1
                print(f"Frame {frame_idx:05d} → az={best_az:.1f}°, el={best_el:.1f}°")

                # --- Publish to ROS2 ---
                doa_publisher.publish_doa(best_az, best_el, False)

                # Trim buffer occasionally
                if next_frame_start > 4 * FRAME_LEN:
                    buffer = buffer[next_frame_start:, :]
                    next_frame_start = 0

    pt = threading.Thread(target=processing_loop, daemon=True)
    kt = threading.Thread(target=wait_for_quit, args=(stop_flag,), daemon=True)
    pt.start()
    kt.start()

    # =========================================================================
    # 7) Graceful shutdown
    # =========================================================================
    try:
        while not stop_flag["stop"]:
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop_flag["stop"] = True

    stop_flag["stop"] = True
    client.deactivate()
    client.close()

    doa_publisher.destroy_node()
    rclpy.shutdown()
    print("\n[INFO] Clean shutdown complete.")


# -------------------------------------------------------------------------
#  Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
