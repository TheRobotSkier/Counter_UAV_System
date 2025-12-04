#!/usr/bin/env python3
"""
doa_node.py
============

ROS2 node for real-time Direction-of-Arrival (DOA) estimation using a
4-channel microphone array and SRP-PHAT processing.

This is the ROS2 successor to the original `uav_doa_node.py`.

----------------------------------------------------------------------
High-Level Responsibilities
----------------------------------------------------------------------
1) Initialize ROS2 and publish DOA estimates on the topic:
       /acoustic_doa
   Message type:
       cuav_interfaces/msg/DOA
       - float32 azimuth
       - float32 elevation
       - bool uav_detected   (placeholder for future classifier)

2) Optionally call the service:
       /get_speed_of_sound
   Service type:
       cuav_interfaces/srv/SpeedOfSound
   The serial node requests temp/humidity from the Arduino and computes
   the current speed-of-sound using Cramer's formula.

3) Initialize JACK for low-latency, multi-channel audio capture.

4) Precompute SRP-PHAT lookup tables (τ-grid) based on:
       - microphone array geometry
       - search grid resolution
       - speed of sound

5) Collect audio frames (100 ms, 50% overlap by default) and perform
   SRP-PHAT DOA estimation for each frame.

6) Publish DOA messages continuously until shutdown.

7) Graceful cleanup: deactivate JACK, stop threads, shutdown ROS2.

----------------------------------------------------------------------
Differences from legacy uav_doa_node.py
----------------------------------------------------------------------
✔ Clean ROS2 package structure (cuav_acoustic)  
✔ ROS2 parameters replace hard-coded config  
✔ Imports use the new cuav_* packages  
✔ Improved shutdown handling (no tracebacks)  
✔ Threading and callback organization simplified  
✔ All ROS-specific logic isolated inside the node class  
✔ Speed-of-sound configuration moved out of config.py  
✔ Code is now readable by team members with clear sections  
"""

# ----------------------------------------------------------------------
# Standard Libraries
# ----------------------------------------------------------------------
import time
import threading
import queue
import numpy as np
import jack

# ----------------------------------------------------------------------
# ROS2 Libraries
# ----------------------------------------------------------------------
import rclpy
from rclpy.node import Node

# ----------------------------------------------------------------------
# Project Modules
# ----------------------------------------------------------------------
from cuav_acoustic import config, doa_core
from cuav_interfaces.msg import DOA
from cuav_interfaces.srv import SpeedOfSound


# ======================================================================
#  ROS2 Acoustic DOA Node
# ======================================================================
class DOANode(Node):
    """
    Node performing:
      - Speed-of-sound acquisition (fixed or service mode)
      - JACK-based audio capture
      - SRP-PHAT DOA estimation
      - Publishing DOA estimates to /acoustic_doa
    """

    def __init__(self):
        super().__init__("acoustic_doa_node")

        # ------------------------------------------------------------------
        # Declare & read ROS2 parameters
        # ------------------------------------------------------------------
        self.declare_parameter("speed_of_sound_mode", "service")
        self.declare_parameter("fixed_speed_of_sound", 343.0)
        self.declare_parameter("speed_of_sound_service_name", "/get_speed_of_sound")
        self.declare_parameter("doa_topic", "/acoustic_doa")

        self.speed_mode = self.get_parameter("speed_of_sound_mode").get_parameter_value().string_value
        self.fixed_c = self.get_parameter("fixed_speed_of_sound").get_parameter_value().double_value
        self.sos_service_name = self.get_parameter("speed_of_sound_service_name").get_parameter_value().string_value
        doa_topic = self.get_parameter("doa_topic").get_parameter_value().string_value

        self.get_logger().info(f"DOA will publish on: {doa_topic}")
        self.get_logger().info(f"Speed-of-sound mode: {self.speed_mode}")

        # ------------------------------------------------------------------
        # Publisher for DOA messages
        # ------------------------------------------------------------------
        self.pub = self.create_publisher(DOA, doa_topic, 10)

        # ------------------------------------------------------------------
        # Determine speed of sound
        # ------------------------------------------------------------------
        c = self._determine_speed_of_sound()
        config.SPEED_OF_SOUND = float(c)

        self.get_logger().info(
            f"Using SPEED_OF_SOUND = {config.SPEED_OF_SOUND:.3f} m/s\n"
        )

        # ------------------------------------------------------------------
        # Start JACK + processing thread
        # ------------------------------------------------------------------
        self._start_audio_processing()

    # ==================================================================
    #  Speed of Sound Acquisition
    # ==================================================================
    def _determine_speed_of_sound(self) -> float:
        """
        Returns the speed of sound either:
            - from a fixed ROS parameter
            - via the /get_speed_of_sound service
        """
        if self.speed_mode == "fixed":
            self.get_logger().info("Using fixed speed of sound.")
            return self.fixed_c

        # -------- Service mode -----------
        if self.speed_mode != "service":
            self.get_logger().warn(
                f"Invalid speed_mode '{self.speed_mode}', defaulting to fixed."
            )
            return self.fixed_c

        self.get_logger().info(
            f"Requesting speed of sound from service {self.sos_service_name}..."
        )

        client = self.create_client(SpeedOfSound, self.sos_service_name)
        while not client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info("Waiting for SpeedOfSound service...")

        req = SpeedOfSound.Request()
        future = client.call_async(req)

        # Block until service responds
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(
                "SpeedOfSound service failed. Falling back to fixed."
            )
            return self.fixed_c

        return future.result().speed_of_sound

    # ==================================================================
    #  Audio Capture + Processing Thread
    # ==================================================================
    def _start_audio_processing(self) -> None:
        """Creates JACK client and launches the processing loop."""
        NUM_CHANNELS = config.NUM_CHANNELS

        # ---------------------- JACK Client Setup -----------------------
        self.jack = jack.Client("doa_realtime")
        fs = self.jack.samplerate

        frame_len = int(round(config.FRAME_DUR_SEC * fs))
        hop_len = frame_len // 2 if config.OVERLAP_50 else frame_len

        # ---------------------- Precompute τ-grid -----------------------
        pairs = self._build_pairs(NUM_CHANNELS)
        tau_grid, max_tdoa = doa_core.precompute_tau_grid(
            config.MIC_POSITIONS,
            pairs,
            config.AZIMUTHS,
            config.ELEVATIONS,
            config.SPEED_OF_SOUND,
        )

        # ---------------------- JACK Input Ports ------------------------
        inports = [
            self.jack.inports.register(f"in_{i+1}")
            for i in range(NUM_CHANNELS)
        ]

        q_blocks: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)

        @self.jack.set_process_callback
        def process(frames: int):
            """Realtime audio callback: capture block into queue."""
            try:
                block = np.column_stack([
                    np.frombuffer(p.get_array(), dtype=np.float32).copy()
                    for p in inports
                ])
                q_blocks.put_nowait(block)
            except queue.Full:
                pass  # Drop block silently

        self.jack.activate()

        # Try connecting system:capture_X ports
        for i in range(NUM_CHANNELS):
            try:
                self.jack.connect(
                    f"system:capture_{i+1}",
                    f"{self.jack.name}:in_{i+1}"
                )
            except jack.JackError:
                self.get_logger().warn(f"Could not connect capture_{i+1}")

        self.get_logger().info("JACK activated. Starting DOA thread...\n")

        # ---------------------- Processing Thread -----------------------
        def loop():
            buffer = np.zeros((0, NUM_CHANNELS), dtype=np.float32)
            next_start = 0

            while rclpy.ok():
                try:
                    block = q_blocks.get(timeout=0.25)
                except queue.Empty:
                    continue

                buffer = np.vstack((buffer, block))

                while next_start + frame_len <= buffer.shape[0]:
                    frame = buffer[next_start: next_start + frame_len]
                    next_start += hop_len

                    az, el = doa_core.doa_from_frame(
                        frame,
                        fs,
                        pairs,
                        tau_grid,
                        max_tdoa,
                        config.AZIMUTHS,
                        config.ELEVATIONS,
                        interp=config.INTERP_GCC,
                    )

                    self._publish_doa(az, el)

                    if next_start > 4 * frame_len:
                        buffer = buffer[next_start:, :]
                        next_start = 0

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    # ==================================================================
    #  Helper Functions
    # ==================================================================
    @staticmethod
    def _build_pairs(n):
        """Return all microphone index pairs."""
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def _publish_doa(self, az, el):
        """Publish DOA message to ROS2."""
        msg = DOA()
        msg.azimuth = float(az)
        msg.elevation = float(el)
        msg.uav_detected = False
        self.pub.publish(msg)

    # ==================================================================
    #  Shutdown
    # ==================================================================
    def destroy_node(self):
        """Ensure JACK client is properly closed."""
        try:
            self.jack.deactivate()
            self.jack.close()
        except Exception as exc:
            self.get_logger().warn(f"Error closing JACK: {exc}")
        super().destroy_node()


# ======================================================================
#  Entry Point
# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DOANode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
