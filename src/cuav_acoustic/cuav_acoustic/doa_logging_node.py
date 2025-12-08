#!/usr/bin/env python3
"""
doa_logging_node.py

A ROS2 DOA node that:
- Runs the real-time DOA estimator (same as doa_node.py)
- Logs all DOA estimates with timestamps to CSV
- Records raw multichannel audio to a WAV file while still running in real time
- Designed for post-analysis with UAV RTK truth data

This node is NOT meant for deployment on the target system — it is for
experiments, testing, dataset creation, and offline algorithm development.
"""

import rclpy
from rclpy.node import Node

import numpy as np
import jack
import queue
import threading
import time
import os
from datetime import datetime
import soundfile as sf
from pathlib import Path


from cuav_interfaces.msg import DOA
from cuav_interfaces.srv import SpeedOfSound

from cuav_acoustic import doa_core, config


class DOALoggingNode(Node):
    """
    Real-time DOA node with:
    - CSV logging
    - Multichannel audio recording

    Similar to DOANode, but with major additions:
    - Creates a log directory for each run
    - Writes DOA estimates to CSV
    - Records 4-channel WAV while still running real-time detection
    """

    def __init__(self):
        super().__init__("acoustic_doa_logger")

        # -----------------------------------------------------------
        # Load parameters
        # -----------------------------------------------------------
        self.declare_parameter("speed_of_sound_mode", "service")
        self.declare_parameter("fixed_speed_of_sound", 343.0)
        self.declare_parameter("speed_of_sound_service_name", "/get_speed_of_sound")
        self.declare_parameter("doa_topic", "/acoustic_doa")

        self.speed_mode = self.get_parameter("speed_of_sound_mode").value
        self.fixed_c = float(self.get_parameter("fixed_speed_of_sound").value)
        self.sos_service_name = self.get_parameter("speed_of_sound_service_name").value
        self.doa_topic = self.get_parameter("doa_topic").value

        self.get_logger().info(f"DOA publish topic: {self.doa_topic}")
        self.get_logger().info(f"SOS mode: {self.speed_mode}")

        # -----------------------------------------------------------
        # Publisher
        # -----------------------------------------------------------
        self.pub = self.create_publisher(DOA, self.doa_topic, 10)

        # -----------------------------------------------------------
        # Determine speed of sound
        # -----------------------------------------------------------
        self.c = self._determine_speed_of_sound()
        config.SPEED_OF_SOUND = self.c

        # -----------------------------------------------------------
        # JACK setup
        # -----------------------------------------------------------
        try:
            self.jack = jack.Client("doa_logger")
        except jack.JackOpenError:
            self.get_logger().error("JACK not running — cannot start logging node.")
            rclpy.shutdown()
            return

        fs = self.jack.samplerate
        self.fs = fs
        self.get_logger().info(f"JACK samplerate: {fs}")

        # Compute frame parameters
        self.frame_len = int(round(config.FRAME_DUR_SEC * fs))
        self.hop_len = self.frame_len // 2 if config.OVERLAP_50 else self.frame_len

        # Prepare MIC geometry + SRP grid
        self._prepare_geometry()

        # -----------------------------------------------------------
        # Create session logging directory
        # -----------------------------------------------------------
        self._prepare_logging_directories()

        # -----------------------------------------------------------
        # Prepare audio writer (WAV)
        # -----------------------------------------------------------
        self._prepare_audio_recording()

        # -----------------------------------------------------------
        # Start JACK processing + DOA processing thread
        # -----------------------------------------------------------
        self._start_processing_threads()

    # ==============================================================
    #   SPEED OF SOUND
    # ==============================================================
    def _determine_speed_of_sound(self):
        if self.speed_mode == "fixed":
            return self.fixed_c

        # Service mode
        client = self.create_client(SpeedOfSound, self.sos_service_name)
        while not client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info("Waiting for SpeedOfSound service...")

        req = SpeedOfSound.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None:
            self.get_logger().warn("Service failed — falling back to fixed SOS.")
            return self.fixed_c

        self.get_logger().info(f"Using SPEED_OF_SOUND = {result.speed_of_sound:.3f} m/s")
        return float(result.speed_of_sound)

    # ==============================================================
    #   PRECOMPUTE GEOMETRY
    # ==============================================================
    def _prepare_geometry(self):
        pairs = []
        for i in range(config.NUM_CHANNELS):
            for j in range(i + 1, config.NUM_CHANNELS):
                pairs.append((i, j))
        self.pairs = pairs

        tau_grid, max_tdoa = doa_core.precompute_tau_grid(
            config.MIC_POSITIONS,
            pairs,
            config.AZIMUTHS,
            config.ELEVATIONS,
            config.SPEED_OF_SOUND
        )
        self.tau_grid = tau_grid
        self.max_tdoa = max_tdoa

    # ==============================================================
    #   SESSION LOGGING DIRECTORY
    # ==============================================================
    def _prepare_logging_directories(self):
        # Resolve workspace root (2 levels above this file)
        workspace_root = Path(__file__).resolve().parents[2]

        # Create cuav_system_logs folder - Path: Counter_UAV_System/cuav_system_logs/
        base_dir = workspace_root / "cuav_system_logs"
        base_dir.mkdir(exist_ok=True)

        # Create timestamped session directory based on timestamp YYYY-MM-DD_HH-MM-SS-ffffff (Year-Month-Day_Hour-Minute-Second-Microsecond)
        session_name = "doa_session_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.session_dir = base_dir / session_name
        self.session_dir.mkdir()

        # CSV log file
        self.csv_path = self.session_dir / "doa_log.csv"
        self.csv_file = open(self.csv_path, "w")

        # Metadata header
        self.csv_file.write(f"# SpeedOfSound={self.c:.3f}\n")
        self.csv_file.write("# Columns: timestamp(sec), azimuth(deg), elevation(deg), uav_detected (bool)\n")

        # CSV header line
        self.csv_file.write("timestamp,azimuth,elevation,uav_detected\n")

        self.get_logger().info(f"Logging to: {self.session_dir}")

    # ==============================================================
    #   PREPARE AUDIO RECORDING
    # ==============================================================
    def _prepare_audio_recording(self):
        self.audio_path = os.path.join(
            self.session_dir,
            f"audio_{self.fs}Hz_{config.NUM_CHANNELS}ch.wav"
        )
        self.outfile = sf.SoundFile(
            self.audio_path,
            mode="w",
            samplerate=self.fs,
            channels=config.NUM_CHANNELS,
            subtype="FLOAT"
        )

        self.audio_queue = queue.Queue(maxsize=128)
        self.drop_count = 0

        # JACK input ports
        self.inports = [
            self.jack.inports.register(f"in_{i+1}")
            for i in range(config.NUM_CHANNELS)
        ]

    # ==============================================================
    #   THREADS SETUP
    # ==============================================================
    def _start_processing_threads(self):
        # JACK callback
        @self.jack.set_process_callback
        def process(frames):
            try:
                cols = [
                    np.frombuffer(p.get_array(), dtype=np.float32).copy()
                    for p in self.inports
                ]
                block = np.stack(cols, axis=1)
                self.audio_queue.put_nowait(block)
            except queue.Full:
                self.drop_count += 1

        # Activate and auto-connect
        self.jack.activate()
        for i in range(config.NUM_CHANNELS):
            try:
                self.jack.connect(f"system:capture_{i+1}", f"{self.jack.name}:in_{i+1}")
            except jack.JackError:
                self.get_logger().warn(f"Could not connect capture_{i+1}")

        # Start DOA + writer threads
        self.stop_flag = False

        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        self.doa_thread = threading.Thread(target=self._doa_loop, daemon=True)
        self.doa_thread.start()

        self.get_logger().info("JACK activated. DOA Logging started.")

    # ==============================================================
    #   WAV RECORDING LOOP
    # ==============================================================
    def _writer_loop(self):
        while not self.stop_flag or not self.audio_queue.empty():
            try:
                block = self.audio_queue.get(timeout=0.25)
                self.outfile.write(block)
            except queue.Empty:
                pass

    # ==============================================================
    #   DOA LOOP
    # ==============================================================
    def _doa_loop(self):
        buffer = np.zeros((0, config.NUM_CHANNELS), dtype=np.float32)
        next_start = 0

        while not self.stop_flag:
            try:
                block = self.audio_queue.get(timeout=0.1)
                buffer = np.vstack((buffer, block))
            except queue.Empty:
                continue

            while next_start + self.frame_len <= buffer.shape[0]:
                frame = buffer[next_start:next_start + self.frame_len]
                next_start += self.hop_len

                az, el = doa_core.doa_from_frame(
                    frame,
                    self.fs,
                    self.pairs,
                    self.tau_grid,
                    self.max_tdoa,
                    config.AZIMUTHS,
                    config.ELEVATIONS,
                    interp=config.INTERP_GCC,
                )

                msg = DOA()
                msg.azimuth = float(az)
                msg.elevation = float(el)
                msg.uav_detected = False  # classifier later

                self.pub.publish(msg)

                # CSV log
                ts = time.time()
                self.csv_file.write(
                    f"{ts:.6f},{az:.3f},{el:.3f},{msg.uav_detected}\n"
                )

                # Compress buffer occasionally
                if next_start > 4 * self.frame_len:
                    buffer = buffer[next_start:, :]
                    next_start = 0

    # ==============================================================
    #   CLEAN SHUTDOWN
    # ==============================================================
    def destroy_node(self):
        self.get_logger().info("Shutting down logging node...")

        self.stop_flag = True
        time.sleep(0.5)

        try:
            self.jack.deactivate()
            self.jack.close()
        except Exception as exc:
            self.get_logger().warn(f"JACK close error: {exc}")

        try:
            self.outfile.close()
        except:
            pass

        try:
            self.csv_file.close()
        except:
            pass

        super().destroy_node()


# ==============================================================
#   ENTRY POINT
# ==============================================================
def main(args=None):
    rclpy.init(args=args)
    node = DOALoggingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
