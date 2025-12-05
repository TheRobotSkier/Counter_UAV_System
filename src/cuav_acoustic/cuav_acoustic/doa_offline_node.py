#!/usr/bin/env python3
"""
doa_offline_node.py

Offline DOA replay node.

Reads:
  - Recorded multichannel WAV file from a logging session
  - Speed of sound from the log file header

Recomputes DOA estimates from the recording and publishes them on a ROS2 topic
(similar to doa_node.py), but without JACK or live audio.

Useful for:
  - Re-running DOA after algorithm changes
  - Comparing different DOA algorithms on the same dataset
  - Debugging, visualization, offline experiments
"""

import os
import time
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

import rclpy
from rclpy.node import Node

from cuav_interfaces.msg import DOA
from cuav_acoustic import config, doa_core


class DOAOfflineNode(Node):
    def __init__(self):
        super().__init__("acoustic_doa_offline")

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------
        self.declare_parameter("session_dir", "")
        self.declare_parameter("doa_topic", "/acoustic_doa")
        self.declare_parameter("realtime", False)

        session_dir_param = self.get_parameter("session_dir").get_parameter_value().string_value
        if not session_dir_param:
            self.get_logger().error("Parameter 'session_dir' must be set.")
            raise RuntimeError("session_dir not provided")

        self.session_dir = Path(session_dir_param).expanduser().resolve()
        if not self.session_dir.is_dir():
            self.get_logger().error(f"Session directory does not exist: {self.session_dir}")
            raise RuntimeError("Invalid session_dir")

        self.doa_topic = self.get_parameter("doa_topic").get_parameter_value().string_value
        self.realtime = self.get_parameter("realtime").get_parameter_value().bool_value

        self.pub = self.create_publisher(DOA, self.doa_topic, 10)

        # ----------------------------------------------------------
        # Load metadata (speed of sound) and audio
        # ----------------------------------------------------------
        self._load_session_metadata()
        self._prepare_geometry()
        self._start_replay_thread()

    # --------------------------------------------------------------
    # Load CSV header and WAV audio
    # --------------------------------------------------------------
    def _load_session_metadata(self):
        csv_path = self.session_dir / "doa_log.csv"
        if not csv_path.exists():
            self.get_logger().error(f"doa_log.csv not found in {self.session_dir}")
            raise RuntimeError("Missing doa_log.csv")

        # SpeedOfSound from header (if present)
        sos = None
        with csv_path.open("r") as f:
            first_line = f.readline().strip()
            if first_line.startswith("# SpeedOfSound="):
                try:
                    sos = float(first_line.split("=")[1])
                except ValueError:
                    pass

        if sos is None:
            self.get_logger().warn(
                "Could not parse SpeedOfSound from doa_log.csv. "
                f"Using config.SPEED_OF_SOUND={config.SPEED_OF_SOUND:.3f}"
            )
            sos = config.SPEED_OF_SOUND

        config.SPEED_OF_SOUND = sos
        self.get_logger().info(f"Using SPEED_OF_SOUND = {sos:.3f} m/s")

        # Find WAV file (take first *.wav in session_dir)
        wav_files = [p for p in self.session_dir.iterdir() if p.suffix.lower() == ".wav"]
        if not wav_files:
            self.get_logger().error(f"No WAV file found in {self.session_dir}")
            raise RuntimeError("Missing audio file")

        self.audio_path = wav_files[0]
        self.get_logger().info(f"Loading audio: {self.audio_path}")

        data, fs = sf.read(self.audio_path, dtype="float32", always_2d=True)
        self.audio = data  # shape (samples, channels)
        self.fs = fs
        self.n_samples, self.n_channels = self.audio.shape

        self.get_logger().info(
            f"Audio loaded: fs={self.fs} Hz, channels={self.n_channels}, samples={self.n_samples}"
        )

    # --------------------------------------------------------------
    # Precompute SRP-PHAT geometry
    # --------------------------------------------------------------
    def _prepare_geometry(self):
        # You can assert that n_channels == config.NUM_CHANNELS, but
        # for flexibility, we just use the audio's channel count.
        pairs = []
        for i in range(self.n_channels):
            for j in range(i + 1, self.n_channels):
                pairs.append((i, j))
        self.pairs = pairs

        tau_grid, max_tdoa = doa_core.precompute_tau_grid(
            config.MIC_POSITIONS,
            pairs,
            config.AZIMUTHS,
            config.ELEVATIONS,
            config.SPEED_OF_SOUND,
        )
        self.tau_grid = tau_grid
        self.max_tdoa = max_tdoa

        self.frame_len = int(round(config.FRAME_DUR_SEC * self.fs))
        self.hop_len = self.frame_len // 2 if config.OVERLAP_50 else self.frame_len
        self.get_logger().info(
            f"Frame_len={self.frame_len} samples, hop_len={self.hop_len} samples"
        )

    # --------------------------------------------------------------
    # Start offline replay in background thread
    # --------------------------------------------------------------
    def _start_replay_thread(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self._replay_loop, daemon=True)
        self.thread.start()

    def _replay_loop(self):
        idx = 0
        hop_sec = self.hop_len / float(self.fs)

        self.get_logger().info(
            f"Starting offline DOA replay (realtime={self.realtime})..."
        )

        while idx + self.frame_len <= self.n_samples and rclpy.ok():
            frame = self.audio[idx:idx + self.frame_len, :]
            idx += self.hop_len

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
            msg.uav_detected = False  # classifier can be added later
            self.pub.publish(msg)

            if self.realtime:
                time.sleep(hop_sec)

        self.get_logger().info("Offline DOA replay finished.")
        # Optionally shutdown after finishing:
        # rclpy.shutdown()

    # --------------------------------------------------------------
    # Clean shutdown
    # --------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down offline DOA node...")
        self.stop_flag = True
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DOAOfflineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
