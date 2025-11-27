#!/usr/bin/env python3
"""
main_real.py

Real-time direction-of-arrival (DOA) estimation using a 4-channel microphone
array and JACK for low-latency audio capture. This program:

1) Connects to JACK and receives 4-channel streaming audio
2) Buffers audio into overlapping frames (default: 100 ms frames, 50% overlap)
3) Performs DOA estimation per frame using SRP-PHAT
4) Prints azimuth/elevation estimates in real time

This script uses:
    - config.py                (global DOA and array parameters)
    - processing/doa_core.py   (SRP-PHAT core routines)

Designed for ~20 Hz DOA output under real-time constraints.
"""

import os
import sys
import time
import threading
import queue
import numpy as np
import jack


# -------------------------------------------------------------------------
#  Make parent directory importable so config/processing modules work
# -------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import config
from processing import doa_core


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

    Steps:
        1. Initialize JACK client and input ports
        2. Precompute SRP-PHAT τ-grid (geometry only)
        3. Start JACK callback → queue
        4. Run processing loop in a separate thread
        5. Estimate DOA every hop (≈20 Hz for 50% overlap)
        6. Print azimuth/elevation in real time
    """

    # =========================================================================
    # 1) Setup and parameters
    # =========================================================================
    NUM_CHANNELS = config.NUM_CHANNELS
    FRAME_LEN = int(round(config.FRAME_DUR_SEC * config.SPEED_OF_SOUND))  # overridden later
    PRINT_TIMING = config.PRINT_DSP_TIMING

    # Connect to JACK
    client = jack.Client("doa_realtime")
    fs = client.samplerate
    blocksize = client.blocksize

    # Compute frame & hop in samples
    FRAME_LEN = int(round(config.FRAME_DUR_SEC * fs))
    HOP_LEN = FRAME_LEN // 2 if config.OVERLAP_50 else FRAME_LEN

    print(f"[INFO] JACK samplerate: {fs} Hz")
    print(f"[INFO] JACK blocksize:  {blocksize} samples")
    print(f"[INFO] Frame length:    {FRAME_LEN} samples "
          f"({config.FRAME_DUR_SEC*1000:.1f} ms)")
    print(f"[INFO] Hop length:      {HOP_LEN} samples")

    # =========================================================================
    # 2) Precompute SRP-PHAT geometry (τ-grid)
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
    print(f"[INFO] Maximum TDOA = {max_tdoa_sec*1e3:.3f} ms\n")

    # =========================================================================
    # 3) Setup JACK input ports
    # =========================================================================
    inports = [
        client.inports.register(f"in_{i+1}")
        for i in range(NUM_CHANNELS)
    ]

    # Queue for passing samples from JACK → processing thread
    q_blocks: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
    cb_stats = {"dropped": 0}

    @client.set_process_callback
    def process(frames: int) -> None:
        """
        JACK real-time callback.

        Reads `frames × NUM_CHANNELS` samples from JACK and places them in
        a queue for the processing thread. Must be lock-free and fast.

        Parameters
        ----------
        frames : int
            Number of frames provided by JACK (should equal blocksize)
        """
        try:
            cols = [
                np.frombuffer(p.get_array(), dtype=np.float32).copy()
                for p in inports
            ]
            block = np.stack(cols, axis=1)  # shape (frames, channels)
            q_blocks.put_nowait(block)
        except queue.Full:
            cb_stats["dropped"] += 1

    # Activate and connect system capture ports
    client.activate()
    for i in range(NUM_CHANNELS):
        src = f"system:capture_{i+1}"
        dst = f"{client.name}:in_{i+1}"
        try:
            client.connect(src, dst)
            print(f"[INFO] Connected {src} → {dst}")
        except jack.JackError:
            print(f"[WARN] Could not connect {src} → {dst}")

    print("\n[INFO] Real-time DOA estimation started.")
    print(f"[INFO] Expected update rate ≈ {fs / HOP_LEN:.1f} Hz\n")

    # =========================================================================
    # 4) Processing thread: frame extraction + DOA estimation
    # =========================================================================
    stop_flag = {"stop": False}

    def processing_loop() -> None:
        """
        Main processing loop running in a separate thread.

        Continuously:
            - Pulls blocks from queue
            - Appends to rolling buffer
            - Extracts frames with overlap
            - Calls doa_core.doa_from_frame()
            - Prints az / el estimate per frame

        Runs until stop_flag["stop"] is set AND the queue drains.
        """
        buffer = np.zeros((0, NUM_CHANNELS), dtype=np.float32)
        next_frame_start = 0
        frame_idx = 0

        while not stop_flag["stop"] or not q_blocks.empty():
            # Fetch block from queue
            try:
                block = q_blocks.get(timeout=0.25)
            except queue.Empty:
                continue

            buffer = np.vstack((buffer, block))

            # Process frames as soon as enough samples exist
            while next_frame_start + FRAME_LEN <= buffer.shape[0]:
                frame = buffer[next_frame_start: next_frame_start + FRAME_LEN]
                next_frame_start += HOP_LEN

                # ----------------------------------------------------------
                # DOA estimation (single call)
                # ----------------------------------------------------------
                t0 = time.perf_counter()

                best_az, best_el = doa_core.doa_from_frame(
                    frame,
                    fs,
                    pairs,
                    tau_grid,
                    max_tdoa_sec,
                    config.AZIMUTHS,
                    config.ELEVATIONS,
                    interp=config.INTERP_GCC,
                )

                t1 = time.perf_counter()
                dt_ms = (t1 - t0) * 1000.0

                frame_idx += 1
                if PRINT_TIMING:
                    print(
                        f"Frame {frame_idx:05d} → "
                        f"az={best_az:6.1f}°, el={best_el:5.1f}° "
                        f"| proc={dt_ms:6.2f} ms"
                    )
                else:
                    print(
                        f"Frame {frame_idx:05d} → "
                        f"az={best_az:6.1f}°, el={best_el:5.1f}°"
                    )

                # Trim buffer occasionally to avoid unbounded growth
                if next_frame_start > 4 * FRAME_LEN:
                    buffer = buffer[next_frame_start:, :]
                    next_frame_start = 0

    # Start background threads
    pt = threading.Thread(target=processing_loop, daemon=True)
    kt = threading.Thread(target=wait_for_quit, args=(stop_flag,), daemon=True)
    pt.start()
    kt.start()

    # =========================================================================
    # 5) Wait for stop, then clean up
    # =========================================================================
    try:
        while not stop_flag["stop"]:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C received → stopping.")
        stop_flag["stop"] = True
    finally:
        stop_flag["stop"] = True
        client.deactivate()
        client.close()
        pt.join(timeout=2.0)

    print("\n[INFO] DOA processing stopped.")
    if cb_stats["dropped"]:
        print(f"[WARN] Dropped {cb_stats['dropped']} JACK blocks (queue overrun).")


# -------------------------------------------------------------------------
#  Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
