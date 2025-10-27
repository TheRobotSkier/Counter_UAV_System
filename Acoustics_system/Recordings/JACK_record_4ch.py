#!/usr/bin/env python3
import jack
import numpy as np
import soundfile as sf
import threading, queue, time, sys, os
from datetime import datetime

# ---------- Config ----------
NUM_CHANNELS = 4
FILE_DIR = "Acoustics_system/Recordings/Four_mic_recordings"
SUBTYPE = "FLOAT"      # 32-bit float WAV to preserve dynamic range
QUEUE_MAX = 128        # blocks; increase if your disk is slow

# ---------- Keypress (q / ESC) ----------
def wait_for_quit(flag):
    input("Press Enter to stop recording...\n")
    flag["stop"] = True


def main():
    # Create JACK client
    client = jack.Client("recorder_4ch")
    fs = client.samplerate
    blocksize = client.blocksize

    # Register 4 input ports
    inports = [client.inports.register(f"in_{i+1}") for i in range(NUM_CHANNELS)]

    # A lock-free-ish handoff: callback -> writer thread
    q = queue.Queue(maxsize=QUEUE_MAX)
    stats = {"dropped": 0}

    @client.set_process_callback
    def process(frames):
        # JACK guarantees frames == blocksize (unless changed)
        # Copy the float32 input buffers, stack to (frames, channels)
        try:
            cols = [np.frombuffer(p.get_array(), dtype=np.float32).copy() for p in inports]
            block = np.stack(cols, axis=1)
            q.put_nowait(block)
        except queue.Full:
            stats["dropped"] += 1  # disk/CPU too slow; block dropped (no xrun to JACK)

    # Activate and auto-connect to system capture ports
    client.activate()
    # Try to connect system:capture_1..4 → our in_1..in_4
    for i in range(NUM_CHANNELS):
        src = f"system:capture_{i+1}"
        dst = f"{client.name}:in_{i+1}"
        try:
            client.connect(src, dst)
        except jack.JackError:
            print(f"Warning: couldn't connect {src} -> {dst}. Check port names.", file=sys.stderr)

    print(f"Recording {NUM_CHANNELS} channels at {fs} Hz, blocksize {blocksize}.")
    print("Press 'q' or ESC to stop.")

    # Prepare output file
    base = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fs}Hz_{NUM_CHANNELS}ch.wav"
    outpath = os.path.join(FILE_DIR, base)
    outfile = sf.SoundFile(outpath, mode="w", samplerate=fs, channels=NUM_CHANNELS, subtype=SUBTYPE)

    stop_flag = {"stop": False}

    # Writer thread (disk I/O off the realtime callback)
    def writer():
        # Write until stop flag and queue drained
        while not stop_flag["stop"] or not q.empty():
            try:
                block = q.get(timeout=0.25)
                outfile.write(block)  # (frames, channels)
            except queue.Empty:
                pass

    wt = threading.Thread(target=writer, daemon=True)
    kt = threading.Thread(target=wait_for_quit, args=(stop_flag,), daemon=True)
    wt.start()
    kt.start()

    try:
        # Keep main thread alive
        while not stop_flag["stop"]:
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop_flag["stop"] = True
    finally:
        # Wind down
        stop_flag["stop"] = True
        kt.join(timeout=1.0)
        client.deactivate()
        wt.join(timeout=2.0)
        outfile.close()
        client.close()

    print(f"\nSaved: {outpath}")
    if stats["dropped"]:
        print(f"Note: dropped {stats['dropped']} blocks (writer queue overrun). "
              f"Consider faster disk, larger QUEUE_MAX, or larger JACK buffers.")

if __name__ == "__main__":
    main()
