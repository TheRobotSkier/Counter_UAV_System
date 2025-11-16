import json
import io
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio


# ---------------------------------------------
# Load dataset WITHOUT decoding (no torchcodec)
# ---------------------------------------------
def load_dataset_no_decode(path_pattern):
    ds = load_dataset(
        "parquet",
        data_files={"train": path_pattern}
    )
    return ds["train"].cast_column("audio", Audio(decode=False))


# ---------------------------------------------
# Decode audio from HF raw bytes
# ---------------------------------------------
def decode_audio_from_dataset(dataset, idx):
    record = dataset[idx]
    audio_bytes = record["audio"]["bytes"]
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return y, sr, record["label"]


# ---------------------------------------------
# Load sample index from split (train.json etc.)
# ---------------------------------------------
def load_sample_from_split(split_json_path, dataset, position):
    with open(split_json_path, "r") as f:
        indices = json.load(f)["indices"]

    if position >= len(indices):
        raise IndexError("Requested position outside split index list.")

    dataset_idx = indices[position]
    return decode_audio_from_dataset(dataset, dataset_idx)


# ---------------------------------------------
# 100 ms / 200 ms framing function
# ---------------------------------------------
def frame_audio(y, sr, frame_ms):
    frame_size = int(sr * (frame_ms / 1000.0))  # 100ms → 1600 samples
    total_samples = len(y)

    frames = []

    # Only include COMPLETE frames
    for start in range(0, total_samples - frame_size + 1, frame_size):
        end = start + frame_size
        frame = y[start:end]
        frames.append(frame)

    return frames


# ---------------------------------------------
# DEMO
# ---------------------------------------------
if __name__ == "__main__":
    parquet_path = "/home/ehb/Documents/GitHub/drone-audio-detection-samples/data/*.parquet"

    # Load dataset
    dataset = load_dataset_no_decode(parquet_path)

    # Load one sample from train split
    y, sr, label = load_sample_from_split("train.json", dataset, position=0)

    print(f"Loaded waveform: {y.shape}, SR={sr}, label={label}")

    # Frame into 100 ms windows
    frames_100 = frame_audio(y, sr, frame_ms=100)
    frames_200 = frame_audio(y, sr, frame_ms=200)

    print(f"100 ms frames: {len(frames_100)}  (each frame = {len(frames_100[0])} samples)")
    print(f"200 ms frames: {len(frames_200)}  (each frame = {len(frames_200[0])} samples)")
