#!/usr/bin/env python3
"""data_loader.py"""
from datasets import load_dataset, Audio
import soundfile as sf
import numpy as np
import io
import json


def load_hf_dataset(parquet_path_pattern):
    """
    Loads the Hugging Face parquet dataset (raw WAV bytes).
    """
    dataset = load_dataset(
        "parquet",
        data_files={"train": parquet_path_pattern},
    )

    # Disable torchcodec decoding entirely
    dataset = dataset.cast_column("audio", Audio(decode=False))

    return dataset["train"]


def split_dataset(ds, train_pct=0.7, val_pct=0.15, test_pct=0.15, seed=42):

    # Add original indices so we don't lose track
    ds = ds.add_column("orig_idx", list(range(len(ds))))

    # Split off test
    split1 = ds.train_test_split(test_size=test_pct, seed=seed)

    # Split remaining into train/val
    remaining = 1.0 - test_pct
    val_relative = val_pct / remaining

    split2 = split1["train"].train_test_split(test_size=val_relative, seed=seed)

    return {
        "train": split2["train"],
        "val": split2["test"],
        "test": split1["test"]
    }


def save_split_indices(split_dataset, out_path):
    indices = list(split_dataset["orig_idx"])   # <-- FINAL FIX
    with open(out_path, "w") as f:
        json.dump({"indices": indices}, f)
    print(f"Saved indices: {out_path} ({len(indices)} samples)")



def decode_audio(record):
    """
    Convert HF raw bytes -> numpy waveform, sampling rate.
    """
    audio_bytes = record["audio"]["bytes"]
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return y, sr, record["label"]




if __name__ == "__main__":
    parquet_path = "/home/ehb/Documents/GitHub/drone-audio-detection-samples/data/*.parquet"

    ds = load_hf_dataset(parquet_path)
    splits = split_dataset(ds)

    save_split_indices(splits["train"], "train.json")
    save_split_indices(splits["val"], "val.json")
    save_split_indices(splits["test"], "test.json")
    print("Done.")