import json
from datasets import load_dataset, Audio

def load_dataset_no_decode(path):
    ds = load_dataset("parquet", data_files={"train": path})
    return ds["train"].cast_column("audio", Audio(decode=False))


def count_labels(dataset, indices):
    """Count label=0 and label=1 using the HF dataset."""
    count_0 = 0
    count_1 = 0

    for idx in indices:
        label = dataset[idx]["label"]
        if label == 0:
            count_0 += 1
        else:
            count_1 += 1

    return count_0, count_1


if __name__ == "__main__":
    parquet_path = "/home/ehb/Documents/GitHub/drone-audio-detection-samples/data/*.parquet"

    print("Loading dataset...")
    dataset = load_dataset_no_decode(parquet_path)

    # Load splits
    with open("train.json") as f:
        train_idx = json.load(f)["indices"]

    with open("val.json") as f:
        val_idx = json.load(f)["indices"]

    with open("test.json") as f:
        test_idx = json.load(f)["indices"]

    # Count labels
    train_0, train_1 = count_labels(dataset, train_idx)
    val_0, val_1 = count_labels(dataset, val_idx)
    test_0, test_1 = count_labels(dataset, test_idx)

    # Print summary
    print("\nDataset Label Distribution:")
    print("----------------------------------")
    print(f"TRAIN:   No-Drone(0): {train_0:6}   Drone(1): {train_1:6}")
    print(f"VAL:     No-Drone(0): {val_0:6}   Drone(1): {val_1:6}")
    print(f"TEST:    No-Drone(0): {test_0:6}   Drone(1): {test_1:6}")
    print("----------------------------------")
    total_0 = train_0 + val_0 + test_0
    total_1 = train_1 + val_1 + test_1
    print(f"TOTAL:   No-Drone(0): {total_0:6}   Drone(1): {total_1:6}")
