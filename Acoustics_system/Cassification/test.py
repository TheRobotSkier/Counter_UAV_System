from datasets import load_dataset, Audio

dataset = load_dataset(
    "parquet",
    data_files={"train": "/home/ehb/Documents/GitHub/drone-audio-detection-samples/data/*.parquet"},
)

# Disable torchcodec decoding:
dataset = dataset.cast_column("audio", Audio(decode=False))

print(dataset)
print(dataset["train"][0])