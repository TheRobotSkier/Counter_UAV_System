import json
import io
import soundfile as sf
import simpleaudio as sa
from datasets import load_dataset, Audio

# Disable torchcodec
def load_dataset_no_decode(path):
    ds = load_dataset("parquet", data_files={"train": path})
    return ds["train"].cast_column("audio", Audio(decode=False))

def decode_record_audio(record):
    y, sr = sf.read(io.BytesIO(record["audio"]["bytes"]), dtype="float32")
    return y, sr, record["label"]

def play_audio(y, sr):
    audio_int16 = (y * 32767).astype("int16")
    sa.play_buffer(audio_int16, 1, 2, sr).wait_done()

if __name__ == "__main__":
    # Load dataset
    parquet = "/home/ehb/Documents/GitHub/drone-audio-detection-samples/data/*.parquet"
    dataset = load_dataset_no_decode(parquet)

    # Load split indices
    data = json.load(open("test.json"))
    indices = data["indices"]

    # Pick a sample
    idx = indices[0]   # FIX: correctly index into the list
    record = dataset[idx]

    y, sr, label = decode_record_audio(record)

    print("Label:", label)
    print("Playing audio…")
    play_audio(y, sr)
