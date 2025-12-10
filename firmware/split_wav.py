from pathlib import Path
import soundfile as sf
import numpy as np

def split_wav_channels(input_wav: str | Path) -> list[Path]:
    input_wav = Path(input_wav)

    # Read audio (always get a 2D array: frames x channels)
    data, sr = sf.read(input_wav, always_2d=True)
    n_frames, n_channels = data.shape

    if n_channels < 2:
        raise ValueError(f"File has {n_channels} channel(s); nothing to split.")

    out_files = []
    stem = input_wav.stem
    suffix = input_wav.suffix  # ".wav"
    parent = input_wav.parent

    # Use the same subtype/format as the input file when writing
    info = sf.info(input_wav)

    for ch in range(n_channels):
        out_path = parent / f"{stem}_ch{ch+1}{suffix}"
        mono = data[:, ch]  # 1D array (frames,)

        sf.write(out_path, mono, sr, subtype=info.subtype, format=info.format)
        out_files.append(out_path)

    return out_files

if __name__ == "__main__":
    created = split_wav_channels("src/cuav_system_logs/doa_session_2025-12-09_15-25-58-645387/audio_96000Hz_4ch.wav")
    print("Wrote:")
    for p in created:
        print(" -", p)
