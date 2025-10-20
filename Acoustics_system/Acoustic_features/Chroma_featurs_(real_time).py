"""
Acoustic feature extraction and analysis for UAV and background noise audio files.

Note:
- GPU acceleration:
    - Use librosa.onset.onset_strength(..., aggregate=None) or PyTorch STFT for faster FFTs if scaling up.
"""

import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import time
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
SR_TARGET = 16000            # Target sample rate
FRAME_DURATION = 0.1         # 100 ms
FRAME_SIZE = int(SR_TARGET * FRAME_DURATION)
HOP_SIZE = FRAME_SIZE
BANDPASS = (100, 6000)       # UAV-relevant band
NORMALIZE = True             # Normalize amplitude
USE_LIVE = False             # True = mic stream, False = file test
UAV_PATH= 0                  # 1 = UAV, 0 = No UAV (for offline mode)

if not USE_LIVE:
    if UAV_PATH:
        AUDIO_PATH = "Acoustics_system\Dataset_DronePoint\DS1\ExperimentallyCollected\Phantom_Test_File3.wav"
    else:
        AUDIO_PATH = "Acoustics_system\Dataset_DronePoint\DS2\Experimentally Collected Data\CalmEnvironment_Train_File1.wav"

# --------------------------------------------------
# FILTER DESIGN
# --------------------------------------------------
def design_bandpass(lowcut, highcut, sr, order=6):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', output='sos')

sos = design_bandpass(BANDPASS[0], BANDPASS[1], SR_TARGET)

def apply_bandpass(y):
    return sosfilt(sos, y)

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
def preprocess_audio(y, sr_in):
    """Resample, convert to mono, normalize, and filter."""
    # Convert to mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    # Resample
    if sr_in != SR_TARGET:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=SR_TARGET)
    # Band-pass filter
    y = apply_bandpass(y)
    # Normalize amplitude
    if NORMALIZE:
        y = y / (np.max(np.abs(y)) + 1e-9)
    return y

# --------------------------------------------------
# CHROMA FEATURE EXTRACTION
# --------------------------------------------------
def extract_chroma(y, sr):
    """Compute 12-bin Chroma features for a short frame."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=1024, hop_length=512)
    return chroma

# --------------------------------------------------
# DUMMY CLASSIFIER (replace with your ML model)
# --------------------------------------------------
def classify(chroma_features):
    f_mean = np.mean(chroma_features, axis=1)
    f_std = np.std(chroma_features, axis=1)
    features = np.concatenate([f_mean, f_std])
    # Dummy logic: replace with your ML model
    pred = "UAV" if np.max(f_mean) > 0.86 and np.sum(f_std) < 1.5 else "Noise"
    return pred, features

# --------------------------------------------------
# STREAM PROCESSING THREAD
# --------------------------------------------------
def process_audio_stream(q):
    print(">>> Starting real-time UAV Chroma classification (10 Hz) <<<")
    while True:
        frame = q.get()
        if frame is None:
            break
        y = preprocess_audio(frame, SR_TARGET)
        chroma = extract_chroma(y, SR_TARGET)
        pred, feat = classify(chroma)
        print(f"[{time.strftime('%H:%M:%S')}] {pred:5s} | Feature shape: {chroma.shape}")

# --------------------------------------------------
# SOUNDDEVICE CALLBACK
# --------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    q.put(indata[:, 0].copy())

# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    q = queue.Queue()

    if USE_LIVE:
        # --- Live microphone mode ---
        t = threading.Thread(target=process_audio_stream, args=(q,))
        t.start()

        with sd.InputStream(channels=1, samplerate=SR_TARGET,
                            blocksize=FRAME_SIZE, callback=audio_callback):
            print("Listening... Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    else:
        # --- Offline mode (audio file) ---
        y, sr = librosa.load(AUDIO_PATH, sr=None, mono=False)
        y = preprocess_audio(y, sr)

        n_frames = len(y) // FRAME_SIZE
        print(f"Processing {n_frames} frames ({len(y)/SR_TARGET:.1f}s audio)")

        for i in range(n_frames):
            frame = y[i * FRAME_SIZE:(i + 1) * FRAME_SIZE]
            chroma = extract_chroma(frame, SR_TARGET)
            pred, feat = classify(chroma)
            print(f"Frame {i:04d} | {pred:5s} | Chroma shape: {chroma.shape}")

        #plot chroma features for last frame
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=SR_TARGET, hop_length=512)
        plt.colorbar()
        plt.title('Chroma Features')
        plt.tight_layout()
        plt.show()