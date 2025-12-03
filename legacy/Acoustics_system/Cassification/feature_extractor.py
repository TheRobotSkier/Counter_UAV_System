import numpy as np
from scipy.signal import butter, sosfilt
import librosa


# ============================================================
#                   BANDPASS FILTER
# ============================================================
def design_bandpass(lowcut, highcut, sr, order=6):
    """Design a Butterworth bandpass filter in SOS form."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', output='sos')


def apply_bandpass(y, sos):
    """Apply pre-designed bandpass filter to audio."""
    return sosfilt(sos, y)


# ============================================================
#                   FEATURE EXTRACTION
# ============================================================
def extract_mfcc(frame, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def extract_logmel(frame, sr, n_mels=40):
    mel = librosa.feature.melspectrogram(
        y=frame, sr=sr, n_mels=n_mels, fmax=sr//2
    )
    logmel = librosa.power_to_db(mel)
    return np.mean(logmel, axis=1)


def extract_chroma(frame, sr, n_chroma=12):
    chroma = librosa.feature.chroma_stft(
        y=frame, sr=sr, n_chroma=n_chroma
    )
    return np.mean(chroma, axis=1)


# ============================================================
#     MASTER FUNCTION — SELECT FEATURE COMBINATIONS
# ============================================================
def extract_features(frame, sr, sos=None, mode="mfcc_logmel_chroma"):
    """
    mode options:
      "mfcc"
      "logmel"
      "chroma"
      "mfcc_logmel"
      "logmel_chroma"
      "mfcc_logmel_chroma"  <-- default
    """
    if sos is not None:
        frame = apply_bandpass(frame, sos)

    features = []

    if "mfcc" in mode:
        features.append(extract_mfcc(frame, sr))

    if "logmel" in mode:
        features.append(extract_logmel(frame, sr))

    if "chroma" in mode:
        features.append(extract_chroma(frame, sr))

    return np.concatenate(features)


# ============================================================
# Example usage — Extract features for one frame
# ============================================================
if __name__ == "__main__":
    SR = 16000
    BANDPASS = (200, 6000)

    sos = design_bandpass(BANDPASS[0], BANDPASS[1], SR)

    dummy = np.random.randn(1600)  # example frame

    feat1 = extract_features(dummy, SR, sos, mode="mfcc")
    feat2 = extract_features(dummy, SR, sos, mode="logmel")
    feat3 = extract_features(dummy, SR, sos, mode="chroma")
    feat4 = extract_features(dummy, SR, sos, mode="mfcc_logmel")
    feat5 = extract_features(dummy, SR, sos, mode="logmel_chroma")
    feat6 = extract_features(dummy, SR, sos, mode="mfcc_logmel_chroma")

    print("MFCC:", feat1.shape)
    print("LogMel:", feat2.shape)
    print("Chroma:", feat3.shape)
    print("MFCC+LogMel:", feat4.shape)
    print("LogMel+Chroma:", feat5.shape)
    print("MFCC+LogMel+Chroma:", feat6.shape)
