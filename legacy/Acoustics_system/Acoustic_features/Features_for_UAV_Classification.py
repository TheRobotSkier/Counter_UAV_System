import os
from typing import Tuple, Dict

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:#44100
    """Load an audio file and normalize it.

    Returns:
        y: audio time series
        sr: sample rate
    """
    y, sr = librosa.load(path, sr=sr)
    y = librosa.util.normalize(y)
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Extract common audio features used for UAV classification.

    Returns a dict with keys: mfcc, spec_centroid, spec_bandwidth, zcr, rms
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    return {
        "mfcc": mfcc,
        "spec_centroid": spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "zcr": zcr,
        "rms": rms,
    }


def aggregate_features(feat: Dict[str, np.ndarray]) -> np.ndarray:
    """Aggregate features into a 1D feature vector (mean + std where appropriate)."""
    mfcc = feat["mfcc"]
    spec_centroid = feat["spec_centroid"]
    spec_bandwidth = feat["spec_bandwidth"]
    zcr = feat["zcr"]
    rms = feat["rms"]

    features = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(spec_centroid), np.std(spec_centroid),
        np.mean(spec_bandwidth), np.std(spec_bandwidth),
        np.mean(zcr), np.std(zcr),
        np.mean(rms), np.std(rms)
    ])
    return features


def plot_waveform(y: np.ndarray, sr: int, outpath: str) -> None:
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_spectrogram(y: np.ndarray, sr: int, outpath: str) -> None:
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_mfcc(mfcc: np.ndarray, sr: int, outpath: str) -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_feature_summary(feat: Dict[str, np.ndarray], outpath: str) -> None:
    """Create a simple bar plot summarizing mean and std for key features."""
    # We'll show only mean values for clarity; ensure labels match values
    mfcc = feat['mfcc']
    labels = [f'mfcc_{i+1}' for i in range(mfcc.shape[0])]
    values_mean = [np.mean(mfcc[i]) for i in range(mfcc.shape[0])]

    # Add single-valued feature means
    other_labels = ['spec_centroid', 'spec_bandwidth', 'zcr', 'rms']
    other_means = [
        np.mean(feat['spec_centroid']),
        np.mean(feat['spec_bandwidth']),
        np.mean(feat['zcr']),
        np.mean(feat['rms'])
    ]

    labels += other_labels
    values_mean += other_means

    x = np.arange(len(values_mean))
    plt.figure(figsize=(12, 4))
    plt.bar(x, values_mean)
    plt.title('Feature summary (means)')
    plt.ylabel('Value')
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main(audio_path: str = None) -> None:
    """Demo runner. If audio_path is None, uses a default file in the dataset if present."""
    root = os.path.dirname(__file__)
    if audio_path is None:
        default = os.path.join(root, 'Dataset_DronePoint', 'DS1', 'ExperimentallyCollected', 'Spark_Test_File1.wav')
        audio_path = default

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    y, sr = load_audio(audio_path)
    feats = extract_features(y, sr)
    agg = aggregate_features(feats)

    base = os.path.join(root, 'output', f'plots_{os.path.splitext(os.path.basename(audio_path))[0]}')
    ensure_dir(base)

    plot_waveform(y, sr, os.path.join(base, 'waveform.png'))
    plot_spectrogram(y, sr, os.path.join(base, 'spectrogram.png'))
    plot_mfcc(feats['mfcc'], sr, os.path.join(base, 'mfcc.png'))
    plot_feature_summary(feats, os.path.join(base, 'feature_summary.png'))

    # Save aggregated features as numpy file
    np.save(os.path.join(base, 'aggregated_features.npy'), agg)


if __name__ == '__main__':
    # run demo with default
    try:
        main()
        print('Plots and features saved under output/plots_<file>/')
    except Exception as e:
        print(f'Error running demo: {e}')
