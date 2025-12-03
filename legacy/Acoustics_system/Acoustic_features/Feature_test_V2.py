import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import python_speech_features as psf
import pywt

# -------------------------------
# Utility functions
# -------------------------------
def preprocess_audio(path, sr=16000, bandpass=(100, 5000)):
    """Load, resample, convert to mono, normalize, and bandpass filter audio."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.normalize(y)

    # Bandpass filter
    sos = signal.butter(10, bandpass, btype='bandpass', fs=sr, output='sos')
    y_filt = signal.sosfilt(sos, y)
    return y_filt, sr

def plot_feature(title, feature, sr, hop_length=512, y_axis='linear'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feature, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(y, sr):
    features = {}

    # Spectral features
    features['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features['spec_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spec_flatness'] = librosa.feature.spectral_flatness(y=y)
    features['spec_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)

    # Temporal features
    features['zcr'] = librosa.feature.zero_crossing_rate(y)
    features['rms'] = librosa.feature.rms(y=y)

    # Cepstral & harmonic features
    lpc_order = 10
    lpc = librosa.lpc(y, order=lpc_order)
    features['lpc'] = lpc

    spectrum = np.abs(librosa.stft(y))
    log_spectrum = np.log(spectrum + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum, axis=0).real
    features['cepstrum'] = cepstrum

    features['hnr'] = librosa.effects.harmonic(y) / (np.abs(y) + 1e-6)

    # Time-frequency representations
    D = np.abs(librosa.stft(y))
    features['spectrogram'] = librosa.amplitude_to_db(D, ref=np.max)
    features['mel_spec'] = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    features['cqt'] = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr)), ref=np.max)

    # Wavelet transform
    coeffs, freqs = pywt.cwt(y, scales=np.arange(1, 128), wavelet='morl')
    features['wavelet'] = np.abs(coeffs)

    return features

# -------------------------------
# Main analysis and plotting
# -------------------------------
def analyze_audio(uav_path, noise_path):
    uav_y, sr = preprocess_audio(uav_path)
    noise_y, _ = preprocess_audio(noise_path, sr=sr)

    # --- Trim to same length ---
    min_len = min(len(uav_y), len(noise_y))
    uav_y = uav_y[:min_len]
    noise_y = noise_y[:min_len]

    uav_feats = extract_features(uav_y, sr)
    noise_feats = extract_features(noise_y, sr)

    # Plot comparisons
    print("Plotting spectral features...")
    plot_feature("UAV - MFCC", uav_feats['mfcc'], sr, y_axis='mel')
    plot_feature("Background - MFCC", noise_feats['mfcc'], sr, y_axis='mel')

    plot_feature("UAV - Spectrogram (STFT)", uav_feats['spectrogram'], sr)
    plot_feature("Background - Spectrogram (STFT)", noise_feats['spectrogram'], sr)

    plot_feature("UAV - Mel Spectrogram", uav_feats['mel_spec'], sr, y_axis='mel')
    plot_feature("Background - Mel Spectrogram", noise_feats['mel_spec'], sr, y_axis='mel')

    plot_feature("UAV - Constant-Q Transform (CQT)", uav_feats['cqt'], sr, y_axis='cqt_note')
    plot_feature("Background - Constant-Q Transform (CQT)", noise_feats['cqt'], sr, y_axis='cqt_note')

    # Temporal
    plt.figure(figsize=(10, 4))
    plt.plot(uav_feats['zcr'][0], label='UAV ZCR')
    plt.plot(noise_feats['zcr'][0], label='Noise ZCR', alpha=0.7)
    plt.legend()
    plt.title("Zero Crossing Rate Comparison")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(uav_feats['rms'][0], label='UAV RMS Energy')
    plt.plot(noise_feats['rms'][0], label='Noise RMS Energy', alpha=0.7)
    plt.legend()
    plt.title("Short-Time Energy Comparison")
    plt.tight_layout()
    plt.show()

    # Wavelet comparison
    plt.figure(figsize=(10, 6))
    plt.imshow(uav_feats['wavelet'], aspect='auto', cmap='viridis', extent=[0, len(uav_y)/sr, 1, 128])
    plt.title("UAV Wavelet Transform (CWT)")
    plt.xlabel("Time (s)")
    plt.ylabel("Scale")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print("Feature extraction completed.")

# -------------------------------
# Example usage
# -------------------------------
UAV_sound = "Acoustics_system\Dataset_DronePoint\DS1\ExperimentallyCollected\Spark_Test_File1.wav"
Background_noise = "Acoustics_system\Dataset_DronePoint\DS2\Experimentally Collected Data\CalmEnvironment_Train_File1.wav"
analyze_audio(UAV_sound, Background_noise)

