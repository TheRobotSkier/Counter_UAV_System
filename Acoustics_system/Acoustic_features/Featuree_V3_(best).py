import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pywt

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_audio(path, sr=16000, bandpass=(100, 5000)):
    """Load, resample, convert to mono, normalize, and bandpass filter audio."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.normalize(y)

    # Safe bandpass
    low, high = bandpass
    nyquist = sr / 2
    if high >= nyquist:
        high = nyquist * 0.99
    sos = signal.butter(10, [low, high], btype='bandpass', fs=sr, output='sos')
    y_filt = signal.sosfilt(sos, y)
    return y_filt, sr

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(y, sr):
    features = {}

    # --- Spectral features ---
    features['mfcc'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features['spec_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spec_flatness'] = librosa.feature.spectral_flatness(y=y)
    features['spec_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)

    # --- Temporal features ---
    features['zcr'] = librosa.feature.zero_crossing_rate(y)
    features['rms'] = librosa.feature.rms(y=y)

    # --- Cepstral & Harmonic features ---
    lpc_order = 10
    features['lpc'] = librosa.lpc(y, order=lpc_order)

    spectrum = np.abs(librosa.stft(y))
    log_spectrum = np.log(spectrum + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum, axis=0).real
    features['cepstrum'] = cepstrum

    harmonic = librosa.effects.harmonic(y)
    features['hnr'] = np.mean(harmonic ** 2) / (np.mean((y - harmonic) ** 2) + 1e-8)

    # --- Time-Frequency representations ---
    D = np.abs(librosa.stft(y))
    features['spectrogram'] = librosa.amplitude_to_db(D, ref=np.max)
    features['mel_spec'] = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    features['cqt'] = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr)), ref=np.max)

    # --- Wavelet transform ---
    coeffs, freqs = pywt.cwt(y, scales=np.arange(1, 64), wavelet='morl')
    features['wavelet'] = np.abs(coeffs)

    return features

def compute_hnr_series(y, sr, frame_length=2048, hop_length=512):
    """Compute short-time harmonic-to-noise ratio over frames."""
    hnr_series = []
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i:i + frame_length]
        harmonic = librosa.effects.harmonic(frame)
        noise = frame - harmonic
        hnr = np.mean(harmonic ** 2) / (np.mean(noise ** 2) + 1e-8)
        hnr_db = 10 * np.log10(hnr + 1e-8)
        hnr_series.append(hnr_db)
    return np.array(hnr_series)

# -------------------------------
# Plot utilities
# -------------------------------
def plot_side_by_side(title, uav_feat, noise_feat, sr, hop_length=512, y_axis='linear'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
    librosa.display.specshow(uav_feat, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, ax=axs[0])
    axs[0].set_title(f'UAV {title}')
    librosa.display.specshow(noise_feat, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, ax=axs[1])
    axs[1].set_title(f'Noise {title}')
    fig.colorbar(axs[1].collections[0], ax=axs, orientation='vertical', fraction=0.02)
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

# -------------------------------
# Main analysis and plotting
# -------------------------------
def analyze_audio(uav_path, noise_path):
    # Preprocess
    uav_y, sr = preprocess_audio(uav_path)
    noise_y, _ = preprocess_audio(noise_path, sr=sr)

    # --- Trim to same length ---
    min_len = min(len(uav_y), len(noise_y))
    uav_y = uav_y[:min_len]
    noise_y = noise_y[:min_len]

    # Extract features
    uav_feats = extract_features(uav_y, sr)
    noise_feats = extract_features(noise_y, sr)

    # --- Spectral features ---
    plot_side_by_side("MFCC", uav_feats['mfcc'], noise_feats['mfcc'], sr, y_axis='mel')
    plot_side_by_side("Spectrogram (STFT)", uav_feats['spectrogram'], noise_feats['spectrogram'], sr)
    plot_side_by_side("Mel-Spectrogram", uav_feats['mel_spec'], noise_feats['mel_spec'], sr, y_axis='mel')
    plot_side_by_side("CQT", uav_feats['cqt'], noise_feats['cqt'], sr, y_axis='cqt_note')
    plot_side_by_side("Chroma", uav_feats['chroma'], noise_feats['chroma'], sr, y_axis='chroma')

    # --- Temporal features ---
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

    # --- Cepstral & Harmonic features ---
    plt.figure(figsize=(10, 4))
    plt.plot(uav_feats['lpc'], label='UAV LPC')
    plt.plot(noise_feats['lpc'], label='Noise LPC', alpha=0.7)
    plt.legend()
    plt.title("Linear Predictive Coding (LPC) Coefficients")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(np.mean(uav_feats['cepstrum'], axis=1), label='UAV Cepstrum')
    plt.plot(np.mean(noise_feats['cepstrum'], axis=1), label='Noise Cepstrum', alpha=0.7)
    plt.legend()
    plt.title("Cepstrum Comparison")
    plt.tight_layout()
    plt.show()

    print(f"Harmonic-to-Noise Ratio (HNR): UAV={uav_feats['hnr']:.3f}, Noise={noise_feats['hnr']:.3f}")

    # --- Time-varying HNR plot ---
    uav_hnr_series = compute_hnr_series(uav_y, sr)
    noise_hnr_series = compute_hnr_series(noise_y, sr)

    plt.figure(figsize=(10, 4))
    plt.plot(uav_hnr_series, label='UAV HNR (dB)')
    plt.plot(noise_hnr_series, label='Noise HNR (dB)', alpha=0.7)
    plt.legend()
    plt.title("Harmonic-to-Noise Ratio (HNR) over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("HNR (dB)")
    plt.tight_layout()
    plt.show()

    # --- Wavelet transform (CWT) ---
    plot_side_by_side("Wavelet Transform (CWT)", uav_feats['wavelet'], noise_feats['wavelet'], sr)

    print("✅ Feature extraction and plotting completed.")

# -------------------------------
# Example usage
# -------------------------------
#UAV_sound = "Acoustics_system\Dataset_DronePoint\DS1\ExperimentallyCollected\Spark_Test_File1.wav"
UAV_sound= "Acoustics_system\Dataset_DronePoint\DS1\ExperimentallyCollected\Phantom_Test_File3.wav"
Background_noise = "Acoustics_system\Dataset_DronePoint\DS2\Experimentally Collected Data\CalmEnvironment_Train_File1.wav"
analyze_audio(UAV_sound, Background_noise)
