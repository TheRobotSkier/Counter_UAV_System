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
    y, sr = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.normalize(y)
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
    feats = {}

    # Spectral features
    feats['MFCC'] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats['Spectrogram'] = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    feats['Mel-Spectrogram'] = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    feats['CQT'] = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr)), ref=np.max)
    feats['Chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)

    # Temporal
    feats['ZCR'] = librosa.feature.zero_crossing_rate(y)
    feats['RMS'] = librosa.feature.rms(y=y)

    # Cepstral & Harmonics
    feats['LPC'] = librosa.lpc(y, order=10)
    spectrum = np.abs(librosa.stft(y))
    log_spectrum = np.log(spectrum + 1e-10)
    feats['Cepstrum'] = np.mean(np.fft.ifft(log_spectrum, axis=0).real, axis=1)
    harmonic = librosa.effects.harmonic(y)
    feats['HNR'] = compute_hnr_series(y, sr)

    # Wavelet
    coeffs, _ = pywt.cwt(y, scales=np.arange(1, 64), wavelet='morl')
    feats['Wavelet'] = np.abs(coeffs)

    return feats

# -------------------------------
# HNR helper
# -------------------------------
def compute_hnr_series(y, sr, frame_length=2048, hop_length=512):
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
# Interactive plotting
# -------------------------------
def interactive_plot(uav_feats, noise_feats, sr):
    feature_names = list(uav_feats.keys())
    n_features = len(feature_names)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    overview_axes = []

    # --- Overview: small subplots ---
    for i, name in enumerate(feature_names):
        ax = fig.add_subplot(3, 4, i+1)
        overview_axes.append(ax)
        uav_feat = uav_feats[name]
        noise_feat = noise_feats[name]

        # --- Check if feature is 1D or small vector ---
        if uav_feat.ndim == 1 or (uav_feat.ndim == 2 and uav_feat.shape[0] == 1):
            ax.plot(uav_feat, label='UAV')
            ax.plot(noise_feat, label='Noise', alpha=0.7)
            ax.set_ylabel("Value")
            ax.set_xlabel("Frame / Sample Index")
        else:
            librosa.display.specshow(uav_feat, sr=sr, x_axis='time', y_axis='linear', ax=ax)

        ax.set_title(f"{i+1}: {name}")
        ax.axis('off')


    fig.suptitle("Overview of all features", fontsize=16)

    current = -1  # -1 = overview

    def on_key(event):
        nonlocal current
        if event.key in ['q', 'Q', 'escape']:
            plt.close(fig)
        elif event.key == 'enter':
            current = -1
            fig.clf()
            for i, name in enumerate(feature_names):
                ax = fig.add_subplot(3, 4, i+1)
                uav_feat = uav_feats[name]
                if uav_feat.ndim == 1:
                    ax.plot(uav_feat, label='UAV')
                    ax.plot(noise_feats[name], label='Noise', alpha=0.7)
                else:
                    librosa.display.specshow(uav_feat, sr=sr, x_axis='time', y_axis='linear', ax=ax)
                ax.set_title(f"{i+1}: {name}")
                ax.axis('off')
            fig.suptitle("Overview of all features", fontsize=16)
            fig.canvas.draw()
        elif event.key.isdigit():
            idx = int(event.key) - 1
            if 0 <= idx < n_features:
                current = idx
                fig.clf()
                ax = fig.add_subplot(1,1,1)
                uav_feat = uav_feats[feature_names[idx]]
                noise_feat = noise_feats[feature_names[idx]]
                if uav_feat.ndim == 1 or (uav_feat.ndim == 2 and uav_feat.shape[0] == 1):
                    ax.plot(uav_feat, label='UAV')
                    ax.plot(noise_feat, label='Noise', alpha=0.7)
                    ax.set_ylabel("Value")
                    ax.set_xlabel("Frame / Sample Index")
                else:
                    librosa.display.specshow(uav_feat, sr=sr, x_axis='time', y_axis='linear', ax=ax)
                ax.set_title(feature_names[idx])
                ax.legend()
                fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# -------------------------------
# Main analysis
# -------------------------------
def analyze_audio(uav_path, noise_path):
    uav_y, sr = preprocess_audio(uav_path)
    noise_y, _ = preprocess_audio(noise_path, sr=sr)

    # Trim to same length
    min_len = min(len(uav_y), len(noise_y))
    uav_y = uav_y[:min_len]
    noise_y = noise_y[:min_len]

    # --- Extract features first ---
    uav_feats = extract_features(uav_y, sr)
    noise_feats = extract_features(noise_y, sr)

    # --- Interactive plot ---
    interactive_plot(uav_feats, noise_feats, sr)

# -------------------------------
# Example usage
# -------------------------------
# analyze_audio("uav_sound.wav", "background_noise.wav")
UAV_sound= "Acoustics_system\Dataset_DronePoint\DS1\ExperimentallyCollected\Phantom_Test_File3.wav"
Background_noise = "Acoustics_system\Dataset_DronePoint\DS2\Experimentally Collected Data\CalmEnvironment_Train_File1.wav"
analyze_audio(UAV_sound, Background_noise)
