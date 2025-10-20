
import librosa
import librosa.display
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

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
# Plot utilities
# -------------------------------
def plot_side_by_side(title, uav_feat, noise_feat, sr, hop_length=512, y_axis='linear'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
    librosa.display.specshow(uav_feat, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, ax=axs[0])
    axs[0].set_title(f'UAV {title}')
    librosa.display.specshow(noise_feat, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, ax=axs[1])
    axs[1].set_title(f'Noise {title}')
    fig.colorbar(axs[1].collections[0], ax=axs, orientation='vertical', fraction=0.02)
    plt.show()

# -------------------------------
# Main analysis and plotting
# -------------------------------
def analyze_audio_chroma_only(uav_path, noise_path):
    # Preprocess
    uav_y, sr = preprocess_audio(uav_path)
    noise_y, _ = preprocess_audio(noise_path, sr=sr)

    # Trim to same length
    min_len = min(len(uav_y), len(noise_y))
    uav_y = uav_y[:min_len]
    noise_y = noise_y[:min_len]

    # Extract only chroma features
    uav_chroma = librosa.feature.chroma_stft(y=uav_y, sr=sr, n_chroma=12*12)
    noise_chroma = librosa.feature.chroma_stft(y=noise_y, sr=sr, n_chroma=12*12)

    # Plot chroma side by side
    plot_side_by_side("Chroma", uav_chroma, noise_chroma, sr, y_axis='chroma')

    print("✅ Chroma feature plotted.")

# -------------------------------
# Example usage
# -------------------------------
UAV_sound = "Acoustics_system\\Dataset_DronePoint\\DS1\\ExperimentallyCollected\\Phantom_Test_File3.wav"
Background_noise = "Acoustics_system\\Dataset_DronePoint\\DS2\\Experimentally Collected Data\\CalmEnvironment_Train_File1.wav"
analyze_audio_chroma_only(UAV_sound, Background_noise)
