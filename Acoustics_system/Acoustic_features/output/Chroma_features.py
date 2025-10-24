
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
def analyze_audio_chroma_only(uav_path, noise_path,n_bins=12):
    # Preprocess
    uav_y, sr = preprocess_audio(uav_path)
    noise_y, _ = preprocess_audio(noise_path, sr=sr)

    # Trim to same length
    min_len = min(len(uav_y), len(noise_y))
    uav_y = uav_y[:min_len]
    noise_y = noise_y[:min_len]

    # Extract only chroma features
    uav_chroma = librosa.feature.chroma_stft(y=uav_y, sr=sr, n_chroma=n_bins)
    noise_chroma = librosa.feature.chroma_stft(y=noise_y, sr=sr, n_chroma=n_bins)

    # Plot chroma side by side
    plot_side_by_side("Chroma", uav_chroma, noise_chroma, sr, y_axis='chroma')

    print("✅ Chroma feature plotted.")

def analyze_audio_freq_time(uav_path, noise_path, n_fft=2048, hop_length=512, n_bins=128, fmin=100, fmax=2000):
    # Preprocess
    uav_y, sr = preprocess_audio(uav_path)
    noise_y, _ = preprocess_audio(noise_path, sr=sr)

    # Trim to same length
    min_len = min(len(uav_y), len(noise_y))
    uav_y = uav_y[:min_len]
    noise_y = noise_y[:min_len]

    # STFT -> power
    uav_stft = librosa.stft(uav_y, n_fft=n_fft, hop_length=hop_length)
    noise_stft = librosa.stft(noise_y, n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # frequency for each row

    uav_power = np.abs(uav_stft) ** 2
    noise_power = np.abs(noise_stft) ** 2

    # Create frequency bin edges and aggregate power into those bins
    edges = np.linspace(fmin, fmax, n_bins + 1)
    uav_binned = np.zeros((n_bins, uav_power.shape[1]), dtype=np.float32)
    noise_binned = np.zeros((n_bins, noise_power.shape[1]), dtype=np.float32)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            # if no FFT bin falls inside this range, pick the nearest FFT row
            center = 0.5 * (lo + hi)
            idx = np.array([np.argmin(np.abs(freqs - center))])
        # aggregate by mean power across the selected FFT rows
        uav_binned[i, :] = uav_power[idx, :].mean(axis=0)
        noise_binned[i, :] = noise_power[idx, :].mean(axis=0)

    # Convert to dB
    uav_db = librosa.power_to_db(uav_binned, ref=np.max)
    noise_db = librosa.power_to_db(noise_binned, ref=np.max)

    # Time & frequency extents for plotting
    times = librosa.frames_to_time(np.arange(uav_db.shape[1]), sr=sr, hop_length=hop_length)
    extent = [times[0], times[-1], fmin, fmax]

    # Plot side by side with correct frequency axis mapping
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
    im0 = axs[0].imshow(uav_db, origin='lower', aspect='auto', extent=extent, cmap='magma')
    axs[0].set_title(f'UAV Spectrogram ({n_bins} bins {fmin}-{fmax} Hz)')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Frequency (Hz)')

    im1 = axs[1].imshow(noise_db, origin='lower', aspect='auto', extent=extent, cmap='magma')
    axs[1].set_title(f'Noise Spectrogram ({n_bins} bins {fmin}-{fmax} Hz)')
    axs[1].set_xlabel('Time (s)')

    fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02)
    plt.show()

    print("✅ Frequency-time spectrogram plotted with", n_bins, "bins between", fmin, "and", fmax, "Hz.")


# -------------------------------
# Example usage
# -------------------------------
UAV_sound = "Acoustics_system\\Dataset_DronePoint\\DS1\\ExperimentallyCollected\\Phantom_Test_File3.wav"
Background_noise = "Acoustics_system\\Dataset_DronePoint\\DS2\\Experimentally Collected Data\\CalmEnvironment_Train_File1.wav"
analyze_audio_chroma_only(UAV_sound, Background_noise,12*12)
analyze_audio_freq_time(UAV_sound, Background_noise)