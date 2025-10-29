
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ---- Path to your recording ----
PATH_RECORDING = "Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_151227_48000Hz_4ch.wav"

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    Estimate time delay using the GCC-PHAT method.

    Args:
        sig: signal 1 (numpy array)
        refsig: signal 2 (numpy array)
        fs: sampling rate
        max_tau: maximum time delay to consider (in seconds)
        interp: interpolation factor for increased precision

    Returns:
        tau: estimated time delay (in seconds)
        cross_correlation: GCC-PHAT cross-correlation array
    """
    # Compute the FFT size (next power of 2)
    n = sig.shape[0] + refsig.shape[0]
    n = 1 << (n - 1).bit_length()

    # FFT of both signals
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    # Compute cross-spectrum
    R = SIG * np.conj(REFSIG)

    # Apply the PHAT weighting
    R /= np.abs(R) + np.finfo(float).eps

    # Inverse FFT to get cross-correlation
    cc = np.fft.irfft(R, n=(interp * n))
    max_shift = int(interp * n / 2)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc, max_shift

# === MAIN SCRIPT ===
if __name__ == "__main__":
    filename = "Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_151227_48000Hz_4ch.wav"

    # Load the multichannel WAV file
    data, fs = sf.read(filename)

    print(f"Loaded {filename}")
    print(f"Sample rate: {fs} Hz, Shape: {data.shape}")

    # Extract first two channels
    ch1 = data[:, 0]
    ch2 = data[:, 1]

    # Extract from A seconds to B seconds
    start = int(2 * fs) # A=2s
    end = int(2.1 * fs) # B=2.1s
    ch1_segment = ch1[start:end]
    ch2_segment = ch2[start:end]
    

    # Create a delayed version (2 ms = 0.002 s)
    delay_samples = int(0.002 * fs)
    #ch2B_delayed =ch2[start+delay_samples:end+delay_samples]
    ch2B_delayed = np.concatenate((
        np.zeros(delay_samples),
        ch1_segment[:-delay_samples]
    ))

    print(f"Created delayed version with {delay_samples} samples ({delay_samples/fs:.6f} s) delay")

    # Estimate delay using GCC-PHAT
    interp_ = 16

    tau, cc, max_shift = gcc_phat(ch1_segment, ch2B_delayed, fs=fs, interp=interp_)
    print(f"Estimated delay: {tau:.6f} s = {tau*fs:.1f} samples")
    lags = np.linspace(-max_shift, max_shift, num=len(cc)) / (interp_ * fs)

    # --- Plot FULL correlation ---
    plt.figure(figsize=(9,4))
    plt.plot(lags*1000, np.abs(cc))
    plt.title("GCC-PHAT Cross-Correlation (Full Range)")
    plt.xlabel("Lag (ms)")
    plt.ylabel("|Cross-correlation|")
    plt.grid(True)

    # --- Zoom around estimated peak ±5 ms ---
    zoom_window = 0.005  # seconds (±5 ms)
    plt.figure(figsize=(8,4))
    plt.plot(lags*1000, np.abs(cc))
    plt.xlim([(tau - zoom_window)*1000, (tau + zoom_window)*1000])
    plt.title(f"GCC-PHAT Zoomed Around Peak ({zoom_window*1000*2:.0f} ms window)")
    plt.xlabel("Lag (ms)")
    plt.ylabel("|Cross-correlation|")
    plt.grid(True)
    plt.show()
