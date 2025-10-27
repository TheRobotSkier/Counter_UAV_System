
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
    return tau, cc

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

    # Estimate time delay using GCC-PHAT
    tau, _ = gcc_phat(ch1, ch2, fs=fs, interp=16)

    print(f"Estimated delay: {tau:.6f} seconds")
    print(f"= {tau * fs:.2f} samples")