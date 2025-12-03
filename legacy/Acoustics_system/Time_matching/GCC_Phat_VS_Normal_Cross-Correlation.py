import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def gcc_phat(sig, refsig, fs=1, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    n = 1 << (n - 1).bit_length()
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + np.finfo(float).eps
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc, max_shift

# === MAIN ===
filename = "Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_151227_48000Hz_4ch.wav"
data, fs = sf.read(filename)

# Use channel 1 (2s→3s)
start, end = int(2*fs), int(3*fs)
ch1 = data[start:end, 0]

# Create a delayed copy (2 ms)
delay_sec = 0.002
delay_samples = int(delay_sec * fs)
ch1B = np.concatenate((np.zeros(delay_samples), ch1[:-delay_samples]))

# --- GCC-PHAT ---
tau_gcc, cc_gcc, max_shift = gcc_phat(ch1, ch1B, fs, interp=16)
lags = np.linspace(-max_shift, max_shift, len(cc_gcc)) / (16 * fs)

# --- Regular cross-correlation (no PHAT weighting) ---
cc_normal = np.correlate(ch1, ch1B, mode='full')
lags_normal = np.arange(-len(ch1B)+1, len(ch1)) / fs

# Normalize both for easier visual comparison
cc_gcc_norm = np.abs(cc_gcc) / np.max(np.abs(cc_gcc))
cc_normal_norm = np.abs(cc_normal) / np.max(np.abs(cc_normal))

# --- Plot zoomed view ---
plt.figure(figsize=(8,4))
plt.plot(lags*1000, cc_gcc_norm, label="GCC-PHAT")
plt.plot(lags_normal*1000, cc_normal_norm, label="Cross-correlation", color='orange')
plt.xlim([(tau_gcc - 0.005)*1000, (tau_gcc + 0.005)*1000])  # ±5 ms window
plt.title("GCC-PHAT vs. Cross-correlation")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"True delay: {delay_sec*1000:.3f} ms")
print(f"Estimated delay (GCC-PHAT): {tau_gcc*1000:.3f} ms")
