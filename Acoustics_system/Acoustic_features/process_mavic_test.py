import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

# Paths
root = os.path.dirname(__file__)
input_wav = os.path.join(root, 'Dataset_DronePoint', 'DS1', 'ExperimentallyCollected', 'Mavic_Test_File2.wav')
output_dir = os.path.join(root, 'output')
os.makedirs(output_dir, exist_ok=True)

# Parameters
lowcut = 50.0
highcut = 2500.0
n_samples = 4096

# Read WAV
sr, data = wavfile.read(input_wav)
print(f"Sample rate: {sr}, dtype: {data.dtype}, shape: {data.shape}")

# Convert to float32 in range -1..1
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0
elif data.dtype == np.int32:
    data = data.astype(np.float32) / 2147483648.0
elif data.dtype == np.uint8:
    data = (data.astype(np.float32) - 128) / 128.0
else:
    data = data.astype(np.float32)

# Mono
if data.ndim > 1:
    data = np.mean(data, axis=1)

# Trim or pad to n_samples
if len(data) >= n_samples:
    segment = data[:n_samples]
else:
    segment = np.zeros(n_samples, dtype=np.float32)
    segment[:len(data)] = data

# Bandpass filter design (Butterworth)
sos = butter(4, [lowcut, highcut], btype='band', fs=sr, output='sos')
filtered = sosfiltfilt(sos, segment)

# FFT
fft_vals = np.fft.rfft(filtered)
fft_freq = np.fft.rfftfreq(n_samples, 1.0/sr)
fft_mag = np.abs(fft_vals)

# Save filtered WAV (convert back to int16)
out_wav = os.path.join(output_dir, 'Mavic_Test_File2_filtered.wav')
wavfile.write(out_wav, sr, np.int16(np.clip(filtered * 32767, -32768, 32767)))

# Plot
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(np.arange(n_samples)/sr, filtered)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered signal (100-5000 Hz) - 4096 samples')

plt.subplot(2,1,2)
plt.semilogy(fft_freq, fft_mag + 1e-12)
plt.xlim(0, sr/2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT magnitude')
plt.tight_layout()

out_png = os.path.join(output_dir, 'Mavic_Test_File2_fft.png')
plt.savefig(out_png)
print(f"Saved filtered WAV to: {out_wav}")
print(f"Saved FFT plot to: {out_png}")

# Basic summary
peak_freq = fft_freq[np.argmax(fft_mag)]
print(f"Peak frequency in segment: {peak_freq:.1f} Hz")
