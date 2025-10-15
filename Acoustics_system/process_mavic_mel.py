import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

# librosa is optional; try import and give a helpful message if missing
try:
    import librosa
    import librosa.display
except Exception as e:
    raise SystemExit("librosa is required. Install with: pip install librosa")

root = os.path.dirname(__file__)
input_wav = os.path.join(root, 'Dataset_DronePoint', 'DS1', 'ExperimentallyCollected', 'Mavic_Test_File2.wav')
output_dir = os.path.join(root, 'output')
os.makedirs(output_dir, exist_ok=True)

# Parameters
lowcut = 20.0
highcut = 5000.0
n_samples = 10*4096
n_mels = 128

# Read WAV
sr, data = wavfile.read(input_wav)
# Convert to float
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

# Segment
if len(data) >= n_samples:
    segment = data[:n_samples]
else:
    segment = np.zeros(n_samples, dtype=np.float32)
    segment[:len(data)] = data

# Bandpass
sos = butter(4, [lowcut, highcut], btype='band', fs=sr, output='sos')
filtered = sosfiltfilt(sos, segment)

# Compute Mel-spectrogram with librosa
S = librosa.feature.melspectrogram(y=filtered, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels, fmin=lowcut, fmax=highcut)
S_db = librosa.power_to_db(S, ref=np.max)

# Plot
plt.figure(figsize=(8,4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmin=lowcut, fmax=highcut, hop_length=256)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (100-5000 Hz)')
plt.tight_layout()

out_png = os.path.join(output_dir, 'Mavic_Test_File2_mel.png')
plt.savefig(out_png)
print(f"Saved Mel-spectrogram to: {out_png}")

# Save numpy array if desired
out_npy = os.path.join(output_dir, 'Mavic_Test_File2_mel.npy')
np.save(out_npy, S_db)
print(f"Saved Mel spectrogram array to: {out_npy}")
