#!/usr/bin/env python3
"""
Plot 4-channel WAV (e.g. from jack_record_4ch.py) with channels stacked vertically.
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ---- Path to your recording ----
#wav_path = "Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_151227_48000Hz_4ch.wav" #"Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_141301_48000Hz_4ch.wav"#"Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_141112_48000Hz_4ch.wav"#"Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_140932_48000Hz_4ch.wav"#"Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_140817_48000Hz_4ch.wav"#"Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_140550_48000Hz_4ch.wav"#"Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_132236_48000Hz_4ch.wav"
wav_path = "Acoustics_system/Recordings/Four_mic_recordings/recording_20251127_152305_48000Hz_4ch.wav"#"Acoustics_system/Recordings/Four_mic_recordings/recording_20251127_151844_48000Hz_4ch.wav"
# ---- Read WAV ----
data, fs = sf.read(wav_path, always_2d=True)   # shape: (samples, channels)
n_samples, n_channels = data.shape
duration = n_samples / fs
t = np.linspace(0, duration, n_samples)

print(f"Loaded '{wav_path}'")
print(f"Channels: {n_channels}, Sample rate: {fs} Hz, Duration: {duration:.2f} s")

# ---- Plot stacked vertically ----
fig, axes = plt.subplots(n_channels, 1, figsize=(12, 8), sharex=True)

# Make sure axes is iterable
if n_channels == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.plot(t, data[:, i], linewidth=0.6)
    ax.set_ylabel(f"Ch {i+1}")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_xlim(0, duration)

# --- Make all y-axes equal ---
y_min = np.min(data)
y_max = np.max(data)
for ax in axes:
    ax.set_ylim(y_min, y_max)

axes[-1].set_xlabel("Time [s]")
fig.suptitle(f"Waveform ({n_channels} channels @ {fs/1000:.1f} kHz)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
