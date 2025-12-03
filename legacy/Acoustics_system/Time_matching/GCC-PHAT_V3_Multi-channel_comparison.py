import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# === GCC-PHAT function ===
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
    return tau


# === MAIN ===
filename = "Acoustics_system/Recordings/Four_mic_recordings/Re-recording_of_Phantom_Test_File1.wav"
data, fs = sf.read(filename)
print(f"Loaded {filename}, fs={fs} Hz, shape={data.shape}")

# --- Base channels ---
ch1, ch2, ch3, ch4 = [data[:, i] for i in range(4)]

# --- Simulate constant delays (for testing) ---
true_delays = {"ch2": 0.002, "ch3": 0.004, "ch4": 0.006}  # in seconds
delayed = {"ch1": ch1}
for ch_name, delay in true_delays.items():
    samples = int(delay * fs)
    base = data[:, int(ch_name[-1]) - 1]
    delayed[ch_name + "_delayed"] = np.concatenate((np.zeros(samples), base[:-samples]))

# --- Parameters ---
segment_dur = 0.1      # seconds (10 Hz update rate)
segment_len = int(fs * segment_dur)
num_segments = len(ch1) // segment_len
tolerance = 0.0003     # 0.3 ms

# --- Channel pairs to compare ---
pairs = [
    ("ch1", "ch2_delayed"),
    ("ch1", "ch3_delayed"),
    ("ch1", "ch4_delayed"),
    ("ch2_delayed", "ch3_delayed"),
    ("ch2_delayed", "ch4_delayed"),
    ("ch3_delayed", "ch4_delayed"),  # new pair
]

# --- GCC-PHAT over all segments ---
results = {pair: [] for pair in pairs}
correct = {pair: 0 for pair in pairs}
wrong = {pair: 0 for pair in pairs}

for i in range(num_segments):
    s, e = i * segment_len, (i + 1) * segment_len
    for pair in pairs:
        sig = delayed[pair[0]][s:e]
        ref = delayed[pair[1]][s:e]
        tau = gcc_phat(sig, ref, fs, interp=16)
        results[pair].append(tau)

        # Determine true relative delay if both have known offsets
        true_delay = abs(
            true_delays.get(pair[0].replace("_delayed", ""), 0)
            - true_delays.get(pair[1].replace("_delayed", ""), 0)
        )
        if abs(abs(tau) - true_delay) <= tolerance:
            correct[pair] += 1
        else:
            wrong[pair] += 1

# --- Print results ---
print("\n=== Accuracy per pair ===")
for pair in pairs:
    total = correct[pair] + wrong[pair]
    acc = 100 * correct[pair] / total if total else 0
    print(f"{pair[0]} vs {pair[1]}: "
          f"Accuracy {acc:5.1f}%  (Correct {correct[pair]}, Wrong {wrong[pair]})")

# --- Plot estimated delays over time ---
time_axis = np.arange(num_segments) * segment_dur
plt.figure(figsize=(10,6))
for pair in pairs:
    plt.plot(time_axis, np.array(results[pair])*1000, label=f"{pair[0]} vs {pair[1]}")
plt.title("Estimated GCC-PHAT Delays Between Channel Pairs")
plt.xlabel("Time (s)")
plt.ylabel("Estimated delay (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
