import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# --- GCC-PHAT function ---
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


# --- MAIN ---
filename = "Acoustics_system/Recordings/Four_mic_recordings/Re-recording_of_Phantom_Test_File1.wav"#"Acoustics_system/Recordings/Four_mic_recordings/recording_20251027_151227_48000Hz_4ch.wav"
data, fs = sf.read(filename)
print(f"Loaded {filename}, fs={fs} Hz, shape={data.shape}")

# --- Use ch1 and a delayed version of ch2 for testing ---
ch1 = data[:, 0]
ch2 = data[:, 1].copy()

# Simulate a constant delay of 2 ms for testing (you can remove later)
true_delay_sec = 0.002
delay_samples = int(true_delay_sec * fs)
ch2_delayed = np.concatenate((np.zeros(delay_samples), ch2[:-delay_samples]))

# --- Process in 0.1 s segments ---
segment_dur = 0.1      # seconds
segment_len = int(fs * segment_dur)
num_segments = len(ch1) // segment_len
print(f"Processing {num_segments} segments of {segment_dur}s each")

estimated_delays = []
correct = 0
wrong = 0
tolerance = 0.0002  # 0.2 ms tolerance for "correct" estimate

for i in range(num_segments):
    s = i * segment_len
    e = s + segment_len
    seg1 = ch1[s:e]
    seg2 = ch2_delayed[s:e]

    # Skip short last block
    if len(seg1) < segment_len:
        continue

    tau = gcc_phat(seg1, seg2, fs, interp=16)
    estimated_delays.append(tau)
    print(f"Segment {i+1}/{num_segments}: Estimated delay = {tau*1000:.3f} ms, difference from true = {(abs(tau) - true_delay_sec)*1000:.3f} ms")

    if abs(tau) - true_delay_sec <= tolerance:
        correct += 1
    else:
        wrong += 1

# --- Results ---
total = correct + wrong
accuracy = 100 * correct / total if total > 0 else 0
print(f"\nTrue delay: {true_delay_sec*1000:.3f} ms")
print(f"Correct: {correct}, Wrong: {wrong}, Accuracy: {accuracy:.1f}%")

# --- Plot delay vs time ---
times = np.arange(num_segments) * segment_dur
plt.figure(figsize=(10,4))
plt.plot(times, np.array(estimated_delays)*1000, 'b.-')
plt.axhline(true_delay_sec*1000, color='r', linestyle='--', label='True delay')
plt.title("Estimated Time Delay per 0.1 s Segment")
plt.xlabel("Time (s)")
plt.ylabel("Estimated Delay (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
