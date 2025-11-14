# simulation.py
"""
Geometry & simulation utilities for a tetrahedral microphone array.

- Build regular tetrahedron mic array
- Compute travel times / TDOAs for a given 3D source position
- Optionally simulate multi-channel signals with the correct delays
- Visualization to plot array & source in 3D
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt
import soundfile as sf
from numpy.fft import rfft, irfft

def regular_tetrahedron_array(d_m: float) -> np.ndarray:
    """
    Return 4x3 array of mic positions (regular tetrahedron) centered at origin.
    d_m: edge length [m]
    """
    mic_1 = np.array([d_m * np.sqrt(3) / 3, 0.0, 0.0])
    mic_2 = np.array([-d_m * np.sqrt(3) / 6, 0.5 * d_m, 0.0])
    mic_3 = np.array([-d_m * np.sqrt(3) / 6, -0.5 * d_m, 0.0])
    mic_4 = np.array([0.0, 0.0, d_m * np.sqrt(6) / 3])

    mic_positions = np.array([mic_1, mic_2, mic_3, mic_4])
    return mic_positions

def compute_travel_times(source_pos: np.ndarray,
                         mic_positions: np.ndarray,
                         c: float):
    """
    Compute travel distances, absolute arrival times and pairwise TDOAs.

    source_pos : [3]
    mic_positions: [4, 3]
    c: speed of sound [m/s]

    Returns
    -------
    distances: [4]
        in meters
    times : [4] 
        absolute arrival times [s]
    relative_times : [4] 
        (0 = earliest mic) [s]
    pairs: 
        list of (i, j)
    pair_tdoas: 
        dict {(i,j): t_i - t_j}
    """
    # Distances and absolute times
    distances = np.linalg.norm(source_pos[None, :] - mic_positions, axis=1)
    times = distances / c

    min_time = np.min(times)
    relative_times = times - min_time

    # Unique mic pairs
        # Using itertools.combinations to avoid duplicate pairs
            # itertools.combinations(range(4), 2) → generates all unique 2-element combinations of those indices (without repetition, and order doesn’t matter).
    pairs = list(itertools.combinations(range(mic_positions.shape[0]), 2)) # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pair_tdoas = {}
    for i, j in pairs:
        pair_tdoas[(i, j)] = times[i] - times[j]

    return distances, times, relative_times, pairs, pair_tdoas

# ------------- Simulation of delayed signals  ------------- #

def load_soundfile(filename: str):
    """Load multi-channel WAV file.
    
    Expected shape: [T, 4]

    Returns
    -------
    data : np.ndarray [T, 4]
    fs : int
    """
    data, fs = sf.read(filename)
    print(f"Loaded {filename}, fs={fs} Hz, shape={data.shape}")

    # Validate shape [T, 4]
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("Expected a 4-channel WAV file: shape [T, 4].")

    return data, fs


# Helper: Fractional delay using frequency-domain phase rotation
def fractional_delay_fd(signal: np.ndarray, delay_sec: float, fs: float) -> np.ndarray:
    """
    Apply a fractional time delay using a frequency-domain phase shift.
    Output length equals input length (no padding).

    Parameters
    ----------
    signal : np.ndarray
        1-D array, shape (T,)
    delay_sec : float
        Delay in seconds (positive = shift later)
    fs : float
        Sampling rate (Hz)

    Returns
    -------
    delayed : np.ndarray
        Delayed signal, same length as input
    """
    T = len(signal)
    # Use FFT size = next pow2 >= T (no huge inflation needed)
    nfft = 1 << (T - 1).bit_length()

    SIG = rfft(signal, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Apply phase shift exp(-j*2πfτ)
    phase = np.exp(-1j * 2.0 * np.pi * freqs * delay_sec)
    shifted = irfft(SIG * phase, n=nfft)

    return shifted[:T]

# Main function: delay multichannel audio + trim edges
def delay_channel_soundfile(data: np.ndarray, fs: int, relative_times: np.ndarray) -> np.ndarray:
    """
    Apply fractional time delays to a multi-channel signal, then trim
    invalid regions caused by time shifting.

    Parameters
    ----------
    data : np.ndarray
        Multi-channel audio [T, 4]
    fs : int
        Sampling rate
    relative_times : np.ndarray
        Delay per channel in SECONDS, shape [4]

    Returns
    -------
    delayed_trimmed : np.ndarray
        Delayed and trimmed audio, shape [T', 4]
    """
    # Validate input shape?
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("Expected data of shape [T, 4].")

    T, M = data.shape

    # Convert to sample delays for trimming only
    delays_samples = relative_times * fs
    max_shift = int(np.ceil(np.abs(delays_samples).max()))

    # Apply fractional delay BEFORE trimming
    delayed = np.zeros_like(data)
    for ch in range(M):
        delayed[:, ch] = fractional_delay_fd(
            data[:, ch],     # input channel
            relative_times[ch],  # delay SEC
            fs
        )

    # Now trim after delay
    start = max_shift
    end   = T - max_shift

    # Validate trimming range
    if end <= start:
        raise ValueError(
            f"Signal too short to support trimming for delays of ±{max_shift} samples."
        )

    delayed_trimmed = delayed[start:end, :]
    return delayed_trimmed




# ------------- Visualization  ------------- #

def _set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_array_and_source(mic_positions: np.ndarray, source_pos: np.ndarray):
    """Quick 3D visualization of array & source."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Origin
    ax.scatter(0, 0, 0, c="k", marker="x", s=80, label="Origin")

    # Mics
    ax.scatter(mic_positions[:, 0],
               mic_positions[:, 1],
               mic_positions[:, 2],
               c="b", marker="o", s=80, label="Microphones")

    # Source
    ax.scatter(source_pos[0], source_pos[1], source_pos[2],
               c="r", marker="*", s=150, label="Sound Source")

    # Connect microphones (edges of tetrahedron)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        ax.plot([mic_positions[i, 0], mic_positions[j, 0]],
                [mic_positions[i, 1], mic_positions[j, 1]],
                [mic_positions[i, 2], mic_positions[j, 2]], "k--", linewidth=0.8)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    ax.set_title("Tetrahedral Microphone Array Simulation")
    ax.grid(True)
    _set_axes_equal(ax)
    plt.tight_layout()
    plt.show()