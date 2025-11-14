import numpy as np
import soundfile as sf
from scipy.signal import get_window
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
import cv2

# ---------------------------
# Parameters (edit as needed)
# ---------------------------
filename = "Acoustics_system/Recordings/Four_mic_recordings/Re-recording_of_Phantom_Test_File1.wav"
skip_seconds = 1.0
c = 343.0  # speed of sound [m/s] (20°C)
D_M = 1.0#0.20 # tetrahedron edge length [m] (example)

# Search grid
az_deg_step = 1.0#2.0     # azimuth step [deg]
el_deg_step = 1.0#2.0     # elevation step [deg]
azimuths = np.arange(0.0, 360.0, az_deg_step)
elevations = np.arange(0.0, 90.0 + 1e-6, el_deg_step)  # 0..90 inclusive

# GCC-PHAT interpolation factor (higher -> finer delay resolution, more CPU)
interp = 16

# Optional: process a single long block (simple & robust for stationary UAV segment)
# If you want frame-wise estimation, add framing and average SRP over frames.
use_hann_window = False#True

# ---------------------------
# Geometry: regular tetrahedron centered at origin
# Provided by you (kept as-is)
# ---------------------------
MIC_1 = np.array([D_M*np.sqrt(3)/3, 0.0, 0.0])
MIC_2 = np.array([-D_M*np.sqrt(3)/6, 0.5*D_M, 0.0])
MIC_3 = np.array([-D_M*np.sqrt(3)/6, -0.5*D_M, 0.0])
MIC_4 = np.array([0.0, 0.0, D_M*np.sqrt(6)/3])

mic_positions = np.stack([MIC_1, MIC_2, MIC_3, MIC_4], axis=0)
num_mics = mic_positions.shape[0]
assert num_mics == 4

# All unique mic pairs (i < j)
pairs = [(i, j) for i in range(num_mics) for j in range(i+1, num_mics)]

# ---------------------------
# Utility: angle <-> unit-vector (elevation=0 at ground, +90 up)
# ---------------------------
def sph_to_unit(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    # elevation measured from ground plane (0) to zenith (+90):
    # vertical component is sin(el), horizontal radius is cos(el)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])

def unit_to_sph_0to360_0to90(u):
    # u normalized
    x, y, z = u
    # elevation from ground plane: el = arcsin(z)
    el = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    # wrap azimuth to [0, 360)
    az = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
    return az, el

# ---------------------------
# GCC-PHAT cross-correlation (full cc + lags) for a pair
# ---------------------------
def gcc_phat_cc(sig, refsig, fs, interp=16):
    """
    Returns:
      cc: PHAT-weighted cross-correlation array (interpolated, centered)
      lags: time lags [s] corresponding to cc indices
    """
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()  # next pow2

    SIG = rfft(sig, n=nfft)
    REFSIG = rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + np.finfo(float).eps

    cc = irfft(R, n=interp * nfft)
    max_shift = int(interp * nfft / 2)
    # center it so zero lag is in the middle
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    # time axis for lags
    lags = np.arange(-max_shift, max_shift + 1, dtype=float) / (interp * fs)
    return cc, lags

# ---------------------------
# SRP-PHAT scoring
# ---------------------------
def srp_phat_direction(signals, fs, mic_positions, azimuths, elevations, c=343.0, interp=16, window=None):
    """
    signals: shape [T, 4]
    Returns: best_az_deg, best_el_deg, srp_map (El x Az), azimuths, elevations
    """
    assert signals.ndim == 2 and signals.shape[1] == 4
    # Optional windowing to reduce edge effects
    if window is not None:
        win = window(len(signals))
        signals = signals * win[:, None]

    # Precompute GCC-PHAT for all pairs once (single-block SRP)
    pair_cc = []
    pair_lags = []
    for (i, j) in pairs:
        cc, lags = gcc_phat_cc(signals[:, j], signals[:, i], fs, interp=interp)
        pair_cc.append(cc)
        pair_lags.append(lags)
    pair_cc = np.array(pair_cc, dtype=float)   # [num_pairs, L]
    pair_lags = np.array(pair_lags, dtype=float)  # [num_pairs, L]
    num_pairs = len(pairs)

    # Precompute mic dot products with candidate directions for speed
    # Build grid of unit vectors [El x Az x 3]
    U = np.zeros((len(elevations), len(azimuths), 3), dtype=float)
    for ei, el in enumerate(elevations):
        for ai, az in enumerate(azimuths):
            U[ei, ai, :] = sph_to_unit(az, el)

    # For each direction, compute predicted pair delays and sum the cc values
    srp = np.zeros((len(elevations), len(azimuths)), dtype=float)

    for ei in range(len(elevations)):
        for ai in range(len(azimuths)):
            u = U[ei, ai, :]  # unit vector
            # per-mic delays (relative to array origin) tau_i = -(r_i ⋅ u)/c
            tau = -np.dot(mic_positions, u) / c  # shape [4]
            s = 0.0
            for p_idx, (i, j) in enumerate(pairs):
                dt = tau[j] - tau[i]  # predicted TDOA for (i->j)
                # Interpolate cc at lag dt
                l = pair_lags[p_idx]
                cc = pair_cc[p_idx]
                # Find insertion index
                k = np.searchsorted(l, dt)
                if k <= 0:
                    val = cc[0]
                elif k >= len(l):
                    val = cc[-1]
                else:
                    # linear interpolation
                    t0, t1 = l[k-1], l[k]
                    w = (dt - t0) / (t1 - t0 + 1e-12)
                    val = (1.0 - w) * cc[k-1] + w * cc[k]
                s += val
            srp[ei, ai] = s

    # Locate maximum
    max_idx = np.unravel_index(np.argmax(srp), srp.shape)
    best_el = elevations[max_idx[0]]
    best_az = azimuths[max_idx[1]]
    return best_az, best_el, srp, azimuths, elevations

# ---------------------------
# Delay channels by predicted per-mic delays (optional beamformed check)
# ---------------------------
def fractional_delay(sig, delay_sec, fs):
    """Apply a fractional delay using frequency-domain phase shift."""
    N = len(sig)
    nfft = 1 << (N - 1).bit_length()
    SIG = rfft(sig, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    phase = np.exp(-1j * 2.0 * np.pi * freqs * delay_sec)
    y = irfft(SIG * phase, n=nfft)[:N]
    return y

def delay_and_sum(signals, fs, mic_positions, az_deg, el_deg, c=343.0):
    """Form a simple delay-and-sum beam towards (az,el)."""
    u = sph_to_unit(az_deg, el_deg)
    tau = -np.dot(mic_positions, u) / c  # per-mic delays
    aligned = []
    for m in range(signals.shape[1]):
        aligned.append(fractional_delay(signals[:, m], -tau[m], fs))  # negative to align (advance)
    aligned = np.stack(aligned, axis=1)
    return np.mean(aligned, axis=1), tau

def plot_srp_map(srp_map, azimuths, elevations, best_az, best_el,hexbin=False):
    plt.figure(figsize=(8, 5))

    ax = plt.gca()

    # Use hexbin or pcolormesh to plot SRP vs az/el
    if hexbin:
        x = azimuths.repeat(len(elevations))
        y = np.tile(elevations, len(azimuths))
        ax.hexbin(x, y, C=srp_map.T.flatten(), gridsize=50, cmap='viridis')
    else:
        # pcolormesh with azimuths (x) and elevations (y)
        ax.pcolormesh(azimuths, elevations, srp_map / (srp_map.max() + 1e-12), shading='auto', cmap='viridis')

    ax.set_xlim(azimuths[0], azimuths[-1])
    ax.set_ylim(elevations[0], elevations[-1])

    # Ensure both axes use the same data scale (1 deg in x equals 1 deg in y)
    ax.set_aspect('equal', adjustable='box')

    plt.colorbar(ax.collections[0], label='Normalized SRP-PHAT score')
    plt.title('SRP map')
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (deg)')
    plt.scatter([best_az], [best_el], color='white', marker='v', s=80, edgecolors='black', label='Estimated DOA')
    plt.legend()
    plt.tight_layout()
    plt.show()

def srp_to_image(srp_map):
    # Normalize to the range 0–255
    srp_norm = srp_map - np.min(srp_map)
    srp_norm = srp_norm / (np.max(srp_norm) + 1e-9)
    srp_img = (srp_norm * 255).astype(np.uint8)

    # Apply colormap (e.g., VIRIDIS equivalent)
    srp_color = cv2.applyColorMap(srp_img, cv2.COLORMAP_VIRIDIS)

    return srp_color

def draw_doa_marker(img, best_az, best_el, azimuths, elevations):
    H, W, _ = img.shape

    # convert azimuth to pixel (x)
    x = int((best_az - azimuths[0]) / (azimuths[-1] - azimuths[0]) * (W - 1))

    # convert elevation to pixel (y)
    y = int((best_el - elevations[0]) / (elevations[-1] - elevations[0]) * (H - 1))

    # draw a white circle
    cv2.circle(img, (x, y), 6, (255, 255, 255), -1)
    cv2.circle(img, (x, y), 10, (0, 0, 0), 2)

    return img

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    data, fs = sf.read(filename)
    print(f"Loaded {filename}, fs={fs} Hz, shape={data.shape}")

    if data.ndim == 1 or data.shape[1] != 4:
        raise ValueError("Expected a 4-channel WAV: shape [T, 4].")

    # Skip the first 1 s
    start = int(skip_seconds * fs)
    end=int(1.1 * fs)
    #signals = data[start:, :4].astype(float)
    signals = data[start:end, :4].astype(float)

    # Convert delays to seconds
    MIC_delays = {
        "MIC_1": 0.000,
        "MIC_2": 1414.658,
        "MIC_3": 2402.560,
        "MIC_4": 121.560
    }
    delays_s = {k: v * 1e-6 for k, v in MIC_delays.items()}

    # Apply each delay (frequency-domain phase shift)
    aligned_signals = np.zeros_like(signals)
    for i, mic in enumerate(["MIC_1", "MIC_2", "MIC_3", "MIC_4"]):
        d = delays_s[mic]
        aligned_signals[:, i] = fractional_delay(signals[:, i], d, fs)

    signals = aligned_signals

    # Optional window to stabilize correlations
    window_fn = get_window if use_hann_window else None
    window = (lambda N: get_window("hann", N)) if use_hann_window else None

    best_az, best_el, srp_map, az_grid, el_grid = srp_phat_direction(
        signals, fs, mic_positions, azimuths, elevations, c=c, interp=interp, window=window
    )

    print(f"\nEstimated DOA:")
    print(f"  Azimuth   : {best_az:.1f} deg (0..360)")
    print(f"  Elevation : {best_el:.1f} deg (0..90 from ground)")

    # srp_map has shape [len(elevations), len(azimuths)]
    # azimuths and elevations are 1D arrays of grid values (degrees)

    plot_srp_map(srp_map, az_grid, el_grid, best_az, best_el, hexbin=False)

    # Convert to image
    srp_img = srp_to_image(srp_map)

    # Draw marker
    #srp_img = draw_doa_marker(srp_img, best_az, best_el, azimuths, elevations)

    # Resize for nicer display (optional)
    #srp_img_big = cv2.resize(srp_img, (800, 600), interpolation=cv2.INTER_NEAREST)

    # Show image
    plt.imshow(cv2.cvtColor(srp_img, cv2.COLOR_BGR2RGB))
    plt.title("SRP Map")
    plt.show()
    


    # (Optional) form a beam in the estimated direction and report per-mic delays
    y_steered, tau_mics = delay_and_sum(signals, fs, mic_positions, best_az, best_el, c=c)
    print("\nPer-mic delays to steer at the estimate (seconds):")
    for i, tau_i in enumerate(tau_mics, 1):
        print(f"  MIC_{i}: {tau_i:+.6e} s")
"""
if __name__ == "__main__":
    data, fs = sf.read(filename)
    print(f"Loaded {filename}, fs={fs} Hz, shape={data.shape}")

    if data.ndim == 1 or data.shape[1] != 4:
        raise ValueError("Expected a 4-channel WAV: shape [T, 4].")

    # Skip the first second
    start = int(skip_seconds * fs)
    signals = data[start:, :4].astype(float)
    total_len = signals.shape[0]

    # --- Frame (segment) setup ---
    segment_dur = 0.1      # seconds per segment (10 Hz update rate)
    segment_len = int(fs * segment_dur)
    num_segments = total_len // segment_len

    print(f"Processing {num_segments} segments of {segment_dur:.3f} s each...")

    doa_results = []   # list to hold (time_center, az, el)

    for seg_idx in range(num_segments):
        start_idx = seg_idx * segment_len
        end_idx = start_idx + segment_len
        frame = signals[start_idx:end_idx, :]

        if frame.shape[0] < segment_len:
            break  # skip incomplete last frame

        # Optional window function
        window = (lambda N: get_window("hann", N)) if use_hann_window else None

        # Run SRP-PHAT on this frame
        best_az, best_el, _, _, _ = srp_phat_direction(
            frame, fs, mic_positions, azimuths, elevations,
            c=c, interp=interp, window=window
        )

        t_center = (start_idx + end_idx) / 2.0 / fs
        doa_results.append((t_center, best_az, best_el))
        print(f"Frame {seg_idx+1:03d}/{num_segments}: t={t_center:6.3f}s | "
              f"Az={best_az:6.1f}°, El={best_el:5.1f}°")

    # Convert to NumPy array for plotting or saving
    doa_results = np.array(doa_results)
    print("\nDone. Example result sample:")
    print(doa_results[:5])  # print first few rows: [time, az, el]"""