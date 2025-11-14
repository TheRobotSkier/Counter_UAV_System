# srp_doa.py
"""
Vectorized SRP-PHAT implementation with:

1) precompute_tau_grid   – geometry-only, computed once
2) srp_phat_precompute   – GCC-PHAT for each frame/block
3) compute_srp_map       – vectorized SRP-PHAT map using tau_grid
4) find_best_direction   – argmax search
5) plot_srp_map          – visualize results

This module is optimized for real-time DOA estimation (≥10 Hz)
on tetrahedral microphone arrays or arbitrary geometries.
"""

import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt



def build_direction_grid(
    azimuths: np.ndarray,
    elevations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a grid of unit vectors for all (azimuth, elevation) pairs.

    Parameters
    ----------
    azimuths : np.ndarray
        Array of azimuth angles in degrees, shape (A,)
    elevations : np.ndarray
        Array of elevation angles in degrees, shape (E,)

    Returns
    -------
    U_flat : np.ndarray
        Flattened array of unit vectors for each grid point, shape (E*A, 3)
    AZ_grid : np.ndarray
        Azimuth grid mesh, shape (E, A)
    EL_grid : np.ndarray
        Elevation grid mesh, shape (E, A)
    """
    AZ_grid, EL_grid = np.meshgrid(azimuths, elevations)  # shapes [E, A]
    az_rad = np.deg2rad(AZ_grid)
    el_rad = np.deg2rad(EL_grid)

    # Spherical to unit Cartesian conversion
    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)

    U_flat = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # [G,3]
    return U_flat, AZ_grid, EL_grid


# -------------------------------------------------------------------------
#  1) Precompute τ-grid (geometry only, compute ONCE)
# -------------------------------------------------------------------------
def precompute_tau_grid(
    mic_positions: np.ndarray,
    pairs: list[tuple[int, int]],
    azimuths: np.ndarray,
    elevations: np.ndarray,
    c: float
) -> tuple[np.ndarray, float]:
    """
    Compute theoretical TDOAs τ_ij(az,el) for all mic pairs (i,j)
    and all grid directions. This depends ONLY on geometry and speed
    of sound, so compute once per session.

    τ_i(az,el) = -(r_i ⋅ u(az,el)) / c
    τ_ij = τ_j - τ_i

    Parameters
    ----------
    mic_positions : np.ndarray
        Array of mic coordinates [M,3]
    pairs : list[(i,j)]
        Microphone index pairs
    azimuths : np.ndarray
        Grid of azimuths in degrees [A]
    elevations : np.ndarray
        Grid of elevations in degrees [E]
    c : float
        Speed of sound

    Returns
    -------
    tau_grid : np.ndarray
        Theoretical TDOA grid, shape [P, E, A]
    max_tdoa_sec : float
        Maximum physically possible TDOA, i.e. max_distance / c
    """
    num_pairs = len(pairs)
    E = len(elevations)
    A = len(azimuths)

    # Build flattened direction grid U_flat [E*A,3]
    U_flat, _, _ = build_direction_grid(azimuths, elevations)  # [G,3]

    # tau_mic[m, g] = -(r_m ⋅ u_g) / c
    Tau_mic = - (mic_positions @ U_flat.T) / c  # [M, G]

    # Convert to pair TDOAs: tau_ij = tau_j - tau_i
    tau_pairs = np.empty((num_pairs, Tau_mic.shape[1]), dtype=float)  # [P,G]
    d_max = 0.0

    for p_idx, (i, j) in enumerate(pairs):
        tau_pairs[p_idx, :] = Tau_mic[j, :] - Tau_mic[i, :]

        # Compute actual physical distance once (for lag cropping)
        d = np.linalg.norm(mic_positions[j] - mic_positions[i])
        d_max = max(d_max, d)

    # Maximum possible TDOA
    max_tdoa_sec = d_max / c
    
    # Reshape to [P, E, A]
    tau_grid = tau_pairs.reshape(num_pairs, E, A)

    return tau_grid, max_tdoa_sec


# ---------------------------------------------------------------------
# 2) GCC-PHAT with lag cropping
# ---------------------------------------------------------------------

def gcc_phat_cc(sig: np.ndarray,
                refsig: np.ndarray,
                fs: float,
                interp: int = 16,
                lag_limit_sec: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GCC-PHAT between two microphone signals.

    Parameters
    ----------
    sig : np.ndarray
        Signal from microphone j, shape (T,)
    refsig : np.ndarray
        Signal from microphone i, shape (T,)
    fs : float
        Sampling frequency in Hz
    interp : int
        Interpolation factor (upsampling in frequency domain)
    lag_limit_sec : float or None
        If provided, limits output lags to ±lag_limit_sec (physically possible TDOAs)

    Returns
    -------
    cc : np.ndarray
        PHAT-weighted cross-correlation, shape (L,)
    lags : np.ndarray
        Time lags (in seconds), shape (L,)
    """
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()

    SIG = rfft(sig, n=nfft)
    REFSIG = rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + np.finfo(float).eps

    cc = irfft(R, n=interp * nfft)
    max_shift = interp * nfft // 2

    # center it so zero lag is in the middle
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1, dtype=float) / (interp * fs)

    # Crop to physically possible TDOAs
    if lag_limit_sec is not None:
        # mask = True for lags inside ±lag_limit_sec, False otherwise
        # This removes impossible TDOAs (those caused by FFT wrap-around)
        mask = np.abs(lags) <= lag_limit_sec
        lags = lags[mask]
        cc = cc[mask]

    return cc, lags



def srp_phat_precompute(
    signals: np.ndarray,
    fs: float,
    pairs: list[tuple[int, int]],
    max_tdoa_sec: float,
    interp: int = 16
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Precompute GCC-PHAT for all microphone pairs.

    Parameters
    ----------
    signals : np.ndarray
        Audio frame, shape (T, M)
    fs : float
        Sampling frequency
    pairs : list[(i,j)]
        Microphone index pairs
    max_tdoa_sec : float
        Maximum possible TDOA from geometry
    interp : int
        Interpolation factor

    Returns
    -------
    pair_cc : list[np.ndarray]
        List of cross-correlations per pair
    pair_lags : list[np.ndarray]
        List of lag arrays per pair
    """
    pair_cc: list[np.ndarray] = []
    pair_lags: list[np.ndarray] = []

    for (i, j) in pairs:
        cc, lags = gcc_phat_cc(
            signals[:, j], signals[:, i],
            fs, interp=interp,
            lag_limit_sec=max_tdoa_sec
        )
        pair_cc.append(cc)
        pair_lags.append(lags)

    return pair_cc, pair_lags

# -------------------------------------------------------------------------
#  3) Vectorized SRP-PHAT map
# -------------------------------------------------------------------------

def compute_srp_map(
    pair_cc: list[np.ndarray],
    pair_lags: list[np.ndarray],
    tau_grid: np.ndarray
) -> np.ndarray:
    """
    Compute SRP-PHAT map using vectorized interpolation.

    SRP(az,el) = sum over all pairs p:
                       C_p( tau_p(az,el) )

    Parameters
    ----------
    pair_cc : list[np.ndarray]
        GCC-PHAT cross-correlations per pair
    pair_lags : list[np.ndarray]
        Corresponding lags per pair
    tau_grid : np.ndarray
        Precomputed τ_ij grid, shape [P, E, A]

    Returns
    -------
    srp_map : np.ndarray
        SRP-PHAT score map, shape [E, A]
    """
    P, E, A = tau_grid.shape
    srp_map = np.zeros((E, A), dtype=float)

    # Loop over mic pairs (only ~6 pairs for tetrahedron)
    # Inside: vectorized interpolation over all grid points.
    for p in range(P):
        dt = tau_grid[p].ravel()                          # [E*A]
        lags = pair_lags[p]                               # [L_p]
        cc = pair_cc[p]                                   # [L_p]

        # Vectorized interpolation
        contrib_flat = np.interp(dt, lags, cc)            # [E*A]

        # Accumulate into SRP map
        srp_map += contrib_flat.reshape(E, A)

    return srp_map

# -------------------------------------------------------------------------
#  4) Grid search
# -------------------------------------------------------------------------

def find_best_direction(
    srp_map: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray
) -> tuple[float, float]:
    """
    Find the az/el corresponding to highest SRP score.

    Returns
    -------
    best_az : float
    best_el : float
    """
    ei, ai = np.unravel_index(np.argmax(srp_map), srp_map.shape)
    return float(azimuths[ai]), float(elevations[ei])

# -------------------------------------------------------------------------
#  5) Plotting
# -------------------------------------------------------------------------

def plot_srp_map(
    srp_map: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    best_az: float,
    best_el: float
) -> None:
    """
    Plot SRP-PHAT map and highlight best direction.
    """
    plt.figure(figsize=(8, 5))
    mesh = plt.pcolormesh(azimuths, elevations, srp_map,
                           shading="auto", cmap="viridis")
    plt.colorbar(mesh, label="SRP-PHAT score")

    plt.scatter(best_az, best_el,
                c="white", edgecolors="black",
                marker="v", s=120, label="Estimated DOA")

    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("SRP-PHAT Map")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.show()