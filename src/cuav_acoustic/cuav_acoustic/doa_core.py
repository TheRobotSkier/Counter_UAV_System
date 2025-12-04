"""
doa_core.py

Core SRP-PHAT direction-of-arrival (DOA) estimation routines.

This module provides a vectorized, real-time–capable implementation of
Steered Response Power with Phase Transform (SRP-PHAT). It includes:

1) build_direction_grid       – build az/el → unit-vector grid
2) precompute_tau_grid        – geometry-only, compute once per session
3) gcc_phat_cc                – GCC-PHAT between two signals
4) srp_phat_precompute        – GCC-PHAT for all mic pairs (per frame)
5) compute_srp_map            – compute SRP-PHAT power map
6) find_best_direction        – argmax search of SRP map
7) doa_from_frame             – high-level wrapper for a single frame
8) plot_srp_map               – visualization helper (optional)

This file replaces the old srp_doa.py and follows the same documentation style.
"""

import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
#  Grid builder (unit-vector grid for azimuth/elevation)
# -------------------------------------------------------------------------
def build_direction_grid(
    azimuths: np.ndarray,
    elevations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a grid of unit direction vectors for all (azimuth, elevation)
    combinations.

    Parameters
    ----------
    azimuths : np.ndarray
        Array of azimuth angles in degrees, shape (A,)
    elevations : np.ndarray
        Array of elevation angles in degrees, shape (E,)

    Returns
    -------
    U_flat : np.ndarray
        Flattened array of unit vectors (x,y,z) for each grid point,
        shape (E*A, 3)
    AZ_grid : np.ndarray
        Azimuth grid mesh, shape (E, A)
    EL_grid : np.ndarray
        Elevation grid mesh, shape (E, A)
    """
    AZ_grid, EL_grid = np.meshgrid(azimuths, elevations)
    az_rad = np.deg2rad(AZ_grid)
    el_rad = np.deg2rad(EL_grid)

    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)

    U_flat = np.stack([x, y, z], axis=-1).reshape(-1, 3)
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
    Compute theoretical TDOAs τ_ij(az,el) for all microphone pairs (i,j)
    over a spherical grid. This depends ONLY on geometry and the speed of
    sound, so it is computed once per session.

    τ_i(az,el) = -(r_i ⋅ u(az,el)) / c
    τ_ij       = τ_j - τ_i

    Parameters
    ----------
    mic_positions : np.ndarray
        Array of microphone coordinates [M, 3]
    pairs : list[(i,j)]
        List of microphone index pairs
    azimuths : np.ndarray
        Grid of azimuth angles in degrees [A]
    elevations : np.ndarray
        Grid of elevation angles in degrees [E]
    c : float
        Speed of sound (m/s)

    Returns
    -------
    tau_grid : np.ndarray
        Theoretical TDOA grid, shape [P, E, A]
    max_tdoa_sec : float
        Maximum physically possible TDOA in seconds (max distance / c)
    """
    num_pairs = len(pairs)
    E = len(elevations)
    A = len(azimuths)

    U_flat, _, _ = build_direction_grid(azimuths, elevations)

    Tau_mic = - (mic_positions @ U_flat.T) / c  # [M, G]

    tau_pairs = np.empty((num_pairs, Tau_mic.shape[1]), dtype=float)
    d_max = 0.0

    for p_idx, (i, j) in enumerate(pairs):
        tau_pairs[p_idx, :] = Tau_mic[j] - Tau_mic[i]
        d = np.linalg.norm(mic_positions[j] - mic_positions[i])
        d_max = max(d_max, d)

    max_tdoa_sec = d_max / c
    tau_grid = tau_pairs.reshape(num_pairs, E, A)

    return tau_grid, max_tdoa_sec


# -------------------------------------------------------------------------
#  2) GCC-PHAT with lag cropping
# -------------------------------------------------------------------------
def gcc_phat_cc(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: float,
    interp: int = 16,
    lag_limit_sec: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GCC-PHAT between two microphone signals.

    Parameters
    ----------
    sig : np.ndarray
        Signal from microphone j, shape (T,)
    refsig : np.ndarray
        Signal from microphone i, shape (T,)
    fs : float
        Sampling rate (Hz)
    interp : int
        Interpolation factor (frequency-domain upsampling)
    lag_limit_sec : float or None
        Limit the output lags to ±lag_limit_sec seconds. If None, keep all.

    Returns
    -------
    cc : np.ndarray
        PHAT-weighted cross-correlation, shape (L,)
    lags : np.ndarray
        Time lags in seconds, shape (L,)
    """
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()

    SIG = rfft(sig, n=nfft)
    REFSIG = rfft(refsig, n=nfft)

    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + np.finfo(float).eps

    cc = irfft(R, n=interp * nfft)
    max_shift = interp * nfft // 2

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1, dtype=float) / (interp * fs)

    if lag_limit_sec is not None:
        mask = np.abs(lags) <= lag_limit_sec
        cc = cc[mask]
        lags = lags[mask]

    return cc, lags


# -------------------------------------------------------------------------
#  3) GCC-PHAT for all mic pairs (per frame)
# -------------------------------------------------------------------------
def srp_phat_precompute(
    signals: np.ndarray,
    fs: float,
    pairs: list[tuple[int, int]],
    max_tdoa_sec: float,
    interp: int = 16
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute GCC-PHAT for all microphone pairs in a given frame.

    Parameters
    ----------
    signals : np.ndarray
        Multichannel audio frame, shape (T, M)
    fs : float
        Sampling rate (Hz)
    pairs : list[(i,j)]
        Microphone index pairs
    max_tdoa_sec : float
        Maximum physically valid TDOA (for cropping)
    interp : int
        Interpolation factor

    Returns
    -------
    pair_cc : list[np.ndarray]
        List of PHAT-weighted cross-correlations, one per pair
    pair_lags : list[np.ndarray]
        Corresponding lag arrays
    """
    pair_cc = []
    pair_lags = []

    for (i, j) in pairs:
        cc, lags = gcc_phat_cc(
            signals[:, j],
            signals[:, i],
            fs,
            interp=interp,
            lag_limit_sec=max_tdoa_sec,
        )
        pair_cc.append(cc)
        pair_lags.append(lags)

    return pair_cc, pair_lags


# -------------------------------------------------------------------------
#  4) Compute SRP-PHAT power map
# -------------------------------------------------------------------------
def compute_srp_map(
    pair_cc: list[np.ndarray],
    pair_lags: list[np.ndarray],
    tau_grid: np.ndarray
) -> np.ndarray:
    """
    Compute the SRP-PHAT spatial power map using vectorized interpolation.

    SRP(az,el) = sum over mic pairs p of  C_p( τ_ij(az,el) )

    Parameters
    ----------
    pair_cc : list[np.ndarray]
        Cross-correlations per microphone pair
    pair_lags : list[np.ndarray]
        Corresponding lag arrays
    tau_grid : np.ndarray
        Precomputed TDOA grid, shape [P, E, A]

    Returns
    -------
    srp_map : np.ndarray
        SRP-PHAT score map, shape [E, A]
    """
    P, E, A = tau_grid.shape
    srp_map = np.zeros((E, A))

    for p in range(P):
        dt = tau_grid[p].ravel()
        cc = pair_cc[p]
        lags = pair_lags[p]

        contrib_flat = np.interp(dt, lags, cc)
        srp_map += contrib_flat.reshape(E, A)

    return srp_map


# -------------------------------------------------------------------------
#  5) Best azimuth/elevation
# -------------------------------------------------------------------------
def find_best_direction(
    srp_map: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray
) -> tuple[float, float]:
    """
    Locate the (azimuth, elevation) corresponding to the maximum SRP score.

    Parameters
    ----------
    srp_map : np.ndarray
        SRP-PHAT map, shape [E, A]
    azimuths : np.ndarray
        Azimuth grid [A]
    elevations : np.ndarray
        Elevation grid [E]

    Returns
    -------
    best_az : float
        Best azimuth angle in degrees
    best_el : float
        Best elevation angle in degrees
    """
    ei, ai = np.unravel_index(np.argmax(srp_map), srp_map.shape)
    return float(azimuths[ai]), float(elevations[ei])


# -------------------------------------------------------------------------
#  6) High-level wrapper — DOA from a single frame
# -------------------------------------------------------------------------
def doa_from_frame(
    frame: np.ndarray,
    fs: float,
    pairs: list[tuple[int, int]],
    tau_grid: np.ndarray,
    max_tdoa_sec: float,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    interp: int = 16
) -> tuple[float, float]:
    """
    High-level convenience function to compute DOA from a single
    multichannel audio frame.

    Performs:
        - GCC-PHAT for all microphone pairs
        - SRP-PHAT map computation
        - Argmax search for best (az, el)

    Parameters
    ----------
    frame : np.ndarray
        Multichannel frame, shape (T, M)
    fs : float
        Sampling rate (Hz)
    pairs : list[(i,j)]
        Microphone index pairs
    tau_grid : np.ndarray
        Precomputed geometry-based TDOA grid
    max_tdoa_sec : float
        Maximum physical TDOA
    azimuths : np.ndarray
        Azimuth search grid
    elevations : np.ndarray
        Elevation search grid
    interp : int
        GCC-PHAT interpolation factor

    Returns
    -------
    best_az : float
        DOA azimuth estimate in degrees
    best_el : float
        DOA elevation estimate in degrees
    """
    pair_cc, pair_lags = srp_phat_precompute(
        frame,
        fs,
        pairs,
        max_tdoa_sec,
        interp=interp
    )

    srp_map = compute_srp_map(pair_cc, pair_lags, tau_grid)
    return find_best_direction(srp_map, azimuths, elevations)


# -------------------------------------------------------------------------
#  7) Plotting helper
# -------------------------------------------------------------------------
def plot_srp_map(
    srp_map: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    best_az: float,
    best_el: float
) -> None:
    """
    Visualize SRP-PHAT map and highlight best DOA estimate.
    """
    plt.figure(figsize=(8, 5))
    mesh = plt.pcolormesh(
        azimuths, elevations, srp_map,
        shading="auto", cmap="viridis"
    )
    plt.colorbar(mesh, label="SRP-PHAT score")
    plt.scatter(best_az, best_el, c="white", edgecolors="black",
                marker="v", s=120, label="Estimated DOA")

    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("SRP-PHAT Map")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()
