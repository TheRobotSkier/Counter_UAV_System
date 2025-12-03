import numpy as np
from scipy.signal import butter, sosfiltfilt


def design_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Design a Butterworth bandpass filter using second-order sections (SOS).

    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    sos : np.ndarray
        SOS representation of the bandpass filter.

    Notes
    -----
    The SOS format is numerically stable for higher-order IIR filters.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def apply_bandpass(data: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Apply zero-phase bandpass filtering to mono or multi-channel audio.

    Parameters
    ----------
    data : np.ndarray
        Input signal, shape (T,) for mono or (T, M) for multi-channel.
    sos : np.ndarray
        Second-order sections filter representation from `design_bandpass`.

    Returns
    -------
    filtered : np.ndarray
        Filtered output signal, same shape as `data`.

    Notes
    -----
    This uses `sosfiltfilt` which applies forward-reverse filtering.
    This ensures zero-phase distortion, which is required for DOA
    algorithms such as SRP-PHAT where inter-channel phase differences
    must remain intact.
    """
    return sosfiltfilt(sos, data, axis=0)