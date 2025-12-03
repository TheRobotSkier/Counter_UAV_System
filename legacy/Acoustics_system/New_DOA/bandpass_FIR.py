import numpy as np
from scipy.signal import firwin, lfilter

class FIRBandpass4Ch:
    """
    Stateful linear-phase FIR bandpass filter for 4-channel audio.
    Safe for block/stream processing and 50% overlap decoding.

    Expected input shape: [T, 4]  (time × channels)
    """

    def __init__(self, sr: int, lowcut: float, highcut: float, numtaps: int | None = None):
        """
        Parameters
        ----------
        sr : int
            Sample rate (supports 48000 or 96000 Hz)
        lowcut : float
            Lower cutoff frequency in Hz.
        highcut : float
            Upper cutoff frequency in Hz.
        numtaps : int, optional
            Length of FIR filter. Longer = better frequency precision, higher latency.
            Defaults: 401 @ 48 kHz, 801 @ 96 kHz (≈8–9 ms latency)
        """

        if sr not in (48000, 96000):
            raise ValueError("Sample rate must be 48000 or 96000 Hz.")

        # Recommended tap counts by SR:
        if numtaps is None:
            numtaps = 401 if sr == 48000 else 801

        self.sr = sr
        self.lowcut = lowcut
        self.highcut = highcut
        self.numtaps = numtaps

        # Design linear-phase FIR
        self.coeffs = firwin(
            numtaps,
            [lowcut, highcut],
            pass_zero=False,
            fs=sr
        )

        # One zi vector per channel (size = numtaps - 1)
        self.zi = np.zeros((self.numtaps - 1, 4), dtype=np.float32)

    def reset(self):
        """Reset filter state for streaming restart."""
        self.zi[:] = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Filter 4-channel frame.

        Parameters
        ----------
        frame : np.ndarray
            Input audio block, shape [T, 4]

        Returns
        -------
        out : np.ndarray
            Filtered frame, same shape [T, 4]
        """
        if frame.ndim != 2 or frame.shape[1] != 4:
            raise ValueError(f"Expected shape [T, 4], got {frame.shape}")

        frame = frame.astype(np.float32, copy=False)

        # Apply FIR filter channel-wise in streaming mode
        out = np.empty_like(frame)
        for ch in range(4):
            out[:, ch], self.zi[:, ch] = lfilter(
                self.coeffs, [1.0],
                frame[:, ch],
                zi=self.zi[:, ch]
            )

        return out
