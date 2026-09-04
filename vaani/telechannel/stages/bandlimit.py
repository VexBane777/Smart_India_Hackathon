"""
bandlimit.py — Band-limiting stage.

Simulates the narrow telephony passband (300-3400 Hz, the classic
"plain old telephone service" bandwidth) using a 5th-order Butterworth
bandpass filter, applied zero-phase via `sosfiltfilt` to avoid phase
distortion.

See DOC1_TELECHANNEL_SPEC.md for the TeleChannel stage pipeline this module
plugs into (a later task chains these stages together; this module is
independent and does not do that chaining).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

LOW_HZ = 300
HIGH_HZ = 3400
ORDER = 5


def apply_bandlimit(x, sr=16000):
    """
    Band-limit `x` to the 300-3400 Hz telephony passband using a 5th-order
    Butterworth filter (zero-phase, via second-order-sections filtfilt).

    Args:
        x: 1-D input signal (numpy array).
        sr: Sample rate in Hz. Defaults to 16000.

    Returns:
        numpy array the same length as `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    nyquist = sr / 2
    low = LOW_HZ / nyquist
    high = HIGH_HZ / nyquist

    sos = butter(ORDER, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)
