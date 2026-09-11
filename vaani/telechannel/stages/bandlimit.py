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


def apply_bandlimit(x, sr=16000, low_hz=None, high_hz=None):
    """
    Band-limit `x` using a 5th-order Butterworth filter (zero-phase, via
    second-order-sections filtfilt).

    Defaults to the 300-3400 Hz telephony passband (the classic POTS
    bandwidth). Both cutoffs are overridable so a recipe can model channels
    with a wider/different passband — e.g. the `playback` recipe uses
    200-3800 Hz to approximate a small loudspeaker into a phone microphone
    (laptop-speaker low end + phone-mic high-end rolloff), which is NOT a
    telephony bandlimit but is exactly the kind of acoustic-loop shaping the
    retrained model needs to have seen.

    Args:
        x: 1-D input signal (numpy array).
        sr: Sample rate in Hz. Defaults to 16000.
        low_hz: Low cutoff in Hz; defaults to `LOW_HZ` (300).
        high_hz: High cutoff in Hz; defaults to `HIGH_HZ` (3400).

    Returns:
        numpy array the same length as `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    nyquist = sr / 2
    low = (LOW_HZ if low_hz is None else low_hz) / nyquist
    high = (HIGH_HZ if high_hz is None else high_hz) / nyquist

    sos = butter(ORDER, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)
