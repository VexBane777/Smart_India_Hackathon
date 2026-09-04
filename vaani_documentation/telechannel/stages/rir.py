"""
rir.py — Room Impulse Response (RIR) convolution stage.

Simulates the acoustic effect of a room by convolving the dry input signal
with a measured/synthetic room impulse response, then aligning the wet
signal back to the direct path (the RIR's peak sample) so the convolution
doesn't introduce extra latency relative to the input.

See DOC1_TELECHANNEL_SPEC.md for the TeleChannel stage pipeline this module
plugs into (a later task chains these stages together; this module is
independent and does not do that chaining).
"""

import numpy as np
from scipy.signal import fftconvolve


def apply_rir(x, rir, wet_gain=0.7):
    """
    Convolve `x` with the room impulse response `rir` and mix the wet
    (reverberant) signal back with the dry signal.

    Args:
        x: 1-D input signal (numpy array).
        rir: 1-D room impulse response (numpy array). Despite the parameter
            name matching the brief's `rir_id`, this implementation takes
            the RIR samples directly rather than looking them up by id —
            no RIR database exists yet in this codebase, so id-based
            lookup is out of scope for this independent stage.
        wet_gain: Mix ratio in [0, 1] between dry (1 - wet_gain) and wet
            (wet_gain) signal. Defaults to 0.7.

    Returns:
        numpy array the same length as `x`: the dry/wet mix, time-aligned
        to the RIR's direct path (its peak sample).
    """
    rir = np.asarray(rir, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    rir = rir / np.max(np.abs(rir)) * 0.9
    d0 = np.argmax(np.abs(rir))
    wet = fftconvolve(x, rir)[d0:d0 + len(x)]
    return (1 - wet_gain) * x + wet_gain * wet
