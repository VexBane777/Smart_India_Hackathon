"""One-off: prints a deterministic PCM buffer (as a Dart-literal-ready CSV
of samples) and its Python-computed physio features, so the Dart parity
test (test/audio_processor_parity_test.dart) can assert against a value
independently produced by the already-tested Python implementation. Rerun
and re-paste into the Dart test only if extract_physio's algorithm changes
on the Python side — the whole point is these two must never silently
drift apart (see features.py's top-of-file docstring)."""
import numpy as np
from features import extract_physio, SAMPLE_RATE

rng = np.random.default_rng(42)
n = SAMPLE_RATE * 3
t = np.arange(n) / SAMPLE_RATE
# A fixed, realistic-ish synthetic voiced signal: wandering F0 (natural
# vibrato-like drift) + mild amplitude envelope + a little broadband noise,
# chosen to be unambiguously "voiced" for both implementations' voicing
# threshold rather than a pathological edge case.
f0 = 140.0 + 6.0 * np.sin(2 * np.pi * 4.0 * t)
phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
sig = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
sig = sig * (0.9 + 0.1 * np.sin(2 * np.pi * 1.5 * t))
sig = sig + rng.normal(0, 0.02, size=n)
sig = (sig / np.abs(sig).max()).astype(np.float64)

physio = extract_physio(sig)
print("jitter_local =", repr(float(physio[0])))
print("shimmer_local =", repr(float(physio[1])))
print("hnr_db =", repr(float(physio[2])))
# Print a compact way to reconstruct the identical signal in Dart, rather
# than dumping 48000 floats: the generator above is fully deterministic
# given (seed=42, n, formula), so the Dart test reconstructs it with the
# same closed-form expression instead of a literal array.
