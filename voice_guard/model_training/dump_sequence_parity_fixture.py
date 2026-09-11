"""Same approach as dump_parity_fixture.py (track 2), extended to the
per-frame sequence. Prints the first and last frame's LFCC coefficients
(not all 184 frames — too much to hand-paste) plus the 6 scalars, so the
Dart parity test can assert against Python-computed ground truth without
shipping a large fixture file.

Deliberately NOISE-FREE (unlike dump_parity_fixture.py's fixture): LFCC
coefficients are computed per-frame from raw spectral content, so any
per-sample noise lands directly in every frame's values — unlike physio's
frame-*aggregate* statistics, which average noise-realization differences
out. A noisy fixture made Python/Dart per-frame LFCC values diverge by
several units on ~half the coefficients purely from numpy's and Dart's
PRNGs producing different noise sequences (confirmed by inspection, not
a real algorithm bug) — a noise-free deterministic signal sidesteps that
entirely and lets this test hold LFCC to a tight, meaningful tolerance."""
import numpy as np
from features import extract_lfcc_sequence, extract_scalars, SAMPLE_RATE

n = SAMPLE_RATE * 3
t = np.arange(n) / SAMPLE_RATE
f0 = 140.0 + 6.0 * np.sin(2 * np.pi * 4.0 * t)
phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
sig = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
sig = (sig / np.abs(sig).max()).astype(np.float64)

seq = extract_lfcc_sequence(sig)
scalars = extract_scalars(sig)
print("n_frames =", seq.shape[0])
print("first_frame =", repr(seq[0].tolist()))
print("last_frame =", repr(seq[-1].tolist()))
print("scalars =", repr(scalars.tolist()))
