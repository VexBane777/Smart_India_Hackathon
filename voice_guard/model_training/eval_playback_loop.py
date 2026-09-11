"""Validates the deployed model against the acoustic "loudspeaker -> phone mic"
re-recording loop that the 2026-09-11 report identified as the live-capture
failure mode.

Background: direct (clean) .wav scoring works, but playing the same file
through a computer speaker and having the phone's recorder pick it up drops
the model's AI-probability to ~0.01-0.30 even on fakes that score 0.95+
clean. The mechanism (measured with the deployed v11_seqcnn ONNX): room
reverb, speaker/phone-mic coloration and ambient noise flatten exactly the
prosody/LFCC cues the model keys on. Pure level attenuation is NOT the
cause (a -24 dB scale barely moves the score).

This script scores each test asset twice:
  1. clean        — the exact score the app SHOULD produce for a direct file
  2. simulation   — the same audio after the speaker->mic loop approximation
     below (the same transform used during the diagnosis)

and prints both. `--gate-loop-min` exits nonzero if a retrained model still
cannot push looped fakes above the given threshold, so a model can't be
shipped (copied into assets/models/voice_detector.onnx) on clean accuracy
alone.

Usage (run from voice_guard/model_training/):
    python eval_playback_loop.py \
        --onnx ../assets/models/voice_detector.onnx \
        --assets test_assets \
        --gate-clean-min 0.60 \
        --gate-loop-min 0.50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _softmax_ai_prob(logits: np.ndarray) -> float:
    """2-class softmax over [real, fake], returning P(fake)."""
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max()
    e = np.exp(logits)
    return float((e / e.sum())[1])


def _last_3s_window(pcm: np.ndarray) -> np.ndarray:
    """The app scores the LAST 48,000 samples (3s @ 16k) of its 5s buffer."""
    pcm = np.asarray(pcm, dtype=np.float64)
    if len(pcm) < 48000:
        pcm = np.pad(pcm, (48000 - len(pcm), 0))
    return pcm[-48000:]


def simulate_speaker_to_mic_loop(x: np.ndarray, seed: int = 7) -> np.ndarray:
    """Approximate laptop loudspeaker -> room -> phone microphone.

    Kept as a faithful, seeded approximation of the 2026-09-11 diagnosis
    harness (not a telechannel recipe — that's for training):
      * phone-mic high-pass at 120 Hz and small-speaker low-pass at 3.8 kHz
      * room reverb: synthetic decaying-noise RIR, RT60 ~ 350 ms
      * ambient room noise at ~15 dB SNR
      * mic preamp gain -15..-3 dB (AGC-free capture; the phone's own DSP
        adds more, making real loops harsher than this)
    """
    from scipy import signal as dsp

    rng = np.random.default_rng(seed)
    sr = 16000
    sos = dsp.butter(2, 120 / (sr / 2), "highpass", output="sos")
    y = dsp.sosfilt(sos, x)
    sos = dsp.butter(4, 3800 / (sr / 2), "lowpass", output="sos")
    y = dsp.sosfilt(sos, y)

    rt60 = 0.35
    length = int(0.7 * sr)
    t = np.arange(length) / sr
    rir = rng.standard_normal(length) * np.exp(-6.91 * t / rt60)
    rir[0] = 1.0
    rir = rir / (np.sqrt((rir ** 2).sum()) + 1e-9)
    y = 0.75 * y + 0.45 * np.convolve(y, rir)[: len(y)]

    y = y / (np.sqrt((y ** 2).mean()) + 1e-9)
    y = y + rng.standard_normal(len(y)) * 10 ** (-15 / 20)
    y = y * 10 ** (rng.uniform(-15, -3) / 20)
    return np.clip(y, -1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=Path("../assets/models/voice_detector.onnx"))
    ap.add_argument("--assets", type=Path, default=Path("test_assets"))
    ap.add_argument("--gate-clean-min", type=float, default=0.60,
                    help="fail if any fake asset scores below this clean. Use this "
                         "to catch a retrain that regressed the direct-file path.")
    ap.add_argument("--gate-loop-min", type=float, default=None,
                    help="fail if any fake asset's simulated-loop score is below this. "
                         "This is the fix gate: a model that still collapses on the "
                         "speaker->mic loop must not be shipped.")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # local import: heavy-ish dep, optional for feature checks
    except ImportError:
        raise SystemExit("onnxruntime not installed - this harness scores the deployed ONNX model.")

    # Import AFTER parse so `--help` works without the repo-path hack.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import extract_lfcc_sequence, extract_scalars

    session = ort.InferenceSession(str(args.onnx))
    print(f"Model: {args.onnx}")
    print(f"{'asset':<32}{'clean':>8}{'loop':>8}")
    failures = []
    for wav_path in sorted(Path(args.assets).glob("*.wav")):
        x, sr = sf.read(str(wav_path), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            sr = 16000

        clean = _last_3s_window(x)
        looped = simulate_speaker_to_mic_loop(clean)

        def score(pcm: np.ndarray) -> float:
            seq = extract_lfcc_sequence(pcm).astype(np.float32)[None, ...]
            sc = extract_scalars(pcm).astype(np.float32)[None, ...]
            out = session.run(None, {"lfcc_sequence": seq, "scalars": sc})
            return _softmax_ai_prob(out[0][0])

        sc_clean = score(clean)
        sc_loop = score(looped)
        print(f"{wav_path.name:<32}{sc_clean:>8.3f}{sc_loop:>8.3f}")

        if sc_clean < args.gate_clean_min:
            failures.append(f"{wav_path.name}: clean score {sc_clean:.3f} < {args.gate_clean_min}")
        if args.gate_loop_min is not None and sc_loop < args.gate_loop_min:
            failures.append(
                f"{wav_path.name}: speaker->mic loop score {sc_loop:.3f} < {args.gate_loop_min}"
                " - the model still collapses on acoustic re-recording"
            )

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nAll gates passed.")


if __name__ == "__main__":
    main()