"""Mix real recorded noise (MUSAN, SLR17) into real speech clips, addressing
the root cause found in voice_guard/state.md's "Accent/clone data pipeline"
session: the deployed model was trained on studio-quality (ASVspoof) and
pre-recorded (In-the-Wild) audio, never against real speech captured with
real ambient room noise — so live-mic energyVariance/zcrVariance features
looked "foreign" to it and it scored noisy genuine speech as fake.

This augments the REAL side only (not fake) — the goal is teaching the
model "noisy ≠ synthetic", not adding new fake examples. Deliberately
separate from TeleChannel's noise stage (vaani/telechannel/stages/noise.py),
which only generates synthetic white/pink noise and has no real-recorded-
noise mixing capability — extending that shared, tested module was judged
out of scope for this pass; this script achieves the same goal standalone,
writing already-mixed WAVs that need no further --channel processing.

MUSAN noise categories used: free-sound + sound-bible (real-world ambient/
environmental recordings — traffic, crowds, appliances, etc.), not the
music or speech subsets (those would teach the wrong thing here). License:
MUSAN is CC-BY-4.0.

Usage (after data/musan/musan/noise/ exists — see runs/download_openslr_musan.ps1):
    python augment_with_noise.py --real-dir data/accents/real/en_native \
        --out-dir data/real_noise_aug/en_native --snr-db-range 5 20
    # repeat per real/<cell> dir you want noise-augmented, or point --real-dir
    # at data/real (the reconstructed ASVspoof2019 LA train bonafide set) too

Output feeds into train.py as an additional --real-clean dir (already
noise-mixed, no further --channel degradation needed).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MUSAN_NOISE_SUBDIRS = ("free-sound", "sound-bible")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _load_mono(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    # Loop/trim noise to match speech length.
    if len(noise) < len(speech):
        reps = len(speech) // len(noise) + 1
        noise = np.tile(noise, reps)
    start = random.randint(0, max(0, len(noise) - len(speech)))
    noise = noise[start : start + len(speech)]

    speech_rms = _rms(speech)
    noise_rms = _rms(noise)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    if noise_rms > 0:
        noise = noise * (target_noise_rms / noise_rms)
    mixed = speech + noise
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak  # avoid clipping; keeps relative SNR intact
    return mixed.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, required=True, help="source real/ WAV dir")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--musan-root", type=Path, default=Path("musan/musan"),
                     help="dir containing noise/free-sound, noise/sound-bible (relative to data/)")
    ap.add_argument("--snr-db-range", type=float, nargs=2, default=[5.0, 20.0],
                     help="uniform random SNR range in dB (lower = noisier)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap on clips augmented (random sample); omit for all clips in --real-dir")
    args = ap.parse_args()

    noise_files: list[Path] = []
    for sub in MUSAN_NOISE_SUBDIRS:
        d = args.musan_root / "noise" / sub
        if d.exists():
            noise_files.extend(sorted(d.glob("*.wav")))
    if not noise_files:
        print(f"No MUSAN noise files found under {args.musan_root}/noise/{{free-sound,sound-bible}} — "
              "run runs/download_openslr_musan.ps1 first.", file=sys.stderr)
        raise SystemExit(1)

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    speech_files = sorted(args.real_dir.glob("*.wav"))
    if not speech_files:
        print(f"No WAVs found in {args.real_dir}", file=sys.stderr)
        raise SystemExit(1)
    if args.limit is not None and len(speech_files) > args.limit:
        speech_files = rng.sample(speech_files, args.limit)

    n_done = 0
    for speech_path in speech_files:
        speech = _load_mono(speech_path, args.sample_rate)
        noise_path = rng.choice(noise_files)
        noise = _load_mono(noise_path, args.sample_rate)
        snr_db = rng.uniform(*args.snr_db_range)
        mixed = _mix_at_snr(speech, noise, snr_db)

        out_path = args.out_dir / f"noisy_{speech_path.stem}.wav"
        sf.write(out_path, mixed, args.sample_rate)
        n_done += 1
        if n_done % 200 == 0:
            print(f"{n_done}/{len(speech_files)} augmented", file=sys.stderr)

    print(f"Done. {n_done} noise-augmented clips written to {args.out_dir}")


if __name__ == "__main__":
    main()
