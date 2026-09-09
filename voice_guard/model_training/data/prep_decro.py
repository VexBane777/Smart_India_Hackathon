"""Ingest DECRO's English spoof subset into data/accents/fake/en_native —
yet another distinct generator family (HiFiGAN, Multiband-MelGAN, PWG,
Tacotron, FastSpeech2, VITS, StarGANv2-VC, plus commercial Baidu/Xunfei
TTS) on top of ASVspoof/XTTS/CodecFake/MLAAD/FoR already in this pipeline.

Source: Zenodo record 7603208 (DECRO-dataset-v1.2.zip, 15.5GB — actually a
GitHub-repo-shaped archive, not a flat audio dump; top-level folder is
`petrichorwq-DECRO-dataset-<hash>/`, containing `en_train/`, `en_dev/`,
`en_eval/`, `ch_train/`, `ch_dev/`, `ch_eval/` audio folders plus matching
`.txt` protocol files). Only the **English spoof** rows are pulled:
- Chinese is out of this project's en/hi scope entirely.
- English **bona-fide** rows are explicitly documented (DECRO's own README)
  as literally re-partitioned ASVspoof2019 LA utterances — already in this
  pipeline's `data/real` from `split_flac_to_wav.py`, so pulling them again
  here would just duplicate existing real data under a new filename.

Protocol line format (`en_{train,dev,eval}.txt`, space-separated):
    <speaker_id> <utt_id> - <generator> <bonafide|spoof>
e.g. `4 4-727-124443-0055 - baidu spoof`. Audio filename in the matching
folder is `<utt_id>[_<generator-ish-suffix>].wav` — not a fixed pattern
across rows (observed both bare `<utt_id>.wav` and `<utt_id>_<tag>.wav` in
the same folder), so this script glob-matches `<utt_id>*.wav` rather than
reconstructing the exact filename.

Usage (after extracting DECRO-dataset-v1.2.zip into data/decro/):
    python prep_decro.py --split train --limit 3000
    python prep_decro.py --split dev --limit 1000
    python prep_decro.py --split eval --limit 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from accent_common import ACCENTS_ROOT, ensure_dirs, cell_dir, manifest_writer_append

HERE = Path(__file__).resolve().parent
SOURCE_SAMPLE_RATE = 16000
CELL = "en_native"


def _find_repo_root(decro_root: Path) -> Path:
    candidates = list(decro_root.glob("petrichorwq-DECRO-dataset-*"))
    if candidates:
        return candidates[0]
    return decro_root  # already-flattened layout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decro-root", type=Path, default=HERE / "decro")
    ap.add_argument("--split", choices=["train", "dev", "eval"], default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    repo_root = _find_repo_root(args.decro_root)
    protocol_path = repo_root / f"en_{args.split}.txt"
    audio_dir = repo_root / f"en_{args.split}"
    if not protocol_path.exists() or not audio_dir.exists():
        print(f"Missing {protocol_path} or {audio_dir} — check DECRO extraction "
              "(see module docstring for the expected layout).", file=sys.stderr)
        raise SystemExit(1)

    ensure_dirs()
    rows_to_pull = []
    with open(protocol_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            speaker_id, utt_id, label = parts[0], parts[1], parts[-1]
            if label != "spoof":
                continue  # bona-fide is duplicate ASVspoof2019 data — see docstring
            rows_to_pull.append((speaker_id, utt_id))
    if args.limit is not None:
        rows_to_pull = rows_to_pull[: args.limit]

    rows = []
    n_missing = 0
    for speaker_id, utt_id in rows_to_pull:
        matches = list(audio_dir.glob(f"{utt_id}*.wav"))
        if not matches:
            n_missing += 1
            continue
        src = matches[0]
        try:
            array, sr = sf.read(str(src), dtype="float32", always_2d=False)
        except Exception as e:
            print(f"skip {src.name}: {e}", file=sys.stderr)
            continue
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != SOURCE_SAMPLE_RATE:
            import librosa  # local import: heavy dep, only needed for resampling

            array = librosa.resample(array, orig_sr=sr, target_sr=SOURCE_SAMPLE_RATE)
            sr = SOURCE_SAMPLE_RATE

        out_name = f"decro_{args.split}_{utt_id}.wav"
        out_path = cell_dir("fake", CELL) / out_name
        sf.write(out_path, array, sr)
        duration_s = len(array) / sr
        rows.append({
            "file": str(out_path.relative_to(ACCENTS_ROOT.parent)),
            "label": "fake",
            "cell": CELL,
            "source": "decro",
            "speaker_id": speaker_id,
            "duration_s": f"{duration_s:.3f}",
        })
        if len(rows) % 500 == 0:
            print(f"{len(rows)}/{len(rows_to_pull)} converted", file=sys.stderr)

    manifest_writer_append(rows)
    print(f"Done. {len(rows)} manifest rows appended to fake/{CELL} "
          f"(split={args.split}, {n_missing} protocol rows had no matching audio file)")


if __name__ == "__main__":
    main()
