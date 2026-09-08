"""One-off: split ASVspoof2021 LA eval FLACs (channel-degraded-by-construction,
real telephony codecs) and the bundled ASVspoof2019 LA dev FLACs into
real2021/ fake2021/ WAV dirs, using the released key files for labels.

2021 LA eval protocol line (LA-keys-full/keys/LA/CM/trial_metadata.txt):
    speaker_id utt_id codec source attack_id key trim_type subset
e.g. LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
`key` is "bonafide" or "spoof" — column index 5 (0-based).

2019 LA dev protocol line (ASVspoof2019.LA.cm.dev.trl.txt):
    speaker_id utt_id - attack_id key
e.g. LA_0009 LA_D_1234567 - A01 spoof — key is column index 4.
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "asvspoof2021_la"

REAL_DIR = HERE / "real2021"
FAKE_DIR = HERE / "fake2021"
REAL_DIR.mkdir(exist_ok=True)
FAKE_DIR.mkdir(exist_ok=True)


def _convert_one(task: tuple[Path, Path, str]) -> tuple[str, bool]:
    flac_path, out_path, utt_id = task
    # This corpus's re-encoded FLACs choke libsndfile (soundfile) on ~44% of
    # files ("unknown error in flac decoder" / fseek failures) despite being
    # valid — ffmpeg's decoder handles them fine, so shell out instead of sf.read.
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(flac_path), str(out_path)],
        capture_output=True,
    )
    if result.returncode != 0 or not out_path.exists():
        print(f"skip {utt_id}: {result.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
        return utt_id, False
    return utt_id, True


def split(flac_dir: Path, protocol: Path, utt_col: int, key_col: int, limit: int | None = None,
          workers: int = 32):
    with open(protocol) as f:
        lines = f.readlines()
    if limit:
        lines = lines[:limit]

    tasks: list[tuple[Path, Path, str]] = []
    keys: dict[str, str] = {}
    n_missing = 0
    for line in lines:
        parts = line.split()
        utt_id, key = parts[utt_col], parts[key_col]
        flac_path = flac_dir / f"{utt_id}.flac"
        if not flac_path.exists():
            n_missing += 1
            continue
        out_dir = REAL_DIR if key == "bonafide" else FAKE_DIR
        tasks.append((flac_path, out_dir / f"{utt_id}.wav", utt_id))
        keys[utt_id] = key

    n_real = n_fake = 0
    n_skip = n_missing
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for utt_id, ok in pool.map(_convert_one, tasks):
            done += 1
            if ok:
                if keys[utt_id] == "bonafide":
                    n_real += 1
                else:
                    n_fake += 1
            else:
                n_skip += 1
            if done % 5000 == 0:
                print(f"{flac_dir.name}: {done}/{len(tasks)}", file=sys.stderr)
    return n_real, n_fake, n_skip


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-2021-eval", type=int, default=None,
                     help="cap on 2021 eval utterances processed (it's 181k files; "
                          "omit for the full set, pass e.g. 40000 to bound runtime)")
    args = ap.parse_args()

    r1, f1, s1 = split(
        ROOT / "ASVspoof2021_LA_eval" / "ASVspoof2021_LA_eval" / "flac",
        ROOT / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt",
        utt_col=1,
        key_col=5,
        limit=args.limit_2021_eval,
    )
    print(f"2021 eval: real={r1} fake={f1} skipped={s1}")

    r2, f2, s2 = split(
        ROOT / "LA" / "LA" / "ASVspoof2019_LA_dev" / "flac",
        ROOT / "LA" / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
        utt_col=1,
        key_col=4,
    )
    print(f"2019 dev: real={r2} fake={f2} skipped={s2}")
    print(f"TOTAL real2021={r1 + r2} fake2021={f1 + f2}")
