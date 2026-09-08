"""One-off: split ASVspoof2019 LA train FLACs into real/ fake/ WAV dirs per protocol."""
import sys
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
FLAC_DIR = HERE / "extracted" / "LA" / "ASVspoof2019_LA_train" / "flac"
PROTOCOL = HERE / "extracted" / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
REAL_DIR = HERE / "real"
FAKE_DIR = HERE / "fake"
REAL_DIR.mkdir(exist_ok=True)
FAKE_DIR.mkdir(exist_ok=True)

n_real = n_fake = 0
with open(PROTOCOL) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    parts = line.split()
    utt_id, label = parts[1], parts[4]
    flac_path = FLAC_DIR / f"{utt_id}.flac"
    if not flac_path.exists():
        continue
    data, sr = sf.read(str(flac_path))
    out_dir = REAL_DIR if label == "bonafide" else FAKE_DIR
    sf.write(str(out_dir / f"{utt_id}.wav"), data, sr)
    if label == "bonafide":
        n_real += 1
    else:
        n_fake += 1
    if i % 2000 == 0:
        print(f"{i}/{len(lines)}", file=sys.stderr)

print(f"real={n_real} fake={n_fake}")
