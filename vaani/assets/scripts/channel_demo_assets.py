"""
channel_demo_assets.py — Module C Task 2: run the raw demo calls through
TeleChannel's `whatsapp` recipe and update assets/manifest.json.

Consumes:  assets/raw/call_{A,N}_raw.wav (48 kHz mono; Task 1 placeholders)
Produces:  assets/demo/call_{A,N}_whatsapp.wav (16 kHz mono; whatsapp recipe:
           Opus 16k -> white noise 20dB SNR -> near-room RIR -> packet loss)

The plan's "Interfaces" line says FLAC; this script writes WAV to match the
plan's own "Files" section (`call_A_whatsapp.wav`) and because
`telechannel.pipeline.process_clip` returns raw float PCM, not compressed
frames — WAV is the zero-dependency choice.

Usage:
    python assets/scripts/channel_demo_assets.py
"""

import json
import sys
import wave
from datetime import date
from pathlib import Path

import numpy as np

VAANI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(VAANI_ROOT))

from telechannel.pipeline import process_clip  # noqa: E402

ASSETS_DIR = VAANI_ROOT / "assets"
RAW_DIR = ASSETS_DIR / "raw"
DEMO_DIR = ASSETS_DIR / "demo"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"

RECIPE = "whatsapp"
SR = 16000  # telechannel's native rate
CALL_KEYS = ("call_A", "call_N")  # the paired experiment (Doc 4 §4.6); call_B is the accent variant, not needed for the A/N alert demo


def _read_wav_mono_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float64) / 32768.0, sr


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    n_out = int(round(len(audio) * dst_sr / src_sr))
    return np.interp(
        np.linspace(0.0, len(audio) - 1.0, n_out),
        np.arange(len(audio), dtype=np.float64),
        audio,
    )


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"missing {MANIFEST_PATH} — run build_demo_placeholders.py first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    rng = np.random.default_rng(0)  # deterministic channeling across runs

    for call_key in CALL_KEYS:
        raw_path = RAW_DIR / f"{call_key}_raw.wav"
        if not raw_path.exists():
            raise SystemExit(f"missing {raw_path} — run build_demo_placeholders.py first")
        audio, src_sr = _read_wav_mono_float(raw_path)
        audio = _resample_linear(audio, src_sr, SR)
        channeled = process_clip(audio, RECIPE, sr=SR, rng=rng)
        out_path = DEMO_DIR / f"{call_key}_whatsapp.wav"
        _write_wav(out_path, channeled, SR)

        manifest["assets"].setdefault(call_key, {})
        manifest["assets"][call_key]["demo"] = {
            "path": f"assets/demo/{out_path.name}",
            "recipe": RECIPE,
            "sample_rate": SR,
            "source": f"assets/raw/{raw_path.name}",
        }
        print(f"{out_path.name}: peak={np.max(np.abs(channeled)):.3f}")

    manifest["demo_generated_on"] = date.today().isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
