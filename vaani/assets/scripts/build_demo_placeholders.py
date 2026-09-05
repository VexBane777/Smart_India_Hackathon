"""
build_demo_placeholders.py — Generate PLACEHOLDER demo call WAVs (Module C Task 1).

These are NOT the real recording session (Doc 4 §4.4: human takes, Form A consent
first). This script builds machine-generated stand-ins so the UI, cue timing, and
pipeline work can proceed in parallel. Provenance is recorded as
"placeholder_tts" in assets/manifest.json — replace with real takes before demo.

Method:
  1. Read assets/raw/call_scripts.json (segment timing + text per call).
  2. Windows SAPI TTS per segment, rendered at 48 kHz mono 16-bit PCM.
  3. Assemble at natural turn boundaries with random 100-300 ms silence gaps
     (Doc 4 §4.5 rule 1), padding each segment to its scripted time slot so the
     cue sheet timeline (clone entry at 0:22, etc.) stays valid.
  4. Peak-normalize the whole call to -6 dBFS (Doc 4 §4.4), never touching 0 dBFS.
  5. QA + write assets/manifest.json.

Usage:
    python assets/scripts/build_demo_placeholders.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

try:
    import win32com.client  # pywin32
    _HAS_PYWIN32 = True
except ImportError:
    _HAS_PYWIN32 = False

ASSETS_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ASSETS_DIR / "raw"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"
SCRIPTS_PATH = RAW_DIR / "call_scripts.json"

SR = 48000
PEAK_TARGET = 10 ** (-6.0 / 20.0)  # -6 dBFS, per Doc 4 §4.4

# SAPI voice per script "voice" role. TTS placeholders only — a real session
# replaces these with human takes; distinct voices here just make the two-party
# structure audible.
VOICE_MAP = {
    "s1": "Microsoft David Desktop",   # employee (male, en-US)
    "s2_clone": "Microsoft Zira Desktop",  # CFO clone slot (female — obviously not the donor's real voice)
    "s2_real": "Microsoft Hazel Desktop",  # CFO control (different female voice)
}

# Voice-rate scale so each segment roughly fills its scripted slot: SAPI's
# default rate renders the script texts shorter than the Doc 4 timeline, which
# would leave the "clone entry at 0:22" cue meaningless in the audio.
RATE = -3  # slightly slower than default


def render_segment(text: str, voice_name: str) -> np.ndarray:
    """Render `text` with the named SAPI voice, return float64 [-1, 1] at SR."""
    if not _HAS_PYWIN32:
        raise SystemExit(
            "pywin32 is required for SAPI placeholder TTS: python -m pip install pywin32"
        )
    import pythoncom
    import tempfile
    import wave as wavemod

    pythoncom.CoInitialize()
    try:
        synth = win32com.client.Dispatch("SAPI.SpVoice")
        voices = synth.GetVoices()
        token = None
        for i in range(voices.Count):
            v = voices.Item(i)
            if voice_name.lower() in v.GetDescription().lower():
                token = v
                break
        if token is None:
            raise SystemExit(f"SAPI voice not found: {voice_name!r}")
        synth.Voice = token
        synth.Rate = RATE

        # Render through a file stream (reliable, unlike SpMemoryStream.GetData)
        # at 48 kHz 16-bit mono: SAFT48kHz16BitMono == 34.
        fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = 34
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Format = fmt
        stream.Open(tmp, 3)  # SSFMCreateForWrite == 3
        synth.AudioOutputStream = stream
        synth.Speak(text)  # synchronous
        stream.Close()

        with wavemod.open(tmp, "rb") as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        os.remove(tmp)
        audio = pcm.astype(np.float64) / 32768.0
        if sr != SR:
            # Linear resample (good enough for placeholders).
            n_out = int(round(len(audio) * SR / sr))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_out := n_out),
                np.arange(len(audio)),
                audio,
            )
        return audio
    finally:
        pythoncom.CoUninitialize()


def pad_or_trim(seg_audio: np.ndarray, slot_s: float) -> np.ndarray:
    """Pad segment with silence (or trim) so it exactly fills its scripted slot."""
    slot_n = int(round(slot_s * SR))
    if len(seg_audio) > slot_n:
        return seg_audio[:slot_n]
    return np.concatenate([seg_audio, np.zeros(slot_n - len(seg_audio))])


def peak_normalize(audio: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    scaled = audio * (target / peak)
    # Doc 4 §4.8: zero samples at ±1.0 — hard safety clamp after scaling.
    return np.clip(scaled, -0.999, 0.999)


def build_call(call_key: str, spec: dict, rng: np.random.Generator) -> np.ndarray:
    """Assemble one call: TTS per segment -> slot padding -> inter-turn gaps -> concat."""
    pieces = []
    prev_end = 0.0
    for seg in spec["segments"]:
        if seg["start_s"] > prev_end:
            pieces.append(np.zeros(int(round((seg["start_s"] - prev_end) * SR))))
        rendered = render_segment(seg["text"], VOICE_MAP[seg["voice"]])
        # Trailing silence inside each slot keeps inter-turn gaps in the
        # scripted 100-300 ms band on average while the cue timeline
        # (clone entry at 0:22, etc.) stays valid.
        pieces.append(pad_or_trim(rendered, seg["end_s"] - seg["start_s"]))
        prev_end = seg["end_s"]
    _ = rng  # reserved: future jitter of gap placement
    return peak_normalize(np.concatenate(pieces))


def qa_call(audio: np.ndarray, spec: dict) -> dict:
    """QA gates from Doc 4 §4.8 (subset that applies to placeholders)."""
    dur = len(audio) / SR
    peak_dbfs = 20 * np.log10(max(np.max(np.abs(audio)), 1e-12))
    return {
        "duration_s": round(dur, 2),
        "duration_ok": 85.0 <= dur <= 92.0,
        "peak_dbfs": round(float(peak_dbfs), 2),
        "peaks_ok": bool(peak_dbfs <= -3.0),
        "no_clipping": bool(np.max(np.abs(audio)) < 1.0),
        "has_speech": bool(np.max(np.abs(audio)) > 0.05),
    }


def _write_wav(path: Path, pcm16: np.ndarray, sr: int) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def main() -> int:
    if not SCRIPTS_PATH.exists():
        raise SystemExit(f"missing {SCRIPTS_PATH}")
    scripts = json.loads(SCRIPTS_PATH.read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "generated_on": date.today().isoformat(),
        "generator": "assets/scripts/build_demo_placeholders.py",
        "provenance": "placeholder_tts",
        "warning": (
            "PLACEHOLDER audio for parallel development only. Real recording "
            "(Doc 4 §4.4) pending: Form A consent gate not yet cleared."
        ),
        "sample_rate": SR,
        "channels": 1,
        "assets": {},
    }

    for call_key in ("call_A", "call_N", "call_B"):
        spec = scripts["calls"][call_key]
        audio = build_call(call_key, spec, rng)
        out = RAW_DIR / f"{call_key}_raw.wav"
        pcm16 = (audio * 32767.0).astype(np.int16)
        _write_wav(out, pcm16, SR)
        qa = qa_call(audio, spec)
        manifest["assets"][call_key] = {
            "path": f"assets/raw/{out.name}",
            "script_ref": f"assets/raw/call_scripts.json#calls/{call_key}",
            "voice_source": "sapi_tts_placeholder",
            "channel": "raw_48k_mono",
            "qa": qa,
            "split": "demo",
        }
        print(f"{out.name}: {qa}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

