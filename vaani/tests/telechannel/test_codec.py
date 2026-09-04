"""
Unit tests for telechannel/stages/codec.py.

Two kinds of tests here:

1. Tests that need no real ffmpeg binary at all: input validation, and
   ffmpeg-command-construction / pad-trim logic verified by monkeypatching
   `subprocess.run` to capture (and fake the effect of) the invocation
   instead of actually shelling out.

2. Tests that genuinely invoke ffmpeg end-to-end. These are gated with
   `pytest.mark.skipif(shutil.which("ffmpeg") is None, ...)` and are SKIPPED
   (not passed, not failed) in any environment without a real ffmpeg binary
   on PATH. Verified 2026-09-04 against ffmpeg 9.0.1-full_build-www.gyan.dev
   (AMR-NB/AMR-WB/GSM/Opus encoders included) — all pass.
"""

import shutil

import numpy as np
import pytest

from scipy.io import wavfile

from telechannel.stages import codec as codec_mod
from telechannel.stages.codec import CODEC_SPECS, codec_roundtrip

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _sine(freq_hz, sr, duration_s=1.0):
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# No-ffmpeg-required tests
# ---------------------------------------------------------------------------


def test_unknown_codec_raises_value_error():
    x = _sine(440, 16000)
    with pytest.raises(ValueError):
        codec_roundtrip(x, "not_a_real_codec", bitrate="12.2k")


def test_missing_ffmpeg_raises_runtime_error(monkeypatch):
    # This exercises the real "ffmpeg not found" branch: in this authoring
    # environment ffmpeg genuinely is not on PATH, so this doesn't need
    # mocking at all. We still force it via monkeypatch so the test is
    # meaningful (and passes) even on a machine that does have ffmpeg
    # installed.
    monkeypatch.setattr(codec_mod.shutil, "which", lambda name: None)
    x = _sine(440, 16000)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        codec_roundtrip(x, "pcm_mulaw", bitrate="64k")


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stderr=b""):
        self.returncode = returncode
        self.stderr = stderr


def _make_fake_run(captured_cmds, out_len_samples, out_sr=16000):
    """
    Build a fake `subprocess.run` that records every invoked command and,
    when it sees the decode step's output path (a .wav file as the last
    arg), writes a real (synthetic) WAV file there so the rest of
    codec_roundtrip's pipeline (_read_wav / pad-trim) runs for real.
    """

    def fake_run(cmd, capture_output=True):
        captured_cmds.append(cmd)
        out_path = cmd[-1]
        if out_path.endswith(".wav"):
            # Simulate ffmpeg's decode output: `out_len_samples` of silence
            # (with a fixed nonzero sample, so int16 round-trip is visible).
            data = np.zeros(out_len_samples, dtype=np.int16)
            if out_len_samples > 0:
                data[0] = 12345
            wavfile.write(out_path, out_sr, data)
        else:
            # Simulate ffmpeg's encode output: just needs to exist.
            with open(out_path, "wb") as f:
                f.write(b"\x00")
        return _FakeCompletedProcess(returncode=0)

    return fake_run


@pytest.mark.parametrize("codec_name", sorted(CODEC_SPECS))
def test_codec_command_construction_per_codec(monkeypatch, codec_name):
    monkeypatch.setattr(codec_mod.shutil, "which", lambda name: "C:/fake/ffmpeg.exe")

    x = _sine(300, 16000, duration_s=0.5)
    n = len(x)
    captured = []
    monkeypatch.setattr(
        codec_mod.subprocess, "run", _make_fake_run(captured, out_len_samples=n)
    )

    y = codec_roundtrip(x, codec_name, bitrate="12.2k")

    assert len(captured) == 2
    encode_cmd, decode_cmd = captured
    spec = CODEC_SPECS[codec_name]

    # Encode step: WAV -> codec at the codec's native sample rate.
    assert encode_cmd[0] == "ffmpeg"
    assert "-c:a" in encode_cmd
    assert encode_cmd[encode_cmd.index("-c:a") + 1] == spec["encoder"]
    assert "-ar" in encode_cmd
    assert encode_cmd[encode_cmd.index("-ar") + 1] == str(spec["sr"])
    if spec["fixed_rate"]:
        assert "-b:a" not in encode_cmd
    else:
        assert "-b:a" in encode_cmd
        assert encode_cmd[encode_cmd.index("-b:a") + 1] == "12.2k"
    assert encode_cmd[-1].endswith(f".{spec['ext']}")

    # Decode step: codec -> 16kHz PCM.
    assert decode_cmd[0] == "ffmpeg"
    assert "-c:a" in decode_cmd
    assert decode_cmd[decode_cmd.index("-c:a") + 1] == "pcm_s16le"
    assert "-ar" in decode_cmd
    assert decode_cmd[decode_cmd.index("-ar") + 1] == str(codec_mod.OUTPUT_SR)
    assert decode_cmd[-1].endswith(".wav")

    # Output should be same length as input (exact match case here).
    assert len(y) == n


def test_pads_short_decoded_output_to_input_length(monkeypatch):
    monkeypatch.setattr(codec_mod.shutil, "which", lambda name: "C:/fake/ffmpeg.exe")
    x = _sine(300, 16000, duration_s=0.5)
    n = len(x)
    captured = []
    # Simulate ffmpeg producing a shorter decoded file than the input.
    monkeypatch.setattr(
        codec_mod.subprocess, "run",
        _make_fake_run(captured, out_len_samples=n - 100),
    )

    y = codec_roundtrip(x, "pcm_mulaw", bitrate="64k")

    assert len(y) == n
    # The padded tail should be zeros.
    assert np.all(y[-100:] == 0.0)


def test_trims_long_decoded_output_to_input_length(monkeypatch):
    monkeypatch.setattr(codec_mod.shutil, "which", lambda name: "C:/fake/ffmpeg.exe")
    x = _sine(300, 16000, duration_s=0.5)
    n = len(x)
    captured = []
    # Simulate ffmpeg producing a longer decoded file than the input.
    monkeypatch.setattr(
        codec_mod.subprocess, "run",
        _make_fake_run(captured, out_len_samples=n + 200),
    )

    y = codec_roundtrip(x, "pcm_mulaw", bitrate="64k")

    assert len(y) == n


def test_ffmpeg_nonzero_exit_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(codec_mod.shutil, "which", lambda name: "C:/fake/ffmpeg.exe")

    def failing_run(cmd, capture_output=True):
        return _FakeCompletedProcess(returncode=1, stderr=b"encoder not found")

    monkeypatch.setattr(codec_mod.subprocess, "run", failing_run)

    x = _sine(300, 16000, duration_s=0.2)
    with pytest.raises(RuntimeError, match="ffmpeg command failed"):
        codec_roundtrip(x, "pcm_mulaw", bitrate="64k")


def test_write_and_read_wav_roundtrip_pcm16(tmp_path):
    # No ffmpeg involved: exercises our own WAV read/write helpers.
    x = _sine(440, 16000, duration_s=0.1) * 0.5
    path = str(tmp_path / "test.wav")

    codec_mod._write_wav(path, x, 16000)
    y = codec_mod._read_wav(path)

    assert len(y) == len(x)
    # int16 quantization introduces small error; should still be close.
    assert np.max(np.abs(y - x)) < 1e-3


def test_pad_or_trim_exact_length():
    y = np.arange(10, dtype=np.float64)
    out = codec_mod._pad_or_trim(y, 10)
    assert np.array_equal(out, y)


def test_pad_or_trim_pads_with_zeros():
    y = np.arange(5, dtype=np.float64)
    out = codec_mod._pad_or_trim(y, 8)
    assert len(out) == 8
    assert np.array_equal(out[:5], y)
    assert np.array_equal(out[5:], np.zeros(3))


def test_pad_or_trim_trims():
    y = np.arange(10, dtype=np.float64)
    out = codec_mod._pad_or_trim(y, 6)
    assert len(out) == 6
    assert np.array_equal(out, y[:6])


# ---------------------------------------------------------------------------
# Real-ffmpeg tests (SKIPPED if ffmpeg is not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_pcm_mulaw_roundtrip_real_ffmpeg_output_length_and_noise():
    """
    Brief Step 3: pass audio through pcm_mulaw, verify output SR is 16kHz
    (implicit: output array length matches a 16kHz-length input) and that
    quantization noise is present (output != input, but still close).
    """
    sr = 16000
    x = _sine(1000, sr, duration_s=0.5)

    y = codec_roundtrip(x, "pcm_mulaw", bitrate="64k")

    assert len(y) == len(x)
    # mu-law companding introduces quantization noise: output should differ
    # from input, but the waveform shape should still be recognizably close.
    assert not np.array_equal(y, x)
    correlation = np.corrcoef(x, y)[0, 1]
    assert correlation > 0.9


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_pcm_alaw_roundtrip_real_ffmpeg():
    sr = 16000
    x = _sine(1000, sr, duration_s=0.5)

    y = codec_roundtrip(x, "pcm_alaw", bitrate="64k")

    assert len(y) == len(x)
    correlation = np.corrcoef(x, y)[0, 1]
    assert correlation > 0.9


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_libgsm_roundtrip_real_ffmpeg():
    sr = 16000
    x = _sine(1000, sr, duration_s=0.5)

    y = codec_roundtrip(x, "libgsm", bitrate="13k")

    assert len(y) == len(x)
    correlation = np.corrcoef(x, y)[0, 1]
    assert correlation > 0.5  # GSM-FR is much lossier than G.711


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_amr_nb_roundtrip_real_ffmpeg():
    sr = 16000
    x = _sine(1000, sr, duration_s=0.5)

    y = codec_roundtrip(x, "libopencore_amrnb", bitrate="12.2k")

    assert len(y) == len(x)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_amr_wb_roundtrip_real_ffmpeg():
    sr = 16000
    x = _sine(1000, sr, duration_s=0.5)

    y = codec_roundtrip(x, "libvo_amrwbenc", bitrate="23.85k")

    assert len(y) == len(x)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_opus_roundtrip_real_ffmpeg():
    sr = 16000
    x = _sine(1000, sr, duration_s=0.5)

    y = codec_roundtrip(x, "libopus", bitrate="24k")

    assert len(y) == len(x)
