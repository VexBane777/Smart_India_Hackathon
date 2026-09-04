"""
codec.py — Codec round-trip stage.

Simulates the lossy compression a real phone call goes through by shelling
out to FFmpeg: encode the input PCM signal into a lossy telephony codec at
a target bitrate, then decode it back to 16kHz PCM. This captures the
quantization/compression artifacts (G.711 companding noise, GSM-FR/AMR
spectral distortion, Opus's lower-rate coding noise, etc.) that a real
transmission path introduces and that cannot be modeled by DSP alone.

Requires a real `ffmpeg` binary on PATH, built with the relevant encoders
(`pcm_mulaw`/`pcm_alaw` are built into every ffmpeg; `libgsm`,
`libopencore_amrnb`, `libvo_amrwbenc`, and `libopus` are optional encoders
that many public/Windows ffmpeg builds omit due to licensing — check with
`ffmpeg -encoders | grep -Ei "gsm|amr|opus|alaw|mulaw"` before relying on
this module for those codecs).

Verified end-to-end (2026-09-04) against ffmpeg 9.0.1-full_build-www.gyan.dev
(installed per-user via `winget install Gyan.FFmpeg`, no admin rights
needed), which includes the AMR-NB/AMR-WB/GSM/Opus encoders. All real-ffmpeg
tests in tests/telechannel/test_codec.py pass. Note: ffmpeg's AMR muxer only
recognizes the ".amr" extension (it auto-detects NB vs. WB from the encoded
stream) — see CODEC_SPECS below.

See DOC1_TELECHANNEL_SPEC.md for the TeleChannel stage pipeline this module
plugs into (a later task chains these stages together; this module is
independent and does not do that chaining).
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile

INPUT_SR = 16000
OUTPUT_SR = 16000

# codec name -> ffmpeg encoder codec name, container extension for the
# intermediate encoded file, and the sample rate to encode at (the classic
# rate each codec is defined/used at over real telephony networks).
CODEC_SPECS = {
    "pcm_mulaw": {"encoder": "pcm_mulaw", "ext": "wav", "sr": 8000, "fixed_rate": True},
    "pcm_alaw": {"encoder": "pcm_alaw", "ext": "wav", "sr": 8000, "fixed_rate": True},
    "libgsm": {"encoder": "libgsm", "ext": "gsm", "sr": 8000, "fixed_rate": True},
    "libopencore_amrnb": {"encoder": "libopencore_amrnb", "ext": "amr", "sr": 8000, "fixed_rate": False},
    # ffmpeg's AMR muxer only recognizes the ".amr" extension (it auto-detects
    # NB vs. WB from the encoded stream); ".awb" is not a registered muxer
    # extension and fails with "Unable to choose an output format".
    "libvo_amrwbenc": {"encoder": "libvo_amrwbenc", "ext": "amr", "sr": 16000, "fixed_rate": False},
    "libopus": {"encoder": "libopus", "ext": "opus", "sr": 16000, "fixed_rate": False},
}


def codec_roundtrip(x, codec, bitrate):
    """
    Round-trip `x` through a lossy telephony codec via FFmpeg: encode
    WAV -> codec at `bitrate` (and the codec's native sample rate), decode
    codec -> 16kHz PCM, then pad/trim the result back to the original
    input length.

    Args:
        x: 1-D input signal (numpy array), float64 in [-1, 1], assumed
            16kHz PCM.
        codec: One of "pcm_mulaw" (G.711 mu-law), "pcm_alaw" (G.711
            A-law), "libgsm" (GSM-FR), "libopencore_amrnb" (AMR-NB),
            "libvo_amrwbenc" (AMR-WB), "libopus" (Opus).
        bitrate: Target bitrate passed to ffmpeg's `-b:a` (e.g. "12.2k").
            Ignored for the fixed-rate G.711 codecs (pcm_mulaw/pcm_alaw).

    Returns:
        numpy array the same length as `x`, float64, decoded at 16kHz.

    Raises:
        ValueError: if `codec` is not a recognized codec name.
        RuntimeError: if ffmpeg is not found on PATH, or either ffmpeg
            invocation exits non-zero.
    """
    if codec not in CODEC_SPECS:
        raise ValueError(
            f"Unknown codec: {codec!r}. Supported: {sorted(CODEC_SPECS)}"
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; codec_roundtrip requires a real "
            "ffmpeg binary (with the AMR/GSM/Opus encoders enabled, "
            "depending on the codec requested)."
        )

    spec = CODEC_SPECS[codec]
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    tmpdir = tempfile.mkdtemp(prefix="codec_roundtrip_")
    try:
        in_wav = os.path.join(tmpdir, "in.wav")
        encoded = os.path.join(tmpdir, f"encoded.{spec['ext']}")
        out_wav = os.path.join(tmpdir, "out.wav")

        _write_wav(in_wav, x, INPUT_SR)

        encode_cmd = [
            "ffmpeg", "-y", "-i", in_wav,
            "-ar", str(spec["sr"]),
            "-ac", "1",
            "-c:a", spec["encoder"],
        ]
        if not spec["fixed_rate"]:
            encode_cmd += ["-b:a", str(bitrate)]
        encode_cmd.append(encoded)
        _run_ffmpeg(encode_cmd)

        decode_cmd = [
            "ffmpeg", "-y", "-i", encoded,
            "-ar", str(OUTPUT_SR),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            out_wav,
        ]
        _run_ffmpeg(decode_cmd)

        y = _read_wav(out_wav)
        return _pad_or_trim(y, n)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        raise RuntimeError(f"ffmpeg command failed ({' '.join(cmd)}): {stderr}")


def _write_wav(path, x, sr):
    x_clipped = np.clip(x, -1.0, 1.0)
    pcm16 = (x_clipped * 32767.0).astype(np.int16)
    wavfile.write(path, sr, pcm16)


def _read_wav(path):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]

    if data.dtype == np.int16:
        y = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float64) - 128) / 128.0
    else:
        y = data.astype(np.float64)
    return y


def _pad_or_trim(y, n):
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - len(y))])
