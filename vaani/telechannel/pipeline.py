"""
pipeline.py — TeleChannel pipeline orchestrator.

Chains the six independent stage functions in `telechannel/stages/` into
named "recipes" loaded from `telechannel/configs/channels.yaml`, in the
fixed physical order a real phone call is degraded in:

    RIR (room) -> Noise -> Mic -> Codec (0..N round-trips) -> Loss -> Bandlimit

Each stage is applied only if the recipe supplies a (non-null) config for
it, so a recipe can skip stages it doesn't need (e.g. `clean` skips codec
entirely; `volte`/`whatsapp` both skip bandlimit, since Doc 1's
narrowband-only band-limiting rule only applies to the narrowband recipes
(pstn/gsm_2g/cellular_3g/tandem_xnet) — see channels.yaml's comment).

`codec` is always a list of `{codec, bitrate}` entries applied in sequence:
zero entries skips the codec stage, one entry is a normal single round-trip,
and more than one is tandem transcoding (compress/decompress more than
once across different codecs, as a real cross-network call does).

See DOC1_TELECHANNEL_SPEC.md for the full pipeline spec this implements.
"""

from pathlib import Path

import numpy as np
import yaml

from telechannel.stages.rir import apply_rir
from telechannel.stages.noise import apply_noise
from telechannel.stages.mic import apply_mic
from telechannel.stages.codec import codec_roundtrip
from telechannel.stages.packetloss import apply_packet_loss
from telechannel.stages.bandlimit import apply_bandlimit

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "channels.yaml"

# RIR direct-path impulse amplitude (see _synth_rir): forced to exceed the
# decaying-noise tail so `apply_rir`'s peak-alignment (np.argmax(abs(rir)))
# reliably lands on sample 0, matching a real RIR's direct-path-first shape.
_DIRECT_PATH_AMPLITUDE = 1.0


def load_config(config_path=None):
    """
    Load and return the recipes config dict from `config_path` (defaults to
    `telechannel/configs/channels.yaml`).
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _synth_rir(room_cfg, sr, rng):
    """
    Synthesize a stand-in room impulse response: an exponentially decaying
    noise tail (shaped to roughly hit -60dB by `rt60_ms`) with a unit
    impulse forced at sample 0 as the direct path.

    There is no real RIR/SLR28 library wired up in this codebase yet (see
    rir.py's docstring) — this is a placeholder fingerprint generator, not
    a measured impulse response.
    """
    length = int(room_cfg["length_samples"])
    rt60_s = float(room_cfg["rt60_ms"]) / 1000.0
    t = np.arange(length) / sr
    # exp(-6.91 * t / rt60) reaches ~-60dB (10**(-60/20)) at t == rt60.
    decay = np.exp(-6.91 * t / rt60_s)
    rir = rng.standard_normal(length) * decay
    rir[0] = _DIRECT_PATH_AMPLITUDE
    return rir


def process_clip(audio, recipe_name, sr=16000, config=None, config_path=None, rng=None):
    """
    Run `audio` through the named channel recipe.

    Args:
        audio: 1-D input signal (numpy array or array-like), assumed mono
            PCM at `sr` Hz in [-1, 1].
        recipe_name: Key into the config's `recipes` mapping (e.g. "pstn",
            "whatsapp", "clean").
        sr: Sample rate in Hz. Defaults to 16000 (the pipeline's native
            rate — codec_roundtrip's decode side and apply_bandlimit's
            default both assume 16kHz).
        config: Optional pre-loaded config dict (as returned by
            `load_config`), to avoid re-reading/re-parsing the YAML file
            on every call. If omitted, `config_path` (or the default
            channels.yaml) is loaded fresh.
        config_path: Optional path to a recipes YAML file. Ignored if
            `config` is given.
        rng: Optional numpy.random.Generator, threaded through to every
            stage that needs randomness (RIR synthesis, noise, mic, packet
            loss), for reproducible output. Defaults to a fresh
            `np.random.default_rng()` if not given.

    Returns:
        numpy array, float64, 1-D, mono, same sample rate as the input
        (`sr`) and (up to codec pad/trim jitter, which is itself realistic
        per Doc 1) the same length as the input. Always clamped to
        [-1, 1] (see the 2026-09-06 safety-clamp note below).

    Raises:
        KeyError: if `recipe_name` is not defined in the config.
    """
    if config is None:
        config = load_config(config_path)

    recipes = config["recipes"]
    if recipe_name not in recipes:
        raise KeyError(
            f"Unknown recipe: {recipe_name!r}. Available: {sorted(recipes)}"
        )
    recipe = recipes[recipe_name]

    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(audio, dtype=np.float64)

    rir_cfg = recipe.get("rir")
    if rir_cfg:
        room_cfg = config["rooms"][rir_cfg["room"]]
        rir = _synth_rir(room_cfg, sr, rng)
        x = apply_rir(x, rir, wet_gain=rir_cfg.get("wet_gain", 0.7))

    noise_cfg = recipe.get("noise")
    if noise_cfg:
        x = apply_noise(x, noise_cfg["type"], noise_cfg["snr_db"], rng=rng)

    mic_cfg = recipe.get("mic")
    if mic_cfg:
        x = apply_mic(x, clip_prob=mic_cfg.get("clip_prob", 0.2), rng=rng)

    for codec_cfg in recipe.get("codec") or []:
        x = codec_roundtrip(x, codec_cfg["codec"], codec_cfg.get("bitrate"))

    loss_cfg = recipe.get("loss")
    if loss_cfg:
        x = apply_packet_loss(x, loss_cfg["p_gb"], loss_cfg["p_bg"], sr=sr, rng=rng)

    band_cfg = recipe.get("bandlimit")
    if band_cfg and band_cfg.get("enabled", False):
        x = apply_bandlimit(x, sr=sr)

    # Final full-scale safety clamp (2026-09-06): every downstream consumer
    # (FLAC write, mel-spectrogram extraction) assumes output in [-1, 1].
    # apply_mic() already clamps its own output, but this catches overshoot
    # from any other stage -- e.g. bandlimit filter ringing on a
    # near-full-scale signal -- regardless of which stage(s) caused it.
    x = np.clip(x, -1.0, 1.0)

    return x
