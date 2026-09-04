<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 1 — TeleChannel Generation Script Spec · Owner: Corpus Lead · Status: Build-ready
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 1 — TeleChannel Generation Script Spec

**Owner:** Corpus Lead · **Status:** Build-ready · **Suite:** v1.0 · **Generated:** 2026-09-03
**Depends on:** Doc 6 consent forms before recording anything

---

## 1.1 What this builds
A scripted pipeline pushing clean real/fake audio through a physically ordered simulation
of an Indian phone network: room → noise → mic → codec → packet loss → band-limiting.
Every clip carries full metadata, so every split and held-out protocol is a *filter on
metadata*, never a folder shuffle.

## 1.2 Repo layout
```
telechannel/
├── configs/   channels.yaml  run_corpus.yaml
├── telechannel/
│   ├── manifest.py   stages/{rir,noise,mic,codec,packetloss,bandlimit}.py
│   ├── pipeline.py   qa.py   cli.py
├── scripts/   check_ffmpeg.py  benchmark_stage.py  fingerprint_validate.py
└── README.md  (the released "recipe")
```

## 1.3 Day 0 — verify codec support before anything else
```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep -Ei "gsm|amr|opus|alaw|mulaw"
```
Expected: `pcm_mulaw`, `pcm_alaw`, `libgsm`, `libopus`, `libopencore_amrnb`,
`libvo_amrwbenc`. If AMR encoders are missing (common in minimal builds): install full
static builds (gyan.dev "full" Windows / johnvansickle Linux / conda-forge ffmpeg),
re-run check. Missing codecs silently shrink the dataset — check first.

## 1.4 Stage implementations

**Codec round-trip — the star. Encode with the real codec, decode back. The damage is
authentic, not approximated:**

```python
CODECS = {
    # name:      (ffmpeg encoder,      sr,     ext,    bitrates)
    "g711u":  ("pcm_mulaw",         8000, "wav",  [None]),
    "g711a":  ("pcm_alaw",          8000, "wav",  [None]),
    "gsm_fr": ("libgsm",            8000, "wav",  [None]),
    "amr_nb": ("libopencore_amrnb", 8000, "amr",  ["4.75k","5.9k","7.4k","12.2k"]),
    "amr_wb": ("libvo_amrwbenc",   16000, "amr",  ["8k","12.4k","16k","23.85k"]),
    "opus":   ("libopus",          48000, "opus", ["6k","8k","12k","16k"]),
}

def codec_roundtrip(x, sr, codec, bitrate, workdir):
    """Encode to the transport codec, decode back to PCM — exactly what a
    network does. Tandem transcoding = call this twice, different codecs."""
    # write x -> a.wav; ffmpeg encode a.wav -> b.<ext> at codec sr + bitrate;
    # ffmpeg decode b -> c.wav at 16000 Hz pcm_s16le; read y from c.wav
    return pad_or_trim(y, len(x))   # length shifts are real (jitter buffers)
```

**Packet loss — bursty, not random:**

```python
def gilbert_elliott(n, rng, p_gb=0.03, p_bg=0.30, eps_g=0.001, eps_b=0.6):
    """Two-state Markov chain: a 'good' state that rarely drops packets and
    a sticky 'bad' state that drops heavily — losses arrive in bursts, like
    real networks."""
    out, bad = np.zeros(n, bool), False
    for i in range(n):
        out[i] = rng.random() < (eps_b if bad else eps_g)
        bad = (rng.random() < p_gb) if not bad else (rng.random() > p_bg)
    return out

def conceal_loss(x, sr, dropped):
    """Fill each dropped 20 ms frame by repeating the last received frame
    with 5 ms crossfades at 0.9 gain — a simple version of the concealment
    every phone already performs."""
```

**Room, mic, band (order matters — mic BEFORE codec, because μ-law quantization
noise depends on signal level):**

```python
def apply_rir(x, rir, wet_gain=0.7):
    """Convolve with a room impulse response (RIR = the room's recorded echo
    fingerprint, SLR28 library), aligned to the direct path, blend wet/dry."""
    rir = rir / np.max(np.abs(rir)) * 0.9
    d0 = int(np.argmax(np.abs(rir)))
    wet = fftconvolve(x, rir)[d0:d0+len(x)]
    return (1-wet_gain)*x + wet_gain*wet

def mic_stage(x, rng, gain_db=(-6, 12), clip_prob=0.2):
    y = x * 10**(rng.uniform(*gain_db)/20)
    return np.clip(y, -0.5, 0.5) if rng.random() < clip_prob else y

def telephone_band(x, sr=16000):
    """Classic narrowband: keep only 300–3400 Hz. ONLY for narrowband recipes
    (PSTN/2G/AMR-NB) — AMR-WB and Opus are wideband."""
    sos = butter(5, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)
```

Noise pools: MUSAN + ESC-50 + CC0 Freesound Indian street/market/horn, license-filtered,
mixed at target SNR (speech-to-noise loudness ratio).

## 1.5 Channel recipes (named presets — the dataset's structure)
```yaml
recipes:
  clean:        {}                                   # control / debugging
  pstn:         { room: medium, noise: office,  net: [band, g711] }
  gsm_2g:       { room: far,    noise: outdoor, net: [band, gsm_fr] }
  cellular_3g:  { room: medium, noise: street,  net: [band, amr_nb*] }  # *random bitrate
  volte:        { room: near,   noise: light,   net: [amr_wb] }
  whatsapp:     { room: near,   noise: home,    net: [opus], loss: higher }
  tandem_xnet:  { room: medium, noise: street,  net: [band, amr_nb, g711] }
# each field carries a sampled parameter range, versioned with the output
```

## 1.6 Manifest schema (one row per output clip)
```json
{
  "clip_id": "tc_000123",
  "source":  { "file": "...", "origin": "common_voice_hi|librispeech|generated",
               "label": "real|fake", "generator": "melo|bark|rvc_team|indic_tts|null",
               "speaker_anon": "spk_041", "language": "hi-en|ta-en|en-in",
               "license": "CC0|MIT|apache2|team_consent_v2" },
  "channel": { "recipe": "cellular_3g", "rir_id": "slr28_sim_012",
               "noise": "musan_babble", "snr_db": 11.3,
               "codec": "amr_nb", "bitrate": "7.4k",
               "loss_pct": 2.1, "loss_burst_max": 5 },
  "audio":   { "dur_s": 2.0, "sr": 16000, "peak_dbfs": -3.0 },
  "provenance": { "pipeline_version": "1.0.0", "config_hash": "a1b2c3",
                  "rng_seed": 12345, "timestamp": "2026-05-01T22:14Z" }
}
```
**Hard rules:** (1) splits filter on manifest fields, never folder names; the four
hold-outs are one-liners (`generator == "rvc_team"`, `codec == "amr_wb"`,
`noise == "freesound_horn"`, `origin == "real_call"`). (2) split at source-clip level,
not window level — same-clip leakage inflates metrics. (3) real speech splits at speaker
level. (4) license field mandatory — anything without a clean value fails QA.

## 1.7 Sizing & execution
~50–60 h real + ~50–60 h fake × 6 recipes ≈ 500–600 h ≈ ~1M two-second windows.
FFmpeg round-trips are subprocess calls → one worker process per core. RIR convolution
is the slow stage. **Do not trust speed estimates:** benchmark on 100 clips Day 1 and
extrapolate. Distribute across team laptops + Kaggle CPU sessions. FLAC only (~35 GB) —
never lossy codecs. Peak-normalize to −3 dBFS *after* mic/codec stages.

## 1.8 QA gates (every clip)
| Check | Fail if |
|---|---|
| Not silent | RMS < −50 dBFS |
| No wraparound clipping | any sample at ±1.0 post-normalize |
| Narrowband recipes actually narrowband | energy >3.5 kHz above −25 dB relative |
| Duration | outside 1.9–2.1 s |
| Manifest | any required field missing/invalid |
Failures log to `qa_failures.jsonl` — never silently dropped.

## 1.9 Validation against reality (the killer slide)
Compare 25 real consented calls (Doc 5) vs. simulated corpus on: long-term average
spectrum, bandwidth cutoff (−20 dB point), high-frequency energy ratio.
**Targets:** median cutoff within ±300 Hz; LTAS correlation >0.9. Output figure goes to
Doc 3 Slide 8. A mismatch is a *finding* — it tells you which recipe parameter to fix.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
