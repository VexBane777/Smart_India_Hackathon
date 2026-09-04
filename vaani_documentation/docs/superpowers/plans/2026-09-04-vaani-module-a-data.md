# VAANI Module A (Data Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the legal consent firewall, the TeleChannel telephony-degradation pipeline, and the real-call validation suite.

**Architecture:** A sequential DSP pipeline where audio is processed through physically ordered stages (RIR $\rightarrow$ Noise $\rightarrow$ Mic $\rightarrow$ Codec $\rightarrow$ Loss $\rightarrow$ Bandlimit). Results are indexed in a Parquet manifest filtered by a DPDP-compliant consent register.

**Tech Stack:** Python 3.11, FFmpeg, librosa, pandas, pyarrow, numpy, scipy.

## Execution Order & Cross-Module Dependencies
This module has **no upstream dependency** on B/C/D — it can start immediately and should,
since Module B (training) and Module D (`make data`) both consume its outputs.

- **Downstream consumers:** Module B needs Task 9 (manifest/splits) before real training
  runs begin. Module C needs Task 8 (`pipeline.py`) before it can channel demo assets
  (its Task 2). Module D's `make data` target (its Task 2/3/4) wraps this module's
  pipeline and manifest code directly.
- **Week 1 note (master plan §8):** the Monday demo slice does **not** require this
  module's full depth. Only a minimal recipe (raw TTS/RVC fakes + one or two channel
  recipes, ~3–5k windows) is needed to unblock Module B's Tuesday TinyCNN training —
  Tasks 10 (QA gates) and 11 (real-call validation) are Week 2–4 work, not Week 1
  blockers. Don't gate Week 1 on this module reaching Task 11.

## Global Constraints
- **Privacy**: DPDP Act 2023 aligned; no audio retained without itemized consent.
- **Audio Spec**: All processed output must be 16kHz mono FLAC.
- **Git Policy**: Audio files must NEVER enter git; only manifests and configs.
- **Sampling**: Split at source-clip level, real speech at speaker level.

---

### Task 1: Consent Register & Legal Firewall

**Files:**
- Create: `legal/consent_register.json`
- Create: `legal/revocation.py`
- Create: `legal/data_purge.py`

**Interfaces:**
- Produces: `consent_register.json` (Speaker ID $\rightarrow$ Consent Metadata)
- Consumes: Signed PDF/Paper forms.

- [ ] **Step 1: Initialize the register**
  Create `legal/consent_register.json` with the schema defined in TDD_MOD_A_01.
  ```json
  { "speakers": { "spk_001": { "consent_id": "c_260312_test", "opt_ins": ["A1", "A3"], "withdrawn": false } } }
  ```
- [ ] **Step 2: Implement revocation logic**
  In `legal/revocation.py`, implement a function `revoke_consent(speaker_id)` that sets `withdrawn: true` in `consent_register.json`.
- [ ] **Step 3: Implement data purge**
  In `legal/data_purge.py`, implement `purge_speaker(speaker_id)` that reads all `.parquet` manifests in `data/manifests/` and removes rows where `speaker_anon == speaker_id`.
- [ ] **Step 4: Verify revocation flow**
  Run: `python legal/revocation.py --id spk_001 && python legal/data_purge.py --id spk_001`.
  Verify: `consent_register.json` is updated and the speaker is gone from manifests.
- [ ] **Step 5: Commit**
  `git add legal/`
  `git commit -m "feat: implement DPDP consent firewall and revocation flow"`

### Task 2: TeleChannel Stage - RIR (Room Impulse Response)

**Files:**
- Create: `telechannel/stages/rir.py`
- Test: `tests/telechannel/test_rir.py`

**Interfaces:**
- Produces: `apply_rir(x, rir_id, wet_gain=0.7) -> y`

- [ ] **Step 1: Write failing test**
  Create `tests/telechannel/test_rir.py`. Test that `apply_rir` returns an array of the same length as input.
- [ ] **Step 2: Run test to verify failure**
  `pytest tests/telechannel/test_rir.py` $\rightarrow$ FAIL.
- [ ] **Step 3: Implement RIR convolution**
  In `telechannel/stages/rir.py`, use `scipy.signal.fftconvolve`. Align to the direct path (max peak of RIR).
  ```python
  def apply_rir(x, rir, wet_gain=0.7):
      rir = rir / np.max(np.abs(rir)) * 0.9
      d0 = np.argmax(np.abs(rir))
      wet = fftconvolve(x, rir)[d0:d0+len(x)]
      return (1-wet_gain)*x + wet_gain*wet
  ```
- [ ] **Step 4: Verify with impulse**
  Pass a Dirac delta $[1, 0, 0...]$ through the function. Verify output is exactly the RIR filter.
- [ ] **Step 5: Commit**
  `git add telechannel/stages/rir.py tests/telechannel/test_rir.py`
  `git commit -m "feat: add RIR convolution stage"`

### Task 3: TeleChannel Stage - Additive Noise

**Files:**
- Create: `telechannel/stages/noise.py`
- Test: `tests/telechannel/test_noise.py`

**Interfaces:**
- Produces: `apply_noise(x, noise_type, snr_db) -> y`

- [ ] **Step 1: Write failing test**
  Test that `apply_noise` results in the requested SNR $\pm 1$ dB.
- [ ] **Step 2: Implement SNR-based mixing**
  In `telechannel/stages/noise.py`, calculate RMS of signal and noise $\rightarrow$ scale noise $\rightarrow$ add.
- [ ] **Step 3: Verify SNR**
  Run test. Verify a 10dB SNR results in signal power being 10x noise power.
- [ ] **Step 4: Commit**
  `git add telechannel/stages/noise.py tests/telechannel/test_noise.py`
  `git commit -m "feat: add additive noise stage"`

### Task 4: TeleChannel Stage - Mic Stage

**Files:**
- Create: `telechannel/stages/mic.py`
- Test: `tests/telechannel/test_mic.py`

**Interfaces:**
- Produces: `apply_mic(x, clip_prob=0.2) -> y`

- [ ] **Step 1: Implement random gain and clipping**
  In `telechannel/stages/mic.py`, apply random gain between $-6$ and $+12$ dB. Apply `np.clip(y, -0.5, 0.5)` with probability `clip_prob`.
- [ ] **Step 2: Verify clipping**
  Test that for a high-gain signal, samples are capped at exactly $\pm 0.5$.
- [ ] **Step 3: Commit**
  `git add telechannel/stages/mic.py tests/telechannel/test_mic.py`
  `git commit -m "feat: add mic simulation stage"`

### Task 5: TeleChannel Stage - Codec Round-trip

**Files:**
- Create: `telechannel/stages/codec.py`
- Test: `tests/telechannel/test_codec.py`

**Interfaces:**
- Produces: `codec_roundtrip(x, codec, bitrate) -> y`

- [ ] **Step 1: Verify FFmpeg codecs**
  Run `ffmpeg -encoders | grep -Ei "gsm|amr|opus|alaw|mulaw"`. Verify `libopencore_amrnb` and `libvo_amrwbenc` are present.
- [ ] **Step 2: Implement round-trip logic**
  In `telechannel/stages/codec.py`, use `subprocess.run` to call FFmpeg:
  1. WAV $\rightarrow$ Codec (at target bitrate/SR).
  2. Codec $\rightarrow$ PCM (16kHz).
  3. Pad/Trim to original length.
- [ ] **Step 3: Verify G.711**
  Pass audio through `pcm_mulaw`. Verify output SR is 16kHz and quantization noise is present.
- [ ] **Step 4: Commit**
  `git add telechannel/stages/codec.py tests/telechannel/test_codec.py`
  `git commit -m "feat: add codec round-trip stage"`

### Task 6: TeleChannel Stage - Packet Loss

**Files:**
- Create: `telechannel/stages/packetloss.py`
- Test: `tests/telechannel/test_packetloss.py`

**Interfaces:**
- Produces: `apply_packet_loss(x, p_gb, p_bg) -> y`

- [ ] **Step 1: Implement Gilbert-Elliott mask**
  In `telechannel/stages/packetloss.py`, implement the two-state Markov chain to generate a binary mask of 20ms frames.
- [ ] **Step 2: Implement loss concealment**
  For each dropped frame, repeat the previous frame with a 5ms linear crossfade.
- [ ] **Step 3: Verify burstiness**
  Generate 1000 frames. Plot histogram of "consecutive drops". Verify it follows a geometric distribution (bursty), not a binomial one.
- [ ] **Step 4: Commit**
  `git add telechannel/stages/packetloss.py tests/telechannel/test_packetloss.py`
  `git commit -m "feat: add packet loss stage"`

### Task 7: TeleChannel Stage - Band-Limiting

**Files:**
- Create: `telechannel/stages/bandlimit.py`
- Test: `tests/telechannel/test_bandlimit.py`

**Interfaces:**
- Produces: `apply_bandlimit(x, sr=16000) -> y`

- [ ] **Step 1: Implement Butterworth filter**
  In `telechannel/stages/bandlimit.py`, use `scipy.signal.butter` (5th order) and `sosfiltfilt` for 300-3400 Hz.
- [ ] **Step 2: Verify cutoff**
  Pass a 5kHz sine wave. Verify output energy is $<-40$ dB relative to input.
- [ ] **Step 3: Commit**
  `git add telechannel/stages/bandlimit.py tests/telechannel/test_bandlimit.py`
  `git commit -m "feat: add band-limit stage"`

### Task 8: TeleChannel Pipeline Orchestrator

**Files:**
- Create: `telechannel/configs/channels.yaml`
- Create: `telechannel/pipeline.py`
- Test: `tests/telechannel/test_pipeline.py`

**Interfaces:**
- Produces: `process_clip(audio, recipe_name) -> processed_audio`

- [ ] **Step 1: Define recipes**
  In `channels.yaml`, define `pstn`, `gsm_2g`, `cellular_3g`, `volte`, `whatsapp`, and `tandem_xnet`.
- [ ] **Step 2: Implement pipeline chain**
  In `pipeline.py`, load `channels.yaml`. For a given recipe, apply stages in order: RIR $\rightarrow$ Noise $\rightarrow$ Mic $\rightarrow$ Codec $\rightarrow$ Loss $\rightarrow$ Bandlimit.
- [ ] **Step 3: Implement Tandem Transcoding**
  Allow `codec` stage to be called multiple times if the recipe contains a list of codecs.
- [ ] **Step 4: End-to-End Test**
  Process a 2s clip using `recipe: whatsapp`. Verify it passes through all stages and output is 16kHz mono.
- [ ] **Step 5: Commit**
  `git add telechannel/pipeline.py telechannel/configs/channels.yaml tests/telechannel/test_pipeline.py`
  `git commit -m "feat: implement pipeline orchestrator"`

### Task 9: Manifest & Data Split System

**Files:**
- Create: `telechannel/manifest.py`
- Create: `telechannel/configs/run_corpus.yaml`
- Test: `tests/telechannel/test_manifest.py`

**Interfaces:**
- Produces: `write_manifest_row(metadata)` $\rightarrow$ appends to `manifest.parquet`.

- [ ] **Step 1: Implement Parquet Logger**
  In `manifest.py`, implement logic to append metadata rows to a Parquet file using `pandas` and `pyarrow`.
- [ ] **Step 2: Implement Leakage-Safe Splits**
  Implement `generate_splits()`:
  1. Split by `source_clip` (not window).
  2. Split real speech by `speaker_anon`.
  3. Assign `train`, `val`, `test`, `demo` tags.
- [ ] **Step 3: Implement Hold-out Filters**
  Implement filters for: `generator == "rvc_team"`, `codec == "amr_wb"`, `noise == "freesound_horn"`.
- [ ] **Step 4: Verify Splits**
  Run `test_manifest.py`. Assert that no `source_clip` exists in both `train` and `test`.
- [ ] **Step 5: Commit**
  `git add telechannel/manifest.py telechannel/configs/run_corpus.yaml tests/telechannel/test_manifest.py`
  `git commit -m "feat: implement manifest and leakage-safe split system"`

### Task 10: TeleChannel QA Gates

**Files:**
- Create: `telechannel/qa.py`
- Test: `tests/telechannel/test_qa.py`

**Interfaces:**
- Produces: `verify_clip(audio) -> (bool, reason)`

- [ ] **Step 1: Implement Silence/Clipping Checks**
  In `qa.py`, check:
  1. RMS $> -50$ dBFS (not silent).
  2. No samples at exactly $\pm 1.0$ (no wraparound clipping).
- [ ] **Step 2: Implement Narrowband Check**
  Verify that for narrowband recipes, energy $> 3.5$ kHz is $<-25$ dB.
- [ ] **Step 3: Implement Duration Check**
  Verify duration is within $1.9\text{s} - 2.1\text{s}$.
- [ ] **Step 4: Implement Failure Logging**
  Write failures to `qa_failures.jsonl` with clip ID and reason.
- [ ] **Step 5: Commit**
  `git add telechannel/qa.py tests/telechannel/test_qa.py`
  `git commit -m "feat: add QA gates to pipeline"`

### Task 11: Real-Call Validation Tooling

**Files:**
- Create: `telechannel/validate_reality.py`
- Test: `tests/telechannel/test_validate.py`

**Interfaces:**
- Produces: `analyze_call(wav_path) -> {cutoff, ltas, hf_ratio}`

- [ ] **Step 1: Implement Spectral Analysis**
  In `validate_reality.py`, implement:
  1. VAD to strip silence.
  2. PSD computation (LTAS).
  3. Bandwidth cutoff finding ($-20$ dB point).
- [ ] **Step 2: Implement Comparison Logic**
  Create `compare_to_sim(real_metrics, sim_metrics)` $\rightarrow$ compute absolute difference and correlation.
- [ ] **Step 3: Generate Validation Plot**
  Use `matplotlib` to overlay the LTAS of a real call vs a simulated one.
- [ ] **Step 4: Verify against Baseline**
  Run against a "clean" recording and the "clean" recipe. Correlation should be $\approx 1.0$.
- [ ] **Step 5: Commit**
  `git add telechannel/validate_reality.py tests/telechannel/test_validate.py`
  `git commit -m "feat: implement real-call validation tooling"`
