# Frame-level sequence model — a custom, India-tuned alternative to LCNN

**Date:** 2026-09-11
**Status:** Plan / scope only. Not started. This is the concrete next step
for remediation track 3 in `voice_guard/docs/CRITICAL-entity-vs-style-
confound.md` — read that first if you haven't.
**Why this doc exists:** the entity-vs-style confound is a representational
ceiling (mean-pooled statistics can't distinguish "controlled human
delivery" from "vocoder output"), not a data-quality bug — no further
retraining of the current 63-feature model will fix it. This plan replaces
the pooling step with a small, custom-designed sequence model, trained on
this project's own accent-diverse corpus — not a port of academic LCNN.

---

## 1. The key fact that makes this cheaper than it sounds

**Both the Python training pipeline and the Dart on-device pipeline already
compute the full per-frame LFCC sequence and then throw it away.**

- `model_training/features.py::extract_lfcc` computes `lfcc_frames`, shape
  `(184, 60)` for every 3-second window (184 frames at hop 256 / 16kHz;
  confirmed directly: `1 + (48000 - 1024) // 256 = 184`), then
  mean-pools it to a single 60-vector at the last two lines of the function.
- `lib/utils/audio_processor.dart::extractLfcc` computes the identical
  `lfccFrames` (a `List<List<double>>`) and mean-pools it the same way
  (lines 34–38 of that file).

So this is **not** "add a new feature-extraction stage" — it's "stop
discarding a sequence that already exists in both languages' code today,"
and feed it to a small model instead of a mean.

## 2. Proposed design

### 2.1 Input representation

- **Sequence branch**: the `(184, 60)` LFCC frame matrix, per 3s window —
  unchanged math, just not pooled.
- **Scalar branch**: keep the existing 3 prosody scalars (`pauseRatio`,
  `energyVariance`, `zcrVariance`) computed exactly as today, unpooled
  changes not required there. Kept separate rather than forced into the
  sequence branch — deliberately limits the blast radius of this change;
  prosody extraction/normalization code doesn't need to be touched.

### 2.2 Model architecture — small 1D-CNN, not a GRU/LSTM and not LCNN

Recommended: 2–3 `Conv1d` layers over the time axis (kernel size 3–5,
channels roughly `60 → 32 → 16`), followed by global average+max pooling
over the remaining time axis to get a fixed-size learned embedding, then
concatenated with the 3 prosody scalars before a small classifier head
(mirrors today's `64 → 32 → 2` shape, just with a richer input).

**Why CNN over recurrent (GRU/LSTM):**
- Stateless per-window inference (matches how the app already scores
  independent 3s windows — no need to manage or export RNN hidden state
  across calls).
- Materially simpler and more reliable ONNX export/mobile-runtime support
  — Conv1d is a standard, well-covered op; recurrent ops have more
  historical friction in mobile ONNX Runtime builds. **Verify
  `flutter_onnxruntime`'s actual op coverage for `Conv`/`Conv1d` before
  committing to this — not yet checked as of this plan.**
- Still captures exactly the kind of local, short-range temporal pattern
  (formant-transition smoothness, vocoder buzziness) the CRITICAL doc
  argues no pooled statistic can expose — a CNN's receptive field over
  neighboring frames is the right shape for that, without needing
  long-range memory.

**Why NOT a port of academic LCNN specifically:** LCNN's defining
ingredient is the Max-Feature-Map (MFM) activation (channel-halving
max-out), tuned for ASVspoof's original studio-quality, narrow-generator-
family setting. There's no requirement to reuse it — a plain CNN designed
against *this* project's own corpus (accent-diverse, telephony-degraded,
noise-augmented) and *this* project's own deployment constraint (real-time
mobile, ONNX Runtime, sub-6KB-parameter class today) is more directly
tunable to what we actually need, and avoids importing assumptions (MFM's
specific inductive bias) validated on a dataset we've already found doesn't
transfer cleanly to our target distribution (the whole ITW-generalization
saga from the prior session).

### 2.3 Parameter budget

Current `VoiceGuardMLP` (64,32 hidden) is ~6K params. A 3-layer 1D-CNN at
the channel sizes above is comfortably under 50–100K params — still
trivial for real-time mobile inference (sub-millisecond), no realistic
latency risk.

### 2.4 Optional, complementary enhancement (not required for v1)

The CRITICAL doc's remediation track 2 (jitter/shimmer/harmonic-to-noise-
ratio — genuine physiological markers) can be folded into this same design
cheaply, as **extra per-frame input channels** alongside LFCC, rather than
pursued as a separate, sequential effort. Worth prototyping only after v1
(sequence model on LFCC alone) is validated, to isolate which part of any
improvement is doing the work.

## 3. What has to change, by file

| File | Change |
|---|---|
| `model_training/features.py` | `extract_lfcc` returns the full `(184, 60)` frame matrix (new function or a flag); keep a thin wrapper that still mean-pools for anything not yet migrated (backward compat during the transition). |
| `model_training/dataset.py` | `Example.features` becomes `(sequence: (184,60) ndarray, prosody: (3,) ndarray)` instead of a flat `(63,)` vector. `to_arrays` needs both arrays batched. Fixed shape per window (chunk_audio already guarantees constant-length windows) — **no padding/masking needed**, meaningfully simpler than general variable-length sequence modeling. |
| `model_training/model.py` | New model class (e.g. `VoiceGuardSeqCNN`) — Conv1d stack + prosody concat + classifier head + `FixedNormalize`-equivalent (needs per-frame-feature normalization for the sequence branch, separate stats for the scalar branch). |
| `model_training/train.py` / `train_curriculum.py` | Adjust batching/DataLoader for the two-branch input; otherwise the training loop shape is unchanged. |
| `lib/utils/audio_processor.dart` | `extractLfcc` stops mean-pooling; returns the frame matrix (flattened or nested, matching whatever ONNX input layout is chosen). |
| `lib/services/src/tflite_io.dart` | Build a 2-input (or 1 concatenated-flattened-input) ONNX tensor instead of the current 63-length vector; verify `flutter_onnxruntime` handles the new input shape/dtype cleanly. |
| `model_training/eval_held_out_dirs.py`, `select_best_checkpoint.py`, `dataset_audit.py` | Need shape updates wherever they call `build_examples`/`to_arrays` directly — mechanical, not conceptual, changes. |

## 4. Validation plan — reuse this session's hard-won discipline, don't skip it

1. **Phase 1, Python-only, no Dart/mobile changes yet.** Prototype the new
   model entirely offline: modify `features.py` to emit sequences, build
   the CNN, train on the existing corpus, evaluate against:
   - The **exact acceptance criterion for the confound itself**: rerun this
     session's median-split correlation analysis (real clips split by
     `energyVariance`/`pauseRatio`/`zcrVariance`, compare mean fake-
     probability between halves) — the new model must show a **much
     weaker** split than the current model's measured 25.1%-vs-12.9%
     (`energyVariance`), 14.7%-vs-23.6% (`pauseRatio`), 13.1%-vs-24.9%
     (`zcrVariance`). This is the actual thing being fixed — don't accept
     "ITW EER improved" alone as evidence this is resolved.
   - **No regression** on noise-FPR (`data/real_noise_aug_split/held`,
     currently 10.4%) or ITW held-out EER (currently 0.1602, CI
     [0.1431,0.1775]) relative to the currently-deployed `v9_noisefix`
     model — use `eval_stats.py`'s bootstrap CI, not raw point estimates,
     per this session's own pipeline-versioning-bug lesson.
   This phase is cheap and fully de-risks the ML question — whether a
   sequence model actually fixes the confound — before touching any
   production Dart/mobile code.
2. **Phase 2, only if Phase 1 passes all three criteria above.** Port to
   Dart, verify Dart/Python numerical equivalence for the (unchanged) LFCC
   math up to the point where pooling used to happen, export ONNX with the
   new input shape, rebuild, and **live-test on-device with the exact same
   protocol used this session**: real speech in both a controlled/
   recitation register and a natural/varied register (the two that
   produced 60-80s vs 0-20s on the current model), real ambient noise, and
   the genuine AI-clone clip (`model_training/test_assets/
   ai_clone_test_clip.wav`).
3. **Phase 3, optional.** Only after Phase 2 ships: prototype folding in
   jitter/shimmer/HNR (§2.4) if the pure-sequence model alone doesn't fully
   close the gap.

## 5. Open risks / unresolved questions, flagged rather than assumed away

- `flutter_onnxruntime`'s Conv1d op coverage on the actual Android ONNX
  Runtime mobile build — not yet checked.
- Whether per-frame LFCC normalization needs its own baked statistics
  (analogous to `FixedNormalize`) computed per-feature-per-frame-position,
  or just per-feature-across-all-frames — needs a design decision before
  implementation, not yet made.
- This plan does not address the accent-expansion / curriculum-training
  open question from the prior session (whether `en_foreign`/`hi_native`/
  `hi_foreign` should be folded back into the training mix) — orthogonal to
  this change, worth revisiting together once the new architecture exists,
  not before.
