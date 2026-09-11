# VoiceGuard evaluation protocol (v12, 2026-09-11)

This document governs every VoiceGuard model evaluation, checkpoint
selection and training run. The code enforces it; the pointers below show
where. If you change the protocol, change this file, the code and the tests
together.

## 1. Policy: no clean-only evaluation

> **No eval runs purely on clean audio, unless the stated application is
> retraining the model for banks' use.** (User directive, 2026-09-11.)

VoiceGuard is a phone-call product. v11_seqcnn was trained `--channel none`
and every one of its evals ran on clean audio, so its headline EER (0.0624)
said nothing about calls. The same rule applies to training.

Enforcement:
- `model_training/eval_protocol.py`: `resolve_channels(channels,
  application, purpose)` raises `CleanOnlyEvalError` if the channel set has
  no phone channel and `--application` is not `bank`. Every entry point
  (`train_seq_cnn.py`, `build_caches.py`, `select_best_checkpoint_seqcnn.py`,
  `evaluate.py`, `validate_fp16.py`) goes through it.
- `test_eval_protocol.py` AST-scans every non-test script and fails on a
  literal clean-only channel list (`channel_recipes=[None]`,
  `default=[None]`, `("none",)`): the exact pattern the v11-era scripts used.
- TeleChannel's `clean` recipe is rejected everywhere. It is a debug recipe
  (RIR + noise + clipping + packet loss, no codec), neither undegraded audio
  nor a phone path. Undegraded audio is spelled `none`.

## 2. Channels

| group | channels | used for |
|---|---|---|
| reference | `none` | a reference row in every report, never a result on its own |
| seen phone | `whatsapp`, `volte`, `cellular_3g` | training + eval |
| unseen phone | `gsm_2g`, `pstn`, `tandem_xnet` | eval only (never trained on) |

Recipes: `vaani/telechannel/configs/channels.yaml`. Default eval =
`none` + all six phone channels. **Headline numbers pool the phone
channels.** Training renders each file twice: `none` plus one phone
channel from the seen group, chosen by file hash.

## 3. Data splits

- Committed manifest: `model_training/eval_splits/held_out_split_v1.json`
  (written once by `make_eval_splits.py`, which refuses to overwrite it;
  bump the version for a new split). Lists every file explicitly, so files
  added to a directory later never silently enter an eval.
- Core sets, split 50/50 by a stable hash of the file id into `select` and
  `test`: `itw_real` (real_itw_held), `itw_fake` (fake_itw_held),
  `noiseaug_real_en`, `noiseaug_real_hi` (real_noise_aug_split/held).
- Test-only sets: `mlaad_fake` (580 files, ~20 modern TTS systems never
  trained on, FLAC) and the accent cells (accents_split/held, capped at 300
  files each; the successor to eval_accent_cells.py).
- **`select` is the only data checkpoint selection may read**
  (`select_best_checkpoint_seqcnn.py`; a test fails if it references the test
  split). **`test` is read once per candidate** for the final report.
  v11 selected on fake_itw_held, which was also its test set.
- `make_eval_splits.py` refuses to write a manifest where any eval file is
  also a training file, or where an ITW held basename appears in ITW train.
- Limitation: In-the-Wild's meta.csv (speaker labels) is gone from this
  machine (its HF mirror is empty), so select/test is split by file, not
  speaker. Train vs held-out *is* speaker-disjoint (prep_in_the_wild.py).

## 4. Windowing (dataset.py, identical for train and eval)

The app scores a 3 s window every second and never pads. Training/eval
windows are built to match:
1. trim edge silence (randomized pad, per-file seed);
2. `< 1.0 s` after trimming: dropped, and counted in the cache manifest;
3. `1.0–3.0 s`: padded to 3 s (+0.1 s margin) with Gaussian noise at the
   clip's own noise floor, before the channel, so the pad carries channel
   noise like the quiet part of a real call window;
4. `>= 3.0 s`: channel on the whole clip, non-overlapping 3 s windows,
   plus an end-aligned tail window if >= 1 s remains.

Each window records `pad_fraction`. Under the old 3 s cutoff, 44–75% of
held-out files and ~74% of training files were silently dropped.

Padding could become a shortcut: the share of short clips differs by source
and label (fake2021 ~68% vs real2021 ~45%). Training equalizes each source
set's padded-window share at ~50% (`compute_pad_policy`: some long-clip
windows become padded random crops, or some short clips are skipped).
Eval never rebalances; `pad_fraction` is a gated confound feature (§6).

All randomness is seeded by a stable hash of (data-relative file id,
purpose, channel, seed), not by task index. Before v12 the same directory
gave 762 files in one script and 757 in another.

## 5. Feature cache (feature_cache.py)

Per (file list, channel, seed) unit: `seq.npy` (float16, memory-mapped),
`scalars.npy`, `meta.npz`, `manifest.json`. Built in resumable shards with
bounded RAM (this fixes v11's OOM). The manifest stores a **feature-version
hash** over the code of features.py, dataset.py and TeleChannel (AST,
docstrings stripped), channels.yaml's content, library versions and the
ffmpeg build. A stale cache raises `StaleCacheError` and is never silently
reused. float16 storage is validated by `validate_fp16.py` (gate: max
|Δp| <= 0.01).

## 6. Metrics and gates (evaluate.py)

Operating threshold per model = its phone-pooled EER threshold on
`select`. Reported on `test`:

- **Headline**: EER on core sets, pooled over phone channels, with a 95%
  bootstrap CI resampled by source file (all windows and channels of a file
  together); FPR/FNR at the select threshold and at the app's 0.60.
- EER per channel, seen vs unseen phone groups, padded windows only,
  core+MLAAD, per set (FPR for real sets, FNR for fake sets), accent cells.
- **Confound v2**, reals AND fakes, features pauseRatio, energyVariance,
  zcrVariance, jitter, shimmer, HNR, pad_fraction: half-means (median
  split), absolute gap, ratio, Spearman ρ with p_fake, and FPR (reals) /
  FNR (fakes) per half.
  **Gates** (phone-pooled; fixed before any model was scored on this
  protocol):
  - |ρ| <= 0.10;
  - a rate row fails only if worse/better ratio > 1.25 AND |difference| >
    1 point AND the file-level bootstrap CI of the difference excludes 0 at
    a Bonferroni-corrected level (alpha 0.05 over all 14 rows).
  The significance clause exists because the bare ratio gate failed a
  synthetic model with no confound, purely by chance
  (`test_evaluate.py`).
- **Attack-type head**: MLAAD (all TTS, unseen systems) share predicted
  "tts"; balanced accuracy on a training run's val fakes, in-distribution
  vs leave-attack-out (A11 TTS + A18 VC are masked from the attack-type loss
  in training) vs the TTS+VC hybrids A13–A15; coverage and accuracy of the
  sub-label the UI actually shows (p_fake >= 0.60 and confidence >= 0.70).
  **Gates**: leave-out balanced accuracy >= 0.70 and MLAAD tts share >= 0.70.
  If they fail, hide the sub-label in the app.
- **Deploy gate** (`--candidate --reference`): the candidate passes every
  confound gate AND beats the reference's test headline EER. CI overlap is
  reported alongside.

Deprecated metric: the v11-era "gap" (absolute difference of real-clip
half-means on clean audio) depends on the model's score scale and hides
direction. It is kept only as one column of confound v2.

## 7. Commands

```bash
cd voice_guard/model_training
PY=.venv313/Scripts/python.exe          # main checkout's venv
export VOICEGUARD_CACHE_ROOT=...        # optional; default model_training/cache
$PY build_caches.py --eval              # held-out caches, both splits x 7 channels
$PY build_caches.py --train             # training caches (~2x files)
$PY train_seq_cnn.py --out runs/voice_guard_v12
$PY select_best_checkpoint_seqcnn.py --run runs/voice_guard_v12 --out runs/voice_guard_v12_selected
$PY validate_fp16.py --model runs/voice_guard_v11_seqcnn_selected/model.pt --out runs/fp16_validation.json
$PY evaluate.py --split test --out runs/eval_v12_test \
    --model v9=runs/voice_guard_v9_noisefix_final/model.pt \
    --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt \
    --model v12=runs/voice_guard_v12_selected/model.pt \
    --attack-val-run runs/voice_guard_v12 --candidate v12 --reference v11 \
    --fp16-report runs/fp16_validation.json
```

## 8. Retired scripts (2026-09-11)

| retired | why | replaced by |
|---|---|---|
| train.py (MLP) | broken since the seq refactor (2-tuple to_arrays) | train_seq_cnn.py; compute_eer → eval_stats.py, parse_channel_arg → eval_protocol.py |
| train_curriculum.py | same breakage, MLP only | train_seq_cnn.py (warmup+cosine, EMA) |
| eval_held_out.py | 66-d MLP could not load v9 (63-d) | evaluate.py |
| eval_held_out_dirs.py, eval_held_out_dirs_seqcnn.py | clean-only, selection/test leakage | evaluate.py |
| measure_confound.py, measure_confound_seqcnn.py | clean-only, reals only, scale-dependent gap | evaluate.py confound v2 |
| select_best_checkpoint.py (MLP) | broken; selected on the test fakes | select_best_checkpoint_seqcnn.py (select split only) |
| eval_accent_cells.py | broken 2-tuple unpack | evaluate.py accent sets |
| export_tflite.py | app runs ONNX directly since 2026-09-09; asserted the 66-d MLP shape | model.export_onnx |
| features.chunk_audio | dropped every clip < 3 s | dataset.process_file |
