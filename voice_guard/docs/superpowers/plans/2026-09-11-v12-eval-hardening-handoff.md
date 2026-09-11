# v12 eval-hardening + retrain — plan and handoff (2026-09-11)

**Status at handoff: design done, no code written yet. Paused for a machine
reboot** (see "Why the reboot" below). Resume with `claude --resume` in
`SIH/.worktrees/voiceguard-v12-hardening`, which restores the full
conversation. If that's not possible, a fresh session can continue from this
document alone.

## The request (user, 2026-09-11 ~21:19 IST)

Fix every item (1–8) from the v11 review "thoroughly in one single pass such
that we need never visit these again or have them surface unexpectedly in
future runs". **Hard policy: no eval from now on runs purely on clean audio,
unless the stated application is retraining the model for banks' use.** Do it
the way I (Claude) judge best, and document it well. Coordinate with the
UI-repair session over SendMessage.

The v11 review findings being fixed:
1. `state.md`'s v11 confound table listed v9's *half-means* as "baseline gaps".
   The real v9 gaps are energyVariance 12.2pt, pauseRatio 8.9pt and
   zcrVariance 11.8pt. The absolute-point gap metric also depends on scale
   (v11's mean fake-prob on reals is about half of v9's). As a ratio,
   pauseRatio only moved from 1.61× to 1.36×.
2. Checkpoint selection used `fake_itw_held`, which is also the fake half of
   the headline EER test, so the 0.0624 figure is optimistic.
3. The fake side of the confound (the CRITICAL doc's 20.3% missed fakes) was
   never re-measured.
4. The attack-type head was never evaluated, but it is shipped in the UI.
5. v11 was trained `--channel none` and every eval was on clean audio. That
   is the item the user's policy targets.
6. The receptive field is about 5 frames (~130 ms), too little temporal
   context.
7. The model underfits: still improving at epoch 25, constant LR, epoch-to-epoch
   selection noise.
8. OOM: `to_arrays` loads every window into RAM. That is what forced
   `--channel none`.
9. (Also found) The ≥3 s cutoff drops 44–75% of held-out files **and about 74%
   of training files** (v11 kept 69,044 of ~267k source files).
10. (Also found) The same `fake_itw_held` dir gave 762 files in one script and
    757 in another. Cause: per-file RNG seeds come from the *task index*, so
    `trim_edge_silence`'s random pad differs by run composition, and
    borderline ~3.0 s files flip in or out.
11. (Also found) The legacy MLP scripts are **broken at runtime** since the
    seq-CNN refactor. `eval_held_out_dirs.py` and `select_best_checkpoint.py`
    unpack 2 values from the 4-tuple `to_arrays`, `measure_confound.py` reads
    the removed `Example.features`, and `train.py`/`train_curriculum.py` do the
    same. `eval_held_out.py` builds a 66-d MLP that can't load v9 (63-d).
    `eval_accent_cells.py` also unpacks 2 values.

## Environment facts (verified this session)

- Python: `voice_guard/model_training/.venv313/Scripts/python.exe` (torch
  2.11.0+cu128, CUDA OK, soundfile/librosa/onnxruntime/pytest). The PATH
  `python` interpreters do NOT have soundfile.
- GPU: RTX 5050 Laptop, 8 GB. CPU: 20 logical cores. RAM: 15.4 GB. Disk C:
  52 GB free.
- Worktree `SIH/.worktrees/voiceguard-v12-hardening`, branch
  `voiceguard-v12-hardening`, cut from origin/vaani `e3d7e4d` (includes the new
  shipped UI). The worktree's `voice_guard/model_training/data/<dir>` entries
  are **NTFS junctions** to the main worktree's gitignored corpus dirs (git
  status stays clean).
- Baseline test run in the worktree before any change: **41 passed, 8
  skipped** (the skips were data-dependent tests from before the junctions).
- The ITW `meta.csv` (speaker labels) is **gone**. `data/in_the_wild` is
  empty and the HF mirror `sarkarbkl/In_the_wild_audio_deepfake` has only
  `.gitattributes`. So the new held-out select/test split is **file-level
  (deterministic hash), not speaker-level**. Train and held were already
  speaker-disjoint (prep_in_the_wild.py). Only select vs test may share
  speakers. Document this limitation.
- File counts: real 2,580; real2021 21,000; real_itw_train 15,525; fake 22,800
  (ASV2019 train, A01–A06, 3,800 each); fake2021 185,410 (A07–A19 × 7 codecs;
  labeled via trial_metadata.txt); fake_itw_train 10,455;
  real_noise_aug_split/train en 6,816 / hi 2,400; held en 1,703 / hi 600;
  real_itw_held 4,438; fake_itw_held 1,361; mlaad_en500 580 **FLAC** files,
  nested `fake/en/<system>/`, about 20 modern TTS systems (ElevenLabs, F5,
  Chatterbox, …), a good unseen-generator test.
- ASVspoof2019 taxonomy checked against Wang et al., arXiv 1911.01601: A13–A15
  are TTS+VC hybrids, A17–A19 are VC, A16 = A04 (TTS), A19 = A06 (VC). The
  `attack_labels.py` table (hybrids → vc) is consistent with the paper.
  Update its "not verified" docstring.
- App facts: alert threshold default 0.60 on an EMA (alpha 0.7) of per-window
  scores (`risk_score_provider.dart`, `settings_provider.dart`). The
  attack-type sub-label is gated at confidence ≥0.70 (`call_screen.dart:929`).
  It scores 3 s windows every 1 s and never pads (always 3 s of real audio).
- Other session: "Vaani app live test and monitoring" (Remote Control, another
  device) is the UI-repair session. It is editing only
  `risk_score_provider.dart`, `logs_screen.dart` and `home_screen.dart`
  (fake-data removal), and will commit straight to vaani. **Ping it before
  touching `call_screen.dart`, `lib/services/*` or `assets/models/*`.**

## Design (decided — implement as written unless evidence says otherwise)

**A. `eval_protocol.py` (policy, single source of truth)**
- `PHONE_CHANNELS = (whatsapp, volte, cellular_3g, gsm_2g, pstn, tandem_xnet)`.
  `clean` is a debug recipe and is never used.
- `TRAIN_CHANNELS = (none, whatsapp, volte, cellular_3g)`. The held-out
  "unseen channel" eval group is gsm_2g, pstn and tandem_xnet.
- `DEFAULT_EVAL_CHANNELS = (none,) + PHONE_CHANNELS`. `none` is a reference
  row only, never used alone.
- `--application {phone,bank}` (default phone).
  `resolve_channels(channels, application, purpose)` raises
  `CleanOnlyEvalError` if the resolved set has no phone channel and
  application != bank. Training uses the same rule: no clean-only training
  unless bank.
- `parse_channel_arg` moves here from train.py.
- A policy test does an AST scan of every script calling
  `build_examples`/`build_feature_cache`/eval entry points and fails on a
  literal clean-only channel list. Behavioral tests cover `resolve_channels`.

**B. `dataset.py`**
- Per-file RNG seed = stable hash(resolved path relative to the data root,
  recipe, base seed), not the task index. This fixes item 10 and makes caches
  reusable.
- Windowing (`window_clip`): trim edges, then:
  - clips shorter than 1.0 s are dropped (reported);
  - clips from 1.0 to 3.0 s are padded to 3 s (+0.1 s margin) with Gaussian
    noise at the clip's own noise floor (the quietest-10% frame RMS, floor
    1e-4), split randomly between front and back, *before* the channel
    (so the pad gets channel noise, like the silence in a real call's 3 s
    window);
  - long clips get non-overlapping 3 s windows **plus** an end-aligned tail
    window if the remainder is ≥1 s.
- `pad_fraction` is recorded per window. Confound analysis includes it, and a
  corpus report shows real-vs-fake pad distributions, so padding can't become
  a hidden shortcut.
- Loads FLAC and recurses (for MLAAD) via explicit file lists.
- Per-dir deterministic caps: fake2021 capped at 40,000 files (hash
  subsample). Otherwise the corpus is dominated 4.5:1 by fakes.
- Training channel assignment: each file gets `none` plus ONE deterministic
  random phone channel from TRAIN_CHANNELS. That's about 2× the files, a
  compromise between v9's 3× and memory/time.

**C. `feature_cache.py` (fixes OOM, item 8)**
- On-disk memmap cache per (source dir, recipe): `seq.npy` float16 (N×184×60),
  `scalars.npy` float32, `meta.npz` (source_file id, label, attack_type,
  attack_id, window index, pad_fraction, duration), `manifest.json` with a
  **feature-version hash** (features.py + dataset windowing constants +
  channels.yaml + telechannel stage sources). A stale cache is rejected
  loudly, never silently reused.
- Workers write shards, the parent consolidates. Peak RAM is about one shard.
- Validate that float16 storage changes model outputs negligibly (v11 fp32 vs
  fp16 inputs on a sample, report the max |Δprob|).

**D. `model.py`**
- New `VoiceGuardSeqTCN`: stem Conv1d(60→64, k3)+BN+ReLU, then 5 residual
  blocks, each Conv1d(64, k3, dilation 1/2/4/8/16)+BN+ReLU+Dropout(0.1), then
  mean+std+max pooling, concatenated with the normalized scalars, then
  Linear→64+ReLU+Dropout, feeding two heads. RF 65 frames ≈ 1.1 s, about 90k
  params. **Same ONNX I/O contract** as v11 (`lfcc_sequence` (1,184,60),
  `scalars` (1,6) → `real_fake_logits`, `attack_type_logits`), so the Dart side
  is unchanged.
- `build_model_from_norm_stats()` factory. `arch` is stored in norm_stats
  ("seqcnn_v1" for v11, "seqtcn_v2" for v12). An MLP adapter lets v3/v9
  baselines be scored from the cached sequences (pooled LFCC math = features.
  extract_lfcc on the cached frames).

**E. `train_seq_cnn.py`**
- Cache-backed memmap Dataset, AdamW (wd 0.01), 1-epoch warmup then cosine
  decay, 30 epochs, batch 256, EMA weights (0.999). EMA checkpoints are saved
  per epoch.
- Class-weighted real/fake CE, label smoothing 0.05, attack-type loss weight
  0.5.
- **Leave-attack-out for the attack-type head**: A11 (TTS) and A18 (VC) are
  masked from the attack-type loss (kept in real/fake), so attack-type
  generalization to unseen systems can be measured.
- Per-epoch log: val EER (in-distribution, includes channels) and attack-type
  balanced accuracy.

**F. `evaluate.py` (one harness, replaces every one-off eval script)**
- Splits come from a committed manifest (`eval_splits/held_out_split_v1.json`,
  made by `make_eval_splits.py`). Every held dir (real_itw_held,
  fake_itw_held, real_noise_aug_split/held/{en,hi}_native) is split 50/50 by
  file hash into `select` / `test`.
  - `select_best_checkpoint_seqcnn.py` may read **only** `select`.
  - `evaluate.py --split test` is run once per candidate for the final report.
  - MLAAD (all) is an extra unseen-generator fake set in `test`.
- Metrics, per channel and pooled (the phone-pooled headline):
  - EER with a file-level bootstrap CI (windows from all channels of one file
    grouped as one source);
  - FPR/FNR at the threshold fixed on `select` and at the app default 0.60;
  - EER on padded (short-clip) windows only;
  - seen vs unseen channel groups.
- Confound v2, real AND fake side, for pauseRatio, energyVariance,
  zcrVariance, jitter, shimmer, HNR and pad_fraction:
  - half-means, absolute gap, **high/low ratio**, **Spearman ρ**;
  - FPR (reals) and FNR (fakes) per half at the operating threshold.
  - **Gates, fixed before measuring:** for each confound feature, pooled over
    phone channels: |ρ| ≤ 0.10, and the worse/better half ratio of the
    FPR (reals) and of the FNR (fakes) is ≤ 1.25.
- Attack-type: confusion matrix and balanced accuracy on in-distribution val
  fakes, on the leave-attack-out fakes (A11/A18), and on MLAAD (all TTS:
  fraction predicted "tts"). Also measured at the UI's 0.70 confidence gate:
  coverage and accuracy of the labels actually shown.
- Writes `report.json` and `report.md` into the run dir.
- v9_noisefix (MLP), v11_seqcnn and v12 are all re-scored on the same new
  protocol. That comparison is the deliverable.

**G. Retire the broken legacy scripts** (item 11): train.py (MLP),
train_curriculum.py, eval_held_out.py, eval_held_out_dirs.py,
eval_held_out_dirs_seqcnn.py, measure_confound.py, measure_confound_seqcnn.py,
select_best_checkpoint.py (MLP), eval_accent_cells.py (accent cells become
eval sets in the harness). Move `compute_eer` into `eval_stats.py`/`metrics`.
Record in state.md which scripts went where.

**H. Docs**
- `voice_guard/docs/EVAL-PROTOCOL.md` (the policy and protocol);
- a new state.md section, including the correction of the v11 confound table;
- README run instructions;
- a pointer in the root CLAUDE.md voice_guard section;
- `attack_labels.py` docstring (taxonomy verified).

**I. Deploy decision**: ship v12 to `assets/models/voice_detector.onnx` only if
it passes the confound gates and beats v11 on the `test` split under the
channel protocol. The I/O contract is unchanged. Message the UI session
first. If the attack-type head fails (e.g. balanced accuracy on
leave-attack-out below 0.70), recommend hiding the sub-label and coordinate
the `call_screen.dart` change with the UI session.

## Order of work
1. Code A–G with tests (low RAM needed).
2. Build caches: held-out select/test sets × DEFAULT_EVAL_CHANNELS first
   (smaller), then the training corpus.
3. Re-score v9 and v11 on the new protocol (already informative: this is
   items 1–5 answered for the deployed model).
4. Train v12, select on `select`, evaluate on `test`.
5. Docs, state.md, commit, merge to vaani, push, deploy decision.

## Why the reboot
At 21:25 IST available RAM was 0–1.85 GB of 15.4 GB with no large user
process running. `\Memory\Pool Nonpaged Bytes` was **5.9 GB** (normal is <1
GB), a kernel/driver nonpaged-pool leak. Committed was 21.7 GB of 32.2 GB.
Only a reboot clears it. After reboot, re-check with PowerShell:
`Get-Counter '\Memory\Pool Nonpaged Bytes','\Memory\Available MBytes'`.
If the pool climbs back toward multiple GB within hours, it is a driver that
needs identifying (poolmon.exe from the WDK, sorting by nonpaged bytes).
Candidates on this machine: network, NVIDIA or PredatorSense drivers.
