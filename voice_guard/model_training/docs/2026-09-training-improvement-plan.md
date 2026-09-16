# VoiceGuard training — improvement plan (refine before implementation)

> v13 (seqtcn_v2, 30 epochs) trained; v13/v12 **not deployed** (confound + playback-loop gates fail).
> Every suggestion below must clear the deploy gates in `evaluate.py` / `eval_playback_loop.py`,
> **not** just lower headline EER. All paths preserve the ONNX I/O contract.

## 1. Findings from the existing runs (why "more epochs" isn't the lever)

- **Already converged & mild overfit.** v13 `runs/voice_guard_v13/history.json`:
  val_eer `0.143(e6) -> 0.121(e22) -> 0.122(e30)`; train_loss still falling `0.378 -> 0.344`.
  v12 `runs/voice_guard_v12/history.json`: val_eer flat ~0.104 from e20, train loss `0.329 -> 0.296`.
  => mild overfitting, **not** underfitting.
- **LR already at `min_lr=1e-5` by epoch 30.** Cosine `total_steps = steps_per_epoch * --epochs`
  (`train_seq_cnn.py:237`). Bumping `--epochs` *re-stretches* the schedule (LR stays higher longer
  at what was epoch 30) => more overfit, not finer convergence.
- **Selection is the noisy part.** `runs/voice_guard_v13_selected/checkpoint_sweep.json`: pooled-EER
  0.3160 - 0.3200 over epochs 6-30 (flat); **argmin = epoch 6 (0.3160), not the training-best
  epoch 22 (0.3164)** — the "best" is a chance dip. More checkpoints => more chance minima =>
  selection-overfit to the `select` split.
- **Channels dominate the metric.** Worst channels (gsm_2g ~0.45, tandem_xnet ~0.40) swamp
  pooled EER while `none` ~0.06; selection chases noise channels, not generalization.

## 2. Your three axes, refined (with the *paired* levers that make them work)

### B. Increase model size — and make the extra capacity *generalize*

`model.py::VoiceGuardSeqTCN.__init__`: `channels=64, dilations=(1,2,4,8,16), dropout=0.1,
hidden=64` (~87k params; v13 train_config.json `n_params: 87108`).

- Concrete widening paths: `channels 64->96/128`; `dilations +32` (RF 65->129 frames ~= 1.1s->2.2s);
  `hidden 64->128`; scale the trunk/heads with `channels`.
- **Paired regularizers (required, else bigger = more overfit):** spatial dropout on conv outputs;
  **stochastic depth** in `_ResidualBlock`; per-utterance CMVN inside the graph (see F); stronger
  wd (0.01->0.03) + **label-smoothing tune (0.05->0.1)**.
- **Attention blocks:** SE (squeeze-excitation) per residual block; **learnable (attention)
  pooling** instead of mean/std/max. All ONNX-contract-compatible (internal ops only).
- **Trade-off**: don't widen without also widening regularization and validating the *gap*
  (val vs test EER), not just val-EER.
- **Constraint:** ONNX I/O fixed — `lfcc_sequence (1,184,60) + scalars (1,6) ->
    real_fake_logits (1,2) + attack_type_logits (1,2)` (`README.md` Feature contract, `model.py`
  constants). Do not change n_frames/n_lfcc/n_scalars or the I/O names.

### C. Different neural network — minimal-risk steps before a full swap

- `seqcnn_v1` (Arm B, `launch_armB_seqcnn.cmd`): RF=5 frames (~130 ms) — the model.py docstring
  calls this "too little temporal context." Keep as a **baseline/comparison**, not a replacement.
- **Safe first step:** dilation-32 TCN (same contract, +RF) — measure against `seqcnn_v1` and
  `seqtcn_v2`.
- **Next step:** **Conformer / hybrid Conv1d + self-attention encoder** over the 184x60 sequence.
  Risk: more params/overfit; must keep the two-head output and export to the fixed ONNX contract
  (add a parity test vs `features.py` <-> `audio_processor.dart`).
- **Ensemble option:** logits-average `seqtcn_v2 + seqcnn_v1` (or top-k epochs) for selection —
  cheap way to gain robustness without a new arch.

### D. Reduce / refine the dataset to cut overfit — "refine," don't shrink

298k windows is already large; the gap isn't data *quantity*, it's **channel/quality imbalance
and augmentation poverty.**

- **D1 — Selection metric de-bias:** select on a channel-balanced objective (e.g. mean of
  per-channel EER, or phone+reference channels only) instead of pooled EER dominated by
  gsm_2g/tandem_xnet. (Pairs with B/C — a bigger model needs a non-noisy argmin.)
- **D2 — Data quality audit:** `check_corpus.py` / `dataset_audit.py` exist. Add
  **near-duplicate dedup across train/select/test** (by audio hash or speaker x phrase) — v11
  famously selected its epoch on `fake_itw_held` which was *also* the test-half (leakage the new
  protocol was built to prevent).
- **D3 — Augmentation beyond channels:** today augmentation = TeleChannel recipes only. Add
  **SpecAugment** (time + freq masking on LFCC, training-only), **Mixup** (interpolate sequence +
  labels — very effective for spoofing), and **speed/tempo perturbation** (training-only). These
  diversify without shrinking data.
- **D4 — Room/mic realism:** the confound analysis (`docs/CRITICAL-entity-vs-style-confound.md`)
  shows room reverb + mic coloration flatten the prosody/LFCC cues. Expand the
  `playback`-style far-room + mic simulation to **more training channels** (not just the
  half-fraction playback), not just codecs.
- **D5 — Sampling:** channel-balanced / stratified batching (today it's plain index permutation
  `train_seq_cnn.py:267`) so each batch sees a spread of channels; tune `no_pad_balance` /
  pad balancing in `dataset.py`.
- **D6 — Generator diversity:** add more TTS/vocoder engines to training (MLAAD already covers
     unseen generators at eval; train-side diversity reduces the entity-vs-style confound).

## 3. New axes (compounds the above)

**F. Loss & optimization (addresses overfit without more epochs)**
- Add **gradient clipping** — currently absent: `train_seq_cnn.py:285-288` loop is
  `zero_grad -> backward -> ema.update -> step`, optimizer `train_seq_cnn.py:231` has no clip,
  and `train_config.json` has no `clip` field. Start at 1.0; stabilizes larger-capacity /
  larger-batch runs.
- Try **focal loss** (gamma ~= 2) in place of (or with) class-weighted CE — standard in
  spoofing, down-weights easy negatives.
- **Warm the attack-type head** (start weight 0 -> 0.5) so the rf head isn't perturbed by the
  auxiliary task early (`train_seq_cnn.py:283-284`).
- **LR-peak tuning / LR finder** instead of fixed 0.002; optionally a **cosine-then-hold**
  (`--min_lr_epochs`) so *if* epochs are extended, they run at min_lr (real polish) rather than
  re-stretching the schedule.
- (AdamW + lookahead is marginal; skip.)

**G. Regularization paired with capacity** *(the "make it bigger safely" guardrails):*
spatial dropout, stochastic depth, per-utterance CMVN in-graph, label-smoothing tune, wd tune,
Mixup, SpecAugment.

**H. Selection robustness (highest leverage, lowest cost)**
- **Multi-seed** (several `--seed`, currently only `0`) + pick/ensemble — attacks run-to-run
  variance directly.
- **Stability selection**: earliest epoch within 1 bootstrap-CI of the min, or **top-k average** —
  cut chance-minima bias. (`eval_stats.py` has `bootstrap_eer_ci`.)
- **Multi-objective selection**: lowest EER subject to `val_attack_bacc >= k` — don't pick an
  EER-min checkpoint with a weak attack head.

**I. Calibration / operating point**
- Model-side **temperature scaling** on the `select` split (tunes the 0.60 app-threshold gap;
  the existing `CalibrationProvider` is Dart-side per-speaker, not this).
- Gate-aware threshold: optimize threshold on `select` for rate-balance, then apply to `test`.

**K. Ensembling (cheap robustness)**
- Top-k epoch-average; multi-arch average; multi-seed average. Validate under the same gates.

## 4. Suggested sequencing / priority

1. **(cheap, high-leverage)** Multi-seed (3-5 seeds, 30 ep) + stability selector -> proves whether
   "best each run" is seed-variance-bound.
2. **(medium)** Add `--min_lr_epochs` (cosine-then-hold) + channel-balanced selection metric ->
   *then* re-test "do more epochs help?" with the schedule decoupled.
3. **(medium-high)** Grad clip (1.0) + focal loss + SpecAugment/Mixup + per-utterance CMVN;
   paired with the capacity bump (channels 64->96, +dilation 32) **and** stronger reg.
4. **(invest)** dilation-32 TCN, then Conformer/hybrid behind the fixed ONNX contract + parity test.
5. **Avoid:** `--epochs` alone; raw argmin over more checkpoints.

## 5. Risks / non-negotiables

- CPU-only (16 GB box, ~16-48 s/epoch); capacity x seeds x epochs compounds runtime (memmap data
  helps but CPU is the constraint).
- ONNX export parity (`features.py` <-> `audio_processor.dart`) — gate it; `export_onnx` uses
  opset 13, dynamo=False (`model.py:222`).
- `playback` only half-folded in v13 (0.5 fraction, `train_config.json`); full fold is a recipe
  change.
- Any improvement scored against the **gates** (confound v2, acoustic, playback-loop, attack-head
  MLAAD) — not just EER.

## 6. Where the changes land (file map)

- `train_seq_cnn.py`: argparse `--min_lr_epochs`; grad clip; focal-loss flag; attack-head warmup;
  document that `--epochs` re-stretches the schedule.
- `model.py`: width/dilation/pooling kwargs (already parametrized) + optional SE / attention-pool
  + in-graph per-utterance CMVN.
- `train_seq_cnn.py` loop: SpecAugment/Mixup (training-only branches).
- `select_best_checkpoint_seqcnn.py`: channel-balanced objective + top-k / earliest-within-CI
  selection.
- `eval_stats.py`: multi-objective + bootstrap-CI helpers (mostly present).
- `docs/CRITICAL-entity-vs-style-confound.md`: reference for the room/mic + prosody confound.

## 7. Open refinement questions

1. Priority — agree with the "multi-seed -> schedule-hold -> capacity+reg" ordering, or lead with a
   capacity/architecture experiment?
2. Scope implementation of anything now (e.g. the cosine-then-hold + grad-clip patch), or keep this
   purely as the planning doc?
3. File location/name — `model_training/docs/2026-09-training-improvement-plan.md` (chosen).

## 8. Implemented in this pass (act-mode follow-up)

Two backward-compatible changes that directly address the original question ("would
increasing epochs give a more precise best each run?"):

**A. `train_seq_cnn.py` — cosine-then-hold + grad clip**
- Added `--min-lr-epochs` (float, default 0.0). The cosine `total_steps` is now
  `steps_per_epoch * (epochs - min_lr_epochs)`; the trailing `min_lr_epochs` run at
  `min_lr` via `lr_at`'s existing progress clamp. Growing `--epochs` no longer re-stretches
  the cosine *if* you also set `--min-lr-epochs`; without it, behavior is **identical** to
  before (default 0.0). Rationale: "more epochs" is only meaningful as low-LR polish when the
  schedule endpoint is held fixed.
- Added `--grad-clip` (float, default None = no clip; recommend 1.0 for bigger/louder runs).
  Absent before (`train_seq_cnn.py` loop had zero_grad→backward→step with no clip).
- Both flags flow into `train_config.json` automatically (argparse vars) for provenance.
- Tests: `test_lr_holds_at_min_past_cosine_endpoint` (lr_at clamps past endpoint) +
  `test_train_smoke_with_min_lr_hold_and_grad_clip` (3-epoch real run with both flags,
  asserts `history.json`/`train_config.json` + ONNX export shapes). All 8 trainer tests pass.

**B. `select_best_checkpoint_seqcnn.py` — stability selection**
- Added `--stability-tolerance` (float, default 0.0 = raw argmin, unchanged). When >0, the
  selector picks the **earliest** epoch whose pooled_eer is within `tolerance` of the argmin,
  instead of the raw argmin — directly attacks the "epoch 6 vs epoch 22 chance-dip" noise.
  Refactored into a pure, testable `select_best(sweep, tolerance)` (+ `_epoch_of`).
- `selection_mode` + `stability_tolerance` now written into `checkpoint_sweep.json`.
- Tests: `test_epoch_of_parses_names`, `test_select_best_default_is_raw_argmin`,
  `test_select_best_stability_picks_earliest_within_tolerance`,
  `test_select_best_too_small_tolerance_returns_argmin`. The AST test
  `test_checkpoint_selection_never_reads_the_test_split` still passes (selector still reads
  only `SPLIT = "select"`).

**How to use now:**
```bash
# Meaningful "more epochs": 30 cosine + 10 epochs of low-LR hold (instead of re-stretching).
python train_seq_cnn.py --out runs/voice_guard_v13x --cache-root "$VG_CACHE" \
    --epochs 40 --min-lr-epochs 10 --grad-clip 1.0
# Pick a stability-selected checkpoint, not the chance-dip argmin.
python select_best_checkpoint_seqcnn.py --run runs/voice_guard_v13x \
    --out runs/voice_guard_v13x_selected --stability-tolerance 0.005
```

**Test results:** 13 passed (8 trainer incl. new; 1 selector split-leak guard; 4 new selector unit tests).

## 9. Implemented in the second pass (2026-09-16) — capacity/reg + loss + selection

State on entry: the §8 pass left this worktree mid-edit and **syntactically broken**
(`IndentationError: unexpected indent` in `train_seq_cnn.py` and
`test_model_seq_cnn.py`, so `pytest` could not even collect), and the §B capacity
wiring was half-done. Repair first, then complete the axes.

**0. Repairs (blocking everything).**
- `train_seq_cnn.py`: two over-indented statements (`stats = {...}` in
  `compute_norm_stats`, and the `norm_stats = compute_norm_stats(...)` call site).
  The capacity keys were also not persisted for `stochastic_depth`/`cmvn`/`se`.
- `test_model_seq_cnn.py`: over-indented `ns = _norm(...)`; and the expected
  receptive field for dilations `(1,2,4,8,16,32)` was wrong (97 -> 129, i.e.
  `1 + 2*(1 + sum(dilations))`).

**1. Axis B/G — capacity + paired regularization, persisted in `norm_stats.npz`.**
- `model.py`: `_ResidualBlock` takes `stochastic_depth`/`se`; new
  `_SqueezeExcite` (1x1-Conv1d SE, ONNX-clean) and `per_utterance_cmvn()`;
  `VoiceGuardSeqTCN` gains `stochastic_depth`, `cmvn`, `se` and a
  `normalize_sequence()` helper. Stochastic depth uses the dense/linear rule
  (`p_l = p * (l+1) / L`) and is **train-mode only**, so an exported graph takes
  the same path as before.
- `build_model_from_norm_stats` reads all seven keys back (absent = v12/v13
  defaults), so select / evaluate / validate_fp16 / the app all rebuild exactly
  the trained architecture. The ONNX I/O contract is untouched.
- `train_seq_cnn.py`: `--model-channels`, `--model-dilations`, `--model-hidden`,
  `--model-dropout` (from the interrupted pass) plus `--model-stochastic-depth`,
  `--model-cmvn`, `--model-se`; all flow into `train_config.json` for provenance.

**2. Axis F/D3 — loss + augmentation levers (all default-OFF).**
- `--focal-gamma`: `focal_cross_entropy()`. gamma=0 calls
  `nn.CrossEntropyLoss(weight, label_smoothing)` verbatim, so the default loss is
  bit-identical to v12/v13; gamma>0 applies `(1 - p_t)^gamma` with PyTorch's
  weighted-mean normalization.
- `--attack-type-warmup-epochs`: `attack_type_weight()` ramps the auxiliary head
  0 -> `--attack-type-loss-weight` (recorded per epoch as `attack_weight` in
  `history.json`).
- `--mixup-alpha`: `mixup_batch()` + `soft_target_cross_entropy()`. Soft real/fake
  targets; the attack head mixes one-hot labels only where **both** partners carry
  a real attack label (mixing `IGNORE_ATTACK_TYPE` is undefined).
- `--specaugment-freq-masks/-freq-width/-time-masks/-time-width`: `specaugment()`
  on the raw LFCC batch before normalization.
- Augmentation draws from its own RNG stream (`seed + 1`), so batch order — and
  therefore every legacy run — is unchanged.

**3. Axis D1/H — selection robustness (`select_best_checkpoint_seqcnn.py`).**
- `--metric {pooled,channel_balanced}`: `pooled` is the pre-registered v13
  objective (unchanged default); `channel_balanced` averages the per-channel EERs
  so gsm_2g/tandem_xnet stop dominating the argmin.
- `--min-attack-bacc`: multi-objective floor read per epoch from the run's
  `history.json` (`attack_bacc_from_history`), so an EER-min checkpoint with a weak
  attack head cannot win. A floor nothing clears is dropped with a warning.
- The stability tolerance from §8 is kept and composes with the floor; the
  objective treats NaN as +inf, so rows with no usable objective are never picked.
- Still reads only `SPLIT = "select"` (the split-leak guard test still passes).

**How to run the rework candidate** (v13's recipe + these levers; every flag below
is independent — pick the ones under test):
```bash
cd voice_guard/model_training
PY=.venv313/Scripts/python.exe
$PY train_seq_cnn.py --out runs/voice_guard_v13x --cache-root "$VG_CACHE" \
    --epochs 40 --min-lr-epochs 10 --grad-clip 1.0 \
    --model-channels 96 --model-dilations 1,2,4,8,16,32 --model-hidden 96 \
    --model-dropout 0.15 --model-stochastic-depth 0.1 --model-cmvn \
    --focal-gamma 1.5 --attack-type-warmup-epochs 3 --mixup-alpha 0.2 \
    --specaugment-freq-masks 2 --specaugment-time-masks 2
$PY select_best_checkpoint_seqcnn.py --run runs/voice_guard_v13x \
    --out runs/voice_guard_v13x_selected --stability-tolerance 0.005 \
    --metric channel_balanced --min-attack-bacc 0.75
$PY validate_fp16.py --model runs/voice_guard_v13x_selected/model.pt --out runs/fp16_validation_v13x.json
$PY evaluate.py --split test --out runs/eval_v13x_test --candidate v13x --reference v11 \
    --attack-val-run runs/voice_guard_v13x --fp16-report runs/fp16_validation_v13x.json \
    --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt \
    --model v13x=runs/voice_guard_v13x_selected/model.pt
$PY eval_playback_loop.py --onnx runs/voice_guard_v13x_selected/model.onnx --assets test_assets \
    --gate-clean-min 0.60 --gate-loop-min 0.50
```
`--model-channels 96` + dilation 32 is ~2.4x v13's params; on the 16 GB CPU box
that is a real runtime cost, so the capacities in that command are the upper end
of the sequencing, not a default.

**Test results (2026-09-16):** 48 on
`eval_protocol`/`feature_cache`/`eval_stats`/`attack_labels`; 29 passed + 1 skipped
on `model`+`selector` (the skip is the retired `voice_guard_v9_noisefix_final`
checkpoint, absent on disk — **v12 and v13 checkpoints both load**, which is the
backward-compatibility proof for the new default kwargs); 35 on
`evaluate`+`eval_stats`+`eval_protocol`; 16 on `train_seq_cnn` (13 unit +
`clean_only` + `end_to_end_smoke` + `min_lr_hold` + a new `rework_levers` run that
exercises every new flag end to end and re-loads the checkpoint through
`norm_stats`). `_rework_tests.cmd` runs the whole set detached, because the Cline
shell caps a command at 30 s and the trainer tests spawn real trainings.

**Still open after this pass (none of it blocks the rework run):**
multi-seed runner + ensembling (H/K), channel-stratified batching (D5),
near-duplicate dedup across train/select/test (D2), room/mic augmentation on more
channels (D4), temperature scaling / gate-aware threshold (I), dilation-32 vs
Conformer comparison (C/4), and the `--playback-fraction` full fold (0.5 -> 1.0).

