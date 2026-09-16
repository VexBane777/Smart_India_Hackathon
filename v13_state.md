# v13 state — living document for the v13 VoiceGuard retrain

> **For newer sessions:** read this file FIRST. It tracks every v13-prep step
> earlier sessions did and any that remain, so you do not re-investigate.
> Update it continuously as you work. Source of truth: the handoff
> `v13_handoff..txt`, `voice_guard/docs/EVAL-PROTOCOL.md`, and
> `voice_guard/docs/superpowers/plans/2026-09-12-post-v12-plan.md`.
>
> Status stamps: `[x] done` / `[~] in progress` / `[ ] todo`.

## Why v13 (tl;dr)
v12 (SeqTCN, `seqtcn_v2`) is the first model above chance on *phone* audio and
beats the deployed v11 on the headline (0.3223 vs 0.4853 EER, CIs don't
overlap), **but** it fails 9 of 14 confound gates — it detects fakes largely by
acoustic texture (zcrVariance ρ=0.544, jitter 3.21, HNR 2.59 on the fake side),
i.e. the entity-vs-style confound, measured on fakes. It is **NOT deployed**.
It also never saw the new `playback` (acoustic loop) channel, so it cannot meet
the new playback deploy gate. **v13 = v12's recipe + a playback rendition +
fake-side confound remediation**, gated by all four criteria.

## v13 RESULT (2026-09-14) — trained, NOT deployed

- Ran end to end via `v13_stageB_train_eval.cmd` (log
  `model_training/runs/v13_stageB_train_eval.log`): 30 epochs ->
  `runs/voice_guard_v13`, selected `epoch_06.pt` -> `runs/voice_guard_v13_selected`,
  fp16 storage validation PASS, report `model_training/runs/eval_v13_test/report.md`.
- **Gate scorecard** (all four criteria, from that report):
  - phone headline **PASS** — EER 0.3149 [0.3050, 0.3250] vs v11 0.4853
    [0.4766, 0.4939], CIs disjoint;
  - acoustic/playback headline **PASS** — 0.2755 [0.2582, 0.2934] vs v11 0.4925
    (the new playback channel worked: 0.4925 -> 0.2755);
  - attack-head gates **PASS** — leave-attack-out balanced acc 0.750 (>= 0.70),
    MLAAD tts share 76.6% (>= 70%);
  - **confound gates FAIL** — 9 of 14 rows (real: zcrVariance/shimmer/hnr_db;
    fake: energyVariance/zcrVariance/jitter/shimmer/hnr_db/pad_fraction;
    fake zcrVariance rate ratio 3.91, jitter 2.72, hnr_db 2.27);
  - **playback-loop gate FAIL** — `tts_elevenlabs_sample.wav` speaker->mic loop
    0.248 < 0.50 (clean 0.642, already near the 0.60 clean gate; the other three
    assets pass).
  - => `deploy: False`. The fake-side **style confound is still the blocker** and
    the acoustic loop still collapses for at least one real TTS sample; v11 stays
    deployed.
- Open items unaffected by this run: unseen-channel generalization
  (gsm_2g 0.4364), accent cells at chance (en_native 0.4879, hi_native 0.5142).

## Working set
- Branch: `vaani` (this repo). v13 protocol commits: `e2d6874` (acoustic
  group, scoped caches, playback rendition, selection objective,
  EVAL-PROTOCOL §2/§6) and `f350840` (`--playback-fraction` provenance +
  acoustic val log; legacy `train_` units re-derive pad params from
  raw-header durations so revalidate can re-stamp them exact), plus
  `407c055` (this session: fixes a `FrozenInstanceError` that made
  `revalidate()` crash on EVERY legacy unit — FileSpec is frozen — so step
  6's `--revalidate-stale` path actually works; adds
  `v13_stageA_cache.cmd` / `v13_stageB_train_eval.cmd`).
- Code dirs: `voice_guard/model_training/` (training/eval code) and
  `vaani/telechannel/` (pipeline/stages). Python venv:
  `voice_guard/model_training/.venv313/Scripts/python.exe`.
- v12 run lives in
  `.worktrees/voiceguard-v12-hardening/voice_guard/model_training/runs/`
  (main checkout's `runs/` is not the v12-run worktree).

## Post-v12 plan -> implementation status
Steps from `2026-09-12-post-v12-plan.md`:

- [x] 1. Let v12 finish + apply decision rule -> **NOT deployed** (confound fail).
- [x] 2. Merge `origin/vaani` into `voiceguard-v12-hardening`, then into `vaani`
      (`7fdfb51`, `bd2a553`). Textual: no conflicts.
- [x] 3. Precise cache invalidation:
  - [x] 3a. recipe-scoped config hash (`feature_version`)
  - [x] 3b. `feature_cache.revalidate(unit)` + tests
- [x] 4. Add `playback` as `acoustic` group to protocol (`eval_protocol.py`),
  evaluate.py acoustic row, **pre-register acoustic deploy gate** in
  `EVAL-PROTOCOL.md` sec 6.
- [x] 5. Bring `eval_playback_loop.py` under the protocol (policy-discovery test).
- [~] 6. Build playback eval caches -> re-score v9/v11/v12 on `acoustic`
      (RUNNING detached since 2026-09-14 ~16:00 IST: `v13_stageA_cache.cmd`,
      log `model_training/runs/v13_stageA_cache.log`; read-only inventory
      first: 130 STALE legacy units (cache_format 1 + global hash, re-stamped
      in place by `--revalidate-stale`, proven equivalent max_abs_diff 0 on a
      trial unit) + 22 MISSING playback units — 14 eval + 8 train — 0
      non-playback MISSING; 98 orphaned old-key dirs remain on disk,
      harmless). v13_stageA also builds the 8 train playback units.
- [x] 7. Train v13: third `playback` rendition for hash-selected ~50% of
      training files; selection objective pooled over phone+acoustic.
      (staged: `v13_stageB_train_eval.cmd` waits on stage A, then
      train -> select -> fp16 validate -> evaluate (all four gates) ->
      playback-loop ONNX gate.)
- [x] 8. v13 deploy decision -> **NOT deployed** (`deploy: False`, 2026-09-14):
      confound gates fail 9/14 and the playback-loop gate fails on
      `tts_elevenlabs_sample.wav` (0.248 < 0.50). v11 stays deployed.
- [ ] 9. On-device verification (Test-with-audio-file + Live Mic loop), record
      in state.md. (Deferred: nothing new ships until a candidate clears all four
      gates.)
- [~] 10. **Post-v13 rework** (plan
      `model_training/docs/2026-09-training-improvement-plan.md`, second pass
      2026-09-16): axes B/G (capacity + stochastic depth/CMVN/SE, all persisted in
      `norm_stats.npz`), F/D3 (focal loss, attack-head warmup, Mixup,
      SpecAugment), D1/H (channel-balanced selection objective + attack-head
      floor), plus the earlier min-LR hold/grad clip/stability tolerance. The
      worktree was found mid-edit and syntactically broken and was repaired
      first. **Remaining:** actually launch the rework candidate
      (`runs/voice_guard_v13x`, §9 of the plan) and score it against all four
      gates; then multi-seed (H), data work (D2/D4/D5), calibration/ensembling
      (I/K), and the dilation-32/Conformer comparison (C/4).


## Run (v13) — runbook
```bash
cd voice_guard/model_training
PY=.venv313/Scripts/python.exe
# 1) eval caches, incl. the NEW playback units. --revalidate-stale re-stamps
#    unchanged none/phone units (measure, don't assume) and builds only what
#    is truly new/changed. Default --channels now = none+phone+acoustic (8 per set).
$PY build_caches.py --eval --revalidate-stale
# 2) re-score v9/v11/v12 on acoustic (step 6) -> runs/eval_acoustic_reference
# 3) training caches incl. the NEW playback rendition (~half a train build):
$PY build_caches.py --train --revalidate-stale
# 4) train v13:
$PY train_seq_cnn.py --out runs/voice_guard_v13
# 5) select on select (phone+acoustic pooled):
$PY select_best_checkpoint_seqcnn.py --run runs/voice_guard_v13 --out runs/voice_guard_v13_selected
# 6) fp16 validate + evaluate (all four gates):
$PY validate_fp16.py --model runs/voice_guard_v13_selected/model.pt --out runs/fp16_validation_v13.json
$PY evaluate.py --split test --out runs/eval_v13_test --candidate v13 --reference v11 --attack-val-run runs/voice_guard_v13 --fp16-report runs/fp16_validation_v13.json --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt --model v13=runs/voice_guard_v13_selected/model.pt
# 7) acoustic loop gate on the shipped ONNX:
$PY eval_playback_loop.py --onnx runs/voice_guard_v13_selected/model.onnx --assets test_assets --gate-clean-min 0.60 --gate-loop-min 0.50
```
v13 ships only if it clears **all four**: (1) every confound gate,
(2) beats reference test headline EER, (3) beats reference acoustic EER,
(4) both `eval_playback_loop.py` gates.

## Run (post-v13 rework candidate) — runbook

Full command block, per-axis rationale and the expected runtime cost live in
`model_training/docs/2026-09-training-improvement-plan.md` §9. The short form
(every lever is independent and default-OFF):

```bash
cd voice_guard/model_training
PY=.venv313/Scripts/python.exe
$PY train_seq_cnn.py --out runs/voice_guard_v13x \
    --epochs 40 --min-lr-epochs 10 --grad-clip 1.0 \
    --model-channels 96 --model-dilations 1,2,4,8,16,32 --model-hidden 96 \
    --model-dropout 0.15 --model-stochastic-depth 0.1 --model-cmvn \
    --focal-gamma 1.5 --attack-type-warmup-epochs 3 --mixup-alpha 0.2 \
    --specaugment-freq-masks 2 --specaugment-time-masks 2
$PY select_best_checkpoint_seqcnn.py --run runs/voice_guard_v13x \
    --out runs/voice_guard_v13x_selected --stability-tolerance 0.005 \
    --metric channel_balanced --min-attack-bacc 0.75
# then the same four gates as v13: validate_fp16.py -> evaluate.py --split test
# -> eval_playback_loop.py (see the v13 runbook above; swap in the v13x paths).
```

## Known open items / notes

- **Test status this session (post-v13 rework, 2026-09-16):** 48 green on
  `eval_protocol`/`feature_cache`/`eval_stats`/`attack_labels`; 29 passed + 1 skipped
  on `model`+`select_best_checkpoint_seqcnn` (the skip is the retired v9 checkpoint
  that is no longer on disk; **v12 and v13 checkpoints both load**, so the new
  default model kwargs are backward compatible with them); 35 on
  `evaluate`/`eval_stats`/`eval_protocol`; 16 on `train_seq_cnn`, including a new
  end-to-end run that exercises every new flag and re-loads its checkpoint through
  `norm_stats.npz`. Use `model_training/_rework_tests.cmd` (detached) for the whole
  set, and `model_training/runs/_rework_tests.log` for the result: this shell caps
  a command at 30 s and the trainer tests spawn real trainings.
- **Test status this session (v13 protocol work):** 48 green on the fast files
  (`eval_protocol`/`feature_cache`/`eval_stats`/`attack_labels`), 21 on
  `evaluate`/`model`/`train_seq_cnn`, then `evaluate`+`features` (12),
  `features_sequence`+`dataset_sequence`+`windowing` (25), `pipeline_smoke` (2),
  `dataset_integrity` (18) — all green. Full-suite run timed out in-terminal;
  moving to chunked/offline validation like above, and recording counts here.
- The fake-side confound remediation is the hard part (the plan requires fixing
  it, not just adding channels). The 60->66-d physio widening (jitter/shimmer/HNR)
  is already in and did NOT fix it (v10_physio failed). Track it separately from
  the acoustic channel work; see `docs/CRITICAL-entity-vs-style-confound.md`.
- Unseen-channel generalization (gsm_2g 0.4469) and accent cells (chance) remain
  open, not v13-blocking.
- On-device ONNX must pass the deterministic "Test with audio file" path and the
  Live Mic loop through a real speaker before shipping.

