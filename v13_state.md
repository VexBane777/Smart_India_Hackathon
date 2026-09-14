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

## Working set
- Branch: `vaani` (this repo). Latest upstream at session start: `bd2a553`.
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
- [ ] 6. Build playback eval caches -> re-score v9/v11/v12 on `acoustic`
      (long-running; runbook below). ~14 units.
- [x] 7. Train v13: third `playback` rendition for hash-selected ~50% of
      training files; selection objective pooled over phone+acoustic.
- [ ] 8. v13 deploy decision (all four gates); then message UI session before
      copying to `assets/models/voice_detector.onnx`.
- [ ] 9. On-device verification (Test-with-audio-file + Live Mic loop), record
      in state.md.


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

## Known open items / notes
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

