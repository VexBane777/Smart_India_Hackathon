# FLOW.md — How data flows through this repo (`vaani` branch)

> **Heads-up: there is no SQL / Mongo / Postgres "database" in this repo.** The persistent "database" is a set of stores: (1) training corpora on disk (`voice_guard/model_training/data/`), (2) the committed eval-split manifest, (3) the hash-pinned feature cache, (4) training runs/checkpoints + the shipped ONNX asset, (5) on-device prefs/call-logs/WAV dumps, (6) backend monitor log + signaling rooms. This file traces every store: what writes it, what reads it, and the logic between.
> **Maintenance rule:** same as DECISIONS.md — any session that changes a flow below MUST update this file in the same commit. (DECISIONS.md logs *why*; this file logs *how it works now*.)
> Last verified 2026-09-22 at `vaani` tip `3a13b29` (`git pull origin vaani` → Already up to date).

## 0. Map — the six stores at a glance

```
RAW AUDIO corpora (data/real*, data/fake*, data/mlaad*, accents_split/ ...)
   │  make_eval_splits.py splits held-out 50/50 → committed manifest
   ▼
SPLIT MANIFEST (eval_splits/held_out_split_v1.json) — select vs test, frozen
   │  corpus.py training_units() + eval_specs() resolve files × channels
   ▼
TELECHANNEL (vaani/telechannel: RIR→Noise→Mic→Codec→Loss→Bandlimit, real ffmpeg)
   │  dataset.py windows into 3 s units, seeded per file_id (deterministic)
   ▼
FEATURE CACHE (model_training/cache/, hash-pinned, recipe-scoped revalidate)
   │  train_seq_cnn.py trains VoiceGuardSeqCNN/TCN/Conformer + attack head
   ▼
RUNS (runs/voice_guard_v*/) → select_best_checkpoint (select split ONLY)
   → validate_fp16 → evaluate.py on test → eval_playback_loop.py
   → SHIPPED ASSET voice_guard/assets/models/voice_detector.onnx ──┐
                                                                    │
ON-DEVICE (Flutter): mic/WebRTC/file → 16 kHz mono → LFCC 184×60 + 6 scalars
   → ONNX (or heuristic) → EMA + 2-window alert → banner/haptic/logs/WAV dump
   → prefs (SharedPreferences) persist settings; call logs in RAM provider
                                                                    │
BACKEND (FastAPI :8001): /v1/analyze-chunk (features only, never raw audio)
   + /v1/signal/{room} relay (SDP/ICE only) + monitor log (5000-line ring)
```

## 1. Training data — corpora on disk (STORE 1)

- **Where:** `voice_guard/model_training/data/` — `real/`, `real2021/`, `real_itw_train/`, `real_noise_aug_split/{train,held}/{en,hi}_native`, `fake/`, `fake2021/` (capped 40k by stable hash), `fake_itw_train/`, `mlaad_en500/fake/` (test-only, ~20 modern TTS), `accents_split/{train,held}/` (8 cells, test-only). Extensions `.wav`/`.flac` only. **Audio never enters git** (`.gitignore`: `*.wav *.flac *.mp3 *.ogg *.opus *.m4a`).
- **Who fills it:** `prep_*.py` ingestion scripts (ASVspoof2019 LA protocol re-download for attack labels, DECRO spoof-only, MUSAN noise aug, Hindi XTTS clones, `split_accents.py` speaker-disjoint). Attack-type maps: `build_attack_type_maps.py` (fake2021 88% labeled; fake_itw 0% by design — no ground truth).
- **Logic that matters:** fake:real ≈ 4.5:1 uncapped → cap + `compute_pad_policy` per-set pad balancing (short clips differ by label: fake2021 ~68% short vs real2021 ~45%). `hi_native` train capped to random 8000 (was 92k, would skew corpus + take forever).
- **Readers:** `dataset.py::make_file_specs` (explicit file lists only) and `corpus.py::training_units` / `eval_specs`.

## 2. Eval-split manifest (STORE 2 — frozen)

- **File:** `model_training/eval_splits/held_out_split_v1.json`, written once by `make_eval_splits.py` (refuses overwrite; bump version for a new split).
- **Content:** every eval file listed explicitly per set (`itw_real`, `itw_fake`, `noiseaug_real_en/hi` → 50/50 `select`/`test` by stable file-hash; `mlaad_fake` + accent cells test-only). Refuses manifests with train/eval overlap or ITW basename leakage.
- **Logic:** `select` = only data checkpoint selection may read (`select_best_checkpoint_seqcnn.py` fails if it touches test). `test` = read once per candidate (`evaluate.py --split test`). Files added to a directory later NEVER silently enter an eval.
- **Known limit:** select/test split by file, not speaker (ITW `meta.csv` gone); train-vs-held IS speaker-disjoint.

## 3. TeleChannel DSP (the "channel database" — recipes, STORE 3)

- **Where:** `vaani/telechannel/` — `pipeline.py` orchestrates six stages in fixed physical order `RIR → Noise → Mic → Codec (0..N round-trips, real ffmpeg) → Loss → Bandlimit`, recipes from `configs/channels.yaml`.
- **Recipes (groups):** `none` (reference row only, never a result); seen-phone `whatsapp` (Opus 16k), `volte` (AMR-WB 23.85k), `cellular_3g` (AMR-NB 7.4k); unseen-phone `gsm_2g`, `pstn`, `tandem_xnet` (eval only); `playback` (analog loop: far-room RT60 600 ms + pink 12 dB + clip 0.3 + 200–3800 Hz, NO codec/loss); training-only `*_room` (D4, codec THROUGH room); `clean` = DEBUG recipe (rejected by channel policy everywhere).
- **Logic:** training renders each file as `none` + ONE phone channel by file hash + (v13) `playback` for hash-selected ~50% (`stable_unit < playback_fraction`; `room_fraction` adds D4). Eval default = `none` + 6 phone + `playback` (8 units per set); **headlines pool phone channels**. Narrowband-only bandlimit rule (pstn/gsm_2g/cellular_3g/tandem_xnet).
- **Readers/writers:** `dataset.py` calls `process_clip(pcm, recipe, SAMPLE_RATE)` per unit; `feature_cache.py` hashes pipeline+stages+channels.yaml into `feature_version`.

## 4. Windowing + features (dataset.py → features.py)

- **Windowing contract** (identical train/eval; mirrors app's 3 s window / 1 s tick, never pads): trim edge silence (seeded pad) → <1.0 s dropped+counted → 1–3 s padded to 3 s+0.1 s margin with Gaussian noise at clip's own floor BEFORE channel (pad carries channel noise) → ≥3 s: channel on whole clip, then non-overlapping 3 s windows + end-aligned tail if ≥1 s left. Records `pad_fraction` per window (eval measures it as confound; training rebalances via `compute_pad_policy`).
- **Determinism:** every draw seeded from stable hash of (file_id, purpose, channel, seed) — file ids relative to `data/` (junction-safe). Same file → same windows regardless of run composition.
- **Features per window:** `features.py` (MUST match Dart bit-for-bit; parity tests enforce): 1024-pt FFT, hop 256, 513 linear banks (identity), DCT-II orthonormal → per-frame 60-d LFCC sequence (184 frames for 3 s) + 6 scalars `[pauseRatio, energyVar, zcrVar, jitterLocal, shimmerLocal, hnrDb]`. Label: 0 = real, 1 = fake. Attack head: tts/vc, masked (`IGNORE=-100`) for real/unlabeled.

## 5. Feature cache (STORE 4 — hash-pinned)

- **Where:** `model_training/cache/` (one unit per recipe; gitignored, GB-scale). Manifest records file specs + `feature_version` + pad policy.
- **Writers:** `build_caches.py` (Stage A: `v13_stageA_cache.cmd`). **Readers:** `train_seq_cnn.py`, `evaluate.py`, `eval_playback_loop.py` via `CacheCollection.iter_batches`.
- **Invalidation:** version hash covers ALL of `pipeline.py` + `stages/*.py` + parsed `channels.yaml`. Recipe-scoped `revalidate()` rebuilds only affected units (frozen-`FileSpec` fix lets legacy units re-stamp exact pad params). Adding `playback` invalidated everything once — by design.
- **Gate:** `eval_protocol.resolve_channels` raises `CleanOnlyEvalError` on clean-only sets unless `--application bank`; AST test bans literal clean-only lists.

## 6. Training → selection → eval → shipped asset (STORE 5)

```
train_seq_cnn.py --out runs/voice_guard_v13x [flags]   (Stage B)
  → runs/voice_guard_v13x/{model.pt per epoch, norm_stats.npz, metrics}
select_best_checkpoint_seqcnn.py --run ... --metric channel_balanced
  --min-attack-bacc 0.75  (reads SELECT split only)
  → runs/voice_guard_v13x_selected/model.pt (+ model.onnx export)
validate_fp16.py → fp16_validation.json (storage check)
evaluate.py --split test --candidate v13 --reference v11 (+attack-val-run, fp16-report)
  → runs/eval_v13_test/report.md  (phone EER + acoustic EER + confound_table + attack heads)
eval_playback_loop.py --onnx ... --assets test_assets  (clean ≥0.60, loop ≥0.50)
  → deploy: True/False  →  v11 stays unless ALL FOUR pass
```

- **Architectures** (`model.py`, one ONNX I/O contract for all): MLP (retired, baseline-only) → SeqCNN `seqcnn_v1` (v11, RF 5 frames) → SeqTCN `seqtcn_v2` (v12/v13, dilated 1,2,4,8,16,32, RF 65 ≈1.1 s, ~87k params, mean+std+max pool) → Conformer `conformer_v1` (rework axis, same I/O). I/O: `lfcc_sequence (1,184,60)` + `scalars (1,6)` in; `real_fake_logits (1,2)` + `attack_type_logits (1,2)` raw logits out. `FixedNormalize*` buffers bake norm INSIDE graph — callers send raw features.
- **Losses/options:** masked multi-task (attack loss skips real/unlabeled); rework levers all default-OFF (`--model-channels/dilations/hidden/dropout/stochastic-depth/cmvn`, `--focal-gamma`, `--attack-type-warmup-epochs`, `--mixup-alpha`, `--specaugment-*-masks`); OC-Softmax fine-tune (`oc_softmax.py` + `finetune_oc_softmax_v13.py`, `--freeze-backbone` default True, scores rescaled `(1-sim)/2` into same `confound_table`).
- **Current truth (v13, 2026-09-14):** phone 0.3149 PASS, acoustic 0.2755 PASS, attack heads PASS, confound 9/14 FAIL, loop `tts_elevenlabs_sample.wav` 0.248 FAIL → `deploy: False`, v11 (0.4853) stays shipped at `voice_guard/assets/models/voice_detector.onnx` (the ONE `.onnx` excepted from `.gitignore`).

## 7. On-device runtime flow (Flutter app)

```
CAPTURE (one of three, all → AudioService._buffer, RAM only, last 5 s / 80000 samples)
  A) Live mic: CallService EventChannel com.voiceguard/audio_stream → ingestBytes
  B) Protected Call: RemoteAudioTap.kt (native AudioTrack.addSink on remote track)
       → AudioResampler16k (48 kHz stereo → 16 kHz mono, 100 ms/3200-B chunks)
       → MethodChannel → audioService.ingestBytes
  C) File scan: WAV decode (audio_processor_wav_decode) → same buffer
         │
         ▼  every 1 s tick (startScoring): need full 48000 (3 s); skip if _scoring in flight
  SILENCE GATE: chunk RMS < 50/32768 → hasSignal=false, skip (never score zeros)
         │
         ▼  compute(extractFeaturesIsolate) OFF UI THREAD → (184×60 seq, 6 scalars)
  INFERENCE: TFLiteService (name kept; actually flutter_onnxruntime)
       → ONNX session or heuristic fallback (heuristic never claims attack type)
       → (realFakeProb, attackLabel tts/vc/null, attackConf)
         │
         ▼  RiskScoreProvider.update(raw, alertThreshold=settings.sensitivity)
  DECISION (mirrors engine_mock.AlertStateMachine): EMA α=0.7 seed-on-first
       → ≥threshold counts consecutive; state = alert (≥2) / warn (1) / normal (0)
       → Verdict bands on EMA score: <0.30 verified, <0.70 suspicious, else detected
         │
         ▼  ON alert transition: banner (Emerald/Amber/Crimson) + haptic
            + 5 s forensic WAV dump + CallLog entry → LogsScreen
            + overlay/notification if enabled (AudioPipeline)
```

- **Parity rule:** Dart `AudioProcessor` MUST equal Python `features.py` (same 1024/256/513/DCT-II/prosody+physio math); enforced by `audio_processor_*_parity_test.dart` + `dump_*_parity_fixture.py` (LFCC fixture noise-free — numpy vs Dart PRNGs diverge otherwise).
- **Why no raw audio leaves the phone:** payload to backend is LFCC features only; WAV dumps stay in internal cache for `LogsScreen` audit.

## 8. On-device persistence (STORE 6 — prefs, logs, dumps)

| Store | Where | Writes | Reads | Notes |
|---|---|---|---|---|
| Settings | `SharedPreferences` via `SettingsProvider` | `setProtection/Overlay/Sound/Sensitivity/Onboarding/SignalingHost/Port` (+ `load()` at boot) | `AudioPipeline` (protection gate), `ProtectedCall` (host/port), alert path (threshold/overlay/sound) | Defaults: protection on, threshold 0.60, host `10.0.2.2:8001`. Room codes typed manually |
| Risk history | `RiskScoreProvider` RAM (`_history` cap 200, `_callLogs` newest-first) | `update()` per tick, `addCallLog()` on alert/call end | Home/call/logs screens, charts | `reset()` keeps history; `clearHistory()` wipes. NOT persisted across restarts |
| Call state | `CallStateProvider` RAM + 1 s ticker | `setStatus()` from telephony callbacks / WebRTC states | Call UI (elapsed label, status) | `disconnected` auto-resets to idle after 2 s |
| Forensic WAV | internal cache `recordingPath` on `CallLog` | `WavEncoder` 5 s PCM16 dump on alert (`2f069c5`) | `LogsScreen` audit | Cache-only; never git, never uploaded |
| Calibration | `CalibrationProvider` + `AudioService.captureCalibrationSample` | N×3 s windows of user's own voice (mean/std of raw scores) | Per-speaker threshold tuning | Opt-in; `scoreStream` stays `Stream<double>` so calibration type contract holds |
| Attack-type | `attackTypeStream` `(String?, double)` | `TFLiteService.infer` ONNX head | `call_screen.dart` (own confidence gate) | Raw model output, unfiltered; null when heuristic |

## 9. Backend API + signaling + monitor (STORE 7)

- **Base:** FastAPI `voice_guard/backend/main.py` on `:8001` (uvicorn). Pinned: fastapi 0.110.0, uvicorn 0.29.0, pydantic 2.6.0, onnxruntime ≥1.17, httpx (test-only).
- **Two scoring paths, labelled in every response** (`modelBackend` field, never inferred): `onnx` — caller sends full window (`lfcc_sequence` 184×60 + `scalars` 6), scored by `model_backend.py` with the SAME `voice_detector.onnx` the phone runs (load failure is non-fatal: serves heuristic, reports `loaded=false` + reason); `heuristic` — legacy 60-d mean-pooled + 3 prosody through `_heuristic` (kept: SDK contract in README + `api_service.dart` sends that shape). BOTH share the ONE `AlertStateMachine` (EMA + 2-window) imported from `vaani/app/engine_mock.py` — rawScore may differ by scorer, decision policy never does.
- **Endpoints:** `POST /v1/analyze-chunk` (features only — never raw audio, privacy story), `POST /v1/alert` (log+ack; SMS/SIEM mocked), `POST /v1/control` (only mutating endpoint: `reload-model` re-reads ONNX from disk / `clear-monitor`), `GET /v1/model-info` (artifact identity: sha, size, load error), `POST /v1/session/{id}/reset` (clears EMA at call start, logged), `WS /v1/signal/{room_id}` (dumb 2-per-room relay: SDP offer/answer + ICE candidates verbatim, no inspect/persist, full → close 4000), monitor surface below.
- **Monitor log** (`monitor.py`, mounted by BOTH backend and `laptop_monitor/`): `MonitorEvent(seq, ts, mono_ms, stage, severity, message, t, data)` in a 5000-line lock-guarded ring; fan-out via bounded subscriber queues (drop-oldest, never stall pipeline). Serializations from ONE flattened mapping: `GET /monitor.json`, `GET /monitor.txt` (`Monitor:`-prefixed line `call_screen.dart` greps), `GET /logs.ndjson?since=N` (monotonic cursor tail), `GET /events` SSE, `WS /ws/monitor`.
- **Media path (P2P, never via server):** `WebRtcCallService` (STUN `stun.l.google.com:19302` only — fails symmetric NAT; TURN/FCM/ConnectionService are Voip.md phases, not code) — offer/answer/candidates over signaling relay, Opus audio direct, remote PCM tapped natively (§7 path B).

## 10. The `vaani/` doc-suite side (separate pipeline, same policy)

- `vaani/app/` Streamlit demo: `server.py` streams `assets/raw/call_{A,N,B}_raw.wav` (machine TTS placeholders, `provenance: placeholder_tts` — real consented recording NOT done) through `engine_mock.py` (`MockBackend` position-aware: clone-entry time from `call_scripts.json` segments; `AlertStateMachine` EMA 0.7 / 0.6 / 2-windows — the SAME constants `RiskScoreProvider` and backend mirror) → WebSocket UI. `registry.py` merges `configs/*.yaml` over `base.yaml` (deep-merge + `${name}` interpolation; `type` top-level by design, docs differ — intentional).
- Consent/DPDP: `vaani/tests/legal/` gates; signaling relay stores nothing (DPDP-friendly by construction).

## 11. Change protocol (enforced going forward)

1. `git pull origin vaani` before work; note tip hash.
2. Code/config/protocol/dep change → append DECISIONS.md `§ Future Log` (D-0NN, template fields).
3. If §1–§10 above is now wrong → update THIS file in the SAME commit (one-line fix or new subsection; bump `Last verified` line).
4. Never force-push `vaani`; never `git add` across the `vaani/`↔`nyaya` worktree boundary (separate histories).

<!-- APPEND-FLOW -->