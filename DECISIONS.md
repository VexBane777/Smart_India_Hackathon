# DECISIONS.md — Living Log of Meaningful AI Code Decisions (`vaani` branch)

> **Rule for every AI session (human-enforced):** (1) `git pull origin vaani`
> before work and note the tip hash. (2) Any session that changes code,
> configs, protocols, or deps MUST append a new entry to `§ Future Log`
> below using the template in `§ How To Log`. (3) If the pull brought new
> commits, or your own changes alter a flow, update `FLOW.md` (§1–§10 +
> `Last verified` line) in the SAME commit. No silent tradeoffs. State:
> *context → options considered → decision → why this over that → tradeoff
> accepted → evidence (commit/test/doc)*.
> Backfilled on 2026-09-22 from `git log --all` (213 commits), `CLAUDE.md`, `voice_guard/state.md`, `v13_state.md`, `Voip.md`, `EVAL-PROTOCOL.md`, and code comments. `git pull origin vaani` verified `Already up to date` on `3a13b29` — this file starts from that tip.

## How To Log (template — copy/paste)

```markdown
### D-NNN — <short title> — YYYY-MM-DD — <area: app|model|audio|eval|backend|devops>
- **Context:** <what problem forced a choice>
- **Options considered:** A) ... B) ... C) ...
- **Decision:** <what was done>
- **Why this over that:** <library/pattern rationale, with mechanism not vibes>
- **Tradeoff accepted:** <what got worse, and why that's OK / how it's mitigated>
- **Evidence:** <commit hash + test/doc that proves it, e.g. `4c9961d` + `test_oc_softmax.py`>
```

Rules: one entry per meaningful choice (library swap, pattern, gate, threshold, protocol rule). Trivial refactors don't log. If you change a past decision, add a NEW entry that links `Supersedes: D-XXX` — never rewrite history.

## D-000 — This file exists — 2026-09-22 — devops
- **Context:** User asked: pull `vaani`, self-update, then make AI log every meaningful decision + reason (why library/pattern/tradeoff) going forward.
- **Options considered:** A) Rely on `git log` bodies alone B) New `DECISIONS.md` living log at repo root C) Per-folder logs (`vaani/`, `voice_guard/`).
- **Decision:** Single repo-root `DECISIONS.md` with enforcement header + template + backfill from history.
- **Why this over that:** Git bodies are scattered across 213 commits and miss *rejected* options; per-folder logs split cross-cutting decisions (e.g. ONNX contract touches `model_training/` + `lib/` + `backend/`). One root file is discoverable from `CLAUDE.md` workflow.
- **Tradeoff accepted:** Backfill is reconstruction (no time machine); marked as such. Future entries will be first-hand. Mitigation: every backfilled entry cites commit/doc/test so it can be checked.
- **Evidence:** `git fetch --all --prune; git pull origin vaani` → `Already up to date`; branch `vaani` tip `3a13b29`; this file (new, untracked until committed).

## Backfill — Eval protocol & deploy gates (pre-registered, code-enforced)

### D-001 — No clean-only eval/training unless `--application bank` — 2026-09-11 — eval
- **Context:** v11 was trained `--channel none` and evaled on clean; headline EER 0.0624 said nothing about phone calls, the actual product.
- **Options considered:** A) Keep clean evals as headline B) Phone-only headline with clean as reference row C) Dual headlines.
- **Decision:** `eval_protocol.py::resolve_channels` raises `CleanOnlyEvalError` on clean-only sets unless bank use; `test_eval_protocol.py` AST-scans scripts for literal clean-only lists; `clean` recipe rejected (debug recipe, not undegraded — undegraded is `none`).
- **Why this over that:** A headline must predict deployment. Clean audio flatters texture-keyed models and hid the entity-vs-style confound until on-device loops collapsed (0.95→0.01). One rule in code+tests beats documentation pleading.
- **Tradeoff accepted:** Bank use-case (clean) needs an explicit flag; slightly more CLI friction. Mitigation: `--application bank` escape hatch, documented.
- **Evidence:** `EVAL-PROTOCOL.md §1`; `eval_protocol.py` + `test_eval_protocol.py`; commits `bcda9ac`, `e2d6874`.

### D-002 — Committed select/test split; selection reads `select` only — 2026-09-11 — eval
- **Context:** v11 selected checkpoint on `fake_itw_held`, also its test set — selection overfit to test.
- **Options considered:** A) Single held-out set B) 50/50 `select`/`test` by stable file-hash, selection gated to `select`, `test` read once per candidate C) K-fold (expensive).
- **Decision:** B, with explicit manifest `held_out_split_v1.json` (refuses overwrite; version bump for new split); `select_best_checkpoint_seqcnn.py` fails if it references test; `make_eval_splits.py` refuses train/eval overlap.
- **Why this over that:** Prevents the exact v11 mistake structurally, not socially. K-fold rejected: corpus is GB-scale with ffmpeg round-trips per file; 1 split × 8 channel units is already heavy.
- **Tradeoff accepted:** File-hash split, not speaker split (ITW `meta.csv` gone from machine/HF mirror empty) — some speaker leakage inside select/test. Mitigation: train-vs-held IS speaker-disjoint (`prep_in_the_wild.py`); limitation documented in protocol §3, not hidden.
- **Evidence:** `EVAL-PROTOCOL.md §3`; `086f981` select-split baseline; `d09f6ac` channel-balanced selection (`--min-attack-bacc`, `--metric channel_balanced`).

### D-003 — Confound gates (|ρ|≤0.10 + rate ratios) AND headline must pass; v12/v13 NOT deployed on gate fail — 2026-09-12/14 — eval
- **Context:** v12 beat v11 headline decisively (0.3223 vs 0.4853 phone EER, disjoint CIs) yet keyed to acoustic texture (fake zcrVariance ρ=0.544, rate 5.13; jitter 3.21; HNR 2.59) — the entity-vs-style confound.
- **Options considered:** A) Ship on headline alone (v4/v5-style regression risk) B) Pre-registered conjunctive gate: all confound rows + phone headline + acoustic EER + playback-loop gates C) Ship with warning label.
- **Decision:** B. `evaluate.py::confound_table` is the single harness; `deploy: False` recorded for v12 (`3133c04`) and v13 (`runs/eval_v13_test/report.md` via `v13_state.md`).
- **Why this over that:** Policy was fixed in `EVAL-PROTOCOL §6` BEFORE any model was scored — avoids post-hoc rationalization. Shipping a style-keyed detector false-flags monotone humans and misses natural fakes: worse than chance for trust.
- **Tradeoff accepted:** Leaves v11 (0.4853, ~chance) deployed while better-headline models sit out; looks bad on slides. Mitigation: honest `state.md`/report trail; remediation tracks (specialist heads, OC-Softmax, conformer) instead of silent ship.
- **Evidence:** `3133c04`; `v13_handoff..txt`; `v13_state.md` gate scorecard; `d4ac4cf` full confound-rate gate + bootstrap significance.

### D-004 — Playback acoustic channel + loop gates added after on-device collapse — 2026-09-12 — eval/audio
- **Context:** Fakes scoring 0.95–0.99 direct dropped to ~0.01–0.30 through speaker→mic loop (level ruled out: −24 dB only 0.986→0.924). v12 never saw playback in training (only `none` + hashed phone).
- **Options considered:** A) Treat loop as out-of-scope B) Add `playback` TeleChannel recipe (far-room RT60 600 ms, pink 12 dB, clip 0.3, 200–3800 Hz) + hash-selected ~50% third rendition in training + `eval_playback_loop.py` gates (fake clean ≥0.60, loop ≥0.50) C) Only eval playback, never train on it.
- **Decision:** B — protocol §2/§6, `e2d6874` (acoustic group, recipe-scoped caches, playback rendition) + `f350840` (provenance, acoustic val log) + `b927c57` recipe.
- **Why this over that:** The failure was representational (reverb/mic coloration flatten cues), not gain — only exposure + a gate fixes it. Eval-only would guarantee another `deploy: False`.
- **Tradeoff accepted:** +~50% training renders (ffmpeg cost) + cache invalidation of everything (feature_version hashes pipeline+stages+channels.yaml). Mitigation: recipe-scoped hashing + `revalidate()` (frozen-`FileSpec` fix `407c055`) so only affected units rebuild; `v13_stageA_cache.cmd`/`v13_stageB_train_eval.cmd` runbooks.
- **Evidence:** `e2d6874`, `f350840`, `b927c57`, `bd2a553`; `eval_playback_loop.py`; v13 scorecard (acoustic 0.2755 vs 0.4925 PASS, loop `tts_elevenlabs_sample.wav` 0.248 FAIL).

## Backfill — Model architecture (why each arch over the alternative)

### D-005 — SeqCNN (v11) over MLP: frame-level sequence in, pooled decision out — 2026-09-11 — model
- **Context:** MLP (v3–v10) mean-pooled 63/66-d LFCC+prosody(+physio) over whole clip — no temporal structure; v9 at chance on phone (0.5087).
- **Options considered:** A) Bigger MLP + more features B) Frame-level sequence model (Conv over 184×60 LFCC + 6 scalars) C) Raw-waveform end-to-end (AASIST-style, Idea 1).
- **Decision:** B as `VoiceGuardSeqCNN` (`seqcnn_v1`, 2×Conv1d k=3, RF 5 frames ≈130 ms). C deferred to Phase-0 spike (Tasks 1–11, unexecuted — no code on disk).
- **Why this over that:** Pool-then-classify destroys order (pauses vs. stop-consonants look identical); convolution preserves local dynamics with ~10k params vs. raw-waveform's data/compute appetite on a GB-scale ffmpeg corpus. Raw waveform kept as spike, not baseline.
- **Tradeoff accepted:** RF 130 ms too short for prosody-scale cues (later fixed by TCN). Mitigation: shipped as v11 baseline to unblock the two-I/O ONNX contract + attack-type head work.
- **Evidence:** `model.py` arch table; `73cbbba` per-frame LFCC+scalars; `68aa14a` masked multi-task trainer; `e2d8a32` two-I/O ONNX inference.

### D-006 — SeqTCN (v12/v13) over SeqCNN: dilated residual TCN, RF 65 frames ≈1.1 s — 2026-09-12 — model
- **Context:** SeqCNN RF 5 frames can't see syllable/prosody-scale structure; v11 phone EER 0.4853 (~chance).
- **Options considered:** A) Deeper SeqCNN (RF grows linearly, params explode) B) Dilated residual TCN (`seqtcn_v2`: dilations 1,2,4,8,16,32 → RF 65, ~87k params, mean+std+max pooling) C) Transformer/Conformer immediately.
- **Decision:** B — v12 first model above chance on phone (0.3223, CIs disjoint from v11) + first to pass attack-head gates (leave-attack-out BAcc 0.763, MLAAD-TTS 74.3%).
- **Why this over that:** Dilation gives exponential RF per param — 1.1 s context at 87k params, still exportable to ONNX + runnable on-device. Transformer rejected *then*: needs more data + positional machinery for marginal gain at this corpus scale; kept as later axis (C/4 conformer `54b0e3c`).
- **Tradeoff accepted:** Still failed 9/14 confound gates (texture-keying got *worse* on fake side) + gsm_2g 0.4469 + accents at chance. Mitigation: documented as confound problem (D-003/D-007), not arch problem alone; v13 = same recipe + playback + remediation.
- **Evidence:** `model.py`; `4d0e935` handoff; `3133c04` v12 report; `d09f6ac`/`54b0e3c` rework axes.

### D-007 — OC-Softmax (Idea 6) over two-class BCE: one-class objective on frozen v13 embedding — 2026-09-18 — model
- **Context:** BCE is free to pick *any* separating axis — it picked speaking style (D-003). Need an objective where style variance *within* genuine speech must fit the same bound.
- **Options considered:** A) More BCE data/augmentation B) Per-attack specialist heads, max-score combiner (Idea 5, `Task 12` — no learned combiner, flag-selectable variant) C) OC-Softmax: single learned genuine direction w0, genuines inside tight margin, spoofs outside looser one (`oc_softmax.py`, Zhang et al. 2021).
- **Decision:** C as cheapest causal test: `finetune_oc_softmax_v13.py::embed_v13` reads `VoiceGuardSeqTCN.forward` up to `trunk_out` (model.py untouched), `--freeze-backbone` default True trains ONLY w0 — isolates “does the objective move the gate” from “does more training move it.”
- **Why this over that:** B still uses BCE per head (same axis-picking freedom, just partitioned). C changes the *geometry*: one genuine bound forces trunk to represent entity, not style, by construction. Frozen-first follows the one-variable-at-a-time discipline that caught the `clean`-recipe regression.
- **Tradeoff accepted:** Cosine-to-w0 score `(1-sim)/2` must rescale into `confound_table` as fake-prob — a mapping assumption. Clamp to [-1,1] needed (float32 overshoot ~1e-7). Real run blocked: no corpus/cache on this machine (`itw_real/select: 2177 manifest files missing`). Mitigation: ships as Task-15 standalone (no AASIST-spike dependency — correction in `8236685`); unit-tested (`test_oc_softmax.py`, `test_finetune_oc_softmax_v13.py` shape/determinism/value-parity); runbook in `state.md` for training box.
- **Evidence:** `3a4b421`, `4c9961d`, `3a13b29`; `8236685` (decouple from unrun AASIST spike); `ef0b165` (Ideas 5/6).

### D-008 — NOT shipping dataset-pruning / AASIST-spike-gated work speculatively — 2026-09-18 — model
- **Context:** Brainstorm proposed dataset pruning (Idea 4) + raw-waveform AASIST spike (Idea 1, Tasks 1–11) as remediations; temptation to build dependents speculatively.
- **Options considered:** A) Build everything now B) Gate: Task 14 records which corpus variant was trained on; Task 15 (OC-Softmax on v13) ships standalone with zero spike dependency.
- **Decision:** B — `spike_model.py`/`train_spike_aasist.py` untouched (don't exist); pruning doc (`DATASET-PRUNING-PLAN.md`) proposed-not-started.
- **Why this over that:** Unran-spike-gated code rots and confuses the gate trail; the pooled→frame-level swap Idea 6 actually needed already shipped (2026-09-11, deployed as v13) — the brainstorm overstated the dependency and was corrected, not built upon.
- **Tradeoff accepted:** Slower-looking progress (docs before code). Mitigation: Task 10 runs all three variants (BCE / specialist / OC-Softmax) on the SAME corpus + gate when the spike exists — a real comparison, not three anecdotes.
- **Evidence:** `8236685`; `state.md` 2026-09-18 session; `DATASET-PRUNING-PLAN.md` (no code).

## Backfill — On-device inference & audio path (1/2)

### D-009 — ONNX Runtime over tflite_flutter — 2026-09-11 — app
- **Context:** Training exports ONNX; TFLite path needed conversion plus single-I/O assumptions.
- **Options considered:** A) Keep TFLite + convert each export B) `flutter_onnxruntime` ^1.8.5, ship training artifact directly.
- **Decision:** B. Verified Dart API + platform channel + native impl pass arbitrary Map straight to ORT session.run. Softmax in Dart on raw logits; heuristic fallback when asset missing.
- **Why this over that:** Eliminates lossy conversion; one I/O contract ((1,184,60)+(1,6) in, two (1,2) logits out) shared by training, backend, app. Normalization inside graph.
- **Tradeoff accepted:** `TFLiteService` names kept (confusing) to avoid touching every call site; infer/scoreChunk became async.
- **Evidence:** `969fd7e`; `d822a84`; `e2d8a32`; `1149ede`.

### D-010 — compute() isolate + separate attackTypeStream — 2026-09-11 — app
- **Context:** Pure-Dart extraction stalls 1-2.5 s per tick; attack output needed without breaking calibration Stream<double>.
- **Options considered:** A) UI thread B) compute() off-thread C) Merge attack type into scoreStream.
- **Decision:** B + additive `attackTypeStream` in AudioService.
- **Why this over that:** A janks every tick; C breaks calibration contract. Copy cost fits 3 s window / 1 s tick budget.
- **Tradeoff accepted:** Per-chunk isolate copy cost. Mitigation: budget absorbs it; jank is binding constraint.
- **Evidence:** `tflite_io.dart::scoreChunk`; `e2d8a32`; `a66eec6`.

## Backfill — audio path (2/2)

### D-011 — Noise-free LFCC parity fixture — 2026-09-11 — app
- **Context:** Dart must bit-match Python features.py or scores diverge silently.
- **Options considered:** A) Noisy fixture B) Noise-free with tight tolerance.
- **Decision:** B for LFCC sequence parity; physio kept noisy fixture.
- **Why this over that:** numpy vs Dart PRNGs differ, LFCC is per-frame spectral so noise lands in every frame (several units on half the coeffs). Physio frame-aggregates average it out. Tight tolerance beats vacuous wide one.
- **Tradeoff accepted:** Less realistic fixture. Realism lives in channel/playback renditions, not the parity test.
- **Evidence:** `a66eec6`; parity fixtures + audio_processor parity tests.

### D-012 — Native AudioResampler16k + ring buffer + WAV forensics — audio
- **Context:** RemoteAudioTap dumped 48kHz stereo into a 16kHz-mono pipeline (3x squash, invalid LFCC). Plus big-endian read of little-endian PCM16.
- **Options considered:** A) Resample in Dart B) Native Kotlin downmix + 48-to-16k with bounded ring buffer C) Assume 16kHz.
- **Decision:** B (`4afe244` + `ec69d3b`), byte-order fix `7a6132a`, ring snapshot `af27f54`, WAV encoder `37812da`, 5s forensic dump on alert `2f069c5`.
- **Why this over that:** Per-sample arithmetic at 48kHz belongs native (GC overhead); guarantees 100ms/3200-byte chunks regardless of source. Dumps make alerts auditable in LogsScreen.
- **Tradeoff accepted:** Two implementations to sync (Kotlin vs Dart). Mitigation: unit-tested resampler + WAV ground truth.
- **Evidence:** `4afe244`, `ec69d3b`, `7a6132a`, `af27f54`, `37812da`, `2f069c5`.

### D-013 — flutter_webrtc + native AudioTrack.addSink tap — app/audio
- **Context:** No Dart remote-audio tap in flutter_webrtc history; need remote PCM without telephony mic restrictions.
- **Options considered:** A) MediaProjection loopback (OS-gated, fragile) B) Self-owned WebRTC call + native addSink via getRemoteTrack C) Reflection into plugin.
- **Decision:** B (RemoteAudioTap.kt + eb9c362, fcd9e47, 1000f52 stack).
- **Why this over that:** addSink is public JNI-backed, direction-agnostic; same mechanism plugin uses locally. Public accessor, no reflection. Self-owned call has no telephony-call restriction by construction.
- **Tradeoff accepted:** Protects VAANI-to-VAANI only, not carrier/WhatsApp calls. Documented as still-open, not hand-waved.
- **Evidence:** RemoteAudioTap.kt header; `eb9c362`, `fcd9e47`, `c5b4d76`.

## Backfill — backend signaling + app state

### D-014 — Dumb 2-per-room WebSocket relay over stateful VoIP server — backend
- **Context:** Protected Call needs SDP/ICE handshake; media is P2P WebRTC, never via server.
- **Options considered:** A) Stateful INVITE/RINGING/ACCEPT/BYE + TURN + FCM (Voip.md target) B) Dumb pipe: relay verbatim, no inspect/persist, cap 2 per room C) Third-party CPaaS.
- **Decision:** B now (`signaling.py`: _rooms dict, room-full 4000), A as phased plan (Voip.md phases 1-6).
- **Why this over that:** FastAPI APIRouter + WebSocket needs no new infra; 2-cap keeps demo threat model trivial (no MITM without occupying room first). No persistence means no PII store (DPDP-friendly).
- **Tradeoff accepted:** Manual 4-digit room codes both sides; hardcoded 10.0.2.2:8001 default; Google STUN only (fails symmetric NAT/CGNAT); no push/background/lock-screen. Mitigation: Settings host/port (D-015), error surfacing + safe teardown (D-016), Voip.md roadmap to TURN/FCM/ConnectionService.
- **Evidence:** `signaling.py`; `a4f7e02`, `e4b3cc1`; `Voip.md` gap table.

### D-015 — Provider (ChangeNotifier) + SettingsProvider persisted host/port — app
- **Context:** Signaling host/port was hardcoded; needed per-device config without rewrite.
- **Options considered:** A) Riverpod/Bloc migration B) provider ^6.1.1 ChangeNotifier + shared_preferences (already in pubspec) C) Hardcode per build flavor.
- **Decision:** B: SettingsProvider persist + SettingsScreen fields + ProtectedCall uses configured values (`c095cf2`, `812c9c0`, `75b3974`, `6007e80` focus-loss persist).
- **Why this over that:** provider already wired (RiskScore/CallState/Calibration); smallest diff that solves emulator (10.0.2.2) vs device (LAN/Tailscale) split. No new dep, no pattern churn pre-demo.
- **Tradeoff accepted:** ChangeNotifier rebuilds coarsely vs Bloc event purity. Fine at this screen count; risk banner + score already flow through it (`9c3a888`, `72b83d0`).
- **Evidence:** `c095cf2`, `812c9c0`, `75b3974`, `6007e80`; pubspec provider/shared_preferences.

### D-016 — Surface signaling errors + safe teardown over silent hang — app
- **Context:** Wrong/unreachable host left call UI hanging; partial call state leaked across retries.
- **Options considered:** A) Silent retry B) errors stream on SignalingService + user-visible failure + endCall teardown detaches sink, disposes stream/pc C) Crash log only.
- **Decision:** B (`2516db7` errors stream; `b867c48` teardown).
- **Why this over that:** Demo on LAN/emulator/Tailscale means wrong-host is the common case, not edge case. Fail-loud beats hang-silent; teardown order (cancel sub, detach native, dispose local, close pc, null refs) prevents ghost taps double-scoring.
- **Tradeoff accepted:** Extra error plumbing in every call path. Worth it: one silent-hang demo kills trust faster than one error banner.
- **Evidence:** `2516db7`, `b867c48`; signaling_service_test.

### D-017 — Alert policy: EMA 0.35 + 3s speech precond + 0.70 x2 cycles + banner/haptic/forensics — app
- **Context:** Raw per-chunk scores jitter; noise/false-positive-on-noise history (`3a2fc32` shipped validated model for it).
- **Options considered:** A) Alert on single chunk > threshold B) EMA smooth + min-speech gate + 2-consecutive-cycles + tiered banner (Emerald/Amber/Crimson) + haptic + 5s dump C) Server-side adjudication.
- **Decision:** B (Voip.md Phase 5: alpha 0.35, 3s valid speech, >0.70 twice; `72b83d0` banner, `d07aa74` haptic, `2f069c5` dump, `c5b4d76` VoIP screen, `211353a` CalibrationProvider per-speaker threshold).
- **Why this over that:** A pages on every noise burst; C adds latency + PII off-device. Local deterministic policy keeps call real-time and auditable.
- **Tradeoff accepted:** Adds ~2-3 s detection latency by design; per-speaker calibration adds UX step. Mitigation: documented thresholds, not magic; calibration is opt-in.
- **Evidence:** `72b83d0`, `d07aa74`, `2f069c5`, `211353a`, `3a2fc32`; Voip.md Phase 5.

## Backfill — TeleChannel DSP + deps

### D-018 — Real ffmpeg round-trips over mock codecs — 2026-09-04 — audio
- **Context:** Module A codec stage claimed G.711/GSM/AMR/Opus support on mocks; invisible muxer bugs possible.
- **Options considered:** A) Keep mocked codecs B) Shell out to real ffmpeg full GPL build (Gyan, per-user winget, no admin).
- **Decision:** B (`codec.py` shells out; 148/148 pytest pass; found + fixed AMR-WB ext awb-to-amr: no muxer for .awb, .amr auto-detects NB/WB).
- **Why this over that:** Mock-verified codecs lie about muxers/encoders; the AMR bug class is invisible until a real binary runs. Lesson recorded: treat mock-only-verified paths as unproven.
- **Tradeoff accepted:** Per-file subprocess round-trips make corpus builds multi-hour; Windows PATH dependency. Mitigation: feature cache (D-019) so renders happen once; PATH documented.
- **Evidence:** `4459f31`, `915e198`; CLAUDE.md FFmpeg section; telechannel codec/pipeline tests.

### D-019 — Hash-pinned feature cache + recipe-scoped invalidation — model
- **Context:** GB-scale corpus x 8 channel units x ffmpeg renders; naive rebuild every protocol tweak = days.
- **Options considered:** A) Rebuild all on any change B) feature_version hash over pipeline+stages+channels.yaml; recipe-scoped revalidate, rebuild only affected units C) No cache (render on fly).
- **Decision:** B (`feature_cache.py` + `build_caches.py` + `revalidate()`; frozen-FileSpec fix so legacy units re-stamp exact).
- **Why this over that:** A wastes GPU-box days; C makes train/eval nondeterministic and slow. Content-hash invalidation is the only scheme that survives concurrent multi-session edits (CLAUDE.md warns many parallel sessions touch this FS).
- **Tradeoff accepted:** Hash covers whole parsed channels.yaml, so adding playback invalidated everything once (D-004 cost). Mitigation: scoped re-hash after; staged cmds separate cache (StageA) from train/eval (StageB).
- **Evidence:** `e2d6874`, `f350840`, `407c055`; v13_stageA/B cmds; feature_cache tests (48 green).

### D-020 — Pinned small deps over bleeding edge — devops
- **Context:** Demo must build offline-ish on judge hardware; unpinned deps break flutter/onnx/fastapi together.
- **Options considered:** A) Caret-range latest B) Pin server trio (fastapi 0.110.0, uvicorn 0.29.0, pydantic 2.6.0, onnxruntime >=1.17) + app (provider 6.1.1, flutter_onnxruntime 1.8.5, flutter_webrtc 0.12.5, web_socket_channel 3.0.1) C) Vendor everything.
- **Decision:** B (backend/requirements.txt, pubspec.yaml). Test-only httpx listed so pip install -r suffices for suite.
- **Why this over that:** A broke the two-I/O contract path once (ORT API drift); C bloats repo with .onnx-scale binaries (gitignored except shipped asset). Pins are the middle: reproducible without vendoring.
- **Tradeoff accepted:** Pins rot (google_fonts 8.2.1, permission_handler 11.3.1 will age). Mitigation: lockfiles (pubspec.lock) + `.gitignore` keeps runs/audio out; bump deliberately, not by surprise.
- **Evidence:** backend/requirements.txt; voice_guard/pubspec.yaml; `.gitignore` (audio/runs/onnx ignored, shipped asset excepted).

## Future Log (new AI decisions go below — use template)

### D-021 — FLOW.md created + pull/update rule extended to it — 2026-09-22 — devops
- **Context:** User ordered: every pull/update must also update the md; plus a FLOW.md explaining the entire "database", its logic and everything.
- **Options considered:** A) Only update DECISIONS.md on pulls B) New FLOW.md (no SQL exists — document the six real stores end-to-end) + extend the DECISIONS.md header rule to cover both files.
- **Decision:** B. `git pull origin vaani` → Already up to date (`3a13b29`); created `FLOW.md` (§0 map + §1 corpora + §2 manifest + §3 TeleChannel + §4 windowing/features + §5 cache + §6 train→eval→ONNX + §7 on-device flow + §8 prefs/logs/dumps + §9 backend/signaling/monitor + §10 vaani/ side + §11 change protocol); header rule now mandates pull-note + DECISIONS entry + FLOW update in the same commit.
- **Why this over that:** "Database" as SQL doesn't exist — pretending otherwise would be a lie. The six-store trace (with writers/readers per store) is what a newcomer actually needs; coupling both files to one rule keeps why (DECISIONS) and how-it-works-now (FLOW) from drifting apart.
- **Tradeoff accepted:** FLOW.md duplicates some DECISIONS backfill content in flow form. Accepted: different question (how vs why); cross-linked, and §11 forces joint updates so they can't rot independently.
- **Evidence:** This commit (FLOW.md new + DECISIONS.md header + D-021); pull output `Already up to date` on `3a13b29`.

<!-- APPEND-BELOW -->
