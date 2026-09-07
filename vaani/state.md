# Project State: VAANI Documentation Suite

## Current Status
- **Phase**: Module B Week-1 slice + Task 3 (SSL Teacher) implemented; Module C
  fully implemented (all code-buildable tasks; see "Module C implemented per
  plan" below for what's deliberately still gated/flagged) and **committed**
  (`1cfec2b`, contradicting this file's older "not yet committed" note below —
  trust `git log`, not that note). 270/270 tests passing as of 2026-09-07.
- A second, parallel demo surface (`voice_guard/`, a Flutter Android POC +
  its own FastAPI backend on port 8001) landed 2026-09-07, now wired onto
  the same `AlertStateMachine` decision policy as `vaani/app/engine_mock.py`
  (commit `2ec7450`). It is a separate git-tracked directory at repo root,
  not part of the `vaani/` Python package. Model is still heuristic — no
  `assets/models/*.tflite` dropped in yet.
- **Goal**: Execute implementation plans.
- **Last Updated**: 2026-09-07

## Progress Tracking
- [x] Explore project context
- [x] Ask clarifying questions
- [x] Propose approaches
- [x] Present design
- [x] Write design docs
- [x] Spec self-review
- [x] User review of specs
- [x] Transition to implementation

## Notes
- User confirmed hybrid approach: TDDs + Granular Backlogs.
- User approved Approach 3: Integrated Component-Based Expansion.
- All TDDs and Backlogs completed and stored in `VAANI_documentation_suite_v1.0/TDDs/`.
- All implementation plans created in `VAANI_documentation_suite_v1.0/docs/superpowers/plans/`:
    - `2026-09-04-vaani-module-a-data.md`
    - `2026-09-04-vaani-module-b-model.md`
    - `2026-09-04-vaani-module-c-product.md`
    - `2026-09-04-vaani-module-d-ops.md`
- System is now ready for execution. Session paused by user.

## FFmpeg installation gap resolved (2026-09-04)
- Installed FFmpeg 9.0.1-full_build-www.gyan.dev via `winget install --id Gyan.FFmpeg -e`
  (per-user, no admin rights required). Full build includes libopencore-amrnb,
  libvo-amrwbenc, libgsm, libopus — all encoders MOD-A-06 requires.
- Ran full test suite (`pytest tests/`): **148 passed**, 0 skipped, 0 failed —
  every ffmpeg-gated real-encoder test in `tests/telechannel/test_codec.py`
  and `tests/telechannel/test_pipeline.py` (previously skipped for lack of
  ffmpeg) now executes and passes.
- **Bug found and fixed by this run:** `telechannel/stages/codec.py`
  `CODEC_SPECS["libvo_amrwbenc"]["ext"]` was `"awb"`; ffmpeg has no muxer
  registered for that extension (only `.amr`, which auto-detects NB vs. WB
  from the encoded stream), so `test_amr_wb_roundtrip_real_ffmpeg` failed
  with "Unable to choose an output format" until the extension was changed
  to `"amr"`. Updated the stale "not independently verified" caveats in
  `codec.py` and `tests/telechannel/test_codec.py` to reflect the pass.
- MOD-A-06 (`Verify Codec Support`, Backlog_MOD_A_TeleChannel.md) is now
  satisfied: `ffmpeg -encoders | grep -Ei "gsm|amr|opus|alaw|mulaw"` shows
  libopencore_amrnb, libvo_amrwbenc, libgsm, libopus, pcm_alaw, pcm_mulaw
  all present.

## Module C parallel-safe slice (2026-09-05)
- **Task 1 — assets, as PLACEHOLDERS only:** real human recording (Doc 4 §4.4,
  Form A consent gate) has NOT happened. `assets/raw/call_{A,N,B}_raw.wav` are
  machine-generated stand-ins (Windows SAPI TTS, 48 kHz mono, 90.0 s, −6.0 dBFS
  peaks, QA-passing) built by `assets/scripts/build_demo_placeholders.py` from
  `assets/raw/call_scripts.json` (the Doc 4 §4.2/§4.6 script source of truth).
  `assets/manifest.json` records `provenance: placeholder_tts` and the consent
  warning — replace with real takes before any pitch/demo use.
- **Task 4 — skeleton done, mock backend:** `app/server.py` (FastAPI WebSocket
  streamer, 0.5 s chunks), `app/engine_mock.py` (mock score backend implementing
  the master plan §6 decision logic: EMA + 2-consecutive-window rule;
  EMA α=0.7 — with α=0.3 the EMA's memory defeats the 2-window rule, caught by
  tests), `app/ws_client.py`, `app/components/{gauge,spectrogram}.py` (pure
  matplotlib mel — no librosa), `app.py` (Streamlit + st.fragment). Verified
  live: server boot → WS stream of call_A → first alert at t=22.5 s vs clone
  entry at 22 s (onset ≈3–4 s incl. window, per spec). Single swap point:
  `MockBackend` → real Module B backend; nothing else changes.
- **Task 7 Steps 1–2:** `docs/pitch/deck_structure.md` (12 slides per Doc 3
  §3.2) + `docs/pitch/lockdown.json` (number→source mapping). All values still
  `[M]`/pending — Step 3 remains gated on Module B's measured runs.
- **Tests:** +20 in `tests/demo/` (engine contract, alert timing, A-alerts/
  N-never, spectrogram, WS streaming). Full suite: **168 passed** (148 prior).
  Had to `pip install librosa` (Module A's `validate_reality.py` imports it at
  module scope; was missing in this env — pre-existing gap, now resolved).
- Committed as 9925963 on `vaani`; audio files stay untracked per .gitignore.

## Review pass (2026-09-04)
- Fixed a typo in Module D's plan (`la-Cuda` → `CUDA`, Global Constraints section).
- Added an **"Execution Order & Cross-Module Dependencies"** section to each of the four
  module plans — they previously read as independently executable, but C depends on
  A (pipeline) + B (trained model), B depends on A (manifest/splits), and D's Makefile
  targets are wrappers around A/B/C scripts. Each section also states what's actually
  Week-1-critical-path vs. Week 5+/freeze-time scope, since the Week 1 sprint (master
  plan §8) needs a thin slice through all four modules, not depth-first completion of A
  before B before C.
- **Known open issue, not fixed:** every doc in this suite carries a
  `<!-- GENERATED by generate_vaani_docs.py — do not hand-edit -->` header, but that
  script does not exist anywhere under this directory tree. The edits above were
  necessarily hand-edits. Before further changes, decide whether to reconstruct the
  generator (and re-apply these edits as data in it) or drop the "generated" framing and
  treat these files as the source of truth going forward.

## Module B Week-1 slice implemented (2026-09-05)
- Scope: Tasks 1 (`registry.py` + `configs/*.yaml`), 2 (`models/cnn.py::TinyCNN`), and
  Task 4 Step 1 only (`train.py`'s basic training loop) from
  `docs/superpowers/plans/2026-09-04-vaani-module-b-model.md`. Tasks 3 (SSL Teacher),
  5-10, and Task 4 Steps 2-4 (distillation, MLflow, HF Hub sync) are **not** implemented
  — deferred for lack of GPU/HF Hub access in that session; user chose this scope
  explicitly over fuller CPU-feasible or full-10-task alternatives.
- Built via superpowers:subagent-driven-development in worktree `vaani-module-b`
  (branched off `vaani`), 3 tasks + 2 fix rounds + a final whole-branch review + one
  final fix wave, all reviewed clean. Fast-forward merged into `vaani` at `d3e0f1e`.
  187/187 tests passing on `vaani` after merge.
- `registry.load()` now requires importing `models` before lookups resolve — fixed a
  real bug where it silently failed from a clean process (only worked before because
  test files happened to import the model module first). Config merge is a deep/
  recursive merge, not shallow.
- **Known deviations from `DOC2_TRAINING_CONFIGS.md`/`TDD_MOD_B_01`**, documented in
  `registry.py`'s module docstring: `type` is hoisted to the config's top level (spec
  has it nested under `model:`); `audio.sr`/`win_s` renamed to `sample_rate`/
  `window_seconds`; `features.hop` renamed to `hop_length`.
- **Known gaps, deferred (not bugs — tracked for whoever picks up Module B next):**
  `train.py`'s training loop is dataset-agnostic (takes pre-built DataLoaders, not a
  manifest-backed AudioDataset — none exists yet, no real audio corpus is present under
  `data/`, and `torchaudio` isn't installed, only `librosa`/`soundfile`); it also doesn't
  consume `cnn_week1.yaml`'s `optim` block (hardcodes Adam/lr/epochs as kwargs); the
  config schema's `${name}` interpolation, `ckpt.every_steps`/`keep_last`, and the whole
  `tracking:` block are unimplemented (noted in `base.yaml` and `train.py`); no shape/chs
  validation on `TinyCNN`; `register_model` silently overwrites on duplicate keys;
  checkpoint format is `state_dict`-only, not the richer resume payload DOC2 §2.3
  specifies. None of these block Module B's Week-1 scope; they matter once Tasks 3/5-10
  or a real training run start.
- **`SSLHead` (Task 3) takes raw waveform, `train.py`/`TinyCNN` assume mel spectrograms**:
  `SSLHead.forward` expects `(B, num_samples)` waveform while `train.py`'s documented
  data contract and `TinyCNN.forward` expect `(B, 1, n_mels, T)` mel tensors (both models
  now declare this via an `input_kind` class attribute -- `"waveform"` vs. `"mel"` -- for
  callers to branch on). `registry.load("ssl_teacher")` therefore cannot be dropped into
  `train.py` as-is; whoever wires Task 4 Step 2+ or Task 5 needs to either branch
  `train.py` on `input_kind` or give `SSLHead` its own training entry point.
- **`configs/ssl_teacher.yaml`'s `freeze:`/fine-tune settings are inert** (same class of
  gap as the `optim:`/`tracking:` blocks above): `freeze: [feature_extractor]` (HF idiom:
  conv front-end only) and `bfloat16`/`grad_checkpointing`/`accum` are read by nobody --
  `SSLHead` unconditionally freezes the entire wav2vec2 backbone via `no_grad()`,
  making it a ~0.2%-trainable linear probe regardless of what the config says. See
  `registry.py`'s module docstring and `ssl_head.py`'s module docstring for the full
  writeup. Deliberate for Task 3's brief ("freeze the SSL feature extractor"); revisiting
  the freeze scope is a decision for whoever does real teacher training (Task 5+).

## Module B Task 3 (SSL Teacher) implemented (2026-09-05)
- `vaani/models/ssl_head.py::SSLHead` wraps the real `facebook/wav2vec2-base`
  checkpoint (`transformers.Wav2Vec2Model`, loaded lazily inside `__init__` so
  `import transformers` isn't a side effect of every `registry.load()` call),
  registered as `"ssl_head"` (config `configs/ssl_teacher.yaml` has `type: ssl_head`).
  A learnable softmax-normalized layer-weight vector `w` (sized to
  `num_hidden_layers + 1` at construction time, not hardcoded) mixes the
  checkpoint's hidden states before a 768→256→2 MLP head. Feature extractor is
  frozen (`requires_grad=False`, `no_grad()`) and pinned in `.eval()` even under
  `model.train()` (a `train()` override prevents SpecAugment/dropout from
  reactivating during training).
- Built via superpowers:subagent-driven-development in worktree
  `vaani-module-b-ssl` (branched off `vaani`): 1 task + task review (approved
  clean) + final whole-branch review (1 Critical + 4 Important findings) + one
  fix wave + scoped re-review (all addressed, no new breakage). Fast-forward
  merged into `vaani` at `9e54522`. 209/209 tests passing after merge.
- The final whole-branch review caught real cross-cutting issues a task-scoped
  review missed by construction (see known-gaps entries above for the two that
  remain as documented, deliberate gaps: waveform-vs-mel input mismatch with
  `train.py`, and the inert `freeze:`/fine-tune config block). The train-mode
  SpecAugment leak (teacher silently nondeterministic under `model.train()`)
  and a hardcoded hidden-state count (crash on non-12-layer bases) were both
  fixed, not just documented.

## CPU-only closing tasks addressed (2026-09-06)

User asked to determine whether a Kaggle-download command works, whether the
existing pipeline handles real audio, and to close out CPU-feasible pending
items across Modules A/B/C that were previously deferred for lack of a real
corpus/GPU. GPU remains a separate rota machine (per Doc 8), not this
session's machine (`torch 2.10.0+cpu`, no `nvidia-smi`) — so anything
actually requiring GPU compute or a real training corpus is still deferred;
only genuinely CPU-only gaps were closed this session.

- **Kaggle download command (`sripaadsrinivasan/audio-mnist`) does not work
  as given**: no `kaggle` CLI installed, no `~/.kaggle/kaggle.json` —
  `kaggle.com/api/v1/datasets/download/...` requires Basic Auth, so the bare
  `curl -L` gets a 401. Needs `pip install kaggle` + a `kaggle.json` API key
  from the user's Kaggle account (a credential this session can't supply).
  Not yet done — blocked on the user's Kaggle credentials.
- **Real-audio pipeline smoke test found, then fixed, a full-scale overshoot
  bug (2026-09-06)**: `tests/telechannel/*` only ever exercised synthetic
  signals. Ran a real recorded clip (librosa's bundled `trumpet` example,
  16 kHz mono) through all 7 `channels.yaml` recipes via
  `telechannel/pipeline.py::process_clip` end-to-end. All produced
  finite-valued output, but **`gsm_2g` and `cellular_3g` peaked above 1.0**
  (1.18, 1.13) — no clamp/normalize existed anywhere in
  `mic.py`/`codec.py`/`bandlimit.py`/`pipeline.py`. `mic.py`'s gain stage
  applies up to +12 dB unclamped (only clipped 20% of the time via
  `clip_prob`), and bandlimit filter ringing could push it further. Real
  speech/instrument audio has enough amplitude headroom to expose this;
  the test suite's low-amplitude synthetic fixtures never would.
  **Fixed** (user chose belt-and-suspenders, option C, after a layman's
  explanation): `apply_mic()` (`telechannel/stages/mic.py`) now always
  clamps its output to `[-1, 1]` after gain, on top of (not instead of) the
  existing probabilistic ±0.5 "cheap preamp" clip; `process_clip()`
  (`telechannel/pipeline.py`) additionally clamps its final return value to
  `[-1, 1]` as an end-of-chain safety net catching overshoot from any other
  stage (e.g. bandlimit ringing), not just mic gain. Re-ran the same
  trumpet-clip smoke test after the fix: every recipe now peaks at exactly
  1.000 (correctly clamped) instead of overshooting. 2 new regression tests
  added (`test_mic.py`, `test_pipeline.py`); full telechannel suite
  145/145 passing.
- **Module B CPU-only gaps closed** (`registry.py`, `train.py`,
  `models/cnn.py`), via TDD, 220/220 tests passing after:
    - `TinyCNN` now raises `ValueError` on `n_mels <= 0` or an empty/
      non-positive `chs` tuple, instead of failing obscurely later
      (`chs[-1]` IndexError or an invalid `Conv2d`).
    - `registry.load_config()` now resolves `${name}` placeholders (e.g.
      `ckpt.dir: runs/${name}` -> `runs/cnn_week1`) via a new
      `_interpolate()` step, recursing through the merged config's string
      values using its own top-level `name` field. No-op (placeholder left
      literal) if the config has no `name`.
    - New `train.py::train_from_config(model, config, train_loader,
      val_loader, device=None)`: reads `optim.opt`/`optim.lr`/`optim.epochs`
      to build the optimizer and epoch count (adam/adamw only), and
      `ckpt.dir`/`ckpt.every_steps`/`ckpt.keep_last` to write periodic,
      rotated checkpoints with the richer `{model, optimizer, step, epoch,
      config_hash}` resume payload DOC2 sec2.3 specifies — instead of
      `train()`'s single state-dict-only save at the very end. The original
      `train()` is untouched (still used by existing tests/callers).
    - Deliberately still NOT done (needs either GPU time or is out of
      scope for this pass): `optim.sched` (LR scheduling), `optim.clip`
      (gradient clipping), reading a checkpoint back in to actually resume,
      the `register_model` silent-overwrite "gap" (turns out the test
      suite's fixtures rely on that overwrite behavior for test isolation —
      making it strict would break ~5 existing tests for no real benefit,
      so left as-is), and the entire `tracking:`/MLflow block (needs a
      reachable MLflow server — a GPU-rota-machine concern, not CPU-only).

## Module C implemented per plan (2026-09-06)

User asked to "implement Module C fully". Scoped explicitly (user's choice)
to all code-buildable work, with physical-world-only steps mocked/automated
or flagged rather than attempted: real human recording + RVC cloning (Task
1) was already placeholder-only from the prior session and left as-is;
Task 8's screen-recorded backup video and Task 6's literal "physically pull
the network cable" step are flagged as needing a human, not attempted.

- **Task 2 (Demo Asset Channeling)**: new `assets/scripts/channel_demo_assets.py`
  runs `call_A_raw.wav`/`call_N_raw.wav` through `telechannel.pipeline`'s
  `whatsapp` recipe -> `assets/demo/call_{A,N}_whatsapp.wav` (16kHz mono
  WAV, not FLAC — plan's own "Files" section names `.wav`, followed that
  over the "Interfaces" line's FLAC mention). `manifest.json` gets a
  `demo` sub-entry per call. `server.py`'s `available_calls()` now prefers
  the channeled variant over raw when present, so the live demo actually
  streams phone-quality audio.
  - **Real bug found and fixed**: `server.py`'s `stream_call` hardcoded
    `MockBackend()` (default `clone_entry_s=22.0`) for every call
    regardless of `call_key`. `call_N` (Doc 4 §4.6's all-real-voice
    control) has no clone segment at all, so streaming it through the
    *actual server* would have falsely spiked risk after 22s — only the
    existing unit test (which manually passed `clone_entry_s=1e9`) caught
    the *intended* behavior; the wiring into `server.py` never had it.
    Fixed via new `engine_mock.clone_entry_for_call(call_key)`, which
    reads `call_scripts.json`'s per-segment `voice` tags and returns
    `None` for calls with no `"*_clone"` segment. `MockBackend.clone_entry_s`
    now accepts `None` (never spikes). Regression tests added in
    `test_server.py` (`test_call_N_control_never_alerts`,
    `TestChanneledDemoAssets`).
- **Task 3 (Demo Cue Generation)**: `assets/scripts/record_cues.py` (CLI:
  `--call --input --out`) scores an audio file in 2s/0.5s-hop windows via
  the same mock backend and writes a `{timestamp: score}` JSON cue sheet
  plus `first_alert_t`/`clone_entry_s` metadata. Deviation from the plan
  text ("modify vaani.engine") documented in the script's docstring — no
  `vaani.engine` module exists anywhere in this repo; reused
  `app.engine_mock` instead (the same swap point Task 4 already
  established). Generated `assets/demo/call_A_whatsapp.cues.json`
  (committed); verified `first_alert_t == clone_entry_s == 22.0` matches
  the script's clone-entry timestamp exactly.
- **Task 4 (Streamlit UI)**: found already fully implemented by the prior
  session (Steps 1-4: server, gauge, spectrogram, risk curve) — verified
  via the existing test suite, no changes needed beyond the Task 2 fix
  above (which Task 4's UI now benefits from — it streams channeled audio).
- **Task 5 (Bank Sim + OTP) + hash-chained audit log**: new
  `app/components/audit_log.py` (`AuditLog`/`AuditEntry`: SHA-256 chain,
  each entry embeds the previous entry's hash; `verify_chain()` detects
  tampering and returns the first broken seq) implementing the master
  plan §1 item 5 / §2 Friday-milestone requirement ("hash-chained decision
  log... our honest answer to the Blockchain theme") — not explicit in the
  Module C plan's Task 5 file list, but explicitly named as part of the
  same Friday bank-sim milestone in `00_MASTER_PLAN.md`, so built alongside
  it. New `app/components/bank_modal.py`: `decide_transfer_outcome(state)`
  (pure function: "alert" -> held, anything else -> approved) +
  `render_bank_panel()` (Initiate Transfer -> HOLD -> simulated OTP entry
  -> release, or immediate approval if no risk; every transition appended
  to the audit log; OTP is a disclosed on-screen demo constant, never
  actually sent anywhere, matching the master plan's "SIMULATED and
  disclosed" requirement). Wired into `app.py`. Tested via
  `streamlit.testing.v1.AppTest` (full HOLD->OTP->release and
  immediate-approval flows) plus pure-function unit tests.
- **Task 6 (Airplane Mode Verification)**: `tests/demo/test_offline.py`
  automates the CPU-only slice — monkeypatches `socket.socket.connect`/
  `connect_ex` to raise on any *non-loopback* connection attempt (loopback
  allowed: that's how the local demo and the test's own WS transport work,
  neither is "external"), then runs the full call-A/call-N stream and the
  bank-sim workflow end-to-end, asserting zero non-loopback connection
  attempts. Documented in the test's docstring that this does **not**
  replace a real physical rehearsal (literally disabling WiFi/Ethernet on
  the demo machine) — that manual pass is out of scope for an autonomous
  session and still needs a human before the actual pitch.
- **Task 7 (Pitch Deck)**: Steps 1-2 already done by the prior session
  (`deck_structure.md`, `lockdown.json`). Step 3 (fill measured values)
  deliberately left gated/pending — no real Module B training run or
  measured latency/EER numbers exist yet (no GPU, no real corpus this
  session either), and inventing placeholder numbers would violate the
  plan's own Global Constraint ("No estimated numbers"). Not a gap to
  close later without real measurements.
- **Task 8 (Final Demo Production)**: not attempted — needs a human to
  perform and screen-record the actual 90s demo rehearsal. Flagged, not
  faked.
- Full suite: **270 passed** (was 220 before this session; +50 across
  Tasks 2/3/5/6's new tests). Not yet committed — awaiting user's go-ahead
  per this session's "always confirm before pushing/committing" default.

## Module C: prior-session context (2026-09-05 note, preserved as written)
- **Module C: confirmed zero code existed at the time this was checked**
  (no FastAPI/Streamlit/bank-sim/audit-log anywhere in the repo) — true as
  of this session's investigation, but a separate parallel session landed
  a Module C slice (`app/server.py`, `app/engine_mock.py`, placeholder demo
  assets; see "Module C parallel-safe slice" above, commit `9925963`)
  concurrently, merged into this same push. User's call on further Module C
  work stands regardless: most of it (UI, bank sim, hash-chained log,
  airplane-mode test) was never actually GPU/corpus-gated — only Task 1-3
  (demo asset recording/channeling) depend on having real demo audio, and
  the merged slice's assets are explicitly placeholder TTS, not real
  consented recordings.

## Demo-prep sweep (2026-09-07)

User asked to walk through remaining demo prep and finish everything not
gated on a human. Findings:

- Module C's "not yet committed" note above was stale — it was committed
  as `1cfec2b` sometime between 2026-09-06 and now (by this session or
  another concurrent one; `git log` is authoritative, this file lags).
- **Real bug found and fixed in `voice_guard/`**: `pubspec.yaml` declares
  three asset directories (`assets/models/`, `assets/sounds/`, `assets/images/`)
  that didn't exist on disk — `flutter test`/`flutter build`/`flutter run`
  all errored on it ("unable to find directory entry in pubspec.yaml").
  Fixed by creating the three directories with `.gitkeep` placeholders.
  Verified clean after: `flutter test` (3 tests pass, no errors),
  `flutter analyze` (no issues), backend `/v1/analyze-chunk` smoke-tested
  live via `uvicorn` + `curl` (200 OK, correct response shape), `/docs`
  loads, and `vaani/app/server.py` still imports cleanly alongside it
  (no port/state collision between the two FastAPI apps — 8000 vs 8001).
- Everything else identified as remaining is human-only and was not
  attempted: real consented demo-audio recording (Doc 4), physical
  rehearsal + literal airplane-mode/cable-pull test, Kaggle API
  credentials, GPU-rota training time, judge Q&A drills, and the product
  decision of whether `voice_guard` (Android) supplements or replaces the
  Streamlit demo as the primary demo surface for judges — not decided
  yet, ask before assuming either way.
- `docs/pitch/lockdown.json` Step 3 (measured numbers) remains
  deliberately unfilled — still no real training run or measured
  latency/EER exists to fill it with, and the master plan's Global
  Constraint forbids estimated numbers on slides.