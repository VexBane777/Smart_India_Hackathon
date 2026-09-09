<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: VAANI — Master Plan (Real-Time Voice Clone Detection Gateway) · Owner: Whole team · Status: Approved — execution begins
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# VAANI — Master Plan (Real-Time Voice Clone Detection Gateway)

**Owner:** Whole team · **Status:** Approved — execution begins · **Suite:** v1.0 · **Generated:** 2026-09-03

---

## 0. North Star & Project Frame

> **A real-time gatekeeper that listens to a live phone call, flags synthetic
> voice, and holds the transaction until a human re-verifies — running on
> ordinary hardware, backed only by measured numbers.**

Six deliverables:
1. **Streaming detector** — scores a call live, window by window.
2. **TeleChannel dataset** — our phone-network-degraded audio corpus (the differentiator).
3. **Honest explainability** — shows *which moments* triggered the alarm, never fake reasons.
4. **Decision layer** — holds transactions and triggers step-up verification; never silently blocks.
5. **Tamper-evident audit trail** — hash-chained decision log (our honest answer to the Blockchain theme).
6. **Edge deployment** — runs on laptop CPU and a Raspberry Pi 5 (a credit-card-sized $80 computer).

**Frame:** SIH 2026 · Problem SIH26104 · Theme: Blockchain & Cybersecurity · Sponsor: AICTE ·
Budget ₹0 · working demo in 1 week · finished model in 12 weeks (hard freeze week 10) ·
Hardware: 1× RTX 5050 (8 GB) + 2× RTX 4050 (~6 GB) + free Kaggle GPUs.

*Reading convention:* technical terms carry a plain-language bracket on first use.

## 1. Non-Negotiables

1. **`main` branch always runs the full demo.** New capability lands behind a config flag (a settings toggle).
2. **Honest numbers only.** Nothing reaches a slide before it is measured. No invented statistics.
3. **Fail gracefully.** We *hold* and step up verification; we never claim to block fraud outright.
4. **Privacy by design.** On-device inference (audio processed where the call happens), no audio retained, consent documented — DPDP Act 2023 aligned (India's personal-data law).
5. **Model freeze at week 10.** Weeks 11–12 ship; they do not train.
6. **The demo never depends on the venue.** No stage Wi-Fi, no live mic, no cloud. Backup video exists.
7. **Ship public artifacts.** Dataset recipe, model card (standard document describing data/performance/limits), reproducible leaderboard (results anyone can re-run and verify).

## 2. Product Story & Positioning

**The pitch (verified facts only):**
> "In February 2024, a finance employee at Arup in Hong Kong transferred **$25 million**
> after a video call in which every colleague was an AI deepfake — documented and widely
> reported. Indian regulators have publicly warned that AI voice-clone fraud now targets
> families and banks here. Caller ID can be faked in seconds. The voice on the call is the
> last line of defense — and today it is undefended. VAANI defends it."

Cite a specific Indian victim case **only** after verifying a dated news source; otherwise generalize.

**Claims discipline:**

| We say | We never say |
|---|---|
| "First risk score ~2.5 s, updated every 0.5 s — measured" | "150–250 ms total latency" |
| "We hold the transaction and trigger step-up verification" | "We automatically halt fraud" |
| "Cross-generator EER of X% on held-out generator Y" (EER = equal error rate, where misses equal false alarms; lower is better) | "Detects any deepfake, even one cloned minutes ago" |
| "Channel simulation validated against 25 real recorded calls" | "The heatmap says 'unnatural pitch transition'" |
| "Runs on CPU / Pi 5 at measured p95 latency" (p95 = the delay 95% of requests beat) | "Moat," "100% detection" |

**Competitors (one slide):** Pindrop/NICE (enterprise call-center analytics — expensive, slow to deploy) · academic anti-spoofing models (offline, black-box, no workflow) · STIR/SHAKEN (verifies the phone *number*, not the *voice*).

## 3. System Architecture

```
 [Live mic] OR [pre-recorded scam call, streamed in real time
                 through the identical pipeline — disclosed]
   → FastAPI capture service (Python web-server framework)
   → Silero VAD (tiny free model that knows when speech occurs)
   → 2-second windows, new one every 0.5 s (the "hop")
   → Mel-spectrogram (a picture of the sound: pitch energy over time)
   → Student model scores EVERY window (small & fast)
       └─ score in "unsure band" → Teacher model (big, slow, accurate,
          runs in a background thread) = the cascade
   → EMA smoothing (a smoothed trend that ignores one weird moment)
   → Call-level state machine: ALERT only if smoothed score stays above
      threshold for 2+ consecutive windows
   → Mock bank dashboard: transaction → HOLD → simulated OTP step-up
      (OTP = one-time password; SIMULATED and disclosed — real bulk SMS
      needs TRAI registration, not possible at ₹0)
   → Hash-chained audit log (each entry embeds the previous entry's
      fingerprint — tampering becomes detectable)
   → UI: live risk gauge · live spectrogram · per-window risk curve ·
      occlusion highlights (§6)
```

**Stack (all free):** Python · librosa (audio-to-picture) · sounddevice · torch · FastAPI ·
Streamlit with `st.fragment(run_every=0.5)` (refreshes one panel, not the whole page — fixes
the Streamlit freeze) polling over WebSocket (a live two-way connection).
**No WebRTC** (browser live-audio — needs TURN servers and Wi-Fi negotiation; three stage-failure
points for zero benefit on one laptop). **Demo machine rule:** CUDA (NVIDIA GPU mode) disabled —
force CPU, or the "runs on ordinary hardware" claim dies when a judge opens task manager.

## 4. Model Plan — the evolution ladder

Each rung must beat the previous on our own leaderboard before becoming default.

| Stage | Model | Role | Size | Runs on |
|---|---|---|---|---|
| Week 1 | **TinyCNN** (compact pattern-finder trained from scratch on spectrograms) | permanent fallback; the demo engine | 2–5M params (trainable knobs) | any 4050, minutes |
| Wks 5–8 | **AASIST / RawNet2** (published anti-spoofing baselines) | credibility benchmarks | 1–4M | 4050 #1 |
| Wks 5–8 | **SSL teacher** — wav2vec2/WavLM/XLS-R base (models pre-trained on mountains of unlabeled speech) + small head | the accuracy ceiling | ~95M | 5050 overnight; ≥300M variants on Kaggle T4 or LoRA (train small adapter layers only — big memory savings) |
| Wks 6–8 | **Distilled student** (the big teacher trains a small fast imitator) | the deployable streaming model | 2–5M | 4050 #2 |
| Week 9 | **Cascade** (fast model screens; unsure cases escalate to slow model) | production accuracy + speed | both | — |
| Reference | **AST** (the original proposal's model; fixed ~10 s input) | offline reference only, never streaming | 86M | — |

**Cross-cutting:** model-swap abstraction (one config line switches engines — the survival
mechanism) · calibration as a deliverable (temperature scaling = a one-number fix making
confidences match reality; reliability diagrams; ECE = expected calibration error) ·
threshold from an **FPR budget** (false-positive rate — how often we wrongly flag a real
person), e.g. "≤1% legitimate calls held at TPR ≥90%", never a magic 85% ·
**ECAPA-TDNN speaker verification** via SpeechBrain (free, CPU-friendly "is this the enrolled
person?" — a second, orthogonal alarm) · 3-seed final runs (seed = random start; averaging
three removes luck) → mean ± std · ONNX export (universal model format) + INT8 quantization
(8-bit compression for weak chips), measured on laptop CPU and Raspberry Pi 5.

## 5. Data Plan

### 5.1 Free existing datasets (get — for credibility)
| Dataset | Gives us | Access |
|---|---|---|
| ASVspoof 2019 LA (classic anti-spoofing benchmark, clean) | baseline comparability | research license — **register Monday 9 AM** |
| ASVspoof 2021 LA (with simulated telephony degradation) | channel-robustness benchmark | same registration |
| ASVspoof 5 (2024, newest attacks) | "unseen future generator" proxy | same |
| Half-truth / PartialSpoof (real audio spliced with fake sentences) | partial-spoof eval | registration |
| In-the-Wild (~40 h of messy internet audio) | harshest external test | research license |
| LibriSpeech, Common Voice (CC0), IndicVoices, MUCS | Indian-accent/multilingual real speech | free; per-corpus license check |

### 5.2 Generated fakes (create — license-safe only)
MeloTTS (MIT), Bark (MIT), Parler-TTS (Apache-2.0), AI4Bharat Indic-TTS — Hindi-English,
Tamil-English, Indian-accented English. **RVC** (free voice-cloning tool) cloning our own
consented team voices → powers the "cloned five minutes ago" judge demo.
**XTTS-v2 excluded** (non-commercial license) from anything released; **ElevenLabs**
output internal-eval only (free tier ~10 min/month anyway). Script the whole pipeline.

### 5.3 TeleChannel pipeline (create — the differentiator)
Physical-order simulation of what an Indian network does to a voice — scripted end-to-end
(Doc 1 is the full spec):

```
clean real/fake audio
 → RIR convolution (room echo fingerprint; SLR28 library)
 → additive noise (MUSAN, ESC-50, CC0 Indian street/market/horn)
 → mic stage (random gain + clipping) — BEFORE the codec, because
   μ-law quantization noise depends on signal level
 → codec round-trips via FFmpeg: G.711 μ/A-law (landline),
   GSM-FR (2G), AMR-NB (Indian cellular workhorse), AMR-WB (VoLTE),
   Opus 6–16 kbps (WhatsApp) + random TANDEM transcoding
   (compress-decompress twice across codecs — real cross-network calls)
 → bursty packet loss (Gilbert-Elliott, 0–10%, concealed)
 → P.56 300–3400 Hz band-limit for narrowband recipes
 → 8/16 kHz FLAC output + per-clip metadata JSON
```
Plus **RawBoost** (open-source on-the-fly corruption used by top ASVspoof 2021 systems)
during training; it lacks codec transcoding — which is why our FFmpeg chain is the star.

### 5.4 Protocol (what makes it a dataset, not a pile of files)
- **Four held-out test sets:** unseen generator (RVC-team), unseen codec (AMR-WB), unseen
  noise (freesound_horn), all real recorded calls. These four numbers ARE the generalization story.
- **Manifest per clip:** generator, accent, codec chain, bitrate, loss %, SNR (speech-to-noise
  loudness ratio), RIR id, license, consent. **Splits filter on manifest fields, never folders.**
- **Split at source-clip level, and real speech at speaker level** — same-clip or same-speaker
  leakage across train/test is the most common self-sabotage in audio ML.
- **Sizing (₹0-feasible):** ~50–60 h real + ~50–60 h fake × 6 recipes ≈ 500–600 h ≈ ~1M
  two-second windows. FLAC only (lossless) — **never Opus/MP3**, which would inject codec
  damage into the real class and poison our own story. ~35 GB.

### 5.5 Validation against reality (the killer slide)
Record 25 consented real team calls (Indian cellular + WhatsApp; Doc 5 is the protocol).
Compare long-term average spectra, bandwidth cutoffs, and high-frequency energy ratios
between simulation and reality. Targets: median cutoff difference ≤300 Hz; spectrum-shape
correlation ≥0.9. *"Our channel simulation is validated against real Indian network
recordings."* Nobody at SIH will have that.

### 5.6 Release
HuggingFace Datasets (free dataset hosting) + a datasheet (sources, consent, license,
intended use). ASVspoof audio cannot be redistributed — we release the **recipe + our
generated portion**, which is the valuable part anyway.

## 6. Decision Logic & Explainability (honest by construction)

Per-window probability → EMA smoothing → **ALERT only when smoothed score exceeds threshold
for 2+ consecutive windows** (kills single-moment false alarms; makes the threshold defensible).

Explainability, priority order:
1. **Per-window risk curve** on the UI (spikes exactly when the cloned voice speaks).
2. **Occlusion sensitivity** (hide 0.5 s slices, measure the alarm drop — shows *which
   moments* drove the decision). Primary; no library version hell.
3. Simple rule-based stats for honest text like "high synthetic-voice likelihood 00:14–00:16."
4. **Grad-CAM** (heatmap of where the network looked) — stretch goal, timeboxed 3 hours, never load-bearing.

**Never** generated sentences claiming *why* the voice is fake. The UI says *where* and
*how confident* — nothing more.

## 7. Hardware & Training Infrastructure

| GPU | VRAM (GPU memory) | Critical notes |
|---|---|---|
| RTX 5050 (Blackwell, sm_120 chip code) | 8 GB | **needs PyTorch ≥2.7 cu128 wheels** — older builds silently fall back to CPU. Pin one env on ALL machines. If desktop → the always-on **master**. |
| 2× RTX 4050 (Ada, sm_89) | ~6 GB | fine except big-model full fine-tuning; laptops throttle (heat-slowdown) — expect 60–80% of nominal; plugged in, lids open, never-sleep power plan, Windows auto-restart disabled. |

**Why NOT one merged cluster:** DDP (splitting one training run across GPUs) needs gradient
all-reduce (GPUs syncing what they learned) over the home network — ~2–4 s/step vs ~0.5 s of
compute for a 95M model: *slower than single-GPU*. Heterogeneous cards sync to the slowest;
Wi-Fi drops kill overnight runs. Verdict: documented stretch option only; in that scenario
Kaggle's T4 (16 GB) is the better answer anyway.

**What "connected" means instead — federated cluster:**

```
MASTER (5050):  canonical dataset (FLAC) · MLflow (free experiment tracker —
                one shared leaderboard) · git remote
WORKERS (all 3): SSH + tmux (terminal sessions that survive disconnects);
                data replicated via Syncthing (free P2P sync)
SHARED:         HuggingFace Hub private repos — every run auto-checkpoints
                there; any machine or Kaggle resumes any run, one command
```

Overnight insurance (crashes cost one checkpoint interval, never a night):

```bash
while ! python train.py --resume auto --ckpt hf:<repo>; do
  echo "crashed $(date), retrying"; sleep 30
done   # inside tmux
```

**Job routing:** ≤6 GB → any local GPU (unlimited hours is our real edge over Kaggle's
30 h/week) · 7–8 GB → 5050 · >8 GB (300M-class) → Kaggle T4 or 5050+LoRA · same-day +
all busy → Kaggle · GPU dies → one-line resume elsewhere · finals → local AND one Kaggle
run (proves environment-independence) · **the demo itself → any laptop, CUDA disabled.**
Kaggle uploads cap ~20 GB → curated subsets only.

**Day-0 checks:** capability printout on all three machines
(`torch.cuda.get_device_capability()` — catches the sm_120 trap) · fixed 10k-step benchmark
on each GPU + one T4 (samples/sec, VRAM, temps, 30 min) — every capacity claim becomes a
measured number. Doc 8 owns all of this.

## 8. Timeline (12 weeks · demo week 1 · freeze week 10)

**Week 1 — the demo that must exist by Sunday:**

| Day | Work | Done means |
|---|---|---|
| Mon | ASVspoof registration 9 AM; consent templates (Doc 6); pull LibriSpeech/Common Voice; TTS/RVC generation script; augmentation chain | ~3–5k balanced windows incl. phone-channel variants |
| Tue | Train TinyCNN; hold-out + cross-generator eval; benchmark pretrained HF baseline; export ONNX | metrics table exists — EER measured, not guessed |
| Wed | Streaming service: VAD, chunker, EMA, state machine, latency instrumentation | WAV → live-updating score end to end |
| Thu | Streamlit UI: gauge, spectrogram, risk curve, occlusion highlights | 5-minute demo, zero crashes |
| Fri | Mock bank: transfer → HOLD → OTP modal → hash-chained log; metrics page | full workflow E2E |
| Fri (parallel) | Module E: Flutter Android app shell, file import + mic capture, gauge/spectrogram/risk curve/occlusion wired to a clearly-labeled stub scorer (bank sim/audit-log UI deferred to after the demo) | Phone runs the identical pipeline as the laptop demo, labeled "simulated — real model pending" |
| Sat | Record 90 s scam call (Doc 4); 3 rehearsals; backup video; measure latency; airplane-mode test | pitch numbers are real |
| Sun | Buffer: preload model, README, diagram, Q&A drills | Definition of Done met (§9/§10) |

*If the finale is a 36-hour SIH event: build all of this BEFORE the event; the finale is
assembly + polish only.*

**Weeks 2–12:**

| Phase | Weeks | Deliverables | GPUs | Done means |
|---|---|---|---|---|
| Data foundation | 2–4 | ASVspoof processed; TeleChannel v1; ~100 h clean Indic corpus; consents + 25 real calls; cluster + benchmarks | light (one 4050 on TTS/RVC) | corpus + augmentation + real-call validation set |
| Models | 5–8 | AASIST/RawNet2; SSL teacher; distilled student; ONNX/INT8; calibration | **peak** — 5050 teacher nights, 4050s sweeps + distillation, Kaggle big-VRAM run | leaderboard exists; student within ~2–3× teacher EER; latency measured |
| Robustness + FREEZE | 9–10 | four hold-out protocols; partial-spoof; cascade; ECAPA mode; Pi 5 numbers; 3-seed finals | all three in parallel | every protocol reported; **frozen end of week 10** |
| Ship | 11–12 | model card; dataset release; docs; Docker; CI; demo assets; judge drills; reproducible leaderboard (local + Kaggle) | light | one command retrains and reproduces; demo 3× clean offline |
| Mobile (Module E) | whenever Module B ships ONNX | swap StubScorer → OnnxScorer at one call site (mobile/lib/scoring); bank HOLD/OTP/release + audit-log UI (Task 12) also picked up here | none (CPU/NPU on-device) | phone app runs real on-device inference, same thresholds as desktop |

**Roles (4):** Corpus lead (data, TeleChannel, releases, consent) · Model lead (baselines,
teacher, distillation, calibration) · Systems lead (engine, cascade, cluster, CI, Pi) ·
Product lead (UI, bank sim, demo assets, pitch, Q&A).

## 9. The Demo (90 seconds, fully controlled)

| T | Beat |
|---|---|
| 0:00–0:10 | Hook: Arup $25M + Indian advisory (verified facts only) |
| 0:10–0:25 | "Pre-recorded call, streamed through the *exact* live pipeline in real time — disclosed, not hidden" |
| 0:25–0:55 | Real victim voice (gauge low) → cloned voice enters (gauge climbs, curve spikes) → 2 occlusion highlights |
| 0:55–1:10 | Transfer attempted → **HOLD** → OTP step-up modal → "simulated; production plugs into the bank's existing MFA" |
| 1:10–1:15 | Hand a phone to a judge running the identical pipeline on the bundled demo call — same gauge, same alert (mobile bank/audit-log panel not built yet — flagged post-demo) |
| 1:15–1:30 | Audit log + metrics slide (EER table, measured latency, calibration) → close |

Fallbacks in order: live app → backup video → annotated screenshots. Model preloaded,
airplane mode verified, everything on one laptop with CUDA off.

## 10. Metrics & Definition of "Finished"

| Metric | Minimum | Stretch |
|---|---|---|
| EER, ASVspoof 2019 LA eval | ≤2% | ≤1% |
| EER, ASVspoof 2021 LA (telephony) | ≤5% | ≤2.5% |
| EER, our Indian telephony hold-out | ≤6% | ≤3% |
| EER, unseen generator + In-the-Wild | ≤10% | ≤6% |
| FPR @ TPR≥90% (operating point) | ≤1% | ≤0.5% |
| Student p95 latency/window, laptop CPU | ≤30 ms | ≤15 ms |
| Raspberry Pi 5 p95 | ≤100 ms | ≤50 ms |
| Calibration ECE | ≤5% | ≤3% |

**"Finished" = minimum column hit + all artifacts:** student weights (ONNX + INT8) + model
card · TeleChannel recipe + corpus + datasheet · reproducible leaderboard (3 seeds, all
protocols) · repo with green CI, Docker, one-command training · demo (app + video) · deck
with measured numbers only.

**Standing caveats:** generalization to unseen generators is an open research problem — our
goal is *measured, disclosed* generalization; and the week-10 freeze exists so "finished"
never becomes "still training on day 89."

## 11. Risk Register

| Risk | Mitigation |
|---|---|
| ASVspoof approval delayed | registered Monday 9 AM; generated corpus + TeleChannel carries Month 1 |
| 5050 env quirk (sm_120 → silent CPU fallback) | Day-0 capability check; pinned requirements everywhere |
| Overnight laptop death / OS reboot | auto-resume wrapper; checkpoints to HF Hub; power settings; GPU rota (Doc 8) |
| 6 GB VRAM ceiling mid-plan | LoRA + gradient checkpointing; Kaggle structurally |
| Master machine failure | MLflow/git/HF replicated off-machine — master is convenience, not dependency |
| Custom model underperforms | model-swap abstraction → pretrained baseline in one line; week-1 CNN never deleted |
| Grad-CAM breaks | occlusion sensitivity is primary and independent |
| App crashes on stage | backup video + screenshots; preloaded model; fully offline |
| License landmines in released data | permissive generators only; manifest tracks license/consent per clip |
| Team exams/holidays | freeze earlier, not later |
| Temptation to inflate numbers | claims table (§2); only measured numbers reach slides |

## 12. Judge Defense (rehearsed, honest)

| Attack | Answer |
|---|---|
| "Voice cloned 5 minutes ago?" | "No one can guarantee detection of unseen generators — we publish our cross-generator numbers rather than hide them. That's why we don't block: we hold and trigger step-up human verification, plus a second orthogonal signal — speaker verification." |
| "Compromised SIM?" | "STIR/SHAKEN verifies the number; we verify the voice. We deploy as an API inside call routing or on the agent's screen, edge-side." |
| "Latency?" | "First score ~2.5 s, refreshed every 0.5 s, per-window compute [measured] ms on CPU — we show the instrumentation, not a claim." |
| "Why trust it?" | "Calibrated probabilities, published FPR budget, 3-seed reproducible leaderboard, released dataset recipe." |
| "Privacy?" | "On-device, no audio retained — features and scores only; consent under DPDP 2023; tamper-evident audit trail." |
| "How is this blockchain?" | "We don't bolt on a chain — we use cryptographic hashing where it genuinely fits: a tamper-evident decision log." |

## 13. Deliverables Checklist

- [ ] Demo: app (offline, CPU-forced) + backup video + screenshots
- [ ] Models: distilled student (ONNX + INT8), week-1 CNN fallback, model card
- [ ] Data: TeleChannel pipeline + recipe + corpus + datasheet + manifest, on HuggingFace
- [ ] Evaluation: reproducible leaderboard, four hold-out protocols, partial-spoof, calibration, 25-call validation
- [ ] Engineering: green CI, Docker, one-command retrain, auto-resume cluster docs
- [ ] Mobile: Android app (Flutter) — Module E, Phase 1 stub-scored + Phase 2 real-model swap point (bank HOLD/OTP/release + audit-log UI deferred to after the demo, Task 12)
- [ ] Compliance: consent forms, license manifest, DPDP privacy note
- [ ] Pitch: measured numbers only, Q&A drills done, demo rehearsed 3× clean

**Working documents:** Doc 1 (TeleChannel spec) · Doc 2 (training configs) · Doc 3 (pitch
deck) · Doc 4 (demo assets) · Doc 5 (real calls) · Doc 6 (consent/privacy) · Doc 7 (repo &
repro) · Doc 8 (GPU rota). All generated from the same source as this file.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
