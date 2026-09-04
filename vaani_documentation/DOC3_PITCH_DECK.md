<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 3 — Pitch Deck Outline (SIH Presentation Pack) · Owner: Product Lead · Status: Build-ready; numbers gated
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 3 — Pitch Deck Outline (SIH Presentation Pack)

**Owner:** Product Lead · **Status:** Build-ready; numbers gated · **Suite:** v1.0 · **Generated:** 2026-09-03
**Depends on:** Measured artifacts from Doc 2 runs; Doc 4 demo assets

---

## 3.1 Deck rules
1. **One assertion per slide** — headline is the message; slide is the proof.
2. **Every number carries a [M] tag until replaced by its measured value** from a source
   artifact (`latency.json`, leaderboard runs, calibration report). No number enters the
   deck without a source path.
3. Demo lives at Slide 5 — the audience judges the rest through what they just saw.
4. Fonts ≥24 pt; no paragraphs; speaker notes carry the words.
5. Verify the actual slot length against SIH rules; timings below assume ~6 minutes.

## 3.2 Slide-by-slide
| # | Time | Headline | Content |
|---|---|---|---|
| S1 | 10s | VAANI — real-time defense against AI voice-clone fraud | Team, problem code SIH26104, product screenshot at 70% |
| S2 | 30s | Caller ID lies. The voice is the last undefended layer | Arup, Hong Kong, Feb 2024 — $25M transferred after an all-deepfake video call (documented) + [verified Indian advisory — insert only after confirming a dated source] |
| S3 | 25s | Every current answer guards the wrong thing | Pindrop/NICE (cost + months) · STIR/SHAKEN (verifies the number, not the voice) · academic detectors (offline, black-box) |
| S4 | 15s | A gatekeeper that listens, explains, and holds | Live-synthetic risk scoring + explainable evidence + transaction hold with step-up — on ordinary CPUs |
| S5 | **90s** | **LIVE DEMO** | Run Doc 4 asset through the real pipeline (§3.3 cue sheet) |
| S6 | 30s | Streamed, not batched. Cascaded, not monolithic | Architecture diagram; "first score ~[M] s, refreshed 0.5 s, per-window [M] ms on CPU" |
| S7 | 45s | Every number measured. Every failure mode disclosed | Leaderboard: EER ×4 protocols + In-the-Wild, FPR@TPR90, p95 latency (CPU + Pi 5), ECE, 3-seed mean ± std |
| S8 | 30s | A telephone-channel corpus, validated against real Indian calls | Fingerprint overlay figure (Doc 1 §1.9 / Doc 5) + recipe chain + "released publicly with datasheet" |
| S9 | 20s | Hold, don't block. Prove, don't hide | Step-up MFA · hash-chained audit log (the honest blockchain answer) · DPDP privacy |
| S10 | 25s | Runs where the call happens | Measured p95 on laptop CPU + Pi 5, ONNX + INT8, zero cloud; deploys as an API in call routing |
| S11 | 20s | Built in 12 weeks. What production needs next | Done: demo, corpus, calibrated models, leaderboard, public artifacts. Next: bank MFA integration, TRAI-registered SMS, telecom partnership, retraining pipeline |
| S12 | 15s | Close | "A dataset, a detector, and a decision workflow — measured, reproducible, running on the laptop in front of you." Team + contact |

**Appendix (pulled during Q&A):** per-generator breakdown · known failure modes
(unseen generators, replay, adversarial) · reproducibility card · team roles ·
license/consent manifest.

## 3.3 Demo cue sheet (Slide 5)
| T | Driver (laptop) | Presenter says |
|---|---|---|
| 0:00 | App loaded, CUDA off, airplane mode ON, volume pre-set | "A pre-recorded call, streamed live through the identical pipeline. Disclosed — we control audio quality, not the hotel Wi-Fi." |
| 0:10 | Start stream | (silence — let the gauge move) |
| 0:25 | — | Real voice: "gauge stays low." Clone enters: "watch the curve spike." |
| 0:55 | Attempt transfer on mock bank | "Transaction → held. Step-up verification, not a silent block." |
| 1:10 | Show audit log panel | "Tamper-evident — that's our blockchain." |
| 1:20 | — | "Everything you saw ran on this CPU." |

Pre-flight (Sat rehearsal): model preloaded ✅ CUDA off ✅ airplane mode tested ✅ backup
video cued to the same timestamps ✅ 3 screenshots in appendix ✅. If the app dies:
*"And this is why we recorded the backup"* → video. No apology, no improvisation.
All cue times must be **measured** from `cues.json` (Doc 4 §4.9), not script estimates.

## 3.4 Q&A drill sheet
From the master plan: unseen-generator honesty · compromised SIM · latency (measured
only) · why trust it (calibration + published failures) · privacy (DPDP) ·
why-blockchain (hash chain). Add:

| Attack | Answer |
|---|---|
| "What if the scammer plays a *recording* of the real person?" (replay) | "Correct — replay is a different attack class; our synthetic detector won't flag it, because it isn't synthetic. Roadmap: liveness + pause-pattern signals; speaker verification partially covers splices. Disclosed as a known limit." |
| "Adversarial attacks?" | "Unserved by us and by most published systems — that's why the safety property lives in the *workflow*: cascade, second signal, human step-up, not model immunity." |
| "Why not OTP every high-value transfer?" | "Banks partly do — and OTP interception via SIM-swap is itself documented fraud. VAANI makes step-up *targeted*: friction only when voice risk is high." |
| "Your dataset is self-generated — circular?" | "Four held-out protocols, an external benchmark, validation against real recorded calls. Generalization is measured and disclosed — and the recipe is public." |
| "Banks won't integrate a student project." | "It runs today as an edge API on one screen, no integration. Production is a call-routing hook — honestly scoped on the roadmap slide." |

## 3.5 Numbers lockdown table
| Slide | Number | Source artifact |
|---|---|---|
| S5/S6 | first-score latency, per-window compute | `runs/<model>/latency.json` |
| S7 | EER ×5, FPR@TPR90, ECE | MLflow finals + `leaderboard.md` |
| S7/S10 | Pi 5 p95 | Pi bench log |
| S8 | cutoff match, LTAS correlation | `fingerprint_validate.py` output |
| S2 | Arup $25M | named dated news source (URL in appendix) |

**Gate: any [M] still unfilled = the slide is not ready. Ship nothing estimated.**

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
