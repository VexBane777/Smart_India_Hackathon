# VAANI Pitch Deck Structure (Module C Task 7 Step 1)

> Drafted from Doc 3 §3.2. **All numbers remain `[M]` placeholders** — they are
> filled only in Task 7 Step 3, from the measured artifacts mapped in
> `lockdown.json`. Gate: *any `[M]` still unfilled = the slide is not ready*
> (ship nothing estimated).
>
> Deck rules (Doc 3 §3.1): one assertion per slide; every number carries a
> source artifact; demo at Slide 5; fonts ≥24 pt; speaker notes carry the words.

Assumed slot: ~6 minutes (verify against actual SIH rules before rehearsal).

| # | Time | Headline | Content | Numbers (source) |
|---|---|---|---|---|
| S1 | 10s | **VAANI — real-time defense against AI voice-clone fraud** | Team, problem code SIH26104, product screenshot at 70% | — |
| S2 | 30s | **Caller ID lies. The voice is the last undefended layer** | Arup, Hong Kong, Feb 2024 — $25M transferred after an all-deepfake video call (documented) + [verified Indian advisory — insert only after confirming a dated source] | Arup $25M → dated news URL (appendix) |
| S3 | 25s | **Every current answer guards the wrong thing** | Pindrop/NICE (cost + months) · STIR/SHAKEN (verifies the number, not the voice) · academic detectors (offline, black-box) | — |
| S4 | 15s | **A gatekeeper that listens, explains, and holds** | Live-synthetic risk scoring + explainable evidence + transaction hold with step-up — on ordinary CPUs | — |
| S5 | **90s** | **LIVE DEMO** | Run Doc 4 asset through the real pipeline (§3.3 cue sheet; measured cue times from `cues.json`, not script estimates) | — |
| S6 | 30s | **Streamed, not batched. Cascaded, not monolithic** | Architecture diagram; "first score ~[M] s, refreshed 0.5 s, per-window [M] ms on CPU" | `[M]` ×2 → `runs/<model>/latency.json` |
| S7 | 45s | **Every number measured. Every failure mode disclosed** | Leaderboard: EER ×4 protocols + In-the-Wild, FPR@TPR90, p95 latency (CPU + Pi 5), ECE, 3-seed mean ± std | `[M]` ×5 → MLflow finals + `leaderboard.md` |
| S8 | 30s | **A telephone-channel corpus, validated against real Indian calls** | Fingerprint overlay figure (Doc 1 §1.9 / Doc 5) + recipe chain + "released publicly with datasheet" | `[M]` ×2 → `fingerprint_validate.py` output |
| S9 | 20s | **Hold, don't block. Prove, don't hide** | Step-up MFA · hash-chained audit log (the honest blockchain answer) · DPDP privacy | — |
| S10 | 25s | **Runs where the call happens** | Measured p95 on laptop CPU + Pi 5, ONNX + INT8, zero cloud; deploys as an API in call routing | `[M]` ×1 → Pi bench log |
| S11 | 20s | **Built in 12 weeks. What production needs next** | Done: demo, corpus, calibrated models, leaderboard, public artifacts. Next: bank MFA integration, TRAI-registered SMS, telecom partnership, retraining pipeline | — |
| S12 | 15s | **Close** | "A dataset, a detector, and a decision workflow — measured, reproducible, running on the laptop in front of you." Team + contact | — |

**Appendix (pulled during Q&A):** per-generator breakdown · known failure modes
(unseen generators, replay, adversarial) · reproducibility card · team roles ·
license/consent manifest.

## Demo cue sheet (Slide 5, from Doc 3 §3.3)

| T | Driver (laptop) | Presenter says |
|---|---|---|
| 0:00 | App loaded, CUDA off, airplane mode ON, volume pre-set | "A pre-recorded call, streamed live through the identical pipeline. Disclosed — we control audio quality, not the hotel Wi-Fi." |
| 0:10 | Start stream | (silence — let the gauge move) |
| 0:25 | — | Real voice: "gauge stays low." Clone enters: "watch the curve spike." |
| 0:55 | Attempt transfer on mock bank | "Transaction → held. Step-up verification, not a silent block." |
| 1:10 | Show audit log panel | "Tamper-evident — that's our blockchain." |
| 1:20 | — | "Everything you saw ran on this CPU." |

*All cue times must be replaced by measured values from `cues.json` (Doc 4 §4.9) before the deck is built.*

## Q&A drill sheet (from Doc 3 §3.4)

Carried over from the master plan: unseen-generator honesty · compromised SIM ·
latency (measured only) · why trust it (calibration + published failures) ·
privacy (DPDP) · why-blockchain (hash chain). Plus the four drills in Doc 3 §3.4
(replay, adversarial, "why not OTP always", "self-generated dataset circular",
"banks won't integrate a student project").

## Number lockdown (summary — full mapping in `lockdown.json`)

| Slide | Number | Source artifact |
|---|---|---|
| S5/S6 | first-score latency, per-window compute | `runs/<model>/latency.json` |
| S7 | EER ×5, FPR@TPR90, ECE | MLflow finals + `leaderboard.md` |
| S7/S10 | Pi 5 p95 | Pi bench log |
| S8 | cutoff match, LTAS correlation | `fingerprint_validate.py` output |
| S2 | Arup $25M | named dated news source (URL in appendix) |

**Gate: any `[M]` still unfilled = the slide is not ready. Ship nothing estimated.**
