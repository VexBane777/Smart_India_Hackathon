# VAANI model architecture brainstorm

Living brainstorm doc, one idea per top-level section, added to across sessions.
Also maintained as a Claude Doc for inline comments/collaboration:
https://claude.ai/code/artifact/a3be9e1f-25e2-48c6-827f-ba84eb43e330

---

# Idea 1: Laptop/Mobile Model Architecture Split — Cost-Benefit

As of 2026-09-17.

## Current state (corrects the starting premise)

There are not currently two working models to split apart. Only one real, trained model exists today.

| Component | What it actually is today |
| --- | --- |
| `voice_guard` (Flutter mobile app) | Real, deployed model: **v13**, a frame-level SeqTCN over LFCC sequences + 6 physiological scalars (pauseRatio, energyVariance, zcrVariance, jitter, shimmer, hnr_db). Deployed 2026-09-17 by explicit user override, despite failing its own confound gate. |
| `vaani/mobile` (separate Flutter app, `org.vaani.mobile`) | No trained model. `OnnxScorer` loads `identity_model.onnx`, a placeholder test fixture, fed a zero-filled tensor. Code comments mark it as waiting on "Module B's real exported model," which has not landed. |
| `vaani/app` (Python Streamlit/FastAPI PC diagnostic UI) | No trained model. Wired to `MockBackend`, also waiting on the same unshipped Module B model. |

So "both models hallucinate" isn't quite right — one model hallucinates (documented, measured); the other two client apps have no model to hallucinate with yet.

### The hallucination is a known, named, actively-worked bug

`voice_guard/docs/CRITICAL-entity-vs-style-confound.md` (2026-09-11) found and quantified that the model detects speaking **style** (how much pacing/energy/pitch vary) rather than speaker **entity** (human vs. AI). A human in a controlled/monotone register gets false-flagged; a fake with natural pacing slips through — confirmed live on-device and on held-out data. It proposed three remediation tracks, ranked by cost:

| Track | Approach | Status |
| --- | --- | --- |
| 1. Per-speaker calibration | Score deviation from a captured per-speaker baseline, not a fixed global threshold | **Not attempted** |
| 2. Add physiological features | jitter, shimmer, HNR alongside existing prosodic features | Tried (`v10_physio`) — did not close the confound |
| 3. Frame-level sequence model | Raw/frame-level LFCC sequence through a SeqTCN instead of a pooled-vector MLP | Tried (`v11`–`v13`) — cut headline EER roughly in half (0.49 → 0.31), confound gate still fails 9 of 14 rows in every version shipped so far |

The practical upshot: the two most expensive remediation tracks — more features, then more model capacity — have both already been spent on this exact model, and both left the specific hallucination mechanism intact. That's the baseline this proposal has to beat, not a green field.

## Proposed architecture

Two client apps, one shared data/eval pipeline, model family decided by a feasibility spike rather than assumed up front.

- **`vaani_laptop`** (rename of `voice_guard`): becomes the PC diagnostic UI, replacing `vaani/app`'s currently mock-backed Streamlit/FastAPI interface. Runs the large model **locally** (laptop-class CPU/GPU), cloud hosting deferred as a future option, not built in this scope — avoids taking on hosting infra and DPDP data-leaving-device compliance work (the `vaani/tests/legal` DPDP consent/revocation framework already exists specifically to gate audio leaving a device; a real hosted cloud service would need to run through it, a local model doesn't).
- **`vaani/mobile`**: becomes the primary mobile app. Its `OnnxScorer` stops pointing at the `identity_model.onnx` placeholder and gets a real trained small model — this is also the first time "Module B" actually ships into this app, not just a rename.
- **Shared underneath both**: the existing TeleChannel channel-degradation pipeline (`vaani/telechannel/configs/channels.yaml`), the corpus, the `select`/`test` split discipline, and the four-gate `evaluate.py` protocol. These are architecture-agnostic — they operate on raw audio and don't care whether the model consuming it is an MLP, a SeqTCN, or a raw-waveform network. Switching feature extraction only touches `dataset.py`'s feature step and the model class; the hard-won correctness infrastructure carries over unchanged.

**Open by design at this point:** whether the laptop and mobile models share one architecture family (large model full-size, mobile a distilled/pruned version) or diverge into two separate designs. Real-time latency, model-size and battery budget on target Android hardware aren't known yet — this doc treats that as the first thing to establish, not an assumption (see Phased plan, below).

The `voice_guard → vaani_laptop` rename itself is a separate, low-risk, bounded task (path/package renames, asset references, CI if any) independent of which model architecture gets chosen — it doesn't block or get blocked by the model work and could happen on its own timeline.

## Approaches considered

| | A: AASIST-family backbone | B: SSL fine-tune + classical features | C: Calibration only (track 1) |
| --- | --- | --- | --- |
| What it is | Raw-waveform + SincNet frontend + spectro-temporal graph attention, purpose-built for anti-spoofing generalization. Laptop gets it full-size; mobile gets a distilled/pruned/quantized version — pending the feasibility spike. | Laptop: fine-tune a pretrained wav2vec2/WavLM backbone (90M+ params) as the large model. Mobile: refined classical features (LFCC kept, MFCC/LPC/LSP/delta added, VAD-based snippet selection instead of "compressed sensing") on roughly the current architecture family. | Keep the current `v13` SeqTCN as-is. Add per-speaker relative scoring: brief on-device baseline capture, score subsequent audio as deviation from that speaker's own norm rather than a fixed threshold. No new model architecture. |
| Architecture novelty | Low-to-moderate — adapting a published, purpose-built design to this corpus/pipeline, not inventing one. | Low on laptop (transfer learning); the mobile-side feature work is genuine but same family as what already failed (LFCC/physio). | None. |
| Plausibility of closing the confound | **Higher** — targets vocoder-artifact-level signal (phase discontinuities, formant-transition smoothness) that no pooled cepstral statistic exposes; this is specifically why the anti-spoofing literature moved to this family. | **Low on the mobile side** — MFCC/LPC/LSP are the same representational family as LFCC+physio, which state.md already documents didn't fix it. SSL embeddings on the laptop side are more promising than classical features but still unproven against this specific confound. | Doesn't fix the representational ceiling, only compensates for it per-speaker. A naturally low-variance speaker still needs to be "calibrated in." Cheapest and fastest to ship, lowest ceiling on how far it can go. |
| Rough engineering effort (3-month budget) | High: new frontend, retrain per phone-channel recipe already in the pipeline, confound-gate re-evaluation, mobile distillation as its own workstream. | Laptop: moderate (fine-tuning is well-trodden). Mobile: low-to-moderate, but plausibly wasted effort per the plausibility row above. | Low: mostly app-side UX (baseline capture flow) + a scoring-layer change, no retraining pipeline needed. |
| Mobile feasibility risk | Real and unresolved — full raw-waveform + graph attention at real-time on a phone CPU is unverified; a distillation/quantization pass would be its own scoped task. | Lower risk technically (classical features are already proven feasible on this hardware), but risks spending mobile effort on a lever already shown not to work. | Lowest — runs on top of whatever model already ships. |

A fourth option — do nothing architectural and just keep iterating v13's existing SeqTCN on data/regularization levers (the `post-v13 rework` already in flight: CMVN, SpecAugment, mixup, focal loss) — isn't listed as a full approach because it's already the team's current default path, not a new decision; it's the implicit baseline every approach above is measured against.

## Recommendation

**Approach A (AASIST-family backbone) for the laptop model, plus Approach C (per-speaker calibration) layered on top of whatever ships, on both apps.** Not Approach B's classical-feature path — MFCC/LPC/LSP/delta are the same representational family as the physiological features already shown not to fix this (`v10_physio`), so spending mobile-side effort there is the lowest-expected-value use of the 3 months, not a genuinely different attempt. SSL fine-tuning (the other half of B) is a reasonable fallback for the laptop model specifically if A stalls, but AASIST is purpose-built for this exact generalization failure and is small enough to plausibly serve both apps, so it's the better first bet.

Calibration (C) is not a consolation prize — it's cheap, orthogonal, and helps regardless of which model family wins, so it should ship early and in parallel, not be treated as the fallback if A fails.

### Is this the most efficient path? — explicit verdict

**Partially, with one gate that should come before committing the full 3 months.** Two corrections change the calculus from how the idea was originally framed:

1. **The current model already detects TTS and voice conversion, imperfectly, not "can't."** v13's attack-type head passes its gate (MLAAD TTS share 76.6%, leave-attack-out balanced accuracy 0.75); MLAAD false-negative rate was 40.4% for v12 (the last measured number), down from v11's 53.7%. The gap is real but it's a miss-rate problem, not zero capability — and it's plausibly the *same* entity-vs-style confound expressed on the fake side (v13's fake-side confound rows are exactly where it still fails). That means closing the confound is likely to move TTS/clone detection too, not a separate problem needing separate work.
2. **The riskiest unknown isn't the architecture, it's whether this confound is representational or data-driven at all.** Two rounds of representational upgrades (more features, more capacity) both failed to close it. AASIST is a genuinely different representation, which is why it's recommended — but if it *also* fails to close the confound gate, that's evidence the bug is in the training data (not enough hard negatives: monotone real speech, naturally-paced fakes) rather than in any model architecture, and no further architecture spend would be efficient.

**So the efficient version of this plan is gated, not committed up front:** run a scoped early spike (small-scale AASIST trained on the existing corpus, scored through the existing `measure_confound.py`/`evaluate.py` gates) before committing to the full build-out on both apps. If it closes the gate, proceed with the full 3-month plan below. If it doesn't, redirect the remaining budget toward corpus/hard-negative work instead of a bigger model — that pivot is cheap to make early and expensive to discover in month three.

## Budget: where the 3 months actually goes

Rough phase-level time budget for the recommended path (spike → AASIST → calibration → both app integrations), assuming solo development at AI-assisted coding speed, not a committed schedule:

| Phase | Est. time | Compute/data need | Go/no-go criterion |
| --- | --- | --- | --- |
| 0. Feasibility spike | 1–2 weeks | Existing corpus, small-scale AASIST run, one Android test device | Confound-gate rows improve meaningfully AND on-device latency is real-time-viable → proceed; either fails → redirect per Recommendation section |
| 1. Full AASIST training (laptop) | 3–5 weeks | Reuse existing TeleChannel/corpus pipeline; GPU for training (existing training box) | Passes the same four-gate `evaluate.py` protocol already in use |
| 2. Mobile distillation/quantization | 2–3 weeks | Contingent on phase 0; separate workstream if divergent architectures needed | Real-time inference on target hardware within battery/latency budget |
| 3. `vaani_laptop` app integration | 1–2 weeks | None beyond dev time | Replaces `vaani/app`'s `MockBackend` with real scores end-to-end |
| 4. `vaani/mobile` app integration | 1–2 weeks | None beyond dev time | Replaces `identity_model.onnx` placeholder with real scores end-to-end |
| 5. Per-speaker calibration (both apps) | ~1 week, run in parallel early | None beyond dev time | Ships independent of model-family outcome |

That totals 9–15 weeks against a ~12-week budget — plausible but not slack-heavy, and optimistic against this project's own history: closing the confound has already taken three training rounds (`v11`→`v12`→`v13`) without succeeding, so phase 1's 3–5 weeks assumes AASIST needs meaningfully fewer iteration cycles than the SeqTCN work did — true if the representational hypothesis is right, not guaranteed.

**Expected benefit, quantified from this project's own trajectory:** headline EER has already improved from 0.51 (v9) → 0.49 (v11) → 0.32 (v13) across the representational upgrades tried so far. A materially different representation (AASIST) landing further gains on raw EER is a reasonable bet even independent of the confound question. The confound-specific gain (fixing the false-flag/miss-flag mechanism itself, not just aggregate EER) is the genuinely uncertain part — it has resisted two prior representational changes, so treat any confound improvement from this path as a hypothesis being tested, not an expected outcome to plan around.

## Phased plan with decision gates

```mermaid
flowchart TD
    Start["3-month start"] --> Spike["Phase 0: feasibility spike"]
    Start --> Calib["Calibration (Track 1)"]
    Spike --> Gate{"Gate: confound rows improve\nAND mobile hits real-time?"}
    Gate -- yes --> Train["Phase 1: full AASIST train"]
    Train --> Distill["Phase 2: mobile distill"]
    Train --> IntLaptop["Phase 3: vaani_laptop integration"]
    Distill --> IntMobile["Phase 4: vaani/mobile integration"]
    IntLaptop --> OnDevice["On-device verification"]
    IntMobile --> OnDevice
    OnDevice --> Ship["Ship both apps"]
    Calib --> Ship
    Gate -- no --> Pivot["Pivot: hard-negative data work"]
    Pivot --> ShipCalib["Ship calibration-only improvement"]
    Calib --> ShipCalib
```

Calibration ships on its own track regardless of which branch the gate takes — it's cheap enough not to wait on the architecture decision.

**The gate's pass criteria should be concrete, not a feeling:** re-run the existing `measure_confound.py` rows (the same ones v11–v13 have been scored against) and require the spike model to pass materially more than the current 5-of-14, not just "looks better." For mobile real-time: `EVAL-PROTOCOL.md`'s own windowing rule already sets the bar — the app scores a 3 s window every second, so inference has to comfortably clear well under 1 second per window on target hardware, not just complete eventually. Reusing these existing, already-calibrated thresholds (rather than inventing new success criteria for this project) keeps the gate honest and comparable to the project's own history.

## Risks, open questions, what would change this recommendation

- **`vaani/mobile`'s existing ONNX contract doesn't match `voice_guard`'s model at all.** `OnnxScorer` expects an input named `mel`, shape `[1, 48, 10]`. `voice_guard`'s deployed `v13` contract is `lfcc_sequence` `[1, 184, 60]` + `scalars` `[1, 6]` → `real_fake_logits`/`attack_type_logits`. Whatever ships to `vaani/mobile` — AASIST, SSL, or anything else — needs either to match the existing `mel_bridge.dart`/`window_pipeline.dart` contract or those files need to change too. Don't assume "drop in the new model" is a small step; check this before phase 4.
- **Two other known gaps are orthogonal to the entity-vs-style confound and won't be fixed by any architecture change alone:** accent cells (`en_native`, `hi_native`, etc.) sit at chance for every model version tried so far — this reads as a data-coverage gap (Hindi has far less training data than English), not a representational one. Unseen-channel generalization (`gsm_2g`, `tandem_xnet`) is similarly weak across the board. Neither should be expected to improve as a side effect of the AASIST work; they'd need their own data/coverage effort.
- **Training cadence risk:** each of `v11`→`v12`→`v13` took real multi-day sessions with iteration on failed gates. A new architecture may need a similar number of iteration cycles before it's trustworthy — confirm GPU/compute availability can sustain that cadence before committing to the phase-1 timeline.

**What would change this recommendation:**
- Spike shows AASIST (even distilled/quantized) can't hit real-time on target Android hardware → keep mobile on the current/refined classical pipeline, spend laptop-only effort on AASIST or SSL, and accept a laptop/mobile accuracy gap rather than force a shared architecture.
- Spike shows the confound gate doesn't move even on the laptop-scale model → stop spending on architecture entirely; redirect remaining budget to corpus/hard-negative collection (monotone real speech, naturally-paced fakes), since that would be strong evidence the bug is in the data, not the representation.

---

# Idea 2: Spiking Neural Networks vs. other alternatives

As of 2026-09-17.

## Short answer

SNNs are not a good fit for this project right now — not because the
representational idea is bad in the abstract, but because every practical
precondition for SNNs paying off is absent here, and the reasons are almost
entirely deployment/tooling, not modeling.

## What would have to be true for SNNs to earn their keep

| Precondition | SNN's actual selling point | This project's reality |
| --- | --- | --- |
| Neuromorphic or event-driven hardware target (Loihi 2, Akida, SynSense) | Spiking layers only realize their claimed power/latency advantage on hardware built for asynchronous spike events | Both deployment targets are conventional von Neumann hardware — the phone (Android CPU, ONNX Runtime) and the laptop. Simulated on ordinary silicon, a spiking layer is just a recurrent network run for T timesteps — no efficiency win, likely worse latency than the current SeqTCN. |
| Mature export path to the runtime this pipeline already committed to | — | ONNX has no native spiking-neuron op set; snnTorch/SpikingJelly/Lava models don't export cleanly to ONNX Runtime. Idea 1's proposed architecture leans hard on one shared ONNX pipeline across `vaani_laptop` and `vaani/mobile` — an SNN would break that, needing its own runtime (e.g. Lava, Loihi SDK) on top of everything else being built. |
| A mechanistic reason to expect it closes the entity-vs-style confound | Biologically-inspired temporal spike-timing coding | No anti-spoofing literature precedent. SNNs are proposed for energy/latency, not for exposing vocoder-specific artifacts (phase discontinuities, formant-transition smoothness) — the exact signal AASIST/raw-waveform CNNs target and pooled cepstral stats miss (`docs/CRITICAL-entity-vs-style-confound.md` §3). |
| Stable, well-trodden training recipe | — | Surrogate-gradient training is materially less mature than standard backprop; this project has already burned three training rounds (`v11`→`v12`→`v13`) on a conventional architecture without closing the confound — adding training-method risk on top of representation risk is the wrong lever to pull right now. |

Every row fails, and they fail independently — this isn't "one blocker to
solve," it's four separate reasons stacking.

## Verdict

Don't spike SNNs, not even as a cheap trial. Idea 1's Phase 0 spike is
worth doing (small-scale AASIST) precisely because it's cheap **and**
targeted at the actual known bug; an SNN spike would be more expensive (new
toolchain, no existing ONNX bridge) while being untargeted at the bug (no
reason to expect it addresses style-vs-entity confusion specifically). It
fails on cost before it even reaches the confound question.

The one scenario that would revive this: if `vaani_laptop`/`vaani/mobile`
ever targets always-on, battery-constrained inference on hardware that
actually has a neuromorphic co-processor (some newer phone SoCs are
starting to ship small always-on DSP/NPU islands, though not
spiking-specific ones) — that's a hardware decision, not a modeling one,
and nothing on the roadmap points that way today.

## If not SNNs — other neural options worth considering

Two candidates beyond Idea 1's AASIST (Approach A) and SSL fine-tune
(Approach B), aimed specifically at the open gap in Idea 1's plan: what the
**mobile** model should actually be if full AASIST doesn't distill down to
real-time (Idea 1 Phase 2, explicitly left open: "whether the laptop and
mobile models share one architecture family... isn't known yet").

| | RawNet2 / RawNet3 | LCNN (Light CNN) |
| --- | --- | --- |
| What it is | Raw-waveform sinc-filter frontend + residual blocks + GRU/attention pooling — same raw-waveform family as AASIST but explicitly designed lighter | Max-feature-map activation over log-spectrogram/CQT input; a long-standing ASVspoof baseline, already named and deferred once in this project (`docs/CRITICAL-entity-vs-style-confound.md` §4, track 3: "LCNN/frame-level sequence-model reconsideration") |
| Why it's relevant here | Purpose-built as a smaller sibling to the raw-waveform anti-spoofing family — a concrete mobile distillation *target* rather than an unspecified "distilled/pruned AASIST" | Already scoped once in this project's own history as an alternative to the pooled-MLP design before the SeqTCN path (`v11`-`v13`) was chosen instead; frame-level input, same class of fix SeqTCN was tried for |
| Track record on the actual problem | Established ASVspoof 2019/2021 baseline, published generalization results specifically for raw-waveform spoofing detection | Historically one of the strongest classical ASVspoof baselines pre-AASIST; weaker than AASIST/RawNet on the hardest generalization splits, but far cheaper to train and run |
| Where it fits the phased plan | Best candidate to name explicitly in Phase 2 (mobile distillation) instead of leaving it as "distilled AASIST, TBD" | Reasonable fallback if Phase 0's spike shows AASIST can't hit real-time on Android even distilled — swap only the mobile side, keep AASIST on laptop |

Neither changes the Idea 1 recommendation — they're refinements of Phase 2
(already flagged there as the least-resolved part of the plan), not
competitors to Approach A.

---

# Idea 3: Depth-staged auxiliary heads + adversarial style suppression

As of 2026-09-17.

## Origin and scope

Originating idea: train early layers on general voice-presence, middle
layers on human-vs-fake "affectation" features (to be determined via
feature-analysis research), and late layers on language/accent/dialect —
both to shrink the trainable area per fine-tuning run and, via gradient
analysis on the resulting weights, potentially expose the entity-vs-style
confound directly.

This is **backbone-agnostic** — it layers onto whichever model Idea 1's
gate selects (AASIST-family) or, if that spike fails, onto the
already-deployed v13 SeqTCN. It is not a competitor to Idea 1; it's a
training-procedure addition that most naturally lands inside Idea 1's
Phase 1 (full training), not a separate phase.

**Primary goal, as scoped with the user**: closing the entity-vs-style
confound is the real objective. Training-cost reduction (the original
motivating benefit) is real but secondary — it shows up in *later*
fine-tuning cycles, not the first training run (see Training procedure,
below).

## Why the literal staged-freeze version was revised

The original framing — train layers 1-3 on VAD, freeze, train layers 4-6
on human/fake with 1-3 frozen, freeze, train layers 7-9 on language — is
the cheapest version to run per-stage, but has no mechanism that forces
style out of the human/fake decision. It just trains in sequence; nothing
stops the "human/fake" layers from tangling in the same style cues that
already caused the confound in the flat (non-staged) v11-v13 architecture.
That's the same risk profile as Idea 1's "more capacity" attempts, which
already failed to close the gate twice.

On the depth ordering itself: the general → specific intuition (early
layers = universal, late layers = task-specific) is well-supported —
voice-presence really is the most general/lowest-level signal, and SSL
speech-model layer-probing studies (e.g. Pasad et al.) find phonetic
content dominates mid-depth while more contextual/language-level content
shows up later. So the original early→VAD, mid→human/fake, late→language
ordering was kept as the working hypothesis rather than revised.

## Approaches considered

| | A: literal staged freeze | B: staged auxiliary heads, joint training | C: B + adversarial style suppression (recommended) |
| --- | --- | --- | --- |
| What it is | Train early layers on VAD, freeze; train mid layers on human/fake, freeze; train late layers on language, freeze. | Shared trunk, three depth-tapped auxiliary loss heads (VAD / human-fake / language), all trained **jointly** in one pass, not sequentially frozen. | B, plus a small style-predictor head tapped at the same mid-band point as the human/fake head, behind a gradient-reversal layer (GRL), trained to make style features unpredictable from the shared mid-band representation. |
| Disentanglement mechanism | None beyond task ordering. | Task separation by depth alone — hopes disentanglement falls out of having distinct loss signals per band. | Explicit: GRL actively trains the mid-band features to be *uninformative* about style, directly targeting the documented style→false-flag mechanism. |
| Relation to prior failed attempts | Same risk class as "more capacity" (v11-v13), which didn't close the gate. | Weaker bet than C — task separation alone, no direct pressure against the known confound mechanism. | Structurally different from both prior remediation tracks (more features, more capacity) — actively suppresses the specific signal shown to cause the confound. |
| Training-cost benefit | Cheapest per-stage compute, available from the first run. | Comparable cost to training the backbone once, plus three small heads. Cost benefit shows up on *later* fine-tunes (freeze early+mid, retrain only late band). | Same as B — cost benefit deferred to later fine-tune cycles, not the first run. |
| Fragility | Frozen early layers may not carry forward the info later stages need; most fragile to get working. | Low — standard multi-task training. | Adversarial term can destabilize early training if λ_style isn't ramped; mitigated by gradual ramp schedule (DANN-style) and per-branch LR tuning. |

## Recommendation: Approach C

Chosen with the user: shared trunk, three depth-staged auxiliary heads
(VAD / human-fake / language) trained jointly, plus a gradient-reversal
adversarial style-predictor head at the human/fake tap point.

```mermaid
flowchart LR
    Input["Audio input"] --> Early["Early band"]
    Early --> Mid["Mid band"]
    Mid --> Late["Late band"]
    Early -- aux loss --> VAD["VAD head\n(voice / no-voice)"]
    Mid -- aux loss --> HF["Human/fake head"]
    Mid -- GRL, adversarial --> Style["Style predictor\n(pauseRatio, energyVariance,\npitch-variance, ...)"]
    Late -- aux loss --> Lang["Language/accent/\ndialect head"]
```

Loss: `L = L_vad + L_human_fake + λ_lang·L_language − λ_style·L_style_adv`,
all backpropagated jointly from step one — not staged/frozen, since
freezing the mid band before the adversarial term converges would lock in
whatever style/entity tangling already existed.

## Training procedure and the gradient-analysis hook

- **Joint training, not staged freezing.** All heads and the trunk train
  together in one pass.
- **Where the cost savings actually land**: not the first training run
  (comparable cost to training the backbone alone plus three small heads),
  but *later* fine-tuning cycles — e.g. a future pass aimed at closing
  Idea 1's known accent-cell gap can freeze early+mid bands and fine-tune
  only the late band + language head, instead of risking the whole
  network. This relocates the original "smaller trainable area per run"
  benefit from the first run to iteration on an already-jointly-trained
  model.
- **Gradient analysis, concretely** (this is the mechanism that could
  directly expose the confound, per the original idea): log, every N
  steps, the gradient norm each loss term contributes at each band
  (`∂L_human_fake/∂mid_band`, `∂L_style_adv/∂mid_band`,
  `∂L_language/∂late_band`) and their pairwise cosine similarity. If
  `∂L_human_fake` and `∂L_style_adv` are highly correlated early in
  training and that correlation drops as λ_style ramps up, that's
  quantitative, mid-run evidence the confound is unwinding — sharper and
  cheaper than waiting for `measure_confound.py`'s end-of-run rows.
  Conversely, if the two gradients show little correlation even before
  the GRL term is applied, that's useful negative evidence the confound
  isn't a simple linear entanglement at that depth — pointing back
  toward Idea 1's own fallback conclusion (the problem may be data
  coverage, not representation).

### Hyperparameters to tune

Structural:
- Band boundaries (where early/mid/late splits fall in the trunk)
- λ_lang, λ_style, and the GRL ramp schedule (shape, not just final value
  — linear vs. sigmoid/DANN-style vs. step changes how early the
  adversarial pressure bites and whether it destabilizes convergence)
- Auxiliary head capacity/depth (a single linear probe forces the trunk to
  do the disentangling work; a deeper MLP head lets a head solve its task
  without changing the trunk — the opposite of the goal here, so err small)

Secondary, but real:
- λ_vad and λ_human_fake themselves (not just λ_lang/λ_style relative to
  them — an easy VAD task can crowd out the human/fake gradient early
  unless downweighted)
- Per-branch learning rates (the style predictor in particular may need a
  different LR than the trunk to provide a useful adversarial signal
  without "winning" too fast)
- Style-predictor target definition (continuous vs. binned/discretized
  pauseRatio/energyVariance/pitch-variance — changes what "style-invariant"
  means)
- Batch composition/sampling (a batch with low style variance degenerates
  the adversarial signal; may need stratified sampling by style-variance
  bucket)

Roughly 8-10 tunable knobs total — worth a scoped hyperparameter sweep of
its own rather than picking defaults and running once.

## Risks, open questions

- **Untested combination.** Depth-staged auxiliary heads and
  GRL-adversarial training are each individually established techniques
  (deep supervision; domain-adversarial training / DANN) but this specific
  combination, against this specific confound, in this domain, has no
  known precedent here — same epistemic status as Idea 1's AASIST bet.
  Should go through a small-scale spike before a full commit, not be
  assumed to work.
- **Style-predictor target circularity, addressed**: the style features
  being predicted (pauseRatio, energyVariance, pitch-variance) are close
  cousins of the `v10_physio` features that didn't close the confound as
  *input* features. That's fine here — they're used as an adversarial
  *target to suppress*, not an added input — but worth stating explicitly
  so this isn't mistaken for repeating the v10 attempt.
- **Language/accent head inherits Idea 1's known data-coverage gap**
  (accent cells sit at chance for every model version tried so far). Idea
  3 doesn't fix that; it doesn't make it worse either.
- **Suggested spike scope**: small-scale run of the shared-trunk + 3-head
  + GRL setup on the existing corpus, checked via (a) the same
  confound-gate rows Idea 1 uses, and (b) the gradient-correlation
  diagnostic itself as an early, cheap secondary signal. This could
  plausibly run *inside* Idea 1's Phase 0 spike rather than as a separate
  phase, since both need the same small-scale training infrastructure.

## What would change this recommendation

- Gradient-correlation diagnostic shows `∂L_human_fake` and `∂L_style_adv`
  aren't meaningfully correlated even pre-GRL → the confound likely isn't
  a simple linear entanglement at this depth; redirect toward Idea 1's
  data/hard-negative pivot rather than iterating on this architecture.
- Adversarial term proves unstable across reasonable λ_style ramp
  schedules (training collapses or the human/fake head's own accuracy
  degrades) → fall back to Approach B (auxiliary heads without the
  adversarial term) and treat any confound improvement as incidental
  rather than designed-for.

---

# Idea 4: Manual dataset pruning as a controlled ablation

As of 2026-09-18. Full plan: `voice_guard/docs/DATASET-PRUNING-PLAN.md`
(recovered from an uncommitted file this session after the original
in-doc write-up was lost — see that file's own header).

## What it is

A **data-quality track, not an architecture track** — the natural
complement to Ideas 1/3's "if the confound doesn't move, redirect to
corpus/hard-negative work" fallback, run *before* concluding that fallback
is even necessary. Full manual review of every corpus source
(mislabeled / low_quality / redundant / off_scenario, tagged via a new
`prune_manifest.csv` column, never silently deleted), then three ordered
experiments on the result: a reason-category ablation (drop each category
in isolation, see which one actually moves held-out EER), a
scenario-relevance sweep (how much off-scenario data helps or hurts, as a
number not an intuition), and only then — on a fixed, pruned corpus — the
existing hyperparameter search.

Motivated by evidence already in hand, not a fresh hypothesis: every
attempt to add *more* training data (`attempt1`, `attempt2`, `ablation`,
`english_only`, state.md "Attempt 2 + follow-up ablations") made
cross-generator held-out EER *worse*, not better. A smaller, curated
corpus generalizing better than a larger noisy one is the natural next
hypothesis to test, using the same isolate-one-variable-at-a-time method
that already found the `'clean'`-recipe regression.

## Relationship to Ideas 1/3/5/6

Explicitly orthogonal to the entity-vs-style representational ceiling
(`docs/CRITICAL-entity-vs-style-confound.md`) — its own header says so: "a
null result here confirms that ceiling is architectural, not a pruning
failure." It doesn't compete with Ideas 1/3/5/6's architecture/objective
work; it's the thing that should probably run **first**, or at least in
parallel, because it's cheap relative to any retrain and because it
changes what every later architecture experiment is trained on. Running
Idea 1's AASIST spike (or Idea 6's OC-Softmax test) on a still-noisy,
unpruned corpus risks the same trap already hit once: crediting an
architecture change for a gain that was actually a data artifact (or
missing a real architecture gain because data noise masked it).

## What would change this recommendation

- If the reason-category ablation shows none of the four prune categories
  move held-out EER meaningfully, that's evidence the corpus itself isn't
  the bottleneck — proceed with Ideas 1/3/5/6's architecture/objective
  work on the existing corpus without waiting further on pruning.
- If it does move EER substantially, every architecture experiment above
  should be re-run (or at least re-validated) on the pruned corpus before
  trusting its numbers as final.

---

# Idea 5: Per-attack-family specialist classifier heads

As of 2026-09-18.

## Origin and scope

Originating idea: instead of one model doing everything, have "one model
serving each individual task," so a failure can be isolated per sub-model
rather than debugged as one opaque score. Refined through discussion down
to a concrete, scoped version: split the single human/fake decision into
three specialist binary heads — **TTS-vs-human**, **voice-clone-vs-human**,
**other-vs-human** — combined into the final score, rather than splitting
the whole pipeline (feature extraction and gating are not models today;
see the discussion that scoped this down, this session).

This is **backbone-agnostic and additive to Idea 3**, not a competitor to
either Idea 1 or Idea 3. Idea 3's `MultiHeadSpike` already taps a single
`human_fake_head` off the mid-band representation (`spike_model.py`,
`docs/superpowers/plans/2026-09-17-vaani-model-architecture-phase0-spike.md`
Task 3). Idea 5 proposes replacing that one 2-way head with three 2-way
specialist heads reading the same mid-band pooled tensor, combined into
one fake-probability output — a refinement one level inside Idea 3's
architecture, not a parallel design. If Idea 1's Phase 0 spike doesn't
clear its gate and the project stays on the deployed v13 SeqTCN instead,
Idea 5 lands the same way on top of v13's existing pooled representation.

## Why this is a real complement, not just relabeling

The corpus already documents family-specific pathology that one shared
head has to average over:
- `en_foreign`/`hi_native`/`hi_foreign` have a sample-rate/generator
  confound baked into their fake side (XTTS-v2 fakes natively 22050Hz vs.
  16000Hz reals) — a TTS-specific artifact, invisible to a voice-clone
  specialist and vice versa.
- `fake2021` (ASVspoof2021, the largest single source) behaves very
  differently in chunk-survival rate and EER than DECRO, CodecFake, or
  MLAAD — different generator families, different failure modes.
- MLAAD alone spans 116 distinct TTS architectures; lumping all of it plus
  XTTS voice-cloning plus CodecFake neural-codec-resynthesis into one
  binary decision forces the shared head to find one axis that works for
  all three attack families at once, which is exactly the kind of pressure
  that produced the style-variance shortcut in the first place
  (`docs/CRITICAL-entity-vs-style-confound.md`).

Splitting the decision doesn't force any of that averaging. Each
specialist can key on family-specific artifacts (TTS vocoder smoothness,
voice-conversion spectral discontinuities, codec quantization noise)
instead of one blended boundary — and directly delivers the original
motivation: a failing specialist localizes which attack family degraded,
instead of one opaque combined score moving for an unknown reason.

## What it does NOT fix

Layered on top of Idea 1's pooled-MLP-era 63-feature vector, three heads
reading the identical style-sensitive input would still be entity-blind in
the same way the single MLP is — this was the caveat given when the idea
was first floated this session, and it still holds for that representation.
It matters less layered on Idea 3's frame-level trunk instead (richer
mid-band representation, not just three pooled scalars), but it is still
not a targeted fix for the entity-vs-style confound the way Idea 3's GRL
term is — it's an orthogonal generalization/debuggability lever, not a
replacement for Idea 3's adversarial suppression.

## Approaches considered

| | A: single shared head (status quo) | B: 3 specialist heads + max-score combiner (recommended) | C: 3 specialist heads + learned meta-combiner |
| --- | --- | --- | --- |
| What it is | Idea 3's current `human_fake_head`, one 2-way decision. | Three 2-way heads off the same mid-band tap; final fake-probability = max of the three specialist fake-probabilities. | Same three heads, plus a small linear/MLP combiner trained on the three specialist logits (optionally + the pooled mid-band vector) instead of a fixed max. |
| Debuggability | None — one score, no attribution. | Full — each head's own accuracy/EER is directly loggable and gate-able per attack family. | Same as B, plus the combiner itself can be inspected (learned weights) but is one more component to debug. |
| Param/compute cost | Baseline. | Trivial — two more tiny linear heads (`nn.Sequential(Linear(2*mid_ch,32), ReLU, Linear(32,2))` ×2 more), same mid-band pooled input already computed. | B's cost plus a small combiner (a handful of parameters) — still trivial in absolute terms. |
| Calibration risk | None (single head, single threshold). | Real: if the three specialists' score distributions aren't comparable in scale, `max` over-triggers or under-triggers relative to a single-head threshold — needs its own threshold calibration pass in `evaluate.py`, not reuse of v13's existing threshold. | Lower — the combiner can learn to correct for scale differences between heads, at the cost of one more thing to overfit/miscalibrate on a small held-out set. |
| Label requirement | None beyond existing binary real/fake labels. | Per-fake-sample attack-family label (tts / voice_clone / other), mapped from existing metadata (ASVspoof attack-type tags, XTTS = voice_clone, CodecFake/DECRO/MLAAD → other or finer-grained if their own attack-type tags are usable) — real samples don't need a family label, only fake ones do. | Same as B. |

## Recommendation: Approach B first, escalate to C only if calibration demands it

Start with the max-score combiner — it's the cheapest possible version and
directly tests whether specialization helps before spending effort on a
learned combiner. Reuse `dataset.IGNORE_ATTACK_TYPE`'s existing masking
convention (already used for leave-out attacks) for any fake sample whose
attack family can't be confidently mapped, rather than forcing a guess
into "other." Escalate to Approach C only if `evaluate.py`'s confound/EER
gates show the three specialists' score distributions are poorly
comparable on a fixed threshold — don't build the learned combiner
speculatively.

## Risks, open questions

- **"other" is a grab-bag category** (CodecFake's neural-codec-resynthesis
  plus any future/unseen generator that isn't cleanly TTS or voice-clone).
  A specialist trained on a heterogeneous bucket like this is the weakest
  of the three by construction, and it's also the one most likely to see a
  genuinely novel attack in deployment — the case that matters most. Don't
  expect it to perform as well as the TTS/voice-clone specialists, and
  don't let its EER stand in for "we're covered against unknown attacks."
- **Needs per-family gate rows added to the eval harness** (`evaluate.py`,
  `measure_confound.py` lineage) — today's gates score one binary decision;
  scoring three specialists plus the combined output means extending, not
  replacing, the existing four-gate protocol.
- **Not independently useful without attack-family labels on the fake
  side of the corpus** — mostly already present (see label-requirement row
  above) but not audited yet for completeness/consistency across all
  fake sources; that audit is a prerequisite task, not part of this design.

## What would change this recommendation

- If Idea 1's Phase 0 spike doesn't clear its gate and the project pivots
  to corpus/hard-negative work instead of any architecture change (Idea
  1's own stated fallback), Idea 5 still applies on top of whatever model
  ships next — it isn't gated on Idea 1's outcome the way Idea 6 partly is
  (see below).
- If the attack-family label audit turns up too few labeled fakes in one
  family to train a specialist meaningfully (most likely risk: "other"),
  scope that specialist down or merge it back into a two-way split
  (tts-vs-human, everything-else-vs-human) rather than forcing three heads
  regardless of data support.

---

# Idea 6: Post-feature-extraction backend — one-class learning, not just a bigger two-class classifier

As of 2026-09-18. Based on a deep-research pass into audio anti-spoofing
classification/pattern-recognition backends (sources at the end).

## Origin and scope

**Correction to how this was originally scoped:** the trunk swap from a
pooled feature vector to something richer is not an open question — it's
already done. `voice_guard/docs/superpowers/plans/2026-09-11-frame-level-
seq-model-and-attack-type-plan.md` replaced the pooled 66-feature MLP with
a per-frame LFCC sequence CNN back on 2026-09-11; that shipped as
`v11`→`v13`, and v13 (frame-level SeqTCN + 6 physio scalars) is the
**currently deployed** model. Idea 1's AASIST-family raw-waveform trunk is
a *further*, still-hypothetical swap on top of that — and, checked this
session, it has **zero code written**: no `spike_model.py`/`raw_pcm_cache.py`/
etc. exist on disk, only the unexecuted Task list in
`docs/superpowers/plans/2026-09-17-vaani-model-architecture-phase0-spike.md`.
There is no evidence yet — from that spike or anywhere else — that
raw-waveform AASIST decidedly helps; it's an explicit bet gated behind
Task 10's not-yet-run feasibility spike, precisely because the two
remediation tracks already spent (physio features, then frame-level SeqTCN
capacity) both failed to close the confound gate. Treat "AASIST trunk" as
unresolved, not decided, anywhere this doc references it.

Idea 6 asks a different question, one level further down the pipeline:
independent of which trunk wins, what should the final **decision rule**
be — is a bigger/different two-class classifier actually the right
complement to richer features, or is the two-class framing itself part of
the problem?

## The core idea: reframe human/fake as one-class human-verification, not two-class discrimination

A standard binary classifier (softmax/cross-entropy over "human" vs. "fake"
logits — what every version tried so far, v9 through v13, and what Idea 3's
`human_fake_head` still does) is free to pick *whatever axis best separates
the two training classes*. Empirically, on this project, it picked
style/pacing variance (`docs/CRITICAL-entity-vs-style-confound.md`) — not
because the data lacked entity-level signal, but because the objective
never required the model to find it specifically.

**OC-Softmax** (Zhang et al. 2021, "One-Class Learning Towards Synthetic
Voice Spoofing Detection"; refined by a 2024 follow-up on adaptive centroid
shift, arXiv:2406.16716) trains instead to bound a compact embedding
manifold of *genuine human speech only*, and scores distance from that
manifold as the fake-probability. This is a **training-objective change on
top of an embedding**, not necessarily a much bigger model — it can sit on
top of whichever trunk Idea 1 lands on (AASIST-lite spike embedding, or the
deployed v13 SeqTCN's pooled representation) with a loss-function and
final-head change, not a full architecture swap. That makes it
meaningfully cheaper to test than Idea 1's own AASIST bet, while targeting
the confound mechanism directly rather than hoping a richer representation
incidentally fixes it.

## Why this is different from — and complements — Idea 3's GRL approach

Idea 3 suppresses style information *within* a two-class objective (GRL
makes the mid-band representation unable to predict style, while a
separate head still does two-class human/fake discrimination on what's
left). Idea 6 instead changes what "fake" means at the objective level —
distance from a genuine-human manifold, not a boundary between two labeled
classes. **These are not mutually exclusive.** A combined spike variant —
OC-Softmax objective on an embedding that also passes through Idea 3's GRL
style-suppression at the same tap point — tests both levers at once and is
a natural extension of the Phase 0 spike infrastructure already scoped
(same `measure_confound.py` gate, same gradient-diagnostic hook could be
adapted to log style-predictability of the one-class embedding too).

## Approaches considered

| | A: status quo (two-class, whatever trunk) | B: OC-Softmax one-class objective (recommended) | C: AASIST-L / SpAArSIST full architecture escalation |
| --- | --- | --- | --- |
| What it is | Binary cross-entropy over human/fake logits — pooled features (v9 only) or the deployed frame-level SeqTCN (v11-v13), or Idea 3's mid-band tap if that spike ships. | Bound a genuine-human embedding manifold; score = distance from it. Loss/head change on top of an existing trunk. | Purpose-built graph-attention spectro-temporal backbone (Jung et al. 2021; SpAArSIST 2026 follow-up trims compute further, 85K params / ~332kB, 4.64% EER on In-the-Wild for the lite variant). |
| Targets the confound directly? | No — this is the objective that produced the confound. | Yes, structurally — style variance within humans still sits inside the "normal" manifold; the model isn't rewarded for using it as the primary signal. | Indirectly — targets vocoder-artifact-level signal (phase discontinuities, formant-transition smoothness) that no pooled statistic exposes, which is a different (also promising) lever, not a competing explanation. |
| Integration cost relative to already-deferred LCNN | N/A (baseline) | **Lower** — loss/head change on an existing embedding, no new frontend/architecture required. | **Same class of cost as LCNN, already deferred once for exactly this reason** — needs frame-level input, no ONNX export precedent in this stack, own training/iteration cycles. This is already Idea 1 Approach A / Idea 2's RawNet2/LCNN comparison table — Idea 6 doesn't duplicate that decision, just names it as the escalation path if B underperforms. |
| Feature-extraction dependency | None (already shipped). | **None — testable today.** v13's frame-level SeqTCN embedding is already deployed; OC-Softmax is a loss/head change on top of it, no new feature-extraction work required. If the AASIST spike ever ships, swap in its embedding instead, but B doesn't need to wait on that. | Hard dependency on raw-waveform input — gated on Idea 1's Phase 0 spike, which is **unexecuted** (no code on disk as of 2026-09-18) and not decided to help (see the correction above) — this is a real, currently-unresolved blocker for C, not a formality. |

## Recommendation: Approach B first, testable today on v13, no need to wait on Idea 1's gate

Test OC-Softmax before reaching for AASIST-L/SpAArSIST (Approach C) —
it's cheaper, targets the confound by construction rather than by hope,
and doesn't duplicate the AASIST decision already tracked under Idea 1.
**It doesn't have to wait on that decision either**: v13's frame-level
embedding is already deployed, so Approach B can be spiked against it
immediately, independent of whether Idea 1's AASIST Phase 0 spike ever
runs or clears its gate. If that spike *does* ship an embedding (from the
AASIST-lite trunk), re-test OC-Softmax on top of it too as a natural
Phase-0-adjacent experiment, ideally as a fourth combination alongside
Idea 3's B/C variants (B: OC-Softmax alone; C: OC-Softmax + Idea 3's GRL
together) — reusing the
same `measure_confound.py` gate and held-out split, no new infrastructure.

**One near-zero-cost methodology add regardless of which backend wins:**
a 2026 paper, "An Intervention-Based Framework for Shortcut Diagnosis in
Spoofing Countermeasures" (arXiv:2607.03150), proposes controlled acoustic
interventions (perturbing non-speech intervals, spectral content, energy)
to distinguish genuine generalization from shortcut-driven confounds —
methodologically close to how this project already root-caused the
entity-vs-style problem by hand. Worth folding into `EVAL-PROTOCOL.md` as
a pre-ship gate regardless of which backend (B or C) ships, so a claimed
confound fix has to survive deliberate probing, not just move the EER
number.

## Risks, open questions

- **Classical statistical backends (GMM-UBM, x-vector+PLDA) were
  considered and deprioritized**, not for cost but because neural backends
  already outperform them for logical-access spoofing (TTS/voice-clone) —
  this project's actual threat model. GMM-UBM stays competitive mainly for
  physical-access/replay spoofing, not relevant here.
- **One-class framing has its own failure mode**: if the "genuine human"
  manifold is defined too broadly (e.g. it has to cover every accent/
  register in the corpus, including the already-known thin Hindi/foreign-
  accent cells), the boundary may end up loose enough that it doesn't
  actually exclude natural-sounding fakes — the same problem in a
  different shape. This needs the same per-cell EER breakdown Idea 1
  already tracks, not just an aggregate number.
- **Untested combination** (OC-Softmax + GRL together): each piece has
  independent literature support, but stacking them on this specific
  corpus/confound has no precedent — treat as a spike, not an assumed win,
  same epistemic status Idea 3 already holds itself to.

## What would change this recommendation

- OC-Softmax alone closes the confound gate → don't bother with the
  combined GRL variant, ship the simpler single-objective change.
- OC-Softmax doesn't move the confound gate at all, even combined with
  GRL → that's evidence the confound is more about data coverage
  (per-cell accent/register gaps) than any objective-level fix, redirecting
  toward Idea 1's own data/hard-negative pivot rather than further backend
  experimentation.

## Sources

[AASIST arXiv:2110.01200](https://arxiv.org/abs/2110.01200) ·
[SpAArSIST arXiv:2606.11674](https://arxiv.org/abs/2606.11674) ·
[One-Class Learning Towards Synthetic Voice Spoofing Detection (Zhang et al. 2021)](https://www.researchgate.net/publication/351174426_One-Class_Learning_Towards_Synthetic_Voice_Spoofing_Detection) ·
[One-class learning with adaptive centroid shift, arXiv:2406.16716](https://arxiv.org/pdf/2406.16716) ·
[Intervention-Based Framework for Shortcut Diagnosis, arXiv:2607.03150](https://arxiv.org/abs/2607.03150) ·
[Impact of Channel Variation on One-Class Learning, arXiv:2109.14900](https://arxiv.org/pdf/2109.14900)
