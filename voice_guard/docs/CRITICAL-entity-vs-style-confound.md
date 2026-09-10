# ⚠️ CRITICAL — the model confuses speaking STYLE with speaker ENTITY

**Read this before doing any further VoiceGuard model work, on any device.**
This is the most concerning open finding from the 2026-09-10/11 session —
more fundamental than any of that session's data-pipeline bugs, and **not
fixed by anything shipped so far**. It is the load-bearing reason the
current model, however improved, should not be treated as production-ready
without a deliberate decision on one of the remediation tracks in §4.

**One-line summary:** the model does not detect "human vs. AI" — it detects
"does this clip's pacing/energy/pitch-variability statistics match the
training set's typical real speech." Those are two different things, and
they come apart exactly when a real human speaks in an unusual register
(controlled, recitation-style, monotone) or a fake happens to have natural-
sounding pacing. **Confirmed quantitatively on real held-out data, not just
anecdote** — see §2.

---

## 1. How this was found

Live on-device testing, post-noise-fix (see
`2026-09-10-noise-fix-session-report.md` for that fix's own validation).
User spoke into Live Mic Test in two deliberately different registers:

- **Tight, high, eloquent, "oral recitation" style** → score spiked
  immediately to the **60s–80s** (false "AI" territory), despite being
  genuine human speech throughout.
- **Lower, baritone, calmer, slower delivery with longer pauses** →
  reliably scored **0–20s** (correctly "human"), same speaker, same
  content register otherwise.
- A genuine AI voice-clone clip (`model_training/test_assets/
  ai_clone_test_clip.wav`, a real In-the-Wild deepfake, held out from all
  training/eval) scored consistently high, climbing from the low 80s to a
  sustained **97–100%**.

The user's own diagnosis, verbatim, was correct: *"the model judges both
humans and AI according to the same parameters — it doesn't know what
entity-specific markers to look for yet."*

## 2. Quantified confirmation (not just 3 anecdotes — done same session)

The live test is 3 data points from one speaker in one sitting. Before
trusting it, the exact mechanism was checked against the model's own real
held-out benchmark set (`data/real_itw_held`, hundreds of distinct clips)
and its fake held-out set (`data/fake_itw_held`), using the actual deployed
weights (`runs/voice_guard_v9_noisefix_final/model.pt`).

**On real (genuinely human) speech**, split at the median of each prosody
feature:

| feature | low half → mean fake-prob | high half → mean fake-prob |
|---|---|---|
| `energyVariance` | **25.1%** | **12.9%** |
| `pauseRatio` | 14.7% | 23.6% |
| `zcrVariance` | 13.1% | 24.9% |

The worst false-positive-risk real clips (top 10% by fake-probability,
0.55–0.995) are specifically characterized by unusually high `pauseRatio`
(0.47 vs. 0.38 baseline) and high `zcrVariance` (0.021 vs. 0.014) — a
*delivery-style* signature, not anything about who is actually speaking.

**On fake (genuine AI) speech, the mirror image**: of 1,592 held-out fake
windows, median fake-probability is 0.941 (broad, reliable detection — the
Obama clip was not a one-off). But the **20.3% that are missed** are
systematically the ones with *lower* `zcrVariance` (0.014 vs. 0.021 for
caught fakes) and lower `pauseRatio` (0.34 vs. 0.37) — fakes that happen to
have natural-sounding pacing get a pass, for the same reason a controlled-
register human gets flagged.

**This is one axis, working in both directions, and it is now proven, not
suspected.**

## 3. Why this is more serious than the session's other bugs

Every other bug found this session (chunk-yield unevenness, the
`split_by_source` basename bug, the `'clean'`-recipe misuse, the sample-
rate/silence confounds in the accent cells) was a **data-pipeline defect**
— fixable by fixing the data or the training recipe, without touching the
model's representational capacity. This one is different: it is a direct
consequence of the **feature set's design**, flagged abstractly earlier the
same session (`docs/superpowers/specs/2026-09-10-model-regression-design.md`,
idea I1: *"the 63-d mean-pooled feature vector has a hard information
ceiling"*) and now demonstrated concretely, by a human, in real time.

The model's only temporal-dynamics signal is three pooled scalars —
`pauseRatio`, `energyVariance`, `zcrVariance` — computed over a 3-second
window with all time-structure otherwise collapsed (60 LFCC coefficients
are themselves mean-pooled per window). These scalars measure **how much a
speaker's delivery varies**, which is a stylistic choice available to any
human, not a physiological property of a human vocal tract vs. a vocoder.
**No amount of more/cleaner training data fixes this** — it is a ceiling on
what this specific 63-number representation can ever distinguish, not a
gap in what it has seen. This is why it belongs in a category by itself:
every other finding this session was "fix the data, retrain"; this one is
"the representation itself needs to change."

## 4. Remediation tracks — not yet started, ranked by cost

1. **Cheap, partial, immediate**: per-speaker relative calibration. A brief
   on-device "voice baseline" capture at call/session start; score
   subsequent audio as deviation from *that speaker's own* baseline rather
   than a fixed global threshold. Does not fix the underlying feature
   confound — a naturally low-variance speaker would still need to be
   "calibrated in" — but stops permanent false-flagging of a consistent
   individual.
2. **Medium cost**: add genuine physiological features — jitter, shimmer,
   harmonic-to-noise ratio (standard, well-understood measures from speech
   pathology, computable without a full architecture change) — alongside
   the existing 63, giving the model *some* signal that isn't purely a
   style statistic.
3. **High cost, most likely to actually resolve it**: the LCNN/frame-level
   sequence-model reconsideration already flagged and deferred earlier this
   session (see `voice_guard/state.md`, "LCNN backup" section). A model
   that sees the real time-series, not a pooled summary of it, can in
   principle learn vocoder-specific artifacts (formant-transition
   smoothness, phase artifacts) that exist only in the trajectory, not in
   any variance statistic derived from it.

**Nothing here is implemented.** This document is the starting point for
that decision, not a resolution of it. Whoever picks this up next should
decide which track (if any) to pursue before doing further model work on
this project — continuing to retrain the current 63-feature representation
on more/different data will not move this specific problem, per §3.

**Track 3 now has a full scoping/design pass** — see
`voice_guard/docs/superpowers/plans/2026-09-11-frame-level-sequence-model-plan.md`
(2026-09-11). Key finding from that scoping: both the Python and Dart
feature-extraction code already compute the full per-frame LFCC sequence
and then discard it via mean-pooling — this makes a custom, India-tuned
frame-level model a smaller lift than importing an academic architecture
like LCNN would be. Plan only, not yet implemented.

## 5. Cross-references

- `voice_guard/state.md` — full project state; this file's existence is
  flagged at the top of its "Current status" section.
- `voice_guard/docs/2026-09-10-noise-fix-session-report.md` — the session
  that produced the currently-deployed model and did the live testing that
  surfaced this finding.
- `voice_guard/docs/superpowers/specs/2026-09-10-model-regression-design.md`
  — the earlier, more abstract version of this same representational-
  ceiling concern (idea I1), from the code-orange investigation into why
  more training data kept making cross-corpus generalization worse.
