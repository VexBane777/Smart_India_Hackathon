# Attack-type differentiator (TTS vs. Voice Conversion) — Design

**Date:** 2026-09-11
**Status:** Design approved by user in conversation. Not yet an
implementation plan (see "Next step" below).
**Relationship to other tracks:** this is **remediation track 4** in the
entity-vs-style confound lineage (`docs/CRITICAL-entity-vs-style-
confound.md`), and it is **integrated with track 3**
(`docs/superpowers/plans/2026-09-11-frame-level-sequence-model-plan.md`,
frame-level CNN), not built on the current mean-pooled MLP. Track 2
(physio features, `docs/superpowers/plans/2026-09-11-physiological-
features-plan.md`) is also folded in, per that plan's own §2.4 note that
physio features can later become per-frame/scalar channels in the track 3
architecture.

## 1. Motivation

Manual testing (2026-09-11) surfaced a real gap: a voice-**transformed**
(voice-conversion) clip of a real person's speech reliably triggers AI
detection, but plain **TTS**-synthesized clips of the same general quality
do not. This means the model's fake-detection signal is currently more
sensitive to voice-conversion artifacts than TTS artifacts — an
undocumented, uneven blind spot distinct from (but related to) the
entity-vs-style confound already tracked. Confirmed empirically via
`voice_guard/model_training/test_assets/` (`tts_elevenlabs_sample.wav`,
`tts_chattts_sample.wav` vs. `voice_conversion_asvspoof_a17.wav`) — see
that directory's README for provenance.

Goal (per user decision): **both**
1. Use attack-type as an auxiliary training signal (multi-task learning)
   to push the model toward features that generalize across attack
   families, rather than overfitting to whichever family dominates the
   training mix.
2. Once reliable, surface the attack-type distinction in the UI (e.g. "AI
   DETECTED (voice conversion)" vs. "AI DETECTED (synthetic voice)") — real
   fraud-relevance, since live voice conversion during an active call is a
   different threat model than a pre-recorded TTS clip played over the
   line.

**Explicitly out of scope:** replay attacks (a recording of the genuine
target's own voice played back — no AI involved at all; different feature
space — channel/playback artifacts, not synthesis artifacts — and no
labeled data exists in this corpus for it, since only the ASVspoof LA
partition was ever ingested, not the PA/replay partition). If replay
detection is wanted later, it is a separate effort with its own
data-sourcing work, not folded into this track.

## 2. Training consolidation decision

Track 3's per-frame representation is **not compatible** with the
mean-pooled 66-d vector track 2 currently extracts — they are different
feature representations, not different models over the same input. This
means:

- The `v10_physio` training run in flight as of this writing (mean-pooled
  MLP + physio features) is **diagnostic-only**: it validates
  `extract_physio` works correctly at full-corpus scale and produces a
  standalone measurement of whether physio features alone shrink the
  entity-vs-style confound gap (via `measure_confound.py`). Its resulting
  `model.pt`/`model.onnx` will **not** be deployed
  (`assets/models/voice_detector.onnx` stays unchanged) — recorded in
  `state.md` once it finishes.
- Tracks 2 (physio, as extra channels), 3 (frame-level CNN), and 4 (TTS/VC
  head) are implemented **together** and trained in **one** final
  consolidated run, rather than three separate retrains. This avoids
  re-extracting features three times and avoids shipping an intermediate
  model that's immediately superseded.

## 3. Data labeling

Per-file attack-type ground truth is patchy across the corpus — this is
the central implementation risk, not the model architecture:

| Source | Labelable? | How |
|---|---|---|
| `data/fake2021` (ASVspoof2021 LA eval + 2019 LA dev, 185K+ files) | **Yes** | `data/asvspoof2021_la/LA-keys-full/keys/LA/CM/trial_metadata.txt` — format `speaker_id utt_id codec tx attack_id key trim subset`, still on disk. Attack IDs present: A07–A19 (confirmed via `grep`). |
| `data/fake` (ASVspoof2019 LA train, 22,800 files) | **Recoverable, not yet recovered** | Original protocol file was deleted in an earlier disk-cleanup pass (audio itself is kept). It's a small text file — re-download just the protocol from the same Kaggle source (`anishsarkar22/asvpoof-2019-dataset-la`) to recover A01–A19 labels; do not re-pull the multi-GB audio archive. |
| `data/mlaad_en500` | **Yes, trivially** | Every entry is TTS by construction (MLAAD is a TTS-only spoofing dataset) — `metadata.csv`'s `architecture` column confirms, but the whole directory maps to `tts` regardless. |
| `data/accents/fake/*` (XTTS-v2, YourTTS, MMS-TTS-hin) | **Yes, trivially** | All are TTS (text-driven synthesis) by construction, including the "cloning" variants — voice cloning via speaker-embedding conditioning is still TTS, not conversion of an existing audio signal. Verified against each `gen_*.py` script's own docstring. |
| `data/fake_itw_train`/`fake_itw_held` (In-the-Wild) | **No** | Real-world scraped deepfakes; generation method is not documented per-clip in the released dataset. |
| CodecFake (`data/accents/fake/en_native` subset) | **No** (and doesn't fit the taxonomy) | Neural-codec resynthesis — architecturally neither TTS nor VC (no text input, no identity-conversion model). |

**Attack-ID → {tts, vc} mapping** (ASVspoof2019 LA taxonomy, A01–A19):
draft below, based on the well-established convention that A05/A06 and
A17–A19 are voice-conversion systems and the rest are TTS, with A13 a
TTS+VC hybrid. **This table MUST be verified against the official
ASVspoof2019 evaluation plan (Wang et al. 2020) before use** — an error
here silently mislabels thousands of training examples, so implementation
must not proceed on this table alone without that check:

```
A01–A04, A07–A12, A16   -> tts
A05, A06, A13, A14, A15 -> vc      (A13 is a TTS+VC hybrid; verify whether
                                     to bucket as vc, tts, or exclude)
A17, A18, A19           -> vc      (confirmed empirically: these are the
                                     samples used in test_assets/)
```

Unlabelable sources (`fake_itw_*`, CodecFake) get attack-type label
`unknown`, excluded from the auxiliary loss via `ignore_index` — they
still train the primary real/fake head normally.

## 4. Architecture

Extends track 3's frame-level CNN (`(184, 60)` LFCC sequence → Conv1d
stack → pooled embedding), not the current MLP:

- **Shared trunk**: track 3's CNN embedding, concatenated with the 3
  prosody scalars and the 3 physio scalars (jitter/shimmer/HNR — folded in
  as extra scalar inputs alongside prosody, per track 2 §2.4's own note;
  whether physio becomes per-frame channels instead is an open
  implementation question for whoever picks up track 3, not decided here).
- **Head 1 (existing)**: real/fake, 2 logits, unchanged semantics.
- **Head 2 (new)**: attack-type, 2 logits (tts/vc), a small linear layer
  off the same shared embedding — cheap to add, no separate backbone.

## 5. Training — masked multi-task loss

- Head 1 loss: `CrossEntropyLoss` over every example, as today.
- Head 2 loss: `CrossEntropyLoss` with `ignore_index` for every real
  example and every unlabeled fake (`unknown`) — only backprops through
  genuinely-labeled TTS/VC examples.
- Total loss: `real_fake_loss + lambda * attack_type_loss`, `lambda`
  starting at `1.0`, tuned empirically — watch for the auxiliary task
  *helping* head 1's held-out EER (the hoped-for effect, since it forces
  attack-family-invariant features) vs. *hurting* it (if attack-type is
  fighting the primary task); this is an empirical question for the
  implementation phase, not decided here.
- Gate: same discipline as every other track — do not deploy the combined
  model unless it beats the currently-deployed baseline on held-out EER
  and noise-FPR, in addition to whatever attack-type accuracy it achieves.

## 6. UI integration

- Attack-type sub-label only ever shown **conditional on** the primary
  head already firing `ALERT` — never shown standalone, never contradicts
  or precedes the primary verdict.
- Only shown when head 2's own softmax confidence clears a threshold
  (draft: 70%, tune during implementation) — below that, fall back to the
  current generic "AI DETECTED" text rather than presenting an uncertain
  sub-classification as certain.
- Exact copy/placement is a UI-implementation decision, not fixed here —
  should follow whatever the new shadcn design system
  (`lib/design/tokens.dart`, `ShadBadge`, etc., merged 2026-09-11) does for
  secondary status labels, for visual consistency with the rest of the
  just-redesigned UI.

## 7. Testing

- Unit tests for the attack-ID → {tts, vc, unknown} mapping function
  (pure, easily testable in isolation).
- Unit tests for the masked loss (verify `ignore_index` actually excludes
  real/unknown examples from head 2's gradient — a synthetic-batch test,
  not an integration test).
- The existing `check_corpus.py`/`dataset_audit.py` preflight tooling
  should be extended to also report attack-type label coverage per source
  directory (what fraction of each dir's fakes got a real tts/vc label vs.
  falling to `unknown`) — visibility into how much of the corpus can even
  participate in the auxiliary task before committing to a training run.
- On-device verification: re-run the exact `test_assets/` TTS vs. VC clips
  through Live Mic Test once deployed, confirm the attack-type sub-label
  matches expectation for each.

## 8. Open questions for the implementation plan

- Exact A01–A19 mapping needs verification (§3).
- Whether `data/fake`'s protocol re-download is a hard prerequisite or the
  auxiliary task can ship usefully on `data/fake2021` + MLAAD + accents
  alone (a smaller but still substantial labeled subset).
- Whether physio scalars become per-frame channels or stay scalar-appended
  in track 3's architecture (§4) — a track 3 implementation decision this
  spec doesn't resolve.
- Final `lambda` weighting and confidence threshold (§5, §6) — empirical,
  tuned during implementation, not fixed here.

## Next step

Invoke `superpowers:writing-plans` to turn this spec (combined with track
3's existing plan document) into a single implementation plan covering:
track 3's CNN + track 2's physio-as-channels integration + this
document's TTS/VC head, data labeling, and the consolidated training/
deploy-gate pass described in §2.
