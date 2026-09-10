# VoiceGuard model regression — why more training data made it worse (design)

**Date:** 2026-09-10
**Topic:** why every retraining attempt this session on a larger, more diverse corpus regressed
cross-generator held-out EER, even after fixing every concrete bug found — and what the wide-net
literature review says the actual, still-unaddressed root cause is.
**Status:** Design. Written through the code-orange workflow (invent uninformed → wide-net
literature review → revise marking every idea `kept/modified/replaced/refuted` with its citation
→ spec → plan). Pre-research artifact: [`notes/2026-09-10-model-regression-invention-uninformed.md`](notes/2026-09-10-model-regression-invention-uninformed.md).
**Model/effort/exec:** Sonnet 5 · inline, same session.

---

## 0. What this is, and what it is not

**It is** a re-confrontation of six full training runs' worth of evidence (all in
`voice_guard/state.md`) against the actual published/documented state of the art on why
synthetic-speech detectors fail to generalize across corpora — because two rounds of
concrete, verified, individually-correct bug fixes (a channel-recipe misuse, a sample-rate
generator confound) not only failed to recover the baseline, the second one produced the
single worst result of the whole session.

**It is not** a proposal for a new model architecture yet — Tier A ideas below are flagged for a
follow-up plan, not implemented here. This document's job is to say, with citations, *what is
actually going on*, and to leave behind a working detector (`dataset_audit.py`'s new
`find_acoustic_shortcuts`) that already found something real.

---

## 1. The headline result of this document

**The wide-net literature review found this project's exact failure mode is a named, published
phenomenon in the audio-deepfake-detection field — not a mystery specific to our pipeline — and,
searching our own corpus for the specific confound class the literature named, found a second,
still-unfixed shortcut that plausibly explains why the first fix didn't help.**

Kwak et al. 2021, *"Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really
Learn?"* (arXiv:2106.12914) — directly on point, not adjacent — established that ASVspoof2019
bonafide clips have systematically longer leading/trailing silence than spoofed clips, that models
can reach up to 85% accuracy / 15.1% EER from silence duration *alone*, and that correcting for
silence moves a real detector's EER from **3.6% → 15.5%** — an order-of-magnitude-scale
consequence from one confound class, on the exact corpus family (ASVspoof2019/2021) this project's
`data/real`/`data/fake`/`data/real2021`/`data/fake2021` are built from.

We measured our own corpus for exactly this (§4) and found:
- The ASVspoof-derived base corpus **and the ITW held-out benchmark itself** both show the
  literature's documented direction: real has somewhat more lead/trail silence than fake.
- This project's own TTS-generated accent cells (`en_foreign`, `hi_native`) show the **opposite**
  direction — fake has ~3x more trailing silence than real, and fake clips run measurably longer
  overall (AUC 0.04–0.08, i.e. near-perfectly separable, on both trailing silence and duration).

A corpus mixing both isn't "no shortcut, therefore safe" — it's an *inconsistent* one, which recent
work on negative transfer in deepfake-audio training (§3) describes as actively harmful, not
neutral, when mixed into one undifferentiated training set.

---

## 2. Score

Of 7 uninformed ideas (I1–I7) — **2 kept, 3 modified, 1 replaced, 1 refuted**; 1 new finding
(the silence/duration confound itself) added directly from the corpus-measurement step this
review's own methodology called for.

| # | Idea | Verdict | Citation / reasoning |
|---|---|---|---|
| I1 | The 63-d mean-pooled feature has a hard information ceiling more data can't buy past | **MODIFIED** | Partially right, wrong emphasis. [Das/Interspeech 2025 (isca-archive)](https://www.isca-archive.org/interspeech_2025/das25_interspeech.pdf) and the IndicSynth cross-lingual benchmark confirm hand-crafted features "cannot capture sufficient variability between datasets to train back-end classifiers," and that even end-to-end RawNet2/AASIST — strictly more expressive than our pooled-LFCC MLP — show EER >50% cross-lingually without domain adaptation. So representation *does* matter, but the literature's framing isn't "ceiling reached, more data is inert" — it's "more/more-diverse data actively teaches new spurious correlations faster than it teaches transferable signal," which is a different, more actionable claim (see I4/new finding). Kept as a real factor, demoted from "the" explanation. |
| I2 | Bumping raw hidden-layer width was the wrong kind of capacity | **KEPT** | No literature directly refutes this; our own `english_only` result (capacity bumped, held-out EER got worse, in-distribution barely moved) is consistent with "more parameters over the same impoverished/confounded input fits the confound harder, not genuine signal" — exactly the shortcut-learning framing the other citations establish generally. |
| I3 | `FixedNormalize`'s baked constants could be pushing ITW's features into a bad regime | **REFUTED (deprioritized)** | Nothing in the literature review flagged input normalization drift as a documented mechanism in this literature (the confounds found are all at the *feature-value* level — silence, sample rate, duration — not the normalization-statistics level). Not worth further investigation ahead of the confounds below; would need its own isolated test if everything else is fixed and the gap persists. |
| I4 | Multi-domain interference: cramming acoustically distinct populations into one undifferentiated training set actively degrades a single target domain | **KEPT, and now the load-bearing idea** | Directly confirmed by name: the "1+1<2" negative-transfer finding (AUDETER-lineage work) states plainly that jointly training on diverse forgery sources "may result in degraded performance, which contradicts the common belief that incorporating more source-domain data should enhance detection accuracy," names "harmful systems" with "strong system-specific fingerprints" that "bias the training signal," and proposes a **curriculum strategy** (weak-fingerprint systems first, harmful/strong-fingerprint systems folded in gradually) as the mitigation — not proposed or tried here yet. Müller et al. 2024 ("Harder or Different?", ISCA Interspeech) independently confirms "more diverse training data" is a genuine trade-off, not a free win, and that heterogeneous training sometimes underperforms more focused training on a specific held-out target. |
| I5 | v3's 0.1624 was never a stable target; cross-corpus generalization in this field is this volatile generally | **KEPT** | Strongly corroborated, not just plausible: cross-dataset ASVspoof→ITW evaluations in the literature show some published methods scoring **worse than a random classifier** on In-the-Wild after strong in-distribution performance, attributed directly to overfitting on ASVspoof2019's own artifacts (silence included). Our v3 baseline being *any* usable number on ITW, given it's largely ASVspoof-derived, is itself consistent with "got a shortcut that happened to point the right way," not "learned genuine transferable spoof detection." |
| I6 | Unweighted `CrossEntropyLoss` over a class/domain-imbalanced corpus shifts the EER-relevant threshold | **NOT ADDRESSED BY REVIEW — carried forward untested** | No paper surfaced speaking to this specific mechanism for our exact setup; still plausible, still cheap to test (class/domain-balanced sampling), just not validated or refuted by literature. Lowest-priority remaining open item. |
| I7 | The two isolated single-variable ablation effects (channel-recipe, sample-rate) don't compose additively with the full-corpus setting | **REPLACED** | The literature gives a sharper, mechanistic replacement for this vague "maybe they don't compose" idea: it's not that the effects fail to add — it's that **fixing one confound (sample rate) doesn't remove the others already present** (silence duration, duration itself), so "the fix didn't help" is fully explained without needing a non-additivity hypothesis at all. Occam's razor: don't invoke interaction effects when an unaddressed independent confound explains the same observation. |
| **NEW** | **Leading/trailing silence duration, and overall clip duration, are measurably confounded with label in `en_foreign`/`hi_native`, in the *opposite* direction from the ASVspoof-derived base corpus and the ITW held-out set itself** | **NEW FINDING, not yet fixed** | Measured directly this session (§4) using the exact confound class Kwak et al. 2021 established for ASVspoof-lineage corpora specifically. This is the single most concrete, actionable, well-cited output of this whole review. |

---

## 3. What the literature says, organized by the five questions Phase 1 asked

**(1) Documented cross-corpus generalization gaps in anti-spoofing.** Extensively documented, not
an edge case: LFCC-based models scoring 4.86% EER in-distribution (ASVspoof2019 LA) degrade sharply
on VCC2020/In-the-Wild/ADD2022; foundation-model backends (HuBERT-XLarge) reaching 0% EER
in-distribution still degrade substantially on FoR; some ASVspoof2019-trained methods score
*worse than random* on In-the-Wild specifically, which is why Müller et al. built ITW in the first
place — "to better investigate the generalization performance of methods trained on ASVspoof2019."
([Kulkarni 2024](https://www.isca-archive.org/asvspoof_2024/kulkarni24_asvspoof.pdf),
[arXiv:2603.05852](https://arxiv.org/pdf/2603.05852),
[Müller et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/muller24b_interspeech.pdf))

**(2) Shortcut learning / spurious correlation in audio deepfake detection.** A recognized,
actively-researched failure category, with multiple independently-discovered instances beyond
silence: **leading-silence artifacts in audio-visual datasets** reaching 98% separability from a
silence-only classifier ([arXiv:2412.00175](https://arxiv.org/abs/2412.00175)); **watermark-based
shortcuts** where "synthetic speech is watermarked by default and human speech is not," so a
detector learns "watermark → fake" ([arXiv:2606.23335](https://arxiv.org/html/2606.23335)); and,
most directly relevant to us, **non-speech-interval shortcuts diagnosed via controlled acoustic
intervention** — the exact same class of confound as our silence finding, with a proposed causal
methodology ([arXiv:2607.03150](https://arxiv.org/html/2607.03150v1)): compute distributional
divergence between bonafide/spoofed on acoustic descriptors, then verify via targeted perturbation
that removing/altering the suspect region collapses performance. Our `find_acoustic_shortcuts`
(§4) is a simpler version of the same idea (rank-based AUC instead of JS-divergence, no
intervention step yet) — a legitimate, literature-aligned methodology, not an ad hoc metric.

**(3) Hand-crafted pooled features (LFCC) vs. sequence/end-to-end architectures for cross-dataset
generalization.** Confirmed as a real, still-unsolved gap even for strictly more powerful
architectures: RawNet2/AASIST (frame-level, no mean-pooling) on unseen Indic languages showed EER
**>50%** without domain adaptation, vs. sub-1% on ASVspoof2019 — a cross-lingual gap that dwarfs
anything an architecture change alone would fix
([IndicSynth benchmark, arXiv:2608.12536](https://arxiv.org/html/2608.12536)). This tempers I1/I13
(the LCNN-reconsideration idea from Phase 1's evidence table): a bigger/sequence-aware architecture
is not a guaranteed fix for *this specific* cross-lingual, cross-generator gap — it would need to
be paired with the domain-structure and shortcut-removal work below, not substituted for it.

**(4) "More/more-diverse training data hurts generalization" as a named phenomenon.** Confirmed,
named, and mechanistically explained, not just observed: the **"1+1<2" negative-transfer**
framing states plainly that "training on highly diverse deepfake sources can induce negative
transfer" and that certain "harmful systems" with "strong system-specific fingerprints... bias the
training signal and encourage models to overfit to these cues, limiting the learning of
representations that generalise across most systems" — with a proposed **curriculum-based fix**
(train on weak-fingerprint systems first, add strong-fingerprint ones gradually), not tried here.
Müller et al.'s "Harder or Different?" independently frames "more diverse training data" as a
genuine trade-off (broader robustness vs. specialization on any one held-out target), not a free
win, and recommends matching training composition to the actual deployment/evaluation target rather
than maximizing diversity indiscriminately.

**(5) Practical/forum-level reports of the same wall.** Nothing found at the specific granularity
of "small on-device detector, GitHub issue" — the search surfaced only peer-reviewed material at
this specificity, which itself is informative: this looks like a research-grade problem being
independently rediscovered in production, not a known, casually-documented gotcha with an
off-the-shelf fix on a forum somewhere.

---

## 4. New empirical finding this review produced: an inconsistent silence/duration confound

Measured directly (2026-09-10, this session), 80–150 file samples per directory, header reads +
full decode for leading/trailing silence via a >0.01-amplitude threshold:

| corpus | lead silence (real vs fake) | trail silence (real vs fake) | direction vs. Kwak et al.'s documented ASVspoof bias |
|---|---|---|---|
| `data/real` vs `data/fake` (ASVspoof2019 base) | 0.612s vs 0.429s | 0.333s vs 0.367s | **matches** (real > fake, roughly) |
| `data/real2021` vs `data/fake2021` | 0.680s vs 0.237s | 0.414s vs 0.202s | **matches strongly** (real >> fake) |
| `data/real_itw_held` vs `data/fake_itw_held` (**the held-out benchmark itself**) | 0.337s vs 0.287s | 0.398s vs 0.311s | **matches, mildly** (real slightly > fake) |
| `accents_split/.../en_foreign` (real vs fake, incl. this session's new YourTTS clips) | 0.185s vs 0.010s | 0.211s vs **0.666s** | **reversed** (fake >> real on trailing silence) |
| `accents_split/.../hi_native` (real vs fake, incl. this session's new MMS-TTS clips) | 0.360s vs 0.224s | 0.178s vs **0.560s** | **reversed** (fake >> real on trailing silence) |
| `accents_split/.../hi_foreign` (real vs fake) | 0.501s vs 0.287s | 0.901s vs 0.589s | matches (real > fake), but `duration_s` AUC=0.08 (fake clips run longer) |
| `accents_split/.../en_native` (real vs fake) | 0.083s vs 0.179s | 0.167s vs 0.151s | no strong signal either way |

Confirmed the `en_foreign`/`hi_native` reversed pattern **predates this session's fixes** — measured
the original, pre-session XTTS-only fake pool (`data/accents/fake/en_foreign`, before this
session's YourTTS additions) at trail silence mean=0.668s, essentially identical to the post-fix
mixed pool's 0.666s. **This project's TTS-generation pipeline (XTTS originally, and this session's
YourTTS/MMS-TTS additions, which did not address this) has always produced fake audio with
systematically different trailing-silence/pacing than the real recordings it's compared against —
this session's fixes preserved rather than removed it.**

`find_acoustic_shortcuts` (new, in `dataset_audit.py`) generalizes the existing categorical
technical-metadata check to continuous descriptors via a rank-based AUC separability measure (no
scipy dependency — implements the Mann-Whitney U / rank-sum identity directly), analogous to but
simpler than the JS-divergence approach in the shortcut-diagnosis paper above. Wired into
`audit_directory_pair`/`check_corpus.py` automatically. **Currently, correctly, fails for
`en_foreign` (trailing silence + duration) and `hi_native`/`hi_foreign` (duration)** — this is the
gate working as designed, the same status the sample-rate check had before that confound was fixed.

---

## 5. What this changes about the plan

Not implemented in this document — this is diagnosis, not a fix, per the code-orange workflow's own
staging (spec now, plan next). But the shape of the next plan is now much clearer than "try
something else and see":

1. **Fix the silence/duration confound before any further retrain.** The paper's own recommended
   practice — trim/normalize leading and trailing silence consistently across *all* clips, real
   and fake, all sources, before feature extraction — is the direct, literature-endorsed fix, not
   a guess. This also means revisiting whether `pauseRatio` (one of only 3 prosody features,
   computed on the untrimmed signal) should be recomputed post-trim, or reconsidered entirely,
   given it's a within-clip pause measure closely related to the exact confound class found.
2. **Consider the curriculum-based negative-transfer mitigation (I4)** for any future corpus
   expansion: introduce weak-fingerprint/low-risk sources first, fold stronger-fingerprint
   generators (any single-TTS-engine source) in gradually rather than all at once — directly
   sourced from the "1+1<2" paper's proposed fix, not invented here.
3. **Treat v3's 0.1624 as a number to beat cautiously, not a stable ground truth** (I5) — the
   literature's own evidence that ASVspoof-trained models can score worse than random on ITW
   means a small amount of luck in exactly how much of ASVspoof's own silence-bias happened to
   transfer is a live possibility, not paranoia.
4. **LCNN/sequence-model reconsideration (I1) stays a live but lower-priority option** — the
   IndicSynth finding means it is not a guaranteed fix for the cross-lingual gap specifically, so
   it shouldn't jump the queue ahead of fixing the now-concrete, cited confounds above.
5. I6 (loss weighting) remains untested and low-priority — worth a cheap experiment once 1–2 are
   done, not before.
