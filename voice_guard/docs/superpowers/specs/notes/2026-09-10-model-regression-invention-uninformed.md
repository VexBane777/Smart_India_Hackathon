# VoiceGuard model regression — Phase 1: the uninformed invention set

> **Code orange, step 1.** Written 2026-09-10 *before* any literature search, deliberately, so
> step 2's review has something real to refute. Nothing here is approved. Every idea below is
> re-marked `kept / modified / replaced / refuted` with its citation in the design doc that
> follows. **Do not cite this file as a decision.** It is the pre-research artifact, kept only so
> the revision is auditable.
>
> Scope: why every attempt this session to retrain `model_training`'s VoiceGuardMLP on a larger,
> more diverse corpus made cross-generator held-out generalization *worse*, never better — even
> after two rounds of finding and fixing concrete, verified bugs.

---

## 0. The evidence this was invented against

All verified in-session (measured directly, not recalled) unless noted.

| # | Fact | Where |
|---|---|---|
| E1 | Six full training runs this session, all evaluated on the same held-out set (`data/real_itw_held`/`fake_itw_held`, a speaker-disjoint, seed-fixed split of the In-the-Wild corpus, confirmed untouched/stable across the whole session): **v3 (deployed baseline) 0.1624 → attempt1 0.2028 → attempt2 0.2079 → ablation (old corpus, only the channel-recipe changed) 0.1877 → english_only (Hindi dropped, capacity doubled) 0.2137 → v5_fixed (confounds fixed, corrected recipe) 0.2206 (worst of all six)**. Every single post-v3 attempt is worse than baseline. | `voice_guard/state.md`, this session's `eval_held_out_dirs.py` runs |
| E2 | In-distribution `val_eer` stayed essentially flat and *good* across every attempt regardless of corpus size/composition: v3 0.0793, attempt2 0.0853, ablation 0.0905, english_only 0.0815, **v5_fixed 0.0805 — the closest to v3 of any post-v3 run**. The dissociation is stark: in-distribution fit never got meaningfully worse (if anything, v5_fixed is the best-fitting post-v3 model), while held-out fit only ever got worse. | same eval runs |
| E3 | `features.chunk_audio` silently drops any clip <3s, and survival rates are wildly uneven and source-correlated: `fake2021` (185K files, the single largest source) — 69% zero-yield; `en_native` real — 74% zero-yield; `en_foreign`/`hi_native`/`hi_foreign` — ~0% zero-yield. Nobody had measured this before this session. | `dataset_audit.py`/`check_corpus.py`, header-only `sf.info` sampling |
| E4 | TeleChannel's `'clean'` recipe (used by `attempt2`/`ablation`/`english_only`, believed to mean "no degradation") actually applies real RIR reverb, white noise (25dB SNR), 10% mic-clip probability, and simulated packet loss — verified directly against `vaani/telechannel/configs/channels.yaml` and a runtime probe (`process_clip(pcm, "clean", ...)` measurably changes a synthetic tone). Isolated on the *same* old (v3-era) corpus, changing only `None`→`'clean'` moved held-out EER 0.1624→0.1877 (+0.0253) — the single largest isolated-variable effect measured all session, on its own. | `test_dataset_integrity.py::test_clean_recipe_is_not_a_noop`, `ablation` run |
| E5 | `en_foreign`/`hi_native`/`hi_foreign`'s entire fake side was, until this session, 100% XTTS-v2 (native 22050Hz) while every real clip in those cells is native 16000Hz — a perfect, deterministic technical-metadata confound (verified via `sf.info` on 150-file samples per cell, zero exceptions). `en_native` was *not* confounded this way (mixed generators/rates). | `dataset_audit.py::find_technical_shortcuts`, manual `sf.info` scan |
| E6 | **E5 was directly fixed this session** (real alternative-generator fake audio added at matching 16000Hz: `facebook/mms-tts-hin` for Hindi cells, Coqui YourTTS voice-cloning for `en_foreign`) — verified via `check_corpus.py` returning clean (exit 0) across every real/fake pair used in training, and all 19 `test_dataset_integrity.py`/`test_pipeline_smoke.py` tests passing. **v5_fixed trained on this fixed corpus, with the E4 fix also applied (`--channel whatsapp volte none`, the genuine no-op) — and still scored the single worst held-out EER of the entire session (0.2206).** | this session |
| E7 | `split_by_source` keyed on bare filename, not full path — checked across all 14×~300K basenames used this session, **zero actual collisions found**. Fixed anyway (correct on principle), but confirmed *not* a live contributor to any of E1's numbers. | full-corpus Python scan, this session |
| E8 | The model is a tiny MLP: 63-d input (60 mean-pooled LFCC coefficients + 3 prosody scalars: pauseRatio, energyVariance, zcrVariance) → 64 → 32 → 2, ReLU, softmax. No temporal/sequence modeling survives past feature extraction — `extract_lfcc` mean-pools every LFCC frame in a 3s window into a single 60-vector before the network ever sees it. Bumping hidden dims to 128→64 (`english_only`) did not help (E1) and, if anything, coincided with the second-worst result. | `model.py`, `features.py::extract_lfcc`, this session |
| E9 | `FixedNormalize` bakes the *training set's own* mean/std into the exported model as fixed buffers — a different training corpus composition produces different baked normalization constants, applied identically at both train-val time and held-out-eval time. Not yet tested in isolation. | `model.py::FixedNormalize` |
| E10 | The held-out benchmark (In-the-Wild) is explicitly documented in this project's own code as "the harshest external test" / "messy internet audio" — i.e. deliberately the most out-of-distribution benchmark available, not merely a held-out split of the same distribution. | `eval_held_out.py` docstring |
| E11 | v3's own exact training invocation is **not reconstructible with current code**: its training log printed `channels=['whatsapp', 'volte', None]`, but nothing in the current `train.py`/argparse can produce a literal `None` from a CLI string, and `README.md`'s documented "official" v3 command (`--channel whatsapp volte clean`) is now known (E4) to not even mean what it was assumed to mean. The exact provenance of the 0.1624 baseline is unverified. | `train_v3_stdout.log`, `train.py` (this session) |
| E12 | Total corpus size roughly doubled to tripled between v3 (~98,415 unique source files) and the later attempts (~121,782+ for attempt2/english_only, plus more again for v5_fixed's added generator-diversity clips), spanning far more generator families (ASVspoof2019/2021, In-the-Wild, VCTK, Common Voice, Svarah, OpenSLR103, IndicTTS, IndicVoices-R, MLAAD (116 architectures), CodecFake, DECRO, FoR, XTTS-v2, YourTTS, MMS-TTS-hin) and two languages (en, hi) with native/foreign accent splits. | manifest/dir counts, this session |
| E13 | This project already evaluated adopting a published anti-spoofing architecture (LCNN, frame-level CNN+BiLSTM over LFCC, from the ASVspoof-era `project-NN-Pytorch-scripts` reference implementation) and explicitly deferred it, with a stated re-trigger condition: *"revisit only if, after fine-tuning the current MLP on the new data, eval still isn't enough."* That condition has now been met twice (attempt2, and now v5_fixed after real bug fixes). | `voice_guard/state.md`, "LCNN backup" section |

**The one-line summary:** *every attempt to fix a specific, verified, measured bug in the training pipeline (channel-recipe misuse, a sample-rate shortcut) made the fix's own target metric behave exactly as predicted in isolation — yet the end-to-end held-out number never recovered, and in fact kept getting worse. Something about how this model/feature pipeline handles a larger, more heterogeneous corpus is broken in a way no single bug found so far explains.*

---

## 1. Thesis (uninformed): **this is a representation-capacity problem wearing a data-quality costume**

Every bug found this session (E3, E4, E5, E7) was real, worth fixing, and individually moved its
own isolated metric in the predicted direction (E4's ablation, E5's shortcut-test-now-passes). But
fixing all of them simultaneously (v5_fixed, E6) produced the *worst* outcome yet, while
in-distribution fit stayed excellent throughout (E2). That combination — bugs fixed, in-distribution
fit fine or improving, held-out generalization still monotonically getting worse as the corpus grows
— does not look like "there's one more bug to find." It looks like a fixed, small hypothesis space
(63 hand-crafted, mean-pooled features; a 2-hidden-layer MLP; a single global decision boundary; no
domain/channel conditioning) is structurally unable to absorb a more heterogeneous training
distribution without its boundary drifting away from whatever narrow region happened to generalize
to one specific external benchmark (In-the-Wild) at the smaller corpus size. More data, more
generator families, more languages/accents — all of it is being asked to fit through a 63-number
bottleneck with no temporal structure, and the model's only lever for "explaining" a bigger, weirder
training set is to spend capacity on whatever in-distribution split it's given, which is a different
split every time and evidently not the same thing as "whatever makes In-the-Wild's fakes look fake."

## 2. The invention set

Each idea is `Ix`; each will be re-marked in the design doc that follows.

### Tier A — representation/architecture

- **I1 — The 63-d mean-pooled feature vector has a hard information ceiling that a bigger/more
  diverse corpus cannot buy past, and every additional training example beyond that ceiling is
  pure noise/spurious-correlation fuel for whatever decision boundary best fits *that specific*
  training draw.** Grounds: E2 (in-distribution EER never meaningfully improves past ~0.08 no
  matter how much bigger/cleaner the corpus gets — a classic representation ceiling signature),
  E8 (temporal structure is destroyed before the model ever sees it), E13 (this project already
  flagged LCNN as the fallback for exactly this scenario).
- **I2 — Bumping raw hidden-layer width (E8's capacity experiment) was the wrong kind of capacity.**
  More parameters over the *same* impoverished 63-d input can only fit that input's noise harder,
  not recover information chunk_audio's mean-pooling already discarded. This predicts *width alone
  never helps*, matching the english_only result, without needing E1's full "ceiling" framing.
- **I3 — `FixedNormalize`'s baked constants (E9) are recomputed from a different distribution every
  run and could be quietly pushing In-the-Wild's differently-shaped features into a bad regime of
  the network, independent of what the network learned** — untested in isolation this session.

### Tier B — data/domain structure

- **I4 — Multi-domain interference: cramming ASVspoof-era, telephony-Hindi, VCTK, Common Voice,
  and half a dozen TTS-architecture families' worth of acoustically distinct "real" and "fake"
  populations into one undifferentiated training set, with no domain-conditioning/adapter, actively
  degrades any single target domain's (ITW's) decision boundary** — a bigger, "more diverse" corpus
  isn't simply neutral-or-helpful; it can be actively harmful for a model with no way to represent
  which domain an example came from.
- **I5 — Cross-corpus/cross-generator generalization in synthetic-speech detection is inherently
  this volatile, and v3's 0.1624 was never a stable target to begin with** (compounded by E11 — we
  can't even reconstruct v3's exact recipe). If true, chasing "beat 0.1624" with more data is
  chasing noise, and the right fix is a fundamentally more robust architecture/eval protocol, not
  more data curation.
- **I6 — Unweighted `CrossEntropyLoss` over a class-imbalanced, per-domain-imbalanced corpus is
  shifting the decision threshold in a way that specifically hurts EER (a threshold-sensitive
  metric) on a differently-imbalanced held-out set** — not measured this session at all.

### Tier C — process

- **I7 — The two "isolated ablation" findings (E4's channel-recipe effect, E5's shortcut fix) do
  not compose additively, and treating them as independent, stackable fixes was the actual process
  error** — each was validated in a controlled, single-variable comparison, but v5_fixed changed
  both simultaneously against a *also-changed* corpus (the full accent expansion, not the isolated
  old-v3 corpus), so the isolated effect sizes may not transfer to the full-corpus setting at all.

---

## 3. What step 2 needs to find

Per the code-orange workflow: a wide-net literature review (peer-reviewed papers, but also
GitHub issues/READMEs and developer-forum discussion where a "we hit this exact wall" report is
often more directly useful than a paper) specifically on:

1. Documented cross-corpus/cross-generator generalization gaps in synthetic-speech/anti-spoofing
   detection (ASVspoof-lineage literature, In-the-Wild's own paper, any published baseline numbers
   for models trained on ASVspoof and evaluated on In-the-Wild or vice versa).
2. Shortcut learning / spurious correlation in audio deepfake detection specifically (channel,
   codec, or dataset-identity confounds as the majority of a detector's apparent accuracy).
3. Known failure modes of pooled hand-crafted features (LFCC/CQCC without a sequence model) vs.
   frame-level/sequence architectures (LCNN, RawNet2, AASIST, wav2vec2-based) for cross-dataset
   generalization specifically, not just in-distribution EER.
4. "More/more-diverse training data made generalization worse" as a named phenomenon in ML more
   broadly (domain generalization, multi-task/multi-domain interference, negative transfer) —
   does this have known mitigations, and do they match any of Tier B/C above.
5. Practical, forum/GitHub-reported experience (not just papers) from anyone who's built a small
   on-device anti-spoofing/deepfake-audio detector and hit a similar wall.
