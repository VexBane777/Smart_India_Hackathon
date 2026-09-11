# voice_guard model training

Produces `../assets/models/voice_detector.onnx` (see "Run" below — the app
runs ONNX directly as of 2026-09-09, no `.tflite` conversion step).

**2026-09-08 update — full-corpus retrain (`runs/voice_guard_v3`), now deployed:**
Trained on the full ASVspoof2019 LA train partition (2,580 bonafide + all
22,800 spoof, not a 5k subsample) degraded through `--channel whatsapp volte
clean`, plus ASVspoof2021 LA eval + ASVspoof2019 LA dev (21,000 bonafide +
185,410 spoof, already telephony-degraded so used clean/undegraded), plus
the training portion of In-the-Wild (Muller et al., 44/54 speakers,
speaker-disjoint from its own held-out split) — see `data/prep_asvspoof2021.py`
and `data/prep_in_the_wild.py`.

- `final_val_eer = 0.0793` (in-distribution, same-corpus held-out split).
- **Cross-generator held-out EER = 0.1624** on the 10 In-the-Wild speakers
  never seen during training (`data/real_itw_held` / `fake_itw_held`,
  `eval_held_out_dirs.py`) — this is the number that actually matters for
  "does this generalize to messy real-world audio."
- Worth reporting honestly: **without** any In-the-Wild data in training,
  the same architecture scored EER=0.63 (worse than chance) on the full
  In-the-Wild set — i.e. an ASVspoof-only-trained model does not transfer
  to real-world audio at all. Folding ~40% of In-the-Wild's speakers into
  training (while keeping the rest speaker-disjoint for eval) is what
  closed most of that gap. This is a large domain-shift finding worth
  keeping in mind before trusting any future EER number that isn't
  validated against non-ASVspoof audio.
- GPU (RTX 5050, cu128 torch) was used for the MLP training loop itself
  (`--device cuda`); feature extraction and channel degradation stayed
  CPU-bound (ffmpeg subprocesses), parallelized via `--workers`.
- **Windows gotcha found the hard way:** `ProcessPoolExecutor` on Windows
  re-execs the entry-point script's module-level code in every worker
  (no fork). `train.py` used to `import torch` at module level, so every
  worker also paid torch's CUDA-DLL-loading cost — with `--workers 20` on
  a 16GB-RAM box this exhausted the page file (`OSError: [WinError 1455]`).
  Fixed by moving all torch/model imports into `main()`; workers now only
  need numpy/soundfile/the TeleChannel pipeline. Keep heavy imports out of
  module level in any script whose functions get sent through a Windows
  process pool.
- Previous entries below (the 5k-fake-subsample, no-channel-degradation
  `voice_guard_v1` run, and the placeholder `.tflite` before that) are kept
  for history but are superseded by the above.
- **Merged with a concurrent session's changes after this run finished**
  (RNG-seeded channel degradation via `--seed`, `process_clip(..., rng=)`).
  That merge also picked up a semantic fix: `"clean"` is a real, registered
  TeleChannel recipe (light rir/noise/mic/loss, no codec/bandlimit — see
  `telechannel/configs/channels.yaml`), not a synonym for "no processing."
  The `voice_guard_v3` run above used the *old* code, where `--channel
  whatsapp volte clean` silently treated `clean` as zero processing for
  that third of the corpus. Going forward `--channel whatsapp volte clean`
  means what it says — pass no `--channel` flag at all (or an empty list)
  for truly undegraded audio. `voice_guard_v3`'s reported numbers stand as
  measured, but a fresh run under the merged code would build a slightly
  different training mix and shouldn't be assumed to reproduce them exactly.

**Feature contract (must not drift):** `features.py` is a numpy port of
`lib/utils/audio_processor.dart`'s `extractLfccSequence`/`extractScalars`
— **two** ONNX inputs, not one flat vector (changed from the single-vector
MLP contract by
`docs/superpowers/plans/2026-09-11-frame-level-seq-model-and-attack-type-plan.md`,
consolidating remediation tracks 2/3/4):

- `lfcc_sequence`: shape `(1, n_frames, 60)` — the full per-frame LFCC
  matrix (1024-pt/256-hop frames), unpooled.
- `scalars`: shape `(1, 6)` — `[pauseRatio, energyVariance, zcrVariance,
  jitter_local, shimmer_local, hnr_db]` (3 prosody + 3 physio, see
  `docs/CRITICAL-entity-vs-style-confound.md` §4 item 2).

`tflite_io.dart`'s `TFLiteService.infer` sends exactly these two tensors
and expects **two** `(1, 2)` raw-logit outputs: `real_fake_logits` and
`attack_type_logits` (softmax applied on the Dart side for both). If you
change frame size / hop / feature order / model I/O names on either side,
change it on both — a Python-trained model is only as good as its parity
with what the phone actually computes at inference time.

Normalization (mean/std) is baked into the exported model itself
(`model.py::FixedNormalize`), not applied separately — there's no
preprocessing step on the Dart side to apply it, so it has to travel
inside the graph.

## Pipeline (v12, 2026-09-11)

**Read `../docs/EVAL-PROTOCOL.md` first.** It is the policy (no clean-only
eval or training unless `--application bank`), the held-out splits, the
windowing contract, the metrics and the deploy gates. The code enforces it.

```
eval_protocol.py  channel policy: resolve_channels(), CleanOnlyEvalError, channel groups
dataset.py        audio -> 3 s windows (short clips padded, tail windows, pad_fraction),
                  stable per-file seeds, pad balancing for training
features.py       LFCC sequence + 6 scalars; numpy mirror of lib/utils/audio_processor.dart
feature_cache.py  sharded on-disk memmap cache with a feature-version hash (StaleCacheError)
corpus.py         committed training sets (TRAIN_SETS_V12) + eval sets, train channel assignment
make_eval_splits.py  writes eval_splits/held_out_split_v1.json (select/test), refuses overwrite
build_caches.py   builds eval/train caches ahead of time (resumable)
model.py          VoiceGuardSeqTCN (v12), VoiceGuardSeqCNN (v11), MLP (v9, scoring only),
                  build_model_from_norm_stats / load_scoring_model / export_onnx
train_seq_cnn.py  cache-backed training (AdamW, warmup+cosine, EMA, leave-attack-out)
select_best_checkpoint_seqcnn.py  epoch selection on the `select` split ONLY
evaluate.py       the one eval harness -> report.json + report.md
validate_fp16.py  checks float16 cache storage doesn't move model outputs
eval_stats.py     compute_eer (vectorized), bootstrap CIs, Spearman, AUC, balanced accuracy
check_corpus.py / dataset_audit.py  pre-training corpus shortcut audits
```

Retired 2026-09-11, with no drop-in replacement needed: train.py (MLP),
train_curriculum.py, eval_held_out*.py, measure_confound*.py,
select_best_checkpoint.py, eval_accent_cells.py, export_tflite.py (see the
EVAL-PROTOCOL.md §8 table for where each went).

## Getting real training data

**2026-09-08: `data/` was deleted after the `voice_guard_v3` run** (tens of
GB of audio, not worth keeping around when it's fully reproducible from
these three Kaggle sources — `kaggle.json` at `~/.kaggle/kaggle.json` is set
up and verified working). To reconstruct the exact corpus `voice_guard_v3`
was trained on:

```bash
# 1. ASVspoof 2019 LA (train partition -> real/ fake/) — master plan §5.1
kaggle datasets download -d anishsarkar22/asvpoof-2019-dataset-la -p data/extracted --unzip
python data/split_flac_to_wav.py   # extracted/... -> data/real/, data/fake/
```

**Attack-type protocol only** (for track 4's TTS/VC labeling — does NOT
need the multi-GB audio archive, `data/real`/`data/fake` are already
extracted): re-download just the protocol file from the same Kaggle
source used originally:

```bash
kaggle datasets download -d anishsarkar22/asvpoof-2019-dataset-la -p data/asvspoof2019_la_protocol_only --unzip -f "ASVspoof2019.LA.cm.train.trn.txt"
```

(If `-f` isn't supported by the installed `kaggle` CLI version, download
the full dataset to a throwaway directory and delete everything except
the protocol file afterward — it's a few hundred KB, the audio is
multiple GB; do not leave the audio archive on disk.)

The train protocol's column layout differs slightly from
`trial_metadata.txt` (columns: `speaker_id utt_id - attack_id key`, no
codec/tx/trim/subset columns) — `attack_labels.py::load_asvspoof_2019_train_attack_map`
handles this format; `load_asvspoof_attack_map` handles the 2021 format.

```bash
# 2. ASVspoof 2021 LA eval + bundled 2019 LA dev (-> real2021/ fake2021/)
#    already telephony-degraded by construction (real codec roundtrips),
#    so these are NOT run through --channel in training.
kaggle datasets download -d wagiartono/asvspoof-2019-and-2021-la -p data/asvspoof2021_la --unzip
python data/prep_asvspoof2021.py   # asvspoof2021_la/... -> data/real2021/, data/fake2021/
#    NOTE: this corpus's re-encoded FLACs fail to decode in ~44% of files via
#    soundfile/libsndfile ("unknown error in flac decoder") despite being
#    valid FLAC — prep_asvspoof2021.py shells out to ffmpeg instead, which
#    handles them fine. Don't swap that back to sf.read().

# 3. In-the-Wild (Muller et al., ~31.8k real-world clips, 54 speakers)
kaggle datasets download -d abdallamohamed312/in-the-wild-dataset -p data/in_the_wild --unzip
unzip data/in_the_wild/download -d data/in_the_wild   # the dataset is a zip-in-a-zip
python data/prep_in_the_wild.py --held-out-fraction 0.2   # speaker-disjoint 44/10 split
#    -> data/real_itw_train/, fake_itw_train/ (fold into training)
#       data/real_itw_held/,  fake_itw_held/  (KEEP OUT of training — this is
#       the only thing that catches "does it generalize to messy real audio")
```

Recommended: degrade the ASVspoof2019 LA train portion through TeleChannel
recipes (`--channel whatsapp volte none`) so the model sees phone-channel-
shaped audio like it will on a real call, not just clean studio recordings,
*while still including a genuinely undegraded pass* — the 2021/In-the-Wild
portions skip this since they're already realistically degraded or already
real-world audio.

**Do not use `clean` here expecting "no degradation."** Despite the name,
`clean` (`vaani/telechannel/configs/channels.yaml`) applies real RIR reverb,
white noise, mic clipping, and simulated packet loss — it only skips the
codec/ffmpeg step, so TeleChannel's orchestration can be exercised without
ffmpeg installed. Using `clean` where "no processing" was intended is a real
mistake this project made (2026-09-10): four retraining attempts in a row
regressed cross-generator held-out EER, and the single largest isolated
contributor (measured directly, ablating every other variable) was exactly
this — training on `None`-processed audio generalizes meaningfully better
than training on `'clean'`-processed audio. `train.py --channel` now accepts
the literal string `none` for a true no-op pass; see its `--help` and
`voice_guard/state.md`'s "Attempt 2 + follow-up ablations" section for the
full writeup, evidence, and the other data-integrity bugs found alongside it
(uneven chunk-yield across sources, a sample-rate/generator confound in
`en_foreign`/`hi_native`/`hi_foreign`). **Run `check_corpus.py` on every
real/fake directory pair before a training run** — it catches both classes
of issue cheaply, before committing to a 20-40 minute feature-extraction +
training job.

Large downloads on Windows: use a **detached** process (PowerShell
`Start-Process`, not `run_in_background`/`&`) for anything that runs longer
than ~25 minutes — background tasks got silently killed mid-download at
that mark during the v3 run, twice, corrupting the target file both times.
Verify with `unzip -t` before trusting a large download.

## Run (v12)

Python: the main checkout's `.venv313` (torch cu128, soundfile, librosa,
onnxruntime, pytest). The PATH `python` has no soundfile. Caches default to
`model_training/cache` (gitignored) or `$VOICEGUARD_CACHE_ROOT`. Put them in
the main checkout, not a throwaway worktree.

```bash
python check_corpus.py --pair data/real data/fake --pair data/real2021 data/fake2021 \
    --pair data/real_itw_train data/fake_itw_train          # cheap shortcut preflight
python build_caches.py --eval                               # ~25 min, resumable
python build_caches.py --train                              # ~1.5-2 h, resumable
python train_seq_cnn.py --out runs/voice_guard_v12          # 30 epochs, GPU
python select_best_checkpoint_seqcnn.py --run runs/voice_guard_v12 --out runs/voice_guard_v12_selected
python evaluate.py --split test --out runs/eval_v12_test \
    --model v9=runs/voice_guard_v9_noisefix_final/model.pt \
    --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt \
    --model v12=runs/voice_guard_v12_selected/model.pt \
    --attack-val-run runs/voice_guard_v12 --candidate v12 --reference v11
# deploy only if report.md's decision says so:
cp runs/voice_guard_v12_selected/model.onnx ../assets/models/voice_detector.onnx
```

Long jobs on Windows: launch detached (PowerShell `Start-Process
-WindowStyle Hidden -RedirectStandardError log`). Tool-managed background
jobs have been killed at ~25 min before. `--workers 10` is safe on this
16 GB / 20-thread machine because workers never import torch (keep it that
way: see the gotcha above).

**2026-09-09: the app now runs `model.onnx` directly via `flutter_onnxruntime`**
(`lib/services/src/tflite_io.dart`), not a converted `.tflite`. `export_tflite.py`
is kept only as a legacy/optional script (e.g. if a future need for actual
on-device TFLite specifically comes up) — it is no longer part of the
recommended flow, and `assets/models/voice_detector.tflite` was removed.
Just copy `model.onnx` straight into `assets/models/voice_detector.onnx`;
`flutter pub get` / rebuild picks it up automatically (directory already
declared in `pubspec.yaml`). `.onnx` is gitignored project-wide *except*
this one shipped asset (see root `.gitignore`'s explicit exception) — every
other `.onnx` (training runs, intermediates) stays ignored as before.

## Known gaps (deliberate, not oversights)

- `backend/main.py`'s response label (`"mobilenetv3-small-int8-demo-v0.1"`)
  is a placeholder string, not a model description. The shipped model is
  float32 ONNX; no quantization has been done.
- Covered since v12: unseen generators (MLAAD) and unseen codecs/channels
  (gsm_2g, pstn, tandem_xnet are never trained on). Still not covered:
  **real recorded phone calls** (master plan §5.4). Every channel here is
  TeleChannel-simulated.
- In-the-Wild select/test is file-level, not speaker-level (meta.csv is
  gone). See EVAL-PROTOCOL.md §3.
