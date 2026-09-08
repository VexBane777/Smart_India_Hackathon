# voice_guard model training

Produces `../assets/models/voice_detector.tflite`.

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
`lib/utils/audio_processor.dart`'s `extractLfcc`/`extractProsody` — 60 LFCC
(mean-pooled over 1024-pt/256-hop frames) + 3 prosody scalars
(`[pauseRatio, energyVariance, zcrVariance]`), 63 floats total, in that
concatenation order. `tflite_io.dart`'s `TFLiteService.infer` sends exactly
this vector and expects a `(1, 2)` raw-logit output (it applies softmax
itself). If you change frame size / hop / feature order on either side,
change it on both — a Python-trained model is only as good as its parity
with what the phone actually computes at inference time.

Normalization (mean/std) is baked into the exported model itself
(`model.py::FixedNormalize`), not applied separately — there's no
preprocessing step on the Dart side to apply it, so it has to travel
inside the graph.

## Pipeline

```
dataset.py   — real/ + fake/ WAV dirs -> (features, label) examples,
                split at the SOURCE FILE level (no chunk leakage),
                optional TeleChannel degradation (--channel whatsapp ...)
model.py     — VoiceGuardMLP: FixedNormalize -> 63->64->32->2 MLP
train.py     — trains, reports EER per epoch, exports model.onnx
export_tflite.py — model.onnx -> voice_detector.tflite (needs tensorflow +
                onnx2tf — NOT installed on the CPU dev laptop this was
                written on; run on whichever machine has room for it)
test_pipeline_smoke.py — synthetic-data smoke test (mechanics only, not
                accuracy) proving the pipeline runs before pointing it at
                a real corpus
```

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
recipes (`--channel whatsapp volte cellular_3g clean`) so the model sees
phone-channel-shaped audio like it will on a real call, not just clean
studio recordings — the 2021/In-the-Wild portions skip this since they're
already realistically degraded or already real-world audio.

Large downloads on Windows: use a **detached** process (PowerShell
`Start-Process`, not `run_in_background`/`&`) for anything that runs longer
than ~25 minutes — background tasks got silently killed mid-download at
that mark during the v3 run, twice, corrupting the target file both times.
Verify with `unzip -t` before trusting a large download.

## Run

The full `voice_guard_v3` command (2019 LA degraded + 2021 LA/2019 dev clean
+ In-the-Wild training split clean, GPU-accelerated MLP step):

```bash
python train.py --real data/real --fake data/fake \
    --real-clean data/real2021 data/real_itw_train \
    --fake-clean data/fake2021 data/fake_itw_train \
    --out runs/voice_guard_v3 --epochs 30 \
    --channel whatsapp volte clean --device auto --workers 8
python export_tflite.py --onnx runs/voice_guard_v3/model.onnx \
    --out ../assets/models/voice_detector.tflite

# then check it actually generalizes, not just fits ASVspoof:
python eval_held_out_dirs.py --model runs/voice_guard_v3/model.pt \
    --real data/real_itw_held --fake data/fake_itw_held
```

`--workers` on Windows: keep it well under your core count if you're on a
16GB-RAM machine — see the `ProcessPoolExecutor` gotcha below.

`tflite_io.dart` already tries loading from `assets/models/voice_detector.tflite`
first — no code change needed on the Flutter side once the file exists;
`flutter pub get` / rebuild picks it up automatically since the directory
is already declared in `pubspec.yaml`.

## Known gaps (deliberate, not oversights)

- No CNN/temporal model — the feature vector is already mean-pooled (no
  time axis survives), so a small MLP is the right-sized model; a
  CNN/MobileNet would need frame-level features and a matching change to
  `audio_processor.dart`, out of scope for this pass.
- `export_tflite.py` always picks the float32 TFLite variant, not the
  int8-quantized one `backend/main.py`'s response labels
  (`"mobilenetv3-small-int8-demo-v0.1"`) imply — that name is currently
  just a placeholder string, not a real model description. Quantization
  is a follow-up once a float32 model's accuracy is known to be worth
  preserving through int8.
- Cross-generator held-out eval now exists (In-the-Wild speaker-disjoint
  split, see above) but master plan §5.4's other three protocols — unseen
  codec, unseen noise, real recorded calls — are still not covered. Still
  fine for "a real model exists, and we know its real-world number," not
  yet "the leaderboard story."
