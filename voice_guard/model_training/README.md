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

## Pipeline

```
dataset.py   — real/ + fake/ WAV dirs -> (features, label) examples,
                split at the SOURCE FILE level (no chunk leakage),
                optional TeleChannel degradation (--channel whatsapp ...)
model.py     — VoiceGuardMLP: FixedNormalize -> 66->64->32->2 MLP
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

## Run

The full `voice_guard_v3` command (2019 LA degraded + 2021 LA/2019 dev clean
+ In-the-Wild training split clean, GPU-accelerated MLP step):

```bash
# preflight: cheap, catches uneven chunk-yield and technical shortcuts
# BEFORE the 20-40 minute feature-extraction + training job below.
python check_corpus.py --pair data/real data/fake \
    --pair data/real2021 data/fake2021 --pair data/real_itw_train data/fake_itw_train

python train.py --real data/real --fake data/fake \
    --real-clean data/real2021 data/real_itw_train \
    --fake-clean data/fake2021 data/fake_itw_train \
    --out runs/voice_guard_v3 --epochs 30 \
    --channel whatsapp volte none --device auto --workers 8
cp runs/voice_guard_v3/model.onnx ../assets/models/voice_detector.onnx

# then check it actually generalizes, not just fits ASVspoof — and gate on it:
python eval_held_out_dirs.py --model runs/voice_guard_v3/model.pt \
    --real data/real_itw_held --fake data/fake_itw_held \
    --baseline-eer 0.1624   # fails loudly instead of silently shipping a regression
```

Note: this documented command uses `none` (a real no-op), not `clean` — see
the warning above. `train.py`'s own note about the historical v3 model: its
training log printed `channels=['whatsapp', 'volte', None]`, i.e. it likely
*already* used a true no-op for that slot despite this file previously
(wrongly) documenting `clean` — meaning this corrected command is closer to
what actually produced the deployed model's weights than the old docs were,
though the exact historical invocation couldn't be fully reconstructed.

`--workers` on Windows: keep it well under your core count if you're on a
16GB-RAM machine — see the `ProcessPoolExecutor` gotcha below.

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

## Acoustic playback loop (speaker -> phone mic) — v12_seqcnn plan + deploy gate

**2026-09-11 finding, measured on the *deployed* v11_seqcnn ONNX with
`eval_playback_loop.py`:** the model scores the bundled fake assets
correctly when they are scored *directly* (clean), but collapses when the
same audio is played through a loudspeaker and captured by a phone
microphone — on `test_assets/ai_clone_test_clip.wav` and
`tts_chattts_sample.wav` the simulated speaker->mic loop drops the
AI-probability to ~0.30 and below (real rooms/AGC push it lower, ~0.01-0.30,
which is exactly the "needle barely moves" the user saw live). This is a
channel-domain gap, not a capture bug: the Dart/Python feature extractors
are parity-tested, the gauge is `EMA(AI-prob)*100` with no inversion, and a
-24 dB level change alone barely moves the score (0.986 -> 0.924). Room
reverb + speaker/phone-mic coloration + ambient noise flatten the prosody/
LFCC cues the model keys on (see `docs/CRITICAL-entity-vs-style-confound.md`).

The fix is model-side, and it is wired but NOT yet trained (the training
corpus was deleted after v3 — reconstruct it with the "Getting real training
data" commands above):

1. **`vaani/telechannel/configs/channels.yaml` gained a `playback` recipe**
   (2026-09-11) modeling the acoustic loop: far room (RT60 600 ms, wet 0.6),
   pink noise at 12 dB, mic clip 0.3, no digital codec/packet loss, and a
   200-3800 Hz speaker+mic passband (uses the bandlimit stage's new
   `low_hz`/`high_hz` overrides — defaults remain 300-3400).
2. **Retrain** — same command as `voice_guard_v11_seqcnn` but with
   `playback` added to the channel mix:
   ```bash
   python train_seq_cnn.py \
       --real data/real data/real2021 data/real_itw_train \
       --fake data/fake data/fake2021 data/fake_itw_train \
       --real-clean data/real_noise_aug_split/train/en_native data/real_noise_aug_split/train/hi_native \
       --channel whatsapp volte playback \
       --weight-decay 1e-4 --label-smoothing 0.05 --attack-type-loss-weight 1.0 \
       --save-every-epoch-checkpoints \
       --out runs/voice_guard_v12_playback --epochs 25
   ```
3. **Deploy gate** — a model that still collapses on the speaker->mic loop
   must not be shipped. Run BOTH, and only copy `model.onnx` into
   `assets/models/voice_detector.onnx` when both pass:
   ```bash
   python eval_held_out_dirs_seqcnn.py --model runs/voice_guard_v12_playback/model.pt \
       --norm-stats runs/voice_guard_v12_playback/norm_stats.npz \
       --real data/real_itw_held --fake data/fake_itw_held --baseline-eer 0.1538
   python eval_playback_loop.py --onnx runs/voice_guard_v12_playback/model.onnx \
       --assets test_assets --gate-clean-min 0.60 --gate-loop-min 0.50
   ```
   `eval_playback_loop.py` scores each test asset clean vs. through
   `simulate_speaker_to_mic_loop()` (the same seeded approximation used for
   the diagnosis) and exits nonzero on either gate — the current deployed
   model fails the loop gate today, which is the point of the gate.

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
