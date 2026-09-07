# voice_guard model training

Produces `../assets/models/voice_detector.tflite`.

**2026-09-07 update:** a teammate (`DeveloperAJ2799`, commit `98738b2`)
landed a first real `.tflite` there concurrently with this pipeline being
written (10 KB — genuinely "lightweight," almost certainly not trained on
ASVspoof-scale data yet). This pipeline is still the way to replace it
with one trained on a real corpus once the GPU-session training run
finishes — check `../assets/models/voice_detector.tflite`'s size/date
before assuming it still needs building from scratch.

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

`kaggle.json` is set up (`~/.kaggle/kaggle.json`) and verified working.
Nearest match to the master plan's ASVspoof 2019 LA target
(`vaani/00_MASTER_PLAN.md` §5.1):

```bash
kaggle datasets download -d anishsarkar22/asvpoof-2019-dataset-la
```

Real (bonafide) and fake (spoof) protocol files are inside; split them into
`real/` and `fake/` WAV directories per `dataset.py`'s expectation before
running `train.py`. Recommended: degrade a portion through TeleChannel
recipes (`--channel whatsapp volte cellular_3g clean`) so the model sees
phone-channel-shaped audio like it will on a real call, not just clean
studio recordings.

## Run

```bash
python train.py --real path/to/real_wavs --fake path/to/fake_wavs \
    --out runs/voice_guard_v1 --epochs 30 --channel whatsapp volte clean
python export_tflite.py --onnx runs/voice_guard_v1/model.onnx \
    --out ../assets/models/voice_detector.tflite
```

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
- No cross-generator/cross-accent held-out eval yet (master plan §5.4's
  four protocols) — this trains one MLP on whatever corpus is handed to
  it. Fine for "a real model exists," not yet "the leaderboard story."
