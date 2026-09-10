# VoiceGuard model session — 2026-09-10/11: from "doesn't work" to a validated fix

**Status:** Closing report for this session. A dedicated final analysis pass is
still pending (explicitly deferred by the user at session close) — treat the
"Open items" section below as the starting point for that pass, not as
already resolved.

**Bottom line:** the model deployed at session start (`assets/models/
voice_detector.onnx`, dated 2026-09-09) fired 80–100% false "AI DETECTED"
alerts on real human speech whenever ambient noise was present — this was
the actual, demo-blocking production bug. It is now fixed and **verified
live, on-device, with real ambient noise**, not just offline. The fix
shipped is `voice_guard_v9_noisefix` (epoch 22), currently in
`assets/models/voice_detector.onnx`.

---

## 1. What was live-validated (do this again on any new device/build)

Two on-device tests, both with real ambient noise present (not a silent
room) and the phone's own capture path (Live Mic Test, no real call):

1. **Real human speech + real ambient noise, ~114s cumulative across
   several takes**: mostly `VERIFIED HUMAN` (raw scores mostly 0.02–0.3).
   **4 false `ALERT fired` events total, all during continuous
   uniniterrupted stretches of 20s+** — roughly one false alert per 25–30s
   of continuous talking. **Zero false alerts in isolated ~10–15s bursts**
   (the realistic length of one demo turn). This is a dramatic improvement
   over the pre-fix baseline (constant, near-100% false alerts under any
   noise) but **not a perfect elimination** — say this plainly if asked.
2. **Genuine AI voice-clone clip** (`model_training/test_assets/
   ai_clone_test_clip.wav` — a real clip from the In-the-Wild deepfake
   corpus, held out from all training/eval this session), played through an
   **external speaker** (not the phone's own — see the pre-existing
   acoustic-loopback-clamp finding below) into Live Mic Test: correctly and
   confidently detected — `ALERT fired`, ema climbed to 0.92 then held at
   0.96–0.97 for the rest of the clip. Confirms the false-positive fix did
   not cost genuine detection capability.

**Demo guidance derived from this**: keep spoken turns to ~10–15s with
pauses, which is both the natural conversational rhythm and the exact
window where zero false alerts were observed. Have the AI-clone clip ready
as a deliberate "watch it catch a real fake" beat.

## 2. Why this took as long as it did (full narrative, for the record)

The session did not go straight to the fix. In order:

1. **Chased the wrong benchmark for most of the session.** Six-plus
   training attempts (`attempt1`, `attempt2`, `ablation`, `english_only`,
   `v5_fixed`, `v6`, `curriculum`, `v7`, `v8`/`v8b`) were all evaluated
   against **cross-generator spoof-detection accuracy on the In-the-Wild
   held-out set (ITW EER)** — a real, legitimate metric, but **not the
   metric behind the actual reported production bug** (false positives on
   noisy *real* speech). This should have been caught earlier by re-reading
   this project's own `state.md`, which already documented the false-
   positive-on-noise root cause from the prior (2026-09-09) session.
2. **Found and fixed four real, separate bugs along the way** (all still
   valid, all still fixed, all still worth keeping regardless of the ITW-
   vs-actual-bug detour):
   - `features.chunk_audio` silently drops any clip <3s, with wildly uneven
     survival rates across sources (some ~70% loss, some ~0%) — nobody was
     measuring this before. Fixed: `dataset.build_examples` now reports
     per-source-directory yield.
   - `dataset.split_by_source` keyed on bare filename, not full path — a
     latent cross-directory collision risk (checked: zero actual collisions
     in this corpus, but wrong on principle). Fixed: keys on resolved path.
   - TeleChannel's `'clean'` recipe is **not** a no-degradation pass (it
     applies real RIR reverb + noise + mic clipping + packet loss) — using
     it where "no processing" was intended measurably hurt generalization.
     Fixed: `train.py --channel` now accepts a real `none` sentinel, with a
     runtime warning if `'clean'` is used instead.
   - Several accent-expansion cells (`en_foreign`, `hi_native`,
     `hi_foreign`) had every fake example at one TTS engine's native sample
     rate (22050Hz) while every real example was native 16000Hz — a
     perfect, trivially learnable shortcut. Fixed by sourcing real
     alternative-generator fake audio at matching 16000Hz
     (`facebook/mms-tts-hin` for Hindi, Coqui YourTTS for `en_foreign`).
   - A follow-up **code-orange literature review** (see
     `docs/superpowers/specs/2026-09-10-model-regression-design.md`) found
     a second, subtler confound class in the same cells: leading/trailing
     silence duration correlating with label (a documented ASVspoof-lineage
     issue, Kwak et al. 2021, arXiv:2106.12914) — fixed via
     `dataset.trim_edge_silence()`, applied unconditionally to every clip
     at load time.
   - New permanent test/tooling from this work: `dataset_audit.py`
     (technical + acoustic shortcut detection), `check_corpus.py` (preflight
     gate), `test_dataset_integrity.py` (19 tests), `eval_stats.py`
     (bootstrap confidence intervals for held-out EER — built after a
     fine-grained checkpoint sweep produced an apparent "beat baseline"
     result that failed to replicate across an independent run; the CI
     tooling showed it was noise, not signal, and — separately — caught a
     **real pipeline-versioning bug**: re-evaluating the unchanged v3
     checkpoint through the post-silence-trim-fix pipeline gives a
     different EER (0.1538) than originally reported (0.1624), because the
     eval pipeline itself changed mid-session).
3. **The actual fix, once found, was fast.** `data/real_noise_aug/
   {en_native,hi_native}` (11,519 real speech clips + real MUSAN ambient
   noise, built in the *prior* session specifically for this bug) had never
   been included in a single training run this session. Folding it in as
   additional `--real-clean` data and retraining (`v9_noisefix`) dropped
   measured false-positive rate on held-out noisy real speech from **47.3%
   to 10.4%** (`epoch 22`, selected by sweeping every saved checkpoint
   against the held-out noise set directly, not by in-distribution
   val_eer — those two can diverge, per finding #2 above).

## 3. What's actually deployed right now

- `assets/models/voice_detector.onnx` = `model_training/runs/
  voice_guard_v9_noisefix_final/model.onnx` (epoch 22 of `v9_noisefix`).
- Backup of the previous (pre-session) model at `model_training/runs/
  voice_detector_v3_backup.onnx` — restore with a straight file copy +
  rebuild if a rollback is ever needed.
- Training data mix for this model: base ASVspoof2019 (`data/real`/`fake`,
  degraded through `--channel whatsapp volte none`) + ASVspoof2021/2019dev
  (`data/real2021`/`fake2021`) + In-the-Wild train + **`data/
  real_noise_aug_split/train/{en_native,hi_native}`** (the noise-fix
  addition). Does **not** include the accent-expansion cells
  (`en_native`/`en_foreign`/`hi_native`/`hi_foreign` fake/real) — that was a
  deliberate scope decision to keep this run fast and on-target for the
  noise bug specifically, not a statement that accent coverage work is
  abandoned.
- Metrics: ITW held-out EER 0.1602 (CI [0.1431, 0.1775] — statistically
  tied with the re-measured v3 baseline of 0.1538, not a real regression).
  Noise-FPR (real ambient-noise speech misclassified as fake) 10.4%, down
  from 47.3%. Both measured offline; the on-device live test (§1) is the
  real confirmation.

## 4. Environment setup — full replication instructions for another device

This session set up a complete build environment from scratch, persisted
via user-level environment variables (registry, not just the shell
session) so it survives reboots and new sessions **on this machine**. A
**different** device needs all of this repeated. In order:

### 4.1 JDK 17

```powershell
winget install --id Microsoft.OpenJDK.17 -e --accept-source-agreements --accept-package-agreements --scope user
```

Installs to `%LOCALAPPDATA%\Programs\Microsoft\jdk-17.x.x.x-hotspot`. Set
`JAVA_HOME` to that path (User-level env var) and add its `bin/` to PATH.

### 4.2 Android SDK (command-line tools + platform-tools + NDK + build-tools)

No full Android Studio needed. Download the command-line tools
(`https://dl.google.com/android/repository/commandlinetools-win-*_latest.zip`
— get the current version number from
`https://developer.android.com/studio#command-tools`), extract so the
`cmdline-tools` folder inside becomes `<ANDROID_HOME>\cmdline-tools\latest\`
(the `latest` folder name matters). Suggested `ANDROID_HOME`:
`%LOCALAPPDATA%\Android\Sdk`.

```powershell
$sdkmanager = "$env:ANDROID_HOME\cmdline-tools\latest\bin\sdkmanager.bat"
& $sdkmanager --sdk_root=$env:ANDROID_HOME --licenses   # feed "y" repeatedly, or pipe (1..20 | %{"y"}) -join "`n"
& $sdkmanager --sdk_root=$env:ANDROID_HOME "platform-tools" "platforms;android-36" "platforms;android-35" "build-tools;36.0.0" "build-tools;35.0.0"
```

**The NDK needs a separate, more resilient download** — `sdkmanager`'s own
NDK downloader hit repeated connection resets during this session, even on
a working connection. Use `Start-BitsTransfer` instead (resumable, handles
stalls far better than `Invoke-WebRequest` for this specific large file),
then place it manually at the side-by-side path Gradle expects:

```powershell
Start-BitsTransfer -Source "https://dl.google.com/android/repository/android-ndk-r28c-windows.zip" -Destination "$env:TEMP\ndk.zip"
Expand-Archive "$env:TEMP\ndk.zip" "$env:TEMP\ndk-extract"
Move-Item "$env:TEMP\ndk-extract\android-ndk-r28c" "$env:ANDROID_HOME\ndk\28.2.13676358"
```

(Confirm the exact NDK version Gradle wants from a first `flutter build apk`
attempt's error output if `28.2.13676358` is no longer current.)

Set `ANDROID_HOME` and `ANDROID_SDK_ROOT` (User-level env vars) to the SDK
root, and add `platform-tools` to PATH.

### 4.3 Flutter

No official winget package. Clone stable directly:

```powershell
git clone https://github.com/flutter/flutter.git -b stable --depth 1 <path>\flutter
```

Add `<path>\flutter\bin` to PATH. First `flutter doctor` run downloads the
Dart SDK and bootstraps the tool — expect this to take a few minutes.

### 4.4 The space-in-username problem — check this first on a new device

**If the Windows username has a space in it** (this machine's did:
`Tuviksh Murudkar`), Flutter/Gradle native-asset build hooks for at least
one dependency (`objective_c`, pulled in transitively) **will fail** with
`'C:\Users\Firstname' is not recognized as an internal or external command`.
This is a real, reproducible failure, not a one-off — `flutter doctor`
itself warns about it for the Android SDK path specifically, but the actual
break was in the *project/Pub-cache* path, which `flutter doctor` doesn't
check. **8.3 short-name workarounds (`C:\Users\USERNA~1`) do NOT fix this**
— Dart's own path canonicalization on Windows resolves short names straight
back to the long (spaced) form internally, so the space reappears no matter
which alias you `cd` through.

**The only fix found**: physically relocate Flutter, the Pub cache, and a
working copy of the project to a path with no spaces anywhere in it. This
session used `C:\dev\`:

```powershell
git clone https://github.com/flutter/flutter.git -b stable --depth 1 C:\dev\flutter
# Move Pub cache to C:\dev\pub-cache (or set PUB_CACHE fresh there)
# Copy the project (excluding .git, build, .dart_tool, and anything huge/
# irrelevant to the Flutter build — model_training/, backend/, docs/,
# magisk-privileged-module/) to C:\dev\voice_guard:
robocopy "<original project path>" "C:\dev\voice_guard" /E /XD ".git" "build" ".dart_tool" "model_training" "magisk-privileged-module" "backend" "docs" ".gradle"
```

Then set `PUB_CACHE=C:\dev\pub-cache` and run all `flutter` commands from
inside `C:\dev\voice_guard`, not the original (spaced) path. If the new
device's username has no space, **skip this entire step** — build directly
in place.

### 4.5 Build + install

```bash
cd <space-free project path>
export JAVA_HOME="<jdk path>"
export ANDROID_HOME="<sdk path>"
export PUB_CACHE="<space-free pub cache path>"   # only if you relocated (§4.4)
export PATH="<flutter path>/bin:$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"
flutter pub get
flutter build apk --flavor privileged --release
adb devices                                       # confirm phone shows "device", not "unauthorized"
adb install -r build/app/outputs/flutter-apk/app-privileged-release.apk
```

If `adb devices` shows `unauthorized`: check the phone screen for an "Allow
USB debugging?" prompt and accept it (tick "always allow from this
computer").

### 4.6 Watching live behavior

```bash
adb logcat -c
adb logcat -s flutter:V
```

Look for `ONNX model loaded from assets/models/voice_detector.onnx` on
launch, then `Monitor: raw=... ema=... state=... label=...` lines as audio
streams, and `ALERT fired` on escalation. (Debug prints route through
logcat's `flutter` tag on Android.) If piping this through `grep` and
seeing `Binary file (standard input) matches` instead of actual lines, add
`--text` to the grep invocation — a stray non-UTF8 byte in the logcat
stream trips binary detection otherwise.

## 5. Open items for the pending final analysis pass

Not resolved this session, listed here rather than silently dropped:

- The 4 false alerts observed live all occurred during continuous 20s+
  speech, never in isolated bursts — worth deciding whether to tune the
  alert-gating logic (currently requires 2+ consecutive high windows) to be
  less sensitive to brief runs of elevated `raw` scores during long
  speech, rather than only mitigating via "keep turns short" demo guidance.
- The accent-expansion work (Hindi, foreign-accented English coverage) is
  validated as *not worse* than baseline on ITW but never proven *better*,
  and is **not included** in the currently-deployed noise-fix model at all.
  Whether/how to fold it back in (the curriculum-training approach, or the
  new-data-only approach, both explored this session) is still open.
- `data/real_noise_aug_split` was a quick 80/20 split for this session's
  purposes — not integrated into the project's existing `split_accents.py`
  speaker-disjoint-split infrastructure. Worth reconciling if this becomes
  a permanent part of the training recipe rather than a one-off.
- The real (not just offline-estimated) false-alert rate under sustained
  speech longer than ~114s cumulative test coverage is still an
  extrapolation, not directly measured.
