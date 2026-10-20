# Accent/clone data-collection subsystem

Closes a gap surfaced during real-device testing (see `voice_guard/state.md`):
the deployed model had never been evaluated against genuine live-mic speech
with real ambient noise, nor against accents outside ASVspoof/In-the-Wild's
coverage. Target matrix, 8 cells: `{en, hi} x {native, foreign} x {real, fake}`.

**Status as of this writing: scaffolding only, not run.** Written on a
machine without a GPU, `datasets`, or `TTS` installed, per instruction not to
download or generate anything here — everything needed to actually run this
is captured below and in each script's docstring instead. Run it on the GPU
machine (state.md mentions an RTX 5050 available).

## Layout

```
data/accents/real/{en_native,en_foreign,hi_native,hi_foreign}/*.wav
data/accents/fake/{en_native,en_foreign,hi_native,hi_foreign}/*.wav
data/accent_manifest.csv   # file,label,cell,source,speaker_id,duration_s
```

Schema lives in `accent_common.py` — import from there, don't redefine it in
a new script. `*.wav` is git-ignored (`voice_guard/.gitignore`); the CSV
manifest is not, and is the intended durable record of what's been collected.

## Scripts, in run order

1. **`prep_accents_real.py`** — pulls "real" clips from Mozilla Common Voice
   (`mozilla-foundation/common_voice_17_0` on Hugging Face). Needs a free HF
   account, `huggingface-cli login`, and accepting the dataset's terms once at
   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 .
   `pip install datasets`.
   ```
   python prep_accents_real.py --lang en --limit 200
   python prep_accents_real.py --lang hi --limit 200
   ```
   Expect `hi_foreign` to come out thin — Common Voice's Hindi accent tagging
   is sparse. That's expected, not a bug; see step 4.

2. **`gen_accents_fake.py`** — clones each real clip's speaker via Coqui
   XTTS-v2 (open-weight, free, local GPU inference) reading a stock sentence
   in the same language, landing in the matching `fake/<cell>/`.
   `pip install TTS`. Model: https://huggingface.co/coqui/XTTS-v2 — **license
   is Coqui's CPML (non-commercial)**, fine for this prototype, flag before
   any commercial use.
   ```
   python gen_accents_fake.py --lang en --limit 200
   python gen_accents_fake.py --lang hi --limit 200
   ```

3. **`ingest_self_recordings.py`** — no network/GPU needed, safe anywhere.
   Record ~10-30s clips any way convenient (voice memo, WhatsApp note, etc.),
   including some noisy/background-chatter takes on purpose, drop the raw
   files in `data/incoming_self/`, then:
   ```
   python ingest_self_recordings.py --lang en
   python ingest_self_recordings.py --lang hi
   ```

4. **`report_accent_coverage.py`** — no network/GPU needed. Tallies minutes
   per cell against a 30-minute-per-cell target and prints which cells are
   still `THIN` after steps 1-3:
   ```
   python report_accent_coverage.py
   ```
   Whatever's flagged is the manual-supplement checklist — go find more of
   that specific cell and drop `.wav` files straight into
   `data/accents/<real|fake>/<cell>/` plus a manifest row (`source: manual`
   is fine). Candidates worth searching manually for the thin cells (not yet
   verified to have usable licenses/format — check before bulk-downloading):
   OpenSLR's Hindi corpora, IIT Madras IndicTTS, VCTK (accented English).

5. **`split_accents.py`** — no network/GPU needed. Speaker-disjoint
   train/held split per cell (same rationale as `prep_in_the_wild.py`: fold
   everything into training and there's nothing left to measure "does it
   generalize" with). A cloned clip inherits its source speaker's id, so a
   real/clone pair never splits across train/held.
   ```
   python split_accents.py --held-out-fraction 0.2
   ```
   Produces `data/accents_split/{train,held}/{real,fake}/<cell>/`. Run
   `report_accent_coverage.py` first — a cell with only one speaker (e.g.
   early on, `en_native` with just your own recordings) goes entirely to
   `train` rather than losing its only speaker to `held`.

6. **Per-cell evaluation**: `../evaluate.py` scores the held accent cells
   (`accents_split/held/`, capped at 300 files each, test split only) as
   their own eval sets under the channel protocol, with FPR/FNR per cell and
   EER for cells that have both real and fake. `eval_accent_cells.py` was
   retired 2026-09-11 (it was broken and clean-only). See
   `../../docs/EVAL-PROTOCOL.md`.

### Folding into training

The accent cells are **not** in the v12 training corpus
(`../corpus.py::TRAIN_SETS_V12`). v6 trained on them and regressed, and
their held splits now serve as an Indian-accent generalization check. To
add them, append `SetDef`s to a new corpus tuple in `corpus.py`. Training
then renders them through `none` plus one phone channel like every other
set. There is no longer a "clean-only" training path (policy:
EVAL-PROTOCOL.md §1).

## Known caveats

- "native"/"foreign" is a coarse proxy from Common Voice's self-reported
  accent tag, not a verified L1 claim about any speaker — see
  `accent_common.py`'s docstring.
- `gen_accents_fake.py`'s clones share content (stock sentences) across
  speakers within a language — fine for a detector (which shouldn't care
  about text content), but don't reuse these clips for anything
  content-sensitive.
- The 30-minute-per-cell target in `accent_common.TARGET_MINUTES_PER_CELL` is
  a starting guess, not derived from an experiment — revisit once the next
  subsystem (training-pipeline extension) shows what actually moves held-out
  EER per cell.
- The accent cells carry known shortcut history (sample-rate and silence
  confounds, fixed 2026-09-10; see `../test_dataset_integrity.py`). Treat
  per-cell results as a check, not a leaderboard.
