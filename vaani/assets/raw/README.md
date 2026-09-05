# assets/raw/README.md — Demo Asset Recording (Module C, Task 1)

> **Status: PLACEHOLDER audio.** The plan's real recording step (3 human takes per
> segment, earbud mic, quiet room, ~−6 dBFS peaks, Form A consent signed first) has
> **not** been done. These WAVs were machine-generated so the pipeline, UI, and cue
> timing work could be developed in parallel without blocking on a recording session.
> Replace with real takes before any pitch/demo use; the manifest records provenance.

## What lives here

48 kHz mono raw WAVs, per Module C Task 1 (48kHz mono RAW, peaks ~−6 dBFS):

| File | Content | Voice source |
|---|---|---|
| `call_A_raw.wav` | "The Vendor Payment" script — clone attack call (primary) | TTS placeholder (Speaker 1 + "CFO" clone slot) |
| `call_N_raw.wav` | Identical script, control (donor's real voice) | TTS placeholder (different voice for Speaker 2) |
| `call_B_raw.wav` | Accent variant (Tamil-English flavour, per Doc 4 §4.6) | TTS placeholder |

Scripts and timing live in `call_scripts.json` (single source of truth — assembled
from Doc 4 §4.2 and §4.6). `assets/manifest.json` records provenance, QA status,
and the hold-out rule (Doc 4 §4.7: `split: demo`, never trains).

## How they were generated

`python assets/scripts/build_demo_placeholders.py` — Windows SAPI TTS per segment,
assembled at natural turn boundaries with 100–300 ms gaps (Doc 4 §4.5 rule 1),
peak-normalized to −6 dBFS. Regenerate anytime; script is deterministic.

## Before real recordings (checklist from Doc 4 §4.4/§4.8)

1. Doc 6 Form A signed for every speaker (cloning + posting opt-in) — nothing records
   a human before this gate.
2. 48 kHz mono, earbud mic ~15 cm / 30–45° off-axis, smallest room, fan/AC off.
3. Peaks ~−6 dBFS, never at ceiling; 3 takes per segment; 30 s room tone.
4. Donor: 10 min clean reference audio for RVC.
5. QA: duration 85–92 s, VAD coverage 55–85%, ≥20 s continuous synthetic speech,
   ≥15 s leading real speech, transcription spot-check ≥95%.
