<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 7 — Repo README & Reproducibility Card Spec · Owner: Systems Lead · Status: Build-ready
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 7 — Repo README & Reproducibility Card Spec

**Owner:** Systems Lead · **Status:** Build-ready · **Suite:** v1.0 · **Generated:** 2026-09-03
**Depends on:** Doc 2 configs exist

---

## 7.1 Final repo tree
```
vaani/
├── README.md  Makefile  Dockerfile  .github/workflows/ci.yml
├── configs/            # Doc 2
├── vaani/              # package: models/, train.py, engine/, evaluate.py,
│                       #   calibrate.py, export_onnx.py, bench_latency.py,
│                       #   telechannel/ (Doc 1)
├── tests/              # unit tests with synthetic fixtures (no audio in CI)
├── ops/                # rota.md (Doc 8), runbook.md, gpu_check.py
├── legal/              # Doc 6 templates, privacy note, license manifest
├── assets/             # Doc 4 demo assets + cues.json; Doc 5 passages
├── data/manifests/     # parquet manifests (in git — small, reviewable)
├── scripts/            # verify_data.py, make_release.py, hf_upload.py
└── runs/               # (gitignored) checkpoints, metrics, latency.json
```
**Audio never enters git.** Two reproducibility levels declared in the README:
**R1** — reproduce public results from the public dataset release; **R2** — regenerate
the corpus from sources via the recipe (requires individual dataset registrations,
honestly stated).

## 7.2 README skeleton
1. What this is (3 lines + demo GIF) · 2. Quickstart (`make setup && make data && make demo`)
· 3. Reproducibility Card (§7.3) · 4. Results leaderboard (four protocols + external,
3-seed mean ± std, FPR@TPR90, ECE, latency) · 5. Architecture (one diagram) ·
6. TeleChannel: recipe + validation figure (Doc 5) · 7. Privacy & consent (Doc 6) ·
8. Known limits & threats to validity (unseen generators · replay · adversarial) ·
9. R1/R2 + release instructions.

## 7.3 The Reproducibility Card (identical block in README + every model card)
```
REPRO CARD — model: student_v1 (tag v1.2.0, config_hash a1b2c3)
Environment: torch 2.7.x+cu128 · python 3.11 · [OS] · GPU: RTX 4050
             (also verified: T4/Kaggle)
Data snapshot: manifest sha256 [hash] · 1,032,441 windows
               splits: train/val/test/demo
Commands:
    make train CONFIG=student_distill SEED=42      # ~4.1 h on 4050 #2
    make eval  PROTOCOL=all                        # writes leaderboard.md
Expected results (mean ± std over seeds 42/7/2026):
    EER in_domain         [M]% ± [M]   tolerance: |yours − mean| ≤ 2σ or 0.5% abs
    EER unseen_generator  [M]% ± [M]
    FPR@TPR90             [M]% ± [M]
    p95 CPU latency       [M] ms       tolerance: ± 20%
Verification runs (append a row when you reproduce):
    | who | machine | date | tag | pass? |
```
The tolerance column is the honesty mechanism: anyone — including a judge — reruns and
checks against a band, not a vibes-match. A Kaggle row proves environment-independence.

## 7.4 Makefile (spec)
```makefile
setup:   ## pin deps incl. cu128 build; ops/gpu_check.py; print capability table
data:    ## pull HF dataset (R1) or regenerate via TeleChannel (R2); verify_data.py
train:   ## python -m vaani.train --config configs/$(CONFIG).yaml --seed $(SEED)
eval:    ## run eval_protocols.yaml -> leaderboard.md; call-level + window-level
demo:    ## force CPU; preload model; launch FastAPI + Streamlit; offline-safe
bench:   ## latency.json (p50/p95/p99) + INT8 parity check
release: ## model card + repro card + hf_upload.py + tag vX.Y.Z
```

## 7.5 CI (GitHub Actions, free)
On push: ruff lint · pytest with synthetic fixtures (sines + noise — CI needs zero
audio) · `verify_data.py --manifest-only` (schema-valid, license non-null, no clip in
two splits) · guard rejecting any file >5 MB (audio never enters git) · config sanity.
Weekly: Docker build test · 50-step fixed-seed mini train on CPU asserting loss within
tolerance (catches environment drift). Green badge in the README — a slide-proof artifact.

## 7.6 Versioning & release
SemVer on the pipeline; `config_hash` on every run; model tags on HF Hub; dataset
release with datasheet. Recommended: **Zenodo archive of the public dataset — free, and
gives a permanent DOI** (a citable identifier), making "we published a dataset" literally
verifiable.

## 7.7 Done means (the new-user test)
A teammate who has never seen the repo: fresh clone → `make setup && make data && make
demo` → working demo in one sitting — **performed once for real, recorded as a
verification-run row.** CI green Monday morning with no human intervention. Repro card
rows exist for a local GPU and one Kaggle run.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
