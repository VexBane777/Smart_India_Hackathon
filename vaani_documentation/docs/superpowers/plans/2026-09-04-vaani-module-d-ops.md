# VAANI Module D (Ops & Reproducibility) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a "one-command" reproducibility pipeline and a federated GPU cluster management system.

**Architecture:** A Makefile-driven lifecycle that abstracts complex setup (dependencies, data fetching, training) into simple targets. Reproducibility is ensured via a "Repro Card" and a two-tier data access path (R1 Fast / R2 Deep).

**Tech Stack:** GNU Make, Docker, GitHub Actions, SSH/tmux, PyTorch.

## Execution Order & Cross-Module Dependencies
This module is largely **orthogonal** and can start Week 1 in parallel with A/B/C — Tasks
1 (GPU check), 2 (Makefile skeleton), 6 (CI), and 9 (GPU rota) need no other module's
output. But the Makefile targets are thin wrappers, so they only become *runnable*
end-to-end once their target module lands: `make data` (Task 2 Step 2, Tasks 3–4) needs
Module A's pipeline; `make train`/`make eval` (Task 2 Steps 3–4) need Module B's
`train.py`/`evaluate.py`; `make demo` (Task 2 Step 5, Task 8 Dockerfile entrypoint) needs
Module C's `app.py` stack. Write the Makefile skeleton early; wire each target as its
dependency module delivers the underlying script, rather than blocking this module's start
on the others finishing.

## Global Constraints
- **Git Hygiene**: No audio files in git; use HF Hub or local data paths.
- **Env Parity**: All machines must use PyTorch $\ge 2.7$ with cu128 wheels.
- **Repro Levels**: R1 (Fast - HF download) and R2 (Deep - TeleChannel regenerate).
- **Hardware**: CUDA disabled for the final demo.

---

### Task 1: GPU Health & Capability Check

**Files:**
- Create: `ops/gpu_check.py`

**Interfaces:**
- Produces: Capability table (Model, VRAM, sm_version).

- [ ] **Step 1: Implement Capability Probe**
  In `gpu_check.py`, use `torch.cuda.get_device_capability()` to detect the chip version.
- [ ] **Step 2: Implement VRAM & Driver Check**
  Use `nvidia-smi` or `torch.cuda.get_device_properties` to verify available VRAM.
- [ ] **Step 3: Verify sm_120 (RTX 5050)**
  Explicitly flag if the GPU is a 5050 but PyTorch is not cu128 (detecting the "silent CPU fallback").
- [ ] **Step 4: Commit**
  `git add ops/gpu_check.py`
  `git commit -m "feat: implement GPU capability probe"`

### Task 2: The Unified Makefile

**Files:**
- Create: `Makefile`

**Interfaces:**
- Produces: `make setup`, `make data`, `make train`, `make eval`, `make demo`.

- [ ] **Step 1: Implement `make setup`**
  Install pinned requirements $\rightarrow$ run `ops/gpu_check.py`.
- [ ] **Step 2: Implement `make data`**
  Check for `R1` vs `R2` flag $\rightarrow$ call HF download or TeleChannel pipeline.
- [ ] **Step 3: Implement `make train`**
  Execute `python -m vaani.train --config configs/$(CONFIG).yaml`.
- [ ] **Step 4: Implement `make eval`**
  Run `evaluate.py` $\rightarrow$ write `leaderboard.md`.
- [ ] **Step 5: Implement `make demo`**
  Force CUDA off $\rightarrow$ launch FastAPI $\rightarrow$ launch Streamlit.
- [ ] **Step 6: Commit**
  `git add Makefile`
  `git commit -m "feat: implement unified lifecycle Makefile"`

### Task 3: Reproducibility Path R1 (Fast)

**Files:**
- Create: `scripts/hf_download.py`

**Interfaces:**
- Produces: `data/processed/` (HF Hub $\rightarrow$ Local).

- [ ] **Step 1: Implement HF Hub Downloader**
  Use `huggingface_hub.snapshot_download` to pull the processed dataset.
- [ ] **Step 2: Verify Checksums**
  Run a SHA256 check on the downloaded manifests to ensure integrity.
- [ ] **Step 3: Integrate with `make data`**
  Ensure `make data R1=1` triggers this script.
- [ ] **Step 4: Commit**
  `git add scripts/hf_download.py`
  `git commit -m "feat: implement R1 fast-repro data path"`

### Task 4: Reproducibility Path R2 (Deep)

**Files:**
- Modify: `Makefile`
- Modify: `telechannel/pipeline.py`

**Interfaces:**
- Produces: `data/processed/` (Clean $\rightarrow$ TeleChannel $\rightarrow$ Local).

- [ ] **Step 1: Implement R2 Hook**
  Ensure `make data R2=1` triggers the full `TeleChannel` pipeline.
- [ ] **Step 2: Verify Parity**
  Run R2 on a subset $\rightarrow$ compare results with R1 (HF).
- [ ] **Step 3: Commit**
  `git commit -m "feat: implement R2 deep-repro data path"`

### Task 5: Cluster Resume Wrapper

**Files:**
- Create: `scripts/run_resume.sh`

**Interfaces:**
- Consumes: `vaani/train.py`
- Produces: Resumed training session.

- [ ] **Step 1: Implement Resume Loop**
  Build a `while` loop that calls `train.py --resume auto`.
- [ ] **Step 2: Integrate HF Hub Checkpoints**
  Ensure the script checks `hf:<repo>` for the latest `.pth` file before starting.
- [ ] **Step 3: Add Crash Logging**
  Capture stderr on failure $\rightarrow$ write to `runs/<name>/crash/<date>.log`.
- [ ] **Step 4: Commit**
  `git add scripts/run_resume.sh`
  `git commit -m "feat: implement overnight resume wrapper"`

### Task 6: CI Pipeline (GitHub Actions)

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: Green badge on README.

- [ ] **Step 1: Implement Linting & Tests**
  Add `ruff check` and `pytest` (using synthetic sine-wave fixtures).
- [ ] **Step 2: Implement Audio Guard**
  Add a step that fails the build if any file $> 5$ MB is added to git.
- [ ] **Step 3: Implement Docker Build Test**
  Add a job that builds the image and runs `make demo` to verify environment parity.
- [ ] **Step 4: Commit**
  `git add .github/workflows/ci.yml`
  `git commit -m "feat: implement CI pipeline with audio guard"`

### Task 7: Repro Card Automation

**Files:**
- Create: `scripts/make_repro_card.py`

**Interfaces:**
- Produces: `repro_card.md`.

- [ ] **Step 1: Implement Config Hashing**
  Create a function to hash the YAML config and manifest SHA256.
- [ ] **Step 2: Implement Metrics Extraction**
  Pull EER and Latency from `leaderboard.md` and `latency.json`.
- [ ] **Step 3: Generate Markdown**
  Write the result to the `REPRO CARD` format defined in TDD_MOD_D_01.
- [ ] **Step 4: Commit**
  `git add scripts/make_repro_card.py`
  `git commit -m "feat: implement repro card automation"`

### Task 8: Dockerization

**Files:**
- Create: `Dockerfile`

**Interfaces:**
- Produces: `vaani-demo:latest` image.

- [ ] **Step 1: Define Base Image**
  Use `nvidia/cuda:12.x-base-ubuntu22.04` (for training) or `python:3.11-slim` (for demo).
- [ ] **Step 2: Install System Deps**
  Install `ffmpeg` and `git`.
- [ ] **Step 3: Copy Code & Entrypoint**
  Set `ENTRYPOINT ["make", "demo"]`.
- [ ] **Step 4: Commit**
  `git add Dockerfile`
  `git commit -m "feat: add Dockerfile for environment parity"`

### Task 9: GPU Rota & Rituals

**Files:**
- Create: `ops/rota.md`

**Interfaces:**
- Produces: Weekly schedule.

- [ ] **Step 1: Define Rota Template**
  Implement the "Slot $\times$ Machine" table from Doc 8 §8.2.
- [ ] **Step 2: Establish Sunday Ritual**
  Document the a-priori conflict check and GPU mapping process.
- [ ] **Step 3: Commit**
  `git add ops/rota.md`
  `git commit -m "docs: initialize GPU rota and social contract"`
