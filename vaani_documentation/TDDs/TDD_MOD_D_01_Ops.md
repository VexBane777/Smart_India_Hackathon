# TDD: Ops & Reproducibility System (MOD-D-01)
**Source Spec:** DOC7_REPO_REPRO.md + DOC8_GPU_ROTA.md
**Module:** D (Ops)

## 1. System Overview
The Ops system ensures that VAANI is not a "black box" but a reproducible science project. It manages the borrowed hardware (the "federated cluster") and provides a one-command path for any external reviewer to verify the results.

**Goal:** "Fresh clone $\rightarrow$ `make setup && make data && make demo` $\rightarrow$ Working app."

## 2. Component Architecture

### 2.1 The Reproducibility Card (`repro_card.md`)
A structured manifest attached to every model release.
- **Fields:** Model tag, config hash, environment (PyTorch/CUDA version), data snapshot (SHA256), and expected results.
- **Tolerance Band:** Defines a $\pm \sigma$ range for result verification.

### 2.2 The Makefile (The "One-Command" Interface)
Standardizes the lifecycle:
- `make setup`: Installs pinned deps; runs `ops/gpu_check.py`.
- `make data`: Fetches HF dataset (R1) or runs TeleChannel (R2).
- `make train`: Executes `vaani.train` with the given config.
- `make eval`: Generates `leaderboard.md` using the four hold-out protocols.
- `make demo`: Launches the CPU-forced Streamlit app.

### 2.3 Cluster Ops (`ops/`)
- **GPU Rota (`rota.md`)**: A social contract for machine usage.
- **Resume Wrapper (`run_resume.sh`)**: A `while` loop that restarts training from the last HF Hub checkpoint.
- **Health Check (`gpu_check.py`)**: Verifies `sm_120` (RTX 5050) support to prevent silent CPU fallback.

## 3. Implementation Deep-Dive

### 3.1 The "R1 vs R2" Reproducibility
- **R1 (Fast)**: `make data` $\rightarrow$ downloads pre-processed FLACs from HuggingFace.
- **R2 (Deep)**: `make data` $\rightarrow$ runs the TeleChannel pipeline from scratch using the recipe.

### 3.2 CI/CD Pipeline (GitHub Actions)
- **Linting**: Ruff for code style.
- **Testing**: Pytest with synthetic audio (sines) to avoid large binary uploads.
- **Guard**: Rejects any file $> 5$ MB to prevent audio leakage into git.
- **Verification**: Weekly Docker build test to catch environment drift.

## 4. Verification Plan

| Component | Test Case | Expected Outcome |
|---|---|---|
| **Setup** | `make setup` | `gpu_check.py` correctly identifies GPU capability (e.g., sm_89). |
| **Repro R1** | Fresh clone $\rightarrow$ `make data` | Dataset downloads from HF; manifest is valid. |
| **Repro R2** | Fresh clone $\rightarrow$ `make data` | TeleChannel runs; output matches HF checksums. |
| **Resume** | Kill `train.py` $\rightarrow$ rerun | Training resumes from last checkpoint with zero loss of epochs. |
| **CI Guard** | Git add 10MB file | GitHub Action fails the build. |
| **Demo** | `make demo` | App launches in $< 10$s; CUDA is disabled. |
