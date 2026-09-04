# VAANI Module B (Model Engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the "evolution ladder" of detectors: from a TinyCNN fallback to a heavyweight SSL Teacher, and finally a distilled Student model for real-time deployment.

**Architecture:** A registry-based system where models are instantiated from YAML. The launcheable engine supports a "Cascade" mode where a fast student screens inputs and escalates unsure cases to a slow, accurate teacher.

**Tech Stack:** PyTorch, ONNX, MLflow, HuggingFace Hub, NumPy.

## Execution Order & Cross-Module Dependencies
**Depends on Module A:** Tasks 2–4 (TinyCNN, SSL Teacher, Trainer) need data produced by
Module A Task 9 (manifest & splits) to train against — at minimum a small balanced window
set, not the full corpus. Task 5 (Teacher training) needs Module A's full recipe set to be
meaningful, so it is a Weeks 5–8 task, not Week 1.

- **Downstream consumers:** Module C Task 2/4 need an exported/trained model
  (this module's Task 2 minimum, Task 9 for the deployable ONNX/INT8 build) before UI
  and demo-asset verification can run end to end. Module D's `make train`/`make eval`
  wrap this module's `train.py`/`evaluate.py` directly.
- **Week 1 note (master plan §8):** only Tasks 1–2 (registry + TinyCNN) plus enough of
  Task 4 (basic training loop, no distillation) to produce a working checkpoint are
  Week 1 scope — the Tuesday demo needs a trained TinyCNN, not the full evolution
  ladder. Tasks 3, 5–8 (SSL teacher, distillation, calibration, cascade) start Week 5+.

## Global Constraints
- **VRAM Ceiling**: Models must run on 6-8GB VRAM (use LoRA/Grad-Checkpointing for Teacher).
- **Latency**: Student p95 $\leq 30$ms per 2s window on laptop CPU.
- **Interchangeability**: All models must be swappable via `models.meta.json`.
- **Repro**: All finals must be 3-seed mean $\pm$ std.

---

### Task 1: Model Registry & Configuration

**Files:**
- Create: `vaani/registry.py`
- Create: `vaani/configs/base.yaml`
- Create: `vaani/configs/cnn_week1.yaml`
- Create: `vaani/configs/ssl_teacher.yaml`
- Create: `vaani/configs/student_distill.yaml`

**Interfaces:**
- Produces: `registry.load(config_path) -> nn.Module`

- [ ] **Step 1: Implement Registry**
  In `registry.py`, create a `MODEL_REGISTRY` dictionary mapping strings to classes.
- [ ] **Step 2: Implement Config Loader**
  Implement `load_config(path)` that reads YAML and merges it with `base.yaml`.
- [ ] **Step 3: Define Base Configs**
  Create `base.yaml` with shared audio specs (16kHz, 2s win, 64 mels).
- [ ] **Step 4: Define Model Configs**
  Create `cnn_week1.yaml`, `ssl_teacher.yaml`, and `student_distill.yaml` as per TDD_MOD_B_01.
- [ ] **Step 5: Commit**
  `git add vaani/registry.py vaani/configs/`
  `git commit -m "feat: implement model registry and configuration system"`

### Task 2: TinyCNN Implementation

**Files:**
- Create: `vaani/models/cnn.py`
- Test: `tests/models/test_cnn.py`

**Interfaces:**
- Produces: `TinyCNN(n_mels=64, chs=(32,64,128,256))`

- [ ] **Step 1: Write failing test**
  Create `tests/models/test_cnn.py`. Verify that passing a (1, 1, 64, 200) tensor returns a (1, 2) logit tensor.
- [ ] **Step 2: Implement TinyCNN Architecture**
  In `cnn.py`, implement 4x [Conv2d $\rightarrow$ BN $\rightarrow$ GELU $\rightarrow$ MaxPool] followed by AdaptiveAvgPool and a Linear head.
- [ ] **Step 3: Verify Output Shape**
  Run test. Ensure output is exactly (Batch, 2).
- [ ] **Step 4: Commit**
  `git add vaani/models/cnn.py tests/models/test_cnn.py`
  `git commit -m "feat: implement TinyCNN architecture"`

### Task 3: SSL Teacher Implementation

**Files:**
- Create: `vaani/models/ssl_head.py`
- Test: `tests/models/test_ssl.py`

**Interfaces:**
- Produces: `SSLHead(base="facebook/wav2vec2-base")`

- [ ] **Step 1: Implement SSL Head**
  In `ssl_head.py`, load `Wav2Vec2Model`. Implement a learnable vector `w` (size 12) to weight-sum the hidden states.
- [ ] **Step 2: Implement Classifier**
  Add a 2-layer MLP (768 $\rightarrow$ 256 $\rightarrow$ 2) to the mixed hidden state.
- [ ] **Step 3: Freeze Feature Extractor**
  Set `requires_grad = False` for the SSL base feature extractor.
- [ ] **Step 4: Verify Forward Pass**
  Test that passing (1, 32000) raw audio returns (1, 2) logits.
- [ ] **Step 5: Commit**
  `git add vaani/models/ssl_head.py tests/models/test_ssl.py`
  `git commit -m "feat: implement SSL Teacher architecture"`

### Task 4: Unified Trainer (Standard & Distillation)

**Files:**
- Create: `vaani/train.py`
- Test: `tests/train/test_trainer.py`

**Interfaces:**
- Consumes: `registry.load(config)`
- Produces: `model.pth` + MLflow logs.

- [ ] **Step 1: Implement Basic Training Loop**
  In `train.py`, implement the boilerplate: DataLoader $\rightarrow$ Optimizer $\rightarrow$ Epoch loop $\rightarrow$ Validation.
- [ ] **Step 2: Implement Distillation Loss**
  Implement `distill_loss(s_logits, t_logits, y, T=2.0, alpha=0.5)` using KL-Divergence.
- [ ] **Step 3: Integrate MLflow**
  Add `mlflow.log_metrics` for EER, Loss, and Accuracy.
- [ ] **Step 4: Implement HF Hub Sync**
  Add logic to push checkpoints to `vaani/models/<name>` on the HuggingFace Hub.
- [ ] **Step 5: Commit**
  `git add vaani/train.py tests/train/test_trainer.py`
  `git commit -m "feat: implement unified trainer with distillation support"`

### Task 5: Teacher Training & Logits Cache

**Files:**
- Create: `scripts/generate_cache.py`
- Modify: `vaani/train.py`

**Interfaces:**
- Produces: `runs/ssl_teacher/logits_cache.parquet`

- [ ] **Step 1: Train SSL Teacher**
  Run `python vaani/train.py --config configs/ssl_teacher.yaml`. Use 5050 GPU.
- [ ] **Step 2: Implement Logits Cache Generation**
  In `generate_cache.py`, run the trained Teacher over the full training set.
- [ ] **Step 3: Save to Parquet**
  Export the `(clip_id, logits)` pairs to `logits_cache.parquet`.
- [ ] **Step 4: Verify Cache**
  Verify that the Parquet file size matches the expected number of windows.
- [ ] **Step 5: Commit**
  `git add scripts/generate_cache.py`
  `git commit -m "feat: generate teacher logits cache for distillation"`

### Task 6: Student Distillation & Training

**Files:**
- Modify: `vaani/train.py`
- Test: `tests/models/test_distillation.py`

**Interfaces:**
- Consumes: `logits_cache.parquet`
- Produces: `student_v1.pth`

- [ ] **Step 1: Implement Cache-based Loading**
  Modify `train.py` to load teacher logits from the Parquet file during the Student loop.
- [ ] **Step 2: Run Distillation Training**
  Execute `python vaani/train.py --config configs/student_distill.yaml`.
- [ ] **Step 3: Measure EER**
  Evaluate Student on the held-out test set. Verify it is within $3\times$ Teacher EER.
- [ ] **Step 4: Commit**
  `git commit -m "feat: train distilled student model"`

### Task 7: Calibration (Temperature Scaling)

**Files:**
- Create: `vaani/calibrate.py`
- Test: `tests/models/test_calibration.py`

**Interfaces:**
- Produces: `runs/<name>/calibration/T_value`

- [ ] **Step 1: Implement Temperature Scaling**
  In `calibrate.py`, find $T$ that minimizes Negative Log-Likelihood (NLL) on the validation set.
- [ ] **Step 2: Compute ECE**
  Implement Expected Calibration Error (ECE) calculation.
- [ ] **Step 3: Generate Reliability Diagram**
  Plot "Confidence vs Accuracy" bins. Verify ECE $\leq 5\%$.
- [ ] **Step 4: Commit**
  `git add vaani/calibrate.py tests/models/test_calibration.py`
  `git commit -m "feat: implement model calibration"`

### Task 8: The Cascade Router

**Files:**
- Create: `vaani/engine/cascade.py`
- Test: `tests/engine/test_cascade.py`

**Interfaces:**
- Consumes: `StudentModel`, `TeacherModel`
- Produces: `final_score`

- [ ] **Step 1: Implement Router Logic**
  In `cascade.py`, implement the "Unsure Band" check: if $0.35 \leq \text{score} \leq 0.70$, call Teacher.
- [ ] **Step 2: Implement Timeout Guard**
  Wrap the Teacher call in a thread with a `teacher_budget_ms` timeout.
- [ ] **Step 3: Verify Accuracy Boost**
  Run a test set. Verify that Cascade EER $<$ Student-only EER.
- [ ] **Step 4: Commit**
  `git add vaani/engine/cascade.py tests/engine/test_cascade.py`
  `git commit -m "feat: implement cascade routing logic"`

### Task 9: ONNX Export & Quantization

**Files:**
- Create: `vaani/export_onnx.py`
- Test: `tests/models/test_onnx.py`

**Interfaces:**
- Produces: `model_int8.onnx`

- [ ] **Step 1: Implement ONNX Export**
  Use `torch.onnx.export` with `dynamic_axes` for the time dimension.
- [ ] **Step 2: Implement Dynamic Quantization**
  Use `torch.quantization.quantize_dynamic` to convert weights to `qint8`.
- [ ] **Step 3: Run Parity Gate**
  Compare scores of FP32 vs INT8 on 100 clips. Verify $\text{max\_delta} \leq 0.02$.
- [ ] **Step 4: Commit**
  `git add vaani/export_onnx.py tests/models/test_onnx.py`
  `git commit -m "feat: implement ONNX export and INT8 quantization"`

### Task 10: Latency Benchmarking

**Files:**
- Create: `vaani/bench_latency.py`

**Interfaces:**
- Produces: `runs/<name>/latency.json`

- [ ] **Step 1: Implement Benchmarking Loop**
  Run 500 windows $\rightarrow$ 50 warmup $\rightarrow$ measure `perf_counter`.
- [ ] **Step 2: Force CPU Mode**
  Ensure `torch.set_num_threads(4)` and CUDA is disabled.
- [ ] **Step 3: Export Metrics**
  Save p50, p95, p99 and cold-start time to `latency.json`.
- [ ] **Step 4: Commit**
  `git add vaani/bench_latency.py`
  `git commit -m "feat: implement CPU latency benchmarking"`
