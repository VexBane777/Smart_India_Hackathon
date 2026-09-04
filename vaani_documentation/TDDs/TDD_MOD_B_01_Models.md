# TDD: Model Training & Inference Engine (MOD-B-01)
**Source Spec:** DOC2_TRAINING_CONFIGS.md
**Module:** B (Model)

## 1. System Overview
The Model Engine implements the "evolution ladder" of detectors. It moves from a simple TinyCNN for the demo to a heavyweight SSL Teacher for the accuracy ceiling, finally distilling that knowledge into a fast, deployable Student model.

**Key Goal:** Every model must be interchangeable via a single config line (`models.meta.json`).

## 2. Component Architecture

### 2.1 Configuration & Registry (`registry.py`)
A centralized way to instantiate models from YAML.
- **Registry:** A dictionary mapping `type` (e.g., `tinycnn`) to a class.
- **Config Loader:** Reads `base.yaml` $\rightarrow$ merges specific config (e.g., `cnn_week1.yaml`) $\rightarrow$ returns a config object.

### 2.2 Model Architectures (`models/`)

#### TinyCNN (The Fallback/Student)
- **Structure:** 4x [Conv2d $\rightarrow$ BatchNorm $\rightarrow$ GELU $\rightarrow$ MaxPool] $\rightarrow$ AdaptiveAvgPool $\rightarrow$ Linear.
- **Input:** (B, 1, 64, T) Mel-spectrogram.
- **Output:** 2-class logits (Real vs Fake).

#### SSL Head (The Teacher)
- **Structure:** Pretrained `wav2vec2-base` $\rightarrow$ Learnable weighted sum of 12 hidden layers $\rightarrow$ 2-layer MLP.
- **Input:** (B, T) Raw Audio.
- **Output:** 2-class logits.

#### The Cascade (Logic)
- **Router:** If `student_score` $\in [0.35, 0.70]$, trigger `teacher_model.predict()`.
- **Timeout:** Teacher must return within `teacher_budget_ms`.

### 2.3 Training Pipeline (`train.py`)
A unified trainer supporting three modes:
1. **Standard:** Cross-Entropy loss on labels.
2. **Distillation:** $L = \alpha \cdot CE(\text{student}, y) + (1-\alpha) \cdot T^2 \cdot KL(\text{student}, \text{teacher})$.
3. **Calibration:** Temperature scaling $T$ to align confidence with EER.

## 3. Implementation Deep-Dive

### 3.1 The Distillation Cache
To avoid running the heavyweight Teacher during every Student epoch:
1. **Offline Pass:** Run Teacher on the full training set $\rightarrow$ save `logits_cache.parquet`.
2. **Training:** Student reads pre-computed teacher logits from the cache.

### 3.2 Inference Optimization
- **ONNX Export:** `torch.onnx.export` with dynamic time axis.
- **Quantization:** `torch.quantization.quantize_dynamic` for INT8 weights.
- **Parity Gate:** Max difference between FP32 and INT8 scores must be $< 0.02$.

## 4. Verification Plan

| Component | Test Case | Expected Outcome |
|---|---|---|
| **Registry** | Change `streaming_default` in `.json` | App immediately uses the new model without restart. |
| **Distillation** | Student vs Teacher EER | Student within $3\times$ Teacher EER on in-domain set. |
| **Calibration** | Reliability Diagram | Bins show confidence $\approx$ accuracy; ECE $\leq 5\%$. |
| **Latency** | CPU-forced bench | p95 $\leq 30$ ms per 2s window. |
| **Quantization** | FP32 vs INT8 | Max score delta $\leq 0.02$. |
