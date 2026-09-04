# Implementation Backlog: Model Engine (MOD-B)

| Task ID | Subject | Technical Requirement | Definition of Done (DoD) | Deps |
|---|---|---|---|---|
| MOD-B-01 | Build Model Registry | Implement `registry.py` and YAML config loader. | `model = registry.load("cnn_week1")` works. | None |
| MOD-B-02 | Implement TinyCNN | Build `models/cnn.py` with the specified 4-layer Conv architecture. | Model initializes and accepts (B,1,64,T) input. | MOD-B-01 |
| MOD-B-03 | Implement SSL Head | Build `models/ssl_head.py` using wav2vec2-base + layer-weighting. | Model initializes and accepts (B,T) raw audio. | MOD-B-01 |
| MOD-B-04 | Create Unified Trainer | Build `train.py` supporting CE and KL-Distillation losses. | Training run completes 1 epoch; logs to MLflow. | MOD-B-02, MOD-B-03 |
| MOD-B-05 | Generate Logits Cache | Script to run Teacher over train set $\rightarrow$ `logits_cache.parquet`. | Parquet file exists and matches manifest size. | MOD-B-03, MOD-B-04 |
| MOD-B-06 | Train Student v1 | Run distillation training using the logits cache. | Student weights saved to HF Hub; EER measured. | MOD-B-04, MOD-B-05 |
| MOD-B-07 | Implement Calibration | Build `calibrate.py` for temperature scaling. | Reliability diagram produced; ECE $\le 5\%$. | MOD-B-06 |
| MOD-B-08 | Implement Cascade | Build the Router logic (uncertainty band $\rightarrow$ Teacher). | Cascade reduces EER compared to Student-only. | MOD-B-02, MOD-B-03 |
| MOD-B-09 | Export & Quantize | Implement `export_onnx.py` with INT8 quantization. | `.onnx` file produced; parity gate $\le 0.02$ passed. | MOD-B-06 |
| MOD-B-10 | Bench Latency | Implement `bench_latency.py` (p50/p95/p99 on CPU). | `latency.json` generated for all models. | MOD-B-09 |
