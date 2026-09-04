# Implementation Backlog: TeleChannel Pipeline (MOD-A)

| Task ID | Subject | Technical Requirement | Definition of Done (DoD) | Deps |
|---|---|---|---|---|
| MOD-A-06 | Verify Codec Support | Run `ffmpeg -encoders` check from Doc 1 §1.3. | All target codecs (GSM, AMR, Opus, G.711) present. | None |
| MOD-A-07 | Implement RIR Stage | Create `stages/rir.py` with FFT convolution and direct-path alignment. | `test_rir` passes; output matches SLR28 sample. | None |
| MOD-A-08 | Implement Noise Stage | Create `stages/noise.py` with SNR-based additive mixing. | `test_noise` verifies target SNR $\pm 1$ dB. | None |
| MOD-A-09 | Implement Mic Stage | Create `stages/mic.py` with random gain and probabilistic clipping. | Clipping occurs exactly at $\pm 0.5$ as specified. | None |
| MOD-A-10 | Implement Codec Stage | Create `stages/codec.py` using FFmpeg subprocess for round-trips. | `test_codec` verifies output SR is 16kHz. | MOD-A-06 |
| MOD-A-11 | Implement Loss Stage | Create `stages/packetloss.py` with Gilbert-Elliott and concealment. | Loss burstiness verified via length histogram. | None |
| MOD-A-12 | Implement BandLimit | Create `stages/bandlimit.py` with Butterworth 300-3400Hz filter. | Frequency response plot shows sharp cutoff. | None |
| MOD-A-13 | Build Pipeline Core | Create `pipeline.py` to chain stages based on `channels.yaml` recipes. | `python pipeline.py --recipe pstn` produces a valid WAV. | MOD-A-07..12 |
| MOD-A-14 | Implement Manifest | Create `manifest.py` to log metadata to Parquet. | Parquet file contains all fields from Doc 1 §1.6. | MOD-A-13 |
| MOD-A-15 | Build Split Logic | Implement leakage-safe splits in `manifest.py` (source-clip level). | No overlap between `train` and `test` for same clip. | MOD-A-14 |
| MOD-A-16 | Implement QA Gates | Create `qa.py` to check for silence, clipping, and duration. | `qa_failures.jsonl` correctly captures bad clips. | MOD-A-13 |
