# Implementation Backlog: Ops & Reproducibility (MOD-D)

| Task ID | Subject | Technical Requirement | Definition of Done (DoD) | Deps |
|---|---|---|---|---|
| MOD-D-01 | Implement GPU Check | Build `ops/gpu_check.py` to verify `torch.cuda.get_device_capability()`. | Correctly flags sm_120 vs sm_89. | None |
| MOD-D-02 | Build Makefile | Implement `setup`, `data`, `train`, `eval`, `demo` targets. | `make demo` launches the app on a fresh clone. | MOD-D-01 |
| MOD-D-03 | Implement R1 Data Path | Script to pull processed dataset from HuggingFace Hub. | Data downloads and manifest is verified. | MOD-D-02 |
| MOD-D-04 | Implement R2 Data Path | Hook `make data` to run TeleChannel pipeline from scratch. | Local dataset matches HF checksums. | MOD-A-13, MOD-D-02 |
| MOD-D-05 | Create Resume Wrapper | Build `scripts/run_resume.sh` with the `while` loop and HF Hub sync. | Training survives a manual process kill. | MOD-B-04 |
| MOD-D-06 | Setup CI Pipeline | Configure `.github/workflows/ci.yml` (Ruff, Pytest, File Guard). | Green badge on README; 10MB file upload is blocked. | None |
| MOD-D-07 | Implement Repro Card | Create `repro_card.md` template and automation for `config_hash`. | Every model release includes a populated Repro Card. | MOD-B-09 |
| MOD-D-08 | Build Docker Image | Create `Dockerfile` for one-command environment reproduction. | Container runs `make demo` on any machine. | MOD-D-02 |
| MOD-D-09 | Final Rota Setup | Establish `ops/rota.md` and Sunday Ritual process. | Rota updated for Week 1. | None |
