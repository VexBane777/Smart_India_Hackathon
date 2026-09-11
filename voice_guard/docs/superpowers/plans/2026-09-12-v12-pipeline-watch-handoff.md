# v12 pipeline watch — handoff for the next agent (2026-09-12 ~01:45 IST)

Worktree: `C:\Users\Tuviksh Murudkar\SIH\.worktrees\voiceguard-v12-hardening`
(branch `voiceguard-v12-hardening`, from `origin/vaani`).
Main checkout: `C:\Users\Tuviksh Murudkar\SIH`.
Venv python (ONLY one with soundfile/torch):
`C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe`
Feature cache (shared, gitignored, survives worktrees):
`C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache`

## 1. What we are doing

Proving — honestly — whether VoiceGuard works on **phone-channel audio**
(gsm_2g, volte, pstn, cellular_3g, tandem, whatsapp), not just clean mic
audio. All past evals were clean-only, which flattered the numbers. Committed
baseline already shows the damage: **deployed v9 (MLP) and v11 (SeqCNN) are
at chance on phone channels** (~0.48-0.52 EER pooled over phone channels on
the `select` split; see
`voice_guard/model_training/runs/eval_v9_v11_select/report.md`).

Three steps: (1) harden the eval — DONE, committed (committed select/test
split `eval_splits/held_out_split_v1.json`, channel policy, fp16 memmap
cache, SeqTCN with leave-attack-out attack-type head, single `evaluate.py`
harness, legacy scripts retired; spec `voice_guard/docs/EVAL-PROTOCOL.md`,
plan `docs/superpowers/plans/2026-09-11-v12-eval-hardening-handoff.md`,
status in `voice_guard/state.md` v12 section); (2) build feature caches —
RUNNING NOW; (3) train v12, select on `select`, final v9/v11/v12 report on
`test`, then deploy decision.

## 2. Already committed (don't redo)

- `4976a8d` v12 eval-hardening feature commit (tests: 116 passed).
- `3df5624` state.md v12 section (v11 confound-table correction etc).
- `05a4435` re-score of deployed v11 on `select` (chance-level baseline).
- `a134d30` + `0f60c20` parallelize/cache pad-policy pre-pass; hash pad
  policy into cache unit keys (stale-cache correctness fix).
- `086f981` (HEAD) select-split baseline report (`runs/eval_v9_v11_select/`).
- Working tree CLEAN at handoff (`git status --short` empty).

## 3. RUNNING NOW — 3-stage scheduled-task pipeline

| Task | Script | Does |
|---|---|---|
| `v12_pipeline_stage1` | `model_training/pipeline_stage1.cmd` | check_corpus preflight → `build_caches.py --eval --rebuild-stale` → `build_caches.py --train`. Log `model_training/pipeline_stage1.log` |
| `v12_pipeline_stage2` | `pipeline_stage2.cmd` | waits for `stage 1 complete`, then `train_seq_cnn.py --out runs/voice_guard_v12` (GPU, 30 epochs). Log `pipeline_stage2.log` |
| `v12_pipeline_stage3` | `pipeline_stage3.cmd` | waits for `stage 2 complete`, then select-on-`select` → `validate_fp16.py` → `evaluate.py --split test` → `runs/eval_v12_test/report.md`. Log `pipeline_stage3.log` |

State at handoff: all three tasks `Running`; stages 2-3 in their wait loops.
Stage 1 (started 23:40, relaunched 01:08 after log-quoting fix) genuinely
working: ~10 python3.13 workers, CPU-seconds climbing. Tail: unit 26/98 done
(34.3 min elapsed), ~1.3-1.6 min/unit on small eval units; train units later
are slower. Quiet log (30+ s silence) is NORMAL — units log on completion.
`LastTaskResult 267009` = still running, NOT a failure.
(Part 2 continues in 2026-09-12-v12-pipeline-watch-handoff-part2.md)
