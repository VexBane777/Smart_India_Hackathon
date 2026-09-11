# v12 pipeline watch handoff, part 2 — monitor / finish / pitfalls

## 4. Monitor (copy-paste PowerShell, always quote paths)

$m='C:\Users\Tuviksh Murudkar\SIH\.worktrees\voiceguard-v12-hardening\voice_guard\model_training'
Get-Content "$m\pipeline_stage1.log" -Tail 4
Get-Content "$m\pipeline_stage2.log" -Tail 4
Get-Content "$m\pipeline_stage3.log" -Tail 4
Get-ScheduledTask -TaskName 'v12_pipeline_stage*' | Select-Object TaskName, State | Format-Table -AutoSize
Get-Process | Where-Object { $_.ProcessName -like 'python*' } | Select-Object Id, ProcessName, @{n='CPU_s';e={$_.CPU}}, StartTime | Format-Table -AutoSize

Healthy = stage-1 log tail timestamp fresh (< ~5 min) AND worker CPU-seconds
climbing between checks. Transitions: `stage 1 complete` → `stage 1 complete
detected - starting training` → `stage 2 complete` → stage-3 lines.

## 5. When stages complete

1. Stage 1 done: verify both build_caches exit lines say `exit 0`; stage 2
   picks up alone within ~60 s. Non-zero → read tail traceback, fix forward.
2. Stage 2 done: check `runs/voice_guard_v12/` checkpoints + training log;
   stage 3 picks up automatically.
3. Stage 3 done (`stage 3 complete - see runs/eval_v12_test/report.md`):
   read `runs/eval_v12_test/report.md` + `report.json`. Decision rule:
   ship v12 to `assets/models/voice_detector.onnx` ONLY if it passes confound
   gates (|rho| <= 0.10 per confound feature pooled over phone channels,
   worse/better half error-rate ratio <= 1.25) AND beats v11 on `test` under
   the channel protocol. Attack-type head below 0.70 balanced accuracy on
   leave-attack-out → recommend hiding the UI sub-label; coordinate the
   `call_screen.dart` change with the UI session, don't touch UI code alone.
   Then: state.md section, commit, merge to `vaani`, push, message UI session.

## 6. Hard-won lessons (read before touching anything)

- QUOTING: every path in the `.cmd` files MUST stay double-quoted (tree is
  under `Tuviksh Murudkar`). `cd` tolerates spaces; `>>` redirections don't.
- Run from the WORKTREE code: existing caches carry the worktree code hash;
  another checkout's hash = StaleCacheError → full rebuild.
- Never two `build_caches` at once on one cache root (shard races).
- Caches are gitignored in the MAIN checkout; never `git clean` them, never
  commit `*.pt`/`*.onnx`/cache dirs. `runs/eval_*` reports ARE committed.
- `LastTaskResult 267009` = still running, not an error.
- Nightly-reboot risk: machine once had a kernel nonpaged-pool leak (5.9 GB).
  Tasks are one-shot (no repetition trigger): after a reboot, re-run the
  killed stage's `.cmd` manually — it resumes (caches skip done units).
- NEVER eval clean-only (user policy, unless `--application bank`). NEVER
  select checkpoints on `test` (select script reads only `select`).
- NEVER touch the corpus junctions: worktree `data/<dir>` are NTFS junctions
  to the main checkout's gitignored corpus; deleting through them destroys it.

## 7. Resume checklist

- [ ] Read `voice_guard/state.md` v12 section + `docs/EVAL-PROTOCOL.md` first.
- [ ] `git status --short` (expect clean), `git log --oneline -6` (HEAD 086f981).
- [ ] Tail all three stage logs; confirm `unit N/98` progress or transitions.
- [ ] All quiet + no python workers + no completion markers → tail traceback,
      fix, re-run that stage's `.cmd` (it resumes).
- [ ] `runs/eval_v12_test/report.md` exists → apply section 5 rule, update
      state.md, commit, merge to `vaani`, push.
