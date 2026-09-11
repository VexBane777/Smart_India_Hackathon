@echo off
REM v12 pipeline, stage 1b (2026-09-12 ~03:25): resumes stage 1's training-
REM cache build with 18 workers instead of 10. Stage 1's build was stopped at
REM the start of the 32-unit / 243k-rendition training build because the box
REM sat at ~27% total CPU (20 cores, each worker saturating one) — about 7 h
REM at 10 workers. Finished shards are kept (shards are written tmp ->
REM os.replace; a unit without manifest.json resumes from its done shards).
REM On success this writes the "stage 1 complete" marker into
REM pipeline_stage1.log, which is the line stage 2 waits for; on failure it
REM writes "stage 1 FAILED", which stage 2 also watches for.
REM All paths quoted (the tree path contains spaces).
set "PY=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "MT=C:\Users\Tuviksh Murudkar\SIH\.worktrees\voiceguard-v12-hardening\voice_guard\model_training"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "LOG=%MT%\pipeline_stage1.log"
cd /d "%MT%"
echo [%DATE% %TIME%] stage 1b: build_caches --train --workers 18 (resume) >> "%LOG%"
"%PY%" build_caches.py --train --workers 18 --cache-root "%CACHE%" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [%DATE% %TIME%] train caches 1b exit nonzero >> "%LOG%"
    echo [%DATE% %TIME%] stage 1 FAILED >> "%LOG%"
    exit /b 1
)
echo [%DATE% %TIME%] train caches 1b exit 0 >> "%LOG%"
echo [%DATE% %TIME%] stage 1 complete >> "%LOG%"
