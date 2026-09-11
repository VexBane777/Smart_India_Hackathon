@echo off
REM v12 pipeline, stage 1: cheap corpus preflight, then finish the eval
REM caches (42/46 units already built by the earlier session; this resumes
REM the rest) and build the training caches. Logs to pipeline_stage1.log.
REM Run from the worktree (its code hash is what the existing caches were
REM built with); caches live in the MAIN checkout so they outlive worktrees.
REM NOTE: every path below MUST stay quoted — the tree lives under a path
REM with spaces ("Tuviksh Murudkar"), and cmd redirections do not tolerate
REM unquoted spaced paths (cd does, which is why this once half-worked).
set "PY=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "MT=C:\Users\Tuviksh Murudkar\SIH\.worktrees\voiceguard-v12-hardening\voice_guard\model_training"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "LOG=%MT%\pipeline_stage1.log"
cd /d "%MT%"
echo [%DATE% %TIME%] stage 1 start >> "%LOG%"

echo [%DATE% %TIME%] check_corpus preflight >> "%LOG%"
"%PY%" check_corpus.py --pair data/real data/fake --pair data/real2021 data/fake2021 --pair data/real_itw_train data/fake_itw_train >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [%DATE% %TIME%] check_corpus FAILED - stopping before long jobs >> "%LOG%"
    exit /b 1
)

echo [%DATE% %TIME%] build_caches --eval >> "%LOG%"
"%PY%" build_caches.py --eval --rebuild-stale --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] eval caches exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1

echo [%DATE% %TIME%] build_caches --train >> "%LOG%"
"%PY%" build_caches.py --train --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] train caches exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1

echo [%DATE% %TIME%] stage 1 complete >> "%LOG%"

