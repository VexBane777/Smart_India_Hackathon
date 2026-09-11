@echo off
REM v12 pipeline, stage 2: wait for stage 1 ("stage 1 complete" marker), then
REM train VoiceGuardSeqTCN (v12) from the memmap cache. GPU, 30 epochs.
REM All paths quoted (the tree path contains spaces).
set "PY=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "MT=C:\Users\Tuviksh Murudkar\SIH\.worktrees\voiceguard-v12-hardening\voice_guard\model_training"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "LOG=%MT%\pipeline_stage2.log"
cd /d "%MT%"
echo [%DATE% %TIME%] stage 2 waiting for stage 1 >> "%LOG%"
:wait1
findstr /c:"stage 1 complete" "%MT%\pipeline_stage1.log" > nul 2>&1
if not errorlevel 1 goto built
findstr /c:"stage 1 FAILED" "%MT%\pipeline_stage1.log" > nul 2>&1
if not errorlevel 1 (
    echo [%DATE% %TIME%] stage 1 failed - aborting stage 2 >> "%LOG%"
    exit /b 1
)
timeout /t 60 /nobreak > nul
goto wait1
:built
echo [%DATE% %TIME%] stage 1 complete detected - starting training >> "%LOG%"
"%PY%" train_seq_cnn.py --out runs/voice_guard_v12 --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] training exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] stage 2 complete >> "%LOG%"
