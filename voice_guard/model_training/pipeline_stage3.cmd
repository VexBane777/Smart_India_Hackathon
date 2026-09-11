@echo off
REM v12 pipeline, stage 3: wait for stage 2, then checkpoint selection on the
REM `select` split only, fp16-storage validation, and the final evaluate.py
REM run (test split, v9+v11+v12, attack-val run, decision vs v11).
set "PY=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "MT=C:\Users\Tuviksh Murudkar\SIH\.worktrees\voiceguard-v12-hardening\voice_guard\model_training"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "LOG=%MT%\pipeline_stage3.log"
cd /d "%MT%"
echo [%DATE% %TIME%] stage 3 waiting for stage 2 >> "%LOG%"
:wait2
findstr /c:"stage 2 complete" "%MT%\pipeline_stage2.log" > nul 2>&1
if not errorlevel 1 goto trained
findstr /c:"exit 1" "%MT%\pipeline_stage2.log" > nul 2>&1
if not errorlevel 1 (
    echo [%DATE% %TIME%] stage 2 failed - aborting stage 3 >> "%LOG%"
    exit /b 1
)
timeout /t 60 /nobreak > nul
goto wait2
:trained
echo [%DATE% %TIME%] stage 2 complete detected - selecting checkpoint (select split only) >> "%LOG%"
"%PY%" select_best_checkpoint_seqcnn.py --run runs/voice_guard_v12 --out runs/voice_guard_v12_selected --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] selection exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1

echo [%DATE% %TIME%] fp16 storage validation >> "%LOG%"
"%PY%" validate_fp16.py --model runs/voice_guard_v11_seqcnn_selected/model.pt --model runs/voice_guard_v9_noisefix_final/model.pt --out runs/fp16_validation.json >> "%LOG%" 2>&1
echo [%DATE% %TIME%] fp16 exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 (
    echo [%DATE% %TIME%] fp16 gate FAILED - continuing to eval anyway; report will show it >> "%LOG%"
)

echo [%DATE% %TIME%] final evaluate.py run (test split, v9/v11/v12) >> "%LOG%"
"%PY%" evaluate.py --split test --out runs/eval_v12_test --cache-root "%CACHE%" ^
    --model v9=runs/voice_guard_v9_noisefix_final/model.pt ^
    --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt ^
    --model v12=runs/voice_guard_v12_selected/model.pt ^
    --attack-val-run runs/voice_guard_v12 --candidate v12 --reference v11 ^
    --fp16-report runs/fp16_validation.json >> "%LOG%" 2>&1
echo [%DATE% %TIME%] evaluate exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] stage 3 complete - see runs/eval_v12_test/report.md >> "%LOG%"
