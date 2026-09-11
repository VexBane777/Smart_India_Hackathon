@echo off
REM Smoke: evaluate.py on the SELECT split for v9+v11 only, using the
REM already-built select caches. Validates the whole harness end-to-end on
REM real data (cache load, scoring, thresholds, confound table, CIs, report)
REM before stage 3 depends on it, and records the operating thresholds the
REM final test-split run will use. select-only, so no leakage.
set "PY=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "MT=C:\Users\Tuviksh Murudkar\SIH\.worktrees\voiceguard-v12-hardening\voice_guard\model_training"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "LOG=%MT%\eval_select_smoke.log"
cd /d "%MT%"
echo [%DATE% %TIME%] select-split eval smoke start >> "%LOG%"
"%PY%" evaluate.py --split select --out runs/eval_v9_v11_select --cache-root "%CACHE%" --model v9=runs/voice_guard_v9_noisefix_final/model.pt --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt >> "%LOG%" 2>&1
echo [%DATE% %TIME%] evaluate exit %ERRORLEVEL% >> "%LOG%"
