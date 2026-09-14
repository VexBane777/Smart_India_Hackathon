@echo off
REM v13 pipeline, stage B: wait for stage A, then train v13 (v12 recipe +
REM hash-selected ~50%% playback rendition), select on select
REM (phone+acoustic pooled), fp16 validate, evaluate all four deploy gates,
REM and run the supplementary playback-loop gate on the shipped ONNX.
set "PY=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "MT=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "LOG=%MT%\runs\v13_stageB_train_eval.log"
cd /d "%MT%"
echo [%DATE% %TIME%] stage B waiting for stage A >> "%LOG%"
:waitA
findstr /c:"stage A complete" "%MT%\runs\v13_stageA_cache.log" > nul 2>&1
if not errorlevel 1 goto cached
timeout /t 60 /nobreak > nul
goto waitA
:cached
echo [%DATE% %TIME%] stage A complete detected - training v13 >> "%LOG%"
"%PY%" train_seq_cnn.py --out runs/voice_guard_v13 --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] training exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] stage B training complete >> "%LOG%"
echo [%DATE% %TIME%] selecting checkpoint (select split only, phone+acoustic pooled) >> "%LOG%"
"%PY%" select_best_checkpoint_seqcnn.py --run runs/voice_guard_v13 --out runs/voice_guard_v13_selected --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] selection exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] fp16 storage validation >> "%LOG%"
"%PY%" validate_fp16.py --model runs/voice_guard_v13_selected/model.pt --out runs/fp16_validation_v13.json >> "%LOG%" 2>&1
echo [%DATE% %TIME%] fp16 exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] final evaluate.py (test split, v11 reference, v13 candidate) >> "%LOG%"
"%PY%" evaluate.py --split test --out runs/eval_v13_test --cache-root "%CACHE%" --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt --model v13=runs/voice_guard_v13_selected/model.pt --attack-val-run runs/voice_guard_v13 --candidate v13 --reference v11 --fp16-report runs/fp16_validation_v13.json >> "%LOG%" 2>&1
echo [%DATE% %TIME%] evaluate exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] playback-loop gate on shipped ONNX >> "%LOG%"
"%PY%" eval_playback_loop.py --onnx runs/voice_guard_v13_selected/model.onnx --assets test_assets --gate-clean-min 0.60 --gate-loop-min 0.50 >> "%LOG%" 2>&1
echo [%DATE% %TIME%] playback-loop exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] stage B complete - see runs/eval_v13_test/report.md >> "%LOG%"
