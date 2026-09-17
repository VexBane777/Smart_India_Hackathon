@echo off
REM Arm B: Train v13 with seqcnn_v1 architecture (alternative to seqtcn_v2)
REM Alternative architecture arm - simpler 2-layer Conv1d vs dilated TCN

cd /d "C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training"

set "PYTHON=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "OUTDIR=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\runs\voice_guard_v13_seqcnn"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "STDOUT_LOG=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\runs\v13_armB_seqcnn_train.log"
set "STDERR_LOG=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\runs\v13_armB_seqcnn_stderr.log"

REM Clear old logs
break > "%STDOUT_LOG%"
break > "%STDERR_LOG%"

echo [%DATE% %TIME%] Arm B (seqcnn_v1) starting on CPU >> "%STDOUT_LOG%"

REM Run on CPU with unbuffered output
"%PYTHON%" -u "C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\train_seq_cnn.py" ^
    --out "%OUTDIR%" ^
    --arch seqcnn_v1 ^
    --cache-root "%CACHE%" ^
    --seed 0 ^
    --workers 4 ^
    --device cpu ^
    >> "%STDOUT_LOG%" 2>> "%STDERR_LOG%"

set "EXIT_CODE=%ERRORLEVEL%"
echo [%DATE% %TIME%] Arm B exit %EXIT_CODE% >> "%STDOUT_LOG%"
echo [%DATE% %TIME%] Arm B exit %EXIT_CODE% >> "%STDERR_LOG%"

echo Arm B finished with exit code %EXIT_CODE%
exit /b %EXIT_CODE%