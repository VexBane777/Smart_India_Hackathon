@echo off
REM Arm B: Train v13 with seqcnn_v1 architecture on CPU
cd /d C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training
set PYTHON=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe
set OUTDIR=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\runs\voice_guard_v13_seqcnn
set CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache
echo %date% %time% Arm B starting >> runs\v13_armB_seqcnn_train.log
"%PYTHON%" -u train_seq_cnn.py --out "%OUTDIR%" --arch seqcnn_v1 --cache-root "%CACHE%" --seed 0 --workers 4 --device cpu >> runs\v13_armB_seqcnn_train.log 2>&1
echo %date% %time% Arm B exit %errorlevel% >> runs\v13_armB_seqcnn_train.log