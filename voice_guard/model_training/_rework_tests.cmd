@echo off
rem Detached runner for the post-v13 rework test suites.
rem The trainer tests spawn real training subprocesses, which outlive a 30 s
rem foreground command in the Cline shell, so run this detached:
rem   cmd /c start "" /b C:\...\model_training\_rework_tests.cmd
rem or:  Start-Process cmd -ArgumentList '/c','_rework_tests.cmd' -WindowStyle Hidden
rem Results land in runs\_rework_tests.log (and the .err variant).
cd /d "%~dp0"
set PY=.venv313\Scripts\python.exe
echo === post-v13 rework suite %DATE% %TIME% === > runs\_rework_tests.log
%PY% -m pytest test_train_seq_cnn.py test_model_seq_cnn.py test_select_best_checkpoint_seqcnn.py test_eval_protocol.py test_eval_stats.py test_evaluate.py test_feature_cache.py test_features.py test_features_sequence.py test_dataset_sequence.py test_windowing.py test_attack_labels.py -q --no-header -p no:randomly >> runs\_rework_tests.log 2>&1
echo ---- exit %ERRORLEVEL% ---- >> runs\_rework_tests.log
echo DONE >> runs\_rework_tests.log