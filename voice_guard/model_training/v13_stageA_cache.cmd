@echo off
REM v13 pipeline, stage A: revalidate legacy stale units (format 1, global
REM hash) under the current code and re-stamp the unchanged ones, then build
REM the NEW playback eval + training units. Resumable: re-run after a crash.
REM All paths quoted (the tree path contains spaces).
set "PY=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\.venv313\Scripts\python.exe"
set "MT=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training"
set "CACHE=C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training\cache"
set "LOG=%MT%\runs\v13_stageA_cache.log"
cd /d "%MT%"
echo [%DATE% %TIME%] stage A start: eval caches incl playback >> "%LOG%"
"%PY%" build_caches.py --eval --revalidate-stale --workers 10 --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] eval caches exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] train caches incl playback rendition >> "%LOG%"
"%PY%" build_caches.py --train --revalidate-stale --workers 10 --cache-root "%CACHE%" >> "%LOG%" 2>&1
echo [%DATE% %TIME%] train caches exit %ERRORLEVEL% >> "%LOG%"
if errorlevel 1 exit /b 1
echo [%DATE% %TIME%] stage A complete >> "%LOG%"
