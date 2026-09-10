@echo off
REM A-share intraday scout (v8) - one-click launcher
REM Usage: double-click this file, or run from cmd
setlocal
cd /d "%~dp0"

set PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [1/1] Running live_scout.py ...
"%PY%" -u live_scout.py --pool=400 --workers=12
echo.
echo Done. Output -> results_live\
pause
