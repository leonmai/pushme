@echo off
REM Position tracker (v8) - one-click launcher
setlocal
cd /d "%~dp0"

set PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe
if not exist "%PY%" set PY=python

set TODAY=%date:~0,4%-%date:~5,2%-%date:~8,2%
echo Today = %TODAY%

echo [1/3] rescan full-day signals ...
"%PY%" -u live_scout.py --pool=400 --workers=12

echo [2/3] init new positions ...
"%PY%" -u track_positions.py --init=%TODAY% --max-positions=5

echo [3/3] update and check exits ...
"%PY%" -u track_positions.py

echo.
echo Done. State -> results_live\positions.json
pause
