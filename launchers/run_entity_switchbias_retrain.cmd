@echo off
setlocal

for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "PY=C:\Users\jeanj\Documents\School - Research\deepLearning\Scripts\python.exe"
set "DATA=C:\Users\jeanj\.cache\kagglehub\datasets\thephilliplin\pokemon-showdown-battles-gen9-randbats\versions\1"
set "RUN=entity_action_bc_v1_20260403_switchbias_run3"
set "PYTHONUNBUFFERED=1"

set "LOG_DIR=%REPO%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\%RUN%.log"

>> "%LOG_FILE%" echo [launcher] started %DATE% %TIME%
>> "%LOG_FILE%" echo [launcher] repo=%REPO%
>> "%LOG_FILE%" echo [launcher] python=%PY%
>> "%LOG_FILE%" echo [launcher] data=%DATA%

"%PY%" -u "%REPO%\train_entity_action.py" "%DATA%" ^
  --output-dir "%REPO%\artifacts" ^
  --model-name "%RUN%" ^
  --max-battles 5000 ^
  --epochs 20 ^
  --patience 2 ^
  --learning-rate 0.0005 ^
  --switch-logit-bias 0.15 ^
  --policy-return-weighting exp ^
  --policy-return-weight-scale 1.0 ^
  --policy-return-weight-min 0.25 ^
  --policy-return-weight-max 4.0 ^
  --predict-turn-outcome ^
  --predict-value >> "%LOG_FILE%" 2>&1

>> "%LOG_FILE%" echo [launcher] exit_code=%ERRORLEVEL%

endlocal
