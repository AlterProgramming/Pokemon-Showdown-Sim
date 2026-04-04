@echo off
setlocal

for %%I in ("%~dp0..") do set "REPO=%%~fI"
if defined PS_AGENT_PYTHON (
  set "PY=%PS_AGENT_PYTHON%"
) else if exist "%REPO%\.venv\Scripts\python.exe" (
  set "PY=%REPO%\.venv\Scripts\python.exe"
) else if exist "%REPO%\venv\Scripts\python.exe" (
  set "PY=%REPO%\venv\Scripts\python.exe"
) else (
  set "PY=python"
)
if defined PS_AGENT_DATA (
  set "DATA=%PS_AGENT_DATA%"
)
set "RUN=entity_action_bc_v1_20260403_switchbias_run1"
set "LOGDIR=%REPO%\logs"
set "DATA_ARGS="
if defined DATA set "DATA_ARGS=\"%DATA%\""

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

"%PY%" "%REPO%\train_entity_action.py" %DATA_ARGS% --output-dir "%REPO%\artifacts" --model-name "%RUN%" --max-battles 5000 --epochs 20 --patience 2 --learning-rate 0.0005 --switch-logit-bias 0.15 --policy-return-weighting exp --policy-return-weight-scale 1.0 --policy-return-weight-min 0.25 --policy-return-weight-max 4.0 --predict-turn-outcome --predict-value > "%LOGDIR%\%RUN%.log" 2>&1

