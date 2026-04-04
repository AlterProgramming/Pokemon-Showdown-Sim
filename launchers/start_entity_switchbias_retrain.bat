@echo off
setlocal

set "REPO=C:\Users\jeanj\Documents\School - Research\CSCI 8590 Introduction to Deep Learning\Pokemon Showdown Agent"
set "PY=C:\Users\jeanj\Documents\School - Research\deepLearning\Scripts\python.exe"
set "DATA=C:\Users\jeanj\.cache\kagglehub\datasets\thephilliplin\pokemon-showdown-battles-gen9-randbats\versions\1"
set "RUN=entity_action_bc_v1_20260403_switchbias_run1"
set "LOGDIR=%REPO%\logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

"%PY%" "%REPO%\train_entity_action.py" "%DATA%" --output-dir "%REPO%\artifacts" --model-name "%RUN%" --max-battles 5000 --epochs 20 --patience 2 --learning-rate 0.0005 --switch-logit-bias 0.15 --policy-return-weighting exp --policy-return-weight-scale 1.0 --policy-return-weight-min 0.25 --policy-return-weight-max 4.0 --predict-turn-outcome --predict-value > "%LOGDIR%\%RUN%.log" 2>&1

