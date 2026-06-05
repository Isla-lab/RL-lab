@echo off
setlocal enabledelayedexpansion

REM Setup Environments list
set ENVS=Pendulum-v1 


REM OTHER ENVIRONMENTS: Pendulum-v1 MountainCarContinuous-v0 BipedalWalker-v3 Hopper-v4

echo Running CleanRL SAC training on multiple environments...
for %%E in (%ENVS%) do (
    echo.
    echo ==============================================
    echo Training on %%E
    echo ==============================================
    python lesson12_cleanRL_sac_train_code.py --env-id %%E --total-timesteps 50000
)

echo All environments trained via CleanRL Solution script!
pause
