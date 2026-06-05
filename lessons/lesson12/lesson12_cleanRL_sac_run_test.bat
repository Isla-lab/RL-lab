@echo off
echo ==============================================
echo Running CleanRL SAC Test
echo ==============================================

REM Set the environment ID
set ENV_ID=Pendulum-v1

REM IMPORTANT: Replace the path below with the actual path to your trained model
REM Example: runs\Pendulum-v1__sac_cleanRL_solution__1__1684594234\sac_cleanRL_solution.cleanrl_model
set MODEL_PATH="C:\Users\rikyf\OneDrive\Desktop\RL Labs\L12_CleanRL_SAC\models\sac_Pendulum-v1_actor.pth"

echo Testing on %ENV_ID% with model %MODEL_PATH%
echo.

python lesson12_cleanRL_sac_test_sac.py --env-id %ENV_ID% --model-path %MODEL_PATH%

pause
