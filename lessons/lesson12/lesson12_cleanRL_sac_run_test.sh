#!/bin/bash

echo "=============================================="
echo "Running CleanRL SAC Test"
echo "=============================================="

# Set the environment ID
ENV_ID="Pendulum-v1"

# IMPORTANT: Replace the path below with the actual path to your trained model
# Example: runs/Pendulum-v1__sac_cleanRL_solution__1__1684594234/sac_cleanRL_solution.cleanrl_model
MODEL_PATH=""

echo "Testing on $ENV_ID with model $MODEL_PATH"
echo ""

python lesson12_cleanRL_sac_test_sac.py --env-id $ENV_ID --model-path "$MODEL_PATH"
