#!/bin/bash

# Setup Environments list
ENVS=("Pendulum-v1")

# OTHER ENVIRONMENTS: "Pendulum-v1" "MountainCarContinuous-v0" "BipedalWalker-v3" "Hopper-v4"

echo "Running CleanRL SAC training on multiple environments..."
for ENV in ${ENVS[@]}; do
    echo ""
    echo "=============================================="
    echo "Training on $ENV"
    echo "=============================================="
    python lesson12_cleanRL_sac_train_code.py --env-id $ENV --total-timesteps 50000
done

echo "All environments trained via CleanRL Solution script!"
