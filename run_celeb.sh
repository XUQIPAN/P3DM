#!/bin/bash

# Set the working directory to your project's directory
cd /data/xinxi/Project/DPgan_model

# Set the environment variables as specified in launch.json
export EXP_NAME=CELEBA
export CUDA_VISIBLE_DEVICES=1

# Specify the exact Python interpreter used in VSCode
PYTHON_PATH="/common/users/xz657/envs/anaconda3/envs/py39/bin/python"

# Specify the path to your main Python script
SCRIPT="/data/xinxi/Project/DPgan_model/main.py"

# Your script's arguments as specified in launch.json
SCRIPT_ARGS="--config celeba.yml --doc $EXP_NAME --exp /data/xinxi/Project/DPgan_model/logs/exp_celeba"

# Execute the script with the same configuration as VSCode
$PYTHON_PATH $SCRIPT $SCRIPT_ARGS
