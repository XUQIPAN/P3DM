#!/bin/bash

# Set the working directory to your project's directory
cd /data/local/xinxi/Project/DPgan_model

# Set the environment variables as specified in launch.json
export EXP_NAME=CUB
export CUDA_VISIBLE_DEVICES=0

# Specify the exact Python interpreter used in VSCode
PYTHON_PATH="/common/users/xz657/envs/anaconda3/envs/py39/bin/python"

# Specify the path to your main Python script
SCRIPT="/data/local/xinxi/Project/DPgan_model/main.py"

# Your script's arguments as specified in launch.json
SCRIPT_ARGS="--config cub.yml --doc $EXP_NAME --exp /data/local/xinxi/Project/DPgan_model/logs/exp_cub --fast_fid" 

# Execute the script with the same configuration as VSCode
$PYTHON_PATH $SCRIPT $SCRIPT_ARGS
