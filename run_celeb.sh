#!/bin/bash

# Set the working directory to your project's directory
# cd /data/xinxi/Project/DPgan_model

# Set the environment variables as specified in launch.json
# export EXP_NAME=CELEBA
export CUDA_VISIBLE_DEVICES=0

# Specify the exact Python interpreter used in VSCode
# PYTHON_PATH="/common/users/xz657/envs/anaconda3/envs/py39/bin/python"

# Specify the path to your main Python script
# SCRIPT="/data/local/qipan/DPgan_model/main.py"

# Your script's arguments as specified in launch.json
# SCRIPT_ARGS="--config celeba.yml --doc $EXP_NAME --exp /data/xinxi/Project/DPgan_model/logs/exp_celeba"

# Execute the script with the same configuration as VSCode
# $PYTHON_PATH $SCRIPT $SCRIPT_ARGS

# export EXP_NAME=RANDOM_0116_CELEBA_EXP_EXP10_K256_IMAGE_SIZE_28_BATCH_128
export EXP_NAME=LFW_IMAGE_SIZE_64_BATCH_128

OUTPUT_DIR="/data/local/ml01/qipan/exp_lfw/DPGEN_smile_samples_1"
# CLASSIFIER_STATE_DICT="/data/local/ml01/qipan/exp_celeba/logs/NOISY_CLASSIFIER_GENDER/checkpoint_108000.pth"
CLASSIFIER_STATE_DICT="/data/local/ml01/qipan/exp_celeba/logs/NOISY_CLASSIFIER_SMILE/checkpoint_81000.pth"
# EVAL_DIR="/data/local/ml01/qipan/exp_celeba/CG_inf_smile_samples_1"
python main.py --config celeba.yml --doc $EXP_NAME --exp /data/local/ml01/qipan/exp_lfw  --fast_fid --classifier_state_dict $CLASSIFIER_STATE_DICT --output_dir $OUTPUT_DIR --seed 1234
# python main.py --config celeba.yml --doc $EXP_NAME --exp /data/local/ml01/qipan/exp_celeba --privacy_eval --eval_dir $EVAL_DIR --classifier_state_dict $CLASSIFIER_STATE_DICT