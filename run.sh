# +
export EXP_NAME=training_logs
# export CUDA_VISIBLE_DEVICES=4, 3, 5, 6, 7;
python main.py --config celeba.yml --doc $EXP_NAME --exp /workspace --resume_training


