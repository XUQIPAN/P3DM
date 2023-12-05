# +
export EXP_NAME=EXP1_GENDER
# export CUDA_VISIBLE_DEVICES=4, 3, 5, 6, 7;
python main.py --config celeba.yml --doc $EXP_NAME --exp /data/local/qipan/exp_celeba --privacy_eval


