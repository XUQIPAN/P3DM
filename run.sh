# +
export EXP_NAME=RANDOM_0116_CELEBA_EXP_EXP10_K256_IMAGE_SIZE_28_BATCH_128_IMPROVED_2
# export CUDA_VISIBLE_DEVICES=4, 3, 5, 6, 7;
python main.py --config celeba.yml --doc $EXP_NAME --exp /data/local/qipan/exp_celeba


