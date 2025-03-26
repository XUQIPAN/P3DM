# PAC Privacy Preserving Diffusion Models

[Paper Link](https://arxiv.org/pdf/2312.01201.pdf)

### Package Installation

On the basis of `Python >= 3.8` environment, install the necessary dependencies by running the following command:

```shell
pip install -r requirements.txt
```

### Training

To train the model, run the following command:

```shell
python main.py --config celeba.yml --doc $LOG_PATH --exp $YOUR_PATH  --train
```
### Sampling

To sample from the model, run the following command:

```shell
python main.py --config celeba.yml --doc $EXP_NAME --exp $YOUR_PATH  --fast_fid --classifier_state_dict $CLASSIFIER_STATE_DICT --output_dir $OUTPUT_DIR --seed $SEED
```

### Evaluation

To evaluate the model, run the following command:

```shell
python main.py --config celeba.yml --doc $EXP_NAME --exp /data/local/ml01/qipan/exp_celeba --privacy_eval --eval_dir $EVAL_DIR --classifier_state_dict $CLASSIFIER_STATE_DICT
```

