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
python main.py --config celeba.yml --doc $LOG_PATH --exp $IMAGE_OUTPUT_PATH  --train
```

### Training Classifier

To train the classifier, run the following command:

```shell
python main.py --config celeba.yml --doc $LOG_PATH --exp $CLASSIFIER_STATE_DICT_PATH  --train_cls
```

### Sampling

To sample from the model, run the following command:

```shell
python main.py --config celeba.yml --doc $LOG_PATH --exp $IMAGE_OUTPUT_PATH  --fast_fid --classifier_state_dict $CLASSIFIER_STATE_DICT_PATH --output_dir $OUTPUT_DIR --seed $SEED
```

### Privacy Score Evaluation

To evaluate the model with the privacy score, run the following command:

```shell
python main.py --config celeba.yml --doc $LOG_PATH --exp $IMAGE_OUTPUT_PATH --privacy_eval --eval_dir $EVAL_DIR --classifier_state_dict $CLASSIFIER_STATE_DICT
```

### PAC Privacy Evaluation
Run the following notebook:
```shell
pac_privacy.ipynb
```