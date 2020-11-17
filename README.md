# Digital Medicine 2020 Fall: RSNA ICH Detection

## Task

Design and present an analysis flow for RSNA ICH classifiers.

## Environment

- Ubuntu 20.04.1 LTS
- Python 3.7.5+
- GeForce GTX 1080 Ti
- CUDA version 11.0+

### Packages

- tqdm
- numpy
- pandas
- pydicom
- seaborn
- openpyxl
- matplotlib
- scikit-learn 0.23.2+
- pytorch 1.7.0+
- torchvision 0.8.0+

### Prerequsites

```bash
pip3 install -r requirements.txt
```

## Usage

### Training example

```bash
python3 train.py \
--exp_name='' \
--model_name googlenet \
--generate_exp_name --name_split_file --pretrained --radam \
--random_apply_aug --random_horizontal_flip --random_rotation --random_erasing --random_order \
--cutmix
```

### Testing example

```bash
python3 test.py \
--model_name googlenet \
--exp_name='_Pretrain_googlenet_E100_lr0.0005_b1_0.9_b2_0.999_bs_16_splt_0.7_prekeral_bsb_radam_cutmixbeta_1_cutmixprob_0.5_rand_app_rand_flip_rand_rota_rand_ord_rand_eras' \
--cpt_name='googlenet_E_79_iter_21040.cpt'
```

### Supported models

- resnet18, resnet34, resnet50, resnet101, resnet152
- resnext50_32x4d, resnext101_32x8d
- wide_resnet50_2, wide_resnet101_2
- densenet121, densenet169, densenet161, densenet201
- googlenet

## Our method

### Data preprocessing
- Brain/Subdural/Bone Window (https://www.kaggle.com/reppic/gradient-sigmoid-windowing)

### Data Augmentation
- Random Apply
- Random Order
- Random Horizontal Flip
- Random Rotation
- Random Erasing / Cutout
    - Zhong, Zhun, et al. "Random Erasing Data Augmentation." *AAAI*. 2020.
    - DeVries, Terrance et al. "Improved regularization of convolutional neural networks with cutout." *arXiv preprint* 2017.
- Mixup
    - Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk Minimization." *ICLR* 2018
- Cutmix
    - Yun, Sangdoo, et al. "Cutmix: Regularization strategy to train strong classifiers with localizable features." *ICCV*. 2019.

### Model

Fine-tune models in `torchvision.models` with ImageNet pretrained weight
- Simply replace last fully-connected layer to fit our data
- RAdam optimizer (https://github.com/LiyuanLucasLiu/RAdam)

## Experiment results

https://docs.google.com/spreadsheets/d/1HslC0VL4noqPN9LpZZlKMbwCwXAd38pN6IiANSGjKDk/edit?usp=sharing
