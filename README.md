# Digital Medicine 2020 Fall: RSNA ICH Detection

## Task

Design and present an analysis flow for RSNA ICH classifiers.

## Environment

- Ubuntu 20.04.1 LTS
- Python 3.7.5+
- GeForce GTX 1080 Ti
- CUDA version 11.0

### Packages

- tqdm
- numpy
- pydicom
- seaborn
- matplotlib
- scikit-learn
- pytorch 1.7.0+
- torchvision 0.8.0+

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

## Our method

> 可以貼 spreadsheet 網址
