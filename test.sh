#!/bin/bash
python3 test.py --model_name resnext50_32x4d --batch_size 2 --output_dir output_temp \
--exp_name='_Pretrain_resnext50_32x4d_E100_lr0.0005_b1_0.9_b2_0.999_bs_12_splt_0.7_prekeral_bsb_radam_rand_app_rand_flip_rand_rota_rand_ord_rand_eras' \
--cpt_name='resnext50_32x4d_E_48_iter_17150.cpt'
