#!/bin/bash
DATASET_NAME="CUHK-PEDES" # CUHK-PEDES, ICFG-PEDES, RSTPReid

CUDA_VISIBLE_DEVICES=4 \
python train.py \
--name fmfa \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'a-sdm+mlm+id+efa' \
--dataset_name $DATASET_NAME \
--root_dir 'path to your data' \
--num_epoch 60
