#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name fmfa \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'a-sdm+id+mlm+efa' \
--num_epoch 60
