#!/bin/bash
DATASET_NAME="CUHK-PEDES" # CUHK-PEDES, ICFG-PEDES, RSTPReid

CUDA_VISIBLE_DEVICES=4 \
python finetune.py \
--name fmfa \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'a-sdm+id+mlm+efa' \
--num_epoch 60 \
--root_dir 'path to your data' \
--finetune 'path to your pretrain model checkpoint'

