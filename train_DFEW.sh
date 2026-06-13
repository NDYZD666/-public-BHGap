#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python main.py \
--dataset 'DFEW' \
--workers 8 \
--epochs 25 \
--batch-size 8 \
--lr 3e-5 \
--weight-decay 1e-2 \
--print-freq 10 \
--img-size 224 \
--exper-name FINAL_224 \
