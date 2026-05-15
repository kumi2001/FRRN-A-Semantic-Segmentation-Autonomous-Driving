#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

cd /notebooks/docker

PYTHONPATH=. python3 main.py \
    --mode train \
    --dataset /notebooks/cityscape \
    --model FRRN-A \
    --batch_size 1 \
    --num_epochs 100 \
    --checkpoint_step 1 \
    --crop_height 320 \
    --crop_width 640 \
    --continue_training False