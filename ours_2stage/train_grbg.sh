#!/bin/bash

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_in_conda_env ".*556.*" ; then
    echo ENV HAS ALREADY BEEN CREATED!
    eval "$(conda shell.bash hook)"
    conda activate 556
    python train_b2u.py \
        --noisetype bayer \
        --data_dir /home/ethan/Code/SelfSup_DeBayer/data/train/Imagenet_val/  \
        --val_dirs /home/ethan/Code/SelfSup_DeBayer//data/validation/ \
        --save_model_path /home/ethan/Code/SelfSup_DeBayer/experiments/results/ \
        --log_name bayer_unet_imagenetVal_2Stage \
        --checkpoint /home/ethan/Code/SelfSup_DeBayer/saved_models/bayer_pick_6x3080_v1.pth \
        --Lambda1 1.0 \
        --Lambda2 2.0 \
        --increase_ratio 20.0 \
        --gpu_devices 0 \
        --batchsize 4 \
        --num_epochs 100 \
        --warmup_epoch 0
else
    conda create -n 556 -y python=3.8.5
    eval "$(conda shell.bash hook)"
    conda activate 556
    pip install -r requirements.txt
fi
