#!/bin/bash

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_in_conda_env ".*b2ub.*" ; then
    echo ENV HAS ALREADY BEEN CREATED!
    eval "$(conda shell.bash hook)"
    conda activate b2ub
    python train_b2u.py --noisetype bayer_grbg --data_dir /home/featurize/data/pick/ --val_dirs /home/featurize/data/validation --save_model_path ../experiments/results --log_name bayer_unet_pick_6x3080 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0 --gpu_devices '0,1,2,3,4,5' --parallel --batchsize 24
else
    conda create -n b2ub -y python=3.8.5
    eval "$(conda shell.bash hook)"
    conda activate b2ub
    pip install -r requirements.txt
fi
