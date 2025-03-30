#!/bin/bash

python test_b2u.py --noisetype bayer_grbg --checkpoint ./pretrained_models/bayer_pick_6x3080_v1.pth --test_dirs /home/featurize/data/validation --save_test_path ./test --log_name bayer_unet_pick_6x3080
