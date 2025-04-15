#!/bin/bash
#SBATCH --job-name=b2u_train                # 任务名称
#SBATCH --output=logs/%x_%j.out             # 输出日志文件，%x为任务名，%j为任务号
#SBATCH --error=logs/%x_%j.err              # 错误日志文件
#SBATCH --partition=spgpu                   # 分区名称
#SBATCH --gres=gpu:1                        # 请求 GPU 核心数
#SBATCH --cpus-per-task=16                  # 分配 CPU 核心数
#SBATCH --mem=32G                           # 分配内存
#SBATCH --time=0-08:00:00                   # 最长运行时间，格式为 天-小时:分钟:秒
#SBATCH --mail-type=END,FAIL                # 任务完成或失败时通知
#SBATCH --mail-user=ruijiech@umich.edu   

# ====== 环境 ======
# source /home/ethan/.bashrc
# cd /home/ethan/SelfSup_DeBayer/
# conda activate 556

# ====== 日志目录创建 ======
mkdir -p logs

# ====== 训练命令======
python train_new.py \
    --data_dir ./data/train/SIDD_Medium_Raw_noisy_sub512_rggb/ \
    --val_dirs ./data/validation/SIDD_Validation_Blocks/ \
    --save_model_path ../experiments/results \
    --log_name b2u_new_mask \
    --gpu_devices 0 \
    --in_channel 4 \
    --out_channel 3 \
    --n_feature 48 \
    --batchsize 4 \
    --n_epoch 100 \
    --use_mask \
    --remosaic_mode random \
    --warmup_epoch 20 \
    --resume /home/ethan/Code/experiments/results/b2u_new_mask/2025-04-15-03-56/training_states/20.state \
    --checkpoint /home/ethan/Code/experiments/results/b2u_new_mask/2025-04-15-03-56/models/epoch_model_020.pth