#!/bin/bash
#SBATCH --job-name=rankings
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/fvolvert/INFO8010-DeepLearning-Nvidia-Project/reid/logs/rankings_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

cd ~/INFO8010-DeepLearning-Nvidia-Project/reid
python generate_rankings.py --checkpoint runs/run_007/best_model.pth