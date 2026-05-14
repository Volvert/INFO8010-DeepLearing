#!/bin/bash
#SBATCH --job-name=vehicle_reid
#SBATCH --output=logs/train_%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-6:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export WANDB_MODE=online   # change en offline si le compute node bloque internet
cd ~/INFO8010-DeepLearning-Nvidia-Project/reid
python main.py --config config/tiny_vit.yaml