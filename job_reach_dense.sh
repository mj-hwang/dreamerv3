#! /bin/bash
#SBATCH --time=70:00:00
#SBATCH --exclude=node01,node03
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:2

export MUJOCO_GL='egl'
conda activate mbgcrl
module swap cuda/12.1
python python example_reach.py
