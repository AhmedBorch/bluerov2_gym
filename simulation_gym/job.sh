#!/bin/bash
#SBATCH --partition=cpu          # Specify the cpu partition
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks (this may vary depending on your application)
#SBATCH --cpus-per-task=32       # Specify number of CPU cores per task (cassiopee1 has 32 cores per socket)
#SBATCH --time=10:00:00        # 10hours
#SBATCH --mem=64G                # Memory (adjust based on your job requirements)
#SBATCH --job-name=BlueRov_sim       # Job name


# -------------------------------------
# 1) Environment setup
# -------------------------------------

module purge

# (Skip CUDA/cuDNN since you're running CPU-only)

# -------------------------------------
# 2) Activate virtual environment
# -------------------------------------

source /baie/projects/mir/RED_BALL/bluerov2_gym/simulation_gym/.venv/bin/activate

# -------------------------------------
# 3) Run your Python script
# -------------------------------------

cd /baie/projects/mir/RED_BALL/bluerov2_gym/simulation_gym
python3 examples/train.py
