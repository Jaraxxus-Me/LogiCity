#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a5000:5
#SBATCH --mem=150G
#SBATCH --nodelist=node200
#SBATCH --output=./lip_gen_sd_%j.out

export OMP_NUM_THREADS=5
module load cuda/11.8
python -u lip_gen_sd.py