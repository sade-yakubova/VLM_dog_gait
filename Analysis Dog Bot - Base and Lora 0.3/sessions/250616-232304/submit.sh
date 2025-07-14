#!/bin/bash

#SBATCH --job-name=MOGWAI
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=02:59:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --mem=40gb

source /etc/profile.d/modules.sh
module load nvidia/cuda-12.4
source /home/s2425823/test8/bin/activate

export CUDA_HOME=/deepstore/software/nvidia/cuda-12.4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 analysis.py