#!/bin/bash
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --output=gpu_info.out

rocminfo
rocm-smi

