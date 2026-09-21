#!/bin/bash
#SBATCH -A vikramj
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -c 64
#SBATCH -t 08:00:00
#SBATCH -J openvla_full_eval
#SBATCH -o openvla_eval_%j.out
#SBATCH -e openvla_eval_%j.err

# Load Conda module & activate scratch environment
module load conda
conda activate /scratch/gautschi/vphamkha/conda_envs/openvla

# Dynamic library & cache exports
export HF_HOME="/scratch/gautschi/vphamkha/hf_cache"
export PIP_CACHE_DIR="/scratch/gautschi/vphamkha/.cache/pip"
export PYOPENGL_PLATFORM="egl"
export MUJOCO_GL="egl"
export PYOPENGL_ACCELERATE="FALSE"
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
