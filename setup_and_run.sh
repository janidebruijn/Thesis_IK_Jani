#!/bin/bash
#SBATCH --job-name=run_models
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --cpus-per-task=4

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/thesis_env/bin/activate
#python scripts/phi.py
#python scripts/llama.py
#python scripts/qwen.py
#python scripts/binary_phi.py
#python scripts/binary_llama.py
python scripts/binary_qwen.py
