#!/usr/bin/env bash
#SBATCH --job-name=mlip_test
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --account=a_smp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd /scratch/user/s4802880/mlip_phonons

module purge
module load miniforge/24.11.3-0 # fix this version as needed
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"

export XDG_CACHE_HOME="/scratch/user/$USER/.cache"
export HF_HOME="/scratch/user/$USER/huggingface"
export TORCH_HOME="/scratch/user/$USER/torch"

conda activate mace_env
srun mlip-phonons small-omat-0
conda deactivate
