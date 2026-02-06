#!/usr/bin/env bash
#SBATCH --job-name=mlip_phonons_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=a_smp
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail
cd /scratch/user/s4802880/mlip_phonons

models_index="models_index.txt"

nvidia-smi

module purge
module load miniforge/25.3.0-3 # fix this version as needed
source "$ROOTMINIFORGE/etc/profile.d/conda.sh"

export XDG_CACHE_HOME="/scratch/user/$USER/.cache"
export HF_HOME="/scratch/user/$USER/huggingface"
export TORCH_HOME="/scratch/user/$USER/torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME"

read -r model env < <(
python - <<PY
import os
i = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open("$models_index", "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]
model, env = lines[i].split("\t")
print(model, env)
PY
)

conda activate "$env"
srun mlip-phonons "$model"
conda deactivate
