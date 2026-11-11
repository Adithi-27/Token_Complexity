#!/bin/bash
# Usage: sbatch [--time=...] [--array=0-2] slurm/generate.sh [generate.py args]
# With --array, task i runs only the i-th tier, two GPUs each.
#SBATCH --job-name=tok_gen
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e
source "/users/student/pg/pg25/adithi/miniconda3/etc/profile.d/conda.sh"
conda activate baseenv
# home is near its quota: model weights and HF caches live on /janaki/backup
export HF_HOME=/janaki/backup/users/student/pg/pg25/adithi/hf_home
export HF_HUB_OFFLINE=1      # weights are already in the cache
# vLLM 0.6.6 breaks with the transformers 5 in baseenv; vendor_vllm/ holds 4.48
export PYTHONPATH="$PWD/vendor_vllm:$PWD/vendor:$PYTHONPATH"
export NCCL_P2P_DISABLE=1     # see disable_custom_all_reduce in config.yaml
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1    # vLLM's usage-stats file write hit the home quota

# Fail fast on a GPU that needs a reset (seen on anandi in Oct 2026)
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv
if nvidia-smi --query-gpu=temperature.gpu --format=csv | grep -q "requires reset"; then
    echo "a GPU needs a reset; aborting" >&2; exit 1
fi
python -c "import torch; n = torch.cuda.device_count(); assert n == 2, n; [torch.ones(1, device=i).sum().item() for i in range(n)]; print('GPUs OK')"
python -m unittest discover -s tests

TIERS=(easy moderate hard)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    python -m tok_exam.generate --tiers "${TIERS[$SLURM_ARRAY_TASK_ID]}" "$@"
else
    python -m tok_exam.generate "$@"
fi
