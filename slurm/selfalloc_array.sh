#!/bin/bash
#SBATCH --job-name=tok_p5
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e
source "/users/student/pg/pg25/adithi/miniconda3/etc/profile.d/conda.sh"
conda activate baseenv
export HF_HOME=/janaki/backup/users/student/pg/pg25/adithi/hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/vendor_vllm:$PWD/vendor:$PYTHONPATH"
export NCCL_P2P_DISABLE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if nvidia-smi --query-gpu=temperature.gpu --format=csv | grep -q "requires reset"; then
    echo "a GPU needs a reset; aborting" >&2; exit 1
fi
python -c "import torch; n = torch.cuda.device_count(); assert n == 2, n; [torch.ones(1, device=i).sum().item() for i in range(n)]; print('GPUs OK')"
python -m unittest discover -s tests

K=${SLURM_ARRAY_TASK_ID:-0}
N=${SLURM_ARRAY_TASK_COUNT:-1}
TOTAL=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['selfalloc']['n_papers'])")
A=$(( K * TOTAL / N )); B=$(( (K + 1) * TOTAL / N ))
echo "=== task $K/$N: papers $A:$B -> outputs/selfalloc/test_part$K.jsonl"
python -m tok_exam.selfalloc --papers "$A:$B" --out "test_part$K.jsonl"
