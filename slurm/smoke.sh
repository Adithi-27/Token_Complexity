#!/bin/bash
#SBATCH --job-name=tok_smoke
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

python -m tok_exam.generate --out-dir outputs/smoke --limit-questions 2 --limit-samples 2
python -m tok_exam.sanity --in-dir outputs/smoke || echo "(sanity flags are expected on 2 questions)"
python -m tok_exam.curves --in-dir outputs/smoke --out-dir outputs/smoke_curves
