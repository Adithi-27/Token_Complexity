#!/bin/bash
# The whole GPU critical path in one slot (QOS allows few jobs):
#   curves -> difficulty (CPU, needed.jsonl labels) -> ICL with curve labels -> probe
#   -> difficulty again (adds icl / self / probe predictors).
# Each stage is skipped if its outputs already exist, so a resubmit continues.
# Usage: sbatch slurm/pipeline_tc.sh
#SBATCH --job-name=tok_pipeline
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source "/users/student/pg/pg25/adithi/miniconda3/etc/profile.d/conda.sh"
conda activate baseenv
export HF_HOME=/janaki/backup/users/student/pg/pg25/adithi/hf_home   # see slurm/generate.sh
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

stage() { echo "=== $(date '+%F %T') $*"; python -m "$@"; }

[ -s outputs/curves/hard.jsonl ] || stage tok_exam.curves
stage tok_exam.sanity || true
stage tok_exam.difficulty                       # writes needed.jsonl: ICL labels from curves
ls outputs/icl/question_*_k16.jsonl >/dev/null 2>&1 || stage tok_exam.icl
[ -s outputs/probe/probe.jsonl ] || stage tok_exam.probe
stage tok_exam.difficulty                       # now with icl, self, probe predictors
echo "=== $(date '+%F %T') done; run bandit/evaluate/valid/plots on CPU"
