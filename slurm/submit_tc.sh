#!/bin/bash
# Submit the Token Complexity GPU stages. Needs free submit slots (QOS anandi: 4 per user).
# Usage: bash slurm/submit_tc.sh [curves] [icl] [probe] [selfalloc]   (default: curves icl)
#   curves     step 7 truncation curves (everything downstream needs these)  ~6-10 h
#   icl        ICL budgets, question + paper level, k x selection grid       ~2-4 h
#              (labels from curves if outputs/difficulty/needed.jsonl exists, else full-budget traces)
#   probe      self-rating and 128-token probe features                      ~1 h
#   selfalloc  P5 whole-paper self-allocation                                ~4-6 h
#   realbudget prefix determinism + budget-aware-prompt pilot (after curves) ~2-3 h
#   latency    wall clock per paper at batch size 1, 3 papers x 4 policies     ~2 h
# After curves finish, run the CPU stages on the login node:
#   python -m tok_exam.difficulty && python -m tok_exam.bandit && python -m tok_exam.evaluate \
#     && python -m tok_exam.valid && python -m tok_exam.plots
set -e
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then stages=(curves icl); else stages=("$@"); fi
used=$(squeue -u "$USER" -h | wc -l)
echo "jobs submitted now: $used / 4"
for s in "${stages[@]}"; do
    if [ "$used" -ge 4 ]; then echo "no submit slot left for $s; rerun later" >&2; exit 1; fi
    case $s in
        curves)    sbatch --job-name=tok_curves --time=24:00:00 slurm/stage.sh tok_exam.curves ;;
        icl)       sbatch --job-name=tok_icl --time=12:00:00 slurm/stage.sh tok_exam.icl ;;
        probe)     sbatch --job-name=tok_probe --time=04:00:00 slurm/stage.sh tok_exam.probe ;;
        selfalloc) sbatch --job-name=tok_p5 --time=12:00:00 slurm/stage.sh tok_exam.selfalloc ;;
        realbudget) sbatch --job-name=tok_realbudget --time=08:00:00 slurm/stage.sh tok_exam.realbudget ;;
        latency)   sbatch --job-name=tok_latency --time=04:00:00 slurm/stage.sh tok_exam.latency ;;
        *) echo "unknown stage $s" >&2; exit 1 ;;
    esac
    used=$((used + 1))
done
