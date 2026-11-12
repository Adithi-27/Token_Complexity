#!/bin/bash
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
