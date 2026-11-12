#!/bin/bash
cd "$(dirname "$0")/.."
ARRAY=$1; N=$2; MAX_RETRIES=2
log() { echo "$(date '+%F %T') $*"; }
declare -A JOB TRIES
for ((k = 0; k < N; k++)); do JOB[$k]="${ARRAY}_$k"; TRIES[$k]=0; done
deadline=$(( $(date +%s) + 5 * 24 * 3600 ))
log "watching array $ARRAY, $N tasks"
while true; do
    pending=0
    for ((k = 0; k < N; k++)); do
        [ "${JOB[$k]}" = done ] && continue
        st=$(sacct -j "${JOB[$k]}" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
        case "$st" in
            COMPLETED) JOB[$k]=done; log "task $k completed" ;;
            CANCELLED*) JOB[$k]=done; log "task $k cancelled by hand; leaving it" ;;
            TIMEOUT|FAILED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|PREEMPTED)
                pending=1
                if [ "${TRIES[$k]}" -ge $MAX_RETRIES ]; then
                    JOB[$k]=done; log "task $k ended $st; giving up after $MAX_RETRIES resubmits"
                elif [ "$(squeue -u "$USER" -h | wc -l)" -lt 4 ]; then
                    id=$(sbatch --parsable --array="$k" slurm/selfalloc_array.sh) \
                        && { JOB[$k]="${id%%;*}_$k"; TRIES[$k]=$(( TRIES[$k] + 1 )); \
                             log "task $k ended $st; resubmitted as ${JOB[$k]} (resumes), try ${TRIES[$k]}"; }
                else
                    log "task $k ended $st; waiting for a submit slot"
                fi ;;
            *) pending=1 ;;
        esac
    done
    [ $pending -eq 0 ] && break
    [ "$(date +%s)" -ge $deadline ] && { log "deadline reached"; exit 1; }
    sleep "${POLL:-600}"
done
log "all tasks settled; merging and comparing"
( source /users/student/pg/pg25/adithi/miniconda3/etc/profile.d/conda.sh && conda activate baseenv \
  && export PYTHONPATH="$PWD/vendor:$PYTHONPATH" \
  && python -m tok_exam.selfalloc --summarise > logs/p5_summary.txt 2>&1 \
  && python -m tok_exam.plots > logs/cpu_plots.log 2>&1 ) && log "done: see logs/p5_summary.txt" \
  || log "summarise or plots failed; see logs/p5_summary.txt and logs/cpu_plots.log"
