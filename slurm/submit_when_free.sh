#!/bin/bash
# Wait for free QOS submit slots, then submit each stage with slurm/submit_tc.sh.
# Usage: nohup bash slurm/submit_when_free.sh selfalloc realbudget > logs/submit_when_free.log 2>&1 &
# Polls every 10 minutes; gives up after 4 days. Never touches other jobs.
cd "$(dirname "$0")/.."
stages=("$@")
[ ${#stages[@]} -eq 0 ] && { echo "name at least one stage" >&2; exit 1; }
deadline=$(( $(date +%s) + 4 * 24 * 3600 ))
for s in "${stages[@]}"; do
    while true; do
        used=$(squeue -u "$USER" -h | wc -l)
        if [ "$used" -lt 4 ]; then
            echo "$(date '+%F %T') $used/4 slots used: submitting $s"
            bash slurm/submit_tc.sh "$s" || { echo "submit of $s failed" >&2; exit 1; }
            break
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "$(date '+%F %T') gave up waiting for a slot ($s not submitted)" >&2; exit 1
        fi
        sleep 600
    done
done
echo "$(date '+%F %T') all submitted: ${stages[*]}"
