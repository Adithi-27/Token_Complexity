#!/bin/bash
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
