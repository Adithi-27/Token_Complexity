#!/bin/bash
cd "$(dirname "$0")/.."
MAX_RETRIES=2
deadline=$(( $(date +%s) + 5 * 24 * 3600 ))
declare -A NAME=([selfalloc]=tok_p5 [realbudget]=tok_realbudget [latency]=tok_latency)
declare -A TLIM=([selfalloc]=12:00:00 [realbudget]=08:00:00 [latency]=04:00:00)
declare -A OUT=([selfalloc]=outputs/selfalloc/test.jsonl [realbudget]=outputs/realbudget/summary.json
                [latency]=outputs/latency/summary.json)
declare -A JOB TRIES STATUS
for arg in "$@"; do
    s=${arg%%:*}; JOB[$s]=${arg#*:}; TRIES[$s]=0; STATUS[$s]=active
    [ -n "${NAME[$s]}" ] || { echo "unknown stage $s" >&2; exit 1; }
done
log() { echo "$(date '+%F %T') $*"; }

cpu() {
    ( source /users/student/pg/pg25/adithi/miniconda3/etc/profile.d/conda.sh && conda activate baseenv \
      && export PYTHONPATH="$PWD/vendor:$PYTHONPATH" && python -m "$@" )
}

double_time() {
    IFS=: read -r h m s <<<"$1"; t=$(( 2 * (10#$h * 3600 + 10#$m * 60 + 10#$s) ))
    [ $t -gt 86400 ] && t=86400
    printf '%02d:%02d:%02d' $((t / 3600)) $((t % 3600 / 60)) $((t % 60))
}

resubmit() {
    local s=$1 tl=$2
    if [ "$(squeue -u "$USER" -h | wc -l)" -ge 4 ]; then
        log "$s: waiting for a free submit slot"; return 1
    fi
    local id
    id=$(sbatch --parsable --job-name="${NAME[$s]}" --time="$tl" slurm/stage.sh "tok_exam.$s") || return 1
    TRIES[$s]=$(( TRIES[$s] + 1 )); JOB[$s]=${id%%;*}; TLIM[$s]=$tl
    log "$s: resubmitted as ${JOB[$s]} (try ${TRIES[$s]}/$MAX_RETRIES, limit $tl)"
}

log "watching: $(for s in "${!JOB[@]}"; do printf '%s=%s ' "$s" "${JOB[$s]}"; done)"
declare -A WANT
while true; do
    active=0
    for s in "${!JOB[@]}"; do
        [ "${STATUS[$s]}" = active ] || continue
        active=1
        if [ -n "${WANT[$s]}" ]; then
            resubmit "$s" "${WANT[$s]}" && unset "WANT[$s]"
            continue
        fi
        st=$(sacct -j "${JOB[$s]}" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
        [ "$st" = COMPLETED ] && [ ! -s "${OUT[$s]}" ] && st=NO_OUTPUT
        case "$st" in
            PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|"") ;;
            COMPLETED)
                STATUS[$s]=done; log "$s: completed (${JOB[$s]})"
                if [ "$s" = selfalloc ]; then
                    cpu tok_exam.plots > logs/cpu_plots.log 2>&1 && log "plots rerun with P5" \
                        || log "plots failed, see logs/cpu_plots.log"
                fi ;;
            CANCELLED*)
                STATUS[$s]=cancelled; log "$s: ${JOB[$s]} was cancelled by hand; not resubmitting" ;;
            TIMEOUT|FAILED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|PREEMPTED|DEADLINE|NO_OUTPUT)
                log "$s: ${JOB[$s]} ended $st (see logs/${NAME[$s]}_${JOB[$s]}.err)"
                if [ "${TRIES[$s]}" -ge $MAX_RETRIES ]; then
                    STATUS[$s]=gave_up; log "$s: giving up after $MAX_RETRIES resubmits"
                else
                    tl=${TLIM[$s]}; [ "$st" = TIMEOUT ] && tl=$(double_time "$tl")
                    resubmit "$s" "$tl" || WANT[$s]=$tl
                fi ;;
        esac
    done
    [ $active -eq 0 ] && break
    [ "$(date +%s)" -ge $deadline ] && { log "deadline reached; stopping"; break; }
    sleep "${POLL:-600}"
done

{
    echo "Autopilot digest, $(date '+%F %T')"
    for s in "${!JOB[@]}"; do echo "  $s: ${STATUS[$s]} (last job ${JOB[$s]}, resubmits ${TRIES[$s]})"; done
    for f in outputs/realbudget/summary.json outputs/latency/summary.json; do
        [ -s "$f" ] && { echo; echo "== $f"; cat "$f"; }
    done
    if [ -s outputs/selfalloc/test.jsonl ]; then
        echo; echo "== P5 self-allocation (score %, thinking, cut runs per budget)"
        python3 -c "
import json, collections
rows = [json.loads(l) for l in open('outputs/selfalloc/test.jsonl')]
by = collections.defaultdict(list)
for r in rows: by[r['budget']].append(r)
for B in sorted(by):
    rs = by[B]
    print(B, round(sum(r['score'] for r in rs) / len(rs), 1),
          round(sum(r['thinking'] for r in rs) / len(rs)), sum(r['cut'] for r in rs), '/', len(rs))
"
    fi
} > logs/autopilot_results.txt
log "all stages settled; digest in logs/autopilot_results.txt"
