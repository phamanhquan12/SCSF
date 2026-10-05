#!/usr/bin/env bash
# =============================================================================
# run_medical_day.sh
# Run the thesis medical suite one day-sized batch at a time (shared GPU server).
#
# All days write into the same run (results/paper/medical_suite/$RUN_ID) with
# --skip-existing, so an interrupted day resumes where it stopped and tables/
# always reflects every finished run.
#
# Usage:
#   bash run_medical_day.sh status            # progress per day
#   bash run_medical_day.sh next [--gpu 0]    # run the first unfinished day
#   bash run_medical_day.sh 4 [--gpu 0]       # run a specific day
#   bash run_medical_day.sh aggregate         # rebuild tables/ only
#   Extra args are forwarded to run_medical_suite.sh (e.g. --gpu 0 --workers 16).
#
# Plan (seed-major: seed 0 is complete after day 3, seed 1 after day 6, ...):
#   group A: covid_qu_ex
#   group B: malaria_microscopic brain_tumor_mri_masoud breast_ultrasound chest_ct_scan
#   group C: ham10000 aptos2019
#   day 1-3 = seed 0 x groups A,B,C; day 4-6 = seed 1; day 7-9 = seed 2.
#   Each day is 7 methods x its datasets (about 6 H100-hours per day).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-thesis}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/paper/medical_suite}"
DATASET_GROUPS=(
    "covid_qu_ex"
    "malaria_microscopic brain_tumor_mri_masoud breast_ultrasound chest_ct_scan"
    "ham10000 aptos2019"
)
SEEDS=(0 1 2)
METHODS=(sr ccl_sc sat dg selectivenet scsf dualaug)
NUM_DAYS=$(( ${#SEEDS[@]} * ${#DATASET_GROUPS[@]} ))
METRICS_DIR="$RESULTS_ROOT/$RUN_ID/metrics"

day_seed()     { echo "${SEEDS[$(( ($1 - 1) / ${#DATASET_GROUPS[@]} ))]}"; }
day_datasets() { echo "${DATASET_GROUPS[$(( ($1 - 1) % ${#DATASET_GROUPS[@]} ))]}"; }

day_done_count() {
    local seed datasets count=0 ds method
    seed="$(day_seed "$1")"
    datasets="$(day_datasets "$1")"
    for ds in $datasets; do
        for method in "${METHODS[@]}"; do
            if compgen -G "$METRICS_DIR/$ds/*/$method/*/seed_$seed/last/summary.csv" > /dev/null; then
                count=$(( count + 1 ))
            fi
        done
    done
    echo "$count"
}

day_total() { local n; n=$(day_datasets "$1" | wc -w); echo $(( n * ${#METHODS[@]} )); }

find_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then echo "$VIRTUAL_ENV/bin/python"
    elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then echo "$SCRIPT_DIR/.venv/bin/python"
    else echo python3
    fi
}

print_status() {
    echo "Run: $RESULTS_ROOT/$RUN_ID"
    local day
    for (( day = 1; day <= NUM_DAYS; day++ )); do
        printf "day %d  seed %s  %2d/%2d runs  %s\n" \
            "$day" "$(day_seed "$day")" "$(day_done_count "$day")" "$(day_total "$day")" "$(day_datasets "$day")"
    done
}

run_day() {
    local day="$1"; shift
    if (( day < 1 || day > NUM_DAYS )); then
        echo "Day must be between 1 and $NUM_DAYS" >&2
        exit 1
    fi
    echo "=== Day $day: seed $(day_seed "$day"), datasets: $(day_datasets "$day") ==="
    # shellcheck disable=SC2046
    bash "$SCRIPT_DIR/run_medical_suite.sh" \
        --run-id "$RUN_ID" \
        --results-root "$RESULTS_ROOT" \
        --skip-existing \
        --methods "${METHODS[@]}" \
        --seeds "$(day_seed "$day")" \
        --datasets $(day_datasets "$day") \
        "$@"
    echo "=== Day $day finished: $(day_done_count "$day")/$(day_total "$day") runs ==="
}

cmd="${1:-status}"
shift || true
case "$cmd" in
    status)
        print_status
        ;;
    aggregate)
        "$(find_python)" "$SCRIPT_DIR/run_paper_medical_suite.py" \
            --results-root "$RESULTS_ROOT" --run-id "$RUN_ID" --aggregate-only
        ;;
    next)
        for (( day = 1; day <= NUM_DAYS; day++ )); do
            if (( $(day_done_count "$day") < $(day_total "$day") )); then
                run_day "$day" "$@"
                exit 0
            fi
        done
        echo "All $NUM_DAYS days are finished."
        ;;
    ''|*[!0-9]*)
        echo "Unknown command: $cmd (use status, next, aggregate, or a day number)" >&2
        exit 1
        ;;
    *)
        run_day "$cmd" "$@"
        ;;
esac
