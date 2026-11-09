#!/usr/bin/env bash
# =============================================================================
# run_medical_day.sh
# Run the thesis medical suite one method per day (shared GPU server).
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
# Plan: day N = METHODS[N] on every dataset in datasets_thesis.md x seeds 0 1 2
# (7 datasets x 3 seeds = 21 runs per day). dualaug does two forward passes per
# step, so its day takes roughly twice as long as the others.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-thesis}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/paper/medical_suite}"
METHODS=(sr dualaug ccl_sc scsf sat dg selectivenet)
SEEDS=(0 1 2)
DATASETS=(covid_qu_ex ham10000 malaria_microscopic aptos2019 brain_tumor_mri_masoud breast_ultrasound chest_ct_scan)
NUM_DAYS=${#METHODS[@]}
METRICS_DIR="$RESULTS_ROOT/$RUN_ID/metrics"

day_method() { echo "${METHODS[$(( $1 - 1 ))]}"; }

day_done_count() {
    local method count=0 ds seed
    method="$(day_method "$1")"
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if compgen -G "$METRICS_DIR/$ds/*/$method/*/seed_$seed/last/summary.csv" > /dev/null; then
                count=$(( count + 1 ))
            fi
        done
    done
    echo "$count"
}

day_total() { echo $(( ${#DATASETS[@]} * ${#SEEDS[@]} )); }

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
        printf "day %d  %-12s %2d/%2d runs\n" "$day" "$(day_method "$day")" "$(day_done_count "$day")" "$(day_total "$day")"
    done
}

run_day() {
    local day="$1"; shift
    if (( day < 1 || day > NUM_DAYS )); then
        echo "Day must be between 1 and $NUM_DAYS" >&2
        exit 1
    fi
    echo "=== Day $day: $(day_method "$day") on ${#DATASETS[@]} datasets x seeds ${SEEDS[*]} ==="
    bash "$SCRIPT_DIR/run_medical_suite.sh" \
        --run-id "$RUN_ID" \
        --results-root "$RESULTS_ROOT" \
        --skip-existing \
        --methods "$(day_method "$day")" \
        --seeds "${SEEDS[@]}" \
        --datasets "${DATASETS[@]}" \
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
