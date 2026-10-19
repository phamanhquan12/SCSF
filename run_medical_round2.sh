#!/usr/bin/env bash
# =============================================================================
# run_medical_round2.sh
# Round-2 deep-supervision candidates (docs/method_proposals.md):
#   ro_ds, ag_ds, cm_ds
#
# Same medical-suite protocol as run_medical_suite.sh / run_medical_day.sh
# (ResNet-50 pretrained, 224, 100 epochs, pretrain 20, batch 64, SGD 0.01,
# milestones 40/70/90, seeds 0 1 2, 7 thesis datasets). Does NOT modify
# run_medical_day.sh. Default site is layer3+layer4; designated score is MSP
# (aux margin is always logged alongside).
#
# Usage:
#   bash run_medical_round2.sh status
#   bash run_medical_round2.sh next [--gpu 0]
#   bash run_medical_round2.sh 2 [--gpu 0]
#   bash run_medical_round2.sh aggregate
#
# Plans (ROUND2_PLAN=full by default):
#   full      Full candidates only (3 days x 21 runs):
#             ROUND2_PLAN=full bash run_medical_round2.sh next --gpu 0
#   controls  Detached + mechanism controls (9 days):
#             All-DS-detached (1-3), RO rank=0 (4), AG align off (5),
#             CM complement off/random (6-7), then Full again if needed? No —
#             days 1-3 detached Full heads; 4-7 mechanism controls active.
#   screen    First screen: COVID seed 0 only for the three Full candidates.
#
# ROUND2_DATASETS / ROUND2_SEEDS / RUN_ID / RESULTS_ROOT override paths.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-round2}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/paper/medical_suite}"
ROUND2_PLAN="${ROUND2_PLAN:-full}"
read -r -a SEEDS <<< "${ROUND2_SEEDS:-0 1 2}"
read -r -a DATASETS <<< "${ROUND2_DATASETS:-covid_qu_ex ham10000 malaria_microscopic aptos2019 brain_tumor_mri_masoud breast_ultrasound chest_ct_scan}"

# Shared Full-candidate settings: layer3+layer4, MSP primary, detach-none.
FULL_SITE="--ds-sites layer3+layer4 --ds-primary-score msp --ds-detach-modes none"
DETACHED="--ds-sites layer3+layer4 --ds-primary-score msp --ds-detach-modes all"

case "$ROUND2_PLAN" in
    full)
        PLAN_METHODS=(ro_ds ag_ds cm_ds)
        PLAN_FLAGS=(
            "$FULL_SITE --ro-rank-weights 0.5"
            "$FULL_SITE --ag-align-modes on"
            "$FULL_SITE --cm-complements targeted"
        )
        PLAN_VARIANTS=(
            "l34-rank0.5-d128__w0.1r5__detach-none__score-msp"
            "l34-alignon-r0.25-d128__w0.1r5__detach-none__score-msp"
            "l34-comp-targeted-k0.1-d128__w0.1r5__detach-none__score-msp"
        )
        ;;
    controls)
        PLAN_METHODS=(ro_ds ag_ds cm_ds ro_ds ag_ds cm_ds cm_ds)
        PLAN_FLAGS=(
            "$DETACHED --ro-rank-weights 0.5"
            "$DETACHED --ag-align-modes on"
            "$DETACHED --cm-complements targeted"
            "$FULL_SITE --ro-rank-weights 0"
            "$FULL_SITE --ag-align-modes off"
            "$FULL_SITE --cm-complements off"
            "$FULL_SITE --cm-complements random"
        )
        PLAN_VARIANTS=(
            "l34-rank0.5-d128__w0.1r5__detach-all__score-msp"
            "l34-alignon-r0.25-d128__w0.1r5__detach-all__score-msp"
            "l34-comp-targeted-k0.1-d128__w0.1r5__detach-all__score-msp"
            "l34-rank0-d128__w0.1r5__detach-none__score-msp"
            "l34-alignoff-r0.25-d128__w0.1r5__detach-none__score-msp"
            "l34-comp-off-k0.1-d128__w0.1r5__detach-none__score-msp"
            "l34-comp-random-k0.1-d128__w0.1r5__detach-none__score-msp"
        )
        ;;
    screen)
        PLAN_METHODS=(ro_ds ag_ds cm_ds)
        PLAN_FLAGS=(
            "$FULL_SITE --ro-rank-weights 0.5"
            "$FULL_SITE --ag-align-modes on"
            "$FULL_SITE --cm-complements targeted"
        )
        PLAN_VARIANTS=(
            "l34-rank0.5-d128__w0.1r5__detach-none__score-msp"
            "l34-alignon-r0.25-d128__w0.1r5__detach-none__score-msp"
            "l34-comp-targeted-k0.1-d128__w0.1r5__detach-none__score-msp"
        )
        SEEDS=(0)
        DATASETS=(covid_qu_ex)
        ;;
    *)
        echo "Unknown ROUND2_PLAN '$ROUND2_PLAN' (use full, controls or screen)" >&2
        exit 1
        ;;
esac
NUM_DAYS=${#PLAN_METHODS[@]}

cmd="${1:-status}"
shift || true

FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)         RUN_ID="$2"; shift 2 ;;
        --run-id=*)       RUN_ID="${1#--run-id=}"; shift ;;
        --results-root)   RESULTS_ROOT="$2"; shift 2 ;;
        --results-root=*) RESULTS_ROOT="${1#--results-root=}"; shift ;;
        *)                FORWARD_ARGS+=("$1"); shift ;;
    esac
done
RUN_ROOT="$RESULTS_ROOT/$RUN_ID"

find_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then echo "$VIRTUAL_ENV/bin/python"
    elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then echo "$SCRIPT_DIR/.venv/bin/python"
    else echo python3
    fi
}
PYTHON="$(find_python)"

day_method() { echo "${PLAN_METHODS[$(( $1 - 1 ))]}"; }
day_flags() { echo "${PLAN_FLAGS[$(( $1 - 1 ))]}"; }
day_variants() { echo "${PLAN_VARIANTS[$(( $1 - 1 ))]}"; }

day_label() {
    local variants
    variants="$(day_variants "$1")"
    if [[ -n "$variants" ]]; then echo "$(day_method "$1") [$variants]"; else day_method "$1"; fi
}

day_done_count() {
    local variants
    read -r -a variants <<< "$(day_variants "$1")"
    "$PYTHON" -m scsf.run_status --run-root "$RUN_ROOT" --method "$(day_method "$1")" \
        --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" --variants "${variants[@]}"
}

day_total() {
    local variants
    read -r -a variants <<< "$(day_variants "$1")"
    local n_variants=${#variants[@]}
    (( n_variants == 0 )) && n_variants=1
    echo $(( ${#DATASETS[@]} * ${#SEEDS[@]} * n_variants ))
}

print_status() {
    echo "Run: $RUN_ROOT (round2 plan: $ROUND2_PLAN)"
    local day
    for (( day = 1; day <= NUM_DAYS; day++ )); do
        printf "day %d  %-14s %3d/%3d runs\n" "$day" "$(day_label "$day")" "$(day_done_count "$day")" "$(day_total "$day")"
    done
}

run_day() {
    local day="$1"; shift
    if (( day < 1 || day > NUM_DAYS )); then
        echo "Day must be between 1 and $NUM_DAYS" >&2
        exit 1
    fi
    local flags
    read -r -a flags <<< "$(day_flags "$day")"
    echo "=== Round2 day $day ($ROUND2_PLAN): $(day_label "$day") on ${#DATASETS[@]} datasets x seeds ${SEEDS[*]} ==="
    bash "$SCRIPT_DIR/run_medical_suite.sh" \
        --run-id "$RUN_ID" \
        --results-root "$RESULTS_ROOT" \
        --skip-existing \
        --methods "$(day_method "$day")" \
        --seeds "${SEEDS[@]}" \
        --datasets "${DATASETS[@]}" \
        "${flags[@]}" \
        "$@"
    echo "=== Round2 day $day finished: $(day_done_count "$day")/$(day_total "$day") runs ==="
}

case "$cmd" in
    status)
        print_status
        ;;
    aggregate)
        "$PYTHON" "$SCRIPT_DIR/run_paper_medical_suite.py" \
            --results-root "$RESULTS_ROOT" --run-id "$RUN_ID" --aggregate-only
        ;;
    next)
        for (( day = 1; day <= NUM_DAYS; day++ )); do
            if (( $(day_done_count "$day") < $(day_total "$day") )); then
                run_day "$day" "${FORWARD_ARGS[@]}"
                exit 0
            fi
        done
        echo "All $NUM_DAYS days of Round2 plan '$ROUND2_PLAN' are finished."
        ;;
    ''|*[!0-9]*)
        echo "Unknown command: $cmd (use status, next, aggregate, or a day number)" >&2
        exit 1
        ;;
    *)
        run_day "$cmd" "${FORWARD_ARGS[@]}"
        ;;
esac
