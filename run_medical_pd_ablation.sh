#!/usr/bin/env bash
# =============================================================================
# run_medical_pd_ablation.sh
# Ablation suite for pd_scsf on the thesis medical datasets:
#   Day 1  detach-new   cosine depth heads, DS grads into backbone detached
#                       (isolates deep-supervision representation effects)
#   Day 2  linear       standard linear depth heads (GAP -> LN -> Linear ->
#                       ReLU -> Linear), detach=none, same depths/beta/schedule
#                       as full pd_scsf (isolates cosine vs linear head)
#
# Protocol matches run_medical_round3.sh / run_medical_suite.sh
# (ResNet-50 pretrained, 224, 100 epochs, pretrain 20, batch 64, SGD 0.01,
# milestones 40/70/90, seeds 0 1 2, 7 thesis datasets).
#
# Usage:
#   bash run_medical_pd_ablation.sh status
#   bash run_medical_pd_ablation.sh next [--gpu 0]
#   bash run_medical_pd_ablation.sh 1 [--gpu 0]          # detach-new only
#   bash run_medical_pd_ablation.sh 2 [--gpu 0]          # linear only
#   bash run_medical_pd_ablation.sh all [--gpu 0]        # both days
#   bash run_medical_pd_ablation.sh aggregate
#
# Overrides: RUN_ID / RESULTS_ROOT / PD_ABLATION_DATASETS / PD_ABLATION_SEEDS
# Default RUN_ID=round3 so detach-new reuses existing complete.json via
# --skip-existing when present.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-round3}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/paper/medical_suite}"
read -r -a SEEDS <<< "${PD_ABLATION_SEEDS:-0 1 2}"
read -r -a DATASETS <<< "${PD_ABLATION_DATASETS:-covid_qu_ex ham10000 malaria_microscopic aptos2019 brain_tumor_mri_masoud breast_ultrasound chest_ct_scan}"

# Shared pd_scsf recipe (depths / beta / schedule); only detach mode or head type changes.
PD_BASE=(--pd-depths layer2+layer3 --pd-betas 0.5 --pd-targets depth)

PLAN_METHODS=(pd_scsf pd_scsf)
PLAN_FLAGS=(
    "${PD_BASE[*]} --pd-head-types cosine --ds-detach-modes new"
    "${PD_BASE[*]} --pd-head-types linear --ds-detach-modes none"
)
PLAN_VARIANTS=(
    "l23-tgtdepth-b0.5-d128__w0.1r5__detach-new"
    "l23-tgtdepth-b0.5-d128-linear__w0.1r5__detach-none"
)
PLAN_LABELS=(
    "detach-new (cosine heads, no DS grad into backbone)"
    "linear head (standard Linear classifier, detach=none)"
)
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
day_label() { echo "${PLAN_LABELS[$(( $1 - 1 ))]}"; }

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
    echo "Run: $RUN_ROOT (pd_scsf ablation: detach-new + linear head)"
    local day
    for (( day = 1; day <= NUM_DAYS; day++ )); do
        printf "day %d  %-60s %3d/%3d runs\n" \
            "$day" "$(day_label "$day")" "$(day_done_count "$day")" "$(day_total "$day")"
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
    echo "=== pd ablation day $day: $(day_label "$day") on ${#DATASETS[@]} datasets x seeds ${SEEDS[*]} ==="
    bash "$SCRIPT_DIR/run_medical_suite.sh" \
        --run-id "$RUN_ID" \
        --results-root "$RESULTS_ROOT" \
        --skip-existing \
        --methods "$(day_method "$day")" \
        --seeds "${SEEDS[@]}" \
        --datasets "${DATASETS[@]}" \
        "${flags[@]}" \
        "$@"
    echo "=== pd ablation day $day finished: $(day_done_count "$day")/$(day_total "$day") runs ==="
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
        echo "All $NUM_DAYS days of the pd_scsf ablation are finished."
        ;;
    all)
        for (( day = 1; day <= NUM_DAYS; day++ )); do
            if (( $(day_done_count "$day") < $(day_total "$day") )); then
                run_day "$day" "${FORWARD_ARGS[@]}"
            else
                echo "=== pd ablation day $day already complete: $(day_done_count "$day")/$(day_total "$day") ==="
            fi
        done
        print_status
        ;;
    ''|*[!0-9]*)
        echo "Unknown command: $cmd (use status, next, all, aggregate, or a day number 1-$NUM_DAYS)" >&2
        exit 1
        ;;
    *)
        run_day "$cmd" "${FORWARD_ARGS[@]}"
        ;;
esac
