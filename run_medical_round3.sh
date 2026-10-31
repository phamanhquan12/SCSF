#!/usr/bin/env bash
# =============================================================================
# run_medical_round3.sh
# Round-3 deep supervision aimed at the SCSF selective head:
#   pd_scsf  Prediction-Depth SCSF: training-only depth heads on layer2+layer3;
#            the SCSF TCP target becomes p_L(y)^(1-beta) * mean_d q_d(y)^beta.
#   ro_scsf  Risk-Ordered SCSF: ro_ds's training-only risk-ordered head on
#            layer3+layer4, the stages the SCSF calibrator reads.
# Both keep SCSF unchanged at inference (raw TCP-head logit is the score);
# the aux score is logged alongside (aux_aurc) and MSP as msp_aurc.
#
# Same medical-suite protocol as run_medical_suite.sh / run_medical_round2.sh
# (ResNet-50 pretrained, 224, 100 epochs, pretrain 20, batch 64, SGD 0.01,
# milestones 40/70/90, seeds 0 1 2, 7 thesis datasets, SCSF defaults).
#
# Usage:
#   bash run_medical_round3.sh status
#   bash run_medical_round3.sh next [--gpu 0]
#   bash run_medical_round3.sh 2 [--gpu 0]
#   bash run_medical_round3.sh aggregate
#
# Plans (ROUND3_PLAN=full by default):
#   full      Full candidates (2 days x 21 runs): pd_scsf, ro_scsf.
#   controls  Mechanism controls (5 days):
#             1 pd_scsf target=final  (same heads + DS gradient, SCSF target)
#             2 ts_scsf               (no DS: SCSF target softened by temperature to
#                                      pd_scsf's per-epoch mean target; needs day-1 full runs)
#             3 pd_scsf detach=new    (depth target, no DS gradient into the backbone)
#             4 ro_scsf rank=0        (ordinary DS on SCSF)
#             5 ro_scsf detach=new    (risk-ordered head, no DS gradient)
#   ablation  pd_scsf with the layer3 depth head only (drops layer2), 1 day.
#   detach_pd pd_scsf only with detach=new (depth TCP target; aux CE does not
#             update the backbone — isolates DS representation effects), 1 day.
#   screen    First screen: COVID seed 0 only for the two Full candidates.
#
# ROUND3_DATASETS / ROUND3_SEEDS / RUN_ID / RESULTS_ROOT override paths.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-round3}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/paper/medical_suite}"
ROUND3_PLAN="${ROUND3_PLAN:-full}"
read -r -a SEEDS <<< "${ROUND3_SEEDS:-0 1 2}"
read -r -a DATASETS <<< "${ROUND3_DATASETS:-covid_qu_ex ham10000 malaria_microscopic aptos2019 brain_tumor_mri_masoud breast_ultrasound chest_ct_scan}"

# Shared Full-candidate settings.
PD_FULL="--pd-depths layer2+layer3 --pd-betas 0.5 --pd-targets depth"
RO_FULL="--ds-sites layer3+layer4 --ro-rank-weights 0.5"

case "$ROUND3_PLAN" in
    full)
        PLAN_METHODS=(pd_scsf ro_scsf)
        PLAN_FLAGS=(
            "$PD_FULL --ds-detach-modes none"
            "$RO_FULL --ds-detach-modes none"
        )
        PLAN_VARIANTS=(
            "l23-tgtdepth-b0.5-d128__w0.1r5__detach-none"
            "l34-rank0.5-d128__w0.1r5__detach-none"
        )
        ;;
    controls)
        PLAN_METHODS=(pd_scsf ts_scsf pd_scsf ro_scsf ro_scsf)
        PLAN_FLAGS=(
            "--pd-depths layer2+layer3 --pd-targets final --ds-detach-modes none"
            "--pd-betas 0.5"
            "$PD_FULL --ds-detach-modes new"
            "--ds-sites layer3+layer4 --ro-rank-weights 0 --ds-detach-modes none"
            "$RO_FULL --ds-detach-modes new"
        )
        PLAN_VARIANTS=(
            "l23-tgtfinal-b0-d128__w0.1r5__detach-none"
            "matched-l23-b0.5"
            "l23-tgtdepth-b0.5-d128__w0.1r5__detach-new"
            "l34-rank0-d128__w0.1r5__detach-none"
            "l34-rank0.5-d128__w0.1r5__detach-new"
        )
        ;;
    ablation)
        PLAN_METHODS=(pd_scsf)
        PLAN_FLAGS=(
            "--pd-depths layer3 --pd-betas 0.5 --pd-targets depth --ds-detach-modes none"
        )
        PLAN_VARIANTS=(
            "l3-tgtdepth-b0.5-d128__w0.1r5__detach-none"
        )
        ;;
    detach_pd)
        # Same recipe as full pd_scsf, but --ds-detach new: depth heads still
        # train and still supply the TCP target; their CE gradient into the
        # backbone is zeroed (isolates deep-supervision representation effects).
        PLAN_METHODS=(pd_scsf)
        PLAN_FLAGS=(
            "$PD_FULL --ds-detach-modes new"
        )
        PLAN_VARIANTS=(
            "l23-tgtdepth-b0.5-d128__w0.1r5__detach-new"
        )
        ;;
    screen)
        PLAN_METHODS=(pd_scsf ro_scsf)
        PLAN_FLAGS=(
            "$PD_FULL --ds-detach-modes none"
            "$RO_FULL --ds-detach-modes none"
        )
        PLAN_VARIANTS=(
            "l23-tgtdepth-b0.5-d128__w0.1r5__detach-none"
            "l34-rank0.5-d128__w0.1r5__detach-none"
        )
        SEEDS=(0)
        DATASETS=(covid_qu_ex)
        ;;
    *)
        echo "Unknown ROUND3_PLAN '$ROUND3_PLAN' (use full, controls, ablation, detach_pd or screen)" >&2
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
    echo "Run: $RUN_ROOT (round3 plan: $ROUND3_PLAN)"
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
    echo "=== Round3 day $day ($ROUND3_PLAN): $(day_label "$day") on ${#DATASETS[@]} datasets x seeds ${SEEDS[*]} ==="
    bash "$SCRIPT_DIR/run_medical_suite.sh" \
        --run-id "$RUN_ID" \
        --results-root "$RESULTS_ROOT" \
        --skip-existing \
        --methods "$(day_method "$day")" \
        --seeds "${SEEDS[@]}" \
        --datasets "${DATASETS[@]}" \
        "${flags[@]}" \
        "$@"
    echo "=== Round3 day $day finished: $(day_done_count "$day")/$(day_total "$day") runs ==="
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
        echo "All $NUM_DAYS days of Round3 plan '$ROUND3_PLAN' are finished."
        ;;
    ''|*[!0-9]*)
        echo "Unknown command: $cmd (use status, next, aggregate, or a day number)" >&2
        exit 1
        ;;
    *)
        run_day "$cmd" "${FORWARD_ARGS[@]}"
        ;;
esac
