#!/usr/bin/env bash
# =============================================================================
# run_medical_day.sh
# Run the thesis medical suite one method configuration per day (shared GPU server).
#
# All days write into the same run (results/paper/medical_suite/$RUN_ID) with
# --skip-existing, so an interrupted day resumes where it stopped (a finished
# last.pt whose evaluation was interrupted is re-evaluated, not retrained) and
# tables/ always reflects every complete run.
#
# Usage:
#   bash run_medical_day.sh status            # progress per day
#   bash run_medical_day.sh next [--gpu 0]    # run the first unfinished day
#   bash run_medical_day.sh 4 [--gpu 0]       # run a specific day
#   bash run_medical_day.sh aggregate         # rebuild tables/ only
#   --run-id ID / --results-root DIR (or RUN_ID= / RESULTS_ROOT=) select the run for every command.
#   Other extra args are forwarded to run_medical_suite.sh (e.g. --gpu 0 --workers 16).
#
# Plans (DAY_PLAN=thesis by default):
#   thesis    day N = one baseline on every dataset in datasets_thesis.md x seeds 0 1 2
#             (21 runs per day). dualaug does two forward passes per step, so its day
#             takes roughly twice as long as the others.
#   new       the three new methods only (verify_global, verify_spatial, dualaug_correctness),
#             main configuration, 21 runs per day:
#             DAY_PLAN=new bash run_medical_day.sh next --gpu 0
#   proposals deep-supervision proposals (docs/medical_deep_supervision_method_proposals.md)
#             with their matched controls; days 1-3 run active and detached features (42 runs).
#             Use the same RUN_ID as the baselines so tables/ compares them directly:
#             DAY_PLAN=proposals bash run_medical_day.sh next --gpu 0
#   ds        single-view deep supervision (docs/medical_deep_supervision_method_proposals (1).md):
#             mg_scsf, mp_ds, se_ds Full candidates, 21 runs per day:
#             DAY_PLAN=ds bash run_medical_day.sh next --gpu 0
#   ds_controls  matched controls for the ds plan: All-DS-detached (days 1-3), mechanism controls
#             (mg CE head, mp K=1, se GAP; days 4-6) and mg_scsf with only the margin branch detached (day 7).
#   ds_l34    mp_ds/se_ds reading layer3+layer4 like SCSF (days 1-2) and their All-DS-detached runs (days 3-4).
#
# DAY_DATASETS / DAY_SEEDS override the dataset and seed lists (status counts follow them), e.g. the
# first screen on one dataset and seed:
#   DAY_PLAN=ds DAY_DATASETS="ham10000" DAY_SEEDS="0" bash run_medical_day.sh next --gpu 0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-thesis}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/paper/medical_suite}"
DAY_PLAN="${DAY_PLAN:-thesis}"
read -r -a SEEDS <<< "${DAY_SEEDS:-0 1 2}"
read -r -a DATASETS <<< "${DAY_DATASETS:-covid_qu_ex ham10000 malaria_microscopic aptos2019 brain_tumor_mri_masoud breast_ultrasound chest_ct_scan}"

# Per day: method, run_paper_medical_suite.py flags selecting its variants, and the variant
# directories that must be complete (empty: any variant counts, as in the thesis baselines).
case "$DAY_PLAN" in
    thesis)
        PLAN_METHODS=(sr dualaug ccl_sc scsf sat dg selectivenet)
        PLAN_FLAGS=("" "" "" "" "" "" "")
        PLAN_VARIANTS=("" "" "" "" "" "" "")
        ;;
    new)
        PLAN_METHODS=(verify_global verify_spatial dualaug_correctness)
        PLAN_FLAGS=(
            "--aux-feature-modes active --verify-losses binary"
            "--aux-feature-modes active --verify-pools topk --verify-losses binary"
            "--aux-feature-modes active"
        )
        PLAN_VARIANTS=("binary__active" "topk-binary__active" "active")
        ;;
    proposals)
        PLAN_METHODS=(verify_global verify_spatial dualaug_correctness verify_global verify_spatial dualaug)
        PLAN_FLAGS=(
            "--aux-feature-modes active detached --verify-losses binary"
            "--aux-feature-modes active detached --verify-pools topk --verify-losses binary"
            "--aux-feature-modes active detached"
            "--aux-feature-modes active --verify-losses softmax"
            "--aux-feature-modes active --verify-pools gap --verify-losses binary"
            "--aux-feature-modes active detached --dualaug-targets agreement correctness"
        )
        PLAN_VARIANTS=(
            "binary__active binary__detached"
            "topk-binary__active topk-binary__detached"
            "active detached"
            "softmax__active"
            "gap-binary__active"
            "default target-agreement__detached target-correctness__active target-correctness__detached"
        )
        ;;
    ds)
        PLAN_METHODS=(mg_scsf mp_ds se_ds)
        PLAN_FLAGS=(
            "--ds-detach-modes none --mg-losses hinge"
            "--ds-detach-modes none --mp-prototypes 3"
            "--ds-detach-modes none --se-pools topk"
        )
        PLAN_VARIANTS=(
            "l3-hinge-m0.2__w0.1r5__detach-none"
            "l3-k3-d128-t0.1__w0.1r5__detach-none"
            "l3-topk0.2__w0.1r5__detach-none"
        )
        ;;
    ds_controls)
        PLAN_METHODS=(mg_scsf mp_ds se_ds mg_scsf mp_ds se_ds mg_scsf)
        PLAN_FLAGS=(
            "--ds-detach-modes all --mg-losses hinge"
            "--ds-detach-modes all --mp-prototypes 3"
            "--ds-detach-modes all --se-pools topk"
            "--ds-detach-modes none --mg-losses ce"
            "--ds-detach-modes none --mp-prototypes 1"
            "--ds-detach-modes none --se-pools gap"
            "--ds-detach-modes new --mg-losses hinge"
        )
        PLAN_VARIANTS=(
            "l3-hinge-m0.2__w0.1r5__detach-all"
            "l3-k3-d128-t0.1__w0.1r5__detach-all"
            "l3-topk0.2__w0.1r5__detach-all"
            "l3-ce-s10__w0.1r5__detach-none"
            "l3-k1-d128-t0.1__w0.1r5__detach-none"
            "l3-gap__w0.1r5__detach-none"
            "l3-hinge-m0.2__w0.1r5__detach-new"
        )
        ;;
    ds_l34)
        PLAN_METHODS=(mp_ds se_ds mp_ds se_ds)
        PLAN_FLAGS=(
            "--ds-sites layer3+layer4 --ds-detach-modes none --mp-prototypes 3"
            "--ds-sites layer3+layer4 --ds-detach-modes none --se-pools topk"
            "--ds-sites layer3+layer4 --ds-detach-modes all --mp-prototypes 3"
            "--ds-sites layer3+layer4 --ds-detach-modes all --se-pools topk"
        )
        PLAN_VARIANTS=(
            "l34-k3-d128-t0.1__w0.1r5__detach-none"
            "l34-topk0.2__w0.1r5__detach-none"
            "l34-k3-d128-t0.1__w0.1r5__detach-all"
            "l34-topk0.2__w0.1r5__detach-all"
        )
        ;;
    *)
        echo "Unknown DAY_PLAN '$DAY_PLAN' (use thesis, new, proposals, ds, ds_controls or ds_l34)" >&2
        exit 1
        ;;
esac
NUM_DAYS=${#PLAN_METHODS[@]}

cmd="${1:-status}"
shift || true

# Run identity overrides apply to progress checks as well as to the launched runs.
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
    echo "Run: $RUN_ROOT (plan: $DAY_PLAN)"
    local day
    for (( day = 1; day <= NUM_DAYS; day++ )); do
        printf "day %d  %-12s %3d/%3d runs\n" "$day" "$(day_label "$day")" "$(day_done_count "$day")" "$(day_total "$day")"
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
    echo "=== Day $day ($DAY_PLAN): $(day_label "$day") on ${#DATASETS[@]} datasets x seeds ${SEEDS[*]} ==="
    bash "$SCRIPT_DIR/run_medical_suite.sh" \
        --run-id "$RUN_ID" \
        --results-root "$RESULTS_ROOT" \
        --skip-existing \
        --methods "$(day_method "$day")" \
        --seeds "${SEEDS[@]}" \
        --datasets "${DATASETS[@]}" \
        "${flags[@]}" \
        "$@"
    echo "=== Day $day finished: $(day_done_count "$day")/$(day_total "$day") runs ==="
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
        echo "All $NUM_DAYS days of plan '$DAY_PLAN' are finished."
        ;;
    ''|*[!0-9]*)
        echo "Unknown command: $cmd (use status, next, aggregate, or a day number)" >&2
        exit 1
        ;;
    *)
        run_day "$cmd" "${FORWARD_ARGS[@]}"
        ;;
esac
