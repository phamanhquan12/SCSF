#!/usr/bin/env bash
# =============================================================================
# run_medical_r101.sh
# ResNet-101 rerun of the 14 methods in the final comparison table, on the 7
# thesis datasets plus PatchCamelyon (PCam).
#
# Methods (one "day" each, same flags/variants as the ResNet-50 runs):
#    1 sr            2 sat           3 dg            4 selectivenet
#    5 ccl_sc        6 scsf          7 mg_scsf       8 mp_ds
#    9 se_ds        10 ro_ds        11 ag_ds        12 cm_ds
#   13 pd_scsf      14 ro_scsf
# Not included (dropped from the comparison): dualaug, dualaug_correctness,
# verify_global, verify_spatial; the ts_scsf control is not rerun either.
#
# Protocol is the medical-suite protocol with only the backbone changed:
# ResNet-101 ImageNet-pretrained, 224 px, 100 epochs (20 pretrain), batch 64,
# SGD lr 0.01, milestones 40/70/90, seeds 0 1 2, last checkpoint.
# PCam keeps its official train/valid/test split (262144/32768/32768 patches).
#
# Usage:
#   bash run_medical_r101.sh prepare            # download + export PCam (once)
#   bash run_medical_r101.sh status
#   bash run_medical_r101.sh queue 0 4          # shared job queue, one slot per GPU index
#   bash run_medical_r101.sh next [--gpu 0]     # first unfinished method, all datasets
#   bash run_medical_r101.sh 13 [--gpu 0]       # one method (day number), all datasets
#   bash run_medical_r101.sh aggregate
#
# The queue runs small datasets first and PCam last (one PCam run is roughly
# 10 GPU-hours on ResNet-101). Several queue instances can share the claims,
# e.g. start "queue 0" now and "queue 4" later when GPU 4 frees up.
#
# Overrides (environment): R101_DATASETS, R101_SEEDS, R101_METHODS (day numbers
# for queue), PCAM_WORKERS (default 8; other datasets keep the suite's 4),
# RUN_ID (default r101), RESULTS_ROOT, DATA_DIR, NFS_ROOT.
# Results and checkpoints go to the NFS mount by default; PCam raw/cache data
# are symlinked from data/ to NFS by "prepare".
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NFS_ROOT="${NFS_ROOT:-/mnt/data-hps/namdp36/scsf}"
RUN_ID="${RUN_ID:-r101}"
RESULTS_ROOT="${RESULTS_ROOT:-$NFS_ROOT/results/medical_suite}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
DATASETS_FILE="$SCRIPT_DIR/datasets_r101.md"
ARCH=resnet101
PCAM_WORKERS="${PCAM_WORKERS:-8}"
read -r -a SEEDS <<< "${R101_SEEDS:-0 1 2}"
# Queue order: small datasets first, PCam last.
read -r -a DATASETS <<< "${R101_DATASETS:-chest_ct_scan breast_ultrasound aptos2019 brain_tumor_mri_masoud ham10000 malaria_microscopic covid_qu_ex pcam}"

DS_FULL="--ds-sites layer3+layer4 --ds-primary-score msp --ds-detach-modes none"
PLAN_METHODS=(sr sat dg selectivenet ccl_sc scsf mg_scsf mp_ds se_ds ro_ds ag_ds cm_ds pd_scsf ro_scsf)
PLAN_FLAGS=(
    ""
    ""
    ""
    ""
    ""
    ""
    "--ds-detach-modes none --mg-losses hinge"
    "--ds-sites layer3+layer4 --ds-detach-modes none --mp-prototypes 3"
    "--ds-sites layer3+layer4 --ds-detach-modes none --se-pools topk"
    "$DS_FULL --ro-rank-weights 0.5"
    "$DS_FULL --ag-align-modes on"
    "$DS_FULL --cm-complements targeted"
    "--pd-depths layer2+layer3 --pd-betas 0.5 --pd-targets depth --ds-detach-modes none"
    "--ds-sites layer3+layer4 --ro-rank-weights 0.5 --ds-detach-modes none"
)
PLAN_VARIANTS=(
    "default"
    "default"
    "default"
    "default"
    "ccl-official"
    "features-mid_late_logits__target-tcp__ew-1__mlr-0.001__hd-256__minlam-0.0001__scorer-meta"
    "l3-hinge-m0.2__w0.1r5__detach-none"
    "l34-k3-d128-t0.1__w0.1r5__detach-none"
    "l34-topk0.2__w0.1r5__detach-none"
    "l34-rank0.5-d128__w0.1r5__detach-none__score-msp"
    "l34-alignon-r0.25-d128__w0.1r5__detach-none__score-msp"
    "l34-comp-targeted-k0.1-d128__w0.1r5__detach-none__score-msp"
    "l23-tgtdepth-b0.5-d128__w0.1r5__detach-none"
    "l34-rank0.5-d128__w0.1r5__detach-none"
)
NUM_DAYS=${#PLAN_METHODS[@]}
read -r -a QUEUE_DAYS <<< "${R101_METHODS:-$(seq -s ' ' 1 "$NUM_DAYS")}"

cmd="${1:-status}"
shift || true

find_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then echo "$VIRTUAL_ENV/bin/python"
    elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then echo "$SCRIPT_DIR/.venv/bin/python"
    else echo python3
    fi
}
PYTHON="$(find_python)"
RUN_ROOT="$RESULTS_ROOT/$RUN_ID"

day_method() { echo "${PLAN_METHODS[$(( $1 - 1 ))]}"; }
day_flags() { echo "${PLAN_FLAGS[$(( $1 - 1 ))]}"; }
day_variant() { echo "${PLAN_VARIANTS[$(( $1 - 1 ))]}"; }

check_day() {
    if (( $1 < 1 || $1 > NUM_DAYS )); then
        echo "Day must be between 1 and $NUM_DAYS" >&2
        exit 1
    fi
}

done_count() {  # day [datasets...]
    local day="$1"; shift
    "$PYTHON" -m scsf.run_status --run-root "$RUN_ROOT" --method "$(day_method "$day")" \
        --datasets "$@" --seeds "${SEEDS[@]}" --variants "$(day_variant "$day")"
}

print_status() {
    echo "Run: $RUN_ROOT (arch $ARCH, seeds ${SEEDS[*]})"
    local day total=$(( ${#DATASETS[@]} * ${#SEEDS[@]} ))
    for (( day = 1; day <= NUM_DAYS; day++ )); do
        printf "day %2d  %-13s %3d/%3d runs\n" "$day" "$(day_method "$day")" "$(done_count "$day" "${DATASETS[@]}")" "$total"
    done
}

# One method on the given datasets/seeds through the shared medical-suite launcher.
run_method() {  # day "datasets" "seeds" [extra launcher args...]
    local day="$1" datasets="$2" seeds="$3"; shift 3
    local flags
    read -r -a flags <<< "$(day_flags "$day")"
    read -r -a ds <<< "$datasets"
    read -r -a sd <<< "$seeds"
    # PCam's 262k-patch train set needs more loader workers to keep the GPU fed
    # (measured: 4 workers ~1.2k img/s, 8 ~2.4k img/s); every PCam run uses the same value.
    local workers=()
    [[ "$datasets" == pcam ]] && workers=(--workers "$PCAM_WORKERS")
    bash "$SCRIPT_DIR/run_medical_suite.sh" \
        --arch "$ARCH" \
        --run-id "$RUN_ID" \
        --results-root "$RESULTS_ROOT" \
        --data-dir "$DATA_DIR" \
        --datasets-file "$DATASETS_FILE" \
        --skip-existing \
        --methods "$(day_method "$day")" \
        --seeds "${sd[@]}" \
        --datasets "${ds[@]}" \
        ${flags[@]+"${flags[@]}"} \
        ${workers[@]+"${workers[@]}"} \
        "$@"
}

prepare_pcam() {
    # PCam raw HDF5 + PNG export and its 224-px cache live on NFS; data/ only holds symlinks.
    local nfs_data="$NFS_ROOT/data"
    mkdir -p "$nfs_data/raw/pcam" "$nfs_data/preprocessed/pcam_224" "$DATA_DIR/raw" "$DATA_DIR/preprocessed"
    for sub in raw/pcam preprocessed/pcam_224; do
        if [[ ! -e "$DATA_DIR/$sub" ]]; then
            ln -s "$nfs_data/$sub" "$DATA_DIR/$sub"
        fi
    done
    "$PYTHON" "$SCRIPT_DIR/run_experiment.py" --dataset pcam --data-dir "$DATA_DIR" --datasets-file "$DATASETS_FILE" \
        --arch "$ARCH" --input-size 224 --download --prepare-only
}

queue() {  # gpu indices...
    local logs="$SCRIPT_DIR/logs/$RUN_ID"
    mkdir -p "$logs/claims"
    local plan_log="$logs/queue.log"
    local jobs=() ds seed day
    for ds in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for day in "${QUEUE_DAYS[@]}"; do jobs+=("$day $ds $seed"); done
        done
    done
    slot() {
        local gpu="$1" job day ds seed
        for job in "${jobs[@]}"; do
            read -r day ds seed <<< "$job"
            mkdir "$logs/claims/${day}_${ds}_s${seed}" 2>/dev/null || continue
            echo "[$(date '+%F %T')] START $(day_method "$day") $ds seed$seed gpu$gpu" >> "$plan_log"
            local status=0
            run_method "$day" "$ds" "$seed" --gpu "$gpu" >> "$logs/$(day_method "$day")_${ds}_s${seed}.log" 2>&1 || status=$?
            echo "[$(date '+%F %T')] END $(day_method "$day") $ds seed$seed gpu$gpu exit=$status" >> "$plan_log"
        done
    }
    echo "[$(date '+%F %T')] START queue slots: $* (run $RUN_ROOT)" >> "$plan_log"
    local gpu
    for gpu in "$@"; do slot "$gpu" & done
    wait
    aggregate >> "$plan_log" 2>&1 || true
    echo "[$(date '+%F %T')] QUEUE DONE (slots $*)" >> "$plan_log"
}

aggregate() {
    "$PYTHON" "$SCRIPT_DIR/run_paper_medical_suite.py" --results-root "$RESULTS_ROOT" --run-id "$RUN_ID" --aggregate-only
}

case "$cmd" in
    status)    print_status ;;
    prepare)   prepare_pcam ;;
    aggregate) aggregate ;;
    queue)
        if [[ $# -eq 0 ]]; then echo "Usage: $0 queue <gpu> [<gpu> ...]" >&2; exit 1; fi
        queue "$@"
        ;;
    next)
        for (( day = 1; day <= NUM_DAYS; day++ )); do
            if (( $(done_count "$day" "${DATASETS[@]}") < ${#DATASETS[@]} * ${#SEEDS[@]} )); then
                run_method "$day" "${DATASETS[*]}" "${SEEDS[*]}" "$@"
                exit 0
            fi
        done
        echo "All $NUM_DAYS methods are finished."
        ;;
    ''|*[!0-9]*)
        echo "Unknown command: $cmd (use status, prepare, queue, next, aggregate, or a day number)" >&2
        exit 1
        ;;
    *)
        check_day "$cmd"
        run_method "$cmd" "${DATASETS[*]}" "${SEEDS[*]}" "$@"
        ;;
esac
