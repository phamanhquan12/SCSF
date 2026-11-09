#!/usr/bin/env bash
# New PD ablations; same usage/settings as run_medical_pd_ablation.sh.
#
# bash run_medical_dsspmi_new.sh all --gpu 0      # all 87 new training runs
# bash run_medical_dsspmi_new.sh next --gpu 0     # first unfinished day
# bash run_medical_dsspmi_new.sh 1 --gpu 0        # one day
# bash run_medical_dsspmi_new.sh status
# bash run_medical_dsspmi_new.sh evaluate --gpu 0 # test after training is finished
# bash run_medical_dsspmi_new.sh aggregate
#
# Days: 1 layer2; 2 arithmetic; 3/4/5 beta .25/.75/1;
#       6/7 layer2 weight .25/.75; 8 step gradients; 9 early gradients.
# Protocol: pretrained R50, 224px, 100 epochs, pretrain20, batch64,
# eval128, workers4, SGD .01/milestones40,70,90; seeds0,1,2; AMP off.
# Day1/2: COVID,HAM,APTOS,Chest CT. Day3..9: COVID,HAM,Chest CT.
# Tests stay deferred until 'evaluate'. Trained jobs are skipped on restart.
# No historical-reference JSON or clean-checkout requirement in this wrapper.
# Overrides: RUN_ID, RESULTS_ROOT, DATA_DIR, DSSPMI_DATASETS, DSSPMI_SEEDS.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-dsspmi_new_ablations_v4}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/paper/medical_suite}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"

find_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then echo "$VIRTUAL_ENV/bin/python"
    elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then echo "$SCRIPT_DIR/.venv/bin/python"
    else echo python3
    fi
}
PYTHON="$(find_python)"
cmd="${1:-status}"
shift || true

case "$cmd" in
    status|next|all|aggregate|evaluate|dry-run|smoke|screen|confirm|[1-9])
        exec "$PYTHON" -m scsf.dsspmi_campaign "$cmd" --simple \
            --run-id "$RUN_ID" --results-root "$RESULTS_ROOT" --data-dir "$DATA_DIR" "$@"
        ;;
    *)
        echo "Usage: $0 {status|next|1..9|all|aggregate|evaluate|dry-run|smoke|screen|confirm} [--gpu 0]" >&2
        exit 1
        ;;
esac
