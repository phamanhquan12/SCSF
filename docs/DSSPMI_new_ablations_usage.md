# New PD ablations — simple launcher

Use the same command style as `run_medical_pd_ablation.sh`, with the same activated Python environment and existing prepared data:

```bash
bash run_medical_dsspmi_new.sh all --gpu 0
```

This runs all **87 new ResNet50 training jobs**, including the 18 screen jobs. No historical-reference JSON setup or clean Git checkout is required by this shell wrapper. It delegates training through `run_medical_suite.sh`, retaining pretrained R50, 224px, 100 epochs, confidence warm-up 20, batch 64, evaluation batch 128, workers 4, SGD 0.01, milestones 40/70/90, Adam calibrator 0.001, seeds 0/1/2, split seed 42, and AMP off.

```bash
bash run_medical_dsspmi_new.sh status
bash run_medical_dsspmi_new.sh next --gpu 0
bash run_medical_dsspmi_new.sh 1 --gpu 0
bash run_medical_dsspmi_new.sh dry-run
```

Completed, compatible training jobs are skipped on restart. Jobs are sequential, as in the old script. A process lease prevents concurrent launchers from writing the same run ID.

| Day | New variant | Datasets | Runs |
|---|---|---|---:|
| 1 | Layer2-only | COVID, HAM, APTOS, Chest CT | 12 |
| 2 | Arithmetic target | COVID, HAM, APTOS, Chest CT | 12 |
| 3–5 | Beta .25/.75/1 | COVID, HAM, Chest CT | 27 |
| 6–7 | Layer2 target weight .25/.75 | COVID, HAM, Chest CT | 18 |
| 8–9 | Step/early auxiliary gradients | COVID, HAM, Chest CT | 18 |

As previously selected, official test evaluation stays deferred. When the fixed training matrix is finished and validation decisions are frozen:

```bash
bash run_medical_dsspmi_new.sh evaluate --gpu 0
bash run_medical_dsspmi_new.sh aggregate
```

Evaluation uses `last.pt` without retraining. Aggregation writes only the new run's tables. The original ablation script and old output directories are unchanged.

## Paths and overrides

The defaults follow the existing script's conventions:

- Data: `./data`.
- Results: `./results/paper/medical_suite`.
- New run ID: `dsspmi_new_ablations_v4`.

```bash
RUN_ID=my_new_pd_ablations DATA_DIR=/original/data \
RESULTS_ROOT=/my/results \
bash run_medical_dsspmi_new.sh all --gpu 0
```

`--run-id`, `--results-root`, and `--data-dir` also work. Dataset/seed overrides use `DSSPMI_DATASETS` / `DSSPMI_SEEDS` or `--datasets` / `--seeds`. Overrides freeze a different matrix, so use another run ID. Existing historical IDs such as `round3` are rejected.

The launcher still checks **new-run configuration and cache integrity** before skipping/reusing a checkpoint. It requires an existing format-3 cache and does not download or rebuild it. An incomplete run without a completion marker is preserved; move its incomplete checkpoint directory aside before restarting it. Keep implementation and environment fixed between training and deferred evaluation, since source/environment fingerprints form part of the new-run contract.

The simple wrapper does **not certify historic reference compatibility**. Its report labels new runs `new_run_verified`, not historically matched. The earlier optional strict workflow remains available directly through `python -m scsf.dsspmi_campaign`; that interface requires original checkpoint-bound provenance in `configs/dsspmi_reference_contract.json` and a clean checkout. It is not needed to use the shell command above.

## Separate probes and profiling

E02/E16 remain standalone measurements because they need existing trained checkpoint paths. They are not part of the `all` training command.

```bash
# Configure original checkpoint/provenance paths in configs/E02.json first:
python analysis/export_pd_frozen_features.py --config configs/E02.json
python analysis/train_pd_frozen_probes.py --config configs/E02.json
# After the campaign test release:
python analysis/train_pd_frozen_probes.py --config configs/E02.json --evaluate

# Configure matching SR/scsf/pd_scsf checkpoints in configs/E16.json:
python analysis/benchmark_pd_compute.py --config configs/E16.json
```

Probes use validation-only fitting with a 64-unit MLP, standardization fitted on allowed observations, Adam 0.001, batch 64, 200 epochs, and three-fold CV for weight decay 0/.0001/.01. There are 108 final fits plus up to 324 CV fits for the complete checkpoint matrix. Fewer than 15 correct or 15 incorrect validation predictions marks a job insufficient rather than borrowing test samples.

Profiling requires an H100, matched actual checkpoints and an otherwise idle GPU. It measures FP32 primary-only/diagnostic inference, actual method-specific training steps, memory and real-loader throughput. Projected PCam cost is explicitly an estimate.

## Validation

```bash
bash run_medical_dsspmi_new.sh smoke
python -m unittest discover -p 'test_*.py'
```

Default-path parity tests compare original source from commit `e9366ae` with the new implementation across three optimizer updates. Preserve that Git object in the test checkout. No production H100 training or profiling is launched by smoke tests.
