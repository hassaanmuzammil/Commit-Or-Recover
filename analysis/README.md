# analysis/

Scripts behind Finding 7 (near-misses) in [docs/README.md](../docs/README.md). Data: 220 Bench2Drive routes × traffic seeds 100/200/300, schema-v2 logs in `carla_garage/results/all_boxes_seed{100,200,300}`.

| Step | Script | Output |
|---|---|---|
| 1. Per-step dataset (TF++ outputs, temporal features, labels) | `build_dataset.py RUN_DIR ... --out DATA` | `steps.parquet`, `events.parquet`, `routes.parquet` |
| 2. Box geometry per step (reuses `src/ew_analysis.extract_route`) | `box_features.py RUN_DIR ... --out DATA/box_steps.parquet` | intent-aware geometry, plus GT versions for evaluation |
| 3. Monitors + alarm smoothing + false-alarm breakdown | `monitor_fp.py --data DATA --box-features DATA/box_steps.parquet` | `DATA/monitor_fp/` (out-of-fold scores, results) |
| 4. Boxes vs no boxes, paired route bootstrap | `bootstrap_boxes.py --data DATA` | `bootstrap_*.csv` |
| 5. Near-miss-aware evaluation | `near_miss_eval.py --data DATA --ttc 1.5` | `near_miss_*.csv` |
| 6. Figures f7a / f7b | `near_miss_figures.py --data DATA` | `DATA/figures/` |

Negative-result checks: `context_thresholds.py` (per-context thresholds), `path_features.py` + `monitor_fp.py --path-features` (speed-normalised features).

Run from the repo root with `PYTHONPATH=analysis`. Ground truth (`gt_*`, `privileged`) is used only for labels and evaluation, never as monitor input.

**Logging dependency:** `build_dataset.py` also reads a top-level `carla_frame`, a `privileged` block per step and an `events_<route>.json` file per route. These come from local changes to `sensor_agent.py` and the statistics managers that are not in this PR (to be sent separately).
