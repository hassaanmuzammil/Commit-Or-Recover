# Notebooks

Offline analysis of frozen TransFuser++ (TF++) on the 220 Bench2Drive routes, using the **schema-v2 logs** in `results/bench2drive_v2/`. The notebooks follow the phases of the failure-analysis plan. Run them in the order below: later notebooks reuse the caches and tables written by earlier ones.

| # | Notebook | Phase | Question | Main finding |
|---|---|---|---|---|
| 1 | [early_warning_analysis.ipynb](early_warning_analysis.ipynb) | A | Every collision (exact CARLA frames), its category, and a first look at which signals warn early | 76 counted collisions; 60 with a clean 3 s history are used for prediction |
| 2 | [perception_analysis.ipynb](perception_analysis.ipynb) | B | How good are TF++'s predicted boxes compared with ground truth? | Vehicles: 92% recall on moving actors, 0.2 m error. Pedestrians are weak. Box speed is always 0 |
| 3 | [planning_analysis.ipynb](planning_analysis.ipynb) | C | Was the plan unsafe even though the actor was visible? | TF++ keeps a stable plan and accelerates into a conflict its own boxes show |
| 4 | [control_analysis.ipynb](control_analysis.ipynb) | D | Did the controller follow the plan? | Yes: throttle and brake match TF++'s rule exactly, and path tracking is within about 0.02 m |
| 5 | [failure_attribution.ipynb](failure_attribution.ipynb) | E | Perception, planning or control: which caused each collision? | Planning about 64%, perception 10%, control 7% |
| 6 | [prediction_baselines.ipynb](prediction_baselines.ipynb) | F | Speed vs TF++ confidence vs temporal vs geometric monitors, at equal alarm budgets | Geometry on TF++'s planned path is the significant step up over speed |
| 7 | [ablations.ipynb](ablations.ipynb) | G | History length, feature groups, hard negatives, scenario families, model type | 0.5–1 s of history adds warnings ≥2 s ahead; a non-linear model matters more |

Each notebook ends with a **Takeaways** section written from its own outputs.

[gru_collision_monitor.ipynb](gru_collision_monitor.ipynb) is an earlier pilot. It trains a GRU on the older dataset in `data/monitor_dataset/` and is not part of this sequence.

[gru_v2_case_study.ipynb](gru_v2_case_study.ipynb) is a small follow-up case study: the same GRU idea on the schema-v2 logs, with and without TF++'s predicted boxes, using notebook 1's labels and notebook 6's folds and metrics. It writes to `results/bench2drive_v2/gru_v2/` and runs in about a minute on a GPU.

## Dependencies between notebooks

| Notebook | Uses |
|---|---|
| 1 | Builds `results/bench2drive_v2/early_warning/cache.pkl`, which notebooks 3–7 use |
| 2 | Builds `results/bench2drive_v2/perception/cache.pkl`, which notebook 5 uses |
| 3, 4 | Write the per-collision tables in `results/bench2drive_v2/pce/` (notebook 4 also builds that folder's cache) |
| 5 | Reads notebook 4's control table; run 3 and 4 first |
| 6, 7 | Use only notebook 1's cache |

## Code and outputs

| Notebooks | Code | Outputs |
|---|---|---|
| 1 | `src/ew_analysis.py` | `results/bench2drive_v2/early_warning/` |
| 2 | `src/perception_analysis.py` | `results/bench2drive_v2/perception/` |
| 3–5 | `src/pce_analysis.py` | `results/bench2drive_v2/pce/` |
| 6–7 | `src/fg_analysis.py` | `results/bench2drive_v2/fg/` |

Each output folder holds CSV tables and a `figures/` folder.

The `cache.pkl` files are large (30–300 MB) and can be rebuilt from the logs: the first run of each notebook takes a few minutes. Keep them out of git.

## Rules followed throughout

- **Monitor inputs** use only what TF++ itself outputs or observes. CARLA ground truth is used only for labels, actor matching and diagnosis.
- **Cross-validation** is grouped by route. Alarm thresholds are chosen on validation routes only, and confidence intervals resample routes.
- **No changes to TF++**, no braking interventions, and no GRU training in this sequence.
