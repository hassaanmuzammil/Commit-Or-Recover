#!/usr/bin/env python3
"""Per-step TF++ box-geometry features for schema-v2 runs (TF++ predicted boxes + planned path).

Reuses src/ew_analysis.extract_route (team code): boxes tracked across frames for velocity, ego
rolled out along TF++'s planned path timed by its target speed, then per step:
  ttc_plan, clear_plan, pet_plan, conflict_plan  (planned-path interaction with tracked boxes)
  ttc_ctrv                                        (constant-turn-rate ego rollout, for reference)
  corr_dist, corr_close, n_boxes                  (nearest box in the ego corridor, closing speed)
  gt_ttc_plan, gt_clear_plan                      (same with ground-truth actors: reference only)
All monitor-side features use only TF++ outputs; gt_* is privileged.

Usage: python analysis/box_features.py RUN_DIR [RUN_DIR ...] --out analysis/data/boxes_3seeds/box_steps.parquet
"""
import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import ew_analysis as ew  # noqa: E402

KEYS = ['ttc_plan', 'clear_plan', 'pet_plan', 'conflict_plan', 'ttc_ctrv', 'corr_dist', 'corr_close',
        'n_boxes', 'gt_ttc_plan', 'gt_clear_plan', 'gt_pet_plan', 'gt_ttc_ctrv']
STEP_RE = re.compile(r'"step":(\d+)')


def one(job):
  run_dir, route_dir = job
  logs = sorted(Path(route_dir).glob('*.jsonl'))
  ck_path = Path(run_dir) / 'checkpoints' / f'{Path(route_dir).name}.json'
  if len(logs) != 1 or not ck_path.exists():
    return None
  try:
    ck = ew.parse_checkpoint(ck_path)
    if ck is None:
      return None
    d = ew.extract_route(logs[0], ck)
  except Exception as error:  # pylint: disable=broad-except
    print(f'fail {route_dir}: {error}', flush=True)
    return None
  steps = [int(STEP_RE.search(line).group(1)) for line in logs[0].open() if line.strip()]
  out = pd.DataFrame({k: np.asarray(d[k], float) for k in KEYS})
  out.insert(0, 'step', steps[:len(out)])
  out.insert(0, 'route_id', Path(route_dir).name)
  out.insert(0, 'run', Path(run_dir).name)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('runs', nargs='+', type=Path)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--procs', type=int, default=16)
  args = ap.parse_args()
  jobs = [(str(r), str(d)) for r in args.runs for d in sorted((r / 'logs').iterdir()) if d.is_dir()]
  print(f'{len(jobs)} route logs', flush=True)
  with Pool(args.procs) as pool:
    parts = [p for p in pool.imap_unordered(one, jobs, chunksize=1) if p is not None]
  df = pd.concat(parts, ignore_index=True)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  df.to_parquet(args.out)
  print(f'wrote {len(df):,} steps from {len(parts)} routes -> {args.out}')


if __name__ == '__main__':
  main()
