#!/usr/bin/env python3
"""Speed-normalised, along-path box features (TF++ outputs only) for schema-v2 runs.

Target: false alarms at high speed on an empty road. The idea is to give the monitor
"how many seconds until I reach something on my planned path" instead of raw speed.

Per step, TF++'s planned path (ego frame, x forward, y right) is extended straight to PATH_LEN m.
Each TF++ box (vehicle / walker / bicycle classes) is projected onto the path:
  s    = arc length of the projection (m ahead along the path)
  gap  = lateral distance to the path minus the box's half-width and the ego half-width
A box is "on path" if gap < ON_PATH_GAP. Features:
  path_s_near        arc length to the nearest on-path box (capped at PATH_LEN)
  path_time_to_reach path_s_near / max(speed, 1)          (s; capped at 10)
  path_s_rate        d(path_s_near)/dt over 0.5 s         (m/s; negative = closing)
  path_ttc           path_s_near / closing speed          (s; capped at 10)
  path_n_30m         number of on-path boxes within 30 m
  path_gap_min_30m   smallest lateral gap among boxes within 30 m of arc length
  path_gap_time      path_gap_min_30m-box arc length / speed (time to pass the tightest box)

Usage: python analysis/path_features.py RUN_DIR [...] --out analysis/data/boxes_3seeds/path_steps.parquet
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

PATH_LEN = 50.0
ON_PATH_GAP = 0.5
EGO_HALF_W = 1.0642
CAP_T = 10.0
DYN_CLASSES = (0, 1, 4)


def extended_path(path):
  pts = np.vstack([[0.0, 0.0], np.asarray(path, float)])
  seg = np.diff(pts, axis=0)
  L = np.hypot(seg[:, 0], seg[:, 1])
  keep = L > 1e-3
  if not keep.any():
    pts, seg, L = np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([1.0])
  else:
    pts = np.vstack([pts[0], pts[1:][keep]])
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
  extra = PATH_LEN - L.sum()
  if extra > 0:
    pts = np.vstack([pts, pts[-1] + seg[-1] / L[-1] * extra])
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
  return pts, seg, L, np.r_[0.0, np.cumsum(L)]


def project(points, pts, seg, L, scum):
  """Arc length and unsigned lateral distance of each point to the polyline."""
  ap = points[:, None, :] - pts[None, :-1, :]
  t = np.clip((ap * seg[None]).sum(-1) / np.maximum(L ** 2, 1e-9)[None], 0, 1)
  proj = pts[None, :-1, :] + t[..., None] * seg[None]
  dist = np.hypot(points[:, None, 0] - proj[..., 0], points[:, None, 1] - proj[..., 1])
  j = dist.argmin(axis=1)
  idx = np.arange(len(points))
  return scum[j] + t[idx, j] * L[j], dist[idx, j]


def one(route_dir):
  logs = sorted(Path(route_dir).glob('*.jsonl'))
  if len(logs) != 1:
    return None
  rows = []
  for line in logs[0].open():
    if not line.strip():
      continue
    r = json.loads(line)
    v = float(r['ego_speed'])
    b = np.array([x for x in r['pred_boxes'] if int(round(x[7])) in DYN_CLASSES], float).reshape(-1, 9)
    s_near, n30, gap_min, s_gap = PATH_LEN, 0, 10.0, PATH_LEN
    if len(b):
      pts, seg, L, scum = extended_path(r['predicted_path'])
      s, lat = project(b[:, :2], pts, seg, L, scum)
      half_w = np.minimum(b[:, 2], b[:, 3])  # boxes are mostly seen side-on or end-on; take the narrow half-extent
      gap = lat - half_w - EGO_HALF_W
      ahead = (s > 0.5) & (b[:, 0] > 0)
      on = ahead & (gap < ON_PATH_GAP)
      if on.any():
        s_near = float(min(s[on].min(), PATH_LEN))
      n30 = int((on & (s < 30)).sum())
      near = ahead & (s < 30)
      if near.any():
        k = int(np.argmin(np.where(near, gap, np.inf)))
        gap_min, s_gap = float(min(gap[k], 10.0)), float(s[k])
    rows.append((r['step'], v, s_near, n30, gap_min, s_gap))
  d = pd.DataFrame(rows, columns=['step', 'v', 'path_s_near', 'path_n_30m', 'path_gap_min_30m', 's_gap'])
  vv = np.maximum(d['v'], 1.0)
  d['path_time_to_reach'] = (d['path_s_near'] / vv).clip(upper=CAP_T)
  lagged = d['path_s_near'].shift(10)
  valid = (d['path_s_near'] < PATH_LEN) & (lagged < PATH_LEN)
  d['path_s_rate'] = np.where(valid, (d['path_s_near'] - lagged) / 0.5, 0.0)
  closing = np.maximum(-d['path_s_rate'], 0.1)
  d['path_ttc'] = np.where(d['path_s_near'] < PATH_LEN, d['path_s_near'] / closing, CAP_T).clip(max=CAP_T)
  d['path_gap_time'] = (d['s_gap'] / vv).clip(upper=CAP_T)
  d = d.drop(columns=['v', 's_gap'])
  d.insert(0, 'route_id', Path(route_dir).name)
  d.insert(0, 'run', Path(route_dir).parents[1].name)
  return d


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('runs', nargs='+', type=Path)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--procs', type=int, default=16)
  args = ap.parse_args()
  jobs = [str(d) for r in args.runs for d in sorted((r / 'logs').iterdir()) if d.is_dir()]
  with Pool(args.procs) as pool:
    parts = [p for p in pool.imap_unordered(one, jobs, chunksize=2) if p is not None]
  df = pd.concat(parts, ignore_index=True)
  df.to_parquet(args.out)
  print(f'wrote {len(df):,} steps from {len(parts)} routes -> {args.out}')


if __name__ == '__main__':
  main()
