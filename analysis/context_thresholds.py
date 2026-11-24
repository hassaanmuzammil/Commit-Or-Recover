#!/usr/bin/env python3
"""Per-context alarm thresholds on top of monitor_fp.py's out-of-fold scores.

Contexts = junction (map-derivable) x ego speed band (2-6, 6-10, 10-14, >14 m/s) = 8 cells.
Total budget B = share of normal moving time in alarm (same as monitor_fp.py).

Policies compared on held-out routes (route-grouped 5-fold; thresholds and allocation are
chosen on the other 4 folds only):
  global     one threshold for all steps (monitor_fp.py baseline)
  equal      every cell gets the same alarm share B (per-context quantile normalisation)
  optimized  greedy reallocation of alarm time between cells to maximise failures warned
             >=1 s ahead on the training folds, total alarm time fixed at B

Usage: python analysis/context_thresholds.py [--data analysis/data/all_seeds_220] [--score tfpp+intent+map]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DT = 0.05
SPEED_BINS = [2, 6, 10, 14, np.inf]
SPEED_LABELS = ['2-6', '6-10', '10-14', '>14']


def load(data: Path, score_col: str, span: int) -> pd.DataFrame:
  o = pd.read_parquet(data / 'monitor_fp' / 'oof_scores.parquet')
  s = pd.read_parquet(data / 'steps.parquet', columns=['run', 'route_id', 'step', 'ego_speed', 'gt_is_junction'])
  d = o[['run', 'route_id', 'step', 'label_3s', 'time_to_failure', score_col]].merge(
      s, on=['run', 'route_id', 'step'], how='left')
  d = d.sort_values(['run', 'route_id', 'step']).reset_index(drop=True)
  g = d.groupby(['run', 'route_id'], sort=False)[score_col]
  d['score'] = g.transform(lambda x: x.ewm(span=span, adjust=False).mean()) if span > 1 else d[score_col]
  junc = d['gt_is_junction'].astype(float).fillna(0).astype(int).to_numpy()
  sp = np.digitize(d['ego_speed'].to_numpy(), SPEED_BINS[1:-1])
  d['cell'] = junc * 4 + sp
  ttf = d['time_to_failure'].to_numpy()
  d['normal'] = np.isnan(ttf) | (ttf > 6)
  d['fail_id'] = -1
  pre = (ttf >= 0) & (ttf <= 6)
  ids = d.loc[pre, ['run', 'route_id']].astype(str).agg('|'.join, axis=1)
  d.loc[pre, 'fail_id'] = pd.factorize(ids)[0]
  # rising edges per route run (alarm onset needs previous step of same run)
  d['first'] = ~d.duplicated(['run', 'route_id'])
  return d


class Evaluator:
  """Fast evaluation of a per-cell threshold vector on a subset of rows."""

  def __init__(self, d: pd.DataFrame, n_cells=8):
    self.score = d['score'].to_numpy()
    self.cell = d['cell'].to_numpy()
    self.normal = d['normal'].to_numpy()
    self.first = d['first'].to_numpy()
    ttf = d['time_to_failure'].to_numpy()
    fid = d['fail_id'].to_numpy()
    self.w1 = (ttf >= 1) & (ttf <= 3)
    self.w2 = (ttf >= 2) & (ttf <= 3)
    self.fid = fid
    self.n_fail = len(np.unique(fid[fid >= 0]))
    self.n_cells = n_cells
    self.sorted_normal = [np.sort(self.score[self.normal & (self.cell == c)]) for c in range(n_cells)]
    self.w = np.array([len(x) for x in self.sorted_normal], float)
    self.w /= self.w.sum()
    self.minutes = self.normal.sum() * DT / 60

  def thresholds(self, budgets):
    thr = np.full(self.n_cells, np.inf)
    for c, b in enumerate(budgets):
      x = self.sorted_normal[c]
      if len(x) and b > 0:
        thr[c] = x[min(len(x) - 1, int(np.floor((1 - min(b, 1)) * len(x))))]
    return thr

  def metrics(self, thr):
    alarm = self.score >= thr[self.cell]
    warned1 = len(np.unique(self.fid[alarm & self.w1]))
    warned2 = len(np.unique(self.fid[alarm & self.w2]))
    prev = np.roll(alarm, 1)
    prev[self.first] = False
    onsets = (alarm & ~prev & self.normal).sum()
    return {'warned_ge_1s': warned1 / self.n_fail, 'warned_ge_2s': warned2 / self.n_fail,
            'alarm_onsets_per_min': onsets / self.minutes,
            'alarm_share': (alarm & self.normal).sum() / self.normal.sum(),
            'n_warned_1s': warned1, 'n_failures': self.n_fail}


def global_threshold(ev: Evaluator, B):
  allnorm = np.sort(np.concatenate(ev.sorted_normal))
  t = allnorm[int(np.floor((1 - B) * len(allnorm)))]
  return np.full(ev.n_cells, t)


def optimize(ev: Evaluator, B, iters=60, step_frac=0.05):
  """Greedy: move alarm time (share of total normal time) between cells while it helps."""
  b = np.full(ev.n_cells, B)  # start from equal allocation
  def obj(bb):
    m = ev.metrics(ev.thresholds(bb))
    return m['n_warned_1s'] + 0.5 * m['warned_ge_2s'] * ev.n_fail - 0.01 * m['alarm_onsets_per_min']
  best = obj(b)
  delta = step_frac * B
  for _ in range(iters):
    cand = None
    for i in range(ev.n_cells):
      for j in range(ev.n_cells):
        if i == j or ev.w[i] == 0 or ev.w[j] == 0:
          continue
        nb = b.copy()
        nb[i] -= delta / ev.w[i]
        nb[j] += delta / ev.w[j]
        if nb[i] < 0 or nb[j] > 1:
          continue
        v = obj(nb)
        if v > best + 1e-9 and (cand is None or v > cand[0]):
          cand = (v, nb)
    if cand is None:
      break
    best, b = cand
  return b


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', type=Path, default=Path('analysis/data/all_seeds_220'))
  ap.add_argument('--score', default='tfpp+intent+map')
  ap.add_argument('--span', type=int, default=10)
  args = ap.parse_args()
  d = load(args.data, args.score, args.span)
  folds = list(GroupKFold(n_splits=5).split(d, groups=d['route_id']))
  cell_names = [f"{'junction' if c // 4 else 'non-junction'} {SPEED_LABELS[c % 4]}" for c in range(8)]

  rows, allocs = [], []
  for B in (0.01, 0.02, 0.05):
    pooled = {p: np.zeros(len(d), bool) for p in ('global', 'equal', 'optimized')}
    for k, (tr, te) in enumerate(folds):
      ev_tr = Evaluator(d.iloc[tr])
      thr = {'global': global_threshold(ev_tr, B),
             'equal': ev_tr.thresholds(np.full(8, B))}
      b_opt = optimize(ev_tr, B)
      thr['optimized'] = ev_tr.thresholds(b_opt)
      allocs += [{'budget': B, 'fold': k, 'cell': cell_names[c], 'time_share': ev_tr.w[c],
                  'alarm_share_in_cell': b_opt[c]} for c in range(8)]
      te_cells = d['cell'].to_numpy()[te]
      for p, t in thr.items():
        pooled[p][te] = d['score'].to_numpy()[te] >= t[te_cells]
    ev_all = Evaluator(d)
    for p, alarm in pooled.items():
      # reuse Evaluator.metrics by encoding alarms as score>=0.5 with a constant threshold
      ev_all.score = alarm.astype(float)
      m = ev_all.metrics(np.full(8, 0.5))
      rows.append({'budget': B, 'policy': p, **m})
      ev_all.score = d['score'].to_numpy()

  res = pd.DataFrame(rows)
  al = pd.DataFrame(allocs).groupby(['budget', 'cell'], sort=False)[['time_share', 'alarm_share_in_cell']].mean().reset_index()
  out = args.data / 'monitor_fp'
  res.to_csv(out / 'context_thresholds.csv', index=False)
  al.to_csv(out / 'context_threshold_allocation.csv', index=False)
  pd.set_option('display.width', 200)
  print(res.round(3).to_string(index=False))
  print('\nOptimized allocation (mean over folds): alarm share of normal time within each cell')
  print(al.round(4).to_string(index=False))


if __name__ == '__main__':
  main()
