#!/usr/bin/env python3
"""Paired route-bootstrap: does adding TF++ box features improve the monitor?

Model A = tfpp+intent+map, model B = tfpp+intent+map+boxes (out-of-fold scores from monitor_fp.py),
both smoothed with a causal EMA (span 10 = 0.5 s). For each model the threshold is set on the full
data so that it raises a target number of alarm onsets per minute of normal moving driving
(matched alarm frequency). Routes (all seeds of a route together) are resampled with replacement;
the same resample is applied to both models, giving a paired 95% interval on B - A.

Usage: python analysis/bootstrap_boxes.py [--data analysis/data/boxes_3seeds] [--B 2000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DT = 0.05
A, B_ = 'tfpp+intent+map', 'tfpp+intent+map+boxes'


def smooth(d, col, span=10):
  return d.groupby(['run', 'route_id'], sort=False)[col].transform(
      lambda x: x.ewm(span=span, adjust=False).mean()).to_numpy()


def per_route(d, score, thr):
  """Per route_id cluster: normal minutes, false-alarm onsets, failures, warned >=1 s / >=2 s."""
  alarm = score >= thr
  ttf = d['time_to_failure'].to_numpy()
  normal = np.isnan(ttf) | (ttf > 6)
  key = d['run'].to_numpy() + '|' + d['route_id'].to_numpy()
  first = np.r_[True, key[1:] != key[:-1]]
  prev = np.r_[False, alarm[:-1]]
  prev[first] = False
  onset = alarm & ~prev & normal
  f = pd.DataFrame({'route_id': d['route_id'].to_numpy(), 'key': key, 'normal': normal, 'onset': onset,
                    'w1': alarm & (ttf >= 1) & (ttf <= 3), 'w2': alarm & (ttf >= 2) & (ttf <= 3),
                    'pre': (ttf >= 0) & (ttf <= 6)})
  runs = f.groupby('key').agg(route_id=('route_id', 'first'), normal=('normal', 'sum'), onset=('onset', 'sum'),
                              fail=('pre', 'any'), w1=('w1', 'any'), w2=('w2', 'any'))
  runs['normal_min'] = runs['normal'] * DT / 60
  return runs.groupby('route_id')[['normal_min', 'onset', 'fail', 'w1', 'w2']].sum()


def threshold_for_rate(d, score, rate):
  """Bisection on the threshold so the full data shows `rate` alarm onsets per normal minute."""
  lo, hi = float(np.nanmin(score)), float(np.nanmax(score))
  for _ in range(40):
    mid = 0.5 * (lo + hi)
    r = per_route(d, score, mid)
    got = r['onset'].sum() / r['normal_min'].sum()
    lo, hi = (mid, hi) if got > rate else (lo, mid)
  return 0.5 * (lo + hi)


def stats(r):
  return {'onsets_per_min': r['onset'].sum() / r['normal_min'].sum(),
          'warned_ge_1s': r['w1'].sum() / r['fail'].sum(), 'warned_ge_2s': r['w2'].sum() / r['fail'].sum()}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', type=Path, default=Path('analysis/data/boxes_3seeds'))
  ap.add_argument('--B', type=int, default=2000)
  ap.add_argument('--a', default=A)
  ap.add_argument('--b', default=B_)
  ap.add_argument('--scores-dir', default='monitor_fp')
  args = ap.parse_args()
  a_col, b_col = args.a, args.b
  rng = np.random.default_rng(0)
  d = pd.read_parquet(args.data / args.scores_dir / 'oof_scores.parquet')
  d = d.sort_values(['run', 'route_id', 'step']).reset_index(drop=True)
  d['route_id'] = d['route_id'].astype(str)
  sa, sb = smooth(d, a_col), smooth(d, b_col)
  print('A =', a_col, '| B =', b_col)
  routes = np.array(sorted(d['route_id'].unique()))
  print(f'{len(routes)} routes, {d.groupby(["run","route_id"]).ngroups} route runs, '
        f'{int(per_route(d, sa, np.inf)["fail"].sum())} failures, B={args.B}')

  rows = []
  # ---- event metrics at matched alarm frequency
  for rate in (1.0, 2.0, 3.0, 4.0):
    ra = per_route(d, sa, threshold_for_rate(d, sa, rate)).reindex(routes)
    rb = per_route(d, sb, threshold_for_rate(d, sb, rate)).reindex(routes)
    pa, pb = stats(ra), stats(rb)
    boots = {k: [] for k in ('warned_ge_1s', 'warned_ge_2s')}
    Ma, Mb = ra.to_numpy(float), rb.to_numpy(float)
    cols = list(ra.columns)
    for _ in range(args.B):
      w = np.bincount(rng.integers(0, len(routes), len(routes)), minlength=len(routes)).astype(float)
      ta, tb = w @ Ma, w @ Mb
      fa, fb = ta[cols.index('fail')], tb[cols.index('fail')]
      for k, c in (('warned_ge_1s', 'w1'), ('warned_ge_2s', 'w2')):
        boots[k].append(tb[cols.index(c)] / fb - ta[cols.index(c)] / fa)
    for k, v in boots.items():
      v = np.array(v)
      rows.append({'metric': k, 'onsets_per_min': rate, 'A': pa[k], 'B': pb[k],
                   'diff': pb[k] - pa[k], 'ci_low': np.quantile(v, 0.025), 'ci_high': np.quantile(v, 0.975),
                   'p_diff_le_0': float(np.mean(v <= 0)),
                   'n_warned_A': int(ra[k.replace('warned_ge_', 'w').replace('s', '')].sum()),
                   'n_warned_B': int(rb[k.replace('warned_ge_', 'w').replace('s', '')].sum())})

  # ---- ranking metrics (step level, route-weighted bootstrap)
  y = d['label_3s'].to_numpy()
  ridx = pd.Index(routes).get_indexer(d['route_id'])
  ra_, rb_ = d[a_col].to_numpy(), d[b_col].to_numpy()
  point = {'auroc': (roc_auc_score(y, ra_), roc_auc_score(y, rb_)),
           'auprc': (average_precision_score(y, ra_), average_precision_score(y, rb_))}
  bt = {'auroc': [], 'auprc': []}
  for _ in range(min(args.B, 300)):
    w = np.bincount(rng.integers(0, len(routes), len(routes)), minlength=len(routes)).astype(float)[ridx]
    m = w > 0
    bt['auroc'].append(roc_auc_score(y[m], rb_[m], sample_weight=w[m]) - roc_auc_score(y[m], ra_[m], sample_weight=w[m]))
    bt['auprc'].append(average_precision_score(y[m], rb_[m], sample_weight=w[m]) -
                       average_precision_score(y[m], ra_[m], sample_weight=w[m]))
  for k, v in bt.items():
    v = np.array(v)
    rows.append({'metric': f'{k}_moving (raw score)', 'A': point[k][0], 'B': point[k][1],
                 'diff': point[k][1] - point[k][0], 'ci_low': np.quantile(v, 0.025),
                 'ci_high': np.quantile(v, 0.975), 'p_diff_le_0': float(np.mean(v <= 0))})

  res = pd.DataFrame(rows)
  res.to_csv(args.data / args.scores_dir / f'bootstrap_{b_col}_vs_{a_col}.csv'.replace('+', '_'), index=False)
  pd.set_option('display.width', 220)
  print(res.round(3).to_string(index=False))


if __name__ == '__main__':
  main()
