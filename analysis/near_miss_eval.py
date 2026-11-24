#!/usr/bin/env python3
"""Near-miss-aware alarm evaluation.

The 'collision within 3 s' label calls every alarm outside a failure window false, including
alarms during genuine hazards that resolved (the other actor yielded, TF++ corrected in time).
Here an alarm episode in normal driving is JUSTIFIED if, during the episode or within 1 s after
its onset, TF++'s planned path rolled out against ground-truth actors gives a time-to-collision
below NEAR_MISS_TTC (gt_ttc_plan from src/ew_analysis; privileged, used only for evaluation).

Reported per model (EMA 0.5 s smoothing, threshold swept):
  false_onsets_per_min      unjustified alarm onsets per minute of normal moving driving
  justified_onsets_per_min  near-miss alarm onsets per minute
  warned_ge_1s / ge_2s      share of failures warned, interpolated at matched FALSE onsets/min
Paired route-bootstrap for boxes vs no boxes at matched false onsets/min.

Caveat: the box features include ttc_plan computed from TF++'s boxes, which is the TF++-side
analogue of gt_ttc_plan; models with boxes are partly rewarded for agreeing with the hazard
definition. The oracle row uses gt_ttc_plan directly and is circular (reference only).

Usage: python analysis/near_miss_eval.py [--data analysis/data/boxes_3seeds] [--ttc 1.5]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.05
FWD = 20  # 1 s look-ahead after onset
MODELS = ['speed_only', 'tfpp+intent+map', 'speed+boxes', 'tfpp+intent+map+boxes',
          'oracle: +GT scene (privileged ref)']
BUDGETS = [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15]


def load(data: Path, ttc: float) -> pd.DataFrame:
  o = pd.read_parquet(data / 'monitor_fp' / 'oof_scores.parquet')
  b = pd.read_parquet(data / 'box_steps.parquet', columns=['run', 'route_id', 'step', 'gt_ttc_plan'])
  o = o.merge(b, on=['run', 'route_id', 'step'], how='left')
  o = o.sort_values(['run', 'route_id', 'step']).reset_index(drop=True)
  o['route_id'] = o['route_id'].astype(str)
  o['hazard'] = (o['gt_ttc_plan'] < ttc).astype(np.int8)
  key = o['run'] + '|' + o['route_id']
  o['key'] = key
  # hazard within the next FWD steps (inclusive), same route run
  o['hazard_fwd'] = o.groupby('key', sort=False)['hazard'].transform(
      lambda x: x[::-1].rolling(FWD + 1, min_periods=1).max()[::-1]).astype(bool)
  for m in MODELS:
    o[f'ema:{m}'] = o.groupby('key', sort=False)[m].transform(lambda x: x.ewm(span=10, adjust=False).mean())
  ttf = o['time_to_failure'].to_numpy()
  o['normal'] = np.isnan(ttf) | (ttf > 6)
  return o


def per_route(o: pd.DataFrame, score: np.ndarray, thr: float) -> pd.DataFrame:
  """Per route_id: normal minutes, false / justified onsets in normal driving, failures, warned."""
  alarm = score >= thr
  key = o['key'].to_numpy()
  first = np.r_[True, key[1:] != key[:-1]]
  prev = np.r_[False, alarm[:-1]]
  prev[first] = False
  onset = alarm & ~prev
  ep = np.cumsum(onset)  # episode id (valid where alarm)
  hz = o['hazard'].to_numpy().astype(bool) | o['hazard_fwd'].to_numpy()
  # justified if any alarm step of the episode, or the 1 s after onset, has a GT hazard
  ep_haz = pd.Series(hz & alarm).groupby(ep).transform('max').to_numpy() | o['hazard_fwd'].to_numpy()
  normal = o['normal'].to_numpy()
  ttf = o['time_to_failure'].to_numpy()
  f = pd.DataFrame({'route_id': o['route_id'].to_numpy(), 'key': key, 'normal': normal,
                    'false_on': onset & normal & ~ep_haz, 'just_on': onset & normal & ep_haz,
                    'w1': alarm & (ttf >= 1) & (ttf <= 3), 'w2': alarm & (ttf >= 2) & (ttf <= 3),
                    'pre': (ttf >= 0) & (ttf <= 6)})
  runs = f.groupby('key').agg(route_id=('route_id', 'first'), normal=('normal', 'sum'), false_on=('false_on', 'sum'),
                              just_on=('just_on', 'sum'), fail=('pre', 'any'), w1=('w1', 'any'), w2=('w2', 'any'))
  runs['normal_min'] = runs['normal'] * DT / 60
  return runs.groupby('route_id')[['normal_min', 'false_on', 'just_on', 'fail', 'w1', 'w2']].sum()


def summarize(r):
  m = r['normal_min'].sum()
  return {'false_onsets_per_min': r['false_on'].sum() / m, 'justified_onsets_per_min': r['just_on'].sum() / m,
          'justified_share': r['just_on'].sum() / max(r['just_on'].sum() + r['false_on'].sum(), 1),
          'warned_ge_1s': r['w1'].sum() / r['fail'].sum(), 'warned_ge_2s': r['w2'].sum() / r['fail'].sum()}


def thr_for_false_rate(o, score, rate):
  normal = o['normal'].to_numpy()
  lo, hi = float(np.nanmin(score[normal])), float(np.nanmax(score))
  for _ in range(40):
    mid = 0.5 * (lo + hi)
    s = summarize(per_route(o, score, mid))
    lo, hi = (mid, hi) if s['false_onsets_per_min'] > rate else (lo, mid)
  return 0.5 * (lo + hi)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', type=Path, default=Path('analysis/data/boxes_3seeds'))
  ap.add_argument('--ttc', type=float, default=1.5)
  ap.add_argument('--B', type=int, default=2000)
  args = ap.parse_args()
  o = load(args.data, args.ttc)
  out = args.data / 'monitor_fp'
  normal = o['normal'].to_numpy()
  print(f'near-miss TTC < {args.ttc} s: {o.loc[o.normal, "hazard"].mean():.3f} of normal moving steps')

  # 1) budget sweep: how many "false alarms" were near-misses
  rows = []
  for m in MODELS:
    sc = o[f'ema:{m}'].to_numpy()
    for b in BUDGETS:
      thr = np.quantile(sc[normal], 1 - b)
      s = summarize(per_route(o, sc, thr))
      rows.append({'model': m, 'budget': b, 'all_onsets_per_min': s['false_onsets_per_min'] + s['justified_onsets_per_min'], **s})
  sweep = pd.DataFrame(rows)
  sweep.to_csv(out / f'near_miss_sweep_ttc{args.ttc}.csv', index=False)

  # 2) warned at matched FALSE onsets per minute (interpolated) vs at matched ALL onsets
  pd.set_option('display.width', 220)
  tab = []
  for m in MODELS:
    s = sweep[sweep.model == m]
    for rate in (1, 2, 3):
      a = s.sort_values('all_onsets_per_min'); f = s.sort_values('false_onsets_per_min')
      tab.append({'model': m, 'onsets_per_min': rate,
                  'warned_1s @all-onsets': np.interp(rate, a.all_onsets_per_min, a.warned_ge_1s),
                  'warned_1s @false-onsets': np.interp(rate, f.false_onsets_per_min, f.warned_ge_1s),
                  'warned_2s @false-onsets': np.interp(rate, f.false_onsets_per_min, f.warned_ge_2s),
                  'justified_share @false-rate': np.interp(rate, f.false_onsets_per_min, f.justified_share)})
  tab = pd.DataFrame(tab)
  tab.to_csv(out / f'near_miss_matched_ttc{args.ttc}.csv', index=False)
  print(tab.round(3).to_string(index=False))

  # 3) paired route bootstrap: boxes vs no boxes at matched false onsets/min
  rng = np.random.default_rng(0)
  A, B = 'tfpp+intent+map', 'tfpp+intent+map+boxes'
  routes = np.array(sorted(o['route_id'].unique()))
  brow = []
  for rate in (1.0, 2.0, 3.0):
    ra = per_route(o, o[f'ema:{A}'].to_numpy(), thr_for_false_rate(o, o[f'ema:{A}'].to_numpy(), rate)).reindex(routes)
    rb = per_route(o, o[f'ema:{B}'].to_numpy(), thr_for_false_rate(o, o[f'ema:{B}'].to_numpy(), rate)).reindex(routes)
    cols = list(ra.columns); Ma, Mb = ra.to_numpy(float), rb.to_numpy(float)
    for c, name in (('w1', 'warned_ge_1s'), ('w2', 'warned_ge_2s')):
      pa = ra[c].sum() / ra['fail'].sum(); pb = rb[c].sum() / rb['fail'].sum()
      diffs = []
      for _ in range(args.B):
        w = np.bincount(rng.integers(0, len(routes), len(routes)), minlength=len(routes)).astype(float)
        ta, tb = w @ Ma, w @ Mb
        diffs.append(tb[cols.index(c)] / tb[cols.index('fail')] - ta[cols.index(c)] / ta[cols.index('fail')])
      diffs = np.array(diffs)
      brow.append({'false_onsets_per_min': rate, 'metric': name, 'no_boxes': pa, 'with_boxes': pb, 'diff': pb - pa,
                   'ci_low': np.quantile(diffs, 0.025), 'ci_high': np.quantile(diffs, 0.975),
                   'p_diff_le_0': float(np.mean(diffs <= 0))})
  bt = pd.DataFrame(brow)
  bt.to_csv(out / f'near_miss_bootstrap_ttc{args.ttc}.csv', index=False)
  print('\nBoxes vs no boxes at matched FALSE onsets/min (paired route bootstrap)')
  print(bt.round(3).to_string(index=False))


if __name__ == '__main__':
  main()
