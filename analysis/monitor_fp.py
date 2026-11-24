#!/usr/bin/env python3
"""Collision-monitor improvement + false-positive study on the 4-seed, 220-route dataset.

Differences from analyze_failures.grouped_cv:
  * Evaluation and thresholds use MOVING steps only (ego_speed > 2 m/s). Stopped steps are
    trivially safe and made speed-only look strong at "5% per-step FPR".
  * Budget = share of normal moving time in alarm, after alarm post-processing.
  * A failure counts as warned >=X s only if an alarm fires between 3 s and X s before it
    (the label horizon), so coincidental alarms 5-6 s early are not credited.
  * Models trained on moving steps; new 'intent' features (TF++ speeding up / committing).
  * Alarm post-processing (EMA smoothing, k-consecutive persistence) to cut alarm onsets.
  * Hard-negative breakdown: where false alarms happen (privileged fields used for diagnosis only).

Usage: python analysis/monitor_fp.py [--data analysis/data/all_seeds_220]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from analyze_failures import SINGLE_FRAME, TEMPORAL

DT = 0.05
HORIZON = 3.0
MOVING = 2.0
BUDGETS = (0.01, 0.02, 0.05)


def add_intent_features(s: pd.DataFrame) -> pd.DataFrame:
  s = s.sort_values(['run', 'route_id', 'step']).copy()
  g = s.groupby(['run', 'route_id'], sort=False)
  if s['command'].dtype == object:
    s['command'] = s['command'].astype('category').cat.codes.replace(-1, np.nan)
  s['gt_is_junction'] = s['gt_is_junction'].astype(float)
  s['speed_gap'] = s['target_speed_scalar'] - s['ego_speed']
  for w, n in ((0.5, 10), (1, 20)):
    s[f'd_speed_{w}s'] = s['ego_speed'] - g['ego_speed'].shift(n)
    s[f'd_target_speed_{w}s'] = s['target_speed_scalar'] - g['target_speed_scalar'].shift(n)
    s[f'd_path_end_lat_{w}s'] = s['path_end_lat'] - g['path_end_lat'].shift(n)
    s[f'd_entropy_{w}s'] = s['speed_entropy'] - g['speed_entropy'].shift(n)
  s['brake_frac_1s'] = g['brake'].transform(lambda x: x.rolling(20, min_periods=1).mean())
  s['throttle_mean_1s'] = g['throttle'].transform(lambda x: x.rolling(20, min_periods=1).mean())
  s['abs_yaw_rate_max_1s'] = g['yaw_rate'].transform(lambda x: x.abs().rolling(20, min_periods=1).max())
  s['abs_path_end_lat_max_1s'] = g['path_end_lat'].transform(lambda x: x.abs().rolling(20, min_periods=1).max())
  return s


INTENT = ['speed_gap', 'brake_frac_1s', 'throttle_mean_1s', 'abs_yaw_rate_max_1s', 'abs_path_end_lat_max_1s'] + [
    f'{b}_{w}s' for w in (0.5, 1) for b in ('d_speed', 'd_target_speed', 'd_path_end_lat', 'd_entropy')]
MAP_CTX = ['command', 'gt_is_junction', 'gt_lane_width']  # derivable from HD map + route at deployment
ORACLE_SCENE = ['gt_nearest_front_dist', 'gt_nearest_any_dist', 'gt_actors_within_20m',
                'gt_lateral_offset', 'gt_heading_error_deg']  # privileged: reference only

BOX = ['ttc_plan', 'clear_plan', 'pet_plan', 'conflict_plan', 'ttc_ctrv', 'corr_dist', 'corr_close', 'n_boxes',
       'ttc_plan_min_1s', 'clear_plan_min_1s', 'd_clear_plan_0.5s', 'conflict_frac_1s', 'corr_dist_min_1s']
GT_BOX = ['gt_ttc_plan', 'gt_clear_plan', 'gt_pet_plan', 'gt_ttc_ctrv']  # privileged reference
GATES = {  # alarm allowed only when TF++'s own boxes show something near its planned path
    'no gate': None,
    'box within 2 m of plan (1 s)': lambda d: d['clear_plan_min_1s'] < 2.0,
    'box within 4 m of plan (1 s)': lambda d: d['clear_plan_min_1s'] < 4.0,
    'swept-path conflict (1 s)': lambda d: d['conflict_frac_1s'] > 0,
    'box near plan or in corridor <30 m': lambda d: (d['clear_plan_min_1s'] < 4.0) | (d['corr_dist_min_1s'] < 30),
}


def add_box_features(s: pd.DataFrame, path: Path) -> pd.DataFrame:
  b = pd.read_parquet(path)
  for k in ('ttc_plan', 'pet_plan', 'ttc_ctrv', 'gt_ttc_plan', 'gt_pet_plan', 'gt_ttc_ctrv'):
    b[k] = b[k].replace(np.inf, 5.0).clip(upper=5.0)
  b = b.sort_values(['run', 'route_id', 'step'])
  g = b.groupby(['run', 'route_id'], sort=False)
  b['ttc_plan_min_1s'] = g['ttc_plan'].transform(lambda x: x.rolling(20, min_periods=1).min())
  b['clear_plan_min_1s'] = g['clear_plan'].transform(lambda x: x.rolling(20, min_periods=1).min())
  b['d_clear_plan_0.5s'] = b['clear_plan'] - g['clear_plan'].shift(10)
  b['conflict_frac_1s'] = g['conflict_plan'].transform(lambda x: x.rolling(20, min_periods=1).mean())
  b['corr_dist_min_1s'] = g['corr_dist'].transform(lambda x: x.rolling(20, min_periods=1).min())
  return s.merge(b, on=['run', 'route_id', 'step'], how='left')


SPEEDNORM = ['path_s_near', 'path_time_to_reach', 'path_s_rate', 'path_ttc', 'path_n_30m', 'path_gap_min_30m',
             'path_gap_time', 'path_time_to_reach_min_1s', 'path_ttc_min_1s', 'corr_headway', 'corr_ttc']


def add_path_features(s: pd.DataFrame, path: Path) -> pd.DataFrame:
  p = pd.read_parquet(path).sort_values(['run', 'route_id', 'step'])
  g = p.groupby(['run', 'route_id'], sort=False)
  p['path_time_to_reach_min_1s'] = g['path_time_to_reach'].transform(lambda x: x.rolling(20, min_periods=1).min())
  p['path_ttc_min_1s'] = g['path_ttc'].transform(lambda x: x.rolling(20, min_periods=1).min())
  s = s.merge(p, on=['run', 'route_id', 'step'], how='left')
  v = np.maximum(s['ego_speed'], 1.0)
  s['corr_headway'] = (s['corr_dist'] / v).clip(upper=10.0)
  s['corr_ttc'] = np.where(s['corr_close'] > 0.1, s['corr_dist'] / s['corr_close'].clip(lower=0.1), 10.0).clip(max=10.0)
  return s


FEATURE_SETS = {
    'speed_only': ['ego_speed'],
    'tfpp': SINGLE_FRAME + TEMPORAL,
    'tfpp+intent': SINGLE_FRAME + TEMPORAL + INTENT,
    'tfpp+intent+map': SINGLE_FRAME + TEMPORAL + INTENT + MAP_CTX,
    'oracle_scene (privileged ref)': SINGLE_FRAME + TEMPORAL + INTENT + MAP_CTX + ORACLE_SCENE,
}


def cv_scores(d: pd.DataFrame, cols, folds=5) -> np.ndarray:
  X = d[cols].astype(float).to_numpy()
  y = d['label_3s'].to_numpy()
  groups = d['route_id'].to_numpy()
  oof = np.full(len(y), np.nan)
  for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       min_samples_leaf=200, l2_regularization=1.0,
                                       class_weight='balanced', random_state=0)
    m.fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]
  return oof


def postprocess(d: pd.DataFrame, score: np.ndarray, mode: str) -> np.ndarray:
  """Causal smoothing of the score per route run; alarm = processed score >= threshold."""
  sr = pd.Series(score, index=d.index)
  g = sr.groupby([d['run'], d['route_id']], sort=False)
  if mode == 'raw':
    return score
  if mode.startswith('ema'):
    span = int(mode[3:])
    return g.transform(lambda x: x.ewm(span=span, adjust=False).mean()).to_numpy()
  if mode.startswith('k'):  # alarm only after k consecutive steps above threshold == rolling min
    k = int(mode[1:])
    return g.transform(lambda x: x.rolling(k, min_periods=k).min()).to_numpy()
  raise ValueError(mode)


def onsets(alarm: np.ndarray, d: pd.DataFrame) -> int:
  a = pd.Series(alarm.astype(np.int8), index=d.index)
  prev = a.groupby([d['run'], d['route_id']], sort=False).shift(1).fillna(0)
  return int(((a == 1) & (prev == 0)).sum())


def evaluate(d: pd.DataFrame, proc: np.ndarray, budget: float) -> dict:
  ttf = d['time_to_failure'].to_numpy()
  normal = (np.isnan(ttf) | (ttf > 6)) & (d['ego_speed'].to_numpy() > MOVING)
  valid = ~np.isnan(proc)
  thr = np.quantile(proc[normal & valid], 1 - budget)
  alarm = valid & (proc >= thr) & (proc >= 0)  # gated-out steps are -1 and never alarm
  normal_minutes = normal.sum() * DT / 60
  # Onsets counted on normal moving steps only (alarm rising edges inside normal driving).
  dn = d[normal]
  n_on = onsets(alarm[normal], dn)
  leads = []
  pre = d.assign(alarm=alarm)[(ttf >= 0) & (ttf <= HORIZON)]
  fails = d[(ttf >= 0) & (ttf <= 6)].groupby(['run', 'route_id']).ngroups
  for _, s in pre.groupby(['run', 'route_id']):
    hit = s[s['alarm']]
    leads.append(hit['time_to_failure'].max() if len(hit) else 0.0)
  leads = np.array(leads + [0.0] * (fails - len(leads)))
  return {'budget': budget, 'alarm_onsets_per_min': n_on / normal_minutes,
          'warned_ge_1s': float(np.mean(leads >= 1.0)), 'warned_ge_2s': float(np.mean(leads >= 2.0)),
          'median_lead_s': float(np.median(leads)), 'n_failures': int(leads.size), 'threshold': float(thr)}


def fp_breakdown(d: pd.DataFrame, proc: np.ndarray, budget: float) -> pd.DataFrame:
  ttf = d['time_to_failure'].to_numpy()
  normal = (np.isnan(ttf) | (ttf > 6)) & (d['ego_speed'].to_numpy() > MOVING)
  thr = np.quantile(proc[normal & ~np.isnan(proc)], 1 - budget)
  n = d[normal].copy()
  n['fp'] = proc[normal] >= thr
  n['ctx_junction'] = n['gt_is_junction'].fillna(False).astype(bool)
  n['ctx_front'] = pd.cut(n['gt_nearest_front_dist'].fillna(999), [0, 10, 20, 40, 1e9],
                          labels=['<10m', '10-20m', '20-40m', 'none<40m'])
  n['ctx_speed'] = pd.cut(n['ego_speed'], [2, 6, 10, 14, 100], labels=['2-6', '6-10', '10-14', '>14'])
  n['ctx_route_fails'] = n['run_failed'].map({True: 'route later fails', False: 'route succeeds'})
  rows = []
  for c in ('ctx_junction', 'ctx_front', 'ctx_speed', 'ctx_route_fails', 'family'):
    if c not in n:
      continue
    t = n.groupby(c, observed=True)['fp'].agg(['mean', 'sum', 'size'])
    for k, r in t.iterrows():
      rows.append({'context': c, 'value': str(k), 'share_of_time': r['size'] / len(n),
                   'alarm_rate': r['mean'], 'share_of_false_alarms': r['sum'] / n['fp'].sum()})
  return pd.DataFrame(rows)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', type=Path, default=Path('analysis/data/all_seeds_220'))
  ap.add_argument('--box-features', type=Path, default=None)
  ap.add_argument('--path-features', type=Path, default=None)
  ap.add_argument('--only-sets', nargs='+', default=None, help='Restrict to these feature-set names.')
  ap.add_argument('--out-name', default='monitor_fp')
  args = ap.parse_args()
  boxes = args.box_features is not None
  if boxes:
    base_cols = SINGLE_FRAME + TEMPORAL + INTENT + MAP_CTX
    FEATURE_SETS.pop('oracle_scene (privileged ref)')
    FEATURE_SETS['speed+boxes'] = ['ego_speed'] + BOX
    FEATURE_SETS['tfpp+intent+map+boxes'] = base_cols + BOX
    FEATURE_SETS['oracle: +GT scene (privileged ref)'] = base_cols + BOX + ORACLE_SCENE + GT_BOX
  if args.path_features is not None:
    FEATURE_SETS['tfpp+intent+map+boxes+speednorm'] = base_cols + BOX + SPEEDNORM
  if args.only_sets:
    for k in list(FEATURE_SETS):
      if k not in args.only_sets:
        FEATURE_SETS.pop(k)
  out = args.data / args.out_name
  out.mkdir(exist_ok=True)

  s = pd.read_parquet(args.data / 'steps.parquet')
  routes = pd.read_parquet(args.data / 'routes.parquet')
  s = add_intent_features(s)
  if boxes:
    s = add_box_features(s, args.box_features)
  if args.path_features is not None:
    s = add_path_features(s, args.path_features)
    print('path features missing on', round(float(s['path_s_near'].isna().mean()), 4), 'of steps')
    print('box features missing on', round(float(s['clear_plan'].isna().mean()), 4), 'of steps')
  s = s[(s['post_failure'] == 0) & (s['timestamp'] > 1.0)]
  d = s[s['ego_speed'] > MOVING].copy()  # train + evaluate on moving steps
  # keep pre-failure steps even if slow? no: monitor only matters while moving; failures still
  # have moving steps in their 3 s window (median speed at failure ~10 m/s).
  rkey = routes[['run', 'route_id']].copy()
  rkey['run_failed'] = routes['first_failure_kind'].notna().to_numpy()
  if 'family' in routes:
    rkey['family'] = routes['family'].to_numpy()
  d = d.merge(rkey, on=['run', 'route_id'], how='left')
  print(f'moving steps {len(d):,}, positives {int(d.label_3s.sum()):,}, '
        f'route-runs {d.groupby(["run","route_id"]).ngroups}, routes {d.route_id.nunique()}')

  rows, scores = [], {}
  for name, cols in FEATURE_SETS.items():
    oof = cv_scores(d, cols)
    scores[name] = oof
    y = d['label_3s'].to_numpy()
    base = {'features': name, 'auroc_moving': roc_auc_score(y, oof), 'auprc_moving': average_precision_score(y, oof)}
    for mode in ('raw', 'ema10', 'ema20', 'k5', 'k10'):
      proc = postprocess(d, oof, mode)
      for b in BUDGETS:
        rows.append({**base, 'postproc': mode, **evaluate(d, proc, b)})
    print(name, f"AUROC {base['auroc_moving']:.3f} AUPRC {base['auprc_moving']:.3f}", flush=True)

  res = pd.DataFrame(rows)
  res.to_csv(out / 'monitor_results.csv', index=False)
  pd.DataFrame(scores).assign(run=d['run'].values, route_id=d['route_id'].values, step=d['step'].values,
                              label_3s=d['label_3s'].values,
                              time_to_failure=d['time_to_failure'].values).to_parquet(out / 'oof_scores.parquet')
  best = [k for k in ('tfpp+intent+map+boxes+speednorm', 'tfpp+intent+map+boxes', 'tfpp+intent+map') if k in scores][0]
  if boxes and 'tfpp+intent+map' in scores and 'tfpp+intent+map+boxes' in scores and not args.path_features:
    grows = []
    for sc in ('tfpp+intent+map', 'tfpp+intent+map+boxes'):
      proc = postprocess(d, scores[sc], 'ema10')
      for gname, gate in GATES.items():
        gp = proc if gate is None else np.where(gate(d).to_numpy(), proc, -1.0)
        for b in (0.01, 0.02, 0.03, 0.05, 0.07, 0.1):
          grows.append({'features': sc, 'gate': gname, **evaluate(d, gp, b)})
    gres = pd.DataFrame(grows)
    gres.to_csv(out / 'gate_results.csv', index=False)
    print('\nScene gates (ema10):')
    print(gres[['features', 'gate', 'budget', 'alarm_onsets_per_min', 'warned_ge_1s', 'warned_ge_2s']].round(3).to_string(index=False))
  fp = fp_breakdown(d, postprocess(d, scores[best], 'ema10'), 0.02)
  fp.to_csv(out / 'false_alarm_context.csv', index=False)

  pd.set_option('display.width', 200)
  cols = ['features', 'postproc', 'budget', 'alarm_onsets_per_min', 'warned_ge_1s', 'warned_ge_2s', 'median_lead_s']
  print(res[cols].round(3).to_string(index=False))
  print('\nFalse alarms by context (', best, 'ema10 @2%)')
  print(fp.round(3).to_string(index=False))
  (out / 'summary.json').write_text(json.dumps({'n_moving_steps': len(d), 'results': rows}, indent=1, default=float))


if __name__ == '__main__':
  main()
