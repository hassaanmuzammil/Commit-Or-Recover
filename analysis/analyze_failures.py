#!/usr/bin/env python3
"""Failure deep-dive on the tables written by build_dataset.py.

Writes figures to <data>/figures and a machine-readable summary to <data>/summary.json:

  1. Where TF++ fails: outcome rates by scenario family / scenario type / town / weather.
  2. In which situations: ego + world context at the first failure of every route.
  3. Precursors: event-aligned traces of single-frame and temporal signals over the
     6 s before failure vs. matched non-failure windows.
  4. Predictability: per-signal AUROC/AUPRC for "failure within 3 s" and route-grouped
     cross-validated models (single-frame vs temporal vs both), plus warning time at a
     fixed false-alarm rate.

Privileged (gt_*) columns describe situations only; they are never model inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
from sklearn.ensemble import HistGradientBoostingClassifier  # pylint: disable=wrong-import-position
from sklearn.linear_model import LogisticRegression  # pylint: disable=wrong-import-position
from sklearn.metrics import average_precision_score, roc_auc_score  # pylint: disable=wrong-import-position
from sklearn.model_selection import GroupKFold  # pylint: disable=wrong-import-position
from sklearn.pipeline import make_pipeline  # pylint: disable=wrong-import-position
from sklearn.preprocessing import StandardScaler  # pylint: disable=wrong-import-position

# Non-privileged monitor inputs, grouped as in the proposal.
SINGLE_FRAME = ['speed_entropy', 'speed_top_prob', 'speed_margin', 'target_speed_scalar', 'ego_speed',
                'acceleration', 'yaw_rate', 'steer', 'throttle', 'brake', 'tp_dist', 'tp_angle',
                'path_end_lat', 'path_end_heading', 'path_curvature']
TEMPORAL = [f'{base}_{w}s' for w in (1, 2) for base in (
    'steer_reversals', 'steer_tv', 'path_disagree_mean', 'path_disagree_max', 'path_end_lat_std',
    'speed_js_mean', 'speed_entropy_mean', 'speed_argmax_changes', 'tb_switches', 'jerk_abs')] + [
        'path_disagree_mean', 'speed_js', 'steer_rate']
TRACE_SIGNALS = ['ego_speed', 'speed_entropy', 'path_disagree_mean_1s', 'speed_js_mean_1s',
                 'steer_tv_1s', 'path_end_lat', 'gt_nearest_front_dist', 'gt_lateral_offset']


def outcome_tables(routes: pd.DataFrame) -> dict:
  routes = routes.copy()
  routes['failed'] = routes['first_failure_kind'].notna()
  out = {}
  for key in ('family', 'focus_scenario', 'scenario_types', 'town'):
    if key in routes and routes[key].notna().any():
      g = routes.groupby(key).agg(n=('failed', 'size'), failure_rate=('failed', 'mean'),
                                  mean_route_completion=('score_route', 'mean'),
                                  mean_driving_score=('score_composed', 'mean'))
      out[key] = g.sort_values('failure_rate', ascending=False).round(3)
  kinds = routes['first_failure_kind'].value_counts(dropna=False)
  out['first_failure_kind'] = kinds.rename('n').to_frame()
  return out


def failure_context(steps: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
  """Context at each route's first failure (and 1.5 s before it)."""
  rows = []
  first = (events[events['is_failure']].dropna(subset=['step_index'])
           .sort_values('timestamp').groupby(['run', 'route_id']).head(1))
  for _, ev in first.iterrows():
    s = steps[(steps['run'] == ev['run']) & (steps['route_id'] == ev['route_id'])].reset_index(drop=True)
    i = int(ev['step_index'])
    lead = max(0, i - 30)
    rows.append({
        'run': ev['run'], 'route_id': ev['route_id'], 'kind': ev['kind'],
        'other_actor': (ev['other_actor_type'] or '').split('.')[0] or None,
        'other_actor_type': ev['other_actor_type'],
        't_fail': s.loc[i, 'timestamp'],
        'speed_at_fail': s.loc[i, 'ego_speed'], 'speed_1p5s_before': s.loc[lead, 'ego_speed'],
        'brake_1p5s_before': s.loc[lead:i, 'brake'].mean(),
        'in_junction': s.loc[i, 'gt_is_junction'], 'command': s.loc[lead, 'command'],
        'lateral_offset': s.loc[i, 'gt_lateral_offset'], 'heading_error': s.loc[i, 'gt_heading_error_deg'],
        'nearest_front_type_before': s.loc[lead, 'gt_nearest_front_type'],
        'nearest_front_dist_before': s.loc[lead, 'gt_nearest_front_dist'],
        'nearest_any_type_before': s.loc[lead, 'gt_nearest_any_type'],
        'nearest_any_dist_before': s.loc[lead, 'gt_nearest_any_dist'],
        'actors_within_20m_before': s.loc[lead, 'gt_actors_within_20m'],
        'time_in_junction_before': float(s.loc[max(0, i - 100):i, 'gt_is_junction'].astype(float).sum() * 0.05),
        'dist_to_trigger': s.loc[i, 'dist_to_trigger'] if 'dist_to_trigger' in s else np.nan,
        'speed_entropy_before': s.loc[lead, 'speed_entropy'],
        'path_disagree_1s_before': s.loc[lead, 'path_disagree_mean_1s'],
    })
  return pd.DataFrame(rows)


def aligned_traces(steps: pd.DataFrame, events: pd.DataFrame, fig_dir: Path, window_s=6.0, dt=0.05):
  """Mean +- 95% CI of signals in the window before the first failure vs. control windows."""
  n = int(window_s / dt)
  first = (events[events['is_failure']].dropna(subset=['step_index'])
           .sort_values('timestamp').groupby(['run', 'route_id']).head(1))
  rng = np.random.default_rng(0)
  fail_w, ctrl_w = [], []
  for (run, route), s in steps.groupby(['run', 'route_id']):
    s = s.reset_index(drop=True)
    ev = first[(first['run'] == run) & (first['route_id'] == route)]
    if len(ev):
      i = int(ev['step_index'].iloc[0])
      if i >= n:
        fail_w.append(s.loc[i - n:i - 1, TRACE_SIGNALS].to_numpy(dtype=float))
    # control windows: steps far (>8 s) from any failure, ego moving at the window end
    ok = s.index[(s['time_to_failure'].isna() | (s['time_to_failure'] > 8)) & (s['post_failure'] == 0)
                 & (s.index >= n) & (s['ego_speed'] > 1.0)]
    for j in rng.choice(ok, size=min(3, len(ok)), replace=False) if len(ok) else []:
      ctrl_w.append(s.loc[j - n:j - 1, TRACE_SIGNALS].to_numpy(dtype=float))
  if not fail_w:
    return 0, 0
  t = np.arange(-n, 0) * dt
  fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
  for k, (ax, name) in enumerate(zip(axes.flat, TRACE_SIGNALS)):
    for arr, label, color in ((np.stack(fail_w), f'before failure (n={len(fail_w)})', 'C3'),
                              (np.stack(ctrl_w) if ctrl_w else None, f'control (n={len(ctrl_w)})', 'C0')):
      if arr is None:
        continue
      m = np.nanmean(arr[:, :, k], axis=0)
      se = np.nanstd(arr[:, :, k], axis=0) / np.sqrt(np.maximum(1, np.sum(~np.isnan(arr[:, :, k]), axis=0)))
      ax.plot(t, m, color=color, label=label)
      ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.2)
    ax.axvline(-3, color='grey', ls=':', lw=1)
    ax.set_title(name)
  axes[0, 0].legend(fontsize=8)
  for ax in axes[1]:
    ax.set_xlabel('time before first failure [s]')
  fig.tight_layout()
  fig.savefig(fig_dir / 'precursor_traces.png', dpi=130)
  plt.close(fig)
  return len(fail_w), len(ctrl_w)


def _eval_mask(steps):
  # Exclude post-failure steps and the stationary warm-up (first 1 s) from evaluation.
  return (steps['post_failure'] == 0) & (steps['timestamp'] > 1.0)


def univariate_scores(steps: pd.DataFrame, label='label_3s') -> pd.DataFrame:
  d = steps[_eval_mask(steps)]
  y = d[label].to_numpy()
  rows = []
  if y.sum() == 0 or y.sum() == len(y):
    return pd.DataFrame()
  for col in SINGLE_FRAME + TEMPORAL + ['gt_nearest_front_dist']:
    x = d[col].astype(float)
    ok = x.notna().to_numpy()
    if ok.sum() < 100 or y[ok].sum() == 0:
      continue
    auc = roc_auc_score(y[ok], x[ok])
    direction = 1 if auc >= 0.5 else -1
    xs = x[ok] * direction
    rows.append({'signal': col, 'group': 'single' if col in SINGLE_FRAME else
                 ('temporal' if col in TEMPORAL else 'privileged_geometry'),
                 'direction': '+' if direction > 0 else '-', 'auroc': max(auc, 1 - auc),
                 'auprc': average_precision_score(y[ok], xs), 'base_rate': y[ok].mean()})
  return pd.DataFrame(rows).sort_values('auroc', ascending=False)


def grouped_cv(steps: pd.DataFrame, label='label_3s', folds=5,
               feature_sets: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Route-grouped CV: no route appears in both train and test."""
  d = steps[_eval_mask(steps)].copy()
  # Group by route only: the same route at another seed must not leak across folds.
  groups = d['route_id'].to_numpy()
  y = d[label].to_numpy()
  results, oof_all = [], {}
  n_groups = len(np.unique(groups))
  if y.sum() < 20 or n_groups < folds:
    return pd.DataFrame(), pd.DataFrame()
  feature_sets = feature_sets or {'single_frame': SINGLE_FRAME, 'temporal': TEMPORAL,
                                 'single+temporal': SINGLE_FRAME + TEMPORAL}
  for name, cols in feature_sets.items():
    X = d[cols].astype(float).to_numpy()
    for model_name, make in (
        ('logreg', lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))),
        ('gbm', lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=0))):
      oof = np.full(len(y), np.nan)
      for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        if y[tr].min() == y[tr].max():
          continue  # fold without failures (only with very small datasets)
        Xtr, Xte = X[tr], X[te]
        if model_name == 'logreg':
          med = np.nanmedian(Xtr, axis=0)
          Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        m = make().fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
      oof_all[f'{name}/{model_name}'] = oof
      ok = ~np.isnan(oof)
      results.append({'features': name, 'model': model_name, 'auroc': roc_auc_score(y[ok], oof[ok]),
                      'auprc': average_precision_score(y[ok], oof[ok]), 'base_rate': y[ok].mean(),
                      'coverage': ok.mean(), **warning_time_stats(d[ok], oof[ok], label)})
  return pd.DataFrame(results), pd.DataFrame(oof_all, index=d.index)


def warning_time_stats(d: pd.DataFrame, score: np.ndarray, label: str, fpr=0.05) -> dict:
  """Threshold at the given per-step false-alarm rate on negatives; report how early
  each failure is first flagged (within the 6 s before it)."""
  neg = score[(d[label] == 0).to_numpy() & (d['time_to_failure'].isna() | (d['time_to_failure'] > 6)).to_numpy()]
  if neg.size == 0:
    return {}
  thr = np.quantile(neg, 1 - fpr)
  lead = []
  dd = d.assign(score=score)
  for _, s in dd[dd['time_to_failure'].between(0, 6)].groupby(['run', 'route_id']):
    hit = s[s['score'] >= thr]
    lead.append(hit['time_to_failure'].max() if len(hit) else 0.0)
  lead = np.array(lead)
  return {'fpr': fpr, 'failures': int(lead.size), 'detected_ge_1s': float(np.mean(lead >= 1.0)) if lead.size else np.nan,
          'detected_ge_2s': float(np.mean(lead >= 2.0)) if lead.size else np.nan,
          'median_warning_s': float(np.median(lead)) if lead.size else np.nan}


def plot_outcomes(tables: dict, fig_dir: Path):
  for key in ('family', 'focus_scenario', 'town'):
    if key not in tables or tables[key].empty:
      continue
    t = tables[key]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(t) + 2), 4))
    ax.bar(t.index.astype(str), t['failure_rate'], color='C3')
    for i, (rate, n) in enumerate(zip(t['failure_rate'], t['n'])):
      ax.text(i, rate + 0.01, f'n={n}', ha='center', fontsize=8)
    ax.set_ylabel('routes with a failure'); ax.set_ylim(0, 1.05); ax.set_title(f'Failure rate by {key}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout(); fig.savefig(fig_dir / f'failure_rate_by_{key}.png', dpi=130); plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'data')
  args = parser.parse_args()
  steps = pd.read_parquet(args.data / 'steps.parquet')
  events = pd.read_parquet(args.data / 'events.parquet')
  routes = pd.read_parquet(args.data / 'routes.parquet')
  fig_dir = args.data / 'figures'; fig_dir.mkdir(exist_ok=True)

  summary = {'n_route_runs': len(routes), 'n_steps': len(steps),
             'n_failure_routes': int(routes['first_failure_kind'].notna().sum())}
  tables = outcome_tables(routes)
  plot_outcomes(tables, fig_dir)
  for key, t in tables.items():
    print(f'\n== outcomes by {key} ==\n{t.to_string()}')
    t.to_csv(args.data / f'outcomes_by_{key}.csv')

  ctx = failure_context(steps, events)
  ctx.to_csv(args.data / 'failure_context.csv', index=False)
  if not ctx.empty:
    print('\n== first-failure context ==\n', ctx.drop(columns=['run']).round(2).to_string())
    summary['context'] = {
        'kind': ctx['kind'].value_counts().to_dict(),
        'other_actor': ctx['other_actor'].value_counts(dropna=False).to_dict(),
        'in_junction_rate': float(ctx['in_junction'].astype(float).mean()),
        'median_speed_at_fail': float(ctx['speed_at_fail'].median()),
    }
  n_f, n_c = aligned_traces(steps, events, fig_dir)
  summary['trace_windows'] = {'failure': n_f, 'control': n_c}

  uni = univariate_scores(steps)
  if not uni.empty:
    uni.to_csv(args.data / 'univariate_label3s.csv', index=False)
    print('\n== univariate predictiveness (label_3s) ==\n', uni.round(3).head(25).to_string(index=False))
  cv, _ = grouped_cv(steps)
  if not cv.empty:
    cv.to_csv(args.data / 'grouped_cv_label3s.csv', index=False)
    print('\n== route-grouped CV ==\n', cv.round(3).to_string(index=False))
    summary['grouped_cv'] = cv.round(4).to_dict(orient='records')
  (args.data / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
  main()
